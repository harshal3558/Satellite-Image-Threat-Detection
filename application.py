"""
Flask web application for Satellite Image Threat Detection.

Routes:
    GET  /           — Upload page
    POST /           — Run tiled YOLOv8 inference and render results directly
    GET  /health     — Simple health-check endpoint
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import cv2
import numpy as np
import rasterio
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from ultralytics import YOLO

from src.SITP.logger import backend_logger, detection_logger, log_detection_details, logging
from src.SITP.pipelines.prediction_pipeline import PredictPipeline
from src.SITP.utils import normalize_to_uint8

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB limit
app.config["UPLOAD_FOLDER"] = "uploads"

ALLOWED_EXTENSIONS = {"tif", "tiff"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
backend_logger.info("Initializing Flask Web Application & Inference Service")

# Global PredictPipeline instance & class names (lazily initialized on first request)
_pipeline: PredictPipeline | None = None
_CLASS_NAMES: dict[int, str] = {}


def get_pipeline(conf: float = 0.25, iou: float = 0.45) -> PredictPipeline:
    """Lazily initialize and return the warm PredictPipeline instance."""
    global _pipeline, _CLASS_NAMES
    if _pipeline is None:
        if Path("best.onnx").exists():
            startup_model = "best.onnx"
        elif Path("best.pt").exists():
            startup_model = "best.pt"
        else:
            startup_model = None

        try:
            if startup_model:
                _pipeline = PredictPipeline(model_path=startup_model, conf=conf, iou=iou)
            else:
                _pipeline = PredictPipeline(conf=conf, iou=iou)
            _CLASS_NAMES = _pipeline.model.names or {}
            backend_logger.info(
                f"Loaded warm PredictPipeline [{_pipeline.engine_type}] with {len(_CLASS_NAMES)} classes"
            )
        except Exception as exc:
            backend_logger.error(f"Failed to initialize PredictPipeline: {exc}")
            raise exc
    return _pipeline

# Class colours (BGR for OpenCV) — cycles for any number of classes
_PALETTE = [
    (0, 255, 127),   # spring green
    (0, 191, 255),   # deep sky blue
    (255, 99, 71),   # tomato
    (255, 215, 0),   # gold
    (138, 43, 226),  # blue violet
    (255, 140, 0),   # dark orange
    (0, 255, 255),   # cyan
    (255, 20, 147),  # deep pink
    (127, 255, 0),   # chartreuse
    (255, 165, 0),   # orange
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _tif_to_rgb(image_path: Path, max_dim: int = 1024) -> tuple[np.ndarray, float]:
    """Read a GeoTIFF, subsampling in rasterio if large to minimize RAM, and return uint8 RGB array."""
    with rasterio.open(image_path) as src:
        h, w = src.height, src.width
        scale = min(max_dim / max(h, w), 1.0)
        indexes = [1, 2, 3] if src.count >= 3 else [1]
        if scale < 1.0:
            out_shape = (len(indexes), int(h * scale), int(w * scale))
            img = src.read(indexes=indexes, out_shape=out_shape, resampling=rasterio.enums.Resampling.bilinear)
        else:
            img = src.read(indexes=indexes)

    if img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)
    img = np.transpose(img[:3], (1, 2, 0))   # → (H, W, 3)
    return normalize_to_uint8(img), scale


def _draw_boxes(
    vis: np.ndarray,
    detections: list,
    scale: float = 1.0,
    class_names: dict[int, str] | None = None,
) -> np.ndarray:
    """Overlay bounding boxes with class names on the image."""
    vis = vis.copy()
    vh, vw = vis.shape[:2]
    names = class_names or {}

    for det in detections:
        x1, y1, x2, y2, score, cls = det
        x1 = max(0, min(int(x1 * scale), vw - 1))
        y1 = max(0, min(int(y1 * scale), vh - 1))
        x2 = max(0, min(int(x2 * scale), vw - 1))
        y2 = max(0, min(int(y2 * scale), vh - 1))
        color = _PALETTE[int(cls) % len(_PALETTE)]

        # Box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Label with class name
        cls_name = names.get(int(cls), f"cls{int(cls)}")
        label = f"{cls_name}  {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            vis, label,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 0, 0), 1, cv2.LINE_AA,
        )

    return vis


def _to_base64(img_rgb: np.ndarray) -> str:
    """Encode an RGB numpy array as a JPEG base64 string."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Failed to encode result image.")
    return base64.b64encode(buf).decode("utf-8")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    # ── Handle file upload and prediction ────────────────────────────────────
    if "file" not in request.files:
        return render_template("index.html", error="No file part in the request.")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="No file selected.")

    if not _allowed(file.filename):
        return render_template("index.html", error="Only .tif / .tiff files are accepted.")

    # Save to disk temporarily with a unique UUID to prevent collision race conditions
    import uuid
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    filepath = Path(app.config["UPLOAD_FOLDER"]) / unique_filename
    file.save(str(filepath))

    try:
        conf = float(request.form.get("conf", 0.25))
        iou  = float(request.form.get("iou",  0.45))

        pipeline = get_pipeline(conf=conf, iou=iou)

        # ── Inference with latency measurement ───────────────────────────
        timings: dict = {}
        _t_start = time.perf_counter()
        detections = pipeline.predict(filepath, conf=conf, iou=iou, _timings=timings)
        inference_latency_s = time.perf_counter() - _t_start
        inference_latency_str = (
            f"{inference_latency_s * 1000:.0f} ms"
            if inference_latency_s < 1.0
            else f"{inference_latency_s:.2f} s"
        )

        # Build visualization with timing
        _t_vis_start = time.perf_counter()
        image_rgb, display_scale = _tif_to_rgb(filepath, max_dim=1024)
        vis_image = _draw_boxes(image_rgb, detections, scale=display_scale, class_names=_CLASS_NAMES)
        image_b64 = _to_base64(vis_image)
        del image_rgb, vis_image
        import gc
        gc.collect()
        vis_latency_ms = round((time.perf_counter() - _t_vis_start) * 1000, 1)

        t_load_ms = timings.get("t_load_ms", 0.0)
        t_infer_ms = timings.get("t_inference_ms", 0.0)
        t_nms_ms = timings.get("t_nms_ms", 0.0)
        tiles_processed = timings.get("tiles_processed", 0)
        tiles_skipped = timings.get("tiles_skipped", 0)
        engine_type = getattr(pipeline, "engine_type", "ONNX Runtime")

        total_tracked_ms = max(t_load_ms + t_infer_ms + t_nms_ms + vis_latency_ms, 0.001)
        latency_tracker = {
            "total_str": inference_latency_str,
            "total_ms": round(inference_latency_s * 1000, 1),
            "tiles_processed": tiles_processed,
            "tiles_skipped": tiles_skipped,
            "engine_type": engine_type,
            "phases": [
                {
                    "name": "Raster Slicing & Normalization",
                    "time_str": f"{t_load_ms:.1f} ms",
                    "pct": round((t_load_ms / total_tracked_ms) * 100, 1),
                    "color": "#00d4ff",
                },
                {
                    "name": (
                        f"{engine_type} Forward ({tiles_processed} tiles, {tiles_skipped} empty skipped)"
                        if tiles_skipped > 0
                        else f"{engine_type} Forward ({tiles_processed} tiles)"
                    ),
                    "time_str": f"{t_infer_ms:.1f} ms",
                    "pct": round((t_infer_ms / total_tracked_ms) * 100, 1),
                    "color": "#00ff88",
                },
                {
                    "name": "Batched NMS Suppression",
                    "time_str": f"{t_nms_ms:.1f} ms",
                    "pct": round((t_nms_ms / total_tracked_ms) * 100, 1),
                    "color": "#ffb703",
                },
                {
                    "name": "Overlay Rendering & Bounding Boxes",
                    "time_str": f"{vis_latency_ms:.1f} ms",
                    "pct": round((vis_latency_ms / total_tracked_ms) * 100, 1),
                    "color": "#b5179e",
                },
            ],
        }

        # Extract image metadata
        try:
            with rasterio.open(filepath) as src:
                width = src.width
                height = src.height
                bands = src.count
                driver = src.driver
        except Exception:
            width, height, bands, driver = 0, 0, 0, "Unknown"

        try:
            file_size_bytes = filepath.stat().st_size
            if file_size_bytes >= 1024 * 1024:
                file_size_str = f"{file_size_bytes / (1024 * 1024):.2f} MB"
            else:
                file_size_str = f"{file_size_bytes / 1024:.2f} KB"
        except Exception:
            file_size_str = "Unknown"

        img_metadata = {
            "filename": filename,
            "width": width,
            "height": height,
            "bands": bands,
            "driver": driver,
            "size": file_size_str,
        }

        # Summarize results
        class_counts: dict[str, int] = {}
        for det in detections:
            cls_id = int(det[5])
            key = _CLASS_NAMES.get(cls_id, f"Class {cls_id}")
            class_counts[key] = class_counts.get(key, 0) + 1

        # Sort classes descending for the breakdown chart
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        max_class_count = sorted_classes[0][1] if sorted_classes else 1
        class_breakdown = [
            {
                "name": name,
                "count": count,
                "percentage": round((count / max_class_count) * 100, 1)
            }
            for name, count in sorted_classes
        ]

        avg_conf = float(np.mean([d[4] for d in detections])) if detections else 0.0

        # ── High-threat intelligence (conf >= 0.75) ───────────────────────
        HIGH_CONF_THRESHOLD = 0.75
        high_threat_counts: dict[str, int] = {}
        for det in detections:
            if det[4] >= HIGH_CONF_THRESHOLD:
                cls_id = int(det[5])
                name = _CLASS_NAMES.get(cls_id, f"Class {cls_id}")
                high_threat_counts[name] = high_threat_counts.get(name, 0) + 1

        # Sort by count descending
        high_threat_sorted = sorted(high_threat_counts.items(), key=lambda x: x[1], reverse=True)

        # Build natural-language threat brief
        total_high = sum(high_threat_counts.values())
        if not high_threat_sorted:
            threat_brief = (
                "No objects were detected with high confidence (\u2265 75%). "
                "Consider lowering the confidence threshold for a broader scan."
            )
        elif len(high_threat_sorted) == 1:
            name, cnt = high_threat_sorted[0]
            threat_brief = (
                f"The satellite image contains {cnt} high-confidence "
                f"{name} target{'s' if cnt > 1 else ''} flagged at \u2265 75% certainty, "
                f"representing the primary threat in this scan."
            )
        else:
            # Top 3 for the brief
            parts = []
            for name, cnt in high_threat_sorted[:3]:
                parts.append(f"{cnt}\u00d7 {name}")
            remainder = len(high_threat_sorted) - 3
            body = ", ".join(parts)
            if remainder > 0:
                body += f", and {remainder} additional class{'es' if remainder > 1 else ''}"
            threat_brief = (
                f"The satellite image contains {total_high} high-confidence threats across "
                f"{len(high_threat_sorted)} class{'es' if len(high_threat_sorted) > 1 else ''}: "
                f"{body}. These detections exceed the 75% certainty threshold and "
                f"warrant priority review."
            )

        high_threat_objects = [
            {"name": name, "count": cnt}
            for name, cnt in high_threat_sorted
        ]

        # Log complete detection breakdown to the Detection Details Log
        log_detection_details(
            filename=filename,
            detections=detections,
            class_names=_CLASS_NAMES,
            img_metadata=img_metadata,
            conf_threshold=conf,
            iou_threshold=iou,
            extra_info={
                "pipeline": f"YOLOv8 Tiled Inference ({engine_type})",
                "unique_classes": len(class_counts),
                "inference_latency": inference_latency_str,
                "high_threat_detections": total_high,
                "latency_breakdown": {
                    "raster_load_ms": t_load_ms,
                    "inference_forward_ms": t_infer_ms,
                    "nms_ms": t_nms_ms,
                    "overlay_render_ms": vis_latency_ms,
                    "tiles_processed": tiles_processed,
                    "tiles_skipped": tiles_skipped,
                    "engine_type": engine_type,
                },
            },
        )

        # Log deployment/backend event with latency
        backend_logger.info(
            f"Prediction completed for {filename} [{engine_type}]: {len(detections)} detections, "
            f"avg conf={avg_conf:.3f}, total_latency={inference_latency_str} "
            f"[load={t_load_ms}ms, infer={t_infer_ms}ms, nms={t_nms_ms}ms, vis={vis_latency_ms}ms, "
            f"tiles_processed={tiles_processed}, tiles_skipped={tiles_skipped}], "
            f"high-threat={total_high}"
        )

        return render_template(
            "index.html",
            success=True,
            filename=filename,
            total_detections=len(detections),
            avg_confidence=f"{round(avg_conf * 100)}%",
            unique_classes=len(class_counts),
            class_breakdown=class_breakdown,
            detections=[
                {
                    "idx": i + 1,
                    "x1": round(d[0], 1), "y1": round(d[1], 1),
                    "x2": round(d[2], 1), "y2": round(d[3], 1),
                    "score": f"{round(d[4] * 100)}%",
                    "badge_class": "conf-high" if d[4] >= 0.75 else ("conf-med" if d[4] >= 0.45 else "conf-low"),
                    "class_id": int(d[5]),
                    "class_name": _CLASS_NAMES.get(int(d[5]), f"Class {int(d[5])}"),
                }
                for i, d in enumerate(detections)
            ],
            result_image=image_b64,
            conf_val=conf,
            iou_val=iou,
            img_metadata=img_metadata,
            inference_latency=inference_latency_str,
            latency_tracker=latency_tracker,
            engine_type=engine_type,
            high_threat_objects=high_threat_objects,
            high_threat_total=total_high,
            threat_brief=threat_brief,
        )

    except FileNotFoundError as exc:
        backend_logger.error(f"Model not found: {exc}")
        return render_template(
            "index.html",
            error=(
                "Trained model not found. "
                "Please run the training pipeline (python main.py) first."
            )
        )

    except Exception as exc:
        backend_logger.error(f"Prediction error: {exc}")
        return render_template("index.html", error=str(exc))

    finally:
        # Always clean up the uploaded file
        if filepath.exists():
            filepath.unlink()


@app.route("/health")
def health():
    backend_logger.info("Health check requested - status: ok")
    return jsonify({"status": "ok", "service": "satellite-threat-detection", "deployment": "active"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    backend_logger.info("Starting Flask development server on port 5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
