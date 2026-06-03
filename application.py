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
from pathlib import Path

import cv2
import numpy as np
import rasterio
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from ultralytics import YOLO

from src.SITP.logger import logging
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

# Load class names from the trained model at startup
_CLASS_NAMES: dict[int, str] = {}
if Path("best.pt").exists():
    _model_meta = YOLO("best.pt")
    _CLASS_NAMES = _model_meta.names or {}
    del _model_meta  # free memory — the pipeline loads its own model
    logging.info(f"Loaded {len(_CLASS_NAMES)} class names from best.pt")

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


def _tif_to_rgb(image_path: Path) -> np.ndarray:
    """Read a GeoTIFF and return a uint8 RGB numpy array."""
    with rasterio.open(image_path) as src:
        img = src.read()          # shape: (bands, H, W)
    img = np.transpose(img[:3], (1, 2, 0))   # → (H, W, 3)
    return normalize_to_uint8(img)


def _draw_boxes(
    image_rgb: np.ndarray,
    detections: list,
    class_names: dict[int, str] | None = None,
) -> np.ndarray:
    """Overlay bounding boxes with class names on the image (resized for display)."""
    vis = image_rgb.copy()
    h, w = vis.shape[:2]

    # Downscale for browser display (keep under 1024 px on longest side)
    max_dim = 1024
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)))

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

        pipeline = PredictPipeline(conf=conf, iou=iou)
        detections = pipeline.predict(filepath)

        # Build visualization
        image_rgb   = _tif_to_rgb(filepath)
        vis_image   = _draw_boxes(image_rgb, detections, class_names=_CLASS_NAMES)
        image_b64   = _to_base64(vis_image)

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

        logging.info(
            f"Prediction on {filename}: {len(detections)} detections, "
            f"avg conf={avg_conf:.3f}"
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
            img_metadata=img_metadata
        )

    except FileNotFoundError as exc:
        logging.error(f"Model not found: {exc}")
        return render_template(
            "index.html",
            error=(
                "Trained model not found. "
                "Please run the training pipeline (python main.py) first."
            )
        )

    except Exception as exc:
        logging.error(f"Prediction error: {exc}")
        return render_template("index.html", error=str(exc))

    finally:
        # Always clean up the uploaded file
        if filepath.exists():
            filepath.unlink()


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
