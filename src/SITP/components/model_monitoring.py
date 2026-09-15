"""
Model Monitoring component for the Satellite Image Threat Detection pipeline.

Provides post-training inspection utilities:
  - Per-class mAP50 table
  - Results CSV reader (training curves)
  - Large-image tiled inference with batched NMS
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import torch
import torchvision
from ultralytics import YOLO

from src.SITP.exception import CustomException
from src.SITP.logger import logging
from src.SITP.utils import normalize_to_uint8, tile_starts


# ---------------------------------------------------------------------------
# Component
# ---------------------------------------------------------------------------

class ModelMonitoring:
    """
    Wraps post-training diagnostics and tiled large-image inference,
    mirroring the notebook's inspection and prediction cells.
    """

    # ------------------------------------------------------------------
    # Training curves
    # ------------------------------------------------------------------

    @staticmethod
    def read_results(results_csv: str | Path) -> pd.DataFrame:
        """
        Read the ``results.csv`` written by YOLOv8 during training.

        Parameters
        ----------
        results_csv : path to the ``results.csv`` file inside the run directory.

        Returns
        -------
        pd.DataFrame with stripped column names.
        """
        try:
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            logging.info(f"Loaded training results from {results_csv}")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Per-class mAP
    # ------------------------------------------------------------------

    @staticmethod
    def inspect_per_class_performance(model: YOLO) -> None:
        """Print a per-class mAP50 table to stdout."""
        try:
            metrics = model.val()
            print(f"{'Class ID':<10} | {'Class Name':<20} | {'mAP50':<10}")
            print("-" * 45)
            for i, class_name in enumerate(model.names.values()):
                class_map50 = metrics.box.ap50[i]
                print(f"{i:<10} | {class_name:<20} | {class_map50:.4f}")
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Tiled large-image inference (Optimized with Tile Batching & RAM Slicing)
    # ------------------------------------------------------------------

    @staticmethod
    def predict_large_image(
        image_path: str | Path,
        model_path: str | Path | YOLO,
        tile_size: int = 512,
        overlap: int = 100,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        batch_size: int = 16,
        _timings: dict | None = None,
    ) -> list[list[float]]:
        """
        Run high-performance tiled inference on a large GeoTIFF and return merged detections.

        Optimizations applied:
          - In-memory raster reading (eliminates hundreds of disk window seeks).
          - Tile batching (runs multi-tile forward passes concurrently).
          - Warm model reuse (avoids disk weights reloads).
          - GPU half-precision (FP16) auto-acceleration.

        Parameters
        ----------
        image_path      : Path to the GeoTIFF image.
        model_path      : YOLO instance or path to the trained ``.pt`` weights file.
        tile_size       : Side length of each tile in pixels.
        overlap         : Pixel overlap between adjacent tiles.
        conf_threshold  : Minimum confidence score to keep a detection.
        iou_threshold   : IoU threshold for NMS.
        batch_size      : Number of tiles to batch per model forward pass.
        _timings        : Optional dict; if provided it will be populated with
                          per-phase latency keys: ``t_load_ms``, ``t_inference_ms``,
                          ``t_nms_ms``, ``tiles_processed``.

        Returns
        -------
        list[list[float]]
            Each entry is ``[x1, y1, x2, y2, score, class_id]`` in global
            pixel coordinates.
        """
        try:
            model = model_path if isinstance(model_path, YOLO) else YOLO(str(model_path))
            stride = tile_size - overlap
            if stride <= 0:
                raise ValueError("overlap must be smaller than tile_size")

            all_boxes: list[list[float]] = []
            all_scores: list[float] = []
            all_classes: list[int] = []

            # ── Phase 1: Raster load & normalisation ─────────────────────
            _t0 = time.perf_counter()
            with rasterio.open(image_path) as src:
                width, height = src.width, src.height
                raw_img = src.read()  # (bands, H, W)

            raw_img = np.transpose(raw_img[:3], (1, 2, 0))  # (H, W, 3)
            img_normalized = normalize_to_uint8(raw_img)
            del raw_img  # free memory
            _t_load = time.perf_counter() - _t0

            use_half = torch.cuda.is_available()
            _tiles_processed = 0
            _tiles_skipped = 0

            # ── Phase 2: Batched tile inference ──────────────────────────
            _t1 = time.perf_counter()

            # Helper function to process a batch of tiles
            def _process_batch(chips: list[np.ndarray], offsets: list[tuple[int, int]]) -> None:
                if not chips:
                    return
                results = model.predict(
                    chips,
                    imgsz=tile_size,
                    conf=conf_threshold,
                    verbose=False,
                    half=use_half,
                    batch=len(chips),
                )
                for result, (x_off, y_off) in zip(results, offsets):
                    for pred_box in result.boxes:
                        xyxy = pred_box.xyxy[0].cpu().numpy()
                        conf = float(pred_box.conf[0].cpu().numpy())
                        cls = int(pred_box.cls[0].cpu().numpy())

                        all_boxes.append(
                            [
                                float(xyxy[0] + x_off),
                                float(xyxy[1] + y_off),
                                float(xyxy[2] + x_off),
                                float(xyxy[3] + y_off),
                            ]
                        )
                        all_scores.append(conf)
                        all_classes.append(cls)

            current_chips: list[np.ndarray] = []
            current_offsets: list[tuple[int, int]] = []

            for y in tile_starts(height, tile_size, stride):
                for x in tile_starts(width, tile_size, stride):
                    chip = img_normalized[y : y + tile_size, x : x + tile_size]
                    if chip.shape[0] != tile_size or chip.shape[1] != tile_size:
                        continue

                    # ── Optimization 3: Smart Variance & Background Early-Exit ──
                    # Microsecond subsampled statistics (stride 4)
                    sample = chip[::4, ::4]
                    sample_std = float(np.std(sample))
                    sample_mean = float(np.mean(sample))

                    # Fast early reject for:
                    # 1. Featureless water / ocean / uniform terrain (std < 5.0)
                    # 2. No-data borders / void padding (mean < 4.0 & std < 3.0)
                    # 3. Saturated cloud whiteout (mean > 245.0 & std < 4.0)
                    if sample_std < 5.0 or (sample_mean < 4.0 and sample_std < 3.0) or (sample_mean > 245.0 and sample_std < 4.0):
                        _tiles_skipped += 1
                        continue

                    current_chips.append(chip)
                    current_offsets.append((x, y))
                    _tiles_processed += 1

                    if len(current_chips) >= batch_size:
                        _process_batch(current_chips, current_offsets)
                        current_chips.clear()
                        current_offsets.clear()

            # Process remaining trailing chips
            if current_chips:
                _process_batch(current_chips, current_offsets)
                current_chips.clear()
                current_offsets.clear()

            _t_inference = time.perf_counter() - _t1

            if not all_boxes:
                # No detections — populate timings and return early
                if _timings is not None:
                    _timings.update({
                        "t_load_ms":      round(_t_load * 1000, 1),
                        "t_inference_ms": round(_t_inference * 1000, 1),
                        "t_nms_ms":       0.0,
                        "tiles_processed": _tiles_processed,
                        "tiles_skipped":   _tiles_skipped,
                    })
                return []

            # ── Phase 3: Batched NMS ──────────────────────────────────
            _t2 = time.perf_counter()
            keep = torchvision.ops.batched_nms(
                boxes=torch.tensor(all_boxes, dtype=torch.float32),
                scores=torch.tensor(all_scores, dtype=torch.float32),
                idxs=torch.tensor(all_classes, dtype=torch.int64),
                iou_threshold=iou_threshold,
            )
            _t_nms = time.perf_counter() - _t2

            # ── Populate & log phase timings ──────────────────────────
            if _timings is not None:
                _timings.update({
                    "t_load_ms":       round(_t_load * 1000, 1),
                    "t_inference_ms":  round(_t_inference * 1000, 1),
                    "t_nms_ms":        round(_t_nms * 1000, 1),
                    "tiles_processed": _tiles_processed,
                    "tiles_skipped":   _tiles_skipped,
                })

            logging.info(
                f"[LATENCY] Raster load: {_t_load*1000:.1f} ms | "
                f"Tile inference: {_tiles_processed} processed, {_tiles_skipped} skipped early ({_t_inference*1000:.1f} ms) | "
                f"NMS: {_t_nms*1000:.1f} ms"
            )

            return [
                [
                    *all_boxes[i],
                    float(all_scores[i]),
                    int(all_classes[i]),
                ]
                for i in keep.tolist()
            ]

        except Exception as e:
            raise CustomException(e, sys)
