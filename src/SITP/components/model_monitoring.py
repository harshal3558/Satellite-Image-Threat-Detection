"""
Model Monitoring component for the Satellite Image Threat Detection pipeline.

Provides post-training inspection utilities:
  - Per-class mAP50 table
  - Results CSV reader (training curves)
  - Large-image tiled inference with batched NMS
"""

from __future__ import annotations

import sys
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
    # Tiled large-image inference
    # ------------------------------------------------------------------

    @staticmethod
    def predict_large_image(
        image_path: str | Path,
        model_path: str | Path,
        tile_size: int = 512,
        overlap: int = 100,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> list[list[float]]:
        """
        Run tiled inference on a large GeoTIFF and return merged detections.

        The image is divided into overlapping tiles; detections from all tiles
        are merged into global pixel coordinates and de-duplicated with batched
        NMS from ``torchvision``.

        Parameters
        ----------
        image_path      : Path to the GeoTIFF image.
        model_path      : Path to the trained ``.pt`` weights file.
        tile_size       : Side length of each tile in pixels.
        overlap         : Pixel overlap between adjacent tiles.
        conf_threshold  : Minimum confidence score to keep a detection.
        iou_threshold   : IoU threshold for NMS.

        Returns
        -------
        list[list[float]]
            Each entry is ``[x1, y1, x2, y2, score, class_id]`` in global
            pixel coordinates.
        """
        try:
            model = YOLO(str(model_path))
            stride = tile_size - overlap
            if stride <= 0:
                raise ValueError("overlap must be smaller than tile_size")

            all_boxes: list[list[float]] = []
            all_scores: list[float] = []
            all_classes: list[int] = []

            with rasterio.open(image_path) as src:
                width, height = src.width, src.height

                for y in tile_starts(height, tile_size, stride):
                    for x in tile_starts(width, tile_size, stride):
                        window = Window(x, y, tile_size, tile_size)
                        img = src.read(window=window)
                        if (
                            img.shape[1] != tile_size
                            or img.shape[2] != tile_size
                        ):
                            continue

                        img = np.transpose(img[:3], (1, 2, 0))
                        img = normalize_to_uint8(img)

                        # Skip empty / solid black background chips to speed up inference
                        if np.mean(img) < 2.0 and np.std(img) < 1.0:
                            continue

                        results = model.predict(
                            img,
                            imgsz=tile_size,
                            conf=conf_threshold,
                            verbose=False,
                        )

                        for result in results:
                            for pred_box in result.boxes:
                                xyxy = pred_box.xyxy[0].cpu().numpy()
                                conf = float(pred_box.conf[0].cpu().numpy())
                                cls = int(pred_box.cls[0].cpu().numpy())

                                all_boxes.append(
                                    [
                                        float(xyxy[0] + x),
                                        float(xyxy[1] + y),
                                        float(xyxy[2] + x),
                                        float(xyxy[3] + y),
                                    ]
                                )
                                all_scores.append(conf)
                                all_classes.append(cls)

            if not all_boxes:
                return []

            keep = torchvision.ops.batched_nms(
                boxes=torch.tensor(all_boxes, dtype=torch.float32),
                scores=torch.tensor(all_scores, dtype=torch.float32),
                idxs=torch.tensor(all_classes, dtype=torch.int64),
                iou_threshold=iou_threshold,
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
