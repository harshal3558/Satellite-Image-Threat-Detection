"""
Prediction pipeline for the Satellite Image Threat Detection project.

Provides:
  PredictPipeline  — runs tiled inference on a large GeoTIFF using the
                     trained YOLOv8 model.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ultralytics import YOLO

from src.SITP.components.model_monitoring import ModelMonitoring
from src.SITP.exception import CustomException
from src.SITP.logger import logging
from src.SITP.utils import get_paths


class PredictPipeline:
    """
    Tiled inference pipeline for large satellite GeoTIFF images.

    Parameters
    ----------
    model_path : path to the trained ``.pt`` weights file.
                 Defaults to ``<output_dir>/satellite_detector/weights/best.pt``.
    tile_size  : chip size used during inference (should match training imgsz).
    overlap    : pixel overlap between adjacent inference tiles.
    conf       : confidence threshold.
    iou        : IoU threshold for NMS.
    batch_size : batch size for tile forward passes.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        tile_size: int = 512,
        overlap: int = 100,
        conf: float = 0.25,
        iou: float = 0.45,
        batch_size: int = 4,
    ) -> None:
        _, _, output_dir = get_paths()
        if model_path:
            self.model_path = Path(model_path)
        elif Path("best.onnx").exists():
            self.model_path = Path("best.onnx")
        elif Path("best.pt").exists():
            self.model_path = Path("best.pt")
        else:
            self.model_path = output_dir / "satellite_detector" / "weights" / "best.pt"

        self.tile_size = tile_size
        self.overlap = overlap
        self.conf = conf
        self.iou = iou
        self.batch_size = batch_size

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {self.model_path}, 'best.onnx', or 'best.pt'. "
                "Run the training pipeline first."
            )

        # Load and keep model warm in memory (ONNX Runtime or PyTorch)
        model_str = str(self.model_path)
        if model_str.endswith(".onnx"):
            self.model = YOLO(model_str, task="detect")
            self.engine_type = "ONNX Runtime"
        else:
            self.model = YOLO(model_str)
            self.engine_type = "PyTorch"
        logging.info(
            f"PredictPipeline initialized with warm {self.engine_type} model: {self.model_path}"
        )

    def predict(
        self,
        image_path: str | Path,
        conf: float | None = None,
        iou: float | None = None,
        batch_size: int | None = None,
        _timings: dict | None = None,
    ) -> list[list[float]]:
        """
        Run tiled inference on a GeoTIFF and return global-coordinate detections.

        Parameters
        ----------
        image_path : Path to the ``.tif`` image to run inference on.
        conf       : Optional override for confidence threshold.
        iou        : Optional override for IoU threshold.
        batch_size : Optional override for batch size.
        _timings   : Optional dict; if provided it will be populated with
                     per-phase latency keys from ModelMonitoring.

        Returns
        -------
        list[list[float]]
            Each element is ``[x1, y1, x2, y2, score, class_id]``.
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            conf_val = conf if conf is not None else self.conf
            iou_val = iou if iou is not None else self.iou
            batch_val = batch_size if batch_size is not None else self.batch_size

            logging.info(
                f"Running batched tiled inference on {image_path.name} "
                f"(batch={batch_val}, conf={conf_val}, iou={iou_val})"
            )

            detections = ModelMonitoring.predict_large_image(
                image_path=image_path,
                model_path=self.model,
                tile_size=self.tile_size,
                overlap=self.overlap,
                conf_threshold=conf_val,
                iou_threshold=iou_val,
                batch_size=batch_val,
                _timings=_timings,
            )

            logging.info(f"Detected {len(detections)} objects in {image_path.name}")
            return detections

        except Exception as e:
            raise CustomException(e, sys)