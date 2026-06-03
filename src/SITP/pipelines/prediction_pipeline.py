"""
Prediction pipeline for the Satellite Image Threat Detection project.

Provides:
  PredictPipeline  — runs tiled inference on a large GeoTIFF using the
                     trained YOLOv8 model.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        tile_size: int = 512,
        overlap: int = 100,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> None:
        _, _, output_dir = get_paths()
        if model_path:
            self.model_path = Path(model_path)
        elif Path("best.pt").exists():
            self.model_path = Path("best.pt")
        else:
            self.model_path = output_dir / "satellite_detector" / "weights" / "best.pt"

        self.tile_size = tile_size
        self.overlap = overlap
        self.conf = conf
        self.iou = iou

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {self.model_path} or root 'best.pt'. "
                "Run the training pipeline first."
            )

        logging.info(f"PredictPipeline initialised with model: {self.model_path}")

    def predict(self, image_path: str | Path) -> list[list[float]]:
        """
        Run tiled inference on a GeoTIFF and return global-coordinate detections.

        Parameters
        ----------
        image_path : Path to the ``.tif`` image to run inference on.

        Returns
        -------
        list[list[float]]
            Each element is ``[x1, y1, x2, y2, score, class_id]``.
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            logging.info(f"Running tiled inference on {image_path}")

            detections = ModelMonitoring.predict_large_image(
                image_path=image_path,
                model_path=self.model_path,
                tile_size=self.tile_size,
                overlap=self.overlap,
                conf_threshold=self.conf,
                iou_threshold=self.iou,
            )

            logging.info(f"Detected {len(detections)} objects in {image_path.name}")
            return detections

        except Exception as e:
            raise CustomException(e, sys)