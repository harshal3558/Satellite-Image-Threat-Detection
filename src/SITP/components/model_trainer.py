"""
Model Trainer component for the Satellite Image Threat Detection pipeline.

Trains a YOLOv8 model on the prepared YOLO chips and validates the best
checkpoint — directly mirroring the notebook's train_model / validate_model
functions.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

from src.SITP.exception import CustomException
from src.SITP.logger import logging


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModelTrainerConfig:
    """Training hyper-parameters (mirrors the notebook defaults)."""
    model_weights: str = "yolov8m.pt"      # pretrained weights to start from
    epochs: int = 50
    imgsz: int = 512                        # must match chip_size
    batch: int = 8
    workers: int = 2
    optimizer: str = "auto"
    seed: int = 42
    # Augmentation params (passed to model.train)
    mosaic: float = 1.0
    copy_paste: float = 0.2
    degrees: float = 90.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    scale: float = 0.5
    translate: float = 0.1
    run_name: str = "satellite_detector"


# ---------------------------------------------------------------------------
# Component
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Wraps YOLOv8 training and validation.

    Usage
    -----
    trainer = ModelTrainer()
    model, best_weights = trainer.initiate_model_trainer(data_yaml, output_dir)
    """

    def __init__(self, config: ModelTrainerConfig | None = None) -> None:
        self.config = config or ModelTrainerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initiate_model_trainer(
        self,
        data_yaml: Path,
        output_dir: Path,
    ) -> tuple[YOLO, Path]:
        """
        Train YOLOv8 then validate the best checkpoint.

        Parameters
        ----------
        data_yaml   : Path to the ``data.yaml`` written by DataTransformation.
        output_dir  : Root YOLO output directory (project root for runs).

        Returns
        -------
        tuple[YOLO, Path]
            (trained_model, path_to_best_weights)
        """
        try:
            cfg = self.config

            logging.info(
                f"Starting YOLOv8 training: weights={cfg.model_weights}, "
                f"epochs={cfg.epochs}, imgsz={cfg.imgsz}, batch={cfg.batch}"
            )

            model = self._train_model(data_yaml, output_dir)
            self._validate_model(model)

            # Retrieve actual save directory dynamically to handle auto-incremented names
            if hasattr(model, "trainer") and model.trainer is not None and hasattr(model.trainer, "save_dir"):
                save_dir = Path(model.trainer.save_dir)
                best_weights = save_dir / "weights" / "best.pt"
            else:
                best_weights = output_dir / cfg.run_name / "weights" / "best.pt"

            # Scan output directory for the latest incremented name if the direct check failed
            if not best_weights.exists():
                matching_dirs = list(output_dir.glob(f"{cfg.run_name}*"))
                if matching_dirs:
                    latest_dir = max(matching_dirs, key=lambda p: p.stat().st_mtime)
                    best_weights = latest_dir / "weights" / "best.pt"

            if best_weights.exists():
                logging.info(f"Best model weights at: {best_weights}")
            else:
                logging.warning(
                    f"best.pt not found at expected path: {best_weights}. "
                    "Check if training completed successfully."
                )

            return model, best_weights

        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_model(self, data_yaml: Path, output_dir: Path) -> YOLO:
        """Instantiate and train the YOLOv8 model."""
        cfg = self.config
        model = YOLO(cfg.model_weights)
        model.train(
            data=str(data_yaml),
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            batch=cfg.batch,
            workers=cfg.workers,
            cache=False,
            optimizer=cfg.optimizer,
            mosaic=cfg.mosaic,
            copy_paste=cfg.copy_paste,
            degrees=cfg.degrees,
            hsv_h=cfg.hsv_h,
            hsv_s=cfg.hsv_s,
            hsv_v=cfg.hsv_v,
            scale=cfg.scale,
            translate=cfg.translate,
            project=str(output_dir),
            name=cfg.run_name,
            seed=cfg.seed,
        )
        return model

    @staticmethod
    def _validate_model(model: YOLO) -> None:
        """Run validation and log summary metrics."""
        metrics = model.val()
        logging.info(f"mAP50    : {metrics.box.map50:.4f}")
        logging.info(f"mAP50-95 : {metrics.box.map:.4f}")

    # ------------------------------------------------------------------
    # Note: per-class inspection is available via ModelMonitoring.
    # ------------------------------------------------------------------