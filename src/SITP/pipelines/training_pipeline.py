"""
Training pipeline for the Satellite Image Threat Detection project.

Orchestrates:
  1. DataIngestion   — load xView annotations & build train/val image split
  2. DataTransformation — chip GeoTIFFs into YOLO-format image + label files
  3. ModelTrainer    — train YOLOv8 and validate the best checkpoint
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.SITP.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.SITP.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)
from src.SITP.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.SITP.exception import CustomException
from src.SITP.logger import logging
from src.SITP.utils import print_device_info, seed_everything


class TrainingPipeline:
    """
    End-to-end training pipeline that mirrors the notebook's
    ``Run Full Pipeline`` cell.
    """

    def __init__(
        self,
        ingestion_config: DataIngestionConfig | None = None,
        transformation_config: DataTransformationConfig | None = None,
        trainer_config: ModelTrainerConfig | None = None,
    ) -> None:
        self.ingestion_config = ingestion_config or DataIngestionConfig()
        self.transformation_config = transformation_config or DataTransformationConfig()
        self.trainer_config = trainer_config or ModelTrainerConfig()

    def run(self) -> Path:
        """
        Execute the full pipeline.

        Returns
        -------
        Path
            Path to the best YOLOv8 weight file (``best.pt``).
        """
        try:
            seed_everything(self.ingestion_config.seed)
            print_device_info()

            # ── Stage 1: Data Ingestion ──────────────────────────────────
            logging.info("=" * 60)
            logging.info("STAGE 1: Data Ingestion")
            logging.info("=" * 60)

            ingestion = DataIngestion(self.ingestion_config)
            ann_df, image_split, image_dir, output_dir = (
                ingestion.initiate_data_ingestion()
            )

            logging.info(
                f"Loaded {len(ann_df):,} annotations from "
                f"{ann_df['image_id'].nunique():,} images."
            )

            # ── Stage 2: Data Transformation (chipping) ──────────────────
            logging.info("=" * 60)
            logging.info("STAGE 2: Data Transformation")
            logging.info("=" * 60)

            transformation = DataTransformation(self.transformation_config)
            data_yaml, class_mapping = transformation.initiate_data_transformation(
                ann_df=ann_df,
                image_split=image_split,
                image_dir=image_dir,
                output_dir=output_dir,
            )

            # ── Stage 3: Model Training ───────────────────────────────────
            logging.info("=" * 60)
            logging.info("STAGE 3: Model Training")
            logging.info("=" * 60)

            trainer = ModelTrainer(self.trainer_config)
            model, best_weights = trainer.initiate_model_trainer(
                data_yaml=data_yaml,
                output_dir=output_dir,
            )

            logging.info("Training pipeline completed successfully.")
            logging.info(f"Best weights: {best_weights}")
            print(f"\nBest model weights saved at: {best_weights}")

            return best_weights

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
