import os
import sys
from src.SITD.exception import CustomException
from src.SITD.logger import logging
from src.SITD.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.SITD.components.data_transformation import DataTransformation, DataTransformationConfig
from src.SITD.components.model_tranier import ModelTrainer, ModelTrainerConfig
from src.SITD.components.model_monitering import ModelMonitoring, ModelMonitoringConfig

class TrainingPipeline:
    def __init__(self, raw_images_dir: str, geojson_path: str, output_dir: str, trained_model_dir: str):
        self.raw_images_dir = raw_images_dir
        self.geojson_path = geojson_path
        self.output_dir = output_dir
        self.trained_model_dir = trained_model_dir

    def run_pipeline(self):
        """
        Executes the full training pipeline sequentially.
        """
        logging.info("Training pipeline execution started.")
        try:
            # Step 1: Data Ingestion
            logging.info("Step 1: Running Data Ingestion...")
            ingestion_config = DataIngestionConfig(
                raw_images_dir=self.raw_images_dir,
                geojson_path=self.geojson_path,
                train_split_ratio=0.8
            )
            ingestion = DataIngestion(ingestion_config)
            train_images, val_images, ann_df = ingestion.initiate_data_ingestion()
            logging.info("Data Ingestion step completed successfully.")

            # Step 2: Data Transformation (Chipping)
            logging.info("Step 2: Running Data Transformation...")
            transformation_config = DataTransformationConfig(
                output_dir=self.output_dir,
                chip_size=800,
                stride=640,
                visibility_threshold=0.4
            )
            transformation = DataTransformation(transformation_config)
            data_yaml_path = transformation.initiate_data_transformation(
                train_images=train_images,
                val_images=val_images,
                ann_df=ann_df,
                raw_images_dir=self.raw_images_dir
            )
            logging.info("Data Transformation step completed successfully.")

            # Step 3: Dataset Monitoring
            logging.info("Step 3: Running Dataset Monitoring...")
            monitoring_config = ModelMonitoringConfig(data_dir=self.output_dir)
            monitoring = ModelMonitoring(monitoring_config)
            dataset_stats = monitoring.monitor_dataset()
            logging.info(f"Dataset stats: {dataset_stats}")
            logging.info("Dataset Monitoring step completed successfully.")

            # Step 4: Model Trainer
            logging.info("Step 4: Running Model Training...")
            trainer_config = ModelTrainerConfig(
                trained_model_dir=self.trained_model_dir,
                model_name='yolov8m.pt',
                epochs=50,
                imgsz=800,
                batch=8
            )
            trainer = ModelTrainer(trainer_config)
            best_model_path = trainer.initiate_model_trainer(data_yaml_path)
            logging.info("Model Training step completed successfully.")

            logging.info(f"Training pipeline run complete. Best model weights: {best_model_path}")
            return best_model_path

        except Exception as e:
            logging.error("Exception occurred during training pipeline run.")
            raise CustomException(e, sys)
