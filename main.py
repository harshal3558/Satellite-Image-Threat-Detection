"""
Entry point for the Satellite Image Threat Detection training pipeline.

Run from the project root:
    python main.py
"""

# from src.SITP.pipelines.training_pipeline import TrainingPipeline

# if __name__ == "__main__":
#     pipeline = TrainingPipeline()
#     best_weights = pipeline.run()
#     print(f"\nDone! Best model weights: {best_weights}")


from src.SITP.logger import logging
from src.SITP.exception import CustomException
from src.SITP.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.SITP.components.data_transformation import DataTransformation, DataTransformationConfig
from src.SITP.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.SITP.components.model_monitoring import ModelMonitoring

import sys
from pathlib import Path

if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        # --- Stage 1: Data Ingestion ---
        data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        ann_df, image_split, image_dir, output_dir = data_ingestion.initiate_data_ingestion()
        
        print("\n--- Data Ingestion Successful ---")
        print(f"Total annotations loaded: {len(ann_df)}")
        print(f"Total images split: {len(image_split)}")
        print(f"Image directory: {image_dir}")
        print(f"Output directory: {output_dir}")

        # --- Stage 2: Data Transformation ---
        data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_yaml_path, class_mapping = data_transformation.initiate_data_transformation(
            ann_df=ann_df,
            image_split=image_split,
            image_dir=image_dir,
            output_dir=output_dir
        )

        print("\n--- Data Transformation Successful ---")
        print(f"data.yaml path: {data_yaml_path}")
        print(f"Total classes mapped: {len(class_mapping)}")

        # --- Stage 3: Model Training ---
        logging.info("=" * 60)
        logging.info("STAGE 3: Model Training")
        logging.info("=" * 60)

        model_trainer_config = ModelTrainerConfig()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model, best_weights = model_trainer.initiate_model_trainer(
            data_yaml=Path(data_yaml_path),
            output_dir=Path(output_dir)
        )

        print("\n--- Model Training Successful ---")
        print(f"Best model weights saved at: {best_weights}")

        # --- Stage 4: Model Monitoring ---
        logging.info("=" * 60)
        logging.info("STAGE 4: Model Monitoring")
        logging.info("=" * 60)
        
        print("\n--- Running Model Performance Diagnostics ---")
        ModelMonitoring.inspect_per_class_performance(model)

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)