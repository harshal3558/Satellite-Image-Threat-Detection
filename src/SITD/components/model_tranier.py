import os
import sys
from dataclasses import dataclass
from ultralytics import YOLO
from src.SITD.exception import CustomException
from src.SITD.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_dir: str
    model_name: str = 'yolov8m.pt'
    epochs: int = 50
    imgsz: int = 800
    batch: int = 8
    cache: bool = False
    workers: int = 2
    mosaic: float = 1.0
    copy_paste: float = 0.3
    degrees: float = 90.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    scale: float = 0.5
    translate: float = 0.1
    project: str = 'xview_yolo'
    name: str = 'satellite_detector'

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_trainer(self, data_yaml_path: str):
        """
        Loads the pre-trained model, trains it using configuration parameters,
        runs validation, and saves the trained weights.
        """
        logging.info("Starting model training process.")
        try:
            logging.info(f"Loading YOLO model: {self.config.model_name}")
            model = YOLO(self.config.model_name)

            logging.info("Initiating model training with the following parameters:")
            logging.info(f"Data YAML: {data_yaml_path}")
            logging.info(f"Epochs: {self.config.epochs}, Image Size: {self.config.imgsz}, Batch Size: {self.config.batch}")

            model.train(
                data=data_yaml_path,
                epochs=self.config.epochs,
                imgsz=self.config.imgsz,
                batch=self.config.batch,
                cache=self.config.cache,
                workers=self.config.workers,
                mosaic=self.config.mosaic,
                copy_paste=self.config.copy_paste,
                degrees=self.config.degrees,
                hsv_h=self.config.hsv_h,
                hsv_s=self.config.hsv_s,
                hsv_v=self.config.hsv_v,
                scale=self.config.scale,
                translate=self.config.translate,
                project=os.path.join(self.config.trained_model_dir, self.config.project),
                name=self.config.name
            )

            logging.info("Training complete. Validating model...")
            metrics = model.val()
            logging.info(f"Validation complete. MAP50: {metrics.box.map50}, MAP: {metrics.box.map}")

            # Exporting model
            logging.info("Exporting the model to ONNX format...")
            model.export(format='onnx')
            logging.info("Model exported successfully.")

            return os.path.join(
                self.config.trained_model_dir, 
                self.config.project, 
                self.config.name, 
                'weights', 
                'best.pt'
            )

        except Exception as e:
            raise CustomException(e, sys)
