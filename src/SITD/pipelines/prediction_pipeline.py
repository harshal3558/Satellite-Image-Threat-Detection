import os
import sys
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torchvision
from ultralytics import YOLO
from src.SITD.exception import CustomException
from src.SITD.logger import logging

class PredictionPipeline:
    def __init__(self, model_path: str):
        """
        Initializes the prediction pipeline with a trained YOLO model.
        """
        self.model_path = model_path
        try:
            logging.info(f"Loading YOLOv8 model for inference from: {model_path}")
            self.model = YOLO(model_path)
        except Exception as e:
            raise CustomException(e, sys)

    def predict_large_image(self, image_path: str, tile_size: int = 800, overlap: int = 100, iou_threshold: float = 0.45, conf_threshold: float = 0.25) -> list:
        """
        Runs sliding-window inference on a large satellite image,
        translates detections to global pixel coordinates, and applies global NMS.

        Args:
            image_path (str): Path to the large image (e.g. .tif).
            tile_size (int): Dimensions of the sliding-window tiles.
            overlap (int): Overlap between successive tiles.
            iou_threshold (float): IOU threshold for Non-Maximum Suppression.
            conf_threshold (float): Confidence threshold to keep detections.

        Returns:
            list: List of dicts representing final predictions:
                  [{'box': [x1, y1, x2, y2], 'confidence': float, 'class_id': int}]
        """
        logging.info(f"Starting large image inference for: {image_path}")
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at path: {image_path}")

            stride = tile_size - overlap
            all_boxes = []
            all_scores = []
            all_classes = []

            with rasterio.open(image_path) as src:
                width = src.width
                height = src.height
                logging.info(f"Image dimensions: Width={width}, Height={height}")

                for y in range(0, height, stride):
                    for x in range(0, width, stride):
                        # Calculate current tile dimensions (handle boundaries)
                        w = min(tile_size, width - x)
                        h = min(tile_size, height - y)

                        # Skip tiny fragments
                        if w < 10 or h < 10:
                            continue

                        window = Window(x, y, w, h)
                        # Read the RGB bands
                        img = src.read([1, 2, 3], window=window)
                        # Shape format: CHW to HWC
                        img = np.transpose(img, (1, 2, 0))

                        # Run inference on tile
                        results = self.model.predict(img, imgsz=tile_size, conf=conf_threshold, verbose=False)

                        for r in results:
                            for box_obj in r.boxes:
                                b = box_obj.xyxy[0].cpu().numpy()
                                conf = box_obj.conf[0].cpu().numpy()
                                cls = box_obj.cls[0].cpu().numpy()

                                # Translate local tile coordinates to global coordinates
                                gx1, gy1 = b[0] + x, b[1] + y
                                gx2, gy2 = b[2] + x, b[3] + y

                                all_boxes.append([gx1, gy1, gx2, gy2])
                                all_scores.append(conf)
                                all_classes.append(cls)

            if not all_boxes:
                logging.info("No objects detected in the entire image.")
                return []

            # --- Global Batched Non-Maximum Suppression (NMS) ---
            logging.info(f"Applying Global NMS on {len(all_boxes)} raw detections...")
            boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
            classes_tensor = torch.tensor(all_classes, dtype=torch.float32)

            keep_indices = torchvision.ops.batched_nms(
                boxes=boxes_tensor,
                scores=scores_tensor,
                idxs=classes_tensor,
                iou_threshold=iou_threshold
            )

            final_predictions = []
            for idx in keep_indices:
                i = idx.item()
                final_predictions.append({
                    'box': [
                        float(all_boxes[i][0]),
                        float(all_boxes[i][1]),
                        float(all_boxes[i][2]),
                        float(all_boxes[i][3])
                    ],
                    'confidence': float(all_scores[i]),
                    'class_id': int(all_classes[i])
                })

            logging.info(f"Global NMS complete. Detections reduced from {len(all_boxes)} to {len(final_predictions)}.")
            return final_predictions

        except Exception as e:
            logging.error(f"Error during prediction on large image: {str(e)}")
            raise CustomException(e, sys)
