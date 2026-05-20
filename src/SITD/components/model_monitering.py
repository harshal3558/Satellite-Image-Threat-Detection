import os
import sys
import glob
from dataclasses import dataclass
from src.SITD.exception import CustomException
from src.SITD.logger import logging

@dataclass
class ModelMonitoringConfig:
    data_dir: str

class ModelMonitoring:
    def __init__(self, config: ModelMonitoringConfig):
        self.config = config

    def monitor_dataset(self) -> dict:
        """
        Monitors the state of the generated training and validation datasets,
        summarizing labels and files.
        """
        logging.info("Starting dataset monitoring...")
        try:
            stats = {
                'train_images': 0,
                'train_labels': 0,
                'val_images': 0,
                'val_labels': 0,
                'total_detections': 0,
                'class_counts': {}
            }

            train_img_path = os.path.join(self.config.data_dir, 'images/train/*.jpg')
            train_lbl_path = os.path.join(self.config.data_dir, 'labels/train/*.txt')
            val_img_path = os.path.join(self.config.data_dir, 'images/val/*.jpg')
            val_lbl_path = os.path.join(self.config.data_dir, 'labels/val/*.txt')

            stats['train_images'] = len(glob.glob(train_img_path))
            stats['train_labels'] = len(glob.glob(train_lbl_path))
            stats['val_images'] = len(glob.glob(val_img_path))
            stats['val_labels'] = len(glob.glob(val_lbl_path))

            logging.info(f"Dataset stats: Train Images={stats['train_images']}, Val Images={stats['val_images']}")

            # Count total detections and classes
            for label_file in glob.glob(train_lbl_path) + glob.glob(val_lbl_path):
                if not os.path.exists(label_file):
                    continue
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            stats['total_detections'] += 1
                            cls_id = int(parts[0])
                            stats['class_counts'][cls_id] = stats['class_counts'].get(cls_id, 0) + 1

            logging.info(f"Total labeled objects across chips: {stats['total_detections']}")
            return stats

        except Exception as e:
            raise CustomException(e, sys)
