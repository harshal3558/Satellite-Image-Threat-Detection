import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.SITD.exception import CustomException
from src.SITD.logger import logging
from src.SITD.utils import load_json

@dataclass
class DataIngestionConfig:
    raw_images_dir: str
    geojson_path: str
    train_split_ratio: float = 0.8

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        """
        Loads geojson annotation data, parses boxes and image IDs,
        and splits unique images into train and validation sets.
        """
        logging.info("Starting data ingestion process.")
        try:
            if not os.path.exists(self.config.geojson_path):
                raise FileNotFoundError(f"GeoJSON label file not found at: {self.config.geojson_path}")
            if not os.path.exists(self.config.raw_images_dir):
                raise FileNotFoundError(f"Raw images directory not found at: {self.config.raw_images_dir}")

            logging.info("Loading and parsing GeoJSON annotation file...")
            data = load_json(self.config.geojson_path)
            features = data.get('features', [])

            annotations = []
            for feat in features:
                props = feat.get('properties', {})
                image_id = props.get('image_id')
                type_id = props.get('type_id')
                coords = props.get('bounds_imcoords')

                # Skip empty coordinates
                if not coords or coords == '':
                    continue

                try:
                    x1, y1, x2, y2 = map(int, coords.split(','))
                    annotations.append([image_id, type_id, x1, y1, x2, y2])
                except ValueError:
                    # Skip malformed coordinates
                    continue

            ann_df = pd.DataFrame(
                annotations,
                columns=['image_id', 'type_id', 'x1', 'y1', 'x2', 'y2']
            )

            logging.info(f"Loaded {len(ann_df)} annotations from GeoJSON.")

            # Filter annotations to keep only those with images that exist on disk
            unique_images = ann_df['image_id'].unique()
            existing_images = []
            for img_name in unique_images:
                if os.path.exists(os.path.join(self.config.raw_images_dir, img_name)):
                    existing_images.append(img_name)
            
            logging.info(f"Found {len(existing_images)} existing images out of {len(unique_images)} unique images in annotation file.")
            
            # Split unique images
            train_split = int(self.config.train_split_ratio * len(existing_images))
            train_images = existing_images[:train_split]
            val_images = existing_images[train_split:]

            logging.info(f"Split details: {len(train_images)} train images, {len(val_images)} validation images.")
            
            # Filter annotation dataframe for existing images
            filtered_ann_df = ann_df[ann_df['image_id'].isin(existing_images)].reset_index(drop=True)

            return train_images, val_images, filtered_ann_df

        except Exception as e:
            raise CustomException(e, sys)
