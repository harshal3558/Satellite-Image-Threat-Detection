import os
import sys
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
from dataclasses import dataclass
from tqdm import tqdm
from src.SITD.exception import CustomException
from src.SITD.logger import logging
from src.SITD.utils import save_yaml, calculate_visibility

@dataclass
class DataTransformationConfig:
    output_dir: str
    chip_size: int = 800
    stride: int = 640
    visibility_threshold: float = 0.4
    xview_names: list = None

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        if self.config.xview_names is None:
            self.config.xview_names = [
                'Fixed-wing Aircraft', 'Small Aircraft', 'Cargo Plane', 'Helicopter',
                'Passenger Vehicle', 'Small Car', 'Bus', 'Pickup Truck',
                'Utility Truck', 'Truck', 'Cargo Truck', 'Truck w/Box',
                'Truck Tractor', 'Trailer', 'Truck w/Flatbed',
                'Truck w/Liquid', 'Crane Truck', 'Railway Vehicle',
                'Passenger Car', 'Cargo Car', 'Flat Car', 'Tank car',
                'Locomotive', 'Maritime Vessel', 'Motorboat',
                'Sailboat', 'Tugboat', 'Barge', 'Fishing Vessel',
                'Ferry', 'Yacht', 'Container Ship', 'Oil Tanker',
                'Engineering Vehicle', 'Tower crane', 'Container Crane',
                'Reach Stacker', 'Straddle Carrier', 'Mobile Crane',
                'Dump Truck', 'Haul Truck', 'Scraper/Tractor',
                'Front loader/Bulldozer', 'Excavator', 'Cement Mixer',
                'Ground Grader', 'Hut/Tent', 'Shed', 'Building',
                'Aircraft Hangar', 'Damaged Building', 'Facility',
                'Construction Site', 'Vehicle Lot', 'Helipad',
                'Storage Tank', 'Shipping container lot',
                'Shipping Container', 'Pylon', 'Tower', 
                'Class 60 Placeholder', 'Class 61 Placeholder'
            ]

    def initiate_data_transformation(self, train_images, val_images, ann_df, raw_images_dir):
        """
        Processes images in train_images and val_images by splitting them into chips,
        mapping annotations, and writing the final dataset to YOLOv8 format.
        """
        logging.info("Starting data transformation process.")
        try:
            # 1. Setup output directories
            dirs = [
                os.path.join(self.config.output_dir, 'images/train'),
                os.path.join(self.config.output_dir, 'images/val'),
                os.path.join(self.config.output_dir, 'labels/train'),
                os.path.join(self.config.output_dir, 'labels/val')
            ]
            for d in dirs:
                os.makedirs(d, exist_ok=True)

            # 2. Build class mapping dynamically
            unique_types = sorted(ann_df['type_id'].unique())
            class_mapping = {type_id: idx for idx, type_id in enumerate(unique_types)}
            logging.info(f"Class mapping successfully established with {len(class_mapping)} classes.")

            # 3. Perform Chipping
            chip_count = 0
            
            # Map of splits to the image lists
            splits = {
                'train': train_images,
                'val': val_images
            }

            for split_name, images in splits.items():
                logging.info(f"Chipping images for split: {split_name}")
                for image_name in tqdm(images):
                    image_path = os.path.join(raw_images_dir, image_name)
                    if not os.path.exists(image_path):
                        continue

                    # Filter annotations for this image
                    image_annotations = ann_df[ann_df['image_id'] == image_name]

                    with rasterio.open(image_path) as src:
                        width = src.width
                        height = src.height

                        # Sliding window chipping
                        for y in range(0, height - self.config.chip_size, self.config.stride):
                            for x in range(0, width - self.config.chip_size, self.config.stride):
                                window = Window(x, y, self.config.chip_size, self.config.chip_size)
                                chip = src.read(window=window)
                                
                                # Skip tiny/boundary incomplete chips
                                if chip.shape[1] < self.config.chip_size or chip.shape[2] < self.config.chip_size:
                                    continue
                                    
                                # Convert shape CHW to HWC
                                chip = np.transpose(chip, (1, 2, 0))
                                chip_boxes = []
                                chip_polygon = box(x, y, x + self.config.chip_size, y + self.config.chip_size)

                                for _, row in image_annotations.iterrows():
                                    visibility = calculate_visibility(
                                        [row.x1, row.y1, row.x2, row.y2], 
                                        chip_polygon
                                    )
                                    
                                    if visibility < self.config.visibility_threshold:
                                        continue

                                    # Local coordinates relative to the chip
                                    nx1 = max(0, row.x1 - x)
                                    ny1 = max(0, row.y1 - y)
                                    nx2 = min(self.config.chip_size, row.x2 - x)
                                    ny2 = min(self.config.chip_size, row.y2 - y)
                                    
                                    # Convert to normalized YOLO format (0-1)
                                    dw = 1.0 / self.config.chip_size
                                    dh = 1.0 / self.config.chip_size
                                    x_center = ((nx1 + nx2) / 2.0) * dw
                                    y_center = ((ny1 + ny2) / 2.0) * dh
                                    w = (nx2 - nx1) * dw
                                    h = (ny2 - ny1) * dh
                                    
                                    class_idx = class_mapping[row.type_id]
                                    chip_boxes.append([class_idx, x_center, y_center, w, h])

                                # If chip contains objects, save it
                                if len(chip_boxes) > 0:
                                    chip_filename = f"chip_{chip_count}"
                                    
                                    # Save Image
                                    img_out_path = os.path.join(
                                        self.config.output_dir, 
                                        f'images/{split_name}', 
                                        f"{chip_filename}.jpg"
                                    )
                                    # Convert RGB to BGR for cv2
                                    cv2.imwrite(img_out_path, cv2.cvtColor(chip, cv2.COLOR_RGB2BGR))
                                    
                                    # Save Labels (.txt)
                                    lbl_out_path = os.path.join(
                                        self.config.output_dir, 
                                        f'labels/{split_name}', 
                                        f"{chip_filename}.txt"
                                    )
                                    with open(lbl_out_path, 'w') as f:
                                        for b in chip_boxes:
                                            f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
                                    
                                    chip_count += 1

            logging.info(f"Chipping complete. Total chips generated: {chip_count}")

            # 4. Generate and save data.yaml config file
            # Limit the names to nc classes
            num_classes = max(62, len(class_mapping))
            names_dict = {i: name for i, name in enumerate(self.config.xview_names[:num_classes])}
            
            # Pad if less names provided
            if len(names_dict) < num_classes:
                for i in range(len(names_dict), num_classes):
                    names_dict[i] = f"Class_{i}_Placeholder"

            data_yaml = {
                'path': os.path.abspath(self.config.output_dir),
                'train': 'images/train',
                'val': 'images/val',
                'nc': num_classes,
                'names': names_dict
            }

            yaml_path = os.path.join(self.config.output_dir, 'data.yaml')
            save_yaml(yaml_path, data_yaml)

            return yaml_path

        except Exception as e:
            raise CustomException(e, sys)
