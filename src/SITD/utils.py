import os
import sys
import yaml
import json
from shapely.geometry import box
from src.SITD.exception import CustomException
from src.SITD.logger import logging

def save_yaml(file_path: str, data: dict) -> None:
    """
    Saves a dictionary to a YAML file.
    """
    try:
        logging.info(f"Saving YAML file to: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)
        logging.info(f"YAML file saved successfully at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_yaml(file_path: str) -> dict:
    """
    Loads a YAML file and returns its content as a dictionary.
    """
    try:
        logging.info(f"Loading YAML file from: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML file not found at {file_path}")
        with open(file_path, "r") as f:
            content = yaml.safe_load(f)
        logging.info(f"YAML file loaded successfully from: {file_path}")
        return content
    except Exception as e:
        raise CustomException(e, sys)

def save_json(file_path: str, data: dict) -> None:
    """
    Saves a dictionary to a JSON file.
    """
    try:
        logging.info(f"Saving JSON file to: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logging.info(f"JSON file saved successfully at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_json(file_path: str) -> dict:
    """
    Loads a JSON file.
    """
    try:
        logging.info(f"Loading JSON file from: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON file not found at {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        logging.info(f"JSON file loaded successfully from: {file_path}")
        return content
    except Exception as e:
        raise CustomException(e, sys)

def calculate_visibility(box_coords: list, chip_box: box) -> float:
    """
    Calculates the visibility (intersection area / object area) of a bounding box inside a chip.
    """
    try:
        obj_box = box(*box_coords)
        if obj_box.area <= 0:
            return 0.0
        
        intersection = obj_box.intersection(chip_box)
        if intersection.is_empty:
            return 0.0
            
        return intersection.area / obj_box.area
    except Exception as e:
        raise CustomException(e, sys)
