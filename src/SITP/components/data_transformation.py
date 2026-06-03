"""
Data Transformation component for the Satellite Image Threat Detection pipeline.

Converts GeoTIFF images + annotation DataFrame into YOLO-format image chips,
applying sliding-window tiling, per-class label mapping, optional albumentations
augmentation, and a YAML dataset descriptor — exactly as in the notebook.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.windows import Window
from shapely.geometry import box
from tqdm import tqdm

from src.SITP.exception import CustomException
from src.SITP.logger import logging
from src.SITP.utils import normalize_to_uint8, prepare_dirs, save_chip, tile_starts


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DataTransformationConfig:
    """Hyper-parameters that control the chipping / augmentation process."""
    chip_size: int = 512
    stride: int = 364
    visibility_threshold: float = 0.4
    background_keep_prob: float = 0.08
    use_augmentation: bool = True
    seed: int = 42


# ---------------------------------------------------------------------------
# Component
# ---------------------------------------------------------------------------

class DataTransformation:
    """
    Slices GeoTIFF images into fixed-size chips, writes YOLO labels, and
    produces a ``data.yaml`` that YOLOv8 can consume directly.
    """

    def __init__(self, config: DataTransformationConfig | None = None) -> None:
        self.config = config or DataTransformationConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initiate_data_transformation(
        self,
        ann_df: pd.DataFrame,
        image_split: dict[str, str],
        image_dir: Path,
        output_dir: Path,
    ) -> tuple[Path, dict[int, int]]:
        """
        Run the full chipping pipeline.

        Parameters
        ----------
        ann_df       : Annotation DataFrame (from DataIngestion).
        image_split  : {image_id: "train" | "val"} mapping.
        image_dir    : Directory containing ``.tif`` source images.
        output_dir   : Root directory for YOLO chip output.

        Returns
        -------
        tuple[Path, dict[int, int]]
            (data_yaml_path, class_mapping)
        """
        try:
            prepare_dirs(output_dir)

            class_mapping = self._build_class_mapping(ann_df)
            logging.info(f"Class mapping built: {len(class_mapping)} classes.")

            split_counts = self._create_yolo_chips(
                image_dir=image_dir,
                ann_df=ann_df,
                output_dir=output_dir,
                class_mapping=class_mapping,
                image_split=image_split,
            )
            logging.info(f"Chip counts: {dict(split_counts)}")

            data_yaml_path = self._write_data_yaml(output_dir, class_mapping)
            logging.info(f"data.yaml written to {data_yaml_path}")

            return data_yaml_path, class_mapping

        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Class mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _build_class_mapping(ann_df: pd.DataFrame) -> dict[int, int]:
        """Map raw xView type_ids to contiguous 0-based YOLO class indices."""
        type_ids = sorted(int(t) for t in ann_df["type_id"].unique())
        return {int(t): int(i) for i, t in enumerate(type_ids)}

    # ------------------------------------------------------------------
    # Chipping
    # ------------------------------------------------------------------

    def _make_transform(self) -> A.Compose:
        """Build the albumentations augmentation pipeline (train only)."""
        return A.Compose(
            [
                A.CLAHE(p=0.35),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.35),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.2,
            ),
        )

    @staticmethod
    def _calculate_visibility(
        box_coords: tuple[int, int, int, int], chip_polygon
    ) -> float:
        """Fraction of an annotation box that falls inside a chip polygon."""
        obj_box = box(*box_coords)
        if obj_box.area <= 0:
            return 0.0
        intersection = obj_box.intersection(chip_polygon)
        if intersection.is_empty:
            return 0.0
        return float(intersection.area / obj_box.area)

    def _create_yolo_chips(
        self,
        image_dir: Path,
        ann_df: pd.DataFrame,
        output_dir: Path,
        class_mapping: dict[int, int],
        image_split: dict[str, str],
    ) -> Counter:
        """
        Slide a window over every GeoTIFF, extract chips, convert annotations
        to YOLO format, optionally augment train chips, and write to disk.
        """
        cfg = self.config
        transform = self._make_transform()
        chip_count = 0
        split_counts: Counter = Counter()

        for image_name in tqdm(
            sorted(ann_df["image_id"].unique()), desc="Creating chips"
        ):
            image_path = image_dir / image_name
            if not image_path.exists():
                logging.warning(f"Skipping missing image: {image_path}")
                continue

            split = image_split[image_name]
            image_annotations = ann_df[ann_df["image_id"] == image_name]

            with rasterio.open(image_path) as src:
                width, height = src.width, src.height
                x_starts = tile_starts(width, cfg.chip_size, cfg.stride)
                y_starts = tile_starts(height, cfg.chip_size, cfg.stride)

                for y in y_starts:
                    for x in x_starts:
                        window = Window(x, y, cfg.chip_size, cfg.chip_size)
                        chip = src.read(window=window)
                        if (
                            chip.shape[1] != cfg.chip_size
                            or chip.shape[2] != cfg.chip_size
                        ):
                            continue

                        chip = np.transpose(chip[:3], (1, 2, 0))
                        chip = normalize_to_uint8(chip)
                        chip_polygon = box(
                            x, y, x + cfg.chip_size, y + cfg.chip_size
                        )
                        boxes_yolo: list[list[float]] = []

                        for row in image_annotations.itertuples(index=False):
                            visibility = self._calculate_visibility(
                                (row.x1, row.y1, row.x2, row.y2), chip_polygon
                            )
                            if visibility < cfg.visibility_threshold:
                                continue

                            nx1 = max(0, row.x1 - x)
                            ny1 = max(0, row.y1 - y)
                            nx2 = min(cfg.chip_size, row.x2 - x)
                            ny2 = min(cfg.chip_size, row.y2 - y)

                            bw, bh = nx2 - nx1, ny2 - ny1
                            if bw <= 1 or bh <= 1:
                                continue

                            boxes_yolo.append(
                                [
                                    class_mapping[row.type_id],
                                    ((nx1 + nx2) / 2.0) / cfg.chip_size,
                                    ((ny1 + ny2) / 2.0) / cfg.chip_size,
                                    bw / cfg.chip_size,
                                    bh / cfg.chip_size,
                                ]
                            )

                        keep_background = (
                            not boxes_yolo
                            and random.random() < cfg.background_keep_prob
                        )
                        if not boxes_yolo and not keep_background:
                            continue

                        # Augment training chips that have labels
                        if (
                            cfg.use_augmentation
                            and split == "train"
                            and boxes_yolo
                        ):
                            class_labels = [int(b[0]) for b in boxes_yolo]
                            bboxes = [b[1:] for b in boxes_yolo]
                            augmented = transform(
                                image=chip,
                                bboxes=bboxes,
                                class_labels=class_labels,
                            )
                            chip = augmented["image"]
                            boxes_yolo = [
                                [cls, *bbox]
                                for cls, bbox in zip(
                                    augmented["class_labels"],
                                    augmented["bboxes"],
                                )
                            ]
                            if not boxes_yolo:
                                continue

                        chip_name = (
                            f"{Path(image_name).stem}_{x}_{y}_{chip_count}"
                        )
                        save_chip(chip, boxes_yolo, output_dir, split, chip_name)
                        chip_count += 1
                        split_counts[split] += 1

        return split_counts

    # ------------------------------------------------------------------
    # YAML writer
    # ------------------------------------------------------------------

    @staticmethod
    def _write_data_yaml(
        output_dir: Path, class_mapping: dict[int, int]
    ) -> Path:
        """Write ``data.yaml`` and ``class_mapping.json`` to *output_dir*."""
        clean_mapping = {
            int(type_id): int(idx) for type_id, idx in class_mapping.items()
        }
        names = {
            idx: f"xview_type_{type_id}"
            for type_id, idx in clean_mapping.items()
        }

        data_yaml = {
            "path": str(output_dir),
            "train": "images/train",
            "val": "images/val",
            "nc": len(names),
            "names": names,
        }

        yaml_path = output_dir / "data.yaml"
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data_yaml, f, sort_keys=False)

        mapping_path = output_dir / "class_mapping.json"
        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "type_id_to_yolo_index": clean_mapping,
                    "yolo_index_to_type_id": {
                        idx: type_id
                        for type_id, idx in clean_mapping.items()
                    },
                },
                f,
                indent=2,
            )

        return yaml_path