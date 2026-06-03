"""
Data Ingestion component for the Satellite Image Threat Detection pipeline.

Loads the xView GeoJSON annotations and the GeoTIFF image directory, then
performs a reproducible train / val split at the *image* level (not chip
level).  All downstream work (chipping, augmentation) happens in
DataTransformation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.SITP.exception import CustomException
from src.SITP.logger import logging
from src.SITP.utils import get_paths, seed_everything

import sys
import random


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DataIngestionConfig:
    """Paths produced by the ingestion step."""
    # Where to persist the raw annotation DataFrame (optional, for caching)
    raw_annotations_path: str = str(Path("artifacts") / "raw_annotations.csv")
    # Fraction of images held out for validation
    val_fraction: float = 0.2
    # Random seed for the image-level split
    seed: int = 42


# ---------------------------------------------------------------------------
# Component
# ---------------------------------------------------------------------------

class DataIngestion:
    """
    Reads xView GeoJSON labels + GeoTIFF images and produces an annotation
    DataFrame together with a per-image train/val assignment dict.
    """

    def __init__(self, config: DataIngestionConfig | None = None) -> None:
        self.config = config or DataIngestionConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initiate_data_ingestion(
        self,
    ) -> tuple[pd.DataFrame, dict[str, str], Path, Path]:
        """
        Run the full ingestion workflow.

        Returns
        -------
        tuple
            ann_df        – DataFrame with columns [image_id, type_id, x1, y1, x2, y2]
            image_split   – {image_id: "train" | "val"}
            image_dir     – Path to the GeoTIFF directory
            output_dir    – Root path for YOLO chip output
        """
        try:
            seed_everything(self.config.seed)
            image_dir, label_path, output_dir = get_paths()

            logging.info(f"Image directory : {image_dir}")
            logging.info(f"Label path      : {label_path}")
            logging.info(f"Output directory: {output_dir}")

            # 1. Load annotations
            ann_df = self._load_annotations(label_path)
            logging.info(
                f"Loaded {len(ann_df):,} valid annotations "
                f"from {ann_df['image_id'].nunique():,} images."
            )

            # 2. Remove annotations whose image is missing from disk
            ann_df = self._filter_existing_images(ann_df, image_dir)

            # 3. Image-level train / val split
            image_split = self._split_images(ann_df)

            # 4. Persist raw annotations (optional cache)
            Path(self.config.raw_annotations_path).parent.mkdir(
                parents=True, exist_ok=True
            )
            ann_df.to_csv(self.config.raw_annotations_path, index=False)
            logging.info(
                f"Saved raw annotations to {self.config.raw_annotations_path}"
            )

            return ann_df, image_split, image_dir, output_dir

        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_annotations(label_path: Path) -> pd.DataFrame:
        """Parse the xView GeoJSON into a flat DataFrame."""
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")

        with label_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        rows: list[dict] = []
        for feature in data["features"]:
            props = feature["properties"]
            coords = props.get("bounds_imcoords", "")
            if not coords:
                continue

            x1, y1, x2, y2 = map(int, coords.split(","))
            if x2 <= x1 or y2 <= y1:
                continue

            rows.append(
                {
                    "image_id": props["image_id"],
                    "type_id": int(props["type_id"]),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

        ann_df = pd.DataFrame(rows)
        if ann_df.empty:
            raise ValueError("No valid annotations were found in the GeoJSON file.")

        return ann_df

    @staticmethod
    def _filter_existing_images(ann_df: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
        """Drop annotations that reference images not present on disk."""
        existing_images = {p.name for p in image_dir.glob("*.tif")}
        before_ann = len(ann_df)
        before_img = ann_df["image_id"].nunique()

        filtered = ann_df[ann_df["image_id"].isin(existing_images)].copy()
        removed_ann = before_ann - len(filtered)
        removed_img = before_img - filtered["image_id"].nunique()

        logging.info(
            f"After filtering missing images: "
            f"{len(filtered):,} annotations from "
            f"{filtered['image_id'].nunique():,} images."
        )
        if removed_img:
            logging.warning(
                f"Removed {removed_ann:,} annotations linked to "
                f"{removed_img:,} missing images."
            )

        if filtered.empty:
            raise ValueError(f"No annotated images were found in: {image_dir}")

        return filtered

    def _split_images(self, ann_df: pd.DataFrame) -> dict[str, str]:
        """Reproducibly assign each image to 'train' or 'val'."""
        image_ids = list(sorted(ann_df["image_id"].unique()))
        rng = random.Random(self.config.seed)
        rng.shuffle(image_ids)

        val_count = max(1, int(len(image_ids) * self.config.val_fraction))
        val_images = set(image_ids[:val_count])

        split = {
            img_id: ("val" if img_id in val_images else "train")
            for img_id in image_ids
        }
        train_n = sum(1 for v in split.values() if v == "train")
        val_n = sum(1 for v in split.values() if v == "val")
        logging.info(f"Image split — train: {train_n}, val: {val_n}")
        return split