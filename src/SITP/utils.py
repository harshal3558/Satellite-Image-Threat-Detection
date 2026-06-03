"""
Utility helpers for the Satellite Image Threat Detection pipeline.
All core logic is derived from the Kaggle notebook:
  satellite-image-threat-detection (3).ipynb
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Seed all major random-number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Device info
# ---------------------------------------------------------------------------

def print_device_info() -> None:
    """Print whether CUDA / GPU is available."""
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU device: CPU only")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_paths() -> tuple[Path, Path, Path]:
    """
    Dynamically locate key file paths for the xView dataset.

    Checks for a ``/kaggle`` directory to detect the Kaggle environment;
    otherwise falls back to local ``./data`` and ``./xview_yolo`` folders.

    Returns
    -------
    tuple[Path, Path, Path]
        (image_dir, label_path, output_dir)
    """
    if Path("/kaggle").exists():
        root = Path("/kaggle/input/datasets/hassanmojab/xview-dataset")
        output_dir = Path("/kaggle/working/xview_yolo")
    else:
        root = Path("./data").resolve()
        output_dir = Path("./xview_yolo").resolve()

    image_dir = root / "train_images" / "train_images"
    label_path = root / "train_labels" / "xView_train.geojson"
    return image_dir, label_path, output_dir


def prepare_dirs(output_dir: Path) -> None:
    """Create the YOLO-style train/val image and label directories."""
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def normalize_to_uint8(chip: np.ndarray) -> np.ndarray:
    """
    Normalise a chip to uint8 via 1-99 percentile clipping.

    Parameters
    ----------
    chip : np.ndarray
        Input array of any dtype with shape (H, W, C).

    Returns
    -------
    np.ndarray
        uint8 array with the same spatial shape.
    """
    chip = np.asarray(chip)
    if chip.dtype == np.uint8:
        return chip

    chip = chip.astype(np.float32)
    lo, hi = np.percentile(chip, (1, 99))
    if hi <= lo:
        return np.zeros_like(chip, dtype=np.uint8)
    chip = np.clip((chip - lo) * 255.0 / (hi - lo), 0, 255)
    return chip.astype(np.uint8)


# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------

def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """
    Return the list of top-left pixel positions for sliding-window tiles.

    Always appends a final start at ``length - tile_size`` to ensure full
    coverage even when ``length`` is not an exact multiple of ``stride``.
    """
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


# ---------------------------------------------------------------------------
# Chip I/O
# ---------------------------------------------------------------------------

def save_chip(
    chip: np.ndarray,
    boxes_yolo: list[list[float]],
    output_dir: Path,
    split: str,
    chip_name: str,
) -> None:
    """
    Write a chip image (JPEG) and its YOLO label file to disk.

    Parameters
    ----------
    chip : np.ndarray
        RGB uint8 array of shape (H, W, 3).
    boxes_yolo : list[list[float]]
        Each inner list is ``[class_idx, x_center, y_center, width, height]``
        in YOLO normalised format.
    output_dir : Path
        Root directory that contains ``images/`` and ``labels/`` sub-trees.
    split : str
        Either ``"train"`` or ``"val"``.
    chip_name : str
        Base filename (without extension) for the output files.
    """
    image_path = output_dir / "images" / split / f"{chip_name}.jpg"
    label_path = output_dir / "labels" / split / f"{chip_name}.txt"

    cv2.imwrite(str(image_path), cv2.cvtColor(chip, cv2.COLOR_RGB2BGR))
    with label_path.open("w", encoding="utf-8") as f:
        for cls, x_center, y_center, width, height in boxes_yolo:
            f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
