"""
Logging configuration for Satellite Image Threat Detection.

Provides two dedicated loggers:
1. Backend / Deployment Logger:
   - Records server lifecycle, deployment events, pipeline execution, health checks, and errors.
   - Saved to: logs/backend/backend_<timestamp>.log
2. Detection Details Logger:
   - Dedicated exclusively to recording threat detection events, input image metadata,
     bounding box coordinates, class labels, and confidence scores.
   - Saved to: logs/detection/detection_<timestamp>.log
"""

from __future__ import annotations

import logging as _py_logging
import os
import sys
from datetime import datetime
from typing import Any

# Base log directory structure
BASE_LOG_DIR = os.path.join(os.getcwd(), "logs")
BACKEND_LOG_DIR = os.path.join(BASE_LOG_DIR, "backend")
DETECTION_LOG_DIR = os.path.join(BASE_LOG_DIR, "detection")

os.makedirs(BACKEND_LOG_DIR, exist_ok=True)
os.makedirs(DETECTION_LOG_DIR, exist_ok=True)

# Timestamped log files
_TIMESTAMP = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
BACKEND_LOG_FILE = f"backend_{_TIMESTAMP}.log"
DETECTION_LOG_FILE = f"detection_{_TIMESTAMP}.log"

BACKEND_LOG_FILE_PATH = os.path.join(BACKEND_LOG_DIR, BACKEND_LOG_FILE)
DETECTION_LOG_FILE_PATH = os.path.join(DETECTION_LOG_DIR, DETECTION_LOG_FILE)


def get_backend_logger(name: str = "SITP.Backend") -> _py_logging.Logger:
    """Create and return the Backend and Deployment logger."""
    logger = _py_logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(_py_logging.INFO)
        formatter = _py_logging.Formatter(
            "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
        )

        # File Handler
        file_handler = _py_logging.FileHandler(BACKEND_LOG_FILE_PATH, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(_py_logging.INFO)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = _py_logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(_py_logging.INFO)
        logger.addHandler(console_handler)

        logger.propagate = False

    return logger


def get_detection_logger(name: str = "SITP.Detection") -> _py_logging.Logger:
    """Create and return the Detection Details logger."""
    logger = _py_logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(_py_logging.INFO)
        formatter = _py_logging.Formatter(
            "[ %(asctime)s ] - %(levelname)s - %(message)s"
        )

        # File Handler
        file_handler = _py_logging.FileHandler(DETECTION_LOG_FILE_PATH, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(_py_logging.INFO)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = _py_logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(_py_logging.INFO)
        logger.addHandler(console_handler)

        logger.propagate = False

    return logger


# Initialise logger instances
backend_logger = get_backend_logger()
detection_logger = get_detection_logger()

# Backward-compatibility alias for existing modules:
# `from src.SITP.logger import logging` will use the backend logger.
logging = backend_logger


def log_detection_details(
    filename: str,
    detections: list[list[float]] | list[Any],
    class_names: dict[int, str] | None = None,
    img_metadata: dict[str, Any] | None = None,
    conf_threshold: float | None = None,
    iou_threshold: float | None = None,
    extra_info: dict[str, Any] | None = None,
) -> None:
    """
    Format and log detailed threat detection results to the detection log file.

    Parameters
    ----------
    filename : str
        Name of the analyzed image file.
    detections : list
        List of detections, where each detection is [x1, y1, x2, y2, score, class_id].
    class_names : dict[int, str], optional
        Mapping of class IDs to class names.
    img_metadata : dict, optional
        Image metadata (width, height, bands, file size, driver, etc.).
    conf_threshold : float, optional
        Confidence threshold applied for inference.
    iou_threshold : float, optional
        IoU threshold applied for NMS.
    extra_info : dict, optional
        Any extra context or performance metrics.
    """
    class_names = class_names or {}
    total = len(detections)

    # Compute statistics
    scores = [float(det[4]) for det in detections] if total > 0 else []
    avg_conf = (sum(scores) / total) if total > 0 else 0.0
    max_conf = max(scores) if total > 0 else 0.0
    min_conf = min(scores) if total > 0 else 0.0

    class_counts: dict[str, int] = {}
    for det in detections:
        cls_id = int(det[5])
        name = class_names.get(cls_id, f"Class {cls_id}")
        class_counts[name] = class_counts.get(name, 0) + 1

    lines = [
        "=" * 80,
        f"THREAT DETECTION EVENT: {filename}",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 80,
    ]

    # Image metadata section
    if img_metadata:
        lines.append("IMAGE METADATA:")
        for k, v in img_metadata.items():
            lines.append(f"  * {k.replace('_', ' ').capitalize()}: {v}")
        lines.append("-" * 80)

    # Inference settings section
    lines.append("INFERENCE CONFIGURATION:")
    if conf_threshold is not None:
        lines.append(f"  * Confidence Threshold: {conf_threshold}")
    if iou_threshold is not None:
        lines.append(f"  * IoU Threshold (NMS): {iou_threshold}")
    if extra_info:
        for k, v in extra_info.items():
            lines.append(f"  * {k.replace('_', ' ').capitalize()}: {v}")
    lines.append("-" * 80)

    # Detection summary section
    lines.append("DETECTION SUMMARY:")
    lines.append(f"  * Total Threats Detected: {total}")
    lines.append(f"  * Unique Threat Classes: {len(class_counts)}")
    if total > 0:
        lines.append(f"  * Average Confidence: {avg_conf * 100:.2f}% (Score: {avg_conf:.4f})")
        lines.append(f"  * Highest Confidence: {max_conf * 100:.2f}%")
        lines.append(f"  * Lowest Confidence:  {min_conf * 100:.2f}%")
        lines.append("  * Class Breakdown:")
        for cls_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total) * 100
            lines.append(f"      - {cls_name:<25}: {count:>4} detections ({pct:.1f}%)")
    else:
        lines.append("  * No threats detected matching confidence criteria.")
    lines.append("-" * 80)

    # Detailed bounding boxes & threat coordinates
    if total > 0:
        lines.append("ITEMIZED THREAT DETECTIONS:")
        lines.append(
            f"  {'#':<4} | {'Class Name (ID)':<30} | {'Confidence':<12} | {'Bounding Box [x1, y1, x2, y2]':<35} | {'Box Size (WxH)':<15}"
        )
        lines.append("  " + "-" * 105)

        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2, score, cls_id = det
            cls_int = int(cls_id)
            cls_name = class_names.get(cls_int, f"Class {cls_int}")
            class_str = f"{cls_name} ({cls_int})"
            bbox_str = f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
            w_px = max(0.0, x2 - x1)
            h_px = max(0.0, y2 - y1)
            size_str = f"{w_px:.1f}x{h_px:.1f} px"
            score_str = f"{score * 100:.1f}% ({score:.3f})"

            lines.append(
                f"  {i:<4} | {class_str:<30} | {score_str:<12} | {bbox_str:<35} | {size_str:<15}"
            )

    lines.append("=" * 80)
    lines.append("")  # Blank line separator

    detection_logger.info("\n".join(lines))