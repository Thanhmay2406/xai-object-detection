"""Shared implementation for the XAI-guided pruning experiments."""

from .config import (
    COCO_TO_MODEL_LABEL,
    EXPECTED_BASELINE_PARAMS,
    EXPECTED_MAIN_PARAMS,
    MAIN_TAU,
    MODEL_TO_COCO_LABEL,
    NUM_CLASSES,
)

__all__ = [
    "COCO_TO_MODEL_LABEL",
    "EXPECTED_BASELINE_PARAMS",
    "EXPECTED_MAIN_PARAMS",
    "MAIN_TAU",
    "MODEL_TO_COCO_LABEL",
    "NUM_CLASSES",
]
