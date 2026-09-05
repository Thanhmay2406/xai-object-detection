"""Detector construction and baseline checkpoint loading."""

from .faster_rcnn import build_faster_rcnn, load_baseline_model

__all__ = ["build_faster_rcnn", "load_baseline_model"]
