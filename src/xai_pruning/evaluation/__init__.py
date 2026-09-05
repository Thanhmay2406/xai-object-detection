"""Canonical COCO correctness and efficiency evaluation APIs."""

from .coco_eval import EvaluationResult, evaluate_detector, evaluate_predictions, write_predictions_json
from .efficiency import benchmark_latency, count_parameters, estimate_conv_linear_macs

__all__ = [
    "EvaluationResult",
    "benchmark_latency",
    "count_parameters",
    "estimate_conv_linear_macs",
    "evaluate_detector",
    "evaluate_predictions",
    "write_predictions_json",
]
