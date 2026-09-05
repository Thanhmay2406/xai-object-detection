"""Gradient-times-activation importance helpers."""

from .importance import (
    ActivationBank,
    XAIImportanceAccumulator,
    aggregate_importance,
    aggregate_multi_group_importance,
    channel_xai_score,
    estimate_xai_importance,
    get_best_same_class_matches,
    multi_group_gradient_x_activation,
    normalize_vector,
)

__all__ = [
    "ActivationBank",
    "XAIImportanceAccumulator",
    "aggregate_importance",
    "aggregate_multi_group_importance",
    "channel_xai_score",
    "estimate_xai_importance",
    "get_best_same_class_matches",
    "multi_group_gradient_x_activation",
    "normalize_vector",
]
