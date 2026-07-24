from typing import Any

import torch
from torch import nn


FEATURE_ATTR = "_odam_input_features"
HOOK_ATTR = "_odam_feature_hook_handle"


def _capture_detect_inputs(module: nn.Module, inputs: tuple[Any, ...]) -> None:
    """Capture the P-level tensors before Detect.forward mutates its input list."""

    if not inputs:
        raise RuntimeError("Detect pre-hook received no inputs")
    feature_list = inputs[0]
    if not isinstance(feature_list, (list, tuple)):
        raise TypeError(
            "Expected the Detect head input to be a list/tuple of feature maps, "
            f"got {type(feature_list)!r}"
        )
    features = tuple(feature_list)
    if not features or not all(isinstance(x, torch.Tensor) for x in features):
        raise TypeError("Detect head inputs must be a non-empty tensor sequence")

    # A tuple copy is required because Ultralytics replaces each element of the
    # original list with the concatenated head output during Detect.forward.
    setattr(module, FEATURE_ATTR, features)


def install_feature_tap(head: nn.Module) -> None:
    """Install one forward-pre-hook on the Ultralytics Detect head."""

    if hasattr(head, HOOK_ATTR):
        return
    handle = head.register_forward_pre_hook(_capture_detect_inputs)
    setattr(head, HOOK_ATTR, handle)


def get_captured_features(head: nn.Module) -> tuple[torch.Tensor, ...] | None:
    features = getattr(head, FEATURE_ATTR, None)
    if features is None:
        return None
    return tuple(features)
