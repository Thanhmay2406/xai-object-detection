from typing import Any

import torch
from torch import nn
from ultralytics.nn.tasks import DetectionModel

from .config import OdamConfig, load_odam_config
from .loss import FEATURE_ATTR, OdamDetectionLoss


HOOK_ATTR = "_odam_feature_hook_handle"


def _capture_detect_inputs(module: nn.Module, inputs: tuple[Any, ...]) -> None:
    """Capture P-level tensors before Ultralytics Detect mutates its input list."""

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
    setattr(module, FEATURE_ATTR, features)


def install_feature_tap(head: nn.Module) -> None:
    if hasattr(head, HOOK_ATTR):
        return
    handle = head.register_forward_pre_hook(_capture_detect_inputs)
    setattr(head, HOOK_ATTR, handle)


class OdamDetectionModel(DetectionModel):
    """DetectionModel whose criterion includes ODAM-Train."""

    def __init__(
        self,
        cfg: str | dict = "yolov8n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        odam_cfg: OdamConfig | None = None,
    ):
        self.odam_cfg = odam_cfg or load_odam_config()
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        install_feature_tap(self.model[-1])

    def init_criterion(self) -> OdamDetectionLoss:
        return OdamDetectionLoss(self, self.odam_cfg)
