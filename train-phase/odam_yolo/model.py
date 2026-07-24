from typing import Any

from ultralytics.nn.tasks import DetectionModel

from .config import OdamConfig, load_odam_config
from .feature_tap import install_feature_tap
from .loss import OdamDetectionLoss


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
