from typing import Any

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

from .config import load_odam_config
from .model import OdamDetectionModel


class OdamDetectionTrainer(DetectionTrainer):
    """Ultralytics trainer that constructs :class:`OdamDetectionModel`."""

    def get_model(
        self,
        cfg: str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> OdamDetectionModel:
        model = OdamDetectionModel(
            cfg=cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            odam_cfg=load_odam_config(),
        )
        if weights:
            model.load(weights)
        return model

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess_batch(batch)
        model = unwrap_model(self.model)
        model.model[-1]._odam_current_epoch = int(getattr(self, "epoch", 0))
        return batch

    def get_validator(self):
        validator = super().get_validator()
        # Base implementation resets this to three names; restore four so both
        # training and validation accumulators have matching dimensions.
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "odam_loss")
        return validator
