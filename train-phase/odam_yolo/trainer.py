from typing import Any

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

from .config import load_odam_config
from .live_logging import OdamLiveLogger, set_live_logger
from .model import OdamDetectionModel


class OdamDetectionTrainer(DetectionTrainer):
    """Ultralytics trainer that constructs :class:`OdamDetectionModel`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._odam_live_logger: OdamLiveLogger | None = None
        self._odam_next_batch_index = 0
        self._odam_current_batch_size = 0
        self.add_callback("on_train_start", self._odam_on_train_start)
        self.add_callback("on_train_epoch_start", self._odam_on_train_epoch_start)
        self.add_callback("on_train_batch_end", self._odam_on_train_batch_end)
        self.add_callback("on_train_epoch_end", self._odam_on_train_epoch_end)
        self.add_callback("on_train_end", self._odam_on_train_end)

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
        batch_index = int(getattr(self, "_odam_next_batch_index", 0))
        self._odam_next_batch_index = batch_index + 1
        self._odam_current_batch_size = int(batch["img"].shape[0])
        if self._odam_live_logger is not None:
            self._odam_live_logger.start_batch()
        model = unwrap_model(self.model)
        model.model[-1]._odam_current_epoch = int(getattr(self, "epoch", 0))
        model.model[-1]._odam_current_batch_index = batch_index
        return batch

    def get_validator(self):
        validator = super().get_validator()
        # Base implementation resets this to three names; restore four so both
        # training and validation accumulators have matching dimensions.
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "odam_loss")
        return validator

    def _odam_on_train_start(self, trainer) -> None:
        if not OdamLiveLogger.should_log_on_this_rank():
            set_live_logger(None)
            return
        self._odam_live_logger = OdamLiveLogger(self.save_dir)
        self._odam_live_logger.open()
        set_live_logger(self._odam_live_logger)

    def _odam_on_train_epoch_start(self, trainer) -> None:
        self._odam_next_batch_index = 0
        if self._odam_live_logger is not None:
            self._odam_live_logger.start_epoch(int(getattr(self, "epoch", 0)))

    def _odam_on_train_batch_end(self, trainer) -> None:
        logger = self._odam_live_logger
        if logger is None or getattr(self, "loss_items", None) is None:
            return
        model = unwrap_model(self.model)
        criterion = getattr(model, "criterion", None)
        stats = getattr(criterion, "last_stats", None)
        if stats is None:
            return
        lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        logger.record_batch(
            epoch=int(getattr(self, "epoch", 0)),
            batch_index=max(0, int(getattr(self, "_odam_next_batch_index", 1)) - 1),
            batch_size=max(1, int(getattr(self, "_odam_current_batch_size", 1))),
            loss_items=self.loss_items,
            lr=lrs[0] if lrs else 0.0,
            gpu_memory_gb=float(self._get_memory()),
            stats=stats,
        )

    def _odam_on_train_epoch_end(self, trainer) -> None:
        if self._odam_live_logger is not None:
            self._odam_live_logger.end_epoch(int(getattr(self, "epoch", 0)))

    def _odam_on_train_end(self, trainer) -> None:
        if self._odam_live_logger is not None:
            self._odam_live_logger.close()
            self._odam_live_logger = None
        set_live_logger(None)
