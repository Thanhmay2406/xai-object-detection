from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any
import os

import yaml


@dataclass(frozen=True)
class OdamConfig:
    """Configuration for first-order ODAM-Train.

    The default follows the author's released training implementation in one
    important respect: the gradient used to construct the heatmap is detached.
    Therefore, ODAM loss differentiates through the captured detector feature
    map, without requiring a second-order backward pass through the class head.
    """

    enabled: bool = True
    lambda_odam: float = 0.5
    start_epoch: int = 0
    warmup_epochs: int = 0
    every_n_batches: int = 1

    max_samples_per_image: int = 16
    max_samples_per_object: int = 4
    min_assignment_iou: float = 0.0

    map_height: int = 40
    map_width: int = 40
    negative_overlap_iou: float = 0.0
    include_self_positive: bool = False
    eps: float = 1.0e-6

    # False matches the released ODAM-Train code: grad is detached.
    # True is an experimental, much more memory-intensive second-order variant.
    second_order: bool = False

    strict_p2: bool = True
    expected_num_levels: int = 4
    expected_strides: tuple[int, ...] = (4, 8, 16, 32)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "OdamConfig":
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown ODAM config fields: {unknown}")

        data = dict(raw)
        if "expected_strides" in data:
            data["expected_strides"] = tuple(int(x) for x in data["expected_strides"])
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.lambda_odam < 0:
            raise ValueError("lambda_odam must be >= 0")
        if self.start_epoch < 0 or self.warmup_epochs < 0:
            raise ValueError("start_epoch and warmup_epochs must be >= 0")
        if self.every_n_batches < 1:
            raise ValueError("every_n_batches must be >= 1")
        if self.max_samples_per_image < 1:
            raise ValueError("max_samples_per_image must be >= 1")
        if self.max_samples_per_object < 1:
            raise ValueError("max_samples_per_object must be >= 1")
        if self.map_height < 1 or self.map_width < 1:
            raise ValueError("ODAM map dimensions must be positive")
        if not 0.0 <= self.min_assignment_iou <= 1.0:
            raise ValueError("min_assignment_iou must be in [0, 1]")
        if not 0.0 <= self.negative_overlap_iou <= 1.0:
            raise ValueError("negative_overlap_iou must be in [0, 1]")
        if self.eps <= 0:
            raise ValueError("eps must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_odam_config(path: str | Path | None = None) -> OdamConfig:
    """Load configuration from YAML.

    Resolution order:
      1. Explicit ``path``
      2. ``ODAM_CONFIG_PATH`` environment variable
      3. Dataclass defaults
    """

    resolved = path or os.environ.get("ODAM_CONFIG_PATH")
    if not resolved:
        cfg = OdamConfig()
        cfg.validate()
        return cfg

    config_path = Path(resolved).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"ODAM config does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError("ODAM YAML root must be a mapping")
    raw = raw.get("odam", raw)
    if not isinstance(raw, dict):
        raise TypeError("The 'odam' section must be a mapping")
    return OdamConfig.from_dict(raw)
