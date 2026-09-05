"""Frozen experiment constants and lightweight YAML configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

NUM_CLASSES = 7

COCO_TO_MODEL_LABEL = {
    1: 2,  # Broken
    2: 3,  # Chipped
    3: 4,  # Scratched
    4: 5,  # Severe Rust
    5: 6,  # Tip Wear
}
MODEL_TO_COCO_LABEL = {value: key for key, value in COCO_TO_MODEL_LABEL.items()}

MAIN_TAU = 0.01875
EXPECTED_BASELINE_PARAMS = 41_377_906
EXPECTED_MAIN_PARAMS = 39_182_184

TASK_TO_LOSS = {
    "roi_cls": "loss_classifier",
    "roi_reg": "loss_box_reg",
    "rpn_obj": "loss_objectness",
    "rpn_reg": "loss_rpn_box_reg",
}
DEFAULT_FPN_CONVS = [
    "backbone.fpn.layer_blocks.0.0",
    "backbone.fpn.layer_blocks.1.0",
    "backbone.fpn.layer_blocks.2.0",
    "backbone.fpn.layer_blocks.3.0",
]


@dataclass
class XAIImportanceConfig:
    """Configuration retained from the existing XAI importance scaffold."""

    candidate_modules: list[str] = field(default_factory=lambda: list(DEFAULT_FPN_CONVS))
    tasks: tuple[str, ...] = ("roi_cls", "roi_reg", "rpn_obj", "rpn_reg")
    task_weights: dict[str, float] = field(
        default_factory=lambda: {task: 1.0 for task in TASK_TO_LOSS}
    )
    max_probe_batches: int = 200
    normalize_scores: bool = True
    use_scale_buckets: bool = True
    eps: float = 1e-12


def load_config(path: str | Path) -> dict[str, Any]:
    """Load one experiment YAML file and validate its basic shape."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {config_path}")
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject configuration values that conflict with frozen model semantics."""

    model = config.get("model", {})
    if not isinstance(model, Mapping):
        raise TypeError("config.model must be a mapping")
    if "num_classes" in model and int(model["num_classes"]) != NUM_CLASSES:
        raise ValueError(
            f"model.num_classes must remain {NUM_CLASSES}, found {model['num_classes']}"
        )

    pruning = config.get("pruning", {})
    if not isinstance(pruning, Mapping):
        raise TypeError("config.pruning must be a mapping")
    if pruning.get("candidate") == "main" and "tau" in pruning:
        tau = float(pruning["tau"])
        if tau != MAIN_TAU:
            raise ValueError(f"Main tau must remain {MAIN_TAU}, found {tau}")


def merge_config(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""

    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    validate_config(merged)
    return merged
