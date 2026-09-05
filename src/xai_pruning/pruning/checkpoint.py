"""Backward-compatible checkpoint loading and pruned architecture reconstruction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import torch

from xai_pruning.config import MAIN_TAU, NUM_CLASSES
from xai_pruning.models.faster_rcnn import build_faster_rcnn, validate_checkpoint_label_mapping
from xai_pruning.pruning.groups import discover_resnet_bottleneck_pruning_groups
from xai_pruning.pruning.structural import apply_structural_pruning_plan, validate_pruning_plan


def torch_load_cpu(path: str | Path):
    """Load legacy checkpoints while remaining compatible with older PyTorch."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint) -> dict[str, torch.Tensor]:
    """Extract raw/model/model_state_dict/state_dict formats and strip DDP prefixes."""

    candidate = checkpoint
    if isinstance(candidate, Mapping):
        for key in ("model_state_dict", "model", "state_dict"):
            value = candidate.get(key)
            if isinstance(value, Mapping):
                candidate = value
                break
    if not isinstance(candidate, Mapping):
        raise TypeError("Checkpoint does not contain a state_dict")
    state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in candidate.items()
        if torch.is_tensor(value)
    }
    if not state_dict:
        raise TypeError("Checkpoint does not contain tensor state_dict entries")
    return state_dict


def build_pruned_model_from_checkpoint(
    baseline_checkpoint: str | Path,
    pruned_checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_candidate: str | None = "main",
    expected_tau: float | None = MAIN_TAU,
    strict: bool = True,
) -> tuple[torch.nn.Module, dict]:
    """Reconstruct and strict-load a structurally pruned model.

    The baseline weights are loaded first, then the checkpoint's unchanged
    ``pruning_plan`` is applied to the 32 supported bottleneck groups before the
    pruned state dictionary is loaded.
    """

    baseline_payload = torch_load_cpu(baseline_checkpoint)
    validate_checkpoint_label_mapping(baseline_payload)
    model = build_faster_rcnn(num_classes=NUM_CLASSES)
    model.load_state_dict(extract_state_dict(baseline_payload), strict=strict)

    groups = discover_resnet_bottleneck_pruning_groups(model, include_modules=False)
    payload = torch_load_cpu(pruned_checkpoint)
    if not isinstance(payload, Mapping):
        raise TypeError("Pruned checkpoint must be a mapping with pruning_plan metadata")
    validate_checkpoint_label_mapping(payload)
    if "num_classes" in payload and int(payload["num_classes"]) != NUM_CLASSES:
        raise ValueError(
            f"Expected num_classes={NUM_CLASSES}, found {payload['num_classes']}"
        )

    plan = payload.get("pruning_plan")
    if not isinstance(plan, Mapping) or not plan:
        raise KeyError("Pruned checkpoint must contain a non-empty pruning_plan")
    normalized_plan = validate_pruning_plan(plan, groups)

    candidate = payload.get("candidate_name", expected_candidate)
    tau_value = payload.get("tau", expected_tau)
    tau = float(tau_value) if tau_value is not None else None
    if expected_candidate is not None and candidate not in (expected_candidate, None):
        raise ValueError(
            f"Expected candidate_name={expected_candidate!r}, found {candidate!r}"
        )
    if expected_tau is not None and tau is not None and not math.isclose(
        tau, float(expected_tau), rel_tol=0.0, abs_tol=1e-10
    ):
        raise ValueError(f"Pruning tau mismatch: expected {expected_tau}, found {tau}")

    applied = apply_structural_pruning_plan(model, normalized_plan, groups)
    model.load_state_dict(extract_state_dict(payload), strict=strict)
    model.to(device).eval()

    metadata = {
        "checkpoint": payload,
        "candidate_name": candidate,
        "tau": tau,
        "pruning_plan": normalized_plan,
        "groups": groups,
        "applied_groups": applied,
        "strict": bool(strict),
    }
    return model, metadata
