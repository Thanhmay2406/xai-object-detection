"""Exact local structural edits used by Pipeline 03 and Pipeline 04."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


def get_submodule_by_name(root: nn.Module, path: str) -> nn.Module:
    module = root
    for part in path.split("."):
        module = getattr(module, part)
    return module


def set_submodule_by_name(root: nn.Module, path: str, new_module: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def clone_conv2d_keep_outputs(conv: nn.Conv2d, keep_indices: Sequence[int]) -> nn.Conv2d:
    if not isinstance(conv, nn.Conv2d):
        raise TypeError("Expected nn.Conv2d")
    keep = torch.as_tensor(keep_indices, dtype=torch.long, device=conv.weight.device)
    new_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=len(keep_indices),
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype)
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight[keep, ...])
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias[keep])
    return new_conv


def clone_conv2d_keep_inputs(conv: nn.Conv2d, keep_indices: Sequence[int]) -> nn.Conv2d:
    if not isinstance(conv, nn.Conv2d):
        raise TypeError("Expected nn.Conv2d")
    if conv.groups != 1:
        raise ValueError("Input-channel pruning supports groups=1 only")
    keep = torch.as_tensor(keep_indices, dtype=torch.long, device=conv.weight.device)
    new_conv = nn.Conv2d(
        in_channels=len(keep_indices),
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype)
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight[:, keep, ...])
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def clone_batchnorm_keep_features(
    batchnorm: nn.BatchNorm2d, keep_indices: Sequence[int]
) -> nn.BatchNorm2d:
    if not isinstance(batchnorm, nn.BatchNorm2d):
        raise TypeError("Expected nn.BatchNorm2d")
    reference = batchnorm.weight if batchnorm.affine else batchnorm.running_mean
    if reference is None:
        raise ValueError("BatchNorm without affine parameters or running stats is unsupported")
    keep = torch.as_tensor(keep_indices, dtype=torch.long, device=reference.device)
    new_batchnorm = nn.BatchNorm2d(
        num_features=len(keep_indices),
        eps=batchnorm.eps,
        momentum=batchnorm.momentum,
        affine=batchnorm.affine,
        track_running_stats=batchnorm.track_running_stats,
    ).to(device=reference.device, dtype=reference.dtype)
    with torch.no_grad():
        if batchnorm.affine:
            new_batchnorm.weight.copy_(batchnorm.weight[keep])
            new_batchnorm.bias.copy_(batchnorm.bias[keep])
        if batchnorm.track_running_stats:
            new_batchnorm.running_mean.copy_(batchnorm.running_mean[keep])
            new_batchnorm.running_var.copy_(batchnorm.running_var[keep])
            new_batchnorm.num_batches_tracked.copy_(batchnorm.num_batches_tracked)
    return new_batchnorm


def _metadata_rows(group_metadata) -> list[dict]:
    if hasattr(group_metadata, "iterrows"):
        return [row.to_dict() for _, row in group_metadata.iterrows()]
    return [dict(row) for row in group_metadata]


def build_pruning_plan_from_threshold(
    global_ranking,
    group_metadata,
    tau: float,
    min_remaining: int = 1,
) -> dict[str, list[int]]:
    """Select exactly channels with ``importance_normalized <= tau``."""

    if hasattr(global_ranking, "iterrows"):
        ranking_rows = [row.to_dict() for _, row in global_ranking.iterrows()]
    else:
        ranking_rows = [dict(row) for row in global_ranking]
    plan: dict[str, list[int]] = {}
    for row in ranking_rows:
        if float(row["importance_normalized"]) <= float(tau):
            plan.setdefault(str(row["group_id"]), []).append(int(row["channel"]))
    plan = {
        group_id: sorted(set(channel_indices))
        for group_id, channel_indices in plan.items()
    }
    return validate_pruning_plan(plan, group_metadata, min_remaining=min_remaining)


def validate_pruning_plan(
    plan: Mapping[str, Sequence[int]],
    group_metadata,
    min_remaining: int = 1,
) -> dict[str, list[int]]:
    """Validate group IDs and channel indices without altering the plan."""

    if not isinstance(plan, Mapping):
        raise TypeError("pruning_plan must be a mapping")
    rows = _metadata_rows(group_metadata)
    metadata_by_id = {str(row["group_id"]): row for row in rows}
    normalized = {
        str(group_id): sorted({int(index) for index in indices})
        for group_id, indices in plan.items()
    }
    missing = sorted(set(normalized) - set(metadata_by_id))
    if missing:
        raise KeyError(f"Unknown pruning groups: {missing}")

    for group_id, prune_indices in normalized.items():
        if not prune_indices:
            continue
        channels = int(metadata_by_id[group_id]["channels"])
        if min(prune_indices) < 0 or max(prune_indices) >= channels:
            raise IndexError(f"Invalid channel in group={group_id}")
        if channels - len(prune_indices) < min_remaining:
            raise ValueError(f"Pruning removes too many channels from {group_id}")
    return normalized


def apply_structural_pruning_plan(
    model: nn.Module,
    plan: Mapping[str, Sequence[int]],
    group_metadata,
    min_remaining: int = 1,
) -> list[dict]:
    """Remove exactly the selected hidden channels and update producer/BN/consumer."""

    rows = _metadata_rows(group_metadata)
    normalized = validate_pruning_plan(plan, rows, min_remaining=min_remaining)
    metadata_by_id = {str(row["group_id"]): row for row in rows}
    ordered = [
        row["group_id"]
        for row in sorted(rows, key=lambda row: (row["stage"], row["block"], row["group_kind"]))
        if row["group_id"] in normalized
    ]

    applied = []
    for group_id in ordered:
        prune_indices = normalized[group_id]
        if not prune_indices:
            continue
        metadata = metadata_by_id[group_id]
        producer = get_submodule_by_name(model, metadata["producer_name"])
        batchnorm = get_submodule_by_name(model, metadata["norm_name"])
        consumer = get_submodule_by_name(model, metadata["consumer_name"])
        original_channels = int(producer.out_channels)
        prune_set = set(prune_indices)
        keep_indices = [index for index in range(original_channels) if index not in prune_set]

        set_submodule_by_name(
            model,
            metadata["producer_name"],
            clone_conv2d_keep_outputs(producer, keep_indices),
        )
        set_submodule_by_name(
            model,
            metadata["norm_name"],
            clone_batchnorm_keep_features(batchnorm, keep_indices),
        )
        set_submodule_by_name(
            model,
            metadata["consumer_name"],
            clone_conv2d_keep_inputs(consumer, keep_indices),
        )
        applied.append(
            {
                "group_id": group_id,
                "pruned_channels": prune_indices,
                "num_pruned": len(prune_indices),
                "original_channels": original_channels,
                "remaining_channels": len(keep_indices),
            }
        )
    return applied


def count_parameters(model: nn.Module) -> int:
    """Count every trainable and frozen parameter tensor element."""

    return int(sum(parameter.numel() for parameter in model.parameters()))
