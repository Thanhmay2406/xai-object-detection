from __future__ import annotations

import torch


def aligned_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """IoU for aligned XYXY boxes with shape ``[N, 4]``."""

    if boxes1.shape != boxes2.shape or boxes1.ndim != 2 or boxes1.shape[-1] != 4:
        raise ValueError(f"Expected aligned [N,4] boxes, got {boxes1.shape} and {boxes2.shape}")
    lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp_min(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp_min(0)
    return inter / (area1 + area2 - inter).clamp_min(eps)


def pairwise_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Pairwise IoU for two XYXY box collections."""

    if boxes1.ndim != 2 or boxes2.ndim != 2 or boxes1.shape[-1] != 4 or boxes2.shape[-1] != 4:
        raise ValueError("pairwise_box_iou expects [N,4] and [M,4]")
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0)
        * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
    )[:, None]
    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0)
        * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)
    )[None, :]
    return inter / (area1 + area2 - inter).clamp_min(eps)
