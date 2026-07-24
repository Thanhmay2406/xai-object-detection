from __future__ import annotations

import torch

from odam_yolo.config import OdamConfig
from odam_yolo.loss import OdamDetectionLoss


def make_loss_shell() -> OdamDetectionLoss:
    shell = object.__new__(OdamDetectionLoss)
    shell.odam_cfg = OdamConfig(
        strict_p2=False,
        include_self_positive=False,
        negative_overlap_iou=0.0,
    )
    return shell


def test_pair_loss_prefers_same_object_similarity() -> None:
    loss_fn = make_loss_shell()
    cams = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
        ],
        requires_grad=True,
    )
    cams = cams / cams.norm(dim=1, keepdim=True)
    object_ids = torch.tensor([0, 0, 1, 1])
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 0.0, 15.0, 10.0],
            [5.0, 0.0, 15.0, 10.0],
        ]
    )
    ious = torch.tensor([0.9, 0.8, 0.95, 0.7])
    loss, pos, neg = loss_fn._pair_discrimination_loss(cams, object_ids, boxes, ious)
    assert torch.isfinite(loss)
    assert pos == 2
    assert neg == 4
    loss.backward()
    assert cams.grad_fn is not None
