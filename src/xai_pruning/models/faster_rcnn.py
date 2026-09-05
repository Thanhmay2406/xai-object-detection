"""Faster R-CNN + ResNet-50-FPN construction used by the experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from xai_pruning.config import NUM_CLASSES


def build_faster_rcnn(
    num_classes: int = NUM_CLASSES,
    *,
    pretrained_coco: bool = False,
    pretrained_backbone: bool = False,
    trainable_backbone_layers: int = 3,
    min_size: int = 800,
    max_size: int = 1333,
) -> nn.Module:
    """Build the torchvision architecture used by Baseline and Main.

    Loading research checkpoints should use both pretrained flags as ``False``
    (the defaults), which avoids downloads and produces the exact baseline
    architecture before strict state loading.
    """

    if pretrained_coco:
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
        )
    else:
        backbone_weights = (
            ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        )
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=backbone_weights,
            min_size=min_size,
            max_size=max_size,
            num_classes=num_classes,
            **(
                {"trainable_backbone_layers": trainable_backbone_layers}
                if backbone_weights is not None
                else {}
            ),
        )

    if pretrained_coco:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_baseline_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    *,
    strict: bool = True,
    return_checkpoint: bool = False,
):
    """Build the baseline architecture and load a current or legacy checkpoint."""

    from xai_pruning.pruning.checkpoint import extract_state_dict, torch_load_cpu

    checkpoint = torch_load_cpu(checkpoint_path)
    validate_checkpoint_label_mapping(checkpoint)
    model = build_faster_rcnn(num_classes=NUM_CLASSES)
    model.load_state_dict(extract_state_dict(checkpoint), strict=strict)
    model.to(device).eval()
    if return_checkpoint:
        return model, checkpoint
    return model


def validate_checkpoint_label_mapping(checkpoint) -> None:
    """Check the frozen foreground mapping when checkpoint metadata provides it."""

    from xai_pruning.config import MODEL_TO_COCO_LABEL

    if not isinstance(checkpoint, dict):
        return
    stored = checkpoint.get("label_to_category_id")
    if stored is None and isinstance(checkpoint.get("label_mapping"), dict):
        stored = checkpoint["label_mapping"].get("model_to_coco_label")
    if stored is None:
        return
    stored = {int(key): int(value) for key, value in stored.items()}
    for model_label, coco_label in MODEL_TO_COCO_LABEL.items():
        if stored.get(model_label) != coco_label:
            raise ValueError(
                "Checkpoint label mapping disagrees at model label "
                f"{model_label}: expected COCO {coco_label}, found {stored.get(model_label)}"
            )
