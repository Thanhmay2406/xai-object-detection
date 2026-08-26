from torch import nn

from torchvision.models import ResNet50_Weights
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
)


def build_faster_rcnn(
    num_classes: int,
    pretrained_coco: bool = True,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = 800,
    max_size: int = 1333,
) -> nn.Module:
    """
    Faster R-CNN + ResNet50 + FPN.

    Parameters
    ----------
    num_classes:
        Total number of classes INCLUDING background.

        Example:
            1 object class -> num_classes = 2

    pretrained_coco:
        True:
            initialize from torchvision COCO pretrained
            Faster R-CNN.

    pretrained_backbone:
        Used when pretrained_coco=False.
        Initialize ResNet50 from ImageNet.

    trainable_backbone_layers:
        Number of ResNet stages that remain trainable.

    min_size / max_size:
        Faster R-CNN internal resize parameters.
    """

    if pretrained_coco:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

        model = fasterrcnn_resnet50_fpn(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
        )

    else:
        weights_backbone = (
            ResNet50_Weights.IMAGENET1K_V1
            if pretrained_backbone
            else None
        )

        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=weights_backbone,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
        )

    # Replace COCO classification/regression predictor
    # with predictor for the current dataset.
    in_features = (
        model.roi_heads
        .box_predictor
        .cls_score
        .in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes,
    )

    return model


if __name__ == "__main__":
    model = build_faster_rcnn(
        num_classes=2,
    )

    print(model)
