"""
Self-contained Faster R-CNN + FPN + ODAM-Train network.

Mục tiêu:
- Loại bỏ các import nội bộ của repo ODAM:
    backbone.resnet50
    backbone.fpn
    module.rpn
    layers.pooler
    det_oprs.*
- Giữ API/tensor layout tương thích tối đa với network.py của ODAM.

Phụ thuộc bên ngoài:
    pip install torch torchvision

Lưu ý:
- Backbone chỉ freeze stage sau khi có pretrained weights hoặc checkpoint
  tương thích đã được load.
- Phần RPN/ROI target ở đây là implementation tương thích theo Faster R-CNN
  và cấu hình ODAM, không phụ thuộc các module riêng của repo tác giả.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import roi_align, nms
from torchvision.ops.misc import FrozenBatchNorm2d


INF = 100000000
DPGA_ALLREDUCE_BUCKET_BYTES = 32 * 1024 * 1024


# =============================================================================
# Utility: padding
# =============================================================================

def get_padded_tensor(tensor: torch.Tensor, multiple_number: int, pad_value: float = 0):
    """Pad H,W lên bội số gần nhất của multiple_number."""
    h, w = tensor.shape[-2:]
    padded_h = ((h + multiple_number - 1) // multiple_number) * multiple_number
    padded_w = ((w + multiple_number - 1) // multiple_number) * multiple_number

    if padded_h == h and padded_w == w:
        return tensor

    if tensor.ndim == 4:
        out = tensor.new_full(
            (tensor.shape[0], tensor.shape[1], padded_h, padded_w),
            pad_value,
        )
        out[:, :, :h, :w] = tensor
        return out

    if tensor.ndim == 3:
        out = tensor.new_full(
            (tensor.shape[0], padded_h, padded_w),
            pad_value,
        )
        out[:, :h, :w] = tensor
        return out

    raise ValueError(f"Unsupported tensor ndim={tensor.ndim}")


# =============================================================================
# Bounding-box operations
# =============================================================================

def bbox_transform_opr(bbox: torch.Tensor, gt: torch.Tensor):
    """Encode (x1,y1,x2,y2) GT boxes thành Faster R-CNN deltas."""
    bbox_width = bbox[:, 2] - bbox[:, 0] + 1.0
    bbox_height = bbox[:, 3] - bbox[:, 1] + 1.0
    bbox_ctr_x = bbox[:, 0] + 0.5 * bbox_width
    bbox_ctr_y = bbox[:, 1] + 0.5 * bbox_height

    gt_width = gt[:, 2] - gt[:, 0] + 1.0
    gt_height = gt[:, 3] - gt[:, 1] + 1.0
    gt_ctr_x = gt[:, 0] + 0.5 * gt_width
    gt_ctr_y = gt[:, 1] + 0.5 * gt_height

    dx = (gt_ctr_x - bbox_ctr_x) / bbox_width.clamp(min=1e-6)
    dy = (gt_ctr_y - bbox_ctr_y) / bbox_height.clamp(min=1e-6)
    dw = torch.log(gt_width.clamp(min=1e-6) / bbox_width.clamp(min=1e-6))
    dh = torch.log(gt_height.clamp(min=1e-6) / bbox_height.clamp(min=1e-6))

    return torch.stack((dx, dy, dw, dh), dim=1)


def bbox_transform_inv_opr(bbox: torch.Tensor, deltas: torch.Tensor):
    """Decode Faster R-CNN deltas thành (x1,y1,x2,y2)."""
    max_delta = math.log(1000.0 / 16)

    width = bbox[:, 2] - bbox[:, 0] + 1.0
    height = bbox[:, 3] - bbox[:, 1] + 1.0
    ctr_x = bbox[:, 0] + 0.5 * width
    ctr_y = bbox[:, 1] + 0.5 * height

    pred_ctr_x = ctr_x + deltas[:, 0] * width
    pred_ctr_y = ctr_y + deltas[:, 1] * height

    dw = deltas[:, 2].clamp(max=max_delta)
    dh = deltas[:, 3].clamp(max=max_delta)

    pred_w = width * torch.exp(dw)
    pred_h = height * torch.exp(dh)

    return torch.stack(
        (
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h,
        ),
        dim=1,
    )


def box_overlap_opr(box: torch.Tensor, gt: torch.Tensor):
    """Pairwise IoU: [N,4] x [M,4] -> [N,M]."""
    if box.numel() == 0 or gt.numel() == 0:
        return box.new_zeros((box.shape[0], gt.shape[0]))

    area_box = ((box[:, 2] - box[:, 0] + 1).clamp(min=0) *
                (box[:, 3] - box[:, 1] + 1).clamp(min=0))
    area_gt = ((gt[:, 2] - gt[:, 0] + 1).clamp(min=0) *
               (gt[:, 3] - gt[:, 1] + 1).clamp(min=0))

    wh = (
        torch.minimum(box[:, None, 2:], gt[None, :, 2:])
        - torch.maximum(box[:, None, :2], gt[None, :, :2])
        + 1
    ).clamp(min=0)

    inter = wh[..., 0] * wh[..., 1]
    union = area_box[:, None] + area_gt[None, :] - inter

    return torch.where(
        inter > 0,
        inter / union.clamp(min=1e-12),
        torch.zeros_like(inter),
    )


def paired_box_overlap_opr(box1: torch.Tensor, box2: torch.Tensor):
    """Paired IoU: [N,4] x [N,4] -> [N]."""
    if box1.numel() == 0:
        return box1.new_zeros((0,))

    area1 = ((box1[:, 2] - box1[:, 0] + 1).clamp(min=0) *
             (box1[:, 3] - box1[:, 1] + 1).clamp(min=0))
    area2 = ((box2[:, 2] - box2[:, 0] + 1).clamp(min=0) *
             (box2[:, 3] - box2[:, 1] + 1).clamp(min=0))

    wh = (
        torch.minimum(box1[:, 2:], box2[:, 2:])
        - torch.maximum(box1[:, :2], box2[:, :2])
        + 1
    ).clamp(min=0)

    inter = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - inter

    return torch.where(
        inter > 0,
        inter / union.clamp(min=1e-12),
        torch.zeros_like(inter),
    )


def box_overlap_ignore_opr(box: torch.Tensor, gt: torch.Tensor, ignore_label: int = -1):
    """
    Trả về:
      iou_normal: IoU với GT thường, cột ignore được zero
      ioa_ignore: intersection / area(box), chỉ giữ cột GT ignore
    """
    if box.numel() == 0 or gt.numel() == 0:
        shape = (box.shape[0], gt.shape[0])
        return box.new_zeros(shape), box.new_zeros(shape)

    gt_box = gt[:, :4]
    area_box = ((box[:, 2] - box[:, 0] + 1).clamp(min=0) *
                (box[:, 3] - box[:, 1] + 1).clamp(min=0))
    area_gt = ((gt_box[:, 2] - gt_box[:, 0] + 1).clamp(min=0) *
               (gt_box[:, 3] - gt_box[:, 1] + 1).clamp(min=0))

    wh = (
        torch.minimum(box[:, None, 2:], gt_box[None, :, 2:])
        - torch.maximum(box[:, None, :2], gt_box[None, :, :2])
        + 1
    ).clamp(min=0)

    inter = wh.prod(dim=2)
    union = area_box[:, None] + area_gt[None, :] - inter

    iou = torch.where(
        inter > 0,
        inter / union.clamp(min=1e-12),
        torch.zeros_like(inter),
    )
    ioa = torch.where(
        inter > 0,
        inter / area_box[:, None].clamp(min=1e-12),
        torch.zeros_like(inter),
    )

    ignore = gt[:, 4].eq(ignore_label)[None, :]
    iou = iou.masked_fill(ignore, 0)
    ioa = ioa.masked_fill(~ignore, 0)
    return iou, ioa


def clip_boxes_opr(boxes: torch.Tensor, im_info: torch.Tensor):
    boxes = boxes.clone()
    h = float(im_info[0])
    w = float(im_info[1])
    boxes[:, 0::2].clamp_(min=0, max=max(w - 1, 0))
    boxes[:, 1::2].clamp_(min=0, max=max(h - 1, 0))
    return boxes


def filter_boxes_opr(boxes: torch.Tensor, min_size: float):
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    return (ws >= min_size) & (hs >= min_size)


# =============================================================================
# Losses
# =============================================================================

def softmax_loss(score: torch.Tensor, label: torch.Tensor, ignore_label: int = -1):
    return F.cross_entropy(
        score,
        label.long(),
        reduction="none",
        ignore_index=ignore_label,
    )


def smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, beta: float):
    if pred.numel() == 0:
        return pred.new_zeros((0,))

    abs_x = torch.abs(pred - target)
    if beta < 1e-5:
        loss = abs_x
    else:
        loss = torch.where(
            abs_x < beta,
            0.5 * abs_x.pow(2) / beta,
            abs_x - 0.5 * beta,
        )
    return loss.sum(dim=1)


# =============================================================================
# Backbone: ResNet-50
# =============================================================================

class ResNet50(nn.Module):
    """
    Wrapper torchvision ResNet-50 giữ tên layer gần với implementation ODAM:
      conv1, bn1, layer1..layer4
    Output:
      [C2, C3, C4, C5]
    """

    def __init__(self, pretrained: bool = False, has_bias: bool = False):
        super().__init__()

        # has_bias được giữ trong signature để tương thích code cũ.
        # torchvision ResNet dùng bias=False cho conv khi có norm.
        weights = ResNet50_Weights.DEFAULT if bool(pretrained) else None
        base = resnet50(
            weights=weights,
            norm_layer=FrozenBatchNorm2d,
        )

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.has_pretrained_weights = bool(pretrained)

    def _freeze_module(self, module):
        for p in module.parameters():
            p.requires_grad = False

    def freeze_backbone(self, freeze_at: int):
        freeze_at = int(freeze_at)
        if freeze_at not in (0, 1, 2):
            raise ValueError("backbone_freeze_at must be one of {0, 1, 2}")
        if freeze_at == 0:
            return
        if freeze_at >= 1:
            self._freeze_module(self.conv1)
            self._freeze_module(self.bn1)
        if freeze_at >= 2:
            self._freeze_module(self.layer1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outputs = []

        x = self.layer1(x)  # C2, stride 4
        outputs.append(x)

        x = self.layer2(x)  # C3, stride 8
        outputs.append(x)

        x = self.layer3(x)  # C4, stride 16
        outputs.append(x)

        x = self.layer4(x)  # C5, stride 32
        outputs.append(x)

        return outputs


# =============================================================================
# Feature Pyramid Network
# =============================================================================

class FPN(nn.Module):
    """
    Với FPN(bottom_up, 2, 6), output order:
        [P6, P5, P4, P3, P2]
    strides:
        [64, 32, 16, 8, 4]
    """

    def __init__(self, bottom_up: nn.Module, layers_begin: int, layers_end: int):
        super().__init__()

        if layers_begin != 2:
            raise ValueError("Self-contained implementation hiện hỗ trợ layers_begin=2.")
        if layers_end not in (6,):
            raise ValueError("Self-contained implementation hiện hỗ trợ layers_end=6.")

        self.bottom_up = bottom_up

        in_channels = [256, 512, 1024, 2048]
        fpn_dim = 256

        laterals = []
        outputs = []
        for c in in_channels:
            lat = nn.Conv2d(c, fpn_dim, kernel_size=1)
            out = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)

            nn.init.kaiming_normal_(lat.weight, mode="fan_out")
            nn.init.zeros_(lat.bias)
            nn.init.kaiming_normal_(out.weight, mode="fan_out")
            nn.init.zeros_(out.bias)

            laterals.append(lat)
            outputs.append(out)

        # index 0 tương ứng C5, sau đó C4,C3,C2
        self.lateral_convs = nn.ModuleList(list(reversed(laterals)))
        self.output_convs = nn.ModuleList(list(reversed(outputs)))

    def forward(self, x):
        c2, c3, c4, c5 = self.bottom_up(x)
        bottom_up_features = [c5, c4, c3, c2]

        results = []

        prev = self.lateral_convs[0](bottom_up_features[0])
        p5 = self.output_convs[0](prev)
        results.append(p5)

        for feat, lateral, out_conv in zip(
            bottom_up_features[1:],
            self.lateral_convs[1:],
            self.output_convs[1:],
        ):
            lateral_feat = lateral(feat)
            top_down = F.interpolate(
                prev,
                size=lateral_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            prev = lateral_feat + top_down
            results.append(out_conv(prev))

        # results = [P5,P4,P3,P2]
        p6 = F.max_pool2d(results[0], kernel_size=1, stride=2)
        results.insert(0, p6)

        # [P6,P5,P4,P3,P2], và C5,C4,C3 cho compatibility
        return results, bottom_up_features[:3]


# =============================================================================
# FPN ROI assignment / pooling
# =============================================================================

def assign_boxes_to_levels(
    rois: torch.Tensor,
    min_level: int,
    max_level: int,
    canonical_box_size: int = 224,
    canonical_level: int = 4,
):
    if rois.numel() == 0:
        return rois.new_zeros((0,), dtype=torch.long)

    eps = 1e-6

    # rois: [batch_idx, x1, y1, x2, y2]
    widths = (rois[:, 3] - rois[:, 1]).clamp(min=0)
    heights = (rois[:, 4] - rois[:, 2]).clamp(min=0)
    box_sizes = torch.sqrt(widths * heights)

    levels = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size + eps)
    )
    levels = levels.clamp(min=min_level, max=max_level)
    return levels.to(torch.int64) - min_level


def roi_pooler(fpn_fms, rois, stride, pool_shape, pooler_type):
    aligned = pooler_type == "ROIAlignV2"
    if pooler_type not in ("ROIAlign", "ROIAlignV2"):
        raise ValueError(f"Unknown pooler type: {pooler_type}")

    max_level = int(math.log2(stride[-1]))
    min_level = int(math.log2(stride[0]))
    level_assignments = assign_boxes_to_levels(
        rois, min_level, max_level, 224, 4
    )

    output = fpn_fms[0].new_zeros(
        (len(rois), fpn_fms[0].shape[1], pool_shape[0], pool_shape[1])
    )

    for level, (fm, scale) in enumerate(zip(fpn_fms, stride)):
        inds = torch.nonzero(level_assignments == level, as_tuple=False).squeeze(1)
        if inds.numel() == 0:
            continue
        output[inds] = roi_align(
            fm,
            rois[inds],
            pool_shape,
            spatial_scale=1.0 / scale,
            sampling_ratio=-1,
            aligned=aligned,
        )

    return output


# =============================================================================
# RPN: anchors
# =============================================================================

class AnchorGenerator:
    """
    Anchor generator theo convention ODAM/FPN.
    base_stride=4, off_stride={16,8,4,2,1} cho P6..P2.
    """

    def __init__(self, base_size=16, ratios=(0.5, 1, 2), base_scale=(2,)):
        self.base_size = float(base_size)
        self.ratios = tuple(float(r) for r in ratios)

        if isinstance(base_scale, (int, float)):
            base_scale = (float(base_scale),)
        self.base_scale = tuple(float(s) for s in base_scale)

    def _plane_anchors(self, scale_factor, device, dtype):
        anchors = []

        base_area = self.base_size * self.base_size

        for ratio in self.ratios:
            # ratio = h/w
            w = math.sqrt(base_area / ratio)
            h = w * ratio

            for s in self.base_scale:
                ws = w * s * scale_factor
                hs = h * s * scale_factor

                x1 = -0.5 * (ws - 1)
                y1 = -0.5 * (hs - 1)
                x2 = 0.5 * (ws - 1)
                y2 = 0.5 * (hs - 1)
                anchors.append([x1, y1, x2, y2])

        return torch.tensor(anchors, device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(self, featmap, base_stride, off_stride):
        h, w = featmap.shape[-2:]
        device, dtype = featmap.device, featmap.dtype

        stride = float(base_stride * off_stride)
        shifts_x = torch.arange(w, device=device, dtype=dtype) * stride
        shifts_y = torch.arange(h, device=device, dtype=dtype) * stride

        yy, xx = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shifts = torch.stack(
            (xx.reshape(-1), yy.reshape(-1), xx.reshape(-1), yy.reshape(-1)),
            dim=1,
        )

        base = self._plane_anchors(off_stride, device, dtype)
        anchors = shifts[:, None, :] + base[None, :, :]
        return anchors.reshape(-1, 4)


# =============================================================================
# RPN target assignment
# =============================================================================

def _subsample_rpn_labels(config, labels):
    positive = torch.nonzero(labels == 1, as_tuple=False).squeeze(1)
    negative = torch.nonzero(labels == 0, as_tuple=False).squeeze(1)

    num_pos = min(
        positive.numel(),
        int(config.num_sample_anchors * config.positive_anchor_ratio),
    )
    num_neg = min(
        negative.numel(),
        int(config.num_sample_anchors) - num_pos,
    )

    if positive.numel() > num_pos:
        disable = positive[
            torch.randperm(positive.numel(), device=labels.device)[num_pos:]
        ]
        labels[disable] = config.ignore_label

    if negative.numel() > num_neg:
        disable = negative[
            torch.randperm(negative.numel(), device=labels.device)[num_neg:]
        ]
        labels[disable] = config.ignore_label

    return labels


@torch.no_grad()
def build_rpn_targets(config, gt_boxes, im_info, anchors):
    """
    anchors: [A,4]
    gt_boxes: [max_gt, >=5]
    """
    num_gt = int(im_info[5]) if im_info.numel() > 5 else gt_boxes.shape[0]
    all_gt = gt_boxes[:num_gt]
    if all_gt.numel() > 0:
        all_gt = all_gt.to(device=anchors.device, dtype=anchors.dtype)

    if all_gt.shape[1] >= 5 and all_gt.numel() > 0:
        positive_gt = all_gt[all_gt[:, 4] > 0]
        ignore_gt = all_gt[all_gt[:, 4] == int(config.ignore_label)]
    else:
        positive_gt = all_gt
        ignore_gt = all_gt.new_zeros((0, all_gt.shape[1])) if all_gt.numel() else anchors.new_zeros((0, 5))

    labels = torch.full(
        (anchors.shape[0],),
        int(config.ignore_label),
        dtype=torch.long,
        device=anchors.device,
    )
    targets = anchors.new_zeros((anchors.shape[0], 4))

    def apply_ignore_regions():
        if ignore_gt.numel() == 0:
            return
        _, ignore_ioa = box_overlap_ignore_opr(
            anchors,
            ignore_gt,
            ignore_label=int(config.ignore_label),
        )
        if ignore_ioa.numel() == 0:
            return
        max_ignore_ioa = ignore_ioa.max(dim=1).values
        ignore_anchor = max_ignore_ioa >= float(config.rpn_ignore_overlap)
        # Ignore/crowd regions suppress background anchors but do not erase
        # anchors already assigned to a positive training object.
        labels[(labels == 0) & ignore_anchor] = int(config.ignore_label)

    if positive_gt.numel() == 0:
        labels.fill_(0)
        apply_ignore_regions()
        labels = _subsample_rpn_labels(config, labels)
        return labels, targets

    overlaps = box_overlap_opr(anchors, positive_gt[:, :4])
    max_iou, argmax_gt = overlaps.max(dim=1)

    labels[max_iou < config.rpn_negative_overlap] = 0
    labels[max_iou >= config.rpn_positive_overlap] = 1

    # Low-quality match: mỗi GT phải có ít nhất một positive anchor.
    gt_best_anchor = overlaps.argmax(dim=0)
    labels[gt_best_anchor] = 1
    argmax_gt[gt_best_anchor] = torch.arange(
        positive_gt.shape[0], device=anchors.device, dtype=argmax_gt.dtype
    )

    apply_ignore_regions()

    targets = bbox_transform_opr(
        anchors,
        positive_gt[argmax_gt, :4],
    )

    if getattr(config, "rpn_bbox_normalize_targets", False):
        means = anchors.new_tensor(config.bbox_normalize_means).view(1, 4)
        stds = anchors.new_tensor(config.bbox_normalize_stds).view(1, 4)
        targets = (targets - means) / stds

    labels = _subsample_rpn_labels(config, labels)
    return labels, targets


# =============================================================================
# RPN proposal generation
# =============================================================================

@torch.no_grad()
def generate_rpn_proposals(
    config,
    training: bool,
    pred_bbox_list: List[torch.Tensor],
    pred_cls_list: List[torch.Tensor],
    anchors_list: List[torch.Tensor],
    im_info: torch.Tensor,
):
    pre_nms = (
        config.train_prev_nms_top_n
        if training
        else config.test_prev_nms_top_n
    )
    post_nms = (
        config.train_post_nms_top_n
        if training
        else config.test_post_nms_top_n
    )

    batch_size = pred_cls_list[0].shape[0]
    output = []

    for bid in range(batch_size):
        boxes_all = []
        scores_all = []

        for cls_map, box_map, anchors in zip(
            pred_cls_list, pred_bbox_list, anchors_list
        ):
            cls = cls_map[bid].permute(1, 2, 0).reshape(-1, 2)
            delta = box_map[bid].permute(1, 2, 0).reshape(-1, 4)

            if getattr(config, "rpn_bbox_normalize_targets", False):
                means = delta.new_tensor(config.bbox_normalize_means).view(1, 4)
                stds = delta.new_tensor(config.bbox_normalize_stds).view(1, 4)
                delta = delta * stds + means

            boxes = bbox_transform_inv_opr(anchors, delta)

            # Thực tế nên clip proposal vào ảnh trước NMS.
            boxes = clip_boxes_opr(boxes, im_info[bid])

            score = torch.softmax(cls, dim=1)[:, 1]

            boxes_all.append(boxes)
            scores_all.append(score)

        boxes = torch.cat(boxes_all, dim=0)
        scores = torch.cat(scores_all, dim=0)

        scale = float(im_info[bid, 2]) if im_info.shape[1] > 2 else 1.0
        min_size = float(config.rpn_min_box_size) * scale

        keep = filter_boxes_opr(boxes, min_size)
        boxes = boxes[keep]
        scores = scores[keep]

        if scores.numel() == 0:
            output.append(boxes.new_zeros((0, 5)))
            continue

        topk = min(int(pre_nms), scores.numel())
        scores, order = scores.topk(topk, sorted=True)
        boxes = boxes[order]

        keep = nms(boxes, scores, float(config.rpn_nms_threshold))
        keep = keep[: int(post_nms)]

        boxes = boxes[keep]
        batch_col = boxes.new_full((boxes.shape[0], 1), float(bid))
        output.append(torch.cat((batch_col, boxes), dim=1))

    if len(output) == 1:
        return output[0]

    return torch.cat(output, dim=0)


# =============================================================================
# RPN module
# =============================================================================

class RPN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        rpn_channel = int(config.rpn_channel)
        num_anchors = int(config.num_cell_anchors)

        self.anchors_generator = AnchorGenerator(
            config.anchor_base_size,
            config.anchor_aspect_ratios,
            config.anchor_base_scale,
        )

        self.rpn_conv = nn.Conv2d(
            256, rpn_channel, kernel_size=3, stride=1, padding=1
        )
        self.rpn_cls_score = nn.Conv2d(
            rpn_channel, num_anchors * 2, kernel_size=1
        )
        self.rpn_bbox_offsets = nn.Conv2d(
            rpn_channel, num_anchors * 4, kernel_size=1
        )

        for layer in (
            self.rpn_conv,
            self.rpn_cls_score,
            self.rpn_bbox_offsets,
        ):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, features, im_info, boxes=None):
        pred_cls_list = []
        pred_bbox_list = []

        for x in features:
            t = F.relu(self.rpn_conv(x))
            pred_cls_list.append(self.rpn_cls_score(t))
            pred_bbox_list.append(self.rpn_bbox_offsets(t))

        anchors_list = []
        base_stride = 4
        off_stride = 2 ** (len(features) - 1)

        for fm in features:
            anchors = self.anchors_generator(
                fm,
                base_stride=base_stride,
                off_stride=off_stride,
            )
            anchors_list.append(anchors)
            off_stride //= 2

        rpn_rois = generate_rpn_proposals(
            self.config,
            self.training,
            pred_bbox_list,
            pred_cls_list,
            anchors_list,
            im_info,
        ).type_as(features[0])

        if not self.training:
            return rpn_rois

        if boxes is None:
            raise ValueError("gt boxes are required while RPN is training.")

        cls_flat_all = []
        box_flat_all = []
        labels_all = []
        targets_all = []

        batch_size = pred_cls_list[0].shape[0]

        for bid in range(batch_size):
            per_img_cls = []
            per_img_box = []
            per_img_anchor = []

            for cls_map, box_map, anchors in zip(
                pred_cls_list, pred_bbox_list, anchors_list
            ):
                per_img_cls.append(
                    cls_map[bid].permute(1, 2, 0).reshape(-1, 2)
                )
                per_img_box.append(
                    box_map[bid].permute(1, 2, 0).reshape(-1, 4)
                )
                per_img_anchor.append(anchors)

            cls_flat = torch.cat(per_img_cls, dim=0)
            box_flat = torch.cat(per_img_box, dim=0)
            anchors = torch.cat(per_img_anchor, dim=0)

            labels, targets = build_rpn_targets(
                self.config,
                boxes[bid],
                im_info[bid],
                anchors,
            )

            cls_flat_all.append(cls_flat)
            box_flat_all.append(box_flat)
            labels_all.append(labels)
            targets_all.append(targets)

        pred_cls = torch.cat(cls_flat_all, dim=0)
        pred_bbox = torch.cat(box_flat_all, dim=0)
        labels = torch.cat(labels_all, dim=0)
        targets = torch.cat(targets_all, dim=0)

        valid = labels >= 0
        pos = labels > 0

        if valid.any():
            loss_rpn_cls = F.cross_entropy(
                pred_cls[valid],
                labels[valid],
                reduction="sum",
            ) / valid.sum().clamp(min=1)
        else:
            loss_rpn_cls = pred_cls.sum() * 0.0

        if pos.any():
            loss_rpn_loc = smooth_l1_loss(
                pred_bbox[pos],
                targets[pos],
                float(self.config.rpn_smooth_l1_beta),
            ).sum() / valid.sum().clamp(min=1)
        else:
            loss_rpn_loc = pred_bbox.sum() * 0.0

        return rpn_rois, {
            "loss_rpn_cls": loss_rpn_cls,
            "loss_rpn_loc": loss_rpn_loc,
        }


# =============================================================================
# ROI target assignment
# =============================================================================

def _random_keep(mask: torch.Tensor, max_samples: int):
    inds = torch.nonzero(mask, as_tuple=False).squeeze(1)
    if inds.numel() <= max_samples:
        return inds
    perm = torch.randperm(inds.numel(), device=mask.device)[:max_samples]
    return inds[perm]


@torch.no_grad()
def fpn_roi_target(config, rpn_rois, im_info, gt_boxes, top_k=1):
    if top_k != 1:
        raise NotImplementedError(
            "Self-contained version hiện tối ưu cho top_k=1 như network ODAM."
        )

    return_rois = []
    return_labels = []
    return_bbox_targets = []
    return_gt_assignments = []

    batch_size = int(getattr(config, "train_batch_per_gpu", gt_boxes.shape[0]))

    for bid in range(batch_size):
        num_gt = int(im_info[bid, 5]) if im_info.shape[1] > 5 else gt_boxes.shape[1]
        gt = gt_boxes[bid, :num_gt]

        roi_mask = rpn_rois[:, 0].long() == bid
        proposals = rpn_rois[roi_mask]

        if gt.numel() == 0:
            num_bg = min(int(config.num_rois), proposals.shape[0])
            inds = torch.randperm(
                proposals.shape[0], device=proposals.device
            )[:num_bg]
            rois = proposals[inds]
            labels = torch.zeros(
                len(rois), dtype=torch.long, device=rois.device
            )
            targets = rois.new_zeros((len(rois), 4))
            assignments = torch.zeros(
                len(rois), dtype=torch.long, device=rois.device
            )

            return_rois.append(rois)
            return_labels.append(labels)
            return_bbox_targets.append(targets)
            return_gt_assignments.append(assignments)
            continue

        gt = gt.to(device=proposals.device, dtype=proposals.dtype)

        batch_inds = gt.new_full((gt.shape[0], 1), float(bid))
        gt_rois = torch.cat((batch_inds, gt[:, :4]), dim=1)

        all_rois = torch.cat((proposals, gt_rois), dim=0)

        overlaps_normal, overlaps_ignore = box_overlap_ignore_opr(
            all_rois[:, 1:5],
            gt,
            ignore_label=int(config.ignore_label),
        )

        max_normal, assign_normal = overlaps_normal.max(dim=1)
        max_ignore, assign_ignore = overlaps_ignore.max(dim=1)

        ignore_assign = (
            (max_normal < float(config.fg_threshold))
            & (max_ignore > max_normal)
        )

        max_overlap = torch.where(ignore_assign, max_ignore, max_normal)
        assignment = torch.where(ignore_assign, assign_ignore, assign_normal)

        labels_raw = gt[assignment, 4].long()
        is_ignore_assignment = labels_raw == int(config.ignore_label)

        fg = (
            (max_overlap >= float(config.fg_threshold))
            & (~is_ignore_assignment)
        )

        bg = (
            (max_overlap < float(config.bg_threshold_high))
            & (max_overlap >= float(config.bg_threshold_low))
            & (~is_ignore_assignment)
        )

        max_fg = int(round(float(config.num_rois) * float(config.fg_ratio)))
        fg_inds = _random_keep(fg, max_fg)

        num_bg = max(int(config.num_rois) - fg_inds.numel(), 0)
        bg_inds = _random_keep(bg, num_bg)

        keep = torch.cat((fg_inds, bg_inds), dim=0)

        if keep.numel() == 0:
            return_rois.append(all_rois.new_zeros((0, 5)))
            return_labels.append(
                torch.zeros((0,), dtype=torch.long, device=all_rois.device)
            )
            return_bbox_targets.append(all_rois.new_zeros((0, 4)))
            return_gt_assignments.append(
                torch.zeros((0,), dtype=torch.long, device=all_rois.device)
            )
            continue

        rois = all_rois[keep]
        assignments = assignment[keep]

        labels = labels_raw[keep]
        keep_is_fg = fg[keep]
        labels = torch.where(
            keep_is_fg,
            labels,
            torch.zeros_like(labels),
        )

        target_boxes = gt[assignments, :4]
        bbox_targets = bbox_transform_opr(rois[:, 1:5], target_boxes)

        if getattr(config, "rcnn_bbox_normalize_targets", False):
            means = bbox_targets.new_tensor(
                config.bbox_normalize_means
            ).view(1, 4)
            stds = bbox_targets.new_tensor(
                config.bbox_normalize_stds
            ).view(1, 4)
            bbox_targets = (bbox_targets - means) / stds

        return_rois.append(rois)
        return_labels.append(labels)
        return_bbox_targets.append(bbox_targets)
        return_gt_assignments.append(assignments)

    return (
        torch.cat(return_rois, dim=0),
        torch.cat(return_labels, dim=0),
        torch.cat(return_bbox_targets, dim=0),
        torch.cat(return_gt_assignments, dim=0),
    )


# =============================================================================
# Main detector
# =============================================================================

class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.resnet50 = ResNet50(
            pretrained=bool(getattr(config, "backbone_pretrained", False)),
            has_bias=False,
        )
        self.FPN = FPN(self.resnet50, 2, 6)
        self.RPN = RPN(config)
        self.RCNN = RCNN(config)

    def set_odam_enabled(self, enabled: bool):
        self.RCNN.set_odam_enabled(enabled)

    def set_odam_inference(self, enabled: bool):
        self.RCNN.set_odam_inference(enabled)

    def freeze_backbone(self, freeze_at: int):
        self.resnet50.freeze_backbone(int(freeze_at))

    def forward(self, image, im_info, gt_boxes=None):
        config = self.config

        mean = torch.tensor(
            config.image_mean,
            device=image.device,
            dtype=image.dtype,
        ).view(1, -1, 1, 1)

        std = torch.tensor(
            config.image_std,
            device=image.device,
            dtype=image.dtype,
        ).view(1, -1, 1, 1)

        image = (image - mean) / std
        image = get_padded_tensor(image, 64)

        if self.training:
            if gt_boxes is None:
                raise ValueError("gt_boxes must be provided in training mode.")
            return self._forward_train(image, im_info, gt_boxes)

        return self._forward_test(image, im_info)

    def _forward_train(self, image, im_info, gt_boxes):
        loss_dict = {}

        fpn_fms, _ = self.FPN(image)
        # fpn_fms order: P6,P5,P4,P3,P2
        # strides:       64,32,16,8,4

        rpn_rois, loss_dict_rpn = self.RPN(
            fpn_fms,
            im_info,
            gt_boxes,
        )

        rcnn_rois, rcnn_labels, rcnn_bbox_targets, rcnn_gts = fpn_roi_target(
            self.config,
            rpn_rois,
            im_info,
            gt_boxes,
            top_k=1,
        )

        loss_dict_rcnn = self.RCNN(
            fpn_fms,
            rcnn_rois,
            rcnn_labels,
            rcnn_bbox_targets,
            rcnn_gts,
        )

        loss_dict.update(loss_dict_rpn)
        loss_dict.update(loss_dict_rcnn)
        return loss_dict

    def _forward_test(self, image, im_info):
        fpn_fms, _ = self.FPN(image)
        rpn_rois = self.RPN(fpn_fms, im_info)
        pred_bbox = self.RCNN(fpn_fms, rpn_rois)
        return pred_bbox.detach()


# =============================================================================
# RCNN + ODAM
# =============================================================================

class RCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.odam_enabled = True
        self.odam_inference = False
        self.last_odam_filter_stats = {
            "candidates": 0,
            "kept": 0,
            "keep_ratio": 0.0,
            "reliability_mean": 0.0,
            "reliability_std": 0.0,
            "reliability_p10": 0.0,
            "reliability_p50": 0.0,
            "reliability_p90": 0.0,
            "roi_iou_mean": 0.0,
            "roi_score_mean": 0.0,
            "odam_loss_raw": 0.0,
            "odam_loss_weighted": 0.0,
            "effective_odam_rois": 0.0,
            "low_reliability_fraction": 0.0,
            "high_reliability_fraction": 0.0,
        }

        self.fc1 = nn.Linear(256 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        for layer in (self.fc1, self.fc2):
            nn.init.kaiming_uniform_(layer.weight, a=1)
            nn.init.zeros_(layer.bias)

        self.pred_cls = nn.Linear(1024, config.num_classes)
        self.pred_delta = nn.Linear(1024, config.num_classes * 4)

        nn.init.normal_(self.pred_cls.weight, std=0.01)
        nn.init.zeros_(self.pred_cls.bias)

        nn.init.normal_(self.pred_delta.weight, std=0.001)
        nn.init.zeros_(self.pred_delta.bias)

    def set_odam_enabled(self, enabled: bool):
        self.odam_enabled = bool(enabled)

    def set_odam_inference(self, enabled: bool):
        self.odam_inference = bool(enabled)

    @torch.enable_grad()
    def forward(
        self,
        fpn_fms,
        rcnn_rois,
        labels=None,
        bbox_targets=None,
        assigned_gts=None,
    ):
        config = self.config
        self.last_odam_filter_stats = {
            "candidates": 0,
            "kept": 0,
            "keep_ratio": 0.0,
            "reliability_mean": 0.0,
            "reliability_std": 0.0,
            "reliability_p10": 0.0,
            "reliability_p50": 0.0,
            "reliability_p90": 0.0,
            "roi_iou_mean": 0.0,
            "roi_score_mean": 0.0,
            "odam_loss_raw": 0.0,
            "odam_loss_weighted": 0.0,
            "effective_odam_rois": 0.0,
            "low_reliability_fraction": 0.0,
            "high_reliability_fraction": 0.0,
        }
        bbox_stds = config.bbox_normalize_stds
        bbox_means = config.bbox_normalize_means

        # input order p2-p5
        fpn_fms = fpn_fms[1:][::-1]
        stride = [4, 8, 16, 32]

        max_level = int(math.log2(stride[-1]))
        min_level = int(math.log2(stride[0]))

        level_assignments = assign_boxes_to_levels(
            rcnn_rois,
            min_level,
            max_level,
            224,
            4,
        )

        pool_features = fpn_fms[0].new_zeros(
            (len(rcnn_rois), fpn_fms[0].shape[1], 7, 7)
        )

        for level, (fm_level, scale_level) in enumerate(zip(fpn_fms, stride)):
            inds = torch.nonzero(
                level_assignments == level,
                as_tuple=False,
            ).squeeze(1)

            if inds.numel() == 0:
                continue

            pool_features[inds] = roi_align(
                fm_level,
                rcnn_rois[inds],
                (7, 7),
                spatial_scale=1.0 / scale_level,
                sampling_ratio=-1,
                aligned=True,
            )

        flatten_feature = torch.flatten(pool_features, start_dim=1)
        flatten_feature = F.relu(self.fc1(flatten_feature))
        flatten_feature = F.relu(self.fc2(flatten_feature))

        pred_cls = self.pred_cls(flatten_feature)
        pred_delta = self.pred_delta(flatten_feature)

        if self.training:
            labels = labels.long().flatten()

            fg_masks = labels > 0
            valid_masks = labels >= 0

            # Nếu batch không có foreground, vẫn trả detection losses hợp lệ.
            if not fg_masks.any():
                objectness_loss = softmax_loss(pred_cls, labels)
                objectness_loss = objectness_loss * valid_masks

                normalizer = valid_masks.sum().clamp(min=1).to(pred_cls.dtype)

                return {
                    "loss_rcnn_loc": pred_delta.sum() * 0.0,
                    "loss_rcnn_cls": objectness_loss.sum() / normalizer,
                    # Giữ key ODAM luôn tồn tại để DPGA có API ổn định.
                    "loss_rcnn_match": pred_delta.sum() * 0.0,
                }

            fg_gt_classes = labels[fg_masks]

            pred_delta_reshaped = pred_delta.reshape(
                -1,
                config.num_classes,
                4,
            )
            pred_delta_fg = pred_delta_reshaped[
                fg_masks,
                fg_gt_classes,
                :,
            ]

            localization_loss = smooth_l1_loss(
                pred_delta_fg,
                bbox_targets[fg_masks],
                config.rcnn_smooth_l1_beta,
            )

            pred_bbox = restore_bbox(
                rcnn_rois[fg_masks, 1:5],
                pred_delta_fg,
                bbox_stds,
                bbox_means,
                True,
            )

            gt_bbox = restore_bbox(
                rcnn_rois[fg_masks, 1:5],
                bbox_targets[fg_masks],
                bbox_stds,
                bbox_means,
                True,
            )

            pred_gt_ious = paired_box_overlap_opr(
                pred_bbox,
                gt_bbox,
            )

            objectness_loss = softmax_loss(pred_cls, labels)
            objectness_loss = objectness_loss * valid_masks

            normalizer = valid_masks.sum().clamp(min=1).to(pred_cls.dtype)

            loss_rcnn_loc = localization_loss.sum() / normalizer
            loss_rcnn_cls = objectness_loss.sum() / normalizer

            if not self.odam_enabled:
                return {
                    "loss_rcnn_loc": loss_rcnn_loc,
                    "loss_rcnn_cls": loss_rcnn_cls,
                    "loss_rcnn_match": pred_delta.sum() * 0.0,
                }

            # ODAM gradients
            fg_inds_all = fg_masks.nonzero(as_tuple=True)[0]
            fg_scores = F.softmax(pred_cls, dim=-1)[
                fg_inds_all,
                fg_gt_classes,
            ]
            reliability_all = odam_reliability_weights(
                pred_gt_iou=pred_gt_ious,
                class_scores=fg_scores,
                enabled=bool(getattr(config, "odam_reliability", False)),
                iou_tau=float(
                    getattr(config, "odam_reliability_iou_tau", 0.6)
                ),
                iou_temperature=float(
                    getattr(config, "odam_reliability_iou_temp", 0.1)
                ),
                score_tau=float(
                    getattr(config, "odam_reliability_score_tau", 0.7)
                ),
                score_temperature=float(
                    getattr(config, "odam_reliability_score_temp", 0.1)
                ),
            )
            odam_keep = odam_quality_filter_mask(
                pred_gt_iou=pred_gt_ious,
                class_scores=fg_scores,
                enabled=bool(getattr(config, "odam_filtering", False)),
                min_iou=float(getattr(config, "odam_min_iou", 0.7)),
                min_score=float(getattr(config, "odam_min_score", 0.9)),
            )
            num_candidates = int(odam_keep.numel())
            num_kept = int(odam_keep.sum().detach().cpu())
            kept_reliability = reliability_all[odam_keep]
            kept_ious = pred_gt_ious[odam_keep]
            kept_scores = fg_scores[odam_keep]
            self.last_odam_filter_stats = {
                "candidates": num_candidates,
                "kept": num_kept,
                "keep_ratio": (
                    float(num_kept) / float(max(num_candidates, 1))
                ),
                **odam_reliability_summary(
                    kept_reliability,
                    kept_ious,
                    kept_scores,
                ),
            }
            if num_kept == 0:
                return {
                    "loss_rcnn_loc": loss_rcnn_loc,
                    "loss_rcnn_cls": loss_rcnn_cls,
                    "loss_rcnn_match": pred_delta.sum() * 0.0,
                }

            fg_inds = fg_inds_all[odam_keep]
            fg_gt_classes_odam = fg_gt_classes[odam_keep]
            assigned_gts_fg = assigned_gts[fg_masks][odam_keep]
            fg_batch_ids = rcnn_rois[fg_masks, 0].long()[odam_keep]
            pred_bbox_odam = pred_bbox[odam_keep]
            pred_gt_ious_odam = pred_gt_ious[odam_keep]
            reliability_odam = kept_reliability
            pool_grads = self.get_gradient(
                pred_cls,
                pool_features,
            )

            pool_dams = F.relu(
                (
                    pool_grads[
                        fg_inds,
                        fg_gt_classes_odam - 1,
                        :, :, :
                    ]
                    * pool_features[fg_inds]
                ).sum(1)
            )

            # common DAM size = stride-16 FPN map
            dam_size = fpn_fms[2].size()[-2:]

            rois_fg = rcnn_rois[fg_inds, 1:5]
            bids = rcnn_rois[fg_inds, 0].long()
            level_assignments_fg = level_assignments[fg_inds]

            pred_dams = get_dams(
                pool_dams,
                bids,
                rois_fg,
                fpn_fms,
                stride,
                level_assignments_fg,
                dam_size,
            )

            loss_rcnn_match, loss_rcnn_match_raw = match_loss(
                pred_dams,
                assigned_gts_fg,
                fg_batch_ids,
                pred_bbox_odam,
                pred_gt_ious_odam,
                reliability=reliability_odam,
                return_raw=True,
            )
            self.last_odam_filter_stats["odam_loss_raw"] = float(
                loss_rcnn_match_raw.detach().cpu()
            )
            self.last_odam_filter_stats["odam_loss_weighted"] = float(
                loss_rcnn_match.detach().cpu()
            )

            # DPGA NOTE:
            # Không nhân cố định 0.2 ở đây.
            # Trả raw ODAM loss để DPGA kiểm soát contribution bằng:
            #   warm-up/ramp-up + cosine gate + projection + norm cap.

            loss_dict = {
                "loss_rcnn_loc": loss_rcnn_loc,
                "loss_rcnn_cls": loss_rcnn_cls,
            }

            # match_loss luôn trả Tensor trong self-contained version
            if torch.is_tensor(loss_rcnn_match):
                loss_dict["loss_rcnn_match"] = loss_rcnn_match

            return loss_dict

        # ---------------------------------------------------------------------
        # Inference
        # ---------------------------------------------------------------------
        if pred_cls.shape[0] == 0:
            return pred_cls.new_zeros((0, 8))

        class_num = pred_cls.shape[-1] - 1

        tag = (
            torch.arange(
                1,
                class_num + 1,
                device=pred_cls.device,
                dtype=pred_cls.dtype,
            )
            .repeat(pred_cls.shape[0])
            .reshape(-1, 1)
        )

        pred_scores = (
            F.softmax(pred_cls, dim=-1)[:, 1:]
            .reshape(-1, 1)
        )

        # Bỏ background regression (4 giá trị đầu).
        pred_delta_fg = pred_delta[:, 4:].reshape(-1, 4)

        base_rois = (
            rcnn_rois[:, None, :]
            .repeat(1, class_num, 1)
            .reshape(-1, 5)
        )

        keep = pred_scores[:, 0] > config.pred_cls_threshold

        pred_scores = pred_scores[keep]
        pred_delta_fg = pred_delta_fg[keep]
        base_rois = base_rois[keep]
        tag = tag[keep]

        bids = base_rois[:, 0].long()
        base_boxes = base_rois[:, 1:5]

        pred_bbox = restore_bbox(
            base_boxes,
            pred_delta_fg,
            bbox_stds,
            bbox_means,
            True,
        )

        if not self.odam_inference:
            return torch.cat(
                (
                    pred_bbox,
                    pred_scores,
                    tag,
                ),
                dim=1,
            )

        pool_grads = self.get_gradient(pred_cls, pool_features)

        level_assignments_exp = (
            level_assignments[:, None]
            .repeat(1, class_num)
            .reshape(-1)
        )

        level_assignments_exp = level_assignments_exp[keep]

        pool_grads = pool_grads.reshape(-1, 256, 7, 7)[keep]

        pred_index = (
            torch.arange(
                pred_cls.shape[0],
                device=pred_cls.device,
            )[:, None]
            .repeat(1, class_num)
            .reshape(-1)[keep]
            .long()
        )

        pool_dams = F.relu(
            (
                pool_grads
                * pool_features[pred_index]
            ).sum(1)
        )

        dam_size = fpn_fms[1].size()[-2:]

        pred_dams = get_dams(
            pool_dams,
            bids,
            base_boxes,
            fpn_fms,
            stride,
            level_assignments_exp,
            dam_size,
        )

        dam_size_tensor = pred_dams.new_tensor(
            dam_size
        ).repeat(len(pred_scores), 1)

        return torch.cat(
            (
                pred_bbox,
                pred_scores,
                tag,
                pred_dams,
                dam_size_tensor,
            ),
            dim=1,
        )

    def get_gradient(self, pred, pool_features):
        grads = []

        with torch.enable_grad():
            for c in range(1, self.config.num_classes):
                grad_mask = pred.new_zeros(pred.shape)
                grad_mask[:, c] = 1.0

                grad = torch.autograd.grad(
                    pred,
                    pool_features,
                    grad_outputs=grad_mask,
                    retain_graph=True,
                    create_graph=False,
                )[0]

                grads.append(grad)

        return torch.stack(grads, dim=1)


# =============================================================================
# DAM projection
# =============================================================================

def get_dams(
    pool_maps,
    bids,
    rois,
    fpn_fms,
    stride,
    level_assignments,
    dam_size,
):
    if len(pool_maps) == 0:
        return pool_maps.new_zeros(
            (0, int(dam_size[0]) * int(dam_size[1]))
        )

    resize = transforms.Resize(dam_size)

    pred_dams = pool_maps.new_zeros(
        len(pool_maps),
        int(dam_size[0]) * int(dam_size[1]),
    )

    for bid in bids.unique():
        inds = torch.nonzero(
            bids == bid,
            as_tuple=True,
        )[0]

        for level, (fm_level, scale_level) in enumerate(
            zip(fpn_fms, stride)
        ):
            inds_level = inds[
                level_assignments[inds] == level
            ]

            if len(inds_level) == 0:
                continue

            dam_maps = roi_align_inv(
                pool_maps[inds_level],
                rois[inds_level],
                1.0 / scale_level,
                fm_level.size()[-2:],
            )

            dam_maps = resize(dam_maps)

            flat = dam_maps.reshape(len(inds_level), -1)
            dam_maps = F.normalize(flat, p=2, dim=1)

            pred_dams[inds_level, :] = dam_maps

    return pred_dams


def restore_bbox(rois, deltas, stds, means, unnormalize=True):
    if unnormalize:
        std_opr = deltas.new_tensor(stds).view(1, 4)
        mean_opr = deltas.new_tensor(means).view(1, 4)
        deltas = deltas * std_opr + mean_opr

    return bbox_transform_inv_opr(rois, deltas)


def roi_align_inv(pool_dams, rois, scale, map_size):
    """
    Project DAM 7x7 từ ROI space trở lại feature-map space.

    pool_dams: [N, 7, 7]
    rois:      [N, 4] x1,y1,x2,y2 (image coordinates)
    scale:     1 / feature_stride
    map_size:  (H, W)
    """
    if pool_dams.numel() == 0:
        return pool_dams.new_zeros(
            (0, int(map_size[0]), int(map_size[1]))
        )

    N, h_pool, w_pool = pool_dams.shape

    rois = rois.clone() * scale

    rois[:, 0::2] = rois[:, 0::2].clamp(
        min=0.0,
        max=map_size[1] - 1,
    )
    rois[:, 1::2] = rois[:, 1::2].clamp(
        min=0.0,
        max=map_size[0] - 1,
    )

    rois_x_low = rois[:, 0].floor()
    rois_y_low = rois[:, 1].floor()
    rois_x_high = rois[:, 2].ceil()
    rois_y_high = rois[:, 3].ceil()

    rois_w_max = int(
        (rois_x_high - rois_x_low)
        .max()
        .clamp(min=0)
        .item()
    ) + 1

    rois_h_max = int(
        (rois_y_high - rois_y_low)
        .max()
        .clamp(min=0)
        .item()
    ) + 1

    shift_y, shift_x = torch.meshgrid(
        torch.arange(
            rois_h_max,
            dtype=rois.dtype,
            device=rois.device,
        ),
        torch.arange(
            rois_w_max,
            dtype=rois.dtype,
            device=rois.device,
        ),
        indexing="ij",
    )

    roi_grid = torch.stack(
        (shift_x.reshape(-1), shift_y.reshape(-1)),
        dim=1,
    )

    M = roi_grid.shape[0]

    roi_start = torch.stack(
        (rois_x_low, rois_y_low),
        dim=1,
    )

    roi_grid = (
        roi_grid.unsqueeze(0).repeat(N, 1, 1)
        + roi_start[:, None, :]
    )

    roi_extent = (
        rois[:, 2:] - rois[:, :2]
    ).clamp(min=1e-6)

    grid_on_pool = (
        (roi_grid - rois[:, None, :2])
        / roi_extent[:, None, :]
    )

    grid_on_pool = (
        grid_on_pool
        * rois.new_tensor(
            [w_pool - 1, h_pool - 1]
        ).view(1, 1, 2)
    )

    x = grid_on_pool[:, :, 0]
    y = grid_on_pool[:, :, 1]

    x0 = x.floor()
    x1 = x.ceil()
    y0 = y.floor()
    y1 = y.ceil()

    valid = (
        (x0 >= 0)
        & (x1 < w_pool)
        & (y0 >= 0)
        & (y1 < h_pool)
    )

    ids, vids = valid.nonzero(as_tuple=True)

    flat_pool = pool_dams.reshape(N, -1)

    wx = x - x0
    wy = y - y0

    values = pool_dams.new_zeros((N, M))

    def sample(xx, yy):
        out = pool_dams.new_zeros((N, M))
        loc = (
            yy[ids, vids] * w_pool
            + xx[ids, vids]
        ).long()
        out[ids, vids] = flat_pool[ids, loc]
        return out

    tl = sample(x0, y0)
    tr = sample(x1, y0)
    bl = sample(x0, y1)
    br = sample(x1, y1)

    values = (
        tl * (1 - wx) * (1 - wy)
        + tr * wx * (1 - wy)
        + bl * (1 - wx) * wy
        + br * wx * wy
    )

    px = roi_grid[:, :, 0].long()
    py = roi_grid[:, :, 1].long()

    spatial_valid = (
        (px >= 0)
        & (px < map_size[1])
        & (py >= 0)
        & (py < map_size[0])
    )

    ids, vids = spatial_valid.nonzero(as_tuple=True)

    output = pool_dams.new_zeros(
        (N, map_size[0] * map_size[1])
    )

    flat_index = (
        py[ids, vids] * map_size[1]
        + px[ids, vids]
    )

    output[ids, flat_index] = values[ids, vids]

    return output.reshape(
        N,
        map_size[0],
        map_size[1],
    )


# =============================================================================
# ODAM matching loss
# =============================================================================

def image_aware_object_ids(
    gt_ids: torch.Tensor,
    batch_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Map local GT ids to object ids unique within a multi-image batch.
    """
    gt_ids = gt_ids.long().flatten()
    batch_ids = batch_ids.long().flatten()

    if gt_ids.numel() != batch_ids.numel():
        raise ValueError(
            "gt_ids and batch_ids must have the same number of elements"
        )

    if gt_ids.numel() == 0:
        return gt_ids.new_zeros((0,), dtype=torch.long)

    object_keys = torch.stack(
        (batch_ids, gt_ids),
        dim=1,
    )
    _, object_ids = torch.unique(
        object_keys,
        dim=0,
        return_inverse=True,
    )
    return object_ids.long()


def image_aware_pair_masks(
    gt_ids: torch.Tensor,
    batch_ids: torch.Tensor,
    pred_bbox: torch.Tensor,
    negative_iou_threshold: float = 0.0,
    include_self_pairs: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build ODAM pair masks without crossing image boundaries.

    Returns:
        same_obj: same image and same (batch_id, gt_id) object.
        neg_mask: same image, overlapping boxes, different object.
        object_ids: compact ids for (batch_id, gt_id).
    """
    gt_ids = gt_ids.long().flatten()
    batch_ids = batch_ids.long().flatten()

    if gt_ids.numel() != batch_ids.numel():
        raise ValueError(
            "gt_ids and batch_ids must have the same number of elements"
        )

    object_ids = image_aware_object_ids(gt_ids, batch_ids)
    M = int(gt_ids.numel())
    if M == 0:
        empty = torch.zeros(
            (0, 0),
            dtype=torch.bool,
            device=gt_ids.device,
        )
        return empty, empty, object_ids

    same_image = batch_ids[:, None] == batch_ids[None, :]
    same_object_full = same_image & (
        object_ids[:, None] == object_ids[None, :]
    )

    eye = torch.eye(
        M,
        dtype=torch.bool,
        device=gt_ids.device,
    )

    same_obj = same_object_full
    if not include_self_pairs:
        same_obj = same_object_full & (~eye)

    pred_paired_iou = box_overlap_opr(
        pred_bbox,
        pred_bbox,
    )
    overlap_mask = pred_paired_iou > float(negative_iou_threshold)
    neg_mask = same_image & overlap_mask & (~same_object_full) & (~eye)
    assert not torch.diag(neg_mask).any()

    return same_obj, neg_mask, object_ids


def odam_quality_filter_mask(
    pred_gt_iou: torch.Tensor,
    class_scores: torch.Tensor,
    enabled: bool,
    min_iou: float,
    min_score: float,
) -> torch.Tensor:
    """
    Select foreground proposals allowed to participate in ODAM loss only.
    """
    pred_gt_iou = pred_gt_iou.flatten()
    class_scores = class_scores.flatten()

    if pred_gt_iou.numel() != class_scores.numel():
        raise ValueError(
            "pred_gt_iou and class_scores must have the same number of elements"
        )

    if not enabled:
        return torch.ones(
            pred_gt_iou.shape,
            dtype=torch.bool,
            device=pred_gt_iou.device,
        )

    return (
        (pred_gt_iou >= float(min_iou))
        & (class_scores >= float(min_score))
    )


def odam_reliability_weights(
    pred_gt_iou: torch.Tensor,
    class_scores: torch.Tensor,
    enabled: bool,
    iou_tau: float,
    iou_temperature: float,
    score_tau: float,
    score_temperature: float,
) -> torch.Tensor:
    """
    Soft controller weights for ODAM-only supervision.

    The weights are detached by design: reliability should describe proposal
    quality, not introduce a new path that lets the detector minimize ODAM by
    lowering its own confidence.
    """
    pred_gt_iou = pred_gt_iou.flatten()
    class_scores = class_scores.flatten()

    if pred_gt_iou.numel() != class_scores.numel():
        raise ValueError(
            "pred_gt_iou and class_scores must have the same number of elements"
        )
    if float(iou_temperature) <= 0.0:
        raise ValueError("iou_temperature must be > 0")
    if float(score_temperature) <= 0.0:
        raise ValueError("score_temperature must be > 0")

    if not enabled:
        return torch.ones(
            pred_gt_iou.shape,
            dtype=pred_gt_iou.dtype,
            device=pred_gt_iou.device,
        )

    iou_weight = torch.sigmoid(
        (pred_gt_iou - float(iou_tau)) / float(iou_temperature)
    )
    score_weight = torch.sigmoid(
        (class_scores - float(score_tau)) / float(score_temperature)
    )
    return (iou_weight * score_weight).detach()


def _stat_float(values: torch.Tensor, fn, default: float = 0.0) -> float:
    if values.numel() == 0:
        return default
    return float(fn(values).detach().cpu())


def odam_reliability_summary(
    reliability: torch.Tensor,
    pred_gt_iou: torch.Tensor,
    class_scores: torch.Tensor,
) -> dict:
    reliability = reliability.flatten().detach()
    pred_gt_iou = pred_gt_iou.flatten().detach()
    class_scores = class_scores.flatten().detach()

    if reliability.numel() == 0:
        return {
            "reliability_mean": 0.0,
            "reliability_std": 0.0,
            "reliability_p10": 0.0,
            "reliability_p50": 0.0,
            "reliability_p90": 0.0,
            "roi_iou_mean": 0.0,
            "roi_score_mean": 0.0,
            "effective_odam_rois": 0.0,
            "low_reliability_fraction": 0.0,
            "high_reliability_fraction": 0.0,
            "odam_loss_raw": 0.0,
            "odam_loss_weighted": 0.0,
        }

    rel_float = reliability.float()
    return {
        "reliability_mean": _stat_float(rel_float, torch.mean),
        "reliability_std": _stat_float(
            rel_float,
            lambda value: torch.std(value, unbiased=False),
        ),
        "reliability_p10": _stat_float(
            rel_float,
            lambda value: torch.quantile(value, 0.10),
        ),
        "reliability_p50": _stat_float(
            rel_float,
            lambda value: torch.quantile(value, 0.50),
        ),
        "reliability_p90": _stat_float(
            rel_float,
            lambda value: torch.quantile(value, 0.90),
        ),
        "roi_iou_mean": _stat_float(pred_gt_iou.float(), torch.mean),
        "roi_score_mean": _stat_float(class_scores.float(), torch.mean),
        "effective_odam_rois": float(rel_float.sum().detach().cpu()),
        "low_reliability_fraction": _stat_float(
            (rel_float < 0.25).float(),
            torch.mean,
        ),
        "high_reliability_fraction": _stat_float(
            (rel_float >= 0.75).float(),
            torch.mean,
        ),
        "odam_loss_raw": 0.0,
        "odam_loss_weighted": 0.0,
    }


def match_loss(
    dams: torch.Tensor,
    gt_ids: torch.Tensor,
    batch_ids: torch.Tensor,
    pred_bbox: torch.Tensor,
    pred_gt_iou: torch.Tensor,
    reliability: Optional[torch.Tensor] = None,
    return_raw: bool = False,
):
    """
    Positive:
        prediction thuộc cùng GT -> DAM similarity -> 1
    Negative:
        prediction thuộc GT khác nhưng bbox overlap -> DAM similarity -> 0
    """
    zero_loss = dams.sum() * 0.0 + pred_bbox.sum() * 0.0
    def maybe_tuple(loss: torch.Tensor, raw_loss: torch.Tensor):
        return (loss, raw_loss) if return_raw else loss

    if dams.numel() == 0:
        return maybe_tuple(zero_loss, zero_loss)

    M, _ = dams.shape

    gt_ids = gt_ids.long()
    batch_ids = batch_ids.long()
    if reliability is None:
        reliability = pred_gt_iou.new_ones((M,))
    else:
        reliability = reliability.flatten().to(
            device=pred_gt_iou.device,
            dtype=pred_gt_iou.dtype,
        )
        if reliability.numel() != M:
            raise ValueError("reliability must have one value per DAM/ROI")
        reliability = reliability.detach()

    if gt_ids.numel() == 0:
        return maybe_tuple(zero_loss, zero_loss)

    same_obj, neg_mask, object_ids = image_aware_pair_masks(
        gt_ids,
        batch_ids,
        pred_bbox,
        include_self_pairs=True,
    )

    num_gt = int(object_ids.max().item()) + 1

    ious = pred_gt_iou.new_zeros((M, num_gt))
    ious[
        torch.arange(M, device=object_ids.device),
        object_ids,
    ] = pred_gt_iou

    max_iou, max_position = ious.max(dim=0)

    valid_gt = max_iou > 0
    max_position = max_position[valid_gt]
    max_iou = max_iou[valid_gt]

    if max_position.numel() == 0:
        return maybe_tuple(zero_loss, zero_loss)

    # Rows là reference proposal tốt nhất của từng GT.
    pos_pair1, pos_pair2 = same_obj[
        max_position
    ].nonzero(as_tuple=True)

    neg_pair1, neg_pair2 = neg_mask[
        max_position
    ].nonzero(as_tuple=True)

    ref_dams = dams[max_position]

    pos_sims = (
        ref_dams[pos_pair1]
        * dams[pos_pair2]
    ).sum(-1).clamp(min=1e-4, max=1 - 1e-4)

    neg_sims = (
        ref_dams[neg_pair1]
        * dams[neg_pair2]
    ).sum(-1).clamp(min=1e-4, max=1 - 1e-4)

    if pos_sims.numel() > 0:
        pos_weights = (
            pred_gt_iou[pos_pair2]
            / max_iou[pos_pair1].clamp(min=1e-4)
        )

        pos_losses = (
            pos_weights
            * (-torch.log(pos_sims))
        )
        pos_term = pos_losses.sum()
        pos_reliability = reliability[pos_pair2]
        weighted_pos_term = (pos_reliability * pos_losses).sum()
        pos_weight_denom = pos_reliability.sum()
    else:
        pos_term = zero_loss
        weighted_pos_term = zero_loss
        pos_weight_denom = zero_loss

    if neg_sims.numel() > 0:
        neg_losses = -torch.log(1 - neg_sims)
        neg_term = neg_losses.sum()
        neg_reliability = reliability[neg_pair2]
        weighted_neg_term = (neg_reliability * neg_losses).sum()
        neg_weight_denom = neg_reliability.sum()
    else:
        neg_term = zero_loss
        weighted_neg_term = zero_loss
        neg_weight_denom = zero_loss

    denom = max(
        1,
        int(pos_sims.numel() + neg_sims.numel()),
    )

    raw_loss = (pos_term + neg_term) / denom
    weight_denom = (pos_weight_denom + neg_weight_denom).clamp(min=1e-12)
    weighted_loss = (weighted_pos_term + weighted_neg_term) / weight_denom

    return maybe_tuple(weighted_loss, raw_loss)


# =============================================================================
# Minimal smoke-test helpers
# =============================================================================

def validate_config(config):
    """
    Kiểm tra nhanh các field mà Network này cần.
    """
    required = [
        "image_mean",
        "image_std",
        "backbone_freeze_at",
        "num_classes",
        "rpn_channel",
        "anchor_base_size",
        "anchor_base_scale",
        "anchor_aspect_ratios",
        "num_cell_anchors",
        "rpn_min_box_size",
        "rpn_nms_threshold",
        "train_prev_nms_top_n",
        "train_post_nms_top_n",
        "test_prev_nms_top_n",
        "test_post_nms_top_n",
        "num_sample_anchors",
        "positive_anchor_ratio",
        "rpn_positive_overlap",
        "rpn_negative_overlap",
        "rpn_ignore_overlap",
        "rpn_bbox_normalize_targets",
        "rpn_smooth_l1_beta",
        "ignore_label",
        "num_rois",
        "fg_ratio",
        "fg_threshold",
        "bg_threshold_high",
        "bg_threshold_low",
        "rcnn_bbox_normalize_targets",
        "bbox_normalize_means",
        "bbox_normalize_stds",
        "rcnn_smooth_l1_beta",
        "pred_cls_threshold",
        "train_batch_per_gpu",
    ]

    missing = [
        name for name in required
        if not hasattr(config, name)
    ]

    if missing:
        raise AttributeError(
            "Config thiếu các field: "
            + ", ".join(missing)
        )

    if int(config.num_cell_anchors) != (
        len(config.anchor_aspect_ratios)
        * (
            len(config.anchor_base_scale)
            if isinstance(config.anchor_base_scale, (list, tuple))
            else 1
        )
    ):
        raise ValueError(
            "config.num_cell_anchors không khớp số "
            "ratio × scale."
        )

    if int(config.backbone_freeze_at) not in (0, 1, 2):
        raise ValueError("backbone_freeze_at must be one of {0, 1, 2}")
    if not 0.0 <= float(config.rpn_ignore_overlap) <= 1.0:
        raise ValueError("rpn_ignore_overlap must be in [0, 1]")
    if not 0.0 <= float(config.rpn_negative_overlap) <= 1.0:
        raise ValueError("rpn_negative_overlap must be in [0, 1]")
    if not 0.0 <= float(config.rpn_positive_overlap) <= 1.0:
        raise ValueError("rpn_positive_overlap must be in [0, 1]")
    if float(config.rpn_negative_overlap) > float(config.rpn_positive_overlap):
        raise ValueError("rpn_negative_overlap must be <= rpn_positive_overlap")
    if not 0.0 <= float(getattr(config, "odam_min_iou", 0.7)) <= 1.0:
        raise ValueError("odam_min_iou must be in [0, 1]")
    if not 0.0 <= float(getattr(config, "odam_min_score", 0.9)) <= 1.0:
        raise ValueError("odam_min_score must be in [0, 1]")
    if not 0.0 <= float(getattr(config, "odam_reliability_iou_tau", 0.6)) <= 1.0:
        raise ValueError("odam_reliability_iou_tau must be in [0, 1]")
    if not 0.0 <= float(getattr(config, "odam_reliability_score_tau", 0.7)) <= 1.0:
        raise ValueError("odam_reliability_score_tau must be in [0, 1]")
    if float(getattr(config, "odam_reliability_iou_temp", 0.1)) <= 0.0:
        raise ValueError("odam_reliability_iou_temp must be > 0")
    if float(getattr(config, "odam_reliability_score_temp", 0.1)) <= 0.0:
        raise ValueError("odam_reliability_score_temp must be > 0")

    return True


# =============================================================================
# DPGA: Detection-Priority Gradient Alignment
# =============================================================================
#
# ODAM gốc thường tối ưu:
#
#     L_total = L_det + lambda * L_odam
#
# nên gradient:
#
#     g_total = g_det + lambda * g_odam
#
# DPGA không cộng gradient ODAM một cách không kiểm soát. Với mỗi module m:
#
#     1) g_d^(m) = d L_det  / d theta_m
#     2) g_o^(m) = d L_odam / d theta_m
#     3) đo cosine(g_d, g_o)
#     4) nếu conflict -> project phần ODAM chống lại detection
#     5) norm-cap:
#           ||g_o_safe|| <= rho_m ||g_d||
#     6) adaptive gate + warm-up/ramp-up
#     7) compose:
#           g_final = g_d + alpha(epoch) * gate * g_o_safe
#
# Detection gradient được giữ nguyên; chỉ auxiliary gradient bị điều chỉnh.
# =============================================================================


@dataclass
class DPGAModulePolicy:
    """
    Chính sách DPGA cho một module.

    rho:
        Giới hạn relative norm:
            ||g_odam_safe|| <= rho * ||g_det||

        rho = 0 -> không cho ODAM tác động module này.

    tau:
        Ngưỡng cosine của adaptive gate.

    temperature:
        Độ mềm của sigmoid gate:
            gate = sigmoid((cos - tau) / temperature)
    """
    rho: float = 0.20
    tau: float = 0.0
    temperature: float = 0.20

    def validate(self):
        if self.rho < 0:
            raise ValueError("DPGA rho must be >= 0")
        if self.temperature <= 0:
            raise ValueError("DPGA temperature must be > 0")


@dataclass
class DPGAConfig:
    """
    Cấu hình DPGA.

    warmup_epochs:
        alpha = 0 trong giai đoạn đầu -> detection-only.

    rampup_epochs:
        Sau warm-up, alpha tăng tuyến tính đến alpha_max.

    alpha_max:
        Global ODAM gradient multiplier tối đa.

    conflict_threshold:
        Thường = 0. Nếu cosine < 0, hai gradient xung đột.
    """
    warmup_epochs: int = 3
    rampup_epochs: int = 3
    alpha_max: float = 1.0

    project_if_conflict: bool = True
    use_norm_cap: bool = True
    use_gate: bool = True
    conflict_threshold: float = 0.0
    eps: float = 1e-12

    default_policy: DPGAModulePolicy = field(
        default_factory=lambda: DPGAModulePolicy(
            rho=0.20,
            tau=0.0,
            temperature=0.20,
        )
    )

    # Preset detection-priority bảo thủ.
    #
    # Lưu ý với Network hiện tại:
    # - proposal generation của RPN dùng @torch.no_grad(), nên ODAM gradient
    #   thường không đi ngược vào RPN.
    # - RCNN.get_gradient(... create_graph=False), nên ODAM không dùng
    #   second-order graph qua gradient map; roi_cls có thể nhận ODAM gradient
    #   bằng 0. Đây là hành vi cố ý giữ tương thích ODAM hiện tại.
    module_policies: Dict[str, DPGAModulePolicy] = field(
        default_factory=lambda: {
            "backbone": DPGAModulePolicy(rho=0.10, tau=0.00, temperature=0.20),
            "fpn": DPGAModulePolicy(rho=0.15, tau=0.00, temperature=0.20),
            "rpn": DPGAModulePolicy(rho=0.00, tau=0.00, temperature=0.20),
            "roi_shared": DPGAModulePolicy(rho=0.25, tau=0.00, temperature=0.20),
            "roi_cls": DPGAModulePolicy(rho=0.10, tau=0.00, temperature=0.20),
            "roi_reg": DPGAModulePolicy(rho=0.05, tau=0.00, temperature=0.20),
        }
    )

    def validate(self):
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        if self.rampup_epochs < 0:
            raise ValueError("rampup_epochs must be >= 0")
        if self.alpha_max < 0:
            raise ValueError("alpha_max must be >= 0")
        if self.eps <= 0:
            raise ValueError("eps must be > 0")

        self.default_policy.validate()
        for policy in self.module_policies.values():
            policy.validate()


@dataclass
class DPGAModuleStats:
    name: str
    cosine_before: float
    cosine_projected: float
    cosine_after: float

    det_norm: float
    odam_norm_before: float
    odam_norm_after_projection: float
    odam_norm_after_cap: float
    final_norm: float

    projected: bool
    cap_active: bool
    norm_scale: float
    gate: float
    alpha: float
    effective_weight: float


@dataclass
class DPGAStats:
    alpha: float
    modules: Dict[str, DPGAModuleStats]
    gradient_scope: str = "local"
    world_size: int = 1


# -----------------------------------------------------------------------------
# Gradient-list operations
# -----------------------------------------------------------------------------

def _dpga_replace_none(
    grads: Sequence[Optional[torch.Tensor]],
    params: Sequence[nn.Parameter],
) -> List[torch.Tensor]:
    return [
        torch.zeros_like(p, memory_format=torch.preserve_format)
        if g is None else g
        for p, g in zip(params, grads)
    ]


def _dpga_zeros_like(params: Sequence[nn.Parameter]) -> List[torch.Tensor]:
    return [
        torch.zeros_like(p, memory_format=torch.preserve_format)
        for p in params
    ]


def _dpga_dot(
    xs: Sequence[torch.Tensor],
    ys: Sequence[torch.Tensor],
) -> torch.Tensor:
    if len(xs) != len(ys):
        raise ValueError("DPGA gradient lists must have same length")
    if len(xs) == 0:
        return torch.tensor(0.0)

    out = xs[0].new_zeros(())
    for x, y in zip(xs, ys):
        out = out + torch.sum(x * y)
    return out


def _dpga_sq_norm(xs: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(xs) == 0:
        return torch.tensor(0.0)

    out = xs[0].new_zeros(())
    for x in xs:
        out = out + torch.sum(x * x)
    return out


def _dpga_norm(xs: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(_dpga_sq_norm(xs))


def _dpga_cosine(
    xs: Sequence[torch.Tensor],
    ys: Sequence[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    if len(xs) == 0:
        return torch.tensor(0.0)

    dot = _dpga_dot(xs, ys)
    nx = _dpga_norm(xs)
    ny = _dpga_norm(ys)
    denom = nx * ny

    if float(denom.detach().cpu()) <= eps:
        return dot.new_zeros(())

    return dot / (denom + eps)


def _dpga_distributed_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size())
    return 1


def _dpga_allreduce_buckets(
    grads: Sequence[torch.Tensor],
    max_bucket_bytes: int = DPGA_ALLREDUCE_BUCKET_BYTES,
) -> List[List[int]]:
    if int(max_bucket_bytes) <= 0:
        raise ValueError("max_bucket_bytes must be > 0")

    grouped: Dict[Tuple[torch.device, torch.dtype], List[List[int]]] = {}
    active: Dict[Tuple[torch.device, torch.dtype], List[int]] = {}
    active_bytes: Dict[Tuple[torch.device, torch.dtype], int] = {}

    for idx, grad in enumerate(grads):
        key = (grad.device, grad.dtype)
        nbytes = int(grad.numel()) * int(grad.element_size())
        nbytes = max(nbytes, 1)

        if key not in active:
            active[key] = []
            active_bytes[key] = 0

        if (
            active[key]
            and active_bytes[key] + nbytes > int(max_bucket_bytes)
        ):
            grouped.setdefault(key, []).append(active[key])
            active[key] = []
            active_bytes[key] = 0

        active[key].append(idx)
        active_bytes[key] += nbytes

    for key, indices in active.items():
        if indices:
            grouped.setdefault(key, []).append(indices)

    buckets = []
    for per_dtype_buckets in grouped.values():
        buckets.extend(per_dtype_buckets)
    return buckets


def _dpga_allreduce_mean(
    grads: Sequence[torch.Tensor],
    max_bucket_bytes: int = DPGA_ALLREDUCE_BUCKET_BYTES,
) -> List[torch.Tensor]:
    """
    Average gradient tensors across DDP ranks before DPGA composition.

    DPGA is nonlinear, so averaging the already-composed final gradients is not
    equivalent to composing from globally averaged detection/ODAM gradients.
    Tensors are bucketed by device/dtype and bounded size to avoid one
    collective per parameter without creating a single very large flat buffer.
    """
    world_size = _dpga_distributed_world_size()
    out = [g.detach().clone() for g in grads]
    if world_size <= 1:
        return out

    with torch.no_grad():
        for indices in _dpga_allreduce_buckets(
            out,
            max_bucket_bytes=max_bucket_bytes,
        ):
            if not indices:
                continue
            flat = torch.cat([
                out[idx].reshape(-1)
                for idx in indices
            ])
            dist.all_reduce(
                flat,
                op=dist.ReduceOp.SUM,
            )
            flat.div_(world_size)

            offset = 0
            for idx in indices:
                numel = out[idx].numel()
                out[idx].copy_(
                    flat[offset: offset + numel].view_as(out[idx])
                )
                offset += numel
    return out


def allreduce_gradient_list_mean(
    grads: Sequence[torch.Tensor],
    max_bucket_bytes: int = DPGA_ALLREDUCE_BUCKET_BYTES,
) -> List[torch.Tensor]:
    return _dpga_allreduce_mean(
        grads,
        max_bucket_bytes=max_bucket_bytes,
    )


# -----------------------------------------------------------------------------
# Module grouping
# -----------------------------------------------------------------------------

def build_dpga_groups(model: nn.Module) -> Dict[str, List[nn.Parameter]]:
    """
    Chia Faster R-CNN thành các module DPGA.

    Quan trọng:
        model.FPN.bottom_up chính là model.resnet50, vì vậy không dùng
        model.FPN.parameters() toàn bộ nếu không backbone sẽ bị trùng.
    """
    required = ("resnet50", "FPN", "RPN", "RCNN")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise AttributeError(
            "Model không tương thích DPGA. Thiếu: " + ", ".join(missing)
        )

    groups = {
        "backbone": [
            p for p in model.resnet50.parameters()
            if p.requires_grad
        ],
        "fpn": [
            p
            for module in (
                model.FPN.lateral_convs,
                model.FPN.output_convs,
            )
            for p in module.parameters()
            if p.requires_grad
        ],
        "rpn": [
            p for p in model.RPN.parameters()
            if p.requires_grad
        ],
        "roi_shared": [
            p
            for module in (model.RCNN.fc1, model.RCNN.fc2)
            for p in module.parameters()
            if p.requires_grad
        ],
        "roi_cls": [
            p for p in model.RCNN.pred_cls.parameters()
            if p.requires_grad
        ],
        "roi_reg": [
            p for p in model.RCNN.pred_delta.parameters()
            if p.requires_grad
        ],
    }

    # Không cho một parameter xuất hiện ở hai group.
    seen = {}
    for group_name, params in groups.items():
        for p in params:
            pid = id(p)
            if pid in seen:
                raise ValueError(
                    f"Parameter duplicated in DPGA groups: "
                    f"{seen[pid]} and {group_name}"
                )
            seen[pid] = group_name

    return groups


# -----------------------------------------------------------------------------
# Loss split
# -----------------------------------------------------------------------------

def split_detection_and_odam_loss(
    loss_dict: Mapping[str, torch.Tensor],
    odam_key: str = "loss_rcnn_match",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    L_det:
        tất cả detection losses.

    L_odam:
        raw loss_rcnn_match.
    """
    if odam_key not in loss_dict:
        raise KeyError(
            f"{odam_key!r} không tồn tại trong loss_dict."
        )

    loss_odam = loss_dict[odam_key]

    det_losses = [
        value
        for key, value in loss_dict.items()
        if key != odam_key
    ]
    if not det_losses:
        raise ValueError("Không tìm thấy detection loss")

    loss_det = det_losses[0]
    for loss in det_losses[1:]:
        loss_det = loss_det + loss

    return loss_det, loss_odam


# -----------------------------------------------------------------------------
# DPGA controller
# -----------------------------------------------------------------------------

class DPGAController:
    """
    Controller tạo gradient cuối và ghi trực tiếp vào param.grad.

    Training usage:

        dpga = DPGAController(model, DPGAConfig(...))

        optimizer.zero_grad(set_to_none=True)

        loss_dict = model(image, im_info, gt_boxes)
        loss_det, loss_odam = split_detection_and_odam_loss(loss_dict)

        stats = dpga.backward(
            loss_det,
            loss_odam,
            epoch=epoch,
        )

        optimizer.step()

    KHÔNG gọi total_loss.backward() thêm trong cùng step.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[DPGAConfig] = None,
    ):
        self.model = model
        self.config = config or DPGAConfig()
        self.config.validate()

        self.groups = build_dpga_groups(model)

        self.params: List[nn.Parameter] = []
        self.param_group: Dict[int, str] = {}

        for group_name, params in self.groups.items():
            for p in params:
                self.params.append(p)
                self.param_group[id(p)] = group_name

        if not self.params:
            raise ValueError("DPGA không tìm thấy trainable parameter")

    def alpha(self, epoch: float) -> float:
        """
        Warm-up:
            epoch < warmup_epochs -> alpha=0

        Ramp-up:
            linear từ 0 -> alpha_max

        Sau đó:
            alpha=alpha_max
        """
        cfg = self.config

        if epoch < cfg.warmup_epochs:
            return 0.0

        if cfg.rampup_epochs == 0:
            return cfg.alpha_max

        ramp_start = float(cfg.warmup_epochs)
        ramp_end = ramp_start + float(cfg.rampup_epochs)

        if epoch >= ramp_end:
            return cfg.alpha_max

        progress = (
            (float(epoch) - ramp_start)
            / float(cfg.rampup_epochs)
        )
        progress = min(max(progress, 0.0), 1.0)
        return cfg.alpha_max * progress

    def _policy(self, name: str) -> DPGAModulePolicy:
        return self.config.module_policies.get(
            name,
            self.config.default_policy,
        )

    def _extract_gradients(
        self,
        loss_det: torch.Tensor,
        loss_odam: torch.Tensor,
        extract_odam: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Không gọi backward().
        autograd.grad lấy riêng g_det và g_odam.
        """
        if loss_det.requires_grad:
            g_det_raw = torch.autograd.grad(
                loss_det,
                self.params,
                retain_graph=bool(extract_odam and loss_odam.requires_grad),
                create_graph=False,
                allow_unused=True,
            )
        else:
            g_det_raw = [None for _ in self.params]

        if extract_odam and loss_odam.requires_grad:
            g_odam_raw = torch.autograd.grad(
                loss_odam,
                self.params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
        else:
            g_odam_raw = [None for _ in self.params]

        g_det = _dpga_replace_none(g_det_raw, self.params)
        g_odam = _dpga_replace_none(g_odam_raw, self.params)

        return g_det, g_odam

    def _split_by_group(
        self,
        grads: Sequence[torch.Tensor],
    ) -> Dict[str, List[torch.Tensor]]:
        out = {
            name: []
            for name in self.groups
        }

        for p, g in zip(self.params, grads):
            out[self.param_group[id(p)]].append(g)

        return out

    def _compose_one_group(
        self,
        name: str,
        g_det: Sequence[torch.Tensor],
        g_odam: Sequence[torch.Tensor],
        alpha: float,
    ) -> Tuple[List[torch.Tensor], DPGAModuleStats]:
        """
        Core DPGA math cho một module m.
        """
        cfg = self.config
        policy = self._policy(name)

        det_norm = _dpga_norm(g_det)
        odam_norm_before = _dpga_norm(g_odam)

        cosine_before = _dpga_cosine(
            g_det,
            g_odam,
            cfg.eps,
        )

        det_has_signal = (
            float(det_norm.detach().cpu()) > cfg.eps
        )
        odam_has_signal = (
            float(odam_norm_before.detach().cpu()) > cfg.eps
        )

        # ---------------------------------------------------------------------
        # Step A — conflict projection
        #
        # Nếu <g_o, g_d> < 0:
        #
        #   g_o_proj =
        #       g_o
        #       - <g_o,g_d>/(||g_d||^2 + eps) * g_d
        #
        # Khi eps ~ 0:
        #   <g_d, g_o_proj> ~ 0
        # ---------------------------------------------------------------------
        projected = False
        g_proj = list(g_odam)

        if (
            cfg.project_if_conflict
            and det_has_signal
            and odam_has_signal
            and float(cosine_before.detach().cpu())
                < cfg.conflict_threshold
        ):
            coefficient = (
                _dpga_dot(g_odam, g_det)
                / (_dpga_sq_norm(g_det) + cfg.eps)
            )

            g_proj = [
                go - coefficient * gd
                for go, gd in zip(g_odam, g_det)
            ]
            projected = True

        odam_norm_proj = _dpga_norm(g_proj)
        cosine_projected = _dpga_cosine(
            g_det,
            g_proj,
            cfg.eps,
        )

        # ---------------------------------------------------------------------
        # Step B — norm cap
        #
        #   ||g_o_safe|| <= rho_m ||g_d||
        # ---------------------------------------------------------------------
        cap_active = False
        if not odam_has_signal:
            norm_scale = det_norm.new_zeros(())
            g_safe = _dpga_zeros_like(self.groups[name])
        elif not cfg.use_norm_cap:
            norm_scale = det_norm.new_tensor(1.0)
            g_safe = list(g_proj)
        elif policy.rho <= 0 or not det_has_signal:
            norm_scale = det_norm.new_zeros(())
            g_safe = _dpga_zeros_like(self.groups[name])

        else:
            max_odam_norm = policy.rho * det_norm

            norm_scale = torch.clamp(
                max_odam_norm
                / (odam_norm_proj + cfg.eps),
                max=1.0,
            )

            g_safe = [
                g * norm_scale
                for g in g_proj
            ]
            cap_active = (
                float(norm_scale.detach().cpu()) < 1.0
            )

        odam_norm_safe = _dpga_norm(g_safe)

        # ---------------------------------------------------------------------
        # Step C — adaptive gate
        #
        # Dùng cosine BEFORE projection để vẫn nhớ conflict ban đầu.
        # ---------------------------------------------------------------------
        if cfg.use_gate:
            gate = torch.sigmoid(
                (cosine_before - policy.tau)
                / policy.temperature
            )

            if (
                float(odam_norm_safe.detach().cpu()) <= cfg.eps
                or not det_has_signal
            ):
                gate = gate.new_zeros(())
        else:
            gate = det_norm.new_tensor(
                1.0
                if float(odam_norm_safe.detach().cpu()) > cfg.eps
                else 0.0
            )

        effective_weight = gate * float(alpha)

        # ---------------------------------------------------------------------
        # Step D — final composition
        #
        #   g_final = g_det + alpha * gate * g_safe
        # ---------------------------------------------------------------------
        g_final = [
            gd + effective_weight * go
            for gd, go in zip(g_det, g_safe)
        ]

        cosine_after = _dpga_cosine(
            g_det,
            g_safe,
            cfg.eps,
        )

        stats = DPGAModuleStats(
            name=name,
            cosine_before=float(cosine_before.detach().cpu()),
            cosine_projected=float(cosine_projected.detach().cpu()),
            cosine_after=float(cosine_after.detach().cpu()),
            det_norm=float(det_norm.detach().cpu()),
            odam_norm_before=float(odam_norm_before.detach().cpu()),
            odam_norm_after_projection=float(
                odam_norm_proj.detach().cpu()
            ),
            odam_norm_after_cap=float(
                odam_norm_safe.detach().cpu()
            ),
            final_norm=float(
                _dpga_norm(g_final).detach().cpu()
            ),
            projected=projected,
            cap_active=cap_active,
            norm_scale=float(norm_scale.detach().cpu()),
            gate=float(gate.detach().cpu()),
            alpha=float(alpha),
            effective_weight=float(
                effective_weight.detach().cpu()
            ),
        )

        return g_final, stats

    def backward(
        self,
        loss_det: torch.Tensor,
        loss_odam: torch.Tensor,
        epoch: float,
        sync_distributed: bool = True,
        **unused_kwargs,
    ) -> DPGAStats:
        """
        Tạo final gradients và ghi vào param.grad.

        Trước hàm này:
            optimizer.zero_grad(set_to_none=True)

        Sau hàm này:
            optimizer.step()
        """
        del unused_kwargs

        if loss_det.ndim != 0:
            loss_det = loss_det.sum()

        if loss_odam.ndim != 0:
            loss_odam = loss_odam.sum()

        alpha = self.alpha(epoch)

        extract_odam = (
            alpha > 0.0
            and bool(loss_odam.requires_grad)
        )

        g_det_flat, g_odam_flat = self._extract_gradients(
            loss_det,
            loss_odam,
            extract_odam=extract_odam,
        )

        world_size = _dpga_distributed_world_size()
        gradient_scope = "local"
        if sync_distributed and world_size > 1:
            g_det_flat = _dpga_allreduce_mean(g_det_flat)
            g_odam_flat = _dpga_allreduce_mean(g_odam_flat)
            gradient_scope = "global_ddp_mean"

        det_by_group = self._split_by_group(g_det_flat)
        odam_by_group = self._split_by_group(g_odam_flat)

        stats_dict: Dict[str, DPGAModuleStats] = {}
        final_by_group: Dict[str, List[torch.Tensor]] = {}

        for name in self.groups:
            g_final, stats = self._compose_one_group(
                name=name,
                g_det=det_by_group[name],
                g_odam=odam_by_group[name],
                alpha=alpha,
            )
            final_by_group[name] = g_final
            stats_dict[name] = stats

        # Ghi gradient cuối trực tiếp vào model parameters.
        for name, params in self.groups.items():
            for p, g in zip(params, final_by_group[name]):
                p.grad = g.detach().clone()

        return DPGAStats(
            alpha=alpha,
            modules=stats_dict,
            gradient_scope=gradient_scope,
            world_size=world_size,
        )


# -----------------------------------------------------------------------------
# Logging / training convenience
# -----------------------------------------------------------------------------

def format_dpga_stats(stats: DPGAStats) -> str:
    lines = [
        (
            f"DPGA alpha={stats.alpha:.4f} "
            f"scope={stats.gradient_scope} "
            f"world_size={stats.world_size}"
        )
    ]

    for name, s in stats.modules.items():
        lines.append(
            f"{name:>10s} | "
            f"cos={s.cosine_before:+.4f} | "
            f"proj={int(s.projected)} | "
            f"det={s.det_norm:.3e} | "
            f"odam={s.odam_norm_before:.3e} | "
            f"safe={s.odam_norm_after_cap:.3e} | "
            f"cap_scale={s.norm_scale:.4f} | "
            f"gate={s.gate:.4f} | "
            f"effective={s.effective_weight:.4f}"
        )

    return "\n".join(lines)


def dpga_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dpga: DPGAController,
    image: torch.Tensor,
    im_info: torch.Tensor,
    gt_boxes: torch.Tensor,
    epoch: float,
):
    """
    Deprecated convenience helper for small local DPGA smoke tests.

    Use train.py for final experiments. This helper does not implement the full
    experiment pipeline, including fractional-step scheduling, diagnostics,
    artifact schema guards, or method-specific ODAM warm-up toggling.

    Example
    -------
        dpga = DPGAController(
            model,
            DPGAConfig(
                warmup_epochs=3,
                rampup_epochs=3,
                alpha_max=1.0,
            ),
        )

        for epoch in range(num_epochs):
            for batch in loader:
                loss_dict, stats = dpga_training_step(
                    model,
                    optimizer,
                    dpga,
                    image,
                    im_info,
                    gt_boxes,
                    epoch,
                )
    """
    warnings.warn(
        "dpga_training_step() is deprecated for experiment runs; use train.py "
        "so DPGA scheduling, diagnostics, and artifact guards stay aligned.",
        DeprecationWarning,
        stacklevel=2,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)

    loss_dict = model(
        image,
        im_info,
        gt_boxes,
    )

    loss_det, loss_odam = split_detection_and_odam_loss(
        loss_dict
    )

    stats = dpga.backward(
        loss_det=loss_det,
        loss_odam=loss_odam,
        epoch=epoch,
    )

    optimizer.step()

    return loss_dict, stats


# =============================================================================
# DPGA validation helper
# =============================================================================

def validate_dpga_integration(model: nn.Module):
    """
    Kiểm tra cấu trúc module để đảm bảo DPGA có thể group parameter.
    """
    groups = build_dpga_groups(model)

    summary = {
        name: sum(p.numel() for p in params)
        for name, params in groups.items()
    }

    if sum(summary.values()) <= 0:
        raise RuntimeError("DPGA groups contain no parameters")

    return summary
