import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import batched_nms

from det_oprs.bbox_opr import bbox_transform_inv_opr, bbox_transform_opr, box_overlap_opr


class RPN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.strides = [64, 32, 16, 8, 4]
        self.anchor_sizes = getattr(config, "rpn_anchor_sizes", [256, 128, 64, 32, 16])
        self.anchor_ratios = getattr(config, "rpn_anchor_ratios", [0.5, 1.0, 2.0])
        self.num_anchors = len(self.anchor_ratios)
        self.conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(256, self.num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(256, self.num_anchors * 4, kernel_size=1)

        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, fpn_fms, im_info, gt_boxes=None):
        logits = []
        deltas = []
        anchors = []
        for feature, stride, size in zip(fpn_fms, self.strides, self.anchor_sizes):
            hidden = F.relu(self.conv(feature))
            logits.append(self.cls_logits(hidden).permute(0, 2, 3, 1).reshape(feature.shape[0], -1))
            deltas.append(
                self.bbox_pred(hidden)
                .permute(0, 2, 3, 1)
                .reshape(feature.shape[0], -1, 4)
            )
            anchors.append(self._make_anchors(feature, stride, size))

        logits = torch.cat(logits, dim=1)
        deltas = torch.cat(deltas, dim=1)
        anchors = torch.cat(anchors, dim=0)
        proposals = self._decode_and_filter(anchors, deltas, logits, im_info)

        if self.training:
            losses = self._losses(anchors, logits, deltas, gt_boxes, im_info)
            return proposals, losses
        return proposals

    def _make_anchors(self, feature, stride, size):
        height, width = feature.shape[-2:]
        device = feature.device
        dtype = feature.dtype
        shifts_x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * stride
        shifts_y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        centers = torch.stack((shift_x.reshape(-1), shift_y.reshape(-1)), dim=1)

        base_anchors = []
        area = float(size * size)
        for ratio in self.anchor_ratios:
            anchor_w = (area / ratio) ** 0.5
            anchor_h = anchor_w * ratio
            base_anchors.append(
                feature.new_tensor(
                    [-0.5 * anchor_w, -0.5 * anchor_h, 0.5 * anchor_w, 0.5 * anchor_h]
                )
            )
        base_anchors = torch.stack(base_anchors, dim=0)
        return (centers[:, None, :].repeat(1, self.num_anchors, 2) + base_anchors).reshape(-1, 4)

    def _decode_and_filter(self, anchors, deltas, logits, im_info):
        pre_nms_topk = int(getattr(self.config, "rpn_pre_nms_topk", 1000))
        post_nms_topk = int(getattr(self.config, "rpn_post_nms_topk", 300))
        nms_thresh = float(getattr(self.config, "rpn_nms_threshold", 0.7))
        min_size = float(getattr(self.config, "rpn_min_size", 1.0))
        rois = []
        batch_size = deltas.shape[0]
        for batch_idx in range(batch_size):
            scores = logits[batch_idx].sigmoid()
            num_topk = min(pre_nms_topk, scores.numel())
            top_scores, top_idx = scores.topk(num_topk)
            boxes = bbox_transform_inv_opr(anchors[top_idx], deltas[batch_idx, top_idx]).detach()
            boxes = _clip_boxes(boxes, im_info[batch_idx])
            keep_size = _valid_box_size(boxes, min_size)
            boxes, top_scores = boxes[keep_size], top_scores[keep_size]
            if boxes.numel() == 0:
                continue
            keep = batched_nms(boxes, top_scores, torch.zeros_like(top_scores, dtype=torch.long), nms_thresh)
            keep = keep[:post_nms_topk]
            batch_col = boxes.new_full((len(keep), 1), float(batch_idx))
            rois.append(torch.cat((batch_col, boxes[keep]), dim=1))
        if not rois:
            return anchors.new_zeros((0, 5))
        return torch.cat(rois, dim=0)

    def _losses(self, anchors, logits, deltas, gt_boxes, im_info):
        labels, targets = _assign_targets(anchors, gt_boxes, im_info)
        sampled = _sample_balanced_anchors(labels, self.config)
        valid = sampled & (labels >= 0)
        positive = sampled & (labels == 1)
        if valid.any():
            cls_loss = F.binary_cross_entropy_with_logits(
                logits[valid],
                labels[valid].float(),
                reduction="mean",
            )
        else:
            cls_loss = logits.sum() * 0.0
        if positive.any():
            loc_loss = F.smooth_l1_loss(
                deltas[positive],
                targets[positive],
                beta=1.0,
                reduction="sum",
            ) / valid.sum().clamp(min=1)
        else:
            loc_loss = deltas.sum() * 0.0
        return {"loss_rpn_cls": cls_loss, "loss_rpn_loc": loc_loss}


def _clip_boxes(boxes, info):
    height = float(info[0])
    width = float(info[1])
    return torch.stack(
        (
            boxes[:, 0].clamp(min=0, max=width - 1),
            boxes[:, 1].clamp(min=0, max=height - 1),
            boxes[:, 2].clamp(min=0, max=width - 1),
            boxes[:, 3].clamp(min=0, max=height - 1),
        ),
        dim=1,
    )


def _valid_box_size(boxes, min_size):
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    return (widths >= min_size) & (heights >= min_size)


def _iter_gt(gt_boxes, batch_idx, device):
    if gt_boxes is None:
        return torch.zeros((0, 5), device=device)
    if gt_boxes.dim() == 2:
        current = gt_boxes
    else:
        current = gt_boxes[batch_idx]
    current = current.to(device)
    if current.numel() == 0:
        return current.reshape(0, 5)
    valid = (current[:, 2] > current[:, 0]) & (current[:, 3] > current[:, 1])
    return current[valid]


def _assign_targets(anchors, gt_boxes, im_info):
    batch_size = int(im_info.shape[0])
    labels = anchors.new_full((batch_size, anchors.shape[0]), -1, dtype=torch.long)
    targets = anchors.new_zeros((batch_size, anchors.shape[0], 4))
    for batch_idx in range(batch_size):
        inside = _inside_image(anchors, im_info[batch_idx])
        if not inside.any():
            continue
        current_gt = _iter_gt(gt_boxes, batch_idx, anchors.device)
        if current_gt.numel() == 0:
            labels[batch_idx][inside] = 0
            continue
        anchor_inds = torch.nonzero(inside, as_tuple=False).flatten()
        inside_anchors = anchors[anchor_inds]
        overlaps = box_overlap_opr(inside_anchors, current_gt[:, :4])
        max_iou, matched = overlaps.max(dim=1)
        labels[batch_idx][anchor_inds[max_iou < 0.3]] = 0
        labels[batch_idx][anchor_inds[max_iou >= 0.7]] = 1
        gt_best = overlaps.argmax(dim=0)
        labels[batch_idx][anchor_inds[gt_best]] = 1
        targets[batch_idx][anchor_inds] = bbox_transform_opr(
            inside_anchors,
            current_gt[matched, :4],
        )
    return labels, targets


def _inside_image(anchors, info):
    height = float(info[0])
    width = float(info[1])
    return (
        (anchors[:, 0] >= 0)
        & (anchors[:, 1] >= 0)
        & (anchors[:, 2] <= width - 1)
        & (anchors[:, 3] <= height - 1)
    )


def _sample_balanced_anchors(labels, config):
    batch_size = int(getattr(config, "rpn_batch_size", 256))
    positive_fraction = float(getattr(config, "rpn_fg_fraction", 0.5))
    if batch_size <= 0:
        return labels >= 0

    positive_fraction = max(0.0, min(1.0, positive_fraction))
    sampled = torch.zeros_like(labels, dtype=torch.bool)
    max_pos = int(batch_size * positive_fraction)
    for batch_idx in range(labels.shape[0]):
        positive = torch.nonzero(labels[batch_idx] == 1, as_tuple=False).flatten()
        negative = torch.nonzero(labels[batch_idx] == 0, as_tuple=False).flatten()

        num_pos = min(int(positive.numel()), max_pos)
        num_neg = min(int(negative.numel()), batch_size - num_pos)

        if num_pos > 0:
            perm = torch.randperm(positive.numel(), device=labels.device)[:num_pos]
            sampled[batch_idx, positive[perm]] = True
        if num_neg > 0:
            perm = torch.randperm(negative.numel(), device=labels.device)[:num_neg]
            sampled[batch_idx, negative[perm]] = True
    return sampled
