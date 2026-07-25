import torch
from torchvision.ops import box_iou


def _empty_like_boxes(boxes):
    return boxes.new_zeros((0, 4))


def bbox_transform_opr(src_boxes, target_boxes):
    if src_boxes.numel() == 0:
        return _empty_like_boxes(src_boxes)

    src_widths = (src_boxes[:, 2] - src_boxes[:, 0]).clamp(min=1e-6)
    src_heights = (src_boxes[:, 3] - src_boxes[:, 1]).clamp(min=1e-6)
    src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
    src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights

    target_widths = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=1e-6)
    target_heights = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=1e-6)
    target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
    target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights

    dx = (target_ctr_x - src_ctr_x) / src_widths
    dy = (target_ctr_y - src_ctr_y) / src_heights
    dw = torch.log(target_widths / src_widths)
    dh = torch.log(target_heights / src_heights)
    return torch.stack((dx, dy, dw, dh), dim=1)


def bbox_transform_inv_opr(boxes, deltas):
    if boxes.numel() == 0:
        return _empty_like_boxes(deltas)

    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2].clamp(max=4.135)
    dh = deltas[:, 3].clamp(max=4.135)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h
    return torch.stack((x1, y1, x2, y2), dim=1)


def box_overlap_opr(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    return box_iou(boxes1, boxes2)


def paired_box_overlap_opr(boxes1, boxes2):
    if boxes1.numel() == 0:
        return boxes1.new_zeros((0,))
    inter_x1 = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.maximum(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.minimum(boxes1[:, 3], boxes2[:, 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)
    return inter / (area1 + area2 - inter).clamp(min=1e-6)
