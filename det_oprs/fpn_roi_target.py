import torch

from det_oprs.bbox_opr import bbox_transform_opr, box_overlap_opr


def fpn_roi_target(config, rpn_rois, im_info, gt_boxes, top_k=1):
    rois = _append_gt_boxes(rpn_rois, gt_boxes)
    if rois.numel() == 0:
        rois = _fallback_rois(im_info)

    labels = rois.new_zeros((len(rois),), dtype=torch.long)
    bbox_targets = rois.new_zeros((len(rois), 4))
    assigned_gts = rois.new_zeros((len(rois),), dtype=torch.long)

    batch_size = int(im_info.shape[0])
    gt_offsets = []
    offset = 0
    for batch_idx in range(batch_size):
        current_gt = _valid_gt(gt_boxes, batch_idx, rois.device)
        gt_offsets.append(offset)
        offset += len(current_gt)

    for batch_idx in range(batch_size):
        inds = torch.nonzero(rois[:, 0].long() == batch_idx, as_tuple=False).flatten()
        if inds.numel() == 0:
            continue
        current_gt = _valid_gt(gt_boxes, batch_idx, rois.device)
        if current_gt.numel() == 0:
            labels[inds] = 0
            assigned_gts[inds] = 0
            continue

        overlaps = box_overlap_opr(rois[inds, 1:5], current_gt[:, :4])
        max_iou, matched = overlaps.max(dim=1)
        fg_thresh = float(getattr(config, "rcnn_fg_threshold", 0.5))
        bg_thresh = float(getattr(config, "rcnn_bg_threshold", 0.5))
        labels[inds[max_iou < bg_thresh]] = 0
        fg_inds = inds[max_iou >= fg_thresh]
        if fg_inds.numel() > 0:
            matched_fg = matched[max_iou >= fg_thresh]
            labels[fg_inds] = current_gt[matched_fg, 4].long().clamp(min=1)
            bbox_targets[fg_inds] = bbox_transform_opr(rois[fg_inds, 1:5], current_gt[matched_fg, :4])
            assigned_gts[fg_inds] = matched_fg.long() + gt_offsets[batch_idx]

    rois, labels, bbox_targets, assigned_gts = _sample_rois(config, rois, labels, bbox_targets, assigned_gts)
    means = rois.new_tensor(getattr(config, "bbox_normalize_means", [0.0, 0.0, 0.0, 0.0]))
    stds = rois.new_tensor(getattr(config, "bbox_normalize_stds", [1.0, 1.0, 1.0, 1.0]))
    bbox_targets = (bbox_targets - means) / stds
    return rois, labels, bbox_targets, assigned_gts


def _valid_gt(gt_boxes, batch_idx, device):
    if gt_boxes is None:
        return torch.zeros((0, 5), device=device)
    current = gt_boxes if gt_boxes.dim() == 2 else gt_boxes[batch_idx]
    current = current.to(device)
    if current.numel() == 0:
        return current.reshape(0, 5)
    valid = (current[:, 2] > current[:, 0]) & (current[:, 3] > current[:, 1])
    if current.shape[1] > 4:
        valid = valid & (current[:, 4] >= 0)
    current = current[valid]
    if current.shape[1] == 4:
        current = torch.cat((current, current.new_ones((len(current), 1))), dim=1)
    return current[:, :5]


def _append_gt_boxes(rois, gt_boxes):
    if gt_boxes is None:
        return rois
    gt_rois = []
    batch_size = 1 if gt_boxes.dim() == 2 else gt_boxes.shape[0]
    for batch_idx in range(batch_size):
        current_gt = _valid_gt(gt_boxes, batch_idx, rois.device)
        if current_gt.numel() == 0:
            continue
        batch_col = current_gt.new_full((len(current_gt), 1), float(batch_idx))
        gt_rois.append(torch.cat((batch_col, current_gt[:, :4]), dim=1))
    if not gt_rois:
        return rois
    if rois.numel() == 0:
        return torch.cat(gt_rois, dim=0)
    return torch.cat((rois, torch.cat(gt_rois, dim=0)), dim=0)


def _fallback_rois(im_info):
    rois = []
    for batch_idx in range(int(im_info.shape[0])):
        height = float(im_info[batch_idx, 0])
        width = float(im_info[batch_idx, 1])
        rois.append(im_info.new_tensor([[float(batch_idx), 0.0, 0.0, width - 1.0, height - 1.0]]))
    return torch.cat(rois, dim=0)


def _sample_rois(config, rois, labels, bbox_targets, assigned_gts):
    batch_size = int(getattr(config, "rcnn_batch_size", len(rois)))
    if batch_size <= 0 or len(rois) <= batch_size:
        return rois, labels, bbox_targets, assigned_gts

    fg_fraction = float(getattr(config, "rcnn_fg_fraction", 0.25))
    max_fg = int(batch_size * fg_fraction)
    fg = torch.nonzero(labels > 0, as_tuple=False).flatten()
    bg = torch.nonzero(labels == 0, as_tuple=False).flatten()
    if fg.numel() > max_fg:
        fg = fg[torch.randperm(fg.numel(), device=fg.device)[:max_fg]]
    bg_needed = batch_size - fg.numel()
    if bg.numel() > bg_needed:
        bg = bg[torch.randperm(bg.numel(), device=bg.device)[:bg_needed]]
    keep = torch.cat((fg, bg), dim=0)
    if keep.numel() == 0:
        keep = torch.arange(min(batch_size, len(rois)), device=rois.device)
    return rois[keep], labels[keep], bbox_targets[keep], assigned_gts[keep]
