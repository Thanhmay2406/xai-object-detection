import math

import torch
from torchvision.ops import roi_align


def assign_boxes_to_levels(boxes, min_level, max_level, canonical_box_size=224, canonical_level=4):
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)

    rois = boxes[:, 1:5] if boxes.shape[1] == 5 else boxes
    widths = (rois[:, 2] - rois[:, 0]).clamp(min=1e-6)
    heights = (rois[:, 3] - rois[:, 1]).clamp(min=1e-6)
    box_sizes = torch.sqrt(widths * heights)
    levels = torch.floor(canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-6))
    levels = levels.clamp(min=min_level, max=max_level).long()
    return levels - min_level


def roi_pooler(feature_maps, rois, strides, output_size, mode="ROIAlignV2"):
    if rois.numel() == 0:
        channels = feature_maps[0].shape[1]
        return feature_maps[0].new_zeros((0, channels, output_size[0], output_size[1]))

    min_level = int(math.log2(strides[0]))
    max_level = int(math.log2(strides[-1]))
    levels = assign_boxes_to_levels(rois, min_level, max_level)
    pooled = feature_maps[0].new_zeros((len(rois), feature_maps[0].shape[1], *output_size))
    for level, (feature_map, stride) in enumerate(zip(feature_maps, strides)):
        inds = torch.nonzero(levels == level, as_tuple=False).flatten()
        if inds.numel() == 0:
            continue
        pooled[inds] = roi_align(
            feature_map,
            rois[inds],
            output_size,
            spatial_scale=1.0 / stride,
            sampling_ratio=-1,
            aligned=True,
        )
    return pooled
