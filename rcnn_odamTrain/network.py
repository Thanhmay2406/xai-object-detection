import math

import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
from torchvision import transforms
from torchvision.ops import batched_nms, roi_align
import numpy as np

from backbone.resnet50 import ResNet50
from backbone.fpn import FPN
from module.rpn import RPN
from layers.pooler import assign_boxes_to_levels, roi_pooler
from det_oprs.bbox_opr import bbox_transform_inv_opr, box_overlap_opr, paired_box_overlap_opr
from det_oprs.fpn_roi_target import fpn_roi_target
from det_oprs.loss_opr import softmax_loss, smooth_l1_loss
from det_oprs.utils import get_padded_tensor

INF = 100000000

class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.resnet50 = ResNet50(
            config.backbone_freeze_at,
            False,
            weights=getattr(config, "backbone_weights", "none"),
        )
        self.FPN = FPN(self.resnet50, 2, 6)
        self.RPN = RPN(config)
        self.RCNN = RCNN(config)

    def forward(self, image, im_info, gt_boxes=None):
        config = self.config
        image_mean = torch.as_tensor(config.image_mean, dtype=image.dtype, device=image.device).reshape(1, -1, 1, 1)
        image_std = torch.as_tensor(config.image_std, dtype=image.dtype, device=image.device).reshape(1, -1, 1, 1)
        image = (image - image_mean) / image_std
        image = get_padded_tensor(image, 64)
        if self.training:
            return self._forward_train(image, im_info, gt_boxes)
        else:
            return self._forward_test(image, im_info)

    def _forward_train(self, image, im_info, gt_boxes):
        config = self.config
        loss_dict = {}
        fpn_fms, _ = self.FPN(image)
        # fpn_fms stride: 64,32,16,8,4, p6->p2
        rpn_rois, loss_dict_rpn = self.RPN(fpn_fms, im_info, gt_boxes)

        # top_k=1: for each rpn_roi, only assign the gt object which fits it best 
        rcnn_rois, rcnn_labels, rcnn_bbox_targets, rcnn_gts, rcnn_roi_is_gt = fpn_roi_target(
                config, rpn_rois, im_info, gt_boxes, top_k=1, return_roi_is_gt=True)
        loss_dict_rcnn = self.RCNN(fpn_fms, rcnn_rois,
                rcnn_labels, rcnn_bbox_targets, rcnn_gts, rcnn_roi_is_gt)
        loss_dict.update(loss_dict_rpn)
        loss_dict.update(loss_dict_rcnn)
        return loss_dict

    @torch.enable_grad()
    def _forward_test(self, image, im_info):
        fpn_fms, _ = self.FPN(image)
        rpn_rois = self.RPN(fpn_fms, im_info)
        pred_bbox = self.RCNN(fpn_fms, rpn_rois)
        return pred_bbox.detach()

class RCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # roi head
        self.fc1 = nn.Linear(256*7*7, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        for l in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)
        # box predictor
        self.pred_cls = nn.Linear(1024, config.num_classes)
        self.pred_delta = nn.Linear(1024, config.num_classes * 4)
        for l in [self.pred_cls]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        for l in [self.pred_delta]:
            nn.init.normal_(l.weight, std=0.001)
            nn.init.constant_(l.bias, 0)

    @torch.enable_grad()
    def forward(self, fpn_fms, rcnn_rois, labels=None, bbox_targets=None, assigned_gts=None, roi_is_gt=None):
        config = self.config
        bbox_stds, bbox_means = config.bbox_normalize_stds, config.bbox_normalize_means
        # input p2-p5
        fpn_fms = fpn_fms[1:][::-1]
        stride = [4, 8, 16, 32]
        assert len(fpn_fms) == len(stride)
        max_level = int(math.log2(stride[-1]))
        min_level = int(math.log2(stride[0]))
        assert (len(stride) == max_level - min_level + 1)        
        level_assignments = assign_boxes_to_levels(rcnn_rois, min_level, max_level, 224, 4)
        pool_features = fpn_fms[0].new_zeros((len(rcnn_rois), fpn_fms[0].shape[1], 7, 7))
        for level, (fm_level, scale_level) in enumerate(zip(fpn_fms, stride)):
            inds = torch.nonzero(level_assignments == level, as_tuple=False).squeeze(1)
            rois_level = rcnn_rois[inds]
            pool_features[inds] = roi_align(fm_level, rois_level, (7,7), spatial_scale=1.0/scale_level,
                    sampling_ratio=-1, aligned=True)  
        # pool_features = roi_pooler(fpn_fms, rcnn_rois, stride, (7, 7), "ROIAlignV2")

        flatten_feature = torch.flatten(pool_features, start_dim=1)
        flatten_feature = F.relu(self.fc1(flatten_feature))
        flatten_feature = F.relu(self.fc2(flatten_feature))
        pred_cls = self.pred_cls(flatten_feature)
        pred_delta = self.pred_delta(flatten_feature)

        if self.training:
            labels = labels.long().flatten()  # class
            if roi_is_gt is None:
                roi_is_gt = torch.zeros_like(labels, dtype=torch.bool)
            else:
                roi_is_gt = roi_is_gt.bool().flatten()
            fg_masks = labels > 0
            valid_masks = labels >= 0
            fg_gt_classes = labels[fg_masks]
            odam_weight = getattr(config, "odam_loss_weight_effective", None)
            if odam_weight is None:
                odam_weight = getattr(config, "odam_loss_weight", 1.0)
            odam_weight = float(odam_weight)
            odam_enabled = odam_weight > 0.0

            # loss for regression
            # multi class
            pred_delta = pred_delta.reshape(-1, config.num_classes, 4)
            if fg_masks.any():
                pred_delta_fg = pred_delta[fg_masks, fg_gt_classes, :]
                localization_loss = smooth_l1_loss(
                    pred_delta_fg,
                    bbox_targets[fg_masks],
                    config.rcnn_smooth_l1_beta)

                pred_bbox = restore_bbox(rcnn_rois[fg_masks, 1:5], pred_delta_fg,
                    bbox_stds, bbox_means, True)

                # loss for iou prediction
                gt_bbox = restore_bbox(rcnn_rois[fg_masks, 1:5], bbox_targets[fg_masks],
                        bbox_stds, bbox_means, True)
                pred_gt_ious = paired_box_overlap_opr(pred_bbox, gt_bbox)

                odam_masks = fg_masks
                if getattr(config, "odam_exclude_gt_rois", True):
                    odam_masks = fg_masks & ~roi_is_gt
                if odam_enabled and odam_masks.any():
                    odam_gt_classes = labels[odam_masks]
                    odam_delta = pred_delta[odam_masks, odam_gt_classes, :]
                    odam_pred_bbox = restore_bbox(rcnn_rois[odam_masks, 1:5], odam_delta,
                        bbox_stds, bbox_means, True)
                    odam_gt_bbox = restore_bbox(rcnn_rois[odam_masks, 1:5], bbox_targets[odam_masks],
                            bbox_stds, bbox_means, True)
                    odam_pred_gt_ious = paired_box_overlap_opr(odam_pred_bbox, odam_gt_bbox)

                    target_scores = F.softmax(pred_cls, dim=-1) if getattr(config, "odam_use_confidence_target", True) else pred_cls
                    pool_grads = self.get_gradient(
                        target_scores,
                        pool_features,
                        create_graph=getattr(config, "odam_create_graph", True),
                    ) # N, C-1, 256, 7,7
                    pool_grads = smooth_pool_grads(
                        pool_grads,
                        int(getattr(config, "odam_smooth_kernel", 3)),
                    )
                    odam_inds = odam_masks.nonzero(as_tuple=True)[0]
                    pool_dams = F.relu((pool_grads[odam_inds, odam_gt_classes-1,:,:,:] *
                        pool_features[odam_masks]).sum(1)) # Num_pred,7,7

                    dam_size = fpn_fms[2].size()[-2:] # image_size // 16
                    rois_odam = rcnn_rois[odam_masks, 1:5]
                    bids = rcnn_rois[odam_masks, 0].long()
                    level_assignments_odam = level_assignments[odam_masks]

                    pred_dams = get_dams(
                        pool_dams, bids, rois_odam, fpn_fms, stride, level_assignments_odam, dam_size)

                    assigned_gts_odam = assigned_gts[odam_masks]
                    loss_rcnn_match = match_loss(
                        pred_dams,
                        assigned_gts_odam,
                        bids,
                        odam_pred_bbox,
                        odam_pred_gt_ious,
                    )
                else:
                    loss_rcnn_match = pred_delta_fg.sum() * 0.0
            else:
                localization_loss = pred_delta.sum() * 0.0
                loss_rcnn_match = pred_delta.sum() * 0.0

            # loss for classification
            objectness_loss = softmax_loss(pred_cls, labels)
            objectness_loss = objectness_loss * valid_masks
            normalizer = 1.0 / valid_masks.sum().item()
            loss_rcnn_loc = localization_loss.sum() * normalizer
            loss_rcnn_cls = objectness_loss.sum() * normalizer
            loss_rcnn_match = odam_weight * loss_rcnn_match

            loss_dict = {}
            loss_dict['loss_rcnn_loc'] = loss_rcnn_loc
            loss_dict['loss_rcnn_cls'] = loss_rcnn_cls
            loss_dict['loss_rcnn_match'] = loss_rcnn_match

            return loss_dict
        else:
            pool_grads = self.get_gradient(pred_cls, pool_features)
            class_num = pred_cls.shape[-1] - 1
            level_assignments = level_assignments.repeat_interleave(class_num)
            tag = torch.arange(class_num).type_as(pred_cls)+1
            tag = tag.repeat(pred_cls.shape[0], 1).reshape(-1,1)
            pred_scores = F.softmax(pred_cls, dim=-1)[:, 1:].reshape(-1, 1)
            pred_delta = pred_delta[:, 4:].reshape(-1, 4)
            base_rois = rcnn_rois.repeat_interleave(class_num, dim=0)
            keep = pred_scores[:, 0] > config.pred_cls_threshold
            pred_scores, pred_delta, base_rois, tag, level_assignments = \
                pred_scores[keep],pred_delta[keep],base_rois[keep],tag[keep],level_assignments[keep]
            pool_grads = pool_grads.reshape(-1,256,7,7)[keep]
  
            bids = base_rois[:, 0].long()
            base_rois = base_rois[:, 1:5]
            pred_bbox = restore_bbox(base_rois, pred_delta, bbox_stds, bbox_means, True)
            
            # get pool grads
            pred_index = torch.arange(pred_cls.shape[0], device=tag.device).repeat_interleave(class_num)
            pred_index = pred_index[keep].long()
            # pool_grads = self.get_pool_gradient(tag, pred_index) # Num_pred,256,7,7
            pool_dams = F.relu((pool_grads * pool_features[pred_index]).sum(1)) # Num_pred,7,7
            
            # get dam maps
            dam_size = fpn_fms[1].size()[-2:] # image_size // 8
            pred_dams = get_dams(pool_dams, bids, base_rois, fpn_fms, stride, level_assignments, dam_size)
            keep = postprocess_predictions(
                pred_bbox,
                pred_scores[:, 0],
                tag[:, 0],
                bids,
                config,
                pred_dams,
                dam_size,
            )
            pred_bbox = pred_bbox[keep]
            pred_scores = pred_scores[keep]
            tag = tag[keep]
            pred_dams = pred_dams[keep]

            dam_size = pred_dams.new_tensor(dam_size).repeat(len(pred_scores),1)
            pred_bbox = torch.cat([pred_bbox, pred_scores, tag, pred_dams, dam_size], axis=1)
            return pred_bbox

    def get_gradient(self, pred, pool_features, create_graph=False):
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
                    create_graph=create_graph)[0]
                grads.append(grad)

        return torch.stack(grads, dim=1)  # N, C-1, 256, 7,7

def smooth_pool_grads(pool_grads, kernel_size):
    if kernel_size <= 1:
        return pool_grads
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = max(float(kernel_size) / 3.0, 1e-6)
    coords = torch.arange(kernel_size, dtype=pool_grads.dtype, device=pool_grads.device)
    coords = coords - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp(min=1e-12)
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    channels = pool_grads.shape[2]
    kernel = kernel_2d.reshape(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    flattened = pool_grads.reshape(-1, channels, pool_grads.shape[-2], pool_grads.shape[-1])
    smoothed = F.conv2d(flattened, kernel, padding=kernel_size // 2, groups=channels)
    return smoothed.reshape_as(pool_grads)

def postprocess_predictions(boxes, scores, labels, bids, config, dams=None, dam_size=None):
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    nms_thresh = float(getattr(config, "rcnn_nms_threshold", 0.5))
    detections_per_image = int(getattr(config, "rcnn_detections_per_image", 100))
    use_odam_nms = bool(getattr(config, "odam_nms", False)) and dams is not None
    keep_all = []
    class_count = int(getattr(config, "num_classes", 1))
    for bid in bids.unique(sorted=True):
        image_inds = torch.nonzero(bids == bid, as_tuple=False).flatten()
        if image_inds.numel() == 0:
            continue
        if use_odam_nms:
            keep = odam_nms_image(
                boxes[image_inds],
                scores[image_inds],
                labels[image_inds],
                dams[image_inds],
                nms_thresh,
                float(getattr(config, "odam_nms_low_threshold", 0.2)),
                float(getattr(config, "odam_nms_high_threshold", 0.8)),
                int(getattr(config, "odam_nms_resize_short_edge", 50)),
                dam_size,
            )
        else:
            nms_ids = labels[image_inds].long() + bids[image_inds].long() * max(1, class_count)
            keep = batched_nms(boxes[image_inds], scores[image_inds], nms_ids, nms_thresh)
        if detections_per_image > 0:
            keep = keep[:detections_per_image]
        keep_all.append(image_inds[keep])
    if not keep_all:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    keep = torch.cat(keep_all, dim=0)
    return keep[scores[keep].argsort(descending=True)]


def odam_nms_image(
    boxes,
    scores,
    labels,
    dams,
    iou_threshold,
    corr_low_threshold,
    corr_high_threshold,
    resize_short_edge,
    dam_size,
):
    keep_all = []
    for label in labels.unique(sorted=True):
        class_inds = torch.nonzero(labels == label, as_tuple=False).flatten()
        if class_inds.numel() == 0:
            continue
        keep_class = odam_nms_class(
            boxes[class_inds],
            scores[class_inds],
            dams[class_inds],
            iou_threshold,
            corr_low_threshold,
            corr_high_threshold,
            resize_short_edge,
            dam_size,
        )
        keep_all.append(class_inds[keep_class])
    if not keep_all:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    keep = torch.cat(keep_all, dim=0)
    return keep[scores[keep].argsort(descending=True)]


def odam_nms_class(
    boxes,
    scores,
    dams,
    iou_threshold,
    corr_low_threshold,
    corr_high_threshold,
    resize_short_edge=50,
    dam_size=None,
):
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    selected = []
    dam_vectors = prepare_odam_nms_heatmaps(dams, dam_size, resize_short_edge)
    iou_threshold = float(iou_threshold)
    corr_low_threshold = float(corr_low_threshold)
    corr_high_threshold = float(corr_high_threshold)

    while order.numel() > 0:
        current = order[0]
        is_duplicate = False
        if selected:
            selected_tensor = torch.stack(selected).to(device=boxes.device)
            ious = box_overlap_opr(boxes[current].reshape(1, 4), boxes[selected_tensor]).flatten()
            corr = (dam_vectors[current].reshape(1, -1) @ dam_vectors[selected_tensor].T).flatten()
            duplicate_by_high_iou = (ious >= iou_threshold) & (corr > corr_low_threshold)
            duplicate_by_high_corr = (ious < iou_threshold) & (corr > corr_high_threshold)
            is_duplicate = bool((duplicate_by_high_iou | duplicate_by_high_corr).any().item())
        if not is_duplicate:
            selected.append(current)
        order = order[1:]

    if not selected:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    keep = torch.stack(selected)
    return keep[scores[keep].argsort(descending=True)]


def prepare_odam_nms_heatmaps(dams, dam_size=None, resize_short_edge=50):
    vectors = dams.float()
    if dam_size is not None and int(resize_short_edge) > 0:
        if isinstance(dam_size, torch.Tensor):
            height = int(dam_size[0].item())
            width = int(dam_size[1].item())
        else:
            height = int(dam_size[0])
            width = int(dam_size[1])
        if height > 0 and width > 0 and height * width == vectors.shape[1]:
            short_edge = min(height, width)
            if short_edge > 0 and short_edge != int(resize_short_edge):
                scale = float(resize_short_edge) / float(short_edge)
                new_height = max(1, int(round(height * scale)))
                new_width = max(1, int(round(width * scale)))
                vectors = F.interpolate(
                    vectors.reshape(-1, 1, height, width),
                    size=(new_height, new_width),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(vectors.shape[0], -1)
    return F.normalize(vectors, p=2, dim=1, eps=1e-12)

def get_dams(pool_maps, bids, rois, fpn_fms, stride, level_assignments, dam_size):
    resize = transforms.Resize(dam_size)
    pred_dams = pool_maps.new_zeros(len(pool_maps), dam_size[0]*dam_size[1])
    # project dam to locations on original feature map
    for bid in bids.unique():
        inds = torch.nonzero(bids == bid, as_tuple=True)[0]
        for level, (fm_level, scale_level) in enumerate(zip(fpn_fms, stride)):
            inds_level = inds[level_assignments[inds]==level]
            if len(inds_level)>0:
                dam_maps = roi_align_inv(\
                    pool_maps[inds_level], rois[inds_level], 1.0/scale_level, fm_level.size()[-2:])    
                dam_maps = resize(dam_maps)                           
                 # Num_pred,dam_size
                dam_maps = F.normalize(dam_maps.reshape(len(inds_level), -1))
                pred_dams[inds_level,:] = dam_maps
    return pred_dams

def restore_bbox(rois, deltas, stds, means, unnormalize=True):
    if unnormalize:
        std_opr = torch.as_tensor(stds, dtype=deltas.dtype, device=deltas.device).reshape(1, -1)
        mean_opr = torch.as_tensor(means, dtype=deltas.dtype, device=deltas.device).reshape(1, -1)
        deltas = deltas * std_opr
        deltas = deltas + mean_opr
    pred_bbox = bbox_transform_inv_opr(rois, deltas)
    return pred_bbox

def roi_align_inv(pool_dams, rois, scale, map_size):
    '''
        pool_dams: N,7,7
        rois: N,4 (x1,y1,x2,y2)
        scale: 1.0/scale_level
        map_size: the feature map size before roi_align
    '''
    N, h_pool, w_pool = pool_dams.shape
    rois = rois * scale
    rois = torch.stack((
        rois[:, 0].clamp(min=0., max=map_size[1]-1),
        rois[:, 1].clamp(min=0., max=map_size[0]-1),
        rois[:, 2].clamp(min=0., max=map_size[1]-1),
        rois[:, 3].clamp(min=0., max=map_size[0]-1),
    ), dim=1)
    rois_x_low, rois_y_low = rois[:,0].floor(), rois[:,1].floor()
    rois_x_high, rois_y_high = rois[:,2].ceil(), rois[:,3].ceil()
    rois_w_max = (rois_x_high - rois_x_low).max().int()+1
    rois_h_max = (rois_y_high - rois_y_low).max().int()+1
    M = rois_w_max * rois_h_max

    shift_y, shift_x = torch.meshgrid(
            torch.arange(0, rois_h_max, 1, dtype=torch.float32, device=rois.device),
            torch.arange(0, rois_w_max, 1, dtype=torch.float32, device=rois.device),
            indexing="ij")
    rois_grids = torch.stack((shift_x.reshape(-1), shift_y.reshape(-1)), dim=1) # W_max*H_max, 2

    rois_start_locs = torch.stack((rois_x_low, rois_y_low), dim=1) # N, 2
    rois_grids = rois_grids.repeat(N,1,1) + rois_start_locs.reshape(N, 1, 2) # N, W_max*H_max, 2

    grids_on_pool = ((rois_grids - rois[:,:2].reshape(N,1,2)) / \
                (rois[:,2:]-rois[:,:2]).reshape(N,1,2)) * \
                rois.new_tensor([w_pool-1, h_pool-1]).reshape(1,1,2) # N, W_max*H_max, 2 

    grids_x_low, grids_x_high = grids_on_pool[:,:,0].floor(), grids_on_pool[:,:,0].ceil()
    grids_y_low, grids_y_high = grids_on_pool[:,:,1].floor(), grids_on_pool[:,:,1].ceil()

    x_l_valid = (grids_x_low>=0) * (grids_x_low<w_pool)
    y_l_valid = (grids_y_low>=0) * (grids_y_low<h_pool) 
    x_h_valid = (grids_x_high>=0) * (grids_x_high<w_pool)
    y_h_valid = (grids_y_high>=0) * (grids_y_high<h_pool)

    ids, valid_inds = (x_l_valid * y_l_valid * x_h_valid * y_h_valid).nonzero(as_tuple=True)   
    pool_dams = pool_dams.reshape(N, -1) # N, 49

    x_weight = grids_on_pool[:,:,0] - grids_x_low
    y_weight = grids_on_pool[:,:,1] - grids_y_low

    # the dam value on top left corner
    tl_values = pool_dams.new_zeros(N, M)
    # ids, valid_inds = (x_l_valid * y_l_valid).nonzero(as_tuple=True)
    tl_locs = (grids_y_low[ids, valid_inds] * w_pool + grids_x_low[ids, valid_inds]).long()  # locations on pool_dam map
    tl_values[ids, valid_inds] = pool_dams[ids, tl_locs]
    roi_dam_values = tl_values * (1-x_weight) * (1-y_weight)
    del tl_values
    
    # the dam value on top right corner
    tr_values = pool_dams.new_zeros(N, M)
    # ids, valid_inds = (x_h_valid * y_l_valid).nonzero(as_tuple=True)
    tr_locs = (grids_y_low[ids, valid_inds] * w_pool + grids_x_high[ids, valid_inds]).long()  # locations on pool_dam map
    tr_values[ids, valid_inds] = pool_dams[ids, tr_locs]
    roi_dam_values += tr_values * x_weight * (1-y_weight)
    del tr_values

    # the dam value on bottom left corner
    bl_values = pool_dams.new_zeros(N, M)
    # ids, valid_inds = (x_l_valid * y_h_valid).nonzero(as_tuple=True)
    bl_locs = (grids_y_high[ids, valid_inds] * w_pool + grids_x_low[ids, valid_inds]).long()  # locations on pool_dam map
    bl_values[ids, valid_inds] = pool_dams[ids, bl_locs]
    roi_dam_values += bl_values * y_weight * (1-x_weight)
    del bl_values

    # the dam value on bottom right corner
    br_values = pool_dams.new_zeros(N, M)
    # ids, valid_inds = (x_h_valid * y_h_valid).nonzero(as_tuple=True)
    br_locs = (grids_y_high[ids, valid_inds] * w_pool + grids_x_high[ids, valid_inds]).long()  # locations on pool_dam map
    br_values[ids, valid_inds] = pool_dams[ids, br_locs]
    roi_dam_values += br_values * x_weight * y_weight
    del br_values

    # put the values back to map size
    indices = (rois_grids[:,:,1] * map_size[1] + rois_grids[:,:,0]).long() # N,M
    ids, valid_inds = (indices < map_size[0]*map_size[1]).nonzero(as_tuple=True)
    dam_maps = pool_dams.new_zeros(N, map_size[0]*map_size[1])
    dam_maps[ids, indices[ids, valid_inds]] = roi_dam_values[ids, valid_inds] # N, map_size
    return dam_maps.reshape(N, map_size[0], map_size[1])

def match_loss(dams, objs, bids, pred_bbox, pred_gt_iou):
    M, C = dams.shape
    if M == 0:
        return dams.sum() * 0.0
    objs = objs.long()
    bids = bids.long()

    best_positions = []
    for obj in objs.unique(sorted=True):
        inds = torch.nonzero(objs == obj, as_tuple=False).flatten()
        if inds.numel() == 0:
            continue
        best_positions.append(inds[pred_gt_iou[inds].argmax()])
    if not best_positions:
        return dams.sum() * 0.0
    best_positions = torch.stack(best_positions)
    best_objs = objs[best_positions]
    best_bids = bids[best_positions]
    best_dams = dams[best_positions]

    same_object = best_objs[:, None] == objs[None, :]
    same_image = best_bids[:, None] == bids[None, :]
    different_object_same_image = same_image & ~same_object

    pos_sims = (best_dams[:, None, :] * dams[None, :, :]).sum(-1)[same_object]
    neg_sims = (best_dams[:, None, :] * dams[None, :, :]).sum(-1)[different_object_same_image]
    pos_sims = pos_sims.clamp(min=1e-4, max=1-1e-4)
    neg_sims = neg_sims.clamp(min=1e-4, max=1-1e-4)

    pair_count = pos_sims.numel() + neg_sims.numel()
    if pair_count == 0:
        return dams.sum() * 0.0
    loss = (-pos_sims.log().sum() - (1 - neg_sims).log().sum()) / pair_count
    return loss



