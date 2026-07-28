#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import functional as TF

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rcnn_odamTrain.network import Network


@dataclass(frozen=True)
class CategoryMapping:
    coco_to_train: dict[int, int]
    train_to_coco: dict[int, int]
    train_to_name: dict[int, str]

    @property
    def num_classes(self) -> int:
        return len(self.train_to_name)


@dataclass
class TrainConfig:
    backbone_freeze_at: int
    image_mean: list[float]
    image_std: list[float]
    num_classes: int
    bbox_normalize_stds: list[float]
    bbox_normalize_means: list[float]
    rcnn_smooth_l1_beta: float
    pred_cls_threshold: float
    rpn_pre_nms_topk: int
    rpn_post_nms_topk: int
    rpn_nms_threshold: float
    rpn_min_size: float
    rpn_batch_size: int
    rpn_fg_fraction: float
    rcnn_batch_size: int
    rcnn_fg_fraction: float
    rcnn_fg_threshold: float
    rcnn_bg_threshold: float
    rpn_anchor_sizes: list[int]
    rpn_anchor_ratios: list[float]
    rcnn_nms_threshold: float = 0.5
    rcnn_detections_per_image: int = 100
    odam_nms: bool = False
    odam_nms_low_threshold: float = 0.2
    odam_nms_high_threshold: float = 0.8
    odam_nms_resize_short_edge: int = 50
    odam_loss_weight: float = 1.0
    odam_loss_weight_effective: float | None = None
    odam_loss_start_epoch: int = 1
    odam_loss_warmup_epochs: int = 0
    odam_smooth_kernel: int = 3
    odam_create_graph: bool = True
    odam_use_confidence_target: bool = True
    odam_exclude_gt_rois: bool = True
    dp_odam: bool = False
    dp_odam_min_iou: float = 0.5
    dp_odam_min_confidence: float = 0.5
    dp_odam_min_rois: int = 2
    dp_odam_topk_per_gt: int = 2
    dp_odam_max_rois_per_batch: int = 32
    dp_odam_adaptive_quality_weight: bool = True
    dp_odam_negative_iou_threshold: float = 0.1
    dp_odam_exclude_self_pairs: bool = True
    dp_odam_detach_localization: bool = True
    dp_odam_roi_classifier_only: bool = True
    backbone_weights: str = "none"
    sab_odam: bool = False
    sab_small_area_threshold: float = 0.0025
    sab_medium_area_threshold: float = 0.0225
    sab_small_resolution: int = 28
    sab_medium_resolution: int = 14
    sab_large_resolution: int = 7
    sab_topk_per_gt: int = 2
    sab_max_rois_per_batch: int = 32
    sab_lambda_match: float = 1.0
    sab_lambda_scale: float = 0.1
    sab_lambda_edge: float = 0.1
    sab_lambda_inside: float = 0.05
    sab_boundary_band_ratio: float = 0.08
    sab_small_weight_ref_area: float = 0.0025
    sab_small_weight_gamma: float = 0.0
    sab_small_weight_max: float = 3.0
    sab_gate_hidden_dim: int = 32
    sab_gate_embed_dim: int = 8
    sab_force_fp32: bool = True
    sab_use_confidence_target: bool = True


class CocoDrillBitDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        mapping=None,
        image_size=None,
        keep_empty=True,
        include_empty_categories=False,
    ):
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.annotation_path = self.split_dir / "_annotations.coco.json"
        if not self.annotation_path.is_file():
            raise FileNotFoundError(f"Missing COCO annotations: {self.annotation_path}")

        with self.annotation_path.open("r", encoding="utf-8") as handle:
            self.coco = json.load(handle)

        used_category_ids = {int(ann["category_id"]) for ann in self.coco.get("annotations", [])}
        mapping_used_ids = None if include_empty_categories else used_category_ids
        self.mapping = mapping or build_category_mapping(
            self.coco["categories"],
            mapping_used_ids,
        )
        self.image_size = image_size
        annotations_by_image = {}
        for ann in self.coco.get("annotations", []):
            if int(ann.get("iscrowd", 0)) == 1:
                continue
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue
            x, y, w, h = [float(v) for v in bbox]
            if w <= 0 or h <= 0 or not all(math.isfinite(v) for v in (x, y, w, h)):
                continue
            coco_id = int(ann["category_id"])
            if coco_id not in self.mapping.coco_to_train:
                continue
            annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)

        images = sorted(self.coco.get("images", []), key=lambda item: int(item["id"]))
        if not keep_empty:
            images = [img for img in images if annotations_by_image.get(int(img["id"]))]
        self.images = images
        self.annotations_by_image = annotations_by_image
        self.image_by_id = {int(image["id"]): image for image in images}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_info = self.images[index]
        image_id = int(image_info["id"])
        image_path = self.split_dir / image_info["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            orig_w, orig_h = image.size
            scale_x = scale_y = 1.0
            if self.image_size is not None:
                image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
                scale_x = self.image_size / orig_w
                scale_y = self.image_size / orig_h
            tensor = TF.to_tensor(image)

        boxes = []
        for ann in self.annotations_by_image.get(image_id, []):
            x, y, w, h = [float(v) for v in ann["bbox"]]
            label = self.mapping.coco_to_train[int(ann["category_id"])]
            boxes.append([x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y, float(label)])

        gt_boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 5)
        height, width = tensor.shape[-2:]
        im_info = torch.tensor([float(height), float(width), 1.0], dtype=torch.float32)
        return tensor, im_info, gt_boxes, image_id

    def resize_scale(self, image_id):
        image_info = self.image_by_id[int(image_id)]
        if self.image_size is None:
            return 1.0, 1.0
        return (
            self.image_size / float(image_info["width"]),
            self.image_size / float(image_info["height"]),
        )


def build_category_mapping(categories, used_category_ids=None):
    sorted_categories = sorted(categories, key=lambda category: int(category["id"]))
    if used_category_ids is not None:
        sorted_categories = [
            category
            for category in sorted_categories
            if int(category["id"]) in used_category_ids
        ]
    if not sorted_categories:
        raise ValueError("No COCO categories with annotations were found")
    coco_to_train = {}
    train_to_coco = {}
    train_to_name = {0: "__background__"}
    for train_label, category in enumerate(sorted_categories, start=1):
        coco_id = int(category["id"])
        coco_to_train[coco_id] = train_label
        train_to_coco[train_label] = coco_id
        train_to_name[train_label] = str(category["name"])
    return CategoryMapping(coco_to_train, train_to_coco, train_to_name)


def collate_fn(batch):
    images, im_infos, gt_boxes, image_ids = zip(*batch)
    max_h = max(image.shape[-2] for image in images)
    max_w = max(image.shape[-1] for image in images)
    padded = images[0].new_zeros((len(images), 3, max_h, max_w))
    for idx, image in enumerate(images):
        padded[idx, :, : image.shape[-2], : image.shape[-1]] = image

    max_gt = max((boxes.shape[0] for boxes in gt_boxes), default=0)
    padded_gt = padded.new_full((len(gt_boxes), max_gt, 5), -1.0)
    for idx, boxes in enumerate(gt_boxes):
        if boxes.numel() > 0:
            padded_gt[idx, : boxes.shape[0]] = boxes
    return padded, torch.stack(im_infos), padded_gt, torch.tensor(image_ids, dtype=torch.long)


def make_config(args, mapping):
    return TrainConfig(
        backbone_freeze_at=args.backbone_freeze_at,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        num_classes=mapping.num_classes,
        bbox_normalize_stds=[0.1, 0.1, 0.2, 0.2],
        bbox_normalize_means=[0.0, 0.0, 0.0, 0.0],
        rcnn_smooth_l1_beta=1.0,
        pred_cls_threshold=args.pred_cls_threshold,
        rpn_pre_nms_topk=args.rpn_pre_nms_topk,
        rpn_post_nms_topk=args.rpn_post_nms_topk,
        rpn_nms_threshold=0.7,
        rpn_min_size=1.0,
        rpn_batch_size=args.rpn_batch_size,
        rpn_fg_fraction=args.rpn_fg_fraction,
        rcnn_batch_size=args.rcnn_batch_size,
        rcnn_fg_fraction=0.25,
        rcnn_fg_threshold=0.5,
        rcnn_bg_threshold=0.5,
        rpn_anchor_sizes=args.rpn_anchor_sizes,
        rpn_anchor_ratios=[0.5, 1.0, 2.0],
        rcnn_nms_threshold=args.rcnn_nms_threshold,
        rcnn_detections_per_image=args.rcnn_detections_per_image,
        odam_nms=args.odam_nms,
        odam_nms_low_threshold=args.odam_nms_low_threshold,
        odam_nms_high_threshold=args.odam_nms_high_threshold,
        odam_nms_resize_short_edge=args.odam_nms_resize_short_edge,
        odam_loss_weight=args.odam_loss_weight,
        odam_loss_weight_effective=None,
        odam_loss_start_epoch=args.odam_loss_start_epoch,
        odam_loss_warmup_epochs=args.odam_loss_warmup_epochs,
        odam_smooth_kernel=args.odam_smooth_kernel,
        odam_create_graph=args.odam_create_graph,
        odam_use_confidence_target=args.odam_use_confidence_target,
        odam_exclude_gt_rois=args.odam_exclude_gt_rois,
        dp_odam=args.dp_odam or args.dpga_odam,
        dp_odam_min_iou=args.dp_odam_min_iou,
        dp_odam_min_confidence=args.dp_odam_min_confidence,
        dp_odam_min_rois=args.dp_odam_min_rois,
        dp_odam_topk_per_gt=args.dp_odam_topk_per_gt,
        dp_odam_max_rois_per_batch=args.dp_odam_max_rois_per_batch,
        dp_odam_adaptive_quality_weight=args.dp_odam_adaptive_quality_weight,
        dp_odam_negative_iou_threshold=args.dp_odam_negative_iou_threshold,
        dp_odam_exclude_self_pairs=args.dp_odam_exclude_self_pairs,
        dp_odam_detach_localization=args.dp_odam_detach_localization,
        dp_odam_roi_classifier_only=args.dp_odam_roi_classifier_only,
        backbone_weights=args.backbone_weights,
        sab_odam=args.sab_odam,
        sab_small_area_threshold=args.sab_small_area_threshold,
        sab_medium_area_threshold=args.sab_medium_area_threshold,
        sab_small_resolution=args.sab_small_resolution,
        sab_medium_resolution=args.sab_medium_resolution,
        sab_large_resolution=args.sab_large_resolution,
        sab_topk_per_gt=args.sab_topk_per_gt,
        sab_max_rois_per_batch=args.sab_max_rois_per_batch,
        sab_lambda_match=args.sab_lambda_match,
        sab_lambda_scale=args.sab_lambda_scale,
        sab_lambda_edge=args.sab_lambda_edge,
        sab_lambda_inside=args.sab_lambda_inside,
        sab_boundary_band_ratio=args.sab_boundary_band_ratio,
        sab_small_weight_ref_area=args.sab_small_weight_ref_area,
        sab_small_weight_gamma=args.sab_small_weight_gamma,
        sab_small_weight_max=args.sab_small_weight_max,
        sab_gate_hidden_dim=args.sab_gate_hidden_dim,
        sab_gate_embed_dim=args.sab_gate_embed_dim,
        sab_force_fp32=args.sab_force_fp32,
        sab_use_confidence_target=args.sab_use_confidence_target,
    )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    images, im_info, gt_boxes, image_ids = batch
    return images.to(device), im_info.to(device), gt_boxes.to(device), image_ids.to(device)


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self):
        return self.rank == 0


def env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"Invalid integer environment {name}={value}") from error


def should_enable_distributed(argument):
    if argument is not None:
        return argument
    return env_int("WORLD_SIZE", 1) > 1


def setup_distributed(enabled):
    if not enabled:
        return DistributedContext(False, 0, 0, 1)

    rank = env_int("RANK", 0)
    local_rank = env_int("LOCAL_RANK", rank)
    world_size = env_int("WORLD_SIZE", 1)
    if world_size < 2:
        raise ValueError("Distributed mode requires WORLD_SIZE >= 2")
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return DistributedContext(True, rank, local_rank, world_size)


def cleanup_distributed(context):
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(context):
    if not context.enabled:
        return
    try:
        dist.barrier(device_ids=[context.local_rank])
    except TypeError:
        dist.barrier()


def reduce_metrics_mean(metrics, context):
    if not context.enabled:
        return metrics
    reduced = {}
    for key, value in metrics.items():
        tensor = torch.tensor(
            float(value),
            device=torch.device("cuda", context.local_rank),
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced[key] = float((tensor / context.world_size).detach().cpu())
    return reduced


def resolve_device(args, context):
    if context.enabled:
        device = torch.device(f"cuda:{context.local_rank}")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(f"rank={context.rank} cuda device={device} name={torch.cuda.get_device_name(device)}", flush=True)
    return device


def format_seconds(seconds):
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def gpu_memory_text(device):
    if device.type != "cuda":
        return "gpu_mem=cpu"
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    return f"gpu_mem={allocated:.2f}G reserved={reserved:.2f}G max={max_allocated:.2f}G"


def should_log_step(step, total_steps, args):
    return (
        step <= args.log_first_n
        or step == total_steps
        or step % max(1, args.log_every) == 0
    )


def compute_odam_loss_weight(epoch, args):
    target_weight = float(args.odam_loss_weight)
    if target_weight <= 0.0:
        return 0.0
    recovery_epochs = int(getattr(args, "dp_odam_recovery_epochs", 0))
    if (bool(getattr(args, "dp_odam", False)) or bool(getattr(args, "dpga_odam", False))) and recovery_epochs > 0:
        recovery_start = int(args.epochs) - recovery_epochs + 1
        if epoch >= recovery_start:
            return 0.0
    if epoch < int(args.odam_loss_start_epoch):
        return 0.0
    warmup_epochs = int(args.odam_loss_warmup_epochs)
    if warmup_epochs <= 0:
        return target_weight
    warmup_step = epoch - int(args.odam_loss_start_epoch) + 1
    progress = min(1.0, max(0.0, warmup_step / float(warmup_epochs)))
    return target_weight * progress


def set_odam_loss_weight(model, weight):
    module = model.module if hasattr(model, "module") else model
    module.config.odam_loss_weight_effective = float(weight)


def is_loss_key(key):
    return str(key).startswith("loss_")


def is_odam_loss_key(key):
    return key == "loss_rcnn_match" or str(key).startswith("loss_sab_")


def split_detection_and_odam_losses(losses):
    det_loss = None
    odam_loss = None
    for key, value in losses.items():
        if not is_loss_key(key):
            continue
        if is_odam_loss_key(key):
            odam_loss = value if odam_loss is None else odam_loss + value
        else:
            det_loss = value if det_loss is None else det_loss + value
    zero_source = next((value for value in losses.values() if torch.is_tensor(value)), None)
    if zero_source is None:
        raise ValueError("Model returned no tensor losses")
    if det_loss is None:
        det_loss = zero_source.sum() * 0.0
    if odam_loss is None:
        odam_loss = zero_source.sum() * 0.0
    return det_loss, odam_loss


def dp_odam_probe_parameters(model):
    module = model.module if hasattr(model, "module") else model
    allowed_prefixes = (
        "RCNN.fc1.",
        "RCNN.fc2.",
        "RCNN.pred_cls.",
    )
    return [
        param
        for name, param in module.named_parameters()
        if param.requires_grad and name.startswith(allowed_prefixes)
    ]


def gradient_alignment_stats(det_loss, odam_loss, parameters):
    det_grads = torch.autograd.grad(
        det_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    odam_grads = torch.autograd.grad(
        odam_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    device = det_loss.device
    dtype = torch.float32
    dot = torch.zeros((), device=device, dtype=dtype)
    det_sq = torch.zeros((), device=device, dtype=dtype)
    odam_sq = torch.zeros((), device=device, dtype=dtype)
    used = torch.zeros((), device=device, dtype=dtype)
    for det_grad, odam_grad in zip(det_grads, odam_grads):
        if det_grad is None or odam_grad is None:
            continue
        det_flat = det_grad.detach().float().reshape(-1)
        odam_flat = odam_grad.detach().float().reshape(-1)
        dot = dot + (det_flat * odam_flat).sum()
        det_sq = det_sq + det_flat.pow(2).sum()
        odam_sq = odam_sq + odam_flat.pow(2).sum()
        used = used + 1.0
    det_norm = det_sq.sqrt()
    odam_norm = odam_sq.sqrt()
    cosine = dot / (det_norm * odam_norm + 1e-12)
    finite = torch.isfinite(cosine) & torch.isfinite(det_norm) & torch.isfinite(odam_norm) & (used > 0)
    if not bool(finite.detach().item()):
        cosine = torch.zeros((), device=device, dtype=dtype)
        det_norm = torch.zeros((), device=device, dtype=dtype)
        odam_norm = torch.zeros((), device=device, dtype=dtype)
    return cosine, det_norm, odam_norm, finite.to(dtype=dtype)


def sync_dp_odam_stats(cosine, det_norm, odam_norm, finite, active, context):
    if not context.enabled:
        return cosine, det_norm, odam_norm, bool(finite.detach().item()), bool(active.detach().item())

    weighted_finite = finite * active
    values = torch.stack(
        (
            cosine * active,
            det_norm * active,
            odam_norm * active,
            active,
            weighted_finite,
        )
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    active_count = values[3].clamp(min=1.0)
    cosine = values[0] / active_count
    det_norm = values[1] / active_count
    odam_norm = values[2] / active_count
    any_active = bool((values[3] > 0).detach().item())
    finite_bool = any_active and bool((values[4] == values[3]).detach().item())
    return cosine, det_norm, odam_norm, finite_bool, any_active


def maybe_apply_dp_odam_gradient_gate(det_loss, odam_loss, model, args, context):
    stats = {
        "stat_dp_odam_grad_cosine": 0.0,
        "stat_dp_odam_det_grad_norm": 0.0,
        "stat_dp_odam_odam_grad_norm": 0.0,
        "stat_dp_odam_grad_gate": 0.0,
        "stat_dp_odam_grad_scale": 1.0,
    }
    if (
        not bool(getattr(args, "dp_odam_gradient_gate", False))
        or not bool(getattr(args, "dp_odam", False))
    ):
        return det_loss + odam_loss, stats

    device = det_loss.device
    dtype = torch.float32
    active = torch.tensor(
        1.0 if float(odam_loss.detach()) != 0.0 else 0.0,
        device=device,
        dtype=dtype,
    )
    parameters = dp_odam_probe_parameters(model)
    if not parameters:
        stats["stat_dp_odam_grad_gate"] = 1.0
        stats["stat_dp_odam_grad_scale"] = 0.0
        return det_loss, stats

    if bool(active.detach().item()):
        cosine, det_norm, odam_norm, finite = gradient_alignment_stats(det_loss, odam_loss, parameters)
    else:
        cosine = torch.zeros((), device=device, dtype=dtype)
        det_norm = torch.zeros((), device=device, dtype=dtype)
        odam_norm = torch.zeros((), device=device, dtype=dtype)
        finite = torch.zeros((), device=device, dtype=dtype)
    cosine, det_norm, odam_norm, finite_bool, any_active = sync_dp_odam_stats(
        cosine,
        det_norm,
        odam_norm,
        finite,
        active,
        context,
    )
    stats["stat_dp_odam_grad_cosine"] = float(cosine.detach())
    stats["stat_dp_odam_det_grad_norm"] = float(det_norm.detach())
    stats["stat_dp_odam_odam_grad_norm"] = float(odam_norm.detach())

    if not any_active:
        return det_loss + odam_loss, stats

    if (not finite_bool) or float(cosine.detach()) < float(args.dp_odam_conflict_threshold):
        stats["stat_dp_odam_grad_gate"] = 1.0
        stats["stat_dp_odam_grad_scale"] = 0.0
        return det_loss, stats

    scale = 1.0
    if bool(getattr(args, "dp_odam_adaptive_norm_cap", False)):
        ratio = float(args.dp_odam_norm_ratio) * float(det_norm.detach()) / (float(odam_norm.detach()) + 1e-12)
        scale = min(1.0, max(0.0, ratio))
    stats["stat_dp_odam_grad_scale"] = scale
    return det_loss + odam_loss * scale, stats


DPGA_MODULES = (
    "backbone",
    "fpn",
    "rpn",
    "roi_shared",
    "roi_classifier",
    "roi_regressor",
)


DPGA_DEFAULT_POLICIES = {
    "backbone": {"enabled": True, "max_norm_ratio": 0.05, "reject_cosine": -0.05, "full_cosine": 0.20},
    "fpn": {"enabled": True, "max_norm_ratio": 0.10, "reject_cosine": -0.05, "full_cosine": 0.15},
    "rpn": {"enabled": False, "max_norm_ratio": 0.00, "reject_cosine": -0.05, "full_cosine": 0.15},
    "roi_shared": {"enabled": True, "max_norm_ratio": 0.20, "reject_cosine": -0.10, "full_cosine": 0.10},
    "roi_classifier": {"enabled": True, "max_norm_ratio": 0.20, "reject_cosine": -0.05, "full_cosine": 0.15},
    "roi_regressor": {"enabled": True, "max_norm_ratio": 0.02, "reject_cosine": 0.00, "full_cosine": 0.20},
}


def dpga_parameter_module(name):
    if name.startswith("module."):
        name = name[len("module.") :]
    if name.startswith("resnet50.body.fc."):
        return "unused"
    if name.startswith("resnet50."):
        return "backbone"
    if name.startswith("FPN."):
        return "fpn"
    if name.startswith("RPN."):
        return "rpn"
    if name.startswith("RCNN.fc1.") or name.startswith("RCNN.fc2."):
        return "roi_shared"
    if name.startswith("RCNN.pred_cls."):
        return "roi_classifier"
    if name.startswith("RCNN.pred_delta."):
        return "roi_regressor"
    return "other"


def dpga_module_policies(args):
    policies = {key: dict(value) for key, value in DPGA_DEFAULT_POLICIES.items()}
    policies["backbone"]["max_norm_ratio"] = float(args.dpga_backbone_norm_ratio)
    policies["fpn"]["max_norm_ratio"] = float(args.dpga_fpn_norm_ratio)
    policies["rpn"]["max_norm_ratio"] = float(args.dpga_rpn_norm_ratio)
    policies["roi_shared"]["max_norm_ratio"] = float(args.dpga_roi_shared_norm_ratio)
    policies["roi_classifier"]["max_norm_ratio"] = float(args.dpga_roi_classifier_norm_ratio)
    policies["roi_regressor"]["max_norm_ratio"] = float(args.dpga_roi_regressor_norm_ratio)

    coverage = str(args.dpga_module_coverage)
    if coverage == "roi-head-only":
        for module_name in ("backbone", "fpn", "rpn", "roi_regressor"):
            policies[module_name]["enabled"] = False
    elif coverage == "roi-no-regressor":
        for module_name in ("backbone", "fpn", "rpn", "roi_regressor"):
            policies[module_name]["enabled"] = False
        policies["roi_shared"]["enabled"] = True
        policies["roi_classifier"]["enabled"] = True
    elif coverage == "global":
        for module_name in DPGA_MODULES:
            policies[module_name]["enabled"] = module_name != "rpn"
            policies[module_name]["max_norm_ratio"] = float(args.dpga_global_norm_ratio)

    if not bool(args.dpga_projection):
        for policy in policies.values():
            policy["project_on_negative"] = False
    else:
        for policy in policies.values():
            policy["project_on_negative"] = True
    return policies


def dpga_trainable_named_parameters(model):
    module = model.module if hasattr(model, "module") else model
    named_params = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if dpga_parameter_module(name) == "unused":
            continue
        named_params.append((name, param))
    return named_params


def dpga_piecewise_gate(cosine, reject_threshold, full_threshold):
    if full_threshold <= reject_threshold:
        return torch.where(cosine >= full_threshold, torch.ones_like(cosine), torch.zeros_like(cosine))
    gate = (cosine - float(reject_threshold)) / float(full_threshold - reject_threshold)
    return gate.clamp(0.0, 1.0)


def compose_dpga_module_gradients(det_grads, odam_grads, policy, eps=1e-12):
    device = det_grads[0].device
    dtype = torch.float32
    dot = torch.zeros((), device=device, dtype=dtype)
    det_sq = torch.zeros((), device=device, dtype=dtype)
    odam_sq = torch.zeros((), device=device, dtype=dtype)
    for det_grad, odam_grad in zip(det_grads, odam_grads):
        det_flat = det_grad.detach().float().reshape(-1)
        odam_flat = odam_grad.detach().float().reshape(-1)
        dot = dot + (det_flat * odam_flat).sum()
        det_sq = det_sq + det_flat.pow(2).sum()
        odam_sq = odam_sq + odam_flat.pow(2).sum()

    det_norm = det_sq.sqrt()
    odam_norm = odam_sq.sqrt()
    finite = torch.isfinite(dot) & torch.isfinite(det_norm) & torch.isfinite(odam_norm)
    if (not bool(finite.detach().item())) or float(det_norm.detach()) <= eps or float(odam_norm.detach()) <= eps:
        return [torch.zeros_like(grad) for grad in odam_grads], {
            "valid": 0.0,
            "cosine": 0.0,
            "norm_ratio": 0.0,
            "gate": 0.0,
            "norm_scale": 0.0,
            "effective_scale": 0.0,
            "projected": 0.0,
            "rejected": 1.0,
        }

    cosine = dot / (det_norm * odam_norm + eps)
    projected = bool(policy.get("project_on_negative", True)) and float(cosine.detach()) < 0.0
    safe_odam_grads = []
    if projected:
        coeff = dot / (det_sq + eps)
        for det_grad, odam_grad in zip(det_grads, odam_grads):
            safe_odam_grads.append(odam_grad - coeff.to(dtype=odam_grad.dtype) * det_grad)
    else:
        safe_odam_grads = list(odam_grads)

    safe_sq = torch.zeros((), device=device, dtype=dtype)
    for grad in safe_odam_grads:
        safe_sq = safe_sq + grad.detach().float().pow(2).sum()
    safe_norm = safe_sq.sqrt()

    max_ratio = float(policy.get("max_norm_ratio", 0.0))
    if max_ratio <= 0.0:
        norm_scale = torch.zeros((), device=device, dtype=dtype)
    else:
        norm_scale = torch.clamp(max_ratio * det_norm / (safe_norm + eps), max=1.0)
    gate = dpga_piecewise_gate(
        cosine,
        float(policy.get("reject_cosine", -0.05)),
        float(policy.get("full_cosine", 0.15)),
    )
    effective_scale = gate * norm_scale
    final_odam = [effective_scale.to(dtype=grad.dtype) * grad for grad in safe_odam_grads]
    return final_odam, {
        "valid": 1.0,
        "cosine": float(cosine.detach()),
        "norm_ratio": float((odam_norm / (det_norm + eps)).detach()),
        "gate": float(gate.detach()),
        "norm_scale": float(norm_scale.detach()),
        "effective_scale": float(effective_scale.detach()),
        "projected": 1.0 if projected else 0.0,
        "rejected": 1.0 if float(effective_scale.detach()) <= 0.0 else 0.0,
    }


def sync_dpga_active(active, context):
    if not context.enabled:
        return bool(active.detach().item())
    value = active.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return bool((value > 0).detach().item())


def sync_dpga_stats(local_stats, context):
    if not context.enabled:
        return local_stats

    fields = ("valid", "cosine", "norm_ratio", "gate", "norm_scale", "effective_scale", "projected", "rejected")
    values = []
    for module_name in DPGA_MODULES:
        stats = local_stats["modules"].get(module_name, {})
        valid = float(stats.get("valid", 0.0))
        values.append(valid)
        for field in fields[1:]:
            values.append(float(stats.get(field, 0.0)) * valid)
    values.extend(
        [
            float(local_stats.get("any_active", 0.0)),
            float(local_stats.get("detection_only_fallback", 0.0)),
            float(local_stats.get("final_grad_norm", 0.0)),
        ]
    )
    tensor = torch.tensor(values, device=local_stats["device"], dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    synced = {"modules": {}, "device": local_stats["device"]}
    idx = 0
    for module_name in DPGA_MODULES:
        count = float(tensor[idx].detach())
        idx += 1
        module_stats = {"valid": count}
        denom = max(count, 1.0)
        for field in fields[1:]:
            module_stats[field] = float(tensor[idx].detach()) / denom
            idx += 1
        synced["modules"][module_name] = module_stats
    world_size = max(1, int(context.world_size))
    synced["any_active"] = float(tensor[idx].detach()) / world_size
    synced["detection_only_fallback"] = float(tensor[idx + 1].detach()) / world_size
    synced["final_grad_norm"] = float(tensor[idx + 2].detach()) / world_size
    return synced


def flatten_dpga_stats(synced_stats):
    output = {
        "stat_dpga_any_active": float(synced_stats.get("any_active", 0.0)),
        "stat_dpga_detection_only_fallback": float(synced_stats.get("detection_only_fallback", 0.0)),
        "stat_dpga_final_grad_norm": float(synced_stats.get("final_grad_norm", 0.0)),
    }
    valid_modules = 0
    totals = {"cosine": 0.0, "norm_ratio": 0.0, "gate": 0.0, "norm_scale": 0.0, "effective_scale": 0.0, "projected": 0.0, "rejected": 0.0}
    for module_name in DPGA_MODULES:
        stats = synced_stats.get("modules", {}).get(module_name, {})
        valid = float(stats.get("valid", 0.0))
        prefix = f"stat_dpga_{module_name}"
        output[f"{prefix}_valid"] = 1.0 if valid > 0.0 else 0.0
        for field in totals:
            value = float(stats.get(field, 0.0))
            output[f"{prefix}_{field}"] = value
            if valid > 0.0:
                totals[field] += value
        if valid > 0.0:
            valid_modules += 1
    denom = max(1, valid_modules)
    for field, value in totals.items():
        output[f"stat_dpga_mean_{field}"] = value / denom
    return output


def assign_and_sync_final_gradients(named_params, final_grads, context):
    grad_sq = None
    for (_, param), grad in zip(named_params, final_grads):
        if grad is None:
            param.grad = None
            continue
        if not torch.isfinite(grad).all():
            raise FloatingPointError("Non-finite DPGA-ODAM final gradient")
        param.grad = grad.detach().clone()
        current_sq = param.grad.float().pow(2).sum()
        grad_sq = current_sq if grad_sq is None else grad_sq + current_sq

    if context.enabled:
        flat_grads = [param.grad.reshape(-1) for _, param in named_params if param.grad is not None]
        if flat_grads:
            flat = torch.cat(flat_grads)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat = flat / float(context.world_size)
            offset = 0
            for _, param in named_params:
                if param.grad is None:
                    continue
                numel = param.numel()
                param.grad.copy_(flat[offset : offset + numel].view_as(param))
                offset += numel

    if grad_sq is None:
        return 0.0
    return float(grad_sq.sqrt().detach())


def compose_dpga_odam_gradients(det_loss, odam_loss, model, args, context):
    if not bool(torch.isfinite(det_loss).detach().item()):
        raise RuntimeError(f"Non-finite detection loss: {float(det_loss.detach())}")

    named_params = dpga_trainable_named_parameters(model)
    if not named_params:
        raise RuntimeError("DPGA-ODAM found no trainable parameters")
    params = [param for _, param in named_params]
    module_names = [dpga_parameter_module(name) for name, _ in named_params]
    policies = dpga_module_policies(args)

    local_active = torch.tensor(
        1.0 if bool(torch.isfinite(odam_loss).detach().item()) and float(odam_loss.detach()) != 0.0 else 0.0,
        device=det_loss.device,
        dtype=torch.float32,
    )
    any_active = sync_dpga_active(local_active, context)
    det_grads = torch.autograd.grad(
        det_loss,
        params,
        retain_graph=any_active and bool(local_active.detach().item()),
        allow_unused=True,
    )
    missing_detection = [name for (name, _), grad in zip(named_params, det_grads) if grad is None]
    if missing_detection and bool(args.dpga_fail_on_missing_detection_grad):
        preview = ", ".join(missing_detection[:5])
        raise RuntimeError(f"Missing detection gradient for DPGA parameter(s): {preview}")

    if bool(local_active.detach().item()):
        odam_grads = torch.autograd.grad(
            odam_loss,
            params,
            retain_graph=False,
            allow_unused=True,
        )
    else:
        odam_grads = tuple(None for _ in params)

    grouped_indices = {module_name: [] for module_name in DPGA_MODULES}
    for idx, module_name in enumerate(module_names):
        if module_name in grouped_indices:
            grouped_indices[module_name].append(idx)

    final_grads = []
    for det_grad in det_grads:
        if det_grad is None:
            final_grads.append(None)
        else:
            final_grads.append(det_grad.detach().clone())

    local_stats = {
        "modules": {},
        "device": det_loss.device,
        "any_active": 1.0 if any_active else 0.0,
        "detection_only_fallback": 0.0 if any_active else 1.0,
        "final_grad_norm": 0.0,
    }

    for module_name in DPGA_MODULES:
        policy = policies[module_name]
        indices = grouped_indices[module_name]
        det_group = []
        odam_group = []
        valid_indices = []
        for idx in indices:
            det_grad = det_grads[idx]
            odam_grad = odam_grads[idx]
            if det_grad is None:
                continue
            if (
                not any_active
                or not bool(policy.get("enabled", True))
                or odam_grad is None
                or float(policy.get("max_norm_ratio", 0.0)) <= 0.0
            ):
                continue
            det_group.append(det_grad)
            odam_group.append(odam_grad)
            valid_indices.append(idx)

        if not valid_indices:
            local_stats["modules"][module_name] = {"valid": 0.0}
            continue

        safe_odam_grads, module_stats = compose_dpga_module_gradients(det_group, odam_group, policy)
        local_stats["modules"][module_name] = module_stats
        for idx, safe_grad in zip(valid_indices, safe_odam_grads):
            if final_grads[idx] is None:
                continue
            final_grads[idx] = final_grads[idx] + safe_grad.to(dtype=final_grads[idx].dtype)

    final_grad_norm = assign_and_sync_final_gradients(named_params, final_grads, context)
    local_stats["final_grad_norm"] = final_grad_norm
    synced_stats = sync_dpga_stats(local_stats, context)
    stats = flatten_dpga_stats(synced_stats)
    stats["stat_dpga_missing_detection_grad"] = float(len(missing_detection))
    stats["stat_dpga_amp_disabled"] = 1.0 if args.amp else 0.0
    total_loss = det_loss + (odam_loss if bool(local_active.detach().item()) else det_loss.detach() * 0.0)
    return total_loss, stats


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args, context):
    model.train()
    totals = {}
    start = time.perf_counter()
    previous_end = start
    total_steps = len(loader)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if context.rank == 0:
        model_config = (model.module if hasattr(model, "module") else model).config
        use_amp = args.amp and device.type == "cuda" and not bool(getattr(args, "dpga_odam", False))
        print(
            f"epoch={epoch} phase=train start batches={total_steps} "
            f"batch_size={loader.batch_size} amp={use_amp} "
            f"odam_loss_weight_effective={getattr(model_config, 'odam_loss_weight_effective', None)}",
            flush=True,
        )
    for step, batch in enumerate(loader, start=1):
        step_start = time.perf_counter()
        data_time = step_start - previous_end
        images, im_info, gt_boxes, _ = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        use_amp = args.amp and device.type == "cuda" and not bool(getattr(args, "dpga_odam", False))
        with torch.autocast(device_type=device.type, enabled=use_amp):
            losses = model(images, im_info, gt_boxes)
            det_loss, odam_loss = split_detection_and_odam_losses(losses)
            if bool(getattr(args, "dpga_odam", False)):
                total_loss, dp_grad_stats = compose_dpga_odam_gradients(
                    det_loss,
                    odam_loss,
                    model,
                    args,
                    context,
                )
            else:
                total_loss, dp_grad_stats = maybe_apply_dp_odam_gradient_gate(
                    det_loss,
                    odam_loss,
                    model,
                    args,
                    context,
                )
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}: {float(total_loss.detach())}")
        if bool(getattr(args, "dpga_odam", False)):
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.grad is not None],
                    args.max_grad_norm,
                )
            optimizer.step()
        elif use_amp:
            scaler.scale(total_loss).backward()
            if args.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
        for key, value in dp_grad_stats.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        totals["loss_total"] = totals.get("loss_total", 0.0) + float(total_loss.detach())

        if should_log_step(step, total_steps, args) and device.type == "cuda":
            torch.cuda.synchronize(device)
        step_end = time.perf_counter()
        batch_time = step_end - step_start
        previous_end = step_end

        if should_log_step(step, total_steps, args) and context.rank == 0:
            avg = {key: value / step for key, value in totals.items()}
            current_metrics = {
                **{key: float(value.detach()) for key, value in losses.items()},
                **dp_grad_stats,
                "loss_total": float(total_loss.detach()),
            }
            loss_text = " ".join(
                f"{key}={float(current_metrics.get(key, 0.0)):.4f}/{value:.4f}"
                for key, value in sorted(avg.items())
                if key != "loss_total"
            )
            elapsed = step_end - start
            eta = (elapsed / max(1, step)) * max(0, total_steps - step)
            print(
                f"epoch={epoch} phase=train step={step}/{total_steps} "
                f"progress={100.0 * step / max(1, total_steps):.1f}% "
                f"images={images.shape[0]} gt={(gt_boxes[..., 4] >= 0).sum().item()} "
                f"loss_total={float(total_loss.detach()):.4f}/{avg['loss_total']:.4f} "
                f"{loss_text} lr={optimizer.param_groups[0]['lr']:.6g} "
                f"batch_time={batch_time:.3f}s data_time={data_time:.3f}s "
                f"img_s={images.shape[0] / max(batch_time, 1e-9):.2f} "
                f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)} "
                f"{gpu_memory_text(device)}",
                flush=True,
            )

    seconds = time.perf_counter() - start
    if context.rank == 0:
        print(f"epoch={epoch} phase=train done elapsed={format_seconds(seconds)}", flush=True)
    return {key: value / max(1, len(loader)) for key, value in totals.items()} | {"seconds": seconds}


def freeze_batchnorm_for_loss_eval(module):
    if isinstance(module, _BatchNorm):
        module.eval()


def validate_loss(model, loader, device, args, phase="valid"):
    if loader is None:
        return {}
    was_training = model.training
    model.train()
    model.apply(freeze_batchnorm_for_loss_eval)
    totals = {}
    max_batches = args.val_batches
    start = time.perf_counter()
    total_steps = len(loader) if max_batches is None else min(len(loader), max_batches)
    print(f"phase={phase} loss_eval start batches={total_steps}", flush=True)
    for step, batch in enumerate(loader, start=1):
        step_start = time.perf_counter()
        images, im_info, gt_boxes, _ = move_batch(batch, device)
        losses = model(images, im_info, gt_boxes)
        det_loss, odam_loss = split_detection_and_odam_losses(losses)
        total_loss = det_loss + odam_loss
        for key, value in losses.items():
            totals[f"{phase}_{key}"] = totals.get(f"{phase}_{key}", 0.0) + float(value.detach())
        totals[f"{phase}_loss_total"] = totals.get(f"{phase}_loss_total", 0.0) + float(total_loss.detach())
        if should_log_step(step, total_steps, args):
            avg_total = totals[f"{phase}_loss_total"] / max(1, step)
            print(
                f"phase={phase} loss_eval step={step}/{total_steps} "
                f"loss_total={float(total_loss.detach()):.4f}/{avg_total:.4f} "
                f"batch_time={time.perf_counter() - step_start:.3f}s",
                flush=True,
            )
        if max_batches is not None and step >= max_batches:
            break
    if not was_training:
        model.eval()
    else:
        model.train()
    denom = max(1, step if "step" in locals() else 0)
    metrics = {key: value / denom for key, value in totals.items()}
    print(
        f"phase={phase} loss_eval done elapsed={format_seconds(time.perf_counter() - start)} "
        + " ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items())),
        flush=True,
    )
    return metrics


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, mapping, config, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "mapping": asdict(mapping),
            "config": asdict(config),
            "args": vars(args),
        },
        tmp_path,
    )
    tmp_path.replace(path)


def restore_checkpoint(path, model, optimizer, scheduler, scaler, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint.get("scaler", {}))
    return int(checkpoint["epoch"]) + 1


def load_model_checkpoint_for_eval(path, model, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    epoch = int(checkpoint["epoch"])
    print(f"eval checkpoint loaded={path} epoch={epoch}", flush=True)
    return epoch


def write_epoch_metrics(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {key: row.get(key, "") for key in sorted(row)}
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        existing_fieldnames = list(reader.fieldnames or [])
        existing_rows = []
        for existing_row in reader:
            # Rows written by csv.DictWriter should not contain a None key. If a
            # historical artifact is malformed, keep only named columns here and
            # let the new clean schema take over for subsequent rows.
            existing_row.pop(None, None)
            existing_rows.append(existing_row)

    fieldnames = sorted(set(existing_fieldnames).union(row))
    if fieldnames == existing_fieldnames:
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writerow({key: row.get(key, "") for key in fieldnames})
        return

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for existing_row in existing_rows:
            writer.writerow({key: existing_row.get(key, "") for key in fieldnames})
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    tmp_path.replace(path)


def build_loader(dataset, batch_size, workers, shuffle, sampler=None, pin_memory=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
        persistent_workers=workers > 0,
        collate_fn=collate_fn,
    )


def maybe_subset(dataset, limit):
    if limit is None:
        return dataset
    return Subset(dataset, range(min(limit, len(dataset))))


def get_base_dataset(dataset):
    current = dataset
    while isinstance(current, Subset):
        current = current.dataset
    if not isinstance(current, CocoDrillBitDataset):
        raise TypeError("Expected CocoDrillBitDataset or Subset[CocoDrillBitDataset]")
    return current


def get_dataset_image_ids(dataset):
    if isinstance(dataset, Subset):
        base = get_base_dataset(dataset)
        return [int(base.images[index]["id"]) for index in dataset.indices]
    return get_base_dataset(dataset).image_by_id.keys()


def create_empty_coco_results(coco_gt):
    from pycocotools.coco import COCO

    coco_dt = COCO()
    coco_dt.dataset = {
        "images": list(coco_gt.dataset.get("images", [])),
        "categories": list(coco_gt.dataset.get("categories", [])),
        "annotations": [],
    }
    coco_dt.createIndex()
    return coco_dt


def sanitize_metric_name(name):
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "class"


def evaluate_coco(model, loader, device, mapping, split, score_threshold, log_every, log_first_n):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise RuntimeError("pycocotools is required. Install with: pip install pycocotools") from error

    model.eval()
    base_dataset = get_base_dataset(loader.dataset)
    image_ids = list(get_dataset_image_ids(loader.dataset))
    coco_gt = COCO(str(base_dataset.annotation_path))
    predictions = []
    total_predictions = 0
    total_gt = 0
    start = time.perf_counter()
    total_steps = len(loader)

    print(f"phase={split} coco_eval start batches={total_steps} score_threshold={score_threshold}", flush=True)
    for step, batch in enumerate(loader, start=1):
        step_start = time.perf_counter()
        images, im_info, gt_boxes, image_id_tensor = move_batch(batch, device)
        if images.shape[0] != 1:
            raise ValueError("COCO eval for rcnn_odamTrain requires batch_size=1")
        image_id = int(image_id_tensor.item())
        total_gt += int((gt_boxes[..., 4] >= 0).sum().item())

        # Network test forward computes ODAM gradients internally.
        outputs = model(images, im_info).detach().cpu()
        scale_x, scale_y = base_dataset.resize_scale(image_id)
        image_info = base_dataset.image_by_id[image_id]
        image_width = float(image_info["width"])
        image_height = float(image_info["height"])
        kept = 0
        if outputs.numel() > 0:
            for output in outputs:
                score = float(output[4].item())
                if score < score_threshold:
                    continue
                train_label = int(output[5].item())
                if train_label not in mapping.train_to_coco:
                    continue
                x1 = max(0.0, min(image_width - 1.0, float(output[0].item()) / scale_x))
                y1 = max(0.0, min(image_height - 1.0, float(output[1].item()) / scale_y))
                x2 = max(0.0, min(image_width - 1.0, float(output[2].item()) / scale_x))
                y2 = max(0.0, min(image_height - 1.0, float(output[3].item()) / scale_y))
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                if width <= 0.0 or height <= 0.0:
                    continue
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": mapping.train_to_coco[train_label],
                        "bbox": [x1, y1, width, height],
                        "score": score,
                    }
                )
                kept += 1
        total_predictions += kept

        if step <= log_first_n or step == total_steps or step % max(1, log_every) == 0:
            elapsed = time.perf_counter() - start
            eta = (elapsed / max(1, step)) * max(0, total_steps - step)
            print(
                f"phase={split} coco_eval step={step}/{total_steps} "
                f"image_id={image_id} kept={kept} total_predictions={total_predictions} "
                f"batch_time={time.perf_counter() - step_start:.3f}s "
                f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}",
                flush=True,
            )

    coco_dt = coco_gt.loadRes(predictions) if predictions else create_empty_coco_results(coco_gt)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = sorted(image_ids)
    coco_eval.params.catIds = sorted(mapping.coco_to_train.keys())
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats
    metrics = {
        "map_50_95": float(stats[0]),
        "map50": float(stats[1]),
        "map75": float(stats[2]),
        "map_small": float(stats[3]),
        "map_medium": float(stats[4]),
        "map_large": float(stats[5]),
        "ar_1": float(stats[6]),
        "ar_10": float(stats[7]),
        "ar_100": float(stats[8]),
        "ar_small": float(stats[9]),
        "ar_medium": float(stats[10]),
        "ar_large": float(stats[11]),
        "gt_total": float(total_gt),
        "pred_total": float(total_predictions),
    }
    precision = coco_eval.eval.get("precision")
    if precision is not None:
        iou_thresholds = coco_eval.params.iouThrs
        iou50_index = int(np.argmin(np.abs(iou_thresholds - 0.5)))
        area_all_index = 0
        max_dets_100_index = len(coco_eval.params.maxDets) - 1
        for category_index, coco_category_id in enumerate(coco_eval.params.catIds):
            values = precision[
                iou50_index,
                :,
                category_index,
                area_all_index,
                max_dets_100_index,
            ]
            valid_values = values[values > -1]
            class_ap50 = float(np.mean(valid_values)) if valid_values.size else 0.0
            train_label = mapping.coco_to_train[coco_category_id]
            class_name = mapping.train_to_name[train_label]
            metrics[f"class_{train_label}_{sanitize_metric_name(class_name)}_ap50"] = class_ap50
    print(
        f"phase={split} coco_eval done map_50_95={metrics['map_50_95']:.4f} "
        f"map50={metrics['map50']:.4f} map75={metrics['map75']:.4f} "
        f"gt_total={int(metrics['gt_total'])} pred_total={int(metrics['pred_total'])} "
        f"elapsed={format_seconds(time.perf_counter() - start)}",
        flush=True,
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train rcnn_odamTrain Network on data/drill_bit_coco")
    parser.add_argument("--data-root", default="data/drill_bit_coco")
    parser.add_argument("--output-dir", default="results/rcnn_odam_train")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--step-size", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--distributed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable DDP. Default auto-enables when torchrun sets WORLD_SIZE > 1.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-empty", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--include-empty-categories",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include COCO categories with zero annotations in the label space. "
            "Use this to match baseline/train_faster_rcnn.py outputs."
        ),
    )
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--max-test-images", type=int, default=None)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--log-first-n", type=int, default=3)
    parser.add_argument("--test-after-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--test-checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument(
        "--eval-coco-every",
        type=int,
        default=1,
        help="Run COCO validation every N epochs. Use 0 to disable.",
    )
    parser.add_argument("--max-val-coco-images", type=int, default=None)
    parser.add_argument("--backbone-freeze-at", type=int, default=0)
    parser.add_argument(
        "--backbone-weights",
        choices=("none", "random", "scratch", "default", "coco", "imagenet", "pretrained"),
        default="none",
        help="Use default/pretrained to align the backbone initialization with pretrained baselines.",
    )
    parser.add_argument("--pred-cls-threshold", type=float, default=0.05)
    parser.add_argument("--rcnn-nms-threshold", type=float, default=0.5)
    parser.add_argument("--rcnn-detections-per-image", type=int, default=100)
    parser.add_argument(
        "--odam-nms",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use ODAM-NMS in RCNN post-processing instead of classical IoU-only NMS.",
    )
    parser.add_argument(
        "--odam-nms-low-threshold",
        type=float,
        default=0.2,
        help="Low heatmap-correlation threshold T_l for high-IoU pairs in ODAM-NMS.",
    )
    parser.add_argument(
        "--odam-nms-high-threshold",
        type=float,
        default=0.8,
        help="High heatmap-correlation threshold T_h for low-IoU pairs in ODAM-NMS.",
    )
    parser.add_argument(
        "--odam-nms-resize-short-edge",
        type=int,
        default=50,
        help="Resize ODAM heatmaps to this short-edge length before ODAM-NMS correlation. Use <=0 to disable.",
    )
    parser.add_argument("--odam-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--odam-loss-start-epoch",
        type=int,
        default=1,
        help=(
            "Epoch where ODAM auxiliary loss becomes active. "
            "Earlier epochs use an effective ODAM loss weight of 0."
        ),
    )
    parser.add_argument(
        "--odam-loss-warmup-epochs",
        type=int,
        default=0,
        help=(
            "Linearly ramp ODAM loss from 0 to --odam-loss-weight over this many "
            "active epochs. Use 0 to keep a constant weight after start epoch."
        ),
    )
    parser.add_argument("--odam-smooth-kernel", type=int, default=3)
    parser.add_argument(
        "--odam-create-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep ODAM gradient maps differentiable so the auxiliary loss can optimize them.",
    )
    parser.add_argument(
        "--odam-use-confidence-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use softmax confidence scores, not raw logits, as ODAM explanation targets.",
    )
    parser.add_argument(
        "--odam-exclude-gt-rois",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude GT-appended ROIs from the ODAM auxiliary pair loss.",
    )
    parser.add_argument(
        "--dp-odam",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Detection-Preserving ODAM reliable mining and branch isolation.",
    )
    parser.add_argument("--dp-odam-min-iou", type=float, default=0.5)
    parser.add_argument("--dp-odam-min-confidence", type=float, default=0.5)
    parser.add_argument(
        "--dp-odam-min-rois",
        type=int,
        default=2,
        help="Disable ODAM loss for the batch/rank when fewer reliable ROIs remain.",
    )
    parser.add_argument(
        "--dp-odam-topk-per-gt",
        type=int,
        default=2,
        help="Maximum reliable ODAM proposals kept per assigned GT object.",
    )
    parser.add_argument(
        "--dp-odam-max-rois-per-batch",
        type=int,
        default=32,
        help="Global reliable ODAM ROI cap per batch/rank. Use <=0 for no cap.",
    )
    parser.add_argument(
        "--dp-odam-adaptive-quality-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale ODAM loss by mean sqrt(IoU * class-confidence) of reliable ROIs.",
    )
    parser.add_argument(
        "--dp-odam-negative-iou-threshold",
        type=float,
        default=0.1,
        help="Keep different-object ODAM negative pairs only when predicted boxes overlap above this threshold. Use <0 to keep all.",
    )
    parser.add_argument(
        "--dp-odam-exclude-self-pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove best-ROI self-pairs from the positive ODAM pair loss.",
    )
    parser.add_argument(
        "--dp-odam-detach-localization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detach ODAM bbox/IoU inputs so auxiliary loss does not update box regression through pair selection.",
    )
    parser.add_argument(
        "--dp-odam-roi-classifier-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute ODAM heatmaps from detached ROI features so ODAM updates ROI classifier layers only.",
    )
    parser.add_argument(
        "--dp-odam-recovery-epochs",
        type=int,
        default=0,
        help="Disable ODAM loss for the final N epochs to recover detection/localization.",
    )
    parser.add_argument(
        "--dp-odam-gradient-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Probe ROI-classifier gradients and drop ODAM loss when cosine is below --dp-odam-conflict-threshold.",
    )
    parser.add_argument("--dp-odam-conflict-threshold", type=float, default=0.0)
    parser.add_argument(
        "--dp-odam-adaptive-norm-cap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When gradient gate is enabled, further cap ODAM loss scale by ROI-classifier gradient norm ratio.",
    )
    parser.add_argument(
        "--dp-odam-norm-ratio",
        type=float,
        default=0.1,
        help="Maximum ODAM/detection gradient norm ratio used by --dp-odam-adaptive-norm-cap.",
    )
    parser.add_argument(
        "--dpga-odam",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable Detection-Priority Gradient-Aligned ODAM. This uses the ODAM/DP-ODAM "
            "auxiliary loss but composes detection and ODAM gradients with module-wise "
            "projection, adaptive gate, norm balancing, and manual DDP all-reduce."
        ),
    )
    parser.add_argument(
        "--dpga-module-coverage",
        choices=("full", "roi-head-only", "roi-no-regressor", "global"),
        default="full",
        help="Module coverage for DPGA gradient composition.",
    )
    parser.add_argument(
        "--dpga-projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Project away the ODAM component that points against detection when module cosine is negative.",
    )
    parser.add_argument("--dpga-global-norm-ratio", type=float, default=0.1)
    parser.add_argument("--dpga-backbone-norm-ratio", type=float, default=0.05)
    parser.add_argument("--dpga-fpn-norm-ratio", type=float, default=0.10)
    parser.add_argument("--dpga-rpn-norm-ratio", type=float, default=0.00)
    parser.add_argument("--dpga-roi-shared-norm-ratio", type=float, default=0.20)
    parser.add_argument("--dpga-roi-classifier-norm-ratio", type=float, default=0.20)
    parser.add_argument("--dpga-roi-regressor-norm-ratio", type=float, default=0.02)
    parser.add_argument(
        "--dpga-fail-on-missing-detection-grad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail closed if an included trainable parameter has no detection gradient.",
    )
    parser.add_argument(
        "--sab-odam",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the Scale-Adaptive Boundary-Aware ODAM training branch.",
    )
    parser.add_argument("--sab-small-area-threshold", type=float, default=0.0025)
    parser.add_argument("--sab-medium-area-threshold", type=float, default=0.0225)
    parser.add_argument("--sab-small-resolution", type=int, default=28)
    parser.add_argument("--sab-medium-resolution", type=int, default=14)
    parser.add_argument("--sab-large-resolution", type=int, default=7)
    parser.add_argument(
        "--sab-topk-per-gt",
        type=int,
        default=2,
        help="Maximum SAB positive proposals kept per assigned GT object.",
    )
    parser.add_argument(
        "--sab-max-rois-per-batch",
        type=int,
        default=32,
        help="Global SAB proposal cap per batch/rank. Use <=0 for no cap.",
    )
    parser.add_argument("--sab-lambda-match", type=float, default=1.0)
    parser.add_argument("--sab-lambda-scale", type=float, default=0.1)
    parser.add_argument("--sab-lambda-edge", type=float, default=0.1)
    parser.add_argument("--sab-lambda-inside", type=float, default=0.05)
    parser.add_argument("--sab-boundary-band-ratio", type=float, default=0.08)
    parser.add_argument("--sab-small-weight-ref-area", type=float, default=0.0025)
    parser.add_argument("--sab-small-weight-gamma", type=float, default=0.0)
    parser.add_argument("--sab-small-weight-max", type=float, default=3.0)
    parser.add_argument("--sab-gate-hidden-dim", type=int, default=32)
    parser.add_argument("--sab-gate-embed-dim", type=int, default=8)
    parser.add_argument(
        "--sab-force-fp32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the SAB explanation branch in fp32 even when AMP is enabled.",
    )
    parser.add_argument(
        "--sab-use-confidence-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use class confidence rather than raw class logit for SAB class heatmaps.",
    )
    parser.add_argument("--rpn-pre-nms-topk", type=int, default=1000)
    parser.add_argument("--rpn-post-nms-topk", type=int, default=300)
    parser.add_argument(
        "--rpn-batch-size",
        type=int,
        default=256,
        help="Number of anchors sampled per image for RPN loss. Use <=0 to keep all valid anchors.",
    )
    parser.add_argument(
        "--rpn-fg-fraction",
        type=float,
        default=0.5,
        help="Maximum fraction of positive anchors in the sampled RPN loss batch.",
    )
    parser.add_argument("--rcnn-batch-size", type=int, default=128)
    parser.add_argument("--rpn-anchor-sizes", type=int, nargs="+", default=[256, 128, 64, 32, 16])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.workers < 0:
        raise ValueError("--workers must be >= 0")
    if args.max_test_images is not None and args.max_test_images < 1:
        raise ValueError("--max-test-images must be >= 1")
    if args.max_val_coco_images is not None and args.max_val_coco_images < 1:
        raise ValueError("--max-val-coco-images must be >= 1")
    if args.eval_coco_every < 0:
        raise ValueError("--eval-coco-every must be >= 0")
    if not 0.0 <= args.rcnn_nms_threshold <= 1.0:
        raise ValueError("--rcnn-nms-threshold must be in [0, 1]")
    if args.rcnn_detections_per_image < 1:
        raise ValueError("--rcnn-detections-per-image must be >= 1")
    if not 0.0 <= args.odam_nms_low_threshold <= 1.0:
        raise ValueError("--odam-nms-low-threshold must be in [0, 1]")
    if not 0.0 <= args.odam_nms_high_threshold <= 1.0:
        raise ValueError("--odam-nms-high-threshold must be in [0, 1]")
    if args.odam_nms_low_threshold > args.odam_nms_high_threshold:
        raise ValueError("--odam-nms-low-threshold must be <= --odam-nms-high-threshold")
    if args.odam_nms_resize_short_edge < 0:
        raise ValueError("--odam-nms-resize-short-edge must be >= 0")
    if args.rpn_fg_fraction < 0.0 or args.rpn_fg_fraction > 1.0:
        raise ValueError("--rpn-fg-fraction must be in [0, 1]")
    if args.odam_loss_weight < 0.0:
        raise ValueError("--odam-loss-weight must be >= 0")
    if args.odam_loss_start_epoch < 1:
        raise ValueError("--odam-loss-start-epoch must be >= 1")
    if args.odam_loss_warmup_epochs < 0:
        raise ValueError("--odam-loss-warmup-epochs must be >= 0")
    if args.odam_smooth_kernel < 1:
        raise ValueError("--odam-smooth-kernel must be >= 1")
    if not 0.0 <= args.dp_odam_min_iou <= 1.0:
        raise ValueError("--dp-odam-min-iou must be in [0, 1]")
    if not 0.0 <= args.dp_odam_min_confidence <= 1.0:
        raise ValueError("--dp-odam-min-confidence must be in [0, 1]")
    if args.dp_odam_min_rois < 1:
        raise ValueError("--dp-odam-min-rois must be >= 1")
    if args.dp_odam_topk_per_gt < 1:
        raise ValueError("--dp-odam-topk-per-gt must be >= 1")
    if args.dp_odam_negative_iou_threshold > 1.0:
        raise ValueError("--dp-odam-negative-iou-threshold must be <= 1")
    if args.dp_odam_recovery_epochs < 0:
        raise ValueError("--dp-odam-recovery-epochs must be >= 0")
    if args.dp_odam_recovery_epochs >= args.epochs:
        raise ValueError("--dp-odam-recovery-epochs must be < --epochs")
    if args.dp_odam_norm_ratio <= 0.0:
        raise ValueError("--dp-odam-norm-ratio must be > 0")
    for name in (
        "dpga_global_norm_ratio",
        "dpga_backbone_norm_ratio",
        "dpga_fpn_norm_ratio",
        "dpga_rpn_norm_ratio",
        "dpga_roi_shared_norm_ratio",
        "dpga_roi_classifier_norm_ratio",
        "dpga_roi_regressor_norm_ratio",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    if args.sab_small_area_threshold <= 0.0:
        raise ValueError("--sab-small-area-threshold must be > 0")
    if args.sab_medium_area_threshold <= args.sab_small_area_threshold:
        raise ValueError("--sab-medium-area-threshold must be > --sab-small-area-threshold")
    for name in ("sab_small_resolution", "sab_medium_resolution", "sab_large_resolution"):
        if getattr(args, name) < 7:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 7")
    if args.sab_topk_per_gt < 1:
        raise ValueError("--sab-topk-per-gt must be >= 1")
    if args.sab_lambda_match < 0.0:
        raise ValueError("--sab-lambda-match must be >= 0")
    if args.sab_lambda_scale < 0.0:
        raise ValueError("--sab-lambda-scale must be >= 0")
    if args.sab_lambda_edge < 0.0:
        raise ValueError("--sab-lambda-edge must be >= 0")
    if args.sab_lambda_inside < 0.0:
        raise ValueError("--sab-lambda-inside must be >= 0")
    if args.sab_boundary_band_ratio <= 0.0:
        raise ValueError("--sab-boundary-band-ratio must be > 0")
    if args.sab_small_weight_ref_area <= 0.0:
        raise ValueError("--sab-small-weight-ref-area must be > 0")
    if args.sab_small_weight_gamma < 0.0:
        raise ValueError("--sab-small-weight-gamma must be >= 0")
    if args.sab_small_weight_max < 1.0:
        raise ValueError("--sab-small-weight-max must be >= 1")
    if args.sab_gate_hidden_dim < 1:
        raise ValueError("--sab-gate-hidden-dim must be >= 1")
    if args.sab_gate_embed_dim < 1:
        raise ValueError("--sab-gate-embed-dim must be >= 1")
    distributed = setup_distributed(should_enable_distributed(args.distributed))
    seed_everything(args.seed)
    try:
        output_dir = Path(args.output_dir)
        if distributed.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            if args.overwrite and args.resume is None:
                for name in ("metrics.csv", "last.pt", "best.pt", "config.json", "test_metrics.json"):
                    path = output_dir / name
                    if path.exists():
                        path.unlink()
        distributed_barrier(distributed)

        train_dataset_full = CocoDrillBitDataset(
            args.data_root,
            "train",
            image_size=args.image_size,
            keep_empty=args.keep_empty,
            include_empty_categories=args.include_empty_categories,
        )
        mapping = train_dataset_full.mapping
        val_dataset_full = CocoDrillBitDataset(
            args.data_root,
            "valid",
            mapping=mapping,
            image_size=args.image_size,
            keep_empty=True,
        )
        train_dataset = maybe_subset(train_dataset_full, args.max_train_images)
        val_dataset = maybe_subset(val_dataset_full, args.max_val_images)
        val_coco_dataset = maybe_subset(
            val_dataset_full,
            args.max_val_coco_images or args.max_val_images,
        )

        config = make_config(args, mapping)

        device = resolve_device(args, distributed)
        raw_model = Network(config).to(device)
        optimizer = torch.optim.SGD(
            [param for param in raw_model.parameters() if param.requires_grad],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

        start_epoch = 1
        if args.resume is not None:
            start_epoch = restore_checkpoint(
                Path(args.resume),
                raw_model,
                optimizer,
                scheduler,
                scaler,
                device,
            )

        train_model = raw_model
        if distributed.enabled and bool(args.dpga_odam) and distributed.is_main:
            print("DPGA-ODAM uses manual distributed gradient all-reduce; DDP wrapper disabled.", flush=True)
        if distributed.enabled and not bool(args.dpga_odam):
            train_model = DistributedDataParallel(
                raw_model,
                device_ids=[distributed.local_rank],
                output_device=distributed.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=distributed.world_size,
                rank=distributed.rank,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
            )
            if distributed.enabled
            else None
        )

        train_loader = build_loader(
            train_dataset,
            args.batch_size,
            args.workers,
            shuffle=True,
            sampler=train_sampler,
            pin_memory=device.type == "cuda",
        )
        val_loader = build_loader(
            val_dataset,
            args.batch_size,
            args.workers,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )
        val_coco_loader = build_loader(
            val_coco_dataset,
            batch_size=1,
            workers=args.workers,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )

        initial_odam_loss_weight = compute_odam_loss_weight(start_epoch, args)
        set_odam_loss_weight(raw_model, initial_odam_loss_weight)
        run_config = {
            "mapping": asdict(mapping),
            "config": asdict(config),
            "args": vars(args),
            "train_images": len(train_dataset),
            "valid_images": len(val_dataset),
            "valid_coco_images": len(val_coco_dataset),
            "device": str(device),
            "distributed": asdict(distributed),
            "batch_size_per_rank": args.batch_size,
            "global_train_batch_size": args.batch_size * distributed.world_size,
        }
        if distributed.is_main:
            (output_dir / "config.json").write_text(
                json.dumps(run_config, indent=2),
                encoding="utf-8",
            )
            print(
                f"dataset train={len(train_dataset)} valid={len(val_dataset)} "
                f"valid_coco={len(val_coco_dataset)} "
                f"classes={mapping.train_to_name} device={device} "
                f"distributed={distributed.enabled} world_size={distributed.world_size} "
                f"batch_size_per_rank={args.batch_size}",
                flush=True,
            )

        best_map50 = -math.inf
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            lr_used = optimizer.param_groups[0]["lr"]
            odam_loss_weight_effective = compute_odam_loss_weight(epoch, args)
            set_odam_loss_weight(raw_model, odam_loss_weight_effective)
            if distributed.is_main:
                print(
                    f"epoch={epoch} start lr={lr_used:.6g} "
                    f"odam_loss_weight_effective={odam_loss_weight_effective:.6g}",
                    flush=True,
                )

            train_metrics = train_one_epoch(
                train_model,
                train_loader,
                optimizer,
                scaler,
                device,
                epoch,
                args,
                context=distributed,
            )
            train_metrics = reduce_metrics_mean(train_metrics, distributed)

            val_metrics = {}
            if distributed.is_main:
                val_metrics = validate_loss(raw_model, val_loader, device, args, phase="valid")
                if args.eval_coco_every and (
                    epoch % args.eval_coco_every == 0
                    or epoch == args.epochs
                ):
                    val_coco_metrics = evaluate_coco(
                        raw_model,
                        val_coco_loader,
                        device,
                        mapping,
                        "valid",
                        args.score_threshold,
                        args.log_every,
                        args.log_first_n,
                    )
                    val_metrics.update(
                        {
                            f"val_{key}": value
                            for key, value in val_coco_metrics.items()
                        }
                    )

            scheduler.step()
            row = {
                "epoch": epoch,
                "lr": lr_used,
                "next_lr": optimizer.param_groups[0]["lr"],
                "odam_loss_weight_effective": odam_loss_weight_effective,
                **train_metrics,
                **val_metrics,
            }
            if distributed.is_main:
                write_epoch_metrics(output_dir / "metrics.csv", row)
                save_checkpoint(
                    output_dir / "last.pt",
                    raw_model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    mapping,
                    config,
                    args,
                )
                current_map50 = float(row.get("val_map50", -math.inf))
                is_best = current_map50 > best_map50
                if is_best:
                    best_map50 = current_map50
                    save_checkpoint(
                        output_dir / "best.pt",
                        raw_model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        mapping,
                        config,
                        args,
                    )
                summary_text = " ".join(
                    f"{key}={value:.4f}"
                    for key, value in sorted(row.items())
                    if isinstance(value, float)
                )
                print(
                    f"epoch_summary {summary_text} "
                    f"best_map50={best_map50:.4f}",
                    flush=True,
                )
            distributed_barrier(distributed)

        distributed_barrier(distributed)
        if args.test_after_train and distributed.is_main:
            checkpoint_path = output_dir / f"{args.test_checkpoint}.pt"
            checkpoint_epoch = load_model_checkpoint_for_eval(checkpoint_path, raw_model, device)
            set_odam_loss_weight(
                raw_model,
                compute_odam_loss_weight(checkpoint_epoch, args),
            )
            test_dataset_full = CocoDrillBitDataset(
                args.data_root,
                args.test_split,
                mapping=mapping,
                image_size=args.image_size,
                keep_empty=True,
            )
            test_dataset = maybe_subset(test_dataset_full, args.max_test_images)
            test_loader = build_loader(
                test_dataset,
                batch_size=1,
                workers=args.workers,
                shuffle=False,
                pin_memory=device.type == "cuda",
            )
            test_loss_metrics = validate_loss(raw_model, test_loader, device, args, phase="test")
            test_coco_metrics = evaluate_coco(
                raw_model,
                test_loader,
                device,
                mapping,
                args.test_split,
                args.score_threshold,
                args.log_every,
                args.log_first_n,
            )
            payload = {
                "split": args.test_split,
                "checkpoint": args.test_checkpoint,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_epoch": checkpoint_epoch,
                "images": len(test_dataset),
                "loss_metrics": test_loss_metrics,
                "coco_metrics": test_coco_metrics,
            }
            (output_dir / "test_metrics.json").write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
            print(
                f"phase=test done map_50_95={test_coco_metrics['map_50_95']:.4f} "
                f"map50={test_coco_metrics['map50']:.4f} "
                f"saved={output_dir / 'test_metrics.json'}",
                flush=True,
            )
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
