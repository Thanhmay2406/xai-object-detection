#!/usr/bin/env python3
"""
train.py
========

Huấn luyện và đánh giá 3 biến thể trên CÙNG một pipeline:

    1) baseline : Faster R-CNN
    2) odam     : Faster R-CNN + ODAM-Train
    3) dpga     : Faster R-CNN + DPGA-ODAM

Yêu cầu dataset:
    COCO JSON detection format.

Phụ thuộc:
    pip install torch torchvision pycocotools pillow numpy tqdm

Ví dụ:
--------
# Faster R-CNN baseline
python train.py \
    --method baseline \
    --train-images /data/train \
    --train-ann /data/annotations/train.json \
    --val-images /data/val \
    --val-ann /data/annotations/val.json \
    --output runs/baseline

# Faster R-CNN + ODAM
python train.py \
    --method odam \
    --odam-weight 0.2 \
    --train-images /data/train \
    --train-ann /data/annotations/train.json \
    --val-images /data/val \
    --val-ann /data/annotations/val.json \
    --output runs/odam

# Faster R-CNN + DPGA-ODAM
python train.py \
    --method dpga \
    --dpga-warmup 4 \
    --dpga-rampup 4 \
    --train-images /data/train \
    --train-ann /data/annotations/train.json \
    --val-images /data/val \
    --val-ann /data/annotations/val.json \
    --output runs/dpga

Multi-GPU (torchrun):
---------------------
torchrun --nproc_per_node=2 train.py ...same args...

Metrics:
--------
COCO:
    AP       = AP@[IoU=.50:.95]
    AP50
    AP75
    AP_small / AP_medium / AP_large
    AR1 / AR10 / AR100

Pedestrian-only optional:
    MR-2_generic:
        log-average miss rate tại FPPI 10^-2 ... 10^0, IoU=0.5.

QUAN TRỌNG:
    MR-2_generic trong file này KHÔNG thay thế evaluator chính thức của
    CityPersons "Reasonable" vì protocol CityPersons còn có filtering theo
    chiều cao/visibility/ignore regions. Script xuất COCO prediction JSON để
    bạn có thể chạy evaluator chính thức riêng cho paper.

Scientific fairness:
--------------------
Mọi method dùng cùng:
    - Network architecture
    - train/val split
    - seed
    - optimizer
    - learning-rate schedule
    - resize
    - NMS/evaluation threshold

Chỉ khác:
    baseline:
        L = L_det
        và tắt hoàn toàn nhánh ODAM trong training.

    odam:
        L = L_det + lambda_odam * L_odam

    dpga:
        không gọi total_loss.backward();
        DPGAController tạo:
            g_final^(m)
              = g_det^(m)
              + alpha(epoch) * gate_m * g_odam_safe^(m)
"""

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision.ops import nms
from torchvision.transforms.functional import pil_to_tensor

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as exc:
    raise ImportError(
        "pycocotools is required. Install with: pip install pycocotools"
    ) from exc

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from network import (
    Network,
    DPGAConfig,
    DPGAController,
    DPGAModulePolicy,
    allreduce_gradient_list_mean,
    build_dpga_groups,
    format_dpga_stats,
    split_detection_and_odam_loss,
    validate_config,
)


# =============================================================================
# Detector configuration
# =============================================================================


@dataclass
class DetectorConfig:
    # Image normalization
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Backbone
    backbone_freeze_at: int = 0
    backbone_pretrained: bool = False

    # Filled from dataset at runtime.
    num_classes: int = 2

    # RPN
    rpn_channel: int = 256
    anchor_base_size: int = 16
    anchor_base_scale: Tuple[float, ...] = (2.0,)
    anchor_aspect_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0)
    num_cell_anchors: int = 3

    rpn_min_box_size: float = 0.0
    rpn_nms_threshold: float = 0.7

    train_prev_nms_top_n: int = 2000
    train_post_nms_top_n: int = 1000
    test_prev_nms_top_n: int = 1000
    test_post_nms_top_n: int = 1000

    num_sample_anchors: int = 256
    positive_anchor_ratio: float = 0.5
    rpn_positive_overlap: float = 0.7
    rpn_negative_overlap: float = 0.3
    rpn_ignore_overlap: float = 0.5

    rpn_bbox_normalize_targets: bool = False
    rpn_smooth_l1_beta: float = 1.0 / 9.0

    # Ignore label
    ignore_label: int = -1

    # ROI head
    num_rois: int = 512
    fg_ratio: float = 0.25
    fg_threshold: float = 0.5
    bg_threshold_high: float = 0.5
    bg_threshold_low: float = 0.0

    rcnn_bbox_normalize_targets: bool = True
    bbox_normalize_means: Tuple[float, float, float, float] = (
        0.0, 0.0, 0.0, 0.0
    )
    bbox_normalize_stds: Tuple[float, float, float, float] = (
        0.1, 0.1, 0.2, 0.2
    )
    rcnn_smooth_l1_beta: float = 1.0

    # Low threshold for COCO evaluation; postprocess NMS handles duplicates.
    pred_cls_threshold: float = 0.05

    # ODAM-only proposal quality filtering.
    odam_filtering: bool = False
    odam_min_iou: float = 0.7
    odam_min_score: float = 0.9
    odam_reliability: bool = False
    odam_reliability_iou_tau: float = 0.6
    odam_reliability_iou_temp: float = 0.1
    odam_reliability_score_tau: float = 0.7
    odam_reliability_score_temp: float = 0.1
    odam_reliability_adaptive_score_tau: bool = False
    odam_reliability_score_percentile: float = 0.70
    odam_reliability_budget_enabled: bool = False
    odam_reliability_budget_start: float = 1.0
    odam_reliability_budget_end: float = 1.0
    odam_reliability_budget_fraction: float = 1.0
    odam_reliability_budget_min: int = 1

    # Per-process batch size; set from CLI.
    train_batch_per_gpu: int = 1


@dataclass(frozen=True)
class CheckpointLoadResult:
    loaded_keys: Set[str]
    missing_keys: List[str]
    unexpected_keys: List[str]
    shape_mismatch_keys: List[str]
    matched_tensor_count: int
    total_tensor_count: int

    @property
    def loaded(self) -> bool:
        return self.matched_tensor_count > 0

    @property
    def match_ratio(self) -> float:
        return self.matched_tensor_count / max(self.total_tensor_count, 1)


# =============================================================================
# Reproducibility / distributed
# =============================================================================


def set_seed(seed: int, rank: int = 0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if not distributed:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed mode currently expects CUDA/NCCL.")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    return True, rank, world_size, local_rank


def is_main_process(rank: int) -> bool:
    return rank == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return value

    value = value.detach().clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= dist.get_world_size()
    return value


def current_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def manual_allreduce_grads(model: nn.Module):
    """
    Legacy helper for code paths that manually assign final gradients.

    DPGA must not call this after DPGAController.backward(), because the
    controller already averages raw detection and ODAM gradients before applying
    nonlinear gradient surgery.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return

    world_size = dist.get_world_size()

    for p in unwrap_model(model).parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


def gather_objects(obj, rank: int, world_size: int):
    if world_size == 1:
        return [obj]

    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(
        obj,
        object_gather_list=gathered,
        dst=0,
    )
    return gathered


class DistributedEvalSampler(Sampler):
    """
    Evaluation sampler without padding/duplication.

    torch DistributedSampler(drop_last=False) pads indices when dataset size is
    not divisible by world_size, which can duplicate detections and corrupt
    COCO metrics. This sampler partitions validation indices exactly once.
    """

    def __init__(self, dataset: Dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(
            range(
                self.rank,
                len(self.dataset),
                self.world_size,
            )
        )

    def __len__(self):
        n = len(self.dataset)
        if self.rank >= n:
            return 0
        return (n - 1 - self.rank) // self.world_size + 1


# =============================================================================
# COCO Dataset
# =============================================================================


def _compute_resize(
    height: int,
    width: int,
    min_size: int,
    max_size: int,
) -> Tuple[int, int]:
    if min_size <= 0:
        return height, width

    scale = float(min_size) / float(min(height, width))

    if max_size > 0 and max(height, width) * scale > max_size:
        scale = float(max_size) / float(max(height, width))

    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    return new_h, new_w


class CocoDetectionTrainDataset(Dataset):
    """
    COCO JSON -> tensor format expected by the self-contained detector.

    Internal class labels are remapped:
        COCO category ids -> {1, 2, ..., K}
    background = 0
    ignore = -1
    """

    def __init__(
        self,
        image_root: str,
        annotation_file: str,
        min_size: int = 800,
        max_size: int = 1333,
    ):
        self.image_root = Path(image_root)
        self.coco = COCO(annotation_file)
        self.image_ids = sorted(self.coco.getImgIds())

        self.category_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {
            cat_id: idx + 1
            for idx, cat_id in enumerate(self.category_ids)
        }
        self.label_to_cat_id = {
            label: cat_id
            for cat_id, label in self.cat_id_to_label.items()
        }

        self.min_size = int(min_size)
        self.max_size = int(max_size)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = int(self.image_ids[index])
        info = self.coco.loadImgs([image_id])[0]

        image_path = self.image_root / info["file_name"]
        image = Image.open(image_path).convert("RGB")

        orig_w, orig_h = image.size

        new_h, new_w = _compute_resize(
            orig_h,
            orig_w,
            self.min_size,
            self.max_size,
        )

        if (new_h, new_w) != (orig_h, orig_w):
            image = image.resize(
                (new_w, new_h),
                resample=Image.BILINEAR,
            )

        image_tensor = pil_to_tensor(image).float() / 255.0

        sx = float(new_w) / float(orig_w)
        sy = float(new_h) / float(orig_h)

        ann_ids = self.coco.getAnnIds(
            imgIds=[image_id],
        )
        annotations = self.coco.loadAnns(ann_ids)

        gt_rows = []

        for ann in annotations:
            if "bbox" not in ann:
                continue

            x, y, w, h = map(float, ann["bbox"])
            if w <= 0 or h <= 0:
                continue

            x1 = x * sx
            y1 = y * sy
            x2 = (x + w) * sx
            y2 = (y + h) * sy

            # Ignore/crowd are marked -1 for ROI target policy.
            if int(ann.get("ignore", 0)) == 1 or int(ann.get("iscrowd", 0)) == 1:
                label = -1
            else:
                category_id = int(ann["category_id"])
                label = self.cat_id_to_label[category_id]

            gt_rows.append(
                [x1, y1, x2, y2, float(label)]
            )

        if gt_rows:
            gt_boxes = torch.tensor(
                gt_rows,
                dtype=torch.float32,
            )
        else:
            gt_boxes = torch.zeros(
                (0, 5),
                dtype=torch.float32,
            )

        meta = {
            "image_id": image_id,
            "orig_h": orig_h,
            "orig_w": orig_w,
            "resized_h": new_h,
            "resized_w": new_w,
            "scale_x": sx,
            "scale_y": sy,
        }

        return image_tensor, gt_boxes, meta


def detection_collate(batch):
    """
    Pads image tensors to same H/W within a batch.

    Network will additionally pad H/W to multiple of 64.
    """
    images, gt_list, meta_list = zip(*batch)

    batch_size = len(images)
    channels = images[0].shape[0]
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    image_batch = images[0].new_zeros(
        (batch_size, channels, max_h, max_w)
    )

    for i, image in enumerate(images):
        h, w = image.shape[-2:]
        image_batch[i, :, :h, :w] = image

    max_gt = max(max(gt.shape[0] for gt in gt_list), 1)
    gt_batch = images[0].new_zeros(
        (batch_size, max_gt, 5)
    )

    im_info = images[0].new_zeros(
        (batch_size, 6)
    )

    for i, (gt, meta) in enumerate(zip(gt_list, meta_list)):
        if gt.shape[0] > 0:
            gt_batch[i, :gt.shape[0]] = gt

        # Fields used by detector:
        # 0 = resized height
        # 1 = resized width
        # 2 = scale
        # 5 = number GT
        im_info[i, 0] = float(meta["resized_h"])
        im_info[i, 1] = float(meta["resized_w"])
        im_info[i, 2] = float(
            (meta["scale_x"] + meta["scale_y"]) * 0.5
        )
        im_info[i, 3] = float(meta["orig_h"])
        im_info[i, 4] = float(meta["orig_w"])
        im_info[i, 5] = float(gt.shape[0])

    return image_batch, im_info, gt_batch, list(meta_list)


# =============================================================================
# Post-processing
# =============================================================================


@torch.no_grad()
def postprocess_single_image(
    pred: torch.Tensor,
    meta: Dict,
    label_to_cat_id: Dict[int, int],
    score_threshold: float,
    nms_threshold: float,
    max_detections: int,
) -> List[Dict]:
    """
    pred columns:
        [x1, y1, x2, y2, score, internal_class, ...optional DAM]
    """
    if pred.numel() == 0:
        return []

    pred = pred[:, :6]
    pred = pred[
        torch.isfinite(pred).all(dim=1)
    ]

    if pred.numel() == 0:
        return []

    boxes = pred[:, :4]
    scores = pred[:, 4]
    labels = pred[:, 5].long()

    keep_score = scores >= float(score_threshold)
    boxes = boxes[keep_score]
    scores = scores[keep_score]
    labels = labels[keep_score]

    if boxes.numel() == 0:
        return []

    # Clip in resized-image coordinates.
    rh = float(meta["resized_h"])
    rw = float(meta["resized_w"])

    boxes[:, 0::2].clamp_(min=0, max=max(rw - 1, 0))
    boxes[:, 1::2].clamp_(min=0, max=max(rh - 1, 0))

    keep_all = []

    for cls in labels.unique():
        cls_inds = torch.nonzero(
            labels == cls,
            as_tuple=False,
        ).squeeze(1)

        cls_keep = nms(
            boxes[cls_inds],
            scores[cls_inds],
            float(nms_threshold),
        )

        keep_all.append(cls_inds[cls_keep])

    if not keep_all:
        return []

    keep = torch.cat(keep_all, dim=0)

    # Global top-k after per-class NMS.
    keep = keep[
        scores[keep].argsort(descending=True)
    ][: int(max_detections)]

    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    sx = float(meta["scale_x"])
    sy = float(meta["scale_y"])

    boxes = boxes.clone()
    boxes[:, [0, 2]] /= sx
    boxes[:, [1, 3]] /= sy

    oh = float(meta["orig_h"])
    ow = float(meta["orig_w"])

    boxes[:, 0::2].clamp_(min=0, max=max(ow - 1, 0))
    boxes[:, 1::2].clamp_(min=0, max=max(oh - 1, 0))

    output = []

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(float, box.tolist())

        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)

        internal_label = int(label.item())
        if internal_label not in label_to_cat_id:
            continue

        output.append(
            {
                "image_id": int(meta["image_id"]),
                "category_id": int(
                    label_to_cat_id[internal_label]
                ),
                "bbox": [x1, y1, w, h],
                "score": float(score.item()),
            }
        )

    return output


# =============================================================================
# Metrics
# =============================================================================


def evaluate_coco(
    coco_gt: COCO,
    predictions: List[Dict],
    image_ids: Sequence[int],
) -> Dict[str, float]:
    if len(predictions) == 0:
        return {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
            "AR1": 0.0,
            "AR10": 0.0,
            "AR100": 0.0,
        }

    coco_dt = coco_gt.loadRes(predictions)

    evaluator = COCOeval(
        coco_gt,
        coco_dt,
        iouType="bbox",
    )

    evaluator.params.imgIds = list(map(int, image_ids))

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    s = evaluator.stats

    return {
        "AP": float(s[0]),
        "AP50": float(s[1]),
        "AP75": float(s[2]),
        "AP_small": float(s[3]),
        "AP_medium": float(s[4]),
        "AP_large": float(s[5]),
        "AR1": float(s[6]),
        "AR10": float(s[7]),
        "AR100": float(s[8]),
    }


def _xywh_to_xyxy(box):
    x, y, w, h = map(float, box)
    return np.array(
        [x, y, x + w, y + h],
        dtype=np.float64,
    )


def _iou_numpy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    iw = np.maximum(0.0, x2 - x1)
    ih = np.maximum(0.0, y2 - y1)
    inter = iw * ih

    area_a = max(0.0, box[2] - box[0]) * max(
        0.0, box[3] - box[1]
    )
    area_b = np.maximum(
        0.0,
        boxes[:, 2] - boxes[:, 0],
    ) * np.maximum(
        0.0,
        boxes[:, 3] - boxes[:, 1],
    )

    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-12)


def compute_generic_mr2(
    coco_gt: COCO,
    predictions: List[Dict],
    category_id: int,
    image_ids: Sequence[int],
    iou_threshold: float = 0.5,
) -> float:
    """
    Generic log-average miss rate at FPPI = 10^-2 ... 10^0.

    This is useful as a sanity metric for single-class pedestrian detection,
    but is NOT the full official CityPersons Reasonable protocol.
    """
    gt_by_image: Dict[int, np.ndarray] = {}
    matched_by_image: Dict[int, np.ndarray] = {}

    total_gt = 0

    for image_id in image_ids:
        ann_ids = coco_gt.getAnnIds(
            imgIds=[int(image_id)],
            catIds=[int(category_id)],
            iscrowd=False,
        )
        anns = coco_gt.loadAnns(ann_ids)

        boxes = [
            _xywh_to_xyxy(ann["bbox"])
            for ann in anns
            if int(ann.get("ignore", 0)) == 0
        ]

        if boxes:
            arr = np.stack(boxes, axis=0)
        else:
            arr = np.zeros((0, 4), dtype=np.float64)

        gt_by_image[int(image_id)] = arr
        matched_by_image[int(image_id)] = np.zeros(
            (arr.shape[0],),
            dtype=bool,
        )
        total_gt += arr.shape[0]

    if total_gt == 0:
        return float("nan")

    dets = [
        pred
        for pred in predictions
        if int(pred["category_id"]) == int(category_id)
    ]
    dets.sort(
        key=lambda x: float(x["score"]),
        reverse=True,
    )

    tp = []
    fp = []

    for det in dets:
        image_id = int(det["image_id"])
        det_box = _xywh_to_xyxy(det["bbox"])

        gt_boxes = gt_by_image.get(
            image_id,
            np.zeros((0, 4), dtype=np.float64),
        )
        matched = matched_by_image.get(
            image_id,
            np.zeros((0,), dtype=bool),
        )

        if gt_boxes.shape[0] == 0:
            tp.append(0.0)
            fp.append(1.0)
            continue

        ious = _iou_numpy(
            det_box,
            gt_boxes,
        )

        # Do not rematch already matched GT.
        ious[matched] = -1.0

        best = int(np.argmax(ious))
        best_iou = float(ious[best])

        if best_iou >= iou_threshold:
            matched[best] = True
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)

    if len(tp) == 0:
        return 1.0

    tp = np.cumsum(np.asarray(tp, dtype=np.float64))
    fp = np.cumsum(np.asarray(fp, dtype=np.float64))

    miss_rate = 1.0 - tp / float(total_gt)
    fppi = fp / float(max(len(image_ids), 1))

    refs = np.logspace(-2.0, 0.0, 9)
    sampled_miss = []

    for ref in refs:
        valid = np.where(fppi <= ref)[0]
        if len(valid) == 0:
            sampled_miss.append(1.0)
        else:
            sampled_miss.append(
                float(miss_rate[valid[-1]])
            )

    sampled_miss = np.clip(
        np.asarray(sampled_miss, dtype=np.float64),
        1e-10,
        1.0,
    )

    return float(
        np.exp(np.mean(np.log(sampled_miss)))
    )


def _odam_quality_enabled(args) -> bool:
    configured = getattr(args, "eval_odam_quality", None)
    if configured is not None:
        return bool(configured)
    return getattr(args, "method", None) in ("odam", "dpga")


def _dam_energy_in_box(
    dam_flat: torch.Tensor,
    dam_h: int,
    dam_w: int,
    gt_box: torch.Tensor,
    resized_h: float,
    resized_w: float,
) -> Tuple[float, float]:
    if dam_h <= 0 or dam_w <= 0 or dam_flat.numel() != dam_h * dam_w:
        return float("nan"), 0.0

    dam = dam_flat.reshape(dam_h, dam_w).float().clamp(min=0)
    total = float(dam.sum().detach().cpu())
    if total <= 1e-12:
        return float("nan"), total

    x_scale = float(dam_w) / max(float(resized_w), 1.0)
    y_scale = float(dam_h) / max(float(resized_h), 1.0)

    x1 = int(math.floor(float(gt_box[0]) * x_scale))
    y1 = int(math.floor(float(gt_box[1]) * y_scale))
    x2 = int(math.ceil(float(gt_box[2]) * x_scale))
    y2 = int(math.ceil(float(gt_box[3]) * y_scale))

    x1 = min(max(x1, 0), dam_w)
    x2 = min(max(x2, 0), dam_w)
    y1 = min(max(y1, 0), dam_h)
    y2 = min(max(y2, 0), dam_h)

    if x2 <= x1 or y2 <= y1:
        return 0.0, total

    inside = float(dam[y1:y2, x1:x2].sum().detach().cpu())
    return inside / max(total, 1e-12), total


def compute_odam_quality_rows(
    pred: torch.Tensor,
    gt_boxes: torch.Tensor,
    meta: Dict,
    score_threshold: float,
    iou_threshold: float,
) -> List[Dict]:
    """
    DAM localization sanity metric.

    For matched prediction/GT pairs, measure the fraction of positive DAM energy
    that falls inside the matched GT box. This is an internal ODAM/XAI diagnostic,
    not an official dataset metric.
    """
    if pred.numel() == 0 or pred.shape[1] <= 8:
        return []

    valid_gt = gt_boxes[
        (gt_boxes[:, 4] > 0)
        & torch.isfinite(gt_boxes).all(dim=1)
    ]
    if valid_gt.numel() == 0:
        return []

    pred = pred[
        torch.isfinite(pred[:, :6]).all(dim=1)
        & (pred[:, 4] >= float(score_threshold))
    ]
    if pred.numel() == 0:
        return []

    order = pred[:, 4].argsort(descending=True)
    pred = pred[order]

    gt_xyxy = valid_gt[:, :4]
    gt_labels = valid_gt[:, 4].long()
    gt_matched = torch.zeros(
        (valid_gt.shape[0],),
        dtype=torch.bool,
        device=valid_gt.device,
    )

    rows = []
    for det in pred:
        label = int(det[5].item())
        same_label = gt_labels == label
        available = same_label & (~gt_matched)
        if not available.any():
            continue

        candidate_indices = torch.nonzero(
            available,
            as_tuple=False,
        ).squeeze(1)
        ious = torch.as_tensor(
            _iou_numpy(
                det[:4].detach().cpu().numpy(),
                gt_xyxy[candidate_indices].detach().cpu().numpy(),
            ),
            device=gt_boxes.device,
        )
        best_local = int(torch.argmax(ious).item())
        best_iou = float(ious[best_local].detach().cpu())
        if best_iou < float(iou_threshold):
            continue

        gt_index = candidate_indices[best_local]
        gt_matched[gt_index] = True

        dam_h = int(round(float(det[-2].detach().cpu())))
        dam_w = int(round(float(det[-1].detach().cpu())))
        energy, total = _dam_energy_in_box(
            det[6:-2],
            dam_h,
            dam_w,
            valid_gt[gt_index, :4],
            resized_h=float(meta["resized_h"]),
            resized_w=float(meta["resized_w"]),
        )

        if not math.isfinite(energy):
            continue

        rows.append(
            {
                "image_id": int(meta["image_id"]),
                "category_label": label,
                "score": float(det[4].detach().cpu()),
                "iou": best_iou,
                "dam_energy_in_gt": energy,
                "dam_energy_total": total,
                "dam_h": dam_h,
                "dam_w": dam_w,
            }
        )

    return rows


def summarize_odam_quality(rows: Sequence[Dict]) -> Dict[str, float]:
    if not rows:
        return {
            "ODAM_quality": float("nan"),
            "ODAM_quality_samples": 0.0,
            "ODAM_quality_mean_iou": float("nan"),
        }

    energy = np.asarray(
        [float(row["dam_energy_in_gt"]) for row in rows],
        dtype=np.float64,
    )
    iou = np.asarray(
        [float(row["iou"]) for row in rows],
        dtype=np.float64,
    )
    return {
        "ODAM_quality": float(np.mean(energy)),
        "ODAM_quality_samples": float(len(rows)),
        "ODAM_quality_mean_iou": float(np.mean(iou)),
    }


# =============================================================================
# Validation
# =============================================================================


def validate(
    model: nn.Module,
    loader: DataLoader,
    dataset: CocoDetectionTrainDataset,
    device: torch.device,
    args,
    rank: int,
    world_size: int,
    output_dir: Path,
    epoch: int,
) -> Dict[str, float]:
    model.eval()
    raw_model = unwrap_model(model)
    eval_odam_quality = _odam_quality_enabled(args)
    raw_model.set_odam_inference(eval_odam_quality)

    predictions_local: List[Dict] = []
    odam_quality_rows_local: List[Dict] = []
    image_ids_local: List[int] = []

    iterator = loader
    if tqdm is not None and rank == 0:
        iterator = tqdm(
            loader,
            desc=f"val {epoch:03d}",
            leave=False,
        )

    grad_context = (
        torch.enable_grad()
        if eval_odam_quality
        else torch.no_grad()
    )

    with grad_context:
        for image, im_info, gt_boxes, metas in iterator:
            image = image.to(
                device,
                non_blocking=True,
            )
            im_info = im_info.to(
                device,
                non_blocking=True,
            )
            gt_boxes = gt_boxes.to(
                device,
                non_blocking=True,
            )

            pred = model(
                image,
                im_info,
            )

            # Current Network returns concatenated batch predictions only through
            # rcnn_rois. With batch_size=1 val this is unambiguous.
            # We enforce val batch size 1 below.
            if len(metas) != 1:
                raise RuntimeError(
                    "Validation currently requires --val-batch-size=1 "
                    "because RCNN output has no explicit batch column."
                )

            meta = metas[0]

            pred_list = postprocess_single_image(
                pred,
                meta=meta,
                label_to_cat_id=dataset.label_to_cat_id,
                score_threshold=args.eval_score_threshold,
                nms_threshold=args.eval_nms,
                max_detections=args.max_detections,
            )

            predictions_local.extend(pred_list)
            if eval_odam_quality:
                odam_quality_rows_local.extend(
                    compute_odam_quality_rows(
                        pred=pred,
                        gt_boxes=gt_boxes[0],
                        meta=meta,
                        score_threshold=args.eval_score_threshold,
                        iou_threshold=args.odam_quality_iou,
                    )
                )
            image_ids_local.append(
                int(meta["image_id"])
            )

    raw_model.set_odam_inference(False)

    gathered_predictions = gather_objects(
        predictions_local,
        rank,
        world_size,
    )
    gathered_ids = gather_objects(
        image_ids_local,
        rank,
        world_size,
    )
    gathered_quality = gather_objects(
        odam_quality_rows_local,
        rank,
        world_size,
    )

    if rank != 0:
        return {}

    predictions = []
    image_ids = []
    odam_quality_rows = []

    for part in gathered_predictions:
        predictions.extend(part)

    for part in gathered_ids:
        image_ids.extend(part)

    for part in gathered_quality:
        odam_quality_rows.extend(part)

    image_ids = sorted(set(image_ids))

    metrics = evaluate_coco(
        dataset.coco,
        predictions,
        image_ids,
    )

    if len(dataset.category_ids) == 1:
        metrics["MR-2_generic"] = compute_generic_mr2(
            dataset.coco,
            predictions,
            category_id=dataset.category_ids[0],
            image_ids=image_ids,
            iou_threshold=0.5,
        )

    if eval_odam_quality:
        metrics.update(
            summarize_odam_quality(odam_quality_rows)
        )
        quality_file = output_dir / f"odam_quality_epoch_{epoch:03d}.json"
        quality_file.write_text(
            json.dumps(odam_quality_rows),
            encoding="utf-8",
        )

    pred_file = output_dir / f"predictions_epoch_{epoch:03d}.json"
    pred_file.write_text(
        json.dumps(predictions),
        encoding="utf-8",
    )

    return metrics


# =============================================================================
# Training
# =============================================================================


DPGA_ABLATION_PRESETS = {
    "full": {
        "projection": True,
        "norm_cap": True,
        "gate": True,
        "label": "A6_full_dpga",
    },
    "projection-only": {
        "projection": True,
        "norm_cap": False,
        "gate": False,
        "label": "A2_projection",
    },
    "norm-cap-only": {
        "projection": False,
        "norm_cap": True,
        "gate": False,
        "label": "A3_norm_cap",
    },
    "gate-only": {
        "projection": False,
        "norm_cap": False,
        "gate": True,
        "label": "A4_gate",
    },
    "projection-norm-cap": {
        "projection": True,
        "norm_cap": True,
        "gate": False,
        "label": "A5_projection_norm_cap",
    },
}


@dataclass(frozen=True)
class ExperimentStageConfig:
    stage: str
    warmup_enabled: bool
    filtering_enabled: bool
    reliability_enabled: bool
    projection_enabled: bool
    norm_cap_enabled: bool
    gate_enabled: bool
    odam_min_iou: Optional[float] = None
    odam_min_score: Optional[float] = None
    adaptive_score_tau: bool = False
    score_percentile: Optional[float] = None
    budget_enabled: bool = False
    budget_start: Optional[float] = None
    budget_end: Optional[float] = None
    budget_min: Optional[int] = None

    @property
    def requires_dpga(self) -> bool:
        return (
            self.projection_enabled
            or self.norm_cap_enabled
            or self.gate_enabled
        )


EXPERIMENT_STAGE_PRESETS = {
    "E0": ExperimentStageConfig(
        "E0",
        warmup_enabled=False,
        filtering_enabled=False,
        reliability_enabled=False,
        projection_enabled=False,
        norm_cap_enabled=False,
        gate_enabled=False,
    ),
    "E1": ExperimentStageConfig(
        "E1",
        warmup_enabled=True,
        filtering_enabled=False,
        reliability_enabled=False,
        projection_enabled=False,
        norm_cap_enabled=False,
        gate_enabled=False,
    ),
    "E2": ExperimentStageConfig(
        "E2",
        warmup_enabled=True,
        filtering_enabled=True,
        reliability_enabled=False,
        projection_enabled=False,
        norm_cap_enabled=False,
        gate_enabled=False,
    ),
    "E3": ExperimentStageConfig(
        "E3",
        warmup_enabled=True,
        filtering_enabled=True,
        reliability_enabled=False,
        projection_enabled=True,
        norm_cap_enabled=False,
        gate_enabled=False,
    ),
    "E4": ExperimentStageConfig(
        "E4",
        warmup_enabled=True,
        filtering_enabled=True,
        reliability_enabled=False,
        projection_enabled=True,
        norm_cap_enabled=True,
        gate_enabled=False,
    ),
    "E5": ExperimentStageConfig(
        "E5",
        warmup_enabled=True,
        filtering_enabled=True,
        reliability_enabled=False,
        projection_enabled=True,
        norm_cap_enabled=True,
        gate_enabled=True,
    ),
    "E6": ExperimentStageConfig(
        "E6",
        warmup_enabled=True,
        filtering_enabled=False,
        reliability_enabled=True,
        projection_enabled=True,
        norm_cap_enabled=True,
        gate_enabled=True,
    ),
    "E7": ExperimentStageConfig(
        "E7",
        warmup_enabled=True,
        filtering_enabled=True,
        reliability_enabled=True,
        projection_enabled=True,
        norm_cap_enabled=True,
        gate_enabled=True,
        odam_min_iou=0.5,
        odam_min_score=0.0,
        adaptive_score_tau=True,
        score_percentile=0.70,
        budget_enabled=True,
        budget_start=0.25,
        budget_end=0.50,
        budget_min=1,
    ),
}


AUX_SAFETY_EPS = 1e-8
METRICS_SCHEMA_VERSION = 5
GRADIENT_DIAGNOSTIC_SCHEMA_VERSION = 4


def resolve_experiment_stage_config(stage: str) -> ExperimentStageConfig:
    try:
        return EXPERIMENT_STAGE_PRESETS[str(stage)]
    except KeyError as exc:
        raise ValueError(
            "--experiment-stage must be one of {E0,E1,E2,E3,E4,E5,E6,E7}"
        ) from exc


def _cli_option_supplied(argv: Sequence[str], option: str) -> bool:
    return any(arg == option or arg.startswith(option + "=") for arg in argv)


def _odam_reliability_budget_fraction_for_epoch(args, epoch: float) -> float:
    if not bool(getattr(args, "odam_reliability_budget_enabled", False)):
        return 1.0
    start = float(getattr(args, "odam_reliability_budget_start", 1.0))
    end = float(getattr(args, "odam_reliability_budget_end", 1.0))
    if int(getattr(args, "epochs", 1)) <= 1:
        return end
    progress = float(epoch) / float(max(int(args.epochs) - 1, 1))
    progress = min(max(progress, 0.0), 1.0)
    return start + (end - start) * progress


def _odam_weight_for_epoch(args, epoch: float) -> float:
    if not getattr(args, "warmup_enabled", False):
        return float(args.odam_weight)

    warmup_epochs = int(args.dpga_warmup)
    rampup_epochs = int(args.dpga_rampup)
    alpha_max = float(args.odam_weight)
    epoch = float(epoch)

    if epoch < warmup_epochs:
        return 0.0
    if rampup_epochs <= 0:
        return alpha_max

    progress = (epoch - float(warmup_epochs)) / float(rampup_epochs)
    progress = min(max(progress, 0.0), 1.0)
    return alpha_max * progress


def apply_experiment_stage_preset(
    args,
    argv: Optional[Sequence[str]] = None,
):
    argv = list(sys.argv[1:] if argv is None else argv)
    args.warmup_enabled = False
    args.filtering_enabled = bool(args.odam_filtering)
    args.reliability_enabled = bool(args.odam_reliability)
    args.projection_enabled = bool(args.dpga_projection)
    args.norm_cap_enabled = bool(args.dpga_norm_cap)
    args.gate_enabled = bool(args.dpga_gate)
    args.odam_reliability_budget_fraction = (
        _odam_reliability_budget_fraction_for_epoch(args, 0.0)
    )

    if args.experiment_stage is None:
        args = apply_dpga_ablation_preset(args)
        args.projection_enabled = (
            args.method == "dpga" and bool(args.dpga_projection)
        )
        args.norm_cap_enabled = (
            args.method == "dpga" and bool(args.dpga_norm_cap)
        )
        args.gate_enabled = (
            args.method == "dpga" and bool(args.dpga_gate)
        )
        args.warmup_enabled = (
            args.method == "dpga"
            and (args.dpga_warmup > 0 or args.dpga_rampup > 0)
        )
        args.filtering_enabled = bool(args.odam_filtering)
        args.reliability_enabled = bool(args.odam_reliability)
        return args

    for option in (
        "--dpga-ablation",
        "--no-dpga-projection",
        "--no-dpga-norm-cap",
        "--no-dpga-gate",
    ):
        if _cli_option_supplied(argv, option):
            raise ValueError(
                f"{option} cannot be combined with --experiment-stage; "
                "the experiment stage is the ablation source of truth."
            )

    stage = resolve_experiment_stage_config(args.experiment_stage)
    if args.method == "baseline":
        raise ValueError("--experiment-stage is only valid for ODAM/DPGA runs.")
    if stage.requires_dpga and args.method != "dpga":
        raise ValueError(
            f"--experiment-stage {stage.stage} requires --method dpga."
        )
    if not stage.requires_dpga and args.method != "odam":
        raise ValueError(
            f"--experiment-stage {stage.stage} is an ODAM stage and "
            "requires --method odam."
        )
    if (
        _cli_option_supplied(argv, "--odam-filtering")
        and not stage.filtering_enabled
    ):
        raise ValueError(
            "--odam-filtering conflicts with this --experiment-stage."
        )
    if (
        _cli_option_supplied(argv, "--odam-reliability")
        and not stage.reliability_enabled
    ):
        raise ValueError(
            "--odam-reliability conflicts with this --experiment-stage."
        )

    args.warmup_enabled = stage.warmup_enabled
    args.odam_filtering = stage.filtering_enabled
    args.filtering_enabled = stage.filtering_enabled
    args.odam_reliability = stage.reliability_enabled
    args.reliability_enabled = stage.reliability_enabled
    args.dpga_projection = stage.projection_enabled
    args.dpga_norm_cap = stage.norm_cap_enabled
    args.dpga_gate = stage.gate_enabled
    args.projection_enabled = stage.projection_enabled
    args.norm_cap_enabled = stage.norm_cap_enabled
    args.gate_enabled = stage.gate_enabled
    if stage.odam_min_iou is not None:
        args.odam_min_iou = float(stage.odam_min_iou)
    if stage.odam_min_score is not None:
        args.odam_min_score = float(stage.odam_min_score)
    args.odam_reliability_adaptive_score_tau = bool(
        stage.adaptive_score_tau
    )
    if stage.score_percentile is not None:
        args.odam_reliability_score_percentile = float(
            stage.score_percentile
        )
    args.odam_reliability_budget_enabled = bool(stage.budget_enabled)
    if stage.budget_start is not None:
        args.odam_reliability_budget_start = float(stage.budget_start)
    if stage.budget_end is not None:
        args.odam_reliability_budget_end = float(stage.budget_end)
    if stage.budget_min is not None:
        args.odam_reliability_budget_min = int(stage.budget_min)
    args.odam_reliability_budget_fraction = (
        _odam_reliability_budget_fraction_for_epoch(args, 0.0)
    )
    args.dpga_ablation = None
    args.dpga_ablation_label = f"{stage.stage}_incremental"
    return args


def apply_dpga_ablation_preset(args):
    if args.method != "dpga":
        if args.dpga_ablation != "full":
            raise ValueError(
                "--dpga-ablation is only valid with --method dpga."
            )
        args.dpga_ablation_label = args.method
        return args

    if args.dpga_ablation == "custom":
        args.dpga_ablation_label = f"custom_{args.method}"
        return args

    preset = DPGA_ABLATION_PRESETS[args.dpga_ablation]
    args.dpga_projection = bool(preset["projection"])
    args.dpga_norm_cap = bool(preset["norm_cap"])
    args.dpga_gate = bool(preset["gate"])
    args.dpga_ablation_label = str(preset["label"])
    return args


def make_dpga_config(args) -> DPGAConfig:
    dpga_fields = {
        field.name
        for field in fields(DPGAConfig)
    }
    missing_fields = [
        name
        for name in ("use_norm_cap", "use_gate")
        if name not in dpga_fields
    ]
    if missing_fields:
        raise RuntimeError(
            "ODAM/network.py is older than ODAM/train.py. "
            "Sync ODAM/network.py from this repository before running "
            "--dpga-ablation. Missing DPGAConfig fields: "
            + ", ".join(missing_fields)
        )

    alpha_max = (
        args.odam_weight
        if args.experiment_stage is not None
        else args.dpga_alpha
    )

    return DPGAConfig(
        warmup_epochs=args.dpga_warmup,
        rampup_epochs=args.dpga_rampup,
        alpha_max=alpha_max,
        project_if_conflict=args.dpga_projection,
        use_norm_cap=args.dpga_norm_cap,
        use_gate=args.dpga_gate,
        conflict_threshold=0.0,
        module_policies={
            "backbone": DPGAModulePolicy(
                rho=args.rho_backbone,
                tau=0.0,
                temperature=args.dpga_temperature,
            ),
            "fpn": DPGAModulePolicy(
                rho=args.rho_fpn,
                tau=0.0,
                temperature=args.dpga_temperature,
            ),
            "rpn": DPGAModulePolicy(
                rho=args.rho_rpn,
                tau=0.0,
                temperature=args.dpga_temperature,
            ),
            "roi_shared": DPGAModulePolicy(
                rho=args.rho_roi_shared,
                tau=0.0,
                temperature=args.dpga_temperature,
            ),
            "roi_cls": DPGAModulePolicy(
                rho=args.rho_roi_cls,
                tau=0.0,
                temperature=args.dpga_temperature,
            ),
            "roi_reg": DPGAModulePolicy(
                rho=args.rho_roi_reg,
                tau=0.0,
                temperature=args.dpga_temperature,
            ),
        },
    )


def build_optimizer(model: nn.Module, args):
    params = [
        p for p in unwrap_model(model).parameters()
        if p.requires_grad
    ]

    return torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def build_scheduler(optimizer, args):
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(args.lr_steps),
        gamma=args.lr_gamma,
    )


def load_initial_weights(
    model: nn.Module,
    checkpoint_path: Optional[str],
) -> CheckpointLoadResult:
    raw_model = unwrap_model(model)
    model_state = raw_model.state_dict()

    if not checkpoint_path:
        return CheckpointLoadResult(
            loaded_keys=set(),
            missing_keys=list(model_state.keys()),
            unexpected_keys=[],
            shape_mismatch_keys=[],
            matched_tensor_count=0,
            total_tensor_count=len(model_state),
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise TypeError(
            f"Checkpoint {checkpoint_path} does not contain a state_dict."
        )

    loadable_state = {}
    skipped_shape = []
    for key, value in state.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        if normalized_key not in model_state:
            continue
        if not torch.is_tensor(value):
            continue
        if tuple(value.shape) != tuple(model_state[normalized_key].shape):
            skipped_shape.append(
                (
                    normalized_key,
                    tuple(value.shape),
                    tuple(model_state[normalized_key].shape),
                )
            )
            continue
        loadable_state[normalized_key] = value

    matched = len(loadable_state)
    total = len(model_state)
    match_ratio = matched / max(total, 1)
    if matched == 0 or match_ratio < 0.50:
        raise RuntimeError(
            "Checkpoint is incompatible with this model: "
            f"matched_tensors={matched}/{total} "
            f"match_ratio={match_ratio:.3f} path={checkpoint_path}"
        )

    missing, _ = unwrap_model(model).load_state_dict(
        loadable_state,
        strict=False,
    )

    unexpected = [
        key for key in state
        if (key[7:] if key.startswith("module.") else key) not in model_state
    ]
    print(
        f"[init] loaded {checkpoint_path}; "
        f"matched_tensors={matched}/{total} "
        f"match_ratio={match_ratio:.3f} "
        f"missing={len(missing)}, unexpected={len(unexpected)}, "
        f"shape_mismatch={len(skipped_shape)}"
    )
    if missing:
        print("[init] missing sample: " + ", ".join(missing[:10]))
    if unexpected:
        print("[init] unexpected sample: " + ", ".join(unexpected[:10]))
    if skipped_shape:
        sample = [
            f"{name}: checkpoint{src} != model{dst}"
            for name, src, dst in skipped_shape[:5]
        ]
        print("[init] shape mismatch sample: " + "; ".join(sample))
    return CheckpointLoadResult(
        loaded_keys=set(loadable_state.keys()),
        missing_keys=list(missing),
        unexpected_keys=list(unexpected),
        shape_mismatch_keys=[name for name, _, _ in skipped_shape],
        matched_tensor_count=matched,
        total_tensor_count=total,
    )


def _backbone_freeze_prefixes(freeze_at: int) -> List[str]:
    freeze_at = int(freeze_at)
    if freeze_at not in (0, 1, 2):
        raise ValueError("backbone_freeze_at must be one of {0, 1, 2}")

    prefixes: List[str] = []
    if freeze_at >= 1:
        prefixes.extend(
            [
                "resnet50.conv1.",
                "resnet50.bn1.",
            ]
        )
    if freeze_at >= 2:
        prefixes.append("resnet50.layer1.")
    return prefixes


def _required_backbone_state_keys(
    model: nn.Module,
    freeze_at: int,
) -> List[str]:
    prefixes = _backbone_freeze_prefixes(freeze_at)
    model_state = unwrap_model(model).state_dict()
    return sorted(
        key for key in model_state
        if any(key.startswith(prefix) for prefix in prefixes)
    )


def apply_backbone_freeze_policy(
    model: nn.Module,
    detector_config: DetectorConfig,
    checkpoint_result: CheckpointLoadResult,
):
    freeze_at = int(detector_config.backbone_freeze_at)
    if freeze_at not in (0, 1, 2):
        raise ValueError("backbone_freeze_at must be one of {0, 1, 2}")
    if freeze_at <= 0:
        return

    if not detector_config.backbone_pretrained:
        required_keys = _required_backbone_state_keys(model, freeze_at)
        missing_required = [
            key for key in required_keys
            if key not in checkpoint_result.loaded_keys
        ]
    else:
        missing_required = []

    if missing_required:
        sample = ", ".join(missing_required[:10])
        raise RuntimeError(
            "Refusing to freeze a randomly initialized backbone stage: "
            f"freeze_at={freeze_at} missing_checkpoint_keys="
            f"{len(missing_required)} sample=[{sample}]. Use "
            "--backbone-pretrained, provide a checkpoint that covers the "
            "frozen backbone stages, or set --backbone-freeze-at 0."
        )

    unwrap_model(model).freeze_backbone(freeze_at)
    print(f"[init] froze backbone stages up to {freeze_at}")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    method: str,
    detector_config: DetectorConfig,
    args,
    metrics: Dict[str, float],
    category_ids: Sequence[int],
    label_to_cat_id: Dict[int, int],
):
    torch.save(
        {
            "epoch": epoch,
            "method": method,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "detector_config": asdict(detector_config),
            "category_ids": list(map(int, category_ids)),
            "label_to_cat_id": {
                int(label): int(cat_id)
                for label, cat_id in label_to_cat_id.items()
            },
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def current_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)

    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def cuda_memory_summary(device: torch.device) -> str:
    if device.type != "cuda":
        return "mem=cpu"

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return (
        f"mem={allocated:.2f}G"
        f"/{reserved:.2f}G"
        f" peak={peak:.2f}G"
    )


def write_train_log(iterator, message: str):
    if tqdm is not None and hasattr(iterator, "write"):
        iterator.write(message)
    else:
        print(message)


def _first_nonfinite_tensor(
    tensors: Dict[str, torch.Tensor],
) -> Optional[Tuple[str, str, float]]:
    for name, tensor in tensors.items():
        if tensor is None or not torch.is_tensor(tensor):
            continue
        if not tensor.is_floating_point():
            continue
        finite = torch.isfinite(tensor)
        if bool(finite.all()):
            continue
        bad = tensor.detach()[~finite]
        if bool(torch.isnan(bad).any()):
            kind = "nan"
        elif bool(torch.isinf(bad).any()):
            kind = "inf"
        else:
            kind = "nonfinite"
        sample = float(bad.flatten()[0].detach().cpu())
        return name, kind, sample
    return None


def assert_finite_training_state(
    stage: str,
    epoch: int,
    step: int,
    rank: int,
    method: str,
    tensors: Optional[Dict[str, torch.Tensor]] = None,
    model: Optional[nn.Module] = None,
    check_grads: bool = False,
):
    if tensors:
        bad = _first_nonfinite_tensor(tensors)
        if bad is not None:
            name, kind, sample = bad
            raise FloatingPointError(
                "[nonfinite] "
                f"stage={stage} epoch={epoch} step={step} "
                f"rank={rank} method={method} tensor={name} "
                f"kind={kind} sample={sample}"
            )

    if model is None:
        return

    raw_model = unwrap_model(model)
    items = raw_model.named_parameters()
    for name, param in items:
        tensor = param.grad if check_grads else param
        if tensor is None or not tensor.is_floating_point():
            continue
        finite = torch.isfinite(tensor)
        if bool(finite.all()):
            continue
        bad = tensor.detach()[~finite]
        if bool(torch.isnan(bad).any()):
            kind = "nan"
        elif bool(torch.isinf(bad).any()):
            kind = "inf"
        else:
            kind = "nonfinite"
        sample = float(bad.flatten()[0].detach().cpu())
        target = "grad" if check_grads else "param"
        raise FloatingPointError(
            "[nonfinite] "
            f"stage={stage} epoch={epoch} step={step} "
            f"rank={rank} method={method} {target}={name} "
            f"kind={kind} sample={sample}"
        )


def should_log_step(args, step: int, total_steps: int) -> bool:
    interval = int(args.log_interval)
    return (
        interval > 0
        and (
            step == 0
            or (step + 1) % interval == 0
            or step == total_steps - 1
        )
    )


def format_train_step_log(
    args,
    epoch: int,
    step: int,
    total_steps: int,
    lr: float,
    loss_det: float,
    loss_odam: float,
    raw_loss_sum: float,
    avg_det: float,
    avg_odam: float,
    avg_raw_loss_sum: float,
    step_seconds: float,
    elapsed_seconds: float,
    device: torch.device,
) -> str:
    completed = step + 1
    avg_step_seconds = elapsed_seconds / max(completed, 1)
    eta_seconds = avg_step_seconds * max(total_steps - completed, 0)
    effective_batch = int(args.batch_size) * current_world_size()

    return (
        f"[train] epoch={epoch + 1}/{args.epochs} "
        f"step={completed}/{total_steps} "
        f"method={args.method} "
        f"experiment_stage={args.experiment_stage} "
        f"warmup={int(bool(args.warmup_enabled))} "
        f"filtering={int(bool(args.filtering_enabled))} "
        f"reliability={int(bool(getattr(args, 'reliability_enabled', False)))} "
        f"projection={int(bool(args.projection_enabled))} "
        f"norm_cap={int(bool(args.norm_cap_enabled))} "
        f"gate={int(bool(args.gate_enabled))} "
        f"lr={lr:.3e} "
        f"loss_det={loss_det:.4f} "
        f"loss_odam={loss_odam:.4f} "
        f"raw_loss_sum={raw_loss_sum:.4f} "
        f"avg_det={avg_det:.4f} "
        f"avg_odam={avg_odam:.4f} "
        f"avg_raw_loss_sum={avg_raw_loss_sum:.4f} "
        f"step_time={step_seconds:.2f}s "
        f"elapsed={format_duration(elapsed_seconds)} "
        f"eta={format_duration(eta_seconds)} "
        f"batch_per_gpu={args.batch_size} "
        f"effective_batch={effective_batch} "
        f"{cuda_memory_summary(device)}"
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    dpga: Optional[DPGAController],
    device: torch.device,
    output_dir: Path,
    args,
    epoch: int,
    rank: int,
) -> Dict[str, float]:
    model.train()

    raw_model = unwrap_model(model)

    totals = {
        "loss_det": 0.0,
        "loss_odam": 0.0,
        "raw_loss_sum": 0.0,
        "loss_proxy": 0.0,
        "loss_total_objective": 0.0,
        "odam_num_candidates": 0.0,
        "odam_num_kept": 0.0,
        "odam_keep_ratio": 0.0,
        "odam_reliability_mean": 0.0,
        "odam_reliability_std": 0.0,
        "odam_reliability_p10": 0.0,
        "odam_reliability_p50": 0.0,
        "odam_reliability_p90": 0.0,
        "odam_reliability_score_tau": 0.0,
        "odam_reliability_budget_fraction": 0.0,
        "odam_reliability_budget_keep_ratio": 0.0,
        "odam_roi_iou_mean": 0.0,
        "odam_roi_score_mean": 0.0,
        "odam_loss_raw": 0.0,
        "odam_loss_weighted": 0.0,
        "odam_effective_rois": 0.0,
        "odam_low_reliability_fraction": 0.0,
        "odam_high_reliability_fraction": 0.0,
    }

    num_steps = 0
    total_steps = len(loader)
    epoch_start = time.time()

    iterator = loader
    if tqdm is not None and rank == 0:
        iterator = tqdm(
            loader,
            desc=f"train {epoch:03d}",
            leave=False,
        )

    for step, (image, im_info, gt_boxes, _) in enumerate(iterator):
        step_start = time.time()

        image = image.to(
            device,
            non_blocking=True,
        )
        im_info = im_info.to(
            device,
            non_blocking=True,
        )
        gt_boxes = gt_boxes.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)
        diagnostic_rows = []
        dpga_stats = None
        fractional_epoch = float(epoch) + (
            float(step) / max(float(total_steps), 1.0)
        )
        dpga_epoch = fractional_epoch
        odam_weight = _odam_weight_for_epoch(args, fractional_epoch)
        args.odam_reliability_budget_fraction = (
            _odam_reliability_budget_fraction_for_epoch(
                args,
                fractional_epoch,
            )
        )
        if hasattr(raw_model, "config"):
            raw_model.config.odam_reliability_budget_fraction = (
                args.odam_reliability_budget_fraction
            )
        if args.method == "dpga" and dpga is not None:
            raw_model.set_odam_enabled(dpga.alpha(dpga_epoch) > 0.0)
        else:
            raw_model.set_odam_enabled(
                args.method == "odam" and odam_weight > 0.0
            )

        if args.method == "dpga" and isinstance(model, DDP):
            sync_context = model.no_sync()
        else:
            sync_context = nullcontext()

        with sync_context:
            loss_dict = model(
                image,
                im_info,
                gt_boxes,
            )

            loss_det, loss_odam = split_detection_and_odam_loss(
                loss_dict
            )
            if getattr(args, "finite_checks", True):
                assert_finite_training_state(
                    stage="after_forward_loss",
                    epoch=epoch,
                    step=step,
                    rank=rank,
                    method=args.method,
                    tensors={
                        **loss_dict,
                        "loss_det": loss_det,
                        "loss_odam": loss_odam,
                    },
                )

            if args.method == "baseline":
                backward_objective = loss_det
                raw_loss_sum = loss_det
                scalar_objective_for_logging = backward_objective
                backward_objective.backward()

            elif args.method == "odam":
                backward_objective = (
                    loss_det
                    + odam_weight * loss_odam
                )
                raw_loss_sum = backward_objective
                scalar_objective_for_logging = backward_objective
                if should_log_gradient_diagnostics(args, step):
                    diagnostic_rows = compute_odam_gradient_diagnostics(
                        model=model,
                        loss_det=loss_det,
                        loss_odam=loss_odam,
                        odam_weight=odam_weight,
                        epoch=epoch,
                        step=step,
                        rank=rank,
                    )
                backward_objective.backward()

            elif args.method == "dpga":
                if dpga is None:
                    raise RuntimeError("DPGA controller not initialized")

                dpga_stats = dpga.backward(
                    loss_det=loss_det,
                    loss_odam=loss_odam,
                    epoch=dpga_epoch,
                )

                # For logging only; NOT used as the DPGA backward objective.
                raw_loss_sum = loss_det + loss_odam
                scalar_objective_for_logging = None
                if should_log_gradient_diagnostics(args, step):
                    diagnostic_rows = dpga_stats_to_diagnostic_rows(
                        stats=dpga_stats,
                        loss_det=loss_det,
                        loss_odam=loss_odam,
                        epoch=epoch,
                        step=step,
                        rank=rank,
                        method=args.method,
                    )

            else:
                raise ValueError(args.method)

        if getattr(args, "finite_checks", True):
            assert_finite_training_state(
                stage="after_backward",
                epoch=epoch,
                step=step,
                rank=rank,
                method=args.method,
                tensors={"raw_loss_sum": raw_loss_sum},
                model=model,
                check_grads=True,
            )

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(),
                max_norm=args.grad_clip,
            )
            if getattr(args, "finite_checks", True):
                assert_finite_training_state(
                    stage="after_grad_clip",
                    epoch=epoch,
                    step=step,
                    rank=rank,
                    method=args.method,
                    model=model,
                    check_grads=True,
                )

        optimizer.step()
        if getattr(args, "finite_checks", True):
            assert_finite_training_state(
                stage="after_optimizer_step",
                epoch=epoch,
                step=step,
                rank=rank,
                method=args.method,
                model=model,
            )
        append_gradient_diagnostics(
            output_dir,
            diagnostic_rows,
            rank,
        )

        loss_det_reduced = reduce_mean(
            loss_det.detach()
        )
        loss_odam_reduced = reduce_mean(
            loss_odam.detach()
        )
        raw_loss_sum_reduced = reduce_mean(
            raw_loss_sum.detach()
        )
        if scalar_objective_for_logging is not None:
            loss_total_objective_reduced = reduce_mean(
                scalar_objective_for_logging.detach()
            )
        else:
            loss_total_objective_reduced = None
        odam_filter_stats = getattr(
            getattr(raw_model, "RCNN", None),
            "last_odam_filter_stats",
            {},
        )
        odam_num_candidates = reduce_mean(
            loss_det.new_tensor(
                float(odam_filter_stats.get("candidates", 0.0))
            )
        )
        odam_num_kept = reduce_mean(
            loss_det.new_tensor(
                float(odam_filter_stats.get("kept", 0.0))
            )
        )
        odam_keep_ratio = (
            odam_num_kept
            / odam_num_candidates.clamp(min=1.0)
        )
        odam_reliability_mean = reduce_mean(
            loss_det.new_tensor(
                float(odam_filter_stats.get("reliability_mean", 0.0))
            )
        )
        odam_reliability_std = reduce_mean(
            loss_det.new_tensor(
                float(odam_filter_stats.get("reliability_std", 0.0))
            )
        )
        odam_reliability_p10 = reduce_mean(
            loss_det.new_tensor(
                float(odam_filter_stats.get("reliability_p10", 0.0))
            )
        )
        odam_reliability_p50 = reduce_mean(
            loss_det.new_tensor(
                float(odam_filter_stats.get("reliability_p50", 0.0))
            )
        )
        odam_reliability_p90 = reduce_mean(
            loss_det.new_tensor(
                float(odam_filter_stats.get("reliability_p90", 0.0))
            )
        )
        odam_reliability_score_tau = reduce_mean(
            loss_det.new_tensor(
                float(odam_filter_stats.get("reliability_score_tau", 0.0))
            )
        )
        odam_reliability_budget_fraction = reduce_mean(
            loss_det.new_tensor(
                float(odam_filter_stats.get("reliability_budget_fraction", 0.0))
            )
        )
        odam_reliability_budget_keep_ratio = reduce_mean(
            loss_det.new_tensor(
                float(
                    odam_filter_stats.get(
                        "reliability_budget_keep_ratio",
                        0.0,
                    )
                )
            )
        )
        odam_roi_iou_mean = reduce_mean(
            loss_det.new_tensor(
                float(odam_filter_stats.get("roi_iou_mean", 0.0))
            )
        )
        odam_roi_score_mean = reduce_mean(
            loss_det.new_tensor(
                float(odam_filter_stats.get("roi_score_mean", 0.0))
            )
        )
        odam_loss_raw = reduce_mean(
            loss_det.new_tensor(
                float(odam_filter_stats.get("odam_loss_raw", 0.0))
            )
        )
        odam_loss_weighted = reduce_mean(
            loss_det.new_tensor(
                float(odam_filter_stats.get("odam_loss_weighted", 0.0))
            )
        )
        odam_effective_rois = reduce_mean(
            loss_det.new_tensor(
                float(odam_filter_stats.get("effective_odam_rois", 0.0))
            )
        )
        odam_low_reliability_fraction = reduce_mean(
            loss_det.new_tensor(
                float(odam_filter_stats.get("low_reliability_fraction", 0.0))
            )
        )
        odam_high_reliability_fraction = reduce_mean(
            loss_det.new_tensor(
                float(odam_filter_stats.get("high_reliability_fraction", 0.0))
            )
        )

        totals["loss_det"] += float(
            loss_det_reduced.cpu()
        )
        totals["loss_odam"] += float(
            loss_odam_reduced.cpu()
        )
        totals["raw_loss_sum"] += float(
            raw_loss_sum_reduced.cpu()
        )
        totals["loss_proxy"] += float(
            raw_loss_sum_reduced.cpu()
        )
        if loss_total_objective_reduced is not None:
            totals["loss_total_objective"] += float(
                loss_total_objective_reduced.cpu()
            )
        totals["odam_num_candidates"] += float(
            odam_num_candidates.cpu()
        )
        totals["odam_num_kept"] += float(
            odam_num_kept.cpu()
        )
        totals["odam_keep_ratio"] += float(
            odam_keep_ratio.cpu()
        )
        totals["odam_reliability_mean"] += float(
            odam_reliability_mean.cpu()
        )
        totals["odam_reliability_std"] += float(
            odam_reliability_std.cpu()
        )
        totals["odam_reliability_p10"] += float(
            odam_reliability_p10.cpu()
        )
        totals["odam_reliability_p50"] += float(
            odam_reliability_p50.cpu()
        )
        totals["odam_reliability_p90"] += float(
            odam_reliability_p90.cpu()
        )
        totals["odam_reliability_score_tau"] += float(
            odam_reliability_score_tau.cpu()
        )
        totals["odam_reliability_budget_fraction"] += float(
            odam_reliability_budget_fraction.cpu()
        )
        totals["odam_reliability_budget_keep_ratio"] += float(
            odam_reliability_budget_keep_ratio.cpu()
        )
        totals["odam_roi_iou_mean"] += float(
            odam_roi_iou_mean.cpu()
        )
        totals["odam_roi_score_mean"] += float(
            odam_roi_score_mean.cpu()
        )
        totals["odam_loss_raw"] += float(
            odam_loss_raw.cpu()
        )
        totals["odam_loss_weighted"] += float(
            odam_loss_weighted.cpu()
        )
        totals["odam_effective_rois"] += float(
            odam_effective_rois.cpu()
        )
        totals["odam_low_reliability_fraction"] += float(
            odam_low_reliability_fraction.cpu()
        )
        totals["odam_high_reliability_fraction"] += float(
            odam_high_reliability_fraction.cpu()
        )
        num_steps += 1

        loss_det_value = float(loss_det_reduced.cpu())
        loss_odam_value = float(loss_odam_reduced.cpu())
        raw_loss_sum_value = float(raw_loss_sum_reduced.cpu())
        avg_det = totals["loss_det"] / max(num_steps, 1)
        avg_odam = totals["loss_odam"] / max(num_steps, 1)
        avg_raw_loss_sum = totals["raw_loss_sum"] / max(num_steps, 1)

        if rank == 0 and tqdm is not None:
            iterator.set_postfix(
                det=f"{loss_det_value:.3f}",
                odam=f"{loss_odam_value:.3f}",
                raw=f"{raw_loss_sum_value:.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        if rank == 0 and should_log_step(args, step, total_steps):
            write_train_log(
                iterator,
                format_train_step_log(
                    args=args,
                    epoch=epoch,
                    step=step,
                    total_steps=total_steps,
                    lr=optimizer.param_groups[0]["lr"],
                    loss_det=loss_det_value,
                    loss_odam=loss_odam_value,
                    raw_loss_sum=raw_loss_sum_value,
                    avg_det=avg_det,
                    avg_odam=avg_odam,
                    avg_raw_loss_sum=avg_raw_loss_sum,
                    step_seconds=time.time() - step_start,
                    elapsed_seconds=time.time() - epoch_start,
                    device=device,
                ),
            )

        if (
            rank == 0
            and args.method == "dpga"
            and args.dpga_log_interval > 0
            and step % args.dpga_log_interval == 0
        ):
            print(format_dpga_stats(dpga_stats))

    denom = max(num_steps, 1)

    metrics = {
        key: value / denom
        for key, value in totals.items()
    }
    if args.method == "dpga":
        metrics["loss_total_objective"] = float("nan")
    return metrics


CSV_FIELDS = [
    "epoch",
    "method",
    "lr",
    "seconds",
    "loss_det",
    "loss_odam",
    "raw_loss_sum",
    "loss_proxy",
    "loss_total_objective",
    "odam_num_candidates",
    "odam_num_kept",
    "odam_keep_ratio",
    "odam_reliability_mean",
    "odam_reliability_std",
    "odam_reliability_p10",
    "odam_reliability_p50",
    "odam_reliability_p90",
    "odam_reliability_score_tau",
    "odam_reliability_budget_fraction",
    "odam_reliability_budget_keep_ratio",
    "odam_roi_iou_mean",
    "odam_roi_score_mean",
    "odam_loss_raw",
    "odam_loss_weighted",
    "odam_effective_rois",
    "odam_low_reliability_fraction",
    "odam_high_reliability_fraction",
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "MR-2_generic",
    "ODAM_quality",
    "ODAM_quality_samples",
    "ODAM_quality_mean_iou",
]


GRADIENT_DIAGNOSTIC_FIELDS = [
    "epoch",
    "step",
    "rank",
    "method",
    "module",
    "loss_det",
    "loss_odam",
    "gradient_scope",
    "world_size",
    "cosine_raw",
    "cosine_projected",
    "det_norm",
    "det_gradient_norm",
    "odam_norm_raw",
    "raw_odam_norm",
    "projected_odam_norm",
    "capped_odam_norm",
    "final_odam_norm",
    "odam_norm_safe",
    "aux_to_det_raw",
    "aux_to_det_projected",
    "aux_to_det_capped",
    "aux_to_det_final",
    "aux_to_det_effective",
    "directional_margin",
    "aux_directional_margin",
    "final_alignment_margin",
    "final_cosine_to_det",
    "final_angle_deg",
    "conflict_raw",
    "dominance_raw",
    "dominance_effective",
    "unsafe_descent",
    "aux_unsafe",
    "projected",
    "cap_active",
    "norm_scale",
    "gate",
    "alpha",
    "effective_weight",
]


def append_csv_fields(
    csv_path: Path,
    row: Dict,
    fields: Sequence[str],
):
    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = csv_path.exists()
    expected_fields = list(fields)
    if exists:
        with csv_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
        if existing_header != expected_fields:
            raise RuntimeError(
                "CSV schema mismatch for "
                f"{csv_path}. The output directory contains artifacts from "
                "a different code revision or schema version. Use a fresh "
                "output directory for this run."
            )

    normalized = {
        field: row.get(field, "")
        for field in expected_fields
    }

    with csv_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=expected_fields,
        )

        if not exists:
            writer.writeheader()

        writer.writerow(normalized)


def append_csv(
    csv_path: Path,
    row: Dict,
):
    append_csv_fields(
        csv_path,
        row,
        CSV_FIELDS,
    )


def should_log_gradient_diagnostics(args, step: int) -> bool:
    interval = int(args.gradient_diagnostics_interval)
    return (
        args.method in ("odam", "dpga")
        and interval > 0
        and step % interval == 0
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return 0.0
    if abs(denominator) <= 1e-12:
        return 0.0
    return numerator / denominator


def _final_angle_degrees(cosine: float) -> float:
    cosine = min(max(float(cosine), -1.0), 1.0)
    return math.degrees(math.acos(cosine))


def _tensor_list_dot(
    xs: Sequence[torch.Tensor],
    ys: Sequence[torch.Tensor],
) -> torch.Tensor:
    if not xs:
        return torch.tensor(0.0)

    out = xs[0].new_zeros(())
    for x, y in zip(xs, ys):
        out = out + torch.sum(x * y)
    return out


def _tensor_list_norm(xs: Sequence[torch.Tensor]) -> torch.Tensor:
    if not xs:
        return torch.tensor(0.0)

    out = xs[0].new_zeros(())
    for x in xs:
        out = out + torch.sum(x * x)
    return torch.sqrt(out)


def _replace_none_grads(
    grads: Sequence[Optional[torch.Tensor]],
    params: Sequence[nn.Parameter],
) -> List[torch.Tensor]:
    return [
        torch.zeros_like(param, memory_format=torch.preserve_format)
        if grad is None else grad
        for param, grad in zip(params, grads)
    ]


def compute_odam_gradient_diagnostics(
    model: nn.Module,
    loss_det: torch.Tensor,
    loss_odam: torch.Tensor,
    odam_weight: float,
    epoch: int,
    step: int,
    rank: int,
) -> List[Dict]:
    raw_model = unwrap_model(model)
    groups = build_dpga_groups(raw_model)

    params = [
        param
        for group_params in groups.values()
        for param in group_params
    ]

    g_det_raw = torch.autograd.grad(
        loss_det,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    g_odam_raw = torch.autograd.grad(
        loss_odam,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    g_det_flat = _replace_none_grads(
        g_det_raw,
        params,
    )
    g_odam_flat = _replace_none_grads(
        g_odam_raw,
        params,
    )

    world_size = current_world_size()
    gradient_scope = "local"
    if world_size > 1:
        g_det_flat = allreduce_gradient_list_mean(g_det_flat)
        g_odam_flat = allreduce_gradient_list_mean(g_odam_flat)
        gradient_scope = "global_ddp_mean"

    rows = []
    offset = 0

    for module_name, module_params in groups.items():
        size = len(module_params)
        g_det = g_det_flat[offset: offset + size]
        g_odam = g_odam_flat[offset: offset + size]
        offset += size

        det_norm_t = _tensor_list_norm(g_det)
        odam_norm_t = _tensor_list_norm(g_odam)
        dot_t = _tensor_list_dot(g_det, g_odam)
        final_grads = [
            gd + float(odam_weight) * go
            for gd, go in zip(g_det, g_odam)
        ]
        auxiliary_grads = [
            float(odam_weight) * go
            for go in g_odam
        ]
        final_norm_t = _tensor_list_norm(final_grads)
        final_dot_t = _tensor_list_dot(g_det, final_grads)
        auxiliary_dot_t = _tensor_list_dot(g_det, auxiliary_grads)

        det_norm = float(det_norm_t.detach().cpu())
        odam_norm = float(odam_norm_t.detach().cpu())
        dot = float(dot_t.detach().cpu())
        final_norm = float(final_norm_t.detach().cpu())
        final_dot = float(final_dot_t.detach().cpu())
        auxiliary_dot = float(auxiliary_dot_t.detach().cpu())
        final_odam_norm = abs(float(odam_weight)) * odam_norm

        cosine_raw = _safe_ratio(
            dot,
            det_norm * odam_norm,
        )
        final_cosine = _safe_ratio(
            final_dot,
            det_norm * final_norm,
        )
        directional_margin = _safe_ratio(
            auxiliary_dot,
            det_norm * det_norm,
        )
        final_alignment_margin = _safe_ratio(
            final_dot,
            det_norm * det_norm,
        )

        rows.append(
            {
                "epoch": epoch,
                "step": step,
                "rank": rank,
                "method": "odam",
                "module": module_name,
                "loss_det": float(loss_det.detach().cpu()),
                "loss_odam": float(loss_odam.detach().cpu()),
                "gradient_scope": gradient_scope,
                "world_size": world_size,
                "cosine_raw": cosine_raw,
                "cosine_projected": cosine_raw,
                "det_norm": det_norm,
                "det_gradient_norm": det_norm,
                "odam_norm_raw": odam_norm,
                "raw_odam_norm": odam_norm,
                "projected_odam_norm": odam_norm,
                "capped_odam_norm": odam_norm,
                "final_odam_norm": final_odam_norm,
                "odam_norm_safe": odam_norm,
                "aux_to_det_raw": _safe_ratio(odam_norm, det_norm),
                "aux_to_det_projected": _safe_ratio(odam_norm, det_norm),
                "aux_to_det_capped": _safe_ratio(odam_norm, det_norm),
                "aux_to_det_final": _safe_ratio(
                    final_odam_norm,
                    det_norm,
                ),
                "aux_to_det_effective": _safe_ratio(
                    final_odam_norm,
                    det_norm,
                ),
                "directional_margin": directional_margin,
                "aux_directional_margin": directional_margin,
                "final_alignment_margin": final_alignment_margin,
                "final_cosine_to_det": final_cosine,
                "final_angle_deg": _final_angle_degrees(final_cosine),
                "conflict_raw": int(cosine_raw < 0.0),
                "dominance_raw": int(odam_norm > det_norm),
                "dominance_effective": int(final_odam_norm > det_norm),
                "unsafe_descent": int(directional_margin < -AUX_SAFETY_EPS),
                "aux_unsafe": int(directional_margin < -AUX_SAFETY_EPS),
                "projected": 0,
                "cap_active": 0,
                "norm_scale": 1.0,
                "gate": 1.0,
                "alpha": float(odam_weight),
                "effective_weight": float(odam_weight),
            }
        )

    return rows


def dpga_stats_to_diagnostic_rows(
    stats,
    loss_det: torch.Tensor,
    loss_odam: torch.Tensor,
    epoch: int,
    step: int,
    rank: int,
    method: str = "dpga",
) -> List[Dict]:
    rows = []

    for module_name, module_stats in stats.modules.items():
        det_norm = float(module_stats.det_norm)
        odam_norm_raw = float(module_stats.odam_norm_before)
        odam_norm_projected = float(
            module_stats.odam_norm_after_projection
        )
        odam_norm_safe = float(module_stats.odam_norm_after_cap)
        final_odam_norm = (
            abs(float(module_stats.effective_weight))
            * odam_norm_safe
        )
        effective_aux_norm = (
            abs(float(module_stats.effective_weight))
            * odam_norm_safe
        )
        raw_aux_ratio = _safe_ratio(odam_norm_raw, det_norm)
        projected_aux_ratio = _safe_ratio(
            odam_norm_projected,
            det_norm,
        )
        capped_aux_ratio = _safe_ratio(
            odam_norm_safe,
            det_norm,
        )
        effective_aux_ratio = _safe_ratio(effective_aux_norm, det_norm)

        final_dot = (
            det_norm * det_norm
            + float(module_stats.effective_weight)
            * float(module_stats.cosine_after)
            * det_norm
            * odam_norm_safe
        )
        auxiliary_dot = (
            float(module_stats.effective_weight)
            * float(module_stats.cosine_after)
            * det_norm
            * odam_norm_safe
        )
        directional_margin = _safe_ratio(
            auxiliary_dot,
            det_norm * det_norm,
        )
        final_cosine = _safe_ratio(
            final_dot,
            det_norm * float(module_stats.final_norm),
        )
        final_alignment_margin = _safe_ratio(
            final_dot,
            det_norm * det_norm,
        )

        rows.append(
            {
                "epoch": epoch,
                "step": step,
                "rank": rank,
                "method": method,
                "module": module_name,
                "loss_det": float(loss_det.detach().cpu()),
                "loss_odam": float(loss_odam.detach().cpu()),
                "gradient_scope": getattr(stats, "gradient_scope", "local"),
                "world_size": getattr(stats, "world_size", current_world_size()),
                "cosine_raw": float(module_stats.cosine_before),
                "cosine_projected": float(
                    getattr(
                        module_stats,
                        "cosine_projected",
                        module_stats.cosine_after,
                    )
                ),
                "det_norm": det_norm,
                "det_gradient_norm": det_norm,
                "odam_norm_raw": odam_norm_raw,
                "raw_odam_norm": odam_norm_raw,
                "projected_odam_norm": odam_norm_projected,
                "capped_odam_norm": odam_norm_safe,
                "final_odam_norm": final_odam_norm,
                "odam_norm_safe": odam_norm_safe,
                "aux_to_det_raw": raw_aux_ratio,
                "aux_to_det_projected": projected_aux_ratio,
                "aux_to_det_capped": capped_aux_ratio,
                "aux_to_det_final": effective_aux_ratio,
                "aux_to_det_effective": effective_aux_ratio,
                "directional_margin": directional_margin,
                "aux_directional_margin": directional_margin,
                "final_alignment_margin": final_alignment_margin,
                "final_cosine_to_det": final_cosine,
                "final_angle_deg": _final_angle_degrees(final_cosine),
                "conflict_raw": int(module_stats.cosine_before < 0.0),
                "dominance_raw": int(odam_norm_raw > det_norm),
                "dominance_effective": int(effective_aux_norm > det_norm),
                "unsafe_descent": int(directional_margin < -AUX_SAFETY_EPS),
                "aux_unsafe": int(directional_margin < -AUX_SAFETY_EPS),
                "projected": int(module_stats.projected),
                "cap_active": int(module_stats.cap_active),
                "norm_scale": float(module_stats.norm_scale),
                "gate": float(module_stats.gate),
                "alpha": float(module_stats.alpha),
                "effective_weight": float(module_stats.effective_weight),
            }
        )

    return rows


def append_gradient_diagnostics(
    output_dir: Path,
    rows: Sequence[Dict],
    rank: int,
):
    if not rows:
        return

    csv_path = output_dir / f"gradient_diagnostics_rank{rank}.csv"
    for row in rows:
        append_csv_fields(
            csv_path,
            row,
            GRADIENT_DIAGNOSTIC_FIELDS,
        )


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN / ODAM / DPGA-ODAM"
    )

    parser.add_argument(
        "--method",
        choices=("baseline", "odam", "dpga"),
        required=True,
    )

    parser.add_argument("--train-images", required=True)
    parser.add_argument("--train-ann", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--val-ann", required=True)

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--experiment-stage",
        choices=tuple(EXPERIMENT_STAGE_PRESETS.keys()),
        default=None,
        help=(
            "Progressive ODAM/DPGA ablation stage. E0-E2 require "
            "--method odam; E3-E7 require --method dpga."
        ),
    )

    # Image size
    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)

    # Loader
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)

    # Train
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--lr-steps",
        nargs="+",
        type=int,
        default=[8, 11],
    )
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument(
        "--no-finite-checks",
        dest="finite_checks",
        action="store_false",
        help=(
            "Disable fail-fast NaN/Inf checks for losses, gradients, and "
            "model parameters. Default is fail-closed."
        ),
    )
    parser.set_defaults(finite_checks=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Print rank-0 training progress every N steps. Set 0 to disable.",
    )

    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Same initialization checkpoint should be used for all 3 methods.",
    )
    parser.add_argument(
        "--backbone-pretrained",
        action="store_true",
        help="Initialize ResNet-50 from torchvision pretrained weights.",
    )
    parser.add_argument(
        "--backbone-freeze-at",
        type=int,
        default=0,
        help=(
            "Freeze backbone stages after pretrained/checkpoint weights are "
            "available. 0 means no freeze; 1 freezes stem; 2 freezes stem+layer1."
        ),
    )
    parser.add_argument(
        "--rpn-ignore-overlap",
        type=float,
        default=0.5,
        help=(
            "Anchor IoA threshold for marking anchors that overlap ignore/crowd "
            "regions as ignore instead of background."
        ),
    )

    # Original ODAM scalarization
    parser.add_argument(
        "--odam-weight",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--odam-filtering",
        action="store_true",
        help=(
            "Enable ODAM-only proposal quality filtering for legacy/manual "
            "runs. Stage presets set this automatically."
        ),
    )
    parser.add_argument("--odam-min-iou", type=float, default=0.7)
    parser.add_argument("--odam-min-score", type=float, default=0.9)
    parser.add_argument(
        "--odam-reliability",
        action="store_true",
        help=(
            "Enable soft reliability-aware ODAM weighting for legacy/manual "
            "runs. Stages E6-E7 set this automatically."
        ),
    )
    parser.add_argument(
        "--odam-reliability-iou-tau",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--odam-reliability-iou-temp",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--odam-reliability-score-tau",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--odam-reliability-score-temp",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--odam-reliability-adaptive-score-tau",
        action="store_true",
        help=(
            "Use the current ODAM candidate score percentile as the soft "
            "score reliability tau. Stage E7 sets this automatically."
        ),
    )
    parser.add_argument(
        "--odam-reliability-score-percentile",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--odam-reliability-budget",
        dest="odam_reliability_budget_enabled",
        action="store_true",
        help=(
            "Keep only a per-image/per-GT top reliability budget for ODAM. "
            "Stage E7 sets this automatically."
        ),
    )
    parser.add_argument(
        "--odam-reliability-budget-start",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--odam-reliability-budget-end",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--odam-reliability-budget-min",
        type=int,
        default=1,
    )

    # DPGA schedule
    parser.add_argument("--dpga-warmup", type=int, default=4)
    parser.add_argument("--dpga-rampup", type=int, default=4)
    parser.add_argument("--dpga-alpha", type=float, default=1.0)
    parser.add_argument(
        "--dpga-ablation",
        choices=(
            "full",
            "projection-only",
            "norm-cap-only",
            "gate-only",
            "projection-norm-cap",
            "custom",
        ),
        default="full",
        help=(
            "Named DPGA ablation preset for running one job at a time. "
            "Use custom if you want manual --no-dpga-* flags to define the run."
        ),
    )
    parser.add_argument(
        "--no-dpga-projection",
        dest="dpga_projection",
        action="store_false",
        help=(
            "Manual DPGA ablation: disable conflict projection. "
            "Use --dpga-ablation custom for this flag to take effect."
        ),
    )
    parser.add_argument(
        "--no-dpga-norm-cap",
        dest="dpga_norm_cap",
        action="store_false",
        help=(
            "Manual DPGA ablation: disable module-wise auxiliary norm cap. "
            "Use --dpga-ablation custom for this flag to take effect."
        ),
    )
    parser.add_argument(
        "--no-dpga-gate",
        dest="dpga_gate",
        action="store_false",
        help=(
            "Manual DPGA ablation: disable adaptive cosine gate. "
            "Use --dpga-ablation custom for this flag to take effect."
        ),
    )
    parser.set_defaults(
        dpga_projection=True,
        dpga_norm_cap=True,
        dpga_gate=True,
    )
    parser.add_argument(
        "--dpga-temperature",
        type=float,
        default=0.20,
    )

    # Module-wise norm caps
    parser.add_argument("--rho-backbone", type=float, default=0.10)
    parser.add_argument("--rho-fpn", type=float, default=0.15)
    parser.add_argument("--rho-rpn", type=float, default=0.00)
    parser.add_argument("--rho-roi-shared", type=float, default=0.25)
    parser.add_argument("--rho-roi-cls", type=float, default=0.10)
    parser.add_argument("--rho-roi-reg", type=float, default=0.05)

    parser.add_argument(
        "--dpga-log-interval",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--gradient-diagnostics-interval",
        "--gradient-log-interval",
        dest="gradient_diagnostics_interval",
        type=int,
        default=100,
        help=(
            "Write gradient_diagnostics_rank*.csv every N train steps for "
            "ODAM/DPGA. Set 0 to disable."
        ),
    )

    # Evaluation
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--eval-score-threshold",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--eval-nms",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--best-metric",
        choices=(
            "AP",
            "AP50",
            "AP75",
            "MR-2_generic",
            "ODAM_quality",
        ),
        default="AP",
    )
    parser.add_argument(
        "--eval-odam-quality",
        dest="eval_odam_quality",
        action="store_true",
        default=None,
        help=(
            "Compute DAM energy-in-GT localization metric during validation. "
            "Default: enabled for ODAM/DPGA and disabled for baseline."
        ),
    )
    parser.add_argument(
        "--no-eval-odam-quality",
        dest="eval_odam_quality",
        action="store_false",
        help="Disable DAM localization metric during validation.",
    )
    parser.add_argument(
        "--odam-quality-iou",
        type=float,
        default=0.5,
        help="IoU threshold for matching predictions to GT for ODAM_quality.",
    )

    args = parser.parse_args()
    if args.odam_weight < 0:
        raise ValueError("--odam-weight must be >= 0")
    if args.dpga_warmup < 0:
        raise ValueError("--dpga-warmup must be >= 0")
    if args.dpga_rampup < 0:
        raise ValueError("--dpga-rampup must be >= 0")
    if args.dpga_alpha < 0:
        raise ValueError("--dpga-alpha must be >= 0")
    if args.dpga_temperature <= 0:
        raise ValueError("--dpga-temperature must be > 0")
    if not 0.0 <= args.odam_min_iou <= 1.0:
        raise ValueError("--odam-min-iou must be in [0, 1]")
    if not 0.0 <= args.odam_min_score <= 1.0:
        raise ValueError("--odam-min-score must be in [0, 1]")
    if not 0.0 <= args.odam_reliability_iou_tau <= 1.0:
        raise ValueError("--odam-reliability-iou-tau must be in [0, 1]")
    if not 0.0 <= args.odam_reliability_score_tau <= 1.0:
        raise ValueError("--odam-reliability-score-tau must be in [0, 1]")
    if args.odam_reliability_iou_temp <= 0:
        raise ValueError("--odam-reliability-iou-temp must be > 0")
    if args.odam_reliability_score_temp <= 0:
        raise ValueError("--odam-reliability-score-temp must be > 0")
    if not 0.0 <= args.odam_reliability_score_percentile <= 1.0:
        raise ValueError(
            "--odam-reliability-score-percentile must be in [0, 1]"
        )
    if not 0.0 <= args.odam_reliability_budget_start <= 1.0:
        raise ValueError(
            "--odam-reliability-budget-start must be in [0, 1]"
        )
    if not 0.0 <= args.odam_reliability_budget_end <= 1.0:
        raise ValueError("--odam-reliability-budget-end must be in [0, 1]")
    if args.odam_reliability_budget_min < 1:
        raise ValueError("--odam-reliability-budget-min must be >= 1")
    if args.backbone_freeze_at not in (0, 1, 2):
        raise ValueError("--backbone-freeze-at must be one of {0, 1, 2}")
    if not 0.0 <= args.rpn_ignore_overlap <= 1.0:
        raise ValueError("--rpn-ignore-overlap must be in [0, 1]")
    if args.method == "baseline" and args.odam_filtering:
        raise ValueError("--odam-filtering is only valid for ODAM/DPGA runs.")
    if args.method == "baseline" and args.odam_reliability:
        raise ValueError("--odam-reliability is only valid for ODAM/DPGA runs.")
    if args.method == "baseline" and args.odam_reliability_budget_enabled:
        raise ValueError(
            "--odam-reliability-budget is only valid for ODAM/DPGA runs."
        )
    return apply_experiment_stage_preset(args, argv=argv)


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    distributed, rank, world_size, local_rank = init_distributed()

    set_seed(
        args.seed,
        rank=rank,
    )

    if torch.cuda.is_available():
        if distributed:
            device = torch.device(
                "cuda",
                local_rank,
            )
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    output_dir = Path(args.output)

    if is_main_process(rank):
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    barrier()

    train_dataset = CocoDetectionTrainDataset(
        args.train_images,
        args.train_ann,
        min_size=args.min_size,
        max_size=args.max_size,
    )

    val_dataset = CocoDetectionTrainDataset(
        args.val_images,
        args.val_ann,
        min_size=args.min_size,
        max_size=args.max_size,
    )

    if train_dataset.category_ids != val_dataset.category_ids:
        raise ValueError(
            "Train and val category IDs do not match."
        )

    if args.val_batch_size != 1:
        raise ValueError(
            "Current evaluator requires --val-batch-size 1."
        )

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        if distributed
        else None
    )

    val_sampler = (
        DistributedEvalSampler(
            val_dataset,
            rank=rank,
            world_size=world_size,
        )
        if distributed
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=detection_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=detection_collate,
    )

    config = DetectorConfig(
        num_classes=len(train_dataset.category_ids) + 1,
        train_batch_per_gpu=args.batch_size,
        backbone_freeze_at=args.backbone_freeze_at,
        backbone_pretrained=args.backbone_pretrained,
        rpn_ignore_overlap=args.rpn_ignore_overlap,
        odam_filtering=args.odam_filtering,
        odam_min_iou=args.odam_min_iou,
        odam_min_score=args.odam_min_score,
        odam_reliability=args.odam_reliability,
        odam_reliability_iou_tau=args.odam_reliability_iou_tau,
        odam_reliability_iou_temp=args.odam_reliability_iou_temp,
        odam_reliability_score_tau=args.odam_reliability_score_tau,
        odam_reliability_score_temp=args.odam_reliability_score_temp,
        odam_reliability_adaptive_score_tau=(
            args.odam_reliability_adaptive_score_tau
        ),
        odam_reliability_score_percentile=(
            args.odam_reliability_score_percentile
        ),
        odam_reliability_budget_enabled=(
            args.odam_reliability_budget_enabled
        ),
        odam_reliability_budget_start=args.odam_reliability_budget_start,
        odam_reliability_budget_end=args.odam_reliability_budget_end,
        odam_reliability_budget_fraction=(
            args.odam_reliability_budget_fraction
        ),
        odam_reliability_budget_min=args.odam_reliability_budget_min,
    )
    validate_config(config)

    model = Network(config)
    model.to(device)

    checkpoint_result = load_initial_weights(
        model,
        args.init_checkpoint,
    )
    apply_backbone_freeze_policy(
        model,
        config,
        checkpoint_result=checkpoint_result,
    )

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    raw_model = unwrap_model(model)

    # train_one_epoch updates ODAM enablement at each fractional epoch.
    raw_model.set_odam_enabled(
        args.method == "odam"
        and _odam_weight_for_epoch(args, 0.0) > 0.0
    )
    raw_model.set_odam_inference(False)

    optimizer = build_optimizer(
        model,
        args,
    )
    scheduler = build_scheduler(
        optimizer,
        args,
    )

    dpga = None
    if args.method == "dpga":
        dpga = DPGAController(
            raw_model,
            make_dpga_config(args),
        )

    if is_main_process(rank):
        experiment_meta = {
            "method": args.method,
            "experiment_stage": args.experiment_stage,
            "warmup_enabled": bool(args.warmup_enabled),
            "filtering_enabled": bool(args.filtering_enabled),
            "reliability_enabled": bool(args.reliability_enabled),
            "odam_min_iou": args.odam_min_iou,
            "odam_min_score": args.odam_min_score,
            "odam_reliability": bool(args.odam_reliability),
            "odam_reliability_iou_tau": args.odam_reliability_iou_tau,
            "odam_reliability_iou_temp": args.odam_reliability_iou_temp,
            "odam_reliability_score_tau": args.odam_reliability_score_tau,
            "odam_reliability_score_temp": args.odam_reliability_score_temp,
            "odam_reliability_adaptive_score_tau": (
                args.odam_reliability_adaptive_score_tau
            ),
            "odam_reliability_score_percentile": (
                args.odam_reliability_score_percentile
            ),
            "odam_reliability_budget_enabled": (
                args.odam_reliability_budget_enabled
            ),
            "odam_reliability_budget_start": (
                args.odam_reliability_budget_start
            ),
            "odam_reliability_budget_end": (
                args.odam_reliability_budget_end
            ),
            "odam_reliability_budget_min": (
                args.odam_reliability_budget_min
            ),
            "projection_enabled": bool(args.projection_enabled),
            "norm_cap_enabled": bool(args.norm_cap_enabled),
            "gate_enabled": bool(args.gate_enabled),
            "dpga_alpha_max": (
                dpga.config.alpha_max
                if dpga is not None
                else None
            ),
            "dpga_ablation": getattr(args, "dpga_ablation", None),
            "dpga_ablation_label": getattr(
                args,
                "dpga_ablation_label",
                args.method,
            ),
            "world_size": world_size,
            "device": str(device),
            "train_batch_size_per_gpu": args.batch_size,
            "effective_train_batch_size": args.batch_size * world_size,
            "val_batch_size": args.val_batch_size,
            "categories": train_dataset.category_ids,
            "internal_label_mapping": train_dataset.cat_id_to_label,
            "detector_config": asdict(config),
            "args": vars(args),
            "git_commit": current_git_commit(),
            "metrics_schema_version": METRICS_SCHEMA_VERSION,
            "gradient_diagnostic_schema_version": (
                GRADIENT_DIAGNOSTIC_SCHEMA_VERSION
            ),
            "dpga_gradient_pipeline": (
                "raw_detection_and_odam_gradients_are_allreduced_before_dpga"
                if args.method == "dpga"
                else None
            ),
            "odam_pair_identity": (
                "image_aware_batch_id_gt_id"
                if args.method in ("odam", "dpga")
                else None
            ),
            "odam_pair_scope": (
                "same_image_only_self_pairs_included"
                if args.method in ("odam", "dpga")
                else None
            ),
            "dpga_warmup_semantics": (
                "odam_forward_disabled_when_alpha_is_zero"
                if args.method == "dpga"
                else None
            ),
            "dpga_alpha_timebase": (
                "fractional_epoch"
                if args.method == "dpga"
                else None
            ),
            "metric_note": (
                "MR-2_generic is not the full official CityPersons "
                "Reasonable protocol."
            ),
        }

        (output_dir / "experiment.json").write_text(
            json.dumps(
                experiment_meta,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            "[setup] "
            f"method={args.method} "
            f"experiment_stage={args.experiment_stage} "
            f"device={device} "
            f"world_size={world_size} "
            f"train_images={len(train_dataset)} "
            f"val_images={len(val_dataset)} "
            f"batch_per_gpu={args.batch_size} "
            f"effective_batch={args.batch_size * world_size} "
            f"epochs={args.epochs} "
            f"lr={args.lr:.3e} "
            f"lr_steps={list(args.lr_steps)} "
            f"output={output_dir}"
        )
        print(
            "[setup-stage] "
            f"warmup={int(bool(args.warmup_enabled))} "
            f"filtering={int(bool(args.filtering_enabled))} "
            f"reliability={int(bool(args.reliability_enabled))} "
            f"min_iou={args.odam_min_iou} "
            f"min_score={args.odam_min_score} "
            f"reliability_iou_tau={args.odam_reliability_iou_tau} "
            f"reliability_iou_temp={args.odam_reliability_iou_temp} "
            f"reliability_score_tau={args.odam_reliability_score_tau} "
            f"reliability_score_temp={args.odam_reliability_score_temp} "
            f"adaptive_score_tau={int(bool(args.odam_reliability_adaptive_score_tau))} "
            f"score_percentile={args.odam_reliability_score_percentile} "
            f"budget={int(bool(args.odam_reliability_budget_enabled))} "
            f"budget_start={args.odam_reliability_budget_start} "
            f"budget_end={args.odam_reliability_budget_end} "
            f"projection={int(bool(args.projection_enabled))} "
            f"norm_cap={int(bool(args.norm_cap_enabled))} "
            f"gate={int(bool(args.gate_enabled))}"
        )
        if args.method == "dpga":
            print(
                "[setup-dpga] "
                f"ablation={args.dpga_ablation_label} "
                f"warmup={args.dpga_warmup} "
                f"rampup={args.dpga_rampup} "
                f"alpha={dpga.config.alpha_max} "
                f"projection={int(args.dpga_projection)} "
                f"norm_cap={int(args.dpga_norm_cap)} "
                f"gate={int(args.dpga_gate)} "
                f"temperature={args.dpga_temperature} "
                f"rho_backbone={args.rho_backbone} "
                f"rho_fpn={args.rho_fpn} "
                f"rho_rpn={args.rho_rpn} "
                f"rho_roi_shared={args.rho_roi_shared} "
                f"rho_roi_cls={args.rho_roi_cls} "
                f"rho_roi_reg={args.rho_roi_reg}"
            )
    best_value = (
        float("inf")
        if args.best_metric == "MR-2_generic"
        else -float("inf")
    )

    csv_path = output_dir / "metrics.csv"

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            dpga=dpga,
            device=device,
            output_dir=output_dir,
            args=args,
            epoch=epoch,
            rank=rank,
        )

        scheduler.step()

        do_eval = (
            (epoch + 1) % args.eval_every == 0
            or epoch == args.epochs - 1
        )

        val_metrics = {}

        if do_eval:
            if is_main_process(rank):
                print(
                    "[eval] "
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"split=val "
                    f"images={len(val_dataset)} "
                    f"score_thr={args.eval_score_threshold} "
                    f"nms={args.eval_nms} "
                    f"max_det={args.max_detections}"
                )
            val_metrics = validate(
                model=model,
                loader=val_loader,
                dataset=val_dataset,
                device=device,
                args=args,
                rank=rank,
                world_size=world_size,
                output_dir=output_dir,
                epoch=epoch,
            )

        barrier()

        if is_main_process(rank):
            elapsed = time.time() - start

            row = {
                "epoch": epoch,
                "method": args.method,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": elapsed,
                **train_metrics,
                **val_metrics,
            }

            append_csv(
                csv_path,
                row,
            )

            print(
                f"\n[epoch] epoch={epoch + 1}/{args.epochs} "
                f"method={args.method} "
                f"seconds={format_duration(elapsed)} "
                f"det_loss={train_metrics['loss_det']:.4f} "
                f"odam_loss={train_metrics['loss_odam']:.4f} "
                f"raw_loss_sum={train_metrics['raw_loss_sum']:.4f}"
            )

            if val_metrics:
                print(
                    " | ".join(
                        f"{k}={v:.4f}"
                        for k, v in val_metrics.items()
                    )
                )

            save_checkpoint(
                output_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                method=args.method,
                detector_config=config,
                args=args,
                metrics=val_metrics,
                category_ids=train_dataset.category_ids,
                label_to_cat_id=train_dataset.label_to_cat_id,
            )

            if val_metrics and args.best_metric in val_metrics:
                current = val_metrics[args.best_metric]

                if args.best_metric == "MR-2_generic":
                    improved = current < best_value
                else:
                    improved = current > best_value

                if improved:
                    best_value = current

                    save_checkpoint(
                        output_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        method=args.method,
                        detector_config=config,
                        args=args,
                        metrics=val_metrics,
                        category_ids=train_dataset.category_ids,
                        label_to_cat_id=train_dataset.label_to_cat_id,
                    )

                    print(
                        f"[best] {args.best_metric}={current:.6f}"
                    )

    barrier()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
