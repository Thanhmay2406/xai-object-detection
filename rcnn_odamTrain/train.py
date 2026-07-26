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
    backbone_weights: str = "none"


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
        backbone_weights=args.backbone_weights,
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


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args, rank=0):
    model.train()
    totals = {}
    start = time.perf_counter()
    previous_end = start
    total_steps = len(loader)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if rank == 0:
        model_config = (model.module if hasattr(model, "module") else model).config
        print(
            f"epoch={epoch} phase=train start batches={total_steps} "
            f"batch_size={loader.batch_size} amp={args.amp and device.type == 'cuda'} "
            f"odam_loss_weight_effective={getattr(model_config, 'odam_loss_weight_effective', None)}",
            flush=True,
        )
    for step, batch in enumerate(loader, start=1):
        step_start = time.perf_counter()
        data_time = step_start - previous_end
        images, im_info, gt_boxes, _ = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        use_amp = args.amp and device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=use_amp):
            losses = model(images, im_info, gt_boxes)
            total_loss = sum(losses.values())
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}: {float(total_loss.detach())}")
        if use_amp:
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
        totals["loss_total"] = totals.get("loss_total", 0.0) + float(total_loss.detach())

        if should_log_step(step, total_steps, args) and device.type == "cuda":
            torch.cuda.synchronize(device)
        step_end = time.perf_counter()
        batch_time = step_end - step_start
        previous_end = step_end

        if should_log_step(step, total_steps, args) and rank == 0:
            avg = {key: value / step for key, value in totals.items()}
            loss_text = " ".join(
                f"{key}={float(losses.get(key, total_loss).detach()):.4f}/{value:.4f}"
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
    if rank == 0:
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
        total_loss = sum(losses.values())
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
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(row))
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in sorted(row)})


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
        if distributed.enabled:
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
                rank=distributed.rank,
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
