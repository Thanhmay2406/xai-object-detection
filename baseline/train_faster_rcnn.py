#!/usr/bin/env python3
"""Train Faster R-CNN on a Roboflow-style COCO detection dataset.

Expected directory layout:

    data/drill_bit_coco/
      train/
        _annotations.coco.json
        image_001.jpg
        ...
      valid/
        _annotations.coco.json
        ...
      test/
        _annotations.coco.json
        ...

Main features:
- Correctly remaps every COCO category ID, including category_id=0, to
  Faster R-CNN labels 1..N. Label 0 remains internal background.
- Uses official pycocotools COCOeval for bbox AP/AP50/AP75 and AR.
- Supports detailed per-batch train/evaluation logging.
- Supports AMP, checkpoint resume, scheduler/scaler/RNG restoration.
- Detects non-finite losses before corrupting model weights.
- Supports optional small-object anchors.
- Preserves empty/background-only images by default.
- Supports single-GPU and torchrun/DDP multi-GPU training.

For DDP, launch with torchrun and let the script resolve the per-rank CUDA
device from LOCAL_RANK, for example:
    torchrun --standalone --nproc_per_node=2 baseline/train_faster_rcnn.py ...
"""
import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import tv_tensors
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-faster-rcnn")


@dataclass(frozen=True)
class CategoryMapping:
    """Mapping between source COCO category IDs and training labels."""

    coco_to_train: dict[int, int]
    train_to_coco: dict[int, int]
    train_to_name: dict[int, str]

    @property
    def num_classes(self) -> int:
        """Number of Faster R-CNN classes, including background."""
        return len(self.train_to_name)


class CocoDetectionDataset(Dataset):
    """Minimal COCO-format dataset for torchvision detection models."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        mapping: CategoryMapping | None = None,
        transforms: Any | None = None,
        keep_empty: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.annotation_path = self.split_dir / "_annotations.coco.json"

        if not self.annotation_path.is_file():
            raise FileNotFoundError(
                f"COCO annotation file not found: {self.annotation_path}"
            )

        with self.annotation_path.open("r", encoding="utf-8") as handle:
            self.coco: dict[str, Any] = json.load(handle)

        categories = self.coco.get("categories")
        images = self.coco.get("images")
        annotations = self.coco.get("annotations")

        if not isinstance(categories, list) or not categories:
            raise ValueError(
                f"No valid 'categories' list in {self.annotation_path}"
            )
        if not isinstance(images, list):
            raise ValueError(f"No valid 'images' list in {self.annotation_path}")
        if not isinstance(annotations, list):
            raise ValueError(
                f"No valid 'annotations' list in {self.annotation_path}"
            )

        self.mapping = mapping or build_category_mapping(categories)
        self.transforms = transforms
        self.keep_empty = keep_empty

        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        dropped_annotations = 0

        for annotation in annotations:
            if int(annotation.get("iscrowd", 0)) == 1:
                # Crowd boxes are excluded from training in this baseline.
                dropped_annotations += 1
                continue

            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                dropped_annotations += 1
                continue

            x, y, width, height = (float(value) for value in bbox)
            if not all(math.isfinite(v) for v in (x, y, width, height)):
                dropped_annotations += 1
                continue
            if width <= 0.0 or height <= 0.0:
                dropped_annotations += 1
                continue

            category_id = int(annotation["category_id"])
            if category_id not in self.mapping.coco_to_train:
                dropped_annotations += 1
                continue

            image_id = int(annotation["image_id"])
            annotations_by_image[image_id].append(annotation)

        sorted_images = sorted(images, key=lambda item: int(item["id"]))
        if not keep_empty:
            sorted_images = [
                image_info
                for image_info in sorted_images
                if annotations_by_image.get(int(image_info["id"]))
            ]

        self.images: list[dict[str, Any]] = sorted_images
        self.annotations_by_image = annotations_by_image
        self.dropped_annotations = dropped_annotations

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_info = self.images[index]
        image_id = int(image_info["id"])
        image_path = self.split_dir / str(image_info["file_name"])

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as pil_image:
            image = pil_image.convert("RGB")

        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        iscrowd: list[int] = []

        for annotation in self.annotations_by_image.get(image_id, []):
            x, y, width, height = (
                float(value) for value in annotation["bbox"]
            )
            boxes.append([x, y, x + width, y + height])
            labels.append(
                self.mapping.coco_to_train[int(annotation["category_id"])]
            )
            areas.append(float(annotation.get("area", width * height)))
            iscrowd.append(int(annotation.get("iscrowd", 0)))

        image_width, image_height = image.size

        target: dict[str, torch.Tensor] = {
            "boxes": tv_tensors.BoundingBoxes(
                torch.as_tensor(
                    boxes,
                    dtype=torch.float32,
                ).reshape(-1, 4),
                format="XYXY",
                canvas_size=(image_height, image_width),
            ),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    @property
    def image_ids(self) -> list[int]:
        return [int(image_info["id"]) for image_info in self.images]

    @property
    def empty_image_count(self) -> int:
        return sum(
            1
            for image_info in self.images
            if not self.annotations_by_image.get(int(image_info["id"]))
        )


def build_category_mapping(
    categories: list[dict[str, Any]],
) -> CategoryMapping:
    """Map all source category IDs to labels 1..N.

    Important:
    A custom COCO dataset may use category_id=0 for a real object class.
    Faster R-CNN label 0 is reserved internally for background, so source
    category IDs must be remapped rather than deleting category_id=0.
    """

    sorted_categories = sorted(
        categories,
        key=lambda category: int(category["id"]),
    )

    category_ids = [int(category["id"]) for category in sorted_categories]
    if len(category_ids) != len(set(category_ids)):
        raise ValueError(f"Duplicate COCO category IDs: {category_ids}")

    coco_to_train: dict[int, int] = {}
    train_to_coco: dict[int, int] = {}
    train_to_name: dict[int, str] = {0: "__background__"}

    for train_label, category in enumerate(sorted_categories, start=1):
        coco_id = int(category["id"])
        coco_to_train[coco_id] = train_label
        train_to_coco[train_label] = coco_id
        train_to_name[train_label] = str(category["name"])

    return CategoryMapping(
        coco_to_train=coco_to_train,
        train_to_coco=train_to_coco,
        train_to_name=train_to_name,
    )


def collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def make_transforms(train: bool) -> T.Compose:
    transforms: list[Any] = [T.ToImage()]
    if train:
        transforms.append(T.RandomHorizontalFlip(p=0.5))

    transforms.extend(
        [
            T.ToDtype(torch.float32, scale=True),
            T.SanitizeBoundingBoxes(),
        ]
    )
    return T.Compose(transforms)


def create_model(
    num_classes: int,
    weights: str,
    trainable_backbone_layers: int | None,
    min_size: int,
    max_size: int,
    small_object_anchors: bool,
) -> torch.nn.Module:
    normalized_weights = weights.strip().lower()

    model_kwargs: dict[str, Any] = {
        "min_size": min_size,
        "max_size": max_size,
    }

    if trainable_backbone_layers is not None:
        if not 0 <= trainable_backbone_layers <= 5:
            raise ValueError("--trainable-backbone-layers must be in [0, 5]")
        model_kwargs["trainable_backbone_layers"] = (
            trainable_backbone_layers
        )

    if small_object_anchors:
        model_kwargs["rpn_anchor_generator"] = AnchorGenerator(
            sizes=((8,), (16,), (32,), (64,), (128,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )

    if normalized_weights in {"none", "random", "scratch"}:
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            **model_kwargs,
        )
    elif normalized_weights in {"coco", "default", "pretrained"}:
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            **model_kwargs,
        )
    else:
        raise ValueError(
            "--weights must be one of: "
            "coco, default, pretrained, none, random, scratch"
        )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes,
    )
    return model


def move_targets_to_device(
    targets: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    return [
        {
            key: value.to(device, non_blocking=True)
            for key, value in target.items()
        }
        for target in targets
    ]


def format_seconds(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)

    if hours:
        return f"{hours:d}h{minutes:02d}m{remaining_seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{remaining_seconds:02d}s"
    return f"{remaining_seconds:d}s"


def should_log_step(
    step: int,
    total_steps: int,
    log_every: int,
    log_first_n: int,
) -> bool:
    return (
        step <= log_first_n
        or step == total_steps
        or step % max(1, log_every) == 0
    )


def gpu_memory_text(device: torch.device) -> str:
    if device.type != "cuda":
        return "gpu_mem=cpu"

    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3

    return (
        f"gpu_mem={allocated:.2f}G "
        f"reserved={reserved:.2f}G "
        f"max={max_allocated:.2f}G"
    )


def loss_log_text(
    current: dict[str, float],
    running: dict[str, float],
    batches: int,
) -> str:
    keys = (
        "loss",
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
    )

    parts: list[str] = []
    for key in keys:
        if key not in current:
            continue
        average = running[key] / max(1, batches)
        parts.append(f"{key}={current[key]:.4f}/{average:.4f}")

    return " ".join(parts)


def create_grad_scaler(enabled: bool) -> Any:
    """Create a GradScaler across recent and older PyTorch APIs."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"Invalid integer environment {name}={value}") from error


def should_enable_distributed(argument: bool | None) -> bool:
    if argument is not None:
        return argument
    return env_int("WORLD_SIZE", 1) > 1


def setup_distributed(enabled: bool) -> DistributedContext:
    if not enabled:
        return DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
        )

    rank = env_int("RANK", 0)
    local_rank = env_int("LOCAL_RANK", rank)
    world_size = env_int("WORLD_SIZE", 1)

    if world_size < 2:
        raise ValueError(
            "Distributed mode requires WORLD_SIZE >= 2. "
            "Launch with torchrun or disable --distributed."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training currently requires CUDA")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(context: DistributedContext) -> None:
    if not context.enabled:
        return
    try:
        dist.barrier(device_ids=[context.local_rank])
    except TypeError:
        dist.barrier()


def reduce_metrics_mean(
    metrics: dict[str, float],
    context: DistributedContext,
) -> dict[str, float]:
    if not context.enabled:
        return metrics

    reduced: dict[str, float] = {}
    for key, value in metrics.items():
        tensor = torch.tensor(
            float(value),
            device=torch.device("cuda", context.local_rank),
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced[key] = float((tensor / context.world_size).detach().cpu())
    return reduced


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    amp_enabled: bool,
    log_every: int,
    log_first_n: int,
    scaler: Any,
    max_grad_norm: float | None,
    rank: int = 0,
) -> dict[str, float]:
    model.train()

    running: dict[str, float] = defaultdict(float)
    batches = 0
    epoch_start = time.perf_counter()
    previous_end = epoch_start
    total_steps = len(loader)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if rank == 0:
        print(
            f"epoch={epoch} phase=train start "
            f"batches={total_steps} batch_size={loader.batch_size} "
            f"amp={amp_enabled} log_every={log_every} "
            f"log_first_n={log_first_n}",
            flush=True,
        )

    for step, (images, targets) in enumerate(loader, start=1):
        step_start = time.perf_counter()
        data_time = step_start - previous_end
        instance_count = sum(
            int(target["boxes"].shape[0]) for target in targets
        )

        images = [
            image.to(device, non_blocking=True) for image in images
        ]
        targets = move_targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            loss_components = model(images, targets)
            total_loss = sum(loss_components.values())

        if not torch.isfinite(total_loss):
            detached_components = {
                key: float(value.detach().cpu())
                for key, value in loss_components.items()
            }
            raise FloatingPointError(
                "Non-finite training loss detected: "
                f"epoch={epoch}, step={step}, "
                f"total={float(total_loss.detach().cpu())}, "
                f"components={detached_components}"
            )

        if amp_enabled:
            scaler.scale(total_loss).backward()

            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                )

            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                )

            optimizer.step()

        batches += 1
        current: dict[str, float] = {
            "loss": float(total_loss.detach().cpu())
        }
        running["loss"] += current["loss"]

        for key, value in loss_components.items():
            current[key] = float(value.detach().cpu())
            running[key] += current[key]

        log_this_step = should_log_step(
            step,
            total_steps,
            log_every,
            log_first_n,
        )

        if log_this_step and device.type == "cuda":
            torch.cuda.synchronize(device)

        step_end = time.perf_counter()
        batch_time = step_end - step_start
        previous_end = step_end

        if log_this_step and rank == 0:
            elapsed = step_end - epoch_start
            eta = (
                elapsed / max(1, step)
            ) * max(0, total_steps - step)
            images_per_second = len(images) / max(batch_time, 1.0e-9)
            learning_rate = optimizer.param_groups[0]["lr"]

            print(
                f"epoch={epoch} phase=train "
                f"step={step}/{total_steps} "
                f"progress={100.0 * step / max(1, total_steps):.1f}% "
                f"images={len(images)} instances={instance_count} "
                f"{loss_log_text(current, running, batches)} "
                f"lr={learning_rate:.6g} "
                f"batch_time={batch_time:.3f}s "
                f"data_time={data_time:.3f}s "
                f"img_s={images_per_second:.2f} "
                f"elapsed={format_seconds(elapsed)} "
                f"eta={format_seconds(eta)} "
                f"{gpu_memory_text(device)}",
                flush=True,
            )

    epoch_elapsed = time.perf_counter() - epoch_start
    if rank == 0:
        print(
            f"epoch={epoch} phase=train done "
            f"elapsed={format_seconds(epoch_elapsed)}",
            flush=True,
        )

    return {
        key: value / max(1, batches)
        for key, value in running.items()
    }


def get_base_dataset(dataset: Dataset) -> CocoDetectionDataset:
    current: Dataset = dataset
    while isinstance(current, Subset):
        current = current.dataset
    if not isinstance(current, CocoDetectionDataset):
        raise TypeError(
            "Expected CocoDetectionDataset or Subset[CocoDetectionDataset]"
        )
    return current


def get_dataset_image_ids(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        base = get_base_dataset(dataset)
        return [
            int(base.images[index]["id"])
            for index in dataset.indices
        ]

    base = get_base_dataset(dataset)
    return base.image_ids


def create_empty_coco_results(coco_gt: Any) -> Any:
    """Create an empty COCO result object when there are no predictions."""
    from pycocotools.coco import COCO

    coco_dt = COCO()
    coco_dt.dataset = {
        "images": list(coco_gt.dataset.get("images", [])),
        "categories": list(coco_gt.dataset.get("categories", [])),
        "annotations": [],
    }
    coco_dt.createIndex()
    return coco_dt


@torch.no_grad()
def evaluate_coco(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mapping: CategoryMapping,
    epoch: int,
    log_every: int,
    log_first_n: int,
    score_threshold: float,
) -> dict[str, float]:
    """Evaluate with official pycocotools COCOeval."""

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise RuntimeError(
            "pycocotools is required for official COCO evaluation. "
            "Install it with: pip install pycocotools"
        ) from error

    model.eval()

    base_dataset = get_base_dataset(loader.dataset)
    image_ids = get_dataset_image_ids(loader.dataset)
    coco_gt = COCO(str(base_dataset.annotation_path))

    predictions: list[dict[str, Any]] = []
    total_steps = len(loader)
    total_images = 0
    total_gt = 0
    total_kept_predictions = 0
    evaluation_start = time.perf_counter()

    print(
        f"epoch={epoch} phase=valid start "
        f"batches={total_steps} "
        f"score_threshold={score_threshold}",
        flush=True,
    )

    for step, (images, targets) in enumerate(loader, start=1):
        step_start = time.perf_counter()

        images_on_device = [
            image.to(device, non_blocking=True) for image in images
        ]
        outputs = model(images_on_device)

        batch_gt = sum(
            int(target["boxes"].shape[0]) for target in targets
        )
        batch_prediction_count = 0

        for output, target in zip(outputs, targets):
            image_id = int(target["image_id"].item())

            boxes = output["boxes"].detach().cpu()
            labels = output["labels"].detach().cpu()
            scores = output["scores"].detach().cpu()

            keep = scores >= score_threshold
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]

            batch_prediction_count += int(scores.numel())

            for box, train_label, score in zip(
                boxes,
                labels,
                scores,
            ):
                train_label_int = int(train_label.item())
                if train_label_int not in mapping.train_to_coco:
                    continue

                x1, y1, x2, y2 = (
                    float(value) for value in box.tolist()
                )
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)

                if width <= 0.0 or height <= 0.0:
                    continue

                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": mapping.train_to_coco[
                            train_label_int
                        ],
                        "bbox": [x1, y1, width, height],
                        "score": float(score.item()),
                    }
                )

        total_images += len(images)
        total_gt += batch_gt
        total_kept_predictions += batch_prediction_count

        log_this_step = should_log_step(
            step,
            total_steps,
            log_every,
            log_first_n,
        )

        if log_this_step and device.type == "cuda":
            torch.cuda.synchronize(device)

        if log_this_step:
            now = time.perf_counter()
            elapsed = now - evaluation_start
            eta = (
                elapsed / max(1, step)
            ) * max(0, total_steps - step)
            batch_time = now - step_start

            print(
                f"epoch={epoch} phase=valid "
                f"step={step}/{total_steps} "
                f"progress={100.0 * step / max(1, total_steps):.1f}% "
                f"images={total_images} "
                f"gt={total_gt} "
                f"kept_predictions={total_kept_predictions} "
                f"batch_time={batch_time:.3f}s "
                f"elapsed={format_seconds(elapsed)} "
                f"eta={format_seconds(eta)} "
                f"{gpu_memory_text(device)}",
                flush=True,
            )

    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
    else:
        coco_dt = create_empty_coco_results(coco_gt)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = sorted(image_ids)
    coco_eval.params.catIds = sorted(mapping.coco_to_train.keys())
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats

    metrics: dict[str, float] = {
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
        "pred_total": float(total_kept_predictions),
    }

    # COCOeval precision shape:
    # [IoU threshold, recall threshold, category, area, maxDets].
    precision = coco_eval.eval.get("precision")
    if precision is not None:
        iou_thresholds = coco_eval.params.iouThrs
        iou50_index = int(np.argmin(np.abs(iou_thresholds - 0.5)))
        area_all_index = 0
        max_dets_100_index = len(coco_eval.params.maxDets) - 1

        for category_index, coco_category_id in enumerate(
            coco_eval.params.catIds
        ):
            values = precision[
                iou50_index,
                :,
                category_index,
                area_all_index,
                max_dets_100_index,
            ]
            valid_values = values[values > -1]
            class_ap50 = (
                float(np.mean(valid_values))
                if valid_values.size
                else 0.0
            )

            train_label = mapping.coco_to_train[coco_category_id]
            class_name = mapping.train_to_name[train_label]
            safe_class_name = sanitize_metric_name(class_name)
            metrics[
                f"class_{train_label}_{safe_class_name}_ap50"
            ] = class_ap50

    print(
        f"epoch={epoch} phase=valid done "
        f"map_50_95={metrics['map_50_95']:.4f} "
        f"map50={metrics['map50']:.4f} "
        f"map75={metrics['map75']:.4f} "
        f"map_small={metrics['map_small']:.4f} "
        f"ar100={metrics['ar_100']:.4f} "
        f"gt_total={int(metrics['gt_total'])} "
        f"pred_total={int(metrics['pred_total'])} "
        f"elapsed={format_seconds(time.perf_counter() - evaluation_start)}",
        flush=True,
    )

    for train_label in range(1, mapping.num_classes):
        class_name = mapping.train_to_name[train_label]
        safe_class_name = sanitize_metric_name(class_name)
        key = f"class_{train_label}_{safe_class_name}_ap50"
        if key in metrics:
            print(
                f"epoch={epoch} phase=valid "
                f"class={train_label}:{class_name} "
                f"ap50={metrics[key]:.4f}",
                flush=True,
            )

    return metrics


def sanitize_metric_name(name: str) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() else "_"
        for character in name
    )
    parts = [part for part in normalized.split("_") if part]
    return "_".join(parts) or "unnamed"


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    epoch: int,
    metrics: dict[str, Any],
    mapping: CategoryMapping,
    best_map50: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "metrics": metrics,
        "best_map50": best_map50,
        "mapping": asdict(mapping),
        "args": vars(args),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def restore_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    mapping: CategoryMapping,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    stored_mapping = checkpoint.get("mapping")
    expected_mapping = asdict(mapping)
    if stored_mapping is not None and stored_mapping != expected_mapping:
        raise ValueError(
            "Checkpoint category mapping does not match current dataset.\n"
            f"checkpoint={stored_mapping}\n"
            f"current={expected_mapping}"
        )

    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    scaler_state = checkpoint.get("scaler")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])

    cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
    if torch.cuda.is_available() and cuda_rng_state_all is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state_all)

    completed_epoch = int(checkpoint["epoch"])
    best_map50 = float(checkpoint.get("best_map50", -math.inf))

    print(
        f"resume loaded={path} "
        f"completed_epoch={completed_epoch} "
        f"next_epoch={completed_epoch + 1} "
        f"best_map50={best_map50:.6f}",
        flush=True,
    )

    return completed_epoch + 1, best_map50


def load_model_checkpoint_for_eval(
    path: Path,
    model: torch.nn.Module,
    mapping: CategoryMapping,
    device: torch.device,
) -> int:
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    stored_mapping = checkpoint.get("mapping")
    expected_mapping = asdict(mapping)
    if stored_mapping is not None and stored_mapping != expected_mapping:
        raise ValueError(
            "Checkpoint category mapping does not match current dataset.\n"
            f"checkpoint={stored_mapping}\n"
            f"current={expected_mapping}"
        )

    model.load_state_dict(checkpoint["model"], strict=True)
    epoch = int(checkpoint["epoch"])

    print(
        f"eval checkpoint loaded={path} epoch={epoch}",
        flush=True,
    )
    return epoch


def write_metrics_row(
    path: Path,
    row: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0

    if file_exists:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            existing_header = next(reader)

        current_header = list(row.keys())
        if existing_header != current_header:
            raise ValueError(
                "metrics.csv header differs from the current metrics. "
                "Use a new output directory or --overwrite.\n"
                f"existing={existing_header}\n"
                f"current={current_header}"
            )

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(row.keys()),
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def limit_dataset(
    dataset: Dataset,
    limit: int | None,
) -> Dataset:
    if limit is None:
        return dataset
    if limit < 1:
        raise ValueError("Dataset limit must be at least 1")

    indices = list(range(min(limit, len(dataset))))
    return Subset(dataset, indices)


def count_dataset_instances(dataset: Dataset) -> int:
    if isinstance(dataset, Subset):
        base = get_base_dataset(dataset)
        return sum(
            len(
                base.annotations_by_image.get(
                    int(base.images[index]["id"]),
                    [],
                )
            )
            for index in dataset.indices
        )

    base = get_base_dataset(dataset)
    return sum(
        len(base.annotations_by_image.get(image_id, []))
        for image_id in base.image_ids
    )


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_argument: str) -> torch.device:
    requested = device_argument.strip().lower()

    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"warning: requested device={device_argument}, "
            "but CUDA is unavailable; falling back to CPU",
            flush=True,
        )
        return torch.device("cpu")

    device = torch.device(device_argument)

    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(
            f"cuda device={device} "
            f"name={torch.cuda.get_device_name(device)}",
            flush=True,
        )

    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN on a COCO-format dataset"
    )

    parser.add_argument(
        "--data-root",
        default="data/drill_bit_coco",
    )
    parser.add_argument(
        "--output-dir",
        default="results/baseline/faster_rcnn",
    )
    parser.add_argument(
        "--weights",
        default="coco",
        help=(
            "coco/default/pretrained or "
            "none/random/scratch"
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--step-size", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Optional gradient clipping norm, e.g. 10.0",
    )

    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--distributed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable DDP. Default auto-enables when torchrun sets "
            "WORLD_SIZE > 1."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--log-first-n", type=int, default=3)
    parser.add_argument("--eval-log-every", type=int, default=50)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.001,
        help=(
            "Prediction filtering before COCOeval. "
            "Keep this very low for standard AP evaluation."
        ),
    )

    parser.add_argument(
        "--keep-empty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep background-only images during training",
    )
    parser.add_argument(
        "--trainable-backbone-layers",
        type=int,
        default=None,
    )

    parser.add_argument("--min-size", type=int, default=640)
    parser.add_argument("--max-size", type=int, default=640)
    parser.add_argument(
        "--small-object-anchors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use FPN anchor sizes 8,16,32,64,128",
    )

    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--max-test-images", type=int, default=None)

    parser.add_argument(
        "--test-after-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate the selected checkpoint on the test split after training",
    )
    parser.add_argument(
        "--test-split",
        default="test",
        help="Dataset split name used for final test evaluation",
    )
    parser.add_argument(
        "--test-checkpoint",
        choices=("best", "last"),
        default="best",
        help="Checkpoint to evaluate on the final test split",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to last.pt or another compatible checkpoint",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove old metrics/config/checkpoints in output-dir",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if args.step_size < 1:
        raise ValueError("--step-size must be at least 1")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("--gamma must be in (0, 1]")
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be in [0, 1]")
    if args.min_size < 1 or args.max_size < args.min_size:
        raise ValueError(
            "Require 1 <= --min-size <= --max-size"
        )
    if args.max_grad_norm is not None and args.max_grad_norm <= 0.0:
        raise ValueError("--max-grad-norm must be positive")
    if args.max_test_images is not None and args.max_test_images < 1:
        raise ValueError("--max-test-images must be at least 1")


def prepare_output_directory(
    output_dir: Path,
    resume_path: Path | None,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    managed_files = [
        output_dir / "metrics.csv",
        output_dir / "last.pt",
        output_dir / "best.pt",
        output_dir / "run_config.json",
        output_dir / "test_metrics.json",
    ]

    if overwrite and resume_path is not None:
        raise ValueError("--overwrite and --resume cannot be used together")

    if overwrite:
        for path in managed_files:
            if path.exists():
                path.unlink()
        return

    if resume_path is None:
        existing = [path for path in managed_files if path.exists()]
        if existing:
            formatted = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                "Output directory already contains run files: "
                f"{formatted}. Use --resume or --overwrite."
            )


def main() -> None:
    args = parse_args()
    validate_args(args)
    distributed = setup_distributed(
        should_enable_distributed(args.distributed)
    )
    seed_everything(args.seed)

    try:
        try:
            torch.set_float32_matmul_precision("high")
        except (AttributeError, RuntimeError):
            pass

        if distributed.enabled:
            device = resolve_device(f"cuda:{distributed.local_rank}")
        else:
            device = resolve_device(args.device)
        amp_enabled = bool(args.amp and device.type == "cuda")

        output_dir = Path(args.output_dir)
        resume_path = Path(args.resume) if args.resume else None

        if resume_path is not None and not resume_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}"
            )

        if distributed.is_main:
            prepare_output_directory(
                output_dir=output_dir,
                resume_path=resume_path,
                overwrite=args.overwrite,
            )
        distributed_barrier(distributed)

        train_dataset_full = CocoDetectionDataset(
            root=args.data_root,
            split="train",
            transforms=make_transforms(train=True),
            keep_empty=args.keep_empty,
        )
        mapping = train_dataset_full.mapping

        val_dataset_full = CocoDetectionDataset(
            root=args.data_root,
            split="valid",
            mapping=mapping,
            transforms=make_transforms(train=False),
            keep_empty=True,
        )

        train_dataset = limit_dataset(
            train_dataset_full,
            args.max_train_images,
        )
        val_dataset = limit_dataset(
            val_dataset_full,
            args.max_val_images,
        )

        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty")
        if len(val_dataset) == 0:
            raise ValueError("Validation dataset is empty")

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

        train_generator = torch.Generator()
        train_generator.manual_seed(args.seed)

        loader_common_kwargs: dict[str, Any] = {
            "num_workers": args.workers,
            "collate_fn": collate_fn,
            "pin_memory": device.type == "cuda",
            "persistent_workers": args.workers > 0,
            "worker_init_fn": seed_worker,
        }

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            generator=train_generator if train_sampler is None else None,
            drop_last=False,
            **loader_common_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            **loader_common_kwargs,
        )

        raw_model = create_model(
            num_classes=mapping.num_classes,
            weights=args.weights,
            trainable_backbone_layers=args.trainable_backbone_layers,
            min_size=args.min_size,
            max_size=args.max_size,
            small_object_anchors=args.small_object_anchors,
        )
        raw_model.to(device)

        trainable_parameters = [
            parameter
            for parameter in raw_model.parameters()
            if parameter.requires_grad
        ]

        optimizer = torch.optim.SGD(
            trainable_parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma,
        )
        scaler = create_grad_scaler(enabled=amp_enabled)

        start_epoch = 1
        best_map50 = -math.inf

        if resume_path is not None:
            start_epoch, best_map50 = restore_checkpoint(
                path=resume_path,
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                mapping=mapping,
                device=device,
            )

        train_model: torch.nn.Module = raw_model
        if distributed.enabled:
            train_model = DistributedDataParallel(
                raw_model,
                device_ids=[distributed.local_rank],
                output_device=distributed.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

        run_config = {
            "arguments": vars(args),
            "mapping": asdict(mapping),
            "train_images": len(train_dataset),
            "train_instances": count_dataset_instances(train_dataset),
            "train_empty_images_full": train_dataset_full.empty_image_count,
            "valid_images": len(val_dataset),
            "valid_instances": count_dataset_instances(val_dataset),
            "valid_empty_images_full": val_dataset_full.empty_image_count,
            "torch_version": torch.__version__,
            "torchvision_version": __import__("torchvision").__version__,
            "device": str(device),
            "amp_enabled": amp_enabled,
            "distributed": asdict(distributed),
            "batch_size_per_rank": args.batch_size,
            "global_train_batch_size": args.batch_size
            * distributed.world_size,
        }
        if distributed.is_main:
            write_json(output_dir / "run_config.json", run_config)

            print(
                "Faster R-CNN training "
                f"data={args.data_root} "
                f"train_images={len(train_dataset)} "
                f"train_instances={run_config['train_instances']} "
                f"train_empty_full={train_dataset_full.empty_image_count} "
                f"valid_images={len(val_dataset)} "
                f"valid_instances={run_config['valid_instances']} "
                f"valid_empty_full={val_dataset_full.empty_image_count} "
                f"classes={mapping.train_to_name} "
                f"coco_to_train={mapping.coco_to_train} "
                f"device={device} "
                f"weights={args.weights} "
                f"small_object_anchors={args.small_object_anchors} "
                f"distributed={distributed.enabled} "
                f"world_size={distributed.world_size} "
                f"batch_size_per_rank={args.batch_size}",
                flush=True,
            )

        if start_epoch > args.epochs:
            if distributed.is_main:
                print(
                    f"Nothing to train: checkpoint epoch={start_epoch - 1} "
                    f"is already >= requested epochs={args.epochs}",
                    flush=True,
                )
            if not args.test_after_train:
                return

        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            epoch_learning_rate = optimizer.param_groups[0]["lr"]

            if distributed.is_main:
                print(
                    f"epoch={epoch} start lr={epoch_learning_rate:.6g}",
                    flush=True,
                )

            train_metrics = train_one_epoch(
                model=train_model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                amp_enabled=amp_enabled,
                log_every=max(1, args.log_every),
                log_first_n=max(0, args.log_first_n),
                scaler=scaler,
                max_grad_norm=args.max_grad_norm,
                rank=distributed.rank,
            )
            train_metrics = reduce_metrics_mean(train_metrics, distributed)

            val_metrics: dict[str, float] = {}
            if distributed.is_main:
                val_metrics = evaluate_coco(
                    model=raw_model,
                    loader=val_loader,
                    device=device,
                    mapping=mapping,
                    epoch=epoch,
                    log_every=max(1, args.eval_log_every),
                    log_first_n=max(0, args.log_first_n),
                    score_threshold=args.score_threshold,
                )

            metrics_row: dict[str, Any] = {
                "epoch": epoch,
                "lr": epoch_learning_rate,
                **{
                    f"train_{key}": value
                    for key, value in train_metrics.items()
                },
                **{
                    f"val_{key}": value
                    for key, value in val_metrics.items()
                },
            }

            if distributed.is_main:
                write_metrics_row(
                    output_dir / "metrics.csv",
                    metrics_row,
                )

                current_map50 = val_metrics["map50"]
                is_best = current_map50 > best_map50
                if is_best:
                    best_map50 = current_map50
            else:
                is_best = False

            # Advance scheduler after the epoch. The saved scheduler state is
            # ready for the next epoch; metrics.csv keeps the LR actually used.
            scheduler.step()

            if distributed.is_main:
                save_checkpoint(
                    path=output_dir / "last.pt",
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    metrics=metrics_row,
                    mapping=mapping,
                    best_map50=best_map50,
                    args=args,
                )

                saved_paths = [str(output_dir / "last.pt")]

                if is_best:
                    save_checkpoint(
                        path=output_dir / "best.pt",
                        model=raw_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        metrics=metrics_row,
                        mapping=mapping,
                        best_map50=best_map50,
                        args=args,
                    )
                    saved_paths.append(str(output_dir / "best.pt"))

                train_summary = " ".join(
                    f"{key}={train_metrics[key]:.4f}"
                    for key in (
                        "loss",
                        "loss_classifier",
                        "loss_box_reg",
                        "loss_objectness",
                        "loss_rpn_box_reg",
                    )
                    if key in train_metrics
                )

                print(
                    f"epoch={epoch} summary "
                    f"{train_summary} "
                    f"lr_used={epoch_learning_rate:.6g} "
                    f"next_lr={optimizer.param_groups[0]['lr']:.6g}",
                    flush=True,
                )
                print(
                    f"epoch={epoch} summary "
                    f"val_map_50_95={val_metrics['map_50_95']:.4f} "
                    f"val_map50={val_metrics['map50']:.4f} "
                    f"val_map75={val_metrics['map75']:.4f} "
                    f"val_map_small={val_metrics['map_small']:.4f} "
                    f"val_ar100={val_metrics['ar_100']:.4f} "
                    f"best_map50={best_map50:.4f} "
                    f"saved={','.join(saved_paths)}",
                    flush=True,
                )

            distributed_barrier(distributed)

        distributed_barrier(distributed)

        if args.test_after_train and distributed.is_main:
            test_checkpoint_path = output_dir / f"{args.test_checkpoint}.pt"
            if not test_checkpoint_path.is_file():
                raise FileNotFoundError(
                    "Requested test checkpoint not found: "
                    f"{test_checkpoint_path}"
                )

            checkpoint_epoch = load_model_checkpoint_for_eval(
                path=test_checkpoint_path,
                model=raw_model,
                mapping=mapping,
                device=device,
            )

            test_dataset_full = CocoDetectionDataset(
                root=args.data_root,
                split=args.test_split,
                mapping=mapping,
                transforms=make_transforms(train=False),
                keep_empty=True,
            )
            test_dataset = limit_dataset(
                test_dataset_full,
                args.max_test_images,
            )
            if len(test_dataset) == 0:
                raise ValueError("Test dataset is empty")

            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                **loader_common_kwargs,
            )

            test_metrics = evaluate_coco(
                model=raw_model,
                loader=test_loader,
                device=device,
                mapping=mapping,
                epoch=checkpoint_epoch,
                log_every=max(1, args.eval_log_every),
                log_first_n=max(0, args.log_first_n),
                score_threshold=args.score_threshold,
            )

            test_payload: dict[str, Any] = {
                "split": args.test_split,
                "checkpoint": args.test_checkpoint,
                "checkpoint_path": str(test_checkpoint_path),
                "checkpoint_epoch": checkpoint_epoch,
                "images": len(test_dataset),
                "instances": count_dataset_instances(test_dataset),
                "empty_images_full": test_dataset_full.empty_image_count,
                "metrics": test_metrics,
            }
            write_json(output_dir / "test_metrics.json", test_payload)

            print(
                f"phase=test done "
                f"split={args.test_split} "
                f"checkpoint={args.test_checkpoint} "
                f"checkpoint_epoch={checkpoint_epoch} "
                f"map_50_95={test_metrics['map_50_95']:.4f} "
                f"map50={test_metrics['map50']:.4f} "
                f"map75={test_metrics['map75']:.4f} "
                f"map_small={test_metrics['map_small']:.4f} "
                f"saved={output_dir / 'test_metrics.json'}",
                flush=True,
            )
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
