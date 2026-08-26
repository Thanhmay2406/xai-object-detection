import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.distributed as dist

from pycocotools.cocoeval import COCOeval
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.data import (
    COCODetectionDataset,
    collate_fn,
)
from src.FasterRCNN import build_faster_rcnn


# ============================================================
# Distributed
# ============================================================

def is_distributed():
    return (
        dist.is_available()
        and dist.is_initialized()
    )


def is_main_process():
    return (
        not is_distributed()
        or dist.get_rank() == 0
    )


def setup_distributed(args):
    world_size = int(
        os.environ.get("WORLD_SIZE", "1")
    )

    rank = int(
        os.environ.get("RANK", "0")
    )

    local_rank = int(
        os.environ.get("LOCAL_RANK", "0")
    )

    distributed = world_size > 1

    if args.device == "cuda" and torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)

        device = torch.device(
            "cuda",
            local_rank,
        )

        backend = "nccl"

    else:
        device = torch.device("cpu")
        backend = "gloo"

    if distributed:
        dist.init_process_group(
            backend=backend,
            init_method="env://",
        )

    return {
        "distributed": distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
    }


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def unwrap_model(model):
    if isinstance(model, DistributedDataParallel):
        return model.module

    return model


def rank0_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# Train
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    scaler,
    use_amp=True,
    main_process=True,
):
    model.train()

    running = {
        "loss_total": 0.0,
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
    }

    num_batches = 0

    progress = tqdm(
        dataloader,
        desc=f"Train epoch {epoch}",
        disable=not main_process,
    )

    for images, targets in progress:

        images = [
            image.to(
                device,
                non_blocking=True,
            )
            for image in images
        ]

        targets = [
            {
                key: value.to(
                    device,
                    non_blocking=True,
                )
                for key, value in target.items()
            }
            for target in targets
        ]

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=use_amp,
        ):
            loss_dict = model(
                images,
                targets,
            )

            total_loss = sum(
                loss for loss in loss_dict.values()
            )

        if not torch.isfinite(total_loss):
            rank0_print(
                "\nNon-finite loss detected:"
            )

            rank0_print(loss_dict)

            raise RuntimeError(
                f"Loss became non-finite: "
                f"{total_loss.item()}"
            )

        scaler.scale(
            total_loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        total_value = total_loss.item()

        running["loss_total"] += total_value

        for key in loss_dict:
            if key in running:
                running[key] += (
                    loss_dict[key].item()
                )

        num_batches += 1

        if main_process:
            progress.set_postfix(
                loss=f"{total_value:.4f}"
            )

    if is_distributed():
        keys = list(running.keys())

        reduced = torch.tensor(
            [running[key] for key in keys]
            + [float(num_batches)],
            dtype=torch.float64,
            device=device,
        )

        dist.all_reduce(
            reduced,
            op=dist.ReduceOp.SUM,
        )

        total_batches = max(
            int(reduced[-1].item()),
            1,
        )

        for index, key in enumerate(keys):
            running[key] = (
                reduced[index].item()
                / total_batches
            )

    else:
        for key in running:
            running[key] /= max(
                num_batches,
                1,
            )

    return running


# ============================================================
# COCO Evaluation
# ============================================================

@torch.no_grad()
def evaluate_coco(
    model,
    dataloader,
    dataset,
    device,
):
    model.eval()

    coco_results = []

    inference_times = []

    progress = tqdm(
        dataloader,
        desc="Validation",
    )

    for images, targets in progress:

        images = [
            image.to(
                device,
                non_blocking=True,
            )
            for image in images
        ]

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        outputs = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        inference_times.append(
            elapsed / len(images)
        )

        outputs = [
            {
                key: value.cpu()
                for key, value in output.items()
            }
            for output in outputs
        ]

        for output, target in zip(
            outputs,
            targets,
        ):
            image_id = int(
                target["image_id"].item()
            )

            boxes = output["boxes"]
            scores = output["scores"]
            labels = output["labels"]

            if boxes.numel() == 0:
                continue

            # XYXY -> XYWH
            boxes_xywh = boxes.clone()

            boxes_xywh[:, 2] = (
                boxes[:, 2]
                - boxes[:, 0]
            )

            boxes_xywh[:, 3] = (
                boxes[:, 3]
                - boxes[:, 1]
            )

            boxes_xywh[:, 0] = boxes[:, 0]
            boxes_xywh[:, 1] = boxes[:, 1]

            for box, score, label in zip(
                boxes_xywh,
                scores,
                labels,
            ):
                label = int(label.item())

                category_id = (
                    dataset
                    .label_to_category_id[
                        label
                    ]
                )

                coco_results.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [
                            float(x)
                            for x in box.tolist()
                        ],
                        "score": float(
                            score.item()
                        ),
                    }
                )

    if len(coco_results) == 0:
        print(
            "Warning: model produced "
            "no detections."
        )

        return {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
            "latency_ms_per_image": 0.0,
        }

    coco_gt = dataset.coco

    coco_dt = coco_gt.loadRes(
        coco_results
    )

    evaluator = COCOeval(
        coco_gt,
        coco_dt,
        iouType="bbox",
    )

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats

    latency = (
        np.mean(inference_times)
        * 1000.0
    )

    metrics = {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),

        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),

        "AR_1": float(stats[6]),
        "AR_10": float(stats[7]),
        "AR_100": float(stats[8]),

        "latency_ms_per_image":
            float(latency),
    }

    return metrics


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_ap,
    train_metrics,
    val_metrics,
    dataset,
):
    checkpoint = {
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "best_ap":
            best_ap,

        "train_metrics":
            train_metrics,

        "val_metrics":
            val_metrics,

        "category_id_to_label":
            dataset.category_id_to_label,

        "label_to_category_id":
            dataset.label_to_category_id,

        "class_names":
            dataset.class_names,
    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# CSV log
# ============================================================

def append_csv(
    path,
    epoch,
    lr,
    train_metrics,
    val_metrics,
):
    row = {
        "epoch": epoch,
        "lr": lr,
        **train_metrics,
        **val_metrics,
    }

    file_exists = path.exists()

    with open(
        path,
        "a",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=row.keys(),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# Main
# ============================================================

def main(args):
    dist_info = setup_distributed(args)

    distributed = dist_info["distributed"]
    local_rank = dist_info["local_rank"]
    world_size = dist_info["world_size"]
    device = dist_info["device"]
    main_process = is_main_process()

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(args.seed + dist_info["rank"])

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    rank0_print(f"Device: {device}")
    rank0_print(f"Distributed: {distributed}")

    if distributed:
        rank0_print(f"World size: {world_size}")

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    checkpoint_dir = (
        output_dir
        / "checkpoints"
    )

    if main_process:
        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    if distributed:
        dist.barrier()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = COCODetectionDataset(
        image_dir=args.train_images,
        annotation_file=args.train_annotations,
        train=True,
        horizontal_flip_prob=(
            args.horizontal_flip_prob
        ),
    )

    val_dataset = COCODetectionDataset(
        image_dir=args.val_images,
        annotation_file=args.val_annotations,
        train=False,

        # IMPORTANT:
        # use exactly the same class mapping.
        category_id_to_label=(
            train_dataset
            .category_id_to_label
        ),
    )

    num_classes = (
        train_dataset.num_classes
    )

    rank0_print(
        f"Train images: "
        f"{len(train_dataset)}"
    )

    rank0_print(
        f"Val images: "
        f"{len(val_dataset)}"
    )

    rank0_print(
        f"Classes including background: "
        f"{num_classes}"
    )

    rank0_print(
        "Classes:",
        train_dataset.class_names,
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_sampler = None

    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=dist_info["rank"],
            shuffle=True,
        )

    train_loader = DataLoader(
        train_dataset,

        batch_size=args.batch_size,

        shuffle=(train_sampler is None),

        sampler=train_sampler,

        num_workers=args.workers,

        pin_memory=True,

        persistent_workers=(
            args.workers > 0
        ),

        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=args.workers,

        pin_memory=True,

        persistent_workers=(
            args.workers > 0
        ),

        collate_fn=collate_fn,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_faster_rcnn(
        num_classes=num_classes,

        pretrained_coco=(
            not args.no_coco_pretrained
        ),

        pretrained_backbone=(
            not args.no_backbone_pretrained
        ),

        trainable_backbone_layers=(
            args.trainable_backbone_layers
        ),

        min_size=args.min_size,

        max_size=args.max_size,
    )

    model.to(device)

    if distributed:
        ddp_kwargs = {}

        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [local_rank]
            ddp_kwargs["output_device"] = local_rank

        model = DistributedDataParallel(
            model,
            **ddp_kwargs,
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    trainable_parameters = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.SGD(
        trainable_parameters,

        lr=args.lr,

        momentum=args.momentum,

        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------------
    # LR Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,

        step_size=args.lr_step_size,

        gamma=args.lr_gamma,
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    use_amp = (
        args.amp
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    # --------------------------------------------------------
    # Save experiment config
    # --------------------------------------------------------

    config = vars(args).copy()

    config[
        "num_classes"
    ] = num_classes

    config[
        "class_names"
    ] = train_dataset.class_names

    config["distributed"] = distributed
    config["world_size"] = world_size

    if main_process:
        with open(
            output_dir / "config.json",
            "w",
        ) as f:
            json.dump(
                config,
                f,
                indent=4,
            )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_ap = -1.0

    metrics_csv = (
        output_dir
        / "metrics.csv"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        rank0_print(
            "\n"
            + "=" * 70
        )

        rank0_print(
            f"Epoch "
            f"{epoch}/{args.epochs}"
        )

        rank0_print(
            "=" * 70
        )

        # -------------------------------
        # Train
        # -------------------------------

        train_metrics = (
            train_one_epoch(
                model=model,

                dataloader=train_loader,

                optimizer=optimizer,

                device=device,

                epoch=epoch,

                scaler=scaler,

                use_amp=use_amp,

                main_process=main_process,
            )
        )

        # -------------------------------
        # Validate
        # -------------------------------

        if main_process:
            val_metrics = evaluate_coco(
                model=unwrap_model(model),

                dataloader=val_loader,

                dataset=val_dataset,

                device=device,
            )

        else:
            val_metrics = {}

        # -------------------------------
        # Current LR before scheduler
        # -------------------------------

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        # -------------------------------
        # Print results
        # -------------------------------

        if main_process:
            print("\nTrain:")

            for key, value in (
                train_metrics.items()
            ):
                print(
                    f"  {key:<25}: "
                    f"{value:.6f}"
                )

            print("\nValidation:")

            for key, value in (
                val_metrics.items()
            ):
                print(
                    f"  {key:<25}: "
                    f"{value:.6f}"
                )

        # -------------------------------
        # CSV
        # -------------------------------

        if main_process:
            append_csv(
                metrics_csv,

                epoch,

                current_lr,

                train_metrics,

                val_metrics,
            )

        # -------------------------------
        # Last checkpoint
        # -------------------------------

        if main_process:
            save_checkpoint(
                checkpoint_dir
                / "last.pth",

                unwrap_model(model),

                optimizer,

                scheduler,

                epoch,

                best_ap,

                train_metrics,

                val_metrics,

                train_dataset,
            )

        # -------------------------------
        # Best checkpoint
        # -------------------------------

        current_ap = (
            val_metrics["AP"]
            if main_process
            else best_ap
        )

        if main_process and current_ap > best_ap:

            best_ap = current_ap

            print(
                f"\nNew best AP: "
                f"{best_ap:.4f}"
            )

            save_checkpoint(
                checkpoint_dir
                / "best.pth",

                unwrap_model(model),

                optimizer,

                scheduler,

                epoch,

                best_ap,

                train_metrics,

                val_metrics,

                train_dataset,
            )

        if distributed:
            dist.barrier()

        scheduler.step()

    rank0_print(
        "\nTraining complete."
    )

    rank0_print(
        f"Best AP = "
        f"{best_ap:.4f}"
    )

    rank0_print(
        f"Output: "
        f"{output_dir}"
    )

    cleanup_distributed()


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Baseline Faster R-CNN "
            "+ ResNet50-FPN"
        )
    )

    # Dataset
    parser.add_argument(
        "--train-images",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--train-annotations",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--val-images",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--val-annotations",
        type=str,
        required=True,
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
    )

    parser.add_argument(
        "--lr-step-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.1,
    )

    # Model
    parser.add_argument(
        "--trainable-backbone-layers",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--min-size",
        type=int,
        default=800,
    )

    parser.add_argument(
        "--max-size",
        type=int,
        default=1333,
    )

    parser.add_argument(
        "--no-coco-pretrained",
        action="store_true",
    )

    parser.add_argument(
        "--no-backbone-pretrained",
        action="store_true",
    )

    # Augmentation
    parser.add_argument(
        "--horizontal-flip-prob",
        type=float,
        default=0.5,
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "experiments/"
            "E0_baseline/"
            "seed_42"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
