#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import MISSING, asdict, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rcnn_odamTrain.network import Network
from rcnn_odamTrain.train import (
    CategoryMapping,
    CocoDrillBitDataset,
    TrainConfig,
    build_loader,
    compute_odam_loss_weight,
    evaluate_coco,
    maybe_subset,
    set_odam_loss_weight,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained rcnn_odamTrain checkpoint over inference "
            "threshold/post-processing grids without retraining."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--log-first-n", type=int, default=1)
    parser.add_argument("--max-combinations", type=int, default=64)
    parser.add_argument("--score-thresholds", type=float, nargs="+", default=[0.001])
    parser.add_argument("--pred-cls-thresholds", type=float, nargs="+", default=[0.05, 0.08, 0.1, 0.12])
    parser.add_argument("--rcnn-nms-thresholds", type=float, nargs="+", default=[0.45, 0.5])
    parser.add_argument("--detections-per-image", type=int, nargs="+", default=[75, 100])
    parser.add_argument("--odam-nms-low-thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--odam-nms-high-thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--odam-nms-resize-short-edges", type=int, nargs="+", default=None)
    parser.add_argument("--no-odam-nms", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_thresholds(values: list[float], name: str) -> None:
    for value in values:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} values must be in [0, 1], got {value}")


def normalize_int_dict(obj: dict[Any, Any]) -> dict[int, int]:
    return {int(key): int(value) for key, value in obj.items()}


def normalize_name_dict(obj: dict[Any, Any]) -> dict[int, str]:
    return {int(key): str(value) for key, value in obj.items()}


def mapping_from_checkpoint(checkpoint: dict[str, Any]) -> CategoryMapping:
    raw = checkpoint.get("mapping")
    if not isinstance(raw, dict):
        raise KeyError("Checkpoint does not contain a mapping dictionary")
    return CategoryMapping(
        coco_to_train=normalize_int_dict(raw["coco_to_train"]),
        train_to_coco=normalize_int_dict(raw["train_to_coco"]),
        train_to_name=normalize_name_dict(raw["train_to_name"]),
    )


def config_from_checkpoint(checkpoint: dict[str, Any]) -> TrainConfig:
    raw = dict(checkpoint.get("config") or {})
    kwargs: dict[str, Any] = {}
    for field in fields(TrainConfig):
        if field.name in raw:
            kwargs[field.name] = raw[field.name]
        elif field.default is not MISSING:
            kwargs[field.name] = field.default
        elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
            kwargs[field.name] = field.default_factory()  # type: ignore[misc]
        else:
            raise KeyError(f"Checkpoint config is missing required field: {field.name}")
    config = TrainConfig(**kwargs)
    # The checkpoint strict-loads all model weights, so avoid downloading or
    # resolving torchvision pretrained weights during model construction.
    config.backbone_weights = "none"
    return config


def checkpoint_args_for_weight(checkpoint: dict[str, Any], config: TrainConfig) -> SimpleNamespace:
    raw_args = checkpoint.get("args") or {}
    return SimpleNamespace(
        odam_loss_weight=raw_args.get("odam_loss_weight", config.odam_loss_weight),
        odam_loss_start_epoch=raw_args.get("odam_loss_start_epoch", config.odam_loss_start_epoch),
        odam_loss_warmup_epochs=raw_args.get("odam_loss_warmup_epochs", config.odam_loss_warmup_epochs),
    )


def metric_row(
    combo_index: int,
    checkpoint_path: Path,
    checkpoint_epoch: int,
    split: str,
    parameters: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "combo_index": combo_index,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "split": split,
        **parameters,
    }
    row.update(metrics)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def best_by_metric(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for metric in ("map50", "map_50_95", "map75", "ar_100"):
        values = [row for row in rows if isinstance(row.get(metric), (int, float))]
        if values:
            selected[metric] = max(values, key=lambda row: float(row[metric]))
    if rows:
        selected["pred_total_lowest"] = min(rows, key=lambda row: float(row.get("pred_total", float("inf"))))
    return selected


def main() -> None:
    args = parse_args()
    if args.workers < 0:
        raise ValueError("--workers must be >= 0")
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("--max-images must be >= 1")
    if args.max_combinations < 1:
        raise ValueError("--max-combinations must be >= 1")
    require_thresholds(args.score_thresholds, "--score-thresholds")
    require_thresholds(args.pred_cls_thresholds, "--pred-cls-thresholds")
    require_thresholds(args.rcnn_nms_thresholds, "--rcnn-nms-thresholds")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / f"{checkpoint_path.stem}_threshold_sweep"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --overwrite or choose another directory.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mapping = mapping_from_checkpoint(checkpoint)
    config = config_from_checkpoint(checkpoint)
    checkpoint_epoch = int(checkpoint["epoch"])
    checkpoint_args = checkpoint_args_for_weight(checkpoint, config)
    config.odam_loss_weight_effective = compute_odam_loss_weight(checkpoint_epoch, checkpoint_args)

    image_size = args.image_size or int((checkpoint.get("args") or {}).get("image_size", 640))
    device = torch.device(args.device)
    model = Network(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    set_odam_loss_weight(model, config.odam_loss_weight_effective or 0.0)
    model.eval()

    dataset_full = CocoDrillBitDataset(
        args.data_root,
        args.split,
        mapping=mapping,
        image_size=image_size,
        keep_empty=True,
    )
    dataset = maybe_subset(dataset_full, args.max_images)
    loader = build_loader(
        dataset,
        batch_size=1,
        workers=args.workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    low_thresholds = args.odam_nms_low_thresholds or [config.odam_nms_low_threshold]
    high_thresholds = args.odam_nms_high_thresholds or [config.odam_nms_high_threshold]
    resize_edges = args.odam_nms_resize_short_edges or [config.odam_nms_resize_short_edge]
    require_thresholds(low_thresholds, "--odam-nms-low-thresholds")
    require_thresholds(high_thresholds, "--odam-nms-high-thresholds")

    combinations = list(
        itertools.product(
            args.score_thresholds,
            args.pred_cls_thresholds,
            args.rcnn_nms_thresholds,
            args.detections_per_image,
            low_thresholds,
            high_thresholds,
            resize_edges,
        )
    )
    if len(combinations) > args.max_combinations:
        raise ValueError(
            f"Threshold grid has {len(combinations)} combinations, exceeding --max-combinations={args.max_combinations}."
        )

    rows: list[dict[str, Any]] = []
    for combo_index, (
        score_threshold,
        pred_cls_threshold,
        rcnn_nms_threshold,
        detections_per_image,
        odam_nms_low_threshold,
        odam_nms_high_threshold,
        odam_nms_resize_short_edge,
    ) in enumerate(combinations, start=1):
        if odam_nms_low_threshold > odam_nms_high_threshold:
            print(
                f"skip combo={combo_index} low_threshold={odam_nms_low_threshold} "
                f"> high_threshold={odam_nms_high_threshold}",
                flush=True,
            )
            continue
        config.pred_cls_threshold = float(pred_cls_threshold)
        config.rcnn_nms_threshold = float(rcnn_nms_threshold)
        config.rcnn_detections_per_image = int(detections_per_image)
        config.odam_nms = bool(config.odam_nms and not args.no_odam_nms)
        config.odam_nms_low_threshold = float(odam_nms_low_threshold)
        config.odam_nms_high_threshold = float(odam_nms_high_threshold)
        config.odam_nms_resize_short_edge = int(odam_nms_resize_short_edge)

        parameters = {
            "score_threshold": float(score_threshold),
            "pred_cls_threshold": float(pred_cls_threshold),
            "rcnn_nms_threshold": float(rcnn_nms_threshold),
            "rcnn_detections_per_image": int(detections_per_image),
            "odam_nms": bool(config.odam_nms),
            "odam_nms_low_threshold": float(odam_nms_low_threshold),
            "odam_nms_high_threshold": float(odam_nms_high_threshold),
            "odam_nms_resize_short_edge": int(odam_nms_resize_short_edge),
        }
        print(f"sweep combo={combo_index}/{len(combinations)} {parameters}", flush=True)
        metrics = evaluate_coco(
            model,
            loader,
            device,
            mapping,
            args.split,
            float(score_threshold),
            args.log_every,
            args.log_first_n,
        )
        rows.append(metric_row(combo_index, checkpoint_path, checkpoint_epoch, args.split, parameters, metrics))

    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "data_root": args.data_root,
        "split": args.split,
        "image_size": image_size,
        "max_images": args.max_images,
        "num_combinations": len(rows),
        "config": asdict(config),
        "results": rows,
        "best_by_metric": best_by_metric(rows),
    }
    write_json(output_dir / "threshold_sweep_results.json", payload)
    write_csv(output_dir / "threshold_sweep_results.csv", rows)
    print(f"saved_json={output_dir / 'threshold_sweep_results.json'}", flush=True)
    print(f"saved_csv={output_dir / 'threshold_sweep_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
