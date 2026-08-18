#!/usr/bin/env python3
"""
Export ODAM heatmap quality metrics from saved checkpoints.

This script does not train. It forwards validation images through saved
checkpoints with ODAM inference enabled and writes rows containing:
  - Bounding Box Energy Ratio
  - Pointing Game hit
  - Saliency IoU
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    tqdm = None

import train
from network import Network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export Pointing Game and Saliency IoU from saved ODAM/DPGA "
            "checkpoints without retraining."
        )
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("outputs"),
        help="Directory containing run subdirectories.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Run directory names to export. Defaults to e0 e1 e2 e3 e4 e5 e6 if present.",
    )
    parser.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
        help="Checkpoint file to load from each run directory.",
    )
    parser.add_argument(
        "--val-images",
        type=Path,
        default=Path("data/coco/valid"),
        help="Validation image root.",
    )
    parser.add_argument(
        "--val-ann",
        type=Path,
        default=Path("data/coco/valid/_annotations.coco.json"),
        help="Validation COCO annotation file.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Optional output JSON filename per run. Default is "
            "xai_quality_<checkpoint>.json."
        ),
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=None,
        help="Override resize min_size. Defaults to checkpoint args or 800.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Override resize max_size. Defaults to checkpoint args or 1333.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Detection score threshold. Defaults to checkpoint args or 0.05.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=None,
        help="Prediction-GT IoU threshold for matched XAI rows. Defaults to checkpoint args or 0.5.",
    )
    parser.add_argument(
        "--saliency-threshold-ratio",
        type=float,
        default=0.5,
        help="Saliency mask threshold as ratio of per-DAM max value.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Validation DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device, for example cuda or cpu.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional smoke/debug limit on validation images.",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Also export baseline if selected/discovered. Disabled by default.",
    )
    return parser.parse_args()


def finite_float(value, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def checkpoint_args(checkpoint: Dict) -> Dict:
    args = checkpoint.get("args", {})
    return args if isinstance(args, dict) else {}


def choose(value, fallback):
    return fallback if value is None else value


def discover_run_dirs(outputs: Path, selected: Optional[Sequence[str]], include_baseline: bool) -> List[Path]:
    if selected:
        run_dirs = [outputs / name for name in selected]
    else:
        preferred = ["e0", "e1", "e2", "e3", "e4", "e5", "e6"]
        run_dirs = [outputs / name for name in preferred if (outputs / name).is_dir()]
        if include_baseline and (outputs / "baseline").is_dir():
            run_dirs.insert(0, outputs / "baseline")

    out = []
    for run_dir in run_dirs:
        if run_dir.name.lower() == "baseline" and not include_baseline:
            continue
        out.append(run_dir)
    return out


def detector_config_from_checkpoint(checkpoint: Dict) -> train.DetectorConfig:
    data = checkpoint.get("detector_config")
    if not isinstance(data, dict):
        raise ValueError("checkpoint does not contain detector_config")

    valid_names = {field.name for field in fields(train.DetectorConfig)}
    kwargs = {
        key: value
        for key, value in data.items()
        if key in valid_names
    }
    config = train.DetectorConfig(**kwargs)
    train.validate_config(config)
    return config


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[Network, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{checkpoint_path} is not a training checkpoint dict")

    config = detector_config_from_checkpoint(checkpoint)
    model = Network(config)
    result = train.load_initial_weights(model, str(checkpoint_path))
    if not result.loaded:
        raise RuntimeError(f"no checkpoint tensors loaded from {checkpoint_path}")
    model.to(device)
    model.eval()
    model.set_odam_inference(True)
    return model, checkpoint


def gt_box_to_dam_rect(
    gt_box: torch.Tensor,
    dam_h: int,
    dam_w: int,
    resized_h: float,
    resized_w: float,
) -> Tuple[int, int, int, int]:
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
    return x1, y1, x2, y2


def dam_metrics_in_box(
    dam_flat: torch.Tensor,
    dam_h: int,
    dam_w: int,
    gt_box: torch.Tensor,
    resized_h: float,
    resized_w: float,
    threshold_ratio: float,
) -> Dict[str, float]:
    if dam_h <= 0 or dam_w <= 0 or dam_flat.numel() != dam_h * dam_w:
        return {}

    dam = dam_flat.reshape(dam_h, dam_w).float().clamp(min=0)
    total = float(dam.sum().detach().cpu())
    if total <= 1e-12:
        return {}

    x1, y1, x2, y2 = gt_box_to_dam_rect(
        gt_box,
        dam_h,
        dam_w,
        resized_h,
        resized_w,
    )

    if x2 <= x1 or y2 <= y1:
        energy = 0.0
        gt_mask = dam.new_zeros((dam_h, dam_w), dtype=torch.bool)
    else:
        energy = float(dam[y1:y2, x1:x2].sum().detach().cpu()) / max(total, 1e-12)
        gt_mask = dam.new_zeros((dam_h, dam_w), dtype=torch.bool)
        gt_mask[y1:y2, x1:x2] = True

    flat_idx = int(torch.argmax(dam).detach().cpu())
    peak_y = flat_idx // dam_w
    peak_x = flat_idx % dam_w
    pointing_hit = bool(gt_mask[peak_y, peak_x].detach().cpu())

    max_value = float(dam.max().detach().cpu())
    threshold = max_value * float(threshold_ratio)
    saliency_mask = dam >= threshold if max_value > 0 else dam > 0

    intersection = int((saliency_mask & gt_mask).sum().detach().cpu())
    union = int((saliency_mask | gt_mask).sum().detach().cpu())
    saliency_iou = float(intersection) / float(union) if union > 0 else float("nan")

    return {
        "dam_energy_in_gt": float(energy),
        "dam_energy_total": float(total),
        "pointing_game_hit": 1.0 if pointing_hit else 0.0,
        "saliency_iou": saliency_iou,
        "peak_x": float(peak_x),
        "peak_y": float(peak_y),
        "peak_in_gt": 1.0 if pointing_hit else 0.0,
        "saliency_threshold": float(threshold),
        "saliency_threshold_ratio": float(threshold_ratio),
        "saliency_area": float(int(saliency_mask.sum().detach().cpu())),
        "gt_area_in_dam": float(int(gt_mask.sum().detach().cpu())),
        "saliency_gt_intersection": float(intersection),
        "saliency_gt_union": float(union),
    }


def compute_xai_quality_rows(
    pred: torch.Tensor,
    gt_boxes: torch.Tensor,
    meta: Dict,
    score_threshold: float,
    iou_threshold: float,
    saliency_threshold_ratio: float,
) -> List[Dict]:
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
        available = (gt_labels == label) & (~gt_matched)
        if not available.any():
            continue

        candidate_indices = torch.nonzero(available, as_tuple=False).squeeze(1)
        ious = torch.as_tensor(
            train._iou_numpy(
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
        quality = dam_metrics_in_box(
            det[6:-2],
            dam_h,
            dam_w,
            valid_gt[gt_index, :4],
            resized_h=float(meta["resized_h"]),
            resized_w=float(meta["resized_w"]),
            threshold_ratio=float(saliency_threshold_ratio),
        )
        if not quality:
            continue
        if not math.isfinite(float(quality["dam_energy_in_gt"])):
            continue

        row = {
            "image_id": int(meta["image_id"]),
            "category_label": label,
            "score": float(det[4].detach().cpu()),
            "iou": best_iou,
            "dam_h": dam_h,
            "dam_w": dam_w,
        }
        row.update(quality)
        rows.append(row)

    return rows


def summarize_rows(rows: Sequence[Dict]) -> Dict:
    def values(name: str) -> List[float]:
        out = []
        for row in rows:
            try:
                value = float(row[name])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                out.append(value)
        return out

    def mean(name: str) -> float:
        vals = values(name)
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "samples": int(len(rows)),
        "bbox_energy_ratio": mean("dam_energy_in_gt"),
        "pointing_game": mean("pointing_game_hit"),
        "saliency_iou": mean("saliency_iou"),
        "detection_match_iou": mean("iou"),
        "mean_score": mean("score"),
    }


def build_dataset(args: argparse.Namespace, checkpoint: Dict) -> train.CocoDetectionTrainDataset:
    ckpt_args = checkpoint_args(checkpoint)
    min_size = int(choose(args.min_size, ckpt_args.get("min_size", 800)))
    max_size = int(choose(args.max_size, ckpt_args.get("max_size", 1333)))
    dataset = train.CocoDetectionTrainDataset(
        args.val_images,
        args.val_ann,
        min_size=min_size,
        max_size=max_size,
    )
    if args.max_images is not None:
        max_images = max(0, min(int(args.max_images), len(dataset)))
        dataset = Subset(dataset, list(range(max_images)))
    return dataset


def export_one_run(
    run_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict:
    checkpoint_path = run_dir / f"{args.checkpoint}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")

    model, checkpoint = load_model(checkpoint_path, device)
    ckpt_args = checkpoint_args(checkpoint)
    score_threshold = finite_float(
        choose(args.score_threshold, ckpt_args.get("eval_score_threshold", 0.05)),
        0.05,
    )
    iou_threshold = finite_float(
        choose(args.iou_threshold, ckpt_args.get("odam_quality_iou", 0.5)),
        0.5,
    )

    dataset = build_dataset(args, checkpoint)
    label_to_cat_id = getattr(dataset, "label_to_cat_id", None)
    if label_to_cat_id is None and isinstance(dataset, Subset):
        label_to_cat_id = dataset.dataset.label_to_cat_id

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=train.detection_collate,
    )

    iterator: Iterable = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=f"xai {run_dir.name}", leave=False)

    rows: List[Dict] = []
    predictions: List[Dict] = []
    model.eval()
    model.set_odam_inference(True)

    with torch.enable_grad():
        for image, im_info, gt_boxes, metas in iterator:
            if len(metas) != 1:
                raise RuntimeError("export_xai_metrics requires batch_size=1")
            image = image.to(device, non_blocking=True)
            im_info = im_info.to(device, non_blocking=True)
            gt_boxes = gt_boxes.to(device, non_blocking=True)
            pred = model(image, im_info)
            meta = metas[0]
            rows.extend(
                compute_xai_quality_rows(
                    pred=pred,
                    gt_boxes=gt_boxes[0],
                    meta=meta,
                    score_threshold=score_threshold,
                    iou_threshold=iou_threshold,
                    saliency_threshold_ratio=float(args.saliency_threshold_ratio),
                )
            )
            predictions.extend(
                train.postprocess_single_image(
                    pred,
                    meta=meta,
                    label_to_cat_id=label_to_cat_id,
                    score_threshold=score_threshold,
                    nms_threshold=finite_float(ckpt_args.get("eval_nms", 0.5), 0.5),
                    max_detections=int(finite_float(ckpt_args.get("max_detections", 100), 100)),
                )
            )

    output_name = args.output_name or f"xai_quality_{args.checkpoint}.json"
    output_path = run_dir / output_name
    summary_path = run_dir / output_name.replace(".json", "_summary.json")
    prediction_path = run_dir / output_name.replace(".json", "_predictions.json")

    summary = summarize_rows(rows)
    summary.update(
        {
            "run_dir": run_dir.name,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "method": checkpoint.get("method"),
            "val_images": str(args.val_images),
            "val_ann": str(args.val_ann),
            "num_images": int(len(dataset)),
            "score_threshold": score_threshold,
            "iou_threshold": iou_threshold,
            "saliency_threshold_ratio": float(args.saliency_threshold_ratio),
            "output": str(output_path),
            "predictions_output": str(prediction_path),
        }
    )

    output_path.write_text(json.dumps(rows), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if not args.outputs.exists():
        raise FileNotFoundError(args.outputs)
    if not args.val_images.exists():
        raise FileNotFoundError(args.val_images)
    if not args.val_ann.exists():
        raise FileNotFoundError(args.val_ann)
    if not 0.0 <= float(args.saliency_threshold_ratio) <= 1.0:
        raise ValueError("--saliency-threshold-ratio must be in [0, 1]")

    device = torch.device(args.device)
    run_dirs = discover_run_dirs(args.outputs, args.runs, args.include_baseline)
    if not run_dirs:
        raise RuntimeError("no run directories selected")

    summaries = []
    for run_dir in run_dirs:
        summary = export_one_run(run_dir, args, device)
        summaries.append(summary)
        print(
            "[xai-export] "
            f"run={run_dir.name} "
            f"samples={summary['samples']} "
            f"bbox_energy={summary['bbox_energy_ratio']:.6f} "
            f"pointing_game={summary['pointing_game']:.6f} "
            f"saliency_iou={summary['saliency_iou']:.6f}"
        )

    summary_path = args.outputs / f"xai_quality_{args.checkpoint}_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"[xai-export] wrote {summary_path}")


if __name__ == "__main__":
    main()
