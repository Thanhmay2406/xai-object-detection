#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from det_oprs.bbox_opr import box_overlap_opr
from rcnn_odamTrain.evaluate_threshold_sweep import config_from_checkpoint, mapping_from_checkpoint
from rcnn_odamTrain.network import Network, prepare_odam_nms_heatmaps
from rcnn_odamTrain.train import CocoDrillBitDataset, build_loader, maybe_subset, move_batch, set_odam_loss_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ODAM evidence for instance-specific explanation and object "
            "discrimination on a trained rcnn_odamTrain checkpoint."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--pred-cls-threshold", type=float, default=0.001)
    parser.add_argument("--rcnn-nms-threshold", type=float, default=0.95)
    parser.add_argument("--detections-per-image", type=int, default=300)
    parser.add_argument("--heatmap-resize-short-edge", type=int, default=50)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--crowd-iou-threshold", type=float, default=0.1)
    parser.add_argument("--vea-top-fraction", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--log-first-n", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def finite_mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return sum(finite) / len(finite)


def finite_median(values: list[float]) -> float | None:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    mid = len(finite) // 2
    if len(finite) % 2:
        return finite[mid]
    return 0.5 * (finite[mid - 1] + finite[mid])


def binary_auc(positive_scores: list[float], negative_scores: list[float]) -> float | None:
    positives = [float(value) for value in positive_scores if math.isfinite(float(value))]
    negatives = [float(value) for value in negative_scores if math.isfinite(float(value))]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
            total += 1
    return wins / max(1, total)


def box_mask(height: int, width: int, box: torch.Tensor, image_height: float, image_width: float, device) -> torch.Tensor:
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    if image_height <= 0.0 or image_width <= 0.0:
        return mask
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    left = max(0, min(width, int(math.floor(x1 / image_width * width))))
    right = max(0, min(width, int(math.ceil(x2 / image_width * width))))
    top = max(0, min(height, int(math.floor(y1 / image_height * height))))
    bottom = max(0, min(height, int(math.ceil(y2 / image_height * height))))
    if right <= left or bottom <= top:
        return mask
    mask[top:bottom, left:right] = True
    return mask


def top_fraction_mask(heatmap: torch.Tensor, fraction: float) -> torch.Tensor:
    flat = heatmap.flatten()
    if flat.numel() == 0:
        return torch.zeros_like(heatmap, dtype=torch.bool)
    fraction = min(1.0, max(0.0, float(fraction)))
    if fraction <= 0.0:
        threshold = flat.max()
        return heatmap >= threshold
    k = max(1, int(math.ceil(float(flat.numel()) * fraction)))
    threshold = torch.topk(flat, k).values[-1]
    return heatmap >= threshold


def mask_iou(first: torch.Tensor, second: torch.Tensor) -> float:
    intersection = (first & second).sum().item()
    union = (first | second).sum().item()
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


def safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    denom = float(denominator.detach().item())
    if denom <= 1e-12:
        return 0.0
    return float(numerator.detach().item()) / denom


def detection_heatmap_metrics(
    heatmap: torch.Tensor,
    target_box: torch.Tensor,
    other_boxes: torch.Tensor,
    image_height: float,
    image_width: float,
    top_fraction: float,
) -> dict[str, float]:
    height, width = int(heatmap.shape[0]), int(heatmap.shape[1])
    device = heatmap.device
    heatmap = torch.nan_to_num(heatmap.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    total_energy = heatmap.sum()
    target_mask = box_mask(height, width, target_box, image_height, image_width, device)
    other_mask = torch.zeros_like(target_mask)
    for other_box in other_boxes:
        other_mask |= box_mask(height, width, other_box, image_height, image_width, device)
    outside_target = ~target_mask
    top_mask = top_fraction_mask(heatmap, top_fraction)
    peak_index = int(heatmap.flatten().argmax().item()) if heatmap.numel() else 0
    peak_y, peak_x = divmod(peak_index, max(1, width))

    return {
        "target_energy_ratio": safe_ratio(heatmap[target_mask].sum(), total_energy),
        "other_object_energy_ratio": safe_ratio(heatmap[other_mask].sum(), total_energy),
        "outside_target_energy_ratio": safe_ratio(heatmap[outside_target].sum(), total_energy),
        "pointing_box_hit": 1.0 if bool(target_mask[peak_y, peak_x].item()) else 0.0,
        "other_object_peak": 1.0 if bool(other_mask[peak_y, peak_x].item()) else 0.0,
        "box_proxy_vea_iou": mask_iou(top_mask, target_mask),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"image_count": len(rows)}
    count_fields = [
        "gt_count",
        "prediction_count",
        "matched_prediction_count",
        "same_object_pair_count",
        "different_object_pair_count",
        "crowded_different_pair_count",
    ]
    for field in count_fields:
        summary[field] = int(sum(int(row.get(field, 0)) for row in rows))

    mean_fields = [
        "target_energy_ratio_mean",
        "other_object_energy_ratio_mean",
        "outside_target_energy_ratio_mean",
        "pointing_box_accuracy",
        "other_object_peak_rate",
        "box_proxy_vea_iou_mean",
        "same_object_cosine_mean",
        "same_object_cosine_median",
        "different_object_cosine_mean",
        "different_object_cosine_median",
        "crowded_different_cosine_mean",
        "discrimination_margin",
        "pair_auc",
    ]
    for field in mean_fields:
        values = [float(row[field]) for row in rows if row.get(field, "") not in ("", None)]
        summary[field] = finite_mean(values)
    return summary


def format_optional(value: Any, precision: int = 6) -> str:
    if isinstance(value, (int, float)) and value is not None and math.isfinite(float(value)):
        return f"{float(value):.{precision}f}"
    return "n/a"


def write_report(path: Path, summary: dict[str, Any], config: dict[str, Any]) -> None:
    lines = [
        "# ODAM Evidence Report",
        "",
        "This report evaluates evidence aligned with the original ODAM paper's object-discrimination claim.",
        "",
        "## Run",
        "",
        f"- checkpoint: `{config['checkpoint']}`",
        f"- split: `{config['split']}`",
        f"- images: `{summary.get('image_count', 0)}`",
        f"- match IoU threshold: `{config['match_iou_threshold']}`",
        f"- inference NMS threshold: `{config['rcnn_nms_threshold']}`",
        f"- prediction threshold: `{config['pred_cls_threshold']}`",
        "",
        "## Summary",
        "",
        "| Evidence | Metric | Value |",
        "|---|---|---:|",
        f"| Target localization | target energy ratio | {format_optional(summary.get('target_energy_ratio_mean'))} |",
        f"| Target localization | pointing-box accuracy | {format_optional(summary.get('pointing_box_accuracy'))} |",
        f"| Target localization | box-proxy VEA IoU | {format_optional(summary.get('box_proxy_vea_iou_mean'))} |",
        f"| Leakage | other-object energy ratio | {format_optional(summary.get('other_object_energy_ratio_mean'))} |",
        f"| Leakage | other-object peak rate | {format_optional(summary.get('other_object_peak_rate'))} |",
        f"| Object discrimination | same-object cosine | {format_optional(summary.get('same_object_cosine_mean'))} |",
        f"| Object discrimination | different-object cosine | {format_optional(summary.get('different_object_cosine_mean'))} |",
        f"| Object discrimination | cosine margin | {format_optional(summary.get('discrimination_margin'))} |",
        f"| Object discrimination | pair AUC | {format_optional(summary.get('pair_auc'))} |",
        "",
        "## Counts",
        "",
        "| Count | Value |",
        "|---|---:|",
        f"| GT boxes | {summary.get('gt_count', 0)} |",
        f"| predictions after configured post-processing | {summary.get('prediction_count', 0)} |",
        f"| predictions matched to GT | {summary.get('matched_prediction_count', 0)} |",
        f"| same-object pairs | {summary.get('same_object_pair_count', 0)} |",
        f"| different-object pairs | {summary.get('different_object_pair_count', 0)} |",
        f"| crowded different-object pairs | {summary.get('crowded_different_pair_count', 0)} |",
        "",
        "## Interpretation Guide",
        "",
        "- Higher same-object cosine supports the Odam-Train consistency claim.",
        "- Lower different-object cosine and a positive cosine margin support the separation/object-discrimination claim.",
        "- Lower other-object energy and other-object peak rate indicate less explanation leakage to neighboring objects.",
        "- Higher pointing-box accuracy and box-proxy VEA IoU indicate better localization of the explanation map.",
        "",
        "This is not a replacement for the original paper's full human trust study or CrowdHuman AP/JI/MR/Recall protocol.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_image(
    outputs: torch.Tensor,
    gt_boxes: torch.Tensor,
    image_height: float,
    image_width: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    valid_gt = gt_boxes[gt_boxes[:, 4] >= 0]
    row: dict[str, Any] = {
        "gt_count": int(valid_gt.shape[0]),
        "prediction_count": int(outputs.shape[0]),
        "matched_prediction_count": 0,
        "same_object_pair_count": 0,
        "different_object_pair_count": 0,
        "crowded_different_pair_count": 0,
    }
    if outputs.numel() == 0 or valid_gt.numel() == 0:
        return row

    outputs = outputs[outputs[:, 4] >= float(args.score_threshold)]
    row["prediction_count"] = int(outputs.shape[0])
    if outputs.numel() == 0:
        return row

    boxes = outputs[:, :4]
    labels = outputs[:, 5].long()
    dams = outputs[:, 6:-2]
    dam_size = outputs[0, -2:].long()
    dam_height, dam_width = int(dam_size[0].item()), int(dam_size[1].item())
    if dam_height <= 0 or dam_width <= 0 or dams.shape[1] != dam_height * dam_width:
        row["invalid_heatmap_layout"] = 1
        return row

    gt_labels = valid_gt[:, 4].long()
    overlaps = box_overlap_opr(boxes, valid_gt[:, :4])
    same_class = labels[:, None] == gt_labels[None, :]
    overlaps = torch.where(same_class, overlaps, overlaps.new_zeros(overlaps.shape))
    best_iou, assigned_gt = overlaps.max(dim=1)
    matched = best_iou >= float(args.match_iou_threshold)
    row["matched_prediction_count"] = int(matched.sum().item())
    if int(matched.sum().item()) == 0:
        return row

    matched_indices = torch.nonzero(matched, as_tuple=False).flatten()
    matched_gt = assigned_gt[matched_indices]
    matched_iou = best_iou[matched_indices]
    vectors = prepare_odam_nms_heatmaps(dams[matched_indices], dam_size, int(args.heatmap_resize_short_edge))
    heatmaps = dams[matched_indices].reshape(-1, dam_height, dam_width)

    target_metrics: dict[str, list[float]] = {
        "target_energy_ratio": [],
        "other_object_energy_ratio": [],
        "outside_target_energy_ratio": [],
        "pointing_box_hit": [],
        "other_object_peak": [],
        "box_proxy_vea_iou": [],
    }
    for local_idx, gt_idx in enumerate(matched_gt.tolist()):
        other_mask = torch.arange(valid_gt.shape[0], device=valid_gt.device) != int(gt_idx)
        metrics = detection_heatmap_metrics(
            heatmaps[local_idx],
            valid_gt[int(gt_idx), :4],
            valid_gt[other_mask, :4],
            image_height,
            image_width,
            float(args.vea_top_fraction),
        )
        for key, value in metrics.items():
            target_metrics[key].append(float(value))

    row["target_energy_ratio_mean"] = finite_mean(target_metrics["target_energy_ratio"])
    row["other_object_energy_ratio_mean"] = finite_mean(target_metrics["other_object_energy_ratio"])
    row["outside_target_energy_ratio_mean"] = finite_mean(target_metrics["outside_target_energy_ratio"])
    row["pointing_box_accuracy"] = finite_mean(target_metrics["pointing_box_hit"])
    row["other_object_peak_rate"] = finite_mean(target_metrics["other_object_peak"])
    row["box_proxy_vea_iou_mean"] = finite_mean(target_metrics["box_proxy_vea_iou"])
    row["matched_iou_mean"] = finite_mean([float(value) for value in matched_iou.detach().cpu()])

    pair_sims = vectors @ vectors.T
    same_scores: list[float] = []
    different_scores: list[float] = []
    crowded_different_scores: list[float] = []
    matched_boxes = boxes[matched_indices]
    pair_box_iou = box_overlap_opr(matched_boxes, matched_boxes)
    for left in range(vectors.shape[0]):
        for right in range(left + 1, vectors.shape[0]):
            score = float(pair_sims[left, right].detach().item())
            if int(matched_gt[left].item()) == int(matched_gt[right].item()):
                same_scores.append(score)
            else:
                different_scores.append(score)
                if float(pair_box_iou[left, right].detach().item()) >= float(args.crowd_iou_threshold):
                    crowded_different_scores.append(score)

    row["same_object_pair_count"] = len(same_scores)
    row["different_object_pair_count"] = len(different_scores)
    row["crowded_different_pair_count"] = len(crowded_different_scores)
    row["same_object_cosine_mean"] = finite_mean(same_scores)
    row["same_object_cosine_median"] = finite_median(same_scores)
    row["different_object_cosine_mean"] = finite_mean(different_scores)
    row["different_object_cosine_median"] = finite_median(different_scores)
    row["crowded_different_cosine_mean"] = finite_mean(crowded_different_scores)
    if row["same_object_cosine_mean"] is not None and row["different_object_cosine_mean"] is not None:
        row["discrimination_margin"] = row["same_object_cosine_mean"] - row["different_object_cosine_mean"]
    row["pair_auc"] = binary_auc(same_scores, different_scores)
    return row


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be in [0, 1]")
    if not 0.0 <= args.pred_cls_threshold <= 1.0:
        raise ValueError("--pred-cls-threshold must be in [0, 1]")
    if not 0.0 <= args.rcnn_nms_threshold <= 1.0:
        raise ValueError("--rcnn-nms-threshold must be in [0, 1]")
    if not 0.0 <= args.match_iou_threshold <= 1.0:
        raise ValueError("--match-iou-threshold must be in [0, 1]")
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("--max-images must be >= 1")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "odam_evidence"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --overwrite or choose another directory.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mapping = mapping_from_checkpoint(checkpoint)
    config = config_from_checkpoint(checkpoint)
    config.backbone_weights = "none"
    config.odam_nms = False
    config.pred_cls_threshold = float(args.pred_cls_threshold)
    config.rcnn_nms_threshold = float(args.rcnn_nms_threshold)
    config.rcnn_detections_per_image = int(args.detections_per_image)
    config.odam_loss_weight_effective = 0.0

    image_size = args.image_size or int((checkpoint.get("args") or {}).get("image_size", 640))
    device = torch.device(args.device)
    model = Network(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    set_odam_loss_weight(model, 0.0)
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

    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    total_steps = len(loader)
    for step, batch in enumerate(loader, start=1):
        images, im_info, gt_boxes, image_ids = move_batch(batch, device)
        if images.shape[0] != 1:
            raise ValueError("ODAM evidence evaluation requires batch_size=1")
        outputs = model(images, im_info).detach()
        row = evaluate_image(
            outputs,
            gt_boxes[0],
            float(im_info[0, 0].detach().item()),
            float(im_info[0, 1].detach().item()),
            args,
        )
        row["image_id"] = int(image_ids.item())
        rows.append(row)
        if step <= args.log_first_n or step == total_steps or step % max(1, args.log_every) == 0:
            elapsed = time.perf_counter() - start
            print(
                f"odam_evidence step={step}/{total_steps} image_id={int(image_ids.item())} "
                f"pred={row.get('prediction_count', 0)} matched={row.get('matched_prediction_count', 0)} "
                f"same_pairs={row.get('same_object_pair_count', 0)} diff_pairs={row.get('different_object_pair_count', 0)} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    summary = summarize_rows(rows)
    run_config = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "data_root": str(args.data_root),
        "split": args.split,
        "image_size": image_size,
        "score_threshold": float(args.score_threshold),
        "pred_cls_threshold": float(args.pred_cls_threshold),
        "rcnn_nms_threshold": float(args.rcnn_nms_threshold),
        "detections_per_image": int(args.detections_per_image),
        "match_iou_threshold": float(args.match_iou_threshold),
        "crowd_iou_threshold": float(args.crowd_iou_threshold),
        "heatmap_resize_short_edge": int(args.heatmap_resize_short_edge),
        "vea_top_fraction": float(args.vea_top_fraction),
    }
    payload = {"config": run_config, "summary": summary}
    write_csv(output_dir / "per_image_metrics.csv", rows)
    write_json(output_dir / "summary.json", payload)
    write_report(output_dir / "report.md", summary, run_config)
    print(f"odam_evidence done output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
