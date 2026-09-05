"""Paired image-ID bootstrap used by Pipeline 04."""

from __future__ import annotations

import contextlib
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from xai_pruning.utils.io import load_json


def index_by_image(items, key: str = "image_id") -> defaultdict[int, list]:
    """Index annotations or predictions by integer image ID."""

    indexed = defaultdict(list)
    for item in items:
        indexed[int(item[key])].append(item)
    return indexed


def build_bootstrap_replicate(
    gt_dataset: dict,
    baseline_predictions: list[dict],
    main_predictions: list[dict],
    sampled_image_ids,
) -> tuple[dict, list[dict], list[dict]]:
    """Clone sampled images under synthetic IDs, preserving image as sampling unit."""

    images_by_id = {int(image["id"]): image for image in gt_dataset["images"]}
    annotations_by_image = index_by_image(gt_dataset["annotations"])
    baseline_by_image = index_by_image(baseline_predictions)
    main_by_image = index_by_image(main_predictions)
    new_images = []
    new_annotations = []
    new_baseline = []
    new_main = []
    next_annotation_id = 1

    for synthetic_id, original_id in enumerate(sampled_image_ids, start=1):
        original_id = int(original_id)
        image = dict(images_by_id[original_id])
        image["id"] = synthetic_id
        new_images.append(image)
        for annotation in annotations_by_image.get(original_id, []):
            cloned = dict(annotation)
            cloned["id"] = next_annotation_id
            cloned["image_id"] = synthetic_id
            new_annotations.append(cloned)
            next_annotation_id += 1
        for prediction in baseline_by_image.get(original_id, []):
            cloned = dict(prediction)
            cloned["image_id"] = synthetic_id
            new_baseline.append(cloned)
        for prediction in main_by_image.get(original_id, []):
            cloned = dict(prediction)
            cloned["image_id"] = synthetic_id
            new_main.append(cloned)

    new_ground_truth = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": gt_dataset["categories"],
    }
    for optional_key in ("info", "licenses"):
        if optional_key in gt_dataset:
            new_ground_truth[optional_key] = gt_dataset[optional_key]
    return new_ground_truth, new_baseline, new_main


def coco_eval_in_memory(gt_dataset: dict, predictions: list[dict], category_ids) -> dict:
    """Evaluate one synthetic bootstrap dataset with the Pipeline 04 protocol."""

    coco_gt = COCO()
    coco_gt.dataset = gt_dataset
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.catIds = [int(category_id) for category_id in category_ids]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    return {
        "map": float(coco_eval.stats[0]),
        "ap50": float(coco_eval.stats[1]),
        "ap75": float(coco_eval.stats[2]),
    }


def paired_bootstrap_coco(
    annotation_json: str | Path,
    baseline_predictions,
    main_predictions,
    category_ids,
    *,
    reps: int = 1000,
    seed: int = 20260905,
    log_every: int = 50,
) -> list[dict]:
    """Bootstrap paired Baseline/Main COCO metrics by resampling image IDs."""

    gt_dataset = load_json(annotation_json)
    if isinstance(baseline_predictions, (str, Path)):
        baseline_predictions = load_json(baseline_predictions)
    if isinstance(main_predictions, (str, Path)):
        main_predictions = load_json(main_predictions)
    image_ids = np.asarray([int(image["id"]) for image in gt_dataset["images"]], dtype=np.int64)
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(1, reps + 1):
        sampled = rng.choice(image_ids, size=len(image_ids), replace=True)
        boot_gt, boot_baseline, boot_main = build_bootstrap_replicate(
            gt_dataset, baseline_predictions, main_predictions, sampled
        )
        baseline_metrics = coco_eval_in_memory(boot_gt, boot_baseline, category_ids)
        main_metrics = coco_eval_in_memory(boot_gt, boot_main, category_ids)
        row = {"replicate": replicate}
        for metric in ("map", "ap50", "ap75"):
            row[f"baseline_{metric}"] = baseline_metrics[metric]
            row[f"main_{metric}"] = main_metrics[metric]
            row[f"delta_{metric}"] = main_metrics[metric] - baseline_metrics[metric]
        rows.append(row)
        if log_every and replicate % log_every == 0:
            print(f"bootstrap {replicate}/{reps}")
    return rows


def summarize_bootstrap(rows) -> list[dict]:
    """Summarize Pipeline 04 paired deltas using percentile confidence intervals."""

    summaries = []
    for metric in ("map", "ap50", "ap75"):
        values = np.asarray([row[f"delta_{metric}"] for row in rows], dtype=np.float64)
        summaries.append(
            {
                "metric": metric,
                "mean_delta": float(values.mean()),
                "median_delta": float(np.median(values)),
                "ci95_low": float(np.percentile(values, 2.5)),
                "ci95_high": float(np.percentile(values, 97.5)),
                "p_delta_gt_0": float(np.mean(values > 0)),
                "reps": len(values),
            }
        )
    return summaries


def paired_bootstrap_latency(
    baseline_ms,
    main_ms,
    *,
    reps: int = 5000,
    seed: int = 20260905,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Preserve Pipeline 04b's paired per-image latency bootstrap."""

    baseline = np.asarray(baseline_ms, dtype=np.float64)
    main = np.asarray(main_ms, dtype=np.float64)
    if len(baseline) != len(main):
        raise RuntimeError("Pairing mismatch")
    rng = np.random.default_rng(seed)
    reductions = np.empty(reps, dtype=np.float64)
    delta_ms = np.empty(reps, dtype=np.float64)
    for replicate in range(reps):
        indices = rng.integers(0, len(baseline), size=len(baseline))
        baseline_mean = baseline[indices].mean()
        main_mean = main[indices].mean()
        reductions[replicate] = 100.0 * (baseline_mean - main_mean) / baseline_mean
        delta_ms[replicate] = main_mean - baseline_mean
    point_baseline = baseline.mean()
    point_main = main.mean()
    summary = {
        "n_images": len(baseline),
        "baseline_mean_ms": float(point_baseline),
        "main_mean_ms": float(point_main),
        "delta_main_minus_baseline_ms": float(point_main - point_baseline),
        "latency_reduction_pct": float(100.0 * (point_baseline - point_main) / point_baseline),
        "ci95_low_pct": float(np.percentile(reductions, 2.5)),
        "ci95_high_pct": float(np.percentile(reductions, 97.5)),
        "p_main_faster": float(np.mean(reductions > 0)),
        "bootstrap_reps": reps,
    }
    return summary, reductions, delta_ms
