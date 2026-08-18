#!/usr/bin/env python3
"""
Aggregate experiment metrics for Faster R-CNN, ODAM, and DPGA-ODAM runs.

The script reads completed training artifacts such as:
  - metrics.csv
  - experiment.json
  - gradient_diagnostics_rank*.csv
  - odam_quality_epoch_*.json

It intentionally separates official CityPersons metrics from internal
diagnostics. Official MR^-2 Reasonable/Heavy/Small must come from a
CityPersons-compatible evaluator or an external results file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:  # pragma: no cover - exercised only without eval deps
    COCO = None
    COCOeval = None


STAGE_ORDER = {
    "baseline": 0,
    "E0": 1,
    "E1": 2,
    "E2": 3,
    "E3": 4,
    "E4": 5,
    "E5": 6,
    "E6": 7,
}

HIGHER_IS_BETTER = {
    "AP": True,
    "AP50": True,
    "AP75": True,
    "AP_small": True,
    "AP_medium": True,
    "AP_large": True,
    "AR1": True,
    "AR10": True,
    "AR100": True,
    "MR-2_generic": False,
    "ODAM_quality": True,
    "ODAM_quality_mean_iou": True,
}

DETECTION_METRICS = [
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
]

OFFICIAL_CITYPERSONS_COLUMNS = [
    "MR-2_Reasonable",
    "MR-2_Heavy",
    "MR-2_Small",
]

OFFLINE_DETECTION_METRICS = [
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
    "MR-2_Reasonable",
    "MR-2_Heavy",
    "MR-2_Small",
]

NAN = float("nan")


@dataclass
class RunArtifact:
    label: str
    path: Path
    metrics_path: Path
    experiment: Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute post-training metric tables from output directories "
            "according to metric.txt."
        )
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("outputs"),
        help="Directory containing run subdirectories with metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/metrics_report"),
        help="Directory where metric tables and report will be written.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help=(
            "Optional run directory names to include, for example: "
            "baseline e0 e1 e2 e3 e4 e5 e6."
        ),
    )
    parser.add_argument(
        "--citypersons-mr-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV with external official CityPersons MR columns. "
            "Expected columns: run, MR-2_Reasonable, MR-2_Heavy, MR-2_Small."
        ),
    )
    parser.add_argument(
        "--val-ann",
        type=Path,
        default=Path("data/coco/valid/_annotations.coco.json"),
        help=(
            "COCO validation annotation used to recompute offline metrics "
            "from predictions_epoch_*.json. If the file does not exist, "
            "offline prediction evaluation is skipped."
        ),
    )
    parser.add_argument(
        "--offline-prediction-epoch",
        choices=("final", "best", "both"),
        default="both",
        help=(
            "Which saved prediction epoch to evaluate offline. 'best' uses "
            "--offline-best-metric from metrics.csv."
        ),
    )
    parser.add_argument(
        "--offline-best-metric",
        default="AP",
        choices=tuple(HIGHER_IS_BETTER.keys()),
        help="Metric used to select the best prediction epoch for offline eval.",
    )
    parser.add_argument(
        "--offline-iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for offline MR matching.",
    )
    parser.add_argument(
        "--citypersons-ignore-ioa",
        type=float,
        default=0.5,
        help=(
            "Detection-over-ignore-box IoA threshold for ignoring detections "
            "that fall in excluded/ignored CityPersons-style regions."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value, default: float = NAN) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def finite_values(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if finite(v)]


def mean(values: Iterable[float]) -> float:
    vals = finite_values(values)
    return float(statistics.fmean(vals)) if vals else NAN


def median(values: Iterable[float]) -> float:
    vals = finite_values(values)
    return float(statistics.median(vals)) if vals else NAN


def minimum(values: Iterable[float]) -> float:
    vals = finite_values(values)
    return float(min(vals)) if vals else NAN


def maximum(values: Iterable[float]) -> float:
    vals = finite_values(values)
    return float(max(vals)) if vals else NAN


def normalize_run_label(run_dir: Path, experiment: Dict) -> str:
    stage = experiment.get("experiment_stage")
    if stage:
        return str(stage).upper()

    name = run_dir.name
    lower = name.lower()
    if lower == "baseline":
        return "baseline"
    if lower in {"e0", "e1", "e2", "e3", "e4", "e5", "e6"}:
        return lower.upper()
    return name


def sort_key(run: RunArtifact) -> Tuple[int, str]:
    return STAGE_ORDER.get(run.label, 10_000), run.label


def discover_runs(outputs: Path, selected_names: Optional[Sequence[str]]) -> List[RunArtifact]:
    if not outputs.exists():
        raise FileNotFoundError(f"outputs directory does not exist: {outputs}")

    selected = {name.lower() for name in selected_names} if selected_names else None
    runs: List[RunArtifact] = []
    for run_dir in sorted(outputs.iterdir()):
        if not run_dir.is_dir():
            continue
        if selected is not None and run_dir.name.lower() not in selected:
            continue
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        experiment = load_json(run_dir / "experiment.json")
        runs.append(
            RunArtifact(
                label=normalize_run_label(run_dir, experiment),
                path=run_dir,
                metrics_path=metrics_path,
                experiment=experiment,
            )
        )
    return sorted(runs, key=sort_key)


def best_metric(rows: Sequence[Dict[str, str]], metric: str) -> Tuple[float, float]:
    candidates = []
    for row in rows:
        value = to_float(row.get(metric))
        if finite(value):
            candidates.append((value, to_float(row.get("epoch"), default=NAN)))
    if not candidates:
        return NAN, NAN

    higher = HIGHER_IS_BETTER.get(metric, True)
    value, epoch = max(candidates) if higher else min(candidates)
    return float(value), float(epoch)


def final_metric(rows: Sequence[Dict[str, str]], metric: str) -> float:
    if not rows:
        return NAN
    ordered = sorted(rows, key=lambda r: to_float(r.get("epoch"), default=-1.0))
    return to_float(ordered[-1].get(metric))


def summarize_training_metrics(run: RunArtifact, rows: Sequence[Dict[str, str]]) -> Dict:
    epochs = [to_float(row.get("epoch")) for row in rows]
    seconds = [to_float(row.get("seconds")) for row in rows]
    out = {
        "run": run.label,
        "run_dir": run.path.name,
        "method": run.experiment.get("method", rows[0].get("method") if rows else ""),
        "experiment_stage": run.experiment.get("experiment_stage", ""),
        "epochs": int(max(finite_values(epochs)) + 1) if finite_values(epochs) else 0,
        "rows": len(rows),
        "git_commit": run.experiment.get("git_commit", ""),
        "metrics_schema_version": run.experiment.get("metrics_schema_version", ""),
        "world_size": run.experiment.get("world_size", ""),
        "warmup_enabled": run.experiment.get("warmup_enabled", ""),
        "filtering_enabled": run.experiment.get("filtering_enabled", ""),
        "projection_enabled": run.experiment.get("projection_enabled", ""),
        "norm_cap_enabled": run.experiment.get("norm_cap_enabled", ""),
        "gate_enabled": run.experiment.get("gate_enabled", ""),
        "time_per_epoch_mean_s": mean(seconds),
        "time_per_epoch_median_s": median(seconds),
        "time_total_s": sum(finite_values(seconds)),
        "peak_gpu_memory_gb": NAN,
        "inference_ms_per_image": NAN,
        "fps": NAN,
        "num_parameters": NAN,
    }

    for metric in DETECTION_METRICS + ["ODAM_quality", "ODAM_quality_mean_iou"]:
        best, epoch = best_metric(rows, metric)
        out[f"best_{metric}"] = best
        out[f"best_{metric}_epoch"] = epoch
        out[f"final_{metric}"] = final_metric(rows, metric)

    for metric in ["loss_det", "loss_odam", "raw_loss_sum", "loss_proxy"]:
        out[f"mean_{metric}"] = mean(to_float(row.get(metric)) for row in rows)
        out[f"final_{metric}"] = final_metric(rows, metric)

    for metric in ["odam_num_candidates", "odam_num_kept", "odam_keep_ratio"]:
        out[f"mean_{metric}"] = mean(to_float(row.get(metric)) for row in rows)
        out[f"final_{metric}"] = final_metric(rows, metric)

    for metric in [
        "odam_reliability_mean",
        "odam_reliability_std",
        "odam_reliability_p10",
        "odam_reliability_p50",
        "odam_reliability_p90",
        "odam_roi_iou_mean",
        "odam_roi_score_mean",
        "odam_loss_raw",
        "odam_loss_weighted",
        "odam_effective_rois",
        "odam_low_reliability_fraction",
        "odam_high_reliability_fraction",
    ]:
        out[f"mean_{metric}"] = mean(to_float(row.get(metric)) for row in rows)
        out[f"final_{metric}"] = final_metric(rows, metric)

    return out


def read_citypersons_mr(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    if path is None:
        return {}
    rows = read_csv_dicts(path)
    by_run: Dict[str, Dict[str, float]] = {}
    for row in rows:
        run = (row.get("run") or row.get("method") or "").strip()
        if not run:
            continue
        by_run[run] = {
            col: to_float(row.get(col))
            for col in OFFICIAL_CITYPERSONS_COLUMNS
        }
    return by_run


def with_citypersons_mr(row: Dict, official_mr: Dict[str, Dict[str, float]]) -> Dict:
    out = dict(row)
    values = official_mr.get(str(row["run"]), {})
    for col in OFFICIAL_CITYPERSONS_COLUMNS:
        out[col] = values.get(col, NAN)
    out["citypersons_mr_source"] = "external_csv" if values else "not_available"
    return out


def xywh_to_xyxy(box: Sequence[float]) -> np.ndarray:
    x, y, w, h = map(float, box)
    return np.asarray([x, y, x + w, y + h], dtype=np.float64)


def iou_numpy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0,
        boxes[:, 3] - boxes[:, 1],
    )
    return inter / np.maximum(area_a + area_b - inter, 1e-12)


def ioa_numpy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return inter / max(area, 1e-12)


def ann_height(ann: Dict) -> float:
    value = to_float(ann.get("height"))
    if finite(value):
        return value
    return to_float((ann.get("bbox") or [0, 0, 0, NAN])[3])


def ann_vis_ratio(ann: Dict) -> float:
    value = to_float(ann.get("vis_ratio"))
    if finite(value):
        return value
    bbox = ann.get("bbox") or [0, 0, 0, 0]
    vis_bbox = ann.get("vis_bbox") or []
    if len(vis_bbox) >= 4:
        full = max(float(bbox[2]) * float(bbox[3]), 1e-12)
        visible = max(float(vis_bbox[2]) * float(vis_bbox[3]), 0.0)
        return visible / full
    return NAN


def ann_is_ignored(ann: Dict) -> bool:
    return int(ann.get("ignore", 0)) == 1 or int(ann.get("iscrowd", 0)) == 1


def subset_reasonable(ann: Dict) -> bool:
    return ann_height(ann) >= 50.0 and ann_vis_ratio(ann) >= 0.65


def subset_heavy(ann: Dict) -> bool:
    vis = ann_vis_ratio(ann)
    return ann_height(ann) >= 50.0 and 0.20 <= vis < 0.65


def subset_small(ann: Dict) -> bool:
    return 50.0 <= ann_height(ann) < 75.0 and ann_vis_ratio(ann) >= 0.65


def load_annotation_data(path: Optional[Path]) -> Optional[Dict]:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def default_category_id(annotation_data: Dict) -> Optional[int]:
    categories = annotation_data.get("categories") or []
    if not categories:
        return None
    return int(categories[0]["id"])


def annotation_image_ids(annotation_data: Dict) -> List[int]:
    return sorted(int(img["id"]) for img in annotation_data.get("images", []))


def evaluate_coco_predictions(
    annotation_path: Path,
    predictions: Sequence[Dict],
    image_ids: Sequence[int],
) -> Dict[str, float]:
    if COCO is None or COCOeval is None:
        raise ImportError("pycocotools is required for offline COCO evaluation")

    if not predictions:
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

    with redirect_stdout(StringIO()):
        coco_gt = COCO(str(annotation_path))
        coco_dt = coco_gt.loadRes(list(predictions))
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.params.imgIds = list(map(int, image_ids))
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR1": float(stats[6]),
        "AR10": float(stats[7]),
        "AR100": float(stats[8]),
    }


def compute_mr2_from_predictions(
    annotation_data: Dict,
    predictions: Sequence[Dict],
    category_id: int,
    image_ids: Sequence[int],
    valid_ann: Callable[[Dict], bool],
    iou_threshold: float,
    ignore_ioa_threshold: float,
) -> float:
    anns_by_image: Dict[int, List[Dict]] = {}
    for ann in annotation_data.get("annotations", []):
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    gt_by_image: Dict[int, np.ndarray] = {}
    matched_by_image: Dict[int, np.ndarray] = {}
    ignore_by_image: Dict[int, np.ndarray] = {}
    total_gt = 0

    for image_id in image_ids:
        valid_boxes = []
        ignore_boxes = []
        for ann in anns_by_image.get(int(image_id), []):
            if int(ann.get("category_id", category_id)) != int(category_id):
                continue
            is_valid = (not ann_is_ignored(ann)) and valid_ann(ann)
            if is_valid:
                valid_boxes.append(xywh_to_xyxy(ann["bbox"]))
            else:
                ignore_boxes.append(xywh_to_xyxy(ann["bbox"]))

        valid_arr = (
            np.stack(valid_boxes, axis=0)
            if valid_boxes
            else np.zeros((0, 4), dtype=np.float64)
        )
        ignore_arr = (
            np.stack(ignore_boxes, axis=0)
            if ignore_boxes
            else np.zeros((0, 4), dtype=np.float64)
        )
        gt_by_image[int(image_id)] = valid_arr
        matched_by_image[int(image_id)] = np.zeros((valid_arr.shape[0],), dtype=bool)
        ignore_by_image[int(image_id)] = ignore_arr
        total_gt += valid_arr.shape[0]

    if total_gt == 0:
        return NAN

    detections = [
        pred
        for pred in predictions
        if int(pred.get("category_id", category_id)) == int(category_id)
    ]
    detections.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)

    tp: List[float] = []
    fp: List[float] = []
    for det in detections:
        image_id = int(det["image_id"])
        det_box = xywh_to_xyxy(det["bbox"])
        gt_boxes = gt_by_image.get(image_id, np.zeros((0, 4), dtype=np.float64))
        matched = matched_by_image.get(image_id, np.zeros((0,), dtype=bool))

        if gt_boxes.shape[0] > 0:
            ious = iou_numpy(det_box, gt_boxes)
            ious[matched] = -1.0
            best = int(np.argmax(ious))
            if float(ious[best]) >= float(iou_threshold):
                matched[best] = True
                tp.append(1.0)
                fp.append(0.0)
                continue

        ignore_boxes = ignore_by_image.get(image_id, np.zeros((0, 4), dtype=np.float64))
        if ignore_boxes.shape[0] > 0 and float(np.max(ioa_numpy(det_box, ignore_boxes))) >= float(ignore_ioa_threshold):
            continue

        tp.append(0.0)
        fp.append(1.0)

    if not tp:
        return 1.0

    tp_arr = np.cumsum(np.asarray(tp, dtype=np.float64))
    fp_arr = np.cumsum(np.asarray(fp, dtype=np.float64))
    miss_rate = 1.0 - tp_arr / float(total_gt)
    fppi = fp_arr / float(max(len(image_ids), 1))

    sampled = []
    for ref in np.logspace(-2.0, 0.0, 9):
        valid = np.where(fppi <= ref)[0]
        sampled.append(1.0 if len(valid) == 0 else float(miss_rate[valid[-1]]))
    sampled_arr = np.clip(np.asarray(sampled, dtype=np.float64), 1e-10, 1.0)
    return float(np.exp(np.mean(np.log(sampled_arr))))


def prediction_epoch_files(run: RunArtifact) -> Dict[int, Path]:
    files: Dict[int, Path] = {}
    for path in run.path.glob("predictions_epoch_*.json"):
        suffix = path.stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            files[int(suffix)] = path
    return files


def offline_epoch_plan(
    run: RunArtifact,
    metric_rows: Sequence[Dict[str, str]],
    mode: str,
    best_name: str,
) -> Dict[str, Tuple[int, Path]]:
    files = prediction_epoch_files(run)
    if not files:
        return {}

    out: Dict[str, Tuple[int, Path]] = {}
    if mode in {"final", "both"}:
        epoch = max(files)
        out["final"] = (epoch, files[epoch])
    if mode in {"best", "both"}:
        _, best_epoch = best_metric(metric_rows, best_name)
        if finite(best_epoch):
            epoch_int = int(best_epoch)
            if epoch_int in files:
                out["best"] = (epoch_int, files[epoch_int])
    return out


def evaluate_prediction_file_offline(
    annotation_path: Path,
    annotation_data: Dict,
    prediction_path: Path,
    iou_threshold: float,
    ignore_ioa_threshold: float,
) -> Dict[str, float]:
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    image_ids = annotation_image_ids(annotation_data)
    category_id = default_category_id(annotation_data)
    if category_id is None:
        raise ValueError("annotation file has no categories")

    out = evaluate_coco_predictions(annotation_path, predictions, image_ids)
    out["MR-2_generic"] = compute_mr2_from_predictions(
        annotation_data,
        predictions,
        category_id=category_id,
        image_ids=image_ids,
        valid_ann=lambda ann: not ann_is_ignored(ann),
        iou_threshold=iou_threshold,
        ignore_ioa_threshold=ignore_ioa_threshold,
    )
    out["MR-2_Reasonable"] = compute_mr2_from_predictions(
        annotation_data,
        predictions,
        category_id=category_id,
        image_ids=image_ids,
        valid_ann=subset_reasonable,
        iou_threshold=iou_threshold,
        ignore_ioa_threshold=ignore_ioa_threshold,
    )
    out["MR-2_Heavy"] = compute_mr2_from_predictions(
        annotation_data,
        predictions,
        category_id=category_id,
        image_ids=image_ids,
        valid_ann=subset_heavy,
        iou_threshold=iou_threshold,
        ignore_ioa_threshold=ignore_ioa_threshold,
    )
    out["MR-2_Small"] = compute_mr2_from_predictions(
        annotation_data,
        predictions,
        category_id=category_id,
        image_ids=image_ids,
        valid_ann=subset_small,
        iou_threshold=iou_threshold,
        ignore_ioa_threshold=ignore_ioa_threshold,
    )
    return out


def summarize_offline_predictions(
    run: RunArtifact,
    metric_rows: Sequence[Dict[str, str]],
    args: argparse.Namespace,
    annotation_data: Optional[Dict],
) -> Dict:
    val_ann = getattr(args, "val_ann", None)
    if annotation_data is None or val_ann is None or not val_ann.exists():
        return {
            "offline_detection_source": "not_available",
        }

    out: Dict = {
        "offline_detection_source": "saved_predictions",
        "offline_val_ann": str(val_ann),
        "offline_iou_threshold": float(getattr(args, "offline_iou_threshold", 0.5)),
        "offline_citypersons_ignore_ioa": float(getattr(args, "citypersons_ignore_ioa", 0.5)),
        "offline_citypersons_protocol": (
            "CityPersons-style converted-COCO subsets: Reasonable h>=50 "
            "vis>=0.65; Heavy h>=50 and 0.20<=vis<0.65; Small 50<=h<75 "
            "vis>=0.65. Ignored/excluded boxes suppress detections by IoA."
        ),
    }
    plan = offline_epoch_plan(
        run,
        metric_rows,
        mode=getattr(args, "offline_prediction_epoch", "both"),
        best_name=getattr(args, "offline_best_metric", "AP"),
    )
    for label, (epoch, prediction_path) in plan.items():
        prefix = f"offline_{label}"
        out[f"{prefix}_epoch"] = int(epoch)
        out[f"{prefix}_prediction_file"] = str(prediction_path)
        metrics = evaluate_prediction_file_offline(
            val_ann,
            annotation_data,
            prediction_path,
            iou_threshold=float(getattr(args, "offline_iou_threshold", 0.5)),
            ignore_ioa_threshold=float(getattr(args, "citypersons_ignore_ioa", 0.5)),
        )
        for metric, value in metrics.items():
            out[f"{prefix}_{metric}"] = value
    return out


def group_gradient_steps(rows: Sequence[Dict[str, str]]) -> Dict[Tuple[str, str, str], List[Dict[str, str]]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    for row in rows:
        key = (
            str(row.get("epoch", "")),
            str(row.get("step", "")),
            str(row.get("rank", "")),
        )
        groups.setdefault(key, []).append(row)
    return groups


def is_odam_active_gradient_row(row: Dict[str, str], eps: float = 1e-12) -> bool:
    return (
        abs(to_float(row.get("raw_odam_norm"))) > eps
        or abs(to_float(row.get("odam_norm_raw"))) > eps
        or abs(to_float(row.get("loss_odam"))) > eps
    )


def summarize_gradient_rows(run: RunArtifact) -> Dict:
    files = sorted(run.path.glob("gradient_diagnostics_rank*.csv"))
    rows: List[Dict[str, str]] = []
    for path in files:
        rows.extend(read_csv_dicts(path))

    active_rows = [
        row for row in rows
        if is_odam_active_gradient_row(row)
    ]
    step_groups = group_gradient_steps(rows)
    active_step_groups = [
        group for group in step_groups.values()
        if any(is_odam_active_gradient_row(row) for row in group)
    ]

    def col(name: str, source: Sequence[Dict[str, str]] = active_rows) -> List[float]:
        return [to_float(row.get(name)) for row in source]

    conflict_row_flags = [
        1.0 if to_float(row.get("conflict_raw"), 0.0) > 0.5 else 0.0
        for row in active_rows
    ]
    projected_row_flags = [
        1.0 if to_float(row.get("projected"), 0.0) > 0.5 else 0.0
        for row in active_rows
    ]
    cap_row_flags = [
        1.0 if to_float(row.get("cap_active"), 0.0) > 0.5 else 0.0
        for row in active_rows
    ]
    unsafe_row_flags = [
        1.0 if to_float(row.get("unsafe_descent"), 0.0) > 0.5 else 0.0
        for row in active_rows
    ]

    conflict_step_flags = [
        1.0
        if any(to_float(row.get("conflict_raw"), 0.0) > 0.5 for row in group)
        else 0.0
        for group in active_step_groups
    ]
    projected_step_flags = [
        1.0
        if any(to_float(row.get("projected"), 0.0) > 0.5 for row in group)
        else 0.0
        for group in active_step_groups
    ]
    cap_step_flags = [
        1.0
        if any(to_float(row.get("cap_active"), 0.0) > 0.5 for row in group)
        else 0.0
        for group in active_step_groups
    ]

    return {
        "run": run.label,
        "run_dir": run.path.name,
        "gradient_rows": len(rows),
        "gradient_active_rows": len(active_rows),
        "gradient_steps": len(step_groups),
        "gradient_active_steps": len(active_step_groups),
        "cosine_similarity_mean": mean(col("cosine_raw")),
        "cosine_similarity_median": median(col("cosine_raw")),
        "cosine_similarity_min": minimum(col("cosine_raw")),
        "cosine_similarity_max": maximum(col("cosine_raw")),
        "cosine_projected_mean": mean(col("cosine_projected")),
        "cosine_projected_median": median(col("cosine_projected")),
        "gradient_conflict_rate": mean(conflict_step_flags),
        "gradient_conflict_rate_module_rows": mean(conflict_row_flags),
        "det_gradient_norm_mean": mean(col("det_gradient_norm")),
        "det_gradient_norm_median": median(col("det_gradient_norm")),
        "odam_gradient_norm_mean": mean(col("raw_odam_norm")),
        "odam_gradient_norm_median": median(col("raw_odam_norm")),
        "final_aux_gradient_norm_mean": mean(col("final_odam_norm")),
        "final_aux_gradient_norm_median": median(col("final_odam_norm")),
        "final_gradient_norm": NAN,
        "gradient_norm_ratio_mean": mean(col("aux_to_det_raw")),
        "gradient_norm_ratio_median": median(col("aux_to_det_raw")),
        "gradient_norm_ratio_max": maximum(col("aux_to_det_raw")),
        "projection_rate": mean(projected_step_flags),
        "projection_rate_module_rows": mean(projected_row_flags),
        "norm_cap_rate": mean(cap_step_flags),
        "norm_cap_rate_module_rows": mean(cap_row_flags),
        "unsafe_descent_rate_module_rows": mean(unsafe_row_flags),
        "gate_mean": mean(col("gate")),
        "alpha_mean": mean(col("alpha")),
        "effective_weight_mean": mean(col("effective_weight")),
        "gradient_scope": (
            rows[0].get("gradient_scope", "")
            if rows
            else ""
        ),
    }


def read_odam_quality_files(run: RunArtifact) -> List[Dict]:
    out: List[Dict] = []
    for path in sorted(run.path.glob("odam_quality_epoch_*.json")):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(rows, list):
            epoch = path.stem.rsplit("_", 1)[-1]
            for row in rows:
                if isinstance(row, dict):
                    item = dict(row)
                    item["epoch"] = epoch
                    out.append(item)
    return out


def read_exported_xai_quality_files(run: RunArtifact) -> List[Dict]:
    out: List[Dict] = []
    for path in sorted(run.path.glob("xai_quality_*.json")):
        if path.name.endswith("_summary.json") or path.name.endswith("_predictions.json"):
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(rows, list):
            source = path.stem
            for row in rows:
                if isinstance(row, dict):
                    item = dict(row)
                    item["source_file"] = path.name
                    item["source_export"] = source
                    out.append(item)
    return out


def read_exported_xai_quality_summary(run: RunArtifact) -> Optional[Dict]:
    summary_paths = sorted(run.path.parent.glob("xai_quality_*_summary.json"))
    for path in summary_paths:
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("run_dir", "")).lower() == run.path.name.lower():
                item = dict(row)
                item["source_file"] = path.name
                return item
    return None


def summarize_xai_rows(run: RunArtifact, training_summary: Dict) -> Dict:
    exported_rows = read_exported_xai_quality_files(run)
    exported_summary = None if exported_rows else read_exported_xai_quality_summary(run)
    if exported_summary:
        bbox_energy = to_float(exported_summary.get("bbox_energy_ratio"))
        pointing_game = to_float(exported_summary.get("pointing_game"))
        saliency_iou = to_float(exported_summary.get("saliency_iou"))
        match_iou = to_float(exported_summary.get("detection_match_iou"))
        samples_value = to_float(exported_summary.get("samples"))
        samples = int(samples_value) if math.isfinite(samples_value) else 0
        return {
            "run": run.label,
            "run_dir": run.path.name,
            "xai_source": "export_xai_metrics_summary",
            "bbox_energy_ratio_mean": bbox_energy,
            "bbox_energy_ratio_median": bbox_energy,
            "bbox_energy_ratio_final_epoch": bbox_energy,
            "bbox_energy_samples": samples,
            "bbox_energy_final_samples": samples,
            "pointing_game": pointing_game,
            "pointing_game_final_epoch": pointing_game,
            "saliency_iou": saliency_iou,
            "saliency_iou_final_epoch": saliency_iou,
            "detection_match_iou_mean": match_iou,
            "detection_match_iou_final_epoch": match_iou,
            "xai_metric_note": (
                "XAI metrics were read from the aggregate "
                f"{exported_summary.get('source_file')} artifact."
            ),
        }

    rows = exported_rows if exported_rows else read_odam_quality_files(run)
    final_epoch = int(training_summary.get("epochs", 0)) - 1
    final_rows = [
        row for row in rows
        if str(row.get("epoch", "")).isdigit()
        and int(row["epoch"]) == final_epoch
    ]
    if exported_rows:
        final_rows = rows

    energy_all = [to_float(row.get("dam_energy_in_gt")) for row in rows]
    energy_final = [to_float(row.get("dam_energy_in_gt")) for row in final_rows]
    match_iou_all = [to_float(row.get("iou")) for row in rows]
    match_iou_final = [to_float(row.get("iou")) for row in final_rows]
    pointing_game_all = [to_float(row.get("pointing_game_hit")) for row in rows]
    saliency_iou_all = [to_float(row.get("saliency_iou")) for row in rows]
    pointing_game_final = [
        to_float(row.get("pointing_game_hit")) for row in final_rows
    ]
    saliency_iou_final = [
        to_float(row.get("saliency_iou")) for row in final_rows
    ]

    return {
        "run": run.label,
        "run_dir": run.path.name,
        "xai_source": "export_xai_metrics" if exported_rows else "odam_quality_epoch",
        "bbox_energy_ratio_mean": mean(energy_all),
        "bbox_energy_ratio_median": median(energy_all),
        "bbox_energy_ratio_final_epoch": mean(energy_final),
        "bbox_energy_samples": len(finite_values(energy_all)),
        "bbox_energy_final_samples": len(finite_values(energy_final)),
        "pointing_game": mean(pointing_game_all),
        "pointing_game_final_epoch": mean(pointing_game_final),
        "saliency_iou": mean(saliency_iou_all),
        "saliency_iou_final_epoch": mean(saliency_iou_final),
        "detection_match_iou_mean": mean(match_iou_all),
        "detection_match_iou_final_epoch": mean(match_iou_final),
        "xai_metric_note": (
            "BBox Energy Ratio is computed from dam_energy_in_gt. "
            "Pointing Game and Saliency IoU are populated when "
            "export_xai_metrics.py artifacts are present."
        ),
    }


def clean_for_json(value):
    if isinstance(value, dict):
        return {k: clean_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_for_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_value(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.6g}"
    return str(value)


def markdown_table(rows: Sequence[Dict], columns: Sequence[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(format_value(row.get(col)) for col in columns)
            + " |"
        )
    return "\n".join(lines)


def write_report(
    path: Path,
    detection_rows: Sequence[Dict],
    gradient_rows: Sequence[Dict],
    xai_rows: Sequence[Dict],
    cost_rows: Sequence[Dict],
) -> None:
    text = [
        "# DPGA-ODAM Metric Report",
        "",
        "Generated from completed run artifacts under `outputs/`.",
        "",
        "Important notes:",
        "- `MR-2_Reasonable`, `MR-2_Heavy`, and `MR-2_Small` remain reserved for external official CityPersons evaluator CSV input.",
        "- `offline_best_MR-2_Reasonable`, `offline_best_MR-2_Heavy`, and `offline_best_MR-2_Small` are recomputed from saved predictions using CityPersons-style filters in the converted COCO annotations.",
        "- `MR-2_generic` is the internal generic log-average miss-rate diagnostic from `train.py`, not the official CityPersons protocol.",
        "- `BBox Energy Ratio` is computed from stored ODAM `dam_energy_in_gt` rows.",
        "- `Pointing Game`, `Saliency IoU`, peak GPU memory, FPS, and parameter count are marked `NA` unless the required artifacts are supplied/stored.",
        "- `final_gradient_norm` is marked `NA` because current gradient diagnostics store module-wise detection and ODAM norms, not the full composed gradient vector norm.",
        "",
        "## Detection",
        markdown_table(
            detection_rows,
            [
                "run",
                "method",
                "offline_best_MR-2_Reasonable",
                "offline_best_MR-2_Heavy",
                "offline_best_MR-2_Small",
                "offline_best_AP50",
                "offline_best_AP",
                "best_AP50",
                "best_AP",
                "best_MR-2_generic",
                "offline_final_MR-2_Reasonable",
                "offline_final_AP50",
                "final_AP50",
                "final_AP",
            ],
        ),
        "",
        "## Gradient",
        markdown_table(
            gradient_rows,
            [
                "run",
                "cosine_similarity_mean",
                "cosine_similarity_median",
                "gradient_conflict_rate",
                "gradient_norm_ratio_mean",
                "projection_rate",
                "norm_cap_rate",
                "final_aux_gradient_norm_mean",
            ],
        ),
        "",
        "## XAI / ODAM Quality",
        markdown_table(
            xai_rows,
            [
                "run",
                "bbox_energy_ratio_mean",
                "bbox_energy_ratio_final_epoch",
                "pointing_game",
                "saliency_iou",
                "detection_match_iou_mean",
            ],
        ),
        "",
        "## Computational Cost",
        markdown_table(
            cost_rows,
            [
                "run",
                "time_per_epoch_mean_s",
                "time_total_s",
                "peak_gpu_memory_gb",
                "fps",
                "num_parameters",
            ],
        ),
        "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def build_tables(args: argparse.Namespace) -> Dict[str, List[Dict]]:
    runs = discover_runs(args.outputs, args.runs)
    official_mr = read_citypersons_mr(args.citypersons_mr_csv)
    val_ann = getattr(args, "val_ann", None)
    annotation_data = load_annotation_data(val_ann)

    summary_rows = []
    detection_rows = []
    gradient_rows = []
    xai_rows = []
    cost_rows = []

    for run in runs:
        metric_rows = read_csv_dicts(run.metrics_path)
        training_summary = summarize_training_metrics(run, metric_rows)
        training_summary = with_citypersons_mr(training_summary, official_mr)
        offline_summary = summarize_offline_predictions(
            run,
            metric_rows,
            args,
            annotation_data,
        )
        training_summary.update(offline_summary)
        gradient_summary = summarize_gradient_rows(run)
        xai_summary = summarize_xai_rows(run, training_summary)

        summary_rows.append(
            {
                **training_summary,
                **{
                    k: v
                    for k, v in gradient_summary.items()
                    if k not in {"run", "run_dir"}
                },
                **{
                    k: v
                    for k, v in xai_summary.items()
                    if k not in {"run", "run_dir", "xai_metric_note"}
                },
            }
        )
        detection_rows.append(
            {
                key: training_summary.get(key)
                for key in [
                    "run",
                    "run_dir",
                    "method",
                    "experiment_stage",
                    "MR-2_Reasonable",
                    "MR-2_Heavy",
                    "MR-2_Small",
                    "citypersons_mr_source",
                    "offline_detection_source",
                    "offline_val_ann",
                    "offline_best_epoch",
                    "offline_final_epoch",
                    "offline_best_AP",
                    "offline_best_AP50",
                    "offline_best_AP75",
                    "offline_best_MR-2_generic",
                    "offline_best_MR-2_Reasonable",
                    "offline_best_MR-2_Heavy",
                    "offline_best_MR-2_Small",
                    "offline_final_AP",
                    "offline_final_AP50",
                    "offline_final_AP75",
                    "offline_final_MR-2_generic",
                    "offline_final_MR-2_Reasonable",
                    "offline_final_MR-2_Heavy",
                    "offline_final_MR-2_Small",
                    "best_AP50",
                    "best_AP50_epoch",
                    "final_AP50",
                    "best_AP",
                    "best_AP_epoch",
                    "final_AP",
                    "best_MR-2_generic",
                    "best_MR-2_generic_epoch",
                    "final_MR-2_generic",
                ]
            }
        )
        gradient_rows.append(gradient_summary)
        xai_rows.append(xai_summary)
        cost_rows.append(
            {
                key: training_summary.get(key)
                for key in [
                    "run",
                    "run_dir",
                    "method",
                    "time_per_epoch_mean_s",
                    "time_per_epoch_median_s",
                    "time_total_s",
                    "peak_gpu_memory_gb",
                    "inference_ms_per_image",
                    "fps",
                    "num_parameters",
                ]
            }
        )

    return {
        "summary": summary_rows,
        "detection": detection_rows,
        "gradient": gradient_rows,
        "xai": xai_rows,
        "cost": cost_rows,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tables = build_tables(args)
    write_csv(args.output_dir / "method_metrics_summary.csv", tables["summary"])
    write_csv(args.output_dir / "detection_metrics.csv", tables["detection"])
    write_csv(args.output_dir / "gradient_metrics.csv", tables["gradient"])
    write_csv(args.output_dir / "xai_metrics.csv", tables["xai"])
    write_csv(args.output_dir / "computational_cost_metrics.csv", tables["cost"])

    (args.output_dir / "method_metrics_summary.json").write_text(
        json.dumps(clean_for_json(tables["summary"]), indent=2),
        encoding="utf-8",
    )
    write_report(
        args.output_dir / "metric_report.md",
        tables["detection"],
        tables["gradient"],
        tables["xai"],
        tables["cost"],
    )

    print(f"[metrics] runs={len(tables['summary'])}")
    print(f"[metrics] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
