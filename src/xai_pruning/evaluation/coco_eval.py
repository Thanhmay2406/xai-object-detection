"""Single COCOeval protocol shared by Pipeline 03, 04, 05, and later work."""

from __future__ import annotations

import contextlib
import io
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from xai_pruning.config import MODEL_TO_COCO_LABEL
from xai_pruning.evaluation.per_class import per_class_metrics
from xai_pruning.pruning.structural import count_parameters
from xai_pruning.utils.io import load_json, save_json


@dataclass
class EvaluationResult:
    """COCO metrics plus optional predictions and per-image audit rows."""

    metrics: dict[str, Any]
    per_class: list[dict]
    predictions: list[dict] | None
    per_image: list[dict]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "per_class": self.per_class,
            "predictions": self.predictions,
            "per_image": self.per_image,
        }


def _annotation_category_ids(coco_gt: COCO) -> list[int]:
    return sorted({int(annotation["category_id"]) for annotation in coco_gt.dataset["annotations"]})


def evaluate_predictions(
    annotation_json: str | Path,
    predictions,
    *,
    category_ids=None,
    quiet: bool = False,
) -> tuple[dict, list[dict], COCOeval]:
    """Run the unchanged bbox COCOeval protocol on in-memory or saved predictions."""

    if isinstance(predictions, (str, Path)):
        predictions = load_json(predictions)
    if not predictions:
        raise RuntimeError("COCO evaluation requires at least one mapped prediction")
    output_context = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with output_context:
        coco_gt = COCO(str(annotation_json))
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        selected_category_ids = (
            _annotation_category_ids(coco_gt) if category_ids is None else category_ids
        )
        coco_eval.params.catIds = [int(category_id) for category_id in selected_category_ids]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    stats = coco_eval.stats
    metrics = {
        "map": float(stats[0]),
        "ap50": float(stats[1]),
        "ap75": float(stats[2]),
        "ap_small": float(stats[3]),
        "ap_medium": float(stats[4]),
        "ap_large": float(stats[5]),
        "ar1": float(stats[6]),
        "ar10": float(stats[7]),
        "ar100": float(stats[8]),
    }
    return metrics, per_class_metrics(coco_eval, coco_gt), coco_eval


def write_predictions_json(predictions: list[dict], path: str | Path) -> Path:
    """Write already-computed COCO predictions without rerunning inference."""

    return save_json(predictions, path)


def evaluate_detector(
    model,
    data_loader,
    annotation_json: str | Path,
    device: str | torch.device,
    *,
    model_name: str = "model",
    split: str = "test",
    model_to_coco_label=None,
    category_ids=None,
    log_every: int = 100,
    return_predictions: bool = True,
    prediction_json: str | Path | None = None,
    quiet_coco: bool = False,
) -> EvaluationResult:
    """Run detector inference once, map labels, evaluate, and retain audit counts.

    No additional confidence threshold is applied. This preserves Pipeline 04's
    use of the detections returned by torchvision after its configured post-NMS
    processing.
    """

    resolved_device = torch.device(device)
    label_mapping = {
        int(key): int(value)
        for key, value in (model_to_coco_label or MODEL_TO_COCO_LABEL).items()
    }
    model.to(resolved_device).eval()
    predictions = []
    image_rows = []
    empty_scores = []
    empty_fp_model_counts = Counter()
    empty_fp_coco_counts = Counter()
    empty_images = 0
    empty_images_with_detection = 0
    empty_detection_count = 0
    skipped_unmapped_predictions = 0
    raw_prediction_count = 0
    seen_images = 0

    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    started = time.perf_counter()

    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(data_loader, start=1):
            outputs = model(
                [image.to(resolved_device, non_blocking=True) for image in images]
            )
            for output, target in zip(outputs, targets):
                image_id = int(target["image_id"].item())
                boxes = output["boxes"].detach().cpu()
                scores = output["scores"].detach().cpu()
                labels = output["labels"].detach().cpu()
                raw_prediction_count += len(scores)
                is_empty = int(target["boxes"].shape[0]) == 0
                mapped_count = 0
                per_image_counts = Counter()

                if is_empty:
                    empty_images += 1
                    if len(scores):
                        empty_images_with_detection += 1
                        empty_detection_count += len(scores)
                        empty_scores.extend(float(score) for score in scores.tolist())

                for box, score, model_label_tensor in zip(boxes, scores, labels):
                    model_label = int(model_label_tensor.item())
                    if model_label not in label_mapping:
                        skipped_unmapped_predictions += 1
                        continue
                    coco_label = label_mapping[model_label]
                    x1, y1, x2, y2 = (float(value) for value in box.tolist())
                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": coco_label,
                            "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                            "score": float(score.item()),
                        }
                    )
                    mapped_count += 1
                    per_image_counts[coco_label] += 1
                    if is_empty:
                        empty_fp_model_counts[model_label] += 1
                        empty_fp_coco_counts[coco_label] += 1

                image_rows.append(
                    {
                        "model": model_name,
                        "image_id": image_id,
                        "is_empty": bool(is_empty),
                        "num_gt": int(target["boxes"].shape[0]),
                        "num_predictions": mapped_count,
                        "max_confidence": float(scores.max().item()) if len(scores) else 0.0,
                        "mean_confidence": float(scores.mean().item()) if len(scores) else 0.0,
                        "prediction_counts_coco_json": json.dumps(
                            {str(key): int(value) for key, value in sorted(per_image_counts.items())}
                        ),
                    }
                )
                seen_images += 1
            if log_every and batch_index % log_every == 0:
                print(
                    f"[{model_name} {split.upper()}] batches={batch_index}/{len(data_loader)} "
                    f"images={seen_images}"
                )

    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    elapsed_seconds = time.perf_counter() - started
    coco_metrics, class_rows, _ = evaluate_predictions(
        annotation_json,
        predictions,
        category_ids=category_ids,
        quiet=quiet_coco,
    )
    if prediction_json is not None:
        write_predictions_json(predictions, prediction_json)

    dataset_size = len(data_loader.dataset)
    metrics = {
        "model": model_name,
        "split": split,
        **coco_metrics,
        "num_images": dataset_size,
        "raw_prediction_count": int(raw_prediction_count),
        "mapped_prediction_count": len(predictions),
        "num_predictions": len(predictions),
        "skipped_unmapped_prediction_count": int(skipped_unmapped_predictions),
        "skipped_unmapped_predictions": int(skipped_unmapped_predictions),
        "empty_images": int(empty_images),
        "empty_images_with_detection": int(empty_images_with_detection),
        "empty_image_detection_rate": empty_images_with_detection / max(empty_images, 1),
        "empty_detection_count": int(empty_detection_count),
        "empty_mean_confidence": float(np.mean(empty_scores)) if empty_scores else 0.0,
        "empty_max_confidence": float(np.max(empty_scores)) if empty_scores else 0.0,
        "empty_fp_class_counts_model_space": {
            str(key): int(value) for key, value in sorted(empty_fp_model_counts.items())
        },
        "empty_fp_class_counts_coco_space": {
            str(key): int(value) for key, value in sorted(empty_fp_coco_counts.items())
        },
        "parameter_count": count_parameters(model),
        "evaluation_wall_time_seconds": elapsed_seconds,
        "evaluation_wall_time_ms_per_image": 1000.0 * elapsed_seconds / max(dataset_size, 1),
    }
    return EvaluationResult(
        metrics=metrics,
        per_class=class_rows,
        predictions=predictions if return_predictions else None,
        per_image=image_rows,
    )
