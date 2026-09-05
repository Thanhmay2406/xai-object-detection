"""Per-class AP extraction from one canonical COCOeval result."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def mean_valid(values) -> float | None:
    """Average COCO precision entries while excluding sentinel value -1."""

    values = np.asarray(values)
    values = values[values > -1]
    return float(values.mean()) if values.size else None


def per_class_metrics(coco_eval, coco_gt) -> list[dict]:
    """Extract AP, AP50, AP75, and GT count for every evaluated category."""

    precision = coco_eval.eval["precision"]
    iou_thresholds = coco_eval.params.iouThrs
    ap50_index = int(np.argmin(np.abs(iou_thresholds - 0.50)))
    ap75_index = int(np.argmin(np.abs(iou_thresholds - 0.75)))
    rows = []
    for class_index, category_id in enumerate(coco_eval.params.catIds):
        category_id = int(category_id)
        rows.append(
            {
                "category_id": category_id,
                "class_name": coco_gt.cats[category_id]["name"],
                "gt_objects": sum(
                    int(annotation["category_id"]) == category_id
                    for annotation in coco_gt.dataset["annotations"]
                ),
                "AP": mean_valid(precision[:, :, class_index, 0, -1]),
                "AP50": mean_valid(precision[ap50_index, :, class_index, 0, -1]),
                "AP75": mean_valid(precision[ap75_index, :, class_index, 0, -1]),
            }
        )
    return rows


def evaluate_per_class(
    annotation_json: str | Path,
    prediction_json,
    *,
    category_ids=None,
    quiet: bool = True,
) -> list[dict]:
    """Run COCOeval on an existing prediction JSON and return per-class metrics."""

    from xai_pruning.evaluation.coco_eval import evaluate_predictions

    _, rows, _ = evaluate_predictions(
        annotation_json, prediction_json, category_ids=category_ids, quiet=quiet
    )
    return rows
