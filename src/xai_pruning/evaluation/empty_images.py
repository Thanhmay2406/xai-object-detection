"""False-positive summaries for images with no ground-truth objects."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from xai_pruning.utils.io import load_json


def analyze_empty_images(annotation_json: str | Path, predictions) -> dict:
    """Summarize prediction count/confidence on empty images from saved predictions."""

    ground_truth = load_json(annotation_json)
    if isinstance(predictions, (str, Path)):
        predictions = load_json(predictions)
    annotations_by_image = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    empty_ids = {
        int(image["id"])
        for image in ground_truth["images"]
        if not annotations_by_image[int(image["id"])]
    }
    empty_predictions = [
        prediction for prediction in predictions if int(prediction["image_id"]) in empty_ids
    ]
    detected_ids = {int(prediction["image_id"]) for prediction in empty_predictions}
    scores = [float(prediction["score"]) for prediction in empty_predictions]
    counts = Counter(int(prediction["category_id"]) for prediction in empty_predictions)
    return {
        "empty_images": len(empty_ids),
        "empty_images_with_detection": len(detected_ids),
        "empty_image_detection_rate": len(detected_ids) / max(len(empty_ids), 1),
        "empty_detection_count": len(empty_predictions),
        "empty_mean_confidence": float(np.mean(scores)) if scores else 0.0,
        "empty_max_confidence": float(np.max(scores)) if scores else 0.0,
        "empty_fp_class_counts_coco_space": {
            str(key): int(value) for key, value in sorted(counts.items())
        },
    }


def empty_fp_rows(model_name: str, metrics: dict) -> list[dict]:
    """Convert the evaluator's empty-image COCO class counts to table rows."""

    return [
        {
            "model": model_name,
            "category_id": int(category_id),
            "false_positives": int(count),
        }
        for category_id, count in metrics.get(
            "empty_fp_class_counts_coco_space", {}
        ).items()
    ]
