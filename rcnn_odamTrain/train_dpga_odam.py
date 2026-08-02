#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rcnn_odamTrain.method_runner import run_method


DEFAULT_ARGS = [
    "--output-dir",
    "results/dpga_odam_train",
    "--no-amp",
    "--odam-nms",
    "--odam-nms-low-threshold",
    "0.2",
    "--odam-nms-high-threshold",
    "0.8",
    "--odam-nms-resize-short-edge",
    "50",
    "--odam-loss-start-epoch",
    "10",
    "--odam-loss-warmup-epochs",
    "5",
    "--dpga-odam",
    "--dpga-module-coverage",
    "roi-head-only",
    "--dpga-roi-shared-norm-ratio",
    "0.2",
    "--dpga-roi-classifier-norm-ratio",
    "0.2",
    "--dp-odam-min-iou",
    "0.5",
    "--dp-odam-min-confidence",
    "0.5",
    "--dp-odam-topk-per-gt",
    "2",
    "--dp-odam-max-rois-per-batch",
    "32",
    "--dp-odam-negative-iou-threshold",
    "0.1",
    "--dp-odam-recovery-epochs",
    "5",
    "--no-sab-odam",
]


if __name__ == "__main__":
    run_method("dpga_odam", DEFAULT_ARGS)
