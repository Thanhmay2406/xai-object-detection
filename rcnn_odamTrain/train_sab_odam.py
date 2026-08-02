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
    "results/sab_odam_train",
    "--odam-nms",
    "--odam-nms-low-threshold",
    "0.2",
    "--odam-nms-high-threshold",
    "0.8",
    "--odam-nms-resize-short-edge",
    "50",
    "--odam-loss-start-epoch",
    "4",
    "--odam-loss-warmup-epochs",
    "5",
    "--sab-odam",
    "--sab-small-resolution",
    "28",
    "--sab-medium-resolution",
    "14",
    "--sab-large-resolution",
    "7",
    "--sab-topk-per-gt",
    "2",
    "--sab-max-rois-per-batch",
    "32",
    "--sab-lambda-match",
    "1.0",
    "--sab-lambda-scale",
    "0.1",
    "--sab-lambda-edge",
    "0.1",
    "--sab-lambda-inside",
    "0.05",
    "--sab-small-weight-gamma",
    "0.5",
    "--no-dp-odam",
    "--no-dpga-odam",
]


if __name__ == "__main__":
    run_method("sab_odam", DEFAULT_ARGS)
