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
    "results/rcnn_odam_train_odam_nms",
    "--odam-loss-weight",
    "1.0",
    "--odam-nms",
    "--odam-nms-low-threshold",
    "0.2",
    "--odam-nms-high-threshold",
    "0.8",
    "--odam-nms-resize-short-edge",
    "50",
    "--no-dp-odam",
    "--no-dpga-odam",
    "--no-sab-odam",
]


if __name__ == "__main__":
    run_method("odam_nms", DEFAULT_ARGS)
