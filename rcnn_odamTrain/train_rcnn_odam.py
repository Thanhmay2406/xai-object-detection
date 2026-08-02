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
    "results/rcnn_odam_train",
    "--odam-loss-weight",
    "1.0",
    "--no-odam-nms",
    "--no-dp-odam",
    "--no-dpga-odam",
    "--no-sab-odam",
]


if __name__ == "__main__":
    run_method("rcnn_odam", DEFAULT_ARGS)
