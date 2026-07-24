from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO

from odam_yolo.config import load_odam_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--odam-config", default="configs/odam_yolov8_p2.yaml")
    args = parser.parse_args()

    os.environ["ODAM_CONFIG_PATH"] = str(Path(args.odam_config).resolve())
    cfg = load_odam_config()
    wrapper = YOLO(args.model)
    head = wrapper.model.model[-1]
    strides = tuple(int(round(float(x))) for x in head.stride.detach().cpu().tolist())

    print(f"model: {args.model}")
    print(f"head: {type(head).__name__}")
    print(f"levels: {head.nl}")
    print(f"strides: {strides}")
    print(f"classes: {head.nc}")
    print(f"reg_max: {head.reg_max}")
    print(f"expected levels: {cfg.expected_num_levels}")
    print(f"expected strides: {cfg.expected_strides}")

    if cfg.strict_p2 and (head.nl != cfg.expected_num_levels or strides != cfg.expected_strides):
        raise SystemExit("FAIL: checkpoint/model is not the expected YOLOv8-P2 contract")
    print("PASS: YOLOv8-P2 head contract")


if __name__ == "__main__":
    main()
