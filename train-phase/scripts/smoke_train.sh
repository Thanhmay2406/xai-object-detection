#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: scripts/smoke_train.sh /path/to/best.pt /path/to/data.yaml}"
DATA="${2:?Usage: scripts/smoke_train.sh /path/to/best.pt /path/to/data.yaml}"

python train_odam.py \
  --model "$MODEL" \
  --data "$DATA" \
  --odam-config configs/odam_yolov8_p2.yaml \
  --epochs 1 \
  --imgsz 640 \
  --batch 2 \
  --device 0 \
  --workers 0 \
  --amp false \
  --fraction 0.02 \
  --log-every 1 \
  --log-detail-batches 3 \
  --heartbeat-seconds 20 \
  --project runs/odam_smoke \
  --name gradient_connectivity
