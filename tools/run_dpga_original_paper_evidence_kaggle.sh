#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/kaggle/working/xai-object-detection}"
DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/thanhmay2406/dataset-for-research/coco}"
RESULT_ROOT="${RESULT_ROOT:-/kaggle/working/xai-object-detection/results/city-persons}"
DEVICE="${DEVICE:-cuda:0}"
WORKERS="${WORKERS:-2}"
IMAGE_SIZE="${IMAGE_SIZE:-640}"

cd "${REPO_ROOT}"

run_evidence() {
  local run_name="$1"
  local checkpoint="${RESULT_ROOT}/${run_name}/best.pt"
  local output_dir="${RESULT_ROOT}/${run_name}/odam_evidence"

  python "${REPO_ROOT}/rcnn_odamTrain/evaluate_odam_evidence.py" \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${checkpoint}" \
    --output-dir "${output_dir}" \
    --split valid \
    --device "${DEVICE}" \
    --workers "${WORKERS}" \
    --image-size "${IMAGE_SIZE}" \
    --score-threshold 0.001 \
    --pred-cls-threshold 0.001 \
    --rcnn-nms-threshold 0.95 \
    --detections-per-image 300 \
    --match-iou-threshold 0.5 \
    --crowd-iou-threshold 0.1 \
    --overwrite
}

run_evidence "dpga_odam_citypersons_imagenet"
run_evidence "rcnn_odam_citypersons_imagenet"
run_evidence "dp_odam_citypersons_imagenet"

python "${REPO_ROOT}/rcnn_odamTrain/evaluate_threshold_sweep.py" \
  --data-root "${DATA_ROOT}" \
  --checkpoint "${RESULT_ROOT}/dpga_odam_citypersons_imagenet/best.pt" \
  --output-dir "${RESULT_ROOT}/dpga_odam_citypersons_imagenet/classical_nms_sweep" \
  --split valid \
  --device "${DEVICE}" \
  --workers "${WORKERS}" \
  --image-size "${IMAGE_SIZE}" \
  --score-thresholds 0.001 \
  --pred-cls-thresholds 0.03 0.05 0.08 \
  --rcnn-nms-thresholds 0.45 0.50 0.55 \
  --detections-per-image 50 75 100 \
  --no-odam-nms \
  --max-combinations 64 \
  --overwrite

python "${REPO_ROOT}/rcnn_odamTrain/evaluate_threshold_sweep.py" \
  --data-root "${DATA_ROOT}" \
  --checkpoint "${RESULT_ROOT}/dpga_odam_citypersons_imagenet/best.pt" \
  --output-dir "${RESULT_ROOT}/dpga_odam_citypersons_imagenet/odam_nms_sweep" \
  --split valid \
  --device "${DEVICE}" \
  --workers "${WORKERS}" \
  --image-size "${IMAGE_SIZE}" \
  --score-thresholds 0.001 \
  --pred-cls-thresholds 0.03 0.05 0.08 \
  --rcnn-nms-thresholds 0.45 0.50 0.55 \
  --detections-per-image 50 75 100 \
  --odam-nms \
  --odam-nms-low-thresholds 0.1 0.2 0.3 \
  --odam-nms-high-thresholds 0.7 0.8 0.9 \
  --odam-nms-resize-short-edges 50 \
  --max-combinations 243 \
  --overwrite

python "${REPO_ROOT}/tools/summarize_dpga_original_paper_evidence.py" \
  --results-root "${RESULT_ROOT}" \
  --output "${RESULT_ROOT}/dpga_original_paper_evidence_summary.md"
