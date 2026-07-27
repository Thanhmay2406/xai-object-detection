# xai-object-detection

This repository contains a Faster R-CNN baseline and an RCNN-ODAM training
entrypoint for COCO-format drill-bit defect detection experiments.

## ODAM-Train

The ODAM-Train implementation is in:

- `rcnn_odamTrain/train.py`
- `rcnn_odamTrain/network.py`
- `det_oprs/fpn_roi_target.py`
- `backbone/resnet50.py`

It trains a two-stage detector with an auxiliary ODAM object-discrimination
loss. The implementation keeps standard detector losses, computes ODAM maps
from positive predicted proposals, excludes GT-appended ROIs from the ODAM
auxiliary loss by default, filters ODAM pairs by image id, and applies RCNN
post-processing before COCO evaluation.
RPN training samples a balanced mini-batch of anchors per image by default,
instead of letting the many negative anchors dominate the objectness loss.
ODAM-NMS can be enabled at evaluation/inference time to use both bounding-box
IoU and ODAM heatmap correlation when removing duplicate detections.

### Install

Use the project virtual environment if it already exists:

```bash
./.venv/bin/python -m pip install -r requirements-odam.txt
```

If you need a fresh environment:

```bash
python -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements-odam.txt
```

Install the PyTorch build that matches your CUDA/runtime if the default pip
resolver is not appropriate for your machine.

### Data Layout

The default dataset path is `data/drill_bit_coco`:

```text
data/drill_bit_coco/
  train/_annotations.coco.json
  valid/_annotations.coco.json
  test/_annotations.coco.json
```

Image files must live beside each split annotation file.

### Run RCNN-ODAM

Single-GPU/local command:

```bash
./.venv/bin/python rcnn_odamTrain/train.py \
  --data-root data/drill_bit_coco \
  --output-dir results/rcnn_odam_train_fixed \
  --overwrite \
  --backbone-weights default \
  --epochs 20 \
  --batch-size 4 \
  --workers 2 \
  --image-size 640 \
  --device cuda:0 \
  --amp \
  --include-empty-categories \
  --rpn-batch-size 256 \
  --rpn-fg-fraction 0.5 \
  --odam-nms \
  --odam-nms-low-threshold 0.2 \
  --odam-nms-high-threshold 0.8 \
  --odam-nms-resize-short-edge 50 \
  --odam-loss-start-epoch 4 \
  --odam-loss-warmup-epochs 5 \
  --test-after-train \
  --test-checkpoint best
```

Two-GPU DDP command:

```bash
torchrun --standalone --nproc_per_node=2 rcnn_odamTrain/train.py \
  --data-root data/drill_bit_coco \
  --output-dir results/rcnn_odam_train_fixed \
  --overwrite \
  --backbone-weights default \
  --epochs 20 \
  --batch-size 4 \
  --workers 2 \
  --image-size 640 \
  --amp \
  --include-empty-categories \
  --rpn-batch-size 256 \
  --rpn-fg-fraction 0.5 \
  --odam-nms \
  --odam-nms-low-threshold 0.2 \
  --odam-nms-high-threshold 0.8 \
  --odam-nms-resize-short-edge 50 \
  --odam-loss-start-epoch 4 \
  --odam-loss-warmup-epochs 5 \
  --test-after-train \
  --test-checkpoint best
```

`--backbone-weights default` uses torchvision's default ResNet50 ImageNet
weights. Use `--backbone-weights none` when running fully offline without a
pretrained-weight cache, and run the baseline from scratch as well if you need a
fair scratch-vs-scratch comparison.
`--include-empty-categories` keeps the saved ODAM label mapping aligned with the
baseline when the COCO file contains categories with no annotations.
`--odam-nms-low-threshold 0.2` and `--odam-nms-high-threshold 0.8` follow the
paper's default ODAM-Train setting; disable with `--no-odam-nms` for classical
IoU-only NMS ablations.
`--odam-nms-resize-short-edge 50` matches the paper's heatmap-correlation
preprocessing.
`--odam-loss-start-epoch 4 --odam-loss-warmup-epochs 5` keeps the auxiliary
ODAM loss off for epochs 1-3, then linearly ramps it to `--odam-loss-weight`
from epochs 4-8 so the detector losses can stabilize first.

### Run SAB-ODAM Improved Architecture

`SAB-ODAM` implements the training-time improvement described in
`SAB_ODAM_kien_truc_cai_tien.md`: the Faster R-CNN detection branch still uses
the standard `7x7` RoI head, while the auxiliary explanation branch uses
scale-adaptive multi-level FPN RoIAlign, attribute-specific scale fusion, cross
scale consistency, class-inside-object loss, and boundary-aware losses for
`x1/y1/x2/y2` heatmaps.

Two-GPU scratch command for fair comparison against a scratch Faster R-CNN
baseline:

```bash
torchrun --standalone --nproc_per_node=2 rcnn_odamTrain/train.py \
  --data-root data/drill_bit_coco \
  --output-dir results/sab_odam_train_scratch_ddp_20e \
  --overwrite \
  --backbone-weights none \
  --epochs 20 \
  --batch-size 4 \
  --workers 2 \
  --image-size 640 \
  --amp \
  --include-empty-categories \
  --rpn-batch-size 256 \
  --rpn-fg-fraction 0.5 \
  --odam-nms \
  --odam-nms-low-threshold 0.2 \
  --odam-nms-high-threshold 0.8 \
  --odam-nms-resize-short-edge 50 \
  --odam-loss-start-epoch 4 \
  --odam-loss-warmup-epochs 5 \
  --sab-odam \
  --sab-small-resolution 28 \
  --sab-medium-resolution 14 \
  --sab-large-resolution 7 \
  --sab-topk-per-gt 2 \
  --sab-max-rois-per-batch 32 \
  --sab-lambda-match 1.0 \
  --sab-lambda-scale 0.1 \
  --sab-lambda-edge 0.1 \
  --sab-lambda-inside 0.05 \
  --sab-small-weight-gamma 0.5 \
  --test-after-train \
  --test-checkpoint best
```

The SAB metrics are written as separate columns in `metrics.csv`:
`loss_sab_scale`, `loss_sab_edge`, and `loss_sab_inside`; the class heatmap
consistency/separation component is reported through `loss_rcnn_match`.

### Fair Baseline Command

```bash
./.venv/bin/python baseline/train_faster_rcnn.py \
  --data-root data/drill_bit_coco \
  --output-dir results/baseline/faster_rcnn_imagenet \
  --overwrite \
  --weights imagenet \
  --epochs 30 \
  --batch-size 4 \
  --workers 2 \
  --min-size 640 \
  --max-size 640 \
  --device cuda:0 \
  --amp \
  --test-after-train \
  --test-checkpoint best
```

`--weights imagenet` initializes only the ResNet50 backbone from ImageNet and
keeps the RPN/RoI detector heads randomly initialized. This is the appropriate
baseline setting when comparing against RCNN-ODAM or SAB-ODAM runs that use
`--backbone-weights default`. Use `--weights coco` only as a COCO-pretrained
detector upper-bound, not as a fair detector-training baseline.

For two-GPU DDP, launch the same script with:

```bash
torchrun --standalone --nproc_per_node=2 baseline/train_faster_rcnn.py ...
```

## Outputs

Training writes checkpoints and metrics under the selected `--output-dir`:

- `metrics.csv`
- `best.pt`
- `last.pt`
- `config.json`
- `test_metrics.json` when `--test-after-train` is enabled

Generated datasets, checkpoints, metrics, notebook figures, and PDF papers are
not required source files and should not be committed unless intentionally
publishing a frozen artifact bundle.
