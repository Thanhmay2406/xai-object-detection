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
  --test-after-train \
  --test-checkpoint best
```

`--backbone-weights default` uses torchvision's default ResNet50 ImageNet
weights. Use `--backbone-weights none` when running fully offline without a
pretrained-weight cache, and run the baseline from scratch as well if you need a
fair scratch-vs-scratch comparison.

### Fair Baseline Command

```bash
./.venv/bin/python baseline/train_faster_rcnn.py \
  --data-root data/drill_bit_coco \
  --output-dir results/baseline/faster_rcnn_fixed \
  --overwrite \
  --weights coco \
  --epochs 20 \
  --batch-size 4 \
  --workers 2 \
  --min-size 640 \
  --max-size 640 \
  --device cuda:0 \
  --amp \
  --test-after-train \
  --test-checkpoint best
```

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
