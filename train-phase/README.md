# YOLOv8-P2 + ODAM-Train

A first-order adaptation of the released ODAM-Train objective to an Ultralytics
YOLOv8-P2 detector. The integration uses the detector's existing
TaskAlignedAssigner, captures the P2/P3/P4/P5 tensors immediately before the
Detect head, generates an instance-specific gradient map for selected positive
predictions, and adds a pair-discrimination BCE term to the standard
box/classification/DFL loss vector.

## Reproducible environment

This package is validated in this workspace with **Ultralytics 8.4.75**. Do not silently upgrade the
Ultralytics package because the internal trainer, Detect head, and loss APIs are
not a stable public extension interface.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Kaggle, PyTorch is already installed, so usually this is enough:

```bash
pip install -q ultralytics==8.4.75 PyYAML
```

## 1. Verify the P2 checkpoint

```bash
python scripts/inspect_model.py \
  --model /path/to/experiments/baseline/train/weights/best.pt \
  --odam-config configs/odam_yolov8_p2.yaml
```

Expected contract:

```text
levels: 4
strides: (4, 8, 16, 32)
PASS: YOLOv8-P2 head contract
```

If your P2 YAML uses different strides, edit `expected_strides`. Do not disable
`strict_p2` until you have inspected the actual head.

## 2. Run a gradient-connectivity smoke train

```bash
bash scripts/smoke_train.sh \
  /path/to/experiments/baseline/train/weights/best.pt \
  /path/to/data.yaml
```

Use `amp=false`, one GPU, a tiny fraction, and a small batch for the first run.
The training table should contain four loss components, including `odam_loss`.
A permanently zero value can be valid for individual batches with no eligible
pairs, but it should not remain zero across a dataset containing multiple
positive predictions per object or overlapping objects.

The smoke run writes live ODAM diagnostics into the Ultralytics run directory:

```text
runs/odam_smoke/gradient_connectivity/odam_live.log
runs/odam_smoke/gradient_connectivity/odam_batches.csv
runs/odam_smoke/gradient_connectivity/odam_batches.jsonl
runs/odam_smoke/gradient_connectivity/odam_epochs.jsonl
```

## 3. Full single-GPU run

```bash
python train_odam.py \
  --model /path/to/experiments/baseline/train/weights/best.pt \
  --data /path/to/data.yaml \
  --odam-config configs/odam_yolov8_p2.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --amp true \
  --project experiments/odam \
  --name odam_seed0 \
  --seed 0 \
  --log-every 1 \
  --log-detail-batches 3 \
  --heartbeat-seconds 20
```

## 4. Two-GPU run

After the one-GPU smoke test succeeds:

```bash
python train_odam.py \
  --model /path/to/experiments/baseline/train/weights/best.pt \
  --data /path/to/data.yaml \
  --odam-config configs/odam_yolov8_p2.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0,1 \
  --workers 4 \
  --amp true \
  --project experiments/odam \
  --name odam_seed0_ddp \
  --seed 0 \
  --log-every 1 \
  --log-detail-batches 3 \
  --heartbeat-seconds 20
```

`batch=16` is the global batch size handled by Ultralytics; with two GPUs it is
normally divided between the two ranks.

Only rank 0 writes the live diagnostics in DDP, so the four ODAM log files are
not duplicated across workers. Kaggle often buffers or hides worker stdout, so
`train_odam.py` mirrors rank-zero `odam_live.log` back to the parent process by
default when `--device` contains multiple GPUs. Disable this with
`--tail-live-log false` if the parent-process tail is not wanted.

## Live logging

The ODAM integration adds flush-on-write logging without changing the detector,
matching, loss formula, optimizer, or training control flow. Batch summaries are
printed to stdout and appended to `odam_live.log`, `odam_batches.csv`, and
`odam_batches.jsonl`. Epoch aggregates are appended to `odam_epochs.jsonl`.

Each batch line includes epoch, batch index, `box_loss`, `cls_loss`, `dfl_loss`,
`odam_loss`, total loss, learning rate, GPU memory, batch time, throughput,
foreground anchors, selected predictions, generated CAM count, positive pairs,
negative pairs, raw ODAM loss, weighted ODAM loss, and skip reason.

The first `--log-detail-batches` batches of each epoch also emit per-image
foreground/selection lines and CAM start/done progress. If ODAM CAM generation
runs longer than `--heartbeat-seconds`, rank 0 prints `odam_heartbeat` lines so
Kaggle notebooks do not look stalled.

Useful controls:

```bash
--log-every 1              # write every batch summary
--log-detail-batches 3     # verbose image/CAM detail for first 3 batches/epoch
--heartbeat-seconds 20     # heartbeat while a slow ODAM batch is running
--tail-live-log true       # mirror rank-zero file logs to stdout in DDP
```

## Loss definition

The total training objective is:

```text
L = L_box + L_cls + L_dfl + lambda_odam * L_odam
```

For each image:

1. TaskAlignedAssigner supplies foreground anchors and assigned GT identities.
2. Positive predictions are sorted by predicted-box/assigned-GT IoU.
3. At most `max_samples_per_object` and `max_samples_per_image` are retained.
4. For each prediction, the configured target score is differentiated with
   respect to its corresponding P-level feature map. The default target is
   `sigmoid(class_logit)`, matching the paper's confidence-score target more
   closely than a raw logit for YOLOv8.
5. The gradient map is locally smoothed with a grouped Gaussian `Phi`, then the
   ODAM vector is `ReLU(sum_c(feature_c * Phi(gradient_c)))`, resized and L2
   normalized.
6. The highest-IoU prediction of each GT is the reference.
7. Same-GT maps are pushed toward cosine similarity 1.
8. Different-GT maps are pushed toward similarity 0 only when their predicted
   boxes overlap.

The default `second_order: false` detaches the gradient term. This matches the
released author's Odam-Train implementation and avoids a costly Hessian-vector
backward. Setting `second_order: true` is experimental and changes the method.

The logger writes to both stdout and files. It now emits `batch_start` before
preprocessing enters the model and `odam_start` before CAM generation, so long
ODAM batches no longer look silent while `autograd.grad` is running.

## Important limitations

- This is an adaptation to YOLOv8, not an official author implementation for
  Ultralytics.
- The code depends on Ultralytics internals and is intentionally version-pinned.
- ODAM generation is expensive because it performs one `autograd.grad` call per
  selected prediction. Reduce `max_samples_per_image` or increase
  `every_n_batches` if VRAM/time is excessive.
- Validation runs under disabled gradients, so `val/odam_loss` is zero by
  design. Scientific comparison should use detection metrics and separately
  computed ODAM quality metrics.
- Calibrate `lambda_odam`; `0.5` is only a safe starting value, not a paper- or
  dataset-certified optimum for your drill-bit dataset.
