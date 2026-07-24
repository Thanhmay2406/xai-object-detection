# YOLOv8-P2 + ODAM-Train

A first-order adaptation of the released ODAM-Train objective to an Ultralytics
YOLOv8-P2 detector. The integration uses the detector's existing
TaskAlignedAssigner, captures the P2/P3/P4/P5 tensors immediately before the
Detect head, generates an instance-specific gradient map for selected positive
predictions, and adds a pair-discrimination BCE term to the standard
box/classification/DFL loss vector.

## Reproducible environment

This package targets **Ultralytics 8.3.245**. Do not silently upgrade the
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
pip install -q ultralytics==8.3.245 PyYAML
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
  --seed 0
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
  --seed 0
```

`batch=16` is the global batch size handled by Ultralytics; with two GPUs it is
normally divided between the two ranks.

## Loss definition

The total training objective is:

```text
L = L_box + L_cls + L_dfl + lambda_odam * L_odam
```

For each image:

1. TaskAlignedAssigner supplies foreground anchors and assigned GT identities.
2. Positive predictions are sorted by predicted-box/assigned-GT IoU.
3. At most `max_samples_per_object` and `max_samples_per_image` are retained.
4. For each prediction, the raw class logit is differentiated with respect to
   its corresponding P-level feature map.
5. The ODAM vector is `ReLU(sum_c(feature_c * gradient_c))`, resized and L2
   normalized.
6. The highest-IoU prediction of each GT is the reference.
7. Same-GT maps are pushed toward cosine similarity 1.
8. Different-GT maps are pushed toward similarity 0 only when their predicted
   boxes overlap.

The default `second_order: false` detaches the gradient term. This matches the
released author's Odam-Train implementation and avoids a costly Hessian-vector
backward. Setting `second_order: true` is experimental and changes the method.

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
