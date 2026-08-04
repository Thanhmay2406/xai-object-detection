# DPGA-ODAM Evidence Protocol

This protocol turns the original ODAM paper's argument into runnable evidence for the repository's DPGA-ODAM checkpoint.

The original paper's claim is not only detector mAP. The complete argument is:

```text
instance-specific ODAM heatmaps
-> same-object consistency and different-object separation
-> less leakage to neighboring objects
-> useful duplicate reasoning in crowded post-processing
```

DPGA-ODAM keeps the Odam-Train explanation objective and adds detection-priority gradient composition. To claim it satisfies the original paper's research argument, report all evidence groups below.

## Evidence Matrix

| Evidence group | Question | Repository artifact |
|---|---|---|
| Detection quality | Does DPGA-ODAM preserve/improve detector performance? | `metrics.csv`, `test_metrics.json` |
| Object-discrimination heatmaps | Are same-object heatmaps more similar than different-object heatmaps? | `evaluate_odam_evidence.py` |
| Explanation localization | Does the heatmap focus on the assigned object instead of neighbors? | `evaluate_odam_evidence.py` |
| ODAM-NMS utility | Do heatmap correlations help duplicate removal at inference? | `evaluate_threshold_sweep.py --odam-nms` |
| Gradient safety | Does DPGA prevent unsafe ODAM gradient injection? | `stat_dpga_*` columns in `metrics.csv` |

## 1. Detection Evidence

Use the already produced result files:

- `results/city-persons/dpga_odam_citypersons_imagenet/metrics.csv`
- `results/city-persons/dpga_odam_citypersons_imagenet/test_metrics.json`

Current CityPersons COCO-style result:

```text
DPGA-ODAM best epoch: 30
mAP50:95: 0.285186
mAP50:    0.585013
mAP75:    0.244105
AR100:    0.394837
```

This is currently the best ODAM-family result, but it is still below the Faster R-CNN baseline on mAP50:95 and mAP75.

## 2. ODAM Explanation Evidence

Run this for DPGA-ODAM:

```bash
python /kaggle/working/xai-object-detection/rcnn_odamTrain/evaluate_odam_evidence.py \
  --data-root /kaggle/input/datasets/thanhmay2406/dataset-for-research/coco \
  --checkpoint /kaggle/working/results/dpga_odam_citypersons_imagenet/best.pt \
  --output-dir /kaggle/working/results/dpga_odam_citypersons_imagenet/odam_evidence \
  --split valid \
  --device cuda:0 \
  --workers 2 \
  --image-size 640 \
  --score-threshold 0.001 \
  --pred-cls-threshold 0.001 \
  --rcnn-nms-threshold 0.95 \
  --detections-per-image 300 \
  --match-iou-threshold 0.5 \
  --crowd-iou-threshold 0.1 \
  --overwrite
```

Run the same evidence evaluator on comparison checkpoints:

```bash
python /kaggle/working/xai-object-detection/rcnn_odamTrain/evaluate_odam_evidence.py \
  --data-root /kaggle/input/datasets/thanhmay2406/dataset-for-research/coco \
  --checkpoint /kaggle/working/results/rcnn_odam_citypersons_imagenet/best.pt \
  --output-dir /kaggle/working/results/rcnn_odam_citypersons_imagenet/odam_evidence \
  --split valid \
  --device cuda:0 \
  --workers 2 \
  --image-size 640 \
  --score-threshold 0.001 \
  --pred-cls-threshold 0.001 \
  --rcnn-nms-threshold 0.95 \
  --detections-per-image 300 \
  --match-iou-threshold 0.5 \
  --crowd-iou-threshold 0.1 \
  --overwrite
```

```bash
python /kaggle/working/xai-object-detection/rcnn_odamTrain/evaluate_odam_evidence.py \
  --data-root /kaggle/input/datasets/thanhmay2406/dataset-for-research/coco \
  --checkpoint /kaggle/working/results/dp_odam_citypersons_imagenet/best.pt \
  --output-dir /kaggle/working/results/dp_odam_citypersons_imagenet/odam_evidence \
  --split valid \
  --device cuda:0 \
  --workers 2 \
  --image-size 640 \
  --score-threshold 0.001 \
  --pred-cls-threshold 0.001 \
  --rcnn-nms-threshold 0.95 \
  --detections-per-image 300 \
  --match-iou-threshold 0.5 \
  --crowd-iou-threshold 0.1 \
  --overwrite
```

The evaluator writes:

- `summary.json`
- `per_image_metrics.csv`
- `report.md`

Primary fields to compare:

| Metric | Desired direction | Interpretation |
|---|---:|---|
| `same_object_cosine_mean` | higher | Odam-Train consistency |
| `different_object_cosine_mean` | lower | Odam-Train separation |
| `discrimination_margin` | higher | same-object minus different-object similarity |
| `pair_auc` | higher | heatmap similarity ranks same-object pairs above different-object pairs |
| `target_energy_ratio_mean` | higher | heatmap focuses on assigned object |
| `other_object_energy_ratio_mean` | lower | less leakage to neighboring objects |
| `pointing_box_accuracy` | higher | heatmap peak lies inside assigned object box |
| `box_proxy_vea_iou_mean` | higher | box-based proxy for explanation localization |

Use high `--rcnn-nms-threshold 0.95` and high `--detections-per-image 300` here intentionally. This keeps duplicate and overlapping predictions so same-object/different-object pairs can be measured.

## 3. ODAM-NMS Evidence

Evaluate DPGA-ODAM with classical NMS:

```bash
python /kaggle/working/xai-object-detection/rcnn_odamTrain/evaluate_threshold_sweep.py \
  --data-root /kaggle/input/datasets/thanhmay2406/dataset-for-research/coco \
  --checkpoint /kaggle/working/results/dpga_odam_citypersons_imagenet/best.pt \
  --output-dir /kaggle/working/results/dpga_odam_citypersons_imagenet/classical_nms_sweep \
  --split valid \
  --device cuda:0 \
  --workers 2 \
  --image-size 640 \
  --score-thresholds 0.001 \
  --pred-cls-thresholds 0.03 0.05 0.08 \
  --rcnn-nms-thresholds 0.45 0.50 0.55 \
  --detections-per-image 50 75 100 \
  --no-odam-nms \
  --max-combinations 64 \
  --overwrite
```

Evaluate DPGA-ODAM with ODAM-NMS:

```bash
python /kaggle/working/xai-object-detection/rcnn_odamTrain/evaluate_threshold_sweep.py \
  --data-root /kaggle/input/datasets/thanhmay2406/dataset-for-research/coco \
  --checkpoint /kaggle/working/results/dpga_odam_citypersons_imagenet/best.pt \
  --output-dir /kaggle/working/results/dpga_odam_citypersons_imagenet/odam_nms_sweep \
  --split valid \
  --device cuda:0 \
  --workers 2 \
  --image-size 640 \
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
```

The original paper reports CrowdHuman AP50/JI/MR/Recall. This repository's sweep reports COCO-style `map_50_95`, `map50`, `map75`, AP by scale, AR, GT count, and prediction count. Treat this as repository-aligned ODAM-NMS evidence, not an exact CrowdHuman protocol reproduction.

## 4. Gradient-Safety Evidence

For DPGA-ODAM, report these `metrics.csv` columns:

- `stat_dpga_any_active`
- `stat_dpga_detection_only_fallback`
- `stat_dpga_mean_cosine`
- `stat_dpga_mean_gate`
- `stat_dpga_mean_norm_scale`
- `stat_dpga_mean_effective_scale`
- `stat_dpga_roi_shared_valid`
- `stat_dpga_roi_classifier_valid`
- `stat_dpga_missing_detection_grad`

Current DPGA run summary:

```text
active DPGA epochs: 10-25
recovery epochs: 26-30
missing detection gradients: 0
active ROI shared/classifier valid: 1
full-active mean effective scale: about 0.03
```

This supports the DPGA-specific claim: ODAM gradients are not injected directly; they are conservatively gated and norm-capped under detection-priority constraints.

## Claim Template

Use this only after the evidence scripts have been run on DPGA-ODAM and at least one comparison checkpoint:

```text
DPGA-ODAM preserves the original Odam-Train objective of learning instance-specific, object-discriminative explanations, while adding detection-priority gradient composition. In the CityPersons COCO-style protocol, DPGA-ODAM is the strongest ODAM-family detector. Its ODAM evidence shows [higher/lower] same-object consistency, [higher/lower] different-object separation, [higher/lower] explanation leakage, and [better/worse] ODAM-NMS utility compared with [baseline method]. Therefore, DPGA-ODAM [does/does not] support the original paper's explanation-to-object-discrimination argument under this repository protocol.
```

Do not claim full reproduction of the paper unless CrowdHuman AP50/JI/MR/Recall and the paper's explanation/user-study metrics are also reproduced.
