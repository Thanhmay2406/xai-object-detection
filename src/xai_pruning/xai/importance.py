"""Existing Gradient x Activation channel-importance implementation."""

from __future__ import annotations

import gc
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from xai_pruning.config import TASK_TO_LOSS, XAIImportanceConfig

ImportanceKey = tuple[str, str, str]


def _get_module(model: nn.Module, name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"Module {name!r} not found")
    return modules[name]


def move_batch_to_device(batch, device: str | torch.device):
    """Move a torchvision detection batch without changing non-tensor metadata."""

    images, targets = batch
    images = [image.to(device) for image in images]
    moved_targets = []
    for target in targets:
        moved_targets.append(
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in target.items()
            }
        )
    return images, moved_targets


def dominant_scale(target: Mapping[str, Tensor]) -> str:
    """Return the existing image-level COCO size bucket based on median GT area."""

    boxes = target.get("boxes")
    if boxes is None or boxes.numel() == 0:
        return "empty"
    widths_heights = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    area = torch.median(widths_heights[:, 0] * widths_heights[:, 1]).item()
    if area < 32.0**2:
        return "small"
    if area < 96.0**2:
        return "medium"
    return "large"


class ActivationBank:
    """Store selected activations and retain their gradients during a forward pass."""

    def __init__(self, model: nn.Module, module_names: Sequence[str]):
        self.model = model
        self.module_names = list(module_names)
        self.activations: dict[str, Tensor] = {}
        self.handles = []

    def _hook(self, name: str):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                raise TypeError(f"XAI candidate {name!r} returned {type(output)}")
            self.activations[name] = output
            if output.requires_grad:
                output.retain_grad()

        return hook

    def __enter__(self):
        for name in self.module_names:
            self.handles.append(_get_module(self.model, name).register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear_grads(self) -> None:
        for activation in self.activations.values():
            activation.grad = None


def channel_xai_score(activation: Tensor, gradient: Tensor) -> Tensor:
    """Compute ``mean(|activation * gradient|)`` for each channel."""

    contribution = (activation * gradient).abs()
    if contribution.ndim == 4:
        return contribution.mean(dim=(0, 2, 3))
    if contribution.ndim == 2:
        return contribution.mean(dim=0)
    reduce_dims = tuple(index for index in range(contribution.ndim) if index != 1)
    return contribution.mean(dim=reduce_dims)


class XAIImportanceAccumulator:
    """Accumulate mean channel vectors under module/task/scale keys."""

    def __init__(self):
        self.sums: dict[ImportanceKey, Tensor] = {}
        self.counts: defaultdict[ImportanceKey, int] = defaultdict(int)

    def add(self, key: ImportanceKey, score: Tensor) -> None:
        score = score.detach().cpu()
        if key not in self.sums:
            self.sums[key] = torch.zeros_like(score)
        if self.sums[key].shape != score.shape:
            raise RuntimeError(f"Shape changed for {key}")
        self.sums[key] += score
        self.counts[key] += 1

    def mean_scores(self) -> dict[ImportanceKey, Tensor]:
        return {
            key: value / max(self.counts[key], 1) for key, value in self.sums.items()
        }


def estimate_xai_importance(
    model: nn.Module,
    probe_loader: DataLoader,
    config: XAIImportanceConfig,
    device: str | torch.device,
) -> dict[ImportanceKey, Tensor]:
    """Estimate task-aware channel importance from Faster R-CNN training losses."""

    model.train()
    accumulator = XAIImportanceAccumulator()
    with ActivationBank(model, config.candidate_modules) as bank:
        for batch_index, batch in enumerate(probe_loader):
            if batch_index >= config.max_probe_batches:
                break
            images, targets = move_batch_to_device(batch, device)
            if config.use_scale_buckets:
                scales = [dominant_scale(target) for target in targets]
                nonempty = [scale for scale in scales if scale != "empty"]
                scale = nonempty[0] if len(set(nonempty)) == 1 else "mixed"
            else:
                scale = "all"

            model.zero_grad(set_to_none=True)
            losses = model(images, targets)
            for task_index, task in enumerate(config.tasks):
                loss_name = TASK_TO_LOSS[task]
                if loss_name not in losses:
                    raise KeyError(f"Expected loss {loss_name!r}; found {list(losses)}")
                bank.clear_grads()
                model.zero_grad(set_to_none=True)
                losses[loss_name].backward(retain_graph=task_index < len(config.tasks) - 1)
                for module_name in config.candidate_modules:
                    activation = bank.activations[module_name]
                    gradient = activation.grad
                    score = (
                        torch.zeros(activation.shape[1], device=activation.device)
                        if gradient is None
                        else channel_xai_score(activation.detach(), gradient.detach())
                    )
                    accumulator.add((module_name, task, scale), score)
    return accumulator.mean_scores()


def normalize_vector(score: Tensor, eps: float = 1e-12) -> Tensor:
    """Apply the existing percentile-95 scaling and clamp to [0, 1]."""

    score = score.float()
    if score.numel() == 0:
        return score
    denominator = max(float(torch.quantile(score, 0.95)), eps)
    return (score / denominator).clamp(0.0, 1.0)


def aggregate_importance(
    raw_scores: Mapping[ImportanceKey, Tensor], config: XAIImportanceConfig
) -> dict[str, Tensor]:
    """Use the existing weighted arithmetic mean over task/scale vectors."""

    grouped = defaultdict(list)
    for (module_name, task, scale), score in raw_scores.items():
        if task not in config.tasks:
            continue
        vector = normalize_vector(score, config.eps) if config.normalize_scores else score.float()
        grouped[module_name].append((vector, float(config.task_weights.get(task, 1.0))))

    aggregated = {}
    for module_name, items in grouped.items():
        numerator = torch.zeros_like(items[0][0])
        denominator = 0.0
        for vector, weight in items:
            numerator += weight * vector
            denominator += weight
        aggregated[module_name] = numerator / max(denominator, config.eps)
    return aggregated


def get_best_same_class_matches(output, target) -> tuple[list[dict], dict]:
    """Match each GT to its highest-IoU prediction of the same model-space class."""

    gt_boxes = target["boxes"].to(output["boxes"].device)
    gt_labels = target["labels"].to(output["labels"].device)
    pred_boxes = output["boxes"]
    pred_labels = output["labels"]
    pred_scores = output["scores"]
    if gt_boxes.numel() == 0:
        return [], {
            "mode": "empty",
            "num_gt": 0,
            "num_predictions": len(pred_boxes),
            "num_gt_with_same_class_prediction": 0,
        }
    if pred_boxes.numel() == 0:
        return [], {
            "mode": "no_predictions",
            "num_gt": len(gt_boxes),
            "num_predictions": 0,
            "num_gt_with_same_class_prediction": 0,
        }
    ious = box_iou(gt_boxes, pred_boxes)
    matches = []
    for gt_index in range(len(gt_boxes)):
        same_class = torch.where(pred_labels == gt_labels[gt_index])[0]
        if same_class.numel() == 0:
            continue
        prediction_index = same_class[torch.argmax(ious[gt_index, same_class])]
        matches.append(
            {
                "gt_idx": gt_index,
                "pred_idx": int(prediction_index.item()),
                "iou": float(ious[gt_index, prediction_index].detach().item()),
                "score": float(pred_scores[prediction_index].detach().item()),
                "label": int(gt_labels[gt_index].detach().item()),
            }
        )
    return matches, {
        "mode": "best_same_class" if matches else "no_same_class_prediction",
        "num_gt": len(gt_boxes),
        "num_predictions": len(pred_boxes),
        "num_gt_with_same_class_prediction": len(matches),
    }


def select_gt_matched_target(
    output,
    target,
    min_iou: float,
    *,
    weight_by_iou: bool = True,
):
    """Build Pipeline 01's confidence scalar from unique retained predictions."""

    matches, metadata = get_best_same_class_matches(output, target)
    kept = [match for match in matches if match["iou"] >= min_iou]
    if not kept:
        reason = "no_same_class_prediction" if not matches else "below_iou_threshold"
        return None, {
            **metadata,
            "mode": "no_gt_match",
            "reason": reason,
            "min_iou": float(min_iou),
            "num_matches": 0,
            "matches": matches,
            "kept_matches": [],
            "matched_pred_indices": [],
        }
    prediction_to_iou = {}
    for match in kept:
        prediction_to_iou[match["pred_idx"]] = max(
            prediction_to_iou.get(match["pred_idx"], 0.0), match["iou"]
        )
    prediction_indices = sorted(prediction_to_iou)
    index = torch.tensor(prediction_indices, device=output["scores"].device, dtype=torch.long)
    selected_scores = output["scores"][index]
    if weight_by_iou:
        weights = torch.tensor(
            [prediction_to_iou[item] for item in prediction_indices],
            device=selected_scores.device,
            dtype=selected_scores.dtype,
        )
        scalar = (selected_scores * weights).sum() / weights.sum().clamp_min(1e-12)
    else:
        scalar = selected_scores.mean()
    return scalar, {
        **metadata,
        "mode": "gt_matched",
        "min_iou": float(min_iou),
        "num_matches": len(prediction_indices),
        "matches": matches,
        "kept_matches": kept,
        "matched_pred_indices": prediction_indices,
        "mean_kept_iou": float(sum(match["iou"] for match in kept) / len(kept)),
    }


def select_empty_false_positive_target(output):
    """Use the top surviving score as the separate empty-image FP target."""

    scores = output["scores"]
    if scores.numel() == 0:
        return None, {"mode": "empty_no_prediction"}
    return scores[0], {
        "mode": "empty_top_prediction",
        "score": float(scores[0].detach().item()),
        "label": int(output["labels"][0].detach().item()),
    }


def select_gt_matched_multitask_target(
    output,
    target,
    min_iou: float,
    image_hw,
    *,
    weight_by_iou: bool = True,
    confidence_weight: float = 0.70,
    localization_weight: float = 0.30,
    enable_localization: bool = True,
):
    """Build the unchanged confidence plus normalized-L1 localization target."""

    confidence, metadata = select_gt_matched_target(
        output, target, min_iou, weight_by_iou=weight_by_iou
    )
    if confidence is None:
        return None, metadata
    if not enable_localization or localization_weight <= 0:
        return confidence, {**metadata, "target_mode": "confidence_only"}
    height, width = (int(value) for value in image_hw)
    scale = torch.tensor(
        [width, height, width, height],
        device=output["boxes"].device,
        dtype=output["boxes"].dtype,
    ).clamp_min(1.0)
    gt_boxes = target["boxes"].to(output["boxes"].device)
    localization_terms = []
    localization_weights = []
    for match in metadata.get("kept_matches", []):
        normalized_l1 = (
            (output["boxes"][match["pred_idx"]] - gt_boxes[match["gt_idx"]]).abs() / scale
        ).mean()
        localization_terms.append(torch.exp(-normalized_l1))
        localization_weights.append(match["iou"] if weight_by_iou else 1.0)
    if not localization_terms:
        return confidence, {**metadata, "target_mode": "confidence_only_no_loc_pairs"}
    terms = torch.stack(localization_terms)
    weights = torch.tensor(
        localization_weights, device=terms.device, dtype=terms.dtype
    )
    localization = (terms * weights).sum() / weights.sum().clamp_min(1e-12)
    total_weight = max(float(confidence_weight + localization_weight), 1e-12)
    composite = (
        float(confidence_weight) * confidence + float(localization_weight) * localization
    ) / total_weight
    return composite, {
        **metadata,
        "target_mode": "confidence_plus_localization",
        "confidence_target": float(confidence.detach().item()),
        "localization_target": float(localization.detach().item()),
        "confidence_weight": float(confidence_weight),
        "localization_weight": float(localization_weight),
    }


def multi_group_gradient_x_activation(
    activation_storage: Mapping[str, Tensor], group_order: Sequence[str], scalar_score: Tensor
) -> dict[str, Tensor]:
    """Calculate all bottleneck channel vectors in one autograd call."""

    missing = [group_id for group_id in group_order if group_id not in activation_storage]
    if missing:
        raise RuntimeError(f"Missing hooked activations for groups: {missing}")
    activations = [activation_storage[group_id] for group_id in group_order]
    gradients = torch.autograd.grad(
        scalar_score,
        activations,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    result = {}
    for group_id, activation, gradient in zip(group_order, activations, gradients):
        if gradient is None:
            raise RuntimeError(f"Autograd returned None gradient for group={group_id}")
        result[group_id] = (activation * gradient).abs().mean(dim=(0, 2, 3)).detach().cpu()
    return result


def aggregate_multi_group_importance(
    model,
    probe_loader,
    pruning_groups,
    device,
    min_iou: float,
    *,
    weight_by_iou: bool = True,
    analyze_empty: bool = True,
    empty_fp_weight: float = 0.10,
    confidence_weight: float = 0.70,
    localization_weight: float = 0.30,
    enable_localization: bool = True,
    log_every: int = 20,
) -> dict:
    """Aggregate Pipeline 03 object importance plus weighted empty-image importance."""

    model.eval()
    activation_storage = {}
    group_order = [group["group_id"] for group in pruning_groups]
    handles = [
        group["producer_module"].register_forward_hook(
            lambda _module, _inputs, output, group_id=group["group_id"]: activation_storage.__setitem__(group_id, output)
        )
        for group in pruning_groups
    ]
    object_sums = {group_id: None for group_id in group_order}
    empty_sums = {group_id: None for group_id in group_order}
    per_image_lists = {group_id: [] for group_id in group_order}
    used_image_ids = []
    stats = Counter()
    total_match_iou = 0.0
    total_match_count = 0
    try:
        for batch_index, (images, targets) in enumerate(probe_loader, start=1):
            if len(images) != 1:
                raise ValueError("Multi-group aggregator expects batch_size=1")
            target = targets[0]
            image = images[0].to(device, non_blocking=True)
            is_empty = len(target["boxes"]) == 0
            if is_empty and not analyze_empty:
                stats["empty_images_skipped"] += 1
                continue
            model.zero_grad(set_to_none=True)
            activation_storage.clear()
            output = model([image])[0]
            if is_empty:
                stats["empty_images"] += 1
                scalar, fp_info = select_empty_false_positive_target(output)
                if scalar is None:
                    stats["empty_no_prediction"] += 1
                    continue
                importance = multi_group_gradient_x_activation(
                    activation_storage, group_order, scalar
                )
                for group_id in group_order:
                    if empty_sums[group_id] is None:
                        empty_sums[group_id] = torch.zeros_like(importance[group_id])
                    empty_sums[group_id] += importance[group_id]
                stats["empty_used"] += 1
                stats[f"empty_fp_label_{fp_info['label']}"] += 1
                continue

            stats["object_images"] += 1
            scalar, metadata = select_gt_matched_multitask_target(
                output,
                target,
                min_iou,
                image.shape[-2:],
                weight_by_iou=weight_by_iou,
                confidence_weight=confidence_weight,
                localization_weight=localization_weight,
                enable_localization=enable_localization,
            )
            if scalar is None:
                stats["no_gt_match"] += 1
                stats["skip_" + metadata.get("reason", "unknown")] += 1
                continue
            importance = multi_group_gradient_x_activation(activation_storage, group_order, scalar)
            for group_id in group_order:
                if object_sums[group_id] is None:
                    object_sums[group_id] = torch.zeros_like(importance[group_id])
                object_sums[group_id] += importance[group_id]
                per_image_lists[group_id].append(importance[group_id])
            used_image_ids.append(int(target["image_id"].item()))
            stats["object_used"] += 1
            stats["matched_detections"] += metadata["num_matches"]
            if metadata.get("target_mode") == "confidence_plus_localization":
                stats["localization_target_images"] += 1
            for match in metadata.get("kept_matches", []):
                total_match_iou += match["iou"]
                total_match_count += 1
            if log_every and batch_index % log_every == 0:
                print(
                    f"[multi-group {batch_index:>3}/{len(probe_loader)}] "
                    f"object_used={stats['object_used']} | empty_used={stats['empty_used']} | "
                    f"no_match={stats['no_gt_match']}"
                )

        if stats["object_used"] == 0:
            raise RuntimeError("No Probe object image contributed to multi-group importance")
        object_importance = {}
        empty_importance = {}
        combined = {}
        per_image = {}
        for group_id in group_order:
            if object_sums[group_id] is None:
                raise RuntimeError(f"No object importance accumulated for group={group_id}")
            object_importance[group_id] = object_sums[group_id] / stats["object_used"]
            empty_importance[group_id] = (
                empty_sums[group_id] / stats["empty_used"]
                if stats["empty_used"] > 0 and empty_sums[group_id] is not None
                else torch.zeros_like(object_importance[group_id])
            )
            combined[group_id] = object_importance[group_id] + float(empty_fp_weight) * empty_importance[group_id]
            per_image[group_id] = torch.stack(per_image_lists[group_id], dim=0)
        stats.update(
            {
                "object_image_coverage": stats["object_used"] / max(stats["object_images"], 1),
                "empty_image_coverage": stats["empty_used"] / max(stats["empty_images"], 1),
                "selected_iou_threshold": float(min_iou),
                "mean_kept_gt_iou": total_match_iou / max(total_match_count, 1),
                "empty_fp_importance_weight": float(empty_fp_weight),
                "confidence_target_weight": float(confidence_weight),
                "localization_target_weight": float(localization_weight),
            }
        )
        return {
            "group_order": group_order,
            "aggregate_importance": combined,
            "object_importance": object_importance,
            "empty_fp_importance": empty_importance,
            "per_image_importance": per_image,
            "image_ids": used_image_ids,
            "stats": dict(stats),
        }
    finally:
        for handle in handles:
            handle.remove()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
