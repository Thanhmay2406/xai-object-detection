import os
import inspect
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-odam-train")

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from ultralytics.engine.trainer import BaseTrainer

from .config import OdamConfig
from .live_logging import get_live_logger


FEATURE_ATTR = "_odam_input_features"
_TRAINER_EXPECTS_DICT_LOSS_ITEMS: bool | None = None


def trainer_expects_dict_loss_items() -> bool:
    global _TRAINER_EXPECTS_DICT_LOSS_ITEMS
    if _TRAINER_EXPECTS_DICT_LOSS_ITEMS is None:
        source = inspect.getsource(BaseTrainer._do_train)
        _TRAINER_EXPECTS_DICT_LOSS_ITEMS = (
            "self.loss_items.items()" in source or "self.tloss.values()" in source
        )
    return bool(_TRAINER_EXPECTS_DICT_LOSS_ITEMS)


def get_captured_features(head: torch.nn.Module) -> tuple[torch.Tensor, ...] | None:
    features = getattr(head, FEATURE_ATTR, None)
    if features is None:
        return None
    return tuple(features)


@dataclass
class OdamBatchStats:
    raw_loss: float = 0.0
    weighted_loss: float = 0.0
    foreground_anchors: int = 0
    selected_predictions: int = 0
    cam_count: int = 0
    positive_pairs: int = 0
    negative_pairs: int = 0
    skipped: bool = False
    skip_reason: str = ""


class OdamDetectionLoss(v8DetectionLoss):
    """Ultralytics YOLOv8 detection loss plus ODAM pair discrimination loss.

    The detection part intentionally mirrors ``v8DetectionLoss`` from the
    local Ultralytics runtime. The additional ODAM term uses TaskAlignedAssigner's
    foreground assignments, chooses high-IoU predictions, generates one
    instance-specific map per selected prediction, and applies the BCE pair
    objective released by the ODAM authors.
    """

    def __init__(self, model: torch.nn.Module, odam_cfg: OdamConfig):
        super().__init__(model)
        self.head = model.model[-1]
        self.odam_cfg = odam_cfg
        self.call_index = 0
        self.last_stats = OdamBatchStats()
        self._validate_head_contract()

    def _validate_head_contract(self) -> None:
        head = self.head
        strides = tuple(int(round(float(x))) for x in head.stride.detach().cpu().tolist())
        if self.odam_cfg.strict_p2:
            if int(head.nl) != self.odam_cfg.expected_num_levels:
                raise RuntimeError(
                    "strict_p2=True but the detection head has "
                    f"{head.nl} levels; expected {self.odam_cfg.expected_num_levels}."
                )
            if strides != self.odam_cfg.expected_strides:
                raise RuntimeError(
                    "strict_p2=True but detector strides are "
                    f"{strides}; expected {self.odam_cfg.expected_strides}."
                )

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # box, cls, dfl, weighted_odam
        loss = torch.zeros(4, device=self.device)
        parsed = self.parse_output(preds) if hasattr(self, "parse_output") else preds
        if isinstance(parsed, dict):
            feats = parsed["feats"]
            pred_distri = parsed["boxes"].permute(0, 2, 1).contiguous()
            pred_scores = parsed["scores"].permute(0, 2, 1).contiguous()
        else:
            feats = parsed[1] if isinstance(parsed, tuple) else parsed
            if not isinstance(feats, (list, tuple)):
                raise TypeError(f"Expected training feature outputs, got {type(feats)!r}")
            pred_distri, pred_scores = torch.cat(
                [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
            ).split((self.reg_max * 4, self.nc), 1)
            pred_scores = pred_scores.permute(0, 2, 1).contiguous()
            pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(
            targets,
            batch_size,
            scale_tensor=imgsz[[1, 0, 1, 0]],
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = target_scores.sum().clamp_min(1.0)

        # Standard YOLOv8 detection losses.
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        class_weights = getattr(self, "class_weights", None)
        if class_weights is not None:
            bce_loss *= class_weights
        loss[1] = bce_loss.sum() / target_scores_sum
        if fg_mask.sum():
            try:
                loss[0], loss[2] = self.bbox_loss(
                    pred_distri,
                    pred_bboxes,
                    anchor_points,
                    target_bboxes / stride_tensor,
                    target_scores,
                    target_scores_sum,
                    fg_mask,
                    imgsz,
                    stride_tensor,
                )
            except TypeError:
                loss[0], loss[2] = self.bbox_loss(
                    pred_distri,
                    pred_bboxes,
                    anchor_points,
                    target_bboxes / stride_tensor,
                    target_scores,
                    target_scores_sum,
                    fg_mask,
                )
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        self.call_index += 1
        odam_weight = self._current_odam_weight()
        should_run, skip_reason = self._should_run_odam(odam_weight)
        if should_run:
            raw_odam, stats = self._compute_odam_loss(
                feats=feats,
                pred_scores=pred_scores,
                pred_bboxes_pixels=pred_bboxes * stride_tensor,
                target_labels=target_labels,
                target_bboxes_pixels=target_bboxes,
                fg_mask=fg_mask,
                target_gt_idx=target_gt_idx,
            )
            loss[3] = raw_odam * odam_weight
            stats.raw_loss = float(raw_odam.detach())
            stats.weighted_loss = float(loss[3].detach())
            self.last_stats = stats
        else:
            self.last_stats = OdamBatchStats(
                foreground_anchors=int(fg_mask.sum().detach().item()),
                skipped=True,
                skip_reason=skip_reason,
            )

        # Multiplication by batch size mirrors v8DetectionLoss. Ultralytics
        # changed loss_items from a tensor to a dict in newer 8.4.x trainers,
        # so return the shape the local trainer expects at runtime.
        return loss * batch_size, self._format_loss_items(loss)

    def _current_odam_weight(self) -> float:
        cfg = self.odam_cfg
        epoch = int(getattr(self.head, "_odam_current_epoch", 0))
        if epoch < cfg.start_epoch:
            return 0.0
        if cfg.warmup_epochs <= 0:
            return float(cfg.lambda_odam)
        progress = (epoch - cfg.start_epoch + 1) / float(cfg.warmup_epochs)
        return float(cfg.lambda_odam) * max(0.0, min(1.0, progress))

    @staticmethod
    def _format_loss_items(loss: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        detached = loss.detach()
        if not trainer_expects_dict_loss_items():
            return detached
        return {
            "box_loss": detached[0],
            "cls_loss": detached[1],
            "dfl_loss": detached[2],
            "odam_loss": detached[3],
        }

    def _should_run_odam(self, weight: float) -> tuple[bool, str]:
        if not self.odam_cfg.enabled:
            return False, "disabled"
        if weight <= 0:
            return False, "zero_weight"
        # Validation is normally inside inference_mode/no_grad. ODAM requires
        # autograd, so val/odam_loss is intentionally reported as zero.
        if not torch.is_grad_enabled():
            return False, "grad_disabled"
        if self.call_index % self.odam_cfg.every_n_batches != 0:
            return False, "frequency_gate"
        head = self.head
        if get_captured_features(head) is None:
            return False, "missing_feature_tap"
        return True, ""

    def _compute_odam_loss(
        self,
        feats: list[torch.Tensor] | tuple[torch.Tensor, ...],
        pred_scores: torch.Tensor,
        pred_bboxes_pixels: torch.Tensor,
        target_labels: torch.Tensor,
        target_bboxes_pixels: torch.Tensor,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, OdamBatchStats]:
        cfg = self.odam_cfg
        head = self.head

        # TaskAlignedAssigner in Ultralytics 8.3.245 returns:
        #   target_labels: [B, A]
        #   fg_mask:       [B, A]
        #   target_gt_idx: [B, A]
        # Some nearby Ultralytics/custom variants may retain a trailing
        # singleton class dimension, so normalize that case explicitly.
        if target_labels.ndim == 3 and target_labels.shape[-1] == 1:
            target_labels = target_labels.squeeze(-1)

        expected_assignment_shape = pred_scores.shape[:2]
        if target_labels.ndim != 2 or tuple(target_labels.shape) != tuple(expected_assignment_shape):
            raise RuntimeError(
                "Unexpected target_labels shape: "
                f"got {tuple(target_labels.shape)}, expected {tuple(expected_assignment_shape)} "
                "(or the same shape with a trailing singleton dimension)."
            )
        if fg_mask.ndim != 2 or tuple(fg_mask.shape) != tuple(expected_assignment_shape):
            raise RuntimeError(
                f"Unexpected fg_mask shape: got {tuple(fg_mask.shape)}, "
                f"expected {tuple(expected_assignment_shape)}."
            )
        if target_gt_idx.ndim != 2 or tuple(target_gt_idx.shape) != tuple(expected_assignment_shape):
            raise RuntimeError(
                f"Unexpected target_gt_idx shape: got {tuple(target_gt_idx.shape)}, "
                f"expected {tuple(expected_assignment_shape)}."
            )
        if target_bboxes_pixels.ndim != 3 or target_bboxes_pixels.shape[-1] != 4:
            raise RuntimeError(
                "Unexpected target_bboxes_pixels shape: "
                f"got {tuple(target_bboxes_pixels.shape)}, expected [B, A, 4]."
            )
        if pred_bboxes_pixels.ndim != 3 or pred_bboxes_pixels.shape[-1] != 4:
            raise RuntimeError(
                "Unexpected pred_bboxes_pixels shape: "
                f"got {tuple(pred_bboxes_pixels.shape)}, expected [B, A, 4]."
            )

        captured = get_captured_features(head)
        if captured is None:
            zero = pred_scores.sum() * 0.0
            return zero, OdamBatchStats(skipped=True, skip_reason="missing_feature_tap")
        if len(captured) != len(feats):
            raise RuntimeError(
                f"Captured {len(captured)} feature levels but loss received {len(feats)} outputs"
            )

        level_starts: list[int] = []
        level_ends: list[int] = []
        start = 0
        for level_output in feats:
            count = int(level_output.shape[-2] * level_output.shape[-1])
            level_starts.append(start)
            start += count
            level_ends.append(start)

        image_losses: list[torch.Tensor] = []
        total_foreground = int(fg_mask.sum().detach().item())
        total_selected = 0
        total_cams = 0
        total_pos_pairs = 0
        total_neg_pairs = 0
        logger = get_live_logger()
        epoch = int(getattr(head, "_odam_current_epoch", 0))
        batch_index = int(getattr(head, "_odam_current_batch_index", -1))
        detail_enabled = logger is not None and logger.detail_enabled(batch_index)
        if logger is not None:
            logger.start_odam(epoch, batch_index)

        for batch_id in range(pred_scores.shape[0]):
            pos_anchor_idx = torch.nonzero(fg_mask[batch_id], as_tuple=False).flatten()
            if logger is not None:
                logger.heartbeat(
                    epoch,
                    batch_index,
                    f"image={batch_id} stage=foreground anchors={int(pos_anchor_idx.numel())}",
                )
            if pos_anchor_idx.numel() == 0:
                if detail_enabled:
                    logger.detail(epoch, batch_index, batch_id, "skip=no_foreground_anchors")
                continue

            pred_boxes = pred_bboxes_pixels[batch_id, pos_anchor_idx]
            assigned_boxes = target_bboxes_pixels[batch_id, pos_anchor_idx]
            assignment_iou = aligned_box_iou(pred_boxes.detach(), assigned_boxes.detach())
            object_ids = target_gt_idx[batch_id, pos_anchor_idx].long()
            class_ids = target_labels[batch_id, pos_anchor_idx].long()
            class_ids = class_ids.clamp(min=0, max=self.nc - 1)

            keep = assignment_iou >= cfg.min_assignment_iou
            pos_anchor_idx = pos_anchor_idx[keep]
            pred_boxes = pred_boxes[keep]
            assignment_iou = assignment_iou[keep]
            object_ids = object_ids[keep]
            class_ids = class_ids[keep]
            if pos_anchor_idx.numel() == 0:
                if detail_enabled:
                    logger.detail(epoch, batch_index, batch_id, "skip=no_anchor_after_iou_gate")
                continue

            selected = self._select_predictions(object_ids, assignment_iou)
            pos_anchor_idx = pos_anchor_idx[selected]
            pred_boxes = pred_boxes[selected]
            assignment_iou = assignment_iou[selected]
            object_ids = object_ids[selected]
            class_ids = class_ids[selected]
            total_selected += int(pos_anchor_idx.numel())
            if detail_enabled:
                logger.detail(
                    epoch,
                    batch_index,
                    batch_id,
                    "selected "
                    f"foreground={int(fg_mask[batch_id].sum().detach().item())} "
                    f"after_iou={int(keep.sum().detach().item())} "
                    f"predictions={int(pos_anchor_idx.numel())}",
                )

            cams: list[torch.Tensor] = []
            valid_rows: list[int] = []
            for row, (anchor_idx_t, class_idx_t) in enumerate(zip(pos_anchor_idx, class_ids)):
                anchor_idx = int(anchor_idx_t.item())
                class_idx = int(class_idx_t.item())
                level = self._anchor_level(anchor_idx, level_starts, level_ends)
                if detail_enabled:
                    logger.detail(
                        epoch,
                        batch_index,
                        batch_id,
                        "cam_start "
                        f"cam={row + 1}/{int(pos_anchor_idx.numel())} "
                        f"level={level} anchor={anchor_idx} class={class_idx}",
                    )
                if logger is not None:
                    logger.heartbeat(
                        epoch,
                        batch_index,
                        "stage=cam "
                        f"image={batch_id} cam={row + 1}/{int(pos_anchor_idx.numel())} "
                        f"level={level}",
                    )
                feature = captured[level]
                score = pred_scores[batch_id, anchor_idx, class_idx]
                target = score.sigmoid() if cfg.target_activation == "sigmoid" else score

                grad = torch.autograd.grad(
                    outputs=target,
                    inputs=feature,
                    retain_graph=True,
                    create_graph=cfg.second_order,
                    allow_unused=True,
                )[0]
                if grad is None:
                    if detail_enabled:
                        logger.detail(
                            epoch,
                            batch_index,
                            batch_id,
                            f"cam_skip cam={row + 1}/{int(pos_anchor_idx.numel())} reason=grad_none",
                        )
                    continue
                grad_for_cam = grad[batch_id] if cfg.second_order else grad[batch_id].detach()
                grad_for_cam = self._smooth_gradient(grad_for_cam, cfg.smoothing_kernel)
                cam = (feature[batch_id].float() * grad_for_cam.float()).sum(dim=0)
                cam = F.relu(cam)
                cam = F.interpolate(
                    cam[None, None],
                    size=(cfg.map_height, cfg.map_width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                cam = cam.flatten()
                cam = cam / cam.norm(p=2).clamp_min(cfg.eps)
                cams.append(cam)
                valid_rows.append(row)
                total_cams += 1
                if detail_enabled:
                    logger.detail(
                        epoch,
                        batch_index,
                        batch_id,
                        f"cam_done cam={row + 1}/{int(pos_anchor_idx.numel())}",
                    )

            if not cams:
                if detail_enabled:
                    logger.detail(epoch, batch_index, batch_id, "skip=no_valid_cams")
                continue

            valid = torch.tensor(valid_rows, device=object_ids.device, dtype=torch.long)
            cam_tensor = torch.stack(cams, dim=0)
            object_ids = object_ids[valid]
            pred_boxes = pred_boxes[valid]
            assignment_iou = assignment_iou[valid]

            pair_loss, pos_count, neg_count = self._pair_discrimination_loss(
                cams=cam_tensor,
                object_ids=object_ids,
                pred_boxes=pred_boxes.detach(),
                assignment_iou=assignment_iou,
            )
            if pos_count + neg_count > 0:
                image_losses.append(pair_loss)
                total_pos_pairs += pos_count
                total_neg_pairs += neg_count
                if detail_enabled:
                    logger.detail(
                        epoch,
                        batch_index,
                        batch_id,
                        f"pairs positive={pos_count} negative={neg_count}",
                    )
            elif detail_enabled:
                logger.detail(epoch, batch_index, batch_id, "skip=no_positive_or_negative_pairs")

        if not image_losses:
            # Keep a valid graph-connected zero for all batches where ODAM has
            # no eligible pair, rather than returning a detached scalar.
            zero = sum(feature.sum() * 0.0 for feature in captured)
            return zero, OdamBatchStats(
                foreground_anchors=total_foreground,
                selected_predictions=total_selected,
                cam_count=total_cams,
                positive_pairs=total_pos_pairs,
                negative_pairs=total_neg_pairs,
                skipped=True,
                skip_reason="no_eligible_pairs",
            )

        raw_loss = torch.stack(image_losses).mean()
        return raw_loss, OdamBatchStats(
            foreground_anchors=total_foreground,
            selected_predictions=total_selected,
            cam_count=total_cams,
            positive_pairs=total_pos_pairs,
            negative_pairs=total_neg_pairs,
        )

    def _select_predictions(self, object_ids: torch.Tensor, ious: torch.Tensor) -> torch.Tensor:
        """Global high-IoU ordering with a per-object cap."""

        order = torch.argsort(ious, descending=True)
        counts: dict[int, int] = {}
        selected: list[int] = []
        for idx_t in order:
            idx = int(idx_t.item())
            object_id = int(object_ids[idx].item())
            if counts.get(object_id, 0) >= self.odam_cfg.max_samples_per_object:
                continue
            counts[object_id] = counts.get(object_id, 0) + 1
            selected.append(idx)
            if len(selected) >= self.odam_cfg.max_samples_per_image:
                break
        return torch.tensor(selected, device=object_ids.device, dtype=torch.long)

    @staticmethod
    def _anchor_level(anchor_idx: int, starts: list[int], ends: list[int]) -> int:
        for level, (start, end) in enumerate(zip(starts, ends)):
            if start <= anchor_idx < end:
                return level
        raise IndexError(f"Anchor index {anchor_idx} is outside flattened feature levels")

    def _pair_discrimination_loss(
        self,
        cams: torch.Tensor,
        object_ids: torch.Tensor,
        pred_boxes: torch.Tensor,
        assignment_iou: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        """ODAM-Train BCE over reference-positive and overlapping-negative pairs."""

        cfg = self.odam_cfg
        overlap = pairwise_box_iou(pred_boxes, pred_boxes) > cfg.negative_overlap_iou
        terms: list[torch.Tensor] = []
        positive_count = 0
        negative_count = 0

        for object_id_t in torch.unique(object_ids):
            same = object_ids == object_id_t
            same_idx = torch.nonzero(same, as_tuple=False).flatten()
            reference_idx = same_idx[torch.argmax(assignment_iou[same_idx])]
            ref_cam = cams[reference_idx]

            for candidate_idx_t in same_idx:
                candidate_idx = int(candidate_idx_t.item())
                if candidate_idx == int(reference_idx.item()) and not cfg.include_self_positive:
                    continue
                similarity = torch.dot(ref_cam, cams[candidate_idx]).clamp(cfg.eps, 1.0 - cfg.eps)
                terms.append(-torch.log(similarity))
                positive_count += 1

            different_idx = torch.nonzero(~same, as_tuple=False).flatten()
            for candidate_idx_t in different_idx:
                candidate_idx = int(candidate_idx_t.item())
                if not bool(overlap[reference_idx, candidate_idx]):
                    continue
                similarity = torch.dot(ref_cam, cams[candidate_idx]).clamp(cfg.eps, 1.0 - cfg.eps)
                terms.append(-torch.log1p(-similarity))
                negative_count += 1

        if not terms:
            return cams.sum() * 0.0, 0, 0
        return torch.stack(terms).mean(), positive_count, negative_count

    @staticmethod
    def _smooth_gradient(grad: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Apply the local ODAM smoothing Phi from the paper.

        The paper uses a Gaussian kernel whose size can depend on object scale.
        For YOLOv8-P2 training, a fixed grouped Gaussian is a stable, minimal
        adaptation that preserves spatial gradients while reducing pixel noise.
        """

        if kernel_size <= 1:
            return grad
        channels = int(grad.shape[0])
        device = grad.device
        dtype = grad.dtype
        radius = kernel_size // 2
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        sigma = max(float(kernel_size) / 3.0, 1.0e-6)
        one_d = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
        one_d = one_d / one_d.sum().clamp_min(1.0e-12)
        kernel = (one_d[:, None] * one_d[None, :]).view(1, 1, kernel_size, kernel_size)
        kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
        return F.conv2d(grad[None], kernel, padding=radius, groups=channels)[0]


def aligned_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """IoU for aligned XYXY boxes with shape ``[N, 4]``."""

    if boxes1.shape != boxes2.shape or boxes1.ndim != 2 or boxes1.shape[-1] != 4:
        raise ValueError(f"Expected aligned [N,4] boxes, got {boxes1.shape} and {boxes2.shape}")
    lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp_min(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp_min(0)
    return inter / (area1 + area2 - inter).clamp_min(eps)


def pairwise_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Pairwise IoU for two XYXY box collections."""

    if boxes1.ndim != 2 or boxes2.ndim != 2 or boxes1.shape[-1] != 4 or boxes2.shape[-1] != 4:
        raise ValueError("pairwise_box_iou expects [N,4] and [M,4]")
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0)
        * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
    )[:, None]
    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0)
        * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)
    )[None, :]
    return inter / (area1 + area2 - inter).clamp_min(eps)
