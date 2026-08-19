import inspect
import unittest
from types import SimpleNamespace

try:
    import torch
    import network

    HAS_TORCH = True
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    network = None
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch is required for ODAM filtering tests")
class ODAMFilteringTest(unittest.TestCase):
    def test_filter_rejects_low_iou_and_low_score(self):
        pred_gt_iou = torch.tensor([0.69, 0.70, 0.90, 0.90])
        class_scores = torch.tensor([0.95, 0.89, 0.89, 0.90])

        keep = network.odam_quality_filter_mask(
            pred_gt_iou=pred_gt_iou,
            class_scores=class_scores,
            enabled=True,
            min_iou=0.70,
            min_score=0.90,
        )

        self.assertEqual(keep.tolist(), [False, False, False, True])

    def test_filter_keeps_threshold_boundary(self):
        keep = network.odam_quality_filter_mask(
            pred_gt_iou=torch.tensor([0.70]),
            class_scores=torch.tensor([0.90]),
            enabled=True,
            min_iou=0.70,
            min_score=0.90,
        )

        self.assertEqual(keep.tolist(), [True])

    def test_filter_off_keeps_old_foreground_set(self):
        keep = network.odam_quality_filter_mask(
            pred_gt_iou=torch.tensor([0.0, 0.2, 0.9]),
            class_scores=torch.tensor([0.0, 0.2, 0.9]),
            enabled=False,
            min_iou=0.70,
            min_score=0.90,
        )

        self.assertEqual(keep.tolist(), [True, True, True])

    def test_soft_reliability_is_detached_and_continuous(self):
        class_scores = torch.tensor([0.30, 0.70, 0.97], requires_grad=True)
        reliability = network.odam_reliability_weights(
            pred_gt_iou=torch.tensor([0.25, 0.60, 0.90]),
            class_scores=class_scores,
            enabled=True,
            iou_tau=0.60,
            iou_temperature=0.10,
            score_tau=0.70,
            score_temperature=0.10,
        )

        self.assertFalse(reliability.requires_grad)
        self.assertTrue(torch.all(reliability >= 0.0))
        self.assertTrue(torch.all(reliability <= 1.0))
        self.assertLess(float(reliability[0]), float(reliability[1]))
        self.assertLess(float(reliability[1]), float(reliability[2]))

    def test_adaptive_score_tau_uses_candidate_percentile(self):
        scores = torch.tensor([0.05, 0.10, 0.20, 0.40])

        tau = network.odam_score_tau(
            scores,
            enabled=True,
            percentile=0.50,
            default_tau=0.70,
        )
        fixed = network.odam_score_tau(
            scores,
            enabled=False,
            percentile=0.50,
            default_tau=0.70,
        )

        self.assertAlmostEqual(float(tau), 0.15, places=6)
        self.assertAlmostEqual(float(fixed), 0.70, places=6)

    def test_reliability_budget_is_per_image_and_gt(self):
        reliability = torch.tensor([0.1, 0.9, 0.4, 0.8, 0.2])
        batch_ids = torch.tensor([0, 0, 0, 1, 1])
        gt_ids = torch.tensor([1, 1, 2, 1, 1])
        candidate_mask = torch.tensor([True, True, True, True, True])

        keep = network.odam_reliability_budget_mask(
            reliability=reliability,
            batch_ids=batch_ids,
            gt_ids=gt_ids,
            candidate_mask=candidate_mask,
            enabled=True,
            fraction=0.50,
            min_keep=1,
        )

        self.assertEqual(keep.tolist(), [False, True, True, True, False])

    def test_reliability_budget_fraction_25_keeps_top_two_when_available(self):
        reliability = torch.tensor([0.1, 0.9, 0.4, 0.8])
        ids = torch.zeros(4, dtype=torch.long)
        candidate_mask = torch.ones(4, dtype=torch.bool)

        keep = network.odam_reliability_budget_mask(
            reliability=reliability,
            batch_ids=ids,
            gt_ids=ids,
            candidate_mask=candidate_mask,
            enabled=True,
            fraction=0.25,
            min_keep=2,
        )

        self.assertEqual(keep.tolist(), [False, True, False, True])

    def test_reliability_budget_fraction_50_keeps_half_per_group(self):
        reliability = torch.tensor([0.1, 0.9, 0.4, 0.8])
        ids = torch.zeros(4, dtype=torch.long)
        candidate_mask = torch.ones(4, dtype=torch.bool)

        keep = network.odam_reliability_budget_mask(
            reliability=reliability,
            batch_ids=ids,
            gt_ids=ids,
            candidate_mask=candidate_mask,
            enabled=True,
            fraction=0.50,
            min_keep=2,
        )

        self.assertEqual(keep.tolist(), [False, True, False, True])

    def test_reliability_budget_separates_images_with_same_gt_id(self):
        reliability = torch.tensor([0.1, 0.9, 0.2, 0.8])
        batch_ids = torch.tensor([0, 0, 1, 1])
        gt_ids = torch.tensor([3, 3, 3, 3])
        candidate_mask = torch.ones(4, dtype=torch.bool)

        keep = network.odam_reliability_budget_mask(
            reliability=reliability,
            batch_ids=batch_ids,
            gt_ids=gt_ids,
            candidate_mask=candidate_mask,
            enabled=True,
            fraction=0.25,
            min_keep=1,
        )

        self.assertEqual(keep.tolist(), [False, True, False, True])

    def test_reliability_budget_respects_candidate_mask(self):
        reliability = torch.tensor([0.9, 0.8, 0.7])
        ids = torch.tensor([0, 0, 0])
        candidate_mask = torch.tensor([False, True, True])

        keep = network.odam_reliability_budget_mask(
            reliability=reliability,
            batch_ids=ids,
            gt_ids=ids,
            candidate_mask=candidate_mask,
            enabled=True,
            fraction=1.0,
            min_keep=1,
        )

        self.assertEqual(keep.tolist(), [False, True, True])

    def test_reliability_budget_handles_zero_candidate(self):
        keep = network.odam_reliability_budget_mask(
            reliability=torch.tensor([0.9, 0.8]),
            batch_ids=torch.tensor([0, 0]),
            gt_ids=torch.tensor([0, 0]),
            candidate_mask=torch.tensor([False, False]),
            enabled=True,
            fraction=0.25,
            min_keep=2,
        )

        self.assertEqual(keep.tolist(), [False, False])

    def test_reliability_weighted_match_loss_downweights_noisy_roi(self):
        dams = torch.tensor(
            [
                [1.0, 0.0],
                [0.5, 0.5],
            ],
            requires_grad=True,
        )
        gt_ids = torch.tensor([0, 0])
        batch_ids = torch.tensor([0, 0])
        boxes = torch.tensor(
            [
                [10.0, 10.0, 100.0, 100.0],
                [12.0, 12.0, 102.0, 102.0],
            ]
        )
        pred_gt_iou = torch.tensor([0.9, 0.45])
        reliability = torch.tensor([1.0, 0.1])

        weighted, raw = network.match_loss(
            dams,
            gt_ids,
            batch_ids,
            boxes,
            pred_gt_iou,
            reliability=reliability,
            return_raw=True,
        )
        weighted.backward()

        self.assertTrue(torch.isfinite(weighted))
        self.assertTrue(torch.isfinite(raw))
        self.assertLess(float(weighted.detach()), float(raw.detach()))
        self.assertTrue(torch.isfinite(dams.grad).all())

    def test_pair_reliability_uses_reference_and_target(self):
        target = torch.tensor([0.81])
        high_reference = network.odam_pair_reliability(
            torch.tensor([1.0]),
            target,
        )
        low_reference = network.odam_pair_reliability(
            torch.tensor([0.01]),
            target,
        )

        self.assertLess(
            float(low_reference.detach()),
            float(high_reference.detach()),
        )
        self.assertAlmostEqual(float(high_reference), 0.9)
        self.assertAlmostEqual(float(low_reference), 0.09)

    def test_zero_valid_proposals_can_produce_finite_zero_odam_loss(self):
        keep = network.odam_quality_filter_mask(
            pred_gt_iou=torch.tensor([0.1, 0.2]),
            class_scores=torch.tensor([0.1, 0.2]),
            enabled=True,
            min_iou=0.70,
            min_score=0.90,
        )
        dams = torch.zeros((int(keep.sum()), 49), requires_grad=True)
        gt_ids = torch.zeros((0,), dtype=torch.long)
        batch_ids = torch.zeros((0,), dtype=torch.long)
        boxes = torch.zeros((0, 4))
        pred_gt_iou = torch.zeros((0,))

        loss = network.match_loss(dams, gt_ids, batch_ids, boxes, pred_gt_iou)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertIsNotNone(dams.grad)

    def test_e2_zero_valid_forward_backward_keeps_detection_path(self):
        config = SimpleNamespace(
            bbox_normalize_stds=(0.1, 0.1, 0.2, 0.2),
            bbox_normalize_means=(0.0, 0.0, 0.0, 0.0),
            num_classes=2,
            rcnn_smooth_l1_beta=1.0,
            pred_cls_threshold=0.05,
            odam_filtering=True,
            odam_min_iou=0.7,
            odam_min_score=1.0,
        )
        rcnn = network.RCNN(config)
        rcnn.train()
        rcnn.set_odam_enabled(True)

        fpn_fms = [
            torch.zeros(1, 256, 8, 8),
            torch.randn(1, 256, 64, 64, requires_grad=True),
            torch.randn(1, 256, 32, 32, requires_grad=True),
            torch.randn(1, 256, 16, 16, requires_grad=True),
            torch.randn(1, 256, 8, 8, requires_grad=True),
        ]
        rcnn_rois = torch.tensor([[0.0, 16.0, 16.0, 80.0, 80.0]])
        labels = torch.tensor([1])
        bbox_targets = torch.zeros(1, 4)
        assigned_gts = torch.tensor([0])

        losses = rcnn(
            fpn_fms,
            rcnn_rois,
            labels=labels,
            bbox_targets=bbox_targets,
            assigned_gts=assigned_gts,
        )
        objective = losses["loss_rcnn_loc"] + losses["loss_rcnn_cls"]
        objective.backward()

        self.assertTrue(torch.isfinite(losses["loss_rcnn_loc"]))
        self.assertTrue(torch.isfinite(losses["loss_rcnn_cls"]))
        self.assertTrue(torch.isfinite(losses["loss_rcnn_match"]))
        self.assertEqual(float(losses["loss_rcnn_match"].detach()), 0.0)
        self.assertEqual(rcnn.last_odam_filter_stats["candidates"], 1)
        self.assertEqual(rcnn.last_odam_filter_stats["kept"], 0)

    def test_e7_forward_backward_is_finite_and_logs_selection_stats(self):
        config = SimpleNamespace(
            bbox_normalize_stds=(0.1, 0.1, 0.2, 0.2),
            bbox_normalize_means=(0.0, 0.0, 0.0, 0.0),
            num_classes=2,
            rcnn_smooth_l1_beta=1.0,
            pred_cls_threshold=0.05,
            odam_filtering=True,
            odam_min_iou=0.5,
            odam_min_score=0.0,
            odam_reliability=True,
            odam_reliability_iou_tau=0.6,
            odam_reliability_iou_temp=0.1,
            odam_reliability_score_tau=0.7,
            odam_reliability_score_temp=0.1,
            odam_reliability_adaptive_score_tau=True,
            odam_reliability_score_percentile=0.70,
            odam_reliability_budget_enabled=True,
            odam_reliability_budget_fraction=0.25,
            odam_reliability_budget_min=2,
        )
        rcnn = network.RCNN(config)
        rcnn.train()
        rcnn.set_odam_enabled(True)
        with torch.no_grad():
            rcnn.pred_delta.weight.zero_()
            rcnn.pred_delta.bias.zero_()

        fpn_fms = [
            torch.zeros(1, 256, 8, 8),
            torch.randn(1, 256, 64, 64, requires_grad=True),
            torch.randn(1, 256, 32, 32, requires_grad=True),
            torch.randn(1, 256, 16, 16, requires_grad=True),
            torch.randn(1, 256, 8, 8, requires_grad=True),
        ]
        rcnn_rois = torch.tensor(
            [
                [0.0, 16.0, 16.0, 80.0, 80.0],
                [0.0, 18.0, 18.0, 82.0, 82.0],
            ]
        )
        labels = torch.tensor([1, 1])
        bbox_targets = torch.zeros(2, 4)
        assigned_gts = torch.tensor([0, 0])

        losses = rcnn(
            fpn_fms,
            rcnn_rois,
            labels=labels,
            bbox_targets=bbox_targets,
            assigned_gts=assigned_gts,
        )
        objective = (
            losses["loss_rcnn_loc"]
            + losses["loss_rcnn_cls"]
            + losses["loss_rcnn_match"]
        )
        objective.backward()

        stats = rcnn.last_odam_filter_stats
        self.assertTrue(torch.isfinite(objective))
        self.assertTrue(torch.isfinite(losses["loss_rcnn_match"]))
        self.assertEqual(stats["num_fg"], 2)
        self.assertEqual(stats["num_preselected"], 2)
        self.assertEqual(stats["num_budget_kept"], 2)
        self.assertIn("reliability_pre_p50", stats)
        self.assertIn("reliability_kept_p50", stats)
        self.assertGreaterEqual(stats["reliability_score_tau"], 0.0)

    def test_detection_losses_are_computed_before_odam_filtering(self):
        source = inspect.getsource(network.RCNN.forward)
        self.assertLess(
            source.index("loss_rcnn_loc = localization_loss.sum() / normalizer"),
            source.index("odam_quality_filter_mask("),
        )
        self.assertLess(
            source.index("loss_rcnn_cls = objectness_loss.sum() / normalizer"),
            source.index("odam_quality_filter_mask("),
        )


if __name__ == "__main__":
    unittest.main()
