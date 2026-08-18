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
