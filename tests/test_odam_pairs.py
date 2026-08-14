import inspect
import unittest

try:
    import torch
    import network
    import train

    HAS_DEPS = True
except ModuleNotFoundError:  # pragma: no cover - environment guard
    torch = None
    network = None
    train = None
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "torch and train dependencies are required")
class ODAMPairMaskTest(unittest.TestCase):
    def test_same_object_different_proposals_are_positive_pairs(self):
        gt_ids = torch.tensor([0, 0])
        batch_ids = torch.tensor([0, 0])
        boxes = torch.tensor(
            [
                [10.0, 10.0, 100.0, 100.0],
                [12.0, 12.0, 102.0, 102.0],
            ]
        )

        same_obj, neg_mask, _ = network.image_aware_pair_masks(
            gt_ids,
            batch_ids,
            boxes,
        )

        self.assertTrue(bool(same_obj[0, 1]))
        self.assertTrue(bool(same_obj[1, 0]))
        self.assertFalse(bool(neg_mask[0, 1]))
        self.assertFalse(bool(neg_mask[1, 0]))

    def test_self_pairs_are_positive_and_not_negative(self):
        gt_ids = torch.tensor([0, 0])
        batch_ids = torch.tensor([0, 0])
        boxes = torch.tensor(
            [
                [10.0, 10.0, 100.0, 100.0],
                [12.0, 12.0, 102.0, 102.0],
            ]
        )

        same_obj, neg_mask, _ = network.image_aware_pair_masks(
            gt_ids,
            batch_ids,
            boxes,
        )

        self.assertTrue(bool(same_obj[0, 0]))
        self.assertTrue(bool(same_obj[1, 1]))
        self.assertFalse(bool(neg_mask[0, 0]))
        self.assertFalse(bool(neg_mask[1, 1]))

    def test_different_objects_same_image_do_not_become_positive(self):
        gt_ids = torch.tensor([0, 1])
        batch_ids = torch.tensor([0, 0])
        boxes = torch.tensor(
            [
                [10.0, 10.0, 100.0, 100.0],
                [12.0, 12.0, 102.0, 102.0],
            ]
        )

        same_obj, neg_mask, _ = network.image_aware_pair_masks(
            gt_ids,
            batch_ids,
            boxes,
        )

        self.assertFalse(bool(same_obj[0, 1]))
        self.assertFalse(bool(same_obj[1, 0]))
        self.assertTrue(bool(neg_mask[0, 1]))
        self.assertTrue(bool(neg_mask[1, 0]))

    def test_different_images_do_not_share_positive_or_negative_pairs(self):
        gt_ids = torch.tensor([0, 0])
        batch_ids = torch.tensor([0, 1])
        boxes = torch.tensor(
            [
                [10.0, 10.0, 100.0, 100.0],
                [10.0, 10.0, 100.0, 100.0],
            ]
        )

        same_obj, neg_mask, object_ids = network.image_aware_pair_masks(
            gt_ids,
            batch_ids,
            boxes,
        )

        self.assertNotEqual(int(object_ids[0]), int(object_ids[1]))
        self.assertFalse(bool(same_obj[0, 1]))
        self.assertFalse(bool(same_obj[1, 0]))
        self.assertFalse(bool(neg_mask[0, 1]))
        self.assertFalse(bool(neg_mask[1, 0]))

    def test_single_sample_self_pair_loss_is_finite(self):
        dams = torch.tensor([[1.0, 0.0]], requires_grad=True)
        gt_ids = torch.tensor([0])
        batch_ids = torch.tensor([0])
        boxes = torch.tensor([[10.0, 10.0, 100.0, 100.0]])
        pred_gt_iou = torch.tensor([0.9])

        loss = network.match_loss(dams, gt_ids, batch_ids, boxes, pred_gt_iou)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss.detach()), 1e-3)
        self.assertTrue(torch.isfinite(dams.grad).all())

    def test_empty_pair_loss_is_finite_zero(self):
        dams = torch.zeros((0, 49), requires_grad=True)
        gt_ids = torch.zeros((0,), dtype=torch.long)
        batch_ids = torch.zeros((0,), dtype=torch.long)
        boxes = torch.zeros((0, 4))
        pred_gt_iou = torch.zeros((0,))

        loss = network.match_loss(dams, gt_ids, batch_ids, boxes, pred_gt_iou)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertIsNotNone(dams.grad)

    def test_pair_masks_are_batch_size_invariant(self):
        gt_ids = torch.tensor([0, 0, 0, 0])
        batch_ids = torch.tensor([0, 0, 1, 1])
        boxes = torch.tensor(
            [
                [10.0, 10.0, 100.0, 100.0],
                [12.0, 12.0, 102.0, 102.0],
                [10.0, 10.0, 100.0, 100.0],
                [12.0, 12.0, 102.0, 102.0],
            ]
        )

        same_batched, neg_batched, _ = network.image_aware_pair_masks(
            gt_ids,
            batch_ids,
            boxes,
        )
        same_a, neg_a, _ = network.image_aware_pair_masks(
            gt_ids[:2],
            torch.zeros(2, dtype=torch.long),
            boxes[:2],
        )
        same_b, neg_b, _ = network.image_aware_pair_masks(
            gt_ids[2:],
            torch.zeros(2, dtype=torch.long),
            boxes[2:],
        )

        self.assertEqual(
            int(same_batched.sum()),
            int(same_a.sum() + same_b.sum()),
        )
        self.assertEqual(
            int(neg_batched.sum()),
            int(neg_a.sum() + neg_b.sum()),
        )

    def test_odam_and_dpga_use_same_canonical_odam_loss_source(self):
        loss_dict = {
            "loss_rpn_cls": torch.tensor(1.0),
            "loss_rpn_loc": torch.tensor(2.0),
            "loss_rcnn_cls": torch.tensor(3.0),
            "loss_rcnn_loc": torch.tensor(4.0),
            "loss_rcnn_match": torch.tensor(5.0),
        }

        _, loss_odam = network.split_detection_and_odam_loss(loss_dict)
        self.assertIs(loss_odam, loss_dict["loss_rcnn_match"])

        train_source = inspect.getsource(train.train_one_epoch)
        self.assertIn("loss_det, loss_odam = split_detection_and_odam_loss", train_source)
        self.assertIn("elif args.method == \"odam\":", train_source)
        self.assertIn("elif args.method == \"dpga\":", train_source)
        self.assertIn("loss_odam=loss_odam", train_source)


if __name__ == "__main__":
    unittest.main()
