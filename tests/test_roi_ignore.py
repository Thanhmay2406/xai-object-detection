import json
import tempfile
import unittest
from pathlib import Path

try:
    from PIL import Image
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
class ROIIgnoreRegressionTest(unittest.TestCase):
    def test_dataset_ignore_annotation_can_use_unknown_category_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            ann_path = root / "annotations.json"
            Image.new("RGB", (16, 16), color=(255, 255, 255)).save(image_path)
            ann_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "id": 1,
                                "file_name": image_path.name,
                                "width": 16,
                                "height": 16,
                            }
                        ],
                        "annotations": [
                            {
                                "id": 1,
                                "image_id": 1,
                                "category_id": 999,
                                "bbox": [1, 1, 4, 4],
                                "area": 16,
                                "ignore": 1,
                                "iscrowd": 1,
                            }
                        ],
                        "categories": [
                            {"id": 1, "name": "person"}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            dataset = train.CocoDetectionTrainDataset(
                str(root),
                str(ann_path),
                min_size=16,
                max_size=16,
            )
            _, gt_boxes, _ = dataset[0]

        self.assertEqual(gt_boxes.shape, (1, 5))
        self.assertEqual(float(gt_boxes[0, 4]), -1.0)

    def test_foreground_background_and_ignore_are_separated(self):
        config = train.DetectorConfig(
            num_classes=2,
            num_rois=16,
            fg_ratio=0.5,
            train_batch_per_gpu=1,
        )
        rpn_rois = torch.tensor(
            [
                [0.0, 0.0, 0.0, 9.0, 9.0],
                [0.0, 30.0, 30.0, 39.0, 39.0],
                [0.0, 100.0, 100.0, 109.0, 109.0],
            ]
        )
        gt_boxes = torch.tensor(
            [
                [
                    [0.0, 0.0, 9.0, 9.0, 1.0],
                    [30.0, 30.0, 39.0, 39.0, -1.0],
                ]
            ]
        )
        im_info = torch.tensor([[128.0, 128.0, 1.0, 128.0, 128.0, 2.0]])

        rois, labels, _, _ = network.fpn_roi_target(
            config,
            rpn_rois,
            im_info,
            gt_boxes,
        )

        self.assertIn(1, labels.tolist())
        self.assertIn(0, labels.tolist())
        ignore_box = torch.tensor([30.0, 30.0, 39.0, 39.0])
        self.assertFalse(
            torch.isclose(rois[:, 1:5], ignore_box).all(dim=1).any()
        )

    def test_all_ignored_candidates_return_empty_roi_batch(self):
        config = train.DetectorConfig(
            num_classes=2,
            num_rois=16,
            train_batch_per_gpu=1,
        )
        rpn_rois = torch.tensor([[0.0, 30.0, 30.0, 39.0, 39.0]])
        gt_boxes = torch.tensor([[[30.0, 30.0, 39.0, 39.0, -1.0]]])
        im_info = torch.tensor([[128.0, 128.0, 1.0, 128.0, 128.0, 1.0]])

        rois, labels, bbox_targets, assigned_gts = network.fpn_roi_target(
            config,
            rpn_rois,
            im_info,
            gt_boxes,
        )

        self.assertEqual(tuple(rois.shape), (0, 5))
        self.assertEqual(labels.numel(), 0)
        self.assertEqual(tuple(bbox_targets.shape), (0, 4))
        self.assertEqual(assigned_gts.numel(), 0)

    def test_empty_valid_roi_batch_has_finite_roi_head_losses(self):
        config = train.DetectorConfig(
            num_classes=2,
            train_batch_per_gpu=1,
        )
        rcnn = network.RCNN(config)
        rcnn.train()
        fpn_fms = [
            torch.zeros((1, 256, 2, 2))
            for _ in range(5)
        ]
        loss_dict = rcnn(
            fpn_fms,
            torch.zeros((0, 5)),
            labels=torch.zeros((0,), dtype=torch.long),
            bbox_targets=torch.zeros((0, 4)),
            assigned_gts=torch.zeros((0,), dtype=torch.long),
        )

        self.assertEqual(set(loss_dict), {
            "loss_rcnn_loc",
            "loss_rcnn_cls",
            "loss_rcnn_match",
        })
        for loss in loss_dict.values():
            self.assertTrue(torch.isfinite(loss))

    def test_no_foreground_keeps_valid_background_sampling(self):
        config = train.DetectorConfig(
            num_classes=2,
            num_rois=16,
            train_batch_per_gpu=1,
        )
        rpn_rois = torch.tensor(
            [
                [0.0, 100.0, 100.0, 109.0, 109.0],
                [0.0, 120.0, 120.0, 129.0, 129.0],
            ]
        )
        gt_boxes = torch.zeros((1, 0, 5))
        im_info = torch.tensor([[128.0, 128.0, 1.0, 128.0, 128.0, 0.0]])

        rois, labels, _, _ = network.fpn_roi_target(
            config,
            rpn_rois,
            im_info,
            gt_boxes,
        )

        self.assertEqual(rois.shape[0], 2)
        self.assertEqual(labels.tolist(), [0, 0])

    def test_rpn_ignore_region_does_not_become_negative(self):
        config = train.DetectorConfig(
            num_classes=2,
            rpn_ignore_overlap=0.5,
        )
        anchors = torch.tensor(
            [
                [0.0, 0.0, 9.0, 9.0],
                [100.0, 100.0, 109.0, 109.0],
            ]
        )
        gt_boxes = torch.tensor([[[0.0, 0.0, 9.0, 9.0, -1.0]]])
        im_info = torch.tensor([32.0, 32.0, 1.0, 32.0, 32.0, 1.0])

        labels, _ = network.build_rpn_targets(
            config,
            gt_boxes[0],
            im_info,
            anchors,
        )

        self.assertEqual(int(labels[0]), int(config.ignore_label))
        self.assertEqual(int(labels[1]), 0)


if __name__ == "__main__":
    unittest.main()
