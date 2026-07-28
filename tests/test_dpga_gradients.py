import unittest

import torch

from rcnn_odamTrain.train import compose_dpga_module_gradients, compose_dpga_odam_gradients, dpga_parameter_module


class _Context:
    enabled = False
    world_size = 1


class _Args:
    amp = False
    dpga_backbone_norm_ratio = 0.05
    dpga_fpn_norm_ratio = 0.10
    dpga_rpn_norm_ratio = 0.0
    dpga_roi_shared_norm_ratio = 0.20
    dpga_roi_classifier_norm_ratio = 0.20
    dpga_roi_regressor_norm_ratio = 0.02
    dpga_global_norm_ratio = 0.10
    dpga_module_coverage = "roi-head-only"
    dpga_projection = True
    dpga_fail_on_missing_detection_grad = True


class _TinyRCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.RCNN = torch.nn.Module()
        self.RCNN.fc1 = torch.nn.Linear(2, 2, bias=False)
        self.RCNN.fc2 = torch.nn.Linear(2, 2, bias=False)
        self.RCNN.pred_cls = torch.nn.Linear(2, 1, bias=False)


class DPGAGradientTests(unittest.TestCase):
    def test_negative_cosine_projection_removes_conflict(self):
        det = torch.tensor([1.0, 0.0])
        odam = torch.tensor([-1.0, 1.0])
        safe, stats = compose_dpga_module_gradients(
            [det],
            [odam],
            {
                "enabled": True,
                "max_norm_ratio": 10.0,
                "reject_cosine": -1.0,
                "full_cosine": -0.5,
                "project_on_negative": True,
            },
        )

        self.assertEqual(stats["projected"], 1.0)
        self.assertGreaterEqual(stats["effective_scale"], 0.0)
        self.assertAlmostEqual(float(torch.dot(det, safe[0])), 0.0, places=6)

    def test_norm_ratio_caps_odam_gradient(self):
        det = torch.tensor([2.0, 0.0])
        odam = torch.tensor([0.0, 10.0])
        safe, stats = compose_dpga_module_gradients(
            [det],
            [odam],
            {
                "enabled": True,
                "max_norm_ratio": 0.25,
                "reject_cosine": -0.1,
                "full_cosine": 0.0,
                "project_on_negative": True,
            },
        )

        max_norm = 0.25 * det.norm()
        self.assertLessEqual(float(safe[0].norm()), float(max_norm) + 1e-6)
        self.assertAlmostEqual(stats["norm_scale"], 0.05, places=6)

    def test_parameter_group_mapping(self):
        self.assertEqual(dpga_parameter_module("resnet50.body.layer4.0.conv1.weight"), "backbone")
        self.assertEqual(dpga_parameter_module("FPN.lateral_convs.0.weight"), "fpn")
        self.assertEqual(dpga_parameter_module("RPN.conv.weight"), "rpn")
        self.assertEqual(dpga_parameter_module("RCNN.fc1.weight"), "roi_shared")
        self.assertEqual(dpga_parameter_module("RCNN.pred_cls.weight"), "roi_classifier")
        self.assertEqual(dpga_parameter_module("RCNN.pred_delta.weight"), "roi_regressor")
        self.assertEqual(dpga_parameter_module("resnet50.body.fc.weight"), "unused")

    def test_compose_assigns_parameter_grads_for_active_odam(self):
        model = _TinyRCNN()
        x = torch.tensor([[1.0, -2.0]])
        shared = model.RCNN.fc2(model.RCNN.fc1(x))
        logits = model.RCNN.pred_cls(shared)
        det_loss = logits.square().sum()
        odam_loss = (logits - 1.0).square().sum()

        total_loss, stats = compose_dpga_odam_gradients(det_loss, odam_loss, model, _Args(), _Context())

        self.assertTrue(torch.isfinite(total_loss))
        self.assertEqual(stats["stat_dpga_any_active"], 1.0)
        self.assertGreater(stats["stat_dpga_roi_classifier_valid"], 0.0)
        self.assertIsNotNone(model.RCNN.pred_cls.weight.grad)
        self.assertTrue(torch.isfinite(model.RCNN.pred_cls.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
