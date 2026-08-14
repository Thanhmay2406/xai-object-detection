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
class CheckpointFreezePolicyTest(unittest.TestCase):
    def test_pretrained_backbone_freeze_at_one_is_allowed(self):
        config = train.DetectorConfig(
            num_classes=2,
            backbone_freeze_at=1,
            backbone_pretrained=True,
        )
        model = _TinyBackboneModel()

        train.apply_backbone_freeze_policy(
            model,
            config,
            checkpoint_result=self._checkpoint_result(model, loaded_keys=[]),
        )

        self.assertEqual(model.frozen_at, 1)

    def test_no_pretrained_no_checkpoint_freeze_at_one_fails(self):
        config = train.DetectorConfig(
            num_classes=2,
            backbone_freeze_at=1,
            backbone_pretrained=False,
        )
        model = _TinyBackboneModel()

        with self.assertRaisesRegex(RuntimeError, "randomly initialized backbone"):
            train.apply_backbone_freeze_policy(
                model,
                config,
                checkpoint_result=self._checkpoint_result(model, loaded_keys=[]),
            )

    def test_broad_checkpoint_match_missing_stem_keys_fails(self):
        config = train.DetectorConfig(
            num_classes=2,
            backbone_freeze_at=1,
            backbone_pretrained=False,
        )
        model = _TinyBackboneModel()
        loaded_keys = [
            key for key in model.state_dict()
            if not key.startswith(("resnet50.conv1.", "resnet50.bn1."))
        ]

        with self.assertRaisesRegex(RuntimeError, "missing_checkpoint_keys"):
            train.apply_backbone_freeze_policy(
                model,
                config,
                checkpoint_result=self._checkpoint_result(model, loaded_keys),
            )

    def test_checkpoint_with_every_stem_key_freeze_at_one_is_allowed(self):
        config = train.DetectorConfig(
            num_classes=2,
            backbone_freeze_at=1,
            backbone_pretrained=False,
        )
        model = _TinyBackboneModel()
        loaded_keys = [
            key for key in model.state_dict()
            if key.startswith(("resnet50.conv1.", "resnet50.bn1."))
        ]

        train.apply_backbone_freeze_policy(
            model,
            config,
            checkpoint_result=self._checkpoint_result(model, loaded_keys),
        )

        self.assertEqual(model.frozen_at, 1)

    def test_checkpoint_with_every_stem_and_layer1_key_freeze_at_two_is_allowed(self):
        config = train.DetectorConfig(
            num_classes=2,
            backbone_freeze_at=2,
            backbone_pretrained=False,
        )
        model = _TinyBackboneModel()
        loaded_keys = [
            key for key in model.state_dict()
            if key.startswith(
                (
                    "resnet50.conv1.",
                    "resnet50.bn1.",
                    "resnet50.layer1.",
                )
            )
        ]

        train.apply_backbone_freeze_policy(
            model,
            config,
            checkpoint_result=self._checkpoint_result(model, loaded_keys),
        )

        self.assertEqual(model.frozen_at, 2)

    def test_freeze_at_three_is_rejected_by_config_and_direct_freeze(self):
        config = train.DetectorConfig(
            num_classes=2,
            backbone_freeze_at=3,
            backbone_pretrained=True,
        )
        model = _TinyBackboneModel()

        with self.assertRaisesRegex(ValueError, "one of"):
            train.apply_backbone_freeze_policy(
                model,
                config,
                checkpoint_result=self._checkpoint_result(model, loaded_keys=[]),
            )

        with self.assertRaisesRegex(ValueError, "one of"):
            network.validate_config(config)

        backbone = network.ResNet50.__new__(network.ResNet50)
        with self.assertRaisesRegex(ValueError, "one of"):
            network.ResNet50.freeze_backbone(backbone, 3)

    def test_negative_freeze_at_is_rejected_by_direct_freeze(self):
        backbone = network.ResNet50.__new__(network.ResNet50)

        with self.assertRaisesRegex(ValueError, "one of"):
            network.ResNet50.freeze_backbone(backbone, -1)

    def _checkpoint_result(self, model, loaded_keys):
        loaded_keys = set(loaded_keys)
        return train.CheckpointLoadResult(
            loaded_keys=loaded_keys,
            missing_keys=[
                key for key in model.state_dict()
                if key not in loaded_keys
            ],
            unexpected_keys=[],
            shape_mismatch_keys=[],
            matched_tensor_count=len(loaded_keys),
            total_tensor_count=len(model.state_dict()),
        )


if HAS_DEPS:
    class _TinyBackboneModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.resnet50 = torch.nn.Module()
            self.resnet50.conv1 = torch.nn.Conv2d(
                3,
                2,
                kernel_size=1,
                bias=False,
            )
            self.resnet50.bn1 = torch.nn.BatchNorm2d(2)
            self.resnet50.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(2, 2, kernel_size=1, bias=False),
                torch.nn.BatchNorm2d(2),
            )
            self.frozen_at = None

        def freeze_backbone(self, freeze_at):
            self.frozen_at = int(freeze_at)
else:
    _TinyBackboneModel = object


if __name__ == "__main__":
    unittest.main()
