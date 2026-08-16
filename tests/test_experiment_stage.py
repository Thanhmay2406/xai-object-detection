import unittest
from unittest.mock import patch

try:
    import torch
    import train
    import network

    HAS_TRAIN_DEPS = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    torch = None
    train = None
    network = None
    HAS_TRAIN_DEPS = False


@unittest.skipUnless(HAS_TRAIN_DEPS, "train dependencies are required")
class ExperimentStageTest(unittest.TestCase):
    def test_stage_mapping_is_deterministic(self):
        expected = {
            "E0": (False, False, False, False, False),
            "E1": (True, False, False, False, False),
            "E2": (True, True, False, False, False),
            "E3": (True, True, True, False, False),
            "E4": (True, True, True, True, False),
            "E5": (True, True, True, True, True),
        }

        for stage, values in expected.items():
            with self.subTest(stage=stage):
                cfg = train.resolve_experiment_stage_config(stage)
                self.assertEqual(
                    (
                        cfg.warmup_enabled,
                        cfg.filtering_enabled,
                        cfg.projection_enabled,
                        cfg.norm_cap_enabled,
                        cfg.gate_enabled,
                    ),
                    values,
                )

    def test_e2_cli_sets_warmup_and_filtering_without_dpga_components(self):
        args = self._parse(
            "--method",
            "odam",
            "--experiment-stage",
            "E2",
        )

        self.assertTrue(args.warmup_enabled)
        self.assertTrue(args.filtering_enabled)
        self.assertTrue(args.odam_filtering)
        self.assertFalse(args.projection_enabled)
        self.assertFalse(args.norm_cap_enabled)
        self.assertFalse(args.gate_enabled)

    def test_e4_cli_sets_projection_and_norm_cap_without_gate(self):
        args = self._parse(
            "--method",
            "dpga",
            "--experiment-stage",
            "E4",
        )

        self.assertTrue(args.warmup_enabled)
        self.assertTrue(args.filtering_enabled)
        self.assertTrue(args.projection_enabled)
        self.assertTrue(args.norm_cap_enabled)
        self.assertFalse(args.gate_enabled)
        self.assertTrue(args.dpga_projection)
        self.assertTrue(args.dpga_norm_cap)
        self.assertFalse(args.dpga_gate)
        self.assertIsNone(args.dpga_ablation)
        self.assertEqual(args.dpga_ablation_label, "E4_incremental")

    def test_stage_rejects_legacy_dpga_ablation_flags(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self._parse(
                "--method",
                "dpga",
                "--experiment-stage",
                "E3",
                "--dpga-ablation",
                "projection-only",
            )

    def test_stage_rejects_wrong_method_family(self):
        with self.assertRaisesRegex(ValueError, "requires --method dpga"):
            self._parse(
                "--method",
                "odam",
                "--experiment-stage",
                "E3",
            )

        with self.assertRaisesRegex(ValueError, "requires --method odam"):
            self._parse(
                "--method",
                "dpga",
                "--experiment-stage",
                "E0",
            )

    def test_legacy_dpga_ablation_still_works_without_stage(self):
        args = self._parse(
            "--method",
            "dpga",
            "--dpga-ablation",
            "projection-only",
        )

        self.assertTrue(args.dpga_projection)
        self.assertFalse(args.dpga_norm_cap)
        self.assertFalse(args.dpga_gate)
        self.assertEqual(args.dpga_ablation_label, "A2_projection")

    def test_e0_e1_scalar_smoke_backward(self):
        e0 = self._parse("--method", "odam", "--experiment-stage", "E0")
        e1 = self._parse("--method", "odam", "--experiment-stage", "E1")
        e1_active = self._parse(
            "--method",
            "odam",
            "--experiment-stage",
            "E1",
            "--dpga-warmup",
            "0",
            "--dpga-rampup",
            "0",
        )

        self.assertTrue(
            self._scalar_backward_is_finite(e0, epoch=0.0)
        )
        self.assertEqual(train._odam_weight_for_epoch(e1, 0.0), 0.0)
        self.assertTrue(
            self._scalar_backward_is_finite(e1_active, epoch=0.0)
        )

    def test_e3_e4_e5_dpga_smoke_composition_is_finite(self):
        for stage in ("E3", "E4", "E5"):
            with self.subTest(stage=stage):
                args = self._parse(
                    "--method",
                    "dpga",
                    "--experiment-stage",
                    stage,
                )
                controller = network.DPGAController.__new__(
                    network.DPGAController
                )
                controller.config = network.DPGAConfig(
                    project_if_conflict=args.dpga_projection,
                    use_norm_cap=args.dpga_norm_cap,
                    use_gate=args.dpga_gate,
                    default_policy=network.DPGAModulePolicy(
                        rho=0.5,
                        tau=0.0,
                        temperature=0.2,
                    ),
                    module_policies={},
                )
                controller.groups = {
                    "module": [torch.nn.Parameter(torch.zeros(2))]
                }

                final, stats = controller._compose_one_group(
                    "module",
                    [torch.tensor([1.0, 0.0])],
                    [torch.tensor([-1.0, 2.0])],
                    alpha=1.0,
                )

                self.assertTrue(torch.isfinite(final[0]).all())
                self.assertTrue(torch.isfinite(torch.tensor(stats.gate)))

    def _scalar_backward_is_finite(self, args, epoch):
        param = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
        loss_det = (param.square()).sum()
        loss_odam = (param * torch.tensor([0.5, -0.25])).sum()
        weight = train._odam_weight_for_epoch(args, epoch)
        objective = loss_det + weight * loss_odam
        objective.backward()
        return bool(torch.isfinite(param.grad).all())

    def _parse(self, *extra):
        argv = [
            "train.py",
            "--train-images",
            "train_images",
            "--train-ann",
            "train.json",
            "--val-images",
            "val_images",
            "--val-ann",
            "val.json",
            "--output",
            "out",
            *extra,
        ]
        with patch("sys.argv", argv):
            return train.parse_args()


if __name__ == "__main__":
    unittest.main()
