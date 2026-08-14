import unittest

try:
    import torch
    import network

    HAS_TORCH = True
except ModuleNotFoundError:  # pragma: no cover - environment guard
    torch = None
    network = None
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch is required for DPGA tests")
class DPGATest(unittest.TestCase):
    def test_aligned_gradient_is_not_projected(self):
        controller = self._synthetic_controller(rho=10.0)
        final, stats = controller._compose_one_group(
            "module",
            [torch.tensor([1.0, 0.0])],
            [torch.tensor([0.5, 0.0])],
            alpha=1.0,
        )

        self.assertFalse(stats.projected)
        self.assertTrue(torch.allclose(final[0], torch.tensor([1.5, 0.0])))

    def test_conflicting_auxiliary_component_is_projected(self):
        controller = self._synthetic_controller(rho=10.0)
        final, stats = controller._compose_one_group(
            "module",
            [torch.tensor([1.0, 0.0])],
            [torch.tensor([-1.0, 1.0])],
            alpha=1.0,
        )

        safe_aux = final[0] - torch.tensor([1.0, 0.0])
        self.assertTrue(stats.projected)
        self.assertAlmostEqual(
            float(torch.dot(safe_aux, torch.tensor([1.0, 0.0]))),
            0.0,
            places=5,
        )

    def test_zero_auxiliary_gradient_is_finite_and_preserves_detection(self):
        controller = self._synthetic_controller(rho=1.0)
        final, stats = controller._compose_one_group(
            "module",
            [torch.tensor([1.0, 0.0])],
            [torch.tensor([0.0, 0.0])],
            alpha=1.0,
        )

        self.assertTrue(torch.isfinite(final[0]).all())
        self.assertTrue(torch.isfinite(torch.tensor(stats.gate)))
        self.assertTrue(torch.allclose(final[0], torch.tensor([1.0, 0.0])))

    def test_norm_cap_reduces_unsafe_auxiliary_without_exceeding_cap(self):
        controller = self._synthetic_controller(rho=0.5)
        _, stats = controller._compose_one_group(
            "module",
            [torch.tensor([1.0, 0.0])],
            [torch.tensor([0.0, 10.0])],
            alpha=1.0,
        )

        self.assertLessEqual(stats.odam_norm_after_cap, 0.5 + 1e-6)
        self.assertTrue(stats.cap_active)
        self.assertLessEqual(
            stats.odam_norm_after_cap,
            stats.odam_norm_after_projection + 1e-6,
        )

    def test_norm_cap_does_not_amplify_already_safe_auxiliary(self):
        controller = self._synthetic_controller(rho=2.0)
        _, stats = controller._compose_one_group(
            "module",
            [torch.tensor([1.0, 0.0])],
            [torch.tensor([0.0, 0.25])],
            alpha=1.0,
        )

        self.assertFalse(stats.cap_active)
        self.assertAlmostEqual(stats.norm_scale, 1.0, places=6)
        self.assertAlmostEqual(stats.odam_norm_after_cap, 0.25, places=6)

    def test_alpha_zero_preserves_detection_gradient(self):
        controller = self._synthetic_controller(rho=10.0)
        final, stats = controller._compose_one_group(
            "module",
            [torch.tensor([1.0, 2.0])],
            [torch.tensor([10.0, -10.0])],
            alpha=0.0,
        )

        self.assertTrue(torch.allclose(final[0], torch.tensor([1.0, 2.0])))
        self.assertEqual(stats.alpha, 0.0)
        self.assertEqual(stats.effective_weight, 0.0)

    def _synthetic_controller(self, rho: float):
        controller = network.DPGAController.__new__(network.DPGAController)
        controller.config = network.DPGAConfig(
            use_gate=False,
            default_policy=network.DPGAModulePolicy(
                rho=rho,
                tau=0.0,
                temperature=0.2,
            ),
            module_policies={},
        )
        controller.groups = {
            "module": [torch.nn.Parameter(torch.zeros(2))]
        }
        return controller


if __name__ == "__main__":
    unittest.main()
