import tempfile
import unittest
from pathlib import Path

import torch

import export_xai_metrics


class ExportXaiMetricsTest(unittest.TestCase):
    def test_default_discovery_includes_e6(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            for name in ("baseline", "e0", "e5", "e6"):
                (root / name).mkdir(parents=True)

            runs = export_xai_metrics.discover_run_dirs(
                root,
                selected=None,
                include_baseline=False,
            )

            self.assertEqual(
                [run.name for run in runs],
                ["e0", "e5", "e6"],
            )

    def test_dam_metrics_compute_pointing_game_and_saliency_iou(self):
        dam = torch.tensor(
            [
                0.0,
                0.1,
                0.0,
                0.0,
                0.2,
                1.0,
                0.7,
                0.0,
                0.0,
                0.3,
                0.6,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=torch.float32,
        )
        gt_box = torch.tensor([25.0, 25.0, 75.0, 75.0], dtype=torch.float32)

        metrics = export_xai_metrics.dam_metrics_in_box(
            dam_flat=dam,
            dam_h=4,
            dam_w=4,
            gt_box=gt_box,
            resized_h=100.0,
            resized_w=100.0,
            threshold_ratio=0.5,
        )

        self.assertAlmostEqual(metrics["pointing_game_hit"], 1.0)
        self.assertAlmostEqual(metrics["peak_x"], 1.0)
        self.assertAlmostEqual(metrics["peak_y"], 1.0)
        self.assertAlmostEqual(metrics["saliency_iou"], 3.0 / 4.0)
        self.assertAlmostEqual(metrics["saliency_area"], 3.0)
        self.assertAlmostEqual(metrics["gt_area_in_dam"], 4.0)

    def test_empty_dam_returns_no_metrics(self):
        metrics = export_xai_metrics.dam_metrics_in_box(
            dam_flat=torch.zeros(4),
            dam_h=2,
            dam_w=2,
            gt_box=torch.tensor([0.0, 0.0, 10.0, 10.0]),
            resized_h=10.0,
            resized_w=10.0,
            threshold_ratio=0.5,
        )
        self.assertEqual(metrics, {})


if __name__ == "__main__":
    unittest.main()
