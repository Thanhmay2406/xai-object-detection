import unittest

import torch

from rcnn_odamTrain.evaluate_odam_evidence import binary_auc, detection_heatmap_metrics


class ODAMEvidenceMetricTests(unittest.TestCase):
    def test_binary_auc_ranks_positive_pair_scores_above_negative(self):
        self.assertEqual(binary_auc([0.9, 0.8], [0.2, 0.1]), 1.0)
        self.assertEqual(binary_auc([0.5], [0.5]), 0.5)
        self.assertIsNone(binary_auc([], [0.1]))

    def test_detection_heatmap_metrics_measure_target_energy_and_peak(self):
        heatmap = torch.zeros((4, 4), dtype=torch.float32)
        heatmap[1, 1] = 3.0
        heatmap[3, 3] = 1.0
        target_box = torch.tensor([0.0, 0.0, 2.0, 2.0])
        other_boxes = torch.tensor([[3.0, 3.0, 4.0, 4.0]])

        metrics = detection_heatmap_metrics(
            heatmap,
            target_box,
            other_boxes,
            image_height=4.0,
            image_width=4.0,
            top_fraction=0.25,
        )

        self.assertAlmostEqual(metrics["target_energy_ratio"], 0.75)
        self.assertAlmostEqual(metrics["other_object_energy_ratio"], 0.25)
        self.assertEqual(metrics["pointing_box_hit"], 1.0)
        self.assertEqual(metrics["other_object_peak"], 0.0)


if __name__ == "__main__":
    unittest.main()
