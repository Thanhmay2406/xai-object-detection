import unittest

import torch

from xai_pruning.xai.importance import build_multi_group_importance_tables


def _group(group_id, stage, block, kind):
    return {
        "group_id": group_id,
        "stage": stage,
        "block": block,
        "group_kind": kind,
        "producer_name": f"backbone.body.{group_id}",
        "norm_name": f"backbone.body.{group_id}.norm",
        "consumer_name": f"backbone.body.{group_id}.consumer",
    }


class ReproductionTableTests(unittest.TestCase):
    def test_historical_group_normalization_ties_and_global_order(self):
        groups = [
            _group("layer1.0.conv1", "layer1", 0, "conv1"),
            _group("layer1.0.conv2", "layer1", 0, "conv2"),
        ]
        importance = {
            "layer1.0.conv1": torch.tensor([0.0, 2.0, 2.0, 4.0]),
            "layer1.0.conv2": torch.tensor([1.0, 3.0]),
        }

        group_rows, channel_rows, ranking_rows = build_multi_group_importance_tables(
            groups, importance
        )

        self.assertEqual([row["group_id"] for row in group_rows], [
            "layer1.0.conv1",
            "layer1.0.conv2",
        ])
        first_group = [
            row for row in channel_rows if row["group_id"] == "layer1.0.conv1"
        ]
        self.assertEqual(
            [row["importance_normalized"] for row in first_group],
            [0.0, 0.5, 0.5, 1.0],
        )
        self.assertEqual(
            [row["within_group_percentile"] for row in first_group],
            [0.0, 0.5, 0.5, 1.0],
        )
        self.assertEqual(
            [(row["group_id"], row["channel"]) for row in ranking_rows],
            [
                ("layer1.0.conv1", 0),
                ("layer1.0.conv2", 0),
                ("layer1.0.conv1", 1),
                ("layer1.0.conv1", 2),
                ("layer1.0.conv2", 1),
                ("layer1.0.conv1", 3),
            ],
        )
        self.assertEqual(
            [row["global_rank_least_to_most"] for row in ranking_rows],
            list(range(1, 7)),
        )

    def test_rejects_missing_group_vector(self):
        groups = [_group("layer1.0.conv1", "layer1", 0, "conv1")]
        with self.assertRaisesRegex(ValueError, "Importance/group mismatch"):
            build_multi_group_importance_tables(groups, {})


if __name__ == "__main__":
    unittest.main()
