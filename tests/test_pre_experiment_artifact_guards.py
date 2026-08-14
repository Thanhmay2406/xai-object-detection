import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

try:
    import train

    HAS_TRAIN_DEPS = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover - environment guard
    train = None
    HAS_TRAIN_DEPS = False


@unittest.skipUnless(HAS_TRAIN_DEPS, "train dependencies are required")
class PreExperimentArtifactGuardTest(unittest.TestCase):
    def test_csv_schema_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "metrics.csv"
            csv_path.write_text("epoch,method\n0,dpga\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "CSV schema mismatch"):
                train.append_csv_fields(
                    csv_path,
                    {"epoch": 1, "method": "dpga", "lr": 0.1},
                    ["epoch", "method", "lr"],
                )

    def test_csv_matching_schema_appends(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "metrics.csv"
            fields = ["epoch", "method", "lr"]
            csv_path.write_text(",".join(fields) + "\n", encoding="utf-8")

            train.append_csv_fields(
                csv_path,
                {"epoch": 1, "method": "dpga", "lr": 0.1},
                fields,
            )

            lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(lines[0], "epoch,method,lr")
            self.assertEqual(lines[1], "1,dpga,0.1")

    def test_removed_method_is_rejected_by_cli(self):
        removed_method = "ra" + "pg"
        argv = [
            "train.py",
            "--method",
            removed_method,
            "--train-images",
            "train",
            "--train-ann",
            "train.json",
            "--val-images",
            "val",
            "--val-ann",
            "val.json",
            "--output",
            "out",
        ]
        with patch("sys.argv", argv):
            with self.assertRaises(SystemExit):
                train.parse_args()

    def test_incompatible_checkpoint_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.pt"
            torch.save({"not_a_model_key": torch.zeros(1)}, path)
            model = torch.nn.Linear(2, 2)

            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                train.load_initial_weights(model, str(path))


if __name__ == "__main__":
    unittest.main()
