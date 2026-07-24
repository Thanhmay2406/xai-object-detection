from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from odam_yolo.live_logging import OdamLiveLogConfig, OdamLiveLogger


class LiveLoggerTest(unittest.TestCase):
    def test_live_logger_writes_flushable_batch_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            logger = OdamLiveLogger(
                tmp_path,
                OdamLiveLogConfig(log_every=1, detail_batches=1, heartbeat_seconds=0.0),
            )
            logger.open()
            logger.start_epoch(0)
            logger.start_batch(epoch=0, batch_index=0, batch_size=2)
            logger.start_odam(epoch=0, batch_index=0)
            logger.record_batch(
                epoch=0,
                batch_index=0,
                batch_size=2,
                loss_items=torch.tensor([1.0, 2.0, 3.0, 0.5]),
                lr=0.01,
                gpu_memory_gb=1.25,
                stats=SimpleNamespace(
                    foreground_anchors=7,
                    selected_predictions=4,
                    cam_count=4,
                    positive_pairs=2,
                    negative_pairs=1,
                    raw_loss=1.0,
                    weighted_loss=0.5,
                    skip_reason="",
                ),
            )
            logger.end_epoch(0)
            logger.close()

            live_text = (tmp_path / "odam_live.log").read_text(encoding="utf-8")
            self.assertIn("ODAM live logging enabled", live_text)
            self.assertIn("batch_start epoch=1 batch=0 batch_size=2", live_text)
            self.assertIn("odam_start epoch=1 batch=0", live_text)
            self.assertIn("batch epoch=1 batch=0", live_text)
            self.assertIn("fg=7 selected=4 cams=4 pos=2 neg=1", live_text)

            with (tmp_path / "odam_batches.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["box_loss"], "1.0")
            self.assertEqual(rows[0]["cam_count"], "4")

            batch_rows = [
                json.loads(line)
                for line in (tmp_path / "odam_batches.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(batch_rows[0]["total_loss"], 6.5)
            self.assertEqual(batch_rows[0]["skip_reason"], "")

            epoch_rows = [
                json.loads(line)
                for line in (tmp_path / "odam_epochs.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(epoch_rows[0]["batches"], 1)
            self.assertEqual(epoch_rows[0]["sum_foreground_anchors"], 7)
