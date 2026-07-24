from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class OdamLiveLogConfig:
    log_every: int = 1
    detail_batches: int = 3
    heartbeat_seconds: float = 20.0

    @classmethod
    def from_env(cls) -> "OdamLiveLogConfig":
        return cls(
            log_every=max(1, int(os.environ.get("ODAM_LOG_EVERY", "1"))),
            detail_batches=max(0, int(os.environ.get("ODAM_LOG_DETAIL_BATCHES", "3"))),
            heartbeat_seconds=max(0.0, float(os.environ.get("ODAM_HEARTBEAT_SECONDS", "20"))),
        )


class OdamLiveLogger:
    """Rank-zero, flush-on-write live logger for ODAM training diagnostics."""

    batch_fields = (
        "time",
        "epoch",
        "batch",
        "box_loss",
        "cls_loss",
        "dfl_loss",
        "odam_loss",
        "total_loss",
        "lr",
        "gpu_memory_gb",
        "batch_time_s",
        "throughput_img_s",
        "foreground_anchors",
        "selected_predictions",
        "cam_count",
        "positive_pairs",
        "negative_pairs",
        "raw_odam_loss",
        "weighted_odam_loss",
        "skip_reason",
    )

    def __init__(self, save_dir: str | Path, config: OdamLiveLogConfig | None = None):
        self.save_dir = Path(save_dir)
        self.config = config or OdamLiveLogConfig.from_env()
        self.live_path = self.save_dir / "odam_live.log"
        self.csv_path = self.save_dir / "odam_batches.csv"
        self.batch_jsonl_path = self.save_dir / "odam_batches.jsonl"
        self.epoch_jsonl_path = self.save_dir / "odam_epochs.jsonl"
        self._live_handle = None
        self._csv_handle = None
        self._batch_jsonl_handle = None
        self._epoch_jsonl_handle = None
        self._csv_writer: csv.DictWriter | None = None
        self._batch_start_time: float | None = None
        self._odam_start_time: float | None = None
        self._last_heartbeat_time: float | None = None
        self._epoch_rows: list[dict[str, Any]] = []

    @staticmethod
    def should_log_on_this_rank() -> bool:
        rank = int(os.environ.get("RANK", "-1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        return rank in {-1, 0} and local_rank in {-1, 0}

    def open(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._live_handle = self.live_path.open("a", encoding="utf-8", buffering=1)
        new_csv = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        self._csv_handle = self.csv_path.open("a", encoding="utf-8", newline="", buffering=1)
        self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=self.batch_fields)
        if new_csv:
            self._csv_writer.writeheader()
            self._csv_handle.flush()
        self._batch_jsonl_handle = self.batch_jsonl_path.open("a", encoding="utf-8", buffering=1)
        self._epoch_jsonl_handle = self.epoch_jsonl_path.open("a", encoding="utf-8", buffering=1)
        self.line(
            "ODAM live logging enabled "
            f"log_every={self.config.log_every} "
            f"detail_batches={self.config.detail_batches} "
            f"heartbeat_seconds={self.config.heartbeat_seconds:g}"
        )

    def close(self) -> None:
        self.line("ODAM live logging closed")
        for handle in (
            self._live_handle,
            self._csv_handle,
            self._batch_jsonl_handle,
            self._epoch_jsonl_handle,
        ):
            if handle is not None:
                handle.flush()
                handle.close()

    def line(self, text: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{stamp}] {text}"
        print(msg, flush=True)
        if self._live_handle is not None:
            self._live_handle.write(msg + "\n")
            self._live_handle.flush()

    def jsonl(self, handle: Any, payload: dict[str, Any]) -> None:
        if handle is not None:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()

    def start_epoch(self, epoch: int) -> None:
        self._epoch_rows = []
        self.line(f"epoch_start epoch={epoch + 1}")

    def end_epoch(self, epoch: int) -> None:
        count = len(self._epoch_rows)
        payload: dict[str, Any] = {
            "time": time.time(),
            "epoch": epoch + 1,
            "batches": count,
        }
        if count:
            for key in (
                "box_loss",
                "cls_loss",
                "dfl_loss",
                "odam_loss",
                "total_loss",
                "raw_odam_loss",
                "weighted_odam_loss",
                "batch_time_s",
                "throughput_img_s",
            ):
                payload[f"mean_{key}"] = sum(float(row[key]) for row in self._epoch_rows) / count
            for key in (
                "foreground_anchors",
                "selected_predictions",
                "cam_count",
                "positive_pairs",
                "negative_pairs",
            ):
                payload[f"sum_{key}"] = sum(int(row[key]) for row in self._epoch_rows)
        self.jsonl(self._epoch_jsonl_handle, payload)
        self.line(
            "epoch_end "
            f"epoch={epoch + 1} batches={count} "
            f"mean_total_loss={payload.get('mean_total_loss', 0.0):.6g}"
        )

    def start_batch(self, epoch: int, batch_index: int, batch_size: int) -> None:
        self._batch_start_time = time.perf_counter()
        if self.should_emit_batch(batch_index):
            self.line(f"batch_start epoch={epoch + 1} batch={batch_index} batch_size={batch_size}")

    def start_odam(self, epoch: int, batch_index: int) -> None:
        now = time.perf_counter()
        self._odam_start_time = now
        self._last_heartbeat_time = now
        if self.should_emit_batch(batch_index):
            self.line(f"odam_start epoch={epoch + 1} batch={batch_index}")

    def detail_enabled(self, batch_index: int) -> bool:
        return batch_index < self.config.detail_batches

    def should_emit_batch(self, batch_index: int) -> bool:
        return batch_index < self.config.detail_batches or batch_index % self.config.log_every == 0

    def heartbeat(self, epoch: int, batch_index: int, context: str) -> None:
        interval = self.config.heartbeat_seconds
        if interval <= 0 or self._odam_start_time is None:
            return
        now = time.perf_counter()
        last = self._last_heartbeat_time or self._odam_start_time
        if now - last >= interval:
            elapsed = now - self._odam_start_time
            self.line(
                "odam_heartbeat "
                f"epoch={epoch + 1} batch={batch_index} elapsed_s={elapsed:.1f} {context}"
            )
            self._last_heartbeat_time = now

    def detail(self, epoch: int, batch_index: int, image_index: int, message: str) -> None:
        if self.detail_enabled(batch_index):
            self.line(
                "odam_detail "
                f"epoch={epoch + 1} batch={batch_index} image={image_index} {message}"
            )

    def record_batch(
        self,
        *,
        epoch: int,
        batch_index: int,
        batch_size: int,
        loss_items: torch.Tensor,
        lr: float,
        gpu_memory_gb: float,
        stats: Any,
    ) -> None:
        elapsed = 0.0 if self._batch_start_time is None else time.perf_counter() - self._batch_start_time
        throughput = 0.0 if elapsed <= 0 else float(batch_size) / elapsed
        loss_values = loss_items.detach().float().cpu().tolist()
        while len(loss_values) < 4:
            loss_values.append(0.0)
        row = {
            "time": time.time(),
            "epoch": epoch + 1,
            "batch": batch_index,
            "box_loss": float(loss_values[0]),
            "cls_loss": float(loss_values[1]),
            "dfl_loss": float(loss_values[2]),
            "odam_loss": float(loss_values[3]),
            "total_loss": float(sum(loss_values[:4])),
            "lr": float(lr),
            "gpu_memory_gb": float(gpu_memory_gb),
            "batch_time_s": float(elapsed),
            "throughput_img_s": float(throughput),
            "foreground_anchors": int(getattr(stats, "foreground_anchors", 0)),
            "selected_predictions": int(getattr(stats, "selected_predictions", 0)),
            "cam_count": int(getattr(stats, "cam_count", 0)),
            "positive_pairs": int(getattr(stats, "positive_pairs", 0)),
            "negative_pairs": int(getattr(stats, "negative_pairs", 0)),
            "raw_odam_loss": float(getattr(stats, "raw_loss", 0.0)),
            "weighted_odam_loss": float(getattr(stats, "weighted_loss", 0.0)),
            "skip_reason": str(getattr(stats, "skip_reason", "")),
        }
        self._epoch_rows.append(row)
        if not self.should_emit_batch(batch_index):
            return
        if self._csv_writer is not None and self._csv_handle is not None:
            self._csv_writer.writerow(row)
            self._csv_handle.flush()
        self.jsonl(self._batch_jsonl_handle, row)
        self.line(
            "batch "
            f"epoch={row['epoch']} batch={row['batch']} "
            f"box={row['box_loss']:.6g} cls={row['cls_loss']:.6g} dfl={row['dfl_loss']:.6g} "
            f"odam={row['odam_loss']:.6g} total={row['total_loss']:.6g} "
            f"lr={row['lr']:.6g} gpu={row['gpu_memory_gb']:.3g}G "
            f"time={row['batch_time_s']:.3f}s throughput={row['throughput_img_s']:.2f}img/s "
            f"fg={row['foreground_anchors']} selected={row['selected_predictions']} "
            f"cams={row['cam_count']} pos={row['positive_pairs']} neg={row['negative_pairs']} "
            f"raw_odam={row['raw_odam_loss']:.6g} weighted_odam={row['weighted_odam_loss']:.6g} "
            f"skip={row['skip_reason'] or 'none'}"
        )


_LOGGER: OdamLiveLogger | None = None


def set_live_logger(logger: OdamLiveLogger | None) -> None:
    global _LOGGER
    _LOGGER = logger


def get_live_logger() -> OdamLiveLogger | None:
    return _LOGGER
