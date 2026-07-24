import argparse
import os
import threading
import time
from pathlib import Path
from typing import Any

from odam_yolo.trainer import OdamDetectionTrainer


def prepend_pythonpath(path: Path) -> None:
    """Expose this local package to Ultralytics DDP worker subprocesses."""

    resolved = str(path.resolve())
    current = os.environ.get("PYTHONPATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if resolved not in entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([resolved, *entries])


class LiveLogTailer:
    """Mirror rank-zero ODAM file logs back to the parent process during DDP."""

    def __init__(self, path: Path, interval: float = 1.0):
        self.path = path
        self.interval = max(0.1, float(interval))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="odam-log-tail", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        seek_to_end = self.path.exists()
        while not self._stop.is_set():
            if not self.path.exists():
                time.sleep(self.interval)
                continue
            try:
                with self.path.open("r", encoding="utf-8", errors="replace") as handle:
                    if seek_to_end:
                        handle.seek(0, os.SEEK_END)
                        seek_to_end = False
                    while not self._stop.is_set():
                        line = handle.readline()
                        if line:
                            print(line.rstrip("\n"), flush=True)
                        else:
                            time.sleep(self.interval)
            except OSError as exc:
                print(f"ODAM live log tail waiting: {exc}", flush=True)
                time.sleep(self.interval)


def is_multi_device(device: str) -> bool:
    normalized = str(device).strip().lower()
    return "," in normalized and normalized not in {"cpu", "mps"}


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8-P2 with ODAM-Train")
    parser.add_argument("--model", required=True, help="YOLOv8-P2 YAML or baseline .pt checkpoint")
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML")
    parser.add_argument(
        "--odam-config",
        default="configs/odam_yolov8_p2.yaml",
        help="ODAM YAML configuration",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="Examples: 0, 0,1, cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/odam")
    parser.add_argument("--name", default="yolov8_p2_odam")
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", type=str_to_bool, default=True)
    parser.add_argument("--amp", type=str_to_bool, default=True)
    parser.add_argument("--cache", type=str_to_bool, default=False)
    parser.add_argument("--resume", type=str_to_bool, default=False)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--log-every", type=int, default=1, help="Emit live batch logs every N batches")
    parser.add_argument(
        "--tail-live-log",
        type=str_to_bool,
        default=True,
        help="In multi-GPU DDP, mirror odam_live.log to the parent process stdout.",
    )
    parser.add_argument(
        "--tail-live-log-interval",
        type=float,
        default=1.0,
        help="Seconds between DDP live-log tail polls.",
    )
    parser.add_argument(
        "--log-detail-batches",
        type=int,
        default=3,
        help="Emit per-image/CAM detail for the first N batches of each epoch",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=20.0,
        help="Emit an ODAM heartbeat when CAM generation exceeds this many seconds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_phase_dir = Path(__file__).resolve().parent
    prepend_pythonpath(train_phase_dir)

    odam_path = Path(args.odam_config).expanduser().resolve()
    if not odam_path.is_file():
        raise FileNotFoundError(f"ODAM config not found: {odam_path}")
    os.environ["ODAM_CONFIG_PATH"] = str(odam_path)
    os.environ["ODAM_LOG_EVERY"] = str(max(1, args.log_every))
    os.environ["ODAM_LOG_DETAIL_BATCHES"] = str(max(0, args.log_detail_batches))
    os.environ["ODAM_HEARTBEAT_SECONDS"] = str(max(0.0, args.heartbeat_seconds))

    overrides: dict[str, Any] = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "amp": args.amp,
        "cache": args.cache,
        "resume": args.resume,
        "fraction": args.fraction,
        "patience": args.patience,
        "save_period": args.save_period,
        # Hooks and autograd.grad are not torch.compile-safe in this package.
        "compile": False,
    }
    print(
        "ODAM-Train starting "
        f"model={args.model} data={args.data} config={odam_path} "
        f"epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} device={args.device} "
        f"log_every={args.log_every} detail_batches={args.log_detail_batches}",
        flush=True,
    )
    trainer = OdamDetectionTrainer(overrides=overrides)
    tailer: LiveLogTailer | None = None
    if args.tail_live_log and is_multi_device(args.device):
        live_log_path = Path(trainer.save_dir) / "odam_live.log"
        print(f"ODAM DDP live log mirror watching {live_log_path}", flush=True)
        tailer = LiveLogTailer(live_log_path, interval=args.tail_live_log_interval)
        tailer.start()
    try:
        trainer.train()
    finally:
        if tailer is not None:
            tailer.stop()


if __name__ == "__main__":
    main()
