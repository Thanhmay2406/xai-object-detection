#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AblationVariant:
    name: str
    description: str
    output_dir: str
    flags: tuple[str, ...]


VARIANTS: dict[str, AblationVariant] = {
    "match_only": AblationVariant(
        name="match_only",
        description="Only SAB class/box heatmap matching; disables scale, edge, and inside losses.",
        output_dir="sab_ablation_match_only",
        flags=(
            "--odam-loss-start-epoch",
            "4",
            "--odam-loss-warmup-epochs",
            "5",
            "--sab-lambda-match",
            "1.0",
            "--sab-lambda-scale",
            "0.0",
            "--sab-lambda-edge",
            "0.0",
            "--sab-lambda-inside",
            "0.0",
        ),
    ),
    "match_scale": AblationVariant(
        name="match_scale",
        description="Adds SAB scale consistency while keeping boundary/inside losses disabled.",
        output_dir="sab_ablation_match_scale",
        flags=(
            "--odam-loss-start-epoch",
            "4",
            "--odam-loss-warmup-epochs",
            "5",
            "--sab-lambda-match",
            "1.0",
            "--sab-lambda-scale",
            "0.1",
            "--sab-lambda-edge",
            "0.0",
            "--sab-lambda-inside",
            "0.0",
        ),
    ),
    "tuned_loss": AblationVariant(
        name="tuned_loss",
        description="Lower-pressure full SAB: delayed warmup and reduced edge/inside losses.",
        output_dir="sab_ablation_tuned_loss",
        flags=(
            "--odam-loss-start-epoch",
            "6",
            "--odam-loss-warmup-epochs",
            "8",
            "--sab-lambda-match",
            "1.0",
            "--sab-lambda-scale",
            "0.1",
            "--sab-lambda-edge",
            "0.03",
            "--sab-lambda-inside",
            "0.02",
        ),
    ),
    "late_light": AblationVariant(
        name="late_light",
        description="More conservative SAB: later warmup and lower all auxiliary weights.",
        output_dir="sab_ablation_late_light",
        flags=(
            "--odam-loss-start-epoch",
            "8",
            "--odam-loss-warmup-epochs",
            "10",
            "--sab-lambda-match",
            "0.75",
            "--sab-lambda-scale",
            "0.05",
            "--sab-lambda-edge",
            "0.02",
            "--sab-lambda-inside",
            "0.01",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or run controlled SAB-ODAM ablation training commands."
    )
    parser.add_argument("--data-root", default="data/drill_bit_coco")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--variants", nargs="+", default=["all"], choices=["all", *VARIANTS])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--use-torchrun", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--plan-dir", default="results/sab_ablation_plan")
    return parser.parse_args()


def selected_variants(names: list[str]) -> list[AblationVariant]:
    if "all" in names:
        return list(VARIANTS.values())
    seen: set[str] = set()
    variants: list[AblationVariant] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        variants.append(VARIANTS[name])
    return variants


def repo_resolved_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def base_train_args(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        str(REPO_ROOT / "rcnn_odamTrain" / "train.py"),
        "--data-root",
        str(repo_resolved_path(args.data_root)),
        "--output-dir",
        str(output_dir),
        "--backbone-weights",
        "default",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--image-size",
        str(args.image_size),
        "--lr",
        str(args.lr),
        "--momentum",
        "0.9",
        "--weight-decay",
        "0.0005",
        "--step-size",
        "8",
        "--gamma",
        "0.1",
        "--amp",
        "--include-empty-categories",
        "--odam-nms",
        "--odam-nms-low-threshold",
        "0.2",
        "--odam-nms-high-threshold",
        "0.8",
        "--odam-nms-resize-short-edge",
        "50",
        "--sab-odam",
        "--sab-small-resolution",
        "28",
        "--sab-medium-resolution",
        "14",
        "--sab-large-resolution",
        "7",
        "--sab-topk-per-gt",
        "2",
        "--sab-max-rois-per-batch",
        "32",
        "--sab-small-weight-gamma",
        "0.5",
        "--test-after-train",
        "--test-checkpoint",
        "best",
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def full_command(args: argparse.Namespace, variant: AblationVariant) -> list[str]:
    output_dir = repo_resolved_path(args.output_root) / variant.output_dir
    train_args = base_train_args(args, output_dir)
    train_args.extend(variant.flags)
    if args.use_torchrun:
        return [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            str(args.nproc_per_node),
            *train_args,
        ]
    return [args.python, *train_args]


def shell_line(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def write_plan(plan_dir: Path, variants: list[AblationVariant], commands: list[list[str]], args: argparse.Namespace) -> None:
    plan_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "data_root": args.data_root,
        "output_root": args.output_root,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "image_size": args.image_size,
        "lr": args.lr,
        "use_torchrun": args.use_torchrun,
        "nproc_per_node": args.nproc_per_node,
        "variants": [asdict(variant) for variant in variants],
        "commands": [command for command in commands],
    }
    (plan_dir / "ablation_plan.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for variant, command in zip(variants, commands):
        lines.append(f"# {variant.name}: {variant.description}")
        lines.append(shell_line(command))
        lines.append("")
    commands_path = plan_dir / "commands.sh"
    commands_path.write_text("\n".join(lines), encoding="utf-8")
    commands_path.chmod(0o755)


def run_commands(args: argparse.Namespace, variants: list[AblationVariant], commands: list[list[str]]) -> None:
    for variant, command in zip(variants, commands):
        output_dir = repo_resolved_path(args.output_root) / variant.output_dir
        if args.skip_existing and (output_dir / "test_metrics.json").exists():
            print(f"skip variant={variant.name} existing={output_dir / 'test_metrics.json'}", flush=True)
            continue
        print(f"run variant={variant.name} command={shell_line(command)}", flush=True)
        subprocess.run(command, check=True, cwd=REPO_ROOT)


def main() -> None:
    args = parse_args()
    if args.execute and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "Do not launch run_sab_ablation_plan.py with torchrun. "
            "Run it once with python; this runner launches torchrun for each training arm."
        )
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.workers < 0:
        raise ValueError("--workers must be >= 0")
    if args.nproc_per_node < 1:
        raise ValueError("--nproc-per-node must be >= 1")

    variants = selected_variants(args.variants)
    commands = [full_command(args, variant) for variant in variants]
    plan_dir = repo_resolved_path(args.plan_dir)
    write_plan(plan_dir, variants, commands, args)

    print(f"wrote_plan={plan_dir / 'ablation_plan.json'}", flush=True)
    print(f"wrote_commands={plan_dir / 'commands.sh'}", flush=True)
    for variant, command in zip(variants, commands):
        print(f"{variant.name}: {shell_line(command)}", flush=True)

    if args.execute:
        run_commands(args, variants, commands)
    else:
        print("dry_run=true add --execute to run commands sequentially", flush=True)


if __name__ == "__main__":
    main()
