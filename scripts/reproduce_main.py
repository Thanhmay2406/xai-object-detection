#!/usr/bin/env python3
"""Reproduce the single frozen historical Main pruning experiment.

This entry point never trains or fine-tunes. It evaluates the immutable
baseline, recomputes the 300-image XAI ranking, applies only tau=0.01875,
evaluates the resulting dependency-safe model, and exports comparison evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from xai_pruning.config import (
    COCO_TO_MODEL_LABEL,
    MAIN_TAU,
    MODEL_TO_COCO_LABEL,
    NUM_CLASSES,
)
from xai_pruning.data.coco import COCODetectionDataset, detection_collate_fn
from xai_pruning.evaluation.coco_eval import evaluate_detector
from xai_pruning.models.faster_rcnn import load_baseline_model
from xai_pruning.pruning.groups import discover_resnet_bottleneck_pruning_groups
from xai_pruning.pruning.structural import (
    apply_structural_pruning_plan,
    build_pruning_plan_from_threshold,
    count_parameters,
)
from xai_pruning.utils.io import load_json, save_json
from xai_pruning.utils.seed import seed_everything
from xai_pruning.xai.importance import (
    aggregate_multi_group_importance,
    build_multi_group_importance_tables,
)


METRIC_KEYS = (
    "map",
    "ap50",
    "ap75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar1",
    "ar10",
    "ar100",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exactly one frozen historical Main reproduction without training."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--historical-artifacts", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(rows: list[dict], path: Path) -> Path:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def git_record() -> dict:
    def output(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        return {
            "commit": output("rev-parse", "HEAD"),
            "branch": output("branch", "--show-current"),
            "dirty": bool(output("status", "--porcelain")),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def resolve_paths(args: argparse.Namespace, reference: dict) -> dict[str, Path]:
    return {
        "train_images": args.dataset_root / reference["dataset"]["train_split"],
        "test_images": args.dataset_root / reference["dataset"]["evaluation_split"],
        "test_annotations": args.dataset_root
        / reference["dataset"]["evaluation_split"]
        / "_annotations.coco.json",
        "baseline_checkpoint": args.baseline_checkpoint,
        "probe_json": args.historical_artifacts
        / Path(reference["probe"]["relative_path"]).name,
        "historical_ranking": args.historical_artifacts
        / Path(reference["xai"]["historical_ranking_relative_path"]).name,
        "historical_main_checkpoint": args.historical_artifacts
        / Path(reference["pruning"]["historical_main_checkpoint_relative_path"]).name,
        "historical_plan": args.historical_artifacts
        / Path(reference["pruning"]["historical_plan_relative_path"]).name,
    }


def validate_frozen_reference(reference: dict) -> None:
    if int(reference["seed"]) != 42:
        raise ValueError("Historical Main seed must remain 42")
    if float(reference["pruning"]["tau"]) != MAIN_TAU:
        raise ValueError(f"Historical Main tau must remain {MAIN_TAU}")
    if int(reference["xai"]["ranking_rows"]) != 7_552:
        raise ValueError("Historical Main ranking must contain 7,552 rows")
    if reference["pruning"]["post_pruning_finetuning"] is not False:
        raise ValueError("This historical Main reference must not enable fine-tuning")
    if reference["xai"]["normalization"] != "group_max":
        raise ValueError("Historical Main must use group_max normalization")


def preflight(args: argparse.Namespace, reference: dict, paths: dict[str, Path]) -> dict:
    validate_frozen_reference(reference)
    required = [args.reference, args.config_file, *paths.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Full reproduction assets are missing: {missing}")

    expected_hashes = {
        "baseline_checkpoint": reference["baseline"]["sha256"],
        "probe_json": reference["probe"]["sha256"],
        "historical_ranking": reference["xai"]["historical_ranking_sha256"],
        "historical_main_checkpoint": reference["pruning"][
            "historical_main_checkpoint_sha256"
        ],
        "historical_plan": reference["pruning"]["historical_plan_sha256"],
        "test_annotations": reference["dataset"]["test_annotation_sha256"],
    }
    actual_hashes = {name: sha256_file(paths[name]) for name in expected_hashes}
    mismatches = {
        name: {"expected": expected_hashes[name], "actual": actual_hashes[name]}
        for name in expected_hashes
        if actual_hashes[name] != expected_hashes[name]
    }
    if mismatches:
        raise ValueError(f"Frozen asset hash mismatch: {mismatches}")

    probe = load_json(paths["probe_json"])
    test_annotations = load_json(paths["test_annotations"])
    ranking = read_csv(paths["historical_ranking"])
    historical_plan_payload = load_json(paths["historical_plan"])
    historical_plan = historical_plan_payload["main"]["pruning_plan"]
    plan_channels = sum(len(indices) for indices in historical_plan.values())
    report = {
        "status": "passed",
        "asset_hashes": actual_hashes,
        "probe_images": len(probe["images"]),
        "probe_annotations": len(probe["annotations"]),
        "test_images": len(test_annotations["images"]),
        "test_annotations": len(test_annotations["annotations"]),
        "historical_ranking_rows": len(ranking),
        "historical_plan_groups": len(historical_plan),
        "historical_plan_channels": plan_channels,
    }
    expected = {
        "probe_images": int(reference["probe"]["images"]),
        "test_images": int(reference["dataset"]["test_images"]),
        "test_annotations": int(reference["dataset"]["test_annotations"]),
        "historical_ranking_rows": int(reference["xai"]["ranking_rows"]),
        "historical_plan_groups": int(reference["pruning"]["expected_groups_pruned"]),
        "historical_plan_channels": int(reference["pruning"]["expected_channels_pruned"]),
    }
    deviations = {
        key: {"expected": expected[key], "actual": report[key]}
        for key in expected
        if report[key] != expected[key]
    }
    if deviations:
        raise ValueError(f"Frozen asset structure mismatch: {deviations}")
    return report


def compare_metrics(historical: dict, reproduced: dict) -> dict:
    return {
        key: {
            "historical": float(historical[key]),
            "reproduction": float(reproduced[key]),
            "delta": float(reproduced[key]) - float(historical[key]),
        }
        for key in METRIC_KEYS
    }


def compare_rankings(historical_rows: list[dict], reproduced_rows: list[dict]) -> dict:
    key = lambda row: (str(row["group_id"]), int(row["channel"]))
    historical_by_key = {key(row): row for row in historical_rows}
    reproduced_by_key = {key(row): row for row in reproduced_rows}
    if historical_by_key.keys() != reproduced_by_key.keys():
        raise ValueError("Historical and reproduced ranking channel keys differ")

    ordered_keys = sorted(historical_by_key)
    historical_ranks = np.asarray(
        [float(historical_by_key[item]["global_rank_least_to_most"]) for item in ordered_keys]
    )
    reproduced_ranks = np.asarray(
        [float(reproduced_by_key[item]["global_rank_least_to_most"]) for item in ordered_keys]
    )
    spearman = float(np.corrcoef(historical_ranks, reproduced_ranks)[0, 1])

    historical_order = [key(row) for row in historical_rows]
    reproduced_order = [key(row) for row in reproduced_rows]
    bottom_overlap = {}
    top_overlap = {}
    for k_value in (50, 100, 250, 500):
        bottom_overlap[str(k_value)] = len(
            set(historical_order[:k_value]) & set(reproduced_order[:k_value])
        ) / k_value
        top_overlap[str(k_value)] = len(
            set(historical_order[-k_value:]) & set(reproduced_order[-k_value:])
        ) / k_value

    historical_values = np.asarray(
        [float(historical_by_key[item]["importance_normalized"]) for item in ordered_keys]
    )
    reproduced_values = np.asarray(
        [float(reproduced_by_key[item]["importance_normalized"]) for item in ordered_keys]
    )
    return {
        "historical_rows": len(historical_rows),
        "reproduced_rows": len(reproduced_rows),
        "spearman_rank_correlation": spearman,
        "bottom_k_overlap": bottom_overlap,
        "top_k_overlap": top_overlap,
        "normalized_importance_mean_abs_difference": float(
            np.mean(np.abs(reproduced_values - historical_values))
        ),
        "historical_distribution": {
            "min": float(historical_values.min()),
            "max": float(historical_values.max()),
            "mean": float(historical_values.mean()),
            "median": float(np.median(historical_values)),
        },
        "reproduced_distribution": {
            "min": float(reproduced_values.min()),
            "max": float(reproduced_values.max()),
            "mean": float(reproduced_values.mean()),
            "median": float(np.median(reproduced_values)),
        },
    }


def build_environment(device: torch.device) -> dict:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": package_version("torchvision"),
        "kaggle_package_version": package_version("kaggle"),
        "numpy_version": np.__version__,
        "pycocotools_version": package_version("pycocotools"),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def run_reproduction(
    args: argparse.Namespace,
    reference: dict,
    paths: dict[str, Path],
    preflight_report: dict,
) -> dict:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact_paths = [
        output / "run_manifest.json",
        output / "environment.json",
        output / "baseline/metrics.json",
        output / "xai/ranking.csv",
        output / "pruning/pruning_plan.json",
        output / "evaluation/metrics.json",
        output / "comparison.json",
    ]
    existing = [str(path) for path in artifact_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite reproduction artifacts: {existing}")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The Full Reproduction Run requires an available CUDA device")
    seed = int(reference["seed"])
    seed_everything(seed, deterministic=True)
    environment = build_environment(device)
    save_json(environment, output / "environment.json")
    shutil.copyfile(args.config_file, output / "config.yaml")
    shutil.copyfile(args.reference, output / "historical_main_reference.json")

    manifest = {
        "status": "running",
        "started_at_unix": time.time(),
        "git": git_record(),
        "environment": environment,
        "dataset": {
            **reference["dataset"],
            "root": str(args.dataset_root),
            "train_images": str(paths["train_images"]),
            "test_annotations": str(paths["test_annotations"]),
        },
        "baseline_checkpoint": str(paths["baseline_checkpoint"]),
        "baseline_checkpoint_sha256": preflight_report["asset_hashes"][
            "baseline_checkpoint"
        ],
        "seed": seed,
        "probe_set": str(paths["probe_json"]),
        "probe_images": int(reference["probe"]["images"]),
        "xai": reference["xai"],
        "pruning": reference["pruning"],
        "evaluator": reference["evaluation"],
        "config_file": str(args.config_file),
        "runner_entrypoint": str(Path(__file__).resolve()),
        "full_training_started": False,
        "threshold_search_started": False,
        "ablation_started": False,
        "fine_tuning_started": False,
    }
    save_json(manifest, output / "run_manifest.json")
    save_json(preflight_report, output / "preflight_report.json")

    print("[1/6] Loading immutable baseline checkpoint", flush=True)
    model = load_baseline_model(paths["baseline_checkpoint"], device=device, strict=True)
    params_before = count_parameters(model)
    if params_before != int(reference["baseline"]["parameters"]):
        raise RuntimeError(f"Baseline parameter count changed: {params_before}")
    save_json({"parameters": params_before}, output / "pruning/before_stats.json")

    evaluation = reference["evaluation"]
    test_dataset = COCODetectionDataset(paths["test_images"], paths["test_annotations"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(evaluation["batch_size"]),
        shuffle=False,
        num_workers=int(evaluation["num_workers"]),
        collate_fn=detection_collate_fn,
        pin_memory=True,
    )
    print("[2/6] Evaluating baseline on frozen TEST split", flush=True)
    baseline_result = evaluate_detector(
        model,
        test_loader,
        paths["test_annotations"],
        device,
        model_name="baseline",
        split="test",
        prediction_json=output / "baseline/predictions.coco.json",
        log_every=100,
    )
    save_json(baseline_result.metrics, output / "baseline/metrics.json")
    save_json(baseline_result.per_class, output / "baseline/per_class.json")
    save_json(baseline_result.per_image, output / "baseline/per_image.json")

    print("[3/6] Recomputing 300-image Gradient x Activation importance", flush=True)
    probe_dataset = COCODetectionDataset(paths["train_images"], paths["probe_json"])
    probe_loader = DataLoader(
        probe_dataset,
        batch_size=int(reference["probe"]["batch_size"]),
        shuffle=False,
        num_workers=int(reference["probe"]["num_workers"]),
        collate_fn=detection_collate_fn,
        pin_memory=True,
    )
    groups = discover_resnet_bottleneck_pruning_groups(
        model,
        include_stages=tuple(reference["xai"]["group_stages"]),
        include_convs=tuple(reference["xai"]["group_convs"]),
        include_modules=True,
    )
    xai_result = aggregate_multi_group_importance(
        model,
        probe_loader,
        groups,
        device,
        min_iou=float(reference["xai"]["match_iou_threshold"]),
        weight_by_iou=bool(reference["xai"]["weight_matches_by_iou"]),
        analyze_empty=bool(reference["xai"]["analyze_empty_images"]),
        empty_fp_weight=float(reference["xai"]["empty_fp_importance_weight"]),
        confidence_weight=float(reference["xai"]["confidence_target_weight"]),
        localization_weight=float(reference["xai"]["localization_target_weight"]),
        enable_localization=True,
        log_every=20,
    )
    group_rows, channel_rows, ranking_rows = build_multi_group_importance_tables(
        groups, xai_result["aggregate_importance"]
    )
    if len(ranking_rows) != int(reference["xai"]["ranking_rows"]):
        raise RuntimeError(f"Reproduced ranking has {len(ranking_rows)} rows")
    write_csv(group_rows, output / "xai/group_summary.csv")
    write_csv(channel_rows, output / "xai/importance.csv")
    write_csv(ranking_rows, output / "xai/ranking.csv")
    torch.save(
        {
            "reference": reference["xai"],
            "group_metadata": [
                {key: value for key, value in group.items() if not key.endswith("_module")}
                for group in groups
            ],
            "aggregate_importance": xai_result["aggregate_importance"],
            "object_importance": xai_result["object_importance"],
            "empty_fp_importance": xai_result["empty_fp_importance"],
            "image_ids": xai_result["image_ids"],
            "stats": xai_result["stats"],
        },
        output / "xai/importance.pt",
    )
    torch.save(
        {
            "image_ids": xai_result["image_ids"],
            "group_order": xai_result["group_order"],
            "per_image_importance": xai_result["per_image_importance"],
        },
        output / "xai/per_image_importance.pt",
    )
    save_json(xai_result["stats"], output / "xai/stats.json")

    print("[4/6] Applying the single frozen tau=0.01875 plan", flush=True)
    group_metadata = [
        {key: value for key, value in group.items() if not key.endswith("_module")}
        for group in groups
    ]
    pruning_plan = build_pruning_plan_from_threshold(
        ranking_rows,
        group_metadata,
        float(reference["pruning"]["tau"]),
        min_remaining=int(reference["pruning"]["min_remaining_channels"]),
    )
    channels_pruned = sum(len(indices) for indices in pruning_plan.values())
    historical_plan_payload = load_json(paths["historical_plan"])
    historical_plan = {
        str(group_id): sorted(int(index) for index in indices)
        for group_id, indices in historical_plan_payload["main"]["pruning_plan"].items()
    }
    exact_plan_match = pruning_plan == historical_plan
    plan_report = {
        "candidate": "main",
        "tau": float(reference["pruning"]["tau"]),
        "groups_pruned": len(pruning_plan),
        "channels_pruned": channels_pruned,
        "exact_historical_plan_match": exact_plan_match,
        "plan": pruning_plan,
    }
    save_json(plan_report, output / "pruning/pruning_plan.json")
    if not exact_plan_match:
        raise RuntimeError("Reproduced pruning plan differs from historical Main")

    applied = apply_structural_pruning_plan(
        model,
        pruning_plan,
        group_metadata,
        min_remaining=int(reference["pruning"]["min_remaining_channels"]),
    )
    params_after = count_parameters(model)
    after_stats = {
        "parameters_before": params_before,
        "parameters_after": params_after,
        "parameter_reduction": params_before - params_after,
        "parameter_reduction_fraction": (params_before - params_after) / params_before,
        "parameter_reduction_percent": 100.0 * (params_before - params_after) / params_before,
        "groups_pruned": len(applied),
        "channels_pruned": sum(int(row["num_pruned"]) for row in applied),
        "applied_groups": applied,
    }
    save_json(after_stats, output / "pruning/after_stats.json")
    if params_after != int(reference["pruning"]["expected_parameters_after"]):
        raise RuntimeError(f"Pruned parameter count changed: {params_after}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "candidate_name": "main",
            "tau": float(reference["pruning"]["tau"]),
            "pruning_plan": pruning_plan,
            "num_classes": NUM_CLASSES,
            "label_mapping": {
                "coco_to_model_label": COCO_TO_MODEL_LABEL,
                "model_to_coco_label": MODEL_TO_COCO_LABEL,
            },
            "source_baseline_sha256": preflight_report["asset_hashes"][
                "baseline_checkpoint"
            ],
            "git": manifest["git"],
        },
        output / "pruning/main_reproduced_tau_0.0187500.pth",
    )

    print("[5/6] Evaluating reproduced Main on frozen TEST split", flush=True)
    main_result = evaluate_detector(
        model,
        test_loader,
        paths["test_annotations"],
        device,
        model_name="main_reproduction",
        split="test",
        prediction_json=output / "evaluation/predictions.coco.json",
        log_every=100,
    )
    save_json(main_result.metrics, output / "evaluation/metrics.json")
    save_json(main_result.per_class, output / "evaluation/per_class.json")
    save_json(main_result.per_image, output / "evaluation/per_image.json")

    print("[6/6] Comparing against immutable historical Main evidence", flush=True)
    historical_ranking = read_csv(paths["historical_ranking"])
    xai_comparison = compare_rankings(historical_ranking, ranking_rows)
    baseline_comparison = compare_metrics(
        evaluation["historical_baseline"], baseline_result.metrics
    )
    main_comparison = compare_metrics(evaluation["historical_main"], main_result.metrics)
    metric_tolerance = float(evaluation["metric_abs_tolerance"])
    metric_deltas = [
        abs(item["delta"])
        for comparison in (baseline_comparison, main_comparison)
        for item in comparison.values()
    ]
    ranking_pass = xai_comparison["spearman_rank_correlation"] >= 0.999 and all(
        value >= 0.99
        for family in ("bottom_k_overlap", "top_k_overlap")
        for value in xai_comparison[family].values()
    )
    deterministic_pass = (
        exact_plan_match
        and len(pruning_plan) == int(reference["pruning"]["expected_groups_pruned"])
        and channels_pruned == int(reference["pruning"]["expected_channels_pruned"])
        and params_before == int(reference["baseline"]["parameters"])
        and params_after == int(reference["pruning"]["expected_parameters_after"])
    )
    metrics_pass = max(metric_deltas) <= metric_tolerance
    if deterministic_pass and ranking_pass and metrics_pass:
        status = (
            "REPRODUCTION_PASS"
            if max(metric_deltas) == 0.0
            and xai_comparison["spearman_rank_correlation"] == 1.0
            else "REPRODUCTION_PASS_WITH_NUMERICAL_TOLERANCE"
        )
    else:
        status = "REPRODUCTION_FAIL"

    comparison = {
        "status": status,
        "baseline": baseline_comparison,
        "xai": xai_comparison,
        "pruning": plan_report,
        "complexity": after_stats,
        "main_evaluation": main_comparison,
        "gates": {
            "deterministic_artifacts": deterministic_pass,
            "xai_ranking": ranking_pass,
            "metrics_within_absolute_tolerance": metrics_pass,
            "metric_abs_tolerance": metric_tolerance,
            "maximum_metric_abs_delta": max(metric_deltas),
        },
    }
    save_json(comparison, output / "comparison.json")
    manifest.update(
        {
            "status": status,
            "completed_at_unix": time.time(),
            "artifacts": sorted(
                str(path.relative_to(output))
                for path in output.rglob("*")
                if path.is_file()
            ),
        }
    )
    save_json(manifest, output / "run_manifest.json")
    save_json(comparison, output / "final_report.json")
    if status == "REPRODUCTION_FAIL":
        raise RuntimeError("Full Main reproduction failed one or more comparison gates")
    print(json.dumps({"status": status, "gates": comparison["gates"]}, indent=2))
    return comparison


def main() -> None:
    args = parse_args()
    reference = load_json(args.reference)
    paths = resolve_paths(args, reference)
    preflight_report = preflight(args, reference, paths)
    print(json.dumps({"preflight": preflight_report}, indent=2), flush=True)
    if args.preflight_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_json(preflight_report, args.output_dir / "preflight_report.json")
        return
    run_reproduction(args, reference, paths, preflight_report)


if __name__ == "__main__":
    main()
