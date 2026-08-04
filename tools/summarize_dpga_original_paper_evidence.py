#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EVIDENCE_RUNS = [
    "dpga_odam_citypersons_imagenet",
    "rcnn_odam_citypersons_imagenet",
    "dp_odam_citypersons_imagenet",
]

EVIDENCE_METRICS = [
    "target_energy_ratio_mean",
    "pointing_box_accuracy",
    "other_object_energy_ratio_mean",
    "other_object_peak_rate",
    "same_object_cosine_mean",
    "different_object_cosine_mean",
    "crowded_different_cosine_mean",
    "discrimination_margin",
    "pair_auc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize DPGA-ODAM evidence aligned with the ODAM paper.")
    parser.add_argument("--results-root", default="results/city-persons")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def fmt(value: Any, precision: int = 6) -> str:
    value = number(value)
    if value is None:
        return "missing"
    return f"{value:.{precision}f}"


def load_evidence(results_root: Path) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for run in EVIDENCE_RUNS:
        payload = load_json(results_root / run / "odam_evidence" / "summary.json")
        if payload is None:
            evidence[run] = {"missing": True}
            continue
        evidence[run] = {
            "missing": False,
            "config": payload.get("config", {}),
            "summary": payload.get("summary", {}),
        }
    return evidence


def load_sweep(path: Path) -> dict[str, Any] | None:
    payload = load_json(path)
    if payload is None:
        return None
    if "best" in payload and isinstance(payload["best"], dict):
        return payload
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return payload
    best = {}
    for metric in ("map_50_95", "map50", "map75", "ar_100"):
        candidates = [row for row in rows if number(row.get(metric)) is not None]
        if candidates:
            best[metric] = max(candidates, key=lambda row: float(row[metric]))
    payload["best"] = best
    return payload


def metric_delta(evidence: dict[str, dict[str, Any]], run: str, metric: str) -> float | None:
    dpga = evidence.get("dpga_odam_citypersons_imagenet", {})
    other = evidence.get(run, {})
    if dpga.get("missing") or other.get("missing"):
        return None
    left = number(dpga.get("summary", {}).get(metric))
    right = number(other.get("summary", {}).get(metric))
    if left is None or right is None:
        return None
    return left - right


def best_sweep_metric(sweep: dict[str, Any] | None, metric: str) -> Any:
    if sweep is None:
        return None
    best = sweep.get("best", {})
    row = best.get(metric)
    if isinstance(row, dict):
        return row.get(metric)
    return None


def write_report(results_root: Path, output: Path, evidence: dict[str, dict[str, Any]]) -> None:
    classical_sweep = load_sweep(results_root / "dpga_odam_citypersons_imagenet" / "classical_nms_sweep" / "threshold_sweep_results.json")
    odam_sweep = load_sweep(results_root / "dpga_odam_citypersons_imagenet" / "odam_nms_sweep" / "threshold_sweep_results.json")

    lines = [
        "# DPGA-ODAM Original-Paper Evidence Summary",
        "",
        "This report summarizes repository evidence for the original ODAM paper's chain:",
        "",
        "```text",
        "instance-specific explanation -> object discrimination -> crowded duplicate reasoning",
        "```",
        "",
        "## Evidence Availability",
        "",
        "| Run | ODAM evidence |",
        "|---|---|",
    ]
    for run in EVIDENCE_RUNS:
        lines.append(f"| `{run}` | {'missing' if evidence[run].get('missing') else 'available'} |")

    lines.extend(
        [
            "",
            "| Sweep | Status |",
            "|---|---|",
            f"| DPGA classical NMS sweep | {'available' if classical_sweep else 'missing'} |",
            f"| DPGA ODAM-NMS sweep | {'available' if odam_sweep else 'missing'} |",
            "",
            "## ODAM Evidence Metrics",
            "",
            "| Run | Target energy | Pointing | Other energy | Same cosine | Different cosine | Crowd diff cosine | Margin | Pair AUC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in EVIDENCE_RUNS:
        summary = evidence[run].get("summary", {})
        lines.append(
            f"| `{run}` | {fmt(summary.get('target_energy_ratio_mean'))} | "
            f"{fmt(summary.get('pointing_box_accuracy'))} | "
            f"{fmt(summary.get('other_object_energy_ratio_mean'))} | "
            f"{fmt(summary.get('same_object_cosine_mean'))} | "
            f"{fmt(summary.get('different_object_cosine_mean'))} | "
            f"{fmt(summary.get('crowded_different_cosine_mean'))} | "
            f"{fmt(summary.get('discrimination_margin'))} | "
            f"{fmt(summary.get('pair_auc'))} |"
        )

    lines.extend(
        [
            "",
            "## DPGA Deltas",
            "",
            "| Comparison | Target energy | Other energy | Same cosine | Different cosine | Margin | Pair AUC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in ("rcnn_odam_citypersons_imagenet", "dp_odam_citypersons_imagenet"):
        lines.append(
            f"| DPGA - `{run}` | "
            f"{fmt(metric_delta(evidence, run, 'target_energy_ratio_mean'))} | "
            f"{fmt(metric_delta(evidence, run, 'other_object_energy_ratio_mean'))} | "
            f"{fmt(metric_delta(evidence, run, 'same_object_cosine_mean'))} | "
            f"{fmt(metric_delta(evidence, run, 'different_object_cosine_mean'))} | "
            f"{fmt(metric_delta(evidence, run, 'discrimination_margin'))} | "
            f"{fmt(metric_delta(evidence, run, 'pair_auc'))} |"
        )

    lines.extend(
        [
            "",
            "## ODAM-NMS Sweep",
            "",
            "| Sweep | Best mAP50:95 | Best mAP50 | Best mAP75 | Best AR100 |",
            "|---|---:|---:|---:|---:|",
            f"| Classical NMS | {fmt(best_sweep_metric(classical_sweep, 'map_50_95'))} | "
            f"{fmt(best_sweep_metric(classical_sweep, 'map50'))} | "
            f"{fmt(best_sweep_metric(classical_sweep, 'map75'))} | "
            f"{fmt(best_sweep_metric(classical_sweep, 'ar_100'))} |",
            f"| ODAM-NMS | {fmt(best_sweep_metric(odam_sweep, 'map_50_95'))} | "
            f"{fmt(best_sweep_metric(odam_sweep, 'map50'))} | "
            f"{fmt(best_sweep_metric(odam_sweep, 'map75'))} | "
            f"{fmt(best_sweep_metric(odam_sweep, 'ar_100'))} |",
            "",
            "## Claim Status",
            "",
        ]
    )

    missing = [run for run in EVIDENCE_RUNS if evidence[run].get("missing")]
    if missing or classical_sweep is None or odam_sweep is None:
        lines.extend(
            [
                "Status: **incomplete**.",
                "",
                "The DPGA evidence can be interpreted, but the full original-paper evidence chain needs all comparison evidence and ODAM-NMS sweep artifacts.",
            ]
        )
    else:
        lines.extend(
            [
                "Status: **ready for interpretation**.",
                "",
                "All repository-aligned evidence groups are present. Interpret the direction of the DPGA deltas and the ODAM-NMS sweep before making a final claim.",
            ]
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output = Path(args.output) if args.output else results_root / "dpga_original_paper_evidence_summary.md"
    evidence = load_evidence(results_root)
    write_report(results_root, output, evidence)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
