#!/usr/bin/env python3
"""Thin Kaggle bootstrapper for the existing repository entry points.

The kernel uploads this file, clones the project, installs the project's own
requirements without replacing Kaggle's working PyTorch build, and either runs
the CUDA smoke test or delegates to the configured project script.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_REPO_URL = "https://github.com/Thanhmay2406/xai-object-detection.git"
DEFAULT_BRANCH = "main"
DEFAULT_CONFIG = "configs/kaggle.yaml"
DEFAULT_CHECKOUT_NAME = "xai-object-detection"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Clone/setup this project on Kaggle and delegate to an existing entry point."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--repo-url", default=os.environ.get("XAI_REPO_URL", DEFAULT_REPO_URL))
    parser.add_argument("--branch", default=os.environ.get("XAI_REPO_BRANCH", DEFAULT_BRANCH))
    parser.add_argument("--entry-point")
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument(
        "entry_args",
        nargs=argparse.REMAINDER,
        help="Arguments after '--' are appended to the delegated project command.",
    )
    args = parser.parse_args()
    extra = list(args.entry_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    return args, extra


def is_kaggle_environment() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or (
        Path("/kaggle/working").is_dir() and Path("/kaggle/input").is_dir()
    )


def print_environment(in_kaggle: bool) -> dict[str, Any]:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    environment = {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "current_working_directory": str(Path.cwd()),
        "inside_kaggle": in_kaggle,
    }
    print("=" * 72, flush=True)
    print("XAI pruning Kaggle runner", flush=True)
    print("Python:", environment["python_version"], flush=True)
    print("Platform:", environment["platform"], flush=True)
    print("PyTorch:", environment["pytorch_version"], flush=True)
    print("CUDA available:", cuda_available, flush=True)
    print("GPU:", gpu_name, flush=True)
    print("Current working directory:", environment["current_working_directory"], flush=True)
    print("Running inside Kaggle:", in_kaggle, flush=True)
    print("Kaggle input root:", "/kaggle/input" if in_kaggle else None, flush=True)
    print("Kaggle output root:", "/kaggle/working" if in_kaggle else None, flush=True)
    print("=" * 72, flush=True)
    return environment


def run_checked(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def find_local_repository() -> Path | None:
    candidates = [Path.cwd(), Path(__file__).resolve().parent.parent]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src/xai_pruning"
        ).is_dir():
            return candidate.resolve()
    return None


def prepare_repository(args: argparse.Namespace, in_kaggle: bool) -> Path:
    local_repository = find_local_repository()
    if local_repository is not None and not in_kaggle:
        print("Using local repository:", local_repository, flush=True)
        return local_repository

    checkout_parent = Path("/kaggle/working") if in_kaggle else Path.cwd()
    checkout = checkout_parent / DEFAULT_CHECKOUT_NAME
    if checkout.exists():
        if (checkout / ".git").is_dir():
            print("Using existing checkout:", checkout, flush=True)
            return checkout.resolve()
        raise FileExistsError(f"Checkout path exists but is not a Git repository: {checkout}")

    if not args.repo_url:
        raise ValueError("Repository URL is required via --repo-url or XAI_REPO_URL")
    run_checked(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            args.branch,
            args.repo_url,
            str(checkout),
        ],
        cwd=checkout_parent,
    )
    return checkout.resolve()


def install_project(repository: Path, skip_install: bool) -> None:
    if skip_install:
        print("Dependency installation skipped by --skip-install", flush=True)
        return
    requirements = repository / "requirements.txt"
    if requirements.is_file():
        run_checked(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                str(requirements),
            ],
            cwd=repository,
        )
    else:
        print("No requirements.txt found; using the Kaggle environment as-is", flush=True)

    if (repository / "pyproject.toml").is_file():
        run_checked(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--no-build-isolation",
                "-e",
                str(repository),
            ],
            cwd=repository,
        )


def add_project_source_to_path(repository: Path) -> Path:
    """Expose the freshly cloned src-layout package to this running process."""

    source_root = (repository / "src").resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Project source directory not found: {source_root}")
    source_value = str(source_root)
    if source_value in sys.path:
        sys.path.remove(source_value)
    sys.path.insert(0, source_value)
    print("Project source path:", source_root, flush=True)
    return source_root


def load_yaml_config(repository: Path, config_value: str) -> tuple[dict[str, Any], Path]:
    import yaml

    config_path = Path(config_value)
    if not config_path.is_absolute():
        config_path = repository / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Kaggle experiment config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Kaggle config must contain a mapping: {config_path}")
    return config, config_path


def resolve_output_dir(
    repository: Path,
    in_kaggle: bool,
    cli_output: str | None,
    config: dict[str, Any],
) -> Path:
    configured = cli_output or config.get("artifacts", {}).get(
        "output_dir", "xai_pruning_outputs"
    )
    output = Path(configured)
    if not output.is_absolute():
        base = Path("/kaggle/working") if in_kaggle else repository / "results/kaggle/local"
        output = base / output
    output = output.resolve()
    if in_kaggle:
        kaggle_working = Path("/kaggle/working").resolve()
        if output != kaggle_working and kaggle_working not in output.parents:
            raise ValueError(f"Kaggle artifacts must be under /kaggle/working, found {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    return output


def write_experiment_record(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def append_runner_log(output_dir: Path, message: str) -> None:
    """Keep a small downloadable orchestration log alongside experiment metadata."""

    with (output_dir / "logs/runner.log").open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def run_gpu_smoke_test(repository: Path) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke test failed: torch.cuda.is_available() is False")
    gpu_name = torch.cuda.get_device_name(0)
    tensor = torch.ones((16, 16), device="cuda")
    checksum = float((tensor @ tensor).sum().item())
    del tensor
    torch.cuda.synchronize()
    print("CUDA tensor test: PASS", flush=True)

    # Import the real shared package; no model construction or training occurs.
    import xai_pruning
    print("xai_pruning import: PASS", flush=True)
    from xai_pruning.evaluation.coco_eval import evaluate_detector
    print("evaluation API import: PASS", flush=True)
    from xai_pruning.pruning.checkpoint import build_pruned_model_from_checkpoint
    print("reconstruction API import: PASS", flush=True)

    del evaluate_detector, build_pruned_model_from_checkpoint, xai_pruning
    result = {
        "cuda_available": True,
        "gpu_name": gpu_name,
        "cuda_tensor_test": "passed",
        "cuda_tensor_checksum": checksum,
        "xai_pruning_import": "passed",
        "evaluation_api_import": "passed",
        "reconstruction_api_import": "passed",
        "repository": str(repository),
    }
    print("GPU smoke test: PASS", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    return result


def run_entry_point(
    repository: Path,
    output_dir: Path,
    config: dict[str, Any],
    cli_entry_point: str | None,
    extra_args: list[str],
) -> list[str]:
    execution = config.get("execution", {})
    entry_value = cli_entry_point or execution.get("entry_point")
    if not entry_value:
        raise ValueError("execution.entry_point or --entry-point is required in run mode")
    entry_point = Path(entry_value)
    if not entry_point.is_absolute():
        entry_point = repository / entry_point
    entry_point = entry_point.resolve()
    if not entry_point.is_file():
        raise FileNotFoundError(f"Project entry point not found: {entry_point}")
    if repository != entry_point and repository not in entry_point.parents:
        raise ValueError(f"Entry point must be inside the cloned repository: {entry_point}")

    configured_args = execution.get("args", [])
    if not isinstance(configured_args, list) or not all(
        isinstance(value, (str, int, float)) for value in configured_args
    ):
        raise TypeError("execution.args must be a list of scalar command-line values")
    command = [sys.executable, str(entry_point)] + [str(value) for value in configured_args]
    output_argument = execution.get("output_arg")
    if output_argument:
        command.extend([str(output_argument), str(output_dir)])
    command.extend(extra_args)
    run_checked(command, cwd=repository)
    return command


def main() -> None:
    args, extra_args = parse_args()
    in_kaggle = is_kaggle_environment()
    environment = print_environment(in_kaggle)
    repository = prepare_repository(args, in_kaggle)
    install_project(repository, args.skip_install)
    add_project_source_to_path(repository)
    config, config_path = load_yaml_config(repository, args.config)
    output_dir = resolve_output_dir(repository, in_kaggle, args.output_dir, config)

    mode = "smoke_test" if args.smoke_test else config.get("execution", {}).get(
        "mode", "smoke_test"
    )
    if mode not in {"smoke_test", "run"}:
        raise ValueError(f"execution.mode must be smoke_test or run, found {mode!r}")

    record: dict[str, Any] = {
        "status": "running",
        "mode": mode,
        "started_at_unix": time.time(),
        "repository": str(repository),
        "repo_url": args.repo_url,
        "branch": args.branch,
        "config": str(config_path),
        "output_dir": str(output_dir),
        "inside_kaggle": in_kaggle,
        "runtime": environment,
    }
    record_path = output_dir / "experiment.json"
    write_experiment_record(record_path, record)
    append_runner_log(
        output_dir,
        f"status=running mode={mode} repository={repository} config={config_path}",
    )
    try:
        if mode == "smoke_test":
            record["smoke_test"] = run_gpu_smoke_test(repository)
            smoke = record["smoke_test"]
            append_runner_log(
                output_dir,
                " ".join(
                    [
                        f"python={environment['python_version']}",
                        f"pytorch={environment['pytorch_version']}",
                        f"cuda_available={smoke['cuda_available']}",
                        f"gpu={smoke['gpu_name']}",
                        f"cuda_tensor_test={smoke['cuda_tensor_test']}",
                        f"xai_pruning_import={smoke['xai_pruning_import']}",
                        f"evaluation_api_import={smoke['evaluation_api_import']}",
                        f"reconstruction_api_import={smoke['reconstruction_api_import']}",
                    ]
                ),
            )
        else:
            record["command"] = run_entry_point(
                repository,
                output_dir,
                config,
                args.entry_point,
                extra_args,
            )
        record["status"] = "completed"
        record["completed_at_unix"] = time.time()
        write_experiment_record(record_path, record)
        append_runner_log(output_dir, "status=completed")
    except BaseException as error:
        record["status"] = "failed"
        record["completed_at_unix"] = time.time()
        record["error_type"] = type(error).__name__
        record["error"] = str(error)
        write_experiment_record(record_path, record)
        append_runner_log(
            output_dir,
            f"status=failed error_type={type(error).__name__} error={error}",
        )
        raise

    print("Artifacts:", output_dir, flush=True)


if __name__ == "__main__":
    main()
