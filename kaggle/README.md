# Kaggle GPU runner

This directory is a thin bootstrap layer. It does not contain training,
evaluation, XAI, or pruning implementations. Kaggle uploads `runner.py`, which
clones the configured Git branch and delegates to a script already in the
repository.

The complete Vietnamese operations runbook is available at
[`docs/kaggle_gpu_runbook_vi.md`](../docs/kaggle_gpu_runbook_vi.md).

## Before the first push

1. Commit and push the source and `configs/kaggle.yaml`. The local launch script
   refuses a dirty or unpushed checkout by default because those changes would
   not be visible to `git clone` on Kaggle.
2. Replace `KAGGLE_USERNAME` in `kernel-metadata.json`, or export:

   ```bash
   export KAGGLE_KERNEL_ID="your-username/xai-pruning-runner"
   ```

3. Attach datasets/checkpoints by adding Kaggle dataset slugs to
   `dataset_sources`. Do not add tokens or credentials to metadata or YAML.

The current repository is public and defaults to:

```text
https://github.com/Thanhmay2406/xai-object-detection.git
```

Override it for a direct/manual runner invocation with `--repo-url` or the
`XAI_REPO_URL` environment variable. Never place a private Git token in this
repository. If a future private clone is needed, use Kaggle Secrets.

## Arch Linux setup

```bash
./.venv/bin/python -m pip show kaggle
./.venv/bin/kaggle auth login
./.venv/bin/kaggle kernels list --mine
```

`.venv/bin/kaggle auth login` uses the current OAuth browser flow. The CLI can print a URL
instead with `.venv/bin/kaggle auth login --no-launch-browser`. Credentials stay in the
user account/configuration area and must never be copied into the repository.

## First GPU smoke test

The committed `configs/kaggle.yaml` defaults to `execution.mode: smoke_test`, so
the first push does not train anything:

```bash
export KAGGLE_KERNEL_ID="your-username/xai-pruning-runner"
./scripts/kaggle_run.sh NvidiaTeslaT4
./scripts/kaggle_status.sh
./scripts/kaggle_pull.sh results/kaggle/smoke-001
```

The smoke test requires CUDA, allocates one small CUDA tensor, prints the GPU,
and imports the shared pruning/evaluation package. Its result is written to
`/kaggle/working/xai_pruning_outputs/experiment.json`.

## Configure a real experiment later

Change `execution.mode` to `run` in `configs/kaggle.yaml`, choose an existing
entry point, and place its CLI arguments in `execution.args`. For example,
`scripts/evaluate.py` requires dataset and checkpoint paths below
`/kaggle/input/...`. Attach the corresponding Kaggle dataset slug in
`kernel-metadata.json`.

`execution.output_arg` causes the runner to append an output directory under
`/kaggle/working`. Set it to `null` for an entry point that does not accept an
output argument. Additional arguments can also be used for direct invocation:

```bash
./.venv/bin/python kaggle/runner.py --config configs/kaggle.yaml -- --device cuda
```

Suggested output convention:

```text
/kaggle/working/xai_pruning_outputs/
├── experiment.json
├── metrics.json                 # when produced by the selected entry point
├── importance.csv              # optional
├── pruning.json                # optional
├── logs/
├── plots/                       # optional
└── checkpoints/                 # optional
```

Only `experiment.json` and `logs/runner.log` are established by the runner.
Research entry points remain responsible for their own artifacts.
