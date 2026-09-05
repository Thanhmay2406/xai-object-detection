#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
KAGGLE_DIR="${REPO_ROOT}/kaggle"
METADATA="${KAGGLE_DIR}/kernel-metadata.json"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
KAGGLE_CLI="${REPO_ROOT}/.venv/bin/kaggle"

if [[ ! -x "${VENV_PYTHON}" || ! -x "${KAGGLE_CLI}" ]]; then
    echo "Error: expected project executables ${VENV_PYTHON} and ${KAGGLE_CLI}" >&2
    exit 1
fi

if [[ ! -f "${METADATA}" || ! -f "${KAGGLE_DIR}/runner.py" ]]; then
    echo "Error: expected ${METADATA} and ${KAGGLE_DIR}/runner.py" >&2
    exit 1
fi

if [[ "${KAGGLE_SKIP_GIT_CHECK:-0}" != "1" ]]; then
    DEPLOYMENT_PATHS=(
        .gitignore
        pyproject.toml
        requirements.txt
        configs/historical_main_reference.json
        configs/kaggle.yaml
        kaggle/README.md
        kaggle/kernel-metadata.json
        kaggle/runner.py
        scripts/kaggle_run.sh
        scripts/kaggle_status.sh
        scripts/kaggle_pull.sh
        scripts/reproduce_main.py
        src/xai_pruning
    )
    if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=normal -- "${DEPLOYMENT_PATHS[@]}")" ]]; then
        echo "Error: deployment-related code/config has unpublished changes." >&2
        echo "The Kaggle runner clones Git, so commit and push the intended code/config first." >&2
        echo "Set KAGGLE_SKIP_GIT_CHECK=1 only if the configured remote already contains everything needed." >&2
        exit 1
    fi

    UPSTREAM="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null || true)"
    if [[ -z "${UPSTREAM}" ]]; then
        echo "Error: the current Git branch has no upstream; push it before launching Kaggle." >&2
        exit 1
    fi
    read -r AHEAD BEHIND < <(
        git -C "${REPO_ROOT}" rev-list --left-right --count "HEAD...${UPSTREAM}"
    )
    if [[ "${AHEAD}" != "0" || "${BEHIND}" != "0" ]]; then
        echo "Error: HEAD and ${UPSTREAM} differ (ahead=${AHEAD}, behind=${BEHIND})." >&2
        echo "Push/synchronize the branch before launching because Kaggle will clone the remote." >&2
        exit 1
    fi
fi

METADATA_ID="$("${VENV_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["id"])' "${METADATA}")"
KERNEL_ID="${KAGGLE_KERNEL_ID:-${METADATA_ID}}"

if [[ "${KERNEL_ID}" == *KAGGLE_USERNAME* || ! "${KERNEL_ID}" =~ ^[^/]+/[^/]+$ ]]; then
    echo "Error: replace KAGGLE_USERNAME in kaggle/kernel-metadata.json" >&2
    echo "or export KAGGLE_KERNEL_ID='your-username/xai-pruning-runner'." >&2
    exit 1
fi

STAGING_DIR="$(mktemp -d /tmp/xai-pruning-kaggle.XXXXXX)"
trap 'rm -rf -- "${STAGING_DIR}"' EXIT
cp "${KAGGLE_DIR}/runner.py" "${KAGGLE_DIR}/kernel-metadata.json" "${KAGGLE_DIR}/README.md" "${STAGING_DIR}/"

if [[ "${KERNEL_ID}" != "${METADATA_ID}" ]]; then
    "${VENV_PYTHON}" -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p, encoding="utf-8")); d["id"]=sys.argv[2]; json.dump(d, open(p, "w", encoding="utf-8"), indent=2); open(p, "a", encoding="utf-8").write("\n")' "${STAGING_DIR}/kernel-metadata.json" "${KERNEL_ID}"
fi

echo "Launching Kaggle kernel: ${KERNEL_ID}"
echo "Source directory: ${KAGGLE_DIR}"

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [ACCELERATOR_ID]" >&2
    exit 2
fi

if [[ $# -eq 1 ]]; then
    if ! "${KAGGLE_CLI}" kernels push --help | grep -q -- '--accelerator'; then
        echo "Error: installed Kaggle CLI does not support --accelerator." >&2
        exit 1
    fi
    echo "Accelerator override: $1"
    "${KAGGLE_CLI}" kernels push -p "${STAGING_DIR}" --accelerator "$1"
else
    "${KAGGLE_CLI}" kernels push -p "${STAGING_DIR}"
fi
