#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
METADATA="${REPO_ROOT}/kaggle/kernel-metadata.json"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
KAGGLE_CLI="${REPO_ROOT}/.venv/bin/kaggle"

if [[ ! -x "${VENV_PYTHON}" || ! -x "${KAGGLE_CLI}" ]]; then
    echo "Error: expected project executables ${VENV_PYTHON} and ${KAGGLE_CLI}" >&2
    exit 1
fi
if [[ ! -f "${METADATA}" ]]; then
    echo "Error: missing ${METADATA}" >&2
    exit 1
fi

METADATA_ID="$("${VENV_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["id"])' "${METADATA}")"
KERNEL_ID="${1:-${KAGGLE_KERNEL_ID:-${METADATA_ID}}}"
if [[ "${KERNEL_ID}" == *KAGGLE_USERNAME* || ! "${KERNEL_ID}" =~ ^[^/]+/[^/]+$ ]]; then
    echo "Error: provide a kernel ID or replace KAGGLE_USERNAME in kernel-metadata.json." >&2
    exit 1
fi

echo "Kaggle kernel status: ${KERNEL_ID}"
"${KAGGLE_CLI}" kernels status "${KERNEL_ID}"
