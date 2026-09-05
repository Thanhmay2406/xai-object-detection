#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
METADATA="${REPO_ROOT}/kaggle/kernel-metadata.json"
OUTPUT_DIR="${1:-${REPO_ROOT}/results/kaggle/latest}"
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
KERNEL_ID="${KAGGLE_KERNEL_ID:-${METADATA_ID}}"
if [[ "${KERNEL_ID}" == *KAGGLE_USERNAME* || ! "${KERNEL_ID}" =~ ^[^/]+/[^/]+$ ]]; then
    echo "Error: replace KAGGLE_USERNAME or export KAGGLE_KERNEL_ID." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
echo "Downloading latest output from ${KERNEL_ID}"
echo "Destination: ${OUTPUT_DIR}"
# Deliberately omit --force so existing local result files are not overwritten.
# Use one large page so nested artifacts are not omitted by the CLI default.
"${KAGGLE_CLI}" kernels output "${KERNEL_ID}" -p "${OUTPUT_DIR}" --page-size 200
