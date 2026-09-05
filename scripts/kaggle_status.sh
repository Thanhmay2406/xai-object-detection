#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
METADATA="${REPO_ROOT}/kaggle/kernel-metadata.json"

if ! command -v kaggle >/dev/null 2>&1; then
    echo "Error: Kaggle CLI is not installed. Install it with: pipx install kaggle" >&2
    exit 1
fi
if [[ ! -f "${METADATA}" ]]; then
    echo "Error: missing ${METADATA}" >&2
    exit 1
fi

METADATA_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["id"])' "${METADATA}")"
KERNEL_ID="${1:-${KAGGLE_KERNEL_ID:-${METADATA_ID}}}"
if [[ "${KERNEL_ID}" == *KAGGLE_USERNAME* || ! "${KERNEL_ID}" =~ ^[^/]+/[^/]+$ ]]; then
    echo "Error: provide a kernel ID or replace KAGGLE_USERNAME in kernel-metadata.json." >&2
    exit 1
fi

echo "Kaggle kernel status: ${KERNEL_ID}"
kaggle kernels status "${KERNEL_ID}"
