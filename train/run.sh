#!/usr/bin/env bash
# Launch the Qwen3-VL COCONUT curriculum from the repository root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_CONFIG_PATH="${1:-${REPO_ROOT}/config/train.yaml}"
QWEN_CONFIG_PATH="${2:-${REPO_ROOT}/config/qwen.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

if [[ ! -f "${TRAIN_CONFIG_PATH}" ]]; then
  echo "COCONUT config not found: ${TRAIN_CONFIG_PATH}" >&2
  exit 2
fi
if [[ ! -f "${QWEN_CONFIG_PATH}" ]]; then
  echo "Qwen evaluation config not found: ${QWEN_CONFIG_PATH}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m train.coconut --config "${TRAIN_CONFIG_PATH}" --qwen-config "${QWEN_CONFIG_PATH}"
