#!/usr/bin/env bash
# Launch single-image Qwen3-VL COCONUT training from the repository root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_CONFIG_PATH="${REPO_ROOT}/config/train.yaml"
MODEL_CONFIG_PATH="${REPO_ROOT}/config/qwen.yaml"
DEVICE=""
LOAD_CHECKPOINT=""
SAVE_NAME=""
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

usage() {
  cat <<EOF
Usage: bash train/run.sh [options]

Options:
  --train-config PATH  COCONUT training YAML (default: config/train.yaml)
  --model-config PATH  Qwen/MMMU model YAML (default: config/qwen.yaml)
  --device DEVICE      cuda, cuda:N, or cpu (default: preserve environment)
  --load-checkpoint X  Checkpoint file or history run directory to resume from
  --save-name NAME     New history run name for checkpoints written by this run
  -h, --help           Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-config)
      [[ $# -ge 2 ]] || { echo "--train-config requires a path" >&2; exit 2; }
      TRAIN_CONFIG_PATH="$2"; shift 2 ;;
    --model-config)
      [[ $# -ge 2 ]] || { echo "--model-config requires a path" >&2; exit 2; }
      MODEL_CONFIG_PATH="$2"; shift 2 ;;
    --device)
      [[ $# -ge 2 ]] || { echo "--device requires cuda, cuda:N, or cpu" >&2; exit 2; }
      DEVICE="$2"; shift 2 ;;
    --load-checkpoint)
      [[ $# -ge 2 ]] || { echo "--load-checkpoint requires a checkpoint path or run name" >&2; exit 2; }
      LOAD_CHECKPOINT="$2"; shift 2 ;;
    --save-name)
      [[ $# -ge 2 ]] || { echo "--save-name requires a run name" >&2; exit 2; }
      SAVE_NAME="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "${TRAIN_CONFIG_PATH}" ]]; then
  echo "COCONUT config not found: ${TRAIN_CONFIG_PATH}" >&2
  exit 2
fi
if [[ ! -f "${MODEL_CONFIG_PATH}" ]]; then
  echo "Qwen model config not found: ${MODEL_CONFIG_PATH}" >&2
  exit 2
fi

case "${DEVICE}" in
  "") ;;
  cuda) ;;
  cuda:[0-9]*)
    [[ "${NPROC_PER_NODE}" == "1" ]] || { echo "--device cuda:N requires NPROC_PER_NODE=1." >&2; exit 2; }
    export CUDA_VISIBLE_DEVICES="${DEVICE#cuda:}" ;;
  cpu)
    [[ "${NPROC_PER_NODE}" == "1" ]] || { echo "--device cpu requires NPROC_PER_NODE=1." >&2; exit 2; }
    export CUDA_VISIBLE_DEVICES="" ;;
  *)
    echo "Unsupported --device value: ${DEVICE}; use cuda, cuda:N, or cpu." >&2
    exit 2 ;;
esac

cd "${REPO_ROOT}"
TRAIN_ARGS=(--config "${TRAIN_CONFIG_PATH}" --qwen-config "${MODEL_CONFIG_PATH}")
if [[ -n "${LOAD_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--load-checkpoint "${LOAD_CHECKPOINT}")
fi
if [[ -n "${SAVE_NAME}" ]]; then
  TRAIN_ARGS+=(--save-name "${SAVE_NAME}")
fi
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m train.coconut "${TRAIN_ARGS[@]}"
