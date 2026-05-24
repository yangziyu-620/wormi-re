#!/usr/bin/env bash
set -euo pipefail

# VH stage-1: train the six scene-keyed world models from
# tools/world_curricula_vh.py. Run this on a GPU compute node.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CURRICULA_PATH="${CURRICULA_PATH:-tools/world_curricula_vh.py}"
DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
DATA_ROOT="${DATA_ROOT:-$DATA_DISK/wormi-data/virtualhome}"
CKPT_ROOT="${CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/world-vh}"
LOG_ROOT="${LOG_ROOT:-$DATA_DISK/wormi-logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$LOG_ROOT/vh-world-$RUN_ID"
LOG_FILE="$LOG_DIR/train.log"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WORMI_DATA_DISK="$DATA_DISK"
export WORMI_VH_DATA_ROOT="$DATA_ROOT"
export WORMI_WORLD_VH_OUTPUT_DIR="$CKPT_ROOT"
export HF_HOME="${HF_HOME:-$DATA_DISK/hf-home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_HOME="${TORCH_HOME:-$DATA_DISK/torch-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$DATA_DISK/triton-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_DISK/xdg-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$DATA_DISK/uv-cache}"
export TMPDIR="${TMPDIR:-$DATA_DISK/tmp}"

mkdir -p \
  "$LOG_DIR" "$CKPT_ROOT" "$HF_HOME" "$HF_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$TMPDIR"

echo "== WorMI VH world-model stage-1 training =="
echo "root:       $ROOT_DIR"
echo "curricula:  $CURRICULA_PATH"
echo "data root:  $DATA_ROOT"
echo "data disk:  $DATA_DISK"
echo "ckpt root:  $CKPT_ROOT"
echo "hf home:    $HF_HOME"
echo "uv cache:   $UV_CACHE_DIR"
echo "log file:   $LOG_FILE"
echo "started:    $(date -Is)"

if [[ ! -f "$CURRICULA_PATH" ]]; then
  echo "ERROR: curricula file not found: $CURRICULA_PATH" >&2
  exit 1
fi

for scene in scene_0 scene_1 scene_2 scene_3 scene_4 scene_5; do
  if [[ ! -s "$DATA_ROOT/$scene/train.jsonl" ]]; then
    echo "ERROR: missing or empty $DATA_ROOT/$scene/train.jsonl" >&2
    exit 1
  fi
  if [[ ! -e "$DATA_ROOT/$scene/test.jsonl" ]]; then
    echo "ERROR: missing $DATA_ROOT/$scene/test.jsonl" >&2
    exit 1
  fi
done

if [[ "${RUN_UV_SYNC:-0}" == "1" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not available on PATH" >&2
    exit 1
  fi
  echo "RUN_UV_SYNC=1: running uv sync before training"
  uv sync
elif [[ ! -x ".venv/bin/python" ]]; then
  echo "ERROR: .venv is missing. Run this once on a compute node:" >&2
  echo "  uv sync" >&2
  echo "or rerun this script with RUN_UV_SYNC=1." >&2
  exit 1
fi

WORMI_CMD="${WORMI_CMD:-.venv/bin/wormi}"
if [[ ! -x "$WORMI_CMD" ]]; then
  echo "ERROR: wormi command not found or not executable: $WORMI_CMD" >&2
  exit 1
fi

set -x
"$WORMI_CMD" world train \
  --curricula_path "$CURRICULA_PATH" \
  >> "$LOG_FILE" 2>&1
set +x

echo "finished: $(date -Is)"
