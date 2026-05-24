#!/usr/bin/env bash
set -euo pipefail

# VH stage-2: train WorMI cross-attention adapters using the six frozen
# scene-keyed world models produced by stage 1.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CURRICULA_PATH="${CURRICULA_PATH:-tools/wormi_curricula_vh.py}"
DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
DATA_ROOT="${DATA_ROOT:-$DATA_DISK/wormi-data/virtualhome}"
WORLD_CKPT_ROOT="${WORLD_CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/world-vh}"
CKPT_ROOT="${CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/wormi-vh}"
LOG_ROOT="${LOG_ROOT:-$DATA_DISK/wormi-logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$LOG_ROOT/vh-wormi-$RUN_ID"
LOG_FILE="$LOG_DIR/train.log"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WORMI_VH_STAGE2_BATCH_SIZE="${WORMI_VH_STAGE2_BATCH_SIZE:-1}"
export WORMI_DATA_DISK="$DATA_DISK"
export WORMI_VH_DATA_ROOT="$DATA_ROOT"
export WORMI_WORLD_VH_OUTPUT_DIR="$WORLD_CKPT_ROOT"
export WORMI_VH_OUTPUT_DIR="$CKPT_ROOT"
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

echo "== WorMI VH stage-2 adapter training =="
echo "root:          $ROOT_DIR"
echo "curricula:     $CURRICULA_PATH"
echo "data root:     $DATA_ROOT"
echo "world ckpt:    $WORLD_CKPT_ROOT"
echo "output root:   $CKPT_ROOT"
echo "data disk:     $DATA_DISK"
echo "hf home:       $HF_HOME"
echo "uv cache:      $UV_CACHE_DIR"
echo "batch size:    $WORMI_VH_STAGE2_BATCH_SIZE"
echo "log file:      $LOG_FILE"
echo "started:       $(date -Is)"

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
  if [[ ! -d "$WORLD_CKPT_ROOT/$scene/last" ]]; then
    echo "ERROR: missing stage-1 world model $WORLD_CKPT_ROOT/$scene/last" >&2
    exit 1
  fi
done

for eval_dir in eval_col_1_seen_seen eval_col_2_seen_unseen eval_col_3_unseen_unseen; do
  if [[ ! -e "$DATA_ROOT/$eval_dir/test.jsonl" ]]; then
    echo "ERROR: missing $DATA_ROOT/$eval_dir/test.jsonl" >&2
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
"$WORMI_CMD" train \
  --curricula_path "$CURRICULA_PATH" \
  >> "$LOG_FILE" 2>&1
set +x

echo "finished: $(date -Is)"
