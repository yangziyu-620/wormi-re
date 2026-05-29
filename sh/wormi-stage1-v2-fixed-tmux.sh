#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-paperlike-v2-fixed-stage1-tmux-20260527}"
DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
DATA_ROOT="${DATA_ROOT:-$DATA_DISK/wormi-data/virtualhome-paperlike-v2-fixed-20260527}"
CKPT_ROOT="${CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/world-vh-paperlike-v2-fixed-20260527}"
LOG_ROOT="${LOG_ROOT:-$DATA_DISK/wormi-logs}"
LOG_DIR="$LOG_ROOT/vh-world-$RUN_ID"
LOG_FILE="$LOG_DIR/train.log"

mkdir -p \
  "$LOG_DIR" "$CKPT_ROOT" "$DATA_DISK/hf-home" "$DATA_DISK/hf-home/hub" \
  "$DATA_DISK/hf-home/transformers" "$DATA_DISK/hf-home/datasets" \
  "$DATA_DISK/torch-cache" "$DATA_DISK/triton-cache" "$DATA_DISK/xdg-cache" \
  "$DATA_DISK/uv-cache" "$DATA_DISK/tmp"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "== WorMI VH paperlike-v2-fixed stage1 =="
echo "started:   $(date -Is)"
echo "root:      $ROOT_DIR"
echo "data root: $DATA_ROOT"
echo "ckpt root: $CKPT_ROOT"
echo "log file:  $LOG_FILE"

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

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WORMI_DATA_DISK="$DATA_DISK"
export WORMI_VH_DATA_ROOT="$DATA_ROOT"
export WORMI_WORLD_VH_OUTPUT_DIR="$CKPT_ROOT"
export WORMI_WORLD_VH_BATCH_SIZE="${WORMI_WORLD_VH_BATCH_SIZE:-2}"
export HF_HOME="${HF_HOME:-$DATA_DISK/hf-home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_HOME="${TORCH_HOME:-$DATA_DISK/torch-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$DATA_DISK/triton-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_DISK/xdg-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$DATA_DISK/uv-cache}"
export TMPDIR="${TMPDIR:-$DATA_DISK/tmp}"

echo "batch size: ${WORMI_WORLD_VH_BATCH_SIZE}"
echo "command:    .venv/bin/wormi world train --curricula_path tools/world_curricula_vh.py"

.venv/bin/wormi world train --curricula_path tools/world_curricula_vh.py

echo "finished:  $(date -Is)"
