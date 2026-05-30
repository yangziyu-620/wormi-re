#!/usr/bin/env bash
set -euo pipefail

# VirtualHome rollout evaluation from the trained stage-2 WorMI model.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CURRICULA_PATH="${CURRICULA_PATH:-tools/wormi_curricula_vh.py}"
DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
DATA_ROOT="${DATA_ROOT:-$DATA_DISK/wormi-data/virtualhome-realtasks-v3-20260530}"
VH_SRC="${VH_SRC:-$DATA_DISK/wormi-data/virtualhome-src}"
# Prefer the dataset-local scene_inits.json when available (e.g. v3 flat layout).
# Fall back to the legacy shared path for older datasets that do not embed one.
if [[ -z "${SCENE_INITS_JSON:-}" ]]; then
  if [[ -f "$DATA_ROOT/scene_inits.json" ]]; then
    SCENE_INITS_JSON="$DATA_ROOT/scene_inits.json"
  else
    SCENE_INITS_JSON="$DATA_DISK/wormi-data/scene-inits/init_graphs_20_semantic.json"
  fi
fi
WORLD_CKPT_ROOT="${WORLD_CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/world-vh}"
WORMI_CKPT_ROOT="${WORMI_CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/wormi-vh}"
MODEL_NAME="${MODEL_NAME:-$WORMI_CKPT_ROOT/wormi-vh-n6/last}"
OUTPUT_PATH="${OUTPUT_PATH:-$WORMI_CKPT_ROOT/wormi-vh-n6/vh-rollout}"
LOG_ROOT="${LOG_ROOT:-$DATA_DISK/wormi-logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$LOG_ROOT/vh-rollout-$RUN_ID"
LOG_FILE="$LOG_DIR/eval.log"
MAX_STEPS="${MAX_STEPS:-30}"
TOP_P="${TOP_P:-1.0}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
if [[ "${WORMI_VH_ROLLOUT_ALLOW_SAMPLING:-0}" == "1" ]]; then
  TEMPERATURE="${TEMPERATURE:-1.0}"
else
  TEMPERATURE="0.0"
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WORMI_DATA_DISK="$DATA_DISK"
export WORMI_VH_DATA_ROOT="$DATA_ROOT"
export WORMI_WORLD_VH_OUTPUT_DIR="$WORLD_CKPT_ROOT"
export WORMI_VH_OUTPUT_DIR="$WORMI_CKPT_ROOT"
export WORMI_VH_SRC="$VH_SRC"
export WORMI_VH_SCENE_INITS_JSON="$SCENE_INITS_JSON"
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
  "$LOG_DIR" "$OUTPUT_PATH" "$HF_HOME" "$HF_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$TMPDIR"

echo "== WorMI VH rollout eval =="
echo "root:        $ROOT_DIR"
echo "curricula:   $CURRICULA_PATH"
echo "model:       $MODEL_NAME"
echo "data root:   $DATA_ROOT"
echo "vh src:      $VH_SRC"
echo "scene init:  $SCENE_INITS_JSON"
echo "world ckpt:  $WORLD_CKPT_ROOT"
echo "output:      $OUTPUT_PATH"
echo "max steps:   $MAX_STEPS"
echo "temperature: $TEMPERATURE"
echo "log file:    $LOG_FILE"
echo "started:     $(date -Is)"

if [[ ! -f "$CURRICULA_PATH" ]]; then
  echo "ERROR: curricula file not found: $CURRICULA_PATH" >&2
  exit 1
fi

if [[ ! -d "$MODEL_NAME" ]]; then
  echo "ERROR: trained WorMI model not found: $MODEL_NAME" >&2
  exit 1
fi

if [[ ! -d "$VH_SRC" ]]; then
  echo "ERROR: VirtualHome source tree not found: $VH_SRC" >&2
  exit 1
fi

if [[ ! -f "$SCENE_INITS_JSON" ]]; then
  echo "ERROR: scene init cache not found: $SCENE_INITS_JSON" >&2
  exit 1
fi

for scene in scene_0 scene_1 scene_2 scene_3 scene_4 scene_5; do
  if [[ ! -d "$WORLD_CKPT_ROOT/$scene/last" ]]; then
    echo "ERROR: missing world model $WORLD_CKPT_ROOT/$scene/last" >&2
    exit 1
  fi
done

for eval_dir in eval_col_1_seen_seen eval_col_2_seen_unseen eval_col_3_unseen_unseen; do
  if [[ ! -e "$DATA_ROOT/$eval_dir/test.jsonl" ]]; then
    echo "ERROR: missing $DATA_ROOT/$eval_dir/test.jsonl" >&2
    exit 1
  fi
done

WORMI_CMD="${WORMI_CMD:-.venv/bin/wormi}"
if [[ ! -x "$WORMI_CMD" ]]; then
  echo "ERROR: wormi command not found or not executable: $WORMI_CMD" >&2
  exit 1
fi

CMD=(
  "$WORMI_CMD" eval-vh-rollout
  --curricula_path "$CURRICULA_PATH"
  --model_name "$MODEL_NAME"
  --output_path "$OUTPUT_PATH"
  --scene_inits_json "$SCENE_INITS_JSON"
  --vh_src "$VH_SRC"
  --max_steps "$MAX_STEPS"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
)

if [[ -n "$NUM_SAMPLES" ]]; then
  CMD+=(--num_samples "$NUM_SAMPLES")
fi

set -x
"${CMD[@]}" >> "$LOG_FILE" 2>&1
set +x

echo "finished: $(date -Is)"
