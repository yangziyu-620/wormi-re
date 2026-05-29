#!/usr/bin/env bash
set -euo pipefail

# Full WorMI VirtualHome pipeline for the independent TMoW-style compact data.
# Stage 2 defaults to sequential Reptile, the paper-faithful meta update.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_BASE="${RUN_BASE:-paperlike-tmow-compact-fill17-20260528}"
DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
DATA_ROOT="${DATA_ROOT:-$DATA_DISK/wormi-data/virtualhome-$RUN_BASE}"
WORLD_CKPT_ROOT="${WORLD_CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/world-vh-$RUN_BASE}"
WORMI_CKPT_ROOT="${WORMI_CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/wormi-vh-$RUN_BASE}"
VH_SRC="${VH_SRC:-$DATA_DISK/wormi-data/virtualhome-src}"
SCENE_INITS_JSON="${SCENE_INITS_JSON:-$DATA_DISK/wormi-data/scene-inits/init_graphs_20_semantic.json}"
LOG_ROOT="${LOG_ROOT:-$DATA_DISK/wormi-logs}"
PIPE_DIR="$LOG_ROOT/vh-pipeline-$RUN_BASE"
STATUS_FILE="$PIPE_DIR/status.tsv"
PIPE_LOG="$PIPE_DIR/pipeline.log"
MODEL_NAME="${MODEL_NAME:-$WORMI_CKPT_ROOT/wormi-vh-n6/last}"
TABLE1_OUTPUT="${TABLE1_OUTPUT:-$WORMI_CKPT_ROOT/wormi-vh-n6/table1-$RUN_BASE}"
ROLLOUT_OUTPUT="${ROLLOUT_OUTPUT:-$WORMI_CKPT_ROOT/wormi-vh-n6/vh-rollout-$RUN_BASE}"
VALIDATION_JSON="${VALIDATION_JSON:-reports/virtualhome/validation/vh-$RUN_BASE-validation-2026-05-28.json}"
TOKENIZER="${TOKENIZER:-unsloth/Llama-3.2-3B-Instruct}"

export HF_HOME="${HF_HOME:-$DATA_DISK/hf-home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_DISK/xdg-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$DATA_DISK/uv-cache}"
export TMPDIR="${TMPDIR:-$DATA_DISK/tmp}"

mkdir -p \
  "$PIPE_DIR" "$WORLD_CKPT_ROOT" "$WORMI_CKPT_ROOT" \
  "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
  "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$TMPDIR"
printf 'time\tstage\tstatus\n' > "$STATUS_FILE"
exec > >(tee -a "$PIPE_LOG") 2>&1

log_status() {
  printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >> "$STATUS_FILE"
}

on_error() {
  log_status "${CURRENT_STAGE:-pipeline}" failed
}

STAGE1_CLEANUP_PID=""
STAGE1_CLEANUP_DONE="$PIPE_DIR/stage1-cleanup.done"
STAGE1_CLEANUP_LOG="$PIPE_DIR/stage1-cleanup.log"

start_stage1_cleanup_watcher() {
  if [[ "${WORMI_CLEAN_WORLD_INTERMEDIATE_CHECKPOINTS:-1}" != "1" ]]; then
    return
  fi
  rm -f "$STAGE1_CLEANUP_DONE"
  (
    echo "stage1 cleanup watcher start: $(date -Is)" >> "$STAGE1_CLEANUP_LOG"
    while [[ ! -e "$STAGE1_CLEANUP_DONE" ]]; do
      cleanup_completed_world_checkpoints
      sleep "${WORMI_WORLD_CKPT_CLEANUP_INTERVAL_SECONDS:-120}"
    done
    echo "stage1 cleanup watcher stop: $(date -Is)" >> "$STAGE1_CLEANUP_LOG"
  ) &
  STAGE1_CLEANUP_PID="$!"
}

cleanup_completed_world_checkpoints() {
  for scene_dir in "$WORLD_CKPT_ROOT"/scene_*; do
    [[ -d "$scene_dir/last" ]] || continue
    for ckpt in "$scene_dir"/checkpoint-*; do
      [[ -e "$ckpt" ]] || continue
      echo "removing $ckpt at $(date -Is)" >> "$STAGE1_CLEANUP_LOG"
      rm -rf "$ckpt"
    done
  done
}

stop_stage1_cleanup_watcher() {
  cleanup_completed_world_checkpoints
  [[ -n "${STAGE1_CLEANUP_PID:-}" ]] || return
  touch "$STAGE1_CLEANUP_DONE"
  wait "$STAGE1_CLEANUP_PID" || true
  cleanup_completed_world_checkpoints
  STAGE1_CLEANUP_PID=""
}

on_exit() {
  stop_stage1_cleanup_watcher
}
trap on_error ERR
trap on_exit EXIT

echo "== WorMI VH paperlike TMoW-compact full pipeline =="
echo "started:      $(date -Is)"
echo "root:         $ROOT_DIR"
echo "data root:    $DATA_ROOT"
echo "world ckpt:   $WORLD_CKPT_ROOT"
echo "wormi ckpt:   $WORMI_CKPT_ROOT"
echo "pipeline log: $PIPE_LOG"
echo "status file:  $STATUS_FILE"
echo "seq meta:     ${WORMI_SEQUENTIAL_META_LEARNING:-1}"
echo "tokenizer:    $TOKENIZER"

CURRENT_STAGE=preflight
log_status "$CURRENT_STAGE" start
.venv/bin/python tools/validate_virtualhome_dataset.py \
  --data-root "$DATA_ROOT" \
  --scene-inits-json "$SCENE_INITS_JSON" \
  --vh-src "$VH_SRC" \
  --check-loader \
  --check-chat-template \
  --tokenizer "$TOKENIZER" \
  --tokenizer-local-files-only \
  --output-json "$VALIDATION_JSON"
log_status "$CURRENT_STAGE" done

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  CURRENT_STAGE=pipeline
  log_status "$CURRENT_STAGE" preflight_only
  echo "PREFLIGHT_ONLY=1: stopping before training."
  exit 0
fi

first_missing_scene=6
for i in 0 1 2 3 4 5; do
  if [[ ! -d "$WORLD_CKPT_ROOT/scene_${i}/last" ]]; then
    first_missing_scene="$i"
    break
  fi
done

CURRENT_STAGE=stage1
if [[ "$first_missing_scene" -lt 6 ]]; then
  log_status "$CURRENT_STAGE" "start_scene_${first_missing_scene}"
  start_stage1_cleanup_watcher
  RUN_ID="$RUN_BASE-stage1" \
  DATA_DISK="$DATA_DISK" \
  DATA_ROOT="$DATA_ROOT" \
  CKPT_ROOT="$WORLD_CKPT_ROOT" \
  WORMI_WORLD_VH_BATCH_SIZE="${WORMI_WORLD_VH_BATCH_SIZE:-2}" \
  WORMI_VH_SCENE_START="$first_missing_scene" \
  WORMI_VH_SCENE_END=6 \
  bash sh/wormi-train-vh-world.sh
  stop_stage1_cleanup_watcher
  log_status "$CURRENT_STAGE" done
else
  log_status "$CURRENT_STAGE" skipped
fi

for i in 0 1 2 3 4 5; do
  if [[ ! -d "$WORLD_CKPT_ROOT/scene_${i}/last" ]]; then
    echo "ERROR: missing world model after stage1: $WORLD_CKPT_ROOT/scene_${i}/last" >&2
    exit 1
  fi
done

CURRENT_STAGE=stage2
if [[ ! -d "$MODEL_NAME" ]]; then
  log_status "$CURRENT_STAGE" start
  RUN_ID="$RUN_BASE-stage2" \
  DATA_DISK="$DATA_DISK" \
  DATA_ROOT="$DATA_ROOT" \
  WORLD_CKPT_ROOT="$WORLD_CKPT_ROOT" \
  CKPT_ROOT="$WORMI_CKPT_ROOT" \
  WORMI_VH_STAGE2_BATCH_SIZE="${WORMI_VH_STAGE2_BATCH_SIZE:-1}" \
  WORMI_VH_STAGE2_GRADIENT_ACCUMULATION_STEPS="${WORMI_VH_STAGE2_GRADIENT_ACCUMULATION_STEPS:-4}" \
  WORMI_SEQUENTIAL_META_LEARNING="${WORMI_SEQUENTIAL_META_LEARNING:-1}" \
  WORMI_VH_STAGE2_INNER_STEPS="${WORMI_VH_STAGE2_INNER_STEPS:-30}" \
  WORMI_VH_STAGE2_META_STEPS="${WORMI_VH_STAGE2_META_STEPS:-8}" \
  bash sh/wormi-train-vh-wormi.sh
  log_status "$CURRENT_STAGE" done
else
  log_status "$CURRENT_STAGE" skipped
fi

if [[ ! -d "$MODEL_NAME" ]]; then
  echo "ERROR: stage2 did not produce $MODEL_NAME" >&2
  exit 1
fi

CURRENT_STAGE=table1
log_status "$CURRENT_STAGE" start
RUN_ID="$RUN_BASE-table1" \
DATA_DISK="$DATA_DISK" \
DATA_ROOT="$DATA_ROOT" \
WORLD_CKPT_ROOT="$WORLD_CKPT_ROOT" \
WORMI_CKPT_ROOT="$WORMI_CKPT_ROOT" \
MODEL_NAME="$MODEL_NAME" \
OUTPUT_PATH="$TABLE1_OUTPUT" \
bash sh/wormi-eval-vh-table1.sh
log_status "$CURRENT_STAGE" done

CURRENT_STAGE=rollout
log_status "$CURRENT_STAGE" start
RUN_ID="$RUN_BASE-rollout" \
DATA_DISK="$DATA_DISK" \
DATA_ROOT="$DATA_ROOT" \
VH_SRC="$VH_SRC" \
SCENE_INITS_JSON="$SCENE_INITS_JSON" \
WORLD_CKPT_ROOT="$WORLD_CKPT_ROOT" \
WORMI_CKPT_ROOT="$WORMI_CKPT_ROOT" \
MODEL_NAME="$MODEL_NAME" \
OUTPUT_PATH="$ROLLOUT_OUTPUT" \
MAX_STEPS="${MAX_STEPS:-30}" \
TEMPERATURE="${TEMPERATURE:-1.0}" \
TOP_P="${TOP_P:-1.0}" \
bash sh/wormi-eval-vh-rollout.sh
log_status "$CURRENT_STAGE" done

CURRENT_STAGE=pipeline
log_status "$CURRENT_STAGE" done
echo "finished:     $(date -Is)"
echo "table1 out:   $TABLE1_OUTPUT"
echo "rollout out:  $ROLLOUT_OUTPUT"
