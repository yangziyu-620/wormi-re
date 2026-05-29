#!/usr/bin/env bash
set -euo pipefail

# Resume evaluation for the completed paperlike TMoW compact VirtualHome run.
# This script does not train. It evaluates the existing stage-2 `last` checkpoint.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-paperlike-tmow-compact-aa-fill17-20260528}"
DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
DATA_ROOT="${DATA_ROOT:-$DATA_DISK/wormi-data/virtualhome-$RUN_TAG}"
WORLD_CKPT_ROOT="${WORLD_CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/world-vh-$RUN_TAG}"
WORMI_CKPT_ROOT="${WORMI_CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/wormi-vh-$RUN_TAG}"
MODEL_NAME="${MODEL_NAME:-$WORMI_CKPT_ROOT/wormi-vh-n6/last}"
TABLE1_OUTPUT="${TABLE1_OUTPUT:-$WORMI_CKPT_ROOT/wormi-vh-n6/table1-$RUN_TAG}"
ROLLOUT_OUTPUT="${ROLLOUT_OUTPUT:-$WORMI_CKPT_ROOT/wormi-vh-n6/vh-rollout-$RUN_TAG}"
VH_SRC="${VH_SRC:-$DATA_DISK/wormi-data/virtualhome-src}"
SCENE_INITS_JSON="${SCENE_INITS_JSON:-$DATA_DISK/wormi-data/scene-inits/init_graphs_20_semantic.json}"
LOG_DIR="${LOG_DIR:-$DATA_DISK/wormi-logs/vh-eval-$RUN_TAG}"
STATUS_FILE="$LOG_DIR/status.tsv"
RUN_LOG="$LOG_DIR/launch.log"

mkdir -p "$LOG_DIR"

if [[ ! -f "$STATUS_FILE" ]]; then
  printf "time\tstage\tstatus\n" > "$STATUS_FILE"
fi

log_status() {
  printf "%s\t%s\t%s\n" "$(date -Is)" "$1" "$2" >> "$STATUS_FILE"
}

on_error() {
  local rc=$?
  log_status "${CURRENT_STAGE:-unknown}" "failed_rc_$rc"
  exit "$rc"
}
trap on_error ERR

exec > >(tee -a "$RUN_LOG") 2>&1

echo "== WorMI VH completed-checkpoint eval resume =="
echo "run tag:       $RUN_TAG"
echo "root:          $ROOT_DIR"
echo "data root:     $DATA_ROOT"
echo "world ckpt:    $WORLD_CKPT_ROOT"
echo "model:         $MODEL_NAME"
echo "table1 out:    $TABLE1_OUTPUT"
echo "rollout out:   $ROLLOUT_OUTPUT"
echo "log dir:       $LOG_DIR"
echo "started:       $(date -Is)"

if [[ ! -d "$MODEL_NAME" ]]; then
  echo "ERROR: missing trained stage-2 model: $MODEL_NAME" >&2
  exit 1
fi

CURRENT_STAGE=table1
if [[ "${FORCE_EVAL:-0}" != "1" && -f "$TABLE1_OUTPUT/table1-summary.tsv" ]]; then
  log_status "$CURRENT_STAGE" skipped_existing
  echo "Skipping table1; summary already exists: $TABLE1_OUTPUT/table1-summary.tsv"
else
  log_status "$CURRENT_STAGE" start
  RUN_ID="$RUN_TAG-table1" \
  DATA_DISK="$DATA_DISK" \
  DATA_ROOT="$DATA_ROOT" \
  WORLD_CKPT_ROOT="$WORLD_CKPT_ROOT" \
  WORMI_CKPT_ROOT="$WORMI_CKPT_ROOT" \
  MODEL_NAME="$MODEL_NAME" \
  OUTPUT_PATH="$TABLE1_OUTPUT" \
  bash sh/wormi-eval-vh-table1.sh
  log_status "$CURRENT_STAGE" done
fi

CURRENT_STAGE=rollout
if [[ "${FORCE_EVAL:-0}" != "1" && -f "$ROLLOUT_OUTPUT/vh-rollout-summary.tsv" ]]; then
  log_status "$CURRENT_STAGE" skipped_existing
  echo "Skipping rollout; summary already exists: $ROLLOUT_OUTPUT/vh-rollout-summary.tsv"
else
  log_status "$CURRENT_STAGE" start
  RUN_ID="$RUN_TAG-rollout" \
  DATA_DISK="$DATA_DISK" \
  DATA_ROOT="$DATA_ROOT" \
  VH_SRC="$VH_SRC" \
  SCENE_INITS_JSON="$SCENE_INITS_JSON" \
  WORLD_CKPT_ROOT="$WORLD_CKPT_ROOT" \
  WORMI_CKPT_ROOT="$WORMI_CKPT_ROOT" \
  MODEL_NAME="$MODEL_NAME" \
  OUTPUT_PATH="$ROLLOUT_OUTPUT" \
  MAX_STEPS="${MAX_STEPS:-30}" \
  bash sh/wormi-eval-vh-rollout.sh
  log_status "$CURRENT_STAGE" done
fi

CURRENT_STAGE=pipeline
log_status "$CURRENT_STAGE" done
echo "finished:      $(date -Is)"
