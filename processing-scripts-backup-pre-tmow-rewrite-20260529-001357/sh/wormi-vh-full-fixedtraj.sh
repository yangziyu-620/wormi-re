#!/usr/bin/env bash
set -euo pipefail

# Full VirtualHome reproduction run after fixing trajectory-level data split.
# Runs:
#   1. Rebuild main VH jsonl data under the data disk.
#   2. Retrain six scene-keyed world models.
#   3. Retrain WorMI stage-2 adapters.
#   4. Run VirtualHome rollout evaluation.
#
# Intended for unattended execution, e.g.:
#   nohup bash sh/wormi-vh-full-fixedtraj.sh > /root/autodl-tmp/wormi-logs/vh-full-fixedtraj.nohup.log 2>&1 &

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
RUN_ID="${RUN_ID:-fixedtraj-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$DATA_DISK/wormi-logs}"
LOG_DIR="$LOG_ROOT/vh-full-$RUN_ID"
LOG_FILE="$LOG_DIR/full.log"

DATA_ROOT="${DATA_ROOT:-$DATA_DISK/wormi-data/virtualhome}"
WORLD_CKPT_ROOT="${WORLD_CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/world-vh-fixedtraj}"
WORMI_CKPT_ROOT="${WORMI_CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/wormi-vh-fixedtraj}"
ROLLOUT_OUTPUT="${ROLLOUT_OUTPUT:-$WORMI_CKPT_ROOT/wormi-vh-n6/vh-rollout}"

mkdir -p "$LOG_DIR"

{
  echo "== WorMI VH full fixed-trajectory run =="
  echo "root:         $ROOT_DIR"
  echo "run id:       $RUN_ID"
  echo "data root:    $DATA_ROOT"
  echo "world ckpt:   $WORLD_CKPT_ROOT"
  echo "wormi ckpt:   $WORMI_CKPT_ROOT"
  echo "rollout out:  $ROLLOUT_OUTPUT"
  echo "log file:     $LOG_FILE"
  echo "started:      $(date -Is)"
} | tee "$LOG_FILE"

export DATA_DISK
export DATA_ROOT
export RUN_ID
export WORLD_CKPT_ROOT
export WORMI_CKPT_ROOT

echo "== Stage 0: rebuild VH data ==" | tee -a "$LOG_FILE"
OUTPUT_DIR="$DATA_ROOT" \
  bash sh/wormi-build-vh-data.sh 2>&1 | tee -a "$LOG_FILE"

echo "== Stage 1: train VH world models ==" | tee -a "$LOG_FILE"
WORMI_WORLD_VH_OUTPUT_DIR="$WORLD_CKPT_ROOT" \
  CKPT_ROOT="$WORLD_CKPT_ROOT" \
  bash sh/wormi-train-vh-world.sh 2>&1 | tee -a "$LOG_FILE"

echo "== Stage 2: train WorMI adapters ==" | tee -a "$LOG_FILE"
WORMI_WORLD_VH_OUTPUT_DIR="$WORLD_CKPT_ROOT" \
  WORMI_VH_OUTPUT_DIR="$WORMI_CKPT_ROOT" \
  WORLD_CKPT_ROOT="$WORLD_CKPT_ROOT" \
  CKPT_ROOT="$WORMI_CKPT_ROOT" \
  bash sh/wormi-train-vh-wormi.sh 2>&1 | tee -a "$LOG_FILE"

echo "== Stage 3: rollout eval ==" | tee -a "$LOG_FILE"
WORMI_WORLD_VH_OUTPUT_DIR="$WORLD_CKPT_ROOT" \
  WORMI_VH_OUTPUT_DIR="$WORMI_CKPT_ROOT" \
  WORLD_CKPT_ROOT="$WORLD_CKPT_ROOT" \
  WORMI_CKPT_ROOT="$WORMI_CKPT_ROOT" \
  MODEL_NAME="$WORMI_CKPT_ROOT/wormi-vh-n6/last" \
  OUTPUT_PATH="$ROLLOUT_OUTPUT" \
  bash sh/wormi-eval-vh-rollout.sh 2>&1 | tee -a "$LOG_FILE"

{
  echo "finished: $(date -Is)"
  echo "summary:  $ROLLOUT_OUTPUT/vh-rollout-summary.tsv"
} | tee -a "$LOG_FILE"
