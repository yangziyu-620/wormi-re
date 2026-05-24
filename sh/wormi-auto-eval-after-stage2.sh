#!/usr/bin/env bash
set -euo pipefail

# Wait for a detached VH stage-2 run to finish, then run Table-1 offline eval
# and VirtualHome rollout eval sequentially. Intended to be launched detached.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TRAIN_PID="${TRAIN_PID:?TRAIN_PID is required}"
DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
DATA_ROOT="${DATA_ROOT:-$DATA_DISK/wormi-data/virtualhome}"
WORLD_CKPT_ROOT="${WORLD_CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/world-vh-taskaware-split-20260520}"
WORMI_CKPT_ROOT="${WORMI_CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/wormi-vh-taskaware-split-20260520}"
MODEL_NAME="${MODEL_NAME:-$WORMI_CKPT_ROOT/wormi-vh-n6/last}"
RUN_ID="${RUN_ID:-taskaware-auto-eval-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$DATA_DISK/wormi-logs}"
WATCH_DIR="$LOG_ROOT/vh-auto-eval-$RUN_ID"
WATCH_LOG="$WATCH_DIR/watch.log"
TRAIN_LOG="${TRAIN_LOG:-$LOG_ROOT/vh-wormi-taskaware-split-stage2-bs1-ga4-20260520/train.log}"
TABLE1_OUTPUT="${TABLE1_OUTPUT:-$WORMI_CKPT_ROOT/wormi-vh-n6/table1-auto-$RUN_ID}"
ROLLOUT_OUTPUT="${ROLLOUT_OUTPUT:-$WORMI_CKPT_ROOT/wormi-vh-n6/vh-rollout-auto-$RUN_ID}"
QUICK_NUM_SAMPLES="${QUICK_NUM_SAMPLES:-8}"
QUICK_MIN_SEEN_SEEN_SR="${QUICK_MIN_SEEN_SEEN_SR:-0.50}"
QUICK_OUTPUT="${QUICK_OUTPUT:-$WORMI_CKPT_ROOT/wormi-vh-n6/table1-quick-$RUN_ID}"
POLL_SECONDS="${POLL_SECONDS:-300}"

mkdir -p "$WATCH_DIR"

log() {
  echo "[$(date -Is)] $*" | tee -a "$WATCH_LOG"
}

log "auto eval watcher started"
log "train pid:     $TRAIN_PID"
log "train log:     $TRAIN_LOG"
log "model:         $MODEL_NAME"
log "world ckpt:    $WORLD_CKPT_ROOT"
log "wormi ckpt:    $WORMI_CKPT_ROOT"
log "table1 output: $TABLE1_OUTPUT"
log "rollout out:   $ROLLOUT_OUTPUT"
log "quick output:  $QUICK_OUTPUT"
log "quick gate:    n=$QUICK_NUM_SAMPLES min_seen_seen_sr=$QUICK_MIN_SEEN_SEEN_SR"

while kill -0 "$TRAIN_PID" >/dev/null 2>&1; do
  log "stage-2 still running; sleeping ${POLL_SECONDS}s"
  sleep "$POLL_SECONDS"
done

log "stage-2 pid exited; waiting briefly for wrapper file flush"
sleep 30

if grep -aE "Traceback|CUDA out of memory|RuntimeError:" "$TRAIN_LOG" >/dev/null 2>&1; then
  log "ERROR: stage-2 train log contains an exception marker; skipping eval"
  grep -aE "Traceback|CUDA out of memory|RuntimeError:" "$TRAIN_LOG" | tail -20 | tee -a "$WATCH_LOG" || true
  exit 1
fi

if [[ ! -d "$MODEL_NAME" ]]; then
  log "ERROR: model dir not found after training: $MODEL_NAME"
  exit 1
fi

if [[ "$QUICK_NUM_SAMPLES" != "0" ]]; then
  log "starting quick Table-1 gate"
  RUN_ID="${RUN_ID}-quick" \
  DATA_DISK="$DATA_DISK" \
  DATA_ROOT="$DATA_ROOT" \
  WORLD_CKPT_ROOT="$WORLD_CKPT_ROOT" \
  WORMI_CKPT_ROOT="$WORMI_CKPT_ROOT" \
  MODEL_NAME="$MODEL_NAME" \
  OUTPUT_PATH="$QUICK_OUTPUT" \
  NUM_SAMPLES="$QUICK_NUM_SAMPLES" \
  bash sh/wormi-eval-vh-table1.sh 2>&1 | tee -a "$WATCH_LOG"

  if [[ ! -f "$QUICK_OUTPUT/table1-summary.tsv" ]]; then
    log "ERROR: quick gate summary missing: $QUICK_OUTPUT/table1-summary.tsv"
    exit 1
  fi
  quick_seen_seen_sr=$(awk -F '\t' '$1 == "col_1_seen_seen" {print $4}' "$QUICK_OUTPUT/table1-summary.tsv")
  log "quick gate seen-seen SR: $quick_seen_seen_sr"
  python - "$quick_seen_seen_sr" "$QUICK_MIN_SEEN_SEEN_SR" <<'PY'
import sys
sr = float(sys.argv[1])
threshold = float(sys.argv[2])
if sr < threshold:
    raise SystemExit(1)
PY
  log "quick gate passed"
fi

log "starting Table-1 eval"
RUN_ID="${RUN_ID}-table1" \
DATA_DISK="$DATA_DISK" \
DATA_ROOT="$DATA_ROOT" \
WORLD_CKPT_ROOT="$WORLD_CKPT_ROOT" \
WORMI_CKPT_ROOT="$WORMI_CKPT_ROOT" \
MODEL_NAME="$MODEL_NAME" \
OUTPUT_PATH="$TABLE1_OUTPUT" \
bash sh/wormi-eval-vh-table1.sh 2>&1 | tee -a "$WATCH_LOG"
log "Table-1 eval finished"

log "starting rollout eval"
RUN_ID="${RUN_ID}-rollout" \
DATA_DISK="$DATA_DISK" \
DATA_ROOT="$DATA_ROOT" \
WORLD_CKPT_ROOT="$WORLD_CKPT_ROOT" \
WORMI_CKPT_ROOT="$WORMI_CKPT_ROOT" \
MODEL_NAME="$MODEL_NAME" \
OUTPUT_PATH="$ROLLOUT_OUTPUT" \
bash sh/wormi-eval-vh-rollout.sh 2>&1 | tee -a "$WATCH_LOG"
log "rollout eval finished"

log "auto eval complete"
