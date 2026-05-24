#!/usr/bin/env bash
set -euo pipefail

# Launch VH stage-2 training detached from the current shell/Codex exec
# session. The training script itself writes the main train.log; this wrapper
# only captures launcher stdout/stderr and records the detached process PID.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
LOG_ROOT="${LOG_ROOT:-$DATA_DISK/wormi-logs}"
RUN_ID="${RUN_ID:-semantic1023-rerun-stage2-bs1-detached-$(date +%Y%m%d_%H%M%S)}"
LAUNCH_DIR="$LOG_ROOT/vh-wormi-$RUN_ID"
LAUNCH_LOG="$LAUNCH_DIR/launch.log"
PID_FILE="$LAUNCH_DIR/pid"

mkdir -p "$LAUNCH_DIR"

TRAIN_PATTERN="[.]venv/bin/wormi train|bash sh/wormi-train-vh-wormi.sh"

if pgrep -af "$TRAIN_PATTERN" >/dev/null; then
  echo "ERROR: an existing WorMI training process is running:" >&2
  pgrep -af "$TRAIN_PATTERN" >&2
  exit 1
fi

setsid env \
  RUN_ID="$RUN_ID" \
  DATA_DISK="$DATA_DISK" \
  DATA_ROOT="${DATA_ROOT:-$DATA_DISK/wormi-data/virtualhome}" \
  WORLD_CKPT_ROOT="${WORLD_CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/world-vh-semantic1023-rerun}" \
  CKPT_ROOT="${CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/wormi-vh-semantic1023-rerun}" \
  WORMI_VH_STAGE2_BATCH_SIZE="${WORMI_VH_STAGE2_BATCH_SIZE:-1}" \
  WORMI_VH_STAGE2_GRADIENT_ACCUMULATION_STEPS="${WORMI_VH_STAGE2_GRADIENT_ACCUMULATION_STEPS:-4}" \
  WORMI_SEQUENTIAL_META_LEARNING="${WORMI_SEQUENTIAL_META_LEARNING:-0}" \
  WORMI_VH_STAGE2_INNER_STEPS="${WORMI_VH_STAGE2_INNER_STEPS:-30}" \
  WORMI_VH_STAGE2_META_STEPS="${WORMI_VH_STAGE2_META_STEPS:-8}" \
  bash sh/wormi-train-vh-wormi.sh \
  >"$LAUNCH_LOG" 2>&1 < /dev/null &

pid=$!
echo "$pid" > "$PID_FILE"

echo "launched pid: $pid"
echo "run id:       $RUN_ID"
echo "launch log:   $LAUNCH_LOG"
echo "pid file:     $PID_FILE"
echo "train log:    $LAUNCH_DIR/train.log"
