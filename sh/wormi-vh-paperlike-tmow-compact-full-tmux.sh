#!/usr/bin/env bash
set -euo pipefail

# Detached launcher for the TMoW-compact full pipeline.
# Optional:
#   WAIT_FOR_PID=180560 bash sh/wormi-vh-paperlike-tmow-compact-full-tmux.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SCRIPT_PATH="$ROOT_DIR/sh/wormi-vh-paperlike-tmow-compact-full-tmux.sh"
SESSION_NAME="${SESSION_NAME:-wormi_tmow_compact_full}"
DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
RUN_BASE="${RUN_BASE:-paperlike-tmow-compact-fill17-20260528}"
LOG_ROOT="${LOG_ROOT:-$DATA_DISK/wormi-logs}"
LOG_DIR="$LOG_ROOT/vh-pipeline-$RUN_BASE-detached"
LAUNCH_LOG="$LOG_DIR/launch.log"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-300}"

if [[ "${1:-}" == "--run" ]]; then
  mkdir -p "$LOG_DIR"
  exec > >(tee -a "$LAUNCH_LOG") 2>&1

  echo "== TMoW-compact pipeline detached runner =="
  echo "started:      $(date -Is)"
  echo "root:         $ROOT_DIR"
  echo "run base:     $RUN_BASE"
  echo "launch log:   $LAUNCH_LOG"
  echo "wait pid:     ${WAIT_FOR_PID:-none}"
  echo "poll seconds: $WAIT_POLL_SECONDS"

  if [[ -n "$WAIT_FOR_PID" ]]; then
    while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
      echo "waiting for PID $WAIT_FOR_PID: $(date -Is)"
      sleep "$WAIT_POLL_SECONDS"
    done
    echo "waited PID $WAIT_FOR_PID ended: $(date -Is)"
  fi

  echo "starting compact full pipeline: $(date -Is)"
  WORMI_SEQUENTIAL_META_LEARNING="${WORMI_SEQUENTIAL_META_LEARNING:-1}" \
    bash sh/wormi-vh-paperlike-tmow-compact-full.sh
  echo "compact full pipeline finished: $(date -Is)"
  exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux is not available on PATH" >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "ERROR: tmux session already exists: $SESSION_NAME" >&2
  echo "Attach with: tmux attach -t $SESSION_NAME" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
tmux new-session -d -s "$SESSION_NAME" \
  "cd '$ROOT_DIR' && DATA_DISK='$DATA_DISK' RUN_BASE='$RUN_BASE' LOG_ROOT='$LOG_ROOT' WAIT_FOR_PID='$WAIT_FOR_PID' WAIT_POLL_SECONDS='$WAIT_POLL_SECONDS' WORMI_SEQUENTIAL_META_LEARNING='${WORMI_SEQUENTIAL_META_LEARNING:-1}' bash '$SCRIPT_PATH' --run"

echo "started tmux session: $SESSION_NAME"
echo "attach: tmux attach -t $SESSION_NAME"
echo "log:    $LAUNCH_LOG"
if [[ -n "$WAIT_FOR_PID" ]]; then
  echo "queued after PID: $WAIT_FOR_PID"
fi
