#!/usr/bin/env bash
set -euo pipefail

BACKUP_PID="${BACKUP_PID:-18724}"
REMOTE="${REMOTE:-gdrive_ghub:}"
REMOTE_ROOT="${REMOTE_ROOT:-gdrive_ghub/vh-paperlike-prebackup-20260526}"
DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
LOG_DIR="${LOG_DIR:-$DATA_DISK/wormi-logs/cleanup-vh-paperlike-prebackup-20260526}"
CHECKPOINT_DIR="$DATA_DISK/wormi-checkpoints"

mkdir -p "$LOG_DIR"
log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_DIR/cleanup.log"
}

log "waiting for checkpoint backup pid=$BACKUP_PID"
while kill -0 "$BACKUP_PID" 2>/dev/null; do
  sleep 300
  log "checkpoint backup still running"
done
log "checkpoint backup pid exited; verifying remote backup before cleanup"

rclone check "$CHECKPOINT_DIR" "$REMOTE$REMOTE_ROOT/02_checkpoints/wormi-checkpoints" \
  --one-way --size-only > "$LOG_DIR/rclone-check-checkpoints.log" 2>&1
log "rclone check passed"

rclone size "$REMOTE$REMOTE_ROOT" > "$LOG_DIR/remote-size.txt" 2>&1 || true

du -sh "$CHECKPOINT_DIR" > "$LOG_DIR/local-checkpoints-before-delete.txt" 2>&1 || true
log "deleting local checkpoint contents under $CHECKPOINT_DIR"
find "$CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
mkdir -p "$CHECKPOINT_DIR"
du -sh "$CHECKPOINT_DIR" > "$LOG_DIR/local-checkpoints-after-delete.txt" 2>&1 || true

date -Is > "$LOG_DIR/DONE"
rclone copyto "$LOG_DIR/rclone-check-checkpoints.log" "$REMOTE$REMOTE_ROOT/RCLONE_CHECK_CHECKPOINTS.txt" --retries 8 --low-level-retries 20 > "$LOG_DIR/upload-check-log.log" 2>&1 || true
rclone copyto "$LOG_DIR/DONE" "$REMOTE$REMOTE_ROOT/CLEANUP_DONE" --retries 8 --low-level-retries 20 > "$LOG_DIR/upload-done.log" 2>&1 || true
log "cleanup complete"
