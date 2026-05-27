#!/usr/bin/env bash
set -euo pipefail

# Back up the current WorMI reproduction state before rebuilding the
# VirtualHome data into a paper-like house-configuration benchmark.
#
# This is intentionally broader than the previous single-run upload script:
# it preserves all current data, checkpoints, logs, reports, and code needed
# to audit old results before checkpoint cleanup frees local disk space.

RUN_NAME="${RUN_NAME:-vh-paperlike-prebackup-20260526}"
REMOTE="${REMOTE:-gdrive_ghub:}"
REMOTE_ROOT="${REMOTE_ROOT:-gdrive_ghub/$RUN_NAME}"
DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
REPO_ROOT="${REPO_ROOT:-/root/WorMI}"
STAGING_ROOT="${STAGING_ROOT:-$DATA_DISK/wormi-upload-staging/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$DATA_DISK/wormi-logs/upload-$RUN_NAME}"

UPLOAD_JOBS="${UPLOAD_JOBS:-3}"
RCLONE_TRANSFERS="${RCLONE_TRANSFERS:-4}"
RCLONE_CHECKERS="${RCLONE_CHECKERS:-8}"
DRIVE_CHUNK_SIZE="${DRIVE_CHUNK_SIZE:-256M}"

mkdir -p "$STAGING_ROOT" "$LOG_DIR"

RCLONE_COMMON=(
  --progress
  --transfers "$RCLONE_TRANSFERS"
  --checkers "$RCLONE_CHECKERS"
  --drive-chunk-size "$DRIVE_CHUNK_SIZE"
  --fast-list
  --create-empty-src-dirs
  --retries 8
  --low-level-retries 20
  --stats 30s
)

RCLONE_COPYTO_COMMON=(
  --progress
  --transfers "$RCLONE_TRANSFERS"
  --checkers "$RCLONE_CHECKERS"
  --drive-chunk-size "$DRIVE_CHUNK_SIZE"
  --fast-list
  --retries 8
  --low-level-retries 20
  --stats 30s
)

cat > "$STAGING_ROOT/00_README.md" <<EOF
# WorMI VirtualHome Pre-Paperlike Backup

Run: $RUN_NAME

Purpose: preserve the current data, checkpoints, logs, reports, and code before
rebuilding VirtualHome into a paper-like house-configuration benchmark.

## Layout

- \`01_data/wormi-data\`: all local data under \`$DATA_DISK/wormi-data\`.
- \`02_checkpoints/wormi-checkpoints\`: all local checkpoints under
  \`$DATA_DISK/wormi-checkpoints\`.
- \`03_logs/wormi-logs\`: all logs under \`$DATA_DISK/wormi-logs\`.
- \`04_reports/reports\`: local analysis reports.
- \`05_code_snapshot\`: \`tools\`, \`sh\`, \`wormi\`, and root config files.
- \`SOURCE_INVENTORY.tsv\`: local source file size inventory at backup time.
- \`UPLOAD_SOURCES.tsv\`: source-to-destination mapping.
- \`REMOTE_SIZE.txt\`: remote size after upload.
- \`RCLONE_CHECK.txt\`: rclone check result after upload.
- \`DONE\`: written only after all upload/check steps complete.

This backup is the restore point for old results before deleting local
checkpoints to free space.
EOF

cat > "$STAGING_ROOT/UPLOAD_SOURCES.tsv" <<EOF
destination	source
00_README.md	$STAGING_ROOT/00_README.md
UPLOAD_SOURCES.tsv	$STAGING_ROOT/UPLOAD_SOURCES.tsv
SOURCE_INVENTORY.tsv	$STAGING_ROOT/SOURCE_INVENTORY.tsv
01_data/wormi-data	$DATA_DISK/wormi-data
02_checkpoints/wormi-checkpoints	$DATA_DISK/wormi-checkpoints
03_logs/wormi-logs	$DATA_DISK/wormi-logs
04_reports/reports	$REPO_ROOT/reports
05_code_snapshot/tools	$REPO_ROOT/tools
05_code_snapshot/sh	$REPO_ROOT/sh
05_code_snapshot/wormi	$REPO_ROOT/wormi
05_code_snapshot/root_files/AGENTS.md	$REPO_ROOT/AGENTS.md
05_code_snapshot/root_files/Readme.md	$REPO_ROOT/Readme.md
05_code_snapshot/root_files/pyproject.toml	$REPO_ROOT/pyproject.toml
05_code_snapshot/root_files/uv.lock	$REPO_ROOT/uv.lock
EOF

echo "Generating source inventory..."
{
  printf "bytes\tmtime_epoch\tpath\n"
  find \
    "$DATA_DISK/wormi-data" \
    "$DATA_DISK/wormi-checkpoints" \
    "$DATA_DISK/wormi-logs" \
    "$REPO_ROOT/reports" \
    "$REPO_ROOT/tools" \
    "$REPO_ROOT/sh" \
    "$REPO_ROOT/wormi" \
    -type f -printf '%s\t%T@\t%p\n' \
    | sort
  for file in "$REPO_ROOT/AGENTS.md" "$REPO_ROOT/Readme.md" "$REPO_ROOT/pyproject.toml" "$REPO_ROOT/uv.lock"; do
    if [[ -f "$file" ]]; then
      stat -c '%s	%Y	%n' "$file"
    fi
  done
} > "$STAGING_ROOT/SOURCE_INVENTORY.tsv"

copy_one() {
  local src="$1"
  local dst="$2"
  local name
  name="$(echo "$dst" | tr '/:' '__')"
  echo "Uploading $src -> $REMOTE$REMOTE_ROOT/$dst"
  rclone copy "$src" "$REMOTE$REMOTE_ROOT/$dst" "${RCLONE_COMMON[@]}" \
    > "$LOG_DIR/$name.log" 2>&1
}

copy_file() {
  local src="$1"
  local dst_dir="$2"
  local name
  name="$(echo "$dst_dir" | tr '/:' '__')"
  echo "Uploading file $src -> $REMOTE$REMOTE_ROOT/$dst_dir"
  rclone copyto "$src" "$REMOTE$REMOTE_ROOT/$dst_dir/$(basename "$src")" "${RCLONE_COPYTO_COMMON[@]}" \
    > "$LOG_DIR/${name}_file.log" 2>&1
}

job_count() {
  local jobs_out
  jobs_out="$(jobs -rp || true)"
  if [[ -z "$jobs_out" ]]; then
    echo 0
  else
    printf '%s\n' "$jobs_out" | wc -l
  fi
}

run_limited() {
  while (( "$(job_count)" >= UPLOAD_JOBS )); do
    sleep 5
  done
  "$@" &
}

echo "Remote: $REMOTE$REMOTE_ROOT"
echo "Logs:   $LOG_DIR"

run_limited copy_file "$STAGING_ROOT/00_README.md" "."
run_limited copy_file "$STAGING_ROOT/UPLOAD_SOURCES.tsv" "."
run_limited copy_file "$STAGING_ROOT/SOURCE_INVENTORY.tsv" "."
run_limited copy_one "$DATA_DISK/wormi-data" "01_data/wormi-data"
run_limited copy_one "$DATA_DISK/wormi-checkpoints" "02_checkpoints/wormi-checkpoints"
run_limited copy_one "$DATA_DISK/wormi-logs" "03_logs/wormi-logs"
run_limited copy_one "$REPO_ROOT/reports" "04_reports/reports"
run_limited copy_one "$REPO_ROOT/tools" "05_code_snapshot/tools"
run_limited copy_one "$REPO_ROOT/sh" "05_code_snapshot/sh"
run_limited copy_one "$REPO_ROOT/wormi" "05_code_snapshot/wormi"
run_limited copy_file "$REPO_ROOT/AGENTS.md" "05_code_snapshot/root_files"
run_limited copy_file "$REPO_ROOT/Readme.md" "05_code_snapshot/root_files"
run_limited copy_file "$REPO_ROOT/pyproject.toml" "05_code_snapshot/root_files"
run_limited copy_file "$REPO_ROOT/uv.lock" "05_code_snapshot/root_files"

wait

echo "Computing remote size..."
rclone size "$REMOTE$REMOTE_ROOT" > "$STAGING_ROOT/REMOTE_SIZE.txt"
rclone copyto "$STAGING_ROOT/REMOTE_SIZE.txt" "$REMOTE$REMOTE_ROOT/REMOTE_SIZE.txt" "${RCLONE_COPYTO_COMMON[@]}" \
  > "$LOG_DIR/REMOTE_SIZE.log" 2>&1

echo "Running rclone check..."
{
  echo "# Data"
  rclone check "$DATA_DISK/wormi-data" "$REMOTE$REMOTE_ROOT/01_data/wormi-data" --one-way --size-only
  echo "# Checkpoints"
  rclone check "$DATA_DISK/wormi-checkpoints" "$REMOTE$REMOTE_ROOT/02_checkpoints/wormi-checkpoints" --one-way --size-only
  echo "# Logs"
  rclone check "$DATA_DISK/wormi-logs" "$REMOTE$REMOTE_ROOT/03_logs/wormi-logs" --one-way --size-only
  echo "# Reports"
  rclone check "$REPO_ROOT/reports" "$REMOTE$REMOTE_ROOT/04_reports/reports" --one-way --size-only
} > "$STAGING_ROOT/RCLONE_CHECK.txt" 2>&1
rclone copyto "$STAGING_ROOT/RCLONE_CHECK.txt" "$REMOTE$REMOTE_ROOT/RCLONE_CHECK.txt" "${RCLONE_COPYTO_COMMON[@]}" \
  > "$LOG_DIR/RCLONE_CHECK.log" 2>&1

date -Is > "$STAGING_ROOT/DONE"
rclone copyto "$STAGING_ROOT/DONE" "$REMOTE$REMOTE_ROOT/DONE" "${RCLONE_COPYTO_COMMON[@]}" \
  > "$LOG_DIR/DONE.log" 2>&1

echo "Upload and verification complete: $REMOTE$REMOTE_ROOT"
echo "Per-job logs: $LOG_DIR"
