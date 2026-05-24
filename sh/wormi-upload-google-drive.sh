#!/usr/bin/env bash
set -euo pipefail

# Upload the current VirtualHome reproduction state to Google Drive with a
# reviewable directory layout. This streams from the source directories; it
# does not copy large checkpoints into the repo/root filesystem first.

REMOTE="${REMOTE:-gdrive_ghub:}"
RUN_NAME="${RUN_NAME:-vh-semantic1023-20260520}"
REMOTE_ROOT="${REMOTE_ROOT:-gdrive_ghub/$RUN_NAME}"
DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
REPO_ROOT="${REPO_ROOT:-/root/WorMI}"
STAGING_ROOT="${STAGING_ROOT:-$DATA_DISK/wormi-upload-staging/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$DATA_DISK/wormi-logs/upload-$RUN_NAME}"

# Two levels of parallelism:
# - UPLOAD_JOBS: independent rclone copy jobs.
# - RCLONE_TRANSFERS/CHECKERS: per-rclone file-level parallelism.
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

cat > "$STAGING_ROOT/00_README.md" <<EOF
# WorMI VirtualHome Reproduction State

Run: $RUN_NAME

This upload preserves the current reproduction state before rebuilding the
VirtualHome split with a task-aware Table-1 test design.

## Layout

- \`01_data/virtualhome\`: current rebuilt VirtualHome JSONL data.
- \`01_data/scene-inits\`: semantic scene initial graph cache.
- \`02_stage1_world_models\`: six scene-keyed stage-1 world models.
- \`03_stage2_wormi_adapter\`: trained stage-2 WorMI adapter checkpoint and eval outputs.
- \`04_eval_results\`: copied rollout summaries for quick access.
- \`05_logs\`: training/eval/upload logs.
- \`06_reports_and_analysis\`: local reports and validation JSONs.
- \`07_code_snapshot\`: code files needed to inspect data generation, training, eval, and scripts.
- \`MANIFEST.sha256\`: sha256 checksums for uploaded local source files.
- \`UPLOAD_SOURCES.tsv\`: source-to-destination mapping.

## Key Results

Main rollout summary:

\`03_stage2_wormi_adapter/wormi-vh-semantic1023-rerun/wormi-vh-n6/vh-rollout-semantic1023-rerun-20260520/vh-rollout-summary.tsv\`

Current main rollout:

\`\`\`
col_1_seen_seen      SR=75.00%  PS=13.50  n=8
col_2_seen_unseen    SR=84.95%  PS=10.75  n=186
col_3_unseen_unseen  SR=62.67%  PS=16.15  n=300
\`\`\`

Known issue: current Table-1 col1 seen-seen eval has only 8 episodes and is
not task-balanced. See \`06_reports_and_analysis\` for notes.
EOF

cat > "$STAGING_ROOT/UPLOAD_SOURCES.tsv" <<EOF
destination	source
00_README.md	$STAGING_ROOT/00_README.md
UPLOAD_SOURCES.tsv	$STAGING_ROOT/UPLOAD_SOURCES.tsv
MANIFEST.sha256	$STAGING_ROOT/MANIFEST.sha256
01_data/virtualhome	$DATA_DISK/wormi-data/virtualhome
01_data/scene-inits	$DATA_DISK/wormi-data/scene-inits
02_stage1_world_models/world-vh-semantic1023-rerun	$DATA_DISK/wormi-checkpoints/world-vh-semantic1023-rerun
03_stage2_wormi_adapter/wormi-vh-semantic1023-rerun	$DATA_DISK/wormi-checkpoints/wormi-vh-semantic1023-rerun
05_logs/wormi-logs	$DATA_DISK/wormi-logs
06_reports_and_analysis/reports	$REPO_ROOT/reports
07_code_snapshot/tools	$REPO_ROOT/tools
07_code_snapshot/sh	$REPO_ROOT/sh
07_code_snapshot/wormi	$REPO_ROOT/wormi
07_code_snapshot/root_files/AGENTS.md	$REPO_ROOT/AGENTS.md
07_code_snapshot/root_files/Readme.md	$REPO_ROOT/Readme.md
07_code_snapshot/root_files/pyproject.toml	$REPO_ROOT/pyproject.toml
07_code_snapshot/root_files/uv.lock	$REPO_ROOT/uv.lock
EOF

echo "Generating checksum manifest..."
{
  cd /
  find \
    "$DATA_DISK/wormi-data/virtualhome" \
    "$DATA_DISK/wormi-data/scene-inits" \
    "$DATA_DISK/wormi-checkpoints/world-vh-semantic1023-rerun" \
    "$DATA_DISK/wormi-checkpoints/wormi-vh-semantic1023-rerun" \
    "$DATA_DISK/wormi-logs" \
    "$REPO_ROOT/reports" \
    "$REPO_ROOT/tools" \
    "$REPO_ROOT/sh" \
    "$REPO_ROOT/wormi" \
    -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum
  sha256sum "$REPO_ROOT/AGENTS.md" "$REPO_ROOT/Readme.md" "$REPO_ROOT/pyproject.toml" "$REPO_ROOT/uv.lock"
} > "$STAGING_ROOT/MANIFEST.sha256"

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
  rclone copyto "$src" "$REMOTE$REMOTE_ROOT/$dst_dir/$(basename "$src")" "${RCLONE_COMMON[@]}" \
    > "$LOG_DIR/$name.log" 2>&1
}

run_limited() {
  while (( "$(jobs -rp | wc -l)" >= UPLOAD_JOBS )); do
    sleep 5
  done
  "$@" &
}

echo "Remote: $REMOTE$REMOTE_ROOT"
echo "Logs:   $LOG_DIR"

run_limited copy_file "$STAGING_ROOT/00_README.md" "."
run_limited copy_file "$STAGING_ROOT/UPLOAD_SOURCES.tsv" "."
run_limited copy_file "$STAGING_ROOT/MANIFEST.sha256" "."
run_limited copy_one "$DATA_DISK/wormi-data/virtualhome" "01_data/virtualhome"
run_limited copy_one "$DATA_DISK/wormi-data/scene-inits" "01_data/scene-inits"
run_limited copy_one "$DATA_DISK/wormi-checkpoints/world-vh-semantic1023-rerun" "02_stage1_world_models/world-vh-semantic1023-rerun"
run_limited copy_one "$DATA_DISK/wormi-checkpoints/wormi-vh-semantic1023-rerun" "03_stage2_wormi_adapter/wormi-vh-semantic1023-rerun"
run_limited copy_one "$DATA_DISK/wormi-logs" "05_logs/wormi-logs"
run_limited copy_one "$REPO_ROOT/reports" "06_reports_and_analysis/reports"
run_limited copy_one "$REPO_ROOT/tools" "07_code_snapshot/tools"
run_limited copy_one "$REPO_ROOT/sh" "07_code_snapshot/sh"
run_limited copy_one "$REPO_ROOT/wormi" "07_code_snapshot/wormi"
run_limited copy_file "$REPO_ROOT/AGENTS.md" "07_code_snapshot/root_files"
run_limited copy_file "$REPO_ROOT/Readme.md" "07_code_snapshot/root_files"
run_limited copy_file "$REPO_ROOT/pyproject.toml" "07_code_snapshot/root_files"
run_limited copy_file "$REPO_ROOT/uv.lock" "07_code_snapshot/root_files"

wait

echo "Upload complete: $REMOTE$REMOTE_ROOT"
echo "Per-job logs: $LOG_DIR"
