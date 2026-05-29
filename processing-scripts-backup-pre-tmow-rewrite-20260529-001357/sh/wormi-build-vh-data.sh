#!/usr/bin/env bash
set -euo pipefail

# Rebuild the VirtualHome jsonl data used by WorMI.
# Raw zip, scene-init cache, generated data, logs, and temp files all default
# to the data disk. Run this before VH stage-1 training when the builder logic
# changes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
WORMI_DATA_ROOT="${WORMI_DATA_ROOT:-$DATA_DISK/wormi-data}"
RAW_DIR="${RAW_DIR:-$WORMI_DATA_ROOT/raw}"
SCENE_INIT_DIR="${SCENE_INIT_DIR:-$WORMI_DATA_ROOT/scene-inits}"
SCENE_INITS_JSON="${SCENE_INITS_JSON:-$SCENE_INIT_DIR/init_graphs_20_semantic.json}"
SCENE_INITS_MANIFEST_JSON="${SCENE_INITS_MANIFEST_JSON:-$SCENE_INIT_DIR/init_graphs_20_semantic_manifest.json}"
VH_SRC="${VH_SRC:-$WORMI_DATA_ROOT/virtualhome-src}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORMI_DATA_ROOT/virtualhome}"
LOG_ROOT="${LOG_ROOT:-$DATA_DISK/wormi-logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$LOG_ROOT/vh-data-$RUN_ID"
LOG_FILE="$LOG_DIR/build.log"
TARGET_TRAJECTORIES="${TARGET_TRAJECTORIES:-1023}"
CANDIDATE_MULTIPLIER="${CANDIDATE_MULTIPLIER:-12}"
SEEN_SEEN_EVAL_PER_TASK="${SEEN_SEEN_EVAL_PER_TASK:-2}"
BUILD_MODE="${BUILD_MODE:-paper_like}"

ZIP_PATH="$RAW_DIR/programs_processed_precond_nograb_morepreconds.zip"
VH_ZIP_URL="http://virtual-home.org/release/programs/programs_processed_precond_nograb_morepreconds.zip"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$DATA_DISK/uv-cache}"
export TMPDIR="${TMPDIR:-$DATA_DISK/tmp}"
export HF_HOME="${HF_HOME:-$DATA_DISK/hf-home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_DISK/xdg-cache}"

mkdir -p "$RAW_DIR" "$SCENE_INIT_DIR" "$LOG_DIR" "$UV_CACHE_DIR" "$TMPDIR" "$HF_HOME" "$XDG_CACHE_HOME"

echo "== WorMI VirtualHome data rebuild =="
echo "root:          $ROOT_DIR"
echo "data disk:     $DATA_DISK"
echo "raw zip:       $ZIP_PATH"
echo "scene inits:   $SCENE_INITS_JSON"
echo "scene manifest:$SCENE_INITS_MANIFEST_JSON"
echo "vh source:     $VH_SRC"
echo "output dir:    $OUTPUT_DIR"
echo "target traj:   $TARGET_TRAJECTORIES"
echo "seen-seen eval/task: $SEEN_SEEN_EVAL_PER_TASK"
echo "build mode:    $BUILD_MODE"
echo "log file:      $LOG_FILE"
echo "started:       $(date -Is)"

if [[ ! -s "$ZIP_PATH" ]]; then
  echo "raw zip missing, downloading..."
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 16 -s 16 -k 1M -d "$RAW_DIR" "$VH_ZIP_URL"
  else
    curl -L "$VH_ZIP_URL" -o "$ZIP_PATH"
  fi
fi

if [[ ! -d "$VH_SRC/.git" ]]; then
  if [[ -e "$VH_SRC" ]]; then
    echo "ERROR: $VH_SRC exists but is not a git checkout" >&2
    exit 1
  fi
  git clone --depth 1 https://github.com/xavierpuigf/virtualhome.git "$VH_SRC"
fi

if [[ ! -s "$SCENE_INITS_JSON" || "${REBUILD_SCENE_INITS:-0}" == "1" ]]; then
  echo "scene init cache missing/stale, building 20 semantic init graphs from raw zip..."
  .venv/bin/python tools/build_virtualhome_dataset.py scene-cache \
    --zip-path "$ZIP_PATH" \
    --vh-src "$VH_SRC" \
    --output-json "$SCENE_INITS_JSON" \
    --manifest-json "$SCENE_INITS_MANIFEST_JSON"
fi

if [[ -e "$OUTPUT_DIR" ]]; then
  BACKUP_DIR="${OUTPUT_DIR}.bak.${RUN_ID}"
  echo "backing up existing output: $OUTPUT_DIR -> $BACKUP_DIR"
  mv "$OUTPUT_DIR" "$BACKUP_DIR"
fi

set -x
.venv/bin/python tools/build_virtualhome_dataset.py \
  --scene-inits-json "$SCENE_INITS_JSON" \
  --vh-src "$VH_SRC" \
  --output-dir "$OUTPUT_DIR" \
  --seen-scenes 6 \
  --seen-instructions 16 \
  --candidate-multiplier "$CANDIDATE_MULTIPLIER" \
  --target-trajectories "$TARGET_TRAJECTORIES" \
  --seen-seen-eval-per-task "$SEEN_SEEN_EVAL_PER_TASK" \
  --mode "$BUILD_MODE" \
  2>&1 | tee "$LOG_FILE"
set +x

for scene in scene_0 scene_1 scene_2 scene_3 scene_4 scene_5; do
  test -s "$OUTPUT_DIR/$scene/train.jsonl"
  test -e "$OUTPUT_DIR/$scene/test.jsonl"
done
for eval_dir in eval_col_1_seen_seen eval_col_2_seen_unseen eval_col_3_unseen_unseen; do
  test -e "$OUTPUT_DIR/$eval_dir/test.jsonl"
done

echo "finished: $(date -Is)"
