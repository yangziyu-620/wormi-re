#!/usr/bin/env bash
set -euo pipefail

# Rebuild the TMoW-style compact VirtualHome jsonl data for WorMI.
# This uses an independent graph-state builder, not post-hoc conversion from a
# full-observation JSONL dataset.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
WORMI_DATA_ROOT="${WORMI_DATA_ROOT:-$DATA_DISK/wormi-data}"
RAW_DIR="${RAW_DIR:-$WORMI_DATA_ROOT/raw}"
SCENE_INIT_DIR="${SCENE_INIT_DIR:-$WORMI_DATA_ROOT/scene-inits}"
SCENE_INITS_JSON="${SCENE_INITS_JSON:-$SCENE_INIT_DIR/init_graphs_20_semantic.json}"
SCENE_INITS_MANIFEST_JSON="${SCENE_INITS_MANIFEST_JSON:-$SCENE_INIT_DIR/init_graphs_20_semantic_manifest.json}"
VH_SRC="${VH_SRC:-$WORMI_DATA_ROOT/virtualhome-src}"
DATASET_NAME="${DATASET_NAME:-paperlike-tmow-compact-fill17-20260528}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORMI_DATA_ROOT/virtualhome-$DATASET_NAME}"
LOG_ROOT="${LOG_ROOT:-$DATA_DISK/wormi-logs}"
RUN_ID="${RUN_ID:-tmow-compact-fill17-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$LOG_ROOT/vh-data-$RUN_ID"
LOG_FILE="$LOG_DIR/build.log"
VALIDATION_JSON="${VALIDATION_JSON:-reports/virtualhome/validation/vh-$DATASET_NAME-validation-2026-05-28.json}"
TARGET_TRAJECTORIES="${TARGET_TRAJECTORIES:-1023}"
CANDIDATE_MULTIPLIER="${CANDIDATE_MULTIPLIER:-12}"
SEEN_SEEN_EVAL_PER_TASK="${SEEN_SEEN_EVAL_PER_TASK:-2}"
COMPACT_NUM_EDGES="${COMPACT_NUM_EDGES:-17}"
NEXT_OBSERVATION_MODE="${NEXT_OBSERVATION_MODE:-delta}"
TOKENIZER="${TOKENIZER:-unsloth/Llama-3.2-3B-Instruct}"

ZIP_PATH="$RAW_DIR/programs_processed_precond_nograb_morepreconds.zip"
VH_ZIP_URL="http://virtual-home.org/release/programs/programs_processed_precond_nograb_morepreconds.zip"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$DATA_DISK/uv-cache}"
export TMPDIR="${TMPDIR:-$DATA_DISK/tmp}"
export HF_HOME="${HF_HOME:-$DATA_DISK/hf-home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_DISK/xdg-cache}"

mkdir -p \
  "$RAW_DIR" "$SCENE_INIT_DIR" "$LOG_DIR" "$UV_CACHE_DIR" "$TMPDIR" \
  "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
  "$XDG_CACHE_HOME"

echo "== WorMI VirtualHome TMoW-compact data rebuild =="
echo "root:             $ROOT_DIR"
echo "data disk:        $DATA_DISK"
echo "raw zip:          $ZIP_PATH"
echo "scene inits:      $SCENE_INITS_JSON"
echo "scene manifest:   $SCENE_INITS_MANIFEST_JSON"
echo "vh source:        $VH_SRC"
echo "output dir:       $OUTPUT_DIR"
echo "validation json:  $VALIDATION_JSON"
echo "target traj:      $TARGET_TRAJECTORIES"
echo "compact edges:    $COMPACT_NUM_EDGES"
echo "next obs mode:    $NEXT_OBSERVATION_MODE"
echo "tokenizer:        $TOKENIZER"
echo "log file:         $LOG_FILE"
echo "started:          $(date -Is)"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "ERROR: .venv is missing. Run uv sync first on a compute node." >&2
  exit 1
fi

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
.venv/bin/python tools/build_virtualhome_dataset_tmow_compact.py \
  --scene-inits-json "$SCENE_INITS_JSON" \
  --vh-src "$VH_SRC" \
  --output-dir "$OUTPUT_DIR" \
  --seen-scenes 6 \
  --seen-instructions 16 \
  --candidate-multiplier "$CANDIDATE_MULTIPLIER" \
  --target-trajectories "$TARGET_TRAJECTORIES" \
  --seen-seen-eval-per-task "$SEEN_SEEN_EVAL_PER_TASK" \
  --compact-num-edges "$COMPACT_NUM_EDGES" \
  --next-observation-mode "$NEXT_OBSERVATION_MODE" \
  2>&1 | tee "$LOG_FILE"

.venv/bin/python tools/validate_virtualhome_dataset.py \
  --data-root "$OUTPUT_DIR" \
  --scene-inits-json "$SCENE_INITS_JSON" \
  --vh-src "$VH_SRC" \
  --check-loader \
  --check-chat-template \
  --tokenizer "$TOKENIZER" \
  --tokenizer-local-files-only \
  --output-json "$VALIDATION_JSON"
set +x

echo "finished: $(date -Is)"
