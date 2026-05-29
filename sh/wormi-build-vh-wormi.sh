#!/usr/bin/env bash
# Build WorMI-aligned VirtualHome dataset and validate it.
#
# Output layout (see tools/build_virtualhome_dataset_wormi.py):
#   $DATA_ROOT/scene_0..5/train.jsonl, test.jsonl (symlink)
#   $DATA_ROOT/test_seen_task_seen_scene.jsonl
#   $DATA_ROOT/test_seen_task_unseen_scene.jsonl
#   $DATA_ROOT/test_unseen_task_unseen_scene.jsonl
#   $DATA_ROOT/eval_col_{1,2,3}_<...>/test.jsonl (symlink)
#   $DATA_ROOT/virtualhome_manifest.json
#   $DATA_ROOT/quality_report.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
RUN_ID="${RUN_ID:-wormi-paperaligned-$(date +%Y%m%d)}"
RAW_DIR="${RAW_DIR:-$DATA_DISK/wormi-data/raw/programs_processed_precond_nograb_morepreconds}"
VH_SRC="${VH_SRC:-$DATA_DISK/wormi-data/virtualhome-src}"
DATA_ROOT="${DATA_ROOT:-$DATA_DISK/wormi-data/virtualhome-$RUN_ID}"

SEED="${SEED:-42}"
VARIANTS_PER_DOMAIN="${VARIANTS_PER_DOMAIN:-12}"
TRAIN_EPISODES="${TRAIN_EPISODES:-384}"
EVAL_A_EPISODES="${EVAL_A_EPISODES:-96}"
EVAL_B_EPISODES="${EVAL_B_EPISODES:-224}"
EVAL_C_EPISODES="${EVAL_C_EPISODES:-319}"
EVAL_A_MIN_PER_TASK="${EVAL_A_MIN_PER_TASK:-4}"
CANDIDATE_MULTIPLIER="${CANDIDATE_MULTIPLIER:-8}"
MIN_PROBE_SUCCESSES="${MIN_PROBE_SUCCESSES:-1}"
MAX_PROBE_TASKS="${MAX_PROBE_TASKS:-20}"
MAX_SCAN_PER_BASE="${MAX_SCAN_PER_BASE:-8000}"
FAIL_FIRST_ACTION_TOP1_SHARE="${FAIL_FIRST_ACTION_TOP1_SHARE:-0.35}"

echo "== WorMI VH dataset build =="
echo "root:         $ROOT"
echo "data disk:    $DATA_DISK"
echo "raw_dir:      $RAW_DIR"
echo "vh_src:       $VH_SRC"
echo "output_dir:   $DATA_ROOT"
echo "seed:         $SEED"
echo "variants:     $VARIANTS_PER_DOMAIN"
echo "train/A/B/C:  $TRAIN_EPISODES/$EVAL_A_EPISODES/$EVAL_B_EPISODES/$EVAL_C_EPISODES"
echo "started:      $(date -Iseconds)"

mkdir -p "$DATA_DISK/wormi-logs"
LOG_DIR="$DATA_DISK/wormi-logs/vh-build-$RUN_ID"
mkdir -p "$LOG_DIR"

set -x
python3 -u tools/build_virtualhome_dataset_wormi.py \
  --raw-dir "$RAW_DIR" \
  --vh-src "$VH_SRC" \
  --output-dir "$DATA_ROOT" \
  --seed "$SEED" \
  --variants-per-domain "$VARIANTS_PER_DOMAIN" \
  --train-episodes "$TRAIN_EPISODES" \
  --eval-a-episodes "$EVAL_A_EPISODES" \
  --eval-b-episodes "$EVAL_B_EPISODES" \
  --eval-c-episodes "$EVAL_C_EPISODES" \
  --eval-a-min-per-task "$EVAL_A_MIN_PER_TASK" \
  --candidate-multiplier "$CANDIDATE_MULTIPLIER" \
  --min-probe-successes "$MIN_PROBE_SUCCESSES" \
  --max-probe-tasks "$MAX_PROBE_TASKS" \
  --max-scan-per-base "$MAX_SCAN_PER_BASE" \
  --fail-first-action-top1-share "$FAIL_FIRST_ACTION_TOP1_SHARE" \
  --overwrite 2>&1 | tee "$LOG_DIR/build.log"
set +x

echo "finished:     $(date -Iseconds)"
echo "data root:    $DATA_ROOT"
echo "log:          $LOG_DIR/build.log"
