#!/usr/bin/env bash
set -euo pipefail

# Wait for the currently running AA-fill17 rollout to finish, then run a small
# rollout sample with the current evaluator code. This captures the semantic
# instance-binding fix without overwriting the full rollout output.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-paperlike-tmow-compact-aa-fill17-20260528}"
QUICK_TAG="${QUICK_TAG:-${RUN_TAG}-bindfix-quick}"
DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
DATA_ROOT="${DATA_ROOT:-$DATA_DISK/wormi-data/virtualhome-$RUN_TAG}"
WORLD_CKPT_ROOT="${WORLD_CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/world-vh-$RUN_TAG}"
WORMI_CKPT_ROOT="${WORMI_CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/wormi-vh-$RUN_TAG}"
MODEL_NAME="${MODEL_NAME:-$WORMI_CKPT_ROOT/wormi-vh-n6/last}"
VH_SRC="${VH_SRC:-$DATA_DISK/wormi-data/virtualhome-src}"
SCENE_INITS_JSON="${SCENE_INITS_JSON:-$DATA_DISK/wormi-data/scene-inits/init_graphs_20_semantic.json}"
LOG_ROOT="${LOG_ROOT:-$DATA_DISK/wormi-logs}"
LOG_DIR="$LOG_ROOT/vh-rollout-$QUICK_TAG"
LOG_FILE="$LOG_DIR/launcher.log"
OUTPUT_PATH="${OUTPUT_PATH:-$WORMI_CKPT_ROOT/wormi-vh-n6/vh-rollout-$QUICK_TAG}"
NUM_SAMPLES="${NUM_SAMPLES:-32}"
MAX_STEPS="${MAX_STEPS:-30}"
POLL_SECONDS="${POLL_SECONDS:-120}"

OLD_OUTPUT="$WORMI_CKPT_ROOT/wormi-vh-n6/vh-rollout-$RUN_TAG"
OLD_PATTERN="python3 .*eval-vh-rollout.*$OLD_OUTPUT"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "== AA-fill17 bindfix quick rollout watcher =="
echo "started:       $(date -Is)"
echo "run tag:       $RUN_TAG"
echo "quick tag:     $QUICK_TAG"
echo "old output:    $OLD_OUTPUT"
echo "quick output:  $OUTPUT_PATH"
echo "num samples:   $NUM_SAMPLES per eval column"
echo "poll seconds:  $POLL_SECONDS"

while pgrep -af "$OLD_PATTERN" >/dev/null; do
  echo "waiting for current full rollout to finish: $(date -Is)"
  sleep "$POLL_SECONDS"
done

if [[ -f "$OUTPUT_PATH/vh-rollout-summary.tsv" ]]; then
  echo "quick summary already exists: $OUTPUT_PATH/vh-rollout-summary.tsv"
  exit 0
fi

echo "starting bindfix quick rollout: $(date -Is)"
RUN_ID="$QUICK_TAG" \
DATA_DISK="$DATA_DISK" \
DATA_ROOT="$DATA_ROOT" \
VH_SRC="$VH_SRC" \
SCENE_INITS_JSON="$SCENE_INITS_JSON" \
WORLD_CKPT_ROOT="$WORLD_CKPT_ROOT" \
WORMI_CKPT_ROOT="$WORMI_CKPT_ROOT" \
MODEL_NAME="$MODEL_NAME" \
OUTPUT_PATH="$OUTPUT_PATH" \
NUM_SAMPLES="$NUM_SAMPLES" \
MAX_STEPS="$MAX_STEPS" \
bash sh/wormi-eval-vh-rollout.sh

echo "finished:      $(date -Is)"
echo "summary:       $OUTPUT_PATH/vh-rollout-summary.tsv"
