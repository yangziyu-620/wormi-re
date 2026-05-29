#!/usr/bin/env bash
set -euo pipefail

# Run VirtualHome stage-2 WorMI adapter training, then Table-1 offline eval
# and VirtualHome rollout eval. Designed for detached execution.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
DATA_ROOT="${DATA_ROOT:-$DATA_DISK/wormi-data/virtualhome}"
WORLD_CKPT_ROOT="${WORLD_CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/world-vh-paperlike-v1-20260526}"
WORMI_CKPT_ROOT="${WORMI_CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/wormi-vh-paperlike-v1-threaded-lockfix}"
RUN_BASE="${RUN_BASE:-paperlike-v1-threaded-lockfix-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$DATA_DISK/wormi-logs}"
PIPE_DIR="$LOG_ROOT/vh-pipeline-$RUN_BASE"
STATUS_FILE="$PIPE_DIR/status.tsv"
MODEL_NAME="${MODEL_NAME:-$WORMI_CKPT_ROOT/wormi-vh-n6/last}"
TABLE1_OUTPUT="${TABLE1_OUTPUT:-$WORMI_CKPT_ROOT/wormi-vh-n6/table1-$RUN_BASE}"
ROLLOUT_OUTPUT="${ROLLOUT_OUTPUT:-$WORMI_CKPT_ROOT/wormi-vh-n6/vh-rollout-$RUN_BASE}"

mkdir -p "$PIPE_DIR" "$WORMI_CKPT_ROOT"
printf 'time\tstage\tstatus\n' > "$STATUS_FILE"

log_status() {
  printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >> "$STATUS_FILE"
}

log_status preflight start
export DATA_ROOT WORLD_CKPT_ROOT
.venv/bin/python - <<'PYCHECK'
import json
import os
from pathlib import Path
root = Path(os.environ["DATA_ROOT"])
quality = root / "quality_report.json"
if quality.exists():
    q = json.loads(quality.read_text())
    if q.get("errors"):
        raise SystemExit(q["errors"])
world = Path(os.environ["WORLD_CKPT_ROOT"])
missing = [str(world / f"scene_{i}" / "last") for i in range(6) if not (world / f"scene_{i}" / "last").is_dir()]
if missing:
    raise SystemExit(f"missing world models: {missing}")
for scene in [f"scene_{i}" for i in range(6)]:
    if not (root / scene / "train.jsonl").is_file():
        raise SystemExit(f"missing train split: {root / scene / 'train.jsonl'}")
    if not (root / scene / "test.jsonl").is_file():
        raise SystemExit(f"missing test split: {root / scene / 'test.jsonl'}")
for eval_dir in ["eval_col_1_seen_seen", "eval_col_2_seen_unseen", "eval_col_3_unseen_unseen"]:
    if not (root / eval_dir / "test.jsonl").is_file():
        raise SystemExit(f"missing eval split: {root / eval_dir / 'test.jsonl'}")
print("preflight ok", root, world)
PYCHECK
log_status preflight done

log_status stage2 start
RUN_ID="$RUN_BASE-stage2" \
DATA_DISK="$DATA_DISK" \
DATA_ROOT="$DATA_ROOT" \
WORLD_CKPT_ROOT="$WORLD_CKPT_ROOT" \
CKPT_ROOT="$WORMI_CKPT_ROOT" \
WORMI_VH_STAGE2_BATCH_SIZE="${WORMI_VH_STAGE2_BATCH_SIZE:-1}" \
WORMI_VH_STAGE2_GRADIENT_ACCUMULATION_STEPS="${WORMI_VH_STAGE2_GRADIENT_ACCUMULATION_STEPS:-4}" \
WORMI_SEQUENTIAL_META_LEARNING="${WORMI_SEQUENTIAL_META_LEARNING:-0}" \
WORMI_THREADED_META_USE_BETA="${WORMI_THREADED_META_USE_BETA:-0}" \
WORMI_VH_STAGE2_INNER_STEPS="${WORMI_VH_STAGE2_INNER_STEPS:-30}" \
WORMI_VH_STAGE2_META_STEPS="${WORMI_VH_STAGE2_META_STEPS:-8}" \
bash sh/wormi-train-vh-wormi.sh
log_status stage2 done

if [[ ! -d "$MODEL_NAME" ]]; then
  echo "ERROR: stage2 did not produce $MODEL_NAME" >&2
  exit 1
fi

log_status table1 start
RUN_ID="$RUN_BASE-table1" \
DATA_DISK="$DATA_DISK" \
DATA_ROOT="$DATA_ROOT" \
WORLD_CKPT_ROOT="$WORLD_CKPT_ROOT" \
WORMI_CKPT_ROOT="$WORMI_CKPT_ROOT" \
MODEL_NAME="$MODEL_NAME" \
OUTPUT_PATH="$TABLE1_OUTPUT" \
bash sh/wormi-eval-vh-table1.sh
log_status table1 done

log_status rollout start
RUN_ID="$RUN_BASE-rollout" \
DATA_DISK="$DATA_DISK" \
DATA_ROOT="$DATA_ROOT" \
WORLD_CKPT_ROOT="$WORLD_CKPT_ROOT" \
WORMI_CKPT_ROOT="$WORMI_CKPT_ROOT" \
MODEL_NAME="$MODEL_NAME" \
OUTPUT_PATH="$ROLLOUT_OUTPUT" \
MAX_STEPS="${MAX_STEPS:-30}" \
TEMPERATURE="${TEMPERATURE:-1.0}" \
TOP_P="${TOP_P:-1.0}" \
bash sh/wormi-eval-vh-rollout.sh
log_status rollout done
log_status pipeline done
