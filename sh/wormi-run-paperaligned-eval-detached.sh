#!/usr/bin/env bash
set -euo pipefail

# Fully detached Table-1 + rollout eval for the paper-aligned 20260529 run.
# Launched via setsid+nohup so it survives terminal/session exit.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_BASE="paperaligned-eval-20260529"
DD=/root/autodl-tmp
DR=$DD/wormi-data/virtualhome-wormi-paperaligned-20260529
WC=$DD/wormi-checkpoints/world-vh-wormi-paperaligned-20260529
MC=$DD/wormi-checkpoints/wormi-vh-wormi-paperaligned-20260529
MODEL=$MC/wormi-vh-n6/last
PIPE=$DD/wormi-logs/vh-pipeline-$RUN_BASE
STATUS="$PIPE/status.tsv"

mkdir -p "$PIPE"
printf 'time\tstage\tstatus\n' > "$STATUS"
log_status() { printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >> "$STATUS"; }

log_status table1 start
RUN_ID="$RUN_BASE-table1" DATA_DISK="$DD" DATA_ROOT="$DR" \
  WORLD_CKPT_ROOT="$WC" WORMI_CKPT_ROOT="$MC" MODEL_NAME="$MODEL" \
  OUTPUT_PATH="$MC/wormi-vh-n6/table1-$RUN_BASE" \
  bash sh/wormi-eval-vh-table1.sh
log_status table1 done

if [[ ! -f "$MC/wormi-vh-n6/table1-$RUN_BASE/table1-summary.tsv" ]]; then
  log_status table1 missing-summary
fi

log_status rollout start
RUN_ID="$RUN_BASE-rollout" DATA_DISK="$DD" DATA_ROOT="$DR" \
  WORLD_CKPT_ROOT="$WC" WORMI_CKPT_ROOT="$MC" MODEL_NAME="$MODEL" \
  OUTPUT_PATH="$MC/wormi-vh-n6/vh-rollout-$RUN_BASE" \
  MAX_STEPS=30 TEMPERATURE=1.0 TOP_P=1.0 \
  bash sh/wormi-eval-vh-rollout.sh
log_status rollout done

log_status pipeline done
