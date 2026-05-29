#!/usr/bin/env bash
# Launch stage 1 for the WorMI paper-aligned dataset.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
RUN_ID="${RUN_ID:-wormi-paperaligned-20260529}"
export DATA_ROOT="$DATA_DISK/wormi-data/virtualhome-$RUN_ID"
export CKPT_ROOT="$DATA_DISK/wormi-checkpoints/world-vh-$RUN_ID"
export RUN_ID
bash sh/wormi-train-vh-world.sh
