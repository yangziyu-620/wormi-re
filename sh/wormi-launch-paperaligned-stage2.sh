#!/usr/bin/env bash
# Launch stage 2 for the WorMI paper-aligned dataset.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
RUN_ID="${RUN_ID:-wormi-paperaligned-20260529}"
export DATA_ROOT="$DATA_DISK/wormi-data/virtualhome-$RUN_ID"
export WORLD_CKPT_ROOT="$DATA_DISK/wormi-checkpoints/world-vh-$RUN_ID"
export CKPT_ROOT="$DATA_DISK/wormi-checkpoints/wormi-vh-$RUN_ID"

# Use threaded path with beta=0.1 (closest to paper Algorithm 1 we have).
# sequential_meta is known to collapse the model on this code base; threaded
# default mean(theta_j) was empirically stable on the previous task-aware run.
# Choose threaded + paper beta=0.1 update.
export WORMI_ALLOW_UNSAFE_THREADED_META=1
export WORMI_SEQUENTIAL_META_LEARNING=0
export WORMI_THREADED_META_USE_BETA=1
export WORMI_VH_STAGE2_BATCH_SIZE=1
export WORMI_VH_STAGE2_GRADIENT_ACCUMULATION_STEPS=4
export RUN_ID
bash sh/wormi-train-vh-wormi.sh
