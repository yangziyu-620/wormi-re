#!/usr/bin/env bash
set -euo pipefail
# Beta-retry of paper-aligned stage 2: same stage-1 world models + same data,
# but meta β raised to test the adapter-starvation diagnosis, and the
# in-training eval set capped to a tiny sample count (was ~90% of wall-clock).
# Fully detached (setsid+nohup) so it survives session exit.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
SRC_RUN="wormi-paperaligned-20260529"          # reuse this run's data + world ckpt
RETRY_RUN="${RETRY_RUN:-wormi-paperaligned-beta1-20260529}"

export DATA_DISK
export DATA_ROOT="$DATA_DISK/wormi-data/virtualhome-$SRC_RUN"
export WORLD_CKPT_ROOT="$DATA_DISK/wormi-checkpoints/world-vh-$SRC_RUN"
export CKPT_ROOT="$DATA_DISK/wormi-checkpoints/wormi-vh-$RETRY_RUN"
export RUN_ID="$RETRY_RUN"

# Meta-learning path: same threaded Reptile + β update as before, but β raised.
export WORMI_ALLOW_UNSAFE_THREADED_META=1
export WORMI_SEQUENTIAL_META_LEARNING=0
export WORMI_THREADED_META_USE_BETA=1
export WORMI_VH_STAGE2_META_LR="${WORMI_VH_STAGE2_META_LR:-1.0}"   # β (was 0.1)

# Paper-aligned inner/meta budget (unchanged).
export WORMI_VH_STAGE2_BATCH_SIZE=1
export WORMI_VH_STAGE2_GRADIENT_ACCUMULATION_STEPS=4
export WORMI_VH_STAGE2_INNER_STEPS=30
export WORMI_VH_STAGE2_META_STEPS=8

# Eval cost control: tiny in-training eval set + no mid-inner-loop eval.
export WORMI_VH_STAGE2_EVAL_SAMPLES="${WORMI_VH_STAGE2_EVAL_SAMPLES:-16}"
export WORMI_VH_STAGE2_EVAL_STEPS=100000

bash sh/wormi-train-vh-wormi.sh
