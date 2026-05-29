"""WorMI stage-2 integration curricula for VirtualHome.

Paper §3 / Table A.6 configuration:
- Reasoning model: Llama-3.2-3B-Instruct
- Method: WORLD_WISE_ATTENTION
- Connections: base=[13, 27], world=[7, 15]
- Meta-learning: λ_M=8 outer iterations, λ_I=30 inner steps
- Inner lr α=1e-5
- Prototype size k=15, retrieval K=3
- N=6 world models (one per seen scene; from stage 1)

`train_curricula` enumerates 6 paper-K=3 subsets of {0..5} so every world
model participates in 3 of the 6 subsets (Reptile-style — each "task"
in meta-learning terms is one subset, aggregated by
MetaLearningAggregationCallback). Paper Algorithm 1 line 5 says
"sample the subset"; with `WorMITrainer`'s fixed-curricula architecture
we approximate that with a cyclic enumeration that covers all world
models evenly.

`test_curricula` produces Table 1's three columns. Retrieval is on at
test time (sentence_embedding_model is set), so `target_world_models`
on each test entry is ignored — Algorithm 1 line 28 retrieves K=3 by
Wasserstein distance over prototype sets.
"""

import os
from pathlib import Path

from wormi.curricula import WorldModel, WorMICurricula, WorMICurriculum
from wormi.model import WorMIIntegrateMethod
from wormi.trainer import SchedulerType, WorMITrainerConfig

DATA_DISK = Path(os.environ.get("WORMI_DATA_DISK", "/root/autodl-tmp"))
DATASET_ROOT = Path(
    os.environ.get("WORMI_VH_DATA_ROOT", DATA_DISK / "wormi-data" / "virtualhome")
)
WORLD_CKPT_ROOT = Path(
    os.environ.get("WORMI_WORLD_VH_OUTPUT_DIR", DATA_DISK / "wormi-checkpoints" / "world-vh")
)
OUTPUT_DIR = Path(
    os.environ.get("WORMI_VH_OUTPUT_DIR", DATA_DISK / "wormi-checkpoints" / "wormi-vh")
)

# Stage-2 reasoning model. Same un-gated unsloth mirror used for stage 1
# but the 3B variant per paper Table A.6.
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"

# Paper Table A.6 — base-model implant layers = [13, 27], world-model
# implant layers = [7, 15].
BASE_CONNECTIONS = [13, 27]
WORLD_CONNECTIONS = [7, 15]

# Paper Table A.6: 6 world models. Names match stage-1 output dirs.
SCENES = [f"scene_{i}" for i in range(6)]

# Inner-loop trainer config. Paper Table A.6: λ_I=30 inner steps, α=1e-5.
#
# eval_steps default is intentionally larger than the inner-loop max_steps so
# that NO mid-training eval fires inside an inner loop. Previously eval_steps=30
# == max_steps=30 triggered a full eval (≈11 min on the seen-seen set) after
# every single inner loop; across 6 trainers × λ_M=8 that was ~40 evals = ~7h,
# i.e. 90% of stage-2 wall-clock was eval, not training. Final model quality is
# measured by the dedicated rollout / table1 evaluators, not by these inner
# evals, so they add cost without value. Override with
# WORMI_VH_STAGE2_EVAL_STEPS=30 to restore the old per-inner-loop eval.
trainer_args = WorMITrainerConfig(
    max_steps=int(os.environ.get("WORMI_VH_STAGE2_INNER_STEPS", "30")),
    eval_steps=int(os.environ.get("WORMI_VH_STAGE2_EVAL_STEPS", "100000")),
    save_steps=240,
    logging_steps=5,
    batch_size=int(os.environ.get("WORMI_VH_STAGE2_BATCH_SIZE", "4")),
    gradient_accumulation_steps=int(
        os.environ.get("WORMI_VH_STAGE2_GRADIENT_ACCUMULATION_STEPS", "4")
    ),
    learning_rate=1e-5,
    lr_scheduler_type=SchedulerType.COSINE,
)

# Test-time evaluation uses bigger sampling for stable accuracy.
test_trainer_args = WorMITrainerConfig(
    max_steps=0,
    batch_size=1,
)

# 6 K=3 subsets covering N=6 world models evenly. Each WM appears in 3 of 6.
TRAIN_SUBSETS = [
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 0],
    [5, 0, 1],
]

# curricula.datasets ordering: positions 0..5 paired 1:1 with world models
# (used for prototype computation at retrieval time). Positions 6..8 are
# eval-only dirs (test.jsonl symlinks to the Table 1 col-N source files).
EVAL_DIRS = [
    "eval_col_1_seen_seen",        # idx 6 → Table 1 col 1
    "eval_col_2_seen_unseen",      # idx 7 → Table 1 col 2
    "eval_col_3_unseen_unseen",    # idx 8 → Table 1 col 3
]
DATASETS = [str(DATASET_ROOT / s) for s in SCENES] + [
    str(DATASET_ROOT / d) for d in EVAL_DIRS
]

curricula = WorMICurricula(
    name="wormi-vh-n6",
    output_dir=OUTPUT_DIR,
    base_model=BASE_MODEL,
    connections=BASE_CONNECTIONS,
    method=WorMIIntegrateMethod.WORLD_WISE_ATTENTION,
    num_heads=8,
    self_attention=False,
    meta_learning=True,
    meta_learning_rate=float(
        os.environ.get("WORMI_VH_STAGE2_META_LR", "0.1")
    ),  # β in paper Algorithm 1 / Table A.6 (override to test adapter starvation)
    num_iterations=int(os.environ.get("WORMI_VH_STAGE2_META_STEPS", "8")),   # λ_M
    world_models=[
        WorldModel(
            model_name=str(WORLD_CKPT_ROOT / s / "last"),
            connections=WORLD_CONNECTIONS,
        )
        for s in SCENES
    ],
    datasets=DATASETS,
    train_curricula=[
        WorMICurriculum(
            name=f"subset-{'-'.join(map(str, subset))}",
            target_world_models=list(subset),
            datasets=list(subset),
            trainer_args=trainer_args,
            # Cap the in-training eval set. Previously unbounded (~1470 BC
            # samples/subset → ~658s/eval → ~90% of stage-2 wall-clock). The
            # final quality is measured by table1/rollout, so a tiny eval is
            # enough to watch loss. 0 disables via taking 0 (kept >0 here).
            num_eval_samples=int(
                os.environ.get("WORMI_VH_STAGE2_EVAL_SAMPLES", "16")
            ),
        )
        for subset in TRAIN_SUBSETS
    ],
    test_curricula=[
        # Retrieval is on, so target_world_models is ignored by the
        # retrieval-aware eval. Set to [0,1,2] just so the legacy
        # _inner_test_loop path (no retrieval) would still parse.
        WorMICurriculum(
            name="col_1_seen_seen",
            target_world_models=[0, 1, 2],
            datasets=[6],  # eval_col_1_seen_seen
            trainer_args=test_trainer_args,
        ),
        WorMICurriculum(
            name="col_2_seen_unseen",
            target_world_models=[0, 1, 2],
            datasets=[7],  # eval_col_2_seen_unseen
            trainer_args=test_trainer_args,
            num_eval_samples=300,
        ),
        WorMICurriculum(
            name="col_3_unseen_unseen",
            target_world_models=[0, 1, 2],
            datasets=[8],  # eval_col_3_unseen_unseen
            trainer_args=test_trainer_args,
            num_eval_samples=300,
        ),
    ],
    test_continuously=False,
    sentence_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    retrieval_k=3,
    prototype_size=15,
)
