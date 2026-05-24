"""WorMI stage-2 integration curricula for ALFWorld.

Paper §3 / Table A.6 configuration, same shape as VH:
- Reasoning model: Llama-3.2-3B-Instruct
- Method: WORLD_WISE_ATTENTION
- Connections: base=[13, 27], world=[7, 15]
- Meta-learning: λ_M=8 outer iterations, λ_I=30 inner steps
- Inner lr α=1e-5
- Prototype size k=15, retrieval K=3
- N=6 world models (one per task type; from stage 1)

`train_curricula` enumerates 6 paper-K=3 subsets of {0..5}, where the
6 world-model indices correspond to the 6 task types
(pick_simple / look_at_obj / pick_heat / pick_cool / pick_two / pick_clean).
The first 4 are "seen tasks" per paper §4, the last 2 are "unseen tasks";
but at stage-2 training all 6 participate in the meta-learning rotation
since paper Algorithm 1 retrieves across the full N=6 pool at test time.

`test_curricula` produces Table 1's three columns via retrieval (paper
Algorithm 1 line 28).
"""

from pathlib import Path

from wormi.curricula import WorldModel, WorMICurricula, WorMICurriculum
from wormi.model import WorMIIntegrateMethod
from wormi.trainer import SchedulerType, WorMITrainerConfig

DATASET_ROOT = Path("/srv/scratch/z5524306/wormi-data/alfworld")
WORLD_CKPT_ROOT = Path("/srv/scratch/z5524306/wormi-checkpoints/world-alfworld")
OUTPUT_DIR = Path("/srv/scratch/z5524306/wormi-checkpoints/wormi-alfworld")

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"
BASE_CONNECTIONS = [13, 27]
WORLD_CONNECTIONS = [7, 15]

# Order matches tools/world_curricula_alfworld.py::TASKS. Index → task type:
#   0: task_pick_simple    1: task_look_at_obj   2: task_pick_heat
#   3: task_pick_cool      4: task_pick_two      5: task_pick_clean
TASKS = [
    "task_pick_simple",
    "task_look_at_obj",
    "task_pick_heat",
    "task_pick_cool",
    "task_pick_two",
    "task_pick_clean",
]

trainer_args = WorMITrainerConfig(
    max_steps=30,
    eval_steps=30,
    save_steps=240,
    logging_steps=5,
    batch_size=4,
    learning_rate=1e-5,
    lr_scheduler_type=SchedulerType.COSINE,
)

test_trainer_args = WorMITrainerConfig(
    max_steps=0,
    batch_size=1,
)

TRAIN_SUBSETS = [
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 0],
    [5, 0, 1],
]

EVAL_DIRS = [
    "eval_col_1_seen_seen",
    "eval_col_2_seen_unseen",
    "eval_col_3_unseen_unseen",
]
DATASETS = [str(DATASET_ROOT / t) for t in TASKS] + [
    str(DATASET_ROOT / d) for d in EVAL_DIRS
]

curricula = WorMICurricula(
    name="wormi-alfworld-n6",
    output_dir=OUTPUT_DIR,
    base_model=BASE_MODEL,
    connections=BASE_CONNECTIONS,
    method=WorMIIntegrateMethod.WORLD_WISE_ATTENTION,
    num_heads=8,
    self_attention=False,
    meta_learning=True,
    num_iterations=8,
    world_models=[
        WorldModel(
            model_name=str(WORLD_CKPT_ROOT / t / "last"),
            connections=WORLD_CONNECTIONS,
        )
        for t in TASKS
    ],
    datasets=DATASETS,
    train_curricula=[
        WorMICurriculum(
            name=f"subset-{'-'.join(map(str, subset))}",
            target_world_models=list(subset),
            datasets=list(subset),
            trainer_args=trainer_args,
            # Mid-training eval is a monitoring curve, not the reported
            # metric — cap it. ALFWorld test sets are cumulative-expanded
            # (thousands of rows), and eval fires 48× over the run.
            num_eval_samples=150,
        )
        for subset in TRAIN_SUBSETS
    ],
    test_curricula=[
        WorMICurriculum(
            name="col_1_seen_seen",
            target_world_models=[0, 1, 2],
            datasets=[6],
            trainer_args=test_trainer_args,
        ),
        WorMICurriculum(
            name="col_2_seen_unseen",
            target_world_models=[0, 1, 2],
            datasets=[7],
            trainer_args=test_trainer_args,
            num_eval_samples=300,
        ),
        WorMICurriculum(
            name="col_3_unseen_unseen",
            target_world_models=[0, 1, 2],
            datasets=[8],
            trainer_args=test_trainer_args,
            num_eval_samples=300,
        ),
    ],
    test_continuously=False,
    sentence_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    retrieval_k=3,
    prototype_size=15,
)
