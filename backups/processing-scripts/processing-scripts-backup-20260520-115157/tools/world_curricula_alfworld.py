"""Production ALFWorld world-model curricula — one Llama-3.2-1B world model
per task type (paper §4 / Table A.4 / Table A.6: N=6 world models).

Per paper Section 4 + Table A.4 ALFWorld defines 6 task types:
    pick_and_place_simple          → task_pick_simple   (4 seen)
    look_at_obj_in_light           → task_look_at_obj   (4 seen)
    pick_heat_then_place_in_recep  → task_pick_heat     (4 seen)
    pick_cool_then_place_in_recep  → task_pick_cool     (4 seen)
    pick_two_obj_and_place         → task_pick_two      (2 unseen)
    pick_clean_then_place_in_recep → task_pick_clean    (2 unseen)

All 6 are trained at stage 1 on the 3 seen rooms' data
(bedrooms / kitchens / livingrooms). The seen/unseen distinction applies at
the WorMI integration level: at test time the 2 unseen-task world models
are retrieved for unseen-task queries (Algorithm 1 line 28).

The 1 unseen room (bathrooms) is held out for Table 1 col 2/col 3 evals,
no per-room world model.

Data prepared by `tools/resplit_alfworld_by_task_type.py`.

Used by `sh/wormi-train-alfworld-world.sh` for unattended PBS training.

Output checkpoints land at:
    /srv/scratch/z5524306/wormi-checkpoints/world-alfworld/<task>/last
"""

from pathlib import Path

from wormi.curricula import WorldModelCurricula, WorldModelCurriculum
from wormi.trainer import SchedulerType, WorMITrainerConfig

DATASET_ROOT = Path("/srv/scratch/z5524306/wormi-data/alfworld")
OUTPUT_DIR = Path("/srv/scratch/z5524306/wormi-checkpoints/world-alfworld")

# Paper Table A.4: 6 task types. Order kept stable for reproducibility.
TASKS = [
    "task_pick_simple",
    "task_look_at_obj",
    "task_pick_heat",
    "task_pick_cool",
    "task_pick_two",
    "task_pick_clean",
]

# unsloth mirror = unrestricted Llama-3.2-1B-Instruct weights (already in
# $HF_HOME on this user's scratch). meta-llama/ originals would need a gated
# license + HF token. Matches the VH world-model setup for apples-to-apples
# comparison at the WorMI integration stage.
BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"

# Paper Table A.6: batch=4, 2000 gradient steps, lr=3e-5, cosine.
trainer_args = WorMITrainerConfig(
    max_steps=2000,
    eval_steps=500,
    save_steps=1000,
    logging_steps=20,
    batch_size=4,
    learning_rate=3e-5,
    lr_scheduler_type=SchedulerType.COSINE,
)

curricula = WorldModelCurricula(
    output_dir=OUTPUT_DIR,
    curricula=[
        WorldModelCurriculum(
            name=task,
            base_model=BASE_MODEL,
            tokenizer=BASE_MODEL,
            dataset=DATASET_ROOT / task,
            trainer_args=trainer_args,
            # Paper §3.2 trains world models on transition tuples
            # (instruction, state_t, action_t, state_{t+1}); keep next-state
            # supervision instead of action-only behavior cloning.
            behavior_cloning=False,
        )
        for task in TASKS
    ],
)
