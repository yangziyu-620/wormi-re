"""Production VH world-model curricula — one Llama-3.2-1B world model per
seen scene. Paper §4 / Table A.6 (N=6 world models).

Each world model is trained on (16 seen tasks × that one seen scene)
trajectories. Variant choice per base apartment is fixed in
`tools/build_virtualhome_dataset.py::PAPER_SEEN_SCENE_KEYS`:
TrimmedTestScene1..5 use v0 and Scene6 uses v1 (Scene6_v0 has an invalid
init_graph).

Used by `sh/wormi-train-vh-world.sh` for unattended PBS training.

Output checkpoints land at:
    /srv/scratch/z5524306/wormi-checkpoints/world-vh/<scene>/last
"""

from pathlib import Path
import os

from wormi.curricula import WorldModelCurricula, WorldModelCurriculum
from wormi.trainer import SchedulerType, WorMITrainerConfig

DATA_DISK = Path(os.environ.get("WORMI_DATA_DISK", "/root/autodl-tmp"))
DATASET_ROOT = Path(
    os.environ.get("WORMI_VH_DATA_ROOT", DATA_DISK / "wormi-data" / "virtualhome")
)
OUTPUT_DIR = Path(
    os.environ.get("WORMI_WORLD_VH_OUTPUT_DIR", DATA_DISK / "wormi-checkpoints" / "world-vh")
)

# Paper §4 / Table A.6: N=6 seen scenes. Names match the per-scene dirs
# emitted by the VH builder (scene_0 .. scene_5). The env range allows resuming
# after a completed scene without overwriting its checkpoint.
SCENE_START = int(os.environ.get("WORMI_VH_SCENE_START", "0"))
SCENE_END = int(os.environ.get("WORMI_VH_SCENE_END", "6"))
SCENES = [f"scene_{i}" for i in range(SCENE_START, SCENE_END)]

# unsloth mirror = unrestricted Llama-3.2-1B-Instruct weights (already in
# $HF_HOME on this user's scratch). meta-llama/ originals would need a gated
# license + HF token.
BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"

# Paper Table A.6: batch=4, 2000 gradient steps, lr=3e-5, cosine. Transition
# supervision pushes VH samples close to 4096 tokens, so default to batch=2 on
# a 48GB RTX 4090; override with WORMI_WORLD_VH_BATCH_SIZE if needed.
trainer_args = WorMITrainerConfig(
    max_steps=2000,
    eval_steps=500,
    save_steps=1000,
    logging_steps=20,
    batch_size=int(os.environ.get("WORMI_WORLD_VH_BATCH_SIZE", "2")),
    learning_rate=3e-5,
    lr_scheduler_type=SchedulerType.COSINE,
)

curricula = WorldModelCurricula(
    output_dir=OUTPUT_DIR,
    curricula=[
        WorldModelCurriculum(
            name=scene,
            base_model=BASE_MODEL,
            tokenizer=BASE_MODEL,
            dataset=DATASET_ROOT / scene,
            trainer_args=trainer_args,
            # Paper §3.2 trains world models on transition tuples
            # (instruction, state_t, action_t, state_{t+1}); keep next-state
            # supervision instead of action-only behavior cloning.
            behavior_cloning=False,
        )
        for scene in SCENES
    ],
)
