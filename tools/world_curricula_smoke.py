"""Minimal smoke curricula for `wormi world train` — VirtualHome only.

Tests that the dataset loader, chat-template formatting, SFT trainer wiring and
checkpoint saving all work end-to-end. Uses the Unsloth mirror of Llama-3.2-1B
(un-gated, already in $HF_HOME on this machine) so no HF auth is needed.

Run on a compute node with CUDA available:

    cd /home/z5524306/WorMI
    uv sync          # one-off, populates .venv from pyproject.toml
    uv run wormi world train \
        --curricula_path tools/world_curricula_smoke.py
"""

from pathlib import Path

from wormi.curricula import WorldModelCurricula, WorldModelCurriculum
from wormi.trainer import SchedulerType, WorMITrainerConfig

dataset_root = Path("/srv/scratch/z5524306/wormi-data/virtualhome")

# Smoke the same scene-keyed layout as the paper-aligned VH pipeline.
scenes = ["scene_0"]

base_model = "unsloth/Llama-3.2-1B-Instruct"

aux_trainer_args = WorMITrainerConfig(
    max_steps=20,
    eval_steps=20,
    save_steps=20,
    logging_steps=5,
    batch_size=2,
    lr_scheduler_type=SchedulerType.COSINE,
)

curricula = WorldModelCurricula(
    output_dir=Path("/srv/scratch/z5524306/wormi-checkpoints/smoke-vh"),
    curricula=[
        WorldModelCurriculum(
            name=scene,
            base_model=base_model,
            tokenizer=base_model,
            dataset=dataset_root / scene,
            trainer_args=aux_trainer_args,
            behavior_cloning=False,
        )
        for scene in scenes
    ],
)
