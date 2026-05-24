from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from wormi.curricula import WorldModelCurricula, load_world_model_curricula
from wormi.datasets.auto_jsonl import AutoJsonlDataset
from wormi.trainer import WorMISubTrainer, WorMITrainerConfig


@dataclass
class WorldModelTrainingArgs:
    curricula_path: Path | None = field(
        metadata={"help": "Path to the curricula file."}, default=None
    )
    dataset_path: Path | None = field(
        metadata={"help": "Path to the dataset directory."}, default=None
    )
    model_name: str | None = field(
        metadata={"help": "Name of the pretrained world model to use."},
        default=None,
    )
    test: bool = field(
        metadata={"help": "Whether to run in test mode."}, default=False
    )
    interactive: bool = field(
        metadata={"help": "Whether to run in interactive test mode."},
        default=False,
    )


def train_by_curricula(curricula: WorldModelCurricula):
    for curr in curricula.curricula:
        if curr.base_model is None:
            raise ValueError("Base model must be specified.")

        tokenizer = AutoTokenizer.from_pretrained(curr.tokenizer)
        # Load weights in bf16. The trainer runs bf16 mixed-precision anyway,
        # and this also halves both the on-disk size of every save_pretrained
        # checkpoint (4.7 GiB -> 2.5 GiB for a 1B model) and the GPU memory
        # footprint of the frozen reference parameters.
        target_model = AutoModelForCausalLM.from_pretrained(
            curr.base_model, torch_dtype=torch.bfloat16
        )

        dataset_path = curr.dataset
        assert dataset_path is not None, "Dataset path must be specified."

        train_dataset = AutoJsonlDataset.load(
            dataset_path / "train.jsonl",
            end_with_action=curr.behavior_cloning,
            cumulative=True,
        )
        if (dataset_path / "unknown.jsonl").exists():
            train_dataset = train_dataset.merge(
                AutoJsonlDataset.load(
                    dataset_path / "unknown.jsonl",
                    end_with_action=curr.behavior_cloning,
                    cumulative=True,
                )
            )
        test_dataset = AutoJsonlDataset.load(
            dataset_path / "test.jsonl",
            end_with_action=curr.behavior_cloning,
            cumulative=True,
        )

        train_dataset = train_dataset.as_chat(tokenizer).shuffle()
        test_dataset = test_dataset.as_chat(tokenizer)

        if curr.num_train_samples:
            if curr.num_train_samples > len(train_dataset):
                raise ValueError(
                    f"Number of training samples ({curr.num_train_samples}) "
                    f"exceeds the total number of samples ({len(train_dataset)})."
                )
            train_dataset = train_dataset.take(curr.num_train_samples)
        if curr.num_eval_samples:
            if curr.num_eval_samples > len(test_dataset):
                raise ValueError(
                    f"Number of evaluation samples ({curr.num_eval_samples}) "
                    f"exceeds the total number of samples ({len(test_dataset)})."
                )
            test_dataset = test_dataset.take(curr.num_eval_samples)

        trainer_args = curr.trainer_args or WorMITrainerConfig()
        trainer_args.output_dir = curricula.output_dir
        trainer_args.run_name = curr.name

        trainer = WorMISubTrainer(
            model=target_model,
            args=trainer_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
        )
        trainer.train()
        trainer.test(test_dataset, interactive=False, sample=5)

        del tokenizer
        del target_model
        del train_dataset
        del test_dataset
        del trainer


def main(args):
    """Train a world model."""
    args, trainer_args = parse_args(args)

    if args.curricula_path is not None:
        curricula = load_world_model_curricula(args.curricula_path)
        train_by_curricula(curricula)
        return

    if args.dataset_path is None:
        raise ValueError("Dataset path must be specified.")

    if args.model_name is None:
        raise ValueError("Model name must be specified.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    )

    train_dataset = AutoJsonlDataset.load(args.dataset_path / "train.jsonl")
    if (args.dataset_path / "unknown.jsonl").exists():
        train_dataset = train_dataset.merge(
            AutoJsonlDataset.load(args.dataset_path / "unknown.jsonl")
        )
    test_dataset = AutoJsonlDataset.load(args.dataset_path / "test.jsonl")

    train_dataset = train_dataset.as_chat(tokenizer).shuffle()
    test_dataset = test_dataset.as_chat(tokenizer)

    trainer = WorMISubTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    if not args.test:
        trainer.train()
    trainer.test(
        test_dataset,
        interactive=args.interactive,
        sample=5 if not args.interactive else -1,
    )


def parse_args(args) -> tuple[WorldModelTrainingArgs, WorMITrainerConfig]:
    parser = HfArgumentParser((WorldModelTrainingArgs, WorMITrainerConfig))  # type: ignore
    return parser.parse_args_into_dataclasses(args)  # type: ignore
