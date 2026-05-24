from dataclasses import dataclass, field
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer, HfArgumentParser

from wormi.curricula import load_wormi_curricula
from wormi.model import WorMI, WorMIConfig
from wormi.trainer import WorMIMetaLearningTrainer, WorMITrainer


@dataclass
class WorMITrainingArgs:
    curricula_path: Path | None = field(
        metadata={"help": "Path to the curricula python script file."}
    )
    test: bool = field(
        metadata={"help": "Whether to test the model after training."},
        default=False,
    )
    interactive: bool = field(
        metadata={"help": "Whether to run the model interactively."},
        default=False,
    )


def main(args):
    """Train a WorMI model on a curricula."""
    args = parse_args(args)

    if args.curricula_path is None:
        raise ValueError("Curricula path must be provided.")

    curricula = load_wormi_curricula(args.curricula_path)

    if curricula.resume_from is not None:
        model_name = curricula.resume_from
        config = AutoConfig.from_pretrained(model_name)
        model = WorMI.from_pretrained(model_name, config=config)
        # WorMIConfig field is `base_model`; CLAUDE.md noted the older
        # `main_model` name was a drift between this script and the model.
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    else:
        config = WorMIConfig(
            method=curricula.method.value,
            base_model=curricula.base_model,
            connections=curricula.connections,
            num_heads=curricula.num_heads,
            self_attention=curricula.self_attention,
            world_wise_positional_encoding=curricula.model_wise_positional_encoding,
            vision=curricula.vision,
        )
        model = WorMI(config)
        tokenizer = AutoTokenizer.from_pretrained(curricula.base_model)
    model.to("cuda")

    if curricula.meta_learning:
        trainer = WorMIMetaLearningTrainer(
            model=model, tokenizer=tokenizer, curricula=curricula
        )
    else:
        trainer = WorMITrainer(
            model=model, tokenizer=tokenizer, curricula=curricula
        )
    if not args.test:
        trainer.train(end_with_action=True, cumulative=True)
    trainer.test(interactive=args.interactive, end_with_action=True)


def parse_args(args) -> WorMITrainingArgs:
    parser = HfArgumentParser(WorMITrainingArgs)  # type: ignore
    return parser.parse_args_into_dataclasses(args)[0]  # type: ignore
