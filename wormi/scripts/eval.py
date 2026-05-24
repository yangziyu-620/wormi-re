from __future__ import annotations

import enum
import json
import re
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)

from wormi.curricula import load_wormi_curricula
from wormi.datasets.auto_jsonl import AutoJsonlDataset
from wormi.datasets.jsonl import JsonlDataset
from wormi.model import WorMI, WorMIConfig
from wormi.model_store import ModelStore


@dataclass
class WorMIEvaluationArgs:
    curricula_path: Path = field(
        metadata={"help": "Path to the curricula python script file."}
    )
    model_name: str | None = field(
        metadata={"help": "Name of the model to evaluate."}, default=None
    )


class EvaluationStatus(enum.Enum):
    SUCCESS = enum.auto()
    FAILURE = enum.auto()


class TaskSuccessEvaluator:
    def __init__(self):
        self.__total = 0
        self.__success = 0

    @property
    def accuracy(self) -> float:
        if self.__total == 0:
            return 0
        return self.__success / self.__total

    def update(self, pred: str, answer: str) -> EvaluationStatus:
        self.__total += 1
        if pred == answer:
            status = EvaluationStatus.SUCCESS
            self.__success += 1
        else:
            status = EvaluationStatus.FAILURE
        return status


# Cap on the number of strings we feed through the sentence encoder when
# computing a prototype set. Paper Table A.6 uses k=15 cluster centers, so
# anything above a few hundred samples is well past the saturation point.
_PROTOTYPE_SAMPLE_CAP = 1024
_VH_TRIPLE_RE = re.compile(r"\(([^()]*)\)")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _split_text_state(text: str | None) -> list[str]:
    if not text:
        return []
    parts = []
    for line in text.splitlines():
        for part in re.split(r"(?<=[.!?])\s+", line.strip()):
            if part:
                parts.append(part)
    return parts or [text]


def _virtualhome_state_texts(row: dict) -> list[str]:
    texts = []
    for key in ("observation", "next_observation"):
        value = row.get(key)
        if not value:
            continue
        triples = [f"({m.group(1).strip()})" for m in _VH_TRIPLE_RE.finditer(value)]
        texts.extend(triples or [value])
    return texts


def _alfworld_state_texts(row: dict) -> list[str]:
    texts = []
    for hist in row.get("history", []):
        texts.extend(_split_text_state(hist.get("observation")))
        texts.extend(_split_text_state(hist.get("next_observation")))
    return texts


def _prototype_texts(dataset_paths: list[Path]) -> list[str]:
    """Extract paper-style state/object texts for retrieval prototypes."""
    texts = []
    for path in dataset_paths:
        for row in _read_jsonl(path):
            if {"instruction", "observation", "action", "next_observation"} <= row.keys():
                texts.extend(_virtualhome_state_texts(row))
            elif "history" in row:
                texts.extend(_alfworld_state_texts(row))
            else:
                texts.extend(str(v) for v in row.values() if isinstance(v, str))
    return texts


def _build_world_prototypes(
    curricula,
    se_model,
    se_tokenizer,
):
    """One prototype set per world model, computed from its associated stage-1
    training dataset (the i-th world model is paired with the i-th dataset).
    Cached on the curricula object so re-eval is cheap.
    """
    store = ModelStore(
        se_model, se_tokenizer, n_clusters=curricula.prototype_size
    )
    prototypes = []
    print(
        f"Computing prototypes for {len(curricula.world_models)} world models "
        f"(prototype_size={curricula.prototype_size}, encoder="
        f"{curricula.sentence_embedding_model})..."
    )
    for i, _wm in enumerate(curricula.world_models):
        train_path = Path(curricula.datasets[i]) / "train.jsonl"
        if not train_path.exists():
            raise FileNotFoundError(
                f"World model {i}'s associated dataset train.jsonl not "
                f"found at {train_path}. Required for prototype computation."
            )
        strings = _prototype_texts([train_path])
        if len(strings) > _PROTOTYPE_SAMPLE_CAP:
            strings = strings[:_PROTOTYPE_SAMPLE_CAP]
        p = store._compute_prototype(strings)
        prototypes.append(p)
        print(f"  [{i}] {curricula.datasets[i].name}: {len(strings)} strings → prototype set ready")
    return store, prototypes


def _select_world_models(
    curricula,
    curriculum,
    store,
    world_prototypes,
    test_strings: list[str],
):
    """Return (selected_world_model_indices, selected_world_models). If
    retrieval is configured, do paper-Algorithm-1 Wasserstein top-K retrieval;
    otherwise fall back to the curriculum-hardcoded indices for backward
    compatibility with toy README configs.
    """
    if world_prototypes is None:
        idx = list(curriculum.world_models)
        return idx, [curricula.world_models[j] for j in idx]

    if len(test_strings) > _PROTOTYPE_SAMPLE_CAP:
        test_strings = test_strings[:_PROTOTYPE_SAMPLE_CAP]
    test_p = store._compute_prototype(test_strings)
    order = sorted(
        range(len(world_prototypes)),
        key=lambda j: float(world_prototypes[j].dist(test_p)),
    )
    idx = order[: curricula.retrieval_k]
    return idx, [curricula.world_models[j] for j in idx]


def main(args):
    """Evaluate a WorMI model on a curricula. Implements paper Algorithm 1
    test-time retrieval when curricula.sentence_embedding_model is set.
    """
    args = parse_args(args)

    curricula = load_wormi_curricula(args.curricula_path)

    if args.model_name is not None:
        model_name = args.model_name
    else:
        model_name = str(curricula.output_dir / curricula.run_name / "last")

    config = WorMIConfig.from_pretrained(model_name)

    model = WorMI.from_pretrained(
        model_name, config=config, torch_dtype=torch.bfloat16
    )
    model.eval()
    model.to("cuda")

    # WorMIConfig field is `base_model` (CLAUDE.md noted the older `main_model`
    # name was a drift; fixed here to make the script runnable).
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = "<|end_of_text|>"

    if not curricula.test:
        print("No test curricula found, skipping evaluation")
        return

    # Build sentence encoder + per-world-model prototypes once if retrieval
    # is configured. Reused across all test curricula.
    store = None
    world_prototypes = None
    if curricula.sentence_embedding_model:
        se_tokenizer = AutoTokenizer.from_pretrained(
            curricula.sentence_embedding_model
        )
        se_model = AutoModel.from_pretrained(
            curricula.sentence_embedding_model
        ).to(model.device)
        se_model.eval()
        store, world_prototypes = _build_world_prototypes(
            curricula, se_model, se_tokenizer
        )

    evaluator: TaskSuccessEvaluator | None = None
    for i, curriculum in enumerate(curricula.test):
        name = curriculum.name or f"curriculum-{i + 1}"
        out_dir = curricula.output_dir / curricula.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        f = open(out_dir / f"eval-{name}.jsonl", "w")

        print(f"Testing curriculum {i + 1}/{len(curricula.test)}: {name}")

        test_datasets = [
            AutoJsonlDataset.load(
                Path(curricula.datasets[j]) / "test.jsonl",
                end_with_action=True,
                cumulative=True,
            )
            for j in curriculum.datasets
        ]
        test_dataset = reduce(JsonlDataset.merge, test_datasets)
        test_proto_strings = _prototype_texts(
            [Path(curricula.datasets[j]) / "test.jsonl" for j in curriculum.datasets]
        )

        if not curricula.test_continuously or evaluator is None:
            match test_dataset.dataset_type.split("/")[0]:
                case "virtualhome" | "alfworld":
                    evaluator = TaskSuccessEvaluator()
                case _:
                    raise ValueError(
                        f"Unknown dataset type: {test_dataset.dataset_type}"
                    )

        test_dataset = test_dataset.as_chat(tokenizer)

        if curriculum.num_eval_samples is not None:
            if curriculum.num_eval_samples > len(test_dataset):
                raise ValueError(
                    f"Number of evaluation samples ({curriculum.num_eval_samples}) "
                    f"exceeds the total number of samples ({len(test_dataset)})."
                )
            test_dataset = test_dataset.shuffle().take(
                curriculum.num_eval_samples
            )

        # Paper Algorithm 1 lines 18-31 — pick which world models to implant
        # for this test curriculum.
        selected_idx, test_targets = _select_world_models(
            curricula, curriculum, store, world_prototypes, test_proto_strings
        )
        print(f"  world models used: {selected_idx}")

        # WorMI.implant / WorMI.remove_all are the actual API names
        # (older `plug` / `unplug_all` in the script were never wired up).
        model.remove_all()
        for target in test_targets:
            aux_model = AutoModelForCausalLM.from_pretrained(
                target.model_name, torch_dtype=torch.bfloat16
            )
            model.implant(aux_model, target.connections)
        model.to(model.device)

        with tqdm(
            test_dataset,
            total=len(test_dataset),
            desc=f"Evaluating ({i + 1}/{len(curricula.test)})",
        ) as pbar:
            for elem in pbar:
                prompt = elem["text"]  # type: ignore
                *splits, answer = prompt.split("assistant<|end_header_id|>")
                prompt = "assistant<|end_header_id|>".join(splits)
                prompt += "assistant<|end_header_id|>\n\n"
                answer = answer.split("<|eot_id|>")[0].strip()

                input_ids = tokenizer(prompt, return_tensors="pt")
                input_ids = input_ids.to(model.device)  # type: ignore

                outputs = model.generate(
                    **input_ids,  # type: ignore
                    max_length=input_ids["input_ids"].shape[-1] + 20,  # type: ignore
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    tokenizer=tokenizer,
                    use_cache=False,
                )[0]
                pred = tokenizer.decode(outputs, skip_special_tokens=True)
                pred = pred.split("assistant")[-1]
                if "<|end_header_id|>" in pred:
                    pred = pred.split("<|end_header_id|>")[1]
                if "<|eot_id|>" in pred:
                    pred = pred.split("<|eot_id|>")[0]
                pred = pred.strip()

                match evaluator.update(pred, answer):
                    case EvaluationStatus.SUCCESS:
                        msg = "Prediction success"
                    case EvaluationStatus.FAILURE:
                        msg = "Prediction failed"
                f.write(
                    json.dumps(
                        {
                            "message": msg,
                            "data": {
                                "prompt": prompt,
                                "answer": answer,
                                "pred": pred,
                                "retrieved_world_models": selected_idx,
                            },
                        }
                    )
                )
                f.write("\n")
                f.flush()
                pbar.set_postfix({"accuracy": evaluator.accuracy})

        print(f"Accuracy: {evaluator.accuracy:.2%}")
        f.write(
            json.dumps(
                {"message": "Test success rate", "data": evaluator.accuracy}
            )
        )

        f.close()


def parse_args(args) -> WorMIEvaluationArgs:
    parser = HfArgumentParser(WorMIEvaluationArgs)  # type: ignore
    return parser.parse_args_into_dataclasses(args)[0]  # type: ignore
