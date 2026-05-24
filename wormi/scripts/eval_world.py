from __future__ import annotations

import argparse
import enum
import json
from functools import partial, reduce
from pathlib import Path
from typing import Literal

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from wormi.datasets.auto_jsonl import AutoJsonlDataset
from wormi.datasets.jsonl import JsonlDataset


class WorldModelEvaluationArgs(argparse.Namespace):
    _dataset_path: list[str]
    model_name: str
    _output_path: str
    num_samples: int

    @property
    def dataset_path(self):
        return [Path(x) for x in self._dataset_path]

    @property
    def output_path(self):
        return Path(self._output_path)


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


def main(args):
    """Evaluate a world model."""
    args = parse_args(args)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model.to("cuda")

    dataset = reduce(
        JsonlDataset.merge,
        map(
            partial(
                AutoJsonlDataset.load,
                end_with_action=True,
                cumulative=True,
            ),
            args.dataset_path,
        ),
    )
    dataset = dataset.as_chat(tokenizer)
    if args.num_samples > 0:
        if len(dataset) < args.num_samples:
            raise ValueError(
                f"Number of samples ({len(dataset)}) "
                f"exceeds the total number of samples ({args.num_samples})."
            )
        dataset = dataset.shuffle().take(args.num_samples)

    if not args.output_path.parent.exists():
        args.output_path.parent.mkdir(parents=True)

    f = open(args.output_path, "w")

    evaluator = TaskSuccessEvaluator()
    with tqdm(dataset, total=len(dataset), desc="Evaluating") as pbar:
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
                        },
                    }
                )
            )
            f.write("\n")
            f.flush()
            pbar.set_postfix({"accuracy": evaluator.accuracy})

    print(f"Accuracy: {evaluator.accuracy:.2%}")
    f.write(
        json.dumps({"message": "Test success rate", "data": evaluator.accuracy})
    )

    f.close()


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        dest="_dataset_path",
        nargs="+",
    )
    parser.add_argument("--model-name", "--model", required=True, type=str)
    parser.add_argument(
        "--output-path",
        "--output",
        required=True,
        type=str,
        dest="_output_path",
    )
    parser.add_argument(
        "--num-samples", "--sample", type=int, default=-1, dest="num_samples"
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default="all",
        choices=["all", "word", "word-odd"],
        dest="_method",
    )
    return parser.parse_args(args, namespace=WorldModelEvaluationArgs())
