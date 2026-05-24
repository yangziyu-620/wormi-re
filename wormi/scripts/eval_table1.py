from __future__ import annotations

"""Offline Table-1-style evaluation.

This reports SR and PS over jsonl expert trajectories. It advances along the
recorded trajectory only when the predicted action matches the expert action;
it does not restore simulator state or execute actions inside VirtualHome /
ALFWorld. Use it for reproducible jsonl-based evaluation, not as a replacement
for a full environment rollout.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)

from wormi.curricula import load_wormi_curricula
from wormi.datasets.alfworld import BASE_PROMPT as ALFWORLD_BASE_PROMPT
from wormi.datasets.virtualhome import BASE_PROMPT as VIRTUALHOME_BASE_PROMPT
from wormi.model import WorMI, WorMIConfig
from wormi.scripts.eval import (
    _build_world_prototypes,
    _prototype_texts,
    _select_world_models,
)


@dataclass
class Table1EvaluationArgs:
    curricula_path: Path = field(
        metadata={"help": "Path to the WorMI curricula python script file."}
    )
    model_name: str | None = field(
        metadata={"help": "Path/name of the trained WorMI checkpoint."},
        default=None,
    )
    output_path: Path | None = field(
        metadata={"help": "Directory for eval-table1 outputs."},
        default=None,
    )
    num_samples: int | None = field(
        metadata={"help": "Optional per-column episode sample cap."},
        default=None,
    )
    seed: int = field(default=42, metadata={"help": "Sampling seed."})
    device: str = field(default="cuda", metadata={"help": "Torch device."})
    max_new_tokens: int = field(
        default=24, metadata={"help": "Max generated tokens per action."}
    )


@dataclass
class EpisodeResult:
    success: bool
    pending_steps: int
    total_steps: int
    matched_steps: int
    step_details: list[dict[str, Any]] | None = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _normalize_action(text: str) -> str:
    text = text.strip().lower().strip("\"'`.,")
    return " ".join(text.split())


def _decode_action(tokenizer, outputs) -> str:
    pred = tokenizer.decode(outputs, skip_special_tokens=True)
    pred = pred.split("assistant")[-1]
    if "<|end_header_id|>" in pred:
        pred = pred.split("<|end_header_id|>", 1)[1]
    if "<|eot_id|>" in pred:
        pred = pred.split("<|eot_id|>", 1)[0]
    lines = pred.strip().splitlines()
    if not lines:
        return ""
    return lines[0].strip()


def _render_prompt(tokenizer, chat: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(chat, tokenize=False)
        return text + "<|start_header_id|>assistant<|end_header_id|>\n\n"


def _generate_action(model, tokenizer, chat, max_new_tokens: int) -> str:
    prompt = _render_prompt(tokenizer, chat)
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            max_length=input_ids["input_ids"].shape[-1] + max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            tokenizer=tokenizer,
            use_cache=False,
        )[0]
    return _decode_action(tokenizer, outputs)


def _vh_chat(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": VIRTUALHOME_BASE_PROMPT},
        {
            "role": "user",
            "content": (
                f"Instruction: {row['instruction']}\n\n"
                f"Observation: {row['observation']}\n\n"
                f"Action: "
            ),
        },
    ]


def _alfworld_chat(
    task: str,
    initial_observation: str,
    prefix: list[dict[str, str | None]],
) -> list[dict[str, str]]:
    chat = [
        {"role": "system", "content": ALFWORLD_BASE_PROMPT},
        {
            "role": "user",
            "content": (
                f"Instruction: {task}\n\n"
                f"Observation: {initial_observation}\n\n"
                f"Action: "
            ),
        },
    ]
    for i, elem in enumerate(prefix):
        if i != 0:
            chat.append({"role": "user", "content": "Action: "})
        if elem["action"] is not None:
            chat.append({"role": "assistant", "content": elem["action"]})
        if elem["observation"] is not None:
            chat.append({"role": "user", "content": "Next observation: "})
            chat.append({"role": "assistant", "content": elem["observation"]})
    return chat


def _group_virtualhome(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    fallback = 0
    for row in rows:
        meta = row.get("_meta", {})
        trajectory_id = meta.get("trajectory_id")
        if trajectory_id is None:
            trajectory_id = f"legacy-row-{fallback}"
            fallback += 1
        grouped.setdefault(str(trajectory_id), []).append(row)

    episodes = []
    for episode in grouped.values():
        episodes.append(
            sorted(
                episode,
                key=lambda r: int(r.get("_meta", {}).get("step_index", 0)),
            )
        )
    return episodes


def _eval_virtualhome_episode(
    model,
    tokenizer,
    episode: list[dict[str, Any]],
    max_new_tokens: int,
) -> EpisodeResult:
    matched = 0
    step_details: list[dict[str, Any]] = []
    for local_step, row in enumerate(episode):
        pred = _generate_action(model, tokenizer, _vh_chat(row), max_new_tokens)
        pred_norm = _normalize_action(pred)
        target_norm = _normalize_action(row["action"])
        is_match = pred_norm == target_norm
        meta = row.get("_meta", {})
        step_details.append(
            {
                "local_step": local_step,
                "dataset_step_index": meta.get("step_index"),
                "scene": meta.get("scene"),
                "instruction": row.get("instruction"),
                "target_action": row.get("action"),
                "prediction": pred,
                "target_norm": target_norm,
                "prediction_norm": pred_norm,
                "matched": is_match,
            }
        )
        if not is_match:
            break
        matched += 1
    total = len(episode)
    return EpisodeResult(
        success=matched == total,
        pending_steps=total - matched,
        total_steps=total,
        matched_steps=matched,
        step_details=step_details,
    )


def _eval_alfworld_episode(
    model,
    tokenizer,
    row: dict[str, Any],
    max_new_tokens: int,
) -> EpisodeResult:
    history = row["history"]
    if not history:
        return EpisodeResult(False, 0, 0, 0)

    initial_observation = history[0]["observation"]
    prefix: list[dict[str, str | None]] = []
    matched = 0
    for hist in history:
        prefix.append({"action": None, "observation": None})
        pred = _generate_action(
            model,
            tokenizer,
            _alfworld_chat(row["task"], initial_observation, prefix),
            max_new_tokens,
        )
        if _normalize_action(pred) != _normalize_action(hist["action"]):
            break
        matched += 1
        prefix[-1]["action"] = hist["action"]
        prefix[-1]["observation"] = hist["next_observation"]

    total = len(history)
    return EpisodeResult(
        success=matched == total,
        pending_steps=total - matched,
        total_steps=total,
        matched_steps=matched,
        step_details=None,
    )


def _load_eval_rows(paths: list[Path]) -> tuple[str, list[Any]]:
    rows = []
    for path in paths:
        rows.extend(_read_jsonl(path))
    if not rows:
        raise ValueError(f"No rows found in {paths}")

    first = rows[0]
    if "history" in first:
        return "alfworld", rows
    if {"instruction", "observation", "action", "next_observation"} <= first.keys():
        return "virtualhome", _group_virtualhome(rows)
    raise ValueError(f"Unsupported Table 1 eval schema in {paths[0]}")


def _select_samples(items: list[Any], limit: int | None, seed: int) -> list[Any]:
    if limit is None or limit < 0 or limit >= len(items):
        return items
    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    return [items[i] for i in indices[:limit]]


def _summarize(results: list[EpisodeResult]) -> dict[str, float | int]:
    if not results:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "pending_steps": 0.0,
            "avg_total_steps": 0.0,
        }
    return {
        "episodes": len(results),
        "success_rate": sum(r.success for r in results) / len(results),
        "pending_steps": sum(r.pending_steps for r in results) / len(results),
        "avg_total_steps": sum(r.total_steps for r in results) / len(results),
    }


def main(args):
    """Evaluate WorMI on Table-1-style SR/PS over jsonl trajectories."""
    args = parse_args(args)
    curricula = load_wormi_curricula(args.curricula_path)

    model_name = args.model_name or str(
        curricula.output_dir / curricula.run_name / "last"
    )
    out_dir = args.output_path or curricula.output_dir / curricula.run_name / "table1"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = WorMIConfig.from_pretrained(model_name)
    model = WorMI.from_pretrained(
        model_name, config=config, torch_dtype=torch.bfloat16
    )
    model.eval()
    model.to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = "<|end_of_text|>"

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

    summary_rows = []
    for i, curriculum in enumerate(curricula.test):
        name = curriculum.name or f"curriculum-{i + 1}"
        dataset_paths = [
            Path(curricula.datasets[j]) / "test.jsonl" for j in curriculum.datasets
        ]

        dataset_type, episodes = _load_eval_rows(dataset_paths)
        sample_cap = args.num_samples
        if sample_cap is None:
            sample_cap = curriculum.num_eval_samples
        episodes = _select_samples(episodes, sample_cap, args.seed)

        proto_strings = _prototype_texts(dataset_paths)
        selected_idx, targets = _select_world_models(
            curricula, curriculum, store, world_prototypes, proto_strings
        )
        model.remove_all()
        for target in targets:
            aux_model = AutoModelForCausalLM.from_pretrained(
                target.model_name, torch_dtype=torch.bfloat16
            )
            model.implant(aux_model, target.connections)
        model.to(model.device)

        detail_path = out_dir / f"table1-{name}.jsonl"
        results: list[EpisodeResult] = []
        with detail_path.open("w") as detail_file:
            with tqdm(
                episodes,
                total=len(episodes),
                desc=f"Table1 {name}",
            ) as pbar:
                for episode_idx, episode in enumerate(pbar):
                    if dataset_type == "virtualhome":
                        result = _eval_virtualhome_episode(
                            model, tokenizer, episode, args.max_new_tokens
                        )
                    else:
                        result = _eval_alfworld_episode(
                            model, tokenizer, episode, args.max_new_tokens
                        )
                    results.append(result)
                    metrics = _summarize(results)
                    pbar.set_postfix(
                        {
                            "SR": f"{metrics['success_rate']:.2%}",
                            "PS": f"{metrics['pending_steps']:.2f}",
                        }
                    )
                    detail_file.write(
                        json.dumps(
                            {
                                "episode": episode_idx,
                                "success": result.success,
                                "pending_steps": result.pending_steps,
                                "total_steps": result.total_steps,
                                "matched_steps": result.matched_steps,
                                "retrieved_world_models": selected_idx,
                                "steps": result.step_details,
                            }
                        )
                        + "\n"
                    )
                    detail_file.flush()

        metrics = _summarize(results)
        row = {
            "name": name,
            "dataset_type": dataset_type,
            "retrieved_world_models": selected_idx,
            **metrics,
        }
        summary_rows.append(row)
        print(
            f"{name}: SR={metrics['success_rate']:.2%}, "
            f"PS={metrics['pending_steps']:.2f}, n={metrics['episodes']}"
        )

    summary_path = out_dir / "table1-summary.json"
    with summary_path.open("w") as f:
        json.dump(summary_rows, f, indent=2)

    tsv_path = out_dir / "table1-summary.tsv"
    with tsv_path.open("w") as f:
        f.write("name\tdataset_type\tepisodes\tSR\tPS\tavg_total_steps\tworld_models\n")
        for row in summary_rows:
            f.write(
                f"{row['name']}\t{row['dataset_type']}\t{row['episodes']}\t"
                f"{row['success_rate']:.6f}\t{row['pending_steps']:.6f}\t"
                f"{row['avg_total_steps']:.6f}\t"
                f"{','.join(map(str, row['retrieved_world_models']))}\n"
            )

    print(f"Summary written to {summary_path}")


def parse_args(args) -> Table1EvaluationArgs:
    parser = HfArgumentParser(Table1EvaluationArgs)  # type: ignore
    return parser.parse_args_into_dataclasses(args)[0]  # type: ignore
