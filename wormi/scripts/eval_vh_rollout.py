from __future__ import annotations

"""VirtualHome rollout evaluation using the EvolvingGraph environment.

This evaluator differs from ``eval-table1``: it resets an EvolvingGraph scene,
generates one action at a time, executes valid VirtualHome actions, and scores
task success from the final graph state.
"""

import copy
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.build_virtualhome_dataset import _bootstrap_evolving_graph, format_observation
from tools.compact_virtualhome_observations import (
    DEFAULT_NUM_EDGES,
    _format_triples,
    compact_observation,
    format_instance_grounded_observation,
    select_task_instances,
    selected_instance_ids_from_selection,
)
from wormi.curricula import load_wormi_curricula
from wormi.datasets.virtualhome import BASE_PROMPT as VIRTUALHOME_BASE_PROMPT
from wormi.model import WorMI, WorMIConfig
from wormi.scripts.eval import (
    _build_world_prototypes,
    _prototype_texts,
    _select_world_models,
)
from wormi.scripts.eval_table1 import _group_virtualhome, _read_jsonl, _select_samples


def _default_data_disk() -> Path:
    return Path(os.environ.get("WORMI_DATA_DISK", "/root/autodl-tmp"))


@dataclass
class VirtualHomeRolloutArgs:
    curricula_path: Path = field(
        metadata={"help": "Path to the WorMI curricula python script file."}
    )
    model_name: str | None = field(
        default=None,
        metadata={"help": "Path/name of the trained WorMI checkpoint."},
    )
    output_path: Path | None = field(
        default=None,
        metadata={"help": "Directory for rollout evaluation outputs."},
    )
    scene_inits_json: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "WORMI_VH_SCENE_INITS_JSON",
                str(_default_data_disk() / "wormi-data" / "scene-inits" / "init_graphs_20.json"),
            )
        ),
        metadata={"help": "JSON cache mapping scene keys to initial graph dicts."},
    )
    vh_src: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "WORMI_VH_SRC",
                str(_default_data_disk() / "wormi-data" / "virtualhome-src"),
            )
        ),
        metadata={"help": "Local VirtualHome source tree."},
    )
    num_samples: int | None = field(
        default=None,
        metadata={"help": "Optional per-column episode sample cap."},
    )
    seed: int = field(default=42, metadata={"help": "Sampling/generation seed."})
    device: str = field(default="cuda", metadata={"help": "Torch device."})
    max_steps: int = field(
        default=30, metadata={"help": "Maximum rollout steps per episode."}
    )
    max_new_tokens: int = field(
        default=24, metadata={"help": "Max generated tokens per action."}
    )
    temperature: float = field(
        default=1.0, metadata={"help": "Generation temperature. <=0 disables sampling."}
    )
    top_p: float = field(default=1.0, metadata={"help": "Nucleus sampling p."})
    observation_format: str = field(
        default="auto",
        metadata={
            "help": "Observation renderer for rollout: auto, full, or tmow_compact."
        },
    )
    compact_num_edges: int = field(
        default=DEFAULT_NUM_EDGES,
        metadata={"help": "Fallback edge budget for tmow_compact rollout observations."},
    )


@dataclass
class ParsedAction:
    verb: str
    args: list[str]


@dataclass
class RolloutResult:
    success: bool
    steps: int
    invalid_actions: int
    executed_actions: int
    goal: dict[str, Any]


_ACTION_PREFIX_RE = re.compile(r"\baction\s*:\s*", re.IGNORECASE)
_ARTICLE_WORDS = {"a", "an", "the"}
_ROOM_ALIASES = {
    "livingroom": ["livingroom", "living_room", "home_office"],
    "living room": ["livingroom", "living_room", "home_office"],
    "kitchen": ["kitchen", "dining_room"],
    "dining room": ["dining_room", "kitchen"],
    "bathroom": ["bathroom"],
    "bedroom": ["bedroom", "kids_bedroom"],
}


def _decode_action(tokenizer, outputs) -> str:
    pred = tokenizer.decode(outputs, skip_special_tokens=True)
    pred = pred.split("assistant")[-1]
    if "<|end_header_id|>" in pred:
        pred = pred.split("<|end_header_id|>", 1)[1]
    if "<|eot_id|>" in pred:
        pred = pred.split("<|eot_id|>", 1)[0]
    return pred.strip().splitlines()[0].strip()


def _resolve_observation_format(first: dict[str, Any], args: VirtualHomeRolloutArgs) -> str:
    requested = args.observation_format.lower().strip()
    if requested not in {"auto", "full", "tmow_compact"}:
        raise ValueError(
            "observation_format must be one of: auto, full, tmow_compact"
        )
    if requested != "auto":
        return requested

    prep = (first.get("_meta") or {}).get("observation_preprocessing") or {}
    mode = str(prep.get("mode", "")).lower()
    if mode.startswith("tmow_compact"):
        return "tmow_compact"
    return "full"


def _rollout_task_args(first: dict[str, Any]) -> list[str]:
    meta = first.get("_meta") or {}
    args = meta.get("task_args")
    if isinstance(args, list) and args:
        return [str(arg).replace(" ", "_").lower() for arg in args]
    return []


def _render_rollout_observation(
    graph: dict[str, Any],
    first: dict[str, Any],
    args: VirtualHomeRolloutArgs,
    selected_node_ids: list[int] | None = None,
) -> tuple[str, str]:
    mode = _resolve_observation_format(first, args)
    full = format_observation(graph)
    if mode == "full":
        return full, mode

    meta = first.get("_meta") or {}
    prep = meta.get("observation_preprocessing") or {}
    num_edges = int(prep.get("num_edges") or args.compact_num_edges)
    task_args = _rollout_task_args(first)
    source_observation = full
    if prep.get("instance_grounded", False):
        if selected_node_ids is None:
            selection = select_task_instances(
                graph, _infer_goal(first)["family"], task_args
            )
            selected_node_ids = selected_instance_ids_from_selection(selection)
        source_observation = format_instance_grounded_observation(
            graph,
            task_args=task_args,
            selected_node_ids=[int(node_id) for node_id in selected_node_ids],
        )
    triples = compact_observation(
        source_observation,
        instruction=str(first.get("instruction", "")),
        action="",
        task_args=task_args,
        num_edges=num_edges,
        fill_to_num_edges=bool(prep.get("fill_to_num_edges", True)),
    )
    return _format_triples(triples), mode


def _render_prompt(tokenizer, instruction: str, observation: str) -> str:
    chat = [
        {"role": "system", "content": VIRTUALHOME_BASE_PROMPT},
        {
            "role": "user",
            "content": (
                f"Instruction: {instruction}\n\n"
                f"Observation: {observation}\n\n"
                f"Action: "
            ),
        },
    ]
    try:
        return tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(chat, tokenize=False)
        return text + "<|start_header_id|>assistant<|end_header_id|>\n\n"


def _generate_action(
    model,
    tokenizer,
    instruction: str,
    observation: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    prompt = _render_prompt(tokenizer, instruction, observation)
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_kwargs: dict[str, Any] = {
        "max_length": input_ids["input_ids"].shape[-1] + max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "tokenizer": tokenizer,
        "use_cache": False,
    }
    if temperature > 0:
        generation_kwargs.update(
            {"do_sample": True, "temperature": temperature, "top_p": top_p}
        )
    else:
        generation_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**input_ids, **generation_kwargs)[0]
    return _decode_action(tokenizer, outputs)


def _clean_action_text(text: str) -> str:
    text = text.strip().replace("`", "")
    if _ACTION_PREFIX_RE.search(text):
        text = _ACTION_PREFIX_RE.split(text)[-1]
    lines = text.splitlines()
    if not lines:
        return ""
    text = lines[0]
    text = re.split(r"\s*(?:;|\||\t)\s*", text, maxsplit=1)[0]
    return text.strip().strip("\"'.,")


def _phrase_tokens(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9_ ]+", " ", text.lower().replace("_", " "))
    return [tok for tok in cleaned.split() if tok and tok not in _ARTICLE_WORDS]


def _norm_phrase(text: str) -> str:
    return " ".join(_phrase_tokens(text))


def _graph_classes(graph: dict[str, Any]) -> list[str]:
    classes = sorted({n["class_name"] for n in graph["nodes"]}, key=lambda s: (-len(s), s))
    return classes


def _class_norms(graph: dict[str, Any]) -> list[tuple[str, str]]:
    out = []
    for cls in _graph_classes(graph):
        out.append((_norm_phrase(cls), cls))
        spaced = cls.replace("_", " ")
        norm_spaced = _norm_phrase(spaced)
        if norm_spaced != out[-1][0]:
            out.append((norm_spaced, cls))
    return out


def _find_first_id(graph: dict[str, Any], class_name: str) -> int | None:
    for node in graph["nodes"]:
        if node["class_name"] == class_name:
            return int(node["id"])
    return None


def _nodes_by_id(graph: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(node["id"]): node for node in graph["nodes"]}


def _node_ids_by_class(graph: dict[str, Any], class_name: str) -> list[int]:
    return [int(node["id"]) for node in graph["nodes"] if node["class_name"] == class_name]


def _character_ids(graph: dict[str, Any]) -> set[int]:
    return {
        int(node["id"])
        for node in graph["nodes"]
        if node.get("class_name") == "character"
    }


def _character_room_ids(graph: dict[str, Any]) -> set[int]:
    nodes = _nodes_by_id(graph)
    chars = _character_ids(graph)
    rooms = set()
    for edge in graph["edges"]:
        if (
            int(edge["from_id"]) in chars
            and edge["relation_type"] == "INSIDE"
            and nodes.get(int(edge["to_id"]), {}).get("category") == "Rooms"
        ):
            rooms.add(int(edge["to_id"]))
    return rooms


def _node_room_ids(graph: dict[str, Any], node_id: int) -> set[int]:
    nodes = _nodes_by_id(graph)
    rooms = set()
    for edge in graph["edges"]:
        if (
            int(edge["from_id"]) == node_id
            and edge["relation_type"] == "INSIDE"
            and nodes.get(int(edge["to_id"]), {}).get("category") == "Rooms"
        ):
            rooms.add(int(edge["to_id"]))
    return rooms


def _character_close_ids(graph: dict[str, Any]) -> set[int]:
    chars = _character_ids(graph)
    close = set()
    for edge in graph["edges"]:
        if int(edge["from_id"]) in chars and edge["relation_type"] == "CLOSE":
            close.add(int(edge["to_id"]))
    return close


def _held_object_info(graph: dict[str, Any]) -> tuple[str, int] | None:
    nodes = _nodes_by_id(graph)
    chars = _character_ids(graph)
    for edge in graph["edges"]:
        if (
            int(edge["from_id"]) in chars
            and edge["relation_type"] in {"HOLDS_RH", "HOLDS_LH"}
        ):
            held_id = int(edge["to_id"])
            held = nodes.get(held_id)
            if held is not None:
                return held["class_name"], held_id
    return None


def _choose_node_id(
    graph: dict[str, Any],
    class_name: str,
    *,
    verb: str,
    goal: dict[str, Any] | None = None,
) -> int | None:
    ids = _node_ids_by_class(graph, class_name)
    if not ids:
        return None
    if len(ids) == 1:
        return ids[0]

    nodes = _nodes_by_id(graph)
    if nodes.get(ids[0], {}).get("category") == "Rooms":
        return ids[0]

    close_ids = _character_close_ids(graph)
    char_rooms = _character_room_ids(graph)
    held = _held_object_info(graph)
    goal_args = [str(arg) for arg in (goal or {}).get("args", [])]
    source_cls = goal_args[0] if goal_args else None
    target_cls = goal_args[1] if len(goal_args) > 1 else None

    def score(node_id: int) -> tuple[int, int]:
        value = 0
        if node_id in close_ids:
            value += 100
        if _node_room_ids(graph, node_id) & char_rooms:
            value += 60
        if held is not None and node_id == held[1]:
            value += 120
        if class_name == source_cls and held is None:
            value += 20
        if class_name == target_cls and held is not None:
            value += 40
        if verb in {"grab", "open", "switchon"} and node_id in close_ids:
            value += 40
        return value, -node_id

    return max(ids, key=score)


def _class_category(graph: dict[str, Any], class_name: str) -> str | None:
    for node in graph["nodes"]:
        if node["class_name"] == class_name:
            return node.get("category")
    return None


def _resolve_room_alias(graph: dict[str, Any], phrase: str) -> str | None:
    key = _norm_phrase(phrase)
    candidates = _ROOM_ALIASES.get(key, [])
    graph_classes = {n["class_name"] for n in graph["nodes"]}
    for cls in candidates:
        if cls in graph_classes:
            return cls
    return None


def _match_class(graph: dict[str, Any], phrase: str) -> str | None:
    alias = _resolve_room_alias(graph, phrase)
    if alias is not None:
        return alias

    norm = _norm_phrase(phrase)
    if not norm:
        return None

    for cls_norm, cls in _class_norms(graph):
        if norm == cls_norm:
            return cls

    # Generated actions often include light syntactic glue ("to fridge").
    norm_tokens = set(norm.split())
    for cls_norm, cls in _class_norms(graph):
        cls_tokens = set(cls_norm.split())
        if cls_tokens and cls_tokens <= norm_tokens:
            return cls

    close_matches = [
        (SequenceMatcher(None, norm, cls_norm).ratio(), cls)
        for cls_norm, cls in _class_norms(graph)
        if len(norm) >= 5 and len(cls_norm) >= 5
    ]
    if close_matches:
        score, cls = max(close_matches, key=lambda item: item[0])
        if score >= 0.88:
            return cls
    return None


def _match_pair(graph: dict[str, Any], phrase: str) -> tuple[str, str] | None:
    phrase = phrase.strip()
    for sep in ("into", "in", "onto", "on", "to"):
        match = re.search(rf"\s+{sep}\s+", phrase, flags=re.IGNORECASE)
        if match:
            left = phrase[: match.start()]
            right = phrase[match.end() :]
            left_cls = _match_class(graph, left)
            right_cls = _match_class(graph, right)
            if left_cls is not None and right_cls is not None:
                return left_cls, right_cls

    norm = _norm_phrase(phrase)
    if not norm:
        return None
    norms = _class_norms(graph)
    for left_norm, left_cls in norms:
        if not norm.startswith(left_norm + " "):
            continue
        rest = norm[len(left_norm) :].strip()
        for right_norm, right_cls in norms:
            if rest == right_norm:
                return left_cls, right_cls
    return None


def _held_object(graph: dict[str, Any]) -> str | None:
    held = _held_object_info(graph)
    return held[0] if held is not None else None


def _parse_action(text: str) -> ParsedAction | None:
    cleaned = _clean_action_text(text)
    bracket = re.match(r"^\s*\[([A-Za-z_]+)\]\s*(.*)$", cleaned)
    if bracket:
        verb = bracket.group(1).lower().replace("_", "")
        args = re.findall(r"<([^>]+)>", bracket.group(2))
        verb_map = {"switchon": "switchon", "putback": "put"}
        return ParsedAction(verb_map.get(verb, verb), args)

    lowered = cleaned.lower()

    def consume_prefix(prefix: str) -> str | None:
        if lowered == prefix:
            return ""
        if lowered.startswith(prefix + " "):
            return lowered[len(prefix) :].strip()
        return None

    patterns: list[tuple[tuple[str, ...], str]] = [
        (("switch on", "switchon", "turn on", "switch"), "switchon"),
        (("put in", "putin", "place in"), "putin"),
        (("put back", "put on", "put"), "put"),
        (("walk to", "go to", "walk"), "walk"),
        (("grab", "take", "pick up"), "grab"),
        (("open",), "open"),
    ]
    for prefixes, verb in patterns:
        for prefix in prefixes:
            rest = consume_prefix(prefix)
            if rest is not None:
                if verb == "put" and " in " in f" {rest} ":
                    return ParsedAction("putin", [rest])
                return ParsedAction(verb, [rest] if rest else [])
    return None


def _script_line_from_prediction(
    graph: dict[str, Any], prediction: str, goal: dict[str, Any] | None = None
) -> tuple[str | None, str | None]:
    parsed = _parse_action(prediction)
    if parsed is None:
        return None, "unrecognized action"

    verb = parsed.verb
    raw_arg = " ".join(parsed.args).strip()
    if verb in {"walk", "grab", "open", "switchon"}:
        cls = _match_class(graph, raw_arg)
        if cls is None:
            return None, f"object not found: {raw_arg!r}"
        if _class_category(graph, cls) == "Characters":
            return None, f"invalid target: {cls}"
        node_id = _choose_node_id(graph, cls, verb=verb, goal=goal)
        if node_id is None:
            return None, f"object id not found: {cls}"
        action = {
            "walk": "WALK",
            "grab": "GRAB",
            "open": "OPEN",
            "switchon": "SWITCHON",
        }[verb]
        return f"[{action}] <{cls}> ({node_id})", None

    if verb in {"put", "putin"}:
        pair = _match_pair(graph, raw_arg)
        if pair is None:
            target = _match_class(graph, raw_arg)
            source = _held_object(graph)
            if source is None or target is None:
                return None, f"put arguments not found: {raw_arg!r}"
            pair = (source, target)

        source, target = pair
        if (
            _class_category(graph, source) == "Characters"
            or _class_category(graph, target) == "Characters"
        ):
            return None, f"invalid put target: {source}, {target}"
        held = _held_object_info(graph)
        if held is not None and held[0] == source:
            source_id = held[1]
        else:
            source_id = _choose_node_id(graph, source, verb=verb, goal=goal)
        target_id = _choose_node_id(graph, target, verb=verb, goal=goal)
        if source_id is None or target_id is None:
            return None, f"put ids not found: {source}, {target}"
        action = "PUTIN" if verb == "putin" else "PUTBACK"
        return f"[{action}] <{source}> ({source_id}) <{target}> ({target_id})", None

    return None, f"unsupported action: {verb}"


def _infer_goal(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("_meta", {})
    args = [str(x) for x in meta.get("task_args", [])]
    instruction = str(row["instruction"]).lower()
    if instruction.startswith("turn on "):
        family = "turnon"
    elif instruction.startswith("open "):
        family = "open"
    elif instruction.startswith("put ") and " on " in instruction:
        family = "puton"
    elif instruction.startswith("place ") and " in " in instruction:
        family = "placein"
    else:
        raise ValueError(f"Cannot infer VirtualHome goal from instruction: {row['instruction']}")
    return {"family": family, "args": args, "instruction": row["instruction"]}


def _node_has_state(graph: dict[str, Any], class_name: str, state: str) -> bool:
    state = state.upper()
    return any(
        node["class_name"] == class_name and state in set(node.get("states", []))
        for node in graph["nodes"]
    )


def _has_relation(
    graph: dict[str, Any], source_class: str, relation: str, target_class: str
) -> bool:
    relation = relation.upper()
    nodes = {int(node["id"]): node for node in graph["nodes"]}
    source_ids = {
        int(node["id"]) for node in graph["nodes"] if node["class_name"] == source_class
    }
    target_ids = {
        int(node["id"]) for node in graph["nodes"] if node["class_name"] == target_class
    }
    for edge in graph["edges"]:
        if (
            int(edge["from_id"]) in source_ids
            and int(edge["to_id"]) in target_ids
            and edge["relation_type"] == relation
            and int(edge["from_id"]) in nodes
            and int(edge["to_id"]) in nodes
        ):
            return True
    return False


def _goal_satisfied(graph: dict[str, Any], goal: dict[str, Any]) -> bool:
    family = goal["family"]
    args = goal["args"]
    if family == "turnon":
        return bool(args) and _node_has_state(graph, args[0], "ON")
    if family == "open":
        return bool(args) and _node_has_state(graph, args[0], "OPEN")
    if family == "puton":
        return len(args) >= 2 and _has_relation(graph, args[0], "ON", args[1])
    if family == "placein":
        return len(args) >= 2 and _has_relation(graph, args[0], "INSIDE", args[1])
    raise ValueError(f"Unsupported goal family: {family}")


def _eval_episode(
    model,
    tokenizer,
    eg_modules: dict[str, Any],
    scene_inits: dict[str, Any],
    episode: list[dict[str, Any]],
    args: VirtualHomeRolloutArgs,
    detail_file,
    episode_idx: int,
    retrieved_world_models: list[int],
) -> RolloutResult:
    first = episode[0]
    meta = first.get("_meta", {})
    scene = meta.get("scene")
    if scene not in scene_inits:
        raise KeyError(f"Scene {scene!r} not found in {args.scene_inits_json}")

    goal = _infer_goal(first)
    EnvironmentGraph = eg_modules["environment"].EnvironmentGraph
    EnvironmentState = eg_modules["environment"].EnvironmentState
    ScriptExecutor = eg_modules["execution"].ScriptExecutor
    read_script_from_string = eg_modules["scripts"].read_script_from_string

    env_graph = EnvironmentGraph(copy.deepcopy(scene_inits[scene]))
    # Generated actions are class-level ("grab drawing"), matching the
    # observation format. Let EvolvingGraph bind a feasible instance instead of
    # forcing the first graph id for classes with duplicates.
    state = EnvironmentState(env_graph, {}, instance_selection=False)
    executor = ScriptExecutor(env_graph, {}, char_index=0)

    invalid_actions = 0
    executed_actions = 0
    steps_taken = 0
    final_graph = state.to_dict()

    selection = select_task_instances(scene_inits[scene], goal["family"], goal["args"])
    selected_node_ids = selected_instance_ids_from_selection(selection)

    if _goal_satisfied(final_graph, goal):
        return RolloutResult(True, 0, 0, 0, goal)

    for step_idx in range(args.max_steps):
        graph = state.to_dict()
        observation, observation_format = _render_rollout_observation(
            graph, first, args, selected_node_ids
        )
        prediction = _generate_action(
            model,
            tokenizer,
            first["instruction"],
            observation,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
        )
        script_line, parse_error = _script_line_from_prediction(graph, prediction, goal)
        executed = False
        execution_error = None
        if script_line is None:
            invalid_actions += 1
        else:
            try:
                script = read_script_from_string(script_line)
                ok, new_state = executor.execute_one_step(script, state)
                if ok:
                    state = new_state
                    executed = True
                    executed_actions += 1
                else:
                    invalid_actions += 1
                    execution_error = "precondition failed"
            except Exception as exc:  # VirtualHome raises custom exceptions here.
                invalid_actions += 1
                execution_error = f"{type(exc).__name__}: {exc}"

        steps_taken = step_idx + 1
        final_graph = state.to_dict()
        success = _goal_satisfied(final_graph, goal)
        detail_file.write(
            json.dumps(
                {
                    "episode": episode_idx,
                    "step": step_idx,
                    "scene": scene,
                    "instruction": first["instruction"],
                    "prediction": prediction,
                    "script_line": script_line,
                    "executed": executed,
                    "parse_error": parse_error,
                    "execution_error": execution_error,
                    "success_after_step": success,
                    "retrieved_world_models": retrieved_world_models,
                    "observation_format": observation_format,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        detail_file.flush()
        if success:
            return RolloutResult(True, steps_taken, invalid_actions, executed_actions, goal)

    return RolloutResult(False, args.max_steps, invalid_actions, executed_actions, goal)


def _load_eval_episodes(paths: list[Path]) -> list[list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_jsonl(path))
    if not rows:
        raise ValueError(f"No rows found in {paths}")
    first = rows[0]
    if not {"instruction", "observation", "action", "next_observation"} <= first.keys():
        raise ValueError("eval-vh-rollout only supports VirtualHome jsonl rows")
    return _group_virtualhome(rows)


def _summarize(results: list[RolloutResult]) -> dict[str, float | int]:
    if not results:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "path_steps": 0.0,
            "invalid_actions": 0.0,
            "executed_actions": 0.0,
        }
    return {
        "episodes": len(results),
        "success_rate": sum(r.success for r in results) / len(results),
        "path_steps": sum(r.steps for r in results) / len(results),
        "invalid_actions": sum(r.invalid_actions for r in results) / len(results),
        "executed_actions": sum(r.executed_actions for r in results) / len(results),
    }


def main(argv):
    """Evaluate VirtualHome SR/PS by executing generated actions in EvolvingGraph."""
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    curricula = load_wormi_curricula(args.curricula_path)
    model_name = args.model_name or str(
        curricula.output_dir / curricula.run_name / "last"
    )
    out_dir = args.output_path or curricula.output_dir / curricula.run_name / "vh-rollout"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.scene_inits_json.exists():
        raise FileNotFoundError(args.scene_inits_json)
    if not args.vh_src.exists():
        raise FileNotFoundError(args.vh_src)

    scene_inits = json.loads(args.scene_inits_json.read_text())
    eg_modules = _bootstrap_evolving_graph(args.vh_src)

    config = WorMIConfig.from_pretrained(model_name)
    model = WorMI.from_pretrained(
        model_name, config=config, torch_dtype=torch.bfloat16
    )
    model.eval()
    model.to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = "<|end_of_text|>"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        episodes = _load_eval_episodes(dataset_paths)
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

        detail_path = out_dir / f"vh-rollout-{name}.jsonl"
        episode_path = out_dir / f"vh-rollout-{name}-episodes.jsonl"
        first_row = episodes[0][0]
        observation_format = _resolve_observation_format(first_row, args)
        results: list[RolloutResult] = []
        with detail_path.open("w") as detail_file, episode_path.open("w") as episode_file:
            with tqdm(episodes, total=len(episodes), desc=f"VH rollout {name}") as pbar:
                for episode_idx, episode in enumerate(pbar):
                    result = _eval_episode(
                        model,
                        tokenizer,
                        eg_modules,
                        scene_inits,
                        episode,
                        args,
                        detail_file,
                        episode_idx,
                        selected_idx,
                    )
                    results.append(result)
                    metrics = _summarize(results)
                    pbar.set_postfix(
                        {
                            "SR": f"{metrics['success_rate']:.2%}",
                            "PS": f"{metrics['path_steps']:.2f}",
                        }
                    )
                    first = episode[0]
                    episode_file.write(
                        json.dumps(
                            {
                                "episode": episode_idx,
                                "scene": first.get("_meta", {}).get("scene"),
                                "instruction": first["instruction"],
                                "success": result.success,
                                "steps": result.steps,
                                "invalid_actions": result.invalid_actions,
                                "executed_actions": result.executed_actions,
                                "goal": result.goal,
                                "retrieved_world_models": selected_idx,
                                "observation_format": observation_format,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    episode_file.flush()

        metrics = _summarize(results)
        row = {
            "name": name,
            "dataset_type": "virtualhome_rollout",
            "retrieved_world_models": selected_idx,
            "max_steps": args.max_steps,
            "observation_format": observation_format,
            **metrics,
        }
        summary_rows.append(row)
        print(
            f"{name}: SR={metrics['success_rate']:.2%}, "
            f"PS={metrics['path_steps']:.2f}, n={metrics['episodes']}"
        )

    summary_path = out_dir / "vh-rollout-summary.json"
    with summary_path.open("w") as f:
        json.dump(summary_rows, f, indent=2)

    tsv_path = out_dir / "vh-rollout-summary.tsv"
    with tsv_path.open("w") as f:
        f.write(
            "name\tdataset_type\tepisodes\tSR\tPS\tinvalid_actions\t"
            "executed_actions\tmax_steps\tobservation_format\tworld_models\n"
        )
        for row in summary_rows:
            f.write(
                f"{row['name']}\t{row['dataset_type']}\t{row['episodes']}\t"
                f"{row['success_rate']:.6f}\t{row['path_steps']:.6f}\t"
                f"{row['invalid_actions']:.6f}\t{row['executed_actions']:.6f}\t"
                f"{row['max_steps']}\t{row['observation_format']}\t"
                f"{','.join(map(str, row['retrieved_world_models']))}\n"
            )

    print(f"Summary written to {summary_path}")


def parse_args(args) -> VirtualHomeRolloutArgs:
    parser = HfArgumentParser(VirtualHomeRolloutArgs)  # type: ignore
    return parser.parse_args_into_dataclasses(args)[0]  # type: ignore
