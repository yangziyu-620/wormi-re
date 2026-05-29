#!/usr/bin/env python3
"""Validate the generated VirtualHome jsonl data.

The checks are intentionally data-contract oriented:
- file layout and symlinks used by curricula
- JSON schema and `_meta` consistency
- trajectory completeness and train/test leakage
- action vocabulary and expected per-task action sequence
- loader compatibility with `VirtualHomeDataset`
- optional Llama-3 chat-template supervision compatibility
- replayability in VirtualHome EvolvingGraph
- whether final goal facts are visible in `next_observation`
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import statistics
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.build_virtualhome_dataset import (  # noqa: E402
    PAPER_SEEN_SCENE_KEYS,
    TARGETS,
    _bootstrap_evolving_graph,
    build_instructions,
    format_observation,
)
from tools.compact_virtualhome_observations import (  # noqa: E402
    _format_triples as _format_compact_triples,
    compact_next_observation,
    compact_observation,
    format_instance_grounded_observation,
    select_task_instances,
    selected_instance_ids_from_selection,
)


REQUIRED_ROW_KEYS = {"instruction", "observation", "action", "next_observation", "_meta"}
REQUIRED_META_KEYS = {
    "scene",
    "split",
    "task_args",
    "trajectory_id",
    "step_index",
    "num_steps",
}
ROOT_SPLITS = {
    "seen_seen": "test_seen_task_seen_scene.jsonl",
    "seen_unseen": "test_seen_task_unseen_scene.jsonl",
    "unseen_seen": "test_unseen_task_seen_scene.jsonl",
    "unseen_unseen": "test_unseen_task_unseen_scene.jsonl",
}
EVAL_LINKS = {
    "eval_col_1_seen_seen/test.jsonl": "../test_seen_task_seen_scene.jsonl",
    "eval_col_2_seen_unseen/test.jsonl": "../test_seen_task_unseen_scene.jsonl",
    "eval_col_3_unseen_unseen/test.jsonl": "../test_unseen_task_unseen_scene.jsonl",
}
ALLOWED_VERBS = {"walk", "grab", "open", "switchon", "put", "putin"}
RAW_ROOM_ACTIONS = {"walk dining_room", "walk home_office", "walk living_room"}
MAX_UNCHANGED_WALK_RATE = 0.05
EXPECTED_ACTIONS = {
    "turnon": ["walk", "switchon"],
    "open": ["walk", "open"],
    "puton": ["walk", "grab", "walk", "put"],
    "placein": ["walk", "grab", "walk", "open", "putin"],
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid json: {exc}") from exc
    return rows


def _family_from_instruction(instruction: str) -> str:
    text = instruction.lower()
    if text.startswith("turn on "):
        return "turnon"
    if text.startswith("open "):
        return "open"
    if text.startswith("put ") and " on " in text:
        return "puton"
    if text.startswith("place ") and " in " in text:
        return "placein"
    raise ValueError(f"unsupported instruction: {instruction!r}")


def _parse_action(action: str) -> tuple[str, list[str]]:
    parts = action.strip().split()
    if not parts:
        return "", []
    return parts[0], parts[1:]


def _find_first_id(graph: dict[str, Any], class_name: str) -> int | None:
    for node in graph["nodes"]:
        if node["class_name"] == class_name:
            return int(node["id"])
    return None


def _script_line_from_action(action: str, init_graph: dict[str, Any]) -> str:
    verb, args = _parse_action(action)
    arity = 2 if verb in {"put", "putin"} else 1
    if verb not in ALLOWED_VERBS or len(args) != arity:
        raise ValueError(f"bad action shape: {action!r}")

    ids = []
    for cls in args:
        node_id = _find_first_id(init_graph, cls)
        if node_id is None:
            raise ValueError(f"object {cls!r} from action {action!r} not in init graph")
        ids.append(node_id)

    def tok(cls: str, node_id: int) -> str:
        return f"<{cls}> ({node_id})"

    if verb == "walk":
        return f"[WALK] {tok(args[0], ids[0])}"
    if verb == "grab":
        return f"[GRAB] {tok(args[0], ids[0])}"
    if verb == "open":
        return f"[OPEN] {tok(args[0], ids[0])}"
    if verb == "switchon":
        return f"[SWITCHON] {tok(args[0], ids[0])}"
    if verb == "put":
        return f"[PUTBACK] {tok(args[0], ids[0])} {tok(args[1], ids[1])}"
    if verb == "putin":
        return f"[PUTIN] {tok(args[0], ids[0])} {tok(args[1], ids[1])}"
    raise ValueError(f"unsupported verb: {verb}")


def _node_has_state(graph: dict[str, Any], class_name: str, state: str) -> bool:
    target = state.upper()
    return any(
        node["class_name"] == class_name and target in set(node.get("states", []))
        for node in graph["nodes"]
    )


def _has_relation(graph: dict[str, Any], src: str, relation: str, dst: str) -> bool:
    relation = relation.upper()
    src_ids = {int(n["id"]) for n in graph["nodes"] if n["class_name"] == src}
    dst_ids = {int(n["id"]) for n in graph["nodes"] if n["class_name"] == dst}
    return any(
        int(edge["from_id"]) in src_ids
        and int(edge["to_id"]) in dst_ids
        and edge["relation_type"] == relation
        for edge in graph["edges"]
    )


def _goal_satisfied(graph: dict[str, Any], family: str, args: list[str]) -> bool:
    if family == "turnon":
        return len(args) == 1 and _node_has_state(graph, args[0], "ON")
    if family == "open":
        return len(args) == 1 and _node_has_state(graph, args[0], "OPEN")
    if family == "puton":
        return len(args) == 2 and _has_relation(graph, args[0], "ON", args[1])
    if family == "placein":
        return len(args) == 2 and _has_relation(graph, args[0], "INSIDE", args[1])
    return False


def _parse_triples(observation: str) -> set[tuple[str, str, str]]:
    triples = set()
    for subj, rel, obj in re.findall(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)", observation):
        triples.add((subj.strip(), rel.strip(), obj.strip()))
    return triples


def _goal_triple(family: str, args: list[str]) -> tuple[str, str, str] | None:
    if family == "turnon" and len(args) == 1:
        return (args[0], "is", "on")
    if family == "open" and len(args) == 1:
        return (args[0], "is", "open")
    if family == "puton" and len(args) == 2:
        return (args[0], "on", args[1])
    if family == "placein" and len(args) == 2:
        return (args[0], "inside", args[1])
    return None


def _check_link(path: Path, expected: str, errors: list[str]) -> None:
    if not path.exists() and not path.is_symlink():
        errors.append(f"missing symlink: {path}")
        return
    if not path.is_symlink():
        errors.append(f"expected symlink, found regular path: {path}")
        return
    actual = os.readlink(path)
    if actual != expected:
        errors.append(f"bad symlink target: {path} -> {actual}, expected {expected}")


def _expected_task_split(
    vh_src: Path,
    scene_inits: dict[str, dict[str, Any]],
    seed: int,
    seen_instruction_count: int,
) -> tuple[set[tuple[str, tuple[str, ...]]], set[tuple[str, tuple[str, ...]]]]:
    properties = json.loads(
        (vh_src / "virtualhome" / "resources" / "properties_data.json").read_text()
    )
    rng = random.Random(seed)
    scene_class_sets = [
        {node["class_name"] for node in graph["nodes"]}
        for graph in scene_inits.values()
    ]
    instructions = build_instructions(properties, scene_class_sets, rng)
    inst_by_fam: dict[str, list[tuple[str, tuple[str, ...]]]] = {
        family: [] for family in TARGETS
    }
    for family, task_args in instructions:
        inst_by_fam[family].append((family, task_args))

    total = sum(len(items) for items in inst_by_fam.values())
    fam_quotas = {
        family: max(1, round(len(items) * seen_instruction_count / total))
        for family, items in inst_by_fam.items()
    }
    drift = sum(fam_quotas.values()) - seen_instruction_count
    fams_sorted = sorted(fam_quotas, key=lambda f: -fam_quotas[f])
    while drift > 0:
        for family in fams_sorted:
            if fam_quotas[family] > 1 and drift > 0:
                fam_quotas[family] -= 1
                drift -= 1
    while drift < 0:
        for family in fams_sorted:
            if drift < 0:
                fam_quotas[family] += 1
                drift += 1

    seen: set[tuple[str, tuple[str, ...]]] = set()
    for family, items in inst_by_fam.items():
        rng.shuffle(items)
        seen.update(items[: fam_quotas[family]])
    return seen, set(instructions) - seen


def validate(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root
    errors: list[str] = []
    warnings: list[str] = []

    if not data_root.exists():
        raise FileNotFoundError(data_root)
    scene_inits = json.loads(args.scene_inits_json.read_text())
    manifest_path = data_root / "virtualhome_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    dataset_mode = (manifest or {}).get("dataset_mode", "easy_debug")

    for split, filename in ROOT_SPLITS.items():
        path = data_root / filename
        if not path.is_file():
            errors.append(f"missing root split file: {path}")
    for rel, target in EVAL_LINKS.items():
        _check_link(data_root / rel, target, errors)
    for i in range(6):
        scene_dir = data_root / f"scene_{i}"
        if not (scene_dir / "train.jsonl").is_file():
            errors.append(f"missing scene train: {scene_dir / 'train.jsonl'}")
        _check_link(scene_dir / "test.jsonl", "../test_seen_task_seen_scene.jsonl", errors)

    core_files: list[tuple[str, str, Path]] = []
    for split, filename in ROOT_SPLITS.items():
        core_files.append(("test", split, data_root / filename))
    for i in range(6):
        core_files.append(("train", "seen_seen", data_root / f"scene_{i}" / "train.jsonl"))

    file_stats: dict[str, dict[str, int]] = {}
    rows_by_tid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tid_roles: dict[str, set[str]] = defaultdict(set)
    tid_files: dict[str, set[str]] = defaultdict(set)
    row_keys_by_role: dict[str, set[tuple[str, str, str, str]]] = defaultdict(set)
    trajectory_task: dict[str, tuple[str, tuple[str, ...]]] = {}
    trajectory_action_seq: dict[str, tuple[str, ...]] = {}
    trajectory_split: dict[str, str] = {}
    task_by_seen_bucket: dict[str, set[tuple[str, tuple[str, ...]]]] = {
        "seen": set(),
        "unseen": set(),
    }
    scenes_present: set[str] = set()
    action_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    unchanged_by_action: Counter[str] = Counter()
    total_by_action: Counter[str] = Counter()
    raw_room_actions: Counter[str] = Counter()
    consecutive_duplicate_actions = 0
    consecutive_duplicate_examples: list[dict[str, Any]] = []
    state_relation_triples = 0

    for role, expected_split, path in core_files:
        if not path.exists():
            continue
        rows = _read_jsonl(path)
        rel = str(path.relative_to(data_root))
        file_stats[rel] = {"rows": len(rows), "trajectories": 0}
        for row_idx, row in enumerate(rows, 1):
            missing = REQUIRED_ROW_KEYS - set(row)
            if missing:
                errors.append(f"{rel}:{row_idx}: missing row keys {sorted(missing)}")
                continue
            for key in ["instruction", "observation", "action", "next_observation"]:
                if not isinstance(row[key], str):
                    errors.append(f"{rel}:{row_idx}: {key} is not string")
            meta = row.get("_meta")
            if not isinstance(meta, dict):
                errors.append(f"{rel}:{row_idx}: _meta is not object")
                continue
            missing_meta = REQUIRED_META_KEYS - set(meta)
            if missing_meta:
                errors.append(f"{rel}:{row_idx}: missing _meta keys {sorted(missing_meta)}")
                continue
            if meta["split"] != expected_split:
                errors.append(
                    f"{rel}:{row_idx}: split {meta['split']!r}, expected {expected_split!r}"
                )
            if meta["scene"] not in scene_inits:
                errors.append(f"{rel}:{row_idx}: scene not in scene cache: {meta['scene']}")
            if not isinstance(meta["task_args"], list) or not all(
                isinstance(x, str) for x in meta["task_args"]
            ):
                errors.append(f"{rel}:{row_idx}: bad task_args")
            if not isinstance(meta["step_index"], int) or not isinstance(meta["num_steps"], int):
                errors.append(f"{rel}:{row_idx}: step_index/num_steps must be int")
            elif meta["step_index"] < 0 or meta["step_index"] >= meta["num_steps"]:
                errors.append(f"{rel}:{row_idx}: step_index outside num_steps")

            try:
                family = _family_from_instruction(row["instruction"])
            except ValueError as exc:
                errors.append(f"{rel}:{row_idx}: {exc}")
                continue
            family_counts[family] += 1
            task_id = (family, tuple(meta["task_args"]))
            if expected_split.startswith("seen_"):
                task_by_seen_bucket["seen"].add(task_id)
            if expected_split.startswith("unseen_"):
                task_by_seen_bucket["unseen"].add(task_id)

            verb, action_args = _parse_action(row["action"])
            action_counts[verb] += 1
            total_by_action[verb] += 1
            if verb not in ALLOWED_VERBS:
                errors.append(f"{rel}:{row_idx}: unsupported action verb {verb!r}")
            expected_arity = 2 if verb in {"put", "putin"} else 1
            if verb in ALLOWED_VERBS and len(action_args) != expected_arity:
                errors.append(f"{rel}:{row_idx}: bad action arity for {row['action']!r}")
            if row["action"] in RAW_ROOM_ACTIONS:
                raw_room_actions[row["action"]] += 1
            if row["observation"] == row["next_observation"]:
                unchanged_by_action[verb] += 1
            state_relation_triples += sum(
                1 for _, reln, _ in _parse_triples(row["observation"]) if reln == "is"
            )

            tid = str(meta["trajectory_id"])
            rows_by_tid[tid].append({"row": row, "role": role, "file": rel})
            tid_roles[tid].add(role)
            tid_files[tid].add(rel)
            row_keys_by_role[role].add((
                row["instruction"],
                row["observation"],
                row["action"],
                row["next_observation"],
            ))
            trajectory_task[tid] = task_id
            trajectory_split[tid] = expected_split
            scenes_present.add(str(meta["scene"]))

        file_stats[rel]["trajectories"] = len(
            {str(r["_meta"]["trajectory_id"]) for r in rows if "_meta" in r}
        )

    overlap = {tid for tid, roles in tid_roles.items() if len(roles) > 1}
    if overlap:
        errors.append(f"train/test trajectory overlap: {len(overlap)} examples")

    trajectory_lengths: Counter[int] = Counter()
    goal_visible_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    replay_failures = 0
    obs_mismatches = 0
    next_obs_mismatches = 0
    goal_failures = 0

    eg = _bootstrap_evolving_graph(args.vh_src)
    EnvironmentGraph = eg["environment"].EnvironmentGraph
    read_script = eg["scripts"].read_script_from_string
    ScriptExecutor = eg["execution"].ScriptExecutor

    for tid, wrapped_rows in rows_by_tid.items():
        rows = [x["row"] for x in wrapped_rows]
        rows = sorted(rows, key=lambda r: int(r["_meta"]["step_index"]))
        first = rows[0]
        meta = first["_meta"]
        scene = meta["scene"]
        family = _family_from_instruction(first["instruction"])
        task_args = [str(x) for x in meta["task_args"]]
        num_steps = int(meta["num_steps"])
        trajectory_lengths[num_steps] += 1

        indices = [int(r["_meta"]["step_index"]) for r in rows]
        if indices != list(range(num_steps)):
            errors.append(f"{tid}: incomplete or non-contiguous steps {indices}, num_steps={num_steps}")
            continue
        if len(rows) != num_steps:
            errors.append(f"{tid}: row count {len(rows)} != num_steps {num_steps}")
        if any(r["_meta"]["scene"] != scene for r in rows):
            errors.append(f"{tid}: mixed scene values")
        if any(r["_meta"]["task_args"] != meta["task_args"] for r in rows):
            errors.append(f"{tid}: mixed task_args values")

        actions = [r["action"] for r in rows]
        verbs = [_parse_action(action)[0] for action in actions]
        trajectory_action_seq[tid] = tuple(actions)
        for prev_action, action in zip(actions, actions[1:]):
            if action == prev_action:
                consecutive_duplicate_actions += 1
                if len(consecutive_duplicate_examples) < 10:
                    consecutive_duplicate_examples.append(
                        {
                            "trajectory_id": tid,
                            "action": action,
                            "actions": actions,
                        }
                    )
        expected_verbs = EXPECTED_ACTIONS[family]
        if dataset_mode == "easy_debug":
            if verbs != expected_verbs:
                errors.append(f"{tid}: action sequence {verbs} != expected {expected_verbs}")
        else:
            final_expected = expected_verbs[-1]
            if not verbs or verbs[-1] != final_expected:
                errors.append(
                    f"{tid}: paper_like final verb {verbs[-1] if verbs else None!r} "
                    f"!= expected {final_expected!r}"
                )
            if family in {"puton", "placein"} and "grab" not in verbs:
                errors.append(f"{tid}: paper_like {family} trajectory has no grab action")

        final_triples = _parse_triples(rows[-1]["next_observation"])
        triple = _goal_triple(family, task_args)
        if triple is not None:
            goal_visible_by_family[family]["visible" if triple in final_triples else "hidden"] += 1

        try:
            init_graph = scene_inits[scene]
            script_lines = [
                r.get("_meta", {}).get("script_line")
                or _script_line_from_action(r["action"], init_graph)
                for r in rows
            ]
            env_graph = EnvironmentGraph(copy.deepcopy(init_graph))
            executor = ScriptExecutor(env_graph, name_equivalence={})
            ok, _final, graph_state_list = executor.execute(
                read_script(", ".join(script_lines)), w_graph_list=True
            )
            if not ok:
                replay_failures += 1
                continue
            if len(graph_state_list) != len(rows) + 1:
                errors.append(
                    f"{tid}: replay graph_state_list len {len(graph_state_list)} "
                    f"!= rows+1 {len(rows) + 1}"
                )
                continue
            for i, row in enumerate(rows):
                prep = row.get("_meta", {}).get("observation_preprocessing") or {}
                if str(prep.get("mode", "")).startswith("tmow_compact"):
                    num_edges = int(prep.get("num_edges", 17))
                    next_mode = str(prep.get("next_mode", "delta"))
                    fill_to_num_edges = bool(prep.get("fill_to_num_edges", False))
                    task_args_for_row = [
                        str(arg).lower().replace(" ", "_")
                        for arg in row.get("_meta", {}).get("task_args", [])
                    ]
                    current_action = (
                        row["action"]
                        if prep.get("current_observation_action_conditioned", True)
                        else ""
                    )
                    current_source = format_observation(graph_state_list[i])
                    next_source = format_observation(graph_state_list[i + 1])
                    if prep.get("instance_grounded", False):
                        selection = row.get("_meta", {}).get("instance_selection")
                        if not isinstance(selection, dict) or not selection:
                            selection = select_task_instances(
                                graph_state_list[0], family, task_args_for_row
                            )
                        selected_node_ids = selected_instance_ids_from_selection(selection)
                        current_source = format_instance_grounded_observation(
                            graph_state_list[i],
                            task_args=task_args_for_row,
                            selected_node_ids=selected_node_ids,
                        )
                        next_source = format_instance_grounded_observation(
                            graph_state_list[i + 1],
                            task_args=task_args_for_row,
                            selected_node_ids=selected_node_ids,
                        )
                    current_compact = compact_observation(
                        current_source,
                        instruction=row["instruction"],
                        action=current_action,
                        task_args=task_args_for_row,
                        num_edges=num_edges,
                        fill_to_num_edges=fill_to_num_edges,
                    )
                    next_compact = compact_observation(
                        next_source,
                        instruction=row["instruction"],
                        action=row["action"],
                        task_args=task_args_for_row,
                        num_edges=num_edges,
                        fill_to_num_edges=fill_to_num_edges,
                    )
                    expected_observation = _format_compact_triples(current_compact)
                    expected_next_observation = _format_compact_triples(
                        compact_next_observation(
                            row,
                            current_compact=current_compact,
                            next_compact=next_compact,
                            mode=next_mode,
                        )
                    )
                else:
                    expected_observation = format_observation(graph_state_list[i])
                    expected_next_observation = format_observation(
                        graph_state_list[i + 1]
                    )
                if expected_observation != row["observation"]:
                    obs_mismatches += 1
                if expected_next_observation != row["next_observation"]:
                    next_obs_mismatches += 1
            if not _goal_satisfied(graph_state_list[-1], family, task_args):
                goal_failures += 1
        except Exception as exc:
            replay_failures += 1
            warnings.append(f"{tid}: replay exception {type(exc).__name__}: {exc}")

    train_row_keys = row_keys_by_role.get("train", set())
    test_row_keys = row_keys_by_role.get("test", set())
    exact_row_overlap = train_row_keys & test_row_keys

    train_tids = {tid for tid, roles in tid_roles.items() if "train" in roles}
    test_tids = {tid for tid, roles in tid_roles.items() if "test" in roles}
    train_sequences = {trajectory_action_seq.get(tid, ()) for tid in train_tids}
    test_sequences = {trajectory_action_seq.get(tid, ()) for tid in test_tids}
    exact_sequence_overlap = train_sequences & test_sequences

    train_task_sequences: set[tuple[tuple[str, tuple[str, ...]], tuple[str, ...]]] = {
        (trajectory_task[tid], trajectory_action_seq.get(tid, ()))
        for tid in train_tids
        if tid in trajectory_task
    }
    test_task_sequences: set[tuple[tuple[str, tuple[str, ...]], tuple[str, ...]]] = {
        (trajectory_task[tid], trajectory_action_seq.get(tid, ()))
        for tid in test_tids
        if tid in trajectory_task
    }
    same_task_sequence_overlap = train_task_sequences & test_task_sequences
    if exact_row_overlap:
        errors.append(f"train/test exact row overlap: {len(exact_row_overlap)}")

    generated_trajectories = len(rows_by_tid)
    if generated_trajectories != 1023:
        warnings.append(
            f"generated trajectory count is {generated_trajectories}, paper count is 1023"
        )

    seen_tasks = task_by_seen_bucket["seen"]
    unseen_tasks = task_by_seen_bucket["unseen"]
    if len(seen_tasks) != 16:
        warnings.append(f"seen task count is {len(seen_tasks)}, expected 16")
    if len(unseen_tasks) != 62:
        warnings.append(f"unseen task count is {len(unseen_tasks)}, expected 62")
    if seen_tasks & unseen_tasks:
        errors.append(f"seen/unseen task overlap: {len(seen_tasks & unseen_tasks)}")

    if manifest is not None:
        expected_seen_tasks = {
            (str(task["family"]), tuple(str(x) for x in task["args"]))
            for task in manifest.get("selected_tasks", [])
            if task.get("task_split") == "seen"
        }
        expected_unseen_tasks = {
            (str(task["family"]), tuple(str(x) for x in task["args"]))
            for task in manifest.get("selected_tasks", [])
            if task.get("task_split") == "unseen"
        }
    else:
        expected_seen_tasks, expected_unseen_tasks = _expected_task_split(
            args.vh_src, scene_inits, args.seed, args.seen_instructions
        )
    missing_seen_tasks = sorted(expected_seen_tasks - seen_tasks)
    missing_unseen_tasks = sorted(expected_unseen_tasks - unseen_tasks)
    missing_all_tasks = sorted((expected_seen_tasks | expected_unseen_tasks) - (seen_tasks | unseen_tasks))
    extra_seen_tasks = sorted(seen_tasks - expected_seen_tasks)
    extra_unseen_tasks = sorted(unseen_tasks - expected_unseen_tasks)
    if missing_seen_tasks:
        warnings.append(f"missing seen tasks: {missing_seen_tasks}")
    if missing_unseen_tasks:
        warnings.append(f"missing unseen tasks: {missing_unseen_tasks}")
    if extra_seen_tasks:
        errors.append(f"extra seen tasks not in manifest/expected split: {extra_seen_tasks}")
    if extra_unseen_tasks:
        errors.append(f"extra unseen tasks not in manifest/expected split: {extra_unseen_tasks}")

    missing_scene_rows = set(scene_inits) - scenes_present
    if missing_scene_rows:
        warnings.append(f"scene cache entries with no rows: {sorted(missing_scene_rows)}")
    if len(scenes_present) != 20:
        warnings.append(f"effective scenes with rows: {len(scenes_present)}, expected 20")

    hidden_goal_families = {
        family: dict(counts)
        for family, counts in sorted(goal_visible_by_family.items())
        if counts.get("hidden", 0)
    }
    if hidden_goal_families:
        warnings.append(f"goal facts hidden from final next_observation: {hidden_goal_families}")
    if state_relation_triples == 0:
        warnings.append("no `(object, is, state)` triples found in observations")
    unchanged_report = {
        verb: {"unchanged": unchanged_by_action[verb], "total": total_by_action[verb]}
        for verb in sorted(total_by_action)
        if unchanged_by_action[verb]
    }
    if raw_room_actions:
        errors.append(f"raw room action labels are not canonicalized: {dict(raw_room_actions)}")
    if consecutive_duplicate_actions:
        errors.append(
            f"consecutive duplicate actions: {consecutive_duplicate_actions}; "
            f"examples={consecutive_duplicate_examples[:3]}"
        )
    if unchanged_report:
        warnings.append(f"actions with unchanged observation: {unchanged_report}")
    walk_total = total_by_action.get("walk", 0)
    walk_unchanged = unchanged_by_action.get("walk", 0)
    if walk_total and walk_unchanged / walk_total > MAX_UNCHANGED_WALK_RATE:
        errors.append(
            f"unchanged walk rate {walk_unchanged}/{walk_total} "
            f"exceeds {MAX_UNCHANGED_WALK_RATE:.0%}"
        )

    loader_stats = {}
    if args.check_loader:
        try:
            import wormi.datasets  # noqa: F401
            from wormi.datasets.auto_jsonl import AutoJsonlDataset

            for _, _, path in core_files:
                if not path.exists():
                    continue
                raw_len = file_stats[str(path.relative_to(data_root))]["rows"]
                if raw_len == 0:
                    loader_stats[str(path.relative_to(data_root))] = {
                        "raw_rows": 0,
                        "action_samples": 0,
                        "world_samples": 0,
                        "skipped": "empty_jsonl",
                    }
                    continue
                action_ds = AutoJsonlDataset.load(
                    path, end_with_action=True, cumulative=True
                )
                world_ds = AutoJsonlDataset.load(
                    path, end_with_action=False, cumulative=True
                )
                loader_stats[str(path.relative_to(data_root))] = {
                    "raw_rows": raw_len,
                    "action_samples": len(action_ds),
                    "world_samples": len(world_ds),
                }
                if len(action_ds) != raw_len:
                    errors.append(f"{path}: action loader len {len(action_ds)} != raw rows {raw_len}")
                if len(world_ds) != raw_len * 3:
                    errors.append(f"{path}: world loader len {len(world_ds)} != raw rows*3 {raw_len * 3}")
        except Exception as exc:
            errors.append(f"loader smoke test failed: {type(exc).__name__}: {exc}")

    chat_template_stats = {}
    if args.check_chat_template:
        try:
            import wormi.datasets  # noqa: F401
            from transformers import AutoTokenizer
            from trl import DataCollatorForCompletionOnlyLM
            from wormi.datasets.auto_jsonl import AutoJsonlDataset

            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer,
                local_files_only=args.tokenizer_local_files_only,
            )
            tokenizer.pad_token = "<|end_of_text|>"
            response_template = "<|start_header_id|>assistant<|end_header_id|>"
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template,
                tokenizer=tokenizer,
            )
            bad_examples = []
            total_action_samples = 0
            total_world_samples = 0
            max_action_tokens = 0
            max_world_tokens = 0
            supervised_token_counts = []

            def check_loss_masks(features, meta):
                if not features:
                    return
                batch = data_collator(features)
                for labels, (rel_path, mode, idx) in zip(batch["labels"], meta):
                    supervised = int((labels != -100).sum().item())
                    supervised_token_counts.append(supervised)
                    if supervised == 0:
                        bad_examples.append([rel_path, mode, idx, "empty_loss_mask"])

            for _, _, path in core_files:
                if not path.exists():
                    continue
                if file_stats[str(path.relative_to(data_root))]["rows"] == 0:
                    continue
                for mode, end_with_action in (("action", True), ("world", False)):
                    dataset = AutoJsonlDataset.load(
                        path,
                        end_with_action=end_with_action,
                        cumulative=True,
                    ).as_chat(tokenizer)
                    if mode == "action":
                        total_action_samples += len(dataset)
                    else:
                        total_world_samples += len(dataset)

                    local_max_tokens = 0
                    rel_path = str(path.relative_to(data_root))
                    feature_batch = []
                    meta_batch = []
                    for idx, text in enumerate(dataset["text"]):
                        if response_template not in text:
                            bad_examples.append(
                                [rel_path, mode, idx, "missing_response_template"]
                            )
                            continue
                        tail = text.rsplit(response_template, 1)[-1]
                        assistant_span = tail.split("<|eot_id|>", 1)[0].strip()
                        if not assistant_span:
                            bad_examples.append([rel_path, mode, idx, "empty_assistant_span"])
                        encoded = tokenizer(text, add_special_tokens=False)
                        num_tokens = len(encoded["input_ids"])
                        local_max_tokens = max(local_max_tokens, num_tokens)
                        if num_tokens > 4096:
                            bad_examples.append([rel_path, mode, idx, "over_4096", num_tokens])
                        feature_batch.append(encoded)
                        meta_batch.append((rel_path, mode, idx))
                        if len(feature_batch) >= 32:
                            check_loss_masks(feature_batch, meta_batch)
                            feature_batch = []
                            meta_batch = []
                    check_loss_masks(feature_batch, meta_batch)
                    if mode == "action":
                        max_action_tokens = max(max_action_tokens, local_max_tokens)
                    else:
                        max_world_tokens = max(max_world_tokens, local_max_tokens)

            chat_template_stats = {
                "tokenizer": args.tokenizer,
                "tokenizer_local_files_only": args.tokenizer_local_files_only,
                "response_template": response_template,
                "action_samples": total_action_samples,
                "world_samples": total_world_samples,
                "max_action_tokens": max_action_tokens,
                "max_world_tokens": max_world_tokens,
                "loss_mask_samples": len(supervised_token_counts),
                "min_supervised_tokens": min(supervised_token_counts) if supervised_token_counts else 0,
                "max_supervised_tokens": max(supervised_token_counts) if supervised_token_counts else 0,
                "bad_count": len(bad_examples),
                "bad_examples": bad_examples[:10],
            }
            if bad_examples:
                errors.append(
                    "chat template supervision check failed: "
                    f"{len(bad_examples)} bad samples; examples={bad_examples[:3]}"
                )
        except Exception as exc:
            errors.append(f"chat template check failed: {type(exc).__name__}: {exc}")

    if replay_failures:
        errors.append(f"expert trajectory replay failures: {replay_failures}")
    if obs_mismatches or next_obs_mismatches:
        errors.append(
            f"observation mismatches after replay: obs={obs_mismatches}, "
            f"next_obs={next_obs_mismatches}"
        )
    if goal_failures:
        errors.append(f"expert final graph goal failures: {goal_failures}")

    lengths = [length for length, count in trajectory_lengths.items() for _ in range(count)]
    length_stats = {
        "mean": statistics.mean(lengths) if lengths else 0.0,
        "median": statistics.median(lengths) if lengths else 0.0,
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
    }
    if lengths:
        sorted_lengths = sorted(lengths)
        length_stats["p10"] = sorted_lengths[int(0.10 * (len(sorted_lengths) - 1))]
        length_stats["p90"] = sorted_lengths[int(0.90 * (len(sorted_lengths) - 1))]

    split_episode_counts: Counter[str] = Counter(trajectory_split.values())
    split_length_sums: Counter[str] = Counter()
    for tid, seq in trajectory_action_seq.items():
        split_length_sums[trajectory_split.get(tid, "unknown")] += len(seq)
    split_avg_lengths = {
        split: split_length_sums[split] / count
        for split, count in split_episode_counts.items()
        if count
    }

    summary = {
        "data_root": str(data_root),
        "dataset_mode": dataset_mode,
        "manifest": str(manifest_path) if manifest is not None else None,
        "errors": errors,
        "warnings": warnings,
        "file_stats": file_stats,
        "total_rows": sum(v["rows"] for v in file_stats.values()),
        "trajectories": generated_trajectories,
        "train_test_overlap": len(overlap),
        "leakage": {
            "trajectory_id_overlap": len(overlap),
            "exact_row_overlap": len(exact_row_overlap),
            "exact_action_sequence_overlap": len(exact_sequence_overlap),
            "same_task_exact_action_sequence_overlap": len(same_task_sequence_overlap),
        },
        "family_row_counts": dict(sorted(family_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "trajectory_lengths": dict(sorted(trajectory_lengths.items())),
        "trajectory_length_stats": length_stats,
        "split_episode_counts": dict(sorted(split_episode_counts.items())),
        "split_avg_lengths": dict(sorted(split_avg_lengths.items())),
        "seen_tasks": len(seen_tasks),
        "unseen_tasks": len(unseen_tasks),
        "expected_seen_tasks": len(expected_seen_tasks),
        "expected_unseen_tasks": len(expected_unseen_tasks),
        "missing_seen_tasks": missing_seen_tasks,
        "missing_unseen_tasks": missing_unseen_tasks,
        "missing_all_tasks": missing_all_tasks,
        "extra_seen_tasks": extra_seen_tasks,
        "extra_unseen_tasks": extra_unseen_tasks,
        "scenes_present": sorted(scenes_present),
        "paper_seen_scene_keys": PAPER_SEEN_SCENE_KEYS,
        "goal_visibility": {
            family: dict(counts) for family, counts in sorted(goal_visible_by_family.items())
        },
        "unchanged_observations_by_action": unchanged_report,
        "raw_room_actions": dict(sorted(raw_room_actions.items())),
        "consecutive_duplicate_actions": consecutive_duplicate_actions,
        "consecutive_duplicate_examples": consecutive_duplicate_examples,
        "quality_thresholds": {
            "max_unchanged_walk_rate": MAX_UNCHANGED_WALK_RATE,
        },
        "replay": {
            "failures": replay_failures,
            "obs_mismatches": obs_mismatches,
            "next_obs_mismatches": next_obs_mismatches,
            "goal_failures": goal_failures,
        },
        "loader": loader_stats,
        "chat_template": chat_template_stats,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/root/autodl-tmp/wormi-data/virtualhome"),
    )
    parser.add_argument(
        "--scene-inits-json",
        type=Path,
        default=Path("/root/autodl-tmp/wormi-data/scene-inits/init_graphs_20_semantic.json"),
    )
    parser.add_argument(
        "--vh-src",
        type=Path,
        default=Path("/root/autodl-tmp/wormi-data/virtualhome-src"),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--check-loader", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-chat-template", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tokenizer", default="unsloth/Llama-3.2-3B-Instruct")
    parser.add_argument(
        "--tokenizer-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seen-instructions", type=int, default=16)
    parser.add_argument("--fail-on-warnings", action="store_true")
    args = parser.parse_args()

    summary = validate(args)
    print(f"data_root: {summary['data_root']}")
    print(f"dataset_mode: {summary['dataset_mode']}")
    print(f"rows: {summary['total_rows']}")
    print(f"trajectories: {summary['trajectories']}")
    print(f"train_test_overlap: {summary['train_test_overlap']}")
    print(f"seen_tasks: {summary['seen_tasks']}")
    print(f"unseen_tasks: {summary['unseen_tasks']}")
    print(f"missing_seen_tasks: {summary['missing_seen_tasks']}")
    print(f"missing_unseen_tasks: {summary['missing_unseen_tasks']}")
    print(f"scenes_present: {len(summary['scenes_present'])}")
    print(f"family_row_counts: {summary['family_row_counts']}")
    print(f"action_counts: {summary['action_counts']}")
    print(f"trajectory_lengths: {summary['trajectory_lengths']}")
    print(f"trajectory_length_stats: {summary['trajectory_length_stats']}")
    print(f"split_episode_counts: {summary['split_episode_counts']}")
    print(f"split_avg_lengths: {summary['split_avg_lengths']}")
    print(f"leakage: {summary['leakage']}")
    print(f"replay: {summary['replay']}")
    print(f"goal_visibility: {summary['goal_visibility']}")
    if summary["chat_template"]:
        print(f"chat_template: {summary['chat_template']}")

    if summary["warnings"]:
        print("\nWARNINGS:")
        for item in summary["warnings"]:
            print(f"  - {item}")
    if summary["errors"]:
        print("\nERRORS:")
        for item in summary["errors"]:
            print(f"  - {item}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.output_json}")

    if summary["errors"] or (args.fail_on_warnings and summary["warnings"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
