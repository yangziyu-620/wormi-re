#!/usr/bin/env python3
"""Build a TMoW-style compact VirtualHome dataset for WorMI.

This builder is independent from the post-hoc JSONL compactor: it executes the
VirtualHome graph program, then renders compact observations from each graph
state before rows are written. The output layout and task split follow the
WorMI VirtualHome paper setup used by `tools/build_virtualhome_dataset.py`.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import build_virtualhome_dataset as base  # noqa: E402
from tools.compact_virtualhome_observations import (  # noqa: E402
    DEFAULT_NUM_EDGES,
    _format_triples,
    _parse_triples,
    compact_next_observation,
    compact_observation,
    format_instance_grounded_observation,
    select_task_instances,
    selected_instance_ids_from_selection,
)


def _triple_count(text: str) -> int:
    return len(_parse_triples(text))


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
    return ""


def _compact_row_from_graph_states(
    row: dict[str, Any],
    current_graph: dict,
    next_graph: dict,
    *,
    num_edges: int,
    next_mode: str,
) -> dict[str, Any]:
    """Render one row directly from graph states using TMoW-style compaction."""
    raw_full_observation = base.format_observation(current_graph)
    raw_full_next = base.format_observation(next_graph)

    instruction = str(row["instruction"])
    action = str(row["action"])
    task_args = [str(arg).lower().replace(" ", "_") for arg in row["_meta"]["task_args"]]
    selection = row["_meta"].get("instance_selection")
    if not isinstance(selection, dict) or not selection:
        selection = select_task_instances(
            current_graph,
            _family_from_instruction(instruction),
            task_args,
        )
    selected_node_ids = selected_instance_ids_from_selection(selection)
    full_observation = format_instance_grounded_observation(
        current_graph,
        task_args=task_args,
        selected_node_ids=selected_node_ids,
    )
    full_next = format_instance_grounded_observation(
        next_graph,
        task_args=task_args,
        selected_node_ids=selected_node_ids,
    )

    # The policy prompt must not be conditioned on the target action. During
    # rollout we only know the instruction and current graph state, so the
    # training observation has to be generated from the same information.
    current = compact_observation(
        full_observation,
        instruction=instruction,
        action="",
        task_args=task_args,
        num_edges=num_edges,
        fill_to_num_edges=True,
    )
    nxt = compact_observation(
        full_next,
        instruction=instruction,
        action=action,
        task_args=task_args,
        num_edges=num_edges,
        fill_to_num_edges=True,
    )
    next_out = compact_next_observation(
        row,
        current_compact=current,
        next_compact=nxt,
        mode=next_mode,
    )

    out = copy.deepcopy(row)
    out["observation"] = _format_triples(current)
    out["next_observation"] = _format_triples(next_out)
    meta = out["_meta"]
    meta["observation_preprocessing"] = {
        "mode": "tmow_compact_from_graph_state",
        "source": "virtualhome_evolving_graph",
        "num_edges": num_edges,
        "next_mode": next_mode,
        "fill_to_num_edges": True,
        "current_observation_action_conditioned": False,
        "instance_grounded": True,
        "instance_selection_mode": selection.get("instance_selection_mode"),
        "selection_inputs": selection.get("selection_inputs"),
        "grounding_node_ids": selected_node_ids,
        "source_observation_triples": _triple_count(raw_full_observation),
        "grounded_observation_triples": _triple_count(full_observation),
        "compact_observation_triples": len(current),
        "source_next_observation_triples": _triple_count(raw_full_next),
        "grounded_next_observation_triples": _triple_count(full_next),
        "compact_next_observation_triples": len(next_out),
    }
    return out


def _execute_tmow_compact_candidate(
    family: str,
    args: tuple[str, ...],
    scene_name: str,
    init_graph: dict,
    EnvironmentGraph,
    read_script,
    ScriptExecutor,
    *,
    num_edges: int,
    next_mode: str,
    max_steps: int = 18,
) -> tuple[list[dict] | None, str | None]:
    script_lines, action_texts, debug = base._paperlike_program(family, args, init_graph)
    if script_lines is None or action_texts is None:
        return None, str(debug.get("reason", "planner_failed"))
    if len(action_texts) > max_steps:
        return None, "too_long"

    try:
        env_graph = EnvironmentGraph(copy.deepcopy(init_graph))
        script = read_script(", ".join(script_lines))
        ok, _final, graph_state_list = ScriptExecutor(
            env_graph, name_equivalence={}
        ).execute(script, w_graph_list=True)
    except Exception:
        return None, "execution_exception"

    if not ok:
        return None, "execution_failed"
    if len(graph_state_list) != len(action_texts) + 1:
        return None, "state_action_misaligned"
    if not base._is_semantically_valid_trajectory(family, args, graph_state_list):
        return None, "semantic_invalid"

    instruction = base.instruction_text(family, args)
    trajectory_id = f"{scene_name}:{family}:{'|'.join(args)}"
    rows = []
    for i, action_text in enumerate(action_texts):
        full_meta_row = {
            "instruction": instruction,
            "observation": "",
            "action": action_text,
            "next_observation": "",
            "_meta": {
                "scene": scene_name,
                "split": None,
                "task_args": list(args),
                "trajectory_id": trajectory_id,
                "step_index": i,
                "num_steps": len(action_texts),
                "generator_mode": "paper_like_graph_planner_tmow_compact",
                "script_line": script_lines[i],
                "instance_selection": debug.get("instance_selection", {}),
                "planner_debug": debug,
            },
        }
        rows.append(
            _compact_row_from_graph_states(
                full_meta_row,
                graph_state_list[i],
                graph_state_list[i + 1],
                num_edges=num_edges,
                next_mode=next_mode,
            )
        )
    return rows, None


def _load_scene_inits(raw_dir: Path | None, scene_inits_json: Path | None) -> dict[str, dict]:
    if scene_inits_json is not None:
        return json.loads(scene_inits_json.read_text())
    if raw_dir is None:
        raise ValueError("either --raw-dir or --scene-inits-json must be provided")
    scene_dirs = sorted(
        d for d in (raw_dir / "init_and_final_graphs").iterdir() if d.is_dir()
    )
    scene_inits = {}
    for scene_dir in scene_dirs:
        graph = base.find_scene_init_graph(scene_dir)
        if graph is not None:
            scene_inits[scene_dir.name] = graph
    return scene_inits


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _collect_preprocessing_summary(rows: list[dict]) -> dict[str, Any]:
    obs_before = []
    obs_after = []
    next_before = []
    next_after = []
    no_updates = 0
    missing_task_args = 0
    instance_grounded = 0
    grounded_obs = []
    grounded_next = []

    for row in rows:
        prep = row.get("_meta", {}).get("observation_preprocessing", {})
        obs_before.append(int(prep.get("source_observation_triples", 0)))
        obs_after.append(int(prep.get("compact_observation_triples", 0)))
        next_before.append(int(prep.get("source_next_observation_triples", 0)))
        next_after.append(int(prep.get("compact_next_observation_triples", 0)))
        if prep.get("instance_grounded", False):
            instance_grounded += 1
        if "grounded_observation_triples" in prep:
            grounded_obs.append(int(prep.get("grounded_observation_triples", 0)))
        if "grounded_next_observation_triples" in prep:
            grounded_next.append(int(prep.get("grounded_next_observation_triples", 0)))
        no_updates += int(str(row.get("next_observation", "")) == "No updates")
        obs_text = str(row.get("observation", "")).lower()
        task_args = row.get("_meta", {}).get("task_args", [])
        if any(str(arg).lower().replace(" ", "_") not in obs_text for arg in task_args):
            missing_task_args += 1

    def mean(values: list[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "rows": len(rows),
        "observation_triples_before_mean": mean(obs_before),
        "observation_triples_after_mean": mean(obs_after),
        "next_triples_before_mean": mean(next_before),
        "next_triples_after_mean": mean(next_after),
        "instance_grounded_rows": instance_grounded,
        "grounded_observation_triples_mean": mean(grounded_obs),
        "grounded_next_observation_triples_mean": mean(grounded_next),
        "observation_triples_after_min": min(obs_after) if obs_after else 0,
        "observation_triples_after_max": max(obs_after) if obs_after else 0,
        "next_triples_after_min": min(next_after) if next_after else 0,
        "next_triples_after_max": max(next_after) if next_after else 0,
        "next_no_updates": no_updates,
        "missing_task_args_in_observation": missing_task_args,
    }


def build_tmow_compact(
    raw_dir: Path | None,
    vh_src: Path,
    output_dir: Path,
    *,
    seen_scene_count: int = 6,
    seen_instruction_count: int = 16,
    seed: int = 42,
    scene_inits_json: Path | None = None,
    candidate_multiplier: int = 12,
    target_trajectories: int | None = 1023,
    seen_seen_eval_per_task: int = 2,
    compact_num_edges: int = DEFAULT_NUM_EDGES,
    next_observation_mode: str = "delta",
) -> None:
    if next_observation_mode not in {"delta", "compact"}:
        raise ValueError(f"Unsupported next observation mode: {next_observation_mode}")
    if target_trajectories is None:
        target_trajectories = 1023

    eg = base._bootstrap_evolving_graph(vh_src)
    EnvironmentGraph = eg["environment"].EnvironmentGraph
    read_script = eg["scripts"].read_script_from_string
    ScriptExecutor = eg["execution"].ScriptExecutor

    print("build mode: paper_like_tmow_compact", flush=True)
    print(f"compact num_edges: {compact_num_edges}", flush=True)
    print(f"next observation mode: {next_observation_mode}", flush=True)

    properties = json.loads(
        (vh_src / "virtualhome" / "resources" / "properties_data.json").read_text()
    )
    scene_inits = _load_scene_inits(raw_dir, scene_inits_json)
    print(f"loaded {len(scene_inits)} scene init graphs: {list(scene_inits)}")

    per_scene_classes = [
        {n["class_name"] for n in graph["nodes"]} for graph in scene_inits.values()
    ]
    union_classes = set().union(*per_scene_classes) if per_scene_classes else set()
    print(f"union classes across {len(per_scene_classes)} scenes: {len(union_classes)}")

    seen_scenes = {key for key in base.PAPER_SEEN_SCENE_KEYS if key in scene_inits}
    if len(seen_scenes) != seen_scene_count:
        raise ValueError(
            f"Expected {seen_scene_count} seen scenes {base.PAPER_SEEN_SCENE_KEYS}, "
            f"only {sorted(seen_scenes)} present in scene_inits."
        )
    unseen_scenes = set(scene_inits) - seen_scenes

    candidate_instructions = base.build_candidate_instructions(
        properties, per_scene_classes, candidate_multiplier=candidate_multiplier
    )
    print(f"candidate instructions: {len(candidate_instructions)}", flush=True)
    for family, count in sorted(Counter(f for f, _ in candidate_instructions).items()):
        print(f"  {family}: {count} candidates (target {base.TARGETS[family]})")

    skipped: Counter = Counter()
    valid_by_task: dict[tuple[str, tuple[str, ...]], dict[str, list[dict]]] = {}
    invalid_tasks: set[tuple[str, tuple[str, ...]]] = set()

    def evaluate_task(task: tuple[str, tuple[str, ...]]) -> dict[str, list[dict]]:
        if task in valid_by_task:
            return valid_by_task[task]
        if task in invalid_tasks:
            return {}
        family, task_args = task
        rows_by_scene = {}
        for scene_name, init_graph in scene_inits.items():
            rows, reason = _execute_tmow_compact_candidate(
                family,
                task_args,
                scene_name,
                init_graph,
                EnvironmentGraph,
                read_script,
                ScriptExecutor,
                num_edges=compact_num_edges,
                next_mode=next_observation_mode,
            )
            if rows is None:
                skipped[reason or "unknown"] += 1
                continue
            rows_by_scene[scene_name] = rows
        if rows_by_scene:
            valid_by_task[task] = rows_by_scene
        else:
            invalid_tasks.add(task)
        return rows_by_scene

    fam_quotas = base._seen_family_quotas(seen_instruction_count)
    print(f"per-family seen quota: {fam_quotas}", flush=True)
    seen_inst: set[tuple[str, tuple[str, ...]]] = set()
    unseen_inst: set[tuple[str, tuple[str, ...]]] = set()

    for family in base.TARGETS:
        family_tasks = [task for task in candidate_instructions if task[0] == family]
        seen_candidates = []
        for order, task in enumerate(family_tasks):
            rows_by_scene = evaluate_task(task)
            valid_scenes = set(rows_by_scene)
            seen_coverage = valid_scenes & seen_scenes
            if seen_coverage:
                seen_candidates.append((task, seen_coverage, valid_scenes, order))

        selected_seen = []
        uncovered_seen = set(seen_scenes)
        while len(selected_seen) < fam_quotas[family] and seen_candidates:
            best_idx, (best_task, best_coverage, _valid_scenes, _order) = max(
                enumerate(seen_candidates),
                key=lambda item: (
                    len(item[1][1] & uncovered_seen),
                    len(item[1][1]),
                    len(item[1][2] & unseen_scenes),
                    len(item[1][2]),
                    -item[1][3],
                ),
            )
            selected_seen.append(best_task)
            uncovered_seen -= best_coverage
            seen_candidates.pop(best_idx)
        if len(selected_seen) < fam_quotas[family]:
            raise ValueError(
                f"Not enough semantically valid seen tasks for {family}: "
                f"{len(selected_seen)} < {fam_quotas[family]}"
            )
        seen_inst.update(selected_seen)

        unseen_need = base.TARGETS[family] - fam_quotas[family]
        unseen_candidates = []
        for order, task in enumerate(family_tasks):
            if task in seen_inst:
                continue
            rows_by_scene = evaluate_task(task)
            valid_scenes = set(rows_by_scene)
            unseen_coverage = valid_scenes & unseen_scenes
            if unseen_coverage:
                unseen_candidates.append((task, unseen_coverage, valid_scenes, order))
        unseen_candidates.sort(key=lambda item: (-len(item[1]), -len(item[2]), item[3]))
        selected_unseen = [
            task for task, _unseen, _valid, _order in unseen_candidates[:unseen_need]
        ]
        if len(selected_unseen) < unseen_need:
            raise ValueError(
                f"Not enough semantically valid unseen tasks for {family}: "
                f"{len(selected_unseen)} < {unseen_need}"
            )
        unseen_inst.update(selected_unseen)
        print(
            f"  {family}: selected seen={len(selected_seen)}, "
            f"unseen={len(selected_unseen)}, "
            f"evaluated_valid={sum(1 for task in valid_by_task if task[0] == family)}",
            flush=True,
        )

    selected_seen_scene_coverage = set().union(
        *(set(valid_by_task[task]) & seen_scenes for task in seen_inst)
    )
    if selected_seen_scene_coverage != seen_scenes:
        missing = sorted(seen_scenes - selected_seen_scene_coverage)
        raise ValueError(f"Selected seen tasks do not cover seen scenes: {missing}")

    instructions = list(seen_inst | unseen_inst)
    print("\nselected semantic instructions:")
    for family, count in sorted(Counter(f for f, _ in instructions).items()):
        print(f"  {family}: {count} selected (target {base.TARGETS[family]})")
    print(f"  skipped evaluated candidate-scene pairs: {dict(skipped)}")

    selected_task_split = {
        **{task: "seen" for task in seen_inst},
        **{task: "unseen" for task in unseen_inst},
    }
    bucket_rows: dict[str, list[dict]] = {family: [] for family in base.TARGETS}
    manifest = {
        "seed": seed,
        "dataset_mode": "paper_like_tmow_compact",
        "observation_mode": "tmow_compact_from_graph_state",
        "next_observation_mode": next_observation_mode,
        "compact_num_edges": compact_num_edges,
        "targets": base.TARGETS,
        "seen_instruction_count": seen_instruction_count,
        "seen_family_quotas": fam_quotas,
        "seen_scenes": sorted(seen_scenes),
        "unseen_scenes": sorted(unseen_scenes),
        "selected_tasks": [
            {
                "task_id": f"{family}:{'|'.join(args)}",
                "family": family,
                "args": list(args),
                "task_split": selected_task_split[(family, args)],
                "goal_triple": list(base._goal_triple(family, args)),
                "valid_scenes": sorted(valid_by_task[(family, args)]),
            }
            for family, args in sorted(instructions)
        ],
        "trajectories": [],
    }

    succeeded_by_scene: Counter = Counter()
    for family, args in sorted(instructions):
        for scene_name, rows in valid_by_task[(family, args)].items():
            task_is_unseen = (family, args) in unseen_inst
            tag = "seen_seen"
            if task_is_unseen and scene_name in unseen_scenes:
                tag = "unseen_unseen"
            elif task_is_unseen:
                tag = "unseen_seen"
            elif scene_name in unseen_scenes:
                tag = "seen_unseen"

            for row in rows:
                out_row = copy.deepcopy(row)
                out_row["_meta"]["split"] = tag
                out_row["_meta"]["task_split"] = selected_task_split[(family, args)]
                bucket_rows[family].append(out_row)
            manifest["trajectories"].append(
                {
                    "trajectory_id": rows[0]["_meta"]["trajectory_id"],
                    "family": family,
                    "args": list(args),
                    "task_split": selected_task_split[(family, args)],
                    "scene": scene_name,
                    "split": tag,
                    "num_steps": rows[0]["_meta"]["num_steps"],
                    "goal_triple": list(base._goal_triple(family, args)),
                }
            )
            succeeded_by_scene[scene_name] += 1

    if target_trajectories is not None:
        kept_trajectory_ids = base._downsample_manifest_trajectories(
            manifest, target_trajectories, seed
        )
        bucket_rows = {
            family: [
                row
                for row in rows
                if row["_meta"]["trajectory_id"] in kept_trajectory_ids
            ]
            for family, rows in bucket_rows.items()
        }
        succeeded_by_scene = Counter(
            trajectory["scene"] for trajectory in manifest["trajectories"]
        )

    print("\nexecution summary:")
    print(
        "  selected successful trajectories by family: "
        f"{dict(Counter(t['family'] for t in manifest['trajectories']))}"
    )
    print(f"  succeeded by scene: {dict(sorted(succeeded_by_scene.items()))}")

    seen_scene_list = sorted(seen_scenes)
    seen_to_idx = {scene: i for i, scene in enumerate(seen_scene_list)}
    per_scene_train: dict[int, dict[str, list[dict]]] = {
        i: defaultdict(list) for i in range(len(seen_scene_list))
    }
    test_buckets: dict[str, list[dict]] = {
        "seen_seen": [],
        "seen_unseen": [],
        "unseen_unseen": [],
        "unseen_seen": [],
    }

    for family_rows in bucket_rows.values():
        for row in family_rows:
            meta = row["_meta"]
            if meta["split"] == "seen_seen":
                per_scene_train[seen_to_idx[meta["scene"]]][meta["trajectory_id"]].append(row)
            else:
                test_buckets[meta["split"]].append(row)

    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_seen_test_ids = base._select_seen_seen_eval_ids(
        per_scene_train, seen_seen_eval_per_task, seed
    )
    seen_seen_test_by_scene = Counter()
    seen_seen_test_by_family = Counter()
    for i, traj_rows in per_scene_train.items():
        for tid in sorted(list(traj_rows)):
            rows = traj_rows[tid]
            if tid in seen_seen_test_ids:
                test_buckets["seen_seen"].extend(rows)
                seen_seen_test_by_scene[rows[0]["_meta"]["scene"]] += 1
                seen_seen_test_by_family[tid.split(":", 2)[1]] += 1
                del traj_rows[tid]
    print(
        "  seen_seen task-aware test trajectories: "
        f"{len(seen_seen_test_ids)} "
        f"(per_task={seen_seen_eval_per_task}, "
        f"by_family={dict(sorted(seen_seen_test_by_family.items()))}, "
        f"by_scene={dict(sorted(seen_seen_test_by_scene.items()))})"
    )

    with (output_dir / "virtualhome_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    split_to_filename = {
        "seen_seen": "test_seen_task_seen_scene.jsonl",
        "seen_unseen": "test_seen_task_unseen_scene.jsonl",
        "unseen_unseen": "test_unseen_task_unseen_scene.jsonl",
        "unseen_seen": "test_unseen_task_seen_scene.jsonl",
    }

    print("\noutput counts:")
    all_rows: list[dict] = []
    for split, filename in split_to_filename.items():
        rows = test_buckets[split]
        rng.shuffle(rows)
        _write_jsonl(output_dir / filename, rows)
        all_rows.extend(rows)
        n_traj = len({row["_meta"]["trajectory_id"] for row in rows})
        print(f"  {filename}: {len(rows)} rows, {n_traj} trajectories")

    eval_dirs = {
        "eval_col_1_seen_seen": "test_seen_task_seen_scene.jsonl",
        "eval_col_2_seen_unseen": "test_seen_task_unseen_scene.jsonl",
        "eval_col_3_unseen_unseen": "test_unseen_task_unseen_scene.jsonl",
    }
    for dirname, filename in eval_dirs.items():
        eval_dir = output_dir / dirname
        eval_dir.mkdir(exist_ok=True)
        test_link = eval_dir / "test.jsonl"
        if test_link.exists() or test_link.is_symlink():
            test_link.unlink()
        test_link.symlink_to(Path("..") / filename)

    for i, scene in enumerate(seen_scene_list):
        scene_dir = output_dir / f"scene_{i}"
        scene_dir.mkdir(exist_ok=True)
        rows = [
            row
            for tid in sorted(per_scene_train[i])
            for row in sorted(
                per_scene_train[i][tid],
                key=lambda item: item["_meta"]["step_index"],
            )
        ]
        rng.shuffle(rows)
        _write_jsonl(scene_dir / "train.jsonl", rows)
        all_rows.extend(rows)
        test_link = scene_dir / "test.jsonl"
        if test_link.exists() or test_link.is_symlink():
            test_link.unlink()
        test_link.symlink_to(Path("..") / "test_seen_task_seen_scene.jsonl")
        n_traj = len({row["_meta"]["trajectory_id"] for row in rows})
        print(f"  scene_{i} ({scene}): train={len(rows)} rows, {n_traj} trajectories")

    generated_trajectories = set()
    for rows in test_buckets.values():
        generated_trajectories.update(row["_meta"]["trajectory_id"] for row in rows)
    for traj_rows in per_scene_train.values():
        generated_trajectories.update(traj_rows)
    if len(generated_trajectories) != 1023:
        print(
            "  WARNING generated trajectory count "
            f"{len(generated_trajectories)} != paper count 1023"
        )

    summary = {
        "data_root": str(output_dir),
        "dataset_mode": "paper_like_tmow_compact",
        "observation_mode": "tmow_compact_from_graph_state",
        "next_observation_mode": next_observation_mode,
        "compact_num_edges": compact_num_edges,
        "target_trajectories": target_trajectories,
        "generated_trajectories": len(generated_trajectories),
        "selected_tasks": len(instructions),
        "seen_tasks": len(seen_inst),
        "unseen_tasks": len(unseen_inst),
        "seen_scenes": len(seen_scenes),
        "unseen_scenes": len(unseen_scenes),
        "preprocessing": _collect_preprocessing_summary(all_rows),
    }
    with (output_dir / "tmow_compact_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("\ncompact summary:")
    print(json.dumps(summary["preprocessing"], indent=2, sort_keys=True))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build WorMI VirtualHome data with TMoW-style compact observations "
            "directly from VirtualHome graph states."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--scene-inits-json", type=Path, default=None)
    parser.add_argument("--vh-src", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seen-scenes", type=int, default=6)
    parser.add_argument("--seen-instructions", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-multiplier", type=int, default=12)
    parser.add_argument("--target-trajectories", type=int, default=1023)
    parser.add_argument("--seen-seen-eval-per-task", type=int, default=2)
    parser.add_argument("--compact-num-edges", type=int, default=DEFAULT_NUM_EDGES)
    parser.add_argument(
        "--next-observation-mode",
        choices=["delta", "compact"],
        default="delta",
        help="delta writes TMoW-style compact state updates; compact writes compact next states.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    build_tmow_compact(
        args.raw_dir,
        args.vh_src,
        args.output_dir,
        seen_scene_count=args.seen_scenes,
        seen_instruction_count=args.seen_instructions,
        seed=args.seed,
        scene_inits_json=args.scene_inits_json,
        candidate_multiplier=args.candidate_multiplier,
        target_trajectories=args.target_trajectories,
        seen_seen_eval_per_task=args.seen_seen_eval_per_task,
        compact_num_edges=args.compact_num_edges,
        next_observation_mode=args.next_observation_mode,
    )


if __name__ == "__main__":
    main()
