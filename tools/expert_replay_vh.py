"""T1 Expert Replay: drive the EVAL pipeline with gold actions (no model).

Feeds the dataset's ground-truth class-level `action` strings through the FULL
eval grounding+execution path used by wormi/scripts/eval_vh_rollout.py and
measures whether the data can in principle reach SR=100% through THIS pipeline.

Also computes a control: gold `_meta.script_line` (instance-bound) through the
SAME eval env (execute_one_step, instance_selection=True) -> env-attainable
ceiling. The gap (script_line_SR - expert_action_SR) isolates _choose_node_id
binding loss.

NO model / checkpoint load. cwd = /root/WorMI.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.build_virtualhome_dataset import _bootstrap_evolving_graph
import wormi.scripts.eval_vh_rollout as R
from wormi.scripts.eval_table1 import _group_virtualhome, _read_jsonl

DATASET = Path("/root/autodl-tmp/wormi-data/virtualhome-realtasks-v3-20260530")
VH_SRC = Path("/root/autodl-tmp/wormi-data/virtualhome-src")
SCENE_INITS = DATASET / "scene_inits.json"

SPLITS = [
    "test_seen_task_seen_scene.jsonl",
    "test_seen_task_unseen_scene.jsonl",
    "test_unseen_task_unseen_scene.jsonl",
]


@dataclass
class _Args:
    """Minimal stand-in for VirtualHomeRolloutArgs for _resolve_observation_format."""
    observation_format: str = "auto"


def _run_episode_for_lines(
    eg, scene_inits, scene, lines, goal, max_steps,
):
    """Execute a sequence of fully-grounded script_line strings through the eval env.

    lines: list of script_line strings (already instance-bound), one per step.
    Returns (success, executed, precond_fail, exc_fail, missing_line).
    """
    EnvironmentGraph = eg["environment"].EnvironmentGraph
    EnvironmentState = eg["environment"].EnvironmentState
    ScriptExecutor = eg["execution"].ScriptExecutor
    read_script = eg["scripts"].read_script_from_string

    env_graph = EnvironmentGraph(copy.deepcopy(scene_inits[scene]))
    # Mirror the eval pipeline (instance_selection=True): script_line strings are
    # fully instance-bound (real graph node ids), so the executor honours them
    # exactly instead of re-binding class objects to the first-found instance.
    state = EnvironmentState(env_graph, {}, instance_selection=True)
    executor = ScriptExecutor(env_graph, {}, char_index=0)

    executed = precond_fail = exc_fail = missing_line = 0
    if R._goal_satisfied(state.to_dict(), goal):
        return True, executed, precond_fail, exc_fail, missing_line

    for line in lines[:max_steps]:
        if not line:
            missing_line += 1
        else:
            try:
                ok, new_state = executor.execute_one_step(read_script(line), state)
                if ok:
                    state = new_state
                    executed += 1
                else:
                    precond_fail += 1
            except Exception:
                exc_fail += 1
        if R._goal_satisfied(state.to_dict(), goal):
            return True, executed, precond_fail, exc_fail, missing_line
    return False, executed, precond_fail, exc_fail, missing_line


def _eval_episode_expert(eg, scene_inits, episode, max_steps):
    """Faithful replica of R._eval_episode but with expert action substituted.

    Returns dict of per-episode stats.
    """
    EnvironmentGraph = eg["environment"].EnvironmentGraph
    EnvironmentState = eg["environment"].EnvironmentState
    ScriptExecutor = eg["execution"].ScriptExecutor
    read_script = eg["scripts"].read_script_from_string

    first = episode[0]
    meta = first.get("_meta", {})
    scene = meta.get("scene")
    if scene not in scene_inits:
        raise KeyError(f"Scene {scene!r} not in scene_inits")
    goal = R._infer_goal(first)

    env_graph = EnvironmentGraph(copy.deepcopy(scene_inits[scene]))
    # Mirror the eval pipeline (instance_selection=True): _script_line_from_prediction
    # resolves each class-level expert action to the goal-relevant graph node id,
    # and the executor honours that id exactly.
    state = EnvironmentState(env_graph, {}, instance_selection=True)
    executor = ScriptExecutor(env_graph, {}, char_index=0)

    # Mirror the eval pipeline: goal-aware instance binding is resolved once from
    # the reset (initial) scene graph + goal/task-args -- the same inputs the
    # rollout gives the agent. No gold actions / _meta.script_line are read here.
    goal_binding = R._build_goal_binding(scene_inits[scene], goal)

    parse_fail = precond_fail = exc_fail = executed = 0
    binding_divergence = 0  # eval-grounded script_line != gold _meta.script_line
    steps_with_gold = 0

    if R._goal_satisfied(state.to_dict(), goal):
        return {
            "scene": scene, "success": True, "steps": 0,
            "parse_fail": 0, "precond_fail": 0, "exc_fail": 0, "executed": 0,
            "binding_divergence": 0, "steps_with_gold": 0,
            "fail_only_binding": False, "preloop_success": True,
        }

    max_t = min(max_steps, len(episode))
    for t in range(max_t):
        row = episode[t]
        graph = state.to_dict()
        prediction = row["action"]  # THE EXPERT ACTION (class-level)
        script_line, parse_err = R._script_line_from_prediction(
            graph, prediction, goal, goal_binding
        )

        gold_line = (row.get("_meta") or {}).get("script_line")
        if gold_line:
            steps_with_gold += 1
            if script_line != gold_line:
                binding_divergence += 1

        if script_line is None:
            parse_fail += 1
        else:
            try:
                ok, new_state = executor.execute_one_step(read_script(script_line), state)
                if ok:
                    state = new_state
                    executed += 1
                else:
                    precond_fail += 1
            except Exception:
                exc_fail += 1

        if R._goal_satisfied(state.to_dict(), goal):
            return {
                "scene": scene, "success": True, "steps": t + 1,
                "parse_fail": parse_fail, "precond_fail": precond_fail,
                "exc_fail": exc_fail, "executed": executed,
                "binding_divergence": binding_divergence,
                "steps_with_gold": steps_with_gold,
                "fail_only_binding": False, "preloop_success": False,
            }

    # Failed. Determine if failure is "binding-divergence-only": i.e. the gold
    # script_line path would have succeeded but the eval-grounded path differs.
    gold_lines = [(r.get("_meta") or {}).get("script_line") for r in episode]
    gold_success, *_ = _run_episode_for_lines(
        eg, scene_inits, scene, gold_lines, goal, max_steps
    )
    fail_only_binding = bool(gold_success and binding_divergence > 0)

    return {
        "scene": scene, "success": False, "steps": max_t,
        "parse_fail": parse_fail, "precond_fail": precond_fail,
        "exc_fail": exc_fail, "executed": executed,
        "binding_divergence": binding_divergence,
        "steps_with_gold": steps_with_gold,
        "fail_only_binding": fail_only_binding, "preloop_success": False,
    }


def _control_gold_scriptline(eg, scene_inits, episode, max_steps):
    """Control (a): gold _meta.script_line through the SAME eval env."""
    first = episode[0]
    scene = first["_meta"]["scene"]
    goal = R._infer_goal(first)
    lines = [(r.get("_meta") or {}).get("script_line") for r in episode]
    success, *_ = _run_episode_for_lines(eg, scene_inits, scene, lines, goal, max_steps)
    return success


def run_split(eg, scene_inits, split, max_steps=30, verbose_fail=True):
    path = DATASET / split
    rows = _read_jsonl(path)
    episodes = _group_virtualhome(rows)  # NO sampling

    # Assert observation_format resolves to 'full' for v3 (no obs_preprocessing).
    fmt = R._resolve_observation_format(episodes[0][0], _Args(observation_format="auto"))
    assert fmt == "full", f"expected observation_format=full, got {fmt!r}"

    n = len(episodes)
    expert_success = 0
    gold_success = 0
    total_binding_div = 0
    total_gold_steps = 0
    fail_only_binding = 0
    fail_records = []

    for ep in episodes:
        stats = _eval_episode_expert(eg, scene_inits, ep, max_steps)
        if stats["success"]:
            expert_success += 1
        else:
            if stats["fail_only_binding"]:
                fail_only_binding += 1
            fail_records.append(stats)
        total_binding_div += stats["binding_divergence"]
        total_gold_steps += stats["steps_with_gold"]

        if _control_gold_scriptline(eg, scene_inits, ep, max_steps):
            gold_success += 1

    expert_sr = expert_success / n
    gold_sr = gold_success / n
    binding_div_rate = (total_binding_div / total_gold_steps) if total_gold_steps else 0.0

    print(f"\n===== {split} =====")
    print(f"observation_format resolved: {fmt}")
    print(f"episodes: {n}")
    print(f"EXPERT-ACTION SR : {expert_sr:.4f} ({expert_success}/{n})")
    print(f"GOLD-SCRIPTLINE SR (control, eval env): {gold_sr:.4f} ({gold_success}/{n})")
    print(f"_choose_node_id binding loss (gold_sr - expert_sr): {gold_sr - expert_sr:.4f}")
    print(f"mean binding-divergence rate (per gold step): {binding_div_rate:.4f}")
    print(f"#episodes failing ONLY due to binding divergence: {fail_only_binding}")
    print(f"#total expert-action failures: {n - expert_success}")
    if verbose_fail and fail_records:
        print("--- first failures ---")
        for fr in fail_records[:12]:
            print(f"  scene={fr['scene']} steps={fr['steps']} "
                  f"parse={fr['parse_fail']} precond={fr['precond_fail']} "
                  f"exc={fr['exc_fail']} exec={fr['executed']} "
                  f"binding_div={fr['binding_divergence']} "
                  f"fail_only_binding={fr['fail_only_binding']}")

    return {
        "split": split,
        "episodes": n,
        "expert_sr": expert_sr,
        "expert_success": expert_success,
        "gold_scriptline_sr": gold_sr,
        "gold_success": gold_success,
        "binding_loss": gold_sr - expert_sr,
        "binding_divergence_rate": binding_div_rate,
        "fail_only_binding": fail_only_binding,
        "total_failures": n - expert_success,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None, help="one split filename; default = all three")
    ap.add_argument("--max-steps", type=int, default=30)
    args = ap.parse_args()

    assert SCENE_INITS.exists(), SCENE_INITS
    assert VH_SRC.exists(), VH_SRC
    scene_inits = json.loads(SCENE_INITS.read_text())
    print(f"scene_inits keys: {len(scene_inits)}")
    eg = _bootstrap_evolving_graph(VH_SRC)

    splits = [args.split] if args.split else SPLITS
    summary = []
    for sp in splits:
        summary.append(run_split(eg, scene_inits, sp, max_steps=args.max_steps))

    print("\n===== SUMMARY =====")
    for s in summary:
        print(json.dumps(s))


if __name__ == "__main__":
    main()
