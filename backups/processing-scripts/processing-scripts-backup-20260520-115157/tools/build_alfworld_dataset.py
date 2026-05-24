"""Build the ALFWorld split of WorMI's training data, aligned with paper §B.2.

Per the paper (arxiv 2509.03956 §4 + §B.2 + Tables A.3/A.4):
- 3,554 episodes (== alfworld textual env's `train` split, 3,553 solvable games)
- 6 task types: Pick & Place / Pick Two & Place / Look at Obj in Light /
  Pick Clean & Place / Pick Heat & Place / Pick Cool & Place
- "Following CL-ALFRED benchmark settings (Kim et al., 2024)" — 4 scene types
  (3 seen, 1 unseen), 6 task types (4 seen, 2 unseen).
- Observation/action use ALFWorld textual env's native format (Figure A.3).

For each solvable game we run the textual env and follow `extra.expert_plan`
deterministically, recording the (obs, action, reward, done, next_obs) sequence
as one jsonl row per episode:
    {task, trial_name, history: [{observation, action, reward, dones, next_observation}, ...]}

Bucketed under `<output_dir>/{bathrooms|bedrooms|kitchens|livingrooms}/{train,test}.jsonl`
(by ALFRED scene-number → room mapping).

Train/test split (paper):
- train.jsonl = seen task type × seen scene type
- test.jsonl  = (seen tasks × unseen scenes) ∪ (unseen tasks × any scenes)

Paper does not list which 4/2 task types and 3/1 scene types are "seen". The
HARD constraint we must respect is that paper Table 1 column 3 "Unseen task ×
Unseen scene" must be non-empty, i.e. the 2 unseen task types must each be
physically realisable in bathrooms (the held-out scene type). ALFRED tasks
need specific receptacles: heat→microwave (kitchen only), cool→fridge
(kitchen only), look_at_obj→lamp (bedroom/livingroom only), clean→sink (kitch
+ bath). Only {pick_and_place_simple, pick_two_obj_and_place,
pick_clean_then_place_in_recep} occur in bathrooms. Of the C(3,2)=3 candidate
pairs we pick the only one that is both data-sufficient (every seen room
≥261 train trials) and conceptually coherent (compositional tasks held out,
atomic pick_and_place_simple kept as seen baseline):
- unseen_task_types: pick_two_obj_and_place, pick_clean_then_place_in_recep
- unseen_scene_type: bathrooms (smallest object inventory in CL-ALFRED stats)

Usage (textual mode runs purely on CPU, no GPU/THOR/X server needed):
    /srv/scratch/z5524306/alfworld-venv/bin/python tools/build_alfworld_dataset.py \\
        --alfworld-data /tmp/alfworld-data \\
        --output-dir /srv/scratch/z5524306/wormi-data/alfworld
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


# Paper Table A.4 — 6 task types from ALFRED's standard 6 (NOT including
# CL-ALFRED's extra `pick_and_place_with_movable_recep`)
ALL_TASK_TYPES = [
    "pick_and_place_simple",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
]

# Paper §4: 4 seen + 2 unseen task types. Selection constrained by paper
# Table 1 col-3 physical feasibility — see module docstring. The previous
# choice {heat, cool} produced col-3 = ∅ and was corrected on 2026-05-13.
UNSEEN_TASK_TYPES = {
    "pick_two_obj_and_place",
    "pick_clean_then_place_in_recep",
}
SEEN_TASK_TYPES = set(ALL_TASK_TYPES) - UNSEEN_TASK_TYPES

# alfworld task_type IDs (1-6) per its base_config.yaml comment:
# 1 - Pick & Place, 2 - Examine in Light, 3 - Clean & Place,
# 4 - Heat & Place, 5 - Cool & Place, 6 - Pick Two & Place
ALL_TASK_TYPE_IDS = [1, 2, 3, 4, 5, 6]

# AI2-THOR / ALFRED scene-number → room type mapping
SCENE_TO_ROOM = {
    "kitchens": range(1, 31),       # scenes 1-30
    "livingrooms": range(201, 231), # 201-230
    "bedrooms": range(301, 331),    # 301-330
    "bathrooms": range(401, 431),   # 401-430
}
ALL_SCENE_TYPES = sorted(SCENE_TO_ROOM)

# Paper §4: 3 seen + 1 unseen scene types
UNSEEN_SCENE_TYPES = {"bathrooms"}
SEEN_SCENE_TYPES = set(ALL_SCENE_TYPES) - UNSEEN_SCENE_TYPES


_TRIAL_DIR_RE = re.compile(
    r"^([a-z_]+?)-([^-/]+)-([^-/]+)-([^-/]+)-(\d+)$"
)


def parse_gamefile_path(gamefile: str) -> tuple[str, int, str] | None:
    """Extract (task_type, scene_num, trial_name) from a game.tw-pddl path."""
    parts = Path(gamefile).parts
    # ...json_2.1.1/<split>/<task_dir>/<trial_dir>/game.tw-pddl
    if len(parts) < 4:
        return None
    trial_dir = parts[-2]
    task_dir = parts[-3]
    m = _TRIAL_DIR_RE.match(task_dir)
    if not m:
        return None
    task_type = m.group(1)
    scene_num = int(m.group(5))
    return task_type, scene_num, trial_dir


def scene_num_to_type(num: int) -> str | None:
    for room, rng in SCENE_TO_ROOM.items():
        if num in rng:
            return room
    return None


def build_config(alfworld_data: Path, num_train_games: int) -> dict:
    """Minimal config dict that satisfies AlfredTWEnv's reads, mirroring
    alfworld's base_config.yaml. We avoid generic.load_config() because it
    requires a YAML file path and pulls in modules we don't need.
    """
    return {
        "env": {
            "type": "AlfredTWEnv",
            "task_types": ALL_TASK_TYPE_IDS,
            "expert_timeout_steps": 150,
            "expert_type": "handcoded",
            "goal_desc_human_anns_prob": 0.0,
            "hide_init_receptacles": False,
            "hide_object_locations": False,
            "domain_randomization": False,
        },
        "dataset": {
            "data_path": str(alfworld_data / "json_2.1.1" / "train"),
            "eval_id_data_path": str(alfworld_data / "json_2.1.1" / "valid_seen"),
            "eval_ood_data_path": str(alfworld_data / "json_2.1.1" / "valid_unseen"),
            "num_train_games": num_train_games,
            "num_eval_games": 0,
        },
        "controller": {"type": "oracle", "debug": False, "load_receps": True},
        "general": {
            "random_seed": 42,
            "use_cuda": False,
            "task": "alfred",
            "training_method": "dagger",
            "observation_pool_capacity": 5,
            "hide_init_receptacles": False,
        },
        "logic": {
            "domain": str(alfworld_data / "logic" / "alfred.pddl"),
            "grammar": str(alfworld_data / "logic" / "alfred.twl2"),
        },
        "dagger": {
            "training": {"max_nb_steps_per_episode": 50},
        },
    }


def _unwrap(value):
    """alfworld returns batch values; for batch_size=1 grab the first slot."""
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


def reset_and_identify(env) -> tuple[list, dict, str]:
    """Reset the env and return (obs, infos, gamefile-actually-loaded).

    Identifying the loaded gamefile from the env (NOT from a parallel
    enumerate of `alfred_env.game_files`) is what protects against
    metadata/observation drift: env.reset() advances its own internal queue
    and any control-flow `continue` in the build loop would otherwise leave
    our loop pointer ahead of the env pointer. AlfworldExpertExtractor adds
    `extra.gamefile` to infos for this purpose.
    """
    obs, infos = env.reset()
    gamefile = infos.get("extra.gamefile")
    if gamefile is None:
        raise RuntimeError(
            "env did not expose `extra.gamefile` in infos — alfworld version "
            "may have changed; cannot determine which game was loaded."
        )
    return obs, infos, _unwrap(gamefile)


def collect_episode(obs, infos, env, max_steps: int = 30) -> list[dict] | None:
    """Follow the expert plan starting from a fresh env reset.

    Caller is expected to have already called `reset_and_identify` and pass in
    the resulting `obs` and `infos` so that gamefile identity is captured
    before any env step happens.
    """
    if "extra.expert_plan" not in infos:
        return None
    history: list[dict] = []
    for _step in range(max_steps):
        plan = infos["extra.expert_plan"][0]
        if not plan:
            break
        action = plan[0]
        prev_obs = obs[0]
        obs, scores, dones, infos = env.step([action])
        history.append(
            {
                "observation": prev_obs,
                "action": action,
                "reward": float(scores[0]) if scores else 0.0,
                "dones": bool(dones[0]),
                "next_observation": obs[0],
            }
        )
        if dones[0]:
            break
    return history


def build(
    alfworld_data: Path,
    output_dir: Path,
    limit: int | None = None,
    expert_timeout: int = 150,
) -> None:
    os.environ["ALFWORLD_DATA"] = str(alfworld_data)

    # Make sure logic files are in place
    logic_dir = alfworld_data / "logic"
    logic_dir.mkdir(parents=True, exist_ok=True)
    if not (logic_dir / "alfred.pddl").exists():
        import alfworld.info as info
        import shutil

        shutil.copy(info.ALFRED_PDDL_PATH, logic_dir / "alfred.pddl")
        shutil.copy(info.ALFRED_TWL2_PATH, logic_dir / "alfred.twl2")

    from alfworld.agents.environment import get_environment

    config = build_config(alfworld_data, num_train_games=limit if limit else -1)
    env_cls = get_environment("AlfredTWEnv")
    alfred_env = env_cls(config, train_eval="train")
    env = alfred_env.init_env(batch_size=1)
    print(f"\nALFWorld textual env loaded: {alfred_env.num_games} games\n", flush=True)

    # Open four (scene_type) × {train,test} jsonl files in append mode for
    # streaming writes — no full-dataset accumulation in RAM, and the partial
    # output is salvageable if we get interrupted.
    #
    # Resume support: if jsonl files already contain rows, harvest trial_names
    # from them and skip those games. We use "a" mode so existing rows are kept.
    output_dir.mkdir(parents=True, exist_ok=True)
    done_trials: set[str] = set()
    for st in ALL_SCENE_TYPES:
        (output_dir / st).mkdir(exist_ok=True)
        for split in ("train", "test"):
            f = output_dir / st / f"{split}.jsonl"
            if f.exists():
                with f.open() as fh:
                    for line in fh:
                        try:
                            done_trials.add(json.loads(line)["trial_name"])
                        except Exception:
                            pass
    if done_trials:
        print(f"resume: {len(done_trials)} games already in output", flush=True)
    handles: dict[tuple[str, str], any] = {}
    for st in ALL_SCENE_TYPES:
        for split in ("train", "test"):
            handles[(st, split)] = (output_dir / st / f"{split}.jsonl").open("a")

    skipped: Counter = Counter()
    by_task_seen: Counter = Counter()
    by_scene_seen: Counter = Counter()
    bucket_counts: Counter = Counter()
    import time
    t_start = time.time()

    n_games = alfred_env.num_games
    seen_gamefiles: set[str] = set()
    # Reset more times than we have games so cycling envs eventually visit
    # everything. We exit early once `seen_gamefiles` covers all of them.
    max_iter = n_games * 2 + 10
    iters = 0
    while len(seen_gamefiles) < n_games and iters < max_iter:
        iters += 1
        try:
            obs, infos, gamefile = reset_and_identify(env)
        except Exception as e:
            skipped[f"reset_error:{type(e).__name__}"] += 1
            continue
        if gamefile in seen_gamefiles:
            # env cycled past the end of its queue; nothing new here.
            continue
        seen_gamefiles.add(gamefile)

        parsed = parse_gamefile_path(gamefile)
        if parsed is None:
            skipped["unparseable_path"] += 1
            continue
        task_type, scene_num, trial_name = parsed
        if task_type not in ALL_TASK_TYPES:
            skipped["non_paper_task_type"] += 1
            continue
        scene_type = scene_num_to_type(scene_num)
        if scene_type is None:
            skipped["unmapped_scene"] += 1
            continue
        if trial_name in done_trials:
            # Env reset has already advanced past this game; we ate one
            # reset() but skip writing a duplicate row. This is the correct
            # resume behaviour now that env identity is the source of truth.
            skipped["already_done"] += 1
            continue

        try:
            history = collect_episode(obs, infos, env, max_steps=expert_timeout)
        except Exception as e:
            skipped[f"step_error:{type(e).__name__}"] += 1
            continue
        if not history:
            skipped["no_history"] += 1
            continue

        is_task_seen = task_type in SEEN_TASK_TYPES
        is_scene_seen = scene_type in SEEN_SCENE_TYPES
        split_tag = "train" if (is_task_seen and is_scene_seen) else "test"
        by_task_seen[is_task_seen] += 1
        by_scene_seen[is_scene_seen] += 1
        bucket_counts[(scene_type, split_tag)] += 1

        row = {"task": task_type, "trial_name": trial_name, "history": history}
        f = handles[(scene_type, split_tag)]
        f.write(json.dumps(row) + "\n")
        f.flush()

        if len(seen_gamefiles) % 50 == 0:
            elapsed = time.time() - t_start
            rate = len(seen_gamefiles) / elapsed
            eta_min = (n_games - len(seen_gamefiles)) / rate / 60
            print(
                f"  [{len(seen_gamefiles):4d}/{n_games}] "
                f"{rate:.1f} games/s, ETA {eta_min:.1f} min",
                flush=True,
            )

    for f in handles.values():
        f.close()
    if len(seen_gamefiles) < n_games:
        print(
            f"WARNING: only saw {len(seen_gamefiles)}/{n_games} unique gamefiles "
            f"after {iters} resets; env may have a deterministic shuffled queue "
            f"that doesn't cover everything in 2× passes."
        )

    print(f"\nSkipped: {dict(skipped)}")
    print(f"task_seen distribution: {dict(by_task_seen)}")
    print(f"scene_seen distribution: {dict(by_scene_seen)}")
    for st in ALL_SCENE_TYPES:
        print(f"  {st}: {bucket_counts[(st, 'train')]} train + {bucket_counts[(st, 'test')]} test")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alfworld-data", type=Path, required=True,
                   help="ALFWORLD_DATA path (containing json_2.1.1/{train,...}, logic/, etc.)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="If set, process only first N games (smoke test).")
    p.add_argument("--expert-timeout", type=int, default=150,
                   help="Max expert steps per episode (alfworld default).")
    args = p.parse_args()
    build(args.alfworld_data, args.output_dir, args.limit, args.expert_timeout)


if __name__ == "__main__":
    main()
