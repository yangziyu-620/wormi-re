"""Build validated ALFWorld benchmark releases for method comparison.

The script consumes episode-level ALFWorld JSONL rows produced by
``tools/collect_alfworld_episodes.py`` or the archived
``share/alfworld-initial-*`` package, then materializes reproducible protocols:

1. ``paper-compatible-v1``: keeps the paper's coarse "4 scene types, 3 seen +
   1 unseen" shape. We use kitchens as the unseen scene type and heat/cool as
   unseen tasks because this keeps both Table-1-style col_2 and col_3 non-empty
   and avoids the current bathrooms split's single-task col_2.
2. ``balanced-scene-instance-v1``: splits by ALFRED scene number within each
   room type. This is not a paper-reproduction protocol; it is a stronger
   method-comparison protocol because every physical task family has a chance
   to appear in seen/unseen scene evaluations.

Each protocol is written as canonical episodes plus method-specific views. The
views are all derived from the same zero-shot training pool, so baselines and
WorMI can be compared without accidentally giving one method more episodes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import random
import re
import shutil
import statistics
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ALL_TASKS = [
    "pick_and_place_simple",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
]

TASK_SHORT = {
    "pick_and_place_simple": "pick_simple",
    "pick_two_obj_and_place": "pick_two",
    "look_at_obj_in_light": "look_at_obj",
    "pick_clean_then_place_in_recep": "pick_clean",
    "pick_heat_then_place_in_recep": "pick_heat",
    "pick_cool_then_place_in_recep": "pick_cool",
}

ROOM_ORDER = ["bathrooms", "bedrooms", "kitchens", "livingrooms"]
SCENE_TO_ROOM = {
    "kitchens": range(1, 31),
    "livingrooms": range(201, 231),
    "bedrooms": range(301, 331),
    "bathrooms": range(401, 431),
}

TRIAL_DIR_RE = re.compile(r"^([a-z_]+?)-([^-/]+)-([^-/]+)-([^-/]+)-(\d+)$")


@dataclass(frozen=True)
class ProtocolConfig:
    name: str
    title: str
    task_split_name: str
    seen_tasks: set[str]
    unseen_tasks: set[str]
    scene_mode: str
    rationale: list[str]
    unseen_room: str | None = None
    unseen_scene_fraction: float = 0.2


PAPER_COMPATIBLE = ProtocolConfig(
    name="paper-compatible-v1",
    title="ALFWorld Paper-Compatible Room-Type Protocol v1",
    task_split_name="heat_cool_unseen",
    seen_tasks={
        "pick_and_place_simple",
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
        "pick_clean_then_place_in_recep",
    },
    unseen_tasks={
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
    },
    scene_mode="room_type",
    unseen_room="kitchens",
    rationale=[
        "The paper states 4 scene types with 3 seen and 1 unseen, but does not "
        "publish the held-out scene type or the two held-out task types.",
        "Kitchens are selected as the unseen scene type because heat and cool "
        "tasks are physically kitchen-bound; this makes col_3 non-empty.",
        "The seen-task unseen-scene column still contains multiple tasks "
        "(simple, clean, pick-two), avoiding the bathrooms split where col_2 "
        "collapses to only pick_and_place_simple.",
        "Unseen-task episodes are excluded from all zero-shot training views; "
        "WorMI world models are scene-clustered from the same seen-domain pool "
        "instead of being trained as task-type experts.",
    ],
)

BALANCED_SCENE_INSTANCE = ProtocolConfig(
    name="balanced-scene-instance-v1",
    title="ALFWorld Balanced Scene-Instance Protocol v1",
    task_split_name="compositional_unseen",
    seen_tasks={
        "pick_and_place_simple",
        "look_at_obj_in_light",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
    },
    unseen_tasks={
        "pick_two_obj_and_place",
        "pick_clean_then_place_in_recep",
    },
    scene_mode="scene_num",
    rationale=[
        "This protocol is for fair method comparison, not direct paper-table "
        "reproduction.",
        "The scene split is by ALFRED scene number within every room type, so "
        "unseen-scene evaluation is not reduced to a single room's physical "
        "task affordances.",
        "Pick-two and clean are held out as compositional unseen tasks while "
        "atomic pick-and-place remains a seen-task baseline.",
        "Unseen-task episodes are excluded from all zero-shot training views; "
        "WorMI world models are scene-clustered from the same seen-domain pool.",
    ],
)

PROTOCOLS = {
    PAPER_COMPATIBLE.name: PAPER_COMPATIBLE,
    BALANCED_SCENE_INSTANCE.name: BALANCED_SCENE_INSTANCE,
}


def room_for_scene(scene_num: int) -> str:
    for room, values in SCENE_TO_ROOM.items():
        if scene_num in values:
            return room
    raise ValueError(f"Unknown ALFRED scene number: {scene_num}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            n += 1
    return n


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.relative_to(dst.parent))
    except ValueError:
        shutil.copy2(src, dst)


def stable_episode_id(source_gamefile: str | None, trial_name: str) -> str:
    payload = source_gamefile or trial_name
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def load_tw_pddl_metadata(zip_path: Path) -> dict[str, dict[str, Any]]:
    """Map trial_name to task/scene metadata by reading zip member names."""
    out: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith("/game.tw-pddl"):
                continue
            parts = Path(name).parts
            if len(parts) < 5:
                continue
            split = parts[-4]
            task_dir = parts[-3]
            trial_name = parts[-2]
            match = TRIAL_DIR_RE.match(task_dir)
            if not match:
                continue
            task_type = match.group(1)
            scene_num = int(match.group(5))
            out[trial_name] = {
                "source_gamefile": name,
                "alfworld_split": split,
                "task_type": task_type,
                "scene_num": scene_num,
                "room_type": room_for_scene(scene_num),
            }
    return out


def load_source_episodes(source_root: Path, tw_pddl_zip: Path) -> list[dict[str, Any]]:
    metadata = load_tw_pddl_metadata(tw_pddl_zip)
    rows_by_trial: dict[str, dict[str, Any]] = {}
    for path in sorted(source_root.rglob("*.jsonl")):
        room_from_path = path.parent.name
        for row in read_jsonl(path):
            trial_name = row["trial_name"]
            if trial_name in rows_by_trial:
                continue
            if trial_name not in metadata:
                raise KeyError(
                    f"{trial_name} from {path} is missing from {tw_pddl_zip}"
                )
            meta = metadata[trial_name]
            if row["task"] != meta["task_type"]:
                raise ValueError(
                    f"Task mismatch for {trial_name}: jsonl={row['task']} "
                    f"zip={meta['task_type']}"
                )
            if room_from_path in ROOM_ORDER and room_from_path != meta["room_type"]:
                raise ValueError(
                    f"Room mismatch for {trial_name}: path={room_from_path} "
                    f"zip={meta['room_type']}"
                )

            enriched = copy.deepcopy(row)
            enriched_meta = dict(enriched.get("_meta", {}))
            enriched_meta.update(meta)
            enriched_meta.update(
                {
                    "episode_id": stable_episode_id(
                        meta["source_gamefile"], trial_name
                    ),
                    "task_short": TASK_SHORT[row["task"]],
                    "expert_steps": len(row["history"]),
                    "expert_success": bool(row["history"][-1].get("dones"))
                    if row.get("history")
                    else False,
                    "source_jsonl": str(path),
                }
            )
            enriched["_meta"] = enriched_meta
            rows_by_trial[trial_name] = enriched

    rows = sorted(
        rows_by_trial.values(),
        key=lambda r: (
            int(r["_meta"]["scene_num"]),
            r["task"],
            r["trial_name"],
        ),
    )
    return rows


def row_key(row: dict[str, Any]) -> str:
    return row["_meta"]["episode_id"]


def split_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "episodes": 0,
            "tasks": {},
            "rooms": {},
            "scene_nums": {},
            "expert_steps": {},
        }
    steps = [len(r["history"]) for r in rows]
    return {
        "episodes": len(rows),
        "tasks": dict(sorted(Counter(r["task"] for r in rows).items())),
        "rooms": dict(sorted(Counter(r["_meta"]["room_type"] for r in rows).items())),
        "scene_nums": {
            str(k): v
            for k, v in sorted(Counter(r["_meta"]["scene_num"] for r in rows).items())
        },
        "expert_steps": {
            "mean": statistics.mean(steps),
            "median": statistics.median(steps),
            "min": min(steps),
            "max": max(steps),
            "p90": sorted(steps)[int(0.9 * (len(steps) - 1))],
        },
    }


def task_room_matrix(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {
        task: {room: 0 for room in ROOM_ORDER} for task in ALL_TASKS
    }
    for row in rows:
        matrix[row["task"]][row["_meta"]["room_type"]] += 1
    return matrix


def choose_scene_nums_for_balanced_protocol(
    rows: list[dict[str, Any]], fraction: float
) -> dict[str, list[int]]:
    by_room_scene_task: dict[str, dict[int, Counter[str]]] = {
        room: defaultdict(Counter) for room in ROOM_ORDER
    }
    for row in rows:
        room = row["_meta"]["room_type"]
        scene = int(row["_meta"]["scene_num"])
        by_room_scene_task[room][scene][row["task"]] += 1

    selected: dict[str, list[int]] = {}
    for room in ROOM_ORDER:
        scene_counts = by_room_scene_task[room]
        scenes = sorted(scene_counts)
        if not scenes:
            selected[room] = []
            continue
        k = max(1, round(len(scenes) * fraction))
        desired_tasks = sorted(
            {task for counts in scene_counts.values() for task in counts}
        )
        room_totals = Counter()
        for counts in scene_counts.values():
            room_totals.update(counts)
        target = {
            task: max(1.0, room_totals[task] * k / len(scenes))
            for task in desired_tasks
        }

        best_combo: tuple[int, ...] | None = None
        best_score: tuple[float, ...] | None = None
        for combo in itertools.combinations(scenes, k):
            counts = Counter()
            for scene in combo:
                counts.update(scene_counts[scene])
            coverage = sum(1 for task in desired_tasks if counts[task] > 0)
            min_seen = min((counts[task] for task in desired_tasks), default=0)
            balance_penalty = sum(
                abs(counts[task] - target[task]) / target[task]
                for task in desired_tasks
            )
            total_desired = sum(counts[task] for task in desired_tasks)
            score = (
                float(coverage),
                float(min_seen),
                -float(balance_penalty),
                float(total_desired),
                -float(sum(combo)),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_combo = combo
        selected[room] = list(best_combo or ())
    return selected


def assign_scene_split(
    rows: list[dict[str, Any]],
    protocol: ProtocolConfig,
) -> tuple[dict[str, Any], set[int], set[str]]:
    if protocol.scene_mode == "room_type":
        assert protocol.unseen_room is not None
        unseen_rooms = {protocol.unseen_room}
        unseen_scene_nums = {
            int(row["_meta"]["scene_num"])
            for row in rows
            if row["_meta"]["room_type"] in unseen_rooms
        }
        scene_split = {
            "mode": "room_type",
            "unseen_room_types": sorted(unseen_rooms),
            "seen_room_types": [r for r in ROOM_ORDER if r not in unseen_rooms],
            "unseen_scene_nums": sorted(unseen_scene_nums),
        }
        return scene_split, unseen_scene_nums, unseen_rooms

    if protocol.scene_mode == "scene_num":
        selected = choose_scene_nums_for_balanced_protocol(
            rows, protocol.unseen_scene_fraction
        )
        unseen_scene_nums = {scene for scenes in selected.values() for scene in scenes}
        scene_split = {
            "mode": "scene_num",
            "unseen_scene_fraction": protocol.unseen_scene_fraction,
            "unseen_scene_nums_by_room": {
                room: selected[room] for room in ROOM_ORDER
            },
            "unseen_scene_nums": sorted(unseen_scene_nums),
            "seen_scene_nums": sorted(
                {
                    int(row["_meta"]["scene_num"])
                    for row in rows
                    if int(row["_meta"]["scene_num"]) not in unseen_scene_nums
                }
            ),
        }
        return scene_split, unseen_scene_nums, set()

    raise ValueError(f"Unknown scene split mode: {protocol.scene_mode}")


def tag_protocol_rows(
    rows: list[dict[str, Any]],
    protocol: ProtocolConfig,
    unseen_scene_nums: set[int],
) -> list[dict[str, Any]]:
    tagged = []
    for row in rows:
        out = copy.deepcopy(row)
        meta = out["_meta"]
        task = out["task"]
        scene_num = int(meta["scene_num"])
        task_split = "unseen" if task in protocol.unseen_tasks else "seen"
        scene_split = "unseen" if scene_num in unseen_scene_nums else "seen"
        meta.update(
            {
                "protocol": protocol.name,
                "task_split": task_split,
                "scene_split": scene_split,
                "zero_shot_train_allowed": task_split == "seen"
                and scene_split == "seen",
            }
        )
        tagged.append(out)
    return tagged


def stratified_three_way_split(
    rows: list[dict[str, Any]],
    seed: int,
    monitor_fraction: float,
    eval_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["task"], row["_meta"]["room_type"])].append(row)

    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    monitor: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for key in sorted(by_key):
        group = sorted(by_key[key], key=row_key)
        rng.shuffle(group)
        n = len(group)
        n_eval = round(n * eval_fraction)
        n_monitor = round(n * monitor_fraction)
        if n >= 20:
            n_eval = max(1, n_eval)
            n_monitor = max(1, n_monitor)
        if n_eval + n_monitor >= n:
            n_eval = max(0, min(n_eval, n - 1))
            n_monitor = max(0, min(n_monitor, n - n_eval - 1))
        eval_rows.extend(group[:n_eval])
        monitor.extend(group[n_eval : n_eval + n_monitor])
        train.extend(group[n_eval + n_monitor :])

    return (
        sorted(train, key=row_key),
        sorted(monitor, key=row_key),
        sorted(eval_rows, key=row_key),
    )


def materialize_split_rows(rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = copy.deepcopy(row)
        item["_meta"]["benchmark_split"] = split_name
        out.append(item)
    return out


def build_splits(
    rows: list[dict[str, Any]],
    protocol: ProtocolConfig,
    seed: int,
    monitor_fraction: float,
    eval_seen_seen_fraction: float,
) -> dict[str, list[dict[str, Any]]]:
    seen_seen = [
        r
        for r in rows
        if r["_meta"]["task_split"] == "seen" and r["_meta"]["scene_split"] == "seen"
    ]
    train, monitor, col1 = stratified_three_way_split(
        seen_seen,
        seed=seed,
        monitor_fraction=monitor_fraction,
        eval_fraction=eval_seen_seen_fraction,
    )

    col2 = [
        r
        for r in rows
        if r["_meta"]["task_split"] == "seen" and r["_meta"]["scene_split"] == "unseen"
    ]
    col3 = [
        r
        for r in rows
        if r["_meta"]["task_split"] == "unseen" and r["_meta"]["scene_split"] == "unseen"
    ]
    unseen_task_seen_scene = [
        r
        for r in rows
        if r["_meta"]["task_split"] == "unseen" and r["_meta"]["scene_split"] == "seen"
    ]

    return {
        "train": materialize_split_rows(train, "train"),
        "monitor": materialize_split_rows(monitor, "monitor"),
        "eval_col_1_seen_seen": materialize_split_rows(col1, "eval_col_1_seen_seen"),
        "eval_col_2_seen_unseen": materialize_split_rows(col2, "eval_col_2_seen_unseen"),
        "eval_col_3_unseen_unseen": materialize_split_rows(col3, "eval_col_3_unseen_unseen"),
        "unused_unseen_task_seen_scene": materialize_split_rows(
            unseen_task_seen_scene, "unused_unseen_task_seen_scene"
        ),
    }


def assign_world_clusters(
    train_rows: list[dict[str, Any]],
    monitor_rows: list[dict[str, Any]],
    num_clusters: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    rows_by_scene: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        rows_by_scene[int(row["_meta"]["scene_num"])].append(row)

    clusters = {
        f"cluster_{idx:02d}": {
            "scene_nums": [],
            "monitor_only_scene_nums": [],
            "train_episodes": 0,
            "monitor_episodes": 0,
            "rooms": [],
            "tasks": [],
        }
        for idx in range(num_clusters)
    }
    sorted_scenes = sorted(
        rows_by_scene,
        key=lambda s: (-len(rows_by_scene[s]), room_for_scene(s), s),
    )
    scene_to_cluster: dict[int, str] = {}
    for scene in sorted_scenes:
        target_name = min(
            clusters,
            key=lambda name: (
                int(clusters[name]["train_episodes"]),
                len(clusters[name]["scene_nums"]),
                name,
            ),
        )
        clusters[target_name]["scene_nums"].append(scene)
        clusters[target_name]["train_episodes"] += len(rows_by_scene[scene])
        scene_to_cluster[scene] = target_name

    cluster_train: dict[str, list[dict[str, Any]]] = {name: [] for name in clusters}
    cluster_monitor: dict[str, list[dict[str, Any]]] = {name: [] for name in clusters}
    for scene, rows in rows_by_scene.items():
        cluster_train[scene_to_cluster[scene]].extend(rows)

    # Monitor rows are split separately from train. In small scene/task buckets,
    # a scene can contribute monitor rows but no train rows. Keep those rows in
    # a cluster test view so every method view covers the same monitor split.
    for row in monitor_rows:
        scene = int(row["_meta"]["scene_num"])
        target_name = scene_to_cluster.get(scene)
        if target_name is None:
            target_name = min(
                clusters,
                key=lambda name: (len(cluster_monitor[name]), name),
            )
            if scene not in clusters[target_name]["monitor_only_scene_nums"]:
                clusters[target_name]["monitor_only_scene_nums"].append(scene)
        cluster_monitor[target_name].append(row)

    for name, info in clusters.items():
        train = sorted(cluster_train[name], key=row_key)
        monitor = sorted(cluster_monitor[name], key=row_key)
        info["scene_nums"] = sorted(info["scene_nums"])
        info["monitor_only_scene_nums"] = sorted(info["monitor_only_scene_nums"])
        info["train_episodes"] = len(train)
        info["monitor_episodes"] = len(monitor)
        info["rooms"] = sorted({r["_meta"]["room_type"] for r in train})
        info["tasks"] = sorted({r["task"] for r in train})
        cluster_train[name] = train
        cluster_monitor[name] = monitor
    return clusters, cluster_train, cluster_monitor


def write_method_views(
    out_dir: Path,
    splits: dict[str, list[dict[str, Any]]],
    num_world_clusters: int,
) -> dict[str, Any]:
    views_dir = out_dir / "views"
    manifest: dict[str, Any] = {}

    train = splits["train"]
    monitor = splits["monitor"]

    for view_name in ["llm_ft", "wormi/adapter"]:
        view_dir = views_dir / view_name
        write_jsonl(view_dir / "train.jsonl", train)
        write_jsonl(view_dir / "test.jsonl", monitor)
        manifest[view_name] = {
            "train": len(train),
            "test": len(monitor),
            "description": "Same zero-shot train/monitor pool shared by methods.",
        }

    retrieval_path = views_dir / "planner_retrieval" / "index.jsonl"
    write_jsonl(retrieval_path, train)
    manifest["planner_retrieval"] = {
        "index": len(train),
        "description": "Retrieval index is exactly the zero-shot train pool.",
    }

    clusters, cluster_train, cluster_monitor = assign_world_clusters(
        train, monitor, num_world_clusters
    )
    world_root = views_dir / "wormi" / "world_model"
    for name in sorted(clusters):
        cluster_dir = world_root / name
        write_jsonl(cluster_dir / "train.jsonl", cluster_train[name])
        write_jsonl(cluster_dir / "test.jsonl", cluster_monitor[name])
    write_json(world_root / "world_model_clusters.json", clusters)
    manifest["wormi/world_model"] = {
        "num_clusters": num_world_clusters,
        "clusters": clusters,
        "description": (
            "Scene-domain clusters over the same zero-shot train pool; no "
            "unseen-task or unseen-scene eval episodes are added."
        ),
    }

    eval_manifest = {}
    for split_name in [
        "eval_col_1_seen_seen",
        "eval_col_2_seen_unseen",
        "eval_col_3_unseen_unseen",
    ]:
        eval_dir = views_dir / "eval" / split_name
        write_jsonl(eval_dir / "test.jsonl", splits[split_name])
        eval_manifest[split_name] = len(splits[split_name])
    manifest["eval"] = eval_manifest
    return manifest


def write_compat_dirs(out_dir: Path, splits: dict[str, list[dict[str, Any]]]) -> None:
    """Write shallow dirs compatible with existing eval loaders."""
    for split_name in [
        "eval_col_1_seen_seen",
        "eval_col_2_seen_unseen",
        "eval_col_3_unseen_unseen",
    ]:
        target = out_dir / "views" / "eval" / split_name / "test.jsonl"
        compat = out_dir / split_name / "test.jsonl"
        symlink_or_copy(target, compat)


def build_distribution_report(
    all_rows: list[dict[str, Any]],
    splits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "source_total_episodes": len(all_rows),
        "source_task_room_matrix": task_room_matrix(all_rows),
        "splits": {name: split_counts(rows) for name, rows in sorted(splits.items())},
    }


def build_leakage_report(
    protocol: ProtocolConfig,
    splits: dict[str, list[dict[str, Any]]],
    method_manifest: dict[str, Any],
) -> dict[str, Any]:
    train_ids = {row_key(r) for r in splits["train"]}
    monitor_ids = {row_key(r) for r in splits["monitor"]}
    eval_ids = {
        row_key(r)
        for name in [
            "eval_col_1_seen_seen",
            "eval_col_2_seen_unseen",
            "eval_col_3_unseen_unseen",
        ]
        for r in splits[name]
    }
    train_trials = {r["trial_name"] for r in splits["train"]}
    eval_trials = {
        r["trial_name"]
        for name in [
            "eval_col_1_seen_seen",
            "eval_col_2_seen_unseen",
            "eval_col_3_unseen_unseen",
        ]
        for r in splits[name]
    }
    train_scene_nums = {int(r["_meta"]["scene_num"]) for r in splits["train"]}
    unseen_scene_nums = {
        int(r["_meta"]["scene_num"])
        for name in ["eval_col_2_seen_unseen", "eval_col_3_unseen_unseen"]
        for r in splits[name]
    }
    zero_shot_unseen_task_rows = [
        r for r in splits["train"] if r["task"] in protocol.unseen_tasks
    ]
    cluster_ids = set()
    clusters = method_manifest["wormi/world_model"]["clusters"]
    for name in clusters:
        # Filled below by checking files would be redundant; cluster metadata is
        # derived from the same train split.
        pass
    return {
        "train_eval_episode_overlap": len(train_ids & eval_ids),
        "monitor_eval_episode_overlap": len(monitor_ids & eval_ids),
        "train_eval_trial_overlap": len(train_trials & eval_trials),
        "train_unseen_scene_overlap_for_col2_col3": len(
            train_scene_nums & unseen_scene_nums
        ),
        "zero_shot_train_unseen_task_rows": len(zero_shot_unseen_task_rows),
        "zero_shot_train_unseen_tasks": dict(
            Counter(r["task"] for r in zero_shot_unseen_task_rows)
        ),
        "method_train_episode_counts": {
            "llm_ft": method_manifest["llm_ft"]["train"],
            "planner_retrieval": method_manifest["planner_retrieval"]["index"],
            "wormi_adapter": method_manifest["wormi/adapter"]["train"],
            "wormi_world_model_union": sum(
                int(c["train_episodes"]) for c in clusters.values()
            ),
        },
    }


def validate_release(
    protocol: ProtocolConfig,
    rows: list[dict[str, Any]],
    splits: dict[str, list[dict[str, Any]]],
    method_manifest: dict[str, Any],
    leakage: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if len({r["trial_name"] for r in rows}) != len(rows):
        errors.append("canonical episodes contain duplicate trial_name values")
    if not all(r.get("history") for r in rows):
        errors.append("some canonical episodes have empty history")
    if not all(r["history"][-1].get("dones") for r in rows):
        errors.append("some expert episodes do not end with dones=True")

    for name in ["train", "monitor", "eval_col_1_seen_seen", "eval_col_2_seen_unseen", "eval_col_3_unseen_unseen"]:
        if not splits[name]:
            errors.append(f"{name} is empty")

    if leakage["train_eval_episode_overlap"] != 0:
        errors.append("train/eval episode overlap is non-zero")
    if leakage["monitor_eval_episode_overlap"] != 0:
        errors.append("monitor/eval episode overlap is non-zero")
    if leakage["train_eval_trial_overlap"] != 0:
        errors.append("train/eval trial overlap is non-zero")
    if leakage["train_unseen_scene_overlap_for_col2_col3"] != 0:
        errors.append("train shares scene_num with unseen-scene eval columns")
    if leakage["zero_shot_train_unseen_task_rows"] != 0:
        errors.append("zero-shot train contains unseen-task rows")

    col2_tasks = Counter(r["task"] for r in splits["eval_col_2_seen_unseen"])
    col3_tasks = Counter(r["task"] for r in splits["eval_col_3_unseen_unseen"])
    missing_col3 = sorted(task for task in protocol.unseen_tasks if col3_tasks[task] == 0)
    if missing_col3:
        errors.append(f"col_3 is missing unseen task(s): {missing_col3}")
    if len(col2_tasks) < 2:
        warnings.append(
            "col_2 has fewer than two seen task types; scene generalization may be narrow"
        )

    method_counts = leakage["method_train_episode_counts"]
    train_count = len(splits["train"])
    for method, count in method_counts.items():
        if count != train_count:
            errors.append(
                f"{method} sees {count} train episodes, expected {train_count}"
            )

    clusters = method_manifest["wormi/world_model"]["clusters"]
    for name, info in clusters.items():
        if int(info["train_episodes"]) == 0:
            errors.append(f"{name} has no train episodes")
        if int(info["monitor_episodes"]) == 0:
            warnings.append(f"{name} has no monitor episodes")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "split_episode_counts": {
            name: len(rows_) for name, rows_ in sorted(splits.items())
        },
        "col2_tasks": dict(sorted(col2_tasks.items())),
        "col3_tasks": dict(sorted(col3_tasks.items())),
    }


def write_episode_hashes(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in sorted(rows, key=row_key):
            payload = json.dumps(row, sort_keys=True)
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            f.write(f"{digest}  {row['_meta']['episode_id']}  {row['trial_name']}\n")


def release_readme(protocol: ProtocolConfig, validation: dict[str, Any]) -> str:
    rationale = "\n".join(f"- {item}" for item in protocol.rationale)
    counts = validation["split_episode_counts"]
    return f"""# {protocol.title}

This ALFWorld release was generated by `tools/build_alfworld_dataset.py`.

## Why This Protocol Exists

{rationale}

## Task Split

Seen tasks:

{chr(10).join(f'- `{task}`' for task in sorted(protocol.seen_tasks))}

Unseen tasks:

{chr(10).join(f'- `{task}`' for task in sorted(protocol.unseen_tasks))}

## Split Counts

| Split | Episodes |
|---|---:|
| train | {counts.get('train', 0)} |
| monitor | {counts.get('monitor', 0)} |
| eval_col_1_seen_seen | {counts.get('eval_col_1_seen_seen', 0)} |
| eval_col_2_seen_unseen | {counts.get('eval_col_2_seen_unseen', 0)} |
| eval_col_3_unseen_unseen | {counts.get('eval_col_3_unseen_unseen', 0)} |
| unused_unseen_task_seen_scene | {counts.get('unused_unseen_task_seen_scene', 0)} |

## Method Views

- `views/llm_ft`: fine-tuning baseline view.
- `views/planner_retrieval/index.jsonl`: retrieval/planner index.
- `views/wormi/adapter`: WorMI adapter training view.
- `views/wormi/world_model/cluster_*/`: six scene-domain world-model views.
- `views/eval/eval_col_*`: shared evaluation episodes for every method.

The union of train episodes visible to each method is validated to match
`splits/train.jsonl`.

## Validation

`validation_report.json` contains machine-checkable validity results.
This release is currently marked `valid={validation['valid']}`.
"""


def build_protocol_release(
    rows: list[dict[str, Any]],
    protocol: ProtocolConfig,
    output_root: Path,
    seed: int,
    monitor_fraction: float,
    eval_seen_seen_fraction: float,
    num_world_clusters: int,
) -> dict[str, Any]:
    out_dir = output_root / protocol.name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_split, unseen_scene_nums, _unseen_rooms = assign_scene_split(rows, protocol)
    tagged = tag_protocol_rows(rows, protocol, unseen_scene_nums)
    splits = build_splits(
        tagged,
        protocol,
        seed=seed,
        monitor_fraction=monitor_fraction,
        eval_seen_seen_fraction=eval_seen_seen_fraction,
    )

    write_jsonl(out_dir / "canonical" / "episodes.jsonl", tagged)
    for name, split_rows in splits.items():
        write_jsonl(out_dir / "splits" / f"{name}.jsonl", split_rows)

    method_manifest = write_method_views(out_dir, splits, num_world_clusters)
    write_compat_dirs(out_dir, splits)

    task_split = {
        "name": protocol.task_split_name,
        "seen_tasks": sorted(protocol.seen_tasks),
        "unseen_tasks": sorted(protocol.unseen_tasks),
    }
    distribution = build_distribution_report(tagged, splits)
    leakage = build_leakage_report(protocol, splits, method_manifest)
    validation = validate_release(protocol, tagged, splits, method_manifest, leakage)

    manifest = {
        "protocol": protocol.name,
        "title": protocol.title,
        "seed": seed,
        "source_total_episodes": len(rows),
        "monitor_fraction": monitor_fraction,
        "eval_seen_seen_fraction": eval_seen_seen_fraction,
        "num_world_clusters": num_world_clusters,
        "rationale": protocol.rationale,
        "files": {
            "canonical": "canonical/episodes.jsonl",
            "splits": "splits/*.jsonl",
            "views": "views/",
        },
    }

    write_json(out_dir / "dataset_manifest.json", manifest)
    write_json(out_dir / "task_split.json", task_split)
    write_json(out_dir / "scene_split.json", scene_split)
    write_json(out_dir / "method_views_manifest.json", method_manifest)
    write_json(out_dir / "distribution_report.json", distribution)
    write_json(out_dir / "leakage_report.json", leakage)
    write_json(out_dir / "validation_report.json", validation)
    write_episode_hashes(out_dir / "episode_hashes.sha256", tagged)
    (out_dir / "README.md").write_text(release_readme(protocol, validation))
    return validation


def write_combined_design_report(
    output_root: Path,
    report_path: Path,
    validations: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# ALFWorld Data Processing Protocols",
        "",
        "Date: 2026-05-27",
        "",
        "This report records the two ALFWorld processing protocols generated by "
        "`tools/build_alfworld_dataset.py`, the reasoning behind them, and the "
        "validation status of the generated releases.",
        "",
        "## Shared Principles",
        "",
        "- Start from official ALFWorld textual-env expert episodes.",
        "- Build canonical episode files first, then derive method-specific views.",
        "- Keep every method's zero-shot training union equal to `splits/train.jsonl`.",
        "- Exclude unseen-task episodes from all zero-shot training views.",
        "- Evaluate all methods on the same `eval_col_*` episode files.",
        "- Use scene-domain WorMI world-model clusters instead of task-type experts, "
        "so unseen tasks do not leak into world-model training.",
        "",
        "## Generated Releases",
        "",
    ]
    for protocol in [PAPER_COMPATIBLE, BALANCED_SCENE_INSTANCE]:
        validation = validations[protocol.name]
        lines.extend(
            [
                f"### `{protocol.name}`",
                "",
                protocol.title,
                "",
                "Rationale:",
                "",
                *[f"- {item}" for item in protocol.rationale],
                "",
                "Validation summary:",
                "",
                f"- valid: `{validation['valid']}`",
                f"- errors: `{len(validation['errors'])}`",
                f"- warnings: `{len(validation['warnings'])}`",
                f"- split counts: `{validation['split_episode_counts']}`",
                f"- col2 tasks: `{validation['col2_tasks']}`",
                f"- col3 tasks: `{validation['col3_tasks']}`",
                f"- output: `{output_root / protocol.name}`",
                "",
            ]
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source-root",
        type=Path,
        default=Path("share/alfworld-initial-2026-05-22"),
        help="Episode-level ALFWorld JSONL source root.",
    )
    p.add_argument(
        "--tw-pddl-zip",
        type=Path,
        default=Path("/root/autodl-tmp/wormi-data/alfworld-data/json_2.1.2_tw-pddl.zip"),
        help="Official json_2.1.2_tw-pddl zip used to recover scene_num metadata.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-tmp/wormi-data/alfworld-protocols"),
    )
    p.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/alfworld-data-processing-protocols-2026-05-27.md"),
    )
    p.add_argument(
        "--protocol",
        choices=sorted(PROTOCOLS),
        action="append",
        help="Protocol(s) to build. Defaults to both.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--monitor-fraction", type=float, default=0.1)
    p.add_argument("--eval-seen-seen-fraction", type=float, default=0.1)
    p.add_argument("--num-world-clusters", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    protocols = [PROTOCOLS[name] for name in (args.protocol or sorted(PROTOCOLS))]
    if not args.source_root.is_dir():
        raise FileNotFoundError(args.source_root)
    if not args.tw_pddl_zip.exists():
        raise FileNotFoundError(args.tw_pddl_zip)

    rows = load_source_episodes(args.source_root, args.tw_pddl_zip)
    validations: dict[str, dict[str, Any]] = {}
    args.output_root.mkdir(parents=True, exist_ok=True)
    for protocol in protocols:
        print(f"Building {protocol.name} -> {args.output_root / protocol.name}")
        validations[protocol.name] = build_protocol_release(
            rows=rows,
            protocol=protocol,
            output_root=args.output_root,
            seed=args.seed,
            monitor_fraction=args.monitor_fraction,
            eval_seen_seen_fraction=args.eval_seen_seen_fraction,
            num_world_clusters=args.num_world_clusters,
        )
        status = "OK" if validations[protocol.name]["valid"] else "INVALID"
        print(f"  validation: {status}")

    if set(validations) == set(PROTOCOLS):
        write_combined_design_report(args.output_root, args.report_path, validations)
        print(f"Design report written to {args.report_path}")

    invalid = [name for name, v in validations.items() if not v["valid"]]
    if invalid:
        raise SystemExit(f"Invalid protocol release(s): {', '.join(invalid)}")


if __name__ == "__main__":
    main()
