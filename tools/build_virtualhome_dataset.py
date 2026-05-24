"""Build the VirtualHome split of WorMI's training data, aligned with paper §4 / §B.1.

Per the paper (arxiv 2509.03956):
- 78 atomic instructions across 4 task families: TurnOn (9), Open (7),
  PutOn (30), PlaceIn (32). 16 seen + 62 unseen.
- 20 distinct scenes (Section 4): 6 seen + 14 unseen.
- N=6 world models (Table A.6) → one per seen scene. Each world model is
  trained on (16 seen task × that one seen scene) trajectories.
- Observations rendered as a list of `(subject, relation, object)` triples.
- 6 action verbs: walk / grab / open / switchon / put / putin.

This script does NOT run any graphical simulator. It uses VirtualHome's
EvolvingGraph (pure Python) loaded directly from a cloned source tree, and
each scene's per-program init_graph as the world bootstrap.

Each successful trajectory produces N jsonl rows of shape:
    {instruction, observation, action, next_observation, _meta}

Output layout (paper-aligned):
    <output_dir>/scene_{0..5}/train.jsonl                  ← stage 1 (seen task × that scene)
    <output_dir>/scene_{0..5}/test.jsonl  → symlink to     ← stage 1 eval signal
        ../test_seen_task_seen_scene.jsonl
    <output_dir>/test_seen_task_seen_scene.jsonl           ← Table 1 col 1 (seen × seen)
    <output_dir>/test_seen_task_unseen_scene.jsonl         ← Table 1 col 2 (seen × unseen scene)
    <output_dir>/test_unseen_task_unseen_scene.jsonl       ← Table 1 col 3 (unseen × unseen scene)
    <output_dir>/test_unseen_task_seen_scene.jsonl         ← not in Table 1, kept for completeness
    <output_dir>/eval_col_{1,2,3}_*/test.jsonl             ← symlinks for curricula

`_meta` carries {scene, split, task_args} so downstream resplit/eval tools
can re-bucket without re-running the builder.
"""

import argparse
import copy
import importlib.util
import json
import random
import re
import sys
import types
import zipfile
from collections import Counter
from collections import defaultdict
from pathlib import Path


def _bootstrap_evolving_graph(vh_src: Path) -> dict:
    """Import virtualhome.simulation.evolving_graph submodules without triggering
    the package's `__init__` chain, which pulls in cv2/ipdb/THOR clients we don't
    need. Returns the loaded modules.
    """
    eg_root = vh_src / "virtualhome" / "simulation" / "evolving_graph"
    sim_root = vh_src / "virtualhome" / "simulation"
    sim_pkg = types.ModuleType("simulation")
    sim_pkg.__path__ = [str(sim_root)]
    sys.modules["simulation"] = sim_pkg
    eg_pkg = types.ModuleType("simulation.evolving_graph")
    eg_pkg.__path__ = [str(eg_root)]
    sys.modules["simulation.evolving_graph"] = eg_pkg

    def _load(name: str):
        spec = importlib.util.spec_from_file_location(
            f"simulation.evolving_graph.{name}", eg_root / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"simulation.evolving_graph.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("common")
    _load("utils")
    return {
        "environment": _load("environment"),
        "scripts": _load("scripts"),
        "execution": _load("execution"),
    }


# Paper Table A.2 — number of instructions per family
TARGETS = {"turnon": 9, "open": 7, "puton": 30, "placein": 32}

# The original VirtualHome resource table marks `floor` as a surface. It is a
# structural support, not a useful target receptacle for this reconstruction:
# class-level observations either omit floor nodes or make the task trivially
# broad. Exclude it from generated `puton` targets.
_EXCLUDED_SURFACE_TARGETS = {"floor"}

# These classes either correspond to simulator structure or are filtered from
# observations. Selecting them as goals creates tasks whose final fact cannot
# be represented in the jsonl text.
_EXCLUDED_GOAL_CLASSES = {"door"}

# State triples needed for VH goals and common action preconditions. Do not
# emit every VH state (for example CLEAN/DIRTY), because that bloats prompts
# with task-irrelevant noise.
_OBSERVABLE_STATES = {
    "OPEN": "open",
    "CLOSED": "closed",
    "ON": "on",
    "OFF": "off",
    "PLUGGED_IN": "plugged_in",
    "PLUGGED_OUT": "plugged_out",
}

# Paper Section 4: 6 seen scenes out of 20. We pick one variant per base
# apartment (TrimmedTestScene1..6), so the seen pool spans 6 distinct
# apartments with maximally different layouts. The remaining 14 variants
# (Scene7's two + v1/v2 of Scene1..6) form the unseen scenes evaluated in
# Table 1 col 2/3.
#
# Scene6_v0 is excluded: its init_graph has some invalid initial state
# (every script raises execution_failed, regardless of object presence).
# v1 is the canonical working substitute. Diagnose: empty Scene6_v0
# breakdown in the first build run vs. v1/v2 each producing ~230 rows.
PAPER_SEEN_SCENE_KEYS = [
    "TrimmedTestScene1_graph__v0",
    "TrimmedTestScene2_graph__v0",
    "TrimmedTestScene3_graph__v0",
    "TrimmedTestScene4_graph__v0",
    "TrimmedTestScene5_graph__v0",
    "TrimmedTestScene6_graph__v1",
]


def select_classes_with_property(
    properties: dict[str, list[str]],
    prop: str,
    scene_class_sets: list[set[str]],
) -> list[str]:
    """Return classes with `prop`, sorted by scene-coverage descending.

    Coverage = number of scenes whose init graph contains this class. Sorting
    by coverage means later `rng.sample`-style picks favor classes that exist
    in many scenes — keeping (instruction, scene) success rate high while
    still letting `build_instructions` hit the paper's per-family targets
    (Table A.2: 9/7/30/32 = 78 total).

    Tied coverage breaks alphabetically for determinism.
    """
    candidates = []
    union = set().union(*scene_class_sets) if scene_class_sets else set()
    for c, ps in properties.items():
        if prop not in ps or c not in union:
            continue
        coverage = sum(1 for s in scene_class_sets if c in s)
        candidates.append((-coverage, c))
    candidates.sort()
    return [c for _, c in candidates]


def find_first_id(graph: dict, class_name: str) -> int | None:
    for n in graph["nodes"]:
        if n["class_name"] == class_name:
            return n["id"]
    return None


def build_atomic_program(
    family: str, args: tuple[str, ...], graph: dict
) -> str | None:
    """Return a comma-separated VirtualHome program for the atomic instruction,
    or None if the required objects are missing in this scene.
    """
    ids = []
    for cls in args:
        nid = find_first_id(graph, cls)
        if nid is None:
            return None
        ids.append(nid)

    def tok(cls: str, nid: int) -> str:
        return f"<{cls}> ({nid})"

    if family == "turnon":
        a = args[0]; ai = ids[0]
        return f"[WALK] {tok(a, ai)}, [SWITCHON] {tok(a, ai)}"
    if family == "open":
        a = args[0]; ai = ids[0]
        return f"[WALK] {tok(a, ai)}, [OPEN] {tok(a, ai)}"
    if family == "puton":
        a, b = args; ai, bi = ids
        return (
            f"[WALK] {tok(a, ai)}, [GRAB] {tok(a, ai)}, "
            f"[WALK] {tok(b, bi)}, [PUTBACK] {tok(a, ai)} {tok(b, bi)}"
        )
    if family == "placein":
        a, b = args; ai, bi = ids
        return (
            f"[WALK] {tok(a, ai)}, [GRAB] {tok(a, ai)}, "
            f"[WALK] {tok(b, bi)}, [OPEN] {tok(b, bi)}, "
            f"[PUTIN] {tok(a, ai)} {tok(b, bi)}"
        )
    raise ValueError(family)


def normalize_action(action_line: str) -> str:
    """Map [VERB] <a>(id) <b>(id) → 'verb a b' (lowercase, no instance ids)."""
    import re

    m = re.match(r"^\s*\[([A-Z_]+)\]\s*(.*?)\s*$", action_line)
    if not m:
        return action_line.strip()
    verb_raw = m.group(1).upper()
    args = re.findall(r"<([^>]+)>", m.group(2))
    verb_map = {
        "WALK": "walk",
        "GRAB": "grab",
        "OPEN": "open",
        "SWITCHON": "switchon",
        "PUTBACK": "put",
        "PUTIN": "putin",
    }
    verb = verb_map.get(verb_raw, verb_raw.lower())
    if not args:
        return verb
    return f"{verb} {' '.join(args)}"


def instruction_text(family: str, args: tuple[str, ...]) -> str:
    if family == "turnon":
        return f"Turn on {args[0].replace('_', ' ')}"
    if family == "open":
        return f"Open {args[0].replace('_', ' ')}"
    if family == "puton":
        return f"Put {args[0].replace('_', ' ')} on {args[1].replace('_', ' ')}"
    if family == "placein":
        return f"Place {args[0].replace('_', ' ')} in {args[1].replace('_', ' ')}"
    raise ValueError(family)


_NON_OBJECT_CATEGORIES = {"Floor", "Walls", "Ceiling", "Doors"}

# VH's resources/class_name_equivalence.json declares dining_room -> kitchen
# and home_office -> livingroom. The 7 default TrimmedTestScene graphs use the
# raw names; paper Figure A.2 shows the canonical names. Apply the alias when
# emitting a Room into observations so the room set lines up with the paper's
# `{livingroom, bathroom, kitchen, bedroom}`.
_ROOM_CANONICAL = {
    "dining_room": "kitchen",
    "home_office": "livingroom",
    "living_room": "livingroom",
}


def _canon_room(name: str) -> str:
    return _ROOM_CANONICAL.get(name, name)


def format_observation(graph: dict) -> str:
    """Render the graph as class-level `(subj, rel, obj)` triples.

    The text must expose the facts needed by the four atomic VH goals. In
    addition to room relations from Figure A.2, keep object-container `inside`
    facts and compact state triples for OPEN/CLOSED/ON/OFF/plugged states.
    Every triple is class-level (no instance ids), and duplicates collapse.
    """
    nodes_by_id = {n["id"]: n for n in graph["nodes"]}
    agent_id = next(
        (n["id"] for n in graph["nodes"] if n.get("category") == "Characters"), None
    )
    triples: set[tuple[str, str, str]] = set()
    seen_holds = False

    def name(nid: int) -> str | None:
        n = nodes_by_id.get(nid)
        if n is None or n.get("category") in _NON_OBJECT_CATEGORIES:
            return None
        return n["class_name"]

    for node in graph["nodes"]:
        node_name = name(node["id"])
        if node_name is None:
            continue
        for state in sorted(set(node.get("states", []))):
            if state in _OBSERVABLE_STATES:
                triples.add((node_name, "is", _OBSERVABLE_STATES[state]))

    for e in graph["edges"]:
        rel_native = e["relation_type"]
        sub = nodes_by_id.get(e["from_id"])
        obj = nodes_by_id.get(e["to_id"])
        if sub is None or obj is None:
            continue
        sub_name, obj_name = name(sub["id"]), name(obj["id"])
        if sub_name is None or obj_name is None:
            continue

        if rel_native == "INSIDE":
            target_name = _canon_room(obj_name) if obj.get("category") == "Rooms" else obj_name
            triples.add((sub_name, "inside", target_name))
        elif rel_native == "ON":
            triples.add((sub_name, "on", obj_name))
        elif (
            rel_native == "CLOSE"
            and agent_id is not None
            and sub["id"] == agent_id
        ):
            triples.add((sub_name, "close", obj_name))
        elif rel_native in ("HOLDS_RH", "HOLDS_LH") and sub["id"] == agent_id:
            triples.add((sub_name, "hold", obj_name))
            seen_holds = True
        elif (
            rel_native == "BETWEEN"
            and sub.get("category") == "Rooms"
            and obj.get("category") == "Rooms"
        ):
            triples.add((_canon_room(sub_name), "adjacent", _canon_room(obj_name)))

    if agent_id is not None and not seen_holds:
        triples.add(("character", "hold", "none"))

    return ", ".join(f"({s}, {r}, {o})" for s, r, o in sorted(triples))


def find_scene_init_graph(scene_dir: Path) -> dict | None:
    """Pick any program's init_graph from this scene as the bootstrap world state."""
    for source_dir in sorted(scene_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        for jf in sorted(source_dir.glob("*.json")):
            try:
                d = json.loads(jf.read_text())
            except Exception:
                continue
            if "init_graph" in d:
                return d["init_graph"]
            if "graph_state_list" in d and d["graph_state_list"]:
                return d["graph_state_list"][0]
    return None


def _parse_triples(observation: str) -> set[tuple[str, str, str]]:
    return {
        (subj.strip(), rel.strip(), obj.strip())
        for subj, rel, obj in re.findall(
            r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)", observation
        )
    }


def _goal_triple(family: str, args: tuple[str, ...]) -> tuple[str, str, str]:
    if family == "turnon":
        return (args[0], "is", "on")
    if family == "open":
        return (args[0], "is", "open")
    if family == "puton":
        return (args[0], "on", args[1])
    if family == "placein":
        return (args[0], "inside", args[1])
    raise ValueError(family)


def _node_has_state(graph: dict, class_name: str, state: str) -> bool:
    state = state.upper()
    return any(
        node["class_name"] == class_name and state in set(node.get("states", []))
        for node in graph["nodes"]
    )


def _has_relation(
    graph: dict, source_class: str, relation: str, target_class: str
) -> bool:
    relation = relation.upper()
    source_ids = {
        int(node["id"]) for node in graph["nodes"] if node["class_name"] == source_class
    }
    target_ids = {
        int(node["id"]) for node in graph["nodes"] if node["class_name"] == target_class
    }
    return any(
        int(edge["from_id"]) in source_ids
        and int(edge["to_id"]) in target_ids
        and edge["relation_type"] == relation
        for edge in graph["edges"]
    )


def _goal_satisfied(graph: dict, family: str, args: tuple[str, ...]) -> bool:
    if family == "turnon":
        return _node_has_state(graph, args[0], "ON")
    if family == "open":
        return _node_has_state(graph, args[0], "OPEN")
    if family == "puton":
        return _has_relation(graph, args[0], "ON", args[1])
    if family == "placein":
        return _has_relation(graph, args[0], "INSIDE", args[1])
    raise ValueError(family)


def _goal_visible(observation: str, family: str, args: tuple[str, ...]) -> bool:
    return _goal_triple(family, args) in _parse_triples(observation)


def _is_semantically_valid_trajectory(
    family: str,
    args: tuple[str, ...],
    graph_state_list: list[dict],
) -> bool:
    initial_observation = format_observation(graph_state_list[0])
    final_observation = format_observation(graph_state_list[-1])
    if not _goal_satisfied(graph_state_list[-1], family, args):
        return False
    if not _goal_visible(final_observation, family, args):
        return False
    # Class-level observations cannot distinguish duplicate instances. If the
    # goal fact is already visible before the action sequence, the example is
    # ambiguous as a transition target, so keep it out of the supervised data.
    return not _goal_visible(initial_observation, family, args)


def _seen_family_quotas(seen_instruction_count: int) -> dict[str, int]:
    total = sum(TARGETS.values())
    quotas = {
        fam: max(1, round(count * seen_instruction_count / total))
        for fam, count in TARGETS.items()
    }
    drift = sum(quotas.values()) - seen_instruction_count
    fams_sorted = sorted(quotas, key=lambda f: -quotas[f])
    while drift > 0:
        for fam in fams_sorted:
            if quotas[fam] > 1 and drift > 0:
                quotas[fam] -= 1
                drift -= 1
    while drift < 0:
        for fam in fams_sorted:
            if drift < 0:
                quotas[fam] += 1
                drift += 1
    return quotas


def build_candidate_instructions(
    properties: dict[str, list[str]],
    scene_class_sets: list[set[str]],
    candidate_multiplier: int = 12,
) -> list[tuple[str, tuple[str, ...]]]:
    """Build a coverage-ranked candidate pool larger than the final 78 tasks."""
    switchable = [
        c for c in select_classes_with_property(properties, "HAS_SWITCH", scene_class_sets)
        if c not in _EXCLUDED_GOAL_CLASSES
    ]
    openable = [
        c for c in select_classes_with_property(properties, "CAN_OPEN", scene_class_sets)
        if c not in _EXCLUDED_GOAL_CLASSES
    ]
    grabbable = select_classes_with_property(properties, "GRABBABLE", scene_class_sets)
    surfaces = [
        c for c in select_classes_with_property(properties, "SURFACES", scene_class_sets)
        if c not in _EXCLUDED_SURFACE_TARGETS and c not in _EXCLUDED_GOAL_CLASSES
    ]
    containers = [
        c for c in select_classes_with_property(properties, "CONTAINERS", scene_class_sets)
        if c not in _EXCLUDED_GOAL_CLASSES
    ]

    def joint_coverage(a: str, b: str) -> int:
        return sum(1 for s in scene_class_sets if a in s and b in s)

    def rank_pairs(grabs: list[str], hosts: list[str]) -> list[tuple[str, str]]:
        pairs = []
        for a in grabs:
            for b in hosts:
                if a == b:
                    continue
                jc = joint_coverage(a, b)
                if jc > 0:
                    pairs.append((-jc, a, b))
        pairs.sort()
        return [(a, b) for _, a, b in pairs]

    on_limit = max(TARGETS["puton"] * candidate_multiplier, TARGETS["puton"] + 20)
    in_limit = max(TARGETS["placein"] * candidate_multiplier, TARGETS["placein"] + 20)
    insts: list[tuple[str, tuple[str, ...]]] = [
        *[("turnon", (c,)) for c in switchable],
        *[("open", (c,)) for c in openable],
        *[("puton", p) for p in rank_pairs(grabbable, surfaces)[:on_limit]],
    ]

    on_pairs = {args for fam, args in insts if fam == "puton"}
    placein_pairs = []
    for pair in rank_pairs(grabbable, containers):
        if pair in on_pairs:
            continue
        placein_pairs.append(pair)
        if len(placein_pairs) >= in_limit:
            break
    insts.extend(("placein", p) for p in placein_pairs)
    return insts


def build_instructions(
    properties: dict[str, list[str]],
    scene_class_sets: list[set[str]],
    rng: random.Random,
) -> list[tuple[str, tuple[str, ...]]]:
    """Sample 78 atomic instructions (9 + 7 + 30 + 32). Single-class families
    take the top-coverage candidates; pair families also rank pairs by joint
    scene coverage so most (instruction, scene) pairs succeed at execution.
    """
    candidates = build_candidate_instructions(properties, scene_class_sets)
    insts = []
    for fam in TARGETS:
        insts.extend([item for item in candidates if item[0] == fam][: TARGETS[fam]])
    rng.shuffle(insts)  # shuffle the final list so seen-task split is not
                        # biased by family ordering
    return insts


def _execute_candidate(
    family: str,
    args: tuple[str, ...],
    scene_name: str,
    init_graph: dict,
    EnvironmentGraph,
    read_script,
    ScriptExecutor,
) -> tuple[list[dict] | None, str | None]:
    program_text = build_atomic_program(family, args, init_graph)
    if program_text is None:
        return None, "missing_object"
    try:
        env_graph = EnvironmentGraph(copy.deepcopy(init_graph))
        script = read_script(program_text)
    except Exception:
        return None, "script_parse_error"

    executor = ScriptExecutor(env_graph, name_equivalence={})
    ok, _final, graph_state_list = executor.execute(script, w_graph_list=True)
    if not ok:
        return None, "execution_failed"
    if not _is_semantically_valid_trajectory(family, args, graph_state_list):
        return None, "semantic_invalid"

    action_lines = [s.strip() for s in program_text.split(", ")]
    if len(graph_state_list) != len(action_lines) + 1:
        return None, "state_action_misaligned"

    instruction = instruction_text(family, args)
    rows = []
    for i, action_line in enumerate(action_lines):
        rows.append(
            {
                "instruction": instruction,
                "observation": format_observation(graph_state_list[i]),
                "action": normalize_action(action_line),
                "next_observation": format_observation(graph_state_list[i + 1]),
                "_meta": {
                    "scene": scene_name,
                    "split": None,
                    "task_args": list(args),
                    "trajectory_id": f"{scene_name}:{family}:{'|'.join(args)}",
                    "step_index": i,
                    "num_steps": len(action_lines),
                },
            }
        )
    return rows, None


def _downsample_manifest_trajectories(
    manifest: dict,
    target_trajectories: int,
    seed: int,
) -> set[str]:
    trajectories = manifest["trajectories"]
    if len(trajectories) <= target_trajectories:
        return {t["trajectory_id"] for t in trajectories}

    protected: set[str] = set()
    seen_task: set[str] = set()
    seen_scene: set[str] = set()
    for traj in trajectories:
        if traj["split"] == "seen_seen":
            protected.add(traj["trajectory_id"])
        task_id = f"{traj['family']}:{'|'.join(traj['args'])}"
        if task_id not in seen_task:
            protected.add(traj["trajectory_id"])
            seen_task.add(task_id)
        if traj["scene"] not in seen_scene:
            protected.add(traj["trajectory_id"])
            seen_scene.add(traj["scene"])

    if len(protected) > target_trajectories:
        raise ValueError(
            f"Cannot downsample to {target_trajectories}; protected trajectory "
            f"count is {len(protected)}"
        )

    rng = random.Random(seed)
    split_priority = {
        "unseen_unseen": 0,
        "unseen_seen": 1,
        "seen_unseen": 2,
        "seen_seen": 3,
    }
    removable = [t for t in trajectories if t["trajectory_id"] not in protected]
    keyed = []
    for traj in removable:
        keyed.append((split_priority.get(traj["split"], 9), rng.random(), traj))
    keyed.sort()

    remove_count = len(trajectories) - target_trajectories
    removed = {traj["trajectory_id"] for _priority, _rand, traj in keyed[:remove_count]}
    kept = {traj["trajectory_id"] for traj in trajectories if traj["trajectory_id"] not in removed}
    manifest["downsample"] = {
        "target_trajectories": target_trajectories,
        "original_trajectories": len(trajectories),
        "kept_trajectories": len(kept),
        "removed_trajectories": len(removed),
        "protected_trajectories": len(protected),
        "strategy": (
            "protect seen_seen, one trajectory per task, one trajectory per scene; "
            "remove from unseen_unseen, then unseen_seen, then seen_unseen"
        ),
    }
    manifest["trajectories"] = [
        traj for traj in trajectories if traj["trajectory_id"] in kept
    ]
    return kept


def _select_seen_seen_eval_ids(
    per_scene_train: dict[int, dict[str, list[dict]]],
    per_task: int,
    seed: int,
) -> set[str]:
    """Pick seen-task/seen-scene held-out trajectories by task, not by 9:1.

    Table 1 column 1 is a domain split over seen tasks and seen scenes. A
    random 10% scene split can leave only a few trajectories and miss task
    families entirely. This selector holds out up to `per_task` trajectories
    for every seen task while greedily spreading them across seen scenes.
    """

    if per_task <= 0:
        return set()

    rng = random.Random(seed)
    by_task: dict[tuple[str, tuple[str, ...]], list[tuple[int, str]]] = defaultdict(list)
    for scene_idx, traj_rows in per_scene_train.items():
        for tid, rows in traj_rows.items():
            meta = rows[0]["_meta"]
            task = (tid.split(":", 2)[1], tuple(meta["task_args"]))
            by_task[task].append((scene_idx, tid))

    selected: set[str] = set()
    scene_counts: Counter = Counter()
    for task in sorted(by_task):
        candidates = by_task[task][:]
        rng.shuffle(candidates)
        for _ in range(min(per_task, len(candidates))):
            candidates.sort(key=lambda item: (scene_counts[item[0]], item[0], item[1]))
            scene_idx, tid = candidates.pop(0)
            selected.add(tid)
            scene_counts[scene_idx] += 1

    return selected


def build(
    raw_dir: Path | None,
    vh_src: Path,
    output_dir: Path,
    seen_scene_count: int = 6,
    seen_instruction_count: int = 16,
    train_ratio: float = 0.9,
    seed: int = 42,
    scene_inits_json: Path | None = None,
    candidate_multiplier: int = 12,
    target_trajectories: int | None = None,
    seen_seen_eval_per_task: int = 2,
) -> None:
    eg = _bootstrap_evolving_graph(vh_src)
    EnvironmentGraph = eg["environment"].EnvironmentGraph
    read_script = eg["scripts"].read_script_from_string
    ScriptExecutor = eg["execution"].ScriptExecutor

    properties = json.loads(
        (vh_src / "virtualhome" / "resources" / "properties_data.json").read_text()
    )

    rng = random.Random(seed)
    scene_inits: dict[str, dict]
    if scene_inits_json is not None:
        # Cached path: 7 scene init graphs in a single JSON. Avoids unzipping
        # the 24K-file `init_and_final_graphs` directory.
        scene_inits = json.loads(scene_inits_json.read_text())
    else:
        if raw_dir is None:
            raise ValueError("either --raw-dir or --scene-inits-json must be provided")
        scene_dirs = sorted(
            d for d in (raw_dir / "init_and_final_graphs").iterdir() if d.is_dir()
        )
        scene_inits = {}
        for sd in scene_dirs:
            g = find_scene_init_graph(sd)
            if g is not None:
                scene_inits[sd.name] = g
    print(f"loaded {len(scene_inits)} scene init graphs: {list(scene_inits)}")

    # Per-scene class sets — passed to `build_instructions` so it can rank
    # candidate classes (and pairs) by coverage. Coverage-aware ranking keeps
    # (instruction × scene) success rate high while still hitting paper's
    # 9/7/30/32 per-family targets (Table A.2).
    per_scene_classes = [
        {n["class_name"] for n in g["nodes"]} for g in scene_inits.values()
    ]
    union_classes = set().union(*per_scene_classes) if per_scene_classes else set()
    print(f"union classes across {len(per_scene_classes)} scenes: {len(union_classes)}")

    # Deterministic per-paper partition (see PAPER_SEEN_SCENE_KEYS at module
    # top): one v0 variant from each of TrimmedTestScene1..6. Refuses to run
    # if scene_inits is missing any of these — silently substituting would
    # corrupt the seen/unseen distribution.
    seen_scenes = {k for k in PAPER_SEEN_SCENE_KEYS if k in scene_inits}
    if len(seen_scenes) != seen_scene_count:
        raise ValueError(
            f"Expected {seen_scene_count} seen scenes "
            f"{PAPER_SEEN_SCENE_KEYS}, only {sorted(seen_scenes)} present in "
            f"scene_inits. Refusing to silently substitute."
        )
    unseen_scenes = set(scene_inits) - seen_scenes

    candidate_instructions = build_candidate_instructions(
        properties, per_scene_classes, candidate_multiplier=candidate_multiplier
    )
    print(f"candidate instructions: {len(candidate_instructions)}", flush=True)
    candidate_count: Counter = Counter(f for f, _ in candidate_instructions)
    for fam, n in sorted(candidate_count.items()):
        print(f"  {fam}: {n} candidates (target {TARGETS[fam]})", flush=True)

    skipped: Counter = Counter()
    valid_by_task: dict[tuple[str, tuple[str, ...]], dict[str, list[dict]]] = {}
    invalid_tasks: set[tuple[str, tuple[str, ...]]] = set()

    def evaluate_task(task: tuple[str, tuple[str, ...]]) -> dict[str, list[dict]]:
        if task in valid_by_task:
            return valid_by_task[task]
        if task in invalid_tasks:
            return {}
        fam, task_args = task
        task_key = (fam, task_args)
        rows_by_scene = {}
        for scene_name, init in scene_inits.items():
            rows, reason = _execute_candidate(
                fam,
                task_args,
                scene_name,
                init,
                EnvironmentGraph,
                read_script,
                ScriptExecutor,
            )
            if rows is None:
                skipped[reason or "unknown"] += 1
                continue
            rows_by_scene[scene_name] = rows
        if rows_by_scene:
            valid_by_task[task_key] = rows_by_scene
        else:
            invalid_tasks.add(task_key)
        return rows_by_scene

    # Seen vs unseen split (paper § 4): 16 seen / 62 unseen tasks; 6 / 14 scenes.
    # Select from semantically valid tasks, not from the raw candidate list, so
    # the jsonl's actual task count matches the intended 78 tasks.
    fam_quotas = _seen_family_quotas(seen_instruction_count)
    print(f"per-family seen quota: {fam_quotas}", flush=True)
    seen_inst: set[tuple[str, tuple[str, ...]]] = set()
    unseen_inst: set[tuple[str, tuple[str, ...]]] = set()
    for fam in TARGETS:
        fam_tasks = [
            task for task in candidate_instructions if task[0] == fam
        ]
        seen_candidates: list[
            tuple[tuple[str, tuple[str, ...]], set[str], int]
        ] = []
        selected_unseen: list[tuple[str, tuple[str, ...]]] = []
        for order, task in enumerate(fam_tasks):
            rows_by_scene = evaluate_task(task)
            seen_coverage = set(rows_by_scene) & seen_scenes
            if seen_coverage:
                seen_candidates.append((task, seen_coverage, order))
        selected_seen: list[tuple[str, tuple[str, ...]]] = []
        uncovered_seen = set(seen_scenes)
        while len(selected_seen) < fam_quotas[fam] and seen_candidates:
            best_idx, (best_task, best_coverage, _order) = max(
                enumerate(seen_candidates),
                key=lambda item: (
                    len(item[1][1] & uncovered_seen),
                    len(item[1][1]),
                    -item[1][2],
                ),
            )
            selected_seen.append(best_task)
            uncovered_seen -= best_coverage
            seen_candidates.pop(best_idx)
        if len(selected_seen) < fam_quotas[fam]:
            raise ValueError(
                f"Not enough semantically valid seen tasks for {fam}: "
                f"{len(selected_seen)} < {fam_quotas[fam]}"
            )
        seen_inst.update(selected_seen)

        unseen_need = TARGETS[fam] - fam_quotas[fam]
        for task in fam_tasks:
            if task in seen_inst:
                continue
            rows_by_scene = evaluate_task(task)
            if set(rows_by_scene) & unseen_scenes:
                selected_unseen.append(task)
                if len(selected_unseen) >= unseen_need:
                    break
        if len(selected_unseen) < unseen_need:
            raise ValueError(
                f"Not enough semantically valid unseen tasks for {fam}: "
                f"{len(selected_unseen)} < {unseen_need}"
            )
        unseen_inst.update(selected_unseen)
        print(
            f"  {fam}: selected seen={len(selected_seen)}, "
            f"unseen={len(selected_unseen)}, evaluated_valid={sum(1 for t in valid_by_task if t[0] == fam)}",
            flush=True,
        )

    selected_seen_scene_coverage = set().union(
        *(set(valid_by_task[task]) & seen_scenes for task in seen_inst)
    )
    if selected_seen_scene_coverage != seen_scenes:
        missing = sorted(seen_scenes - selected_seen_scene_coverage)
        raise ValueError(f"Selected seen tasks do not cover seen scenes: {missing}")

    valid_counts = Counter(fam for fam, _ in valid_by_task)
    print("\nvalid semantic tasks evaluated:")
    for fam, n in sorted(valid_counts.items()):
        print(f"  {fam}: {n} valid candidates evaluated")
    print(f"  skipped evaluated candidate-scene pairs: {dict(skipped)}")

    instructions = list(seen_inst | unseen_inst)
    selected_count: Counter = Counter(fam for fam, _ in instructions)
    print("\nselected semantic instructions:")
    for fam, n in sorted(selected_count.items()):
        print(f"  {fam}: {n} selected (target {TARGETS[fam]})")

    bucket_rows: dict[str, list[dict]] = {fam: [] for fam in TARGETS}
    selected_task_split = {
        **{task: "seen" for task in seen_inst},
        **{task: "unseen" for task in unseen_inst},
    }
    succeeded_by_scene: Counter = Counter()
    manifest = {
        "seed": seed,
        "targets": TARGETS,
        "seen_instruction_count": seen_instruction_count,
        "seen_family_quotas": fam_quotas,
        "seen_scenes": sorted(seen_scenes),
        "unseen_scenes": sorted(unseen_scenes),
        "selected_tasks": [
            {
                "task_id": f"{fam}:{'|'.join(args)}",
                "family": fam,
                "args": list(args),
                "task_split": selected_task_split[(fam, args)],
                "goal_triple": list(_goal_triple(fam, args)),
                "valid_scenes": sorted(valid_by_task[(fam, args)]),
            }
            for fam, args in sorted(instructions)
        ],
        "trajectories": [],
    }

    for fam, args in sorted(instructions):
        for scene_name, rows in valid_by_task[(fam, args)].items():
            task_is_unseen = (fam, args) in unseen_inst
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
                out_row["_meta"]["task_split"] = selected_task_split[(fam, args)]
                bucket_rows[fam].append(out_row)
            manifest["trajectories"].append(
                {
                    "trajectory_id": rows[0]["_meta"]["trajectory_id"],
                    "family": fam,
                    "args": list(args),
                    "task_split": selected_task_split[(fam, args)],
                    "scene": scene_name,
                    "split": tag,
                    "num_steps": rows[0]["_meta"]["num_steps"],
                    "goal_triple": list(_goal_triple(fam, args)),
                }
            )
            succeeded_by_scene[scene_name] += 1

    if target_trajectories is not None:
        kept_trajectory_ids = _downsample_manifest_trajectories(
            manifest, target_trajectories, seed
        )
        bucket_rows = {
            fam: [
                row for row in rows
                if row["_meta"]["trajectory_id"] in kept_trajectory_ids
            ]
            for fam, rows in bucket_rows.items()
        }
        succeeded_by_scene = Counter(
            traj["scene"] for traj in manifest["trajectories"]
        )

    print("\nexecution summary:")
    print(
        "  selected successful trajectories by family: "
        f"{dict(Counter(t['family'] for t in manifest['trajectories']))}"
    )
    print(f"  succeeded by scene: {dict(sorted(succeeded_by_scene.items()))}")
    zero_success_scenes = sorted(set(scene_inits) - set(succeeded_by_scene))
    if zero_success_scenes:
        print(f"  WARNING zero-success scenes: {zero_success_scenes}")

    # World models are scene-keyed (paper N=6 = 6 seen scenes), so stage-1
    # training data is split per scene. Test files at the root span across
    # scenes and feed Table 1 col 2 / col 3 evals at the WorMI integration
    # level. _meta is persisted on every row so downstream resplit/eval tools
    # can re-bucket without re-running this script.
    seen_scene_list = sorted(seen_scenes)
    seen_to_idx = {s: i for i, s in enumerate(seen_scene_list)}

    per_scene_train: dict[int, dict[str, list[dict]]] = {
        i: defaultdict(list) for i in range(len(seen_scene_list))
    }
    test_buckets: dict[str, list[dict]] = {
        "seen_seen": [],        # Table 1 col 1: seen task × seen scene
        "seen_unseen": [],      # Table 1 col 2: seen task × unseen scene
        "unseen_unseen": [],    # Table 1 col 3: unseen task × unseen scene
        "unseen_seen": [],      # not in Table 1, kept for completeness
    }

    for fam_rows in bucket_rows.values():
        for r in fam_rows:
            meta = r["_meta"]
            if meta["split"] == "seen_seen":
                per_scene_train[seen_to_idx[meta["scene"]]][
                    meta["trajectory_id"]
                ].append(r)
            else:
                test_buckets[meta["split"]].append(r)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "virtualhome_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    # Table 1 col 1 should cover seen tasks in seen scenes. Do not use a
    # plain 9:1 random split here: with 86 seen_seen trajectories it produced
    # only 8 eval episodes and missed open/turnon entirely. Hold out a fixed
    # number per seen task, then train world models on the remaining rows.
    seen_seen_test_ids = _select_seen_seen_eval_ids(
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
            else:
                for r in rows:
                    assert r["_meta"]["step_index"] < r["_meta"]["num_steps"]
    print(
        "  seen_seen task-aware test trajectories: "
        f"{len(seen_seen_test_ids)} "
        f"(per_task={seen_seen_eval_per_task}, "
        f"by_family={dict(sorted(seen_seen_test_by_family.items()))}, "
        f"by_scene={dict(sorted(seen_seen_test_by_scene.items()))})"
    )

    # Root-level test files first, so per-scene `test.jsonl` symlinks resolve.
    split_to_filename = {
        "seen_seen": "test_seen_task_seen_scene.jsonl",
        "seen_unseen": "test_seen_task_unseen_scene.jsonl",
        "unseen_unseen": "test_unseen_task_unseen_scene.jsonl",
        "unseen_seen": "test_unseen_task_seen_scene.jsonl",
    }
    print("\noutput counts:")
    for split, fname in split_to_filename.items():
        rows = test_buckets[split]
        rng.shuffle(rows)
        with (output_dir / fname).open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n_traj = len({r["_meta"]["trajectory_id"] for r in rows})
        print(f"  {fname}: {len(rows)} rows, {n_traj} trajectories")

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

    # Per-scene stage-1 dirs: train.jsonl + symlink test.jsonl → root col-2
    # file. The trainer's eval_dataset just needs a non-empty held-out set
    # for periodic eval_loss logging; per-scene eval signal is identical
    # (Table 1 col 1 reporting is a separate WorMI-level eval pass).
    for i, scene in enumerate(seen_scene_list):
        scene_dir = output_dir / f"scene_{i}"
        scene_dir.mkdir(exist_ok=True)
        rows = [
            r
            for tid in sorted(per_scene_train[i])
            for r in sorted(
                per_scene_train[i][tid], key=lambda row: row["_meta"]["step_index"]
            )
        ]
        rng.shuffle(rows)
        with (scene_dir / "train.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        test_link = scene_dir / "test.jsonl"
        if test_link.exists() or test_link.is_symlink():
            test_link.unlink()
        test_link.symlink_to(Path("..") / "test_seen_task_seen_scene.jsonl")
        n_traj = len({r["_meta"]["trajectory_id"] for r in rows})
        print(f"  scene_{i} ({scene}): train={len(rows)} rows, {n_traj} trajectories")

    generated_trajectories = set()
    for rows in test_buckets.values():
        generated_trajectories.update(r["_meta"]["trajectory_id"] for r in rows)
    for traj_rows in per_scene_train.values():
        generated_trajectories.update(traj_rows)
    if len(generated_trajectories) != 1023:
        print(
            "  WARNING generated trajectory count "
            f"{len(generated_trajectories)} != paper count 1023"
        )


def _parse_per_base(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("per-base counts must be positive integers")
    return values


def _graph_probe_successes(
    graph: dict,
    properties: dict[str, list[str]],
    eg_modules: dict,
    max_probe_tasks: int,
) -> int:
    class_sets = [{node["class_name"] for node in graph["nodes"]}]
    candidates = build_candidate_instructions(
        properties,
        class_sets,
        candidate_multiplier=2,
    )
    EnvironmentGraph = eg_modules["environment"].EnvironmentGraph
    read_script = eg_modules["scripts"].read_script_from_string
    ScriptExecutor = eg_modules["execution"].ScriptExecutor

    successes = 0
    for family, task_args in candidates[:max_probe_tasks]:
        rows, _reason = _execute_candidate(
            family,
            task_args,
            "probe_scene",
            graph,
            EnvironmentGraph,
            read_script,
            ScriptExecutor,
        )
        if rows is not None:
            successes += 1
    return successes


def build_scene_cache(
    zip_path: Path,
    vh_src: Path,
    output_json: Path,
    manifest_json: Path | None = None,
    seed: int = 42,
    per_base: list[int] | None = None,
    min_probe_successes: int = 3,
    max_probe_tasks: int = 80,
) -> None:
    """Build a 20-scene init-graph cache while skipping unusable graphs."""
    prefix = "programs_processed_precond_nograb_morepreconds/init_and_final_graphs/"
    rng = random.Random(seed)
    per_base = per_base or [3, 3, 3, 3, 3, 3, 2]
    properties = json.loads(
        (vh_src / "virtualhome" / "resources" / "properties_data.json").read_text()
    )
    eg_modules = _bootstrap_evolving_graph(vh_src)

    with zipfile.ZipFile(zip_path) as zf:
        by_base: dict[str, list[str]] = {}
        for name in zf.namelist():
            if not (name.startswith(prefix) and name.endswith(".json")):
                continue
            base = name[len(prefix):].split("/", 1)[0]
            by_base.setdefault(base, []).append(name)

        out: dict[str, dict] = {}
        manifest = {
            "zip_path": str(zip_path),
            "seed": seed,
            "per_base": per_base,
            "min_probe_successes": min_probe_successes,
            "max_probe_tasks": max_probe_tasks,
            "scenes": [],
        }

        bases = sorted(by_base)
        if len(per_base) > len(bases):
            raise ValueError(
                f"per-base has {len(per_base)} entries but archive only has "
                f"{len(bases)} bases"
            )

        for base, target_count in zip(bases, per_base):
            candidates = sorted(by_base[base])
            rng.shuffle(candidates)
            picked = 0
            scanned = 0
            for candidate in candidates:
                if picked >= target_count:
                    break
                scanned += 1
                with zf.open(candidate) as f:
                    data = json.load(f)
                graph = data.get("init_graph")
                if graph is None:
                    continue
                successes = _graph_probe_successes(
                    graph,
                    properties,
                    eg_modules,
                    max_probe_tasks,
                )
                if successes < min_probe_successes:
                    continue
                key = f"{base}__v{picked}"
                out[key] = graph
                manifest["scenes"].append(
                    {
                        "key": key,
                        "base": base,
                        "archive_member": candidate,
                        "probe_successes": successes,
                    }
                )
                picked += 1
            if picked < target_count:
                raise RuntimeError(
                    f"Only picked {picked}/{target_count} valid graphs for {base} "
                    f"after scanning {scanned} candidates"
                )
            print(f"{base}: picked {picked}/{target_count} valid graphs", flush=True)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out))
    if manifest_json is not None:
        manifest_json.parent.mkdir(parents=True, exist_ok=True)
        manifest_json.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(out)} scene init graphs to {output_json}")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--raw-dir", type=Path, default=None,
        help="Path to programs_processed_precond_nograb_morepreconds dir "
        "(unzipped). Optional if --scene-inits-json is given.",
    )
    p.add_argument(
        "--scene-inits-json", type=Path, default=None,
        help="Cached `{scene_dir_name: init_graph}` JSON. Use this to avoid "
        "re-unzipping the 24K-file raw archive.",
    )
    p.add_argument(
        "--vh-src", type=Path, required=True,
        help="Path to a cloned virtualhome git repo (for evolving_graph + resources)",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seen-scenes", type=int, default=6)
    p.add_argument("--seen-instructions", type=int, default=16)
    p.add_argument("--train-ratio", type=float, default=0.9, help="Deprecated for Table-1 seen_seen; retained for compatibility.")
    p.add_argument("--seen-seen-eval-per-task", type=int, default=2, help="Task-aware held-out trajectories per seen task for Table-1 col1.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--candidate-multiplier",
        type=int,
        default=12,
        help="How many ranked pair-task candidates to evaluate per target count.",
    )
    p.add_argument(
        "--target-trajectories",
        type=int,
        default=None,
        help="Optional paper-count downsample target, e.g. 1023.",
    )
    return p


def _scene_cache_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--zip-path", type=Path, required=True)
    p.add_argument("--vh-src", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--manifest-json", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--per-base",
        type=_parse_per_base,
        default=_parse_per_base("3,3,3,3,3,3,2"),
    )
    p.add_argument("--min-probe-successes", type=int, default=3)
    p.add_argument("--max-probe-tasks", type=int, default=80)
    return p


def _run_build(args) -> None:
    build(
        args.raw_dir, args.vh_src, args.output_dir,
        args.seen_scenes, args.seen_instructions, args.train_ratio, args.seed,
        scene_inits_json=args.scene_inits_json,
        candidate_multiplier=args.candidate_multiplier,
        target_trajectories=args.target_trajectories,
        seen_seen_eval_per_task=args.seen_seen_eval_per_task,
    )


def _run_scene_cache(args) -> None:
    build_scene_cache(
        zip_path=args.zip_path,
        vh_src=args.vh_src,
        output_json=args.output_json,
        manifest_json=args.manifest_json,
        seed=args.seed,
        per_base=args.per_base,
        min_probe_successes=args.min_probe_successes,
        max_probe_tasks=args.max_probe_tasks,
    )


def main() -> None:
    # Preserve the old direct builder CLI:
    #   python tools/build_virtualhome_dataset.py --scene-inits-json ...
    # and add explicit subcommands for the consolidated interface:
    #   python tools/build_virtualhome_dataset.py build ...
    #   python tools/build_virtualhome_dataset.py scene-cache ...
    if len(sys.argv) > 1 and sys.argv[1] in {"build", "scene-cache"}:
        command = sys.argv[1]
        rest = sys.argv[2:]
        if command == "build":
            _run_build(_build_arg_parser().parse_args(rest))
        else:
            _run_scene_cache(_scene_cache_arg_parser().parse_args(rest))
        return
    _run_build(_build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
