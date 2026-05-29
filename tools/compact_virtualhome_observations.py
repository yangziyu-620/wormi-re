#!/usr/bin/env python3
"""Create a TMoW-style compact VirtualHome JSONL dataset from WorMI rows.

This is a preprocessing experiment, not a model change. It keeps WorMI's JSONL
schema and directory layout, but rewrites `observation` and `next_observation`
so prompts contain an instruction-conditioned belief-state subset instead of a
full class-level scene graph.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


TRIPLE_RE = re.compile(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)")
WORD_RE = re.compile(r"[A-Za-z0-9_]+")
STOP_TOKENS = {
    "a",
    "an",
    "and",
    "in",
    "is",
    "of",
    "on",
    "open",
    "or",
    "place",
    "put",
    "the",
    "to",
    "turn",
    "walk",
}
ROOMS = {"livingroom", "bathroom", "kitchen", "bedroom"}
IMPORTANT_CHARACTER_RELS = {"inside", "hold"}
LOW_VALUE_RELS = {"adjacent"}
DEFAULT_NUM_EDGES = 17
_ROOM_CANONICAL = {
    "dining_room": "kitchen",
    "kids_bedroom": "bedroom",
    "home_office": "livingroom",
    "living_room": "livingroom",
}
_NON_OBJECT_CATEGORIES = {"Floor", "Walls", "Ceiling", "Doors"}
_OBSERVABLE_STATES = {
    "OPEN": "open",
    "CLOSED": "closed",
    "ON": "on",
    "OFF": "off",
    "PLUGGED_IN": "plugged_in",
    "PLUGGED_OUT": "plugged_out",
}


Triple = tuple[str, str, str]


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(text.replace("_", " "))}


def _content_tokens(text: str) -> set[str]:
    return _tokens(text) - STOP_TOKENS


def _parse_triples(text: str) -> list[Triple]:
    return [
        (subj.strip().lower(), rel.strip().lower(), obj.strip().lower())
        for subj, rel, obj in TRIPLE_RE.findall(text or "")
    ]


def _format_triples(triples: list[Triple]) -> str:
    if not triples:
        return "No updates"
    return ", ".join(f"({s}, {r}, {o})" for s, r, o in triples)


def _canon_room(name: str) -> str:
    return _ROOM_CANONICAL.get(name, name)


def _nodes_by_id(graph: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(node["id"]): node for node in graph.get("nodes", [])}


def _graph_node_name(nodes_by_id: dict[int, dict[str, Any]], node_id: int) -> str | None:
    node = nodes_by_id.get(int(node_id))
    if node is None or node.get("category") in _NON_OBJECT_CATEGORIES:
        return None
    return str(node["class_name"]).lower()


def graph_observation_triples(graph: dict[str, Any]) -> list[Triple]:
    """Render graph triples with the same class-level contract as WorMI VH data."""
    nodes_by_id = _nodes_by_id(graph)
    agent_id = next(
        (
            int(node["id"])
            for node in graph.get("nodes", [])
            if node.get("category") == "Characters"
        ),
        None,
    )
    triples: set[Triple] = set()
    seen_holds = False

    for node in graph.get("nodes", []):
        node_name = _graph_node_name(nodes_by_id, int(node["id"]))
        if node_name is None:
            continue
        for state in sorted(set(node.get("states", []))):
            if state in _OBSERVABLE_STATES:
                triples.add((node_name, "is", _OBSERVABLE_STATES[state]))

    for edge in graph.get("edges", []):
        rel_native = edge["relation_type"]
        sub = nodes_by_id.get(int(edge["from_id"]))
        obj = nodes_by_id.get(int(edge["to_id"]))
        if sub is None or obj is None:
            continue
        sub_name = _graph_node_name(nodes_by_id, int(sub["id"]))
        obj_name = _graph_node_name(nodes_by_id, int(obj["id"]))
        if sub_name is None or obj_name is None:
            continue

        if rel_native == "INSIDE":
            target = _canon_room(obj_name) if obj.get("category") == "Rooms" else obj_name
            triples.add((sub_name, "inside", target))
        elif rel_native == "ON":
            triples.add((sub_name, "on", obj_name))
        elif rel_native == "CLOSE" and agent_id is not None and int(sub["id"]) == agent_id:
            triples.add((sub_name, "close", obj_name))
        elif rel_native in {"HOLDS_RH", "HOLDS_LH"} and agent_id is not None and int(sub["id"]) == agent_id:
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
    return sorted(triples)


def _graph_class_node_ids(graph: dict[str, Any], class_name: str) -> list[int]:
    return [
        int(node["id"])
        for node in graph.get("nodes", [])
        if str(node.get("class_name")) == class_name
    ]


def _graph_node_has_state(graph: dict[str, Any], node_id: int, state: str) -> bool:
    node = _nodes_by_id(graph).get(int(node_id))
    return node is not None and state.upper() in set(node.get("states", []))


def _graph_node_category(graph: dict[str, Any], node_id: int) -> str | None:
    node = _nodes_by_id(graph).get(int(node_id))
    return None if node is None else node.get("category")


def _graph_parents(
    graph: dict[str, Any], node_id: int, relation: str = "INSIDE"
) -> list[int]:
    relation = relation.upper()
    return [
        int(edge["to_id"])
        for edge in graph.get("edges", [])
        if int(edge["from_id"]) == int(node_id)
        and edge.get("relation_type") == relation
    ]


def graph_room_for_node(graph: dict[str, Any], node_id: int) -> int | None:
    nodes = _nodes_by_id(graph)
    seen: set[int] = set()
    frontier = [int(node_id)]
    while frontier:
        current = frontier.pop(0)
        if current in seen:
            continue
        seen.add(current)
        node = nodes.get(current)
        if node is not None and node.get("category") == "Rooms":
            return current
        frontier.extend(_graph_parents(graph, current, "INSIDE"))
    return None


def graph_agent_room(graph: dict[str, Any]) -> int | None:
    for node in graph.get("nodes", []):
        if node.get("category") == "Characters":
            return graph_room_for_node(graph, int(node["id"]))
    return None


def graph_container_parent_for_node(
    graph: dict[str, Any], node_id: int
) -> int | None:
    for parent_id in _graph_parents(graph, node_id, "INSIDE"):
        if _graph_node_category(graph, parent_id) != "Rooms":
            return parent_id
    return None


def select_graph_class_node(
    graph: dict[str, Any],
    class_name: str,
    *,
    prefer_state: str | None = None,
    prefer_container_parent: bool = False,
) -> int | None:
    candidates = _graph_class_node_ids(graph, class_name)
    if not candidates:
        return None

    def score(node_id: int) -> tuple[int, int, int, int]:
        state_score = int(
            prefer_state is not None
            and _graph_node_has_state(graph, node_id, prefer_state)
        )
        container_score = int(
            prefer_container_parent
            and graph_container_parent_for_node(graph, node_id) is not None
        )
        room_score = int(graph_room_for_node(graph, node_id) is not None)
        # Lower node ids make selection deterministic when semantic scores tie.
        return (state_score, container_score, room_score, -int(node_id))

    return max(candidates, key=score)


def _selection_node_ids(selection: dict[str, Any]) -> list[int]:
    out: list[int] = []
    for key in ("source_id", "target_id", "source_container"):
        value = selection.get(key)
        if value is None:
            continue
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def selected_instance_ids_from_selection(selection: dict[str, Any]) -> list[int]:
    return _selection_node_ids(selection)


def select_task_instances(
    graph: dict[str, Any], family: str, args: tuple[str, ...] | list[str]
) -> dict[str, Any]:
    """Select task-bound instances from only task spec and current graph.

    This is the common task-definition selector for expert planning, compact
    observation rendering, validation, and rollout. It must not read expert
    actions or generated trajectories.
    """
    task_args = tuple(str(arg).lower().replace(" ", "_") for arg in args)
    selection: dict[str, Any] = {
        "instance_selection_mode": "deterministic_graph_task_v1",
        "selection_inputs": ["instruction", "current_graph", "task_args"],
        "family": family,
        "args": list(task_args),
        "source_id": None,
        "target_id": None,
        "source_room": None,
        "target_room": None,
        "source_container": None,
        "start_room": graph_agent_room(graph),
    }

    if family in {"turnon", "open"}:
        if not task_args:
            selection["reason"] = "missing_task_args"
            selection["selected_node_ids"] = []
            return selection
        preferred = "OFF" if family == "turnon" else "CLOSED"
        target_id = select_graph_class_node(graph, task_args[0], prefer_state=preferred)
        selection["target_id"] = target_id
        selection["target_room"] = (
            graph_room_for_node(graph, target_id) if target_id is not None else None
        )
        selection["selected_node_ids"] = _selection_node_ids(selection)
        return selection

    if family in {"puton", "placein"}:
        if len(task_args) < 2:
            selection["reason"] = "missing_task_args"
            selection["selected_node_ids"] = []
            return selection
        source_cls, target_cls = task_args[:2]
        source_id = select_graph_class_node(
            graph, source_cls, prefer_container_parent=True
        )
        target_id = select_graph_class_node(
            graph,
            target_cls,
            prefer_state="CLOSED" if family == "placein" else None,
        )
        source_container = (
            graph_container_parent_for_node(graph, source_id)
            if source_id is not None
            else None
        )
        if source_container is not None and source_container == target_id:
            source_container = None
        selection.update(
            {
                "source_id": source_id,
                "target_id": target_id,
                "source_room": (
                    graph_room_for_node(graph, source_id)
                    if source_id is not None
                    else None
                ),
                "target_room": (
                    graph_room_for_node(graph, target_id)
                    if target_id is not None
                    else None
                ),
                "source_container": source_container,
            }
        )
        selection["selected_node_ids"] = _selection_node_ids(selection)
        return selection

    selection["reason"] = f"unsupported_family:{family}"
    selection["selected_node_ids"] = []
    return selection


def selected_instance_ids_from_meta(meta: dict[str, Any]) -> list[int]:
    """Return selected ids from explicit selector metadata, with legacy fallback."""
    selection = meta.get("instance_selection")
    if isinstance(selection, dict):
        return selected_instance_ids_from_selection(selection)
    prep = meta.get("observation_preprocessing") or {}
    grounding = prep.get("grounding_node_ids")
    if isinstance(grounding, list) and grounding:
        return sorted({int(node_id) for node_id in grounding})
    debug = meta.get("planner_debug") or {}
    return _selection_node_ids(debug)


def _selected_instance_triples(
    graph: dict[str, Any],
    selected_node_ids: list[int],
) -> list[Triple]:
    selected = {int(node_id) for node_id in selected_node_ids}
    if not selected:
        return []

    nodes_by_id = _nodes_by_id(graph)
    agent_ids = {
        int(node["id"])
        for node in graph.get("nodes", [])
        if node.get("category") == "Characters"
    }
    triples: set[Triple] = set()

    for node_id in selected:
        node = nodes_by_id.get(node_id)
        if node is None:
            continue
        node_name = _graph_node_name(nodes_by_id, node_id)
        if node_name is None:
            continue
        for state in sorted(set(node.get("states", []))):
            if state in _OBSERVABLE_STATES:
                triples.add((node_name, "is", _OBSERVABLE_STATES[state]))

    for edge in graph.get("edges", []):
        from_id = int(edge["from_id"])
        to_id = int(edge["to_id"])
        rel_native = edge["relation_type"]
        touches_selected = (
            from_id in selected
            or to_id in selected
            or (from_id in agent_ids and to_id in selected)
        )
        if not touches_selected:
            continue

        sub = nodes_by_id.get(from_id)
        obj = nodes_by_id.get(to_id)
        if sub is None or obj is None:
            continue
        sub_name = _graph_node_name(nodes_by_id, from_id)
        obj_name = _graph_node_name(nodes_by_id, to_id)
        if sub_name is None or obj_name is None:
            continue

        if rel_native == "INSIDE":
            target = _canon_room(obj_name) if obj.get("category") == "Rooms" else obj_name
            triples.add((sub_name, "inside", target))
        elif rel_native == "ON":
            triples.add((sub_name, "on", obj_name))
        elif rel_native == "CLOSE" and from_id in agent_ids:
            triples.add((sub_name, "close", obj_name))
        elif rel_native in {"HOLDS_RH", "HOLDS_LH"} and from_id in agent_ids:
            triples.add((sub_name, "hold", obj_name))

    return sorted(triples)


def instance_grounded_observation_triples(
    graph: dict[str, Any],
    *,
    task_args: list[str],
    selected_node_ids: list[int],
) -> list[Triple]:
    """Collapse duplicate class facts for task-bound instances only.

    VirtualHome graphs may contain several nodes with the same class. A
    class-level observation such as `(drawing, inside, bedroom)` and
    `(drawing, inside, kitchen)` is ambiguous when the expert trajectory has
    selected one concrete `drawing` id. This helper removes class-level facts
    for the selected task classes and re-adds only facts produced by the
    selected ids, while leaving unrelated filler triples available for compact
    retrieval.
    """
    full = graph_observation_triples(graph)
    nodes_by_id = _nodes_by_id(graph)
    selected_classes = {
        _graph_node_name(nodes_by_id, int(node_id))
        for node_id in selected_node_ids
    }
    selected_classes = {name for name in selected_classes if name is not None}
    selected_classes |= {str(arg).lower().replace(" ", "_") for arg in task_args}
    if not selected_classes:
        return full

    task_relations = {"inside", "on", "is", "hold", "close"}
    retained = [
        triple
        for triple in full
        if not (
            triple[1] in task_relations
            and (triple[0] in selected_classes or triple[2] in selected_classes)
        )
    ]
    selected = _selected_instance_triples(graph, selected_node_ids)
    return sorted(_dedupe(retained + selected))


def format_instance_grounded_observation(
    graph: dict[str, Any],
    *,
    task_args: list[str],
    selected_node_ids: list[int],
) -> str:
    return _format_triples(
        instance_grounded_observation_triples(
            graph,
            task_args=task_args,
            selected_node_ids=selected_node_ids,
        )
    )


def _dedupe(triples: list[Triple]) -> list[Triple]:
    seen = set()
    out = []
    for triple in triples:
        if triple in seen:
            continue
        seen.add(triple)
        out.append(triple)
    return out


def _action_parts(action: str) -> tuple[str, list[str]]:
    parts = (action or "").strip().lower().split()
    if not parts:
        return "", []
    return parts[0], parts[1:]


def _instruction_family(instruction: str) -> str | None:
    text = instruction.lower()
    if text.startswith("turn on "):
        return "turnon"
    if text.startswith("open "):
        return "open"
    if text.startswith("put ") and " on " in text:
        return "puton"
    if text.startswith("place ") and " in " in text:
        return "placein"
    return None


def _task_args(row: dict[str, Any]) -> list[str]:
    meta = row.get("_meta") or {}
    args = meta.get("task_args")
    if isinstance(args, list) and args:
        return [str(arg).replace(" ", "_").lower() for arg in args]

    instruction = str(row.get("instruction", "")).lower()
    if instruction.startswith("turn on "):
        return [instruction.removeprefix("turn on ").replace(" ", "_")]
    if instruction.startswith("open "):
        return [instruction.removeprefix("open ").replace(" ", "_")]
    if instruction.startswith("put ") and " on " in instruction:
        src, dst = instruction.removeprefix("put ").split(" on ", 1)
        return [src.replace(" ", "_"), dst.replace(" ", "_")]
    if instruction.startswith("place ") and " in " in instruction:
        src, dst = instruction.removeprefix("place ").split(" in ", 1)
        return [src.replace(" ", "_"), dst.replace(" ", "_")]
    return []


def _goal_relations(family: str | None) -> set[str]:
    if family in {"turnon", "open"}:
        return {"is"}
    if family == "puton":
        return {"on"}
    if family == "placein":
        return {"inside"}
    return set()


def _is_named_entity(triple: Triple, names: set[str]) -> bool:
    subj, _rel, obj = triple
    return subj in names or obj in names


def _score_triple(
    triple: Triple,
    *,
    instruction_tokens: set[str],
    action_tokens: set[str],
    important_names: set[str],
    family: str | None,
) -> tuple[int, int, str]:
    subj, rel, obj = triple
    triple_tokens = _content_tokens(" ".join(triple))
    overlap = len(triple_tokens & instruction_tokens)
    action_overlap = len(triple_tokens & action_tokens)
    goal_rels = _goal_relations(family)

    score = 0
    if subj == "character" and rel in IMPORTANT_CHARACTER_RELS:
        score += 120
    if subj == "character" and rel == "close" and obj in important_names:
        score += 100
    if _is_named_entity(triple, important_names):
        score += 80
    if rel in goal_rels and _is_named_entity(triple, important_names):
        score += 70
    if rel in {"inside", "on"} and _is_named_entity(triple, important_names):
        score += 40
    if rel == "is" and subj in important_names:
        score += 40
    if obj in ROOMS and subj in important_names:
        score += 30
    if rel in LOW_VALUE_RELS:
        score -= 10
    score += overlap * 12
    score += action_overlap * 10

    # More specific object facts should beat broad room-adjacency facts on ties.
    specificity = int(subj in important_names) + int(obj in important_names)
    return score, specificity, f"{subj} {rel} {obj}"


def compact_observation(
    observation: str,
    *,
    instruction: str,
    action: str,
    task_args: list[str],
    num_edges: int = DEFAULT_NUM_EDGES,
    fill_to_num_edges: bool = False,
) -> list[Triple]:
    triples = _dedupe(_parse_triples(observation))
    if len(triples) <= num_edges:
        return triples

    verb, action_args = _action_parts(action)
    del verb
    important_names = set(task_args) | set(action_args)
    instruction_tokens = _content_tokens(instruction) | {
        token for name in important_names for token in _content_tokens(name)
    }
    action_tokens = _content_tokens(action)
    family = _instruction_family(instruction)

    mandatory = []
    for triple in triples:
        subj, rel, obj = triple
        if subj == "character" and rel in IMPORTANT_CHARACTER_RELS:
            mandatory.append(triple)
        elif subj == "character" and rel == "close" and obj in important_names:
            mandatory.append(triple)
        elif _is_named_entity(triple, important_names) and rel in {
            "inside",
            "on",
            "is",
            "hold",
            "close",
        }:
            mandatory.append(triple)

    mandatory = _dedupe(mandatory)
    mandatory_set = set(mandatory)
    remaining = [triple for triple in triples if triple not in mandatory_set]
    scored = [
        (
            _score_triple(
                triple,
                instruction_tokens=instruction_tokens,
                action_tokens=action_tokens,
                important_names=important_names,
                family=family,
            ),
            triple,
        )
        for triple in remaining
    ]
    positive = [item for item in scored if item[0][0] > 0]
    ranked_all = [
        triple
        for _score, triple in sorted(
            scored,
            key=lambda item: item[0],
            reverse=True,
        )
    ]
    if fill_to_num_edges:
        ranked = ranked_all
    elif positive:
        ranked = [
            triple
            for _score, triple in sorted(
                positive,
                key=lambda item: item[0],
                reverse=True,
            )
        ]
    elif mandatory:
        ranked = []
    else:
        ranked = ranked_all

    # TMoW uses num_edges for retrieved facts and appends mandatory agent facts.
    selected = _dedupe(mandatory + ranked[:num_edges])
    return sorted(selected, key=lambda triple: (triple[0], triple[1], triple[2]))


def _fallback_update(action: str, next_triples: list[Triple]) -> list[Triple]:
    verb, args = _action_parts(action)
    desired: list[Triple] = []
    if verb == "grab" and len(args) == 1:
        desired.append(("character", "hold", args[0]))
    elif verb == "open" and len(args) == 1:
        desired.append((args[0], "is", "open"))
    elif verb == "switchon" and len(args) == 1:
        desired.append((args[0], "is", "on"))
    elif verb == "put" and len(args) == 2:
        desired.append((args[0], "on", args[1]))
    elif verb == "putin" and len(args) == 2:
        desired.append((args[0], "inside", args[1]))
    elif verb == "walk" and len(args) == 1:
        desired.extend(
            triple
            for triple in next_triples
            if triple[0] == "character"
            and (triple[1] == "inside" or triple == ("character", "close", args[0]))
        )

    next_set = set(next_triples)
    return [triple for triple in desired if triple in next_set]


def compact_next_observation(
    row: dict[str, Any],
    *,
    current_compact: list[Triple],
    next_compact: list[Triple],
    mode: str,
) -> list[Triple]:
    if mode == "compact":
        return next_compact
    if mode != "delta":
        raise ValueError(f"Unsupported next mode: {mode}")

    current_set = set(current_compact)
    changed = [triple for triple in next_compact if triple not in current_set]
    task_names = set(_task_args(row))
    _verb, action_args = _action_parts(str(row.get("action", "")))
    important_names = task_names | set(action_args)

    relevant = []
    for triple in changed:
        subj, rel, obj = triple
        if subj == "character":
            relevant.append(triple)
        elif subj in important_names or obj in important_names:
            relevant.append(triple)
        elif rel in _goal_relations(_instruction_family(str(row.get("instruction", "")))):
            relevant.append(triple)

    relevant = _dedupe(relevant)
    if relevant:
        return sorted(relevant, key=lambda triple: (triple[0], triple[1], triple[2]))

    fallback = _fallback_update(str(row.get("action", "")), next_compact)
    return sorted(_dedupe(fallback), key=lambda triple: (triple[0], triple[1], triple[2]))


def _triple_count(text: str) -> int:
    return len(_parse_triples(text))


def _mean(values: list[int]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def process_row(
    row: dict[str, Any],
    *,
    num_edges: int,
    next_mode: str,
) -> dict[str, Any]:
    task_args = _task_args(row)
    instruction = str(row.get("instruction", ""))
    action = str(row.get("action", ""))
    # Current observations are policy inputs, so they must be reproducible at
    # rollout time without knowing the target action. The action is still used
    # below for next-observation deltas, where it is already present in chat.
    current = compact_observation(
        str(row.get("observation", "")),
        instruction=instruction,
        action="",
        task_args=task_args,
        num_edges=num_edges,
    )
    nxt = compact_observation(
        str(row.get("next_observation", "")),
        instruction=instruction,
        action=action,
        task_args=task_args,
        num_edges=num_edges,
    )
    next_out = compact_next_observation(
        row,
        current_compact=current,
        next_compact=nxt,
        mode=next_mode,
    )

    out = dict(row)
    out["observation"] = _format_triples(current)
    out["next_observation"] = _format_triples(next_out)
    meta = dict(out.get("_meta") or {})
    meta["observation_preprocessing"] = {
        "mode": "tmow_compact_from_jsonl",
        "num_edges": num_edges,
        "next_mode": next_mode,
        "current_observation_action_conditioned": False,
        "source_observation_triples": _triple_count(str(row.get("observation", ""))),
        "compact_observation_triples": len(current),
        "source_next_observation_triples": _triple_count(
            str(row.get("next_observation", ""))
        ),
        "compact_next_observation_triples": len(next_out),
    }
    out["_meta"] = meta
    return out


def _process_jsonl(
    src: Path,
    dst: Path,
    *,
    num_edges: int,
    next_mode: str,
    limit_rows_per_file: int | None,
) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    obs_before: list[int] = []
    obs_after: list[int] = []
    next_before: list[int] = []
    next_after: list[int] = []
    no_updates = 0
    rows = 0

    with src.open() as f_in, dst.open("w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            if limit_rows_per_file is not None and rows >= limit_rows_per_file:
                break
            row = json.loads(line)
            new_row = process_row(row, num_edges=num_edges, next_mode=next_mode)
            obs_before.append(_triple_count(str(row.get("observation", ""))))
            obs_after.append(_triple_count(str(new_row.get("observation", ""))))
            next_before.append(_triple_count(str(row.get("next_observation", ""))))
            next_after.append(_triple_count(str(new_row.get("next_observation", ""))))
            no_updates += int(str(new_row.get("next_observation", "")) == "No updates")
            rows += 1
            f_out.write(json.dumps(new_row, sort_keys=False) + "\n")

    return {
        "rows": rows,
        "observation_triples_before_mean": _mean(obs_before),
        "observation_triples_after_mean": _mean(obs_after),
        "next_triples_before_mean": _mean(next_before),
        "next_triples_after_mean": _mean(next_after),
        "next_no_updates": no_updates,
    }


def compact_dataset(
    input_root: Path,
    output_root: Path,
    *,
    num_edges: int,
    next_mode: str,
    limit_rows_per_file: int | None,
) -> dict[str, Any]:
    if not input_root.exists():
        raise FileNotFoundError(input_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"{output_root} already exists and is not empty. "
            "Use a new output directory."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    file_summaries: dict[str, dict[str, Any]] = {}
    copied = Counter()

    for src in sorted(input_root.rglob("*")):
        rel = src.relative_to(input_root)
        dst = output_root / rel
        if src.is_symlink():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src.readlink())
            copied["symlink"] += 1
            continue
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        if src.suffix == ".jsonl":
            file_summaries[str(rel)] = _process_jsonl(
                src,
                dst,
                num_edges=num_edges,
                next_mode=next_mode,
                limit_rows_per_file=limit_rows_per_file,
            )
            copied["jsonl"] += 1
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied["copied"] += 1

    total_rows = sum(item["rows"] for item in file_summaries.values())
    weighted = {}
    for key in [
        "observation_triples_before_mean",
        "observation_triples_after_mean",
        "next_triples_before_mean",
        "next_triples_after_mean",
    ]:
        weighted[key] = (
            sum(item[key] * item["rows"] for item in file_summaries.values())
            / total_rows
            if total_rows
            else 0.0
        )

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_edges": num_edges,
        "next_mode": next_mode,
        "limit_rows_per_file": limit_rows_per_file,
        "total_rows": total_rows,
        "counts": dict(copied),
        **weighted,
        "next_no_updates": sum(item["next_no_updates"] for item in file_summaries.values()),
        "files": file_summaries,
    }
    with (output_root / "compact_virtualhome_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite WorMI VirtualHome JSONL observations into deterministic "
            "TMoW-style compact triple lists."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-edges", type=int, default=DEFAULT_NUM_EDGES)
    parser.add_argument(
        "--next-mode",
        choices=["delta", "compact"],
        default="delta",
        help="Use compact state updates or compact full next observations.",
    )
    parser.add_argument(
        "--limit-rows-per-file",
        type=int,
        default=None,
        help="Smoke-test option; process at most N rows from each JSONL file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = compact_dataset(
        args.input_root,
        args.output_root,
        num_edges=args.num_edges,
        next_mode=args.next_mode,
        limit_rows_per_file=args.limit_rows_per_file,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
