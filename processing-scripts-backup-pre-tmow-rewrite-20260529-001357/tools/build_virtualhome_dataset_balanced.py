#!/usr/bin/env python3
"""Build a paper-compatible balanced VirtualHome reconstruction for WorMI.

This script is intentionally not an author-exact reproduction: the WorMI paper
reports aggregate VirtualHome counts, but not exact scene/task/episode IDs. The
protocol here is a fixed-seed, auditable reconstruction.

Important split rule:
    train: seen_task   intersect seen_scene
    eval A: seen_task  intersect seen_scene, episode-held-out from train
    eval B: seen_task  intersect unseen_scene
    eval C: unseen_task intersect unseen_scene

Eval C is sampled only from its own legal candidate pool. It is not computed as
"whatever remains" after other splits.

To avoid the old sparse world-model data, scene domains are backed by multiple
official VirtualHome init-graph variants from the same base apartment. Sampling
is soft-balanced by scene/task/family; it does not require a full Cartesian
product for every task-scene cell.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.build_virtualhome_dataset import (  # noqa: E402
    TARGETS,
    _bootstrap_evolving_graph,
    _execute_paperlike_candidate,
    _goal_triple,
    _graph_probe_successes,
    _seen_family_quotas,
    build_candidate_instructions,
)
from tools.compact_virtualhome_observations import (  # noqa: E402
    graph_room_for_node,
    process_row,
)

DATASET_MODE = "paper_compatible_balanced_reconstruction"
DEFAULT_BASE_DOMAIN_COUNTS = [3, 3, 3, 3, 3, 3, 2]
ROOT_SPLIT_FILES = {
    "seen_seen": "test_seen_task_seen_scene.jsonl",
    "seen_unseen": "test_seen_task_unseen_scene.jsonl",
    "unseen_unseen": "test_unseen_task_unseen_scene.jsonl",
    "unseen_seen": "test_unseen_task_seen_scene.jsonl",
}
EVAL_LINKS = {
    "eval_col_1_seen_seen": "test_seen_task_seen_scene.jsonl",
    "eval_col_2_seen_unseen": "test_seen_task_unseen_scene.jsonl",
    "eval_col_3_unseen_unseen": "test_unseen_task_unseen_scene.jsonl",
}

Task = tuple[str, tuple[str, ...]]


def _dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def _read_init_graph(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    graph = data.get("init_graph")
    if isinstance(graph, dict):
        return graph
    states = data.get("graph_state_list")
    if isinstance(states, list) and states and isinstance(states[0], dict):
        return states[0]
    return None


def _base_domain_counts(num_bases: int, scene_domains: int) -> list[int]:
    if scene_domains == 20 and num_bases >= 7:
        return DEFAULT_BASE_DOMAIN_COUNTS[:num_bases]
    counts = [0 for _ in range(num_bases)]
    for i in range(scene_domains):
        counts[i % num_bases] += 1
    return counts


def _task_id(task: Task) -> str:
    family, args = task
    return f"{family}:{'|'.join(args)}"


def _class_node_count(graph: dict[str, Any], class_name: str) -> int:
    return sum(1 for node in graph.get("nodes", []) if node.get("class_name") == class_name)


def _class_room_ids(graph: dict[str, Any], class_name: str) -> set[int]:
    room_ids: set[int] = set()
    for node in graph.get("nodes", []):
        if node.get("class_name") != class_name:
            continue
        room_id = graph_room_for_node(graph, int(node["id"]))
        if room_id is not None:
            room_ids.add(int(room_id))
    return room_ids


def _semantic_gate_failure(
    graph: dict[str, Any],
    family: str,
    task_args: tuple[str, ...],
    mode: str,
) -> str | None:
    if mode == "none":
        return None
    if family in {"turnon", "open"}:
        if not task_args:
            return "semantic_missing_unary_arg"
        count = _class_node_count(graph, task_args[0])
        if count != 1:
            return f"semantic_target_multi_instance:{count}"
        return None
    if family in {"puton", "placein"}:
        if len(task_args) < 2:
            return "semantic_missing_binary_args"
        source_count = _class_node_count(graph, task_args[0])
        if source_count != 1:
            return f"semantic_source_multi_instance:{source_count}"
        if mode == "source_and_target_unique":
            target_count = _class_node_count(graph, task_args[1])
            if target_count != 1:
                return f"semantic_target_multi_instance:{target_count}"
        if mode == "source_unique_target_room_unique":
            target_rooms = _class_room_ids(graph, task_args[1])
            if len(target_rooms) != 1:
                return f"semantic_target_multi_room:{len(target_rooms)}"
        return None
    return f"semantic_unsupported_family:{family}"


def _family_eval_targets(
    tasks: Iterable[Task],
    total: int,
    min_per_task: int,
) -> dict[str, int]:
    counts = Counter(task[0] for task in tasks)
    floors = {family: count * min_per_task for family, count in counts.items()}
    if sum(floors.values()) > total:
        raise ValueError(
            f"min_per_task={min_per_task} requires {sum(floors.values())} "
            f"episodes, above target {total}"
        )
    remaining = total - sum(floors.values())
    total_tasks = sum(counts.values())
    targets = dict(floors)
    remainders = []
    for family, count in counts.items():
        exact = remaining * count / total_tasks if total_tasks else 0
        whole = int(exact)
        targets[family] += whole
        remainders.append((exact - whole, family))
    for _frac, family in sorted(remainders, reverse=True)[: total - sum(targets.values())]:
        targets[family] += 1
    return targets


def _choose_balanced(
    slots: list[dict[str, Any]],
    target: int,
    *,
    seed: int,
    blocked: set[str] | None = None,
    blocked_row_signatures: set[tuple[str, str, str, str]] | None = None,
    min_per_task: int = 0,
    family_targets: dict[str, int] | None = None,
    scene_targets: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Greedy soft-balanced sampler over legal candidate slots."""
    blocked = blocked or set()
    blocked_row_signatures = blocked_row_signatures or set()
    available = [
        s
        for s in slots
        if s["trajectory_id"] not in blocked
        and not (s.get("row_signatures", set()) & blocked_row_signatures)
    ]
    if len(available) < target:
        raise RuntimeError(f"pool has {len(available)} slots, need {target}")

    rng = random.Random(seed)
    for slot in available:
        slot["_rand"] = rng.random()

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    task_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()

    def can_add(slot: dict[str, Any]) -> bool:
        if slot["trajectory_id"] in selected_ids:
            return False
        if family_targets is not None:
            if family_counts[slot["family"]] >= family_targets.get(slot["family"], 0):
                return False
        # Scene targets are soft balancing hints, not hard caps. Some semantic
        # gates legitimately leave one scene domain with fewer legal slots.
        # Treating that as a cap can stop sampling even when the global pool is
        # large enough.
        return True

    def add(slot: dict[str, Any]) -> None:
        selected.append(slot)
        selected_ids.add(slot["trajectory_id"])
        task_counts[slot["task_id"]] += 1
        family_counts[slot["family"]] += 1
        scene_counts[slot["scene_domain"]] += 1

    if min_per_task:
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for slot in available:
            by_task[slot["task_id"]].append(slot)
        for task_id in sorted(by_task):
            for _ in range(min_per_task):
                candidates = [slot for slot in by_task[task_id] if can_add(slot)]
                if not candidates:
                    raise RuntimeError(f"cannot satisfy min_per_task for {task_id}")
                candidates.sort(
                    key=lambda s: (
                        scene_counts[s["scene_domain"]],
                        family_counts[s["family"]],
                        s["_rand"],
                    )
                )
                add(candidates[0])
                if len(selected) > target:
                    raise RuntimeError("min_per_task exceeds target")

    while len(selected) < target:
        candidates = [slot for slot in available if can_add(slot)]
        if not candidates:
            raise RuntimeError(
                f"balanced sampler stopped at {len(selected)}/{target}; "
                f"family_counts={dict(family_counts)}, scene_counts={dict(scene_counts)}"
            )
        candidates.sort(
            key=lambda s: (
                scene_counts[s["scene_domain"]] / max(1, scene_targets.get(s["scene_domain"], 1))
                if scene_targets is not None
                else scene_counts[s["scene_domain"]],
                task_counts[s["task_id"]],
                family_counts[s["family"]]
                / max(1, family_targets.get(s["family"], 1))
                if family_targets is not None
                else family_counts[s["family"]],
                s["_rand"],
            )
        )
        add(candidates[0])

    return selected


def _row_signature(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("instruction", "")),
        str(row.get("observation", "")),
        str(row.get("action", "")),
        str(row.get("next_observation", "")),
    )


def _slot_row_signatures(slot: dict[str, Any]) -> set[tuple[str, str, str, str]]:
    return set(slot.get("row_signatures", set()))


class Builder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = random.Random(args.seed)
        self.eg = _bootstrap_evolving_graph(args.vh_src)
        self.EnvironmentGraph = self.eg["environment"].EnvironmentGraph
        self.read_script = self.eg["scripts"].read_script_from_string
        self.ScriptExecutor = self.eg["execution"].ScriptExecutor
        self.properties = json.loads(
            (args.vh_src / "virtualhome" / "resources" / "properties_data.json").read_text()
        )
        self.domains: list[dict[str, Any]] = []
        self.scene_inits: dict[str, dict[str, Any]] = {}
        self.slot_cache: dict[Task, list[dict[str, Any]]] = {}
        self.reason_counts: Counter[str] = Counter()

    def load_domains(self) -> None:
        init_root = self.args.raw_dir / "init_and_final_graphs"
        bases = sorted(path for path in init_root.iterdir() if path.is_dir())
        if not bases:
            raise FileNotFoundError(init_root)
        counts = _base_domain_counts(len(bases), self.args.scene_domains)
        domain_idx = 0
        domains: list[dict[str, Any]] = []
        scene_inits: dict[str, dict[str, Any]] = {}

        for base_dir, domain_count in zip(bases, counts):
            need = domain_count * self.args.variants_per_domain
            print(f"selecting init variants for {base_dir.name}: need {need}", flush=True)
            candidates = sorted(base_dir.rglob("*.json"))
            self.rng.shuffle(candidates)
            selected = []
            scanned = 0
            for path in candidates:
                if len(selected) >= need:
                    break
                scanned += 1
                if scanned > self.args.max_scan_per_base:
                    break
                graph = _read_init_graph(path)
                if graph is None:
                    continue
                if self.args.min_probe_successes > 0:
                    successes = _graph_probe_successes(
                        graph,
                        self.properties,
                        self.eg,
                        max_probe_tasks=self.args.max_probe_tasks,
                    )
                    if successes < self.args.min_probe_successes:
                        continue
                else:
                    successes = None
                selected.append({"path": str(path), "graph": graph, "probe_successes": successes})
            if len(selected) < need:
                raise RuntimeError(
                    f"{base_dir.name}: selected {len(selected)}/{need} variants "
                    f"after scanning {scanned}"
                )
            print(f"  selected {len(selected)} init variants for {base_dir.name} after scanning {scanned}", flush=True)
            offset = 0
            for _ in range(domain_count):
                domain_id = f"scene_domain_{domain_idx:02d}"
                variants = []
                for variant_idx in range(self.args.variants_per_domain):
                    item = selected[offset]
                    offset += 1
                    key = f"{base_dir.name}__d{domain_idx:02d}_v{variant_idx:02d}"
                    scene_inits[key] = item["graph"]
                    variants.append(
                        {
                            "variant_key": key,
                            "variant_index": variant_idx,
                            "source_path": item["path"],
                            "probe_successes": item["probe_successes"],
                        }
                    )
                domains.append({"domain_id": domain_id, "base": base_dir.name, "variants": variants})
                domain_idx += 1

        shuffled = domains[:]
        self.rng.shuffle(shuffled)
        seen_ids = {domain["domain_id"] for domain in shuffled[: self.args.seen_scenes]}
        for domain in domains:
            domain["scene_split"] = "seen" if domain["domain_id"] in seen_ids else "unseen"
        self.domains = domains
        self.scene_inits = scene_inits
        print(f"scene domains: {len(domains)}; init variants: {len(scene_inits)}", flush=True)
        print(f"seen domains: {sorted(seen_ids)}", flush=True)
        self.seen_domains = [d for d in domains if d["scene_split"] == "seen"]
        self.unseen_domains = [d for d in domains if d["scene_split"] == "unseen"]

    def candidate_tasks(self) -> list[Task]:
        class_sets = [
            {node["class_name"] for node in graph["nodes"]}
            for graph in self.scene_inits.values()
        ]
        return build_candidate_instructions(
            self.properties,
            class_sets,
            candidate_multiplier=self.args.candidate_multiplier,
        )

    def preprocess_row(self, row: dict[str, Any]) -> dict[str, Any]:
        if self.args.observation_mode == "full":
            return row
        if self.args.observation_mode == "tmow_compact":
            return process_row(
                row,
                num_edges=self.args.compact_num_edges,
                next_mode=self.args.compact_next_mode,
            )
        raise ValueError(f"Unsupported observation mode: {self.args.observation_mode}")

    def collect_slots(self, task: Task) -> list[dict[str, Any]]:
        if task in self.slot_cache:
            return self.slot_cache[task]
        family, task_args = task
        slots: list[dict[str, Any]] = []
        for domain in self.domains:
            for variant in domain["variants"]:
                graph = self.scene_inits[variant["variant_key"]]
                semantic_failure = _semantic_gate_failure(
                    graph,
                    family,
                    task_args,
                    self.args.semantic_gate,
                )
                if semantic_failure is not None:
                    self.reason_counts[semantic_failure] += 1
                    continue
                rows, reason = _execute_paperlike_candidate(
                    family,
                    task_args,
                    variant["variant_key"],
                    graph,
                    self.EnvironmentGraph,
                    self.read_script,
                    self.ScriptExecutor,
                )
                if rows is None:
                    self.reason_counts[reason or "unknown"] += 1
                    continue
                if any(row["observation"] == row["next_observation"] for row in rows):
                    self.reason_counts["unchanged_transition"] += 1
                    continue
                trajectory_id = (
                    f"{domain['domain_id']}:{variant['variant_key']}:{family}:"
                    f"{'|'.join(task_args)}"
                )
                rewritten = []
                for row in rows:
                    out = copy.deepcopy(row)
                    meta = out["_meta"]
                    meta["scene"] = variant["variant_key"]
                    meta["scene_domain"] = domain["domain_id"]
                    meta["scene_domain_split"] = domain["scene_split"]
                    meta["scene_base"] = domain["base"]
                    meta["scene_variant_index"] = variant["variant_index"]
                    meta["raw_init_graph_path"] = variant["source_path"]
                    meta["source"] = "planner_on_official_init_graph_variant"
                    meta["protocol"] = DATASET_MODE
                    meta["trajectory_id"] = trajectory_id
                    rewritten.append(self.preprocess_row(out))
                row_signatures = {_row_signature(row) for row in rewritten}
                slots.append(
                    {
                        "task": task,
                        "task_id": _task_id(task),
                        "family": family,
                        "scene_domain": domain["domain_id"],
                        "scene_domain_split": domain["scene_split"],
                        "scene_base": domain["base"],
                        "variant_key": variant["variant_key"],
                        "trajectory_id": trajectory_id,
                        "row_signatures": row_signatures,
                        "rows": rewritten,
                    }
                )
        self.slot_cache[task] = slots
        return slots

    def select_tasks(self, candidates: list[Task]) -> tuple[set[Task], set[Task]]:
        seen_quotas = _seen_family_quotas(self.args.seen_instructions)
        by_family: dict[str, list[Task]] = defaultdict(list)
        for task in candidates:
            by_family[task[0]].append(task)

        selected_seen: set[Task] = set()
        selected_unseen: set[Task] = set()
        gap = {"families": {}, "seen_quotas": seen_quotas, "targets": TARGETS}

        for family in TARGETS:
            print(f"evaluating task family {family}", flush=True)
            seen_candidates = []
            unseen_candidates = []
            seen_need = seen_quotas[family]
            unseen_need = TARGETS[family] - seen_need
            buffer = self.args.task_candidate_buffer
            for task in by_family[family]:
                slots = self.collect_slots(task)
                seen_seen = [s for s in slots if s["scene_domain_split"] == "seen"]
                seen_unseen = [s for s in slots if s["scene_domain_split"] == "unseen"]
                unseen_unseen = seen_unseen
                seen_scene_coverage = len({s["scene_domain"] for s in seen_seen})
                unseen_scene_coverage = len({s["scene_domain"] for s in seen_unseen})
                if (
                    len(seen_seen) >= self.args.min_seen_seen_slots_per_seen_task
                    and len(seen_unseen) >= self.args.min_seen_unseen_slots_per_seen_task
                    and seen_scene_coverage >= self.args.min_seen_scene_coverage_per_seen_task
                    and unseen_scene_coverage >= self.args.min_unseen_scene_coverage_per_seen_task
                ):
                    seen_candidates.append((task, len(seen_seen), len(seen_unseen), seen_scene_coverage, unseen_scene_coverage))
                if (
                    len(unseen_unseen) >= self.args.min_unseen_unseen_slots_per_unseen_task
                    and unseen_scene_coverage >= self.args.min_unseen_scene_coverage_per_unseen_task
                ):
                    unseen_candidates.append((task, len(unseen_unseen), unseen_scene_coverage))
                if (
                    len(seen_candidates) >= seen_need + buffer
                    and len(unseen_candidates) >= unseen_need + seen_need + buffer
                ):
                    break

            seen_candidates.sort(key=lambda x: (-x[3], -x[4], -x[1], x[0][1]))
            chosen_seen = [x[0] for x in seen_candidates[: seen_quotas[family]]]
            if len(chosen_seen) < seen_quotas[family]:
                gap["families"][family] = {
                    "seen_candidates": len(seen_candidates),
                    "seen_need": seen_quotas[family],
                    "unseen_candidates": len(unseen_candidates),
                }
                _dump_json(self.args.output_dir / "balanced_gap_report.json", gap)
                raise RuntimeError(f"not enough seen tasks for {family}")
            selected_seen.update(chosen_seen)
            print(f"  {family}: seen candidates {len(seen_candidates)}, chosen {len(chosen_seen)}", flush=True)

            unseen_need = TARGETS[family] - seen_quotas[family]
            unseen_candidates = [item for item in unseen_candidates if item[0] not in selected_seen]
            unseen_candidates.sort(key=lambda x: (-x[2], -x[1], x[0][1]))
            chosen_unseen = [x[0] for x in unseen_candidates[:unseen_need]]
            if len(chosen_unseen) < unseen_need:
                gap["families"][family] = {
                    "seen_candidates": len(seen_candidates),
                    "unseen_candidates": len(unseen_candidates),
                    "unseen_need": unseen_need,
                }
                _dump_json(self.args.output_dir / "balanced_gap_report.json", gap)
                raise RuntimeError(f"not enough unseen tasks for {family}")
            selected_unseen.update(chosen_unseen)
            print(f"  {family}: unseen candidates {len(unseen_candidates)}, chosen {len(chosen_unseen)}", flush=True)
            gap["families"][family] = {
                "seen_candidates": len(seen_candidates),
                "chosen_seen": len(chosen_seen),
                "unseen_candidates": len(unseen_candidates),
                "chosen_unseen": len(chosen_unseen),
            }

        gap["skipped_candidate_reasons"] = dict(self.reason_counts)
        _dump_json(self.args.output_dir / "balanced_gap_report.json", gap)
        return selected_seen, selected_unseen

    def pools(self, seen_tasks: set[Task], unseen_tasks: set[Task]) -> dict[str, list[dict[str, Any]]]:
        out = {"seen_seen": [], "seen_unseen": [], "unseen_unseen": [], "unseen_seen": []}
        for task in seen_tasks:
            for slot in self.collect_slots(task):
                if slot["scene_domain_split"] == "seen":
                    out["seen_seen"].append(slot)
                else:
                    out["seen_unseen"].append(slot)
        for task in unseen_tasks:
            for slot in self.collect_slots(task):
                if slot["scene_domain_split"] == "seen":
                    out["unseen_seen"].append(slot)
                else:
                    out["unseen_unseen"].append(slot)
        return out

    def tag_rows(self, slot: dict[str, Any], split: str, task_split: str) -> list[dict[str, Any]]:
        rows = []
        for row in slot["rows"]:
            out = copy.deepcopy(row)
            meta = out["_meta"]
            meta["split"] = split
            meta["task_split"] = task_split
            meta["task_family"] = slot["family"]
            meta["task_id"] = slot["task_id"]
            rows.append(out)
        return rows

    def materialize(self) -> dict[str, Any]:
        self.load_domains()
        candidates = self.candidate_tasks()
        seen_tasks, unseen_tasks = self.select_tasks(candidates)
        pools = self.pools(seen_tasks, unseen_tasks)

        seen_domain_order = sorted(self.seen_domains, key=lambda d: d["domain_id"])
        scene_dir_for_domain = {d["domain_id"]: f"scene_{i}" for i, d in enumerate(seen_domain_order)}

        print(f"pool sizes before sampling: { {k: len(v) for k, v in pools.items()} }", flush=True)
        train_scene_targets = {
            d["domain_id"]: self.args.train_episodes // len(seen_domain_order) + 1
            for d in seen_domain_order
        }
        print("sampling train from seen_task x seen_scene pool", flush=True)
        train = _choose_balanced(
            pools["seen_seen"],
            self.args.train_episodes,
            seed=self.args.seed + 17,
            min_per_task=self.args.train_min_per_seen_task,
            scene_targets=train_scene_targets,
        )
        blocked = {slot["trajectory_id"] for slot in train}
        train_row_signatures = set().union(
            *[_slot_row_signatures(slot) for slot in train]
        ) if train else set()

        print("sampling eval A from seen_task x seen_scene pool", flush=True)
        eval_a = _choose_balanced(
            pools["seen_seen"],
            self.args.eval_a_episodes,
            seed=self.args.seed + 11,
            blocked=blocked,
            blocked_row_signatures=(
                train_row_signatures if self.args.prevent_train_test_row_overlap else None
            ),
            min_per_task=self.args.eval_a_min_per_seen_task,
            scene_targets={d["domain_id"]: self.args.eval_a_episodes // len(seen_domain_order) + 1 for d in seen_domain_order},
        )
        blocked.update(slot["trajectory_id"] for slot in eval_a)

        print("sampling eval B from seen_task x unseen_scene pool", flush=True)
        eval_b = _choose_balanced(
            pools["seen_unseen"],
            self.args.eval_b_episodes,
            seed=self.args.seed + 23,
            blocked_row_signatures=(
                train_row_signatures if self.args.prevent_train_test_row_overlap else None
            ),
            min_per_task=self.args.eval_b_min_per_seen_task,
        )
        print("sampling eval C from unseen_task x unseen_scene pool", flush=True)
        eval_c_family_targets = _family_eval_targets(
            unseen_tasks,
            self.args.eval_c_episodes,
            self.args.eval_c_min_per_unseen_task,
        )
        eval_c = _choose_balanced(
            pools["unseen_unseen"],
            self.args.eval_c_episodes,
            seed=self.args.seed + 29,
            blocked_row_signatures=(
                train_row_signatures if self.args.prevent_train_test_row_overlap else None
            ),
            min_per_task=self.args.eval_c_min_per_unseen_task,
            family_targets=eval_c_family_targets,
        )

        train_by_scene_dir: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for slot in train:
            scene_dir = scene_dir_for_domain[slot["scene_domain"]]
            train_by_scene_dir[scene_dir].extend(self.tag_rows(slot, "seen_seen", "seen"))

        test_buckets = {
            "seen_seen": [row for slot in eval_a for row in self.tag_rows(slot, "seen_seen", "seen")],
            "seen_unseen": [row for slot in eval_b for row in self.tag_rows(slot, "seen_unseen", "seen")],
            "unseen_unseen": [row for slot in eval_c for row in self.tag_rows(slot, "unseen_unseen", "unseen")],
            "unseen_seen": [],
        }

        manifest = self.manifest(seen_tasks, unseen_tasks, scene_dir_for_domain, eval_c_family_targets)
        quality = self.quality(train_by_scene_dir, test_buckets, pools, manifest)
        return {
            "train_by_scene_dir": train_by_scene_dir,
            "test_buckets": test_buckets,
            "manifest": manifest,
            "quality": quality,
        }

    def manifest(
        self,
        seen_tasks: set[Task],
        unseen_tasks: set[Task],
        scene_dir_for_domain: dict[str, str],
        eval_c_family_targets: dict[str, int],
    ) -> dict[str, Any]:
        return {
            "dataset_mode": DATASET_MODE,
            "seed": self.args.seed,
            "claim": "paper-compatible fixed-seed reconstruction, not author-exact split",
            "split_rule": {
                "train": "seen_task intersect seen_scene",
                "eval_a": "seen_task intersect seen_scene, episode-held-out from train",
                "eval_b": "seen_task intersect unseen_scene",
                "eval_c": "unseen_task intersect unseen_scene only; not residual sampling",
            },
            "semantic_gate": self.args.semantic_gate,
            "observation_preprocessing": {
                "mode": self.args.observation_mode,
                "compact_num_edges": self.args.compact_num_edges,
                "compact_next_mode": self.args.compact_next_mode,
                "prevent_train_test_row_overlap": self.args.prevent_train_test_row_overlap,
            },
            "world_model_auxiliary_tasks": {
                "behavior_cloning": "predict action from instruction and current observation",
                "affordance": "predict one feasible action from current observation",
                "dynamics": "predict next observation from instruction, current observation, and executed action",
            },
            "assumptions": [
                "paper does not publish exact task, scene, or episode ids",
                "N=6 world models is interpreted as six seen scene domains",
                "a scene domain groups multiple official init-graph variants from one base apartment",
                "expert episodes are planner rollouts on official VirtualHome init graphs",
                "sampling is soft-balanced and does not require a full task-scene Cartesian product",
            ],
            "episode_targets": {
                "train": self.args.train_episodes,
                "eval_seen_seen": self.args.eval_a_episodes,
                "eval_seen_unseen": self.args.eval_b_episodes,
                "eval_unseen_unseen": self.args.eval_c_episodes,
                "total": self.args.train_episodes
                + self.args.eval_a_episodes
                + self.args.eval_b_episodes
                + self.args.eval_c_episodes,
            },
            "eval_c_family_targets": eval_c_family_targets,
            "paper_task_targets": TARGETS,
            "seen_family_quotas": _seen_family_quotas(self.args.seen_instructions),
            "scene_domains": [
                {
                    "domain_id": d["domain_id"],
                    "base": d["base"],
                    "scene_split": d["scene_split"],
                    "world_model_dir": scene_dir_for_domain.get(d["domain_id"]),
                    "variants": d["variants"],
                }
                for d in self.domains
            ],
            "selected_tasks": [
                {
                    "task_id": _task_id(task),
                    "family": task[0],
                    "args": list(task[1]),
                    "task_split": "seen" if task in seen_tasks else "unseen",
                    "goal_triple": list(_goal_triple(task[0], task[1])),
                }
                for task in sorted(seen_tasks | unseen_tasks)
            ],
        }

    def quality(
        self,
        train_by_scene_dir: dict[str, list[dict[str, Any]]],
        test_buckets: dict[str, list[dict[str, Any]]],
        pools: dict[str, list[dict[str, Any]]],
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        all_rows = [row for rows in train_by_scene_dir.values() for row in rows]
        for rows in test_buckets.values():
            all_rows.extend(rows)
        train_ids = {row["_meta"]["trajectory_id"] for rows in train_by_scene_dir.values() for row in rows}
        test_ids = {row["_meta"]["trajectory_id"] for rows in test_buckets.values() for row in rows}
        train_row_signatures = {
            _row_signature(row)
            for rows in train_by_scene_dir.values()
            for row in rows
        }
        test_row_signatures = {
            _row_signature(row)
            for rows in test_buckets.values()
            for row in rows
        }
        all_ids = train_ids | test_ids
        per_wm = {}
        for scene_dir, rows in sorted(train_by_scene_dir.items()):
            ids = {row["_meta"]["trajectory_id"] for row in rows}
            per_wm[scene_dir] = {
                "episodes": len(ids),
                "rows": len(rows),
                "families": dict(Counter(row["_meta"]["task_family"] for row in rows)),
                "tasks": len({row["_meta"]["task_id"] for row in rows}),
            }
        return {
            "dataset_mode": DATASET_MODE,
            "data_root": str(self.args.output_dir),
            "total_episodes": len(all_ids),
            "total_rows": len(all_rows),
            "train_test_trajectory_overlap": len(train_ids & test_ids),
            "train_test_exact_row_overlap": len(train_row_signatures & test_row_signatures),
            "pool_episode_counts_before_sampling": {k: len(v) for k, v in pools.items()},
            "split_episode_counts": {
                "train": len(train_ids),
                **{split: len({r["_meta"]["trajectory_id"] for r in rows}) for split, rows in test_buckets.items()},
            },
            "split_row_counts": {
                "train": sum(len(rows) for rows in train_by_scene_dir.values()),
                **{split: len(rows) for split, rows in test_buckets.items()},
            },
            "selected_task_counts": dict(Counter(t["family"] for t in manifest["selected_tasks"])),
            "selected_seen_task_counts": dict(Counter(t["family"] for t in manifest["selected_tasks"] if t["task_split"] == "seen")),
            "selected_unseen_task_counts": dict(Counter(t["family"] for t in manifest["selected_tasks"] if t["task_split"] == "unseen")),
            "transitions_per_world_model": per_wm,
            "action_counts": dict(Counter(row["action"].split()[0] for row in all_rows)),
            "family_row_counts": dict(Counter(row["_meta"]["task_family"] for row in all_rows)),
            "init_graph_variants_available": len(self.scene_inits),
            "skipped_candidate_reasons": dict(self.reason_counts),
        }

    def write(self, materialized: dict[str, Any]) -> None:
        out = self.args.output_dir
        if out.exists() and any(out.iterdir()):
            if not self.args.overwrite:
                raise FileExistsError(f"{out} is not empty; pass --overwrite to replace")
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        _dump_json(out / "virtualhome_manifest.json", materialized["manifest"])
        _dump_json(out / "quality_report.json", materialized["quality"])
        _dump_json(out / "scene_inits_used.json", self.scene_inits)
        _dump_json(
            out / "task_split.json",
            {task["task_id"]: task["task_split"] for task in materialized["manifest"]["selected_tasks"]},
        )
        _dump_json(
            out / "scene_split.json",
            {
                d["domain_id"]: {
                    "scene_split": d["scene_split"],
                    "world_model_dir": d.get("world_model_dir"),
                    "base": d["base"],
                    "variants": [v["variant_key"] for v in d["variants"]],
                }
                for d in materialized["manifest"]["scene_domains"]
            },
        )

        rng = random.Random(self.args.seed)
        for split, filename in ROOT_SPLIT_FILES.items():
            rows = materialized["test_buckets"].get(split, [])[:]
            rng.shuffle(rows)
            with (out / filename).open("w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

        for dirname, target in EVAL_LINKS.items():
            d = out / dirname
            d.mkdir(exist_ok=True)
            link = d / "test.jsonl"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(Path("..") / target)

        for scene_dir, rows in materialized["train_by_scene_dir"].items():
            d = out / scene_dir
            d.mkdir(exist_ok=True)
            rows = rows[:]
            rng.shuffle(rows)
            with (d / "train.jsonl").open("w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            link = d / "test.jsonl"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(Path("..") / "test_seen_task_seen_scene.jsonl")

    def run(self) -> None:
        materialized = self.materialize()
        self.write(materialized)
        quality = materialized["quality"]
        print("balanced VirtualHome reconstruction written")
        print(f"  output_dir: {self.args.output_dir}")
        print(f"  episodes: {quality['total_episodes']}")
        print(f"  rows: {quality['total_rows']}")
        print(f"  split_episode_counts: {quality['split_episode_counts']}")
        print(f"  pool_episode_counts_before_sampling: {quality['pool_episode_counts_before_sampling']}")
        print(f"  train_test_trajectory_overlap: {quality['train_test_trajectory_overlap']}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--vh-src", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scene-domains", type=int, default=20)
    p.add_argument("--seen-scenes", type=int, default=6)
    p.add_argument("--variants-per-domain", type=int, default=8)
    p.add_argument("--seen-instructions", type=int, default=16)
    p.add_argument("--candidate-multiplier", type=int, default=16)
    p.add_argument("--task-candidate-buffer", type=int, default=8)
    p.add_argument(
        "--semantic-gate",
        choices=[
            "none",
            "source_unique",
            "source_unique_target_room_unique",
            "source_and_target_unique",
        ],
        default="source_unique",
        help=(
            "Filter candidate episodes before split sampling. source_unique keeps "
            "unary targets unique and binary sources unique; "
            "source_unique_target_room_unique also requires binary targets to appear "
            "in only one room; source_and_target_unique also requires binary targets "
            "to have one graph node."
        ),
    )
    p.add_argument(
        "--observation-mode",
        choices=["full", "tmow_compact"],
        default="full",
        help=(
            "Render final JSONL observations as full class-level graph triples or "
            "deterministic TMoW-style compact triples before split sampling."
        ),
    )
    p.add_argument("--compact-num-edges", type=int, default=17)
    p.add_argument(
        "--compact-next-mode",
        choices=["delta", "compact"],
        default="delta",
        help="For tmow_compact, supervise next observation as a delta or compact full state.",
    )
    p.add_argument(
        "--prevent-train-test-row-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When observations are compacted, reject sampled slots whose final "
            "(instruction, observation, action, next_observation) text would appear "
            "in both train and test."
        ),
    )
    p.add_argument("--train-episodes", type=int, default=384)
    p.add_argument("--eval-a-episodes", type=int, default=96)
    p.add_argument("--eval-b-episodes", type=int, default=224)
    p.add_argument("--eval-c-episodes", type=int, default=319)
    p.add_argument("--train-min-per-seen-task", type=int, default=1)
    p.add_argument("--eval-a-min-per-seen-task", type=int, default=1)
    p.add_argument("--eval-b-min-per-seen-task", type=int, default=1)
    p.add_argument("--eval-c-min-per-unseen-task", type=int, default=1)
    p.add_argument("--min-seen-seen-slots-per-seen-task", type=int, default=2)
    p.add_argument("--min-seen-unseen-slots-per-seen-task", type=int, default=1)
    p.add_argument("--min-unseen-unseen-slots-per-unseen-task", type=int, default=1)
    p.add_argument("--min-seen-scene-coverage-per-seen-task", type=int, default=1)
    p.add_argument("--min-unseen-scene-coverage-per-seen-task", type=int, default=1)
    p.add_argument("--min-unseen-scene-coverage-per-unseen-task", type=int, default=1)
    p.add_argument("--min-probe-successes", type=int, default=1)
    p.add_argument("--max-probe-tasks", type=int, default=30)
    p.add_argument("--max-scan-per-base", type=int, default=3000)
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    Builder(args).run()


if __name__ == "__main__":
    main()
