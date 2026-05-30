#!/usr/bin/env python3
"""WorMI-aligned VirtualHome dataset builder.

Strict-WorMI reconstruction. Key design choices:
  - Task list: the 78-task VirtualHome canonical set, hard-coded via
    `tools/tmow_const.py` (which mirrors the public TMoW const.py). This
    replaces the old coverage-greedy `build_candidate_instructions` path.
  - Seen-task split: TMoW stratified index `SEEN_TASKS = [0, 4, ..., 74]`,
    16 tasks, family quota 2 turnon + 2 open + 6 puton + 6 placein.
  - Unseen tasks: the remaining 62 tasks.
  - Observation: full graph triples via `format_observation()` from the
    existing `tools/build_virtualhome_dataset.py`. This matches WorMI paper
    Figure A.2 schema. No BM25 retrieval, no compact-K subset, no
    augmentation.
  - Trajectory: deterministic expert program via EvolvingGraph
    (`_execute_paperlike_candidate`) on a fixed init-graph variant.
  - Scene split: 20 distinct scenes (paper). Default 6 seen / 14 unseen,
    organized as 20 domains, each domain composed of multiple official VH
    init-graph variants from the same base apartment. The dataset count is
    materialized through these variants, not through stochastic retrieval.
  - Auxiliary tasks: behavior-cloning + dynamics + affordance are expanded
    by `wormi/datasets/virtualhome.py` at load time. The builder writes only
    one row per transition.
  - Strict task balance: train holds exactly `train_episodes / 16` trajectories
    per seen task. This is the single mechanism that prevents the previous
    walk-target collapse.

Output layout (compatible with `wormi/curricula`):

    output_dir/
      scene_0/train.jsonl    # world model 0 (seen domain 0)
      ...
      scene_5/train.jsonl
      test_seen_task_seen_scene.jsonl       # eval A
      test_seen_task_unseen_scene.jsonl     # eval B
      test_unseen_task_unseen_scene.jsonl   # eval C
      test_unseen_task_seen_scene.jsonl     # always empty (paper does not use)
      eval_col_{1,2,3}_<...>/test.jsonl     # symlinks
      virtualhome_manifest.json
      quality_report.json
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
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.build_virtualhome_dataset import (  # noqa: E402
    _bootstrap_evolving_graph,
    _execute_paperlike_candidate,
    _graph_probe_successes,
    _goal_triple,
    build_candidate_instructions,
    find_first_id,
    format_observation,
    instruction_text,
    TARGETS,
)

DATASET_MODE = "wormi_paper_aligned_v1"

DEFAULT_BASE_DOMAIN_COUNTS = [3, 3, 3, 3, 3, 3, 2]  # 20 domains over 7 bases

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
    counts = [0] * num_bases
    for i in range(scene_domains):
        counts[i % num_bases] += 1
    return counts


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
        self.slot_cache: dict[tuple[int, str], dict[str, Any] | None] = {}
        self.reason_counts: Counter[str] = Counter()

    # -------------------------------------------------- task selection ----
    def select_tasks(self) -> tuple[list[tuple[str, tuple[str, ...]]],
                                    list[tuple[str, tuple[str, ...]]]]:
        """Pick 78 tasks from VH properties + stratified-by-source-class seen.

        Step 1: build candidate pool from properties_data.json. The same
                joint-scene-coverage ranking used by the original WorMI
                builder produces a candidate pool larger than 78.
        Step 2: take exactly TARGETS[fam] tasks per family (9/7/30/32) by
                walking the candidate pool with source-class diversity.
                The first TARGETS[fam] eligible tasks per family form the
                final 78. This replaces the old "top-K by coverage" cut.
        Step 3: pick the seen 16 (2/2/6/6) by stratified-by-source-class
                greedy: iterate the 78 in family order, prefer tasks whose
                source class has not been picked yet for the seen set.
                Within ties, prefer underrepresented target class. This is
                the single mechanism that prevents the first-action collapse.
        """
        class_sets = [
            {n["class_name"] for n in g["nodes"]} for g in self.scene_inits.values()
        ]
        cand_pool = build_candidate_instructions(
            self.properties, class_sets,
            candidate_multiplier=self.args.candidate_multiplier,
        )
        by_family: dict[str, list[tuple[str, tuple[str, ...]]]] = defaultdict(list)
        for fam, args in cand_pool:
            by_family[fam].append((fam, args))

        # Step 2: pick exactly TARGETS[fam] per family. Walk cand_pool order
        # (which is coverage-ranked) but enforce per-family source-class cap
        # so the 78 final tasks remain diverse.
        all_tasks: list[tuple[str, tuple[str, ...]]] = []
        for fam in TARGETS:
            quota = TARGETS[fam]
            source_caps = max(1, quota // 4)  # at most quota/4 per source class
            picked: list[tuple[str, tuple[str, ...]]] = []
            src_count: Counter[str] = Counter()
            for fam_, args in by_family[fam]:
                if len(picked) >= quota:
                    break
                src = args[0]
                if src_count[src] >= source_caps:
                    continue
                picked.append((fam_, args))
                src_count[src] += 1
            # Fallback: if cap was too tight, fill from remaining
            if len(picked) < quota:
                seen = {t for t in picked}
                for fam_, args in by_family[fam]:
                    if len(picked) >= quota:
                        break
                    if (fam_, args) in seen:
                        continue
                    picked.append((fam_, args))
                    seen.add((fam_, args))
            if len(picked) < quota:
                raise RuntimeError(f"family {fam}: only {len(picked)}/{quota} tasks available")
            all_tasks.extend(picked)
        print(f"selected {len(all_tasks)} tasks: families = "
              f"{ {f: sum(1 for fa,_ in all_tasks if fa==f) for f in TARGETS} }",
              flush=True)

        # Step 3: stratified-by-source seen-task split (2/2/6/6).
        seen_quotas = {"turnon": 2, "open": 2, "puton": 6, "placein": 6}
        seen_tasks_set: set[tuple[str, tuple[str, ...]]] = set()
        seen_src_count: Counter[str] = Counter()
        seen_tgt_count: Counter[str] = Counter()
        for fam in TARGETS:
            quota = seen_quotas[fam]
            fam_tasks = [t for t in all_tasks if t[0] == fam]
            picked = 0
            remaining = list(enumerate(fam_tasks))  # preserve original order via idx
            while picked < quota and remaining:
                def key(item):
                    _idx, t = item
                    src = t[1][0]
                    tgt = t[1][1] if len(t[1]) > 1 else ""
                    return (seen_src_count[src], seen_tgt_count[tgt], _idx)
                remaining.sort(key=key)
                _idx, t = remaining.pop(0)
                seen_tasks_set.add(t)
                src = t[1][0]
                seen_src_count[src] += 1
                if len(t[1]) > 1:
                    seen_tgt_count[t[1][1]] += 1
                picked += 1
        seen_tasks = [t for t in all_tasks if t in seen_tasks_set]
        unseen_tasks = [t for t in all_tasks if t not in seen_tasks_set]
        print(f"seen tasks ({len(seen_tasks)}): "
              f"{ {f: sum(1 for fa,_ in seen_tasks if fa==f) for f in TARGETS} }",
              flush=True)
        print(f"seen sources: {sorted(seen_src_count.items(), key=lambda x: -x[1])}",
              flush=True)
        return seen_tasks, unseen_tasks

    # ------------------------------------------------------------- scenes ---
    def load_domains(self) -> None:
        init_root = self.args.raw_dir / "init_and_final_graphs"
        bases = sorted(p for p in init_root.iterdir() if p.is_dir())
        if not bases:
            raise FileNotFoundError(init_root)
        counts = _base_domain_counts(len(bases), self.args.scene_domains)
        domain_idx = 0
        for base_dir, domain_count in zip(bases, counts):
            need = domain_count * self.args.variants_per_domain
            print(f"selecting init variants for {base_dir.name}: need {need}", flush=True)
            candidates = sorted(base_dir.rglob("*.json"))
            self.rng.shuffle(candidates)
            selected: list[dict[str, Any]] = []
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
                    succ = _graph_probe_successes(
                        graph, self.properties, self.eg,
                        max_probe_tasks=self.args.max_probe_tasks,
                    )
                    if succ < self.args.min_probe_successes:
                        continue
                selected.append({"path": str(path), "graph": graph})
            if len(selected) < need:
                raise RuntimeError(
                    f"{base_dir.name}: only {len(selected)}/{need} valid variants "
                    f"after scanning {scanned}"
                )
            offset = 0
            for _ in range(domain_count):
                did = f"scene_domain_{domain_idx:02d}"
                variants = []
                for v_idx in range(self.args.variants_per_domain):
                    item = selected[offset]
                    offset += 1
                    key = f"{base_dir.name}__d{domain_idx:02d}_v{v_idx:02d}"
                    self.scene_inits[key] = item["graph"]
                    variants.append({"variant_key": key, "variant_index": v_idx,
                                     "source_path": item["path"]})
                self.domains.append({
                    "domain_id": did, "base": base_dir.name, "variants": variants,
                })
                domain_idx += 1
            print(f"  -> {base_dir.name}: scanned {scanned}, selected {len(selected)}", flush=True)

        # Seen / unseen domain split — deterministic by seed.
        order = self.domains[:]
        self.rng.shuffle(order)
        seen_ids = {d["domain_id"] for d in order[: self.args.seen_scenes]}
        for d in self.domains:
            d["scene_split"] = "seen" if d["domain_id"] in seen_ids else "unseen"
        self.seen_domains = [d for d in self.domains if d["scene_split"] == "seen"]
        self.unseen_domains = [d for d in self.domains if d["scene_split"] == "unseen"]
        print(
            f"scene domains: {len(self.domains)} ({len(self.seen_domains)} seen, "
            f"{len(self.unseen_domains)} unseen)", flush=True,
        )

    # ----------------------------------------------------------- execute ---
    def execute_slot(self, task: tuple[str, tuple[str, ...]],
                     variant_key: str) -> dict[str, Any] | None:
        cache_key = (task, variant_key)
        if cache_key in self.slot_cache:
            return self.slot_cache[cache_key]
        family, args = task
        graph = self.scene_inits[variant_key]
        # Quick prefilter: if any required class is missing from the graph,
        # the expert program will fail with missing_object. Skip the heavy
        # EvolvingGraph execution.
        for cls in args:
            if find_first_id(graph, cls) is None:
                self.reason_counts[f"prefilter_missing:{cls}"] += 1
                self.slot_cache[cache_key] = None
                return None
        rows, reason = _execute_paperlike_candidate(
            family, args, variant_key, graph,
            self.EnvironmentGraph, self.read_script, self.ScriptExecutor,
        )
        if rows is None:
            self.reason_counts[reason or "unknown"] += 1
            self.slot_cache[cache_key] = None
            return None
        if any(r["observation"] == r["next_observation"] for r in rows):
            self.reason_counts["unchanged_transition"] += 1
            self.slot_cache[cache_key] = None
            return None
        task_id = f"{family}:{'|'.join(args)}"
        trajectory_id = f"{variant_key}:{task_id}"
        for row in rows:
            meta = row["_meta"]
            meta["task_family"] = family
            meta["task_args"] = list(args)
            meta["task_id"] = task_id
            meta["scene"] = variant_key
            meta["trajectory_id"] = trajectory_id
            meta["protocol"] = DATASET_MODE
        slot = {
            "task": task, "task_id": task_id, "family": family, "args": args,
            "variant_key": variant_key, "trajectory_id": trajectory_id,
            "rows": rows,
        }
        self.slot_cache[cache_key] = slot
        return slot

    # ----------------------------------------------------------- sampling ---
    def materialize(self) -> dict[str, Any]:
        self.load_domains()
        seen_tasks, unseen_tasks = self.select_tasks()
        self.seen_tasks = seen_tasks
        self.unseen_tasks = unseen_tasks

        pools: dict[str, list[dict[str, Any]]] = {
            "seen_task_seen_scene": [],
            "seen_task_unseen_scene": [],
            "unseen_task_seen_scene": [],
            "unseen_task_unseen_scene": [],
        }
        seen_domain_keys = {d["domain_id"] for d in self.seen_domains}
        all_variants: list[tuple[str, str]] = []
        for d in self.domains:
            for v in d["variants"]:
                all_variants.append((v["variant_key"], d["domain_id"]))

        for ti, task in enumerate(seen_tasks + unseen_tasks):
            tag_task = "seen_task" if task in set(seen_tasks) else "unseen_task"
            for variant_key, domain_id in all_variants:
                slot = self.execute_slot(task, variant_key)
                if slot is None:
                    continue
                slot["scene_domain"] = domain_id
                slot["task_split"] = tag_task
                slot["scene_split"] = "seen_scene" if domain_id in seen_domain_keys else "unseen_scene"
                key = f"{slot['task_split']}_{slot['scene_split']}"
                pools[key].append(slot)
            if (ti + 1) % 10 == 0:
                print(f"  enumerated {ti+1}/{len(seen_tasks)+len(unseen_tasks)} tasks, "
                      f"pool sizes: { {k: len(v) for k, v in pools.items()} }", flush=True)

        print(f"FINAL pool sizes: { {k: len(v) for k, v in pools.items()} }", flush=True)

        train_per_task = self.args.train_episodes // len(seen_tasks)
        eval_a_per_task = self.args.eval_a_episodes // len(seen_tasks)
        eval_b_per_task = self.args.eval_b_episodes // len(seen_tasks)
        rng = random.Random(self.args.seed + 17)
        train_slots: list[dict[str, Any]] = []
        eval_a_slots: list[dict[str, Any]] = []
        eval_b_slots: list[dict[str, Any]] = []

        per_task_seen_seen: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for s in pools["seen_task_seen_scene"]:
            per_task_seen_seen[s["task_id"]].append(s)
        per_task_seen_unseen: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for s in pools["seen_task_unseen_scene"]:
            per_task_seen_unseen[s["task_id"]].append(s)

        for task in seen_tasks:
            tid = f"{task[0]}:{'|'.join(task[1])}"
            avail = per_task_seen_seen[tid][:]
            rng.shuffle(avail)
            take_a = min(eval_a_per_task, len(avail))
            eval_a_slots.extend(avail[:take_a])
            avail = avail[take_a:]
            take_tr = min(train_per_task, len(avail))
            train_slots.extend(avail[:take_tr])
            if take_a < eval_a_per_task or take_tr < train_per_task:
                print(
                    f"  WARN {tid}: seen_seen pool={len(per_task_seen_seen[tid])} "
                    f"-> train {take_tr}/{train_per_task} eval_a {take_a}/{eval_a_per_task}",
                    flush=True,
                )
            avail_b = per_task_seen_unseen[tid][:]
            rng.shuffle(avail_b)
            take_b = min(eval_b_per_task, len(avail_b))
            eval_b_slots.extend(avail_b[:take_b])
            if take_b < eval_b_per_task:
                print(
                    f"  WARN {tid}: seen_unseen pool={len(per_task_seen_unseen[tid])} "
                    f"-> eval_b {take_b}/{eval_b_per_task}",
                    flush=True,
                )

        per_task_unseen_unseen: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for s in pools["unseen_task_unseen_scene"]:
            per_task_unseen_unseen[s["task_id"]].append(s)
        eval_c_per_task_floor = max(1, self.args.eval_c_episodes // len(unseen_tasks))
        eval_c_slots: list[dict[str, Any]] = []
        for task in unseen_tasks:
            tid = f"{task[0]}:{'|'.join(task[1])}"
            avail = per_task_unseen_unseen[tid][:]
            if not avail:
                continue
            rng.shuffle(avail)
            eval_c_slots.extend(avail[:eval_c_per_task_floor])
        if len(eval_c_slots) < self.args.eval_c_episodes:
            need = self.args.eval_c_episodes - len(eval_c_slots)
            chosen_ids = {s["trajectory_id"] for s in eval_c_slots}
            leftover = []
            for task in unseen_tasks:
                tid = f"{task[0]}:{'|'.join(task[1])}"
                for s in per_task_unseen_unseen[tid][eval_c_per_task_floor:]:
                    if s["trajectory_id"] not in chosen_ids:
                        leftover.append(s)
            rng.shuffle(leftover)
            eval_c_slots.extend(leftover[:need])

        # Materialize rows + tag splits + organize by scene_domain.
        seen_domain_order = sorted(self.seen_domains, key=lambda d: d["domain_id"])
        scene_dir_for_domain = {d["domain_id"]: f"scene_{i}"
                                for i, d in enumerate(seen_domain_order)}

        train_by_scene_dir: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for slot in train_slots:
            sd = scene_dir_for_domain[slot["scene_domain"]]
            for row in slot["rows"]:
                out = copy.deepcopy(row)
                out["_meta"]["split"] = "train_seen_task_seen_scene"
                train_by_scene_dir[sd].append(out)

        test_buckets = {"seen_seen": [], "seen_unseen": [],
                        "unseen_unseen": [], "unseen_seen": []}
        # Per-scene eval_a rows: scene_dir -> rows for that seen domain only.
        # Used by write() to produce scene_N/test.jsonl containing only that
        # world model's own held-out seen-task seen-scene episodes, instead of
        # a symlink to the global pool.
        eval_a_by_scene_dir: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for slot in eval_a_slots:
            sd = scene_dir_for_domain[slot["scene_domain"]]
            for row in slot["rows"]:
                out = copy.deepcopy(row)
                out["_meta"]["split"] = "eval_seen_task_seen_scene"
                test_buckets["seen_seen"].append(out)
                eval_a_by_scene_dir[sd].append(out)
        for slot in eval_b_slots:
            for row in slot["rows"]:
                out = copy.deepcopy(row)
                out["_meta"]["split"] = "eval_seen_task_unseen_scene"
                test_buckets["seen_unseen"].append(out)
        for slot in eval_c_slots:
            for row in slot["rows"]:
                out = copy.deepcopy(row)
                out["_meta"]["split"] = "eval_unseen_task_unseen_scene"
                test_buckets["unseen_unseen"].append(out)

        manifest = self._manifest(scene_dir_for_domain, train_slots,
                                  eval_a_slots, eval_b_slots, eval_c_slots, pools)
        quality = self._quality(train_by_scene_dir, test_buckets, pools)
        return {
            "train_by_scene_dir": train_by_scene_dir,
            "eval_a_by_scene_dir": dict(eval_a_by_scene_dir),
            "test_buckets": test_buckets,
            "manifest": manifest,
            "quality": quality,
        }

    # ----------------------------------------------------------- manifest ---
    def _manifest(self, scene_dir_for_domain, train_slots, eval_a, eval_b, eval_c, pools):
        return {
            "dataset_mode": DATASET_MODE,
            "seed": self.args.seed,
            "claim": (
                "WorMI-aligned reconstruction. Task list = VH properties-driven "
                "candidate pool, 78 tasks selected with per-source-class diversity. "
                "Seen 16 tasks selected stratified-by-source-class. "
                "20 scenes / 6 seen / 14 unseen. "
                "Observation = WorMI paper Figure A.2 full graph triples."
            ),
            "task_list_source": (
                "tools/build_virtualhome_dataset::build_candidate_instructions, "
                "constrained by per-source diversity cap"
            ),
            "task_counts": {
                "total": len(self.seen_tasks) + len(self.unseen_tasks),
                "seen": len(self.seen_tasks),
                "unseen": len(self.unseen_tasks),
                "seen_tasks": [f"{f}:{'|'.join(a)}" for f, a in self.seen_tasks],
                "unseen_tasks": [f"{f}:{'|'.join(a)}" for f, a in self.unseen_tasks],
            },
            "scene_split": {
                "total": len(self.domains),
                "seen": len(self.seen_domains),
                "unseen": len(self.unseen_domains),
                "variants_per_domain": self.args.variants_per_domain,
            },
            "episode_targets": {
                "train": self.args.train_episodes,
                "eval_a_seen_seen": self.args.eval_a_episodes,
                "eval_b_seen_unseen": self.args.eval_b_episodes,
                "eval_c_unseen_unseen": self.args.eval_c_episodes,
                "total": (self.args.train_episodes + self.args.eval_a_episodes
                          + self.args.eval_b_episodes + self.args.eval_c_episodes),
            },
            "episode_actual": {
                "train": len(train_slots),
                "eval_a_seen_seen": len(eval_a),
                "eval_b_seen_unseen": len(eval_b),
                "eval_c_unseen_unseen": len(eval_c),
            },
            "pool_sizes": {k: len(v) for k, v in pools.items()},
            "scene_domains": [
                {"domain_id": d["domain_id"], "base": d["base"],
                 "scene_split": d["scene_split"],
                 "world_model_dir": scene_dir_for_domain.get(d["domain_id"]),
                 "variants": [v["variant_key"] for v in d["variants"]]}
                for d in self.domains
            ],
            "skipped_candidate_reasons": dict(self.reason_counts),
            "world_model_auxiliary_tasks": {
                "behavior_cloning": "predict action from instruction + observation",
                "dynamics": "predict next observation from instruction + observation + action",
                "affordance": "predict one feasible action from observation",
            },
        }

    # ------------------------------------------------------------ quality ---
    def _quality(self, train_by_scene, test_buckets, pools):
        all_rows = [r for rs in train_by_scene.values() for r in rs]
        for rs in test_buckets.values():
            all_rows.extend(rs)
        train_traj = {r["_meta"]["trajectory_id"] for rs in train_by_scene.values() for r in rs}
        test_traj = {r["_meta"]["trajectory_id"] for rs in test_buckets.values() for r in rs}

        # First-action target distribution (the key signal we are trying to
        # spread out away from the previous walk-kitchen collapse).
        first_actions = Counter()
        first_action_targets = Counter()
        for rs in train_by_scene.values():
            by_traj = defaultdict(list)
            for r in rs:
                by_traj[r["_meta"]["trajectory_id"]].append(r)
            for tid, rows in by_traj.items():
                rows.sort(key=lambda r: r["_meta"]["step_index"])
                a = rows[0]["action"].strip()
                first_actions[a] += 1
                parts = a.split()
                if len(parts) > 1:
                    first_action_targets[parts[1]] += 1
                else:
                    first_action_targets[parts[0]] += 1
        top_first_share = (first_actions.most_common(1)[0][1] / max(1, sum(first_actions.values()))
                           if first_actions else 0.0)

        action_counts = Counter()
        for r in all_rows:
            parts = r["action"].strip().split()
            action_counts[parts[0]] += 1

        per_wm = {}
        for sd, rows in sorted(train_by_scene.items()):
            ids = {r["_meta"]["trajectory_id"] for r in rows}
            per_wm[sd] = {
                "episodes": len(ids),
                "rows": len(rows),
                "tasks": len({r["_meta"]["task_id"] for r in rows}),
                "families": dict(Counter(r["_meta"]["task_family"] for r in rows)),
            }

        return {
            "dataset_mode": DATASET_MODE,
            "data_root": str(self.args.output_dir),
            "total_episodes": len(train_traj | test_traj),
            "total_rows": len(all_rows),
            "train_test_trajectory_overlap": len(train_traj & test_traj),
            "pool_episode_counts_before_sampling": {k: len(v) for k, v in pools.items()},
            "split_episode_counts": {
                "train": len(train_traj),
                **{k: len({r["_meta"]["trajectory_id"] for r in rs})
                   for k, rs in test_buckets.items()},
            },
            "split_row_counts": {
                "train": sum(len(rs) for rs in train_by_scene.values()),
                **{k: len(rs) for k, rs in test_buckets.items()},
            },
            "transitions_per_world_model": per_wm,
            "action_verb_counts": dict(action_counts),
            "train_first_action_top10": dict(first_actions.most_common(10)),
            "train_first_action_target_top10": dict(first_action_targets.most_common(10)),
            "train_first_action_top1_share": round(top_first_share, 4),
            "skipped_candidate_reasons": dict(self.reason_counts),
        }

    # ------------------------------------------------------------- write ---
    def write(self, materialized: dict[str, Any]) -> None:
        out = self.args.output_dir
        if out.exists() and any(out.iterdir()):
            if not self.args.overwrite:
                raise FileExistsError(f"{out} is not empty; pass --overwrite")
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        _dump_json(out / "virtualhome_manifest.json", materialized["manifest"])
        _dump_json(out / "quality_report.json", materialized["quality"])

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
        eval_a_by_scene_dir = materialized.get("eval_a_by_scene_dir", {})
        for sd, rows in materialized["train_by_scene_dir"].items():
            d = out / sd
            d.mkdir(exist_ok=True)
            rows = rows[:]
            rng.shuffle(rows)
            with (d / "train.jsonl").open("w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            # Write per-scene test.jsonl containing ONLY this world model's own
            # held-out seen-task seen-scene episodes, NOT a symlink to the global
            # pool (which mixes all scenes and carries wrong _meta.scene for the
            # world model being evaluated).
            test_path = d / "test.jsonl"
            if test_path.exists() or test_path.is_symlink():
                test_path.unlink()
            scene_test_rows = eval_a_by_scene_dir.get(sd, [])[:]
            rng.shuffle(scene_test_rows)
            with test_path.open("w") as f:
                for row in scene_test_rows:
                    f.write(json.dumps(row) + "\n")

    def run(self) -> None:
        materialized = self.materialize()
        # Hard gate: first-action top-1 share must be < 0.35, otherwise
        # something in the pipeline is collapsed to a single trajectory shape.
        q = materialized["quality"]
        if q["train_first_action_top1_share"] > self.args.fail_first_action_top1_share:
            raise RuntimeError(
                f"train first-action top1 share = {q['train_first_action_top1_share']} "
                f"> threshold {self.args.fail_first_action_top1_share}; "
                f"the train data is collapsed. top10: {q['train_first_action_top10']}"
            )
        self.write(materialized)
        print("WorMI-aligned VirtualHome dataset written")
        print(f"  output_dir: {self.args.output_dir}")
        print(f"  episodes: {q['total_episodes']}, rows: {q['total_rows']}")
        print(f"  pool sizes: {q['pool_episode_counts_before_sampling']}")
        print(f"  split counts: {q['split_episode_counts']}")
        print(f"  train first-action top1 share: {q['train_first_action_top1_share']}")
        print(f"  train first-action top10: {q['train_first_action_top10']}")
        print(f"  train first-action target top10: {q['train_first_action_target_top10']}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--vh-src", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scene-domains", type=int, default=20)
    p.add_argument("--seen-scenes", type=int, default=6)
    p.add_argument("--variants-per-domain", type=int, default=8)
    p.add_argument("--candidate-multiplier", type=int, default=16)
    p.add_argument("--train-episodes", type=int, default=384)
    p.add_argument("--eval-a-episodes", type=int, default=96)
    p.add_argument("--eval-b-episodes", type=int, default=224)
    p.add_argument("--eval-c-episodes", type=int, default=319)
    p.add_argument("--eval-a-min-per-task", type=int, default=4)
    p.add_argument("--min-probe-successes", type=int, default=1)
    p.add_argument("--max-probe-tasks", type=int, default=30)
    p.add_argument("--max-scan-per-base", type=int, default=4000)
    p.add_argument("--fail-first-action-top1-share", type=float, default=0.35)
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    Builder(args).run()


if __name__ == "__main__":
    main()
