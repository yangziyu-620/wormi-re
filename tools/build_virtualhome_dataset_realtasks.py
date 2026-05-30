#!/usr/bin/env python3
"""Real-task VirtualHome dataset builder for WorMI reproduction.

This is a thin subclass of `build_virtualhome_dataset_wormi.Builder`. It changes
exactly ONE thing: the source of the 78 tasks.

  - OLD (build_virtualhome_dataset_wormi): tasks come from
    `build_candidate_instructions(properties_data.json, ...)` — a synthetic,
    coverage-greedy enumeration over object classes. This is what produced the
    "data collapse" (a handful of source objects dominating) across the prior 12
    builds, because nothing tied the task list to what tasks humans actually do.

  - NEW (this file): tasks are MINED from the real VirtualHome crowdsourced
    ActivityPrograms under `raw_dir/executable_programs/**/*.txt`. Each program's
    terminal manipulation actions map to the 4 WorMI families:

        SWITCHON <obj>            -> ("turnon",  (obj,))
        OPEN     <obj>            -> ("open",    (obj,))
        PUTBACK  <src> <tgt>      -> ("puton",   (src, tgt))      # put X on a surface
        PUTIN    <src> <tgt>      -> ("placein", (src, tgt))      # put X in a container

    (PUTON is intentionally NOT mapped — in VirtualHome PUTON means "wear
    clothing", not "put object on surface"; the existing replay machinery uses
    PUTBACK for the paper's puton family. See build_atomic_program /
    normalize_action in build_virtualhome_dataset.py.)

    The 78 tasks are then selected per paper family quota (9/7/30/32) by REAL
    frequency, with a per-source-class diversity cap for turnon/open/puton, and
    a near-full take for placein (only ~35 unique placein pairs exist in the
    corpus, paper wants 32). The seen-16 split reuses the parent's
    stratified-by-source-class logic.

Everything downstream — scene/variant loading, the EvolvingGraph ScriptExecutor
replay (`_execute_paperlike_candidate`), the (instruction, observation, action,
next_observation) row schema, the 16/62 task split, 6/14 scene split, episode
sampling, first-action collapse gate, and the on-disk layout — is INHERITED
unchanged from the parent. The atomic expert trajectory for each (task, scene)
is still produced + replayed by the parent's machinery; this file only decides
*which* (family, src, tgt) tasks are real and worth replaying.

Usage mirrors the parent, with the same --raw-dir (the directory that contains
both `executable_programs/` and `init_and_final_graphs/`):

    uv run python -m tools.build_virtualhome_dataset_realtasks \
        --raw-dir /root/autodl-tmp/wormi-data/raw/programs_processed_precond_nograb_morepreconds \
        --vh-src  /root/autodl-tmp/wormi-data/virtualhome-src \
        --output-dir /root/autodl-tmp/wormi-data/virtualhome-realtasks-YYYYMMDD \
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.build_virtualhome_dataset import TARGETS, find_first_id  # noqa: E402
from tools.build_virtualhome_dataset_wormi import (  # noqa: E402
    Builder,
    build_arg_parser as _parent_arg_parser,
)

DATASET_MODE = "wormi_real_tasks_v1"

# raw VirtualHome action verb -> (family, arity). arity = number of <obj> tokens
# the family's task signature consumes.
_GOAL_VERB_TO_FAMILY = {
    "SWITCHON": ("turnon", 1),
    "OPEN": ("open", 1),
    "PUTBACK": ("puton", 2),
    "PUTIN": ("placein", 2),
}

_ACTION_RE = re.compile(r"\[([A-Z_]+)\]")
_OBJ_RE = re.compile(r"<([^>]+)>")


def _parse_goals_from_program(text: str) -> set[tuple[str, tuple[str, ...]]]:
    """Extract the set of (family, args) goals realized by a single program.

    Each manipulation action with a recognized goal verb contributes one task.
    A composite human plan (e.g. "make coffee") may contribute several; we count
    the SET so a single program inflates a task's frequency by at most 1.
    """
    goals: set[tuple[str, tuple[str, ...]]] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("["):
            continue
        vm = _ACTION_RE.match(line)
        if not vm:
            continue
        verb = vm.group(1).upper()
        fam_arity = _GOAL_VERB_TO_FAMILY.get(verb)
        if fam_arity is None:
            continue
        family, arity = fam_arity
        objs = _OBJ_RE.findall(line)
        if len(objs) < arity:
            continue
        args = tuple(o.strip().lower() for o in objs[:arity])
        if any(not a for a in args):
            continue
        goals.add((family, args))
    return goals


def mine_real_tasks(raw_dir: Path) -> dict[str, Any]:
    """Walk executable_programs and aggregate the real (family, args) catalog.

    Returns a dict with, per family, a frequency-ranked candidate list and the
    set of base apartments each task appears in, plus corpus-level stats.
    """
    exe_root = raw_dir / "executable_programs"
    if not exe_root.is_dir():
        raise FileNotFoundError(exe_root)

    # (family, args) -> program frequency
    freq: Counter[tuple[str, tuple[str, ...]]] = Counter()
    # (family, args) -> set of base apartment dir names it appears in
    scenes: dict[tuple[str, tuple[str, ...]], set[str]] = defaultdict(set)
    n_programs = 0

    for base_dir in sorted(p for p in exe_root.iterdir() if p.is_dir()):
        base = base_dir.name
        for txt in base_dir.rglob("*.txt"):
            try:
                text = txt.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            n_programs += 1
            for goal in _parse_goals_from_program(text):
                freq[goal] += 1
                scenes[goal].add(base)

    per_family: dict[str, list[dict[str, Any]]] = {fam: [] for fam in TARGETS}
    for (family, args), f in freq.items():
        if family not in per_family:
            continue
        per_family[family].append(
            {"family": family, "args": list(args), "freq": f,
             "n_scenes": len(scenes[(family, args)])}
        )
    # Rank: frequency desc, then apartment coverage desc, then name for determinism.
    for fam in per_family:
        per_family[fam].sort(key=lambda d: (-d["freq"], -d["n_scenes"], tuple(d["args"])))

    return {
        "n_programs_scanned": n_programs,
        "n_unique_tasks": {fam: len(per_family[fam]) for fam in per_family},
        "per_family": per_family,
    }


class RealTaskBuilder(Builder):
    """Builder whose 78-task list is mined from real crowdsourced plans."""

    def _mine(self) -> dict[str, Any]:
        if getattr(self, "_mining", None) is None:
            self._mining = mine_real_tasks(self.args.raw_dir)
            mr = getattr(self.args, "mine_report", None)
            if mr:
                Path(mr).parent.mkdir(parents=True, exist_ok=True)
                Path(mr).write_text(json.dumps(self._mining, indent=2))
                print(f"wrote mining report -> {mr}", flush=True)
        return self._mining

    def select_tasks(self):  # type: ignore[override]
        mining = self._mine()
        per_family = mining["per_family"]
        print(
            f"mined {mining['n_programs_scanned']} programs; unique tasks/family = "
            f"{mining['n_unique_tasks']}",
            flush=True,
        )

        # Scene-aware feasibility. The v1 build's dead tasks (a seen task that
        # was infeasible in ALL 6 seen scenes -> 0 train rows, and 7 unseen tasks
        # with 0 rows) came from selecting tasks without checking the *loaded*
        # scenes. Here we count, per task, how many loaded variants and how many
        # SEEN variants can actually instantiate it, and use those counts to (a)
        # prefer high-pool tasks for the 78 and (b) guarantee every seen task is
        # feasible in several seen scenes.
        variant_classes: dict[str, set[str]] = {
            vk: {n["class_name"] for n in g["nodes"]}
            for vk, g in self.scene_inits.items()
        }
        all_keys = list(variant_classes)
        seen_keys = [v["variant_key"] for d in self.seen_domains for v in d["variants"]]

        def n_feas(args: tuple[str, ...], keys: list[str]) -> int:
            return sum(1 for vk in keys if all(a in variant_classes[vk] for a in args))

        # task tuple -> (freq, feas_all, feas_seen)
        stats: dict[tuple[str, tuple[str, ...]], tuple[int, int, int]] = {}
        for fam in TARGETS:
            for c in per_family.get(fam, []):
                args = tuple(c["args"])
                stats[(fam, args)] = (
                    c["freq"], n_feas(args, all_keys), n_feas(args, seen_keys)
                )

        # placein has only ~35 unique pairs corpus-wide and the paper wants 32,
        # so it is taken near-full (no source cap). The other families enforce a
        # per-source-class diversity cap = quota // 4, exactly as the parent does
        # for the synthetic pool — this is what prevents one source object (e.g.
        # the historical drawing/mat/phone collapse) from dominating a family.
        NEAR_FULL = {"placein"}
        all_tasks: list[tuple[str, tuple[str, ...]]] = []
        selection_report: dict[str, Any] = {}
        for fam in TARGETS:
            quota = TARGETS[fam]
            corpus = [(fam, tuple(c["args"])) for c in per_family.get(fam, [])]
            # Rank: scene-feasible first (feas_all > 0), then by corpus frequency,
            # then by how many loaded scenes can host the task (bigger pool), then
            # name. Execution still skips any (task, scene) combo that can't run,
            # so an occasionally-infeasible task costs nothing; this ordering just
            # front-loads the tasks that will actually produce many episodes.
            def rank_key(t: tuple[str, tuple[str, ...]]):
                freq, fa, _fs = stats[t]
                return (0 if fa > 0 else 1, -freq, -fa, t[1])
            cands = sorted(corpus, key=rank_key)
            picked: list[tuple[str, tuple[str, ...]]] = []
            if fam in NEAR_FULL:
                picked = cands[:quota]
            else:
                cap = max(1, quota // 4)
                src_count: Counter[str] = Counter()
                for t in cands:
                    if len(picked) >= quota:
                        break
                    if src_count[t[1][0]] >= cap:
                        continue
                    picked.append(t)
                    src_count[t[1][0]] += 1
                if len(picked) < quota:  # fallback if the cap was too tight
                    have = set(picked)
                    for t in cands:
                        if len(picked) >= quota:
                            break
                        if t not in have:
                            picked.append(t)
                            have.add(t)
            if len(picked) < quota:
                raise RuntimeError(
                    f"family {fam}: only {len(picked)}/{quota} real tasks in corpus "
                    f"({len(corpus)} unique). The corpus lacks this many {fam} tasks."
                )
            all_tasks.extend(picked)
            selection_report[fam] = {
                "quota": quota, "corpus_candidates": len(corpus),
                "picked": [{"args": list(a), "freq": stats[(fam, a)][0],
                            "feas_all_scenes": stats[(fam, a)][1],
                            "feas_seen_scenes": stats[(fam, a)][2]}
                           for _f, a in picked],
            }

        print(
            "selected 78 real tasks: "
            f"{ {f: sum(1 for fa, _ in all_tasks if fa == f) for f in TARGETS} }",
            flush=True,
        )

        # Seen-16 split (2/2/6/6): stratified-by-source-class, but constrained to
        # tasks feasible in at least MIN_SEEN_FEAS seen scenes so no seen task is
        # dead. Within the source-diversity greedy, ties prefer the task with the
        # largest seen-scene pool. If a family lacks enough qualifying tasks the
        # threshold is relaxed step-by-step.
        seen_quotas = {"turnon": 2, "open": 2, "puton": 6, "placein": 6}
        MIN_SEEN_FEAS = 2
        seen_set: set[tuple[str, tuple[str, ...]]] = set()
        seen_src: Counter[str] = Counter()
        seen_tgt: Counter[str] = Counter()
        for fam in TARGETS:
            quota = seen_quotas[fam]
            fam_tasks = [t for t in all_tasks if t[0] == fam]
            picked = 0
            threshold = MIN_SEEN_FEAS
            while picked < quota:
                pool = [t for t in fam_tasks
                        if t not in seen_set and stats[t][2] >= threshold]
                if not pool:
                    if threshold <= 0:
                        break  # nothing left even with no constraint
                    threshold -= 1
                    continue

                def key(t: tuple[str, tuple[str, ...]]):
                    src = t[1][0]
                    tgt = t[1][1] if len(t[1]) > 1 else ""
                    # source diversity, then target diversity, then bigger seen
                    # pool (-feas_seen), then name for determinism.
                    return (seen_src[src], seen_tgt[tgt], -stats[t][2], t[1])
                t = min(pool, key=key)
                seen_set.add(t)
                seen_src[t[1][0]] += 1
                if len(t[1]) > 1:
                    seen_tgt[t[1][1]] += 1
                picked += 1

        seen_tasks = [t for t in all_tasks if t in seen_set]
        unseen_tasks = [t for t in all_tasks if t not in seen_set]
        self._selection_report = selection_report
        print(
            f"seen tasks ({len(seen_tasks)}): "
            f"{ {f: sum(1 for fa, _ in seen_tasks if fa == f) for f in TARGETS} }; "
            f"seen sources: {sorted(seen_src.items(), key=lambda x: -x[1])}",
            flush=True,
        )
        return seen_tasks, unseen_tasks


    # ----------------------------------------------------------- gate/run ---
    def run(self) -> None:  # type: ignore[override]
        """Same as parent, but gate on SOURCE-OBJECT diversity, not the first
        room-walk.

        The parent gates on first-action top-1 share. With real (kitchen-heavy)
        tasks and the room-first expert program, the first action is `walk
        <room>` and `walk kitchen` legitimately dominates (~46%) because most
        real manipulation tasks happen in the kitchen — and that walk is
        instruction-conditioned and correct. The historical collapse this gate
        was meant to catch was a SOURCE-OBJECT collapse (seen tasks covering
        only 2 source objects, e.g. drawing/mat). So we gate on the share of the
        single most common train source object instead, and keep the room-walk
        share as an informational signal.
        """
        materialized = self.materialize()
        q = materialized["quality"]

        train_rows = [
            r for rs in materialized["train_by_scene_dir"].values() for r in rs
        ]
        per_traj_source: dict[str, str] = {}
        for r in train_rows:
            meta = r["_meta"]
            args = meta.get("task_args") or []
            per_traj_source[meta["trajectory_id"]] = args[0] if args else "?"
        source_counts = Counter(per_traj_source.values())
        n_traj = sum(source_counts.values())
        top_source_share = (
            source_counts.most_common(1)[0][1] / n_traj if n_traj else 0.0
        )
        q["train_source_object_top1_share"] = round(top_source_share, 4)
        q["train_source_object_top10"] = dict(source_counts.most_common(10))
        q["train_distinct_source_objects"] = len(source_counts)

        room_walk_share = q.get("train_first_action_top1_share", 0.0)
        print(
            f"  train distinct source objects: {len(source_counts)}; "
            f"source top1 share: {top_source_share:.4f}; "
            f"(informational) first room-walk top1 share: {room_walk_share}",
            flush=True,
        )
        if top_source_share > self.args.fail_source_top1_share:
            raise RuntimeError(
                f"train source-object top1 share = {top_source_share:.4f} "
                f"> threshold {self.args.fail_source_top1_share}; a single source "
                f"object dominates the train set (real collapse). "
                f"top10: {dict(source_counts.most_common(10))}"
            )

        self.write(materialized)
        # Export the variant_key -> init_graph map so
        # tools/validate_virtualhome_dataset.py can replay every trajectory.
        scene_inits_path = self.args.output_dir / "scene_inits.json"
        scene_inits_path.write_text(json.dumps(self.scene_inits))
        print(f"  wrote scene_inits map -> {scene_inits_path}")
        print("WorMI real-task VirtualHome dataset written")
        print(f"  output_dir: {self.args.output_dir}")
        print(f"  episodes: {q['total_episodes']}, rows: {q['total_rows']}")
        print(f"  split counts: {q['split_episode_counts']}")
        print(f"  train distinct sources: {q['train_distinct_source_objects']}, "
              f"source top1 share: {q['train_source_object_top1_share']}")
        print(f"  (info) first room-walk top1 share: {room_walk_share}, "
              f"top10: {q['train_first_action_top10']}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = _parent_arg_parser()
    p.add_argument(
        "--mine-report", type=str, default=None,
        help="optional path to dump the mined real-task catalog JSON",
    )
    p.add_argument(
        "--fail-source-top1-share", type=float, default=0.35,
        help="abort if one source object exceeds this share of train trajectories",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    RealTaskBuilder(args).run()


if __name__ == "__main__":
    main()
