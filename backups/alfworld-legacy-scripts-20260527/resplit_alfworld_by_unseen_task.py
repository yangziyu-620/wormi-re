"""Re-bucket ALFWorld jsonl per a corrected `UNSEEN_TASK_TYPES` choice, in
place, without re-running alfworld's textual env.

WHY: `tools/build_alfworld_dataset.py` originally picked
{pick_heat_then_place_in_recep, pick_cool_then_place_in_recep} as the 2 unseen
task types because they have the longest expert plans. But heat/cool tasks
require microwave / fridge and are physically constrained to kitchen scenes
only, so they never appear in bathrooms (the held-out unseen scene type).
That makes paper Table 1 column 3 "Unseen task × Unseen scene" empty by
construction, which contradicts paper §4 reporting that column.

To make column 3 non-empty under the paper's stated setup ("6 task types, 4
seen, 2 unseen") the 2 unseen task types MUST be selected from the tasks that
physically occur in bathrooms: pick_and_place_simple, pick_two_obj_and_place,
pick_clean_then_place_in_recep. Of the 3 candidate pairs only
{pick_two_obj_and_place, pick_clean_then_place_in_recep} is

  - physically feasible (Table 1 col-3 = 233 + 149 = 382 bathrooms trials)
  - conceptually coherent (compositional tasks as unseen, with atomic
    pick_and_place_simple kept as a seen baseline)
  - data-sufficient (every seen room >= 261 train trials after split)

This script reads the union of {train,test,test_unseen_task}.jsonl per room
(splitter-processed state), then re-emits new buckets per the corrected
UNSEEN_TASK_TYPES, using the same 10%-in-room holdout splitter behaviour
(seed=42) for the 3 seen scene types. bathrooms (held-out scene) gets a full
test.jsonl plus a test_unseen_task.jsonl subset (paper Table 1 col-3 source).

Output is staged into <root>.tmp/ first, validated against expected row
counts and a trial-total invariant, then atomically swapped into <root>/
(with the prior contents moved to <root>.bak.<ts>/).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

UNSEEN_TASK_TYPES = {
    "pick_two_obj_and_place",
    "pick_clean_then_place_in_recep",
}
SEEN_SCENE_TYPES = ["bedrooms", "kitchens", "livingrooms"]
UNSEEN_SCENE_TYPES = ["bathrooms"]
ALL_SCENE_TYPES = SEEN_SCENE_TYPES + UNSEEN_SCENE_TYPES
SOURCE_FILES = ["train.jsonl", "test.jsonl", "test_unseen_task.jsonl"]

# Expected row counts after resplit, derived from the (task, scene) cross-tab
# of the currently-built jsonl. Computed from the physical-feasibility matrix:
#   pre-split seen-task-only counts: bedrooms=494, kitchens=1071, livingrooms=261
#   10% holdout (round-half-to-even on int): 49 / 107 / 26
# If actuals diverge, abort the swap.
EXPECTED = {
    ("bedrooms", "train.jsonl"): 445,
    ("bedrooms", "test.jsonl"): 49,
    ("bedrooms", "test_unseen_task.jsonl"): 240,
    ("kitchens", "train.jsonl"): 964,
    ("kitchens", "test.jsonl"): 107,
    ("kitchens", "test_unseen_task.jsonl"): 625,
    ("livingrooms", "train.jsonl"): 235,
    ("livingrooms", "test.jsonl"): 26,
    ("livingrooms", "test_unseen_task.jsonl"): 216,
    ("bathrooms", "test.jsonl"): 646,
    ("bathrooms", "test_unseen_task.jsonl"): 382,
}
EXPECTED_TOTAL = 3553


def load_room_rows(room_dir: Path) -> list[dict]:
    """Read all of a room's existing jsonl rows, dedup by trial_name."""
    seen: set[str] = set()
    rows: list[dict] = []
    for fname in SOURCE_FILES:
        path = room_dir / fname
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                tn = r.get("trial_name")
                if tn in seen:
                    continue
                seen.add(tn)
                rows.append(r)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def carve_holdout(rows: list[dict], fraction: float, rng: random.Random) -> tuple[list[dict], list[dict]]:
    n_test = max(1, round(len(rows) * fraction))
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    test_idx = set(indices[:n_test])
    train = [r for i, r in enumerate(rows) if i not in test_idx]
    test = [r for i, r in enumerate(rows) if i in test_idx]
    return train, test


def resplit(root: Path, tmp_root: Path, test_fraction: float, seed: int) -> None:
    rng = random.Random(seed)
    total_trials = 0
    actual_counts: dict[tuple[str, str], int] = {}

    for room in ALL_SCENE_TYPES:
        rows = load_room_rows(root / room)
        total_trials += len(rows)
        ctr = Counter(r["task"] for r in rows)
        print(f"  {room}: {len(rows)} trials | tasks: {dict(ctr)}")

        if room in SEEN_SCENE_TYPES:
            seen_task_rows = [r for r in rows if r["task"] not in UNSEEN_TASK_TYPES]
            unseen_task_rows = [r for r in rows if r["task"] in UNSEEN_TASK_TYPES]
            seen_task_rows.sort(key=lambda r: r.get("trial_name", ""))
            train, test = carve_holdout(seen_task_rows, test_fraction, rng)
            write_jsonl(tmp_root / room / "train.jsonl", train)
            write_jsonl(tmp_root / room / "test.jsonl", test)
            write_jsonl(tmp_root / room / "test_unseen_task.jsonl", unseen_task_rows)
            actual_counts[(room, "train.jsonl")] = len(train)
            actual_counts[(room, "test.jsonl")] = len(test)
            actual_counts[(room, "test_unseen_task.jsonl")] = len(unseen_task_rows)
        else:  # bathrooms — held-out scene type, no train.jsonl
            unseen_task_subset = [r for r in rows if r["task"] in UNSEEN_TASK_TYPES]
            write_jsonl(tmp_root / room / "test.jsonl", rows)
            write_jsonl(tmp_root / room / "test_unseen_task.jsonl", unseen_task_subset)
            actual_counts[(room, "test.jsonl")] = len(rows)
            actual_counts[(room, "test_unseen_task.jsonl")] = len(unseen_task_subset)

    print(f"\nTotal trial count: {total_trials}")
    if total_trials != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Trial-total invariant violated: got {total_trials}, expected "
            f"{EXPECTED_TOTAL}. Aborting before swap; tmp dir preserved."
        )

    print("\nRow-count verification:")
    diffs = []
    for key, expected in EXPECTED.items():
        actual = actual_counts.get(key, 0)
        ok = (actual == expected)
        marker = "OK" if ok else "MISMATCH"
        print(f"  [{marker}] {key[0]}/{key[1]:32s} got={actual:5d}  expected={expected}")
        if not ok:
            diffs.append((key, actual, expected))
    if diffs:
        raise RuntimeError(
            f"{len(diffs)} bucket(s) mismatch — aborting swap. Inspect {tmp_root}."
        )


def atomic_swap(root: Path, tmp_root: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = root.parent / f"{root.name}.bak.{ts}"
    shutil.move(str(root), str(bak))
    shutil.move(str(tmp_root), str(root))
    return bak


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True,
                   help="ALFWorld dataset root with <room>/{train,test,test_unseen_task}.jsonl")
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-swap", action="store_true",
                   help="Stage to <root>.tmp/ but do not atomically swap.")
    args = p.parse_args()

    if not args.root.is_dir():
        sys.exit(f"ERROR: --root {args.root} not a directory")

    tmp_root = args.root.parent / f"{args.root.name}.tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    print(f"Reading from: {args.root}")
    print(f"Staging into: {tmp_root}")
    print(f"NEW_UNSEEN_TASK_TYPES = {sorted(UNSEEN_TASK_TYPES)}\n")

    resplit(args.root, tmp_root, args.test_fraction, args.seed)

    if args.no_swap:
        print(f"\n--no-swap requested; new files staged at: {tmp_root}")
        return

    bak = atomic_swap(args.root, tmp_root)
    print(f"\nSwapped. Old contents preserved at: {bak}")
    print(f"Verify, then `rm -rf {bak}` once happy.")


if __name__ == "__main__":
    main()
