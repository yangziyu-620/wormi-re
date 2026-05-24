"""Split ALFWorld per-room jsonl into the {train,test}.jsonl pair that
`wormi world train` expects.

`tools/build_alfworld_dataset.py` writes paper-aligned buckets:
    bedrooms/train.jsonl     (seen task × seen scene)
    livingrooms/train.jsonl  (seen task × seen scene)
    kitchens/train.jsonl     (seen task × seen scene)
    kitchens/test.jsonl      (unseen task × seen scene)
    bathrooms/test.jsonl     (any task × unseen scene type)

But `wormi/scripts/train_world.py` requires both train.jsonl AND test.jsonl in
each curriculum dir. For bedrooms/livingrooms (no unseen-task data) and for
kitchens (where the existing test.jsonl is the *generalization* probe, not the
within-distribution holdout), this script carves a small fraction of
train.jsonl into test.jsonl so the world model has a valid in-distribution
eval set:

    bedrooms/{train,test}.jsonl
    livingrooms/{train,test}.jsonl
    kitchens/{train,test}.jsonl              (carved from train)
    kitchens/test_unseen_task.jsonl          (preserved generalization probe)
    bathrooms/test.jsonl                     (unchanged — held-out scene type)

Bathrooms is the held-out scene type per paper §4 and has no world-model
training; bathrooms/test.jsonl is the WorMI integration evaluation set.

Usage:
    python3 tools/split_alfworld_train_test.py \
        --root /srv/scratch/z5524306/wormi-data/alfworld \
        --test-fraction 0.1 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def split_room(room_dir: Path, test_fraction: float, rng: random.Random) -> None:
    train_path = room_dir / "train.jsonl"
    if not train_path.exists():
        print(f"  {room_dir.name}: no train.jsonl, skipping")
        return

    existing_test = room_dir / "test.jsonl"
    if existing_test.exists() and not existing_test.is_symlink():
        # kitchens currently has paper-aligned unseen-task test.jsonl — preserve
        # under a more descriptive name so we can recreate test.jsonl as the
        # in-distribution holdout the world-model trainer expects.
        preserved = room_dir / "test_unseen_task.jsonl"
        if not preserved.exists():
            existing_test.rename(preserved)
            print(f"  {room_dir.name}: moved test.jsonl -> test_unseen_task.jsonl")
        else:
            existing_test.unlink()

    with train_path.open() as f:
        rows = [json.loads(line) for line in f]
    if not rows:
        print(f"  {room_dir.name}: train.jsonl empty, skipping")
        return

    n_test = max(1, int(round(len(rows) * test_fraction)))
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    test_idx = set(indices[:n_test])

    train_rows = [r for i, r in enumerate(rows) if i not in test_idx]
    test_rows = [r for i, r in enumerate(rows) if i in test_idx]

    with train_path.open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with (room_dir / "test.jsonl").open("w") as f:
        for r in test_rows:
            f.write(json.dumps(r) + "\n")
    print(
        f"  {room_dir.name}: {len(train_rows)} train + {len(test_rows)} test "
        f"(carved from original {len(rows)})"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True,
                   help="Root containing {kitchens,bedrooms,livingrooms,bathrooms}/")
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    for room in ["kitchens", "bedrooms", "livingrooms"]:
        room_dir = args.root / room
        if not room_dir.exists():
            print(f"  {room}: dir missing, skipping")
            continue
        split_room(room_dir, args.test_fraction, rng)
    print("  bathrooms: kept as held-out scene type (no world model)")


if __name__ == "__main__":
    main()
