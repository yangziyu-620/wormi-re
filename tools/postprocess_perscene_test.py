#!/usr/bin/env python3
"""Post-process an existing WorMI VirtualHome dataset to fix per-scene test.jsonl files.

D3 bug: the builder (build_virtualhome_dataset_wormi.py write()) originally
symlinked every scene_N/test.jsonl to the global ../test_seen_task_seen_scene.jsonl
(a mixed pool of ALL apartments), so per-world-model test sets were byte-identical
and carried the wrong _meta.scene for most rows.

This script reads the existing dataset's manifest (to recover the variant_key ->
scene_dir mapping) and global test_seen_task_seen_scene.jsonl, then writes per-scene
test.jsonl files containing ONLY that scene_domain's own held-out episodes.

Usage:
    python tools/postprocess_perscene_test.py \
        --dataset-dir /root/autodl-tmp/wormi-data/virtualhome-realtasks-v3-20260530

The script is idempotent: running it again on an already-fixed dataset is safe
(it replaces real files or symlinks with the correct per-scene content).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, required=True,
                   help="Root directory of the WorMI VirtualHome dataset.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for shuffling rows within each per-scene file.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done without writing any files.")
    args = p.parse_args(argv)

    ds = args.dataset_dir
    if not ds.is_dir():
        print(f"ERROR: dataset directory not found: {ds}", file=sys.stderr)
        sys.exit(1)

    manifest_path = ds / "virtualhome_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    scene_domains = manifest.get("scene_domains", [])

    # Build variant_key -> scene_dir for seen domains only.
    variant_to_scene_dir: dict[str, str] = {}
    seen_dirs: list[str] = []
    for d in scene_domains:
        wm_dir = d.get("world_model_dir")
        if wm_dir is None:
            continue  # unseen domain, no scene_N directory
        seen_dirs.append(wm_dir)
        for vk in d.get("variants", []):
            variant_to_scene_dir[vk] = wm_dir

    if not variant_to_scene_dir:
        print("ERROR: no seen domains with world_model_dir found in manifest.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(seen_dirs)} seen scene directories: {sorted(seen_dirs)}")
    print(f"Variant-to-scene mapping covers {len(variant_to_scene_dir)} variant keys.")

    # Read global test_seen_task_seen_scene.jsonl.
    global_test_path = ds / "test_seen_task_seen_scene.jsonl"
    if not global_test_path.exists():
        print(f"ERROR: global test file not found: {global_test_path}", file=sys.stderr)
        sys.exit(1)

    global_rows: list[dict] = []
    with global_test_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                global_rows.append(json.loads(line))
    print(f"Loaded {len(global_rows)} rows from global test_seen_task_seen_scene.jsonl.")

    # Partition rows by scene_dir.
    by_scene_dir: dict[str, list[dict]] = defaultdict(list)
    n_unmapped = 0
    for row in global_rows:
        vk = row.get("_meta", {}).get("scene", "")
        sd = variant_to_scene_dir.get(vk)
        if sd is None:
            n_unmapped += 1
        else:
            by_scene_dir[sd].append(row)

    if n_unmapped:
        print(f"WARNING: {n_unmapped} rows have _meta.scene not in variant_to_scene_dir "
              "(they are from unseen domains and will not appear in any per-scene test).")

    # Verify: all seen scene dirs should have at least some rows.
    print("\nPer-scene row counts:")
    for sd in sorted(seen_dirs):
        rows = by_scene_dir.get(sd, [])
        trajs = {r["_meta"]["trajectory_id"] for r in rows}
        print(f"  {sd}: {len(rows)} rows, {len(trajs)} trajectories")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # Write per-scene test.jsonl files.
    rng = random.Random(args.seed)
    for sd in sorted(seen_dirs):
        scene_dir = ds / sd
        if not scene_dir.is_dir():
            print(f"WARNING: directory {scene_dir} does not exist, skipping.")
            continue

        test_path = scene_dir / "test.jsonl"

        # Remove existing symlink or file.
        if test_path.is_symlink():
            test_path.unlink()
            print(f"  Removed symlink: {test_path}")
        elif test_path.exists():
            test_path.unlink()
            print(f"  Removed existing file: {test_path}")

        rows = by_scene_dir.get(sd, [])[:]
        rng.shuffle(rows)
        with test_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"  Wrote {len(rows)} rows -> {test_path}")

    print("\nPost-process complete.")

    # Verification: ensure scene files are NOT byte-identical to each other.
    contents = {}
    for sd in sorted(seen_dirs):
        p = ds / sd / "test.jsonl"
        if p.exists() and not p.is_symlink():
            contents[sd] = p.read_bytes()

    from itertools import combinations
    identical_pairs = []
    for a, b in combinations(sorted(contents), 2):
        if contents[a] == contents[b]:
            identical_pairs.append((a, b))

    if identical_pairs:
        print(f"\nFAIL: {len(identical_pairs)} pairs of scene test files are still "
              f"byte-identical: {identical_pairs}")
        sys.exit(1)
    else:
        print(f"\nPASS: all {len(contents)} per-scene test.jsonl files are distinct.")

    # Verification: _meta.scene in each file maps back to that scene's variants.
    bad = []
    for sd in sorted(seen_dirs):
        p = ds / sd / "test.jsonl"
        if not p.exists() or p.is_symlink():
            continue
        expected_vks = {vk for vk, d in variant_to_scene_dir.items() if d == sd}
        with p.open() as f:
            for i, line in enumerate(f):
                row = json.loads(line.strip())
                vk = row.get("_meta", {}).get("scene", "")
                if vk not in expected_vks:
                    bad.append((sd, i, vk))

    if bad:
        print(f"\nFAIL: {len(bad)} rows have _meta.scene that doesn't match directory:")
        for entry in bad[:10]:
            print(f"  {entry}")
        sys.exit(1)
    else:
        print("PASS: all rows have _meta.scene matching their scene directory.")


if __name__ == "__main__":
    main()
