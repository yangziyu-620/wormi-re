"""Build eval-only VirtualHome split variants without changing training data.

The main use is seen-task/seen-scene diagnostics. The canonical dataset keeps
most seen_seen trajectories in scene_{0..5}/train.jsonl and only a tiny held-out
set in test_seen_task_seen_scene.jsonl. This script materializes alternate
test.jsonl directories from existing rows so a trained checkpoint can be
re-evaluated without retraining.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _trajectory_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["_meta"]["trajectory_id"]].append(row)
    return {
        tid: sorted(items, key=lambda r: r["_meta"]["step_index"])
        for tid, items in grouped.items()
    }


def _family_from_tid(tid: str) -> str:
    return tid.split(":", 2)[1]


def _collect_seen_seen_rows(data_root: Path) -> dict[str, list[dict]]:
    rows: list[dict] = []
    rows.extend(_read_jsonl(data_root / "test_seen_task_seen_scene.jsonl"))
    for scene_dir in sorted(data_root.glob("scene_*")):
        train_path = scene_dir / "train.jsonl"
        if train_path.exists():
            rows.extend(_read_jsonl(train_path))

    grouped = _trajectory_rows(rows)
    return {
        tid: traj_rows
        for tid, traj_rows in grouped.items()
        if traj_rows[0].get("_meta", {}).get("split") == "seen_seen"
    }


def _flatten(grouped: dict[str, list[dict]], tids: list[str]) -> list[dict]:
    return [row for tid in tids for row in grouped[tid]]


def build_eval_variants(data_root: Path, output_root: Path, seed: int, per_family: int) -> None:
    rng = random.Random(seed)
    grouped = _collect_seen_seen_rows(data_root)
    all_tids = sorted(grouped)

    full_rows = _flatten(grouped, all_tids)
    full_dir = output_root / "eval_col_1_seen_seen_full"
    _write_jsonl(full_dir / "test.jsonl", full_rows)

    by_family: dict[str, list[str]] = defaultdict(list)
    for tid in all_tids:
        by_family[_family_from_tid(tid)].append(tid)

    balanced_tids: list[str] = []
    for family in sorted(by_family):
        tids = sorted(by_family[family])
        rng.shuffle(tids)
        balanced_tids.extend(sorted(tids[:per_family]))
    balanced_rows = _flatten(grouped, sorted(balanced_tids))
    balanced_dir = output_root / f"eval_col_1_seen_seen_balanced_{per_family}"
    _write_jsonl(balanced_dir / "test.jsonl", balanced_rows)

    summary = {
        "source_data_root": str(data_root),
        "seed": seed,
        "full": {
            "dir": str(full_dir),
            "trajectories": len(all_tids),
            "rows": len(full_rows),
            "by_family": {
                family: len(tids) for family, tids in sorted(by_family.items())
            },
        },
        "balanced": {
            "dir": str(balanced_dir),
            "per_family": per_family,
            "trajectories": len(balanced_tids),
            "rows": len(balanced_rows),
            "by_family": {
                family: min(per_family, len(tids))
                for family, tids in sorted(by_family.items())
            },
        },
    }
    (output_root / "eval_seen_seen_variants_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/root/autodl-tmp/wormi-data/virtualhome"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to --data-root.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-family", type=int, default=6)
    args = parser.parse_args()
    build_eval_variants(
        data_root=args.data_root,
        output_root=args.output_root or args.data_root,
        seed=args.seed,
        per_family=args.per_family,
    )


if __name__ == "__main__":
    main()
