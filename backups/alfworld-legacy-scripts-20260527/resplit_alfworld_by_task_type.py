"""Re-organize the ALFWorld dataset from room-based to task-type-based layout
to align with paper §4 + Table A.6 (N=6 world models = 6 task types).

Input (current state):
    alfworld/{bedrooms,kitchens,livingrooms}/{train,test,test_unseen_task}.jsonl
    alfworld/bathrooms/{test,test_unseen_task}.jsonl

Output (paper-aligned):
    alfworld/task_<short>/train.jsonl              ← stage 1 training (per task type)
    alfworld/task_<short>/test.jsonl  → symlink    ← stage 1 eval signal
        ../test_seen_task_seen_scene.jsonl
    alfworld/test_seen_task_seen_scene.jsonl       ← Table 1 col 1 source
    alfworld/test_seen_task_unseen_scene.jsonl     ← Table 1 col 2 source
    alfworld/test_unseen_task_unseen_scene.jsonl   ← Table 1 col 3 source
    alfworld/test_unseen_task_seen_scene.jsonl     ← not in Table 1, kept for completeness
    alfworld/eval_col_{1,2,3}_*/test.jsonl         ← symlinks for curricula

Per-row _meta.room is added (extracted from source path) so downstream resplit
or analysis can re-bucket by scene.

Task-type "short names":
    pick_simple  ← pick_and_place_simple
    look_at_obj  ← look_at_obj_in_light
    pick_heat    ← pick_heat_then_place_in_recep
    pick_cool    ← pick_cool_then_place_in_recep
    pick_two     ← pick_two_obj_and_place         (unseen task per paper)
    pick_clean   ← pick_clean_then_place_in_recep (unseen task per paper)

Paper N=6 partitioning rationale: all 6 task types get a world model. The
"unseen task" classification (paper §4) applies at the WorMI integration
level only — at test time these 2 unseen-task world models are retrieved
for unseen-task queries (Algorithm 1 line 28). Stage 1 trains all 6 on
their respective seen-rooms data (3 seen rooms: bedrooms/kitchens/livingrooms).
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

SEEN_ROOMS = ["bedrooms", "kitchens", "livingrooms"]
UNSEEN_ROOM = "bathrooms"

TASK_SHORT = {
    "pick_and_place_simple": "pick_simple",
    "look_at_obj_in_light": "look_at_obj",
    "pick_heat_then_place_in_recep": "pick_heat",
    "pick_cool_then_place_in_recep": "pick_cool",
    "pick_two_obj_and_place": "pick_two",
    "pick_clean_then_place_in_recep": "pick_clean",
}

SEEN_TASKS = {"pick_simple", "look_at_obj", "pick_heat", "pick_cool"}
UNSEEN_TASKS = {"pick_two", "pick_clean"}


def read_jsonl(path: Path, room: str) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            # Persist room metadata so downstream tools can re-bucket by scene
            # without re-reading source-path layout.
            r.setdefault("_meta", {})
            r["_meta"]["room"] = room
            rows.append(r)
    return rows


def write_jsonl(rows: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def resplit(src_root: Path, dst_root: Path) -> None:
    # Stage 1 per-task-type training pool: collected from the 3 seen rooms.
    # Seen-task types come from each room's train.jsonl; unseen-task types
    # come from each room's test_unseen_task.jsonl (the carve-out from the
    # earlier resplit). All 6 task types feed stage 1 because paper Table A.6
    # N=6 covers seen+unseen at world-model level.
    per_task_train: dict[str, list[dict]] = defaultdict(list)

    # Table 1 col 1 source: held-out 10% of seen-task seen-room data.
    # The earlier resplit already carved this into <room>/test.jsonl.
    col1_rows: list[dict] = []
    # Table 1 col 2 source: bathrooms/test.jsonl restricted to seen tasks.
    col2_rows: list[dict] = []
    # Table 1 col 3 source: bathrooms/test_unseen_task.jsonl as-is.
    col3_rows: list[dict] = []
    # not in Table 1, kept for completeness: seen-room test_unseen_task rows
    test_unseen_task_seen_scene_rows: list[dict] = []

    for room in SEEN_ROOMS:
        room_dir = src_root / room
        # Seen-task training pool: room/train.jsonl rows → per-task buckets
        for r in read_jsonl(room_dir / "train.jsonl", room):
            short = TASK_SHORT.get(r["task"])
            if short is None:
                raise ValueError(f"unknown task type in {room}/train.jsonl: {r['task']}")
            per_task_train[short].append(r)
        # Seen-task col 1 holdout: room/test.jsonl rows
        col1_rows.extend(read_jsonl(room_dir / "test.jsonl", room))
        # Unseen-task training pool: room/test_unseen_task.jsonl rows
        # (also kept for the col-4 file)
        for r in read_jsonl(room_dir / "test_unseen_task.jsonl", room):
            short = TASK_SHORT.get(r["task"])
            if short is None:
                raise ValueError(
                    f"unknown task type in {room}/test_unseen_task.jsonl: {r['task']}"
                )
            per_task_train[short].append(r)
            test_unseen_task_seen_scene_rows.append(r)

    # Bathrooms = unseen scene type. test.jsonl mixes seen + unseen tasks;
    # split on task type:
    #   seen task in bathrooms = col 2
    #   unseen task in bathrooms = col 3 (also redundantly carried by
    #                                     test_unseen_task.jsonl but we
    #                                     take it from there as canonical)
    for r in read_jsonl(src_root / UNSEEN_ROOM / "test.jsonl", UNSEEN_ROOM):
        short = TASK_SHORT.get(r["task"])
        if short is None:
            raise ValueError(
                f"unknown task type in {UNSEEN_ROOM}/test.jsonl: {r['task']}"
            )
        if short in SEEN_TASKS:
            col2_rows.append(r)
    col3_rows = read_jsonl(src_root / UNSEEN_ROOM / "test_unseen_task.jsonl", UNSEEN_ROOM)

    # Write Table 1 column files at root first (so per-task test.jsonl
    # symlinks resolve cleanly).
    print("Table 1 column files:")
    print(f"  col 1 (seen task × seen scene):   "
          f"{write_jsonl(col1_rows, dst_root / 'test_seen_task_seen_scene.jsonl')}")
    print(f"  col 2 (seen task × unseen scene): "
          f"{write_jsonl(col2_rows, dst_root / 'test_seen_task_unseen_scene.jsonl')}")
    print(f"  col 3 (unseen task × unseen scene): "
          f"{write_jsonl(col3_rows, dst_root / 'test_unseen_task_unseen_scene.jsonl')}")
    print(f"  (not in Table 1) unseen task × seen scene: "
          f"{write_jsonl(test_unseen_task_seen_scene_rows, dst_root / 'test_unseen_task_seen_scene.jsonl')}")

    eval_dirs = {
        "eval_col_1_seen_seen": "test_seen_task_seen_scene.jsonl",
        "eval_col_2_seen_unseen": "test_seen_task_unseen_scene.jsonl",
        "eval_col_3_unseen_unseen": "test_unseen_task_unseen_scene.jsonl",
    }
    for dirname, filename in eval_dirs.items():
        eval_dir = dst_root / dirname
        eval_dir.mkdir(parents=True, exist_ok=True)
        test_link = eval_dir / "test.jsonl"
        if test_link.exists() or test_link.is_symlink():
            test_link.unlink()
        test_link.symlink_to(Path("..") / filename)

    # Per-task-type stage-1 dirs.
    print("\nPer-task-type stage-1 training pools:")
    for short, rows in sorted(per_task_train.items()):
        task_dir = dst_root / f"task_{short}"
        task_dir.mkdir(parents=True, exist_ok=True)
        n = write_jsonl(rows, task_dir / "train.jsonl")
        # Stage-1 trainer needs a non-empty test.jsonl for periodic eval_loss
        # logging. For seen tasks we point at col 1 (in-distribution); for
        # unseen tasks col 1 has no data of this task type, so point at the
        # bathrooms col-3 file instead. This is just eval-signal plumbing,
        # not a research metric.
        test_link = task_dir / "test.jsonl"
        if test_link.exists() or test_link.is_symlink():
            test_link.unlink()
        if short in SEEN_TASKS:
            test_link.symlink_to(Path("..") / "test_seen_task_seen_scene.jsonl")
        else:
            test_link.symlink_to(Path("..") / "test_unseen_task_unseen_scene.jsonl")
        print(f"  task_{short:12}  train={n:5}  test.jsonl → "
              f"{test_link.resolve().name}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", type=Path,
                   default=Path("/srv/scratch/z5524306/wormi-data/alfworld"))
    p.add_argument("--dst-root", type=Path,
                   default=Path("/srv/scratch/z5524306/wormi-data/alfworld"))
    p.add_argument("--backup-suffix", type=str, default=".pre_task_type_resplit",
                   help="Suffix for backing up old room-based dirs in place. "
                        "Pass empty string to skip backup.")
    args = p.parse_args()

    if args.src_root == args.dst_root:
        # In-place reorganization: back up the old room dirs first so the
        # script is idempotent + recoverable. Old dirs become e.g.
        # bathrooms.pre_task_type_resplit/.
        if args.backup_suffix:
            for room in [*SEEN_ROOMS, UNSEEN_ROOM]:
                src = args.src_root / room
                bak = args.src_root / f"{room}{args.backup_suffix}"
                if src.exists() and not bak.exists():
                    print(f"backup: {src} → {bak}")
                    shutil.move(str(src), str(bak))
            # Read from backups since src dirs no longer exist
            read_root = args.src_root
            # The backups contain the original room dirs renamed; we need
            # to pass a fresh src-root pointing at them. Easiest: read
            # logic uses src/<room>/, so make symlinks back from <room>/
            # to <room><backup_suffix>/ for the read pass.
            for room in [*SEEN_ROOMS, UNSEEN_ROOM]:
                bak = args.src_root / f"{room}{args.backup_suffix}"
                link = args.src_root / room
                if not link.exists() and bak.exists():
                    link.symlink_to(f"{room}{args.backup_suffix}")
            resplit(read_root, args.dst_root)
            # Remove the read-pass symlinks (real backup dirs stay)
            for room in [*SEEN_ROOMS, UNSEEN_ROOM]:
                link = args.src_root / room
                if link.is_symlink():
                    link.unlink()
        else:
            resplit(args.src_root, args.dst_root)
    else:
        resplit(args.src_root, args.dst_root)


if __name__ == "__main__":
    main()
