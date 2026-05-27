# ALFWorld Legacy Split Scripts

Date: 2026-05-27

These scripts were moved out of `tools/` after the ALFWorld data path was
consolidated around:

- `tools/collect_alfworld_episodes.py`
- `tools/build_alfworld_dataset.py`
- `tools/validate_alfworld_dataset.py`

Contents:

- `split_alfworld_train_test.py`: old room-based 10% holdout helper.
- `resplit_alfworld_by_unseen_task.py`: old corrected unseen-task re-bucketer.
- `resplit_alfworld_by_task_type.py`: old task-type directory materializer.

Keep these for historical reproduction of earlier task-type ALFWorld runs. New
experiments should use the protocol builder and validator in `tools/`.
