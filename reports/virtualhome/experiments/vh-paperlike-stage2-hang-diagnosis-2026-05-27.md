# VH Paperlike Stage2 Hang Diagnosis - 2026-05-27

## Run

- Pipeline: `/root/autodl-tmp/wormi-logs/vh-pipeline-paperlike-v1-20260526`
- Stage1 output: `/root/autodl-tmp/wormi-checkpoints/world-vh-paperlike-v1-20260526`
- Stage2 output: `/root/autodl-tmp/wormi-checkpoints/wormi-vh-paperlike-v1-20260526`
- Stage2 log: `/root/autodl-tmp/wormi-logs/vh-wormi-paperlike-v1-20260526-stage2/train.log`

## State Observed

- `status.tsv` reached `stage2 start` but never reached `stage2 done`.
- Stage1 completed and all six world models have `last` checkpoints.
- Stage2 process `34581` remained alive and held about 41 GiB GPU memory.
- GPU utilization stayed at 0%.
- Stage2 log stopped updating at `2026-05-27 01:17:07 +08:00`.
- A 10 second CPU-time sample showed zero CPU tick growth across process threads.
- Main and active Python threads were waiting in `futex_wait_queue_me`.
- No `Traceback`, CUDA OOM, `RuntimeError`, or `ERROR` marker was found.
- No stage2 `last` checkpoint, Table1 summary, or rollout result was produced.

## Per-Subset TensorBoard Progress

The event files show inconsistent sub-trainer termination:

- `subset-0-1-2`: reached step 240 and wrote `train_runtime`.
- `subset-1-2-3`: stopped at step 228 and wrote `train_runtime`.
- `subset-2-3-4`: stopped at step 195 and wrote `train_runtime`.
- `subset-3-4-5`: reached step 240 and wrote `train_runtime`.
- `subset-4-5-0`: reached step 235 but did not write `train_runtime`.
- `subset-5-0-1`: reached step 175 but did not write `train_runtime`.

## Likely Cause

This is a stage2 threaded meta-learning deadlock, not a data-processing failure.

`WorMIMetaLearningTrainer` disables synchronized trainers, but it still inherits the shared lock-ring callback path from `WorMITrainerCallback`. The callback still mutates shared `global_step`, `start_curriculum_step`, and `global_iter` across six concurrently running SFT trainers. Near the final curriculum boundaries, some trainers exit while others are waiting on locks that no live previous trainer will release. That matches the observed state: idle GPU, retained CUDA memory, no log writes, no final checkpoint, and threads waiting on futex locks.

## Consequence

The current stage2 run is not usable for final evaluation because the adapter was never saved to `wormi-vh-n6/last`. Keeping the process alive only keeps GPU memory allocated; it is not making progress.

## Action Taken

- Sent `SIGTERM` to the deadlocked Python process `34581`.
- GPU memory was released afterward: `nvidia-smi` reported 1 MiB used and no running compute processes.
- The pipeline did not write `stage2 done`, so this run must be treated as failed at stage2.

## Fix Direction

Use the already implemented sequential Reptile path:

- set `WORMI_SEQUENTIAL_META_LEARNING=1`
- write to a fresh stage2 output directory
- keep the completed stage1 world-model checkpoints
- avoid frequent internal stage2 eval if fast turnaround is needed, then run final Table1 and rollout eval after the adapter is saved


## Code Fix Applied

Applied a minimal threaded-clean lock-release fix in `wormi/trainer.py`:

- Added `WorMITrainer.release_all_training_locks()`.
- When final training stop is requested in `WorMITrainerCallback.on_step_end`, release all trainer locks.
- In `on_train_end`, release all locks on stop and release both the current and next lock if still held.

This keeps the threaded path and default threaded aggregation behavior, while preventing finished sub-trainers from leaving peers blocked on a lock with no future releaser.

## Threaded-Clean Fix Validation

After the initial lock-release patch, a one-step threaded smoke exposed a shared-model race: concurrent callbacks could call `remove_all()` while another trainer was in `forward()`, producing either `AttributeError` on `__world_models` or an empty world-output stack.

Additional fixes applied:

- `WorMI.remove_all()` is now idempotent and no longer deletes the private attribute before recreating it.
- `WorMI` now exposes a reentrant `state_lock`; `forward()` holds it for the full world+base forward.
- The threaded curriculum switch holds the same `state_lock` while replacing world models.
- Thread target now catches `Exception`, stores it in `self.panic`, releases locks, and lets the main trainer raise `RuntimeError("Training failed")` instead of silently succeeding after a child-thread failure.

Smoke validation:

- Run: `paperlike-threaded-lockfix-smoke3-20260527`
- Config: threaded path, `inner_steps=1`, `meta_steps=1`, `batch=1`, `grad_accum=1`
- Result: all 6 sub-trainers wrote `train_runtime`, no traceback/error markers, `wormi-vh-n6/last` was saved, and GPU memory returned to 1 MiB.
