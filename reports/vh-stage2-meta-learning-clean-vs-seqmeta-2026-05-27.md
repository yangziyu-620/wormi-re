# VirtualHome Stage-2 Meta-Learning: `seqmeta` Failure and `threaded-clean` Interpretation

Date: 2026-05-27

## Purpose

This note records a stage-2 training finding from the VirtualHome WorMI
reproduction work:

- the attempted sequential Reptile implementation, called `seqmeta`, ran
  successfully at the engineering level but produced a nearly unusable adapter
- the later `threaded-clean` run produced usable results, but it is not a
  strict implementation of the paper's Algorithm 1 meta-learning loop
- therefore `threaded-clean` should be described as a stable local
  approximation of the paper stage-2 setup, not as an exact paper
  implementation

The goal is to make the distinction explicit for future reports, result tables,
and paper-comparison claims.

## Background

Stage 2 trains only the WorMI cross-attention adapters. The base LM and all
world models are frozen. For VirtualHome, the intended paper-like setup is:

- base model: Llama-3.2-3B-Instruct mirror
- world models: `N=6`, one per seen scene
- retrieved or trained subset size: `K=3`
- integration method: `WORLD_WISE_ATTENTION`
- base implant layers: `[13, 27]`
- world implant layers: `[7, 15]`
- inner-loop steps: `lambda_I=30`
- meta iterations: `lambda_M=8`
- Reptile-style meta update with `beta=0.1`
- test-time retrieval by prototype/Wasserstein Top-K

The local configuration matches many of these high-level settings in
`tools/wormi_curricula_vh.py`. The main divergence is the exact engineering of
the stage-2 meta-learning loop.

## Paper Algorithm Versus Local Trainer Constraints

The paper-level Reptile loop can be summarized as:

1. Start from current meta adapter parameters `theta`.
2. Sample a subset of `K` world models.
3. Adapt on that subset for `lambda_I` gradient steps, producing `theta_j`.
4. Repeat for multiple sampled tasks or subsets.
5. Aggregate with a Reptile update:

```text
theta <- theta + beta * (mean(theta_j) - theta)
```

This is straightforward as pseudocode, but awkward in this repository because
stage 2 is built on top of Hugging Face/TRL trainers plus dynamically implanted
world-model hooks. A strict implementation must coordinate:

- model parameter snapshots and restores
- optimizer state
- scheduler state
- trainer global step state
- repeated `remove_all()` / `implant()` calls
- hook registration and parameter identity
- device placement and memory release for large world models

Those details are not specified by the paper. They become implementation
choices in this codebase.

## What `seqmeta` Did

`seqmeta` was an attempt to implement the paper's Reptile loop more literally.
It was enabled with:

```text
WORMI_SEQUENTIAL_META_LEARNING=1
```

The implementation is in `WorMIMetaLearningTrainer._train_sequential_reptile`.
For each outer meta iteration it:

1. Built the adapter once.
2. Snapshotted the current meta parameters.
3. For each of the six fixed train curricula:
   - restored the same iteration-start parameters
   - implanted that curriculum's 3 world models
   - built the merged train/test datasets
   - ran `WorMISubTrainer` for 30 inner steps
   - snapshotted the resulting adapter parameters
4. Averaged the six task parameter sets.
5. Applied:

```text
theta_new = theta_old + meta_learning_rate * (mean_theta_task - theta_old)
```

with `meta_learning_rate=0.1`.

The six fixed train curricula were the cyclic K=3 subsets:

```text
[0, 1, 2]
[1, 2, 3]
[2, 3, 4]
[3, 4, 5]
[4, 5, 0]
[5, 0, 1]
```

This is still an approximation because the paper says to sample subsets, while
the local trainer architecture uses a fixed curricula list.

## `seqmeta` Results

The sequential path completed a smoke test:

- run id: `taskaware-seqmeta-smoke-20260521`
- settings: `inner_steps=1`, `meta_steps=1`, `grad_accum=1`
- result: completed all 6 meta tasks, saved `last`, no traceback/runtime error

The full run was:

- run id: `taskaware-seqmeta-stage2-bs1-ga4-20260521`
- output root:
  `/root/autodl-tmp/wormi-checkpoints/wormi-vh-taskaware-seqmeta-20260521`
- train log:
  `/root/autodl-tmp/wormi-logs/vh-wormi-taskaware-seqmeta-stage2-bs1-ga4-20260521/train.log`

The resulting adapter was bad under Table1 evaluation:

| Split | SR | PS | n |
| --- | ---: | ---: | ---: |
| `col_1_seen_seen` | 0.00% | 4.06 | 32 |
| `col_2_seen_unseen` | 1.08% | 4.06 | 186 |
| `col_3_unseen_unseen` | 0.33% | 4.01 | 300 |

An earlier quick gate at `checkpoint-720` was already bad:

- `seen_seen SR=0.0`
- `seen_unseen SR=0.0`
- `unseen_unseen SR=0.0`

Per-step dumps showed first-step action drift such as:

```text
prediction: walk livingroom
target:     walk drawing
```

This means the failure was not only long-horizon compounding. The adapter had
already lost basic first-step object/action grounding.

## What `threaded-clean` Did

`threaded-clean` restored the previously effective threaded stage-2 path and
fixed known engineering issues around it. It was not a new modeling algorithm.

The key changes were:

1. Restore the threaded path as default:

```text
WORMI_SEQUENTIAL_META_LEARNING=0
```

2. Fix the threaded lock-release crash.

The previous threaded path could hit:

```text
RuntimeError: release unlocked lock
```

The clean version checks lock state before release in `on_step_begin` and
`on_train_end`.

3. Restore the original threaded aggregation behavior by default.

The default threaded aggregation computes the mean of task parameters:

```text
theta <- mean(theta_j)
```

The Reptile beta interpolation path exists but is behind:

```text
WORMI_THREADED_META_USE_BETA=1
```

By default, `threaded-clean` does not use:

```text
theta <- theta_old + beta * (mean(theta_j) - theta_old)
```

4. Add evaluation guardrails:

- Table1 per-step prediction dumps
- safe handling for empty outputs
- `NUM_SAMPLES` support for quick gates
- automatic quick gate before full Table1 and rollout
- default quick gate threshold: `seen_seen SR >= 0.50`

## `threaded-clean` Results

The useful threaded-clean run was:

- run id: `taskaware-threaded-clean-bs1-ga4-20260521`
- output root:
  `/root/autodl-tmp/wormi-checkpoints/wormi-vh-taskaware-threaded-clean-20260521`
- train log:
  `/root/autodl-tmp/wormi-logs/vh-wormi-taskaware-threaded-clean-bs1-ga4-20260521/train.log`

Full Table1 results:

| Split | SR | PS | n |
| --- | ---: | ---: | ---: |
| `col_1_seen_seen` | 100.00% | 0.00 | 32 |
| `col_2_seen_unseen` | 89.78% | 0.19 | 186 |
| `col_3_unseen_unseen` | 35.67% | 1.92 | 300 |

Full VirtualHome rollout results:

| Split | SR | PS | n |
| --- | ---: | ---: | ---: |
| `col_1_seen_seen` | 96.88% | 4.84 | 32 |
| `col_2_seen_unseen` | 91.94% | 6.20 | 186 |
| `col_3_unseen_unseen` | 43.33% | 19.30 | 300 |

These results show that the data, world models, and evaluator were not the
primary cause of the `seqmeta` failure. Under the same broad setup, the
threaded path learned usable adapter behavior.

## Difference From the Paper Method

`threaded-clean` aligns with the paper in the broad configuration:

- `N=6` world models
- `K=3` world-model subsets
- `WORLD_WISE_ATTENTION`
- implant layers `[13, 27]` and `[7, 15]`
- `lambda_I=30`
- `lambda_M=8`
- prototype/Wasserstein Top-K retrieval during evaluation

It differs in important implementation details:

| Topic | Paper intent | `threaded-clean` implementation |
| --- | --- | --- |
| subset choice | sample K-world-model subsets | fixed cyclic subsets |
| task inner loop | each task should start from same meta parameters | 6 threaded trainers share one WorMI model and coordinate with locks |
| aggregation | Reptile update with `beta=0.1` | default direct mean of task parameters |
| trainer state | abstract optimizer-free pseudocode | Hugging Face/TRL trainers with optimizer, scheduler, callbacks, global steps |
| world-model switching | abstract subset selection | repeated dynamic hook implant/remove over large frozen CausalLMs |

Therefore `threaded-clean` is not a strict implementation of Algorithm 1. It is
best described as:

```text
a stable local approximation of the paper stage-2 setup
```

or:

```text
paper-configured but not paper-exact
```

## Interpretation

The central finding is not that the paper method is invalid. The finding is
that a literal local Reptile-style rewrite was not reliable in this codebase.

The likely reasons are engineering interactions rather than a simple modeling
failure:

1. `SFTTrainer` does not naturally expose a clean Reptile outer loop.
2. Repeated manual parameter restore can diverge from optimizer/scheduler
   state.
3. Dynamic WorMI hooks make parameter identity and optimizer tracking fragile.
4. Repeated world-model implant/remove operations add device, memory, and hook
   state complexity.
5. The fixed-curricula local implementation is already an approximation of the
   paper's sampling process.
6. The original threaded path may behave more like stable multi-task adapter
   fine-tuning, even if it is less paper-exact.

The `seqmeta` result narrows the failure source:

- not primarily the rebuilt data
- not primarily the Table1 evaluator
- not primarily the frozen world models
- most likely the sequential stage-2 training path, its state handling, or its
  hyperparameter dynamics

## Current Recommendation

For reporting and future experiments:

1. Use `threaded-clean` or the current default threaded path for stable
   VirtualHome stage-2 training.
2. Do not report `seqmeta` as a successful paper-faithful implementation.
3. Do not claim `threaded-clean` is an exact Algorithm 1 reproduction.
4. When comparing to the paper, explicitly state that stage-2 meta-learning is
   an engineering approximation.
5. If strict paper alignment is required, implement a new stage-2 trainer that
   owns the meta-loop directly instead of wrapping multiple `SFTTrainer`
   instances.

## Suggested Strict Reimplementation Plan

A cleaner strict implementation would avoid the current threaded trainer and
avoid repeatedly constructing nested `SFTTrainer` instances. It should:

1. Build the WorMI adapter once and keep a stable list of trainable adapter
   parameters.
2. Build dataloaders for each curriculum/subset outside the meta-loop.
3. For each meta iteration:
   - snapshot adapter parameters
   - for each sampled subset, restore snapshot parameters
   - implant required frozen world models
   - run exactly 30 inner optimizer steps with a local optimizer
   - snapshot adapted adapter parameters
   - remove world models and clear caches
4. Aggregate with:

```text
theta <- theta_old + beta * (mean(theta_j) - theta_old)
```

5. Recreate or reset optimizer state in a controlled way for every inner task.
6. Add an 8-episode quick gate after every checkpoint to catch first-step action
   drift early.

Until that exists, `threaded-clean` is the safest local training path.
