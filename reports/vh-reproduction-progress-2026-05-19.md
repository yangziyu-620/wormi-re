# VirtualHome Reproduction Progress

Date: 2026-05-19

## Current Status

VirtualHome rollout evaluation completed. No WorMI training/evaluation process is running, and the GPU is idle.

Data disk:

- Mount: `/root/autodl-tmp`
- Size: 150G
- Used: 32G
- Available: 119G

## Completed

### VirtualHome Data

Rebuilt and stored at:

- `/root/autodl-tmp/wormi-data/virtualhome`

The rebuilt data includes:

- `test_seen_task_seen_scene.jsonl`
- `test_seen_task_unseen_scene.jsonl`
- `test_unseen_task_unseen_scene.jsonl`
- `test_unseen_task_seen_scene.jsonl`
- `eval_col_1_seen_seen/test.jsonl`
- `eval_col_2_seen_unseen/test.jsonl`
- `eval_col_3_unseen_unseen/test.jsonl`
- per-row `_meta.trajectory_id`, `_meta.step_index`, `_meta.num_steps`

### Stage 1: VH World Models

Completed for 6 seen scenes, one world model per seen scene:

- `/root/autodl-tmp/wormi-checkpoints/world-vh/scene_0/last`
- `/root/autodl-tmp/wormi-checkpoints/world-vh/scene_1/last`
- `/root/autodl-tmp/wormi-checkpoints/world-vh/scene_2/last`
- `/root/autodl-tmp/wormi-checkpoints/world-vh/scene_3/last`
- `/root/autodl-tmp/wormi-checkpoints/world-vh/scene_4/last`
- `/root/autodl-tmp/wormi-checkpoints/world-vh/scene_5/last`

Intermediate stage-1 checkpoints `checkpoint-1000` and `checkpoint-2000` were deleted to free disk space. The `last` directories were kept.

Stage-1 script:

- `sh/wormi-train-vh-world.sh`

### Stage 2: VH WorMI Adapter Training

Completed after restarting with batch size 1 due to OOM at batch size 4.

Outputs:

- `/root/autodl-tmp/wormi-checkpoints/wormi-vh/wormi-vh-n6/checkpoint-240`
- `/root/autodl-tmp/wormi-checkpoints/wormi-vh/wormi-vh-n6/last`

Successful run log:

- `/root/autodl-tmp/wormi-logs/vh-wormi-20260518_235127/train.log`

Launcher log:

- `/root/autodl-tmp/wormi-logs/vh-wormi-launch-20260518_235127.log`

Script:

- `sh/wormi-train-vh-wormi.sh`

Important runtime settings:

- `WORMI_VH_STAGE2_BATCH_SIZE=1`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `DATA_DISK=/root/autodl-tmp`

## Code/Config Changes Relevant To Resume

- `tools/world_curricula_vh.py` now uses `/root/autodl-tmp` defaults and env vars.
- `tools/wormi_curricula_vh.py` now uses `/root/autodl-tmp` defaults and env vars.
- `sh/wormi-train-vh-world.sh` trains stage-1 VH world models.
- `sh/wormi-train-vh-wormi.sh` trains stage-2 VH WorMI adapters.
- `wormi/scripts/eval_table1.py` was added for offline Table-1-style SR/PS evaluation from JSONL expert trajectories.
- `wormi/scripts/main.py` includes `wormi eval-table1`.

## Next Step

Run evaluation from the completed stage-2 model:

```bash
DATA_DISK=/root/autodl-tmp \
.venv/bin/wormi eval \
  --curricula_path tools/wormi_curricula_vh.py \
  --model_name /root/autodl-tmp/wormi-checkpoints/wormi-vh/wormi-vh-n6/last
```

Do not rerun the train scripts unless retraining is intended. For evaluation-only, use the `wormi eval` or `wormi eval-table1` path with:

- curricula: `tools/wormi_curricula_vh.py`
- model: `/root/autodl-tmp/wormi-checkpoints/wormi-vh/wormi-vh-n6/last`

## Notes

- The first stage-2 attempt with batch size 4 failed with CUDA OOM.
- The successful stage-2 run finished at `2026-05-19T00:22:22+08:00`.
- No `Traceback`, `OutOfMemory`, or `RuntimeError` was found in the successful stage-2 log.

## Table-1-Style Offline Eval

Completed at `2026-05-19T01:26:33+08:00`.

Output directory:

- `/root/autodl-tmp/wormi-checkpoints/wormi-vh/wormi-vh-n6/table1`

Logs:

- `/root/autodl-tmp/wormi-logs/vh-table1-20260519_011612/eval.log`
- `/root/autodl-tmp/wormi-logs/vh-table1-launch-20260519_011612.log`

Summary:

| split | episodes | SR | PS | avg_total_steps | retrieved_world_models |
| --- | ---: | ---: | ---: | ---: | --- |
| col_1_seen_seen | 29 | 0.448276 | 0.586207 | 1.034483 | 5,4,3 |
| col_2_seen_unseen | 149 | 0.006711 | 3.060403 | 3.993289 | 2,0,1 |
| col_3_unseen_unseen | 300 | 0.010000 | 3.100000 | 4.030000 | 2,3,1 |

This is the offline JSONL trajectory evaluator from `wormi eval-table1`, not a full VirtualHome simulator rollout.

## Paper Alignment Gap

The current offline results should not be compared directly with paper Table 1.

Paper Table 1 reports VirtualHome **environment rollout** SR/PS:

| split | paper WorMI SR | paper WorMI PS |
| --- | ---: | ---: |
| Seen Tasks & Seen Scenes | 85.78% | 10.76 |
| Seen Tasks & Unseen Scenes | 80.26% | 12.42 |
| Unseen Tasks & Unseen Scenes | 66.12% | 15.17 |

Current `wormi eval-table1` reports **offline JSONL exact-match** SR/PS:

| split | current SR | current PS | episodes |
| --- | ---: | ---: | ---: |
| col_1_seen_seen | 44.8276% | 0.586207 | 29 |
| col_2_seen_unseen | 0.6711% | 3.060403 | 149 |
| col_3_unseen_unseen | 1.0000% | 3.100000 | 300 |

These are different metrics:

- Paper SR: task success after executing generated actions in the VirtualHome environment.
- Current SR: every generated action must exactly match the expert JSONL trajectory until the end.
- Paper PS: average environment timesteps required to complete the task.
- Current PS: remaining unmatched expert trajectory steps after the first mismatch.

The current `avg_total_steps` values are around 1-4, while paper PS values are around 10-29 across methods, confirming that the two evaluations are on different scales.

## Required Work For Full Paper Alignment

1. Implement a VirtualHome rollout evaluator.
   - Reset to the target scene initial graph.
   - Render the same observation format as training.
   - Prompt the model for an action.
   - Parse the action into one of the six VirtualHome skills:
     `walk`, `grab`, `switch`, `open`, `putin`, `put`.
   - Execute the action in the VirtualHome/EvolvingGraph environment.
   - Update the observation after each step.
   - Stop when the task goal is satisfied or max steps is reached.
   - Report SR and PS using the paper's rollout definition.

2. Verify VirtualHome data split against the paper.
   - Paper VirtualHome setting: 1,023 episodes, 78 tasks, 20 scenes.
   - Tasks: 16 seen, 62 unseen.
   - Scenes: 6 seen, 14 unseen.
   - Table 1 columns:
     - seen task + seen scene
     - seen task + unseen scene
     - unseen task + unseen scene

3. Align training hyperparameters.
   - World models:
     - Llama-3.2-1B
     - batch size 4
     - 2000 gradient steps
     - cosine scheduler
     - learning rate 3e-5
   - WorMI adapter:
     - Llama-3.2-3B reasoning model
     - batch size 4
     - meta update steps lambda_M = 8
     - inner-loop gradient steps lambda_I = 30
     - cosine scheduler
     - learning rate alpha = 1e-5
     - meta learning rate beta = 1e-1

4. Check local deviations already known.
   - Stage-1 was run with batch size 2 because batch size 4 OOMed.
   - Stage-2 was run with batch size 1 because batch size 4 OOMed.
   - Local stage-2 meta-learning aggregation appears to average task parameters; it should be checked against the paper's beta = 0.1 update.
   - Local model IDs use `unsloth/...-Instruct` mirrors rather than the exact `meta-llama/...` names in the paper.

5. Align prompt and decoding.
   - Paper VirtualHome prompt is Figure A.4:
     system prompt with 6 skills and `Action:` output.
   - Ensure the local chat template does not add extra text that changes the expected action format.
   - Align generation settings, especially temperature = 1.0 where specified by the paper.

6. Run five seeds.
   - Paper Table 1 reports 95% confidence intervals over 5 random seeds.
   - A single run is not enough for a paper-level comparison.

## Next Concrete Step

Implement `wormi eval-vh-rollout` or equivalent script for real VirtualHome rollout SR/PS. Do this before retraining; otherwise we cannot tell whether training changes are helping under the paper metric.

## Rollout Evaluator Implementation

Implemented `wormi eval-vh-rollout`.

Behavior:

- Load the trained stage-2 WorMI checkpoint.
- Retrieve top-K world models using the same prototype path as `wormi eval-table1`.
- Reset each VirtualHome episode to its initial scene graph from `/root/autodl-tmp/wormi-data/scene-inits/init_graphs_20.json`.
- Render observations with the same `format_observation` used by the data builder.
- Generate one action at a time with the VirtualHome prompt.
- Parse generated actions into VirtualHome EvolvingGraph script lines.
- Execute actions in EvolvingGraph and stop on goal success or `max_steps`.
- Write rollout SR/PS summaries under the data-disk checkpoint output directory.

Files:

- `wormi/scripts/eval_vh_rollout.py`
- `wormi/scripts/main.py` now registers `wormi eval-vh-rollout`
- `sh/wormi-eval-vh-rollout.sh`

Final smoke test:

- Command: `NUM_SAMPLES=1 MAX_STEPS=2 OUTPUT_PATH=/root/autodl-tmp/wormi-checkpoints/wormi-vh/wormi-vh-n6/vh-rollout-smoke-final sh/wormi-eval-vh-rollout.sh`
- Output: `/root/autodl-tmp/wormi-checkpoints/wormi-vh/wormi-vh-n6/vh-rollout-smoke-final/vh-rollout-summary.tsv`
- Result: CLI, checkpoint loading, retrieval, generation, action parsing, EvolvingGraph execution, and summary writing all completed.

Smoke summary:

| split | episodes | SR | PS | invalid_actions | executed_actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| col_1_seen_seen | 1 | 0.000000 | 2.000000 | 1.000000 | 1.000000 |
| col_2_seen_unseen | 1 | 0.000000 | 2.000000 | 1.000000 | 1.000000 |
| col_3_unseen_unseen | 1 | 0.000000 | 2.000000 | 0.000000 | 2.000000 |

Validation:

- Python compile passed for `wormi/scripts/eval_vh_rollout.py`.
- 10 complete expert trajectories from `eval_col_2_seen_unseen/test.jsonl` executed through EvolvingGraph and reached their goals.

## Full VirtualHome Rollout Eval

Completed at `2026-05-19T04:21:47+08:00`.

Command:

```bash
RUN_ID=20260519_015830 sh/wormi-eval-vh-rollout.sh
```

Inputs:

- curricula: `tools/wormi_curricula_vh.py`
- model: `/root/autodl-tmp/wormi-checkpoints/wormi-vh/wormi-vh-n6/last`
- scene init cache: `/root/autodl-tmp/wormi-data/scene-inits/init_graphs_20.json`
- VirtualHome source: `/root/autodl-tmp/wormi-data/virtualhome-src`
- max rollout steps: `30`
- temperature: `1.0`

Outputs:

- summary TSV: `/root/autodl-tmp/wormi-checkpoints/wormi-vh/wormi-vh-n6/vh-rollout/vh-rollout-summary.tsv`
- summary JSON: `/root/autodl-tmp/wormi-checkpoints/wormi-vh/wormi-vh-n6/vh-rollout/vh-rollout-summary.json`
- step details: `/root/autodl-tmp/wormi-checkpoints/wormi-vh/wormi-vh-n6/vh-rollout/vh-rollout-*.jsonl`
- episode details: `/root/autodl-tmp/wormi-checkpoints/wormi-vh/wormi-vh-n6/vh-rollout/vh-rollout-*-episodes.jsonl`
- log: `/root/autodl-tmp/wormi-logs/vh-rollout-20260519_015830/eval.log`

Final rollout summary:

| split | episodes | SR | PS | invalid_actions | executed_actions | world_models |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| col_1_seen_seen | 29 | 0.241379 | 26.724138 | 5.862069 | 20.862069 | 5,4,3 |
| col_2_seen_unseen | 149 | 0.342282 | 24.288591 | 5.006711 | 19.281879 | 2,0,1 |
| col_3_unseen_unseen | 300 | 0.323333 | 25.146667 | 4.756667 | 20.390000 | 2,3,1 |

Paper Table 1 WorMI reference:

| split | paper SR | paper PS | local rollout SR | local rollout PS |
| --- | ---: | ---: | ---: | ---: |
| Seen Tasks & Seen Scenes | 85.78% | 10.76 | 24.14% | 26.72 |
| Seen Tasks & Unseen Scenes | 80.26% | 12.42 | 34.23% | 24.29 |
| Unseen Tasks & Unseen Scenes | 66.12% | 15.17 | 32.33% | 25.15 |

The metric is now a rollout metric, not the previous offline exact-match metric. The gap to the paper remains large and is now more likely due to data/split differences, local training deviations, prompt/action parsing differences, and hyperparameter deviations rather than the offline evaluator mismatch alone.

## Rollout Error Analysis

Performed after the full rollout result above.

Main observed failure pattern:

- The model over-generates `walk`.
  - `col_1_seen_seen`: 623/775 generated steps were parsed as `walk`.
  - `col_2_seen_unseen`: 2901/3616 generated steps were parsed as `walk`.
  - `col_3_unseen_unseen`: 5877/7544 generated steps were parsed as `walk`.
- `placein` is the weakest family.
  - `col_1_seen_seen`: SR 15.38%, PS 28.38
  - `col_2_seen_unseen`: SR 2.13%, PS 29.87
  - `col_3_unseen_unseen`: SR 16.16%, PS 28.54
- `open` is easy under rollout.
  - `col_2_seen_unseen`: SR 100%, PS 6.22
  - `col_3_unseen_unseen`: SR 100%, PS 3.50
- Most failed `puton` / `placein` episodes loop between source and target objects without reliably doing the required `grab` and final `put`/`putin`.
- Most invalid actions are EvolvingGraph precondition failures. Typical examples:
  - `grab drawing` before walking close enough to drawing.
  - `open oven` before walking close enough to oven.
  - `put chair nightstand` before holding the chair.

Evaluator issue found and fixed:

- Bug: generated `walk toilet` matched the longer prefix `walk to`, leaving object text `ilet`.
- Fix: action-prefix parsing now requires token-boundary consumption.
- Added cautious fuzzy object matching for common small model typos such as `coffe maker`, `coffee maker`, `freeer`, and `microwaves`.
- File changed: `wormi/scripts/eval_vh_rollout.py`

Replay using the already saved model predictions, but the patched parser:

| split | old SR | old PS | patched-replay SR | patched-replay PS |
| --- | ---: | ---: | ---: | ---: |
| col_1_seen_seen | 0.2414 | 26.72 | 0.3103 | 26.28 |
| col_2_seen_unseen | 0.3423 | 24.29 | 0.3557 | 24.17 |
| col_3_unseen_unseen | 0.3233 | 25.15 | 0.3367 | 24.96 |

Conclusion:

- The parser bug hurt results, especially `walk toilet`, but fixing it only gives a modest improvement.
- The dominant issue is model policy quality under rollout: it does not consistently produce the multi-step sequences needed for `puton` and `placein`.
- A full rerun with the patched parser is needed for exact final numbers, but replay indicates it will not close the gap to the paper by itself.

## Follow-up Check: Why The Gap Remains Large

Additional checks after the poor rollout result:

- Current local rollout is much lower than paper Table 1 WorMI:
  - paper VirtualHome: 85.78 / 80.26 / 66.12 SR for the three columns.
  - local rollout before parser patch: 24.14 / 34.23 / 32.33 SR.
  - patched-parser replay: 31.03 / 35.57 / 33.67 SR.
- The paper uses a real rollout metric: SR is task completion rate and PS is timesteps to completion.
- The current evaluator now follows this rollout style, so the remaining gap is not explained by offline exact-match evaluation.

High-confidence implementation/training differences found:

- Stage-2 meta-learning was not using paper Algorithm 1's `β=0.1` update. The local callback directly averaged post-inner-loop parameters across curricula. This is now fixed in code:
  - `wormi/curricula.py`: added `meta_learning_rate`.
  - `wormi/trainer.py`: Reptile aggregation now uses `theta_old + beta * (mean(theta_j) - theta_old)`.
  - `tools/wormi_curricula_vh.py`: sets `meta_learning_rate=0.1`.
- This change requires rerunning stage-2 WorMI adapter training. Existing world models do not need to be retrained for this specific fix.
- Local training still used smaller effective batch sizes than paper in earlier runs because of GPU memory constraints. Paper Table A.6 uses batch size 4 for both world-model training and compound-attention training.
- The local split sizes are not obviously paper-identical. Current generated row counts:
  - seen-task/seen-scene eval: 30 rows / 29 rollout episodes.
  - seen-task/unseen-scene eval: 595 rows / 149 rollout episodes.
  - unseen-task/unseen-scene eval: 2389 rows / 300 rollout episodes.
  Paper states 1,023 VirtualHome episodes over 78 tasks and 20 scenes; current reconstruction may still differ from the authors' private episode sampling.

Prompt/action-format note:

- The local system prompt matches paper Figure A.4.
- Paper Table A.1 shows `put` and `putin` as two-argument actions (`put apple table`, `putin apple fridge`), which also matches local labels.
- Since the prompt wording itself is paper-aligned, do not change it yet if the goal is strict alignment. The model should learn the two-argument form from demonstrations.

Next corrective step:

- Rerun stage-2 training with the fixed `β=0.1` meta update, then rerun patched rollout eval.
- World-model retraining is only needed if we decide to restore paper batch size 4/gradient behavior for stage 1, or if the VirtualHome dataset reconstruction changes.

## Stage-2 Rerun With Paper `β=0.1`

Code change tested:

- `WorMICurricula.meta_learning_rate=0.1`
- Meta aggregation changed from direct averaging to:
  `theta_old + beta * (mean(theta_j) - theta_old)`

Training command/output:

- command: `CKPT_ROOT=/root/autodl-tmp/wormi-checkpoints/wormi-vh-beta01 WORMI_VH_STAGE2_BATCH_SIZE=1 .venv/bin/wormi train --curricula_path tools/wormi_curricula_vh.py`
- checkpoint: `/root/autodl-tmp/wormi-checkpoints/wormi-vh-beta01/wormi-vh-n6/last`
- final training loss: `0.5017`
- final logged eval loss: `0.3820`
- run time: about 30 minutes

Full patched rollout eval:

- command: `MODEL_NAME=/root/autodl-tmp/wormi-checkpoints/wormi-vh-beta01/wormi-vh-n6/last OUTPUT_PATH=/root/autodl-tmp/wormi-checkpoints/wormi-vh-beta01/wormi-vh-n6/vh-rollout-full RUN_ID=20260519_beta01_rollout_full sh/wormi-eval-vh-rollout.sh`
- log: `/root/autodl-tmp/wormi-logs/vh-rollout-20260519_beta01_rollout_full/eval.log`
- summary: `/root/autodl-tmp/wormi-checkpoints/wormi-vh-beta01/wormi-vh-n6/vh-rollout-full/vh-rollout-summary.tsv`

Final result:

| split | episodes | SR | PS | invalid_actions | executed_actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| col_1_seen_seen | 29 | 10.34% | 28.41 | 1.52 | 26.90 |
| col_2_seen_unseen | 149 | 17.45% | 26.44 | 1.56 | 24.89 |
| col_3_unseen_unseen | 300 | 13.33% | 27.75 | 2.40 | 25.35 |

This is worse than the previous stage-2 checkpoint, whose patched-parser replay was approximately:

| split | previous patched-replay SR | beta01 full SR |
| --- | ---: | ---: |
| col_1_seen_seen | 31.03% | 10.34% |
| col_2_seen_unseen | 35.57% | 17.45% |
| col_3_unseen_unseen | 33.67% | 13.33% |

Action distribution confirms the same failure mode, now stronger:

| split | generated steps | `walk` predictions |
| --- | ---: | ---: |
| col_1_seen_seen | 824 | 766 |
| col_2_seen_unseen | 3940 | 3632 |
| col_3_unseen_unseen | 8325 | 7428 |

Conclusion from this rerun:

- The missing `β=0.1` implementation was a real paper-alignment bug and should stay fixed.
- However, with local batch size 1, the paper `β=0.1` rerun makes the rollout policy worse, not better.
- The dominant failure is still the learned policy collapsing to repeated `walk`, not evaluator parsing.
- Next highest-priority alignment item is not another stage-2 rerun with the same settings; it is restoring the paper's effective batch/training dynamics or changing the data reconstruction/split to match the authors' VirtualHome episodes more closely.

## VirtualHome Data Pipeline Re-audit

Rechecked the data generation/processing path from scratch, ignoring the previous training-focused hypothesis.

Source constraints from the paper:

- VirtualHome has 1,023 episodes, 78 tasks, and 20 scenes.
- The paper task families are 9 `turnon`, 7 `open`, 30 `puton`, and 32 `placein`.
- Observations are represented as graph triples; actions are `walk`, `grab`, `open`, `put`, `putin`, and `switchon`.

Confirmed local task sampling:

- `tools/build_virtualhome_dataset.py` samples 78 instructions with the correct family counts.
- The local seen/unseen task split is 16 / 62.
- The scene-init cache contains 20 scene graphs.

Definite bug found:

- The old builder split `seen_seen` train/test by individual action rows, not by full trajectory/episode.
- This fragmented episodes across train/test:
  - current `scene_0/train.jsonl`: 12 trajectories, 4 incomplete.
  - current `scene_1/train.jsonl`: 11 trajectories, 5 incomplete.
  - current `scene_2/train.jsonl`: 13 trajectories, 5 incomplete.
  - current `scene_3/train.jsonl`: 14 trajectories, 5 incomplete.
  - current `scene_4/train.jsonl`: 13 trajectories, 5 incomplete.
  - current `scene_5/train.jsonl`: 14 trajectories, 5 incomplete.
  - current `test_seen_task_seen_scene.jsonl`: 29 trajectories, all 29 incomplete.
- This also creates train/test leakage at the trajectory level: 29 trajectories span both scene train files and the root seen/seen test file.

Fix implemented:

- `tools/build_virtualhome_dataset.py` now groups `seen_seen` rows by `_meta.trajectory_id` before holdout.
- Train/test split now moves whole trajectories.
- The builder now prints rows and trajectory counts separately.
- The builder now warns when generated trajectory count differs from the paper's 1,023.
- The builder now prints scenes with zero successful episodes.

Validation build:

- output: `/root/autodl-tmp/wormi-data/virtualhome-check-trajectory-split`
- no current training data was overwritten.
- all generated jsonl files have complete trajectories after the fix.
- train/test trajectory leakage is now 0.

Validation build counts:

| split/file | rows | trajectories |
| --- | ---: | ---: |
| test_seen_task_seen_scene.jsonl | 24 | 6 |
| test_seen_task_unseen_scene.jsonl | 595 | 149 |
| test_unseen_task_seen_scene.jsonl | 1133 | 280 |
| test_unseen_task_unseen_scene.jsonl | 2389 | 590 |
| scene_0/train.jsonl | 46 | 11 |
| scene_1/train.jsonl | 42 | 10 |
| scene_2/train.jsonl | 49 | 12 |
| scene_3/train.jsonl | 53 | 13 |
| scene_4/train.jsonl | 48 | 12 |
| scene_5/train.jsonl | 51 | 13 |

Remaining data-alignment problems:

- `TrimmedTestScene6_graph__v0` has zero successful generated episodes, so although the cache has 20 scene graphs, the actual generated dataset covers only 19 effective scenes.
- The generated trajectory count is 1,096, not the paper's 1,023.
- The Table-1 eval columns from the local generator cover 6 / 149 / 590 trajectories, not an obvious paper-identical split.
- The current observation text loses node states: `open` and `switchon` transitions have identical `observation` and `next_observation` 100% of the time in the current data. This is risky for rollout because the policy cannot observe whether a container is already open or a switchable object is already on. I did not change this yet because the paper's Figure A.2 describes relation triples and does not clearly include node states.

Current conclusion:

- Yes, there is a real data-processing bug: row-level trajectory splitting.
- Fixing it requires rebuilding VirtualHome data and then retraining stage 1 and stage 2, because the training jsonl contents change.
- Even after this fix, the local synthetic data generator still does not match the paper dataset exactly: scene coverage and total episode count remain off.

## VirtualHome Data Functional Validity Audit

Added a reusable validator:

- script: `tools/validate_virtualhome_dataset.py`
- latest JSON report: `reports/vh-data-validation-fixedtraj-2026-05-19.json`
- validated data root: `/root/autodl-tmp/wormi-data/virtualhome`

Scope of validation:

- File layout and symlink targets used by the curricula.
- Row schema: `instruction`, `observation`, `action`, `next_observation`, `_meta`.
- `_meta` consistency: `scene`, `split`, `task_args`, `trajectory_id`, `step_index`, `num_steps`.
- Full-trajectory grouping and train/test leakage.
- Action vocabulary and per-task action sequence.
- `AutoJsonlDataset` / `VirtualHomeDataset` loader compatibility.
- Dynamic replay: convert every jsonl expert action back to a VirtualHome script, execute it from the cached scene init graph, and compare every replayed `observation` / `next_observation` against the jsonl text.
- Final graph goal satisfaction.
- Goal fact visibility in the final `next_observation`.

Hard functional checks passed:

- Total rows checked: 4,430.
- Total trajectories checked: 1,096.
- Train/test trajectory overlap: 0.
- Expert replay failures: 0.
- Replay observation mismatches: 0.
- Replay next-observation mismatches: 0.
- Final graph goal failures: 0.
- Loader smoke test passed for all 10 core jsonl files:
  - `end_with_action=True` produces 1 action sample per raw row.
  - `end_with_action=False` produces 2 samples per raw row (action-only + next-observation supervision).

Counts from the current fixed-trajectory main data:

| item | count |
| --- | ---: |
| rows | 4,430 |
| trajectories | 1,096 |
| seen tasks with data | 15 |
| unseen tasks with data | 55 |
| effective scenes with rows | 19 |

Observed action counts:

| action | rows |
| --- | ---: |
| walk | 2,032 |
| grab | 936 |
| open | 444 |
| put | 570 |
| putin | 366 |
| switchon | 82 |

Task-count validity problem:

- The builder samples 78 tasks with the intended family counts, but only 70 sampled tasks produce at least one successful trajectory.
- Expected seen/unseen task split is still 16 / 62, but observed data contains only 15 / 55 tasks.
- Missing seen task:
  - `placein drawing sink`
- Missing unseen tasks:
  - `open door`
  - `placein chair sink`
  - `placein chair toaster`
  - `placein drawing toaster`
  - `placein keyboard sink`
  - `placein keyboard toaster`
  - `turnon light`

Scene-count validity problem:

- `TrimmedTestScene6_graph__v0` has no successful trajectories.
- The scene cache contains 20 graphs, but only 19 scene keys appear in the generated rows.

Observation-validity problem:

- The data is executable, but current `format_observation()` hides many final goal facts.
- Final goal visibility in the current jsonl:

| family | visible goals | hidden goals |
| --- | ---: | ---: |
| turnon | 0 | 82 |
| open | 0 | 78 |
| placein | 0 | 366 |
| puton | 532 | 38 |

Root causes:

- `open` / `switchon`: node states are not emitted as triples, so there are no `(object, is, open)` or `(object, is, on)` facts.
- `placein`: `INSIDE` relations are only emitted when the target is a room; object-inside-container facts such as `(towel, inside, closet)` are dropped.
- `puton`: the hidden 38 cases are `chair on floor` and `drawing on floor`; some floor nodes are filtered by category, so the final `(object, on, floor)` goal can be absent from the observation text.

Transition-observation issue:

- `open`: 444 / 444 rows have identical `observation` and `next_observation`.
- `switchon`: 82 / 82 rows have identical `observation` and `next_observation`.
- `walk`: 15 / 2,032 rows are unchanged.

What-if formatter check:

- Without changing files, replayed the same 1,096 trajectories with a candidate formatter that adds:
  - object-inside-container triples;
  - `(object, is, open)`;
  - `(object, is, on)`.
- Resulting final goal visibility:
  - `turnon`: 82 / 82 visible.
  - `open`: 78 / 78 visible.
  - `placein`: 366 / 366 visible.
  - `puton`: still 532 / 570 visible unless floor handling is also fixed.
- Average triples per observation would increase from about 177.2 to 224.8.

Current audit conclusion:

- The current fixed-trajectory data is mechanically valid for training and replay: it loads, groups, executes, and reaches the intended hidden graph goals.
- It is not semantically sufficient for a world-model transition objective because key state changes and several final goal facts are missing from the observation text.
- Before rerunning stage 1 / stage 2, the next data fix should be:
  1. add state triples for `OPEN` and `ON`;
  2. preserve object-inside-container triples;
  3. make floor handling consistent, or exclude floor from generated `puton` task targets;
  4. select/filter sampled tasks by successful executable trajectories so the observed task count is actually 78;
  5. rebuild data and rerun this validator before training.

## Planned Semantic Data Fix

Do this before any further stage-1/stage-2 training.

Implementation targets:

- Update `tools/build_virtualhome_dataset.py::format_observation` so the text state exposes the graph facts needed by the four VH goals:
  - keep object-inside-container triples, not only object-inside-room triples;
  - emit `(object, is, open)` / `(object, is, closed)`;
  - emit `(object, is, on)` / `(object, is, off)`;
  - keep `(object, on, floor)` only if floor remains a valid task target.
- Remove floor-like classes from `puton` candidate surfaces. Floor is a structural support, not a good task target for this benchmark reconstruction.
- Change task selection from "sample 78 then discover failures" to "evaluate candidates, then select 78 valid tasks":
  - each selected task must have at least one executable trajectory;
  - each selected task must have its final goal fact visible in `next_observation`;
  - selected seen tasks should have successful seen-scene trajectories so every world model gets meaningful stage-1 rows.
- Persist a manifest next to the jsonl data recording the selected tasks, scenes, split, family, args, and goal triple.
- Rerun `tools/validate_virtualhome_dataset.py` after rebuild and treat these as hard gates before training:
  - replay failures = 0;
  - observation mismatches = 0;
  - final graph goal failures = 0;
  - hidden goal facts = 0;
  - actual task count = 78;
  - effective scenes = 20;
  - train/test trajectory overlap = 0.

## Semantic Data Fix Implemented

Code changes:

- `tools/build_virtualhome_dataset.py`
  - `format_observation()` now emits compact state triples for `open/closed`, `on/off`, and plugged states.
  - `INSIDE` relations now preserve object-inside-container facts, not only object-inside-room facts.
  - `floor` is excluded from generated `puton` targets.
  - `door` is excluded as a goal class because door nodes are structural and filtered from observations.
  - Task selection now evaluates candidate tasks first, then selects 78 semantically valid tasks.
  - Seen-task selection is coverage-aware over the 6 seen scenes.
  - Writes `virtualhome_manifest.json`.
  - Added optional `--target-trajectories`, used with `1023` for paper-count data.
- `tools/build_virtualhome_dataset.py scene-cache`
  - New subcommand that builds a 20-scene init-graph cache from the raw VH zip while skipping init graphs that fail semantic probes.
- `tools/validate_virtualhome_dataset.py`
  - Now reads `virtualhome_manifest.json` when present.
  - Default scene cache path changed to the semantic cache.
- `sh/wormi-build-vh-data.sh`
  - Default scene cache changed to `init_graphs_20_semantic.json`.
  - Builds semantic scene cache if missing or if `REBUILD_SCENE_INITS=1`.
  - Calls `tools/build_virtualhome_dataset.py scene-cache`; the old standalone scene-cache script was removed.
  - Passes `--candidate-multiplier 12`.
  - Passes `--target-trajectories 1023`.
- `sh/wormi-eval-vh-rollout.sh`
  - Default scene cache changed to `init_graphs_20_semantic.json`.

Validated intermediate semantic full build:

- scene cache: `/root/autodl-tmp/wormi-data/scene-inits/init_graphs_20_semantic.json`
- data root: `/root/autodl-tmp/wormi-data/virtualhome-semantic-cache-check`
- report: `reports/vh-data-validation-semantic-cache-check-2026-05-19.json`
- result:
  - 1,225 trajectories;
  - 78 actual tasks;
  - 20 effective scenes;
  - train/test trajectory overlap = 0;
  - replay failures = 0;
  - observation mismatches = 0;
  - final graph goal failures = 0;
  - hidden goal facts = 0.

Validated paper-count semantic build:

- data root: `/root/autodl-tmp/wormi-data/virtualhome-semantic-1023-check`
- report: `reports/vh-data-validation-semantic-1023-check-2026-05-19.json`
- result:
  - 4,172 rows;
  - 1,023 trajectories;
  - 78 actual tasks: 16 seen / 62 unseen;
  - 20 effective scenes;
  - train/test trajectory overlap = 0;
  - replay failures = 0;
  - observation mismatches = 0;
  - next-observation mismatches = 0;
  - final graph goal failures = 0;
  - hidden goal facts = 0.

Paper-count semantic build counts:

| item | count |
| --- | ---: |
| rows | 4,172 |
| trajectories | 1,023 |
| seen tasks | 16 |
| unseen tasks | 62 |
| scenes | 20 |
| open trajectories | 69 |
| placein trajectories | 378 |
| puton trajectories | 496 |
| turnon trajectories | 80 |

Remaining warning:

- `walk` has 6 unchanged transitions out of 1,897. This is acceptable for now because `walk` can target an already-close class-level object or duplicate class; all goal-changing actions now expose their final goal facts.

Next action:

- Main data has now been rebuilt with `sh/wormi-build-vh-data.sh`.
- Old main data backup:
  - `/root/autodl-tmp/wormi-data/virtualhome.bak.semantic1023-20260519`
- Current main data:
  - `/root/autodl-tmp/wormi-data/virtualhome`
- Main validation report:
  - `reports/vh-data-validation-main-semantic-1023-2026-05-19.json`
- Main validation result:
  - 4,172 rows;
  - 1,023 trajectories;
  - 78 actual tasks: 16 seen / 62 unseen;
  - 20 effective scenes;
  - train/test trajectory overlap = 0;
  - replay failures = 0;
  - observation mismatches = 0;
  - next-observation mismatches = 0;
  - final graph goal failures = 0;
  - hidden goal facts = 0.

Next action:

- Retrain stage-1 world models from scratch.
- Then retrain stage-2 WorMI adapters from scratch.

## Data Processing Code Cleanup

The VirtualHome data-processing code has been reduced to two Python entry points:

- `tools/build_virtualhome_dataset.py`
  - canonical builder module;
  - old direct build CLI is still supported;
  - new explicit subcommands:
    - `python tools/build_virtualhome_dataset.py scene-cache ...`
    - `python tools/build_virtualhome_dataset.py build ...`
  - includes semantic scene-cache construction, dataset generation, manifest writing, semantic task selection, and optional 1,023-trajectory downsampling.
- `tools/validate_virtualhome_dataset.py`
  - validator only;
  - checks file layout, schema, split leakage, replay, observation consistency, final goal satisfaction, goal visibility, and loader behavior.

Removed:

- `tools/build_virtualhome_scene_cache.py`
  - functionality moved into `tools/build_virtualhome_dataset.py scene-cache`.

Updated:

- `sh/wormi-build-vh-data.sh` now calls `tools/build_virtualhome_dataset.py scene-cache` when the semantic scene cache is missing/stale.

Sanity checks passed:

- `python -m py_compile tools/build_virtualhome_dataset.py tools/validate_virtualhome_dataset.py wormi/scripts/eval_vh_rollout.py`
- `bash -n sh/wormi-build-vh-data.sh sh/wormi-eval-vh-rollout.sh`
- `python tools/build_virtualhome_dataset.py scene-cache --help`
- `python tools/build_virtualhome_dataset.py build --help`
- `python tools/validate_virtualhome_dataset.py --help`

## Stage-2 Restart After Semantic 1023 Rebuild

Paper / appendix settings rechecked before launch:

- Source checked: arXiv `2509.03956` and PMLR paper page.
- VirtualHome data: 1,023 episodes, 78 tasks, 16 seen tasks, 62 unseen tasks, 20 scenes.
- Table A.6 stage-1 world models:
  - base model Llama-3.2-1B;
  - 2,000 gradient steps;
  - batch size 4;
  - LR 3e-5;
  - cosine scheduler;
  - intermediate connection layer [13, 27].
- Table A.6 stage-2 compound attention:
  - reasoning model Llama-3.2-3B;
  - batch size 4;
  - meta update steps lambda_M = 8;
  - inner-loop gradient steps lambda_I = 30;
  - LR alpha = 1e-5;
  - meta LR beta = 0.1;
  - reasoning connection layers [13, 27];
  - world connection layers [7, 15];
  - prototype size k = 15;
  - world models N = 6;
  - retrieved world models K = 3.

Local alignment before stage-2 restart:

- Curricula: `tools/wormi_curricula_vh.py`
- Data root: `/root/autodl-tmp/wormi-data/virtualhome`
- Stage-1 world root: `/root/autodl-tmp/wormi-checkpoints/world-vh-semantic1023-rerun`
- Stage-2 output root: `/root/autodl-tmp/wormi-checkpoints/wormi-vh-semantic1023-rerun`
- `meta_learning_rate=0.1` is now implemented and enabled.
- N=6, K=3, prototype size 15, lambda_M=8, lambda_I=30, alpha=1e-5, and connection layers match Table A.6.

Strict batch-size attempt:

- Command launched with `WORMI_VH_STAGE2_BATCH_SIZE=4`.
- Log:
  - `/root/autodl-tmp/wormi-logs/vh-wormi-semantic1023-rerun-stage2-20260520/train.log`
- Result: CUDA OOM on L40S before completing the first training step.
- Failure detail: process used about 44.30 GiB and failed when allocating another 5.14 GiB in Llama `lm_head`.
- Failed batch-4 process was terminated and GPU memory was released.

Continuation decision:

- Continue stage-2 with paper settings unchanged except `WORMI_VH_STAGE2_BATCH_SIZE=1`, because batch 4 does not fit on the available 48 GB GPU with the current full-precision model stack.


## Task-Aware VirtualHome Data Rebuild

Backup before cleanup:

- Processing script snapshot: `/root/WorMI/processing-scripts-backup-20260520-115157`
- Backed up `tools/`, `sh/`, `wormi/scripts/eval_table1.py`, and `wormi/scripts/eval_vh_rollout.py`.

Final VirtualHome data-processing entry points:

- `tools/build_virtualhome_dataset.py`
- `tools/validate_virtualhome_dataset.py`

Removed temporary split-variant scripts after backup:

- `tools/build_virtualhome_eval_variants.py`
- `tools/wormi_curricula_vh_eval_seen_seen_variants.py`

Implemented split correction:

- Table 1 col1 (`seen task x seen scene`) no longer uses a plain random 9:1 per-scene holdout.
- The builder now holds out `--seen-seen-eval-per-task 2` trajectories per seen task, greedily spread across seen scenes.
- This gives direct coverage over all 16 seen tasks while keeping the world-model train split on the remaining seen-task/seen-scene trajectories.

Rebuilt dataset:

- Command: `bash sh/wormi-build-vh-data.sh`
- Log: `/root/autodl-tmp/wormi-logs/vh-data-20260520_195342/build.log`
- Data root: `/root/autodl-tmp/wormi-data/virtualhome`
- Target trajectories: 1,023
- Selected tasks: 78 total, 16 seen, 62 unseen
- Selected task-family counts: `open=7`, `placein=32`, `puton=30`, `turnon=9`

New split counts:

- `test_seen_task_seen_scene.jsonl`: 130 rows, 32 trajectories, 16 tasks, 6 scenes, families `open=2`, `placein=14`, `puton=12`, `turnon=4`
- `test_seen_task_unseen_scene.jsonl`: 760 rows, 186 trajectories, 16 tasks, 14 scenes
- `test_unseen_task_unseen_scene.jsonl`: 1847 rows, 457 trajectories, 62 tasks, 14 scenes
- `test_unseen_task_seen_scene.jsonl`: 1216 rows, 294 trajectories, 61 tasks, 6 scenes; not used by Table 1
- World-model train trajectories across seen scenes: 54

Validation:

- Report: `reports/vh-data-validation-taskaware-split-2026-05-20.json`
- Rows: 4,172
- Trajectories: 1,023
- Train/test overlap: 0
- Missing seen tasks: none
- Missing unseen tasks: none
- Replay failures: 0
- Observation mismatches: 0
- Final goal failures: 0
- Warning retained: 6 unchanged `walk` observations out of 1,897 `walk` actions; semantic goal checks still pass.

Training implication:

- Because the seen-task/seen-scene train/test partition changed, stage-1 world models and stage-2 WorMI checkpoints should be retrained before final Table 1 eval.


## Stage-1 Restart On Task-Aware Data

Started detached stage-1 world-model training after the task-aware data rebuild.

- PID wrapper: `30350`
- Training PID: `30355`
- Data root: `/root/autodl-tmp/wormi-data/virtualhome`
- Checkpoint root: `/root/autodl-tmp/wormi-checkpoints/world-vh-taskaware-split-20260520`
- Launch log: `/root/autodl-tmp/wormi-logs/vh-world-taskaware-split-stage1-20260520-detached/launch.log`
- Train log: `/root/autodl-tmp/wormi-logs/vh-world-taskaware-split-stage1-20260520/train.log`
- Initial status: training reached step 37/2000 on the first world-model run; GPU memory about 47.3 GiB in use on RTX 4090.


## Stage-1 Complete And Stage-2 Started

Stage-1 task-aware world-model training completed normally.

- Completed: `2026-05-20T22:45:45+08:00`
- Checkpoint root: `/root/autodl-tmp/wormi-checkpoints/world-vh-taskaware-split-20260520`
- Completed world models: 6/6, each with `checkpoint-2000` and `last`
- GPU released after stage-1.

Started detached stage-2 adapter training on the task-aware split.

- Wrapper PID: `40868`
- Training PID: `40873`
- Data root: `/root/autodl-tmp/wormi-data/virtualhome`
- World checkpoint root: `/root/autodl-tmp/wormi-checkpoints/world-vh-taskaware-split-20260520`
- Stage-2 output root: `/root/autodl-tmp/wormi-checkpoints/wormi-vh-taskaware-split-20260520`
- Train log: `/root/autodl-tmp/wormi-logs/vh-wormi-taskaware-split-stage2-bs1-20260520/train.log`
- Initial status: all 6 trainers reached ready state; no OOM at startup.


## Stage-2 Restart With Gradient Accumulation

Canceled the previous task-aware stage-2 run because `batch_size=1` was not using gradient accumulation and therefore did not approximate the paper batch size 4.

Backups kept:

- Code backup before gradient-accumulation edit: `/root/WorMI/processing-scripts-backup-gradaccum-20260520-232224`
- Aborted no-accum checkpoint backup: `/root/autodl-tmp/wormi-checkpoints/wormi-vh-taskaware-split-20260520.aborted-bs1-noaccum-20260520-232224`
- Aborted no-accum log backup: `/root/autodl-tmp/wormi-logs/vh-wormi-taskaware-split-stage2-bs1-20260520.aborted-bs1-noaccum-20260520-232224`

Code changes:

- `wormi/trainer.py` adds `gradient_accumulation_steps` to `WorMITrainerConfig` and passes it into `SFTConfig`.
- `tools/wormi_curricula_vh.py` reads `WORMI_VH_STAGE2_GRADIENT_ACCUMULATION_STEPS`, defaulting to 4 for VirtualHome stage-2.
- `sh/wormi-train-vh-wormi.sh` and `sh/wormi-train-vh-wormi-background.sh` export/pass the variable and print effective batch size.

New detached stage-2 run:

- Run id: `taskaware-split-stage2-bs1-ga4-20260520`
- Wrapper PID: `45876`
- Training PID: `45881`
- Per-device batch size: 1
- Gradient accumulation steps: 4
- Effective batch size: 4
- Output root: `/root/autodl-tmp/wormi-checkpoints/wormi-vh-taskaware-split-20260520`
- Train log: `/root/autodl-tmp/wormi-logs/vh-wormi-taskaware-split-stage2-bs1-ga4-20260520/train.log`
- Initial status: all 6 trainers ready and training progressed past step 6/240 without startup OOM.


## Auto Eval Watcher

Started a detached watcher to run evaluation automatically after the current task-aware stage-2 run exits.

- Watcher PID: `54151`
- Watched stage-2 training PID: `45881`
- Watch log: `/root/autodl-tmp/wormi-logs/vh-auto-eval-taskaware-bs1-ga4-20260521/watch.log`
- Detached launch log: `/root/autodl-tmp/wormi-logs/vh-auto-eval-taskaware-bs1-ga4-20260521-detached/launch.log`
- Model expected after training: `/root/autodl-tmp/wormi-checkpoints/wormi-vh-taskaware-split-20260520/wormi-vh-n6/last`
- Table1 output: `/root/autodl-tmp/wormi-checkpoints/wormi-vh-taskaware-split-20260520/wormi-vh-n6/table1-auto-taskaware-bs1-ga4-20260521`
- Rollout output: `/root/autodl-tmp/wormi-checkpoints/wormi-vh-taskaware-split-20260520/wormi-vh-n6/vh-rollout-auto-taskaware-bs1-ga4-20260521`

The watcher polls every 300 seconds. It skips evaluation if the stage-2 train log contains `Traceback`, `CUDA out of memory`, or `RuntimeError`, or if the final `last` model directory is missing.


## Salvage Eval Results From Failed Stage-2 Checkpoint

Full eval completed from the saved `last` checkpoint of the task-aware stage-2 run. This checkpoint is usable for inspection, but the stage-2 process ended with `RuntimeError: release unlocked lock`, so these are marked as salvage results rather than clean final reproduction results.

Paths:

- Full eval log: `/root/autodl-tmp/wormi-logs/vh-full-eval-full-salvage-after-failed-stage2-20260521/full-eval.log`
- Table1 output: `/root/autodl-tmp/wormi-checkpoints/wormi-vh-taskaware-split-20260520/wormi-vh-n6/table1-full-salvage-after-failed-stage2-20260521`
- Rollout output: `/root/autodl-tmp/wormi-checkpoints/wormi-vh-taskaware-split-20260520/wormi-vh-n6/vh-rollout-full-salvage-after-failed-stage2-20260521`

Table1-style offline eval:

| Split | Episodes | SR | PS | Avg total steps | World models |
| --- | ---: | ---: | ---: | ---: | --- |
| col_1_seen_seen | 32 | 0.8125 | 0.3750 | 4.0625 | 4,0,1 |
| col_2_seen_unseen | 186 | 0.5484 | 1.0591 | 4.0860 | 2,5,0 |
| col_3_unseen_unseen | 300 | 0.2433 | 1.9500 | 4.0467 | 5,4,3 |

VirtualHome rollout eval:

| Split | Episodes | SR | PS/path steps | Invalid actions | Executed actions | Max steps | World models |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| col_1_seen_seen | 32 | 0.9375 | 6.1563 | 1.8438 | 4.3125 | 30 | 4,0,1 |
| col_2_seen_unseen | 186 | 0.6398 | 13.7527 | 9.5914 | 4.1613 | 30 | 2,5,0 |
| col_3_unseen_unseen | 300 | 0.3600 | 21.1567 | 13.7000 | 7.4567 | 30 | 5,4,3 |

## Stage-2 Meta-Learning Debug And Clean Rerun

Debug finding after the large gap from the paper:

- The previous stage-2 crash (`RuntimeError: release unlocked lock`) was a symptom of the threaded trainer control flow.
- The deeper reproduction risk was that `WorMIMetaLearningTrainer` ran 6 task trainers as threads over one shared WorMI model with ring locks/global steps. With `synchronize_trainers=False`, this is not a clean implementation of the paper's Reptile-style outer loop: each task should start from the same meta parameters, adapt for the inner steps, then contribute one task delta to the meta update.
- Therefore the salvage checkpoint is useful for diagnosis, but should not be treated as a strict paper-aligned stage-2 result.

Code changes for the rerun:

- Added `WORMI_SEQUENTIAL_META_LEARNING=1` path in `wormi/trainer.py`.
- Sequential path builds the adapter once, snapshots the meta parameters, runs each curriculum independently from the same iteration-start parameters, averages the task parameters, then applies `meta_learning_rate * (mean_task_params - old_params)`.
- Stage-2 scripts now default to sequential meta-learning with `inner_steps=30`, `meta_steps=8`, `batch_size=1`, `gradient_accumulation_steps=4` for effective batch size 4.
- `wormi/scripts/train.py` no longer runs post-train test unless explicitly requested, so final evaluation is controlled by the dedicated eval scripts.

Smoke test:

- Run id: `taskaware-seqmeta-smoke-20260521`
- Settings: `inner_steps=1`, `meta_steps=1`, `grad_accum=1`
- Result: completed all 6 meta tasks, saved `/root/autodl-tmp/wormi-checkpoints/wormi-vh-taskaware-seqmeta-smoke-20260521/wormi-vh-n6/last`, and log had no `Traceback`/`RuntimeError`.

Clean full stage-2 rerun now running detached:

- Run id: `taskaware-seqmeta-stage2-bs1-ga4-20260521`
- Training PID: `23602`
- Output root: `/root/autodl-tmp/wormi-checkpoints/wormi-vh-taskaware-seqmeta-20260521`
- Train log: `/root/autodl-tmp/wormi-logs/vh-wormi-taskaware-seqmeta-stage2-bs1-ga4-20260521/train.log`
- Expected final model: `/root/autodl-tmp/wormi-checkpoints/wormi-vh-taskaware-seqmeta-20260521/wormi-vh-n6/last`

Auto eval watcher:

- Watcher PID: `24735`
- Watch log: `/root/autodl-tmp/wormi-logs/vh-auto-eval-taskaware-seqmeta-stage2-bs1-ga4-20260521/watch.log`
- Table1 output: `/root/autodl-tmp/wormi-checkpoints/wormi-vh-taskaware-seqmeta-20260521/wormi-vh-n6/table1-auto-taskaware-seqmeta-stage2-bs1-ga4-20260521`
- Rollout output: `/root/autodl-tmp/wormi-checkpoints/wormi-vh-taskaware-seqmeta-20260521/wormi-vh-n6/vh-rollout-auto-taskaware-seqmeta-stage2-bs1-ga4-20260521`
- Poll interval: 600 seconds. It will skip eval if the training log contains `Traceback`, `CUDA out of memory`, or `RuntimeError`, or if `last` is missing.

## Stage-2 Threaded Clean Fix

Root-cause isolation after the bad sequential run:

- The rebuilt data and Table1 evaluator were not the primary cause. Using the same data/evaluator, the previous salvage checkpoint scored `seen_seen SR=0.875` on an 8-episode quick gate.
- The new sequential checkpoint was already bad at `checkpoint-720`: 8-episode quick gate produced `seen_seen SR=0.0`, `seen_unseen SR=0.0`, `unseen_unseen SR=0.0`. Per-step dumps show first-step action drift such as `walk livingroom` vs target `walk drawing`.
- Therefore the bad result is caused by the new sequential stage-2 training path, not by the data rebuild or eval sampling.

Fix applied for fastest correct result:

- Restored the verified threaded stage-2 path as the default (`WORMI_SEQUENTIAL_META_LEARNING=0`).
- Fixed the threaded trainer lock-release crash by checking `next_lock.locked()` before release in `on_step_begin` and `on_train_end`.
- Restored threaded aggregation to the original mean-of-task-params behavior by default; the beta interpolation path is now behind `WORMI_THREADED_META_USE_BETA=1`.
- Added Table1 per-step prediction dumps and safe empty-output handling.
- Added optional `NUM_SAMPLES` support to `sh/wormi-eval-vh-table1.sh`.
- Added a quick Table1 gate to `sh/wormi-auto-eval-after-stage2.sh`: default 8 episodes per column and minimum `seen_seen SR >= 0.50` before running full Table1 and rollout.

Validation and rerun:

- Threaded smoke run: `taskaware-threaded-smoke-20260521`, `inner_steps=1`, `meta_steps=1`, completed cleanly and saved `last` with no lock crash.
- Full clean threaded stage-2 run started detached.
- Run id: `taskaware-threaded-clean-bs1-ga4-20260521`
- Training PID: `55047`
- Output root: `/root/autodl-tmp/wormi-checkpoints/wormi-vh-taskaware-threaded-clean-20260521`
- Train log: `/root/autodl-tmp/wormi-logs/vh-wormi-taskaware-threaded-clean-bs1-ga4-20260521/train.log`
- Auto-eval watcher PID: `56070`
- Watch log: `/root/autodl-tmp/wormi-logs/vh-auto-eval-taskaware-threaded-clean-bs1-ga4-20260521/watch.log`
- Quick gate output: `/root/autodl-tmp/wormi-checkpoints/wormi-vh-taskaware-threaded-clean-20260521/wormi-vh-n6/table1-quick-taskaware-threaded-clean-bs1-ga4-20260521`
- Full Table1 output: `/root/autodl-tmp/wormi-checkpoints/wormi-vh-taskaware-threaded-clean-20260521/wormi-vh-n6/table1-full-taskaware-threaded-clean-bs1-ga4-20260521`
- Full rollout output: `/root/autodl-tmp/wormi-checkpoints/wormi-vh-taskaware-threaded-clean-20260521/wormi-vh-n6/vh-rollout-full-taskaware-threaded-clean-bs1-ga4-20260521`

