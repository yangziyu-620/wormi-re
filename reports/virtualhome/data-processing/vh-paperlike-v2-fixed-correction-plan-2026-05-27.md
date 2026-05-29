# VirtualHome Paperlike V2 Fixed Correction Plan - 2026-05-27

## Goal

Produce a cleaner VirtualHome reproduction run for WorMI Table 1 style rollout
evaluation, with data semantics and stage2 optimization closer to the paper.

## What To Keep Paper-Faithful

- Keep the paper's `switch`/`switchon` mismatch:
  - Figure A.4 prompt lists `switch [object]`.
  - Table A.1 action strings use `switchon stove`.
  - The local prompt/data now intentionally preserve this instead of rewriting
    both sides to `switchon`.

## Data Corrections

1. Canonicalize room action labels the same way observations are canonicalized.
   - `dining_room -> kitchen`
   - `home_office -> livingroom`
   - `living_room -> livingroom`
   - This prevents examples like an observation mentioning `kitchen` while the
     target action says `walk dining_room`.

2. Remove obvious redundant walk actions from synthetic paperlike programs.
   - No blind `walk start_room` when the agent is already there.
   - No consecutive duplicate walk labels.
   - No post-grab detour back to the source room unless movement is needed.

3. Add the missing world-model affordance auxiliary task.
   - Stage2 behavior cloning remains one row per raw action.
   - Stage1 world-model training now expands each raw row into:
     - behavior cloning: observation/instruction -> action
     - dynamics: observation/action -> next observation
     - affordance: observation -> feasible expert action

## Validation Gates

The rebuilt data must pass `tools/validate_virtualhome_dataset.py` with:

- no train/test trajectory overlap
- all 16 seen tasks and 62 unseen tasks represented
- all 20 scenes represented
- no raw room action labels
- no consecutive duplicate actions
- unchanged-walk rate <= 5%
- replay succeeds against VirtualHome EvolvingGraph
- final replay state satisfies the requested goal
- loader smoke test verifies world samples are exactly `raw_rows * 3`

The current rebuilt data root passed these gates:

`/root/autodl-tmp/wormi-data/virtualhome-paperlike-v2-fixed-20260527`

Validation artifact:

`reports/virtualhome/validation/vh-paperlike-v2-fixed-validation-2026-05-27.json`

## Training/Eval Corrections

Stage2 must not use the threaded default aggregation path for final numbers,
because the threaded default sets the aggregate parameter to the task mean
unless `WORMI_THREADED_META_USE_BETA=1` is enabled. For the main corrected run,
use the sequential meta-learning path:

`WORMI_SEQUENTIAL_META_LEARNING=1`

This follows the Reptile-style update with the curricula's `meta_learning_rate`
(`beta = 0.1`) instead of the threaded default `beta = 1` behavior.

## Active Pipeline

Detached tmux session:

`wormi_v2_fixed_full`

Pipeline script:

`sh/wormi-vh-paperlike-v2-fixed-full-tmux.sh`

Status file:

`/root/autodl-tmp/wormi-logs/vh-pipeline-paperlike-v2-fixed-full-20260527/status.tsv`

Pipeline log:

`/root/autodl-tmp/wormi-logs/vh-pipeline-paperlike-v2-fixed-full-20260527/pipeline.log`

Expected stage order:

1. preflight validation
2. stage1 world-model training for scene_0..scene_5
3. stage2 WorMI adapter training with sequential Reptile
4. offline Table1-style exact-match eval
5. VirtualHome rollout eval

Final rollout summary path:

`/root/autodl-tmp/wormi-checkpoints/wormi-vh-paperlike-v2-fixed-20260527/wormi-vh-n6/vh-rollout-paperlike-v2-fixed-full-20260527/vh-rollout-summary.tsv`

## Runtime Notes

- Stage1 completed for all six world models (`scene_0` through `scene_5`) and
  each scene has a `last` checkpoint.
- Stage2 completed with `WORMI_SEQUENTIAL_META_LEARNING=1` and produced
  `wormi-vh-n6/last`.
- Offline `eval-table1` completed with exact-match SR = 0 for all three VH
  columns. This is not the final paper-style rollout metric. The offline
  evaluator stops an episode at the first action that differs from the expert
  jsonl action, so semantically plausible alternative first steps such as
  `walk bathroom` versus expert `walk drawing` make the whole episode fail.
- The full VirtualHome rollout evaluator is now the authoritative target-state
  check. The current pipeline rollout uses `temperature=1.0`; if the result is
  noisy or unexpectedly poor, run a deterministic follow-up rollout with
  `TEMPERATURE=0`.


### 2026-05-28 early rollout diagnosis

- The `table1-summary.tsv` zero is the offline exact-match evaluator, not
  target-state rollout SR.
- Exact-match is not failing because the model produces empty outputs:
  first-step predictions are non-empty for 100% of episodes.
- First-step exact match remains low:
  - col_1_seen_seen: 6/32 = 18.75%
  - col_2_seen_unseen: 41/212 = 19.34%
  - col_3_unseen_unseen: 66/300 = 22.00%
- Since offline Table1 requires the full expert trajectory prefix to match,
  one plausible but differently ordered action makes the whole episode fail.
  Example: for `Place drawing in toilet`, gold starts with `walk kitchen` or
  `walk drawing`, while the model emits `walk bathroom`.
- The active rollout run uses `temperature=1.0`, so it is a stochastic smoke
  run rather than the best paper-aligned action-selection setting. A detached
  watcher `wormi_v2_fixed_greedy_after_current` was started to run
  `TEMPERATURE=0.0` after the current rollout PID exits.
- TRL collator mask probe passed:
  - world behavior-cloning/affordance samples supervise action tokens;
  - world dynamics samples supervise the next-observation tokens;
  - stage-2 samples supervise action tokens.
  This rules out "all labels are -100" as the cause of zero/low SR.
- Current data size is very small for stage-1/stage-2 seen-scene training:
  scene dirs contain only 10 or 11 train trajectories each after the
  seen_seen holdout, while each scene test symlinks to the 32-trajectory
  seen_seen eval file. This is likely a major contributor if greedy rollout
  also remains near the ZSP floor.


### 2026-05-28 final greedy rollout result

Output:

```text
/root/autodl-tmp/wormi-checkpoints/wormi-vh-paperlike-v2-fixed-20260527/wormi-vh-n6/vh-rollout-paperlike-v2-fixed-full-20260527-greedy/vh-rollout-summary.tsv
```

Result:

```text
name	dataset_type	episodes	SR	PS	invalid_actions	executed_actions	max_steps	world_models
col_1_seen_seen	virtualhome_rollout	32	0.000000	30.000000	2.781250	27.218750	30	3,2,5
col_2_seen_unseen	virtualhome_rollout	212	0.000000	30.000000	1.693396	28.306604	30	4,3,1
col_3_unseen_unseen	virtualhome_rollout	300	0.000000	30.000000	2.000000	28.000000	30	2,5,3
```

Interpretation:

- Greedy removed most stochastic invalid actions, but SR stayed at 0.
- Episodes usually run to the 30-step cap with mostly executable actions.
- Detail traces show repeated valid `walk room` actions, e.g. `Place drawing in toilet`
  repeatedly emits `walk bathroom`, and `Put drawing on bed` repeatedly emits
  `walk bedroom`.
- Therefore the current failure is not an action parser failure, not empty output,
  not TRL loss-mask collapse, and not mainly invalid-action execution. The policy
  has learned target-room/object priors but not state-conditioned multi-step progress
  from source object to target receptacle.
- A separate compact-observation TMoW-style pipeline is now running in tmux
  (`wormi_tmow_compact_full`) to test whether full-scene observations/full next-state
  dynamics are the preprocessing bottleneck.