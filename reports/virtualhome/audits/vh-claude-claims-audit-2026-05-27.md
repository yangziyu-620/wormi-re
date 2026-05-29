# Audit of External Diagnosis Notes - 2026-05-27

This note checks the external diagnosis pasted by the user against the local
code, logs, and data. It should be read together with
`reports/virtualhome/audits/vh-data-construction-audit-2026-05-27.md`.

## Background

Active run:

- pipeline: `/root/autodl-tmp/wormi-logs/vh-pipeline-paperlike-v1-threaded-lockfix2-20260527`
- stage2 log: `/root/autodl-tmp/wormi-logs/vh-wormi-paperlike-v1-threaded-lockfix2-20260527-stage2/train.log`
- active output: `/root/autodl-tmp/wormi-checkpoints/wormi-vh-paperlike-v1-threaded-lockfix2-20260527`

The active run used:

```text
seq meta: 0
inner steps: 30
meta steps: 8
batch size: 1
grad accum: 4
```

So this run used the threaded meta-learning path, not the sequential Reptile
path.

## Claim Checks

### Claim: `switch` vs `switchon` is not a bug under strict paper alignment.

Verdict: confirmed after checking the paper.

Confirmed:

- Paper Figure A.4 uses `switch [object]` in the VirtualHome system prompt.
- Paper Table A.1 uses `SwitchOn [Switchable object]` with example
  `switchon stove`.
- The local `BASE_PROMPT` says `switch [object]`, while the builder and
  validation use `switchon`, which matches this paper inconsistency.
- Current data contains 72 `switchon` actions.

Conclusion:

- Do not change the prompt from `switch [object]` to `switchon [object]` if the
  goal is strict paper alignment.
- `switch/switchon` should be removed from the data-bug list.
- The current data still has larger semantic issues: raw room aliases in action
  labels, redundant/no-op walks, and class-level ambiguity in multi-instance
  scenes.

### Claim: TRL collator only supervises the final assistant turn.

Verdict: confirmed.

Probe:

- Loaded `unsloth/Llama-3.2-1B-Instruct` tokenizer from local cache.
- Used `DataCollatorForCompletionOnlyLM` with
  `response_template="<|start_header_id|>assistant<|end_header_id|>"`.
- Checked `labels != -100`.

Results:

- Stage2/action-only load (`end_with_action=True`) supervises the action only,
  e.g. `open dresser<|eot_id|>`.
- Stage1/world-model load (`end_with_action=False`) produces alternating
  samples:
  - one supervises the action
  - one supervises the next observation
- No all-masked sample was observed in the probe.

Conclusion:

- There is no evidence that the current training is silently unsupervised.
- The world-model stage does spend roughly half its examples on dynamics
  prediction, which is consistent with the paper's world-model objective.

### Claim: Threaded stage2 shares `global_step`, so each trainer only gets about 5 inner steps instead of 30.

Verdict: false for the current code/log behavior.

Code facts:

- `WorMIMetaLearningTrainer` does call `super(..., synchronize_trainers=False)`.
- `WorMITrainerCallback.global_step` is shared through `main_trainer.step`.
- Curriculum boundaries are checked with:

```python
self.global_step - self.main_trainer.start_curriculum_step
>= self.curriculum.trainer_args.max_steps
```

But the lock ring does not switch trainers every single optimizer step. In the
current control flow it switches after the configured curriculum boundary,
which is 30 steps.

Log facts:

- The active threaded run produced 6 `train_runtime` summaries.
- Each subtrainer has `train_runtime * train_steps_per_second` approximately
  equal to 240 optimizer steps.
- 240 = 8 meta iterations * 30 inner steps.
- Different final epoch values are expected because the six merged training
  datasets have different sizes.

Conclusion:

- The pasted diagnosis's "30 steps are split across 6 trainers, about 5 each"
  is not supported by the active logs.
- The shared-step threaded trainer is still brittle and hard to reason about,
  but this specific failure mechanism is not the current explanation.

### Claim: Threaded aggregation defaults to beta=1 instead of paper beta=0.1.

Verdict: confirmed.

Code:

```python
use_reptile_beta = os.environ.get("WORMI_THREADED_META_USE_BETA", "0") == "1"
...
if use_reptile_beta:
    theta <- theta_old + meta_learning_rate * (mean_theta - theta_old)
else:
    theta <- mean_theta
```

Active run:

```text
WORMI_THREADED_META_USE_BETA=0
```

Conclusion:

- The active threaded run is not a strict paper-faithful Reptile update.
- For a paper-faithful threaded run, set `WORMI_THREADED_META_USE_BETA=1`.
- Prior local evidence shows the sequential beta path performed badly on the
  easier task-aware dataset, so beta correctness alone should be validated with
  a quick gate before committing to a full expensive run.


### Claim: World-model action-affordance auxiliary task is missing.

Verdict: confirmed.

Paper section 3.2 says each domain-specific world model is trained with three
auxiliary tasks: dynamics, action affordance, and behavior cloning. The local
VirtualHome and ALFWorld dataset converters only emit:

- behavior cloning prompts: `Instruction + Observation -> Action`;
- dynamics prompts: `Instruction + Observation + Action -> Next observation`.

There is no local prompt/data path for an explicit affordance target such as
`Observation -> feasible actions`. Searches for `affordance`, `feasible`,
`possible action`, and related terms found no implementation in the world-model
training data path.

Impact:

- This is a real paper-alignment gap in stage1 world-model training.
- Current rollout failures show many invalid/precondition-failed actions, which
  is consistent with weak affordance knowledge, though not proof that this is the
  only cause.

### Claim: Current low `col_3` number is rollout SR, not exact-match Table1.

Verdict: confirmed.

The value around 15% came from `eval_vh_rollout`, whose metric checks goal-state
success in the VirtualHome EvolvingGraph environment. It is therefore directionally
comparable to the paper SR metric, unlike the local offline exact-match Table1
helper.

At the time of this check, `col_3_unseen_unseen` was still running. The partial
rollout output had very high invalid/precondition-failed action counts, e.g.
completed col3 episodes averaged more than 20 invalid actions per episode.

## Current Working Interpretation

The current bad result is probably caused by a combination of:

1. Semantic noise in the new paper-like VH data.
2. Threaded stage2 using `theta <- mean(theta_i)` rather than the paper's
   `theta <- theta + beta * (mean(theta_i) - theta)`.
3. Long-horizon exact-match / rollout evaluation amplifying early action errors.

It is not supported that:

- stage2 got only ~5 inner steps per trainer;
- the collator masked away all supervision;
- the only data bug is `switch/switchon`.

## Recommended Next Validation

After the active rollout finishes:

1. Repair VH data semantics:
   - canonicalize room labels in actions to match observations and prompt;
   - remove redundant/no-op walks with a position-aware planner;
   - add validation checks for raw room aliases, duplicate actions, unchanged
     walk rate, and class-id ambiguity.
2. Add the missing world-model action-affordance auxiliary task or explicitly
   document why it is omitted.
3. Rebuild data and run a small rollout quick gate before full retraining.
4. For stage2, compare two controlled runs on the repaired data:
   - threaded default mean update, because it has worked empirically before;
   - threaded beta update with `WORMI_THREADED_META_USE_BETA=1`, because it is
     closer to the paper.

