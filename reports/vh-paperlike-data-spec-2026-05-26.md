# WorMI VirtualHome Paper-Like Data Specification

Date: 2026-05-26

## Objective

Build a fixed VirtualHome benchmark that is comparable across methods and closer
to the data semantics used by WorMI:

- domains are house configurations / scene graphs
- world models are scene-domain experts, not action-family experts
- expert trajectories depend on the concrete house state
- rollout evaluation is the primary metric

This supersedes the current `easy_debug` split, whose expert trajectories are
short atomic templates and are too easy for seen-task evaluation.

## Version Name

```text
WorMI-VH-HouseConfig-v1.0
```

All downstream results must name this dataset version. If the split, generator,
or evaluator changes, the version must change.

## Fixed Axes

### Task Axis

Match the WorMI VirtualHome task scale:

```text
total tasks: 78
seen tasks: 16
unseen tasks: 62
families:
  turnon: 9
  open: 7
  puton: 30
  placein: 32
```

The exact task list must be written to `task_split.json` and
`virtualhome_manifest.json`.

### Scene / House Axis

The domain axis is the full house configuration.

```text
total house configurations: 20
seen house configurations: 6
unseen house configurations: 14
world models: 6, one per seen house configuration
```

A house configuration contains rooms, layout, object placement, receptacles,
object states, and graph connectivity. Room type alone is not a domain.

### Split Semantics

```text
stage-1 world model train:
  seen task x one seen house configuration

stage-2 train:
  seen task x seen house configurations, through K=3 retrieved/selected world models

eval col_1:
  held-out seen task x seen house configuration

eval col_2:
  seen task x unseen house configuration

eval col_3:
  unseen task x unseen house configuration
```

## Expert Trajectory Requirements

Each expert episode should be generated as:

```text
(house_configuration_id, init_state_variant, instruction) -> expert action sequence
```

The expert must depend on graph state, not only on task family.

Allowed public actions remain aligned with the current VirtualHome prompt:

```text
walk, grab, open, switchon, put, putin
```

Internal source programs may contain additional actions such as `find`; those
can only be used as structure hints and should not appear as supervised actions
unless the prompt/evaluator are explicitly expanded.

## Preferred Generation Sources

Generation should use the following priority:

1. **Original VirtualHome executable programs** from
   `programs_processed_precond_nograb_morepreconds.zip`.
   - Use action skeletons and object dependencies from real VH programs.
   - Map object classes to the target house configuration.
   - Replay in EvolvingGraph and keep only successful trajectories.

2. **Graph-aware planner fallback** if original programs do not provide enough
   examples for a required task/scene bucket.
   - The planner may add necessary room/source/target navigation and
     source-container opening steps.
   - The planner must replay successfully in EvolvingGraph.

The old shortest-template generator is allowed only for `easy_debug`, not for
the paper-like split.

## Difficulty Targets

The data should be more difficult than atomic templates but still focused on
the WorMI VirtualHome action set.

Target quality ranges:

```text
expert trajectory length:
  mean successful length: 7-12 high-level actions
  minimum preferred length: 4
  max allowed length: 18

rollout max steps:
  30

episode count:
  1023 total, unless source feasibility forces a documented deviation
```

These are not hard-coded by padding. Longer trajectories should come from
state-dependent requirements such as:

- source object starts inside a closed container
- target receptacle starts closed
- agent starts in a different room
- same task appears in different house configurations with different object
  placements
- source and target are in different rooms
- distractor objects / duplicate classes exist in observations

## Leakage Rules

The benchmark must emit a leakage report with at least:

```text
train/test trajectory_id overlap
train/test exact row overlap
train/test exact full action sequence overlap
same task + exact full action sequence overlap
same task + same scene overlap
```

Required:

- `trajectory_id` overlap must be zero.
- exact row overlap must be zero.
- exact full action-sequence overlap between train and eval should be minimized
  and reported.
- same-task exact action-sequence reuse should be avoided for col_1 whenever
  source feasibility allows it.

## Quality Report

Every built dataset must include:

```text
virtualhome_manifest.json
validation_report.json
quality_report.json
MANIFEST.sha256
```

The quality report must include:

- total rows and episodes
- split-wise rows and episodes
- family distribution by split
- task distribution by split
- scene distribution by split
- trajectory length distribution by split
- mean / median / p10 / p90 length
- replay success count
- goal satisfaction count
- invalid / filtered source-program counts
- leakage report

## Evaluation Policy

The primary paper-comparison table must use environment rollout:

```text
SR = final task goal satisfied in EvolvingGraph
PS = average rollout steps
invalid_actions = separately reported
max_steps = 30
```

The offline JSONL exact-match evaluator may be kept as a diagnostic only.

## Current Status

Current `easy_debug` data:

- has the right high-level task and scene counts
- uses one world model per seen scene
- replays successfully
- but uses short fixed action templates
- produces seen-seen results that are not a faithful paper-like comparison

Next build target:

```text
/root/autodl-tmp/wormi-data/virtualhome-paperlike-v1
```

After validation, this directory can replace:

```text
/root/autodl-tmp/wormi-data/virtualhome
```

for the next full stage-1 / stage-2 training run.
