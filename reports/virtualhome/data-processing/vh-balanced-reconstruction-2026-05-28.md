# VirtualHome Balanced Reconstruction - 2026-05-28

## Status

Built a new VirtualHome dataset at:

```text
/root/autodl-tmp/wormi-data/virtualhome-balanced-reconstruction-20260528
```

This is a **paper-compatible balanced reconstruction**, not an author-exact WorMI split. The WorMI paper reports aggregate counts such as 78 tasks, 16/62 task split, 20 scenes, 6/14 scene split, and N=6 world models, but does not publish exact task IDs, scene IDs, or episode IDs.

## Lessons From Failed Older Data

Older paperlike data had several issues that can explain bad training/eval behavior:

- world-model training was too sparse: each scene WM had only about 10-11 episodes;
- early paperlike data had many no-op walk transitions where observation did not change;
- some variants had replay or observation mismatch problems;
- compact data fixed several noise issues, but it was no longer the paper graph-triple observation format;
- one compact attempt had train/test exact row overlap;
- eval/split accounting was too easy to confuse because seen_seen train and eval shared the same split label.

The new builder addresses these directly:

- each of the 6 world models receives 64 train episodes;
- each WM covers all 16 seen tasks;
- train/eval split is by trajectory ID, never transition row;
- no trajectory with unchanged observation transitions is admitted;
- eval C is sampled only from `unseen_task intersect unseen_scene`, never from residual episodes;
- all split assumptions are written to `virtualhome_manifest.json`.

## Protocol

Output layout remains compatible with the existing WorMI curricula:

```text
scene_0/train.jsonl ... scene_5/train.jsonl
test_seen_task_seen_scene.jsonl
test_seen_task_unseen_scene.jsonl
test_unseen_task_unseen_scene.jsonl
eval_col_1_seen_seen/test.jsonl
eval_col_2_seen_unseen/test.jsonl
eval_col_3_unseen_unseen/test.jsonl
```

Split rule:

```text
train  = seen_task   intersect seen_scene
eval A = seen_task   intersect seen_scene, held out by episode
eval B = seen_task   intersect unseen_scene
eval C = unseen_task intersect unseen_scene only
```

Scene assumption:

```text
N=6 world models is interpreted as six seen scene domains.
Each scene domain groups multiple official VirtualHome init-graph variants from one base apartment.
```

This avoids the previous full Cartesian-product assumption while still giving each WM enough data.

## Final Counts

```text
total episodes: 1023
total rows: 5286
train episodes: 384
eval A seen_seen episodes: 96
eval B seen_unseen episodes: 224
eval C unseen_unseen episodes: 319
unseen_task x seen_scene auxiliary file: 0 episodes
train/test trajectory overlap: 0
```

World-model train distribution:

```text
scene_0: 64 episodes, 311 rows, 16 tasks
scene_1: 64 episodes, 308 rows, 16 tasks
scene_2: 64 episodes, 327 rows, 16 tasks
scene_3: 64 episodes, 309 rows, 16 tasks
scene_4: 64 episodes, 349 rows, 16 tasks
scene_5: 64 episodes, 328 rows, 16 tasks
```

Eval C distribution:

```text
episodes: 319
rows: 1697
families: puton=123, placein=129, turnon=36, open=31
unseen scene domains: 14, each has 22-23 episodes
unseen tasks: 62, each has 5-6 episodes
```

## Validation

Validator output:

```text
reports/virtualhome/validation/vh-balanced-reconstruction-validation-uv-2026-05-28.json
```

Replay checks passed:

```text
replay failures: 0
observation mismatches: 0
next observation mismatches: 0
goal failures: 0
train/test overlap: 0
```

Remaining warnings are protocol-related, not replay errors:

- validator sees 157 effective init-graph variants, while the new protocol has 20 scene domains backed by multiple variants;
- three cached init variants had no selected rows after balanced sampling.

## Code

New builder:

```text
tools/build_virtualhome_dataset_balanced.py
```

Validator adjustment:

```text
tools/validate_virtualhome_dataset.py
```

The validator now skips loader/chat smoke tests for empty optional JSONL files, so the optional `test_unseen_task_seen_scene.jsonl` can remain empty without crashing `JsonlDataset.load()`.
