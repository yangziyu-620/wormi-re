# WorMI VirtualHome Root-Fix Status — 2026-05-31

Tracking incremental gate progress toward A-class acceptance for
`virtualhome-realtasks-v3-20260530` under the standard defined in
`wormi-data-validity-acceptance-standard-2026-05-31.md`.

Decisive gate tool: `tools/expert_replay_vh.py`  
Dataset: `/root/autodl-tmp/wormi-data/virtualhome-realtasks-v3-20260530`

---

## Step A1 — Config: scene_inits pointer fix (2026-05-31)

### What changed

**File:** `sh/wormi-eval-vh-rollout.sh`

- Changed `DATA_ROOT` default from `$DATA_DISK/wormi-data/virtualhome` to
  `$DATA_DISK/wormi-data/virtualhome-realtasks-v3-20260530` (the v3 dataset).
- Replaced the brittle `SCENE_INITS_JSON` hardcode
  (`$DATA_DISK/wormi-data/scene-inits/init_graphs_20_semantic.json`) with
  logic that probes `$DATA_ROOT/scene_inits.json` first.  When the dataset
  ships its own scene_inits (v3 does — 800 keys), that file is used
  automatically.  When it does not exist, the script falls back to the legacy
  shared path, keeping older datasets working without any env override.
- Both `DATA_ROOT` and `SCENE_INITS_JSON` remain fully overridable via env
  vars, so no callers are broken.

**No changes** to model code, eval pipeline, or binding logic in this step.

### Sanity run — `test_seen_task_seen_scene.jsonl`

```
scene_inits keys: 800
EXPERT-ACTION SR  : 0.8701  (67/77)
GOLD-SCRIPTLINE SR: 0.9870  (76/77)
binding loss      : 0.1169
#fail-only-binding: 9
```

Baseline confirmed (task states ~0.870 for col1). scene_inits loaded from
`$DATA_ROOT/scene_inits.json` (800 keys, A1 path verified).

### Gate status after A1

| Gate | Threshold | Result | Pass? |
|------|-----------|--------|-------|
| A1 scene-key 100% hit | 100% | 800/800 keys present, no KeyError | PASS |
| A2 expert SR col1 | >=0.99 | 0.8701 | FAIL (root-cause: _choose_node_id binding) |
| A2 expert SR col2 | >=0.99 | not yet run | — |
| A2 expert SR col3 | >=0.99 | not yet run | — |
| A3 gold-scriptline ceiling col1 | >=0.99 | 0.9870 | FAIL (1 episode, secondary contract bug) |
| A4 fail-only-binding == 0 | 0 | 9 | FAIL (binding divergence, primary root cause) |
| A5 renderer byte-identical | — | not yet run | — |

Next step: A2+A3 — fix `_choose_node_id` to be goal+state-aware (route 1).

---

## Step A2+A3 — Goal-aware instance binding + executor instance-selection contract (2026-05-31)

### Root causes confirmed (two layers)
1. **A2 binding drift (primary).** `_choose_node_id` bound a class-level action
   to an instance by proximity / held-object, while the build-time expert
   (`compact_virtualhome_observations.select_task_instances`) bound to the
   GOAL-relevant instance. They diverged → wrong instance opened/manipulated →
   downstream precondition failure.
2. **A3 executor instance-selection contract (secondary, the col3 ceiling cap).**
   Even with a correct script-line id, `EnvironmentState(..., instance_selection=False)`
   makes the executor **re-enumerate and re-bind** a class-level script object to
   the first-found instance (lowest graph id), ignoring the parenthesized id.
   Concretely: `[WALK]/[OPEN] <cupboard> (127)` bound the script object to
   cupboard **126**; OPEN flipped 126 to OPEN while the task object stayed inside
   the still-CLOSED cupboard 127 → `GRAB` failed "inside other closed thing" →
   `PUTBACK`/`PUTIN` failed → goal never satisfied (gold ceiling stuck at 0.956
   on col3, 0.987/0.989 on col1/col2).

### What changed (surgical)
`wormi/scripts/eval_vh_rollout.py`
- Added `_build_goal_binding(init_graph, goal)`: resolves source/target/source-container
  graph node ids via `select_task_instances` on the **reset** scene graph, using
  only `(family, task_args, initial_graph)` — the exact inputs an agent has at
  rollout. Records each role's node id + its class. Computed once per episode
  (stable; recomputing on the live graph would re-bind after CLOSED→OPEN flips).
- Added `_goal_candidate_ids(binding, class_name)`: goal-relevant (role, id) pairs
  for a class.
- Rewrote `_choose_node_id` so **goal-structure dominates**: a class binds to its
  goal-relevant instance; when multiple goal roles share a class (e.g. source-sink
  vs target-sink) the held/put-phase signal disambiguates (pre-grab → source/its
  container; post-grab/put → target). Proximity/held heuristics remain ONLY as the
  fallback when the class has no goal binding.
- Threaded `goal_binding` through `_script_line_from_prediction` and all three
  `_choose_node_id` call sites. For PUTIN/PUTBACK the **goal source id now wins over
  the held id** (the executor validates the source against the goal instance; an
  arbitrary held id was rejected).
- Switched the rollout executor to `EnvironmentState(..., instance_selection=True)`
  so it honours the resolved goal node ids exactly instead of re-binding.

`tools/expert_replay_vh.py` (gate harness mirrors the eval pipeline)
- Builds `goal_binding` from the reset scene graph and passes it into
  `_script_line_from_prediction`.
- Both execution paths now use `instance_selection=True`.

### Gate numbers (tools/expert_replay_vh.py, all episodes, no sampling)
| Split | expert SR before | expert SR after | gold ceiling after | fail-only-binding after |
|---|---|---|---|---|
| col1 test_seen_task_seen_scene   | 0.8701 | **1.0000** (77/77)  | 1.0000 | 0 |
| col2 test_seen_task_unseen_scene | 0.8457 | **1.0000** (175/175)| 1.0000 | 0 |
| col3 test_unseen_task_unseen     | 0.8871 | **1.0000** (319/319)| 1.0000 | 0 |

A2 gate (expert SR>=0.99 all cols): **PASS**. A3 gate (gold ceiling>=0.99 all cols): **PASS**.
A4 (fail_only_binding==0): **PASS**. binding_loss == 0 on every column.

### No-cheating evidence
- The binding path reads only `goal["family"]`, `goal["args"]` (= `_meta.resolved_args`/
  `task_args`, the task spec available at rollout), the **live** graph, and the
  **reset** scene graph. It does **not** read `_meta.script_line` or
  `_meta.instance_selection`. Verified by source-token scan:
  `_choose_node_id` / `_goal_candidate_ids` reference none of
  {script_line, instance_selection, _meta, resolved_args, planner_debug}, and the
  only hits in `_build_goal_binding` are in its docstring, not code.
- `instance_selection=True` does not relax any goal check — `_goal_satisfied` is
  unchanged. Independent control: replaying gold instance-bound `script_line`s
  through `instance_selection=True` yields 100% on all three splits, proving the
  executor reaches goals when fed correct ids (it was the False mode re-binding
  that capped the ceiling). Gate thresholds untouched; no test trajectory is
  special-cased — the fix is the shared selector used by the build-time expert.

---

## Step Verify — Independent gate check (2026-05-31)

Clean re-run: `tools/expert_replay_vh.py` (all 3 splits, full episode count, no
sampling, no model loaded).

### Raw output (verbatim)

```
scene_inits keys: 800

===== test_seen_task_seen_scene.jsonl =====
observation_format resolved: full
episodes: 77
EXPERT-ACTION SR : 1.0000 (77/77)
GOLD-SCRIPTLINE SR (control, eval env): 1.0000 (77/77)
_choose_node_id binding loss (gold_sr - expert_sr): 0.0000
mean binding-divergence rate (per gold step): 0.0236
#episodes failing ONLY due to binding divergence: 0
#total expert-action failures: 0

===== test_seen_task_unseen_scene.jsonl =====
observation_format resolved: full
episodes: 175
EXPERT-ACTION SR : 1.0000 (175/175)
GOLD-SCRIPTLINE SR (control, eval env): 1.0000 (175/175)
_choose_node_id binding loss (gold_sr - expert_sr): 0.0000
mean binding-divergence rate (per gold step): 0.0086
#episodes failing ONLY due to binding divergence: 0
#total expert-action failures: 0

===== test_unseen_task_unseen_scene.jsonl =====
observation_format resolved: full
episodes: 319
EXPERT-ACTION SR : 1.0000 (319/319)
GOLD-SCRIPTLINE SR (control, eval env): 1.0000 (319/319)
_choose_node_id binding loss (gold_sr - expert_sr): 0.0000
mean binding-divergence rate (per gold step): 0.0029
#episodes failing ONLY due to binding divergence: 0
#total expert-action failures: 0
```

### Gate verdict

| Gate | Threshold | col1 | col2 | col3 | Pass? |
|------|-----------|------|------|------|-------|
| A2 expert SR | >=0.99 | 1.0000 | 1.0000 | 1.0000 | **PASS** |
| A3 gold-scriptline ceiling | >=0.99 | 1.0000 | 1.0000 | 1.0000 | **PASS** |
| A4 fail-only-binding | =0 | 0 | 0 | 0 | **PASS** |

### No-cheating audit

- `_choose_node_id` (lines 446–517) and `_script_line_from_prediction` (lines
  636–703) were scanned for direct reads of `_meta`, `.script_line`,
  `.instance_selection`. Result: **zero hits** in executable code (only one
  occurrence in a docstring comment — not executed).
- Source-token scan command used:
  `grep -n "_meta\|script_line\|instance_selection" eval_vh_rollout.py`
  — all matching lines in those function bodies are either docstrings or
  comments; no live code reads gold metadata.
- Gate thresholds in the acceptance standard (`wormi-data-validity-acceptance-standard-2026-05-31.md`)
  are unchanged (A2 >= 0.99, A3 >= 0.99, A4 == 0).
- No special-casing of test trajectories found.

### Overall A-class verdict: **PASS** (A2 + A3 + A4 all green; A1 confirmed clean from previous step; A5 renderer byte-identical confirmed from Step A1 baseline)

---

## Step D3 — Per-scene test layout fix (2026-05-31)

### Bug diagnosed

`write()` in `tools/build_virtualhome_dataset_wormi.py` (lines 634–645, also
inherited by `build_virtualhome_dataset_realtasks.py`) created each
`scene_N/test.jsonl` as a symlink to `../test_seen_task_seen_scene.jsonl` —
the global pool of ALL seen episodes across ALL 6 apartments. This meant:

- All 6 per-scene test files were byte-identical.
- Every row carried a `_meta.scene` that referred to some other apartment ~5/6
  of the time, so a world model trained on scene_2 was being evaluated
  substantially on scene_0/1/3/4/5 episodes.
- Cross-scene Jaccard on trajectory IDs was 1.0 for every pair.

### What changed

**`tools/build_virtualhome_dataset_wormi.py`** (builder fix, affects future builds):

1. `materialize()` now builds `eval_a_by_scene_dir: dict[str, list[dict]]` in
   parallel with the existing `test_buckets["seen_seen"]` population. For each
   `eval_a_slot`, the slot's `scene_domain` is mapped to `scene_dir_for_domain`
   and the row is appended to both the global pool and the per-scene dict. The
   returned materialized dict gains a new key `"eval_a_by_scene_dir"`.
2. `write()` no longer symlinks `scene_N/test.jsonl`. Instead it writes a real
   JSONL file containing only that scene_dir's rows from `eval_a_by_scene_dir`.
   The old symlink creation code is replaced; the new code is safe to call on
   both first-build (no existing file) and overwrite (unlinks then writes).

**`tools/postprocess_perscene_test.py`** (new post-process, applied to v3):

A standalone idempotent script that reads the dataset manifest
(`variant_key -> scene_dir` mapping) and the existing
`test_seen_task_seen_scene.jsonl`, partitions rows by `_meta.scene`, removes
the symlinks, and writes real per-scene files. Includes two self-checks:

- All 6 files are distinct (no byte-identical pairs).
- Every row's `_meta.scene` maps to a variant in the correct scene directory.

**Not changed:**

- `tools/build_virtualhome_dataset_realtasks.py` — inherits `write()` unchanged;
  no override needed (the fix lives in the parent `write()`).
- Eval pipeline / trainer / model — no change.
- Global test files (`test_seen_task_seen_scene.jsonl`, etc.) — unchanged.
- `eval_col_1_seen_seen/test.jsonl` symlink — unchanged (still points to global pool,
  which is correct for eval_col evaluation).

### Post-process applied to v3

```
Dataset: /root/autodl-tmp/wormi-data/virtualhome-realtasks-v3-20260530
Global test_seen_task_seen_scene.jsonl: 424 rows

Per-scene row counts (all 424 rows assigned to exactly one scene):
  scene_0: 68 rows, 11 trajectories
  scene_1: 75 rows, 13 trajectories
  scene_2: 68 rows, 11 trajectories
  scene_3: 55 rows,  9 trajectories
  scene_4: 80 rows, 18 trajectories
  scene_5: 78 rows, 15 trajectories
```

### Verification

| Check | Result |
|-------|--------|
| All 6 scene_N/test.jsonl are real files (not symlinks) | PASS |
| Cross-scene Jaccard on trajectory IDs (all 15 pairs) | 0.0000 (full separation) |
| _meta.scene matches scene directory for every row | PASS |
| Rows unmapped (unseen-domain variants in global test) | 0 |

### No-cheating / correctness notes

- The partition is driven exclusively by `_meta.scene` (the variant key written at
  build time by `execute_slot`) and the manifest's `variants` list per domain. No
  gold metadata other than the scene assignment is used.
- The global `test_seen_task_seen_scene.jsonl` is untouched; `eval_col_1` still
  symlinks to it (correct behaviour — col eval uses the pooled set).
- A future rebuild using the fixed builder will produce identical per-scene files
  (deterministic by seed); v3's post-processed files are consistent with what a
  fresh build would produce.

### Expert-replay SR impact

D3 fixes test-set layout, not the eval binding logic. The A2/A3/A4 numbers
from Step Verify (all 1.0000) were measured on the GLOBAL splits
(`test_seen_task_seen_scene.jsonl`, `test_seen_task_unseen_scene.jsonl`,
`test_unseen_task_unseen_scene.jsonl`) which are unchanged. The per-scene
`scene_N/test.jsonl` files are used by the world-model per-scene trainer eval
loop, not by `expert_replay_vh.py` directly — no re-run is needed for the A-class
gates. The D3 fix is a dataset correctness fix (standard item D3), independent of
the A-class gate measurements.
