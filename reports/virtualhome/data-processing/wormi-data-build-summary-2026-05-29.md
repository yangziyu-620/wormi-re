# WorMI VirtualHome Dataset Build: End-to-End Pipeline + Key Improvements

Date: 2026-05-29

> One-line framing: this is a **clean rewrite** (`tools/build_virtualhome_dataset_wormi.py`)
> whose purpose is to fix the "data collapse" problem of the old pipeline and make the
> reconstruction strictly aligned with the WorMI paper setup.
> Pre-rewrite scripts are backed up under
> `processing-scripts-backup-pre-tmow-rewrite-20260529-001357/`.

Reference docs:
- Rewrite progress note: `reports/virtualhome/data-processing/wormi-paperaligned-rewrite-2026-05-29.md`
- Paper setup extract: `reports/virtualhome/data-processing/wormi-paper-spec-2026-05-29.md`
- Old method note: `reports/virtualhome/data-processing/vh-current-data-processing-2026-05-28.md`

---

## 1. The 6 build steps (from scratch)

The builder's main flow is `Builder.materialize()`, which decomposes into 6 steps.

### Step 1 — Load scenes (`load_domains`)

- Read 7 base apartments from the raw VirtualHome init-graph library.
- For each apartment, sample several official init-graph variants to form
  **20 scene domains** (default layout `[3,3,3,3,3,3,2]`), each domain holding
  `variants_per_domain` variants (default 8–12).
- A **feasibility probe** (`_graph_probe_successes`) gates variant selection: a graph
  must be able to successfully execute at least a few tasks to be kept, filtering out
  broken graphs.
- After a seed-fixed shuffle, split into **6 seen scenes / 14 unseen scenes** (paper §4).

### Step 2 — Select 78 tasks (`select_tasks` Steps 1–2)

- From `properties_data.json` + the scene object sets, `build_candidate_instructions`
  produces a candidate pool ranked by joint-scene coverage.
- Take exactly 78 tasks by family quota:
  **9 turnon + 7 open + 30 puton + 32 placein** (paper Table A.2).
- **Key constraint**: each family enforces a `source_caps = quota // 4` per-source-class
  cap — a single source object (e.g. book / mug) can occupy at most 1/4 of a family,
  preventing one object from dominating.

### Step 3 — Split out the 16 seen tasks (`select_tasks` Step 3)

- Quota **2 turnon + 2 open + 6 puton + 6 placein = 16 seen**; the remaining 62 are unseen.
- Use a **stratified-by-source-class greedy** rule: at each step prefer the task whose
  source class has not yet been picked and whose target class is least represented.
  **This is the core mechanism that prevents first-action collapse.**

### Step 4 — Generate expert trajectories (`execute_slot`)

- For every (task × scene variant) combination, run a deterministic expert program via
  **EvolvingGraph** (`_execute_paperlike_candidate`).
- Multi-layer filtering: prefilter-skip combinations with missing objects, skip
  execution failures, and skip invalid transitions where
  `observation == next_observation`.
- Each step of each trajectory is written as one row, carrying full `_meta`
  (task_id / scene / trajectory_id / step_index).

### Step 5 — Bucket into 4 quadrants + sample (`materialize`)

- All valid trajectories fall into 4 pools by `{seen/unseen task} × {seen/unseen scene}`.
- Then **task-balanced sampling**:
  - `train` = seen×seen, taking `train_episodes / 16` per seen task (384/16 = 24);
  - `eval A` = seen×seen, **held out** from train at the episode level
    (same task, different trajectory);
  - `eval B` = seen×unseen;
  - `eval C` = unseen×unseen, sampled **only from this legal pool**, never by subtraction.
- `train` is materialized into **6 `scene_0..5/` directories**, matching the paper's
  N=6 world models.

### Step 6 — Quality gate + write (`run` / `write`)

- Compute `quality_report` in memory first, then pass the
  **hard gate `train_first_action_top1_share <= 0.35`**; on failure it `raise`s and
  **writes nothing**.
- On pass, write `scene_*/train.jsonl`, the 3 `test_*.jsonl` files, the `eval_col_*`
  symlinks, `virtualhome_manifest.json`, and `quality_report.json`.
- Auxiliary tasks (BC + dynamics + affordance, paper §3.2) are **expanded at load time**
  by the loader `wormi/datasets/virtualhome.py`; the builder writes only one raw
  transition row.

---

## 2. Key improvements over the previous method

| Dimension | Before (`balanced` builder) | Now (`wormi` builder) | Why it matters |
|---|---|---|---|
| **Task selection** | coverage-ranked **top-K cut** + `semantic-gate=source_unique` filter | family quota + **per-source diversity cap** | high-coverage objects no longer monopolize |
| **Seen-task split** | random / coverage-greedy | **stratified-by-source greedy** (pick least-used source) | source diversity **2 → 9** |
| **Observation** | `tmow_compact`, 17-edge compact graph + compact-K subset | **full class-level graph triples** via `format_observation` (paper Fig A.2) | train / eval rollout render identically, no format mismatch |
| **Retrieval / augmentation** | compact-K selection + augmentation | **all removed** — no BM25, no subset, no augmentation | closer to paper, fully auditable |
| **Quality safeguard** | none | **first-action top1 hard gate <= 0.35** | auto-blocks data collapse; bad data cannot be produced |
| **First-action distribution** | `walk kitchen` ~80% (collapsed) | top1 = 0.30, spread across 6+ rooms | world models learn real adaptation, not "always go to the kitchen" |
| **eval C source** | (correct in both) sampled only from the unseen×unseen pool | same | avoids leakage into the test set |

---

## 3. Talking points (three sentences)

1. **Problem**: the old builder's three layered coverage-greedy biases collapsed
   seen-task training onto 2 source objects, with 80% of trajectories starting with
   `walk kitchen`, so world models failed to learn genuine scene adaptation.
2. **Fix**: rewritten to be paper-aligned — stratified-by-source task selection, full
   graph-triple observations, and removal of all non-paper retrieval / compaction /
   augmentation.
3. **Safeguard**: added a first-action distribution hard gate (top1 <= 0.35) that turns
   "is the data collapsed?" into an automatic build-time check; the smoke build passes at
   0.30 and source diversity rises from 2 to 9.

---

## Appendix: smoke build result (variants_per_domain=6, candidate_multiplier=8)

```text
episodes: 970, rows: 5080
pool sizes: seen×seen=457, seen×unseen=1062, unseen×seen=1819, unseen×unseen=4049
split counts: train=332, eval_a=95, eval_b=224, eval_c=319
train first-action top1 share: 0.3012   (gate threshold 0.35 PASSED)
```

> Note: the first-action / source-diversity numbers in the table come from the smoke
> build. For the production dataset, the authoritative figures are in the corresponding
> `quality_report.json`.
