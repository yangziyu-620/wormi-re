# VirtualHome WorMI Paper-Aligned Rewrite — Progress Note

Date: 2026-05-29

This rewrite is a clean restart on the data pipeline, motivated by the
finding that the old `build_virtualhome_dataset_balanced.py` had three layered
coverage-greedy biases that collapsed seen-task training to a few source
classes (drawing / mat) and pushed 80% of trajectories to start with
`walk kitchen`.

## Reference points

- WorMI paper: arXiv `2509.03956` / OpenReview `tpbtodnI1p`, ICML 2025.
  Verbatim setup extract: `reports/virtualhome/data-processing/wormi-paper-spec-2026-05-29.md`.
- TMoW (ICLR 2026, same author group): used purely as a data-processing
  reference for the 78-task list intent and for the stratified-by-index
  seen-task selection. We did NOT adopt TMoW's BM25 KG retrieve or its
  partial-graph trajectory schema. WorMI uses full graph triples (paper
  Figure A.2).
- Builder source code in TMoW that was inspected:
  - `tmow/utils/virtualhome/const.py` (TASKS_SET, SEEN_TASKS, SEEN_DOMAIN)
  - `tmow/dataset/virtualhome.py` (VirtualHomeDatasetBuilder + Generator)
  - `tmow/scripts/eval_virtualhome.py` (uses Unity rollout)

## What changed

1. **Old artifacts cleaned**:
   - removed `world-vh-balanced-aux-compact17-sourceunique-20260528`,
     `world-vh-paperlike-tmow-compact-aa-fill17-20260528`,
     `world-vh-paperlike-tmow-compact-fill17-20260528`
     (≈42 GB recovered)
   - removed all stale `virtualhome-*` datasets except the canonical raw
     source `virtualhome-src` and `programs_processed_precond_nograb_morepreconds`
   - removed `wormi-vh-balanced-aux-compact17-sourceunique-20260528` stage-2
2. **All data-processing code backed up before edits**:
   `processing-scripts-backup-pre-tmow-rewrite-20260529-001357/`
3. **New files**:
   - `tools/tmow_const.py`: vendored TMoW const for cross-reference (kept,
     even though the builder ended up not using its task list verbatim —
     class names in TMoW const collide with VH raw graph naming).
   - `tools/build_virtualhome_dataset_wormi.py`: clean builder.
   - `sh/wormi-build-vh-wormi.sh`: launcher for the builder.
   - `sh/wormi-launch-paperaligned-stage1.sh`, `sh/wormi-launch-paperaligned-stage2.sh`:
     launchers for the two training stages with the new data root.

## Builder design (`build_virtualhome_dataset_wormi.py`)

- Task pool source: `build_virtualhome_dataset.build_candidate_instructions`.
  78 final tasks: 9 turnon + 7 open + 30 puton + 32 placein, per paper Table A.2.
- Per-family source-class diversity cap (`source_caps = quota // 4`) prevents
  any one source from dominating its family.
- Stratified-by-source-class greedy 16 seen-task split (2 turnon + 2 open +
  6 puton + 6 placein). The greedy step always picks the candidate whose
  source has been picked the fewest times.
- Scene cache: 20 distinct scenes; configurable seen/unseen split (default
  6 seen / 14 unseen per paper §4). 8 variants per domain by default.
- Trajectory: EvolvingGraph expert program through `_execute_paperlike_candidate`.
- Observation: `format_observation(graph)` — full class-level graph triples.
- Auxiliary tasks: BC + dynamics + affordance expanded by
  `wormi/datasets/virtualhome.py` at load time (paper §3.2 says three
  auxiliary objectives).
- Hard gate: `train_first_action_top1_share <= 0.35` or the build aborts.

## Smoke build result (variants_per_domain=6, candidate_multiplier=8)

```
episodes: 970, rows: 5080
pool sizes: seen×seen=457, seen×unseen=1062, unseen×seen=1819, unseen×unseen=4049
split counts: train=332, eval_a=95, eval_b=224, eval_c=319

train first-action top1 share: 0.3012   (gate threshold 0.35 PASSED)

train first-action top10:
  walk kitchen    : 100  (30.1%)
  walk bedroom    :  77  (23.2%)
  walk bathroom   :  42  (12.7%)
  walk livingroom :  29
  walk keyboard   :  22
  walk drawing    :  16
  walk chair      :  13
  walk mat        :   9
  walk mouse      :   9
  walk bathroom_cabinet : 6
```

Compare against the old `balanced-aux-compact17-sourceunique` build:
80% of trajectories started with `walk kitchen` (collapse). The new build
spreads first actions across at least 6 distinct rooms / objects.

Selected 16 seen tasks (sources): chair×3, drawing×3, keyboard×3, mat×2,
coffe_maker, computer, bathroom_cabinet, curtain, mouse. Source diversity
is up from 2 (drawing/mat) to 9.

## Train data per scene (world model size)

| scene | train rows | ratio |
|------|-----------:|------:|
| scene_0 | 310 | |
| scene_1 | 314 | |
| scene_2 | 232 | |
| scene_3 | 265 | |
| scene_4 | 311 | |
| scene_5 | 321 | |

With aux tasks expanded (BC + dynamics + affordance), each scene contributes
≈ 3× the row count to stage-1 supervision.

## Stage 1 training (started)

Run id: `wormi-paperaligned-20260529`

```
Data root:   /root/autodl-tmp/wormi-data/virtualhome-wormi-paperaligned-20260529
Ckpt root:   /root/autodl-tmp/wormi-checkpoints/world-vh-wormi-paperaligned-20260529
Log dir:     /root/autodl-tmp/wormi-logs/vh-world-wormi-paperaligned-20260529
Launch log:  /root/autodl-tmp/wormi-logs/vh-world-launcher-20260529.log
```

Paper Table A.6: Llama-3.2-1B, batch 4 (here batch 2 due to RTX 4090 48 GB),
2000 gradient steps, lr 3e-5, cosine. 6 world models, one per seen scene.
At ≈1.8 it/s observed → roughly 17 min per scene → 1.7 h total.

## Stage 2 training (planned)

Will launch automatically once stage 1 finishes, with:

- threaded path (paper-faithful Reptile is unstable in this codebase per
  `vh-stage2-meta-learning-clean-vs-seqmeta-2026-05-27.md`)
- `WORMI_THREADED_META_USE_BETA=1` so the aggregation rule is
  `θ ← θ + β (mean(θⱼ) − θ)` with β = 0.1 (paper Table A.6)
- batch 1 + grad-accum 4 → effective bs 4 (paper bs)
- inner steps 30, meta steps 8 (paper λ_I / λ_M)
- N=6, K=3 retrieval (paper)

## Rollout evaluator review

`wormi/scripts/eval_vh_rollout.py` was written locally. Audited:

- Goal definition matches paper Table A.2 verbatim:
  - turnon: `(target, is, on)`
  - open: `(target, is, open)`
  - puton: `(source, on, target)` relation in final graph
  - placein: `(source, inside, target)` relation in final graph
- Goal check is class-level (any instance satisfying the goal counts),
  consistent with paper class-level goal triples.
- Observation rendering during rollout uses `format_observation` (full
  graph), matching builder output exactly. No train/test render mismatch.
- Action parser handles `walk toilet` / `walk to` boundary bug fixed in
  earlier work.
- Success Rate / Pending Steps definition matches paper §4 metric.

Verdict: rollout evaluator is paper-aligned. No correction required.

## ALFWorld

Existing data at `/root/autodl-tmp/wormi-data/alfworld/` already conforms to
paper §4 / Table A.4 (6 task types, 4 splits, 3 + 1 seen/unseen scenes).
Train + 4 test files = 4634 episodes vs paper 3554 (paper count likely
excludes our 1081-episode `test_unseen_task_seen_scene.jsonl` which is not
in Table 1). Schema and chat template in `wormi/datasets/alfworld.py` are
already valid. **No rebuild needed.**
