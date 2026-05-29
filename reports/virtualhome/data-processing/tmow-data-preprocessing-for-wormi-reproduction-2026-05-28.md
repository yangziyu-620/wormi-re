# TMoW Data Preprocessing Lessons for WorMI Reproduction

Date: 2026-05-28

This note focuses only on data preprocessing. It does not propose changing the
WorMI model architecture to TMoW's LoRA/MoE structure.

## Executive Summary

The most useful TMoW idea for WorMI reproduction is not the router module. It is
the way TMoW constructs compact, instruction-conditioned observations.

Current WorMI VirtualHome data emits nearly the full class-level scene graph for
both `observation` and `next_observation`. In the current
`virtualhome-paperlike-v2-fixed-20260527` data, `scene_0/train.jsonl` averages
193.5 triples and about 5.3k characters per observation; the Table-1 col-1 test
file averages 263.4 triples and about 7.3k characters per observation.

TMoW instead maintains a refined `KnowledgeGraph` and retrieves a small
instruction-relevant subset of triples. In the local TMoW implementation,
VirtualHome generation calls:

```text
kg.retrieve(instruction, embedding_fns=embedding_fns, num_edges=17)
```

It then appends critical agent state such as character room, held object, and
adjacent room facts. The resulting observation is much closer to a task-focused
belief state than a full scene dump.

For WorMI reproduction, the priority should be:

1. rebuild VirtualHome observations using TMoW-style task-conditioned triple
   retrieval;
2. train world models to predict compact state updates rather than full next
   scene graphs;
3. keep WorMI's JSONL schema and Llama-3 chat envelope, instead of importing
   TMoW's HF-dataset builder directly;
4. add preprocessing diagnostics that report triple counts, prompt lengths,
   contradictory class-level facts, and whether target objects/goals remain
   visible after compaction.

## What TMoW Actually Does

### Dataset Storage

TMoW uses Hugging Face `Dataset.save_to_disk` style directories. Its loader
reads `dataset_info.json` and dispatches by the `"type"` field. WorMI uses plain
JSONL files and schema sniffing via `AutoJsonlDataset`.

This storage difference should not be copied into WorMI. It would force changes
across curricula, validators, and evaluators. The transferable part is the
content of each row, not the storage backend.

### VirtualHome Chat Rows

TMoW's `VirtualHomeDatasetBuilder` expects rows shaped like:

```json
{
  "instruction": "...",
  "observation": "...",
  "action": "...",
  "next_observation": "..."
}
```

The chat shape is mostly compatible with WorMI:

```text
system: home robot skill list
user:   Instruction + Observation + Action:
assistant: action
user:   Next observation:
assistant: next_observation
```

However, the local TMoW code should not be copied blindly. Its
`VirtualHomeDatasetBuilder` uses `BASE_PROMPT` directly even though that string
contains a literal `{instruction}` placeholder. WorMI's prompt is cleaner for
the current trainer and should remain authoritative.

### VirtualHome Observation Construction

TMoW's important preprocessing logic is in `VirtualHomeDatasetGenerator`:

1. initialize a `KnowledgeGraph` from `position_graph`;
2. incrementally extend it with `visible_graph` and `agent_graph`;
3. apply refinement when new edges conflict with stale edges;
4. build `next_kg` using `next_agent_graph` and `next_visible_graph`;
5. retrieve compact observations with `num_edges=17`;
6. convert `next_observation` into a compact update, not a full next graph.

The current local code:

- retrieves `obs` from `kg.retrieve(..., num_edges=17)`;
- retrieves `next_obs` from `next_kg.retrieve(..., num_edges=17)`;
- computes `updated_triples` by comparing current and next retrieved triples;
- keeps updates related to the action, the character, or the held object;
- falls back to `No updates` or a `put`/`putin` state update if no delta is
  found.

This is materially different from current WorMI's full-graph row generation.

### TMoW Triple Retrieval

TMoW's `KnowledgeGraph.retrieve` does several useful things:

- removes low-value routing facts from the candidate pool such as
  `(character, inside, ...)` and `adjacent` before sampling;
- ranks triples with BM25 over triple tokens;
- adds a very small sentence-embedding similarity term;
- samples a fixed number of relevant triples;
- explicitly appends critical agent facts:
  - current character room;
  - current held object, or `(character, hold, none)`;
  - adjacent rooms.

This is a good preprocessing pattern for WorMI because it keeps task-relevant
object and state facts visible without flooding the prompt with every class in
the scene.

### ALFWorld Preprocessing

TMoW's ALFWorld preprocessing has two pieces:

1. chat supervision over expanding histories;
2. an additional `observation_graph` generated from the ALFWorld environment.

The graph field is used by TMoW's router. WorMI's model does not consume it
directly, so it should not be treated as mandatory for WorMI reproduction.

The useful ALFWorld lesson is to keep history expansion explicit and auditable.
WorMI already does a similar cumulative expansion in `AlfworldDataset` when
`cumulative=True`, but the generated ALFWorld demo data previously had a
semantic mismatch: real task metadata was in top-level `initial_observation`,
while the loader uses `history[0]["observation"]`.

## What WorMI Currently Does

### VirtualHome Row Generation

WorMI's `tools/build_virtualhome_dataset.py` currently emits one JSONL row per
action step:

```json
{
  "instruction": "...",
  "observation": "full graph triples",
  "action": "...",
  "next_observation": "full next graph triples",
  "_meta": {...}
}
```

`format_observation` renders all class-level triples from the graph:

- object-room `inside`;
- object-object `inside`;
- `on`;
- compact object states such as `open`, `closed`, `on`, `off`;
- character `close`;
- character `hold`;
- room adjacency.

This was intended to expose all facts needed for the four VirtualHome task
families. The problem is that it is too broad for the current learning setup.

### Current Observation Size

Measured on current local data:

```text
/root/autodl-tmp/wormi-data/virtualhome-paperlike-v2-fixed-20260527/scene_0/train.jsonl
  observation triples: mean 193.5, min 192, max 197
  observation chars:   mean 5290.3

/root/autodl-tmp/wormi-data/virtualhome-paperlike-v2-fixed-20260527/test_seen_task_seen_scene.jsonl
  observation triples: mean 263.4, min 192, max 313
  observation chars:   mean 7281.5
```

This is far larger than TMoW's 17 retrieved triples plus mandatory agent facts.

### Stage-1 vs Stage-2 Supervision

WorMI VirtualHome stage 1 uses `behavior_cloning=False`, which expands each row
into:

1. behavior cloning: predict action only;
2. dynamics: predict action and then full `next_observation`;
3. affordance: predict feasible action from observation.

TMoW expert training is closer to behavior + dynamics, but its dynamics target
is compact. It does not train on a full next scene dump in the code inspected.

WorMI stage 2 uses `end_with_action=True`, so stage 2 is action-only. That part
is already close to TMoW's `action_only=True` usage for MoW training/eval.

## Why This Matters for the Current Failure

The bad `seqmeta` and current v2 outputs show first-step action drift:

```text
target:     walk drawing / walk kitchen
prediction: walk bathroom / walk livingroom
```

This is consistent with a preprocessing issue:

- target objects are present but buried in hundreds of triples;
- class-level observations contain duplicate-object ambiguity;
- class-level state collapse can expose contradictory facts, e.g. one instance
  of an object is open while another is closed;
- world models are asked to predict a full next graph that differs only
  slightly from the input graph, so the supervised signal is dominated by
  copying irrelevant triples rather than learning the actionable state change;
- the adapter receives hidden states produced from these noisy world-model
  prompts.

This does not prove preprocessing is the only cause of `seqmeta` failure. It
does show that our current VirtualHome data is not as close to TMoW's embodied
state representation as it first appears.

## What To Borrow

### Borrow 1: Instruction-Conditioned Observation Compaction

Add a WorMI-side observation compactor:

```text
compact_observation(full_graph, instruction, num_edges=17)
```

It should:

1. parse or construct graph triples;
2. score candidate triples against the instruction;
3. keep top task-relevant triples;
4. always keep:
   - character room;
   - character held object;
   - objects named in the instruction;
   - receptacles named in the instruction;
   - current target object state if the task is `open` or `turnon`;
   - candidate source/target location facts for `puton` and `placein`;
5. preserve a deterministic order or deterministic seed.

TMoW samples with `np.random.choice`; for reproduction, deterministic top-k is
safer unless we intentionally implement augmentation.

### Borrow 2: Compact Next-State Updates

Replace full `next_observation` targets for world-model dynamics with compact
updates:

```text
next_observation = changed triples relevant to action and held object
```

For example:

```text
(character, hold, drawing)
```

or:

```text
(drawing, inside, sink)
```

This better matches the world-model objective: predict how the state changes
after the action, not copy the entire house graph.

### Borrow 3: KnowledgeGraph Refinement

When generating trajectories from graph states, keep an instance-level
`KnowledgeGraph` before rendering class-level text. Use refinement to remove
stale edges for the same instance/relation. Only collapse to class-level text
after retrieval.

This is better than rendering each raw graph independently because it reduces
stale/conflicting state facts and gives a consistent belief-state view.

### Borrow 4: Preprocessing Diagnostics

Every generated dataset should emit a preprocessing report with:

- rows per scene/split;
- trajectories per scene/split;
- observation triple count distribution;
- next-observation triple count distribution;
- prompt character/token length distribution;
- percentage of examples where instruction objects are visible;
- percentage where goal triple is visible in final state;
- count of contradictory class-level state facts;
- action unchanged-observation rate;
- train/test trajectory overlap;
- exact action-sequence overlap.

WorMI already has part of this in `validate_virtualhome_dataset.py`; it should
be extended for compactness and contradiction checks.

## What Not To Borrow Directly

### Do Not Switch WorMI To TMoW Dataset Storage

TMoW's HF dataset builder would require a broad integration change. It is not
needed. Keep WorMI JSONL.

### Do Not Copy TMoW's VirtualHome Prompt As-Is

The local TMoW prompt path contains a likely bug: it uses `BASE_PROMPT` with a
literal `{instruction}` placeholder. WorMI's prompt is already aligned with its
TRL collator and Llama-3 chat template.

### Do Not Depend On TMoW's CUDA-Only Embedding Helper

TMoW's `embedding_fns` calls `.cuda()` internally. A reusable WorMI compactor
should accept a device argument and support CPU/GPU.

### Do Not Use Random Sampling As The Default

TMoW retrieves with probabilistic sampling. For paper reproduction and
debugging, deterministic top-k is easier to validate. Randomized retrieval can
be added later as data augmentation.

## Concrete Implementation Plan

### Phase 1: Build A Reversible Diagnostic Compactor

Add a script such as:

```text
tools/compact_virtualhome_observations.py
```

Input:

```text
--input-root /root/autodl-tmp/wormi-data/virtualhome-paperlike-v2-fixed-20260527
--output-root /root/autodl-tmp/wormi-data/virtualhome-paperlike-v2-fixed-compact17-20260528
--num-edges 17
```

It should read existing JSONL rows, parse the full triple strings, write compact
`observation` and compact `next_observation`, and preserve `_meta`.

This is not the final perfect path, because it works from already-collapsed
class-level text. But it is a low-risk test that can answer whether prompt
compaction improves first-step grounding.

### Phase 1 Status: Implemented Diagnostic Compactor

Implemented:

```text
tools/compact_virtualhome_observations.py
```

The script preserves the current WorMI JSONL schema and directory layout, keeps
symlinks, rewrites only `observation` and `next_observation`, and records the
per-row preprocessing stats in `_meta.observation_preprocessing`.

Full compact dataset generated:

```text
/root/autodl-tmp/wormi-data/virtualhome-paperlike-v2-fixed-compact17-20260528
```

Command used:

```bash
python3 tools/compact_virtualhome_observations.py \
  --input-root /root/autodl-tmp/wormi-data/virtualhome-paperlike-v2-fixed-20260527 \
  --output-root /root/autodl-tmp/wormi-data/virtualhome-paperlike-v2-fixed-compact17-20260528 \
  --num-edges 17 \
  --next-mode delta
```

Full-run summary:

```text
total_rows: 5451
files: 10 jsonl, 9 symlinks, 1 copied metadata file
observation triples mean: 269.45 -> 23.51
next_observation triples mean: 270.17 -> 1.24
next rows with No updates: 0
compact observation min/max triples: 4 / 84
compact next_observation min/max triples: 1 / 3
missing task_args in observation: 0 / 5451
```

Loader compatibility check passed on
`scene_0/train.jsonl`:

```text
AutoJsonlDataset.load(..., end_with_action=True, cumulative=True): 46 rows
as_chat(Llama-3.2-1B-Instruct tokenizer): 46 rows with text column
```

The remaining caveat is important: this is still a post-hoc compactor over the
already serialized class-level observation text. It is good enough for a
controlled training test, but the final reproduction-quality version should
regenerate compact observations from VirtualHome graph states before class-level
collapse.

### Phase 2: Regenerate From Graph States

The reproduction-grade path should not depend on post-hoc conversion of an
already serialized full JSONL dataset. It should execute the VirtualHome graph
program and render compact observations before rows are written.

### Phase 2 Status: Implemented Independent Builder

Implemented:

```text
tools/build_virtualhome_dataset_tmow_compact.py
```

This is intentionally separate from `tools/build_virtualhome_dataset.py`, so the
existing full-observation builder remains available. The independent builder
reuses WorMI's paper-aligned task/scene/program-selection helpers, but it writes
`observation` and `next_observation` directly from replayed graph states using a
TMoW-style retrieval policy:

```text
full graph state
  -> deterministic instruction/action-conditioned retrieved facts
  -> fill to 17 retrieved facts, then append mandatory task/agent facts
  -> next_observation as compact delta update
```

The fill-to-17 step matters. A stricter mandatory-only compact dataset had no
trajectory leakage, but did create 66 train/test exact row overlaps because
seen-task examples across scenes collapsed to identical text. Filling retrieved
facts restores enough scene context to remove exact row overlap while keeping
prompts far smaller than the full scene graph.

Full independent compact dataset generated:

```text
/root/autodl-tmp/wormi-data/virtualhome-paperlike-tmow-compact-fill17-20260528
```

Command used:

```bash
python3 tools/build_virtualhome_dataset_tmow_compact.py \
  --scene-inits-json /root/autodl-tmp/wormi-data/scene-inits/init_graphs_20_semantic.json \
  --vh-src /root/autodl-tmp/wormi-data/virtualhome-src \
  --output-dir /root/autodl-tmp/wormi-data/virtualhome-paperlike-tmow-compact-fill17-20260528 \
  --compact-num-edges 17 \
  --next-observation-mode delta \
  --target-trajectories 1023
```

Validation command:

```bash
.venv/bin/python tools/validate_virtualhome_dataset.py \
  --data-root /root/autodl-tmp/wormi-data/virtualhome-paperlike-tmow-compact-fill17-20260528 \
  --scene-inits-json /root/autodl-tmp/wormi-data/scene-inits/init_graphs_20_semantic.json \
  --vh-src /root/autodl-tmp/wormi-data/virtualhome-src \
  --output-json reports/virtualhome/validation/vh-paperlike-tmow-compact-fill17-validation-2026-05-28.json
```

Full-run validation result:

```text
rows: 5451
trajectories: 1023
selected tasks: 78 = 9 turnon + 7 open + 30 puton + 32 placein
seen/unseen tasks: 16 / 62
seen/unseen scenes: 6 / 14
train/test trajectory overlap: 0
train/test exact row overlap: 0
replay failures: 0
compact obs mismatches after replay: 0
compact next_obs mismatches after replay: 0
goal failures: 0
missing task_args in observation: 0
```

Compactness result:

```text
observation triples mean: 269.45 -> 37.54
next_observation triples mean: 270.17 -> 1.24
compact observation min/max triples: 21 / 95
compact next_observation min/max triples: 1 / 3
```

The compact observation mean is higher than the post-hoc strict version because
this builder keeps TMoW-like retrieved context to avoid cross-split sample
collapse. This is the better training candidate for a paper-style reproduction.

### Phase 2 Status: Experiment Entrypoints

Added rebuild entrypoint:

```text
sh/wormi-build-vh-data-tmow-compact.sh
```

It rebuilds `/root/autodl-tmp/wormi-data/virtualhome-paperlike-tmow-compact-fill17-20260528`
with `tools/build_virtualhome_dataset_tmow_compact.py`, then runs the
compact-aware validator.

Added full pipeline entrypoint:

```text
sh/wormi-vh-paperlike-tmow-compact-full.sh
```

Defaults:

```text
DATA_ROOT=/root/autodl-tmp/wormi-data/virtualhome-paperlike-tmow-compact-fill17-20260528
WORLD_CKPT_ROOT=/root/autodl-tmp/wormi-checkpoints/world-vh-paperlike-tmow-compact-fill17-20260528
WORMI_CKPT_ROOT=/root/autodl-tmp/wormi-checkpoints/wormi-vh-paperlike-tmow-compact-fill17-20260528
WORMI_SEQUENTIAL_META_LEARNING=1
```

The last line is deliberate: stage 2 should use the sequential Reptile path. The threaded path is currently guarded because its shared-step accounting and default beta semantics are not paper-faithful.

Preflight-only check passed:

```bash
PREFLIGHT_ONLY=1 bash sh/wormi-vh-paperlike-tmow-compact-full.sh
```

It reran full data validation and stopped before training. The script printed
`seq meta: 0` in the earlier preflight log came from the old default; after the trainer guard, stage 2 is forced to sequential unless `WORMI_ALLOW_UNSAFE_THREADED_META=1` is set. Validation reported no trajectory overlap, no exact row
overlap, and no replay mismatches.

To run the actual compact-data reproduction attempt directly:

```bash
bash sh/wormi-vh-paperlike-tmow-compact-full.sh
```

Added detached/queued launcher:

```text
sh/wormi-vh-paperlike-tmow-compact-full-tmux.sh
```

Current run status as of 2026-05-28 03:35 local time:

```text
tmux session: wormi_tmow_compact_full
state: waiting for old v2-fixed greedy rollout PID 180560
launch log: /root/autodl-tmp/wormi-logs/vh-pipeline-paperlike-tmow-compact-fill17-20260528-detached/launch.log
```

Command used:

```bash
WAIT_FOR_PID=180560 WAIT_POLL_SECONDS=300 \
  bash sh/wormi-vh-paperlike-tmow-compact-full-tmux.sh
```

Monitor commands:

```bash
tmux attach -t wormi_tmow_compact_full
tail -f /root/autodl-tmp/wormi-logs/vh-pipeline-paperlike-tmow-compact-fill17-20260528-detached/launch.log
tail -f /root/autodl-tmp/wormi-logs/vh-pipeline-paperlike-tmow-compact-fill17-20260528/pipeline.log
```

Added compact status helper:

```text
sh/wormi-vh-paperlike-tmow-compact-status.sh
```

The status helper now also summarizes the saved validation JSON, alignment audit, and reproducibility manifest, including `errors`, `warnings`, replay mismatches, the Llama-3 chat-template gate, compact-observation means, missing task/action arguments, failed alignment verdicts, JSONL row counts, and SHA256-backed manifest totals.

Run:

```bash
bash sh/wormi-vh-paperlike-tmow-compact-status.sh
```

It reports tmux sessions, active GPU process, pipeline `status.tsv`, compact
stage-1/stage-2 checkpoints, detached launch tail, and pipeline log tail.

Latest checked state as of 2026-05-28 04:08 local time:

```text
old rollout PID 180560: still running on GPU, greedy rollout col2 about 190/212
old rollout still has col3_unseen_unseen after col2, so GPU release is not immediate
compact tmux session: still waiting for PID 180560
compact stage1 checkpoints: missing
compact stage2 checkpoint: missing
```

When PID 180560 exits, the queued runner will start the compact full pipeline
with `WORMI_SEQUENTIAL_META_LEARNING=1`.

One direct JSONL sample check from `scene_0/train.jsonl` showed the expected
WorMI fields:

```text
keys: action, instruction, observation, next_observation, _meta
example instruction: Place drawing in sink
example action: walk sink
source observation triples: 193
compact observation triples: 27
source next-observation triples: 196
compact next-observation triples: 2
preprocessing mode: tmow_compact_from_graph_state
```

The validator now has a repeatable `--check-chat-template` gate, and both
compact entrypoint scripts pass it. That gate also instantiates TRL's
`DataCollatorForCompletionOnlyLM` with the same response template used by
`WorMISubTrainer`, so it verifies the actual completion-only loss mask rather
than only checking the chat string. A full offline check with the training
tokenizer produced this saved evidence in
`reports/virtualhome/validation/vh-paperlike-tmow-compact-fill17-validation-2026-05-28.json`:

```text
errors: 0
warnings: 0
action_samples: 5451
world_samples: 16353
response_template: <|start_header_id|>assistant<|end_header_id|>
max_action_tokens: 947
max_world_tokens: 967
loss_mask_samples: 21804
min_supervised_tokens: 4
max_supervised_tokens: 24
bad_count: 0
```

Added a repeatable TMoW-alignment audit:

```text
tools/audit_tmow_compact_alignment.py
reports/virtualhome/validation/vh-paperlike-tmow-compact-alignment-audit-2026-05-28.json
```

The audit checks the concrete preprocessing invariants borrowed from TMoW:

```text
rows: 5451
mode_counts: tmow_compact_from_graph_state = 5451
source_counts: virtualhome_evolving_graph = 5451
next_mode_counts: delta = 5451
fill_to_num_edges=True rows: 5451
source observation triples mean: 269.45
compact observation triples mean: 37.54
source next-observation triples mean: 270.17
compact next-observation triples mean: 1.24
missing task args in observation: 0
missing action args in observation: 0
No updates targets: 0
all alignment verdicts: true
```

This audit maps directly to the TMoW code path: TMoW retrieves compact
instruction-conditioned observations with `KnowledgeGraph.retrieve(...,
num_edges=17)` and turns next state into action-related update triples. The
WorMI version keeps that data idea but makes it deterministic and WorMI-compatible
by rendering from replayed EvolvingGraph states, adding mandatory task/action
facts, and preserving the original `instruction / observation / action /
next_observation` JSONL contract.

Added a reproducibility manifest:

```text
tools/audit_dataset_repro_manifest.py
reports/virtualhome/validation/vh-paperlike-tmow-compact-repro-manifest-2026-05-28.json
```

It records SHA256 hashes, byte counts, and row counts for the generated JSONL
files and hashes for the data/validation scripts used to create this artifact:

```text
jsonl_files: 10
jsonl_rows: 5451
jsonl_bytes: 11230724
metadata_files: 2
source_files: 8
```

### Phase 3: Train A Controlled Pair

Using the same scene cache and same selected tasks:

1. build `full` data;
2. build `tmow_compact` data;
3. train stage 1 on both;
4. train stage 2 with `WORMI_SEQUENTIAL_META_LEARNING=1`;
5. run the same Table1 quick gate and rollout subset.

Do not compare compact data trained with `seqmeta` against full data trained
with a different meta-learning implementation; that would confound preprocessing and training method.

### Phase 4: Revisit `seqmeta`

Only after compact preprocessing is validated should `seqmeta` be retried. If
`seqmeta` still fails on compact data, the failure is much more likely to be the
meta-update implementation itself. If it improves materially, preprocessing was
a major contributor to the failure.

## Acceptance Gates

A compact WorMI dataset should satisfy:

1. loader compatibility:
   - `AutoJsonlDataset.load(path, end_with_action=True, cumulative=True)` works;
   - `as_chat(tokenizer)` produces supervised assistant spans;
2. observation compactness:
   - strict diagnostic compact mode can stay below 30 mean observation triples;
   - TMoW-style fill17 graph-state mode should stay far below full graph size
     while keeping train/test exact row overlap at 0;
   - mean next-observation triples below 10 for delta mode;
3. task visibility:
   - at least one target object/receptacle mention in every `puton` and
     `placein` observation;
   - target object state visible for `open` and `turnon`;
4. semantic validity:
   - final goal triple visible after executing the trajectory;
   - no train/test trajectory overlap;
5. model gate:
   - Table1 first-step match improves over current v2;
   - Table1 col-1 quick SR is nonzero before full rollout.

## Bottom Line

To borrow TMoW for WorMI reproduction, focus on data:

```text
full scene graph prompt
  -> instruction-conditioned compact belief state

full next scene graph target
  -> compact state-update target
```

This is the most plausible preprocessing gap between the current WorMI
reproduction and the style of embodied data used by TMoW. It also attacks the
observed failure directly: first-step object/action grounding drift.


### 2026-05-28 meta-learning guard update

The compact pipeline was originally queued with `WORMI_SEQUENTIAL_META_LEARNING=0`, but the current threaded meta-learning implementation is not paper-faithful: it shares the step boundary across trainers and defaults to direct averaging unless `WORMI_THREADED_META_USE_BETA=1` is set. Before compact stage 2 started, `wormi/trainer.py` was patched so meta-learning defaults to sequential Reptile and only allows the old threaded path when `WORMI_ALLOW_UNSAFE_THREADED_META=1` is explicitly set. The compact launcher defaults were also changed to `WORMI_SEQUENTIAL_META_LEARNING=1`.

The stage-2 entrypoint `sh/wormi-train-vh-wormi.sh` now also forces `WORMI_SEQUENTIAL_META_LEARNING=1` unless `WORMI_ALLOW_UNSAFE_THREADED_META=1` is explicitly set, so queued compact runs will log and execute the sequential path even if the parent tmux environment was created with the old `0` default.

The rollout entrypoint `sh/wormi-eval-vh-rollout.sh` now also defaults to greedy decoding (`TEMPERATURE=0.0`) unless `WORMI_VH_ROLLOUT_ALLOW_SAMPLING=1` is explicitly set. This prevents the compact pipeline from inheriting the old stochastic `TEMPERATURE=1.0` full-script default.


## 2026-05-28 Correction: Why Compact Rollout Still Scored 0

The first `paperlike-tmow-compact-fill17-20260528` rollout was not a valid compact-observation evaluation. Two issues were found after inspecting the live rollout traces:

1. `eval_vh_rollout.py` still rendered observations with `tools.build_virtualhome_dataset.format_observation(graph)`, so the model was trained on compact observations but evaluated on full scene-graph observations. The observed failures were repeated single actions such as `walk bathroom` or repeated precondition-failing `putin toilet drawing`, not empty outputs.
2. The compact builder used the target row action to rank the current `observation` facts. That action is the behavior-cloning label, so the policy input was action-conditioned during training and cannot be reproduced during rollout. Current observations must be generated from only the instruction, task args, and current graph state. The action is still allowed for `next_observation` deltas because that turn occurs after the assistant action in the chat.

Code fixes applied:

- `tools/build_virtualhome_dataset_tmow_compact.py`: current compact observation now calls `compact_observation(..., action="")`; metadata records `current_observation_action_conditioned=false`.
- `tools/compact_virtualhome_observations.py`: post-hoc compaction follows the same action-agnostic rule for current observations.
- `wormi/scripts/eval_vh_rollout.py`: rollout auto-detects `tmow_compact` data from `_meta.observation_preprocessing` and renders runtime observations with the same action-agnostic compact function. Summary rows now record `observation_format`.
- `sh/wormi-vh-paperlike-tmow-compact-full.sh`: `RUN_BASE` now controls data/checkpoint paths, and stage1 removes per-scene intermediate `checkpoint-*` only after `scene_i/last` exists.

The invalid rollout was stopped. A corrected end-to-end run was started in tmux session `wormi_tmow_compact_aa_full` with run base `paperlike-tmow-compact-aa-fill17-20260528`.


<!-- aa-fill17-live-rollout-diagnosis -->

## 2026-05-28 Live Diagnosis During Corrected AA Rollout

The corrected AA rollout is executable and no longer fails from parser/runtime observation mismatch. However, early target-state rollout remains very low: `col_1_seen_seen` finished at 1/32 success and `col_2_seen_unseen` was still 0 success in the first checked chunk.

The dominant observed behavior is repeated executable `walk ...` actions, not invalid action parsing. Example failed traces repeatedly output `walk bathroom`, `walk bedroom`, or `walk dresser` for all 30 rollout steps.

Two concrete semantic issues were identified from the live traces:

1. Class-level duplicate ambiguity. For `Place drawing in toilet` in `TrimmedTestScene6_graph__v1`, the compact observation contains facts such as `(drawing, inside, bedroom)`, `(drawing, inside, kitchen)`, and `(drawing, inside, livingroom)` at the same time. The gold expert trajectory is tied to one instance and begins `walk bedroom -> walk drawing -> grab drawing -> ...`, but the text prompt has no instance id to tell the policy which `drawing` the expert chose. This makes exact expert imitation semantically underdetermined.
2. Rollout instance binding is currently first-id based. `eval_vh_rollout.py` maps `walk/grab/open/put` class names to `_find_first_id(graph, class_name)`. VirtualHome script parsing requires explicit ids, so class-level predictions must be bound by the evaluator. Using the first graph id is not semantically faithful when several nodes share the same class. A better evaluator should choose the held object for `put/putin`, a close object for `grab/open/switchon`, and a same-room/goal-relevant object for `walk`.

The first issue is a data construction issue and likely requires another data rebuild/retrain if we want a semantically clean policy target. The second issue is an evaluator issue and can be fixed without retraining, but it will not by itself solve the repeated-room policy collapse observed in the current model.

<!-- aa-fill17-bindfix-quick-watcher -->

## 2026-05-28 Bind-Fix Quick Rollout Handoff

The full `paperlike-tmow-compact-aa-fill17-20260528` rollout process started at
`2026-05-28T10:38:41+08:00`. The semantic instance-binding evaluator fix in
`wormi/scripts/eval_vh_rollout.py` was written later at
`2026-05-28T10:53:07+08:00`, so the currently running full rollout does not
include that fix.

As of the handoff check, the old-evaluator rollout had:

```text
col_1_seen_seen:     32/32 episodes, SR=3.12%, PS=29.12
col_2_seen_unseen:  181/212 episodes, SR=1.10%, PS≈29.7
```

The step-level traces are still dominated by repeated executable navigation,
especially `walk bathroom`, `walk livingroom`, `walk dresser`, `walk kitchen`,
and `walk bedroom`. This reinforces that the main policy failure is not just a
parser/runtime crash.

To avoid overwriting the full rollout, a follow-up watcher was started:

```text
tmux session: wormi_tmow_compact_aa_bindfix_quick
script:       sh/wormi-run-aa-bindfix-quick-after-current.sh
output:       /root/autodl-tmp/wormi-checkpoints/wormi-vh-paperlike-tmow-compact-aa-fill17-20260528/wormi-vh-n6/vh-rollout-paperlike-tmow-compact-aa-fill17-20260528-bindfix-quick
samples:      32 episodes per eval column
```

That quick rollout will start only after the current full rollout process exits,
and will use the latest evaluator code with semantic class-to-instance binding.

<!-- aa-fill17-instance-grounding-implementation -->

## 2026-05-28 Instance-Grounded Compact Data Fix

A semantic data fix has been implemented but not yet used for a new training run.
The issue addressed is class-level duplicate ambiguity: compact observations can
contain several same-class facts, for example one `drawing` in `bathroom`, one in
`bedroom`, and one in `livingroom`, while the expert trajectory selected one
concrete `drawing` instance.

Implemented code changes:

```text
tools/compact_virtualhome_observations.py
  graph_observation_triples(...)
  selected_instance_ids_from_meta(...)
  instance_grounded_observation_triples(...)
  format_instance_grounded_observation(...)

tools/build_virtualhome_dataset_tmow_compact.py
  compact rows now first render an instance-grounded graph observation using
  planner_debug source_id/target_id/source_container, then apply compact
  retrieval. Metadata records instance_grounded=true and grounding_node_ids.

tools/validate_virtualhome_dataset.py
  replay validation now recomputes compact observations with the same
  instance-grounded source when metadata requests it.

wormi/scripts/eval_vh_rollout.py
  rollout observation rendering now follows instance_grounded metadata, so a
  future rebuilt dataset can be evaluated with the same observation contract.
```

Sanity checks run:

```text
python3 -m py_compile tools/compact_virtualhome_observations.py \
  tools/build_virtualhome_dataset_tmow_compact.py \
  tools/validate_virtualhome_dataset.py \
  wormi/scripts/eval_vh_rollout.py

base.format_observation == graph_observation_triples for all 20 cached init
graphs.
```

A replayed example confirms the intended behavior. For `Place drawing in toilet`
in `TrimmedTestScene6_graph__v1`, raw class-level facts included:

```text
(drawing, inside, bathroom)
(drawing, inside, bedroom)
(drawing, inside, livingroom)
```

With the new instance-grounded renderer and `planner_debug` ids `[118, 302]`,
the task-object facts become:

```text
(character, close, drawing)
(character, close, toilet)
(character, hold, drawing)
(drawing, inside, bathroom)
(toilet, inside, bathroom)
(toilet, is, off)
(toilet, is, open)
```

This removes the contradictory same-class source locations while preserving the
selected expert instance and target state. A new dataset must still be rebuilt
and retrained before this can affect model results.

<!-- aa-fill17-instance-grounding-builder-smoke -->

Builder smoke check after the implementation called
`_execute_tmow_compact_candidate('placein', ('drawing', 'toilet'), ...)` on
`TrimmedTestScene6_graph__v1`. It produced 7 rows with
`instance_grounded=true`, `grounding_node_ids=[37, 108]`, raw source observation
292 triples, grounded source observation 289 triples, compact observation 26
triples, and compact next-observation delta 2 triples. The compact task-object
facts contained one selected `drawing` location rather than multiple same-class
source locations.

<!-- aa-fill17-user-stop-review-20260528 -->

## 2026-05-28 User Stop And Review Point

At the user's request, all AA-fill17 evaluation/background sessions were stopped
around `2026-05-28T11:18:00+08:00`. The GPU was released and `nvidia-smi`
reported no running compute processes. No new data rebuild or training run was
started after this stop request.

Complete result available:

```text
Table1 exact-match:
col_1_seen_seen      SR=0.000000  PS=4.843750   episodes=32
col_2_seen_unseen    SR=0.004717  PS=5.231132   episodes=212
col_3_unseen_unseen  SR=0.003333  PS=5.143333   episodes=300
```

Partial rollout result available at stop time; it should not be reported as a
final full rollout because col3 was interrupted before 300/300 episodes:

```text
col_1_seen_seen       32/32   SR=0.031250  PS=29.12
col_2_seen_unseen    212/212  SR=0.014151  PS=29.60
col_3_unseen_unseen  136/300  SR=0.007353  PS=29.79
```

The stopped rollout was still dominated by repeated executable `walk ...`
actions, not by parser crashes. The bindfix quick watcher was also stopped, so
there is no pending background evaluation job.

