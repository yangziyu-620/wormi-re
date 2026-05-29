# VirtualHome Data Construction Audit - 2026-05-27

Context:
- Active dataset: `/root/autodl-tmp/wormi-data/virtualhome`
- Active run: `wormi-vh-paperlike-v1-threaded-lockfix2-20260527`
- Training/eval were left running while this audit used read-only checks.

Paper alignment anchors:
- Paper reports VirtualHome as 1,023 episodes, 78 tasks, 16 seen tasks, 62 unseen tasks, 20 scenes, 6 seen scenes, 14 unseen scenes.
- Paper Appendix B.1 describes VirtualHome observations as graph triples and the action set as `walk`, `grab`, `open`, `put`, `putin`, `switchon`.
- Paper Table A.6 uses N=6 world models, K=3 retrieved world models, world model steps=2000, compound-attention meta steps=8, inner steps=30.

Current dataset passes structural checks:
- `quality_report.json` has `errors: []`.
- Total rows: 7,868.
- Total trajectories: 1,023.
- Split trajectories:
  - seen_seen: 93
  - seen_unseen: 212
  - unseen_seen: 333
  - unseen_unseen: 385
- Root eval files and `eval_col_*` symlinks are present.
- `train_test_overlap: 0`.
- Selected tasks match 16 seen / 62 unseen.

Critical semantic issues found:

1. Room namespace mismatch between observation, prompt, and action labels.
   - Observation canonicalizes:
     - `dining_room -> kitchen`
     - `home_office -> livingroom`
   - Action labels still emit raw graph names:
     - `walk dining_room`: 1,180 rows
     - `walk home_office`: 718 rows
   - Prompt only lists rooms: `livingroom, bathroom, kitchen, bedroom`.
   - Example: target action is `walk dining_room`, while the observation contains `kitchen` and not `dining_room`.
   - Impact:
     - Table1 exact-match eval penalizes canonical predictions such as `walk kitchen`.
     - Training asks the model to output labels that are not present in the observation namespace.

2. Planner emits redundant/no-op walk steps.
   - Walk rows are 5,752 / 7,868 = 73.1% of all rows.
   - Sorted by `trajectory_id` and `step_index`, unchanged-observation walk counts are high:
     - seen_seen: 180 / 513 walks = 35.1%
     - seen_unseen: 373 / 1,212 walks = 30.8%
     - unseen_seen: 604 / 1,876 walks = 32.2%
     - unseen_unseen: 649 / 2,151 walks = 30.2%
   - Consecutive duplicate actions:
     - seen_seen: 67
     - seen_unseen: 111
     - unseen_seen: 197
     - unseen_unseen: 179
   - First two actions are identical in many trajectories:
     - seen_seen: 41 / 93 = 44.1%
     - seen_unseen: 57 / 212 = 26.9%
     - unseen_seen: 113 / 333 = 33.9%
     - unseen_unseen: 91 / 385 = 23.6%
   - Root cause in `tools/build_virtualhome_dataset.py`:
     - `_paperlike_program` appends `walk start_room` and then `walk source_room` / `walk target_room` without checking whether the agent is already there.
     - For object move tasks it also appends `walk source_room` again after `grab`, even when the agent did not leave.
   - Impact:
     - The model sees many valid labels that do not change the state.
     - Table1 failures show repeated room walking and confusion about when to switch from room navigation to object interaction.

3. Class-level action labels are ambiguous in multi-instance scenes.
   - Actions are class-level (`walk drawing`, `put drawing toaster`) while VirtualHome graphs may contain many nodes with the same class.
   - Across current data:
     - Action script references checked: 8,735
     - References to classes with multiple instances: 3,171 = 36.3%
     - References where expert id is not the rollout evaluator's first id for that class: 397 = 4.5%
   - Examples:
     - `walk drawing` where graph has 8 or 9 drawings.
     - Expert script may use `<drawing> (2009)`, while rollout evaluator would execute the first `<drawing>` id.
   - Impact:
     - Table1 string matching can pass while rollout execution may use the wrong instance.
     - Observations collapse duplicate class triples, so the model cannot always know which instance the expert intended.

4. Current data differs materially from the earlier high-SR task-aware dataset.
   - Earlier task-aware data had average trajectory length around 4 steps and achieved high clean/salvage results.
   - Current paper-like planner has average trajectory length around 7.7, but many extra steps are redundant/no-op walks rather than meaningful state transitions.
   - Therefore the low current SR is not explained by code alone; the current dataset is semantically harder and noisier.

Current Table1 failure signal:
- Latest Table1 offline result:
  - col_1_seen_seen: SR 0.0, PS 6.65625
  - col_2_seen_unseen: SR 0.0, PS 7.627358
  - col_3_unseen_unseen: SR 0.0, PS 7.236667
- In col_1 first failures:
  - 12 / 32 first failures target raw rooms (`walk dining_room` or `walk home_office`).
  - 15 / 32 first failures repeat the previous prediction.
  - Example: target sequence starts `walk bathroom`, `walk bathroom`, `walk drawing`; prediction repeats `walk bathroom` at step 2.

Recommended fix order:

1. Make action and observation namespaces identical.
   - Canonicalize room targets in action labels with the same mapping used in observations:
     - `dining_room -> kitchen`
     - `home_office -> livingroom`
     - `living_room -> livingroom`
   - Keep script lines using raw VirtualHome node ids for simulator replay.

2. Replace blind `_append_walk` calls with a position-aware planner.
   - Track current agent location.
   - Append `walk room` only when the target room differs from current room.
   - Append `walk object` only when it causes a meaningful `CLOSE` relation or is required by VirtualHome execution.
   - Do not append `walk source_room` after `grab` unless the simulated graph moved the character elsewhere.
   - Reject or repair any generated trajectory with consecutive duplicate actions or unchanged-observation walk transitions above a small threshold.

3. Reduce class-level ambiguity.
   - Preferred fast repair: filter candidate tasks/scenes so source and target classes are unique or the selected expert id equals the evaluator's first id.
   - More complete repair: include instance-disambiguating names consistently in observation, action, and evaluator. This is a larger protocol change and should be avoided unless needed.

4. Strengthen validation before retraining.
   - Add hard validation checks:
     - zero raw room aliases in action labels
     - no consecutive duplicate actions
     - unchanged-observation walk rate below a fixed threshold
     - no action class whose expert id disagrees with evaluator id, unless the action is a room alias explicitly handled by evaluator
     - Table1 col_1 quick gate on a small subset before full stage1/stage2 rerun

Conclusion:
- The active training/eval can continue for logging, but these results should be treated as diagnostic, not a valid reproduction.
- A rebuild is required before retraining if the goal is a semantically valid and paper-aligned VirtualHome dataset.
