# VirtualHome Balanced Reconstruction Semantic Audit - 2026-05-28

Dataset:

```text
/root/autodl-tmp/wormi-data/virtualhome-balanced-reconstruction-20260528
```

## Verdict

The dataset is structurally valid and executable, but its current semantics are not fully satisfactory for training. It should be treated as a strong structural prototype, not a final training dataset.

The core issue is not EvolvingGraph replay. Replay passes. The issue is that the expert planner selects concrete object instances, while observations and actions are serialized at class level. When a class has many instances, the observation may say the same class is in several rooms or on several surfaces at once. The model cannot know which instance the expert intends.

## Passed Checks

```text
total episodes: 1023
train/test trajectory overlap: 0
replay failures: 0
observation mismatches: 0
next observation mismatches: 0
goal failures: 0
no unchanged-observation transitions after filtering
```

Split legality also passes:

```text
train: 384 episodes, all seen_task intersect seen_scene
eval A: 96 episodes, all seen_task intersect seen_scene
eval B: 224 episodes, all seen_task intersect unseen_scene
eval C: 319 episodes, all unseen_task intersect unseen_scene
```

Eval C is not residual sampling. It is drawn from the legal `unseen_task intersect unseen_scene` pool.

## Major Semantic Problems

### 1. Class-Level Multi-Instance Ambiguity

Across 1023 episodes:

```text
source class has multiple graph nodes: 781 / 1023 = 76.3%
source class has multiple locations in observation: 961 / 1023 = 93.9%
target class has multiple graph nodes: 305 / 1023 = 29.8%
target class has multiple locations in observation: 586 / 1023 = 57.3%
```

For object-move tasks this is severe:

```text
puton source multi-location: 100%
placein source multi-location: 100%
```

Example:

```text
Instruction: Place mat in toaster
Actions: walk bathroom, walk mat, grab mat, walk kitchen, walk toaster, putin mat toaster
Initial observation includes:
  (mat, inside, bathroom)
  (mat, inside, bedroom)
  (mat, inside, kitchen)
  (mat, inside, livingroom)
  (mat, on, table)
```

The selected expert instance is one concrete mat, but the class-level observation merges all mats. This makes the action target under-specified.

### 2. Task Selection Is Biased Toward High-Coverage Objects

Seen task object distribution is narrow:

```text
seen source top:
  drawing: 10
  mat: 3
```

Seen tasks include:

```text
puton:drawing|bathroom_cabinet
puton:drawing|bathroom_counter
puton:drawing|bed
puton:drawing|chair
puton:drawing|desk
puton:drawing|kitchen_counter
placein:drawing|dresser
placein:drawing|sink
placein:drawing|toaster
placein:drawing|toilet
placein:mat|sink
placein:mat|toaster
placein:mat|toilet
```

This happened because the builder prefers tasks with broad execution coverage. That helps data volume but creates semantic/object bias.

### 3. Some Tasks Are Simulator-Legal But Commonsense Weak

Examples:

```text
Place mat in toaster
Place drawing in toilet
Put keyboard on bathroom counter
```

These are executable in VirtualHome, but they are not natural household goals. If the goal is strict VirtualHome affordance-based reproduction, this may be acceptable. If the goal is embodied commonsense behavior, this is weak.

### 4. Prompt/Label Mismatch Still Exists

Current `wormi/datasets/virtualhome.py` prompt says:

```text
switch [object]
putin [target object]
put [target object]
```

But the dataset labels are:

```text
switchon object
putin source target
put source target
```

This mismatch should be fixed before training. Otherwise the examples teach one protocol while the system prompt describes another.

## Recommendation

Do not train final experiments on this dataset yet.

Best next correction:

1. Keep the split protocol and legal-pool sampling.
2. Add a semantic filter for class-level observations:
   - source object class must have one unambiguous selected location in the rendered observation;
   - target should also be unique or explicitly disambiguated;
   - reject tasks where the class-level observation collapses many instances into contradictory facts.
3. If this makes the 1023 target impossible, choose one of two explicit variants:
   - `unique-class-strict`: fewer episodes, cleaner semantics;
   - `instance-grounded`: keep 1023 episodes, but serialize selected instance facts or instance IDs, which deviates from paper graph-class format.
4. Fix the VirtualHome prompt to match actual action labels.

Current dataset is useful as a structural prototype and split audit. It is not yet semantically clean enough to explain bad training results away.

## Source-Unique Rebuild Result

A construction-time semantic gate was added to `tools/build_virtualhome_dataset_balanced.py`:

```text
--semantic-gate source_unique
```

The gate runs before EvolvingGraph execution, so candidates with ambiguous source classes are rejected before the expensive simulator step.

New dataset:

```text
/root/autodl-tmp/wormi-data/virtualhome-balanced-source-unique-20260528
```

Validation:

```text
reports/virtualhome/validation/vh-balanced-source-unique-validation-uv-2026-05-28.json
```

Counts:

```text
total episodes: 1023
total rows: 5263
train: 384
eval A: 96
eval B: 224
eval C: 319
train/test trajectory overlap: 0
replay failures: 0
obs mismatches: 0
next obs mismatches: 0
goal failures: 0
```

Semantic ambiguity after source gate:

```text
source class has multiple graph nodes: 0 / 1023 = 0.0%
source class has multiple observed locations: 344 / 1023 = 33.6%
target class has multiple graph nodes: 251 / 1023 = 24.5%
target class has multiple observed locations: 607 / 1023 = 59.3%
```

The severe source multi-instance problem is fixed. Remaining ambiguity is mostly target-side or benign multi-relation location facts, for example a unique microwave can be both `inside kitchen` and `on kitchen_counter`.

Tradeoff:

```text
source_unique made train scene balance uneven:
scene_0: 71 episodes
scene_1: 70 episodes
scene_2: 67 episodes
scene_3: 70 episodes
scene_4: 70 episodes
scene_5: 36 episodes
```

This happened because one seen domain has fewer legal source-unique slots. The sampler now treats scene balance as a soft objective rather than a hard cap. If equal WM train size is mandatory, use more variants per domain or lower the total train budget.

## Why It Was Slow

The expensive step is EvolvingGraph replay. For each candidate task, the builder may test it across many init-graph variants. With 20 scene domains and 12 variants per domain, that is 240 graphs. Pair tasks (`puton`, `placein`) dominate runtime because many object pairs must be checked for executability and semantic validity.

The first source-unique attempt was slower than necessary because the semantic filter ran after EvolvingGraph execution. That has been fixed: the semantic gate now runs before replay.

Further speedups if needed:

1. parallelize task/variant execution;
2. persist a slot cache keyed by `(task_id, variant_key, semantic_gate)`;
3. precompute class counts per variant;
4. write intermediate gap/pool reports during construction instead of only at the end.

