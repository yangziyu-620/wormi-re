# VirtualHome Source-Unique Data Quality Analysis - 2026-05-28

Dataset:

```text
/root/autodl-tmp/wormi-data/virtualhome-balanced-source-unique-20260528
```

## Summary

The source-side multi-instance ambiguity that made the previous balanced dataset semantically weak has been fixed at construction time:

```text
source class has multiple graph nodes: 0 / 1023
source class appears in multiple rooms: 0 / 1023
```

The dataset is structurally valid:

```text
episodes: 1023
rows: 5263
train/test overlap: 0
replay failures: 0
obs mismatches: 0
next obs mismatches: 0
goal failures: 0
```

However, several semantic data-quality risks remain.

## Remaining Issues

### 1. Target-Side Ambiguity

The source object is now unique, but targets can still be ambiguous in class-level observations:

```text
target class has multiple graph nodes: 251 / 1023 = 24.5%
target class appears in multiple rooms: 227 / 1023 = 22.2%
target class has multiple observed locations/relations: 607 / 1023 = 59.3%
```

Examples:

```text
Put phone on chair
target chair appears in bedroom and livingroom

Place phone in dresser
target dresser appears in bedroom and livingroom

Place phone in sink
target sink appears in bathroom and kitchen
```

This affects rollout because the model emits class-level actions such as `put phone chair`, while the environment must bind `chair` to an instance. If the evaluator binds a different instance than the expert, replay success can diverge from training supervision.

### 2. Object Diversity Bias

The source-unique gate made the selected tasks much cleaner, but also biased them toward objects that are often unique. Seen tasks are especially narrow:

```text
seen source objects:
phone: 13 / 16 tasks
bathroom_cabinet: 1
microwave: 1
toaster: 1
```

Seen move tasks are mostly phone tasks:

```text
puton: phone -> {bathroom_cabinet, bathroom_counter, bed, bookshelf, chair, couch}
placein: phone -> {coffe_maker, dresser, freezer, microwave, sink, toaster, toilet}
```

This can make the world models learn a `phone`-centric transition distribution rather than general object manipulation.

### 3. Commonsense-Weak Goals

A heuristic audit flags many simulator-legal but semantically unnatural goals:

```text
weak episodes: 455 / 1023 = 44.5%
electronics_into_container: 315
electronics_to_bathroom_surface: 101
nonfood_into_appliance: 242
object_into_plumbing: 112
```

Examples:

```text
Place phone in toilet
Place phone in toaster
Place phone in freezer
Place keyboard in oven
Put phone on bathroom counter
```

These are executable in VirtualHome but weak as embodied commonsense tasks.

### 4. World-Model Train Imbalance

The source-unique gate reduced legal examples in one seen domain:

```text
scene_0: 71 train episodes
scene_1: 70
scene_2: 67
scene_3: 70
scene_4: 70
scene_5: 36
```

This is better semantically than the previous multi-instance data, but it makes `scene_5` a weaker world model.

### 5. Prompt / Label Protocol Mismatch

The current VirtualHome prompt says:

```text
switch [object]
putin [target object]
put [target object]
```

The data labels use:

```text
switchon object
putin source target
put source target
```

This must be fixed before training, otherwise the prompt describes a different action grammar from the supervised labels.

### 6. Template and Exact Sequence Repetition

Validator still reports exact action-sequence overlap:

```text
same-task exact action sequence overlap: 44
```

This is not train/test trajectory leakage, but it means many tasks reduce to the same high-level template. Offline exact-match evaluation may overestimate behavior cloning if the model memorizes common templates.

## Construction-Time Fixes

These issues should be handled during construction, not after training:

1. Add `target_room_unique` or `target_unique` gates.
   - `target_room_unique`: target may have hierarchical relations, but must not appear in multiple rooms.
   - `target_unique`: target class must have exactly one graph node.
   - The first is less destructive; the second is cleaner but may reduce data volume.

2. Add object diversity quotas.
   - Cap max seen tasks per source object, e.g. `phone <= 4`.
   - Require at least several source object classes in seen move tasks.

3. Add commonsense affordance filters.
   - Do not put electronics into toilet/sink/freezer/oven/toaster/coffe_maker.
   - Do not put arbitrary non-food objects into kitchen appliances.
   - Keep simulator-legal but odd tasks only in an explicit stress-test split, not main training.

4. Increase variants per domain or lower train target if strict gates reduce pool size.
   - Current 12 variants/domain keeps 1023 total but makes one WM sparse.
   - More variants can restore balance at the cost of build time and larger scene cache.

5. Fix the prompt before training.
   - Prompt must say `switchon [object]`, `put [object] [target]`, `putin [object] [target]`.

## Recommended Next Dataset Variant

Build a new variant rather than training the current one as final:

```text
virtualhome-balanced-semantic-v2
```

Suggested gates:

```text
semantic_gate = source_unique + target_room_unique
object_diversity_cap = max 4 seen tasks per source object
commonsense_filter = on
variants_per_domain = 16 or 20
```

If this cannot keep 1023 episodes, report the deficit explicitly instead of relaxing silently.
