# VirtualHome Data Construction Notes: House Configuration Axis

Date: 2026-05-22

## Core Understanding

The VirtualHome data should be organized around **house configurations** as the domain axis.

The intended structure is not:

- one expert per action family, such as `open`, `turnon`, `puton`, `placein`
- one expert per room type, such as kitchen, bedroom, bathroom, living room

The intended structure is:

- one world model / expert domain per **house configuration**
- each house configuration contains multiple rooms
- those rooms differ in layout, object placement, receptacles, object states, and graph connectivity

So the six VirtualHome world models should correspond to six seen house configurations:

```text
world_model_0 = seen house configuration 0
world_model_1 = seen house configuration 1
world_model_2 = seen house configuration 2
world_model_3 = seen house configuration 3
world_model_4 = seen house configuration 4
world_model_5 = seen house configuration 5
```

Each world model should see multiple seen-task expert episodes inside its own house configuration.

## What Is Correct In The Current Pipeline

The current directory/model split is mostly on the right axis:

- `scene_0 ... scene_5` represent six seen VirtualHome scene graphs.
- Stage 1 trains one world model per seen scene.
- Stage 2 retrieves/implants world models by scene-domain prototypes.
- The split conceptually matches:
  - seen task + seen house configuration
  - seen task + unseen house configuration
  - unseen task + unseen house configuration

This means the high-level grouping is closer to the paper than an action-family split would be.

## Main Problem

The problem is not mainly the folder split. The problem is that the current expert trajectory generator is too template-like.

Current expert programs are mostly fixed shortest templates:

```text
Open X:
  walk X -> open X

Turn on X:
  walk X -> switchon X

Put A on B:
  walk A -> grab A -> walk B -> put A B

Place A in B:
  walk A -> grab A -> walk B -> open B -> putin A B
```

These trajectories are executable in EvolvingGraph, but they do not use enough information from the house configuration.

As a result:

- The action sequence is mostly determined by the task family
- Same task usually has the same expert action sequence across scenes
- seen-seen test can become nearly memorised
- `col_1_seen_seen` may reach 100% for the wrong reason
- average trajectory length is around 2-5 high-level actions, much shorter than the paper-like rollout PS scale

## Desired Data Semantics

Each expert episode should be a function of:

```text
(house_configuration_id, init_state_variant, instruction) -> action trajectory
```

The important missing component is `init_state_variant`.

A single house configuration should provide multiple initial states, for example:

- object starts in different rooms or containers
- target receptacle is open or closed
- source container may need to be opened first
- agent may start in different rooms
- distractor objects may exist
- same instruction may require different intermediate actions in different graph states

This makes the expert depend on the concrete house configuration, not only on the instruction template.

## Better Paper-Like Split

The final split should remain:

```text
col_1: seen task + seen house configuration
col_2: seen task + unseen house configuration
col_3: unseen task + unseen house configuration
```

But train/test should avoid sharing full action templates too directly.

For example, for the same instruction:

```text
Place cup in cabinet
```

Different house configurations or initial states should produce different valid trajectories:

```text
case A:
  walk cup -> grab cup -> walk cabinet -> open cabinet -> putin cup cabinet

case B:
  walk drawer -> open drawer -> grab cup -> walk cabinet -> putin cup cabinet

case C:
  walk kitchen -> walk cup -> grab cup -> walk cabinet -> open cabinet -> putin cup cabinet
```

The task is still the same, but the trajectory is no longer just a memorized action-family template.

## Implementation Direction

Preferred fix:

1. Keep six world models mapped to six seen house configurations.
2. Keep the paper task counts: 78 tasks, 16 seen / 62 unseen.
3. Replace fixed shortest expert templates with graph-aware expert generation.
4. Use multiple init graphs or state variants per house configuration.
5. Filter only trajectories that EvolvingGraph can replay and whose final goal is satisfied.
6. Target a more paper-like trajectory length distribution, roughly 7-10 successful high-level actions on average.
7. Ensure train/test do not share exact full action sequences for the same task whenever possible.
8. Keep current easy/template data as a debug split, not as the final Table 1 reproduction split.

## Practical Next Step

The next data builder should expose two modes:

```text
easy_debug:
  current atomic templates, fast sanity-check data

paper_like:
  house-configuration-aware expert trajectories with multiple state variants
```

The final reported reproduction should use `paper_like`.

The current `easy_debug` split is useful for debugging stage 1, stage 2, retrieval, and rollout plumbing, but it should not be treated as a faithful VirtualHome Table 1 reproduction.
