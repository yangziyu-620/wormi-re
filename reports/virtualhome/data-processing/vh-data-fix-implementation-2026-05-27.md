# VirtualHome Data Fix Implementation - 2026-05-27

Implemented code-level fixes for the VH data path. The currently running rollout was left untouched and still uses the old already-loaded data/checkpoint.

## Implemented

1. Room action canonicalization in `tools/build_virtualhome_dataset.py`.
   - Room action labels now use the same canonical names as observations.
   - `dining_room -> kitchen`, `home_office -> livingroom`, `living_room -> livingroom`.
   - VirtualHome script lines still use raw node ids/classes, so replay remains bound to the original graph.

2. Position-aware walk generation in `_paperlike_program`.
   - Removed blind `walk start_room`.
   - Removed the post-grab `walk source_room` detour unless movement is actually needed.
   - Skips consecutive duplicate walk labels.

3. World-model affordance auxiliary samples in `wormi/datasets/virtualhome.py`.
   - Stage2 behavior-cloning path is unchanged: `end_with_action=True` still yields one action sample per raw row.
   - Stage1 world-model path now yields three samples per raw row:
     - `behavior_cloning`: observation/instruction -> action
     - `dynamics`: observation/action -> next observation
     - `affordance`: observation -> feasible expert action

4. Stronger VH validation gates in `tools/validate_virtualhome_dataset.py`.
   - Error on raw room action labels (`walk dining_room`, `walk home_office`, `walk living_room`).
   - Error on consecutive duplicate actions.
   - Error when unchanged-walk rate exceeds 5%.
   - Loader smoke test now expects `world_samples = raw_rows * 3`.

## Verification

- `py_compile` passed for:
  - `tools/build_virtualhome_dataset.py`
  - `tools/validate_virtualhome_dataset.py`
  - `wormi/datasets/virtualhome.py`

- Loader smoke check on current `scene_0/train.jsonl`:
  - raw rows: 79
  - stage2 action samples: 79
  - stage1 world samples: 237 = 79 * 3
  - world sample counts: 79 behavior_cloning / 79 dynamics / 79 affordance

- Planner text probe over 234 generated programs:
  - raw room actions: 0
  - consecutive duplicates: 0

- EvolvingGraph execution probe over 40 generated trajectories:
  - successful trajectories: 40
  - raw room actions: 0
  - consecutive duplicates: 0
  - unchanged walk: 1 / 65 = 1.5%

## Next Required Step

Do not overwrite `/root/autodl-tmp/wormi-data/virtualhome` while the old rollout process is still running. Rebuild to a fresh root first, for example:

```bash
RUN_ID=paperlike-v2-fixed-20260527 \
OUTPUT_DIR=/root/autodl-tmp/wormi-data/virtualhome-paperlike-v2-fixed-20260527 \
bash sh/wormi-build-vh-data.sh
```

After validation passes, point stage1/stage2 curricula at the new data root and retrain world models before stage2.
