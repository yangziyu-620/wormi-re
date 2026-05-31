# WorMI Stage-2 Status — 2026-05-31

Living doc for the stage-2 (integrated meta-learning) training run.
Append a new section for each experiment / status update.

---

## Preflight — 2026-05-31

### Step 1 — Base model download

**Result: PASS**

`unsloth/Llama-3.2-3B-Instruct` downloaded to HF cache.

- Cache path: `/root/autodl-tmp/hf-home/hub/models--unsloth--Llama-3.2-3B-Instruct/snapshots/006f5dcd1393c3add266de40994ba96225e9689d`
- Download time: ~5 min 20 s (11 files fetched)
- `HF_HOME=/root/autodl-tmp/hf-home` (on data disk, not root quota)

### Step 2 — Stage-1 world model checkpoints

**Result: PASS — all 6 checkpoints present with weights**

| Scene | Path | Weight size |
|---|---|---|
| scene_0 | `/root/autodl-tmp/wormi-checkpoints/world-vh/scene_0/last` | 2.4 G |
| scene_1 | `/root/autodl-tmp/wormi-checkpoints/world-vh/scene_1/last` | 2.4 G |
| scene_2 | `/root/autodl-tmp/wormi-checkpoints/world-vh/scene_2/last` | 2.4 G |
| scene_3 | `/root/autodl-tmp/wormi-checkpoints/world-vh/scene_3/last` | 2.4 G |
| scene_4 | `/root/autodl-tmp/wormi-checkpoints/world-vh/scene_4/last` | 2.4 G |
| scene_5 | `/root/autodl-tmp/wormi-checkpoints/world-vh/scene_5/last` | 2.4 G |

Each `last/` directory contains `model.safetensors` (or equivalent weight file).
Intermediate checkpoints `checkpoint-1000` and `checkpoint-2000` also present per scene.

### Step 3 — Stage-2 curricula import

**Result: PASS — imports without error, all paths resolve to v3 / world-vh**

Command:
```
WORMI_VH_DATA_ROOT=/root/autodl-tmp/wormi-data/virtualhome-realtasks-v3-20260530 \
WORMI_WORLD_VH_OUTPUT_DIR=/root/autodl-tmp/wormi-checkpoints/world-vh \
.venv/bin/python -c "import tools.wormi_curricula_vh as c; ..."
```

Resolved curricula facts:

| Field | Value |
|---|---|
| `base_model` | `unsloth/Llama-3.2-3B-Instruct` |
| `connections` (base layers) | `[13, 27]` |
| `world connections` (per WM) | `[7, 15]` |
| `method` | `WorMIIntegrateMethod.WORLD_WISE_ATTENTION` |
| N world models | 6 (scene_0..scene_5 from world-vh/scene_*/last) |
| N datasets | 9 (scene_0..5 + eval_col_1/2/3) |
| train curricula | 6 (K=3 subsets cycling all 6 WMs) |
| test curricula | 3 (col_1_seen_seen, col_2_seen_unseen, col_3_unseen_unseen) |
| `meta_learning` | True |
| `meta_learning_rate` (β) | 0.1 |
| `num_iterations` (λ_M) | 8 |
| `output_dir` | `/root/autodl-tmp/wormi-checkpoints/wormi-vh` |
| `name` | `wormi-vh-n6` |

Dataset root resolves to: `/root/autodl-tmp/wormi-data/virtualhome-realtasks-v3-20260530`
All 9 dataset dirs confirmed present (scene_{0..5}: train.jsonl + test.jsonl; eval_col_{1,2,3}: test.jsonl).

### Step 4 — Disk and GPU

| Resource | Status |
|---|---|
| `/root/autodl-tmp` disk | 150 G total, 84 G used, **67 G free** (44%) |
| GPU | NVIDIA GeForce RTX 4090, 49140 MiB total, **48510 MiB free** (1 MiB used) |

GPU is almost fully free. VRAM budget note: 3B base + 6×1B frozen world models = ~21 GB model weights; adapter params ~small; training batch_size=1 + grad_accum=4 should fit within 48 GB per survey §3.7 (OOM forced batch_size=1 historically with bf16 + 8-head cross-attention).

### Known instability notes (from survey §3.7 — carry into training)

- Use **threaded meta-learning path** (`WORMI_ALLOW_UNSAFE_THREADED_META=1`) — sequential Reptile (seqmeta) gives SR≈0.
- Set **`WORMI_THREADED_META_USE_BETA=0`** (direct-mean aggregation, not β-weighted) — only known-stable config.
- Start with **`WORMI_VH_STAGE2_BATCH_SIZE=1`** — OOM observed at batch_size=4 on 48 GB card.
- Launch script: `sh/wormi-train-vh-wormi.sh` (foreground); `sh/wormi-train-vh-wormi-background.sh` (background).

### Preflight verdict

**ALL CHECKS PASS. Ready to launch stage-2 training.**

Blockers: none.

---

## Smoke Test (P5 Crash Gate) — 2026-05-31

### Config

| Env var | Value |
|---|---|
| `WORMI_VH_STAGE2_BATCH_SIZE` | 1 |
| `WORMI_VH_STAGE2_GRADIENT_ACCUMULATION_STEPS` | 4 |
| `WORMI_VH_STAGE2_INNER_STEPS` | 3 |
| `WORMI_VH_STAGE2_META_STEPS` | 1 |
| `WORMI_ALLOW_UNSAFE_THREADED_META` | 1 |
| `WORMI_SEQUENTIAL_META_LEARNING` | 0 |
| `WORMI_THREADED_META_USE_BETA` | 0 (direct-mean aggregation) |
| Output dir | `/root/autodl-tmp/wormi-checkpoints/wormi-vh-SMOKE` |

### Observations

- All 6 sub-trainers started and reached "ready" with no errors.
- All 6 sub-trainers completed 3 inner steps (max_steps=3) concurrently.
- 1 meta aggregation (num_iterations=1) completed without hang or crash.
- No `RuntimeError: ... inplace operation` / autograd conflict.
- No CUDA OOM.
- No futex deadlock or hang.
- `pytorch_model.bin` (adapter weights) written to output `last/` (961 MB — full adapters for 2 base connection layers x 8 heads).
- Exit code: 0.

### Training loss (per trainer, 3 steps)

| Trainer | train_loss | train_runtime |
|---|---|---|
| subset-0-1-2 | 1.683 | 24.8 s |
| subset-1-2-3 | 1.849 | 37.5 s |
| subset-2-3-4 | 1.979 | 28.0 s |
| subset-3-4-5 | 1.447 | 42.7 s |
| subset-4-5-0 | 1.616 | 49.7 s |
| subset-5-0-1 | 1.227 | 43.3 s |

### VRAM

| Measurement | Value |
|---|---|
| Model weights only (base 3B + 6x1B worlds, bf16) | **19.80 GB** |
| Estimated peak during training (+ adapter optimizer + activations bs=1) | ~22-26 GB |
| GPU total VRAM | 47.4 GB (RTX 4090) |
| Headroom | ~21-25 GB free — no OOM observed |

### Verdict

**PASS — P5 crash gate cleared.**

Threaded meta-learning path (`WORMI_ALLOW_UNSAFE_THREADED_META=1`, `WORMI_THREADED_META_USE_BETA=0`) completed 3 inner steps + 1 meta aggregation crash-free. Ready to launch full stage-2 training.

---

## Full Stage-2 Launch — 2026-05-31T11:02:16+08:00

### Launch command

```bash
RUN_ID="realtasks-v3-20260531-110216" \
  DATA_DISK=/root/autodl-tmp \
  DATA_ROOT=/root/autodl-tmp/wormi-data/virtualhome-realtasks-v3-20260530 \
  WORLD_CKPT_ROOT=/root/autodl-tmp/wormi-checkpoints/world-vh \
  CKPT_ROOT=/root/autodl-tmp/wormi-checkpoints/wormi-vh \
  WORMI_THREADED_META_USE_BETA=0 \
  WORMI_ALLOW_UNSAFE_THREADED_META=1 \
  WORMI_SEQUENTIAL_META_LEARNING=0 \
  WORMI_VH_STAGE2_BATCH_SIZE=1 \
  WORMI_VH_STAGE2_GRADIENT_ACCUMULATION_STEPS=4 \
  WORMI_VH_STAGE2_INNER_STEPS=30 \
  WORMI_VH_STAGE2_META_STEPS=8 \
  HF_HOME=/root/autodl-tmp/hf-home \
  bash sh/wormi-train-vh-wormi-background.sh
```

### Run metadata

| Field | Value |
|---|---|
| PID | **986217** |
| Run ID | `realtasks-v3-20260531-110216` |
| Launch time | 2026-05-31T11:02:16+08:00 |
| Logfile | `/root/autodl-tmp/wormi-logs/vh-wormi-realtasks-v3-20260531-110216/train.log` |
| Launch log | `/root/autodl-tmp/wormi-logs/vh-wormi-realtasks-v3-20260531-110216/launch.log` |
| PID file | `/root/autodl-tmp/wormi-logs/vh-wormi-realtasks-v3-20260531-110216/pid` |
| Output ckpt dir | `/root/autodl-tmp/wormi-checkpoints/wormi-vh/wormi-vh-n6/` |
| TensorBoard logs | `/root/autodl-tmp/wormi-checkpoints/wormi-vh/wormi-vh-n6/logs/subset-{0-1-2,...,5-0-1}/` |

### Config

| Env var | Value |
|---|---|
| `WORMI_THREADED_META_USE_BETA` | **0** (direct-mean aggregation — only known-stable config) |
| `WORMI_ALLOW_UNSAFE_THREADED_META` | 1 (threaded path) |
| `WORMI_SEQUENTIAL_META_LEARNING` | 0 (threaded) |
| `WORMI_VH_STAGE2_BATCH_SIZE` | **1** (OOM-safe per survey §3.7) |
| `WORMI_VH_STAGE2_GRADIENT_ACCUMULATION_STEPS` | 4 (effective batch=4) |
| `WORMI_VH_STAGE2_INNER_STEPS` (λ_I) | **30** (full paper value) |
| `WORMI_VH_STAGE2_META_STEPS` (λ_M) | **8** (full paper value) |
| Meta LR (β) | 0.1 (curricula default) |
| Data root | `/root/autodl-tmp/wormi-data/virtualhome-realtasks-v3-20260530` |
| World ckpts | `/root/autodl-tmp/wormi-checkpoints/world-vh/scene_{0..5}/last` |

### Progress confirmation (observed ~2.5 min after launch)

- All 6 sub-trainers initialized and reached "ready" state (logged: `Trainer 1..6 started/ready`, `All trainers ready`).
- Trainer 1: completed inner loop 1 (step 30/240, elapsed ~2:01 min).
- Trainer 2: completed inner loop 1 (step 30/240, elapsed ~2:08 min).
- Trainer 3: at step 15/240 and advancing (~1.8-2.1 s/it).
- Trainers 4-6: queued in threaded ring, waiting their turn.
- TensorBoard event files created for all 6 sub-trainers confirmed present.
- Process alive: `kill -0 986217` returns 0.
- No errors, no OOM, no Traceback in log.

### First progress log lines

```
🏃 Trainer 1 started
🏃 Trainer 1 ready
🏃 Trainer 2 started
🏃 Trainer 2 ready
🏃 Trainer 3 started
🏃 Trainer 3 ready
🏃 Trainer 4 started
🏃 Trainer 4 ready
🏃 Trainer 5 started
🏃 Trainer 5 ready
🏃 Trainer 6 started
🏃 Trainer 6 ready
🚦 All trainers ready
[tqdm] 12%| 30/240 [02:01<06:44, 1.92s/it]  ← Trainer 1 inner loop 1 complete (30 steps)
[tqdm] 12%| 30/240 [02:08<06:26, 1.84s/it]  ← Trainer 2 inner loop 1 complete (30 steps)
[tqdm]  6%| 15/240 [02:33<08:09, 2.18s/it]  ← Trainer 3 at step 15 and advancing
```

### Expected duration

6 trainers × λ_M=8 meta iterations × λ_I=30 inner steps × ~2 s/step ≈ 6-8 hours total wall-clock.

### Verdict

**RUNNING — training confirmed progressing. Do not wait for completion.**

---
