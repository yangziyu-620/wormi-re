# WorMI VH Stage-1 World-Model Training — Launch + Status
**Date:** 2026-05-31  
**Dataset:** virtualhome-realtasks-v3-20260530  
**Branch:** tmow60-noUnity-data-pipeline

---

## Training Launch — CONFIRMED RUNNING

**Launched:** 2026-05-31T01:30:22+08:00  
**RUN_ID:** `realtasks-v3-20260531_013022`  
**PID:** 923583 (`.venv/bin/python3 .venv/bin/wormi world train`)  
**nohup logfile:** `/root/autodl-tmp/wormi-logs/stage1-realtasks-v3.out`  
**Per-scene train.log:** `/root/autodl-tmp/wormi-logs/vh-world-realtasks-v3-20260531_013022/train.log`  
**Dataset root used:** `/root/autodl-tmp/wormi-data/virtualhome-realtasks-v3-20260530`  
**Base model:** `unsloth/Llama-3.2-1B-Instruct`  
**Checkpoint root:** `/root/autodl-tmp/wormi-checkpoints/world-vh`

### Exact launch command

```bash
cd /root/WorMI && DATA_ROOT=/root/autodl-tmp/wormi-data/virtualhome-realtasks-v3-20260530 RUN_ID=realtasks-v3-20260531_013022 nohup bash sh/wormi-train-vh-world.sh > /root/autodl-tmp/wormi-logs/stage1-realtasks-v3.out 2>&1 & disown
```

### First 11 lines of nohup log (startup banner)

```
== WorMI VH world-model stage-1 training ==
root:       /root/WorMI
curricula:  tools/world_curricula_vh.py
data root:  /root/autodl-tmp/wormi-data/virtualhome-realtasks-v3-20260530
data disk:  /root/autodl-tmp
ckpt root:  /root/autodl-tmp/wormi-checkpoints/world-vh
hf home:    /root/autodl-tmp/hf-home
uv cache:   /root/autodl-tmp/uv-cache
log file:   /root/autodl-tmp/wormi-logs/vh-world-realtasks-v3-20260531_013022/train.log
started:    2026-05-31T01:30:22+08:00
+ .venv/bin/wormi world train --curricula_path tools/world_curricula_vh.py
```

### Training progress at confirmation time (~2 min post-launch)

- **scene_0** actively stepping: step 180+/2000 (~9%)
- No errors, no OOM, no traceback
- Loss (from TensorBoard events, scene_0):

| step | loss |
|------|------|
| 20   | 0.7838 |
| 40   | 0.5854 |
| 60   | 0.5772 |
| 80   | 0.5161 |
| 100  | 0.5122 |
| 120  | 0.3680 |
| 140  | 0.5235 |
| 160  | 0.3501 |
| 180  | 0.3119 |

First progress line: `  0%|          | 1/2000 [00:01<38:36,  1.16s/it]`

---

## Preflight Summary

**Result: GO** — all hard checks passed.

---

## Check 1 — DATA ROOT

**PASS**

`/root/autodl-tmp/wormi-data/virtualhome-realtasks-v3-20260530` contains `scene_0` through `scene_5`, each with both `train.jsonl` and `test.jsonl`. No missing or empty files.

Curricula loaded successfully with `WORMI_VH_DATA_ROOT=/root/autodl-tmp/wormi-data/virtualhome-realtasks-v3-20260530`:

- 6 curricula resolved (scene_0 .. scene_5)
- `dataset` paths resolve to `…/virtualhome-realtasks-v3-20260530/scene_*/`
- Output dir: `/root/autodl-tmp/wormi-checkpoints/world-vh`

**Launcher default mismatch (non-blocking, must override):** `sh/wormi-train-vh-world.sh` defaults `DATA_ROOT` to `$DATA_DISK/wormi-data/virtualhome` (wrong). Must invoke with:

```bash
DATA_ROOT=/root/autodl-tmp/wormi-data/virtualhome-realtasks-v3-20260530 \
  bash sh/wormi-train-vh-world.sh
```

This sets `WORMI_VH_DATA_ROOT` inside the script to the correct path.

---

## Check 2 — LOADER

**PASS**

`AutoJsonlDataset.load` dispatched to `VirtualHomeDataset` for `scene_0/train.jsonl`.

- 504 rows loaded, schema: `{instruction, observation, action, next_observation, _meta, auxiliary_task}`
- `as_chat(tokenizer)` produced a `Dataset` with correct Llama-3 chat-template `text` column (header `<|start_header_id|>assistant<|end_header_id|>` visible in output).

---

## Check 3 — BASE MODEL

**PASS** (downloaded during preflight)

`unsloth/Llama-3.2-1B-Instruct` was not present in `/root/autodl-tmp/hf-home/hub/` (only tokenizer files existed in the default `~/.cache` location). `snapshot_download` succeeded over the public network and wrote all 9 files to:

```
/root/autodl-tmp/hf-home/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/5a8abab4a5d6f164389b1079fb721cfab8d7126c
```

Key files confirmed:
- `model.safetensors` — 2.4 GB (weights present and complete)
- `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`, `special_tokens_map.json`, `README.md`

`sh/wormi-train-vh-world.sh` exports `HF_HOME=$DATA_DISK/hf-home` = `/root/autodl-tmp/hf-home`, so the training run will find the weights without re-downloading.

---

## Check 4 — DISK

**PASS**

```
Filesystem      Size  Used  Avail  Use%  Mounted on
/dev/md0         150G   42G   109G   28%  /root/autodl-tmp
```

109 GB free. Six world-model checkpoints (1B-param SFT, ~2.4 GB each, saved at steps 1000 + 2000) ≈ 29 GB total. No disk pressure.

---

## Check 5 — GPU

**PASS**

```
GPU 0: NVIDIA GeForce RTX 4090  49140 MiB total  1 MiB used  GPU-Util 0%
```

4090 is idle and free. 49 GB VRAM available; a 1B-model SFT run at batch=2 / seq≈4096 fits comfortably.

---

## Launch Command

```bash
cd /root/WorMI
DATA_ROOT=/root/autodl-tmp/wormi-data/virtualhome-realtasks-v3-20260530 \
  HF_HOME=/root/autodl-tmp/hf-home \
  bash sh/wormi-train-vh-world.sh
```

Or equivalently (all env vars explicit):

```bash
cd /root/WorMI
DATA_ROOT=/root/autodl-tmp/wormi-data/virtualhome-realtasks-v3-20260530 \
  WORMI_VH_DATA_ROOT=/root/autodl-tmp/wormi-data/virtualhome-realtasks-v3-20260530 \
  HF_HOME=/root/autodl-tmp/hf-home \
  bash sh/wormi-train-vh-world.sh
```

Expected runtime: ~2 h per scene × 6 scenes = ~12 h total (single-GPU sequential).

---

## Per-Check Summary Table

| Check | Result | Notes |
|---|---|---|
| Data root + scene dirs | PASS | scene_0..scene_5 each have train.jsonl + test.jsonl |
| Curricula load | PASS | 6 entries, paths resolve to v3 root |
| Dataset loader (VirtualHomeDataset) | PASS | 504 rows, as_chat OK |
| Base model weights | PASS | 2.4 GB safetensors downloaded to /root/autodl-tmp/hf-home |
| Disk free | PASS | 109 GB free on /root/autodl-tmp |
| GPU | PASS | RTX 4090, 49 GB VRAM, idle |
