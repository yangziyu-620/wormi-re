# WorMI 数据 pipeline 复现指南

复现 WorMI (ICML 2025, arxiv 2509.03956) 训练数据的端到端步骤。**两套数据全部 CPU-only，不需要 Unity / THOR / GPU 渲染**。

## 0. 复现产出

| 数据集 | 输出位置 | 行数 | 严格对齐 paper 的部分 |
|---|---|---|---|
| VirtualHome | `$DATA_ROOT/virtualhome/{turnon,open,puton,placein}/{train,test_seen_task_unseen_scene,test_unseen_task_seen_scene,test_unseen_task_unseen_scene}.jsonl` | ~1.4 K | 78 atomic tasks (9+7+30+32, Table A.2) / 6 actions (Table A.1) / triple-list observation (Figure A.2) / 3-类 split 对应 paper Table 1 三栏 eval |
| ALFWorld | `$DATA_ROOT/alfworld/{kitchens,bedrooms,livingrooms,bathrooms}/{train,test}.jsonl` | ~3.5 K | 3,553 games (≈ paper 3,554) / 6 task types (Table A.4) / textual env native obs (Figure A.3) |

切分 paper 未明示部分：

- VH 16/62 seen/unseen task split：按家族比例分层（turnon=2 / open=1 / puton=6 / placein=7）保证每个 world model 都有 train 数据；非 stratified 的 uniform shuffle 会让 turnon/open 抽到 0 个 seen
- VH 6/14 seen/unseen scene split: 7 个 default scene 按 6/1 (paper 用作者自造的 20 scene)
- ALFWorld 4/2 seen/unseen task: unseen = `pick_two_obj_and_place`, `pick_clean_then_place_in_recep`（更正自 2026-05-13；早期选 `{heat, cool}` 会让 paper Table 1 col-3 物理上为空。完整推导见 [RUNBOOK.md §3](RUNBOOK.md#3-alfworld-unseen-task-resplit-)）
- ALFWorld 3/1 seen/unseen scene type: unseen = `bathrooms`

**已知偏离 paper 的硬约束**（源数据限制，无法在 converter 里修复）：

- 暂无。VH 7 base apartment 各有 ~800 个 program，每个带独立 init_graph（家具构成不同），对应 paper 的"scene 实例"概念。从 7 × 800 = 5600 个 variant 里采 20 个就能严格凑齐 paper 的 20-scene 池。
- ALFWorld prompt / 10-skill 动作词表（Figure A.5）属于 trainer 侧的 chat formatting，不在 jsonl 数据里。jsonl 存原始 textual env observation 和 expert action；prompt 套壳由 `wormi/datasets/alfworld.py::_convert_to_chat` 控制（当前实现没有 system prompt，需另外修）。

---

## 1. 前置环境

```bash
# 必备
- Python 3.10+ (3.12 OK)
- aria2c       # 多连接下载，否则 wget 也行
- git, unzip, curl

# 推荐变量
export SCRATCH=/srv/scratch/$USER         # 大空间持久存储
export DATA_ROOT=$SCRATCH/wormi-data      # 数据落盘根目录
export TMP=/tmp                            # 节点本地 tmp（不占 scratch quota）

mkdir -p $DATA_ROOT
```

### 1.1 inode quota 注意（HPC 用户）

VirtualHome 解压有 ~24,000 个小文件，alfworld 解压有 ~11,000 个。如果你的 scratch 有 inode quota（如 GPFS 默认 489K），**整体能装下但邻近上限**。本指南把 ALFWorld 解压定向到 `/tmp`（节点 local xfs，无 quota）。

---

## 2. VirtualHome 数据

### 2.1 下载静态 dataset zip（1.55 GB）

```bash
mkdir -p $DATA_ROOT/raw
cd $DATA_ROOT/raw

# 16 路并发下载
aria2c -x 16 -s 16 -k 1M \
  'http://virtual-home.org/release/programs/programs_processed_precond_nograb_morepreconds.zip'

# 解压（约 23 GB / 24 K 文件）
unzip -q programs_processed_precond_nograb_morepreconds.zip
```

数据布局：
```
programs_processed_precond_nograb_morepreconds/
├── withoutconds/{source}/{file}.txt          # 原始 program (instruction + actions)
├── initstate/{source}/{file}.json            # 每 program 的 init precondition
├── executable_programs/{scene}/{source}/{file}.txt  # 对齐 scene 的可执行 program
├── init_and_final_graphs/{scene}/{source}/{file}.json  # 每 program init/final graph
└── state_list/{scene}/{source}/{file}.json   # 每步执行后的 graph 序列（已预生成）
```

### 2.2 从 zip 流式抽 20 个 init_graph 变体到单文件

paper §4 写的"20 scene"指的是 (base apartment, init_graph variant) 池子里抽 20 个：每个 TrimmedTestSceneN 下都有 ~800 个 program 文件，每个带独立的 init_graph（家具种类、object 数量都不同）。我们按 3+3+3+3+3+3+2 = 20 在 7 个 base 上分摊采样，全程从 zip 流式读，**不解压**：

```bash
mkdir -p $DATA_ROOT/scene-inits
python3 <<'PY'
import json, zipfile, random
from pathlib import Path
ZIP = '$DATA_ROOT/raw/programs_processed_precond_nograb_morepreconds.zip'
PREFIX = 'programs_processed_precond_nograb_morepreconds/init_and_final_graphs/'
PER_BASE = [3, 3, 3, 3, 3, 3, 2]   # sums to 20
rng = random.Random(42)
out = {}
with zipfile.ZipFile(ZIP) as zf:
    by_base = {}
    for n in zf.namelist():
        if not (n.startswith(PREFIX) and n.endswith('.json')): continue
        base = n[len(PREFIX):].split('/', 1)[0]
        by_base.setdefault(base, []).append(n)
    for base, k in zip(sorted(by_base), PER_BASE):
        cands = sorted(by_base[base]); rng.shuffle(cands); picked = 0
        for c in cands:
            if picked >= k: break
            with zf.open(c) as f:
                d = json.load(f)
            ig = d.get('init_graph')
            if ig is None: continue
            out[f'{base}__v{picked}'] = ig; picked += 1
json.dump(out, open('$DATA_ROOT/scene-inits/init_graphs_20.json', 'w'))
print(f'wrote {len(out)} scene init_graphs')
PY
```

输出 `init_graphs_20.json` 几 MB，可以直接喂给 converter；**raw zip 不需要解压**，全程 inode 占用近零。

```bash
mkdir -p $DATA_ROOT/scene-inits

python3 <<PY
import json
from pathlib import Path
ROOT = '$DATA_ROOT/raw/programs_processed_precond_nograb_morepreconds'
out = {}
for sd in sorted(Path(ROOT, 'init_and_final_graphs').iterdir()):
    if not sd.is_dir(): continue
    found = None
    for src in sd.iterdir():
        if not src.is_dir(): continue
        for jf in src.glob('*.json'):
            try:
                d = json.load(open(jf))
                if 'init_graph' in d:
                    found = d['init_graph']; break
            except Exception: pass
        if found: break
    if found:
        out[sd.name] = found
        print(f"{sd.name}: {len(found['nodes'])} nodes")
json.dump(out, open('$DATA_ROOT/scene-inits/init_graphs.json', 'w'))
PY
```

### 2.3 Clone VirtualHome source（仅用 EvolvingGraph 子模块）

`pip install virtualhome` 在 Python 3.12 下因旧 setuptools API 失败，clone source 直接用：

```bash
git clone --depth 1 https://github.com/xavierpuigf/virtualhome.git $DATA_ROOT/virtualhome-src
```

### 2.4 跑 converter

```bash
# 推荐路径：用 §2.2 生成的 20-scene 缓存
python3 /path/to/wormi/tools/build_virtualhome_dataset.py \
  --scene-inits-json $DATA_ROOT/scene-inits/init_graphs_20.json \
  --vh-src $DATA_ROOT/virtualhome-src \
  --output-dir $DATA_ROOT/virtualhome \
  --seen-scenes 6 \
  --seen-instructions 16
```

约 5 分钟跑完（CPU only）。converter 的工作流：

1. 加载 7 scene 的 init graph
2. 从 `properties_data.json` 抽 4 类 candidate object（HAS_SWITCH / CAN_OPEN / SURFACES / CONTAINERS / GRABBABLE），按 **scene 覆盖率排序** 选 78 个 atomic instruction（按 paper Table A.2 比例 9/7/30/32）—— 覆盖率排序保证 (instruction, scene) 成功率高，否则随机抽到只在 1 个 scene 出现的 class 会让大部分 pair execution_failed
3. 跨 7 scene × 78 instruction = 546 (instruction, scene) pair，每个跑 EvolvingGraph atomic plan
4. 每个成功 trajectory 输出 N 行 jsonl（每 step 一行）
5. **3 类 split**：
   - `train.jsonl` = seen_task × seen_scene（paper Table 1 第 1 列）
   - `test_seen_task_unseen_scene.jsonl` = seen_task × unseen_scene（paper Table 1 第 2 列）
   - `test_unseen_task_unseen_scene.jsonl` = unseen_task × unseen_scene（paper Table 1 第 3 列）
   - `test_unseen_task_seen_scene.jsonl` = unseen_task × seen_scene（paper 没列，但保留方便消融）
6. **family stratified seen split**：16 个 seen task 按家族大小分层（turnon=2 / open=1 / puton=6 / placein=7），不再用全局随机 shuffle 否则 turnon/open 容易抽空

输出 schema:
```json
{
  "instruction": "Place plum in fridge",
  "observation": "(plum, inside, kitchen), (fridge, inside, kitchen), (character, hold, none), ...",
  "action": "walk plum",
  "next_observation": "(plum, inside, kitchen), ..., (character, close, plum)"
}
```

### 2.5 释放 inode（可选）

跑完 converter 后，原始 raw 目录的 24K 小文件可以删（jsonl 已经派生）：

```bash
rm -rf $DATA_ROOT/raw/programs_processed_precond_nograb_morepreconds
# 保留 zip 和 scene-inits/init_graphs.json 即可
```

---

## 3. ALFWorld 数据

### 3.1 创建隔离 venv（推荐放 /tmp）

`alfworld + torch` 装下来 ~25K inode，**强烈建议把 venv 落在 /tmp**（节点 local xfs，2M inode 无 quota），跑完 `rm -rf /tmp/alfworld-venv` 完全释放：

```bash
python3 -m venv /tmp/alfworld-venv
/tmp/alfworld-venv/bin/pip install -U pip wheel

# 装 alfworld（不带 [full]，visdom 在 Python 3.12 build 失败）
/tmp/alfworld-venv/bin/pip install alfworld

# alfworld import 时会拉 torch；CPU 版即可
/tmp/alfworld-venv/bin/pip install torch --no-deps --index-url https://download.pytorch.org/whl/cpu
```

### 3.2 下载 game files (~143 MB) 到 scratch

```bash
mkdir -p $DATA_ROOT/alfworld-data-zips
cd $DATA_ROOT/alfworld-data-zips

aria2c -x 16 -s 16 -k 1M -j 1 \
  'https://github.com/alfworld/alfworld/releases/download/0.4.0/json_2.1.2_tw-pddl.zip'
aria2c -x 16 -s 16 -k 1M -j 1 \
  'https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip'
# json_2.1.1_pddl.zip 是 ALFRED native PDDL，textual env 不需要，跳过
```

### 3.3 解压到 /tmp（不占 scratch inode quota）

```bash
mkdir -p $TMP/alfworld-data
cd $TMP/alfworld-data
unzip -q $DATA_ROOT/alfworld-data-zips/json_2.1.2_tw-pddl.zip
unzip -q $DATA_ROOT/alfworld-data-zips/json_2.1.1_json.zip

# 把 logic files 复制到 ALFWORLD_DATA root（converter 也能自动做这步）
$SCRATCH/alfworld-venv/bin/python <<PY
import os, shutil, alfworld.info as info
os.makedirs('$TMP/alfworld-data/logic', exist_ok=True)
shutil.copy(info.ALFRED_PDDL_PATH, '$TMP/alfworld-data/logic/alfred.pddl')
shutil.copy(info.ALFRED_TWL2_PATH, '$TMP/alfworld-data/logic/alfred.twl2')
PY
```

### 3.4 跑 converter

```bash
/tmp/alfworld-venv/bin/python /path/to/wormi/tools/build_alfworld_dataset.py \
  --alfworld-data /tmp/alfworld-data \
  --output-dir $DATA_ROOT/alfworld

# 强烈建议先 smoke 跑 20 个 game 验证 metadata 对齐：
/tmp/alfworld-venv/bin/python /path/to/wormi/tools/build_alfworld_dataset.py \
  --alfworld-data /tmp/alfworld-data \
  --output-dir /tmp/alfworld-smoke \
  --limit 20
# 跑完抽几行：每行 history[0].observation 里的 "find/clean/heat ..." 任务描述
# 必须和 trial_name 的 task_type 前缀一致；不一致就是 metadata 又错位
```

约 60-160 分钟跑完（取决于 CPU）。converter 的工作流：

1. 用 alfworld 的 `AlfredTWEnv` (textual mode) 加载 3,553 个 solvable game
2. **每次 `env.reset()` 后从 `infos["extra.gamefile"]` 读取 env 实际加载的 game** —— 不再相信循环索引（曾经 bug：循环 `continue` 跳过会让 env 指针和 metadata 错位）
3. 跟随 `infos["extra.expert_plan"]` 的 expert action 序列，每 step 记录 (obs, action, reward, dones, next_obs)
4. 按 ALFRED scene number 分桶：1-30=kitchens / 201-230=livingrooms / 301-330=bedrooms / 401-430=bathrooms
5. 切分：seen task × seen scene (kitchens/bedrooms/livingrooms) → train；其余 → test。**注意**：当前 builder 默认 unseen = `{pick_two_obj_and_place, pick_clean_then_place_in_recep}`（更正后），seen = `{look_at_obj, simple, heat, cool}`。若运行的是更早版本（unseen = `{heat, cool}`），需在 build 完成后跑 `tools/resplit_alfworld_by_unseen_task.py` 重新分桶；详见 [RUNBOOK.md §3](RUNBOOK.md#3-alfworld-unseen-task-resplit-)。
6. 流式写 jsonl，每 50 game 一条 progress
7. resume 安全：已存在的 jsonl 行会被 trial_name 索引，重启后跳过；env 也会跟着每次 reset 推进，不会出现 metadata 错位

输出 schema:
```json
{
  "task": "pick_and_place_simple",
  "trial_name": "trial_T20190907_174127_043461",
  "history": [
    {
      "observation": "-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room. ...\nYour task is to: ...",
      "action": "go to dresser 1",
      "reward": 0.0,
      "dones": false,
      "next_observation": "You arrive at dresser 1. On the dresser 1, you see ..."
    },
    ...
  ]
}
```

### 3.5 释放 /tmp（converter 跑完即可）

```bash
rm -rf /tmp/alfworld-data /tmp/alfworld-venv
```

跑完 scratch 上只多了几 MB jsonl；`/tmp` 释放后整套流程在 scratch 上的 inode 净增 ≈ 0。

---

## 4. 验证数据格式

```bash
# VH schema
head -1 $DATA_ROOT/virtualhome/turnon/train.jsonl | python3 -m json.tool | head -10

# ALFWorld schema
head -1 $DATA_ROOT/alfworld/bedrooms/train.jsonl | python3 -m json.tool | head -10
```

应能看到 paper 对应字段：
- VH: `instruction`, `observation`, `action`, `next_observation` (4 个 string fields)
- ALFWorld: `task`, `trial_name`, `history` (list of `{observation, action, reward, dones, next_observation}`)

---

## 5. 已知问题与对策

### 5.1 `pip install virtualhome` 失败

> `ModuleNotFoundError: No module named 'setuptools.extern.six'`

原因: pypi 的 virtualhome 包用旧 setuptools API。**对策**: 不装包，直接 `git clone` 仓库后用 `tools/build_virtualhome_dataset.py` 里的 `_bootstrap_evolving_graph()` 加载子模块（绕开 `simulation/__init__.py` 的 cv2/ipdb import）。

### 5.2 `pip install alfworld[full]` 失败

> `ModuleNotFoundError: No module named 'pkg_resources'` (visdom)

**对策**: 装基础版 `pip install alfworld`，textual env 不需要 visdom。

### 5.3 inode quota 爆 (`Disk quota exceeded`)

GPFS 默认 489K inode quota，VH 解压 + alfworld-venv + torch 累计接近上限。**对策**:

- ALFWorld 数据**解压到 `/tmp`**（节点 local xfs，2M inodes 无 quota）
- VH 解压完取出 7 scene init graph 到单文件后，**rm -rf 整个解压目录**释放 24K inodes
- alfworld 转换跑完后**整体删除 alfworld-venv**释放 ~25K inodes

### 5.4 ALFWorld converter 慢（每 game 1-3s）

平均每 game expert plan ~30 step × 100ms。优化已在 converter:
- `max_nb_steps_per_episode=30` (与 wormi loader `if len(history) > 30: break` 对齐)
- 流式写 jsonl，可断点续看

### 5.5 ALFWorld textual env 报 `KeyError: 'dagger'` / `'domain_randomization'`

base_config.yaml 里嵌套深层 keys 必须给。converter 已 inline 提供 mock dict（见 `build_config()`）。

---

## 6. 复现规模 vs paper

| 维度 | 我们的复现 | paper |
|---|---|---|
| VH (instruction × scene) rollouts | ~1096 (= 4430 jsonl rows / ~4 step per ep) | 1023 episodes |
| VH unique instructions in jsonl | 70 / 78（剩 8 个 instruction EvolvingGraph 在所有 scene 都 execution_failed，前置条件不可达） | 78 |
| VH instruction pool | 78 (严格对齐 Table A.2: 9+7+30+32) | 78 |
| VH scenes | 20 (= 7 base apartment × 多 init_graph variant，按 3+3+3+3+3+3+2 采样) | 20 |
| VH room set | `{bathroom, bedroom, kitchen, livingroom}` | `{livingroom, bathroom, kitchen, bedroom}` |
| ALFWorld episodes | 3,553 | 3,554 |
| ALFWorld task types | 6 (严格对齐 Table A.4) | 6 |
| ALFWorld scene types | 4 (严格对齐) | 4 |

**剩余差异（不在数据 pipeline 范围）**:
- **ALFWorld prompt / 10-skill 动作词表 (Figure A.5)**：jsonl 里存原始 textual env observation/action，10-skill 限制和 system prompt 是 trainer 侧 (`wormi/datasets/alfworld.py::_convert_to_chat`) 的 chat formatting 决定的。当前 chat 格式有偏离，需另外修。
- **N=6 world model / K=3 retrieval (Table A.6)**：是 curricula 编排问题，不在 jsonl 数据层。VH 4 个 task family + ALFWorld 4 个 room = 8 桶；paper N=6 怎么聚合需要看 §A.6 后再决定。

**严格对齐部分**: VH 78 instruction (9+7+30+32) / 6 action verb / triple-list obs 格式；ALFWorld 6 task type / 4 scene / textual env native obs / 3,553 game (≈ paper 3,554)；3 类 split 文件支持 paper Table 1 三栏 eval。
