# WorMI 端到端复现 Runbook

> 复现 WorMI (ICML 2025, [arxiv:2509.03956](https://arxiv.org/html/2509.03956)) 的完整流程：从原始数据到 paper Table 1 数字。
>
> **范围**：HPC (Katana) 上的 PBS 工作流。本机/laptop 跑也行，把 qsub 换成直接执行 sh 脚本即可。
>
> **配套**：
> - [REPRODUCE_DATA.md](REPRODUCE_DATA.md) — 数据 build 阶段的详细参考（下载、解压、converter 内部逻辑）
> - [../Readme.md](../Readme.md) — wormi CLI 的入口和 curricula 写法
> - [../CLAUDE.md](../CLAUDE.md) — 代码架构（model / trainer / dataset 内部）

## 0. 总览

```
                          ┌─────────────────┐
            VirtualHome   │  raw zip files  │   ALFWorld
                          └────────┬────────┘
                                   │ §1 env + §2 build
                                   ▼
                          ┌─────────────────┐
                          │  per-room jsonl │
                          └────────┬────────┘
                                   │ §3 resplit (ALFWorld only)
                                   ▼
                          ┌─────────────────┐
                          │ paper-aligned   │
                          │ train/test jsonl│
                          └────────┬────────┘
                                   │ §4 wormi world train
                                   ▼
                          ┌─────────────────┐
                          │  4 VH +  3 AFW  │   ← frozen Llama-3.2-1B
                          │  world models   │     world models
                          └────────┬────────┘
                                   │ §5 wormi world eval
                                   ▼
                              paper Table 1
                          ┌─────────────────┐
                          │  wormi train    │   ← §6 integration
                          │  (N=6 / K=3)    │     TODO
                          └────────┬────────┘
                                   │ §7 wormi eval
                                   ▼
                              paper Table A.6
```

### Wall-clock 预算（每阶段）

| 阶段 | 数据集 | 资源 | 实测 walltime |
|---|---|---|---|
| §2 build | VH | 4 cpu / 16 gb | ~5 min |
| §2 build | ALFWorld | 4 cpu / 16 gb | ~2.5 h |
| §3 resplit | ALFWorld | 2 cpu / 8 gb | < 1 min |
| §4 train | VH (4 family) | 1× L40S / 8 cpu / 64 gb | ~45 min |
| §4 train | ALFWorld (3 room) | 1× L40S / 8 cpu / 64 gb | ~6 h 预期 |
| §5 eval | VH (4 fam × 4 split = 16) | 1× L40S | ~2-4 h |
| §5 eval | ALFWorld (3 fam × 3-4 split) | 1× L40S | ~1-3 h 预期 |

### Checkpoint 与数据布局

```
$SCRATCH/wormi-data/                                        ← 数据 root
├── raw/                                                    ← 原始下载（VH zip 1.55 GB + ALFWorld zip ~150 MB）
├── virtualhome-src/                                        ← VH git clone
├── scene-inits/init_graphs.json                            ← 7 base scene init graph 缓存
├── virtualhome/<family>/{train,test,test_*}.jsonl          ← VH 处理后
├── alfworld-data/                                          ← ALFWorld zip 落地
├── alfworld/<room>/{train,test,test_unseen_task}.jsonl     ← ALFWorld 处理后 (resplit 后状态)
├── cl-alfred-splits/                                       ← CL-ALFRED manifest (参考用)
└── logs/<stage>/<job_id>.log                               ← 每个 PBS job 的日志

$SCRATCH/wormi-checkpoints/
├── world-vh/<family>/last/                                 ← VH world model (4×)
└── world-alfworld/<room>/last/                             ← ALFWorld world model (3×)

$SCRATCH/wormi-eval/
├── world-vh/<family>/<split>.jsonl + summary_<ts>.tsv      ← VH eval 结果
└── world-alfworld/<room>/<split>.jsonl + summary_<ts>.tsv  ← ALFWorld eval 结果

$SCRATCH/wormi-venv/                                        ← uv 持久 venv（避免每 job 重装 torch）

$HOME/sh/wormi-*.sh                                         ← 所有 PBS job 脚本
```

---

## 1. 环境准备（一次性）

### 1.1 软件栈

| 工具 | 版本要求 | 备注 |
|---|---|---|
| Python | ≥ 3.12 | `pyproject.toml` 声明，必须 |
| uv | 任意近期版本 | 装在 `~/.local/bin/`，PBS 通过 `$PATH` 继承 |
| CUDA | 12.4+ | L40S 推荐 12.8 |
| `transformers` | **必须 `==4.45.2`** | model code 调内部 API |
| `trl` | **必须 `==0.11.3`** | SFTTrainer signature 锁定 |
| `torch` | `2.5.1` | 通过 pyproject lock |

```bash
# 在 compute node 上做，不要在 login node
export UV_PROJECT_ENVIRONMENT=$SCRATCH/wormi-venv
export UV_CACHE_DIR=$SCRATCH/.cache/uv
cd ~/WorMI
uv sync         # 几分钟，安装 torch 等
uv run wormi --help
```

### 1.2 PBS 环境变量（每个 sh 脚本头都有）

| 变量 | 作用 | 必须？ |
|---|---|---|
| `UV_PROJECT_ENVIRONMENT=$SCRATCH/wormi-venv` | venv 落 scratch，不挤 home quota | ✅ |
| `UV_LINK_MODE=copy` | 跨 fs hardlink 不可靠 | ✅ |
| `HF_HOME=$SCRATCH/.cache/huggingface` | 大模型缓存离开 home | ✅ |
| `HF_HUB_ENABLE_HF_TRANSFER=0` | Katana 某些 node 上不到 HF CDN | ✅ |
| `WANDB_DISABLED=true` | 离线节点 wandb 会卡 | ✅ |

### 1.3 inode quota 注意

GPFS 默认 ~489K inode。VH 解压含 24K 小文件，alfworld+torch venv ~25K。**对策**：

- VH zip **不解压**（§2.1 全程流式读 zip）
- ALFWorld 解压到节点 `/tmp`（local xfs，2M inode 无 quota），跑完 rm
- alfworld 库装在节点 `/tmp/alfworld-venv` 临时 venv，跑完 rm

`wormi-build-*.sh` 已默认这样做。

---

## 2. 数据 build

详细 build 逻辑（converter 内部算法、76 instruction 选取规则、ALFRED scene→room 映射等）见 [REPRODUCE_DATA.md](REPRODUCE_DATA.md)。这里只列 PBS 入口。

### 2.1 VirtualHome

```bash
qsub $HOME/sh/wormi-build-vh-dataset.sh
```

完成后应该看到：

```
$DATA_ROOT/virtualhome/{turnon,open,puton,placein}/
├── train.jsonl                                # seen task × seen scene
├── test.jsonl                                 # = train.jsonl 的镜像 alias（未来可删）
├── test_seen_task_unseen_scene.jsonl          # paper Table 1 col-2 数据源
├── test_unseen_task_seen_scene.jsonl          # 消融用
└── test_unseen_task_unseen_scene.jsonl        # paper Table 1 col-3 数据源
```

VH 的 `train.jsonl` 同时是 trainer 训练源 + paper Table 1 col-1 评估源（in-distribution holdout）；trainer 会自动按 SFT split 内部切。

### 2.2 ALFWorld

```bash
qsub $HOME/sh/wormi-build-alfworld-dataset.sh
```

完成后会 cascade 调用 `tools/split_alfworld_train_test.py` 切 10% in-room holdout（仅 seen scene 三 room）。**注意**：这一步出来的 `<room>/test_unseen_task.jsonl` 含义是"该 room 内被 builder 当 unseen task 的 trial"，会被下一步 §3 resplit 整体改写，**不要直接拿来 eval**。

---

## 3. ALFWorld unseen-task resplit ⚠️

这一节是这次 session 的最大发现。

### 3.1 Trap：builder 默认 unseen task 选错

`build_alfworld_dataset.py` 原默认 `UNSEEN_TASK_TYPES = {pick_heat, pick_cool}`（注释理由 "longest expert plans → toughest generalization probe"）。**这是错的**，原因：

- heat 需 microwave，cool 需 fridge，物理上**只能在 kitchen scene 出现**
- bathrooms 是 unseen scene → bathrooms 永远没有 heat/cool trial
- paper Table 1 column 3 = "Unseen task × Unseen scene" = `unseen_task ∩ bathrooms` = **∅**
- 与 paper 报告 col-3 数字直接矛盾

### 3.2 Why：paper "Following CL-ALFRED" 是 misleading

paper §4 写 "Following CL-ALFRED benchmark settings, ... 6 task types (4 seen, 2 unseen)"，但查证 CL-ALFRED 原论文（Kim et al. ICLR'24, [arxiv:2403.07548](https://arxiv.org/html/2403.07548v1)）：

> "an agent sequentially receives N_j training episodes... for each type of behavior"
> "The same trained models are evaluated in both *seen* and *unseen* environments."

CL-ALFRED 的 seen/unseen 是 **scene 维度**（ALFRED 原生 valid_seen/valid_unseen），**task 维度只有 sequential learning 顺序，没有 seen/unseen 切分**。WorMI 的 "4 seen, 2 unseen task" 是 paper 作者自加，**不指名**，复现者必须独立决定。

### 3.3 推导：unseen pair 的唯一可行解

| task | bathrooms 可行? | 4-room 总数 |
|---|---|---|
| pick_and_place_simple | ✓ | 264 / 237 / 79 / 210 |
| pick_two_obj_and_place | ✓ | 233 / 240 / 124 / 216 |
| pick_clean_then_place_in_recep | ✓ (bath 有 sink) | 149 / 0 / 501 / 0 |
| look_at_obj_in_light | ✗ (无 lamp) | 0 / 257 / 0 / 51 |
| pick_heat_then_place_in_recep | ✗ (无 microwave) | 0 / 0 / 459 / 0 |
| pick_cool_then_place_in_recep | ✗ (无 fridge) | 0 / 0 / 533 / 0 |

unseen pair 候选只能从 `{simple, two_obj, clean}` 选 2 个，C(3,2)=3 种：

| 候选 | bath col-3 | living train | seen task 集合 | 评价 |
|---|---|---|---|---|
| {simple, two_obj} | 497 | **51** | look/clean/heat/cool | ❌ living 训不动 |
| {simple, clean} | 413 | 267 | look/two_obj/heat/cool | ⚠️ 把 atomic 当 unseen，倒置 |
| **{two_obj, clean}** | 382 | 261 | look/simple/heat/cool | ✅ compositional unseen + atomic seen baseline |

→ 选 **`{pick_two_obj_and_place, pick_clean_then_place_in_recep}`**

### 3.4 执行（不重跑 alfworld）

trial 数据已收集；只需按新规则**重新分桶**。`tools/resplit_alfworld_by_unseen_task.py` 干这事：

```bash
qsub $HOME/sh/wormi-resplit-alfworld.sh
```

CPU job < 1 分钟。完成后：

```
$DATA_ROOT/alfworld/
├── bedrooms/
│   ├── train.jsonl          (445)   ← seen task × bedrooms, 10% holdout 后
│   ├── test.jsonl           ( 49)   ← in-room holdout
│   └── test_unseen_task.jsonl (240) ← bedrooms 里的 unseen task (two_obj)
├── kitchens/
│   ├── train.jsonl          (964)   ← look/simple/heat/cool × kitchens
│   ├── test.jsonl           (107)
│   └── test_unseen_task.jsonl (625) ← kitchens 里的 unseen task (two_obj + clean)
├── livingrooms/
│   ├── train.jsonl          (235)   ← look/simple × livingrooms
│   ├── test.jsonl           ( 26)
│   └── test_unseen_task.jsonl (216) ← livingrooms 里的 unseen task (two_obj)
└── bathrooms/                       ← held-out scene type, 不训
    ├── test.jsonl           (646)   ← 所有 task × bathrooms
    └── test_unseen_task.jsonl (382) ← paper Table 1 col-3 数据源 (two_obj + clean ∩ bath)
```

总计 3553 trial（与 paper 3554 ≈ alfworld 自带 train split 大小一致）。

### 3.5 失败回滚

resplit 脚本是 **atomic swap** 设计：staging → `<alfworld>.tmp/`，行数 invariant 全 OK 才 `mv` 原地，原数据保留至 `<alfworld>.bak.<ts>/`。任何 11 个 invariant 失败 → abort，原地数据 untouched。

```bash
# verify OK 后清备份
ls -d $SCRATCH/wormi-data/alfworld.bak.*
rm -rf $SCRATCH/wormi-data/alfworld.bak.<ts>
```

### 3.6 永久修复

`tools/build_alfworld_dataset.py::UNSEEN_TASK_TYPES` 已同步更新为新值，下次 build 默认就对，**不再需要 resplit**（仍可用作 sanity）。

---

## 4. World model 训练

### 4.1 设计要点

| 项 | 值 |
|---|---|
| base model | `unsloth/Llama-3.2-1B-Instruct` (unrestricted mirror，无 HF gated license) |
| 数量 | VH 4 × atomic skill family + ALFWorld 3 × seen scene type = **7 world models** |
| trainer | `WorMISubTrainer(SFTTrainer)`，response_template hardcode Llama-3 chat |
| dataset 模板 | `as_chat(tokenizer)` → Llama-3 `<|start_header_id|>...` |
| dynamics aux | **保留** — `(user) Next observation: → (assistant) {next_obs}` 是 paper §3.2 的训练目标之一 |
| 训练超参 | `max_steps=1000, eval_steps=200, save_steps=500, bs=4, lr=5e-5, cosine` |
| precision | bf16（trainer 默认 + on-disk halve） |
| ckpt 大小 | 每个 ~2.4 GB safetensors |

### 4.2 提交

```bash
# VH 4 world models (~45 min on L40S)
qsub $HOME/sh/wormi-train-vh-world.sh

# ALFWorld 3 world models (~6 h 预期 on L40S)
qsub $HOME/sh/wormi-train-alfworld-world.sh
```

两个 job 独立，可并行（各占一个 GPU）。

### 4.3 Curricula 文件

```
tools/world_curricula_vh.py            ← 4 family
tools/world_curricula_alfworld.py      ← 3 room (bathrooms excluded)
```

curricula 是 **python file，不是 yaml**：`wormi.curricula.load_world_model_curricula` 用 `exec()` 加载，要求 top-level 变量名字面 `curricula` 是 `WorldModelCurricula` 实例。

### 4.4 验证训练完成

job 日志末尾会做 checkpoint roll-up：

```
Checkpoint roll-up:
  [OK]   turnon     -> /srv/.../world-vh/turnon/last  (2.4G)
  [OK]   open       -> /srv/.../world-vh/open/last    (2.4G)
  ...
```

任何一行 `[MISS]` → job 会 exit 1，需要看日志找原因（最常见：OOM、HF download 失败）。

---

## 5. World model eval

### 5.1 与 paper Table 1 列对应

| paper Table 1 column | VH 数据源 | ALFWorld 数据源 |
|---|---|---|
| Seen task × Seen scene | `<family>/train.jsonl` (in-dist holdout, trainer 内部切) | `<room>/test.jsonl` (10% in-room holdout) |
| Seen task × Unseen scene | `<family>/test_seen_task_unseen_scene.jsonl` | （ALFWorld 不报这栏） |
| Unseen task × Seen scene | `<family>/test_unseen_task_seen_scene.jsonl` | `<room>/test_unseen_task.jsonl` |
| Unseen task × Unseen scene | `<family>/test_unseen_task_unseen_scene.jsonl` | `bathrooms/test_unseen_task.jsonl` |

VH 4 split × 4 family = **16 eval cases**；ALFWorld 3 split × 3 room + 1 cross-room (bathrooms) = **10 eval cases**。

### 5.2 提交

```bash
# VH (已写)
qsub $HOME/sh/wormi-eval-vh-world.sh

# ALFWorld (TODO，等训练完后写；脚本结构同 VH，多一个 bathrooms 跨 room eval loop)
qsub $HOME/sh/wormi-eval-alfworld-world.sh
```

### 5.3 输出结构

```
$SCRATCH/wormi-eval/world-vh/
├── <family>/<split>.jsonl                # 每个 sample 一行 {message, data: {prompt, answer, pred}}
└── summary_<timestamp>.tsv               # 4 列: family / split / n_rows / accuracy

$SCRATCH/wormi-eval/world-alfworld/
└── （结构同上）
```

`summary_<ts>.tsv` 直接 `column -t` 看就能对 paper Table 1：

```
family   split                                n_rows  accuracy
turnon   test                                 32       0.875
turnon   test_seen_task_unseen_scene          32       0.812
...
```

---

## 6. WorMI 集成训练（N=6 / K=3 retrieval）— TODO

paper §A.6 描述的 WorMI integration stage（多 world model 通过 retrieval 选 K=3 implant 到 base model）尚未实现。当前状态：

- ✅ 7 个 world model checkpoint 准备好（4 VH + 3 ALFWorld）
- ✅ `tools/world_curricula_smoke.py` 有小型 WorMI curricula 例子可参考
- ⏳ `tools/wormi_curricula.py` 待写：N=6 怎么从 7 个世界模型聚合 + K=3 retrieval 怎么编排
- ⏳ paper Table A.6 没穷尽列；可能需要看 paper Appendix §A.6 原文 + 联系作者

设计草稿：

```python
# 占位 — TODO
from wormi.curricula import WorMICurricula, WorMICurriculum, WorldModel
from wormi.model import WorMIIntegrateMethod
from wormi.trainer import WorMITrainerConfig

VH_FAMILIES = ["turnon", "open", "puton", "placein"]                  # 4
AFW_SEEN_ROOMS = ["bedrooms", "kitchens", "livingrooms"]              # 3
# N=6 怎么聚合？候选: 4 VH + 2 ALFWorld（drop 1 living/bed）？还是 split 维度聚合？
# K=3: trainer 一次 implant 3 个 world model（retrieval-by-task-similarity）

# 完整 curricula 见 Readme.md STEP 4
```

CLAUDE.md 已警告：当前 `wormi/scripts/train.py` 和 `eval.py` 用了**旧 API 名字** (`main_model`, `model_wise_positional_encoding`, `model.plug/unplug_all`)，与现 `wormi/model.py` 的 `base_model`/`world_wise_positional_encoding`/`implant/remove_all` 不一致。**集成训练前先确认 scripts 修复或锁版本**。

---

## 7. WorMI 集成 eval — TODO

`wormi eval --curricula_path` 接同一个 wormi curricula，跑里面 `test_curricula` 列表。结果与 paper Table A.6 / Table 1 对照。等 §6 完成后写。

---

## 附录 A: 设计陷阱清单

实际踩过 + 已修的坑（按 stage 排序）。

### A.1 数据 build 侧

| # | 陷阱 | 根因 | 修复 |
|---|---|---|---|
| 1 | ALFWorld trial 与 metadata 错位 | builder 用循环索引推断 task_type，env reset 跳过会导致指针漂移 | builder 改用 `infos["extra.gamefile"]` 取真值，主循环 `while len(seen_gamefiles) < n_games` |
| 2 | VH 78 instruction 凑不齐（只到 62/78） | `select_classes_with_property` 用全 scene **交集**，多数 property 不能跨 7 scene 都有 | 改用 scene **覆盖率排序**；puton/placein 改 ranked pair joint coverage |
| 3 | VH 16 seen task 抽到 turnon/open 各 0 个 | 全局 uniform shuffle，按家族 size 抽样导致小家族被吃空 | 家族 stratified split（turnon=2 / open=1 / puton=6 / placein=7） |
| 4 | VH 3 类 test split 文件丢失 | converter 只输出 train/test，没分 seen/unseen task × seen/unseen scene 4 种 | 补全 `test_{seen,unseen}_task_{seen,unseen}_scene.jsonl` |
| 5 | VH raw room 名（dining_room/home_office）与 paper 4 room 集不对齐 | VH graph node 用 raw name，paper 用 canonical 4 room | converter 加 `_canon_room` 映射，查 VH `resources/class_name_equivalence.json` |

### A.2 数据 split 侧（**本 session 最大发现**）

| # | 陷阱 | 根因 | 修复 |
|---|---|---|---|
| 6 | **ALFWorld unseen task 选 {heat, cool}** | builder 注释只看 "longest expert plan"，忽略 paper Table 1 col-3 = unseen_task ∩ bathrooms 必须非空 | resplit 为 `{two_obj, clean}`；唯一物理可行 ∩ 数据充足 ∩ 概念自洽的选择 |
| 7 | bedrooms/livingrooms 的 `test_unseen_task.jsonl` 是 0 字节空文件 | trap 6 的衍生物（heat/cool 不在 bed/living） | resplit 后变为 240 / 216 trial（two_obj 子集） |
| 8 | paper "Following CL-ALFRED" 引导误判 | CL-ALFRED 本身没有 task-level seen/unseen，只有 scene-level；"Following" 仅指沿用 ALFRED-6-task 框架 | 翻 CL-ALFRED 原文（arxiv 2403.07548）验证 |

### A.3 Trainer / 模型侧

| # | 陷阱 | 根因 | 缓解 |
|---|---|---|---|
| 9 | `wormi/scripts/train.py / eval.py` 用了旧 API 名字 (`main_model`, `plug/unplug_all`) | 脚本与 model.py 演进未同步 | 已记录 CLAUDE.md；集成阶段使用前必须先修脚本或锁版本 |
| 10 | dynamics aux task（"Next observation: → {next_obs}"）看起来像 chat 模板 bug | Figure A.5 是 **inference/eval-time prompt** 不是 training template；paper §3.2 训三个 aux task，dynamics 必须保留 | `wormi/datasets/alfworld.py::_convert_to_chat` 注释已警告 |
| 11 | ALFWorld trial cumulative=True 展开很长（history 30 step），prompt token 数可能爆 max_seq_length | SFTConfig 默认 max_seq_length=1024，长 prompt 截断 → 实际只 train 前几步 | 当前 max=30 step 是 dataset 层 cap（alfworld.py:43），训练 log 监控是否过短 |
| 12 | `WorMI.save_pretrained` 强制 `safe_serialization=False` | world models 间 shared tensor 触发 safetensors 报错 | model.py 已 override，使用者无感 |

### A.4 PBS / 环境侧

| # | 陷阱 | 根因 | 修复 |
|---|---|---|---|
| 13 | `uv sync` 在 login node 跑会 OOM / 卡 | `torch==2.5.1` ~700 MB 下载 + 解压 | 一律在 compute node 跑（PBS job 第一步） |
| 14 | HF download 卡死在 100% | `hf_transfer` 在 Katana 某些 node 上不可用 | `export HF_HUB_ENABLE_HF_TRANSFER=0` |
| 15 | alfworld 装包占爆 inode quota | alfworld + torch ≈ 25K 文件 | venv 装在节点 `/tmp/`，跑完 rm |
| 16 | `.venv` 软链残留导致 `uv sync` 报错 | 上一个 job 在不同 node 留下断链 | sh 脚本头 `[ -L .venv ] && rm -f .venv` |
| 17 | `UV_LINK_MODE=copy` 必须设 | scratch 和 home 跨 fs，默认 hardlink 失败 | 所有 sh 脚本头部已设 |

---

## 附录 B: paper 数字对照表

复现后实际 vs paper 的预期。**填到这里**需要等所有 stage 跑完。

| 维度 | 我们的复现 | paper 报告 | 偏差 |
|---|---|---|---|
| VH (instruction × scene) episodes | ~1096 | 1023 | +7% (init_graph variant 采样差异) |
| VH unique instructions in jsonl | 70 / 78 | 78 | 8 个 EvolvingGraph 在所有 scene execution_failed |
| VH scenes | 20 | 20 | ✓ |
| ALFWorld episodes (total) | 3553 | 3554 | -1 (alfworld textual env 自带 train split 大小) |
| ALFWorld task types | 6 | 6 | ✓ |
| ALFWorld scene types | 4 | 4 | ✓ |
| ALFWorld Table 1 col-3 evidence | 382 trial | (未明示) | - |
| paper Table 1 accuracy (VH col-1) | TBD | TBD | TBD |
| paper Table 1 accuracy (VH col-2) | TBD | TBD | TBD |
| paper Table 1 accuracy (VH col-3) | TBD | TBD | TBD |
| paper Table 1 accuracy (AFW col-1) | TBD | TBD | TBD |
| paper Table 1 accuracy (AFW col-2) | TBD | TBD | TBD |
| paper Table 1 accuracy (AFW col-3) | TBD | TBD | TBD |

---

## 附录 C: 文件清单

### Python 工具（`tools/`）

| 文件 | 用途 | 阶段 |
|---|---|---|
| `build_virtualhome_dataset.py` | VH converter（zip 流式读 + EvolvingGraph rollout） | §2.1 |
| `build_alfworld_dataset.py` | ALFWorld converter（alfworld textual env + expert plan） | §2.2 |
| `split_alfworld_train_test.py` | 切 10% in-room holdout | §2.2 (cascade) |
| `resplit_alfworld_by_unseen_task.py` | **本 session 新增** — atomic-swap re-bucket per 修正后的 UNSEEN_TASK_TYPES | §3 |
| `world_curricula_vh.py` | VH 4 family 训练 curricula | §4 |
| `world_curricula_alfworld.py` | ALFWorld 3 seen room 训练 curricula | §4 |
| `world_curricula_smoke.py` | 小规模 smoke 测 curricula | dev |

### PBS shell scripts（`$HOME/sh/`）

| 文件 | 资源 | 阶段 |
|---|---|---|
| `wormi-build-vh-dataset.sh` | 4 cpu / 16 gb / 1h | §2.1 |
| `wormi-build-alfworld-dataset.sh` | 4 cpu / 16 gb / 3h | §2.2 |
| `wormi-resplit-alfworld.sh` | 2 cpu / 8 gb / 15 min | §3 |
| `wormi-train-vh-world.sh` | 1× L40S / 8 cpu / 64 gb / 4h | §4 (VH) |
| `wormi-train-alfworld-world.sh` | 1× L40S / 8 cpu / 64 gb / 6h | §4 (ALFWorld) |
| `wormi-eval-vh-world.sh` | 1× L40S / 8 cpu / 64 gb / 4h | §5 (VH) |
| `wormi-eval-alfworld-world.sh` | 同上 | §5 (ALFWorld) — TODO |

### Memory / docs

| 文件 | 用途 |
|---|---|
| [`REPRODUCE_DATA.md`](REPRODUCE_DATA.md) | 数据 build 阶段 deep-dive（converter 算法、5K 行物理矩阵、inode 优化） |
| [`../CLAUDE.md`](../CLAUDE.md) | 代码架构（model.py / trainer.py / datasets/） |
| [`../Readme.md`](../Readme.md) | wormi CLI 入口 + curricula 示例 |

---

## 附录 D: 完整命令序列（cold start → eval done）

下面这串可以从 0 开始一路 qsub 到 §5 完成。每个 step 等前一步 job 完成（看 `qstat -u $USER`）后再提交下一步。

```bash
# §1 一次性环境（不在 PBS 里，需手动在 compute node 跑一次）
ssh <compute_node>
cd ~/WorMI
export UV_PROJECT_ENVIRONMENT=/srv/scratch/$USER/wormi-venv
uv sync

# §2 数据 build（两个 job 独立可并行）
qsub ~/sh/wormi-build-vh-dataset.sh           # ~5 min
qsub ~/sh/wormi-build-alfworld-dataset.sh     # ~2.5 h

# §3 ALFWorld unseen-task resplit（等 §2.2 完成）
qsub ~/sh/wormi-resplit-alfworld.sh           # < 1 min

# §4 world model 训练（两个 job 独立可并行；ALFWorld 需 §3 完成）
qsub ~/sh/wormi-train-vh-world.sh             # ~45 min
qsub ~/sh/wormi-train-alfworld-world.sh       # ~6 h

# §5 world model eval（等对应 §4 完成）
qsub ~/sh/wormi-eval-vh-world.sh              # ~2-4 h
qsub ~/sh/wormi-eval-alfworld-world.sh        # TODO 待写
```

---

📌 **维护**：本文档应在每次 paper-aligned 数据/超参/模型决策变更时同步更新。新陷阱发现 → 加到附录 A；新 stage 实现 → 加正文小节。
