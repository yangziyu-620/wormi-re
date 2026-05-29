# WorMI VirtualHome 数据集构建:从头流程 + 关键改进

日期：2026-05-29

> 一句话定位：这是一次**干净重写**（`tools/build_virtualhome_dataset_wormi.py`），
> 目的是修掉旧 pipeline 的"数据塌缩"问题，让重构数据严格对齐 WorMI 论文设定。
> 重写前的脚本已备份在 `processing-scripts-backup-pre-tmow-rewrite-20260529-001357/`。

参考文档：
- 重写进度记录：`reports/virtualhome/data-processing/wormi-paperaligned-rewrite-2026-05-29.md`
- 论文设定摘录：`reports/virtualhome/data-processing/wormi-paper-spec-2026-05-29.md`
- 旧方法说明：`reports/virtualhome/data-processing/vh-current-data-processing-2026-05-28-zh.md`

---

## 一、从头构建的 6 个步骤

整个 builder 的主流程是 `Builder.materialize()`，可拆成 6 步。

### Step 1 — 加载场景（`load_domains`）

- 从 VirtualHome 原始 init graph 库读入 7 个基础公寓（base apartment）。
- 每个公寓抽多个官方 init-graph variant，组成 **20 个 scene domain**
  （默认布局 `[3,3,3,3,3,3,2]`），每个 domain 含 `variants_per_domain` 个变体
  （默认 8~12）。
- 抽 variant 时做**可行性探针**（`_graph_probe_successes`）：一个图至少要能成功
  执行若干任务才入选，过滤掉残缺图。
- 固定 seed 打乱后切 **6 seen scene / 14 unseen scene**（论文 §4）。

### Step 2 — 选 78 个任务（`select_tasks` Step 1–2）

- 从 `properties_data.json` + 场景物体集合，用 `build_candidate_instructions`
  生成一个按"联合场景覆盖率"排序的候选池。
- 按 family 配额精确取 78 个：**9 turnon + 7 open + 30 puton + 32 placein**
  （论文 Table A.2）。
- **关键约束**：每个 family 加 `source_caps = quota // 4` 的 source 类上限——同一个
  源物体（如 book / mug）在一个 family 里最多占 1/4，防止某个物体刷屏。

### Step 3 — 切 16 个 seen task（`select_tasks` Step 3）

- 配额 **2 turnon + 2 open + 6 puton + 6 placein = 16 seen**，其余 62 为 unseen。
- 用 **stratified-by-source-class greedy**：每次优先选"source 类还没被选过、
  target 类最少"的任务。**这是防首动作塌缩的核心机制。**

### Step 4 — 生成专家轨迹（`execute_slot`）

- 对 (task × scene variant) 的每个组合，用 **EvolvingGraph** 执行确定性专家程序
  （`_execute_paperlike_candidate`）。
- 多层过滤：缺物体的组合预过滤跳过、执行失败跳过、
  `observation == next_observation` 的无效 transition 跳过。
- 每条轨迹的每一步写成一行，带完整 `_meta`
  （task_id / scene / trajectory_id / step_index）。

### Step 5 — 按 4 象限分池 + 采样（`materialize`）

- 所有合法轨迹按 `{seen/unseen task} × {seen/unseen scene}` 落进 4 个池。
- 然后**按任务均衡采样**：
  - `train` = seen×seen，每个 seen task 取 `train_episodes / 16` 条（384/16 = 24）；
  - `eval A` = seen×seen，episode 级从 train **hold out**（同任务但不同轨迹）；
  - `eval B` = seen×unseen；
  - `eval C` = unseen×unseen，**只从这个合法池采**，绝不靠总数相减。
- train 物化成 **6 个 `scene_0..5/` 目录**，对应论文的 N=6 world models。

### Step 6 — 质量闸门 + 写盘（`run` / `write`）

- 先在内存算 `quality_report`，过**硬闸门 `train_first_action_top1_share ≤ 0.35`**，
  不达标直接 `raise` 中止、**一行都不写**。
- 通过后写 `scene_*/train.jsonl`、3 个 `test_*.jsonl`、`eval_col_*` 软链、
  `virtualhome_manifest.json`、`quality_report.json`。
- 辅助任务（BC + dynamics + affordance，论文 §3.2）由 loader
  `wormi/datasets/virtualhome.py` 在**加载时展开**，builder 只写一行原始 transition。

---

## 二、相比之前的关键改进

| 维度 | 之前（`balanced` builder） | 现在（`wormi` builder） | 改进意义 |
|---|---|---|---|
| **任务选取** | coverage 排序后 **top-K 截断** + `semantic-gate=source_unique` 过滤 | 按 family 配额 + **per-source 多样性上限** | 不再让高覆盖率物体垄断 |
| **seen task 切分** | 随机 / 覆盖贪心 | **stratified-by-source greedy**（选最少出现的 source） | source 多样性 **2 → 9** |
| **Observation** | `tmow_compact`，17 边紧凑图 + compact-K 子集 | **完整 class-level 图三元组** `format_observation`（论文 Fig A.2） | train / eval rollout 渲染完全一致，无格式失配 |
| **检索 / 增强** | 含 compact-K 选择、augmentation | **全部移除**，无 BM25、无子集、无增强 | 更贴论文、更可审计 |
| **质量保护** | 无 | **首动作 top1 硬闸门 ≤ 0.35** | 自动拦截数据塌缩，不可能产出坏数据 |
| **首动作分布** | `walk kitchen` 占 **~80%**（塌缩） | top1 占 **0.30**，散布 6+ 房间 | world model 学到真适应而非"无脑去厨房" |
| **eval C 来源** | （两版都正确）只从 unseen×unseen 合法池采 | 同 | 避免泄漏到测试集 |

---

## 三、汇报口径（三句话）

1. **问题**：旧 builder 三层 coverage-greedy 偏置让 seen-task 训练塌缩到 2 个源物体、
   80% 轨迹都 `walk kitchen` 开头，world model 学不到真正的场景适应。
2. **方案**：重写为论文对齐版——按 source 类分层均衡选任务、用完整图三元组观测、
   移除所有非论文的检索 / 压缩 / 增强。
3. **保障**：加了一道首动作分布硬闸门（top1 ≤ 0.35），把"是否塌缩"变成构建阶段的
   自动卡控；smoke 实测 0.30 通过、source 多样性从 2 升到 9。

---

## 附：smoke build 结果（variants_per_domain=6, candidate_multiplier=8）

```text
episodes: 970, rows: 5080
pool sizes: seen×seen=457, seen×unseen=1062, unseen×seen=1819, unseen×unseen=4049
split counts: train=332, eval_a=95, eval_b=224, eval_c=319
train first-action top1 share: 0.3012   (gate threshold 0.35 PASSED)
```

> 注：表中的首动作 / source 多样性数字来自 smoke build。正式数据集的真实数字应以
> 对应 `quality_report.json` 为准。
