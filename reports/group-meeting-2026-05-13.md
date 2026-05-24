# WorMI 复现 — 组会汇报 (2026-05-13)

## 一句话总结

复现 ICML 2025 "WorMI" 工作：把多个独立训练的"world model" CausalLM 通过可训练 cross-attention adapter 注入到一个冻结的 base CausalLM，做 test-time adaptation。当前进度：数据 pipeline 完整、stage-1 训练已 qsub、stage-2 框架就绪等结果。

---

## 1. 论文背景

**WorMI** ("World Model Implanting for Test-time Adaptation of Embodied Agents", ICML 2025, arxiv 2509.03956)

核心思想：

- N 个 frozen world model（每个学一个 domain 的 dynamics + policy）
- 1 个 frozen base reasoning model（Llama-3.2-3B-Instruct）
- 唯一可训练参数：少量 cross-attention adapter（接在 base 的 [13, 27] 层和 world 的 [7, 15] 层之间）
- **测试时**用 Wasserstein 距离从 N 个 world model 里 retrieve top-K=3 装进 base
- 评测在 VirtualHome 和 ALFWorld 两个 embodied benchmark

为什么这种设计有意义：通过分两阶段训练（先训每个 domain 的 world model，再训通用的桥接 adapter），实现**新组合 zero-shot 泛化**——adapter 在测试时遇到训练时未见过的 world model 组合也能用。

---

## 2. 复现挑战

### 2.1 作者没公开数据

- VirtualHome 和 ALFWorld 的处理后 jsonl 不在仓库里
- 联系不上作者
- 必须自建，且要严格对齐 paper §4 / Table A.2 / A.4 数据规模

### 2.2 作者代码有多处 API drift bug

原始 WorMI 仓库的 `wormi/scripts/{train,eval}.py` 调用的 API 名字跟当前 model 类对不上（`main_model` vs `base_model`, `model.plug()` vs `model.implant()`, `model.unplug_all()` vs `model.remove_all()`，等）。**这些脚本现在直接跑不通**。

### 2.3 关键设计细节论文写得含糊

- N=6 world model 怎么按 axis 切分？两个 domain 还不同。
- 测试时是 retrieval 还是固定组合？官方 README 示例用固定组合，但 paper Algorithm 1 是 retrieval。
- 训练时 "sample the subset" 怎么 sample？没说。

---

## 3. 数据 pipeline（这周主要工作量）

### 3.1 VirtualHome

**原始数据**：[programs_processed_precond_nograb_morepreconds](http://virtual-home.org/release/programs/programs_processed_precond_nograb_morepreconds.zip) (1.55 GB)。**不用 Unity 渲染**，纯 Python EvolvingGraph 当 graph 解析器，CPU 节点几分钟跑完。

**输出**（paper §4 对齐）：

- 78 atomic instructions：4 task family × Table A.2 的 9/7/30/32 分配
- 20 distinct scenes：7 个 base apartment 各取 3+3+3+3+3+3+2 个 variant
- 16 seen / 62 unseen tasks：按 task family 分层抽样（保证 turnon/open 不被全局 shuffle 抽空）
- 6 seen / 14 unseen scenes：**每个 base apartment 取 1 个 v0 当 seen，其余当 unseen**
- 共 ~4430 jsonl 行

**N=6 partition**：

```
virtualhome/scene_{0..5}/train.jsonl   ← 每个 scene 一个 world model
virtualhome/test_seen_task_unseen_scene.jsonl    (Table 1 col 2, 595 行)
virtualhome/test_unseen_task_unseen_scene.jsonl  (Table 1 col 3, 2389 行)
```

每行 jsonl 带 `_meta = {scene, split, task_args}` 方便下游分析。

**Output** (aligned with paper §4):

- 78 atomic instructions: 4 task families × the 9/7/30/32 allocation in Table A.2
- 20 distinct scenes: 3+3+3+3+3+3+2 variants for each of the 7 base apartments
- 16 seen / 62 unseen tasks: stratified sampling by task family (ensuring that ‘turnon’ and ‘open’ are not depleted by global shuffling)
- 6 seen / 14 unseen scenes: **1 v0 instance per base apartment is treated as seen, the rest as unseen**
- Total of ~4,430 JSONL lines

**N=6 partitions**:

```

virtualhome/scene_{0..5}/train.jsonl ← One world model per scene

virtualhome/test_seen_task_unseen_scene.jsonl (Table 1 col 2, row 595)

virtualhome/test_unseen_task_unseen_scene.jsonl (Table 1 col 3, 2389 rows)

```

Each JSONL row includes `_meta = {scene, split, task_args}` to facilitate downstream analysis.

**修过的坑（按时间顺序）**：

1. 5/7 第一版 `select_classes_with_property` 用全 scene 交集 → 只生成 62/78 instruction，改成按 scene 覆盖率 ranked + ranked pair 后达 70/78
2. `class_name_equivalence.json` 的 `dining_room→kitchen` / `home_office→livingroom` 没映射 → 加 `_canon_room`
3. `init_and_final_graphs/<scene>/<source>/<file>.json` 是 per-program 存的，每个 program 的 init_graph 是 variant → 流式按 3+3+3+3+3+3+2 采样得 20 scene
4. 5/13 发现 `TrimmedTestScene6_graph__v0` 的 init_graph 有问题，所有 instruction 都 execution_failed → fallback 用 v1

### 3.2 ALFWorld

**原始数据**：`alfworld[full]` 包自带 PDDL game files (~几百 MB)。**textual mode (`AlfredTWEnv`)** 是纯 Python TextWorld 状态机，CPU 节点毫秒级 step，不需要 X11 / THOR / GPU。

**输出**（paper §4 对齐，按 CL-ALFRED setting）：

- 4 scene types: 3 seen (bedrooms/kitchens/livingrooms) + 1 unseen (bathrooms)
- 6 task types (Table A.4): pick_simple / look_at_obj / pick_heat / pick_cool / pick_two / pick_clean
- 4 seen task + 2 unseen task
- 共 ~3935 jsonl 行

**N=6 partition**：按 **task type** 切（不是按 room！见下文 §5.2 决策记录）

```
alfworld/task_{pick_simple,look_at_obj,pick_heat,pick_cool,pick_two,pick_clean}/train.jsonl
alfworld/test_seen_task_seen_scene.jsonl       (col 1, 182 行)
alfworld/test_seen_task_unseen_scene.jsonl     (col 2, 264 行)
alfworld/test_unseen_task_unseen_scene.jsonl   (col 3, 382 行)
```



**修过的坑**：

1. 第一版 builder 在循环里用 `env.game_files[i]` 取 metadata 但 env 内部指针错位 → ~8K 行 jsonl 的 task/scene 对错了 → 重写用 `infos["extra.gamefile"]` 取真值
2. 第一版选 `{heat, cool}` 当 unseen task 但 heat/cool 物理上只发生在 kitchen → bathrooms(unseen scene) 永远没 col 3 数据 → 改为 `{pick_two, pick_clean}`（物理可行 + 数据充足）

---

## 4. Stage 1 训练（已 qsub 在跑）

**12 个 world model = 6 VH scene + 6 ALFWorld task type**

**超参对齐 paper Table A.6**：

- Base: `unsloth/Llama-3.2-1B-Instruct`（unsloth mirror 避免 meta-llama gated license）
- batch=4, 2000 gradient steps, lr=3e-5, cosine
- bf16 + gradient checkpointing（一个 L40S 装得下）
- Behavior cloning style: `DataCollatorForCompletionOnlyLM(response_template="<|start_header_id|>assistant<|end_header_id|>")` 只对 assistant 回复段反传

**当前状态**：

- Job 7939198 (VH N=6) — R, ~20 min
- Job 7939206 (ALFWorld N=6) — R, 刚启动
- 预计 ~2.5h 每个 job，可并行（两个 L40S）

---

## 5. Stage 2 准备（这周后半完成）

### 5.1 修官方代码的 4 处 API drift bug

| 文件                                           | 问题                                                                | 修复                                                                             |
| ---------------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `wormi/scripts/eval.py`                      | `model.unplug_all()`                                              | →`model.remove_all()`                                                         |
| `wormi/scripts/eval.py`                      | `model.plug(...)`                                                 | →`model.implant(...)`                                                         |
| `wormi/scripts/eval.py`                      | `config.main_model`                                               | →`config.base_model`                                                          |
| `wormi/scripts/train.py`                     | `WorMIConfig(main_model=..., model_wise_positional_encoding=...)` | → 用 dataclass 实际字段名 `base_model=` / `world_wise_positional_encoding=` |
| `wormi/curricula.py::WorMICurricula.merge()` | 漏了 `vision` / `decaying_learning_rate` 字段                   | 补全                                                                             |

**没修这些 stage 2 直接不能跑**。

### 5.2 加 paper Algorithm 1 的 retrieval 路径

官方仓库的 `wormi/model_store.py` 已经有 `ModelStore` 类（KMeans on CLS embeddings + Wasserstein dist），但**没接到 CLI**——`wormi eval` 走的是固定 `target_world_models=[0,1,2]` 路径，这违反 paper Algorithm 1 line 28 的 Wasserstein top-K 检索。

修了 `wormi/curricula.py::WorMICurricula` 增加 3 个字段：

```python
sentence_embedding_model: str | None = None  # 启用 retrieval
retrieval_k: int = 3                          # paper Table A.6
prototype_size: int = 15                      # paper Table A.6
```

重写 `wormi/scripts/eval.py`：

1. 加载 N=6 world model 各自的 stage-1 train data → 算 prototype set (KMeans n_clusters=15)
2. 每个 test curriculum 算 test prompt 的 prototype set
3. Wasserstein 距离取 top-3 world model
4. `model.remove_all()` 再 `model.implant(retrieved_K)` 然后 generate
5. eval 输出每行 jsonl 带 `retrieved_world_models` 字段（记录每次 retrieval 选了谁）

向后兼容：如果 `sentence_embedding_model=None` 就退回旧的固定组合路径。

### 5.3 写 stage 2 curricula（paper Table A.6 配置）

```python
WorMICurricula(
    base_model="unsloth/Llama-3.2-3B-Instruct",
    connections=[13, 27],                       # base 端
    world_models=[WorldModel(..., connections=[7, 15])],  # world 端
    method=WorMIIntegrateMethod.WORLD_WISE_ATTENTION,
    meta_learning=True,
    num_iterations=8,                           # λ_M
    train_curricula=[<6 个 K=3 subset 循环>],   # λ_I=30 per subset
    test_curricula=[<col 1/2/3>],
    sentence_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    retrieval_k=3,
    prototype_size=15,
)
```

VH (6 个 scene) + ALFWorld (6 个 task type) 各一份。

---

## 6. 关键决策 & 偏离论文之处

### 6.1 N=6 的 partition axis 论文没明说

**Paper 数字**：Table A.6 列出 N=6 / K=3，两个 domain 都是。

**做出的解读**：

- **VH** N=6 = 6 个 seen scenes（数字跟 paper §4 "20 distinct scenes (6 seen, 14 unseen)" 完全咬合，Figure 4 caption 也写"three world models, each derived from different rooms"）
- **ALFWorld** N=6 = 6 个 task types（数字跟 Table A.4 列出的 6 个 task type 咬合）

**两个 domain partition axis 不同**：VH 按 scene，ALFWorld 按 task type。这是从数字推出来的，不是论文明说。

### 6.2 ALFWorld "unseen task" 在哪一层是 unseen

如果 N=6 = 6 个 task type，但其中 2 个是 unseen，那么这 2 个 unseen task 的 world model **训不训**？

**做出的解读**：训。"unseen task" 是 **WorMI 整合训练 / Table 1 测试层面**的语义，不是 world model 层面：

- Stage 1: 全部 6 个 task type 都训 world model（用 3 seen rooms 的数据）
- Stage 2 train_curricula: 6 个 K=3 subset 循环覆盖全部 6 个 world model
- Stage 2 test 时：col 3 (unseen task × unseen scene = bathrooms) 用 Wasserstein retrieval 召回最相关的 K=3，其中可能包含 unseen-task world model

这样 "unseen" 测的是 compositional generalization（adapter 在新组合上的表现），而不是 stage 1 数据可用性。

### 6.3 测试时强制 retrieval（不是固定组合）

paper Algorithm 1 line 28 明确写 `M_ret = {M_j | j ∈ TopK({-δ(p_j, p)}, K)}`。官方 README 示例和 eval.py 走的是固定 `target_world_models=[0,1,2]`，**直接复现就是 fixed combo，不是 paper 的 test-time adaptation**。这是必须修的，否则数字没意义。

### 6.4 训练时"sample subset"

paper Algorithm 1 line 5 "Sample the subset of world models M_1,⋯ ⊂ {M_1,…,M_N}" 没明说分布。当前实现：6 个 train_curricula 各自是 K=3 的 cyclic subset（[0,1,2], [1,2,3], ..., [5,0,1]），每个 world model 在 3 个 subset 里出现一次。这是 paper-faithfulness 的妥协——理想是每次 inner iteration 随机 sample K，但当前 trainer 架构是固定 curricula list。

如果 Table 1 数字差太多再回来改这个。

---

## 7. 当前进度 & 下一步

### 完成

- [X] Paper 算法理解 + 与代码 diff 审计
- [X] VH 数据 pipeline (6 seen scene × 16 seen task, 4430 行)
- [X] ALFWorld 数据 pipeline (6 task type × 3 seen room, 3935 行)
- [X] 修 4 处 API drift bug
- [X] 加 paper Algorithm 1 retrieval 到 eval CLI
- [X] Stage 1 curricula + PBS 脚本
- [X] Stage 2 curricula (VH + ALFWorld)
- [X] Stage 1 训练已 qsub（两个 GPU 并行）

### 进行中

- [X] Stage 1 训练完成（~2.5h 每个 job）

### 待做

- [ ] Stage 2 训练 PBS 脚本（这次组会后起草）
- [ ] Stage 2 训练（依赖 stage 1 完成）
- [ ] 用 `wormi eval` 跑 Table 1 三栏（依赖 stage 2 完成）
- [ ] 跟 paper 数字对比 + 写 limitations

### 开放问题（想跟组里讨论的）

1. ALFWorld N=6 = task type 是我的解读，组里有没有人对 paper 别的读法？
2. 训练时 random subset sampling vs fixed cyclic subset，是否值得改 trainer 架构？
3. sentence embedding model 用 MiniLM-L6 还是 BERT-base / sentence-transformers 别的 backbone？paper 没说。
4. VH 的 col 1 (seen task × seen scene) test 数据 paper 怎么定义的？我现在拿 train set 当 col 1 test（in-distribution memorization），可能跟 paper 不一致。

---

## 附录: 修改的代码文件清单

**新增**：

- `tools/build_virtualhome_dataset.py` (5/7-5/13 多次迭代)
- `tools/build_alfworld_dataset.py` (5/12)
- `tools/resplit_alfworld_by_unseen_task.py` (5/13)
- `tools/resplit_alfworld_by_task_type.py` (5/13)
- `tools/world_curricula_{vh,alfworld}.py`
- `tools/wormi_curricula_{vh,alfworld}.py` (stage 2)
- `sh/wormi-{train,resplit}-*.sh` (PBS 脚本)

**修改**：

- `wormi/curricula.py`：加 retrieval 3 字段，修 merge() 漏字段
- `wormi/datasets/alfworld.py`：chat 模板对齐 paper Figure A.5
- `wormi/scripts/train_world.py`：bf16 load
- `wormi/scripts/train.py`：3 处 API name 修复
- `wormi/scripts/eval.py`：3 处 API name 修复 + 加 paper Algorithm 1 retrieval
- `wormi/trainer.py`：max_seq_length=4096, bf16, gradient_checkpointing, save_only_model

**删除**：

- `requirements.txt`（改用 `uv` + `pyproject.toml`）

---

## 数据集统计（汇报参考）

### VirtualHome

| split                                    | 行数           |
| ---------------------------------------- | -------------- |
| 6 seen scenes train (合计)               | 313            |
| col 2 (seen task × unseen scene)        | 595            |
| col 3 (unseen task × unseen scene)      | 2389           |
| (not in paper) unseen task × seen scene | 1133           |
| **总计**                           | **4430** |

### ALFWorld

| task type                     | train (3 seen rooms) |
| ----------------------------- | -------------------- |
| task_pick_simple              | 483                  |
| task_look_at_obj              | 272                  |
| task_pick_heat                | 412                  |
| task_pick_cool                | 477                  |
| task_pick_two (unseen task)   | 580                  |
| task_pick_clean (unseen task) | 501                  |
| **train 合计**          | **2725**       |

| Table 1 column                              | 行数            |
| ------------------------------------------- | --------------- |
| col 1 (seen × seen)                        | 182             |
| col 2 (seen × unseen scene)                | 264             |
| col 3 (unseen × unseen)                    | 382             |
| **总计 (含 train + col 1-3 + col 4)** | **~3935** |
