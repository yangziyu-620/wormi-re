# WorMI VirtualHome 数据有效性验收标准（事前 / 永久 / 强制）

Version: 1.0 — 2026-05-31
Status: **Permanent standard.** 任何 VirtualHome 候选数据集在进入 stage1/stage2 训练之前，
必须按本标准逐条判定。**只有全部 HARD 闸通过才允许训练（GO）；任一 HARD 闸不过即 NO-GO。**
本标准是"事前"标准：在训练之前判定，而不是训练失败后回溯。修改本标准需显式 bump version 并记录理由。

相关：失败调查活文档 `reports/virtualhome/wormi-data-construction-failure-survey-2026-05-30.md`；
强制工具 `tools/expert_replay_vh.py`（A 类闸）、`tools/validate_virtualhome_dataset.py`（B/C/D 部分）。

---

## 0. 核心原则（为什么需要这份标准）

> **数据有效 ⇔ EVAL pipeline（不是 build pipeline）能把 expert 真值动作复现到 ~100%。**

历史教训（13 个点的铁证）：旧 validator 用 **build 时 ScriptExecutor** 重放 `_meta.script_line`（已绑定 instance、整脚本一次执行），报告 replay=100%。但 expert 真值**动作字符串**经**真正的 eval pipeline**（`_script_line_from_prediction` → `_choose_node_id` 重新 class→instance 绑定 → `execute_one_step` 逐步 → `_goal_satisfied`）只有 **87/85/89%**。

因此本标准的第一性要求：**验收必须打在 eval-pipeline 这一层，build 自洽是必要但被证明不充分的。** 凡是只测 build 自洽的检查，一律视为"未验收"。

> **强制函数特性**：A2（expert-replay ≥ 99%）这条闸，**任何事后补救（band-aid）都过不了**——只有当 build 的实例绑定/执行契约与 eval 的**真正一致**时才能通过。所以这条标准本身就强制"根本性修复，而非事后补救"。不要试图为了过闸去放宽闸；闸的存在就是为了逼出根因修复。

---

## 1. HARD 闸（任一不过 = NO-GO，禁止训练）

### A. Eval 接口完整性（历史上 12 次失败的真正死因层）

| ID | 准入条件 | 阈值 | 测量方式 | 拦截的历史失败 |
|---|---|---|---|---|
| **A1** | 所有 train/test 行的 `_meta.scene` key **100% 命中** eval 实际使用的 scene_inits 文件 | 命中率 = 100% | 对比数据集 `scene_inits.json` 的 key 与 eval shell 传入的 `--scene_inits_json` | B2：eval shell 指向 `init_graphs_20_semantic.json`（key `__v0`）vs 数据 `__d00_v00`，零重叠 → 每 episode KeyError |
| **A2** | **Expert-replay SR**（喂真值动作走完整 eval pipeline，不加载模型）在**每一列** col1/col2/col3 | **≥ 0.99** | `tools/expert_replay_vh.py` | B1：`_choose_node_id` 绑错 instance → expert SR 仅 87/85/89% |
| **A3** | **Gold-script_line 天花板**（已绑定 instance 的 gold 行走同一 eval env）在每一列 | **≥ 0.99** | `expert_replay_vh.py` 的 control | B3：col3 仅 95.6%（PUTIN 执行了但 `_goal_satisfied` 不认）= eval 执行/goal 契约分歧 |
| **A4** | **fail-only-binding** episode 数（gold 能过但 eval 绑定路径过不了） | **= 0** | `expert_replay_vh.py` | B1 的细分指标，确保绑定 drift 被清零而非被宽松 goal 掩盖 |
| **A5** | **Train/eval 观测 renderer 字符级一致**：同一 (init_graph + GT 脚本到第 t 步) 用 train renderer (`format_observation`) 与 eval renderer (`_render_rollout_observation`) 渲染，逐字节相同；覆盖全部 4 个 family | mismatch = 0 | T3 harness（驱动 eval 侧 `execute_one_step`+`state.to_dict()`，**不得**复用 build 的 `graph_state_list`）| 历史 tmow-compact-fill17：train compact / eval 全图不一致 → SR≈0 |

> A1–A5 的共同要求：**测的是 eval 代码路径**。任何用 build 时 ScriptExecutor 整脚本重放来"代替"的检查不计入验收。

### B. 数据完整性 / 泄漏

| ID | 准入条件 | 阈值 | 测量 |
|---|---|---|---|
| **B1** | train/test **trajectory_id 重叠** | = 0 | `validate_virtualhome_dataset.py` |
| **B2** | train/test **exact row 重叠** | = 0 | 同上 |
| **B3** | **seen/unseen 任务**互斥、**seen/unseen 场景**互斥 | 重叠 = 0 | 同上 |
| **B4** | build 时 ScriptExecutor 重放失败 | = 0 | 同上（**必要非充分**，单独不构成 GO）|

### C. 可学习性 / 覆盖

| ID | 准入条件 | 阈值 | 测量 | 拦截 |
|---|---|---|---|---|
| **C1** | eval 每一列出现的**动作动词**都在 train 中出现过 | verb gap = 0 | T4 | col3 出现 train 没有的动词 → 只能瞎猜 |
| **C2** | eval 每个 **goal 对象类**在其场景图中可达（present） | 100% reachable | T4 | 目标对象不在场景 → episode 不可能赢 |
| **C3** | **agent 状态**（当前房间 + 手持物体）出现在 100% 的行 | = 100% | T2 | 观测缺 agent 位置 → 动作不可推 |
| **C4** | **标签单值**：同一 (instruction, observation) → action 在同一 split 内无冲突 | 冲突 = 0 | T2 | 同输入多标签 → 监督矛盾 |

### D. 结构 / 形状

| ID | 准入条件 | 阈值 | 测量 | 拦截 |
|---|---|---|---|---|
| **D1** | 每个 world-model（scene_N）train **episode 数** | ≥ 20 | quality_report | 历史 paperlike-v2：10-11/scene → WM 数据极稀疏 SR=0 |
| **D2** | train **源对象 top1 share**（反塌缩，测源对象不测 room-walk） | ≤ 0.35 | `build_*_realtasks` gate / quality_report | 历史塌缩：2 源对象垄断、80% walk_kitchen |
| **D3** | 若使用 per-scene 评估/路由：每个 `scene_N/test.jsonl` 必须是**该场景自己的**held-out 集，且行内 `_meta.scene` 与目录一致 | 跨 scene Jaccard < 1.0 且 `_meta.scene` 自洽 | 检查 symlink 与 `_meta.scene` | B4：6 个 `scene_N/test.jsonl` 全软链到同一 mixed pool，字节相同、`_meta.scene` 错位 |

---

## 2. SOFT 项（不阻断训练，但必须**测量并记录**，进 quality_report / 活文档）

| ID | 项 | 记录什么 | 说明 |
|---|---|---|---|
| **S1** | 观测可判别性 | 多房间歧义率、blinded 抽样可判别比例 | T2：歧义被 class-level goal 吸收则非致命，但决定 exact-match 上限 → **训练只看 rollout SR，不看 exact-match** |
| **S2** | 观测膨胀 | 三元组数 median/max | 大但非致命；影响学习难度 |
| **S3** | Prototype 检索可分性 | 用**实际接入的 embedder** 算 retrieval@1/@2 + 距离矩阵 | T5：若用 world-model 路由则必须达 random+margin；当前用 MiniLM 仅 0.53-0.62（弱），且需用真实 Llama embedder 复核 |
| **S4** | episode 总数 / per-split 计数 | 实际 vs 论文 1023 | 论文 per-split 计数未知，接近即可，不作硬闸 |
| **S5** | same-task 跨场景动作序列重叠 | `exact_action_sequence_overlap` | 原子任务跨场景天然重叠，记录不阻断 |

---

## 3. 验收流程（每个候选数据集，按序）

1. **A1 先行**：scene_inits key 必须 100% 命中 eval 将用的文件——否则后续全是 KeyError，无意义。
2. 跑 `tools/expert_replay_vh.py` 得 A2/A3/A4。
3. 跑 A5 格式一致性 harness。
4. 跑 `validate_virtualhome_dataset.py` 得 B/C/D 多数项（注意：它的 replay 只满足 B4，**不**满足 A 类）。
5. 记录 S1–S5。
6. **判定**：全部 HARD 闸通过 → **GO**；任一不过 → **NO-GO**，定位到对应 bug 根因修复后**重跑全部 A 类闸**（不是只跑改的那条）。

> 不允许"绕过闸"或"放宽阈值以通过"。若确有正当理由调阈值，必须 bump 本文件 version 并写明理由与影响。

---

## 4. 当前 v3 数据集对照本标准

### 4.1 修复前快照（2026-05-31 上午）

- **A1 ✗**（eval shell 指错文件）· **A2 ✗**（0.87/0.85/0.89 < 0.99）· **A3 ✗**（col3 0.956 < 0.99）· **A4 ✗**（fail-only-binding 9/25/22）· **A5 ✓**（renderer 0 mismatch）
- **B1–B4 ✓** · **C1 ✓** · **C2 ✓** · **C3 ✓** · **C4 ✓** · **D1 ✓**（26-58）· **D2 ✓**（源 top1 0.094）· **D3 ✗**（scene_N/test.jsonl 全软链 mixed pool）
- 判定：**NO-GO**。阻断项全部在 eval 接口层（A1/A2/A3/A4）+ 数据布局（D3），数据内容本身干净。

### 4.2 根因修复后快照（2026-05-31，已独立复验）

根因修复（详见 `wormi-rootfix-status-2026-05-31.md`）：
- **A2/A4**：eval 新增 `_build_goal_binding`（在 reset 图上复用 build 的 `select_task_instances` 重导出目标实例），重写 `_choose_node_id` 让 goal 结构主导绑定——**build 与 eval 共用同一实例选择器**。仅用 (family, task_args, 图)，不读 `_meta.script_line`/`instance_selection`（rollout 时可得，对真实模型同样有效）。
- **A3**：eval 执行器 `instance_selection=False → True`，使其遵守已解析的目标节点 id（gold-control 证明这是契约修复，非放宽 goal；`_goal_satisfied` 字节未改）。
- **A1**：eval shell 默认指向数据集自带 `scene_inits.json`。
- **D3**：builder `write()` 改写 per-scene test；post-process 就地修好 v3（未重建）。

独立复验结果（`tools/expert_replay_vh.py` 全列）：

- **A1 ✓**（800/800 key 命中）· **A2 ✓**（expert SR **1.00/1.00/1.00**）· **A3 ✓**（gold 天花板 **1.00/1.00/1.00**）· **A4 ✓**（fail-only-binding **0/0/0**）· **A5 ✓**
- **D3 ✓**（scene_N/test.jsonl 为真实 per-scene 文件，md5 各异，`_meta.scene` 自洽）· 其余 B/C/D 仍 ✓
- 反作弊审计：绑定路径无 `_meta.script_line`/`instance_selection` 读取；阈值未放宽。

**判定：GO（通过全部 A 类 HARD 闸）。** 注意区分：通过本"事前"闸 = **数据↔eval 接口干净、可安全进训练**；**不等于**能复现 Table 1 SR——后者需实际训练验证（属下一步、独立问题）。

---

## 5. 根因（供修复参考，非补救清单）

- **A2/A4 根因（B1）**：eval `_choose_node_id`（`eval_vh_rollout.py:380`）按**邻近/手持**打分选 instance；build `select_task_instances`/`_select_class_node`（`build_virtualhome_dataset.py:622`）按**目标结构**选。**两个绑定函数优化目标不同 → 必然 drift。** 根本修复 = 让 build 的专家轨迹生成与 eval 的 rollout 绑定**共用同一个绑定函数 + 同一执行/goal 契约**，使数据 by-construction 与 eval 一致；而不是让 eval 去吃 `_meta.script_line`（那只救 expert replay，救不了真实模型 rollout）。
- **A3 根因（B3）**：build 用 `ScriptExecutor.execute(整脚本)` 验证语义有效，eval 用 `execute_one_step` 逐步 + `_goal_satisfied`。**两套执行契约**。根本修复 = build 的 `_is_semantically_valid_trajectory` 改用 eval 的逐步+goal 契约验证，则 gold 天花板 by-construction = 100%。
- **A1 根因（B2）**：eval shell `wormi-eval-vh-rollout.sh:13` 默认指向外部 scene-inits；应指向数据集自带 `scene_inits.json`。
- **D3 根因（B4）**：parent builder `write()` 把每个 `scene_N/test.jsonl` 软链到全局 `test_seen_task_seen_scene.jsonl`。

> 统一原则：**build 端的数据生成与验证，必须走 eval 端完全相同的 env 接口（绑定 + 逐步执行 + goal 判定）。** 当前 build 与 eval 是两套漂移的实现——这才是 12 次失败的元根因。根本修复 = 让二者共用一个 ground-truth env 接口模块。
