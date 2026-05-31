# WorMI VirtualHome 数据构建复现失败调查报告（活文档）

Date: 2026-05-30
Status: **Living document** — 复现目标是完整对齐论文 Table 1 (WorMI, VirtualHome)。
本文件汇总至今为止的全部数据构建尝试与各自的问题。

> **维护约定**：每次做完一次新的数据构建 / 训练 / 评估实验后，更新本文件——
> 在 §2 时间线追加一行（名称/日期/做了什么/遇到的问题/证据路径），
> 在 §5 主要原因里调整诊断（如某个 P 级原因被证实或排除），
> 并在 §6 开放问题里勾掉已回答的问题。保持"按时间顺序、证据内联"的格式。
> 本文件只做**诊断与总结**，不写解决方案。

调查方法：7-agent workflow 精读 `reports/virtualhome/{data-processing,audits,validation,experiments}/`
全部文件 + `tools/build_*.py` 系列脚本 + git 历史，再由 opus 综合。

---

## 1. 复现目标 (Table 1 到底要什么)

- **数据规模**：1023 episodes，78 个任务（族配额 TurnOn=9 / Open=7 / PutOn=30 / PlaceIn=32），20 个 scenes。
- **划分**：任务 16 seen / 62 unseen；场景 6 seen / 14 unseen；N=6 world models。
- **观测格式**：论文 Figure A.2 / Appendix B.1 = **class-level 全图三元组 (subject, relation, object)**。
- **三列评估**：col1 = seen task × seen scene；col2 = seen task × unseen scene；col3 = unseen task × unseen scene。
- **指标**：环境 rollout 的 Success Rate (SR) 与 Path/Step (PS)。论文目标值
  (`reports/virtualhome/experiments/vh-reproduction-progress-2026-05-19.md:141-143`)：
  - SR：**85.78 / 80.26 / 66.12**；PS：**10.76 / 12.42 / 15.17**。
- **Stage-1 (world model)**：Llama-3.2-1B，batch 4，2000 步，lr 3e-5，cosine。
- **Stage-2 (WorMI 集成)**：Reptile β=0.1，inner_steps λ_I=30，meta_steps λ_M=8，N=6，K=3，batch 4。
- **统计**：5 seed 的 95% CI。

> **关键事实**：论文**未公布** 78 任务名单、16-seen 子集、20 scene 身份、per-(task,scene) episode 数。
> 因此所有数据构建都是"paper-compatible 重构"，不是"author-exact"。这是贯穿全部失败的根本约束。

---

## 2. 尝试时间线 (按时间顺序的每次数据构建尝试)

下表合并了多个 reader 对同一构建的重复描述；冲突已在表后标出。证据路径内联。

| # | 名称 / 时间 | 目标 | 做了什么 | 遇到的问题 |
|---|---|---|---|---|
| 1 | **easy_debug / atomic-template**（≤2026-05-19，初始快照 commit 6e6827b） | 跑通 stage1/stage2 管线的最小数据 | `tools/build_virtualhome_dataset.py` easy_debug 模式；固定原子模板（walk+grab+walk+put 等），2-5 步，与房屋状态无关；单 v0 init-graph/公寓 | open/switchon 动作 100% 不改变观测（444/444 open、82/82 switchon），无 state 三元组；goal fact 隐藏（placein 隐藏 366）；轨迹太短，col1 仅 6 条轨迹；Scene6_v0 init 非法全部 execution_failed；1096 条（≠1023）；只有 15/16 seen 任务。证据：`reports/virtualhome/validation/vh-data-validation-fixedtraj-2026-05-19.json` |
| 2 | **semantic-1023 + task-aware split**（2026-05-19→20） | 达到 78 任务 /1023 条，修 col1 覆盖 | 加 semantic 有效性过滤（跳过 init 已满足 goal 的 episode）；task-aware held-out（每 seen 任务留 2 条）；per-task/scene manifest | 仍 2-5 步；col1 升到 32 条但 per-scene train 降到 6-11 条（**WM 数据极稀疏**，后被定性为 SR=0 主因）；test_seen_seen 仅 36 行。证据：`vh-data-validation-main-semantic-1023-2026-05-19.json`、`vh-data-validation-taskaware-split-2026-05-20.json` |
| 3 | **paperlike-v1 / house-config planner**（2026-05-26→27） | 用图感知 expert planner 替代模板，目标 7-10 步；3 个 v0/v1/v2 变体/公寓 | `_paperlike_program`（`build_virtualhome_dataset.py:703-801`）按 init 图读取房间/容器/放置，发条件导航步 | **房间命名冲突**：观测 canonical（kitchen/livingroom），动作标签用原始 VH 名（1180 行 `walk dining_room`、718 行 `walk home_office`）；**冗余/no-op walk** 占 walk 的 31-35%；class-level 多实例歧义 36.3%；离线 Table1 col1/2/3 SR=0。证据：`vh-paperlike-default-validation-2026-05-26.json`、`audits/vh-data-construction-audit-2026-05-27.md` |
| 4 | **paperlike-v2-fixed**（2026-05-27，commit 5096eee） | 修房间 canonical、去冗余 walk、加 3 类 world-model aux（BC+dynamics+affordance）；跑完整管线 | `_paperlike_program` 房间 canonical 化、去 blind walk、去连续重复；`as_chat()` 每行扩 3 样本；加校验门（raw room、unchanged-walk>5% 报错）。产出 `virtualhome-paperlike-v2-fixed-20260527`，1023 traj/5451 行 | **per-scene 仅 10-11 train 轨迹（43-58 行）**，WM 训练数据极少；greedy rollout SR=0/0/0（PS≈30 撞顶）；离线 first-step exact-match 仅 18-22%。**后续诊断根因**：seen-task 选择只覆盖 2 个 source 对象（drawing/mat），80% 轨迹以 walk_kitchen 开头（数据塌缩）。证据：`vh-paperlike-v2-fixed-validation-2026-05-27.json`、`data-processing/vh-paperlike-v2-fixed-correction-plan-2026-05-27.md` |
| 5 | **post-hoc-compact17**（2026-05-28，Phase 1） | 不重建、事后把全图观测压到 17 三元组测试是否解 walk 塌缩 | `tools/compact_virtualhome_observations.py` 事后改写 JSONL；BM25-like 评分+强制 task/agent fact，next 用 delta。均值 269→23.51 三元组 | 仅对**已序列化的 class-level 文本**操作，矛盾事实（drawing 在 3 个房间）已在文本中，无法知道 expert 用哪个实例；严格 mandatory-only 变体造成 66 行 train/test 重叠；底层任务选择塌缩未解。证据：`tmow-data-preprocessing-for-wormi-reproduction-2026-05-28.md:363-442` |
| 6 | **tmow-compact-fill17**（2026-05-28，Phase 2 第一版） | 从 EvolvingGraph 状态直接生成 17-edge compact 观测 | `tools/build_virtualhome_dataset_tmow_compact.py`，ScriptExecutor 重放后渲染 compact；fill 到 17+mandatory；next=delta。0 轨迹/行重叠 | **两个 bug**：(1) eval_vh_rollout.py 仍用全图 `format_observation` → train/rollout 观测格式不一致；(2) current 观测用 **target action** 排序三元组（rollout 时不可得）→ action-conditioned 泄漏。SR≈0，repeated walk。证据：`tmow-data-preprocessing-...-2026-05-28.md:763-812`、`vh-paperlike-tmow-compact17-validation-2026-05-28.json`（'train/test exact row overlap: 66'） |
| 7 | **tmow-compact-aa-fill17**（2026-05-28，Phase 2 修正） | 修 action-conditioning：current 观测 action-agnostic | `compact_observation(..., action='')`；eval_vh_rollout.py 按 `_meta.observation_preprocessing` 自动检测并同样渲染。审计 verdicts 全 true | **class-level 重复歧义仍在**（drawing 同时 inside bedroom/kitchen/livingroom）；rollout 仍 dominated by repeated walk。Table1 SR 0.00/0.47/0.33；老 evaluator partial 3.12/1.41/0.74；`exact_action_sequence_overlap=42`（同任务跨场景）。证据：`tmow-data-preprocessing-...-2026-05-28.md:785-937`、`validation/vh-paperlike-tmow-compact-alignment-audit-2026-05-28.json` |
| 8 | **instance-grounded-compact**（2026-05-28，代码实现未训练） | 用 planner 选定的 instance node 接地，去 class 歧义 | `select_task_instances`、`instance_grounded_observation_triples` 等；compact 前先 instance-ground；eval 跟随 `instance_grounded` metadata | **未启动新训练**（用户在 2026-05-28T11:18 停止）；只 smoke 验证。证据：`tmow-data-preprocessing-...-2026-05-28.md:829-916` |
| 9 | **balanced-reconstruction**（2026-05-28，无 gate） | 多 init-graph 变体（variants_per_domain=12），每 WM 64 train；eval-C 严格从 unseen×unseen 池采 | `tools/build_virtualhome_dataset_balanced.py`，`_choose_balanced` 软平衡；轨迹级 split；全图观测 | loader smoke 失败（验证环境无 transformers）；无 semantic gate → source 多实例歧义；source 多实例 781/1023=76.3%、多位置 961/1023=93.9%；`exact_action_seq_overlap=71`；2 个 scene cache 空。证据：`vh-balanced-reconstruction-validation-2026-05-28.json`、`audits/vh-balanced-reconstruction-semantic-audit-2026-05-28.md` |
| 9b | **balanced-source-unique**（2026-05-28，中间版） | 加 `--semantic-gate source_unique`（拒绝 source 类多实例），重放前过滤 | source 歧义修到 0/1023 | **target 侧歧义仍大**（多 node 251/1023=24.5%，多位置 607/1023=59.3%）；**对象多样性偏置**：seen 任务被 phone 主导（13/16）；常识弱目标 44.5%（Place phone in toilet 等）；scene_5 仅 36 train；prompt/label 语法不一致。证据：`audits/vh-balanced-source-unique-quality-analysis-2026-05-28.md` |
| 10 | **balanced-aux-compact17-sourceunique**（2026-05-28，最近一个有结果的"非 clean"最佳） | source_unique gate + compact17 + 3 类 aux，并用 uv 修 transformers import | seed=42，variants_per_domain=12，`--observation-mode=tmow_compact --compact-num-edges=17`；world_samples=raw*3=15870 | **stage2 autograd inplace conflict** 中断，eval 用失败前 `last`；Table1 SR **67.71/64.29/18.67**，Rollout **61.46/38.39/17.00**；**80% 仍 walk_kitchen 起手**（数据塌缩未解决）；col3 接近随机先验。证据：`experiments/vh-historical-results-summary-2026-05-29.md`、`data-processing/vh-current-data-processing-2026-05-28.md`。**该数据集已在 05-29 重写中删除（~42GB）** |
| 11 | **wormi-paperaligned-rewrite**（2026-05-29，当前分支并行） | 干净重写：stratified-by-source-class 选任务、per-source cap、全图观测、删除所有 TMoW compact/augment、硬门 top1≤0.35 | `tools/build_virtualhome_dataset_wormi.py`，6 步；smoke build 970 episodes、5080 行、top1=0.301、source 多样性=9。Stage-1 启动 `wormi-paperaligned-20260529` | smoke 仅 970（≠1023）；任务/场景/episode 计数全是重构假设；aux 三任务是论文未指定的实现添加；**尚无 stage2/rollout 结果**；class-level 歧义在回归全图后**未处理**。证据：`data-processing/wormi-paper-spec-2026-05-29.md:53-62`、`wormi-data-build-summary-2026-05-29.md:113-124`、`tools/build_virtualhome_dataset_wormi.py:488-540` |
| 12 | **tmow60-noUnity**（2026-05-29，当前分支 tmow60-noUnity-data-pipeline） | 用 `tmow_const.py` 的 TMoW 78 任务名单 + equivalence mapping，选 20 个 object-rich 变体，无 Unity | `select_tmow60_variants.py`（CAND_PER_APT=30，PER_BASE=[3,3,3,3,3,3,2]，SEEN_PER_BASE=[1,1,1,1,1,1,0]）；`build_tmow60.py`（OBS_MODE 默认 **compact**，17 edges） | **未达 1023/78/16**（"missing objects in available VirtualHome graphs"，`build_tmow60.py:241-243`）；seen-scene 每公寓仅 1 个变体（WM 覆盖弱）；默认 compact 与论文全图冲突（潜在 train/rollout 不一致）；无 first-action 门；把 unseen-task seen-scene 并入 train（与 paperaligned 构建相反）；TMoW 类名与 VH raw 图名冲突；**无 validation JSON、无训练/eval 结果、`/root/WorMI/data` 为空**。证据：`tools/build_tmow60.py`、`tools/select_tmow60_variants.py:20-26`、`wormi-data-build-summary-2026-05-29.md` |

**时间线中需标注的冲突：**
- **"source 多样性 2→9"** 与 **"balanced-aux 仍 80% walk_kitchen / drawing-mat 主导"** 同时出现在不同 reader 中。解读：80%/2-source 描述的是**修复前**（paperlike-v2-fixed / balanced 早期）的状态；2→9 是 **paperaligned-rewrite 与 balanced-source-unique gate 之后**的对比；但 balanced-aux-compact17 **本身仍是 80% walk_kitchen**（coverage-greedy + source_unique 仍塌缩）。即 source_unique gate 解决了 source 多实例歧义，但**未解决 first-action / source-object 分布塌缩**——这是两个不同问题被反复混淆。
- **balanced gate 后的主导对象**：一份说 phone 主导（13/16），另一份说 drawing/mat 主导。差异源于不同数据集版本（source-unique 中间版 vs balanced-aux 最终版），不是矛盾，而是不同 gate 配置导致的不同塌缩对象。
- **1023 是否达到**：初始/paperlike 构建报告 1023；paperaligned smoke 970；tmow60 明确未达。说明 1023 在某些构建可达、在更严格 feasibility 过滤下不可达。

---

## 3. 反复出现的失败模式 (跨尝试的共性问题)

**3.1 数据塌缩（first-action / source-object 分布）** — 80% train 轨迹以 walk_kitchen 起手；seen-task 仅覆盖 2 个 source 对象。出现在 paperlike-v2-fixed、tmow-compact 系列、balanced-aux-compact17。原因：coverage-greedy 无 per-source 多样性 cap；加 source_unique gate 也不解分布塌缩，反而引入新单对象主导（phone）。证据：`wormi-paperaligned-rewrite-2026-05-29.md:6-10`、`wormi-data-build-summary-2026-05-29.md:88-109`。

**3.2 观测膨胀** — 全图 class-level 观测均值 193-269 三元组、5-7k 字符（vs TMoW 17）；WM dynamics 目标≈复制全图，监督被无关三元组淹没。出现在 full-graph-baseline、paperlike-v1/v2、paperaligned-rewrite。原因：论文 Figure A.2 要求全图三元组，忠实复现必然膨胀。

**3.3 class-level 多实例歧义** — 同类对象同时出现在多个房间，expert 绑定单一 instance，文本无 instance id → 模仿语义欠定。几乎全部构建；balanced 中 source 多实例 76.3%、多位置 93.9%。instance-grounding 修复实现了但**从未训练验证**。

**3.4 per-WM 训练数据稀疏** — 每 scene 仅 10-14 train 轨迹（43-58 行）。出现在 semantic-1023、paperlike-v1/v2。balanced 用多变体强制 64/scene 才缓解。

**3.5 train/rollout 观测契约不一致** — 模型在 compact 上训练，evaluator 用全图 `format_observation` 渲染 → rollout 无效。出现在 tmow-compact-fill17、tmow-compact-aa（部分）。叠加 action-conditioned 泄漏与 evaluator first-id binding 两个独立 bug。

**3.6 repeated-walk 策略塌缩** — rollout 全 30 步重复 valid walk，非 parser crash、非空输出。出现在 beta01、paperlike-v2、tmow-compact-aa。是 3.1/3.3 的下游表现。**注意：尚未隔离这是数据问题还是 adapter/meta-learning 问题**（开放问题）。

**3.7 Stage-2 训练不稳定（工程）** — 三种崩溃：`release unlocked lock`（taskaware-split）、futex 死锁（paperlike-v1）、autograd inplace conflict（balanced-aux）；外加 seqmeta（论文忠实）SR≈0。原因：`WorMIMetaLearningTrainer` 线程 ring-lock 在 curriculum 边界脆弱；论文忠实 sequential Reptile 与 SFTTrainer optimizer/scheduler 状态恢复冲突。**使"数据质量 vs 训练 bug"无法干净归因。**

**3.8 prompt/label 语法不一致** — 系统 prompt 用 'switch [object]'/'putin [target]'，label 用 'switchon object'/'putin source target'。所有构建。原因：论文自身 Figure A.4 与 Table A.1 就不一致，复现刻意保留。

---

## 4. 与论文 Table 1 的对齐差距 (仍未解决)

1. **任务名单未知**：78 任务/16-seen 子集论文未公布。
2. **场景身份未知**：20 scene / 6-seen 未公布；paperaligned（seed 采样 7 公寓）与 tmow60（60-变体池）互不一致也都未验证。
3. **episode 计数差**：论文 1023，paperaligned smoke 970，tmow60 未达。
4. **观测格式偏离**：当前有结果的最佳（balanced-aux）用 compact 17-edge，非论文全图；paperaligned 改回全图但 instance 歧义未处理。
5. **aux 任务公式**：BC+dynamics+affordance 三任务是实现添加，论文未指定。
6. **Stage-2 算法**：论文 Algorithm 1 = Reptile β=0.1 隔离 inner loop；可用的 threaded 路径用 direct mean (β=1, `WORMI_THREADED_META_USE_BETA=0`)，非 Algorithm 1；忠实 seqmeta SR≈0。**中心阻塞点：忠实即坏、能用即不忠实。**
7. **超参偏离**：本地 max_steps=1000（论文 2000）、lr=5e-5（论文 3e-5）；batch=1+ga4（论文 4，因 OOM）。
8. **指标差距**：历史最好完整结果 threaded-clean **100/89.78/35.67**（PS 4.84/6.20/19.30）vs 论文 **85.78/80.26/66.12**（PS 10.76/12.42/15.17）。col1=100% 疑似过拟合；**col3 差距 ~31 点**持续存在。
9. **PS 定义不一致**：本地离线 PS = 剩余未匹配 expert 步（越高越差），论文 PS = 到完成的平均步数。
10. **无多 seed**：论文 5-seed 95% CI；本地全是单 seed。
11. **No-Unity 偏离**：tmow60 完全不用 Unity（仅 EvolvingGraph），论文 eval 用 Unity rollout。
12. **eval-C / 符号链接契约**：paperaligned builder 可能不生成 `eval_col_{1,2,3}` 符号链接，validator EVAL_LINKS 检查会失败。

---

## 5. 数据集构建失败的主要原因 (诊断结论，按重要性排序)

**P0 — 论文 ground truth 缺失（根本不可解项）**
78 任务、16-seen、20 scene、per-episode 计数全部未公布。每个构建都是近似，无法验证是否对齐论文训练分布。结构性约束，使后续所有"修复"都在猜测目标。证据：`wormi-paper-spec-2026-05-29.md:53-62`。

**P1 — 任务选择算法导致数据塌缩**
coverage-greedy 无 per-source 多样性 cap → 2 个 source 对象垄断、80% walk_kitchen 起手。从 paperlike-v2 到 balanced-aux **持续存在**，直接造成 repeated-walk 与 col3 泛化失败。source_unique gate 解了多实例歧义但未解分布塌缩。paperaligned-rewrite 的 stratified-by-source + 硬门 top1≤0.35 把 top1 从 0.80 降到 0.30，但该 run **尚无最终结果**。证据：`wormi-data-build-summary-2026-05-29.md:88-109`。

**P2 — class-level 观测与 instance-level expert 动作的根本冲突**
全图 class-level 折叠使 expert 选定的单一 instance 在文本中欠定（多实例 76-94%）。instance-grounding 修复实现了但从未训练验证，paperaligned 回归全图后又放弃。证据：`vh-balanced-reconstruction-semantic-audit-2026-05-28.md`。

**P3 — observation bloat 与 train/rollout 契约不一致**
全图 5-7k 字符淹没监督；转 compact 又引入格式不一致、action-conditioned 泄漏、evaluator first-id binding 三个独立 bug。证据：`tmow-data-preprocessing-...-2026-05-28.md:763-812`。

**P4 — per-WM 数据稀疏（早期）**
10-14 episode/scene 被点名为 paperlike-v2 SR=0 的"major contributor"。balanced 的 64/scene 缓解，但叠加 gate 后 scene_5 又掉到 36。

**P5 — Stage-2 训练不稳定污染归因**
三类崩溃 + seqmeta SR≈0 + threaded 非论文算法，使**无法干净判定剩余 SR 差距是数据还是训练造成**。所有"有结果"的 run 都来自被中断的 `last` checkpoint。证据：`vh-historical-results-summary-2026-05-29.md`。

> **综合判断**：数据集构建失败的最主要原因是 **P0（无 ground truth）叠加 P1（任务选择塌缩）**——前者使目标不可知，后者使即便结构合法的数据也产生退化的训练分布。P2/P3 是 P1 之外的语义/格式诱因，P5 则使整条因果链无法被实验干净证伪。

---

## 6. 仍待人确认的开放问题

1. 作者的确切 78 任务 / 16-seen / 20 scene / 6-seen 身份是什么？有无 code release 或可联系作者？
2. 论文的 class-level 观测是否真含同类多实例？还是作者通过未公开过滤保证对象唯一性？
3. repeated-walk 策略塌缩到底是数据问题还是 adapter/meta-learning 问题？（full-graph 与 compact 同 meta-learning 路径的受控对比尚未跑过）
4. seqmeta（论文忠实 Reptile）为何 SR≈0 而 threaded（不忠实）却 work？是 optimizer 状态恢复 bug 还是超参敏感？（中心阻塞点）
5. balanced-aux 的 67.71/64.29/18.67 是否被中断的 `last` checkpoint 污染？
6. N=6 world models 的"scene"语义：单一 init-graph 变体还是一个公寓的多变体域？决定 64/scene 假设与 tmow60 单变体设计是否过弱。
7. Stage-2 论文报告值用的是 Reptile β=0.1（sequential）还是 threaded direct-mean？
8. paperaligned 生产 build（variants_per_domain=8）能否达 1023？还是 970 是本地 VH 图库 feasibility 上限？
9. rollout temperature：shell 默认 0.0（greedy），dataclass 默认 1.0——各 run 实际用哪个？
10. No-Unity（仅 EvolvingGraph）评估是否构成对论文 Unity rollout 的有效复现信号？
11. `exact_action_sequence_overlap`（42-71）是否实质污染离线 exact-match / 抬高 seen-task SR？

---

证据文件根目录：`/root/WorMI/reports/virtualhome/`（`data-processing/`、`validation/`、`audits/`、`experiments/`）；
构建脚本：`/root/WorMI/tools/build_virtualhome_dataset*.py`、`build_tmow60.py`、`select_tmow60_variants.py`、`compact_virtualhome_observations.py`；
论文目标值：`/root/WorMI/reports/virtualhome/experiments/vh-reproduction-progress-2026-05-19.md:141-143`。

---

## 7. 突破：数据源审计 + 真实 plan pipeline（2026-05-30 更新）

### 7.1 决定性发现 — 病因是"凭空合成 plan"，而非数据缺失本身

前 12 次构建有一个**共同错误前提**：用手写模板 `_paperlike_program`（`tools/build_virtualhome_dataset.py:703`）**合成**专家 plan，再花大量精力修合成带来的副作用（塌缩、歧义、膨胀）。而**真实的 crowdsourced plan 一直躺在 `raw/.../executable_programs/` 里没被用**：

- 现有 pipeline **只读** `init_and_final_graphs`（场景图），从未读 `executable_programs`（真人 plan）/`withoutconds`（grep 确认 0 命中）。
- `executable_programs` 是 VirtualHome ActivityPrograms：**6201 条真人 plan**，自带自然语言指令（第二行）+ **instance-grounded** 动作序列（`<cup> (1.1001)`），`init_and_final_graphs`/`state_list` 各 6201 条 **1:1 全覆盖**。
- 这把 §5 的 P1/P2/P3 重新归因为**同一根因的下游症状**：真人 plan 自带 instance ID（解 P2 歧义）、真实意图分布（解 P1 塌缩）、自然语言指令（解 P3 模板化）。
- **证据**：`tools/build_virtualhome_dataset_realtasks.py` 顶部 docstring；本目录 `wormi-data-construction-failure-survey` 对话审计。

### 7.2 SayCanPay 对照（澄清，非数据源）

WorMI 的 baseline SayCanPay（`github.com/RishiHazra/saycanpay`）公开的 `virtualhome/data/oracle-plans/train` = **795 plan / 14 高层 intention / 7 公寓**，动作 **class-level、无 instance ID**，median 4 步。

- **不是** WorMI 数据的直接来源：795/14/7 ≠ 论文 1023/78/20；SayCanPay 14 个任务里几乎没有 puton/placein（论文 62/78 个主力 family）。
- 价值：① 确认数据血缘（SayCanPay 是从同一批 ActivityPrograms 简化来的，丢了 instance ID 和长度）；② 它的 `parse_action` 给每个对象硬绑 `(1)` —— **这就是 survey §3.5 "evaluator first-id binding" bug 的根**。
- 结论：SayCanPay **没能**递出 78-task / seen-unseen 划分（仓库只 ship train），P0（划分未知）依旧成立。真正该建在 raw `executable_programs` 上。

### 7.3 真实任务挖掘的可行性（6201 → 78 可达）

按论文 4 family 映射 raw 终端动作（**puton←PUTBACK**，PUTON=穿衣已排除；placein←PUTIN；turnon←SWITCHON；open←OPEN），唯一任务数：turnon **25**、open **26**、puton **186**、placein **35**。论文配额 9/7/30/32 **全可达**；placein 35→32 近乎全取（与论文吻合度高）。挖掘的任务全部真实合理（computer/television/plate→dishwasher/food→freezer…），**不再有** "Place phone in toilet" 这类合成怪任务。

### 7.4 attempt #13 — real-task pipeline（`tools/build_virtualhome_dataset_realtasks.py`）

| 项 | 内容 |
|---|---|
| 目标 | 用真实 plan 任务分布替换合成任务源，建一个 replay-clean、无塌缩、可复现的 Table-1-shaped 数据集 |
| 做了什么 | `RealTaskBuilder(Builder)` 子类，**只 override `select_tasks`**：从 6201 条 executable_programs 挖掘 (family,args) 真实目录 → 按配额选 78（placein 近全取、其余 top-K + per-source 多样性 cap）→ 场景感知 seen-16（stratified-by-source，约束在 seen 场景可行）。其余场景加载 / `_execute_paperlike_candidate` 重放 / 切分 / 写盘 **全复用 parent**。把 parent 的首动作 gate 换成**源对象多样性 gate**（真正反映塌缩的轴；room-walk 仅作信息）。 |
| 结果（v3，24→40 变体三版收敛） | **825 episodes / 4385 rows**；split train 254 / eval_a 77 / eval_b 175 / eval_c 319；seen/unseen 任务划分 16/62（有效 14/57）；6/14 场景划分。validator：**replay 0 失败、观测匹配、0 泄漏、loader 兼容**。源对象 top1 share **0.094**（历史 0.80 塌缩**彻底消除**），per-WM episode 26-58（历史 10-14）。 |
| 遗留问题 | ① **count 825 < 1023**：结构性上限——只有 6 个 seen scene，且 parent 把 `train_per_task` 硬均分为 24/任务，可行 slot 少的任务填不满、多的被截。加变体（24→40）几乎不解。② **有效 seen 任务 14/16**：场景感知用的是"类存在性"可行，但执行仍可能失败（物体不可抓/目标已满足）。真正补满需在**枚举后**按真实实现池选 seen 任务（`execute_slot` 已缓存，零额外开销）+ 弹性 per-task fill。③ validator 4420 "errors" 全是 split 命名契约不一致（`train_seen_task_seen_scene` vs 期望 `seen_seen`），继承自 parent，非数据损坏。④ **尚无 stage1/stage2/rollout 结果**——数据质量已验证，但能否真正复现 Table 1 SR 待训练。 |
| 产物 | `/root/autodl-tmp/wormi-data/virtualhome-realtasks-v3-20260530/`（+ `scene_inits.json`、`quality_report.json`、`validation-2026-05-30.json`、`*.mining.json`） |

### 7.5 §5 根因状态更新

- **P0（划分未知）**：仍成立。SayCanPay 未递出划分；78-task/seen-unseen 仍是 paper-compatible 重构。
- **P1（任务选择塌缩）**：**已解**。真实意图分布 + 源多样性 cap → 源对象 top1 0.80→0.094。
- **P2（class 观测 vs instance 歧义）**：**源头消除**。真人 plan 自带 instance ID；当前 pipeline 经 `_execute_paperlike_candidate` 重放仍 class-level 观测，但任务接地来自真实 instance，replay 0 失败。
- **P3（观测膨胀 / 模板化指令）**：指令改用真实分布；观测回归全图（论文 Figure A.2），膨胀本身是论文要求，非 bug。
- **P4（per-WM 稀疏）**：缓解（26-58 vs 10-14）。
- **P5（stage-2 不稳定）**：**未触碰**，仍是能否复现 Table 1 的中心阻塞点；本次只解决数据。

### 7.6 §6 开放问题更新

- #2（class 观测是否真含多实例 / 作者是否过滤）：**部分落定**——raw 真人 plan 自带 instance ID，歧义是合成与 SayCanPay class-level 简化引入的，非数据固有。
- 新增 #12：**6 个 seen scene 是否结构性限制 1023 可达性？** 论文 1023 的 per-split 计数（train/eval_a/eval_b/eval_c）未知，无法确认 825 是否已接近作者 seen 侧实际可达量。
- 新增 #13：**real-task 数据能否真正复现 Table 1 SR？** 需 stage1+stage2+rollout 验证；这是从"数据可用"到"复现成功"的下一步。

---

## 8. T1 专家回放门控 (Expert Replay gate) — 2026-05-31

把数据集 ground-truth 的 class 级 `action` 字符串喂进**完整 eval rollout pipeline**（`/root/WorMI/wormi/scripts/eval_vh_rollout.py`，no-Unity/EvolvingGraph，`_script_line_from_prediction -> _choose_node_id -> execute_one_step(instance_selection=False) -> _goal_satisfied`），不加载任何模型。脚本：`/root/WorMI/tools/expert_replay_vh.py`。全量 episode（无采样），`observation_format` 解析为 `full`（v3 无 obs_preprocessing meta，已断言）。

| split | episodes | EXPERT-action SR | GOLD-scriptline SR (eval env ceiling) | binding loss | fail-only-binding |
|---|---|---|---|---|---|
| test_seen_task_seen_scene | 77 | **87.01%** (67/77) | 98.70% (76/77) | 11.69% | 9 |
| test_seen_task_unseen_scene | 175 | **84.57%** (148/175) | 98.86% (173/175) | 14.29% | 25 |
| test_unseen_task_unseen_scene | 319 | **88.71%** (283/319) | 95.61% (305/319) | 6.90% | 22 |

(test_unseen_task_seen_scene 为空，跳过。)

**判定：FAIL（数据不应进入训练）。** 三列专家 SR 全部 <99%，与 PASS 门槛差 11–15 个百分点。

两类根因：
- **(A) `_choose_node_id` 实例绑定错误（主因，每列都有）**：class 级 action 被绑到错误实例。已逐步追踪 `TrimmedTestScene3_graph__d06_v05`（"Put cup on sink"）：`walk/open cupboard` 被绑到实例 130，gold 用 131；cup(2000) 在 cupboard 131 内，于是 130 被打开而 131 仍关闭，step4 `grab cup` 触发 CLOSE 前置条件失败，episode 永远到不了目标。这正是用户被坑过的 "first-id/启发式 id 绑定" bug。绝大多数失败（9/10、25/27、22/36）属于此类。
- **(B) build-time ScriptExecutor 契约 vs eval execute_one_step 契约分歧（col3 额外 ~4.4%）**：col3 的 gold-scriptline 控制只有 95.61%，即连数据自带的、已实例绑定的 `_meta.script_line` 逐步喂进 eval env 也有 14/319 到不了 `_goal_satisfied`（如 `d19_v29` juice->freezer：PUTIN 执行了但目标未判定满足）。说明 col3 的环境可达上限本身就 <99%，不只是绑定问题。

修复方向（在 EVAL pipeline，不是数据）：(1) 修 `_choose_node_id`/`_match_class` 绑定；或 (2) 让 eval 直接消费 `_meta.script_line` 做专家绑定；(3) 复查 col3 build-time vs eval 的 goal 判定/执行契约分歧。数据本身（class 级 action + gold script_line）是自洽的——validator 用 build-time ScriptExecutor 整脚本 replay=100%，问题在 class->instance 的 eval 重绑定环节。

### 8.1 根因落定 + 修复（2026-05-31，route 1，EVAL pipeline 内）

§8 的 FAIL 已在 EVAL pipeline 内根治（数据未改）。两层根因均确认并修复，决定性门控全绿。

- **(A) 绑定根因（主因）已解**：`_choose_node_id` 原用 proximity/held 把 class 级 action 绑实例，与 build-time expert 的 `select_task_instances`（按 goal 关系选实例）漂移。修复：新增 `_build_goal_binding`（仅用 `family/task_args` + reset 图 + live 图，**不读 `_meta.script_line`/`_meta.instance_selection`**）解析 source/target/source_container 的真实 graph node id，让 goal 结构主导绑定，proximity 仅作同类多角色（如 source-sink vs target-sink）的并列消歧/兜底。PUTIN/PUTBACK 的 source id 改为 goal id 优先于 held id。
- **(B) 执行契约根因（col3 上限封顶）已解**：上限 <99% 不是 goal 判定 bug，而是 `EnvironmentState(instance_selection=False)` 让 executor **无视脚本里的实例 id、按首个枚举实例重绑** class 级 script object（如 `OPEN <cupboard> (127)` 实际翻转 126，task 对象仍封在 127 → `GRAB` "inside other closed thing" → 目标永不达成）。修复：eval 与 gate harness 改用 `instance_selection=True`，executor 严格执行 (A) 解析出的 goal node id。`_goal_satisfied` **未放宽**；独立对照：gold `_meta.script_line` 经 `instance_selection=True` 三列均 100%。

| split | 专家 SR before | 专家 SR after | gold 上限 after | fail-only-binding after |
|---|---|---|---|---|
| col1 test_seen_task_seen_scene | 87.01% | **100.00%** (77/77) | 100.00% | 0 |
| col2 test_seen_task_unseen_scene | 84.57% | **100.00%** (175/175) | 100.00% | 0 |
| col3 test_unseen_task_unseen_scene | 88.71% | **100.00%** (319/319) | 100.00% | 0 |

**判定：A2/A3/A4 门控全部 PASS（>=0.99，binding_loss=0）。** 改动文件：`wormi/scripts/eval_vh_rollout.py`、`tools/expert_replay_vh.py`。详细变更与 no-cheating 证据见 `wormi-rootfix-status-2026-05-31.md` Step A2+A3。

## 9. 首次端到端模型 rollout 评测（2026-05-31）

v3 数据集（825 ep / train 254 / 6 world model）跑通 stage1（per-scene SFT）+ stage2（threaded-meta，`WORMI_ALLOW_UNSAFE_THREADED_META=1`、直接均值、240 step、train_loss≈0.41），用 `eval_vh_rollout.py`（环境 rollout，paper 口径）全量评测。模型 = `wormi-checkpoints/wormi-vh/wormi-vh-n6/last`，检索 K=3，max_steps=30，observation=full。结果文件 `wormi-vh-n6/vh-rollout-full/vh-rollout-summary.json`。

| 列 | 我们 SR | 我们 PS | invalid/ep | exec/ep | 论文 SR | 论文 PS | retrieved WM |
|---|---|---|---|---|---|---|---|
| col_1 seen-seen | **22.08%** (77) | 23.82 | 8.43 | 15.39 | 85.78 | 10.76 | [4,2,3] |
| col_2 seen-unseen | **20.00%** (175) | 24.41 | 8.13 | 16.28 | 80.26 | 12.42 | [4,0,5] |
| col_3 unseen-unseen | **14.33%** (300) | 25.99 | 13.69 | 12.30 | 66.12 | 15.17 | [3,2,4] |

**判定：管线干净、模型很弱（差距数量级）。** eval 这条管线已被 gold 动作证伪过（§8 expert-SR=100%），所以瓶颈在训练/推理侧，非评测 bug。

**新强信号 — invalid_actions 极高（8~14/ep）**：模型自生成时大量动作被环境拒绝（col_3 约半数动作无效），而 expert-replay 用 gold class-action 是 100%。这指向「推理时 模型输出→scriptline 落地/grounding 缺口」，候选根因按先验排序：
- P-INF（待诊断，最便宜）：模型预测的 action 字符串与 `_script_line_from_prediction` 期望格式/对象词表不吻合，或 goal-binding 在 model 路径上未生效（gold 路径已修，model 路径需复核），导致动作打不进环境——若属实，SR 被推理层封顶，与数据量无关。
- P-DATA：v3 仅 254 train / 6 WM，远小于论文 ~1023 / 16 seen task，Reptile 元学习先验不足。
- P-META：threaded-meta 直接均值近似 ≠ 论文 faithful 顺序 Reptile（seqmeta 历史 SR≈0，疑另有 bug）。
- P-SFT：stage1 world model 可能欠拟合（loss≈0.4 量级）。

### 9.1 P-INF 诊断结论（2026-05-31，零 GPU，纯挖 per-step detail 日志）

挖 `vh-rollout-full/vh-rollout-{col}.jsonl`（脚本每步落了 prediction/script_line/parse_error/execution_error）：

- **parse_fail ≈ 0**（col1/2=0%，col3=5%）→ 格式/grounding/§8 binding 在 model 路径也生效，**P-INF（推理落地缺口）排除**。
- invalid 大头是 **"precondition failed"（33–48%）+ 大量可执行但无进展的重复动作**。
- **实锤：模型退化成 majority-action 死循环**。失败 episode 把 `walk bedroom` 连发 14+ 次；**60/77 episode 有某预测重复 ≥5 次**；成功的几乎全是 `walk+switchon` 两步平凡任务。
- **H1（训练 obs 渲染器 ≠ eval 渲染器 / OOD）排除**：两者用**同一个** `tools.build_virtualhome_dataset.format_observation`（eval line34 import；训练数据 line547 渲染），且 obs 含 `(character, hold, *)`/`(character, inside, *)`/`(character, close, *)`，agent 位置手持齐全。
- **根因 = 训练侧策略塌缩**。训练集动作分布：`walk 58.4%` / open 13.4% / grab 12.3% / putin 7.0% / put 5.3% / switchon 3.7%；`walk kitchen` 单动作占 9.3%。仅 1288 transition / 254 traj（median 6 步）。模型学成"无脑 walk <room>"的多数类先验，看到 `(character, inside, bedroom)` 也不切 grab/put。

**根因状态更新**：P-INF=排除；**P-DATA（量小 + 动作极不均衡）= 主因（高置信）**；P-META（直接均值放大主导先验）/ P-SFT（欠拟合）= 共因待隔离。

**下一步（隔离实验，便宜优先）**：单独 rollout 一个 stage1 world-model 检查点（无 meta）在其本场景 test 上，把 SFT 质量 vs meta 均值塌缩拆开 → 再决定扩数据 / 修 meta / 调 recipe。避免盲投数 GPU-小时。

### 9.2 隔离实验结论 —— 根因翻盘到 stage2（2026-05-31，零额外训练）

三个便宜探针（脚本 `tools/stage1_teacher_forcing_probe.py`、`tools/stage1_single_rollout_probe.py`，复用 eval `_eval_episode` 保证环境/binding/goal 契约一致）：

**(1) stage1 teacher-forcing（gold obs 链，无 rollout）—— SFT 是好的**
- scene_0 train: 94.2%（grab/open/put/switchon 全 ~100%，预测动词分布均衡、不塌缩）。
- scene_0 test（held-out）: 66.2%。→ SFT 学到了 obs→action，**P-SFT 欠拟合排除**。

**(2) 单 stage1 world-model 自由 rollout（无 meta、无 base、无 adapter）—— 单模型其实很强**

| scene | 0 | 1 | 2 | 3 | 4 | 5 | 均值 |
|---|---|---|---|---|---|---|---|
| 单模型 rollout SR | 63.6 | 53.8 | 36.4 | 55.6 | 72.2 | 80.0 | **≈60.3%** |

对照：meta 集成的完整 WorMI col_1(seen-seen) = **22.08%**。**单个 1B world model 比集成后的整套 WorMI 高 ~3 倍。**

**(3) 冒烟枪 —— stage2 adapter 停在初始化**
- stage2 `last/pytorch_model.bin` 仅 22 个张量（cross_attention_hooks.{0,1}.*）。所有 MLP 投影 norm ≈ **32.00**（gate/up/down: 32.0187/32.0135/31.9950…）。
- PyTorch 默认 `Linear(d,d)` 初始化权重 norm = `sqrt(d/3)`，d=3072 → `sqrt(1024)=32.0`，与观测**精确吻合**。即 MLP adapter 仅偏离初始化 ~0.06%，**几乎没被训练**。
- 预算：λ_I=30 × λ_M=8 = 每 trainer 仅 **240 个梯度步**，inner LR=1e-5，且 `WORMI_THREADED_META_USE_BETA=0`（直接均值）每 30 步把 6 个 trainer 的 adapter 平均一次 → 更新互相抵消。

**根因落定（覆盖 §9 的 P-DATA 主因假设）**：
- 数据**够用**（足以训出均值 60% SR 的单模型）；stage1 SFT **好**；eval 管线**干净**。
- **真主因 = stage2 meta 集成严重欠训练**：cross-attention adapter 停在初始化，冻结 base 3B 拿不到 world-model 知识，退回自身先验 → majority 动作 walk(58%) 塌缩 → 60/77 episode 死循环。
- 翻盘逻辑：单模型本身是充分 SFT 的策略（直接产动作）；WorMI 的策略是**冻结 base 3B**，全靠 adapter 把 world 知识注进来。adapter≈随机 → 等于裸跑没学过动作格式的 base 3B。

**修复方向（stage2，非数据）**：大幅加 stage2 预算让 adapter 真正离开初始化 —— 提高 adapter LR（1e-5 对 from-init 的 MLP 太小）、增大 λ_M（8→数十）、并复核直接均值是否在抵消更新。重训 stage2（stage1 复用）后重评，看集成 SR 是否向单模型 60% 收敛甚至超过（WorMI 的卖点是集成+检索应 > 任何单模型）。
