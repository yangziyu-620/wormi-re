# VirtualHome 当前数据处理说明

日期：2026-05-28

本文记录当前用于 `virtualhome-balanced-aux-compact17-sourceunique-20260528` 这组实验的 VirtualHome 数据处理流程。这里的结论需要保守表述：这是一份固定随机种子的、可审计的、paper-compatible reconstruction，不是论文作者公开的 exact split。

## 数据位置

生成后的数据集：

```text
/root/autodl-tmp/wormi-data/virtualhome-balanced-aux-compact17-sourceunique-20260528
```

主 manifest：

```text
/root/autodl-tmp/wormi-data/virtualhome-balanced-aux-compact17-sourceunique-20260528/virtualhome_manifest.json
```

验证报告：

```text
/root/WorMI/reports/virtualhome/validation/vh-balanced-aux-compact17-sourceunique-validation-2026-05-28.json
```

构建脚本：

```text
/root/WorMI/tools/build_virtualhome_dataset_balanced.py
```

数据 loader：

```text
/root/WorMI/wormi/datasets/virtualhome.py
```

## 构建命令

当前数据集使用下面的命令生成：

```bash
uv run python tools/build_virtualhome_dataset_balanced.py \
  --raw-dir /root/autodl-tmp/wormi-data/raw/programs_processed_precond_nograb_morepreconds \
  --vh-src /root/autodl-tmp/wormi-data/virtualhome-src \
  --output-dir /root/autodl-tmp/wormi-data/virtualhome-balanced-aux-compact17-sourceunique-20260528 \
  --seed 42 \
  --variants-per-domain 12 \
  --semantic-gate source_unique \
  --observation-mode tmow_compact \
  --compact-num-edges 17 \
  --compact-next-mode compact \
  --overwrite
```

关键参数含义：

- `seed=42`：固定 task、scene、episode 等抽样结果，保证之后可以复现。
- `variants-per-domain=12`：每个 scene domain 使用多个官方 VirtualHome init-graph variant，避免旧方案中单个 scene 数据太稀的问题。
- `semantic-gate=source_unique`：在构建阶段过滤掉 source/action object 过于歧义的候选 episode。
- `observation-mode=tmow_compact`：不直接 dump 完整图，而是使用 TMoW 风格的紧凑图文本。
- `compact-num-edges=17`：限制 compact observation 的边数量，减少文本长度，同时保留局部物体、房间和关系信息。
- `compact-next-mode=compact`：当前 observation 和 next observation 使用同一种 compact 表示，避免输入输出格式不一致。

## 核心 split 规则

当前 split 规则是：

```text
train  = seen_task   intersect seen_scene
eval A = seen_task   intersect seen_scene，并且 episode 级别从 train 中 hold out
eval B = seen_task   intersect unseen_scene
eval C = unseen_task intersect unseen_scene
```

最重要的一点：eval C 不是用总数相减得到的。eval C 只从合法的 `unseen_task intersect unseen_scene` 候选池中采样。这样可以避免把下面这些数据错误混进去：

- `unseen_task intersect seen_scene`
- `seen_task intersect seen_scene` 的剩余 episode
- `seen_task intersect unseen_scene` 的剩余 episode

当前生成的数据规模：

| split | episodes | rows |
| --- | ---: | ---: |
| train | 384 | 1977 |
| eval A: seen task + seen scene | 96 | 431 |
| eval B: seen task + unseen scene | 224 | 1133 |
| eval C: unseen task + unseen scene | 319 | 1749 |
| total | 1023 | 5290 |

训练数据被物化成 6 个 world-model 目录：

| world model dir | train episodes | train rows |
| --- | ---: | ---: |
| `scene_0` | 64 | 356 |
| `scene_1` | 64 | 333 |
| `scene_2` | 64 | 321 |
| `scene_3` | 64 | 331 |
| `scene_4` | 64 | 338 |
| `scene_5` | 64 | 298 |

这对应当前的工作假设：论文里的 `N=6` world models 被解释为 6 个 seen scene domains。论文没有公开 exact scene IDs、task IDs 或 episode IDs，所以这个点必须写成 assumption，不能写成作者官方 split。

## Task split

当前重构遵循论文层面的 task 数量：

```text
seen tasks   = 16
unseen tasks = 62
total tasks  = 78
```

seen task 的四类任务比例尽量贴近论文 VirtualHome 设置：

```text
TurnOn  = 2
Open    = 1
PutOn   = 6
PlaceIn = 7
```

风险：`Open=1` 比较脆弱，因为只选一个 Open task，结果会比较依赖这个具体 task ID。最终写报告时，可以把这个 split 作为主 split，同时如果时间允许再跑几个不同 seed 的 robustness split。

## Scene domain 处理

论文只报告了 `6/14` seen/unseen scene 的数量，没有公开 exact scene split。当前重构把一个 scene domain 定义为来自同一个 base apartment 的多个官方 VirtualHome init-graph variants。

为什么不要求完整的 task-scene 笛卡尔积：

- VirtualHome 总共只有 1023 个 episode，却覆盖 78 个 tasks 和 20 个 scenes。
- 平均每个 task-scene cell 不到 1 条 episode。
- 如果强行要求每个 `seen_scene x seen_task` cell 都有多条 train/eval episode，会迫使我们只选高覆盖 cell，导致选择偏差。

因此当前 builder 使用 scene、task、family 的 soft balancing，而不是假设原始数据是完整笛卡尔积。这能保持规模接近论文，同时避免不现实的数据密度要求。

## Episode 构建方式

每个候选 episode 都是在官方 VirtualHome init graph 上，通过 EvolvingGraph 执行 expert/planner program 得到。对于每个成功执行的 episode，builder 生成 step-level rows：

```json
{
  "instruction": "...",
  "observation": "step i 的 compact graph text",
  "action": "step i 的 expert action",
  "next_observation": "step i+1 的 compact graph text"
}
```

之后 validation 会重新 replay 这些 rows，检查存下来的 observation 和 next_observation 是否和 EvolvingGraph 执行结果一致。

## Semantic gate

当前数据使用：

```text
semantic_gate = source_unique
```

这是构建阶段的语义过滤。目的不是提高指标，而是避免训练目标本身有歧义。旧数据里一个重要问题是：同一个文本 action class 可能对应图里的多个 object instance，导致模型看到的 supervision target 语义不稳定。

我们也尝试过更严格的 `source_unique_target_room_unique`，但它过滤太多，覆盖率下降明显。当前 `source_unique` 是折中方案：去掉最严重的 source-object instance binding 噪声，同时保留足够多的 train/eval episode。

## Observation 处理

当前 observation 设置：

```text
mode = tmow_compact
compact_num_edges = 17
compact_next_mode = compact
```

这样做的原因：

- 完整 VirtualHome graph 很长，而且包含大量和当前动作无关的噪声。
- Compact observation 更强调局部物体、房间、containment、support 和 object state 等与动作相关的信息。
- train 和 rollout eval 使用同一种 compact 格式，避免训练时看到一种 observation，测试时看到另一种 observation。

代价：

- Compact observation 更干净，但可能丢掉长程 rollout 恢复需要的信息。
- 这可以解释为什么 offline action-following 分数高于真实 rollout 分数。

## World model 的三类 auxiliary tasks

当前 `VirtualHomeDataset` 在 world-model training 加载时，会把每条 raw transition 展开成 3 条 world model 样本：

| auxiliary task | 输入 | 目标 |
| --- | --- | --- |
| behavior cloning | instruction + current observation | action |
| dynamics | instruction + current observation + action | next observation |
| affordance | current observation | feasible action |

这对应 reviewer 描述的 world-model 训练信号：

```text
1. 根据 state/action/instruction 预测 next observation，用来学习 transition dynamics
2. 根据 state 识别 feasible action，用来学习 affordance
3. 根据 instruction/state 预测 action，用来学习 instruction-conditioned behavior
```

validation 报告确认了这个展开：

```text
raw rows       = 5290
action samples = 5290
world samples  = 15870
```

也就是说，每条 transition 都贡献了 3 条 world-model auxiliary examples。

## 泄漏控制

builder 在两个层面防止 train/test 泄漏：

```text
trajectory_id_overlap = 0
exact_row_overlap = 0
```

split 是 episode-level，不是 transition-level。这点很关键：如果只按 transition 行切分，同一条轨迹的相邻状态和动作会同时出现在 train/eval 中，eval A 会虚高。

已知残余问题：

```text
exact_action_sequence_overlap = 49
same_task_exact_action_sequence_overlap = 49
```

这不表示同一个 episode 泄漏了，而是不同 episode 可能共享完全相同的 action sequence。在 VirtualHome 中这是合理现象，因为很多任务会退化成类似 `walk, grab, walk, put` 的模板。但它仍然要作为限制写清楚，因为它可能轻微抬高 offline sequence-style 指标。对于 rollout 来说，这种重合帮助较小，因为 rollout 仍然需要正确状态 grounding。

## Validation 证据

当前 validation 结果：

```text
errors = []
total_rows = 5290
trajectories = 1023
train_test_overlap = 0
trajectory_id_overlap = 0
exact_row_overlap = 0
```

Replay 检查：

```text
replay.failures = 0
replay.obs_mismatches = 0
replay.next_obs_mismatches = 0
replay.goal_failures = 0
```

Loader 和 chat-template 检查：

```text
action_samples = 5290
world_samples = 15870
loss_mask_samples = 21160
bad_count = 0
min_supervised_tokens = 4
max_supervised_tokens = 728
```

这些检查说明：

- JSONL 文件满足 WorMI dataset loader 的格式要求；
- LLaMA-3 chat template 能产生有效的 assistant supervised tokens；
- 每个 episode 都能在 EvolvingGraph 中 replay；
- 存储的 observation 和 next_observation 与 replay 后的 graph state 一致；
- expert trajectories 的最终目标状态是可达的。

Warnings：

- 部分 scene cache variants 没有被采样到 row。这是过滤和采样后的正常结果。
- validator 报告 220 个 effective scene variant keys，而论文层面是 20 个 scene domains。这是命名粒度问题：builder 每个 domain 使用多个 variants。
- `walk` 有 42 / 3201 次 observation unchanged，低于当前 warning threshold，主要来自 compact observation 截断，不是执行失败。

## 为什么旧数据失败

之前效果很差的数据主要有这些问题：

- 每个 world model 的训练数据太稀；
- full graph 文本过长、过噪；
- class-to-instance grounding 有歧义；
- train/eval split 容易出现 residual sampling 或泄漏；
- world model training 没有完整实现三类 auxiliary tasks；
- observation/action 文本不匹配时，可能导致监督信号 silently 失效。

当前处理逐项修正了这些问题：

- 6 个 world models 每个都有 64 个 train episodes；
- compact observation 降低无关 graph 噪声；
- `source_unique` 去掉最严重的 source-object 歧义；
- eval C 只从 `unseen_task intersect unseen_scene` 采样；
- 阻断 exact row 和 trajectory overlap；
- loader 展开 behavior cloning、dynamics、affordance 三类任务；
- chat-template validation 检查监督 token 不为空。

## 这份数据能证明什么，不能证明什么

它能证明：

```text
当前数据是 paper-compatible、fixed-seed、可执行、loader-valid、chat-template-valid，
并且没有 episode 级别或 exact-row 级别的 train-test 泄漏。
```

它不能证明：

```text
它不是论文作者的 exact split。
它不能证明 compact observation 包含 rollout 成功需要的全部信息。
它不能单独解释和论文结果之间的差距。
它不能消除模型规模、stage2 训练稳定性、rollout evaluator 实现、超参数等差异。
```

因此，这份数据可以作为下一次 clean experiment 的可靠基础，但除非作者 exact VirtualHome split 可用，否则报告里应该称它为 paper-compatible reconstruction。

## 当前结果背景

最新 eval 已经跑完，但 stage2 训练没有 clean completion。它在 threaded trainer 中因为 autograd inplace conflict 中途失败，不过失败前保存了 `last` checkpoint，后续 eval 使用的是这个 checkpoint。

Offline JSONL-style eval：

| split | SR | PS |
| --- | ---: | ---: |
| seen task + seen scene | 67.71% | 1.34 |
| seen task + unseen scene | 64.29% | 1.25 |
| unseen task + unseen scene | 18.67% | 4.18 |

VirtualHome rollout eval：

| split | SR | PS |
| --- | ---: | ---: |
| seen task + seen scene | 61.46% | 13.72 |
| seen task + unseen scene | 38.39% | 20.18 |
| unseen task + unseen scene | 17.00% | 25.39 |

解释：

- 新数据比旧失败版本明显更好，因为 seen-task performance 已经不是随机水平。
- 和论文的剩余差距不能单独归因于数据，至少要先让 stage2 clean training 完整跑完。
- 下一步最重要的是 clean sequential/non-threaded stage2、base-only baseline、no-world-model baseline，以及 auxiliary-task ablation。
