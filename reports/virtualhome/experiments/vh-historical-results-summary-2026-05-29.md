# VirtualHome 历史结果总表

日期：2026-05-29

本文汇总目前能从 `reports/` 和 `/root/autodl-tmp/wormi-logs/` 追溯到的
VirtualHome 历史实验结果。注意：部分旧 checkpoint 已经按磁盘清理要求删除，
所以有些结果只能从历史报告和日志中恢复；表中的“建议引用”表示是否适合作为
正式实验结果引用。

## 结论速览

| 优先级 | run / 数据版本 | 结论 |
| --- | --- | --- |
| 当前主要可分析结果 | `balanced-aux-compact17-sourceunique-20260528` | 数据方向有效，rollout 有非随机结果；但 stage2 失败，不能算 clean final。 |
| 历史最好可用结果 | `taskaware-threaded-clean-bs1-ga4-20260521` | 历史最强 local result；可作为“曾经可用的工程路径”对照，但不是严格论文 Algorithm 1。 |
| 明确失败结果 | `seqmeta` | 顺序 Reptile 实现导致 action drift，Table1 接近 0。 |
| 明确失败结果 | `paperlike-v2-fixed` / `tmow-compact` | 训练或数据能跑，但策略塌缩到 walk / rollout 极差。 |
| 当前进行中 | `wormi-paperaligned-20260529` | stage2 正在跑，尚无最终 Table1 / rollout。 |

## 仅完整最终数据统计表

纳入标准：至少有一项最终 eval 完整覆盖三列 `c1/c2/c3`。排除 partial、被中断、进行中、无 summary 的结果。`-` 表示该 run 没有找到完整最终数据。

| 日期 | run / 数据版本 | Table1 SR c1/c2/c3 | Table1 平均 SR | Table1 PS c1/c2/c3 | Table1 平均 PS | Rollout SR c1/c2/c3 | Rollout 平均 SR | Rollout PS c1/c2/c3 | Rollout 平均 PS | 是否 clean | 统计备注 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2026-05-19 | 初始 `wormi-vh/wormi-vh-n6` | 44.83 / 0.67 / 1.00 | 15.50 | 0.59 / 3.06 / 3.10 | 2.25 | 24.14 / 34.23 / 32.33 | 30.23 | 26.72 / 24.29 / 25.15 | 25.39 | 否 | 完整 eval，但早期数据/训练/evaluator 仍未对齐论文。 |
| 2026-05-19 | `wormi-vh-beta01` | - | - | - | - | 10.34 / 17.45 / 13.33 | 13.71 | 28.41 / 26.44 / 27.75 | 27.53 | 否 | 只有完整 rollout；策略塌缩到 repeated `walk`。 |
| 2026-05-21 | `taskaware-split` salvage | 81.25 / 54.84 / 24.33 | 53.47 | 0.38 / 1.06 / 1.95 | 1.13 | 93.75 / 63.98 / 36.00 | 64.58 | 6.16 / 13.75 / 21.16 | 13.69 | 否 | 有完整 eval，但 stage2 以 `release unlocked lock` 异常结束。 |
| 2026-05-21 | `taskaware-seqmeta-stage2-bs1-ga4-20260521` | 0.00 / 1.08 / 0.33 | 0.47 | 4.06 / 4.06 / 4.01 | 4.04 | - | - | - | - | 否 | 只有完整 Table1；seqmeta 训练路径失败。 |
| 2026-05-21 | `taskaware-threaded-clean-bs1-ga4-20260521` | 100.00 / 89.78 / 35.67 | 75.15 | 0.00 / 0.19 / 1.92 | 0.70 | 96.88 / 91.94 / 43.33 | 77.38 | 4.84 / 6.20 / 19.30 | 10.11 | 部分否 | 历史最好完整结果；工程可用，但不是严格 Algorithm 1。 |
| 2026-05-27 | `paperlike-v2-fixed-full-20260527` | 0.00 / 0.00 / 0.00 | 0.00 | 4.84 / 5.22 / 5.11 | 5.06 | 0.00 / 0.00 / 0.00 | 0.00 | 30.00 / 30.00 / 30.00 | 30.00 | 否 | greedy rollout 完整但失败；非-greedy rollout 不完整，未统计。 |
| 2026-05-28 | `paperlike-tmow-compact-fill17-20260528` | 0.00 / 0.47 / 0.33 | 0.27 | 4.56 / 4.90 / 4.74 | 4.73 | - | - | - | - | 否 | 只有完整 Table1；rollout 因 observation contract 问题不作为最终数据。 |
| 2026-05-28 | `paperlike-tmow-compact-aa-fill17-20260528` | 0.00 / 0.47 / 0.33 | 0.27 | 4.84 / 5.23 / 5.14 | 5.07 | - | - | - | - | 否 | 只有完整 Table1；rollout partial/停止，不统计。 |
| 2026-05-28 | `balanced-aux-compact17-sourceunique-20260528` | 67.71 / 64.29 / 18.67 | 50.22 | 1.34 / 1.25 / 4.18 | 2.26 | 61.46 / 38.39 / 17.00 | 38.95 | 13.72 / 20.18 / 25.39 | 19.76 | 否 | 有完整 eval；stage2 autograd inplace conflict，非 clean final。 |

## 历史结果表

| 日期 | run / 数据版本 | Table1 SR c1/c2/c3 | Rollout SR c1/c2/c3 | 状态 | 建议引用 | 成功或失败原因 |
| --- | --- | ---: | ---: | --- | --- | --- |
| 2026-05-19 | 初始 `wormi-vh/wormi-vh-n6` | 44.83 / 0.67 / 1.00 | 24.14 / 34.23 / 32.33 | 部分可用，差距大 | 不建议作为最终结果 | 初始 evaluator 从 offline 过渡到 rollout；剩余差距来自 data/split、训练偏差、prompt/action parsing、超参等。 |
| 2026-05-19 | `wormi-vh-beta01` | 未记录完整 Table1 | 10.34 / 17.45 / 13.33 | 失败 | 不建议 | 修了 `β=0.1` paper-alignment bug，但本地 batch/训练设置下策略更差，生成动作进一步塌缩到 repeated `walk`。 |
| 2026-05-21 | `taskaware-split` salvage after failed stage2 | 81.25 / 54.84 / 24.33 | 93.75 / 63.98 / 36.00 | 可诊断，但不是 clean | 只作诊断引用 | stage2 最后 `RuntimeError: release unlocked lock`，保存的 `last` 可评估但不是 clean completion；unseen split invalid/precondition 问题仍明显。 |
| 2026-05-21 | `taskaware-seqmeta-stage2-bs1-ga4-20260521` | 0.00 / 1.08 / 0.33 | 未作为可靠结果记录 | 失败 | 不建议 | 顺序 Reptile 路径看似更 paper-faithful，但训练后 first-step action drift，quick gate 已经 0；失败源在 stage2 训练路径，不是数据或 Table1 evaluator。 |
| 2026-05-21 | `taskaware-threaded-clean-bs1-ga4-20260521` | 100.00 / 89.78 / 35.67 | 96.88 / 91.94 / 43.33 | 历史可用 | 可作历史最好 local 对照 | threaded stage2 工程路径稳定，结果明显可用；但不是严格 Algorithm 1，因为多线程共享模型/锁调度，不是干净的每任务 inner loop 后 Reptile 聚合。 |
| 2026-05-26 | `paperlike-v1-20260526` | 无 | 无 | 失败 | 不建议 | stage2 hang/停滞；无 `last` checkpoint、无 Table1 summary、无 rollout。 |
| 2026-05-27 | `paperlike-v2-fixed-full-20260527` | 0.00 / 0.00 / 0.00 | greedy: 0.00 / 0.00 / 0.00；非 greedy 只见 c1=6.25, c2=3.30，c3 未完整 | 失败 | 不建议 | 数据修了一部分 redundancy，但模型策略仍严重失败；rollout 长时间运行且基本 0，说明动作策略没有学起来。 |
| 2026-05-28 | `paperlike-tmow-compact-fill17-20260528` | 0.00 / 0.47 / 0.33 | 未完成可靠 full rollout | 失败 | 不建议 | 第一版 compact 存在 train/eval observation mismatch 和 action-conditioned observation 风险；rollout 不可作为有效 compact 评估。 |
| 2026-05-28 | `paperlike-tmow-compact-aa-fill17-20260528` | 0.00 / 0.47 / 0.33 | partial: 3.12 / 1.42 / 未完整；另有停止时 3.12 / 1.42 / 0.74 左右的 partial 记录 | 失败 | 不建议 | 修了 action-agnostic compact 和 rollout 渲染，但策略仍 dominated by repeated executable `walk ...`；中途停止，col3 未完整或不可报告。 |
| 2026-05-28 | `balanced-aux-compact17-sourceunique-20260528` | 67.71 / 64.29 / 18.67 | 61.46 / 38.39 / 17.00 | 当前有效诊断结果，但不是 clean final | 可作当前阶段结果，需注明 stage2 失败 | 新数据处理、semantic gate、compact observation、三类 auxiliary tasks 让结果明显非随机；但 stage2 因 autograd inplace conflict 失败，eval 用失败前保存的 `last`。 |
| 2026-05-29 | `wormi-paperaligned-20260529` | 未产出 | 未产出 | 进行中 | 暂不能引用 | 新 paper-aligned full-graph 数据重构；stage1 已有日志，stage2 正在运行，尚无最终 Table1/rollout。 |

## 关键实验说明

### `threaded-clean` 是历史最好结果，但不是严格论文算法

`threaded-clean` 的结果很强：

| split | Table1 SR | Rollout SR |
| --- | ---: | ---: |
| seen task + seen scene | 100.00% | 96.88% |
| seen task + unseen scene | 89.78% | 91.94% |
| unseen task + unseen scene | 35.67% | 43.33% |

它说明数据、world models、rollout evaluator 不是必然导致失败的根因。但是它的
stage2 不是严格论文 Algorithm 1：实现上是 threaded trainers 共享一个模型并用
lock 调度，不能等同于“每个任务从同一 meta 参数复制、inner update、再 Reptile
聚合”的干净实现。

证据：

```text
reports/virtualhome/experiments/vh-stage2-meta-learning-clean-vs-seqmeta-2026-05-27.md
```

### `seqmeta` 说明顺序 Reptile 路径本身有问题

`seqmeta` 是一次尝试更接近论文 Reptile loop 的实现，但结果几乎为 0：

| split | Table1 SR |
| --- | ---: |
| seen task + seen scene | 0.00% |
| seen task + unseen scene | 1.08% |
| unseen task + unseen scene | 0.33% |

早期 checkpoint quick gate 已经是 0，并且 per-step dumps 显示第一步动作漂移，
例如预测 `walk livingroom`，目标却是 `walk drawing`。这说明失败主要来自新的
stage2 训练路径，而不是 Table1 evaluator 或数据抽样。

### `balanced-aux-compact17-sourceunique` 是当前数据方向的证据

这组结果比旧 compact / v2 fixed 明显好：

| split | Table1 SR | Rollout SR |
| --- | ---: | ---: |
| seen task + seen scene | 67.71% | 61.46% |
| seen task + unseen scene | 64.29% | 38.39% |
| unseen task + unseen scene | 18.67% | 17.00% |

成功点：

- eval C 只来自 `unseen_task intersect unseen_scene`，不是总数相减；
- `source_unique` semantic gate 降低了 instance ambiguity；
- compact observation 与 rollout observation contract 对齐；
- world model loader 展开三类辅助任务：BC、dynamics、affordance；
- validation 显示 replay、obs/next_obs mismatch、goal failure 都为 0。

失败/限制：

- stage2 没有 clean finish；
- train log 报 autograd inplace conflict；
- rollout 的 seen/unseen 和 unseen/unseen 仍明显低，说明长程策略和 unseen-task 泛化仍弱。

证据：

```text
/root/autodl-tmp/wormi-logs/vh-table1-balanced-aux-compact17-sourceunique-20260528-table1/eval.log
/root/autodl-tmp/wormi-logs/vh-rollout-balanced-aux-compact17-sourceunique-20260528-rollout/eval.log
/root/autodl-tmp/wormi-logs/vh-wormi-balanced-aux-compact17-sourceunique-20260528-stage2/train.log
reports/virtualhome/data-processing/vh-current-data-processing-2026-05-28-zh.md
```

## 失败原因分类

| 类型 | 相关 run | 具体表现 | 处理结论 |
| --- | --- | --- | --- |
| Stage2 工程失败 | `paperlike-v1`, `taskaware-split salvage`, `balanced-aux` | hang、`release unlocked lock`、autograd inplace conflict | 需要修 stage2 trainer；不应把这类 run 称为 clean final。 |
| 顺序 Reptile 实现失败 | `seqmeta` | quick gate 0，first-step action drift | 不能把当前 `seqmeta` 当 paper-faithful 成功实现。 |
| 数据/observation contract 错误 | early `tmow-compact-fill17` | train 用 compact，rollout 渲染/构建逻辑不一致，甚至 action-conditioned observation | 这类 rollout 无效，应废弃。 |
| 策略塌缩 | `beta01`, `paperlike-v2-fixed`, `tmow-compact-aa` | repeated `walk ...`，PS 接近 30，SR 接近 0 | 说明模型没学到可执行长程策略，不能只看格式通过。 |
| 非 clean 但有诊断价值 | `taskaware salvage`, `balanced-aux` | 有可用 checkpoint，但训练异常结束 | 可用于分析，不可作为最终复现结果。 |

## 当前文件位置

历史实验报告：

```text
reports/virtualhome/experiments/
```

最重要的两个历史报告：

```text
reports/virtualhome/experiments/vh-stage2-meta-learning-clean-vs-seqmeta-2026-05-27.md
reports/virtualhome/experiments/vh-reproduction-progress-2026-05-19.md
```

TMoW / compact 失败分析：

```text
reports/virtualhome/data-processing/tmow-data-preprocessing-for-wormi-reproduction-2026-05-28.md
```

当前数据处理说明：

```text
reports/virtualhome/data-processing/vh-current-data-processing-2026-05-28-zh.md
```

当前 paper-aligned 重构进度：

```text
reports/virtualhome/data-processing/wormi-paperaligned-rewrite-2026-05-29.md
```

日志根目录：

```text
/root/autodl-tmp/wormi-logs/
```

checkpoint 根目录：

```text
/root/autodl-tmp/wormi-checkpoints/
```

注意：2026-05-29 的清理已经删除了部分旧 checkpoint 和 summary 文件，所以之后
追溯旧结果应优先看本报告和 `reports/virtualhome/experiments/`。
