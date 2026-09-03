---
feishu_title: "实验报告"
feishu_url: "https://icnbwz7kd1ui.feishu.cn/wiki/PdLUwsj9FiepqEkVXkLc2ermnmg"
feishu_wiki_token: "PdLUwsj9FiepqEkVXkLc2ermnmg"
feishu_object_token: "RyJVdGcuKo74uJxi1uOc9RC9nne"
feishu_revision: 5
---

<title>实验报告</title>

<callout emoji="⛔">
**S1 / G1 唯一正式标签：`BLOCKED_BY_SUBSTRATE`**

诊断预算几何标签：`PROCEED_JOINT`；条件标记：`VISUAL_GATE_REQUIRES_COARSE_REUSE`

两条轴在当前数值测量中都存在满足预注册 `k/N ≥ 0.20` 的 wall-clock 与 operator-FLOP 合取候选；但测量卡有其他所有者进程共驻，不满足冻结的 `one_owner_safe_cuda_gpu`，因此该数值只能作为诊断证据，不能放行正式 G1。该结论也不是 outcome-boundary 核心机制、joint 增益或论文结论获得验证。
</callout>

# 实验范围与协议变更

- Protocol：`R16-P18-STAGE3-S1-BUDGET-FEASIBILITY-V1`
- 只读基线 commit：`ec3fbbd054f9218332122cc477912d3ddf0ad93b`
- 新产物目录：`experiments/r16p18_stage3/`
- 用户于 2026-09-02 明确将 profiling 地点由 dev05 改为现有 dev14 DSW；变更记录在 `S1_PROTOCOL_AMENDMENT_DEV14.json`，其余 S1 约束未改变。
- 没有创建、reset 或 step 环境；没有 episode、rollout、训练、权重修改或 PAI 作业。
- fresh profile 使用一张 A800、batch 1、50 warmup、200 repeats、每次计时前后同步 CUDA，并保留全部原始样本。

物理 GPU 2 上存在其他用户的内存驻留进程；本实验没有终止、暂停或修改它们。开始时利用率为 0%，168 个 1 秒采样的利用率中位数为 0%、均值 15.34%、最大值 99%。最大值包含本实验自己的前向负载。原始采样保存在 `audits/S1_DEV14_GPU_SAMPLES.csv`。由于用户的 host amendment 只将 dev05 改为 dev14，没有明确取消冻结的 device-isolation 条件，独立审计将此判为协议不合规，而不是仅作为普通误差来源。

# S1.0 底物盘点

## 旧 0.7388 的真实含义

旧值精确为：

`92,438,200,000,000 / 125,120,200,000,000 = 0.738795174560143`。

分子是 576 个 confirmatory state/model-seed unit 的 `CC` estimated-FLOP 总和；分母是每个 state 先按 privileged physical utility 选择 best `FF` tile 后的 estimated-FLOP 总和。单行代理为：

`global_encoder_calls × 1.8e9 + fine_encoder_calls × 1.8e9 + policy_forward_calls × 0.7e9`。

所以 0.7388 不是纯视觉 coarse/full 比，也不是 wall-clock。旧预算分母是 `alpha × sum(best physical FF cost)`，refine 名额由 `cost + dc > budget` 的 guard 决定。

## 为什么旧预算被压扁

代码反解确认，旧 all-coarse 只在前 8 步 treatment 使用 coarse，之后 20 步所有 arm 都恢复 fine visual + fine action。共同 full continuation 同时进入分子和分母，把 coarse/full 比推高到约 0.74；因此 25%/50% budget 连 coarse baseline 都装不下，refine 数自然为 0。

这是 confirmed code semantics，不是新 idea。

## 两条实际可调轴

- 视觉 coarse：每相机 `128→112` global ResNet18。
- 视觉 fine：保留 global branch，再从原始 128 图像切一个 tile 并 resize 到 112，执行共享 crop encoder。原生 tile grid 为 2×2 或 4×4。
- 动作 coarse/fine：同一个 ACT 固定输出 8-step chunk；coarse 每 4 步 query，fine 每步 query。动作分辨率是 runtime query interval 4→1，不是候选 token atlas。
- 两轴在 evaluator 参数上独立；不需要修改 checkpoint 权重或重训。
- 当前 fine path 会重新计算 global feature，因此现行代码是不复用口径；coarse-feature reuse 是本步要求计算的反事实定价列。

# S1.1 复现 0.7388

fresh forward 按旧 576 个 unit 的 query-count 重新加权后得到：

| 口径 | 旧值 | fresh 值 | 相对偏差 | 门槛 |
| --- | ---: | ---: | ---: | --- |
| operator FLOPs | 0.738795175 | 0.735961349 | 0.3836% | PASS |
| wall-clock | 0.750259792 | 0.745154836 | 0.6804% | PASS |

两者的数值都在 5% 内。wall ratio 的记录方差为 `0.0002071641`。但 G1.1 所需的正式复现证据来自不满足冻结 device-isolation 条件的 profile，所以“数值复现”PASS，而正式 G1.1 为 FAIL。

# S1.2 两轴代价曲线

| 轴/格点 | query 数 | wall 中位数 ms | stdev ms | p95 ms | wall/full | operator FLOPs | FLOPs/full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| visual coarse | 2 | 28.158 | 5.399 | 36.844 | 0.631738（相对 grid4） | 4.200932e9 | 0.505344 |
| visual fine-grid2 | 2 | 49.044 | 7.896 | 56.762 | 1.100319（相对 grid4） | 8.313021e9 | 1.000000 |
| visual fine-grid4 | 2 | 44.572 | 6.902 | 53.485 | 1.000000 | 8.313021e9 | 1.000000 |
| action coarse | 2 | 30.567 | 5.064 | 35.752 | 0.247327 | 4.200932e9 | 0.250000 |
| action fine | 8 | 123.591 | 21.706 | 152.605 | 1.000000 | 1.680373e10 | 1.000000 |

JSON 中保留每个 wall-clock 格点的标准差，两个代价曲线 SVG 均绘制了对应误差棒；operator FLOPs 为确定性的算子计数。

FLOP counter 显示 grid2/grid4 的 encoder 图相同；grid2 wall-clock 比 grid4 高约 10%。代码差异只在切出的原始 tile 为 64×64 或 32×32，随后都 resize 到 112，因此差异来自 crop/resize memory path 与共驻计时噪声，而不是更多 encoder FLOPs。不能把它解释为 grid2 模型更复杂。

动作 fine 的 query 数正好是 coarse 的 4 倍，FLOPs 也是精确 4 倍，wall ratio 为约 4.04 倍。这确认动作分辨率成本主要由重复 policy forward 驱动。

# S1.3 可行域

两种公式均按要求计算：

- 不复用：`k/N ≤ alpha − rho`
- 复用 coarse：`k/N ≤ (alpha − rho)/(1 − rho)`

## 预注册 0.20 门槛的正式合取候选

| 轴 | coarse→fine | alpha | 口径 | wall k/N | FLOPs k/N |
| --- | --- | ---: | --- | ---: | ---: |
| visual | global-only→fine-grid2 | 0.75 | 复用 coarse | 0.412951 | 0.494599 |
| visual | global-only→fine-grid4 | 0.75 | 复用 coarse | 0.321135 | 0.494599 |
| action | query-4→query-1 | 0.50 | 不复用 | 0.252673 | 0.250000 |
| action | query-4→query-1 | 0.50 | 复用 | 0.335701 | 0.333333 |
| action | query-4→query-1 | 0.75 | 不复用 | 0.502673 | 0.500000 |
| action | query-4→query-1 | 0.75 | 复用 | 0.667850 | 0.666667 |

both-metrics 候选数量（阈值 0.10/0.20/0.30）：

- visual：`4 / 2 / 2`
- action：`4 / 4 / 3`

## 关键不一致格点

当前实现实际不复用 coarse feature。在 `alpha=0.75` 的视觉不复用格点：

- grid2：wall `k/N=0.175859`，FLOPs `0.244656`
- grid4：wall `k/N=0.118262`，FLOPs `0.244656`

两者在 FLOPs 下越过 0.20、在 wall-clock 下未越过，因此被单列为 wall/FLOP disagreement，不能算正式候选。视觉 G1 的 PASS 明确依赖 coarse-feature reuse 定价；它证明无需改权重或重训即可存在可行预算几何，但不等于当前未复用 evaluator 已经可部署。

# G1

| Gate | 状态 | 依据 |
| --- | --- | --- |
| 1. 复现 0.7388 | FAIL | 数值偏差均小于 1%，但 profile 违反冻结的 one-owner-safe device 条件，不能作为正式复现证据 |
| 2. 视觉 k/N≥0.20 | PASS | alpha 0.75、复用 coarse 时 grid2/grid4 均通过双口径 |
| 3. 动作 k/N≥0.20 | PASS | alpha 0.50 与 0.75 均有双口径候选；不复用也通过 |
| 4. 原生 coarse/fine | PASS | 五个格点均在冻结 checkpoint 上成功执行 fresh forward，无需改权重或重训 |

`BLOCKED_BY_SUBSTRATE`

若后续仅在同样配置的 one-owner-safe A800 上重测且数值保持在 5% 内，当前冻结 G1 算法对应的条件标签才会是 `PROCEED_JOINT`；这不是本次正式结论。

# 机制解释：什么导致提高或降低

- 旧预算降低 refinement 的原因不是 selector 无效，而是共同 20-step full continuation 把 coarse/full ratio 推高，使预算先被 baseline 消耗。
- 视觉 fine 提高计算成本来自第二个 crop ResNet branch；现实现还重复 global branch，所以不复用 wall budget 下无法达到 20% 门槛。
- 复用 coarse 中间结果能提高视觉可 refine 比例，是因为增量成本从完整 fine cost 改为 `fine−coarse`；这只是已有计算图的成本分解，不是新机制或新 idea。
- 动作 fine 的成本降低/提高几乎完全由 query interval 决定：interval 4→1 产生 4 倍 policy calls。它是真正独立于视觉 crop 的运行时轴。
- grid2/grid4 FLOPs 相同但 wall 不同，说明只看解析或 operator FLOPs 会错误合并实际不同的 memory/preprocess latency。

# 审计与停止点

- CPU/unit tests `10 passed`；zero-rollout static audit PASS。
- fresh `S1_PROFILE.json` 保留五个格点各 200 个样本和输入 SHA256。
- dev14 runtime、GPU 共驻、协议地点变更和旧 blocked 快照均独立保留。
- Stage-2.7R 与更早目录未修改。
- 已按终止条件停在 G1：未开始 fresh-env、正任务 screen、S2、Stage-3 最终报告或任何 rollout。

# 结论边界

本步只回答“预算可行域是否非空”。共驻测量显示诊断可行域非空：视觉通过依赖 coarse reuse 定价，动作通过不依赖 reuse；但正式状态因 device substrate 不合规而阻塞。没有 outcome、成功率、joint synergy 或 outcome-boundary selector 数据，因此不能把诊断 `PROCEED_JOINT` 写成 idea validated、accepted evidence 或 joint mechanism 已证实。
