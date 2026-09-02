<title>实验报告</title>

<callout emoji="⛔">
**S1 / G1 唯一标签：`BLOCKED_BY_SUBSTRATE`**

这不是预算机制的负结论。dev05 SSH 入口超时，因而没有获得协议要求的单卡新鲜 wall-clock 与 FLOP-counter 前向测量；归档计费和代码解析只用于诊断，不能替代 S1.1。
</callout>

# 实验范围

- Protocol：`R16-P18-STAGE3-S1-BUDGET-FEASIBILITY-V1`
- 只读基线 commit：`ec3fbbd054f9218332122cc477912d3ddf0ad93b`
- 新产物目录：`experiments/r16p18_stage3/`
- 没有创建、reset 或 step 环境；没有 episode、rollout、训练、权重修改或 PAI 作业。
- 当前 DSW 的四张 A800 均有其他用户进程，按 owner-safe 约束未使用；dev05 `39.101.65.229:1037` 的 10 秒只读连接探针返回 timeout。

# S1.0 底物盘点

## 旧 0.7388 的真实含义

旧值精确为：

`92,438,200,000,000 / 125,120,200,000,000 = 0.738795174560143`。

分子是 576 个 confirmatory state/model-seed unit 的 `CC` estimated-FLOP 总和；分母是每个 state 先按 privileged physical utility 选择 best `FF` tile 后的 estimated-FLOP 总和。代码中的单行代理为：

`global_encoder_calls × 1.8e9 + fine_encoder_calls × 1.8e9 + policy_forward_calls × 0.7e9`。

因此 0.7388 不是 wall-clock。相同归档行的 GPU latency mean-ratio 是 `0.7502597919`；median-ratio 是 `0.761904`。预算分母是 `alpha × sum(best physical FF cost)`，refine 名额由 `cost + dc > budget` 的 guard 决定。

## 为什么旧预算被压扁

代码反解表明，所谓 all-coarse 并不是 28 步都 coarse：只有前 8 步 treatment 可变，后 20 步所有 arm 都恢复为 fine visual + fine action。共同 full continuation 主导了分子和分母，使 coarse/full 比被推高；这也是 25%/50% 预算连全体 coarse baseline 都装不下、refine 数恒为零的直接机制。

这是一条 confirmed code semantics，不是新 idea。

## 两条实际可调轴

- 视觉：coarse 是每相机 `128→112` global ResNet18；fine 保留 global branch，再执行一个 crop branch。2×2 与 4×4 只改变 crop 视野/候选数量，单 tile 的 encoder graph 相同，所以计算档实际只有两档。
- 动作：同一 ACT 固定输出 8-query chunk。coarse 每 4 步 query 并复用 chunk；fine 每步 query、只执行最新 chunk 的第 1 步。没有原生 chunk=2 等中间档。
- 两轴在 evaluator 参数上独立；但训练期 mode dropout 实际只切换视觉模式，动作分辨率只是运行时调度。
- 当前 fine path 会重新计算 global feature，没有 coarse-feature reuse 接口。因此“不复用”是现行实现；复用公式仅是反事实定价。

# S1.1 成本复现

归档解析 FLOP 比精确复算为 `0.738795174560143`，相对偏差为 0；但它不是新鲜实测。新鲜 dev05 profile 所需字段全部保持空值：batch-1 wall-clock median/variance/p95、FLOP-counter、checkpoint/observation runtime readback。

因此 G1.1 为 FAIL。没有把历史 latency 写成新的测量，也没有用解析 FLOPs 冒充 wall-clock。

# S1.2 两轴代价曲线

由于 fresh profile 不可用，曲线中 wall-clock 为 `null`，误差棒也为 `null`。代码确认的固定 8-step 解析代理为：

| 轴 | coarse | fine | FLOP proxy rho |
| --- | ---: | ---: | ---: |
| visual fine-grid2 | 8.6e9 | 15.8e9 | 0.5443037975 |
| visual fine-grid4 | 8.6e9 | 15.8e9 | 0.5443037975 |
| action | 8.6e9 | 34.4e9 | 0.25 |

这些值只用于诊断映射，不通过“wall-clock + FLOPs 两套实测”的合取 gate。

# S1.3 可行域诊断

当前实现不复用 coarse 中间结果，故可部署主口径是 `k/N ≤ alpha-rho`；`(alpha-rho)/(1-rho)` 只作为反事实列出。

## 0.20 门槛的 FLOP-only 候选

| 轴 | coarse→fine | alpha | 不复用 k/N | 复用反事实 k/N |
| --- | --- | ---: | ---: | ---: |
| visual | global-only→global+crop-grid2 | 0.75 | 0.205696 | 0.451389 |
| visual | global-only→global+crop-grid4 | 0.75 | 0.205696 | 0.451389 |
| action | query-4→query-1 | 0.50 | 0.250000 | 0.333333 |
| action | query-4→query-1 | 0.75 | 0.500000 | 0.666667 |

按 `(pair, alpha, sharing)` 计数，视觉门槛 0.10/0.20/0.30 的 FLOP-only 候选数为 4/4/2；动作为 4/4/3。由于 wall-clock 缺失，both-metrics 候选数在两轴均为 0；这表示“未裁决”，不是“预算不存在”。

# G1

| Gate | 状态 | 依据 |
| --- | --- | --- |
| 1. 复现 0.7388 | FAIL | 仅归档复算；缺 dev05 新鲜 wall-clock/FLOP-counter |
| 2. 视觉 k/N≥0.20 | FAIL | FLOP-only 有候选，但缺 wall-clock 合取证据 |
| 3. 动作 k/N≥0.20 | FAIL | FLOP-only 有候选，但缺 wall-clock 合取证据 |
| 4. 原生 coarse/fine | PASS | 源码确认两轴格点均为原生路径、无需改权重或重训；fresh runtime forward 尚未验证 |

`BLOCKED_BY_SUBSTRATE`

# 审计与停止点

- `compileall` PASS；CPU/unit tests `7 passed`；zero-rollout AST static audit PASS。
- `S1_COST_REPRO.json`、`S1_COST_CURVE.json`、`S1_FEASIBILITY.json`、四张 SVG、`S1_DECISION.md` 与 substrate audit 均已生成。
- 每个新产物由 `SHA256SUMS` 绑定；旧 Stage-2.7R 与更早目录未修改。
- 已按停止条件停在 G1；没有 fresh-env 重执行、正任务 screen、S2 准备或 Stage-3 最终报告。

# bounded inference

解析 FLOP 几何说明，两条轴都存在值得在 dev05 恢复后实测的候选：视觉只在 75% 预算边缘越过 0.20，动作在 50% 和 75% 均越过。由于视觉余量只有约 0.0057，wall-clock 开销很可能改变视觉 gate；在实测前不能写 `PROCEED_JOINT`、`PROCEED_VISION_ONLY` 或 `BLOCKED_BY_BUDGET`，也不能据此收窄论文范围。
