# R16-P18 LIBERO Stage-1 初步验证报告

报告日期：2026-08-12（UTC）  
协议：`R16-P18-LIBERO-STAGE1-PILOT-V1`  
最终决策：**`NO_GO_BASELINE_GATE`**

## 1. 结论摘要

本次实验只检验当前 LIBERO 三任务与 Small-BC 基线是否满足进入 R16-P18
自适应实验的必要条件。正式 PAI 作业完成了 9 个模型和 450 个闭环回合，
但 `put_the_wine_bottle_on_the_rack` 的聚合成功率达到 98.0%，超过预注册
上限 80%，产生明显天花板效应。因此三个任务未同时通过 baseline gate。

依据预注册停止规则，本次没有实现或评估 effect predictor、96-state bank、
7×7/13-candidate action atlas、boundary selector 或任何 adaptive arm。该结果
不能用于声称 R16-P18 idea 已验证、已接受或已被否证。

另有一项独立的重大协议偏差：官方 LIBERO 对每个精确任务仅提供 50 条成功
演示，且 HDF5 不含原始 episode seed；原要求的 200 条演示及 160/20/20
episode-seed 划分无法满足。因此即使基线通过，这个 pilot 也不能返回 Stage-1 GO。

## 2. 对研究 idea 的操作化理解

核心假设是：在视觉 token、动作候选、policy call、effect-model call、动作机会
和 timeout 完全匹配的预算下，围绕预测的任务结果边界，同时调整视觉分辨率和
动作分辨率，应优于 uniform、random adaptive、visual-only、action-only 以及
独立组合的 visual+action。

“结果边界”不是普通图像边缘，而是局部动作结果发生离散或高敏感变化的区域，
例如接触建立、碰撞、任务进度跃迁、短时成功和可恢复性变化。测试时 effect
predictor 和 selector 均不得读取 simulator privileged state，也不得预读完整
fine outcome surface 后再分配预算。

本实验是 necessary-condition falsifier 的 baseline gate，不是 world model、
Diffusion Policy、DINO-WM 或 π0.5 实验。

## 3. LIBERO 任务与预注册门槛

| Key | Exact task | 角色 | 基线合格区间 |
|---|---|---|---:|
| `push_plate` | `push_the_plate_to_the_front_of_the_stove` | boundary push | 40%–90% |
| `bottle_rack` | `put_the_wine_bottle_on_the_rack` | constrained placement | 25%–80% |
| `bowl_plate` | `put_the_bowl_on_the_plate` | smooth negative control | 80%–100% |

Gate 使用 3 个模型 seed 与每个 seed 50 个固定 init state 的总体点估计。
95% paired bootstrap CI 用于报告不确定性，不替代点估计判据。三个任务必须全部
通过，才允许进入自适应实现。

## 4. 数据协议与冻结输入

官方精确任务各有 50 条成功演示。本 pilot 没有复制演示，也没有把数据增强轨迹
冒充独立 episode。每条演示以 `SHA256(init_state)` 作为稳定身份，按哈希字典序
做 40/5/5 的 train/validation/test 划分。

| Task | Demo bytes | SHA256 |
|---|---:|---|
| Push | 762,855,139 | `36b4e1bced49d2f4ff6b2fce6b1596a63978e14199e2513cd0df71e127bf47a6` |
| Bottle | 878,958,730 | `f9092aa70734fc4083e97fc58c3ba25f87c614d18326182ddc7a455f0ab4da2e` |
| Bowl | 468,246,288 | `e69528b0cf10dfc59b20698e12ec2affc03f3887309034d3eb74cac3ec929406` |

LIBERO assets 共 585 个文件、422,313,213 bytes，冻结树清单 SHA256 为
`39b115a50b590dfa061d36439702eddf99c178a0fcfeab90c5c60afcc6c05a35`。
原始 demo/assets 是上游输入而非本实验生成产物，仓库保存其哈希和下载/复现说明，
不重复提交约 2.5 GB 上游数据。

## 5. BoundaryBC-S 实现

- 128×128 RGB。
- 随机初始化 ResNet18，取 layer3 的 8×8 spatial feature map。
- 64 个、每个 128 维的 visual microtoken；baseline 固定均匀选取 16 个。
- 15 维 proprio 输入。
- 两层、4-head、hidden size 128 的小型 transformer/action head。
- 7 维 OSC pose action，action horizon 8，execute 4。
- batch size 64，BF16，AdamW；每任务、每 seed 3000 optimizer steps。
- 模型 seeds：16018、16019、16020。
- 两张 A800 分别运行独立单 GPU 队列，不使用 data parallel。

冻结配置位于 `configs/r16_p18_libero_stage1.yaml`，SHA256 为
`d4344c056e1c7682cec9deabd0c82888bd6ec29d6d473b9a1870a262aeaa64bf`。

源码原始提交为 `20d65a9ab7ff42662496c093e14531904cf1fe31`，冻结 tree 为
`b87468210362a81a5d3754630f1ff804f79577e4`。本仓库初始导入提交的 tree 与之
逐字节一致；三个原始实现提交保存在 `provenance/source-commits/`。

## 6. Runtime 与开发机 smoke

正式 runtime：Python 3.11.11、PyTorch 2.5.1+cu124、torchvision
0.20.1+cu124、CUDA 12.4、cuDNN 90100、NumPy 1.26.4、MuJoCo 3.6.0、
robosuite 1.4.0。CUDA 12.4 修订发生在观察任何训练结果之前，未改变模型、数据、
seed、预算或门槛。

最终开发机 smoke 证据位于
`artifacts/dev-smoke/dev14-smoke-20260812-v5/smoke_result.json`：

- 三个任务均完成真实数据读取、forward/backward 和 action-shape 检查。
- 输出 shape 为 `[1, 8, 7]`；microtoken shape 为 `[1, 64, 128]`。
- 三任务 simulator state 重放误差均为 0。
- Push 与 Bowl 的重复渲染逐像素一致；Bottle 最大差 1 LSB、MAE `6.1035e-05`。
- proprio 最大绝对差不超过 `0.001827`。
- 单 A800、batch 1 GPU smoke 通过；`adaptive_components_implemented=false`。

## 7. PAI 作业与审计轨迹

正式作业：

- Run ID：`r16-p18-libero-stage1-bc-gate-20260812-003`
- PAI JobId：`dlcnouq6igkhfyub`
- 终态：`Succeeded`
- 运行时长：1580 秒（26 分 20 秒）
- 资源：1 worker、2×A800、12 CPU、200 GiB memory/shared memory
- AIMaster/自动容错关闭，平台 restart 上限为 0；单一 PodUID，无重启
- exact submit/readback contract、首训练 step、首 rollout、artifact 与 cleanup 检查均通过

审计链也完整保留了两次前置尝试：

- `...-001` 仅完成本地 preflight，因 registry private dir/checkpoint parent 尚未就绪，
  没有创建 PAI job。
- `...-002`（JobId `dlc1eloj62mdzw2y`）在第一个训练 step 前失败：容器 cwd 为
  `/root`，privilege drop 后 GNU find 无法恢复 cwd。修复仅在冻结 launcher 中加入
  `cd "$PROJECT_DIR"`；模型、数据、seed 和阈值未改变。
- `...-003` 成功后，旧 Failed 服务行按两阶段 OpenAPI 删除；删除意图、响应和回读
  证据仍保存在 `artifacts/pai-registry/runs/`。

## 8. Baseline Gate 结果

| Task | Seed 16018 | Seed 16019 | Seed 16020 | Aggregate | 95% paired bootstrap CI | Gate |
|---|---:|---:|---:|---:|---:|---|
| Push | 96% | 50% | 94% | 120/150 = 80.0% | [52.0%, 98.0%] | PASS |
| Bottle | 96% | 100% | 98% | 147/150 = 98.0% | [94.0%, 100.0%] | **FAIL** |
| Bowl | 94% | 80% | 92% | 133/150 = 88.7% | [78.0%, 97.3%] | PASS |

总体为 400/450 = 88.9%，但总体成功率不是 gate 指标。CI 采用固定 seed、10,000
次 paired seed/episode bootstrap；gate 仍使用预注册点估计。

## 9. 调用与时延核算

| Task | Episodes | Policy calls | Calls/episode | Executed steps | Inference seconds | ms/call | Mean wall/ep | P95 wall/ep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Push | 150 | 6,967 | 46.447 | 27,673 | 24.480 | 3.514 | 3.935 s | 6.547 s |
| Bottle | 150 | 6,069 | 40.460 | 24,023 | 21.655 | 3.568 | 3.373 s | 4.254 s |
| Bowl | 150 | 4,613 | 30.753 | 18,228 | 16.789 | 3.639 | 2.667 s | 6.252 s |
| Total | 450 | 17,649 | 39.220 | 69,924 | 62.923 | 3.565 | 3.325 s | 6.436 s |

`wall_seconds` 总和为 1496.265 秒；由于两个 GPU 队列并行，它不等于 PAI 作业
墙钟时间。Gate 失败后没有 effect-model 或 action-atlas call。

## 10. GO / NO-GO 与停止范围

最终决策为 **`NO_GO_BASELINE_GATE`**。直接原因是 Bottle constrained-placement
任务 98.0% 超过 80% 上限，缺少检测联合分辨率提升所需的 headroom。

因此没有生成以下结果：state-bank alias rate、best-action recall、outcome regret、
boundary F1/AUROC、coupling gain、matched-arm budget accounting 或 ID/geometry OOD/
visual-nuisance OOD 对比。这些指标不是“零”，而是依据停止规则**未实施、未评估**。

本次没有自动转向 Diffusion Policy、DINO-WM 或 π0.5。未来若继续，必须重新预注册：

1. 用更难的 constrained-placement 任务替换 Bottle。
2. 解决 200 条独立成功演示及 episode-seed 身份协议。
3. 在观察新任务结果前冻结新 gate 和全部阈值，禁止事后放宽门槛。

## 11. 核心证据哈希

- `baseline_gate.json`: `20d64ae8cec3e5405f79b678710189b0799e3ed4b87614e7e32ab6fd7295b5fd`
- `baseline_gate.md`: `d855e119dcb412fc4a914b79bb1960b9bade3ac65eba00753a3e02567babfff0`
- `run_manifest.json`: `d3ec8cb14033472fe89d3319ee108f82225bb272de1c1f7db9226d690d64d0cc`
- `monitor-result.json`: `2237738c102184521467d34fae470a474f5ff07326ae47425a4f5cda899f12e0`
- `artifact-evidence.json`: `c538d8231071fe5136033696791e6e75db5604dbfd6872ca3c4894ef45eb6b50`
- failed-job cleanup evidence: `af7e9c47591cad08a1da6fd6c0902fde24301a5a3550eab230a2da700a217c14`

完整逐文件校验表见 `provenance/SHA256SUMS`。飞书“实验报告”的原始三段 XML
保存在 `docs/feishu-report-source/`。
