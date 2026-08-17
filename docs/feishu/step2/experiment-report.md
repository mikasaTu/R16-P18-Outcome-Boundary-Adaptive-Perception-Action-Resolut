---
feishu_title: "实验报告"
feishu_url: "https://icnbwz7kd1ui.feishu.cn/wiki/WytawLHgUiNZr6k61UqcpZmQnZc"
feishu_wiki_token: "WytawLHgUiNZr6k61UqcpZmQnZc"
feishu_object_token: "J5z8di0Udo4cOyxntsjcIzopnzh"
feishu_revision: 7
---

# 实验报告

> 状态：进行中。本文档只报告 Stage-2 ManiSkill3 RGB-ACT 任务筛选与 oracle outcome-boundary existence probe；不实现 learned selector、effect predictor、Diffusion Policy、DINO-WM 或 π0.5，也不把当前结果称为“idea 已验证”。

## 1. 冻结协议

- Protocol：`R16-P18-MS3-ACT-BOUNDARY-SCREEN-V1`
- ManiSkill：v3.0.1，commit `a4a4f9272ad64b1564035874b605ceb687b63ed8`
- 官方 demonstrations revision：`d674485bbffdd533914e52d272fdda34c0515608`
- 官方 RGB ACT；模型种子：16018 / 16019 / 16020
- 每任务精确选择 300 条成功轨迹，固定 200/50/50；每条 source trajectory、episode seed、initial-state hash 均唯一
- PAI：最多且恰好 2×A800；UID/GID 2254；AIMaster、elastic、preemptible、平台自动重启均关闭；W&B secret 未注入
- 阈值在观察正式结果前冻结，结果后不调整

## 2. 任务与数据门槛

初始 precision task `PegInsertionSide-v1` 在两次官方 RGB replay 中均只得到 188/200（94%），低于冻结的 95% train-split 门槛；将其标记为 `BLOCKED_DATA_GATE`，没有降低阈值。

严格按预注册 fallback 顺序及允许原因 `data_availability`，在观察 fallback replay 结果前启用首个 fallback `PlugCharger-v1`。当前 active formal tasks：

1. `PlugCharger-v1`（positive，precision grasp/alignment/insertion）
2. `PushT-v1`（positive，contact-rich）
3. `StackCube-v1`（positive，placement/release）
4. `PushCube-v1`（negative control）

四个 active task 均已选出 300 条，完整 source/subset hash 复核通过；总计 1200 条，三重唯一性全部通过。

## 3. Replay 审计

- `PushT-v1`：train 192/200、validation 48/50、test 48/50，三个 split 均为 96%，数据门槛通过。
- `StackCube-v1`：train 已完成 199/200（99.5%）；validation/test 正在续跑。
- `PlugCharger-v1`：官方 archive 将 reward metadata 标为 `dense`，但固定 v3.0.1 的该任务只接受 `none/sparse`。首次运行在第 0 条轨迹、任何物理 replay 发生前退出。由于 replay 不保存/消费 reward，已冻结 metadata-only adapter `dense → sparse`；动作、初态、种子、控制转换、动力学和 policy input 均不变。
- `PushCube-v1`：等待 replay。

正式 PAI 续跑：run `r16p18-msact-v1-20260813k`，job `dlcgr7oy2wuxomtj`。当前正在执行真实 PlugCharger/Stack replay。

## 4. Baseline 与 oracle probe 状态

- 12 个 ACT 训练（4 task × 3 seed）：尚未开始，必须等四任务 replay gate 全部通过。
- 固定 100 test episodes/task/model-seed 的闭环评估代码、paired bootstrap（10,000 次，seed 16018）和严格 call/opportunity accounting 已实现；尚未运行。
- 64-state bank、5×5 local PCA action atlas、4×4 visual tile interventions、4×5 joint probe：受 baseline gate 约束，尚未运行。
- 当前不能给出 Stage-2 GO/NO-GO；最终只按预注册阈值报告。

## 5. 代码与复现

GitHub：`mikasaTu/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut`

工作分支：`maniskill-act-boundary-screen-v1`。配置、代码和中间审计结果通过 SSH 持续推送；截至 run-o 首个完整 checkpoint，`main` 与实验分支均已 fast-forward 至 `eefa8f42a7a0017d85de9624bd481336030a9240`。后续正式结果继续以可审计 fast-forward 追加。

最后更新：2026-08-13 UTC。

## 6. 2026-08-13 数据门最终审计与第二 fallback

正式 PAI run `r16p18-msact-v1-20260813k`（job `dlcgr7oy2wuxomtj`）已结束。冻结的 replay 门为每个 split ≥95%，阈值未调整：

- `PushT-v1`：192/200、48/50、48/50，总计 288/300，PASS。
- `StackCube-v1`：199/200、48/50、50/50，总计 297/300，PASS。
- `PushCube-v1`：199/200、50/50、50/50，总计 299/300，PASS。
- `PlugCharger-v1`：train 163/200（81.5%），低于 190/200，`BLOCKED_DATA_GATE`。失败 RGB H5 SHA256：`12fd261fe6ef0fc1f9cc356fc7d32ce04c18a834c17501e72a59e8c12be46ad6`。

在 PlugCharger 已数学上不可能达到 190/200、且尚未观察下一候选任何 replay/ACT 结果时，严格按预注册顺序启用第二 fallback `PullCubeTool-v1`。其官方源有 1000 条成功轨迹，其中 937 条通过 unique seed/initial-state 资格检查；固定 SHA256 排序选出 300 条并固定为 200/50/50，source/subset hashes 与 trajectory/seed/state 三重唯一性全部复核 PASS。

PullCubeTool 固定配置：`pd_ee_delta_pose`、RGB、PhysX CPU replay、`--use-first-env-state`、每轨迹最多 9 次 retry、ACT 100000 updates。闭环 horizon 在任何 PullCubeTool replay/ACT 结果前固定为 300；依据是官方成功演示 elapsed steps 的 p95=283。ACT protocol dry-run 已 PASS。

新正式 PAI run `r16p18-msact-v1-20260813m`，job `dlc11umo0abex81p` 已创建：2×A800，AIMaster/elastic/preemptible 均关闭，源码 commit `37d54ff1688b688bd7d90e32e0af7473391cd78a`，当前正在进行环境准备/replay。run `...13l` 仅在 CreateJob 前因精确 cache 目录不存在被控制器拒绝，无远端 job、无 GPU 工作。

Baseline gate 与 oracle probe 仍未出结果；当前不能给出 Stage-2 GO/NO-GO，也不把 idea 称为已验证。

## 7. 2026-08-13 replay 矩阵完成与 ACT 正式训练启动

本节覆盖前文中已经过时的“run-m 正在准备”状态。所有失败运行均与科学结果严格分离：

- `run-m`（job `dlc11umo0abex81p`）在任何 replay 前因 launcher 仍绑定旧 demo-lock hash 而 fail-closed；无科学结果。
- `run-n`（job `dlcqh4pvuftvv7kb`，源码 `41e79baaaab396e7f6ad61507445ca7cf35b8f0c`）完成四任务正式 RGB replay：PullCubeTool 200/200、50/50、50/50；PushT 192/200、48/50、48/50；StackCube 199/200、48/50、50/50；PushCube 199/200、50/50、50/50。总计保存 1184/1200，四任务 gate 全部 PASS。
- run-n 在 PushCube seed 16018/16019 的首次训练更新中，于 optimizer/scheduler step 后、指标或完整 checkpoint 持久化前触发 `AttributeError: module transformers has no attribute deepspeed`。因此该 run 标记为 `FAIL_INFRASTRUCTURE_NO_RESULT`，没有接受任何训练结果。

修复仅针对未启用 DeepSpeed 时 Diffusers EMA 的失效属性探测：新增 `NonDeepSpeedEMAModel`，保留上游 decay schedule、更新公式、state dict 与 copy semantics，不改 ACT 架构、数据、损失或超参数。固定依赖环境测试 18/18 PASS；官方 ACT CPU smoke 完成真实 backward、AdamW step、EMA step 和 state round-trip。修复与失败审计已通过 SSH 推送至 commit `f7c6f178142801b84c4d011152958f419df71b7a`。

全新正式 run `r16p18-msact-v1-20260813o`（job `dlc106yxoqy3aa7b`）以 commit `f7c6f178142801b84c4d011152958f419df71b7a` 启动，2×A800、UID/GID 2254、AIMaster/elastic/preemptible/自动重启关闭，无 W&B secret。该 run 已再次完成四任务 replay gate，并持久化两个独立 PushCube seed 的首个真实训练指标：seed 16018，step 1，loss 68.741043；seed 16019，step 1，loss 105.312950。训练矩阵仍在执行。

截至本次更新，闭环 baseline、state bank 与 oracle action/visual/joint probe 尚无结果；不能给出 Stage-2 GO/NO-GO，也不能称 idea 已验证。
