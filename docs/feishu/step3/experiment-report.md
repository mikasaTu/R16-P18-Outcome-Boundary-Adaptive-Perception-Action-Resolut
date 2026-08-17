---
feishu_title: "实验报告"
feishu_url: "https://icnbwz7kd1ui.feishu.cn/wiki/SXEuwFuh5iRjdNkey4IckpnWngd"
feishu_wiki_token: "SXEuwFuh5iRjdNkey4IckpnWngd"
feishu_object_token: "MuvJdSQQrophgYx3qgCciJIwnrc"
feishu_revision: 8
---

<title>实验报告</title>

# Step 3 实验报告（执行中）

**记录时间：**2026-08-13 18:51:04 UTC

**当前结论：**尚无 GO、NO-GO 或 REVISE 结论。Stage-2 正式 baseline 尚未完成，因此没有创建 Stage-3 分支，也没有实现 learned effect predictor、boundary predictor 或 budgeted selector。

## 1. 门禁与范围

- 严格顺序：Stage-2 baseline gate → frozen test state bank 与恢复门禁 → privileged oracle action/visual/joint atlas gate → 仅在双门禁通过时进入 Stage-3。
- 最大同时使用 2 张 GPU；当前正式训练为 2×A800。
- 开发机结果只作为接口、恢复性和计数 smoke，不计入科学指标。
- 所有阈值保持预注册值，未根据 smoke 或训练损失调整。

## 2. 看结果前的代码与契约冻结

GitHub main 与 Stage-2 分支已经通过 SSH 同步至提交 [53fb46d7a3b7538cb7273e55d757e1127eedf45b](https://github.com/mikasaTu/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut/commit/53fb46d7a3b7538cb7273e55d757e1127eedf45b)。

- 冻结 held-out phase predicates、SHA256 状态选择顺序、每相位 16 states、restore repeats=3 与原容差。
- 实现 256-neighbor 标准化局部 PCA、5×5 action atlas、16 tiles×3 visual interventions、4×5 joint probe。
- 实现 outcome class、连续 effect distance、utility、boundary/alias、recall/regret、paired bootstrap 与完整调用/恢复计数。
- 实现按 state 原子落盘与 resume、最多 2 GPU 的 PAI matrix runner 和 secret-free launcher。
- 修复 PullCubeTool 的 table builder 解析，使 contact accounting 同时兼容 scene_builder 与 table_scene。

## 3. 开发机 smoke（均不进入正式统计）

- PushCube frozen ACT 单 episode：seed 150151793，100 steps，success_once=true，first success step=63；仅证明 closed-loop evaluator 可运行。
- 同一 recorded state/action 的三次短回放：PhysX CUDA 的完整最终 state 最大漂移约 0.02495，未通过 1e-4 容差；对象位姿、进度和类别仍稳定。未放宽门槛。
- 同一对照在 pinned PhysX CPU 上：initial restore 最大误差 1.49e-8，完整最终 state、对象位移、旋转、进度的重复差均为 0，类别完全一致。
- 因此所有 Stage-2 四步 counterfactual 固定为 serial PhysX CPU restore；RGB 仍由 SAPIEN CUDA 渲染，ACT 仍在 CUDA 推理，所有候选使用相同 backend。
- 一条 PushCube near-completion state 的完整 atlas smoke：25 action + 48 visual + 20 joint，共 93 restores、372 simulator steps、49 logical policy inputs；执行成功。该单点的 action density=0、joint boundary=false、visual density=0.8125 不是任务级估计，不能支持或否定 idea。

## 4. 代码验证

- 开发机测试：22 passed，2 skipped（skip 为环境依赖）。
- 正式训练 Python/torch 环境中的 oracle 单元测试：8 passed。
- Python compile、Bash syntax 与 Git whitespace 检查均通过。

## 5. PAI 正式训练状态

**PAI run：**r16p18-msact-v1-20260813o；**JobId：**dlc106yxoqy3aa7b；2×A800；AIMaster/elastic 关闭；platform restart=0；W&B 未注入。

- PushCube：3/3 seeds 完成；validation-only 选择 steps 为 25k、25k、30k。
- StackCube：3/3 seeds 完成；3 个 seed 均选择 25k。
- PullCubeTool：seed 16018 完成并由 validation-only 选择 70k；seed 16019 到 93.8k/100k；seed 16020 到 14.3k/100k。
- PushT：尚未开始；位于既定队列的 PullCubeTool 之后。

## 6. 下一门禁

1. 等待 12 个 ACT runs 全部完成并验证完整 checkpoint/selection artifacts。
2. 在相同冻结 test seeds 上运行每 task/model seed 100 episodes 的 baseline closed-loop evaluation。
3. 若 baseline gate 失败，立即报告并停止，不生成 state bank 或 Stage-3。
4. 若 baseline gate 通过，才运行 64-state bank、三次恢复审计和 privileged oracle atlas。
5. 只有 oracle gate 也通过，才创建规定的 Stage-3 分支与目录并冻结 Stage-3 preregistration。

## 7. 2026-08-14 代码加固与正式训练进度

- GitHub 远端 main 与 maniskill-act-boundary-screen-v1 已同步到提交 `323a7d0619aeee57cb74e990cc6b4c93e1fe6945`（tree `c6717de3a6aa22a665f8c0e6ed18f7812cf98f58`），SSH 推送已核验。
- 本次加固：断点续跑结果绑定 state-bank/checkpoint/implementation-contract SHA；每个 action/visual/joint cell 持久化实际执行的前 4 步动作；FIRST_REAL_WORK 仅在首个 simulator 子任务成功完成后写入。
- 回归验证：`22 passed, 2 skipped`；Python compileall、Git diff-check、两个 PAI launcher 的 bash 语法检查均通过。
- PAI 正式训练仍为同一 JobId `dlc106yxoqy3aa7b`、同一 PodUID `9e3db218-331f-41f3-bb96-ff61e8a2794e`、2×A800、状态 Running；AIMaster/elastic/平台自动重启关闭，资源策略 ForbiddenQuotaOverSold。
- 截至本次检查，PushCube 与 StackCube 各 3/3 seed 完成；PullCubeTool seed 16018 完成、16019 到 99.4k/100k、16020 到 20k/100k；PushT 尚未开始。
- baseline-eval PAI 模板静态校验通过。oracle 模板按设计 fail-closed：训练中的 CPFS checkout 尚未包含新 launcher；为避免改变运行中源码，待训练完成后再只做 clean fast-forward。

> 状态：仍为 Stage-2 正式训练进行中。以上 smoke、单元测试与训练损失均不是 baseline、oracle 或 closed-loop 科学结果；当前不作 GO/NO_GO/REVISE 判断，也不创建 Stage-3 实现。

## 8. 正式结果前的证据链加固

- 最终预结果源码提交：`3ee88739fb61ca5ec452143bd3d2f78f2e66095a`，Git tree `39ff94605f93280e093db73d1c3119c84483695a`；已通过 SSH 同步到 GitHub `main` 与 Stage-2 分支。
- Baseline 断点续跑现在同时绑定：评估器 SHA、固定 seed manifest SHA、checkpoint-selection SHA、实际 checkpoint SHA、episodes.jsonl SHA；汇总器重新逐任务核对预注册的 100 个 episode identities。
- State-bank 断点续跑现在绑定：test H5/JSON SHA、builder SHA、phase-contract SHA、state-bank H5 SHA、64-state/16×4 phase 计数及 repeats=3/rollout=4。
- Oracle 断点续跑现在绑定：evaluator SHA、state-bank manifest/H5、train H5、checkpoint step/path/SHA、implementation contract 和 64 个逐状态 surface 的 SHA/字节数。
- 故障注入验证覆盖：改变 episode、test H5、checkpoint 或 surface 后旧 completion marker 均被拒绝。完整测试为 `25 passed, 2 skipped`。
- 最新 PAI baseline-eval 模板静态校验通过；正式训练完成前仍不改动其 CPFS checkout。

> 这些变更只收紧来源与恢复验证，不改变模型、候选集、utility、bootstrap、阈值或 gate，因此不构成科学结果，也不会导致事后调参。

## 9. 2026-08-14 baseline 前终态审计准备

- GitHub `main` 与 Stage-2 分支已通过 SSH 同步到提交 `0eb052f13b2ccba0b7d9fa618cf076d9f1851fba`（tree `8491f603c5fc6d3e60157b5fc0743e142687080a`）。
- Baseline evaluator 现在只在首个真实 20-episode closed-loop batch 完成后写入逐 run marker；matrix runner 将 marker 的 SHA、task/model seed 与 episode seeds 固化为 `FIRST_REAL_WORK.json`，launcher 终态按 fail-closed 方式要求该证据存在。
- 新增只读 formal-training artifact auditor：验证 12 个 run 的 validation-only argmin、5000-step candidate inventory、checkpoint/COMPLETE SHA、有限训练指标、uid/gid 2254:2254 与矩阵终态 artifacts。当前 10 个已完成 run 已逐一实测通过；矩阵完成后将启用全候选 payload SHA 审计。
- 回归验证：`26 passed, 2 skipped`；Python compileall、Bash syntax 与 Git diff-check 通过。
- 同一正式 PAI JobId `dlc106yxoqy3aa7b`、PodUID `9e3db218-331f-41f3-bb96-ff61e8a2794e` 仍在 2×A800 上运行，无重启。当前完成 10/12；剩余 PushT seed 16019/16020 约到 51.7k/31.0k。

> 以上仍是代码/证据链状态，不是 baseline 或 oracle 科学结果。Stage-3 未创建，阈值未调整，当前不作 GO、NO-GO 或 REVISE 判断。

## 10. 正式训练完成与全量 checkpoint 审计

**完成时间：**2026-08-14 02:25:21 UTC；**PAI run：**`r16p18-msact-v1-20260813o`；**JobId：**`dlc106yxoqy3aa7b`；**终态：**Succeeded。

- 12/12 个 ACT 训练 run 完成（4 tasks × 3 model seeds）；同一 PodUID `9e3db218-331f-41f3-bb96-ff61e8a2794e`，2×A800，运行 57,486 秒，观测到的 pod restart=0。
- PAI 源码固定为 commit `f7c6f178142801b84c4d011152958f419df71b7a`、tree `b26e6868230e14e92b054c00cb1720bcb3e5445d`；AIMaster/elastic 关闭，ForbiddenQuotaOverSold。
- 独立审计通过：156/156 个候选 checkpoint payload 全部重算 SHA-256；训练指标均有限；uid/gid 均为 2254:2254；所有正式 checkpoint 均按 validation imitation loss 最小、同值取最早 step 的规则选择，未使用 test 指标。
- 完整审计文件 `FORMAL_TRAINING_ARTIFACT_AUDIT.json` SHA-256=`0fb4efaebac08035b40ff867411466c77bd3d8fbdc7961615354f3a257f54d8b`；审计器 SHA-256=`cc3df75c25d57168aced386622a96949e0d45ef2da0fdaccea2f5de11396456d`。
- GitHub main 与 Stage-2 分支已通过 SSH 同步到 commit [4c0a1d382c68af6cf68f467a9da5fd6b8bfc51f4](https://github.com/mikasaTu/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut/commit/4c0a1d382c68af6cf68f467a9da5fd6b8bfc51f4)。

| Task | seed 16018 | seed 16019 | seed 16020 |
|-|-|-|-|
| PullCubeTool-v1 | 70k / 0.008130 | 70k / 0.008219 | 70k / 0.008089 |
| PushT-v1 | 80k / 0.193354 | 85k / 0.194450 | 5k / 0.195425 |
| StackCube-v1 | 25k / 0.033986 | 25k / 0.036552 | 25k / 0.034599 |
| PushCube-v1 | 25k / 0.010457 | 25k / 0.009699 | 30k / 0.009861 |

表中每格为 selected step / validation loss。PushT seed 16020 的 5k 选择是预注册 validation-only argmin 的结果，未进行人工覆盖。

> **结论边界：**这只证明 baseline 的训练输入及选择制品可用。Baseline gate 尚未运行；state bank 与 oracle 仍被门禁禁止；Stage-3 分支、目录及实现均未创建；当前不作 GO、NO-GO 或 REVISE 判断。

# R16-P18 ManiSkill RGB-ACT Stage-2 最终实验报告

日期：2026-08-14

协议：`R16-P18-MS3-ACT-BOUNDARY-SCREEN-V1`

最终判定：**`NO_GO_BASELINE_GATE`**

## 1. 结论先行

Stage-2 的正式训练与闭环基线评测已经完成并通过独立产物审计，但预注册的 baseline health gate 未通过：三个 positive tasks 中只有 `StackCube-v1` 通过，低于“至少两个 positive tasks 通过”的要求；负对照 `PushCube-v1` 成功率为 57.33%，也低于冻结的 70% 下界。

因此，本轮科学实验在 baseline gate 后按计划停止：没有创建 frozen test state bank，没有运行 privileged oracle，没有创建 Stage-3 分支或目录，也没有实现 learned effect predictor、boundary predictor 或 budgeted joint selector。

这是一项“基线不具备继续资格”的 `NO_GO`，不是对 R16-P18 自适应机制的验证、接受或证伪。该机制在本轮没有被测试。

## 2. 冻结门禁与判定逻辑

正式门禁使用 `success_once`：

- positive task aggregate success 必须在 `[0.25, 0.85]` 内；
- positive task 三个模型 seed 的成功率范围必须 `≤25pp`；
- 至少两个 positive tasks 通过；
- 负对照 `PushCube-v1` aggregate success 必须在 `[0.70, 0.98]` 内；
- 只有以上条件同时成立才允许创建 state bank 并运行 Stage-2 oracle。

统计方法在结果前冻结为：以相同 episode identity 在三个模型 seed 间配对，percentile bootstrap 10,000 次，95% CI，bootstrap seed `16018`。结果观察后没有修改阈值。

## 3. Source/code correctness

### 3.1 固定版本与源码绑定

- 本次正式闭环评测源码 commit：`4c0a1d382c68af6cf68f467a9da5fd6b8bfc51f4`
- 对应 tree：`ae5e1398eaffc621f8b3f09f797e8cf7f41ef487`
- ManiSkill：tag `v3.0.1`，commit `a4a4f9272ad64b1564035874b605ceb687b63ed8`
- Python：3.11.11
- PyTorch：2.5.1+cu124
- torchvision：0.20.1+cu124
- SAPIEN：3.0.0
- 控制模式：`pd_ee_delta_pose`
- RGB：128×128 base camera
- evaluator SHA256：`743ee3da7f94d41dfe9873b0c73a2df6d12b0ee58d2a71bca99e6b9cc48fc925`
- trainer SHA256：`f8b9308c57e6dc298e9672f5f8216b0a2999124fee857f739a2e3c53df2f5311`
- baseline summarizer SHA256：`1afd618ef60c4943a95d98f76cc09a748a67c7b0c6cf71976ec64bd2325ac40e`

### 3.2 Checkpoint selection

4 tasks × 3 model seeds 共 12 个模型。每个 checkpoint 仅由 deterministic mean validation imitation loss 的最小值选择；相同 loss 时选择最早 step。正式 test success 不参与选择。156 个候选 checkpoint payload 与 12 个选中 payload 的 SHA256 均已核验。

### 3.3 正式训练平台终态

- run ID：`r16p18-msact-v1-20260813o`
- JobId：`dlc106yxoqy3aa7b`
- PodUID：`9e3db218-331f-41f3-bb96-ff61e8a2794e`
- 终态：`Succeeded`
- 资源：2×A800，1 pod，12 CPU，200Gi memory / 200Gi shared memory
- 运行时长：57,486 秒
- 完整训练审计 SHA256：`0fb4efaebac08035b40ff867411466c77bd3d8fbdc7961615354f3a257f54d8b`

## 4. Data gate

四个正式任务各冻结 300 个 unique trajectory / episode seed / initial state identity，按 `200 train / 50 validation / 50 test` 切分。闭环评测另对每个任务冻结 100 个 test seeds，并在三个模型 seed 间完全复用。

RGB replay 共尝试 1,200 条，保存 1,184 条成功轨迹；每个 split 都达到不低于 95% 的冻结门槛：

| Task | Train | Validation | Test | 总体 | Gate |
|-|-|-|-|-|-|
| `PullCubeTool-v1` | 200/200 | 50/50 | 50/50 | 300/300 | PASS |
| `PushT-v1` | 192/200 | 48/50 | 48/50 | 288/300 | PASS |
| `StackCube-v1` | 199/200 | 48/50 | 50/50 | 297/300 | PASS |
| `PushCube-v1` | 199/200 | 50/50 | 50/50 | 299/300 | PASS |

数据完整性文件确认所有选中 identity 三重唯一，正式训练和闭环评测的 seed manifest 绑定不变。

## 5. 正式 closed-loop baseline 结果

每个 task/model seed 运行 100 个固定 closed-loop seeds，共 `4×3×100=1,200` episodes。下表中的 CI 是按冻结 identity 配对的 10,000 次 bootstrap 结果。

| Task | 三 seed `success_once` | Aggregate | 95% CI | Seed range | 冻结范围 | Gate |
|-|-|-|-|-|-|-|
| `PullCubeTool-v1` | 1%, 1%, 1% | 1.00% | [0.00%, 2.67%] | 0pp | [25%, 85%] | **FAIL，低于下界** |
| `PushT-v1` | 2%, 7%, 2% | 3.67% | [1.33%, 6.33%] | 5pp | [25%, 85%] | **FAIL，低于下界** |
| `StackCube-v1` | 50%, 63%, 51% | 54.67% | [47.33%, 62.00%] | 13pp | [25%, 85%] | PASS |
| `PushCube-v1` | 59%, 60%, 53% | 57.33% | [48.67%, 66.00%] | 7pp | [70%, 98%] | **FAIL，低于下界** |

顶层门禁：

- passing positive tasks：`[StackCube-v1]`
- passing positive count：`1`，要求至少 `2`
- negative control pass：`false`
- `continue_to_oracle_probe=false`
- 最终状态：`NO_GO_BASELINE_GATE`

### 5.1 Success-at-end 与交互指标

| Task | `success_once` | `success_at_end` | Horizon | Mean collisions | Mean intended contacts |
|-|-|-|-|-|-|
| `PullCubeTool-v1` | 1.00% | 1.00% | 300 | 0.103 | 1.233 |
| `PushT-v1` | 3.67% | 0.33% | 150 | 2.627 | 0.000 |
| `StackCube-v1` | 54.67% | 36.00% | 200 | 0.847 | 1.963 |
| `PushCube-v1` | 57.33% | 57.33% | 100 | 3.463 | 1.677 |

### 5.2 Call 与 latency accounting

| Task | Policy calls | Action opportunities | Total measured policy latency |
|-|-|-|-|
| `PullCubeTool-v1` | 90,000 | 90,000 | 36.763 s |
| `PushT-v1` | 45,000 | 45,000 | 18.281 s |
| `StackCube-v1` | 60,000 | 60,000 | 35.131 s |
| `PushCube-v1` | 30,000 | 30,000 | 12.658 s |
| **合计** | **225,000** | **225,000** | **102.833 s** |

全部 episode 均满足 `policy_calls == action_opportunities`。这里的 latency 是 evaluator 中累计的 policy-call wall time，不等于完整 simulator episode wall time。

## 6. 正式评测平台与产物审计

### 6.1 PAI 终态

- run ID：`r16p18-msact-baseline-v1-20260814d`
- JobId：`dlc141oirugeildi`
- PodUID：`de4b26e7-2ad9-42f2-9c1e-f198226c9953`
- 平台终态：`Succeeded / JobSucceeded`
- 创建：`2026-08-14T02:36:02Z`
- 开始运行：`2026-08-14T02:36:53Z`
- 完成：`2026-08-14T02:46:11Z`
- duration：609 秒
- 资源：2×NVIDIA A800、1 pod、12 CPU、200Gi memory、200Gi shared memory
- ResourceId：`quotakzri8a5wqcp`
- runtime uid/gid：`2254:2254`
- AIMaster：关闭
- elastic training：关闭
- oversold policy：`ForbiddenQuotaOverSold`
- observed pod restarts：0
- 冻结 launcher SHA256：`633695ed89425c7b0275c0733576a1ac4697791a57cba88cb62ac64984681c0c`

### 6.2 独立重算

`audit_formal_baseline.py` 从 12 份 raw `episodes.jsonl` 重新执行：

- 逐一验证 1,200 个 episode seed 及顺序与冻结 manifest 相同；
- 逐一验证 task/model-seed/protocol/source binding；
- 逐一重算 success、contact、collision、latency 和 call accounting；
- 重新执行四个 paired bootstrap；
- 验证 12 个 selected checkpoint payload SHA256；
- 验证 checkpoint selection 未读取 test 指标；
- 重算顶层 baseline gate。

审计结果为 `FORMAL_BASELINE_AUDIT_PASS`，科学判定仍为 `NO_GO_BASELINE_GATE`。

主要持久化证据：

- `baseline_gate.json`：6,343 bytes，SHA256 `ad873bfcc980ab383b31d03b34545f2e07c4110aa2fb72e130c12cdaec79bc6a`
- `EVALUATION_MATRIX_COMPLETE.json`：4,831 bytes，SHA256 `a6e523377bee47f70b022a42eef917108771fc135678bdbb19fea1a05e0ad75f`
- `FIRST_REAL_WORK.json`：890 bytes，SHA256 `3f7afe27e3fe2b1d349c66b7929bc8cc859fbf5493fdeadaf46d4e3d7b34adc7`
- first completed 20-rollout marker：SHA256 `39b5ac8e8273cd15dd88dc684aa759597b665e2213d1206c25931647f11ffb8b`

## 7. Code-first 机理反解

该部分只反解本轮 baseline 的上升/下降机制，不生成新 idea。结论按证据强度分成四类：源码确定语义、正式数据中的相关性、有边界的推断、未测试。

### 7.1 源码确定语义

1. **成功后仍执行到固定 horizon。**`ManiSkillVectorEnv` 使用 `ignore_terminations=True`；`evaluate_batch` 在整个冻结 horizon 内持续执行动作，将任意时刻成功 OR 到 `success_once`，只在循环末读取 `success_at_end`。因此策略可先满足成功条件，随后因继续动作丢失终态成功。
2. **每步调用 ACT 并做 temporal aggregation。** 官方 ACT 输出 30-query chunk；当前时间步的可用预测按 `exp(-0.01*k)` 加权。每个 action opportunity 都有一次 policy call。这是所有 baseline run 共享的执行路径，不是 adaptive-resolution 机制。
3. **validation-only 选 checkpoint。** 选择规则为最小 validation imitation loss 与最早 step tie-break，无法在观察 test 后通过换 checkpoint 修补正式结果。

### 7.2 正式 episode 中观察到的机制信号

- **低成功不是单 seed 异常。** PullCubeTool 三 seed 都是 1%；PushT 是 2%/7%/2%，seed range 分别只有 0pp/5pp。两个 floor failure 都跨 seed 复现。
- **固定 horizon 确实伴随 post-success loss。** StackCube 有 164 个 `success_once`，终态只保留 108 个，56 个为 transient success，retention 65.85%；PushT 有 11 个 `success_once`，终态只保留 1 个，10 个 transient，retention 9.09%。源码允许该现象且数据中出现，但本轮没有隔离每次丢失成功的具体物理动作，因此只作 bounded inference。
- **PushCube 失败与碰撞负担强相关。** never-success episodes 平均 7.398 次 collision，success_once episodes 平均 0.535 次。它是强诊断信号，但不是碰撞造成失败的因果估计，也不能证明碰撞是唯一原因。
- **通过的 StackCube 成功轨迹更“干净”。** success_once episodes 平均 collision 0.250，never-success 为 1.566。这解释了通过与失败样本的差异，但不构成 adaptive selector 的证据。
- **PushT intended-contact counter 在本轮不可辨识。** 冻结 pair-contact predicate 在 300 条 PushT episodes 中全部为零，包括 11 条 `success_once`。所以这个 channel 不能解释 PushT；究竟是 predicate、actor representation 还是 force threshold，需要新的隔离实验，而本计划禁止在门禁后继续。
- **PushT 的 5k 早选 checkpoint 不是充分解释。** seed 16020 选 5k 得 2%，seed 16018 选 80k 同样得 2%，seed 16019 选 85k 得 7%。不能把任务级 floor 单独归因于 5k checkpoint。

### 7.3 没有被测试的机制

由于 baseline gate 失败，下列量没有科学数据：action/visual boundary density、joint coupling density、best-action recall、outcome regret、effect/boundary predictor、budgeted joint selector、outcome-flip margin、BNAR、MVoC router、contact-phase hysteresis、action-commutator proxy。不得从 baseline 诊断外推这些机制是否成立。

## 8. Predictor learnability / offline oracle / deployable selector / OOD

这些阶段均为 **NOT RUN — prohibited by preregistered baseline gate**：

- predictor learnability：未训练 predictor；
- offline oracle：未创建 state bank、未生成 action/visual/joint atlas；
- deployable selector：未实现；
- closed-loop adaptive arms：未实现或评测；
- geometry OOD、visual-nuisance OOD：未运行；
- Stage-3 新线索：未运行。

因此不存在 predictor、oracle、selector 或 OOD 的正负结果，也不存在可报告的 R16-P18 mechanism gain/loss。

## 9. 基础设施失败与科学负结果分离

正式 run `...14d` 之前有三个 registry preflight 被 fail-closed 封存：

- `...14a`：缺少预创建的 baseline evaluation 目录；
- `...14b`：缺少 run-scoped cache 目录；
- `...14c`：缺少 PAI evaluation parent 目录。

三者均停在 `preflight_failed_sealed`，没有 JobId、没有占用 GPU、没有 rollout，也不是科学实验。修复限定路径后，新的 run ID `...14d` 一次提交并成功完成。它们不能计入模型失败或 scientific seed。

## 10. Stop-scope 核验

完成结果后执行了只读核验：

- CPFS data root 下不存在 `state_bank/`；
- CPFS data root 下不存在 `oracle_atlas/`；
- 没有提交 oracle PAI job；
- 本地不存在 `experiments/maniskill_boundary_adaptive_resolution_v1/`；
- remote 不存在 `maniskill-boundary-adaptive-resolution-v1` 分支；
- 没有 learned selector、Stage-3 checkpoint 或 Stage-3 closed-loop 产物。

这不是缺失项，而是预注册停止规则的正确执行。

## 11. Compute 与 latency

正式训练使用 2×A800，运行 57,486 秒；正式闭环评测使用 2×A800，运行 609 秒。评测累计 225,000 次 policy calls，与 225,000 次 action opportunities 完全相等；evaluator 记录的累计 policy latency 为 102.833 秒。

由于没有 adaptive arms，不能进行 matched token/candidate/effect-call budget 比较，也没有 success-latency Pareto 可比较。

## 12. Limitations

- `success_once` 是冻结 gate metric；`success_at_end` 只作诊断。固定 horizon 让二者在 PushT 和 StackCube 上明显不同。
- contact/collision 分组是 observation-level association，不是随机干预后的因果效应。
- PushT intended-contact predicate 在本轮为零，限制了 contact-channel 解释。
- 没有 oracle，因此不能区分“任务没有可利用 outcome boundary”和“基线太弱而不适合继续探测”。
- 没有 Stage-3 数据，因此不能评价 joint selector、非加性交互或任何新线索。

## 13. 最终 GO / NO_GO / REVISE

**最终判定：`NO_GO_BASELINE_GATE`。**

理由是 frozen baseline necessary condition 未满足：只有 1/3 positive tasks 通过，且 PushCube 负对照未过。按照预注册，本轮必须停止，不运行 oracle 或 Stage-3。该判定只适用于本次固定任务、数据、ACT baseline 与门禁，不表示 R16-P18 idea 被验证、接受或证伪。

## 14. 复核命令与 Git 小产物

正式大产物留在 CPFS；Git 保存代码、结果摘要、绝对路径和 SHA256。关键小产物：

- `experiments/maniskill_act_boundary_screen_v1/baseline/baseline_gate_20260814.json`
- `experiments/maniskill_act_boundary_screen_v1/baseline/formal_baseline_completion_audit_20260814.json`
- `experiments/maniskill_act_boundary_screen_v1/baseline/baseline_failure_mechanism_analysis_20260814.json`
- `experiments/maniskill_act_boundary_screen_v1/scripts/audit_formal_baseline.py`
- `experiments/maniskill_act_boundary_screen_v1/scripts/analyze_baseline_mechanisms.py`

只读完整审计命令：

```bash
python experiments/maniskill_act_boundary_screen_v1/scripts/audit_formal_baseline.py \
  --evaluation-root /mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16-p18-maniskill-act-boundary-screen-v1/baseline_evaluation \
  --checkpoint-root /mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16-p18-maniskill-act-boundary-screen-v1 \
  --seed-manifest "$PWD/experiments/maniskill_act_boundary_screen_v1/manifests/data_selection_summary.json" \
  --artifact-dir /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16-p18-maniskill-act-boundary-screen-v1/pai-eval/r16p18-msact-baseline-v1-20260814d \
  --verify-selected-checkpoint-payloads
```

机理诊断命令：

```bash
python experiments/maniskill_act_boundary_screen_v1/scripts/analyze_baseline_mechanisms.py \
  --evaluation-root /mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16-p18-maniskill-act-boundary-screen-v1/baseline_evaluation \
  --checkpoint-root /mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16-p18-maniskill-act-boundary-screen-v1
```

## 15. 最终发布记录

- GitHub `main` 与 `maniskill-act-boundary-screen-v1` 已通过 SSH 同步到 commit [76e71f5eae9771b83906478f0c421183e38cdd9c](https://github.com/mikasaTu/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut/commit/76e71f5eae9771b83906478f0c421183e38cdd9c)，tree `088b74883a53f9577aacd742f4c9ac560704dad9`。
- 提交后验证：`26 passed, 2 skipped`；Python compileall、JSON、Bash syntax、Git diff-check、正式 gate 逐字节副本均通过。
- 正式 baseline 独立审计状态为 `FORMAL_BASELINE_AUDIT_PASS`，科学判定保持 `NO_GO_BASELINE_GATE`；没有在结果后修改阈值，也没有运行 oracle 或 Stage-3。
