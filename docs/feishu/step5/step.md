---
feishu_title: "step5"
feishu_url: "https://icnbwz7kd1ui.feishu.cn/wiki/HtXgwUQFGiKrSZkjCt1ckQnZn9f"
feishu_wiki_token: "HtXgwUQFGiKrSZkjCt1ckQnZn9f"
feishu_object_token: "BdSTdAemdot642x9PYic2H0Jnzb"
feishu_revision: 4
---

# step5

你是一名独立的机器人学习实验与代码审计 agent。你的任务是在 GitHub 仓库

mikasaTu/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut

中实施一个全新的 Stage-2.6 实验：

R16-P18-MS5-STAGE26-COUNTERFACTUAL-COMPLETION-V1

目标不是继续当前 visual × spatial-action joint oracle，也不是证明论文 idea。
唯一科学目标是：

1. 用完全共享的物理、控制器和策略前缀，重新验证“成功后继续执行会破坏成功”的因果关系；
2. 构建只使用 observation/history 的 deployable counterfactual completion gate；
3. 判断 learned gate 能否恢复 privileged stopping oracle 至少一半的闭环增益；
4. 若且仅若本阶段正式通过，输出 Stage-2.7 preregistration 草案，但不要运行 Stage-2.7。

────────────────────────────────────
一、必须先读的冻结来源
────────────────────────────────────

必须先阅读并审计：

- experiments/maniskill_stage25_repair_oracle_v1/preregistration.yaml
- experiments/maniskill_stage25_repair_oracle_v1/docs/STAGE25_FINAL_REPORT.md
- experiments/maniskill_stage25_repair_oracle_v1/scripts/stage25_runtime.py
- experiments/maniskill_stage25_repair_oracle_v1/scripts/evaluate_checkpoint_grid.py
- experiments/maniskill_stage25_repair_oracle_v1/scripts/build_stackcube_state_bank.py
- experiments/maniskill_stage25_repair_oracle_v1/scripts/audit_stage25.py
- sealed v26 stage25_summary_trace_corrected.json
- sealed v26 independent_stage25_audit.json
- sealed v26 success trace terminal audit

不要信任文档描述超过源码和 raw evidence。
记录当前 HEAD、tree、所有 source hashes 和 Stage-2.5 selected checkpoints。

必须在新分支工作，例如：

stage26-counterfactual-completion-v1

新实验目录固定为：

experiments/maniskill_stage26_counterfactual_completion_v1/

禁止修改 Stage-2.5 目录中的任何字节。

────────────────────────────────────
二、Inherited audit notes
────────────────────────────────────

在新目录 audits/INHERITED_STAGE25_ISSUES.md 中记录但不回改旧实验：

1. Stage-2.5 report 写了 pd_ee_delta_pose，但 preregistration/runtime 实际为 pd_ee_delta_pos；
2. Stage-2.5 action gate 的 summarizer/audit 加入了 repeat agreement ≥0.95，而冻结 preregistration 没有该正式条件；
3. Stage-2.5 terminate arms 曾有 vector-wide redundant final snapshot 缺陷，但 trace 与 decision metrics 经重算未变。

这些内容只作继承审计，不允许改变旧 final status。

────────────────────────────────────
三、预注册与冻结
────────────────────────────────────

在任何 confirmatory result 产生前创建并提交：

- preregistration.yaml
- PROTOCOL_FREEZE.json
- manifests/SCIENTIFIC_SHA256SUMS
- seed banks
- checkpoint bindings
- source bindings
- decision precedence

使用 Stage-2.5 已选择的 StackCube checkpoint：
model seeds 16018, 16019, 16020。
不得重新选 checkpoint。

新 seed banks：

- train_source: 512 episode seeds
- calibration: 128 episode seeds
- confirmatory: 200 episode seeds

全部通过确定性 SHA256 算法产生，并与以下内容不相交：

- 所有 demo seeds
- predecessor test seeds
- Stage-2.5 screen/final-validation/confirmatory/oracle seeds
- 三个新 banks 彼此之间

三个 model seeds 使用完全相同的 episode seed 顺序。

────────────────────────────────────
四、完整 rollout capsule
────────────────────────────────────

新增完整状态胶囊，至少存储：

- full simulator state
- elapsed step
- current observation and hashes
- recent four observations or frozen visual features
- ACT temporal aggregation table prefix
- last executed action
- last legal gripper command
- success streak bookkeeping
- Python RNG state
- NumPy RNG state
- Torch CPU RNG state
- Torch CUDA RNG states
- task/model/episode seed
- checkpoint path and SHA
- source step and phase metadata
- current trace prefix hash

禁止只保存 simulator state 后重新初始化 ACT action table。

状态捕获：

成功 episode：

- first near-completion
- first-success - 6
- first-success - 3
- first-success
- first-success + 3
- first-hold5

失败 episode：

- first near-completion false positive
- maximum-progress state

每个 episode 每一类型最多一个状态。

────────────────────────────────────
五、shared-prefix fidelity audit
────────────────────────────────────

先在 64 个 calibration states 上做恢复审计。

从 capsule 恢复 continue_policy branch，前 10 步必须与原 fixed trace 一致：

- executed action max abs <= 1e-6
- object translation error <= 1e-5 m
- object rotation error <= 1e-4 rad
- success/contact/grasp/support categorical agreement = 100%
- ACT table prefix、RNG、observation hashes 可核对

如果不能达到协议要求：
final status = NO_GO_SHARED_PREFIX_FIDELITY
停止所有 predictor 和 confirmatory 工作。

不得为了通过而放宽阈值。

────────────────────────────────────
六、counterfactual branches
────────────────────────────────────

每个 capsule 执行：

1. continue_policy

   - 恢复完整 ACT 状态
   - 继续原 temporal aggregation
2. neutral_hold

   - arm delta = 0
   - gripper 保持最后合法 controller-effective command
   - 不再调用 policy
3. hold_then_reobserve

   - neutral hold 2 steps
   - 用新 observation 重新开始 policy inference
   - 明确重建新的 temporal table，不伪称与 continue matched
4. terminate_oracle

   - privileged diagnostic only
   - 不作为 deployable 方法

训练标签 horizon = 20 steps，记录：

- success_once
- success_hold5 ending at horizon
- success_at_horizon
- hold_success_20
- continue_success_20
- reobserve_success_20
- first success step
- object pose trace and drift
- intended/unintended contact
- policy calls
- latency
- full executed action trace

正式 causal confirmatory 还要从 first-success 或 first-hold5 branch 到原 episode horizon。

────────────────────────────────────
七、predictor
────────────────────────────────────

禁止使用 simulator success、object pose 真值、phase 真值、goal distance、
branch outcome 或 privileged contact 作为输入。

Primary deployable input：

- last 4 frozen ACT visual latents
- last 4 proprio states
- last 4 executed actions
- current base-policy predicted first 5 actions
- gripper history
- temporal action consistency features

训练两个 calibrated binary heads：

Q_hold(history)
Q_continue(history)

分析派生：

- NOT_DONE
- DONE_ROBUST
- DONE_FRAGILE
- APPARENT_BUT_UNSTABLE

候选模型固定为：

- linear probe
- 2-layer MLP
- 1-layer small GRU

只在 calibration bank 上按以下字典序选模型：

1. false-stop <= 5%
2. maximum stop-beneficial AUPRC
3. minimum ECE
4. fewer parameters

每个 model seed 单独训练，但模型结构和超参数必须一致。
secondary：train on two model seeds, test on held-out third seed。

校准方法、threshold、temperature 必须在 confirmatory 前冻结。

决策规则：

stop_or_hold only if

Q_hold >= tau_hold
and Q_hold - Q_continue >= tau_advantage

连续两步成立。

若 Q_hold 与 Q_continue 都低，不得 stop，应标记 unstable_or_not_done。

────────────────────────────────────
八、闭环 arms
────────────────────────────────────

在 200 confirmatory episode seeds × 3 model seeds 上，配对运行：

- fixed_horizon
- fixed_time_matched_stop
- random_matched_stop
- learned_success_only_classifier
- learned_counterfactual_completion_gate
- privileged_neutral_after_hold5
- privileged_terminate_first_success

fixed/random controls 必须匹配 learned arm 的平均 stop rate，
并同时报告 unmatched natural control。

所有 thresholds 在 confirmatory 前冻结。
confirmatory 不能用于 checkpoint、模型、阈值或 early-stopping 选择。

────────────────────────────────────
九、统计与 gates
────────────────────────────────────

Primary closed-loop unit = episode seed。
Offline predictor bootstrap unit = source episode，不是 state row。

使用：

- 10,000 paired bootstrap
- 95% CI
- per-model-seed + aggregate
- secondary comparisons 使用 Holm correction
- 不允许 candidate/state pseudoreplication

Shared-prefix causal gate：

- privileged hold/stop end-success gain >= 10pp
- paired CI lower > 0
- 3/3 model seeds same direction

否则：
NO_GO_STOPPING_NOT_CAUSAL

Offline learnability gate：

- stop-beneficial AUPRC >= 0.60
- ECE <= 0.05
- NOT_DONE false-stop <= 5%
- DONE_FRAGILE recall >= 60%
- at least 2/3 seeds pass
- no catastrophic third seed

否则：
NO_GO_COMPLETION_NOT_LEARNABLE

Closed-loop learned gate：

- end-success gain over fixed >= 8pp
OR recover >= 50% of privileged stopping gain
- paired CI lower > 0
- success_once reduction <= 2pp
- post-success-loss relative reduction >= 30%
- 3/3 seeds nonnegative and at least 2 clearly positive
- policy calls <= fixed
- completion-head wall latency overhead <= 10%

满足：
GO_STOP_NORMALIZED_BASELINE

若 false-stop 过高但有增益：
REVISE_EARLY_STOP_FALSE_POSITIVE

Final precedence：

1. NO_GO_SHARED_PREFIX_FIDELITY
2. NO_GO_STOPPING_NOT_CAUSAL
3. NO_GO_COMPLETION_NOT_LEARNABLE
4. REVISE_EARLY_STOP_FALSE_POSITIVE
5. GO_STOP_NORMALIZED_BASELINE

────────────────────────────────────
十、独立审计
────────────────────────────────────

必须实现独立 audit 脚本，不能调用 summarizer 的 decision 函数。

它必须从 raw：

- rollout capsule
- branch traces
- predictor logits
- closed-loop episode rows

重新计算：

- restore fidelity
- branch labels
- oracle stopping gain
- offline AUPRC/ECE/false-stop
- closed-loop metrics
- bootstrap CI
- final status

不要只读取预先生成的：
boundary flag、utility、final decision 或 aggregate fields。

另外增加：

- capsule serialization unit tests
- ACT table restoration tests
- RNG restoration tests
- branch-prefix identity tests
- no-privileged-input static tests
- train/calibration/confirmatory leakage tests
- trace terminal pose recomputation tests
- fail-on-overwrite tests
- preemption/resume tests
- compileall

────────────────────────────────────
十一、输出结构
────────────────────────────────────

至少产生：

experiments/maniskill_stage26_counterfactual_completion_v1/
README.md
preregistration.yaml
PROTOCOL_FREEZE.json
audits/
manifests/
scripts/
tests/
docs/STAGE26_FINAL_REPORT.md
artifacts/formal-run/<run_id>/
FORMAL_RUN_MANIFEST.json
FORMAL_COMPLETE.json
raw/
key-results/
logs/
SHA256SUMS

最终报告必须明确区分：

- confirmed code semantics
- observed paired evidence
- privileged oracle evidence
- learned deployable evidence
- bounded inference
- not tested

必须用人话回答：

1. shared-prefix 是否真正一致？
2. stopping confound 是否仍然成立？
3. hold 与 continue 的反事实差异能否学习？
4. learned gate 恢复了多少 oracle gain？
5. gain 是否来自简单成功检测，而不是反事实 advantage？
6. false stop 是否伤害 success_once？
7. 三个 model seed 是否一致？
8. 当前是否有资格进入 visual × temporal Stage-2.7？

────────────────────────────────────
十二、禁止事项
────────────────────────────────────

本阶段禁止：

- 修改 Stage-2.5
- 重新选择 checkpoint
- 使用 confirmatory 调参
- 使用 simulator success/phase/object pose 作 predictor input
- 创建 visual tile selector
- 创建 action boundary predictor
- 训练 spatial action router
- OOD
- second task
- pi0.5
- real robot
- 宣称 semantic stopping 本身是新颖 paper idea
- 宣称 token saving 或 wall-clock compute saving
- 在 gate 失败后继续下游并反转上游状态
- 放宽冻结阈值
- 只与 random control 比较
- 把 unit test、CPU smoke 或 oracle 当成 method gain

若 Stage-2.6 得到 GO_STOP_NORMALIZED_BASELINE：
只创建 Stage-2.7 的 preregistration DRAFT，设计 coarse/local-fine × fixed/learned-horizon 的 2×2 factorial。
不要运行 Stage-2.7。

最终提交：

- clean Git commit
- exact commit/tree SHA
- formal run ID
- artifact hashes
- independent audit status
- 简洁的人话总结
