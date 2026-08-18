---
feishu_title: "step6"
feishu_url: "https://icnbwz7kd1ui.feishu.cn/wiki/YzvVwnstMilUKKkazZ5cy65InQe"
feishu_wiki_token: "YzvVwnstMilUKKkazZ5cy65InQe"
feishu_object_token: "FjmTdYMdfovDnYxBzsTck6xQnbd"
feishu_revision: 10
---

<title>step6</title>

你现在负责继续验证项目：

R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolution

GitHub:
https://github.com/mikasaTu/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut

本轮实验名称：

Stage-2.7R Outcome-Marginal-Value Core-Mechanism Reset

你的任务不是继续优化 Stage-2.6 completion/stopping predictor，而是对 R16-P18 的核心 perception-action resolution 机制做一次最终、干净、可裁决的 oracle 验证。

====================
一、首先完整阅读和审核
====================

开始实现前，必须完整阅读：

1. 根目录 README 和当前代码结构；
2. docs/EXPERIMENT_REPORT.md；
3. docs/MANISKILL_STAGE2_FINAL_REPORT.md；
4. experiments/maniskill_stage25_repair_oracle_v1/
   - preregistration.yaml
   - docs/STAGE25_FINAL_REPORT.md
   - scripts/run_visual_resolution_probe.py
   - scripts/action_runtime.py
   - scripts/oracle_math.py
   - scripts/run_joint_factorial_oracle.py
5. experiments/maniskill_stage26_counterfactual_completion_v1/
   - preregistration.yaml
   - docs/STAGE26_FINAL_REPORT.md
   - scripts/stage26_runtime.py
   - scripts/train_completion_predictors.py
   - scripts/evaluate_closed_loop.py

需要明确继承以下结论：

- Stage-2.5 visual-only physical effect 强于 action-only；
- joint coupling=0/64、recall improvement=0pp、regret reduction=3.12%；
- 旧 visual treatment 会持续影响后续 ACT replanning，而 action refinement 只替换前 4 步，二者时间不对称；
- 旧 abstract budget 不是实际 wall-clock/FLOP matched；
- Stage-2.6 mid-state restoration 无法恢复 PhysX hidden contact-solver cache；
- learned stopping gate 闭环为负，不再作为主线。

不要重跑旧实验来改变旧结论，不要修改旧 experiment directory 的任何字节。

====================
二、仓库和协议要求
====================

创建新 branch：

stage27r-core-mechanism-reset-v1

创建唯一可变目录：

experiments/maniskill_stage27r_core_mechanism_reset_v1/

至少包含：

- README.md
- preregistration.yaml
- PROTOCOL_FREEZE.json
- environment_lock.json
- task_selection.json
- manifests/
- scripts/
- tests/
- audits/
- docs/STAGE27R_PLAN.md
- docs/STAGE27R_FINAL_REPORT.md
- artifacts/formal-run/

旧 Stage-1、Stage-2、Stage-2.5、Stage-2.6 目录必须逐字节不可变，并通过 tree/hash audit。

在任何 confirmatory 结果产生之前，先提交并冻结：

- protocol；
- scientific questions；
- candidate task order；
- task selection algorithm；
- data split；
- model seeds；
- screen seeds；
- validation seeds；
- confirmatory seeds；
- state-bank selection；
- treatment definitions；
- success semantics；
- utility weights；
- statistical tests；
- GO/REVISE/NO-GO thresholds；
- final-status precedence。

所有 Stage-2.7R 已预注册的 oracle arms 即使中间 gate 失败也必须执行完成，但失败 gate 仍控制 claim tier，后续结果不得反转上游失败。

Stage-2.7R oracle 失败时，禁止自动执行 learned router、OOD、π0.5 或真机实验。

====================
三、科学假设
====================

定义四个模式：

CC = coarse visual + coarse action
FC = fine visual + coarse action
CF = coarse visual + fine action
FF = fine visual + fine action

对每个 state s 计算：

Delta_visual = U(FC)-U(CC)
Delta_action = U(CF)-U(CC)
Delta_joint = U(FF)-max(U(FC),U(CF))

最终必须在以下状态中做出唯一判定：

1. GO_FULL_JOINT
2. REVISE_SHARED_AXIS_ROUTER
3. REVISE_VISUAL_ONLY
4. NO_GO_CORE_MECHANISM
5. NO_GO_CAUSAL_BACKEND

不要再使用单一 binary “boundary/non-boundary” 作为核心假设。

====================
四、任务和数据 screen
====================

保留 StackCube-v1 作为 anchor positive task。

第二个 positive task 按冻结候选顺序从以下池中 screen：

- PegInsertionSide-v1
- PlugCharger-v1
- PullCubeTool-v1
- PushT-v1
- 一个提前定义并冻结的 harder StackCube fallback

不得直接复用旧失败 checkpoint 作为正式模型。

每个正式任务必须有：

- 至少 300 条独立成功轨迹；
- unique source trajectory id；
- unique episode seed；
- unique initial-state hash；
- 200 train / 50 validation / 50 test；
- augmentation 不得计作独立 episode。

若官方数据不足，可使用 pinned ManiSkill motion-planning/scripted pipeline 生成新的独立成功轨迹，但必须冻结 generator commit、seed 和 manifest。

负对照：

- PushCube-v1 只有在 repaired success_hold5 >=70% 时才可使用；
- 否则从 pinned ManiSkill registry 中通过相同 screen 选择一个健康、平滑、低接触任务；
- 若找不到健康负对照，则正式报告该限制，不得放宽阈值。

任务 mini-screen：

- model seeds: 16018,16019,16020
- 每 seed 32–40 screen episodes
- top-2 checkpoint
- 100 validation episodes
- checkpoint 按以下字典序选择：
  maximize success_hold5
  maximize success_at_end
  minimize post_success_loss
  earliest step

positive task gate：

- aggregate success_hold5 in [0.30,0.80]
- model-seed range <=20pp
- 至少 2/3 seeds success_hold5 >=0.25
- all-fine success 不得超过 0.90
- coarse mode success 至少达到 all-fine 的 50%
- fresh-reset replay categorical agreement 100%

====================
五、实现统一 multi-resolution policy
====================

必须是同一个模型、同一组主要权重支持 CC/FC/CF/FF，不能训练四个独立模型后比较。

Visual modes：

1. coarse visual：
   - 全局低分辨率图像，例如 112x112；
   - 只执行 global coarse encoder；
   - 不执行 fine crop encoder。

2. fine visual：
   - 保留 global coarse branch；
   - 从原始高分辨率图像选择一个 local crop；
   - 执行共享或轻量 fine crop encoder；
   - 将 fine tokens 与 coarse tokens 融合。

正式候选 crop 默认使用 2x2 grid，共 4 tiles。
Calibration 可比较 2x2 与 4x4；只有当 2x2 恢复 4x4 physical oracle 至少 90% 的 gain 时，confirmatory 才冻结为 2x2。

必须实际记录：

- global encoder calls
- fine encoder calls
- visual token count
- estimated FLOPs
- GPU latency
- peak memory

不允许把 blur/downsample-then-upsample 作为主要 visual-resolution 实现；旧式 blur probe 仅可作为 secondary compatibility diagnostic。

Action modes：

1. coarse action：
   - 每 4 步 query policy 一次；
   - 执行 4-step chunk。

2. fine action：
   - 每一步重新观察；
   - 每一步 query policy；
   - receding-horizon 1-step execution。

旧 9-vs-25 PCA candidate atlas 仅保留为 offline diagnostic，不再作为 deployable action-resolution 定义。

训练时进行 resolution-mode dropout，使同一 policy 在所有模式下可用。
加入 free-space consistency regularization，避免 coarse mode 无条件退化。

====================
六、禁止 mid-state restore，使用 lockstep replay
====================

正式 matched-prefix counterfactual 禁止使用 Stage-2.6 式 mid-episode state restore。

实现 replay-from-reset lockstep branching：

1. 多个 shadow env 使用同一 episode seed reset；
2. branch step 前只由一个 canonical driver policy 计算动作；
3. 将完全相同动作广播至全部 shadow env；
4. 同步运行至 branch point；
5. 审计 prefix identity；
6. fidelity 通过后才执行不同 treatment。

Fidelity thresholds：

- broadcast action max abs = 0
- object translation divergence <=1e-5 m
- object rotation divergence <=1e-4 rad
- categorical agreement = 100%
- RGB max pixel difference <=1 LSB
- 至少 95% selected branch states 通过

优先使用 PhysX CPU 完成 confirmatory causal engine。
只有 development smoke 证明 GPU fresh-reset lockstep 满足同样阈值后，才允许 GPU backend。

若 fidelity gate 失败，最终状态最高优先级为：

NO_GO_CAUSAL_BACKEND

下游 oracle 仍执行，但只能标记为 exploratory，不得声称 causal。

====================
七、state bank
====================

Calibration per positive task：

- 48 states
- 4 phases x 12
- expert/on-policy 各半
- 每 source episode 最多一个 state

Confirmatory per positive task：

- 96 states
- 4 phases x 24
- expert/on-policy 各半
- 与 train/validation/screen/test/calibration 完全不相交
- 每 source episode 最多一个 state

Phases：

1. free_space_approach
2. pre_contact_or_pre_grasp
3. object_in_hand_pre_placement
4. contact_placement_near_completion

负对照使用 48 confirmatory states。

Primary bank 禁止包含 post-success state。
Privileged phase 只能用于离线 stratification 和 reporting，禁止进入 learned/deployable inputs。

====================
八、对称 treatment
====================

所有 visual/action treatments 使用完全相同的 8-step treatment window。

8 步后，所有 arms 恢复相同 native/full policy continuation 20 steps，或者运行到 success_hold5。

每个 state 的 treatment conditions：

- CC
- CF
- FC_tile0
- FC_tile1
- FC_tile2
- FC_tile3
- FF_tile0
- FF_tile1
- FF_tile2
- FF_tile3

Calibration repeats = 3
Confirmatory repeats = 5

所有 condition 必须共享：

- source episode
- branch step
- model seed
- prefix
- treatment duration
- continuation duration
- success semantics
- controller semantics
- gripper semantics

Physical oracle：

FC = best physical FC tile
FF = best physical FF tile

Tile candidates 不是独立统计单位。

====================
九、统一 success semantics
====================

Primary metric：

success_hold5 = 任意连续 5 步满足成功 predicate

所有 arms 在首次达到 hold5 后，允许 evaluator 使用完全相同的 privileged terminate，仅用于统一计分，不作为 policy 模块。

同时记录：

- success_once
- success_hold5
- success_at_end
- first_success_step
- longest_success_streak
- post_success_loss
- normalized_progress
- intended contact
- unintended contact
- collision
- dropped_or_slipped
- recoverable
- object translation/rotation drift

禁止将 stopping predictor 作为本阶段 treatment。

====================
十、repeat-aware outcome
====================

不得把 repeat_disagreement 当作一个新的 categorical outcome，然后直接增加 boundary density。

每个 state-treatment 根据 repeats 计算：

- success probability
- mean progress
- progress variance
- drop probability
- collision probability
- recoverability probability
- utility mean
- standard error

Primary scientific claim 以 stable success 和真实 compute 为准。

Secondary utility：

100 * success_hold5
+ 20 * normalized_progress
+ 5 * recoverable
- 10 * dropped_or_slipped
- 5 * collision

同时冻结：

- success-dominant weights
- balanced weights
- progress-dominant weights

方向必须在至少 2/3 weight sets 中一致。

====================
十一、真实预算核算
====================

每个 episode/state/arm 必须记录：

- visual encoder forward calls
- fine encoder calls
- policy forward calls
- policy forward rows
- visual tokens
- action opportunities
- executed steps
- GPU latency
- simulator latency
- estimated FLOPs
- peak memory
- selector latency
- episode total compute

禁止 hard-code budget_compliant=true。
必须由独立 auditor 从 raw accounting 重新计算。

构建 budgets：

- 25% all-fine cost
- 50% all-fine cost
- 75% all-fine cost

每个 budget 下比较：

- all_coarse
- all_fine
- random_state
- phase_heuristic
- visual_only_oracle
- action_only_oracle
- strongest_equal_cost_fixed_axis
- state_axis_oracle
- joint_oracle

如果 FF 单 state 成本高于 FC，则 equal-cost FC control 必须允许 refine 更多 states。

Simulator calls used to构造 physical oracle 要单独报告为 offline labeling cost，不得计为 deployment cost，也不得隐藏。

====================
十二、统计
====================

Primary unit：

source episode identity

要求：

- 每 source episode 最多一个 primary state
- repeats 在 state-treatment 内聚合
- tile/action candidate 不得 pseudoreplicate
- model seed 分别报告
- aggregate 使用相同 episode identity 配对
- paired bootstrap 10,000 replicates
- paired sign-flip test
- 95% CI
- secondary family 使用 Holm correction
- confirmatory data 禁止用于 threshold、task、checkpoint、window、tile-grid 或 utility selection

====================
十三、最终状态
====================

最终状态优先级：

1. NO_GO_CAUSAL_BACKEND
2. NO_GO_CORE_MECHANISM
3. REVISE_VISUAL_ONLY
4. REVISE_SHARED_AXIS_ROUTER
5. GO_FULL_JOINT

GO_FULL_JOINT 要求：

- visual effect 两个正任务均为正
- action effect 至少一个正任务明确为正，另一个非负
- Delta_joint task-level CI lower >0
- 至少 10% confirmatory states 在 2/3 model seeds 上有正 joint advantage
- 50% budget 下 joint oracle 相比 strongest equal-cost single-axis：
  success_hold5 gain >=5pp 且 CI lower >0
  或在 success 差不超过2pp时 compute 至少降低25%

REVISE_SHARED_AXIS_ROUTER：

- visual 和 action 各自存在稳定 marginal value
- joint synergy 不通过
- state+axis allocation 优于 best fixed-axis equal-cost control

REVISE_VISUAL_ONLY：

- visual effect 两个正任务均通过
- action 和 joint 失败
- visual allocation 在 50% budget 下接近 all-fine 或显著优于 random/equal-cost coarse

NO_GO_CORE_MECHANISM：

- visual/action/allocation 均不能稳定优于 equal-cost controls
- 或结果只在单 seed、单任务、单 utility weights 下成立

====================
十四、执行和审计
====================

必须完成：

- unit tests
- compileall
- deterministic smoke
- fresh-reset prefix fidelity smoke
- fail-on-overwrite
- shard autoresume
- data split leakage audit
- no privileged input static audit
- raw outcome recomputation
- compute accounting recomputation
- paired statistics recomputation
- scientific SHA256 manifest
- clean source commit verification
- predecessor immutability audit

正式实验使用现有 PAI/CPFS owner-safe 模式：

- 允许 2–8 张 A800
- 不杀死、不抢占、不共享其他用户 GPU 进程
- 不删除 CPFS 科学证据
- 只有在成功 job 和完整 artifact 被证明后，才允许按 exact job id 清理 superseded Failed/Stopped service records
- 不使用 wildcard 删除
- 正式运行支持 preemption resume
- 所有 raw evidence fail-on-overwrite

====================
十五、最终报告
====================

完成正式实验后生成：

experiments/maniskill_stage27r_core_mechanism_reset_v1/docs/STAGE27R_FINAL_REPORT.md

报告必须使用人话回答：

1. shared prefix 是否真的可信？
2. visual resolution 是否有稳定 marginal value？
3. action replanning resolution 是否有稳定 marginal value？
4. 是否存在真正 joint synergy？
5. adaptive 的收益来自 state selection、axis selection，还是两者？
6. equal-cost 下是否优于 strongest single-axis？
7. 是否接近 all-fine，同时显著降低 compute？
8. 负对照是否会错误 refinement？
9. 结果是否跨任务、跨 model seed、跨 utility weights？
10. 当前 idea 应保留 full joint、shared router、visual-only，还是停止？

必须区分：

- confirmed code semantics
- observed paired evidence
- privileged oracle evidence
- bounded inference
- not tested

同时更新根目录 PROJECT_STATUS.md。

只有最终状态为：

GO_FULL_JOINT
REVISE_SHARED_AXIS_ROUTER
REVISE_VISUAL_ONLY

之一时，才创建：

experiments/maniskill_stage28_learned_router_v1/DRAFT_PREREGISTRATION.md

只创建草案，不执行 Stage-2.8。

不要只完成代码实现或 smoke。必须执行并完成全部 Stage-2.7R 预注册 oracle 实验、独立审计和最终报告。
