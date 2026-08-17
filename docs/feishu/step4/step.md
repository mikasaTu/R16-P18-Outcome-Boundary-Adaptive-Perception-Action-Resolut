---
feishu_title: "step4"
feishu_url: "https://icnbwz7kd1ui.feishu.cn/wiki/Vg0WwZcAfiqhbQkNooKcllXcnkd"
feishu_wiki_token: "Vg0WwZcAfiqhbQkNooKcllXcnkd"
feishu_object_token: "OBq0d61yMoYAlSxQTHNc11wbnZg"
feishu_revision: 5
---

# step4

你是本项目下一阶段的独立实验设计、代码实现、实验执行与结果审计 Agent。

仓库：
[https://github.com/mikasaTu/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut](https://github.com/mikasaTu/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut)

上一阶段已审计证据提交：
76e71f5eae9771b83906478f0c421183e38cdd9c

上一阶段正式结论：
NO_GO_BASELINE_GATE

你的任务不是修改或重解释上一阶段，而是建立一个完全独立的新实验协议：

Protocol ID:
R16-P18-MS4-STAGE25-REPAIR-ORACLE-V1

建议新分支：
stage25-baseline-repair-oracle-v1

建议新目录：
experiments/maniskill_stage25_repair_oracle_v1/

核心目标：

1. 检查 validation imitation loss checkpoint selection 是否掩盖了更好的闭环 ACT checkpoint。
2. 分离“没有到达成功”与“成功后继续执行导致失稳”。
3. 只在健康的 StackCube baseline 上验证 physical action outcome boundary 是否存在。
4. 验证局部视觉信息分辨率是否会使策略跨越 physical action boundary。
5. 验证同时提高视觉和动作分辨率是否优于只提高其中一个。
6. 在 matched refinement budget 下，验证 privileged joint oracle 是否优于 random allocation 和 phase heuristic。
7. 本轮必须在 oracle 结论后停止，不允许实现 learned predictor 或 deployable selector。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
一、不可违反的范围和证据边界
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 以下旧目录必须保持只读，不得修改任何字节：
experiments/maniskill_act_boundary_screen_v1/
2. 不得修改旧 preregistration、baseline_gate、NO_GO 结论或
continue_to_oracle_probe=false。
3. 新实验必须使用新的 protocol ID、run ID、目录、seed manifest 和报告。
4. 不得使用 confirmatory test 指标：

   - 选择 checkpoint；
   - 选择 PCA radius；
   - 选择 effect threshold；
   - 选择 utility weights；
   - 选择 visual tile rule；
   - 修改 GO/NO-GO 阈值。
5. 所有科学阈值必须在产生新 confirmatory 结果前写入并冻结：
preregistration.yaml
PROTOCOL_FREEZE.json
SHA256 manifest
6. CPU smoke、unit test、synthetic fixture、代码可执行性、环境安装成功，
都不能表述为机制增益、baseline reproduction 或方法 acceptance。
7. 基础设施失败和科学负结果必须分离。
修复基础设施后使用新 run ID，不得覆盖旧失败产物。
8. 即使最终 GO，也只能写：
GO_SINGLE_TASK_JOINT_ORACLE
不得写 accepted、validated method、paper-ready、general improvement。
9. 不允许因为结果不好而修改阈值、任务、状态 phase、utility 或 candidate radius。
10. 所有历史 N3/N4 仍是 unaccepted。不得把本轮 exploratory oracle
自动提升成方法接受证据。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
二、Stage 0：源码与证据审计
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

先完成只读审计，至少读取并绑定：

- docs/MANISKILL_STAGE2_FINAL_REPORT.md
- experiments/maniskill_act_boundary_screen_v1/preregistration.yaml
- experiments/maniskill_act_boundary_screen_v1/README.md
- task_selection.json
- train_official_act_protocol.py
- evaluate_official_act_protocol.py
- summarize_baseline.py
- audit_formal_baseline.py
- baseline_failure_mechanism_analysis_20260814.json
- baseline_gate_20260814.json

输出：
audits/STAGE2_SOURCE_AUDIT.md
manifests/source_bindings.json

必须记录：

- 当前 HEAD、tree 和 audited commit；
- 旧目录所有相关脚本 SHA256；
- 所有 checkpoint candidate 路径和 SHA256；
- 旧 100 test seeds；
- 旧 demo identities；
- 现有 state-bank/oracle 脚本是否已经存在；
- 现有脚本与本次新 preregistration 的逐项差异。

可以复用旧代码思想，但必须复制到新目录并重新审计。
不得在旧目录直接修改或执行被旧 stop rule 禁止的下游阶段。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
三、Stage 1：冻结新预注册
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

创建：
experiments/maniskill_stage25_repair_oracle_v1/preregistration.yaml

至少冻结：

A. Seed banks

- checkpoint_screen_val: 32 seeds/task
- checkpoint_final_val: 100 seeds/task，包含前32个
- confirmatory_test: 100 seeds/task
- oracle_source: 独立 seeds

这些集合必须：

- 彼此不相交；
- 与300个 demo seeds不相交；
- 与旧100个正式test seeds不相交；
- 三个model seeds完全复用同一顺序。

B. Primary success metrics

- success_once
- success_hold5
- success_at_end
- post_success_loss
- longest_success_streak
- first_success_step

C. StackCube baseline gate

- success_once ∈ [0.25, 0.85]
- success_hold5 ≥ 0.30
- seed range ≤20pp
- success_at_end / success_once ≥0.60

D. Action boundary gate

- restoration pass 100%
- same-state same-action categorical agreement ≥95%
- candidate validity ≥90%
- placement/contact boundary density ≥0.20
- 比free-space高至少0.10

E. Joint oracle gate

- joint coupling density ≥0.15
- best-action recall 相对最强单侧提高≥10pp
或 outcome regret 降低≥15%
- matched-budget adaptive 优于random和phase heuristic
- paired 95% CI lower bound >0
- 至少2/3 model seeds方向一致
- 排除post-success states后仍成立

冻结后不得修改。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
四、Stage 2：Existing-checkpoint closed-loop repair
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

不要先重新训练。

对StackCube和PushCube的全部现有5k checkpoint：

1. 在32个screen validation seeds上闭环运行。
2. 每个model seed选top-2。
3. top-2在完整100个validation seeds上运行。
4. 按以下字典序选checkpoint：
a. 最大success_hold5
b. 最大success_at_end
c. 最小post_success_loss
d. 最早step

同时保留：

- old validation-loss checkpoint
- final checkpoint
作为正式对照。

可对PullCubeTool和PushT做低成本rescue screen，但不得让它们阻塞
StackCube single-task oracle。

必须报告：

- imitation loss和closed-loop stable success的Spearman相关；
- checkpoint rank inversion；
- 旧checkpoint是否被Pareto dominate；
- screen与final validation的选择一致性。

checkpoint选完后，才可在全新100 confirmatory seeds上运行一次。

如果StackCube baseline gate失败：

- 写 NO_GO_BASELINE_REPAIR；
- 不创建oracle state bank；
- 不运行任何action/visual/joint atlas；
- 进入独立审计和最终报告。

PushCube若仍低于70%，不能作为健康task-level negative control；
但不阻止StackCube single-task exploratory oracle。
最终结论必须明确标记SINGLE_TASK_ONLY。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
五、Stage 3：Success semantics / overshoot diagnostic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

对相同selected checkpoint、相同confirmatory seeds运行：

1. fixed_horizon
2. terminate_first_success
3. terminate_hold5
4. neutral_after_hold5

neutral action：

- arm delta为0；
- gripper保持最后合法命令；
- 不使用新的策略动作。

后三个是privileged simulator diagnostics，不是deployable方法。

保存逐步：

- success predicate
- object pose
- first success step
- success streak
- intended/unintended contact onset
- post-success object drift
- policy action
- neutral action

判断：

- stop arm提升很大 → REVISE_STOPPING_CONFOUND
- neutral仍失稳 → physical instability
- fixed-horizon后续动作造成失稳 → 所有后续方法必须加入stopping baseline

不得使用这些诊断结果重新选择checkpoint。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
六、Stage 4：Contact metric修复
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

当前旧代码中的collisions等于unintended-contact onset，不得继续使用含糊命名。

在新协议中：

- 使用 intended_contact_onsets
- unintended_contact_onsets
- contact_duration
- max_contact_force（若可可靠获取）
- post_success_contact_onsets

为StackCube写脚本化positive/negative contact unit tests。
PushT旧intended-contact channel全零，在没有修复和验证前不得用于科学结论。

Contact metric只作secondary evidence，不得替代task outcome。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
七、Stage 5：构建StackCube state banks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

只有StackCube baseline gate通过才允许创建。

创建：

- calibration: 32 states
- confirmatory: 64 states
- post_success_diagnostic: 16 states

四个phase：

1. free_space_approach
2. pre_grasp_or_pre_contact
3. object_in_hand_pre_placement
4. placement_contact_near_completion

每个confirmatory phase包含：

- 8 expert successful trajectory states
- 8 selected-baseline on-policy states

所有state source episode/seed必须与：

- train
- validation
- old test
- new confirmatory test
完全不相交。

每个state保存：

- full simulator state
- source episode/seed/step
- phase
- expert/on-policy source
- RGB hash
- state hash
- task predicates

每个state在CPU PhysX下恢复3次。
任何一个state恢复不一致，整个state restoration gate失败并停止。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
八、Stage 6：Physical action outcome atlas
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

每个state、每个model seed：

1. 用原始视觉输入生成nominal ACT chunk。
2. 从train split检索256个最近action chunks。
3. 取前4个action steps并拟合2D local PCA。
4. 以nominal action为中心构建嵌套网格：
coarse = 3×3
fine = 5×5
5. PCA radius只允许从预注册候选中在calibration bank选择。
6. confirmatory bank前冻结radius。
7. candidate越界必须标invalid，禁止静默clip。

每个candidate：

- 恢复相同state；
- 执行4-step prefix；
- 继续base policy最多20步；
- 最后neutral hold 5步；
- 重复3次。

记录：

- stable success
- categorical phase outcome
- target-relative pose progress
- grasp/support/drop/slip
- intended/unintended contact
- recoverability
- simulator calls
- policy calls
- latency

构建4-neighbor lattice edge。
boundary edge定义为：

- categorical outcome变化；
- stable success变化；
- 标准化effect distance越过冻结阈值；
- recoverable与unrecoverable变化。

基本统计单位必须是state，不是edge。

若action boundary gate失败：

- 写 NO_GO_NO_ACTION_BOUNDARY；
- 不运行visual/joint probe；
- 独立审计后封账。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
九、Stage 7：Visual information-resolution probe
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

保留旧4×4 tile × destructive interventions仅作diagnostic screen，
不能作为“提高视觉分辨率”的主证据。

正式视觉臂：

V_coarse:

- 128×128降采样到64×64，再上采样回128×128。

V_local_fine:

- V_coarse背景；
- 恢复一个4×4 grid中的32×32原始native-resolution tile；
- tile边缘使用冻结的平滑融合。

V_random_fine:

- 相同预算的随机tile。

V_full_fine:

- 原始128×128，仅作upper bound。

所有输入随后仍通过官方ACT wrapper变成224×224。
报告中必须写明：
这是information-resolution oracle，不是实际token/latency reduction。

对16个tile分别运行policy，把visual-induced action映射到已冻结physical action atlas。
privileged tile oracle可使用physical atlas选择最佳tile，但必须明确：

- 它不可部署；
- 它不等于learned visual router；
- 它只估计headroom。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
十、Stage 8：Joint 2×2 oracle
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心四臂：

CC = coarse visual + 3×3 action
FC = local-fine visual + 3×3 action
CF = coarse visual + 5×5 action
FF = local-fine visual + 5×5 action

每个arm使用privileged simulator选择candidate set中utility最高的action。

计算：
I = U_FF - U_FC - U_CF + U_CC
J = U_FF - max(U_FC, U_CF)

joint-coupled state必须同时满足：

- FF categorical outcome严格优于FC与CF；
- J超过calibration冻结阈值；
- 不是post-success stopping造成；
- candidate/effect-call预算合规。

必须加入：

- random state refinement
- random tile
- phase heuristic
- uniform coarse
- uniform fine
- visual-only
- action-only
- full-fine visual upper bound
- success-stop controls

Matched budget：

- coarse action = 9 candidate calls
- fine action = 25 candidate calls
- local visual fine = 1 tile
- adaptive、random、phase heuristic细化相同数量states

不要声称visual wall-clock compute匹配，因为输入tensor仍是224×224。
只能报告abstract information budget和action candidate budget。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
十一、统计与正式判定
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- primary unit = frozen state identity
- same states across all model seeds and arms
- 10,000 paired bootstrap
- per-model-seed report + aggregate
- primary comparison：
joint adaptive vs strongest matched-budget single-axis control
- secondary comparisons做Holm correction
- 禁止candidate-cell pseudo-replication
- 保存raw episode/state/candidate JSONL
- 独立summarizer从raw重新计算全部结果
- 独立audit脚本不得直接复用主summarizer的决策函数

正式状态只能是以下之一：

NO_GO_BASELINE_REPAIR
REVISE_STOPPING_CONFOUND
NO_GO_STATE_RESTORATION
NO_GO_NO_ACTION_BOUNDARY
REVISE_ACTION_ONLY
REVISE_VISUAL_ONLY
REVISE_NO_JOINT_COUPLING
REVISE_UTILITY_DEPENDENT
GO_SINGLE_TASK_JOINT_ORACLE

即使GO，也必须停止，不得创建：

- learned predictor
- boundary predictor
- budgeted selector
- OOD experiment
- Stage-3 branch
- π0.5 experiment
- real-robot experiment

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
十二、资源与运行安全
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 最多使用2张GPU。
- 只使用明确空闲GPU或正式PAI资源。
- 不杀、暂停、抢占、共享任何已有进程。
- runtime uid/gid保持2254:2254。
- 每次正式run使用唯一run ID。
- CPU smoke通过后才能提交正式任务。
- source、config、dataset、checkpoint、seed manifest、launcher全部SHA256绑定。
- 所有raw结果增量落盘并fsync。
- 中断后允许resume，但必须验证checkpoint与seed cursor。
- 基础设施失败使用新run ID，不能覆盖旧记录。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
十三、必须交付的文件
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

至少创建：

experiments/maniskill_stage25_repair_oracle_v1/
README.md
preregistration.yaml
PROTOCOL_FREEZE.json
manifests/
source_bindings.json
checkpoint_candidates.json
checkpoint_screen_seed_bank.json
checkpoint_final_val_seed_bank.json
confirmatory_test_seed_bank.json
oracle_source_seed_bank.json
state_bank_calibration.json
state_bank_confirmatory.json
state_bank_post_success.json
scripts/
audit_source_bindings.py
generate_disjoint_seed_banks.py
evaluate_checkpoint_grid.py
select_checkpoint_closed_loop.py
evaluate_success_semantics.py
audit_contact_metrics.py
build_stackcube_state_bank.py
audit_state_restoration.py
build_local_action_atlas.py
run_action_boundary_probe.py
run_visual_resolution_probe.py
run_joint_factorial_oracle.py
summarize_stage25.py
audit_stage25.py
baseline_repair/
success_semantics/
action_boundary/
visual_resolution/
joint_oracle/
audits/
docs/
STAGE25_FINAL_REPORT.md

若某个gate失败，不得伪造后续空结果。
必须额外执行stop-scope audit，证明后续目录、job和产物没有被创建或运行。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
十四、最终报告要求
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

最终报告必须用中文，先给人话总结，再给技术结果。

必须明确回答：

1. checkpoint selection是否是上一轮NO-GO的重要原因？
2. StackCube现在是否是健康baseline？
3. success_once与stable success的差异由什么造成？
4. physical action outcome boundary是否存在，集中在哪些phase？
5. visual refinement是否真的改变物理结果，而不只是改变action数值？
6. joint FF是否优于FC和CF？
7. adaptive oracle是否优于matched random和phase heuristic？
8. 结论是否依赖post-success states、某个model seed或某组utility weights？
9. 哪些内容完全没有被测试？
10. 下一步是否有资格进入predictor learnability？

所有科学结论必须分别标记：

- confirmed code semantics
- observed association
- privileged oracle evidence
- bounded inference
- not tested

不要停留在建议层面。完成代码审计、预注册、实现、测试、正式实验、独立重算和最终封账。
遇到科学gate失败时立即停止后续阶段，但仍需完成独立审计与最终报告。

## 当前执行指令的预注册覆盖（确认性结果产生前冻结）

冻结时间：2026-08-14 UTC。

用户在本轮明确要求：必须完成计划内全部实验，不能因为任一科学 gate 未通过而停止验证其他实验；训练与推理可使用 2–8 张 A800。该最新显式指令覆盖上文中“gate 失败立即停止后续阶段”以及“最多 2 张 GPU”的冲突条款，但不修改任何科学阈值，也不放宽最终结论。

因此本轮采用以下不可回溯口径：

- 所有 gate 均照原阈值计算、报告并决定证据层级与最终状态；
- gate 失败不会阻止 checkpoint、success semantics、contact、state restoration、action boundary、visual resolution 与 joint oracle 的预注册实验继续运行；
- 下游结果在上游 gate 未通过时只能作为受限诊断或 privileged oracle evidence，不能反向宣称上游 gate 已通过；
- 不使用 confirmatory 结果选择 checkpoint、PCA radius、effect threshold、utility weights、visual tile rule 或修改任何阈值；
- 仍在 oracle 结论后停止，不实现 learned predictor、deployable selector、OOD、π0.5 或真实机器人实验；
- 资源拓扑允许 2–8 张 A800，实际 formal run 的 GPU 数、资源池、故障恢复与 UID/GID 绑定必须在提交前固定并审计。

最终状态使用上文预注册枚举；当多个 gate 失败时，按 preregistration.yaml 冻结的优先级给出一个主状态，并完整列出所有次级失败，禁止因继续执行后续实验而将失败 gate 解释为通过。
