---
feishu_title: "step3"
feishu_url: "https://icnbwz7kd1ui.feishu.cn/wiki/JEhpwKVxIinKlHkxaCFceTC0n1Z"
feishu_wiki_token: "JEhpwKVxIinKlHkxaCFceTC0n1Z"
feishu_object_token: "RNkWdXQGuoiICHxSPjtcwqqxn2c"
feishu_revision: 10
---

# step3

你是一名独立机器人学习实验 agent。你的任务是在以下仓库中完成

R16-P18 Outcome-Boundary Adaptive Perception–Action Resolution 的下一阶段验证：



Repository:

mikasaTu/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut



当前已知主分支最新提交预计为：

eefa8f42a7a0017d85de9624bd481336030a9240



但你必须在开始时重新读取实际 main HEAD、分支、Git tree 和工作区状态，

不得盲信上述 SHA。



目标：

1. 完成并严格收口当前 ManiSkill RGB-ACT Stage-2。

2. 只有 Stage-2 baseline gate 和 oracle boundary gate 允许时，才创建 Stage-3。

3. Stage-3 第一次实现 learned effect predictor、boundary predictor、

   budgeted joint visual/action resolution selector。

4. 同时验证以下新线索：

   A. outcome-flip boundary margin；

   B. boundary-normal anisotropic action refinement；

   C. marginal value-of-compute joint router；

   D. contact-phase hysteresis；

   E. 可选的 action-commutator boundary proxy。

5. 最终输出 GO、NO_GO 或 REVISE，不得因为代码能运行就声称 idea 成立。



────────────────────────────────────────

一、不可变证据和范围

────────────────────────────────────────



必须首先阅读：



\- README.md

\- docs/EXPERIMENT_REPORT.md

\- experiments/maniskill_act_boundary_screen_v1/README.md

\- experiments/maniskill_act_boundary_screen_v1/preregistration.yaml

\- experiments/maniskill_act_boundary_screen_v1/task_selection.json

\- experiments/maniskill_act_boundary_screen_v1/baseline/

\- experiments/maniskill_act_boundary_screen_v1/manifests/

\- 当前 main 和 maniskill-act-boundary-screen-v1 分支历史



现有 LIBERO Stage-1 和 ManiSkill Stage-2 文件均视为 immutable evidence。

不要重写、删除或偷偷补充旧结果。



新工作必须创建：



branch:

maniskill-boundary-adaptive-resolution-v1



directory:

experiments/maniskill_boundary_adaptive_resolution_v1/



protocol_id:

R16-P18-MS3-ADAPTIVE-RESOLUTION-V1



Git 中只提交：

\- 代码；

\- 配置；

\- preregistration；

\- manifests；

\- SHA256；

\- summaries；

\- small JSON/JSONL/CSV；

\- 报告；

-必要视频索引。



大型数据、checkpoint、state bank、rollout surface 存储在固定 CPFS 路径，

Git 中保存绝对路径、所有权、字节数、哈希和生成命令。



────────────────────────────────────────

二、资源和安全边界

────────────────────────────────────────



\- 最大同时使用 2 张 GPU。

\- 只使用明确空闲 GPU。

\- 不杀死、不暂停、不抢占、不共享其他用户进程。

\- PAI runtime uid/gid 使用 2254:2254。

\- AIMaster、elastic restart、automatic platform restart 关闭。

\- 不将 W&B secret、token 或密码写入 Git。

\- 所有正式 run 必须记录 JobId、PodUID、GPU 型号、CUDA、Python、依赖、

  source commit、数据哈希、首个真实 optimizer step 和完整 checkpoint marker。

\- 基础设施失败与科学负结果严格分开。

\- smoke、synthetic、CPU unit test 不得冒充 baseline、closed-loop 或科学结果。



────────────────────────────────────────

三、先完成 Stage-2

────────────────────────────────────────



当前 GitHub 只显示 FORMAL_TRAINING_IN_PROGRESS。

先检查 PAI/CPFS 是否已有未提交的最终训练或评测结果。



必须完成：



1. 四个 formal tasks：

   \- PullCubeTool-v1

   \- PushT-v1

   \- StackCube-v1

   \- PushCube-v1



2. 三个 ACT model seeds：

   \- 16018

   \- 16019

   \- 16020



3. checkpoint 只能由 validation imitation loss 选择；

   test 指标不得参与 checkpoint selection。



4. 每个 task/model seed 使用 100 个冻结 closed-loop seeds。



5. 记录：

   \- success_once

   \- success_at_end

   \- episode length

   \- intended/unintended contacts

   \- collisions

   \- policy latency

   \- policy calls



6. 执行 Stage-2 baseline gate：

   \- positive task success ∈ [0.25, 0.85]

   \- maximum seed range ≤25pp

   \- PushCube success ∈ [0.70, 0.98]

   \- 至少两个 positive tasks 通过



7. 对通过任务创建 Stage-2 frozen test state bank：

   \- 每任务 64 states

   \- free-space 16

   \- pre-contact/pre-grasp 16

   \- contact/insertion/placement 16

   \- near-completion 16

   \- restore repeats=3



8. 生成 Stage-2：

   \- 5×5 local action atlas

   \- visual intervention atlas

   \- 4×5 joint probe



9. 计算：

   \- action boundary density

   \- visual boundary density

   \- joint coupling density

   \- best-action recall

   \- outcome regret



只有满足：

\- 至少两个 positive tasks action boundary density ≥0.20

\- 至少两个 positive tasks joint coupling density ≥0.15

\- recall +10pp 或 regret -15%

\- PushCube joint boundary density ≤0.10



才允许进入 Stage-3。



若失败：

\- 输出 Stage-2 final report；

\- 给出 GO/NO_GO/REVISE；

\- 不实现 learned selector；

\- 停止科学实验。



不得事后修改 Stage-2 已冻结阈值。



────────────────────────────────────────

四、冻结 Stage-3 preregistration

────────────────────────────────────────



在观察任何 Stage-3 predictor 或 selector 结果前，创建并提交：



\- preregistration.yaml

\- resolution_contract.json

\- budget_contract.json

\- data_split_contract.json

\- statistics_contract.json

\- hypothesis_cards/

\- environment_lock.json

\- SHA256SUMS



Stage-3 假设：



H1:

deployable observation 能预测 action、visual 和 joint outcome boundary。



H2:

predicted marginal value of fine resolution 比 generic uncertainty 更能预测

realized regret reduction。



H3:

在 matched budget 下，joint visual/action selector 优于 random、periodic、

uncertainty、visual-only、action-only 和 independent combination。



H4:

收益集中在 pre-contact/contact/near-completion boundary states。



H5:

PushCube negative control 不退化，也不会频繁错误启用 fine resolution。



────────────────────────────────────────

五、Resolution contract

────────────────────────────────────────



先审查官方 ACT 的视觉 feature/token seam，并在结果前冻结确切实现。



视觉要求：



V0 coarse:

\- 冻结 ResNet；

\- 对空间 feature map 做确定性 pooling；

\- 保留约原生 token 数的 1/3；

\- 不使用 privileged region/object mask。



V1 adaptive:

\- coarse tokens；

\- selector 额外开放 1–4 个局部区域的 fine tokens；

\- selector 在选择区域前不能读取这些 fine tokens。



V2 full-fine:

\- 使用全部原生 spatial tokens；

\- 作为高计算 upper bound，不作为 matched baseline。



若官方输出为 7×7 map，可优先考虑约 16/自适应/49 token 的合同；

若实际结构不同，选择最接近的确定性 1:3 token-ratio，并记录原因。

不得为 proposed 单独更换视觉 backbone。



动作要求：



\- base chunk 来自冻结 ACT；

\- 只编辑最先执行的 4 steps；

\- non-gripper dimensions 用 training action neighbor PCA 得到两个局部方向；

\- A0 coarse = 3×3 = 9 candidates；

\- A1 fine = 5×5 = 25 candidates；

\- gripper continuous value 保持 base；

\- binary gripper controls 单独报告；

\- 所有候选从相同 restored state 评估。



新线索 BNAR：

\- 预测 boundary normal/tangent；

\- normal 5 levels；

\- tangent 3 levels；

\- 总计 15 candidates；

\- matched baselines 必须也是 15 candidates。



────────────────────────────────────────

六、Stage-3 atlas 数据

────────────────────────────────────────



不得使用 Stage-2 frozen test bank 训练 predictor。



对每个通过的 positive task 和 PushCube：



Train atlas:

\- 128 states/task

\- 四个 phase 各 32

\- 只能来自 train identities



Validation atlas:

\- 64 states/task

\- 四个 phase 各 16

\- 只能来自 validation identities



Test atlas:

\- Stage-2 frozen 64 states/task

\- 只用于最终评测



split 单位必须是完整 trajectory/episode/initial-state identity。



每个状态生成并持久化：



\- 25-cell action outcome surface

\- visual tile interventions

\- 4 visual tiles × 5 action candidates joint surface

\- U00, U10, U01, U11

\- categorical outcomes

\- continuous effects

\- utility

\- action boundary margin

\- visual boundary margin

\- nearest outcome-flip witness

\- best action identity

\- all simulator restore/accounting metadata



任何 restore 不稳定状态全部排除并记录 exclusion reason。



────────────────────────────────────────

七、Effect predictor 和 selector

────────────────────────────────────────



冻结 ACT。



训练 Short-Horizon Effect Predictor：



输入：

\- visual tokens

\- proprio

\- base action chunk

\- candidate residual

\- resolution mask



输出：

\- short_success

\- intended_contact

\- unintended_contact

\- collision

\- recoverable

\- object translation/rotation/progress

\- outcome utility



至少训练 3 个 predictor seeds。

可构造 ensemble uncertainty baseline。



训练 Marginal Value-of-Compute Router：



仅使用 coarse-pass 可见输入，预测：



\- U00_hat

\- U10_hat

\- U01_hat

\- U11_hat

\- DeltaV

\- DeltaA

\- interaction

\- boundary margin

\- important visual region

\- action boundary normal



selector 使用：

argmax_r U_hat(r) - lambda \* cost(r)



lambda 只可在 validation 上调节。

不得使用 test/closed-loop 成功率调参数。



────────────────────────────────────────

八、新线索模块

────────────────────────────────────────



A. Outcome-Flip Margin



比较：

\- ensemble uncertainty

\- boundary probability

\- boundary margin



offline gate：

\- AUROC ≥0.75

\- AUPRC 比 prevalence 高 ≥0.15

\- margin 与 fine-resolution gain Spearman ≥0.4

\- top20% critical states 捕获 ≥50% realized regret

\- 相对 uncertainty AUPRC +0.05



未过则仅保留诊断。



B. Boundary-Normal Anisotropic Refinement



比较同样 15 candidates：

\- predicted normal

\- uniform

\- k-means

\- random orientation

\- permuted sensitivity



gate：

\- 至少两个 positive tasks：

  recall +10pp 或 regret -15%

\- 不能输给 uniform/k-means

\- 不得利用 oracle test normal



C. Marginal Value-of-Compute Joint Router



比较：

\- visual-only

\- action-only

\- independent DeltaV + DeltaA

\- direct U00/U10/U01/U11 joint prediction



只有 direct joint 显著优于 independent，才支持 nonadditive interaction claim。



D. Contact-Phase Hysteresis



状态：

free-space → pre-contact → onset → sustain → release → post-contact



冻结：

\- tau_enter

\- tau_exit < tau_enter

\- hold steps ∈ {1,2,3}，只在 validation 选择



比较：

\- framewise

\- hysteresis

\- same-allocation histogram shuffled in time



gate：

\- mode switches -30%

\- allocation count 相同

\- contact success 不低于 framewise

\- 至少一个 positive task 提升



E. Optional Action-Commutator Proxy



仅先做 offline atlas probe：

\- AB vs BA successor/contact outcome difference

\- random pair

\- unsigned magnitude

\- shuffled identity

\- zero-score sham



只有在 held-out boundary prediction 上额外 AUPRC ≥0.05，

才允许进入 selector。

否则停止此支线，不创建新的昂贵 world-model full run。



────────────────────────────────────────

九、实验 arms

────────────────────────────────────────



Offline 必须包括：



C0 coarse/coarse

C1 full/full upper bound

C2 periodic matched

C3 random matched

C4 uncertainty matched

C5 visual-only

C6 action-only

C7 independent visual+action

C8 joint MVoC

C9 joint + margin

C10 joint + BNAR

C11 joint + hysteresis

C12 privileged oracle



Closed-loop 不得直接全部跑满。



先由 validation/offline gate 选择：

\- base ACT

\- coarse/coarse

\- full/full upper bound

\- random matched

\- uncertainty matched

\- independent

\- joint MVoC

\- 最多两个通过 offline gate 的新线索



集成 smoke 每 arm 5–10 episodes，不进入统计。



Formal closed-loop：

\- 3 ACT seeds

\- 每 task/seed 100 identical frozen test seeds

\- all methods 使用相同 initial states

\- stochastic methods 使用预先冻结、可追踪的 RNG lineage

\- test seed 与 demo seed 完全不相交



────────────────────────────────────────

十、预算匹配

────────────────────────────────────────



每个 decision/episode 持久化：



\- visual tokens

\- refined tiles

\- backbone calls

\- policy calls

\- selector calls

\- effect predictor calls

\- action candidate count

\- executed actions

\- fine opportunities

\- fine decisions

\- latency

\- GPU time

\- budget remaining



比较方法必须做到：



\- average visual tokens within 1%

\- average action candidates within 1%

\- effect predictor calls within 1%

\- policy calls equal

\- execution opportunity accounting complete



创建两类 matched control：



1. Rate-matched：

   validation 调节阈值，使 fine rates 相同。



2. Schedule-matched：

   使用相同 fine-allocation histogram，

   但将 allocation 随机或周期地放到其他 decision times。



若 proposed 优势依赖更多 calls/candidates/tokens/retraining，

结论必须判为 mechanism collapse。



────────────────────────────────────────

十一、统计和主 gate

────────────────────────────────────────



使用 paired percentile bootstrap：

\- 10,000 replicates

\- 95% CI

\- bootstrap seed 16018

\- paired by task/model-seed/test-seed



Predictor gate：

\- boundary AUROC ≥0.75

\- AUPRC prevalence +0.15

\- utility Spearman ≥0.5

\- top-5 best-action recall ≥0.70

\- ECE ≤0.10



Offline selector gate：

\- 至少两个 positive tasks

\- recall +10pp 或 regret -15%

\- 2/3 predictor seeds 方向一致

\- budget exact



Closed-loop GO：

\- joint vs strongest matched baseline：

  paired 95% CI lower bound >0 on at least two positive tasks

\- pooled positive-task absolute success gain ≥8pp

\- at least 2/3 ACT seeds same direction

\- PushCube noninferiority margin ≥-3pp

\- budget error ≤1%

\- boundary-state gain > non-boundary-state gain

\- complete call/opportunity accounting



p95 latency 增长超过 25% 时不得隐藏；

必须报告 success-latency Pareto。



────────────────────────────────────────

十二、失败解释与停止规则

────────────────────────────────────────



如果 full-fine oracle 没收益：

NO_GO，当前任务不支持 idea。



如果 oracle 有收益但 predictor gate 失败：

REVISE_INPUT，边界存在但当前 observation/latent 不可辨识。



如果 predictor 成功但 offline selector 失败：

REVISE_RESOLUTION_OPERATOR 或 utility definition。



如果 offline 成功但 closed-loop 失败：

REVISE_EFFECT_MODEL_COMPOUNDING_OR_LATENCY。



如果 visual-only≈joint：

不支持联合动作分辨率 claim。



如果 action-only≈joint：

不支持联合视觉分辨率 claim。



如果 independent≈joint：

不支持非加性交互 claim。



如果 random≈joint：

不支持 outcome-boundary timing claim。



如果 PushCube 大量触发 fine 或退化 >3pp：

selector false-positive，REVISE。



任何阈值不得在看到 test 结果后修改。



────────────────────────────────────────

十三、产物

────────────────────────────────────────



至少生成：



README.md

preregistration.yaml

resolution_contract.json

budget_contract.json

data_split_contract.json

environment_lock.json

task_decision.json

state_bank/

atlas/

predictor/

selector/

budget_ledgers/

offline_gate.json

closed_loop_gate.json

ood_results.json

failure_audits/

STAGE3_REPORT.md

SHA256SUMS



STAGE3_REPORT.md 必须明确分开：



\- source/code correctness

\- data gate

\- predictor learnability

\- offline oracle result

\- deployable selector result

\- closed-loop task result

\- OOD result

\- new clue result

\- compute/latency

\- limitations

\- GO/NO_GO/REVISE



不得把：

\- CPU test

\- synthetic fixture

\- training loss

\- validation imitation loss

\- oracle privileged result



包装为 deployable selector 或 closed-loop 成功。



每完成一个 gate，普通 Git commit 固化：

\- exact source commit

\- artifact SHA

\- run identity

\- result scope



只有全部 formal result 验证完成后，才合并到 main。



请自主推进直到：

1. Stage-2 最终报告完成；以及

2. 若 Stage-2 允许，Stage-3 完成 GO/NO_GO/REVISE 报告。



除非缺少必要权限、数据或凭证，不要中途向用户询问一般性设计问题。

遇到基础设施失败，先生成 fail-closed audit，再修复并使用新 run_id 重跑；

不得覆盖失败证据。
