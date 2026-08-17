---
feishu_title: "实验报告"
feishu_url: "https://icnbwz7kd1ui.feishu.cn/wiki/GnYKwQ63EiWZUjkyeOwcJXvYnRj"
feishu_wiki_token: "GnYKwQ63EiWZUjkyeOwcJXvYnRj"
feishu_object_token: "KVUedo75RoAh7cxgAudcNWPynMb"
feishu_revision: 10
---

# 实验报告

## R16-P18 Step4 / Stage-2.5 最终实验报告

日期：2026-08-14

协议：`R16-P18-MS4-STAGE25-REPAIR-ORACLE-V1`

正式运行：`r16p18-stage25-oracle-20260814-v26`

最终判定：**`REVISE_STOPPING_CONFOUND`**

证据范围：**`SINGLE_TASK_PRIVILEGED_ORACLE_ONLY`**

## 1. 人话总结

本轮把计划中的 checkpoint repair、success stopping、contact、state bank、action atlas、visual information-resolution、joint 2×2 oracle 和 matched allocation 全部跑完了。虽然冻结协议原本要求在 action gate 失败后停止，但用户随后明确要求“必须完成全部实验，不能因为一个 gate 不到就停止”；因此所有下游实验仍被执行，阈值和最终状态优先级保持不变，失败 gate 没有被后续结果反转。

最重要的结果不是 joint oracle 的小幅正 utility，而是一个更早、证据更强的混杂：StackCube 策略经常已经成功，却因为固定 horizon 继续输出动作而把物体弄失稳。固定 horizon 下 `success_once=45.67%`，到终点只剩 `28.00%`；若在首次成功时用 simulator 特权信息立即终止，终点成功率提高 `16.33pp`，paired bootstrap 95% CI 为 `[12.00pp, 21.00pp]`，三个模型 seed 同方向。根据预注册状态优先级，正式结论只能是 `REVISE_STOPPING_CONFOUND`。

在机制层面，局部原生视觉 tile 确实会改变执行后的物理结果，而不只是改变 action 数值；这种变化集中在 pre-contact 和 near-completion。但 action boundary 的 confirmatory gate 没过，joint-coupled state 在跨模型 seed 聚合后为 `0/64`，best-action recall 没有提高，outcome regret 只下降 `3.12%`，远低于 `15%` 门槛。因此当前证据不支持进入 predictor learnability，也不构成 idea 被验证、接受或 paper-ready 的证据。

## 2. 证据标签

- **confirmed code semantics**：冻结源码、manifest、hash 和测试直接确认的执行语义。
- **observed association**：同一冻结 episode/state 上观察到的配对差异。
- **privileged oracle evidence**：使用 simulator outcome 选择停止点、tile、action 或 state 的不可部署上限证据。
- **bounded inference**：代码语义与配对观测共同支持、但受单任务和 state bank 限制的解释。
- **not tested**：本轮没有覆盖，不能外推。

## 3. 协议、源码和平台绑定

### 3.1 冻结科学源

- predecessor 审计提交：`76e71f5eae9771b83906478f0c421183e38cdd9c`
- 正式 v26 科学源码提交：`7be99cd867d27372cfc75095d459b782c0f75a66`
- 正式源码 tree：`31df45e1f28bbba57aebcd804e13c88fe14a0784`
- frozen launcher SHA256：`678db2b187d6ac32b8b34e2b60ff3ae0b43eee28b25ce7daa5c5f75e49742204`
- `PROTOCOL_FREEZE.json` SHA256：`9fd66b86e3d8176baa65da783ee5e3f41df5b1d86a841fa4535e52cd9991c7ca`
- scientific checksum manifest SHA256：`8a606d77d2e70ca6147943c01ba1dc7a8f713852f486698b7998c50c364b5e4f`
- 正式源 worktree 在运行前后均保持 clean；九个冻结科学 hash 全部通过 `sha256sum -c`。
- predecessor 目录 `experiments/maniskill_act_boundary_screen_v1/` 与审计提交相比未修改。

### 3.2 环境

- ManiSkill：tag `v3.0.1`，commit `a4a4f9272ad64b1564035874b605ceb687b63ed8`
- Python：3.11.11
- PyTorch：2.5.1+cu124
- torchvision：0.20.1+cu124
- SAPIEN：3.0.0
- 控制模式：`pd_ee_delta_pose`
- RGB：128×128 base camera；视觉 probe 后仍经官方 ACT wrapper 进入 224×224
- 正式 simulator：PhysX CUDA；state restoration：单环境 serial PhysX CPU

### 3.3 PAI 终态

- JobId：`dlc1ptg07eqpdaxy`
- 平台终态：`Succeeded / JobSucceeded`
- 创建：`2026-08-14T11:05:04Z`
- 开始：`2026-08-14T11:05:53Z`
- 完成：`2026-08-14T12:49:58Z`
- duration：6,294 秒（约 104.9 分钟）
- 资源：1 worker，2×A800，12 CPU，200Gi memory / 200Gi shared memory
- ResourceId：`quotakzri8a5wqcp`，dedicated `exp-efficiency`
- runtime uid/gid：`2254:2254`
- AIMaster、elastic 和 PAI automatic fault tolerance：关闭
- W&B：明确关闭
- PAI smoke：25/25 candidate 有效，标记为非科学实现证据
- exact-job CLI readback 始终可用，未使用浏览器 fallback。
- v26 创建时没有封存同 workflow predecessor 删除集合，因此 monitor 以 `sealed_target_count=0` 封账；没有事后用 wildcard 推断或删除历史任务记录。

## 4. 数据、seed 与执行完整性

四组新 seed bank 在 confirmatory 结果产生前冻结：checkpoint screen 32、final validation 100、confirmatory test 100、oracle source 独立 seeds。它们彼此不相交，并与 300 demo seeds、旧 100 test seeds 不相交；三个 model seeds 复用完全相同的 episode/state 顺序。

本轮实际完成：

- checkpoint screen：2 tasks × 3 model seeds × 6 checkpoints × 32 episodes；
- checkpoint final validation：每组 top-2 × 100 episodes，并保留 old loss-selected 和 final checkpoint 对照；
- baseline confirmatory：2 tasks × 3 seeds × 100 episodes；
- success semantics：4 arms × 3 seeds × 100 StackCube episodes，共 1,200 episodes；
- success trace：219,679 条逐步记录；
- state banks：32 calibration + 64 confirmatory + 16 post-success，共 112 states；
- state restoration：112 states × 3 serial repeats；
- action atlas confirmatory：64 states × 3 model seeds × 25 candidates × 3 repeats；
- visual confirmatory：64 states × 3 seeds，4 个视觉 condition atlas；
- post-success action/visual diagnostics、joint primary 和两组 utility sensitivity 全部完成；
- `FORMAL_COMPLETE.json`：`ALL_PREREGISTERED_STAGE25_EXPERIMENTS_COMPLETE`；
- `prohibited_post_oracle_work_executed=false`。

## 5. Checkpoint closed-loop repair

### 5.1 选择结果

**confirmed code semantics**：每个 task/model seed 先用 32 个 screen seeds 选 top-2，再在完整 100 validation seeds 上按 `success_hold5` 最大、`success_at_end` 最大、`post_success_loss` 最小、step 最早的字典序选 checkpoint。confirmatory test 完全不参与选择。

| Task / seed | 旧 loss checkpoint | 新 checkpoint | 旧 hold5 | 新 hold5 | 提升 | loss/stable rank inversion | 旧 checkpoint Pareto dominated |
|-|-|-|-|-|-|-|-|
| PushCube / 16018 | 25k | 30k | 58% | 63% | +5pp | 是 | 是 |
| PushCube / 16019 | 25k | 30k | 55% | 57% | +2pp | 是 | 是 |
| PushCube / 16020 | 30k | 30k | 51% | 51% | 0pp | 否 | 否 |
| StackCube / 16018 | 25k | 10k | 32% | 37% | +5pp | 是 | 否 |
| StackCube / 16019 | 25k | 15k | 40% | 47% | +7pp | 是 | 是 |
| StackCube / 16020 | 25k | 30k | 36% | 41% | +5pp | 是 | 是 |

StackCube 的 screen Spearman(`validation loss`, `stable success`) 分别为 `+0.1765/-0.5002/-0.9429`；screen winner 与 100-seed final winner 一致 `2/3`。六组中 `5/6` 出现 loss/stable rank inversion，`4/6` 的 predecessor 被 Pareto dominate。

### 5.2 对上一轮 NO-GO 的解释

**observed association**：closed-loop 选择确实把 StackCube 的 validation `success_hold5` 每 seed 提高 5–7pp，因此它是 stable baseline 接近门槛时的重要因素。

**bounded inference**：它不是上一轮顶层 `NO_GO_BASELINE_GATE` 的充分或主要单一解释。上一轮失败还包括仅 1/3 positive tasks 通过，以及 PushCube 57.33% 未达到 70%；本轮 PushCube 仍只有 53.67% `success_once`，没有恢复成健康负对照。本轮和上一轮 confirmatory seed 也不同，不能把两组 aggregate success 作未配对因果比较。

## 6. Baseline confirmatory gate

| Task | 三 seed `success_once` | Aggregate once | hold5 | end | post-success loss | seed range | retention end/once | Gate |
|-|-|-|-|-|-|-|-|-|
| StackCube | 43%, 52%, 42% | 45.67% | 30.67% | 28.00% | 17.67% | 7pp | 61.31% | **PASS，窄幅** |
| PushCube | 54%, 56%, 51% | 53.67% | 52.33% | 52.33% | 1.33% | 7pp | 97.52% | **FAIL health <70%** |

StackCube 同时满足冻结条件：once 位于 `[25%,85%]`、hold5 `≥30%`、seed range `≤20pp`、retention `≥60%`。但 hold5 仅高门槛 `0.67pp`，retention 仅高 `1.31pp`，且后续 stopping diagnostic 发现强混杂，所以只能称为本协议下“gate 通过的单任务 baseline”，不能称为稳健、多任务健康基线。PushCube 不得作为健康 task-level negative control；所有 oracle claim 均为 `SINGLE_TASK_ONLY`。

## 7. Success semantics 与 stopping confound

| Arm | success_once | hold5 | end | post-success loss | end 相对 fixed 增益 | paired 95% CI |
|-|-|-|-|-|-|-|
| fixed horizon | 45.67% | 30.67% | 28.00% | 17.67% | — | — |
| terminate first success | 44.33% | 0%\* | 44.33% | 0% | **+16.33pp** | **[+12.00,+21.00]pp** |
| terminate hold5 | 47.33% | 31.00% | 36.00% | 11.33% | +8.00pp | [+4.33,+11.67]pp |
| neutral after hold5 | 47.33% | 31.00% | 34.67% | 12.67% | +6.67pp | [+3.00,+10.33]pp |

`*` immediate termination 不可能积累五步 streak，因此 hold5=0 是 arm 定义，不是性能失败。

**confirmed code semantics**：fixed arm 持续调用 policy 到 horizon；terminate arms 在 simulator success/hold5 后停止；neutral arm 在 hold5 后不再调用 policy，arm delta 为 0，gripper 保持最后合法命令。后三者均是 privileged diagnostics，不是 deployable stopping 方法。

**privileged oracle evidence**：`terminate_first_success` 超过预注册 `+10pp`、CI lower >0、3/3 seeds 同方向，因此 `stopping_confound=true`。

**bounded inference**：成功后的后续策略动作是失稳的主要来源。对已经达到 hold5 的 episode：

- fixed 终态保留率为 `81.52%`，first-to-terminal drift 为 `5.107mm / 0.1407rad`；
- neutral 保留率为 `95.70%`，drift 降为 `1.013mm / 0.0211rad`；
- terminate-hold5 保留率为 `100%`，terminal drift 为 `0.975mm / 0.0193rad`。

neutral 的小幅非零 drift 说明物理保持/动力学仍有次要贡献，但它明显小于 fixed 下继续策略动作造成的漂移。

### 7.1 终态 trace 缺陷与独立修复审计

v26 的逐步 trace、success 指标和所有 gate 决策正确，但 terminate arms 的冗余顶层 `final_object_position/post_success_object_drift` 使用了 vector-wide 最终 snapshot：某个 vector slot 终止后仍留在共享 simulator，而其他 slot 继续，因此 `terminate_first_success` 有 133 行、`terminate_hold5` 有 93 行冗余字段不对应本 episode 的终止时刻。

修复后的代码把描述性终态字段绑定到每个 episode 的最后一条 trace。独立审计从 object pose/quaternion 重新计算 1,200 个 episode、219,679 个 trace rows，trace 内存储 drift 与 pose 重算的最大误差为 `0`，确认 raw 科学证据未被修改、decision metrics 未受影响。修正后 summary 的正式状态仍是 `REVISE_STOPPING_CONFOUND`。

## 8. Contact metric 和 state restoration

### 8.1 Contact

**confirmed code semantics**：新代码只使用明确字段 `intended/unintended_contact_onsets`、duration、max force 和 post-success onset；旧 `collisions` alias 被禁止，contact 不替代 task outcome。

脚本化审计通过：三条 positive successful replay 各有 1 次 intended onset，duration 为 63/71/61 steps，最大 intended force 为 35.587/26.680/33.238N；三条 10-step neutral negative 均为 0 onset、0 duration、0 force；所有 force 有限且无 unintended contact。`contact_metric_gate_pass=true`。

### 8.2 State banks 与恢复

- calibration：32 states，四 phase 各 8，expert/on-policy 各半；
- confirmatory：64 states，四 phase 各 16，expert/on-policy 各半；
- post-success diagnostic：16 on-policy states；
- 全部 source episode/seed 与 train、validation、旧 test、新 confirmatory test 不相交；
- 112/112 states 在单环境 CPU PhysX 下各恢复 3 次；
- restoration pass rate：100%；same-action categorical agreement：100%；
- 四步 replay 的 state/categorical 检查通过。

## 9. Physical action outcome boundary

calibration 只使用 32-state bank，未读取 confirmatory 结果。冻结选择：PCA radius `1.5`，effect threshold `0.5`；candidate validity 为 100%，5×5 fine grid 含 3×3 coarse 严格子集，越界 candidate 标 invalid 而不 silent clip。

| Phase | confirmatory boundary density | categorical repeat agreement |
|-|-|-|
| free-space approach | 2.34% | 100.00% |
| pre-grasp / pre-contact | 27.60% | 96.75% |
| object-in-hand / pre-placement | 7.55% | 99.83% |
| placement contact / near-completion | **42.14%** | **76.83%** |
| overall | **19.91%** | **93.35%** |

placement-minus-free-space 为 `39.79pp`，candidate validity 为 100%。这表明局部动作 outcome 变化真实存在，并集中在 pre-contact 和 placement/contact；on-policy state density `23.44%` 高于 expert 的 `16.38%`。

但是冻结 action gate **FAIL**：overall density `19.9089% <20%`，少 `0.0911pp`；same-action repeat agreement `93.35% <95%`。repeat disagreement 主要集中于 near-completion：该 phase 有 278/1,200 个 candidate categorical disagreement，其中 186 个涉及 stable-success repeat disagreement。故应写“observed boundary association，但没有通过 confirmatory necessary-condition gate”，不能写“action mechanism validated”。

## 10. Visual information-resolution probe

**confirmed code semantics**：coarse arm 将 128×128 下采样至 64×64 再上采样；local-fine 在 coarse 背景恢复一个 32×32 native tile，并固定两像素 cosine feather。16 个 tile 只用一次 batched policy forward（16 rows）筛选；privileged tile oracle 根据映射到 native physical atlas 的隐藏 utility 选 tile。选定视觉 condition 后重新生成完整 action atlas并实际在 simulator 中执行 candidate，不以 action-vector 距离代替物理 outcome。

| 对比（192 state×seed rows） | physical signature 改变 | phase outcome 改变 | stable success 改变 | 平均 utility Δ |
|-|-|-|-|-|
| FC visual-only vs CC | **35/192 = 18.23%** | 15.63% | 3.13% | +3.374 |
| CF action-only vs CC | 10/192 = 5.21% | 5.21% | 1.04% | +1.041 |
| FF joint vs CC | 40/192 = 20.83% | 18.23% | 4.17% | +4.491 |

FC 的 physical-signature change 按 phase 为：free-space 6.25%、in-hand 10.42%、pre-contact 27.08%、near-completion 29.17%。所有 192 行的 visual-induced first-four arm action 都非零变化；有 physical flip 的 action shift 均值 `0.0434`，无 flip 为 `0.0284`。

**privileged oracle evidence**：视觉细化确实能使策略跨越执行后的物理 outcome，而不仅是改变 action 数值。信号在 3/3 model seeds 都出现。

**边界**：这只是 information-resolution headroom。所有 tensor 仍是 224×224、oracle tile 使用 simulator-derived hidden utility、不可部署，也不是 token/latency reduction。不能声称视觉 wall-clock compute matched。

## 11. Joint FF、matched allocation 与 utility sensitivity

四臂预算：CC=`0 tile+9 actions`，FC=`1 tile+9 actions`，CF=`0 tile+25 actions`，FF=`1 tile+25 actions`。adaptive、random 和 phase-state heuristic 都只细化 32/64 states；joint allocation 复用已生成 atlas，不增加 effect-model 或 simulator calls。

### 11.1 Primary 结果

| 指标 | 结果 | 冻结门槛 | Gate |
|-|-|-|-|
| joint coupling density | **0/64 = 0%** | ≥15% | FAIL |
| best-action recall improvement | **0pp** | ≥10pp，或 regret 条件 | FAIL |
| outcome regret reduction | **3.12%** | ≥15%，或 recall 条件 | FAIL |
| joint adaptive − per-state strongest single-axis | +0.573 | CI `[+0.014,+1.649]` | 描述性正值 |
| adaptive − random-state | +3.788 | CI `[+1.217,+7.001]` | PASS secondary |
| adaptive − phase-state heuristic | +0.084 | CI `[+0.034,+0.147]` | PASS secondary |
| adaptive − random-tile | +1.363 | CI `[-1.113,+3.955]` | 不显著 |
| adaptive − phase-tile | -0.308 | CI `[-2.624,+2.004]` | 不显著 |

primary mean utility：joint adaptive `22.321`、per-state strongest single axis `21.747`、visual-only `21.291`、action-only `18.832`、random-state `18.532`、phase-state `22.236`、uniform-coarse `17.796`、uniform-fine `22.287`、full-native upper `23.761`。

每个 model seed 各出现 1/64 个满足 `categorical strictly better + J≥1` 的 isolated row，但 state 分别是 `confirmatory-051/-035/-057`，彼此不同；按冻结的跨 seed state 聚合规则，没有任何 state 在多数 seed 上复现，因此 formal coupling 为 `0/64`。在 adaptive 选中的 96 个 state×seed rows 中，per-state strongest single axis 是 FC 74 次、CF 22 次，说明视觉与动作 refinement 多数是替代关系，视觉轴占主导。

### 11.2 是否优于 random 和 phase heuristic

答案需要拆开：

- **是，限 state-allocation control**：adaptive 相对 matched random-state 和 phase-state heuristic 的 CI lower 都大于 0，且 3/3 model seeds 方向一致；Holm-adjusted p 均为 `0.00039996`。
- **否，不能扩展成完整 joint-router 优势**：相对 random-tile/phase-tile 不显著；uniform-fine 与 adaptive 只差 `0.034`；最关键的 coupling、recall 和 regret 门槛失败。

因此不能因两个 secondary allocation 对照为正，就把 joint oracle gate 写成通过。

### 11.3 Utility sensitivity、post-success 和 seed

所有 primary confirmatory states 都是 pre-success，`post_success_states_in_primary=0`；16 个 post-success states 仅作 diagnostic，不进入主统计。三组冻结 weights 的结果为：

| Weights | coupling | recall Δ | regret reduction | joint − strongest CI |
|-|-|-|-|-|
| primary | 0% | 0pp | 3.12% | `[+0.014,+1.649]` |
| progress-dominant | 0% | 0pp | 4.07% | `[+0.026,+0.897]` |
| success-dominant | 0% | 0pp | 1.84% | `[+0.008,+3.199]` |

方向对 utility weights 稳健，但失败的 joint necessary conditions 也对 weights 稳健。per-seed regret reduction 为 `-31.10%/+0.97%/+14.27%`，说明效应大小明显依赖 seed，且没有一个达到并稳定复现冻结 15% 标准。

## 12. Code-first 机理反解

本节只解释已经观察到的提升/降低，不生成新 idea。

### 12.1 为什么 checkpoint repair 有提升但不能修复顶层 NO-GO

**confirmed code semantics**：新选择器优化闭环 hold/end，而旧选择器优化 imitation loss；两者目标不一致。`5/6` rank inversion 直接显示 loss 不是 stable-success 的可靠排序变量。

**bounded inference**：StackCube validation 的 5–7pp 改善来自 checkpoint ranking 目标对齐，而不是网络结构改变。但 PushCube 仍不健康、StackCube confirmatory hold5 仅刚过门槛，说明 checkpoint 选择只能修复一部分 baseline 质量。

### 12.2 为什么 success_once 高而 end/hold 低

**confirmed code semantics**：fixed evaluator 在首次 success 后仍继续 temporal-aggregated ACT；neutral arm 切断后续 policy action，gripper 仅保持合法命令。

**observed association**：终止/neutral 明显提高 end，且 conditional drift 大幅下降。

**bounded inference**：继续策略动作是主要失稳机制；物理漂移是小而非零的次要机制。由于 stopping arms 使用 simulator privileged success，本轮没有测试如何从 observation 学到可部署 stop rule。

### 12.3 为什么 action boundary 集中在 near-completion，却 gate 失败

**confirmed code semantics**：每个 state 的 5×5 PCA lattice 执行相同 4-step prefix、20-step base continuation、5-step neutral hold，重复 3 次；boundary 由 categorical、success、recoverability 或 effect-distance crossing 定义。

**observed association**：near-completion density 42.14%，但 repeat agreement 只有 76.83%；free-space density 2.34% 且 agreement 100%。

**bounded inference**：contact/placement 附近确有陡峭 outcome surface，但同一 action 的 stochastic repeat flip 同时抬高 density 并降低可复现性。冻结 gate 正是为了拒绝把不稳定的 repeat noise 当成可靠 boundary，因此总体 FAIL 合理。

### 12.4 为什么视觉轴比动作轴强

**confirmed code semantics**：oracle 在 16 个 tile 上比较 policy chunk，再把 chunk 映射到 native physical atlas；动作 fine arm 只在同一 coarse-visual atlas 中把 9 个候选扩大到 25 个。

**observed association**：FC 改变 physical signature 18.23%，CF 仅 5.21%；adaptive 选中 states 的 strongest axis 74/96 为 FC。oracle tile 只在 7.81% 行与 phase tile 相同、4.69% 与 random tile 相同。

**bounded inference**：局部视觉恢复经 policy 改变了 atlas 中心和完整物理轨迹；单纯增加同一中心附近的 action candidates 边际较小。这个解释只适用于 privileged tile screen 和当前 StackCube state bank。

### 12.5 为什么 adaptive utility 有小幅提升但 joint coupling 为零

**confirmed code semantics**：adaptive 按每个 state 的 `FF−CC` 对 64 states 排序取 32；primary control 在同一 selected state 上取逐 state `max(FC,CF)`，不是较弱的 pooled fixed axis。joint-coupled 还必须同时满足 FF categorical 严格优于 FC/CF 和冻结 `J≥1`。

**observed association**：joint−strongest 的平均值略正，但 coupling=0、recall Δ=0、regret 仅 -3.12%；FC 在 strongest axis 中占 77.08%。

**bounded inference**：小幅 utility 增益主要来自 state ranking 与单轴增益的逐点拾取，不是跨 seed 稳定的视觉×动作非加性交互。由于 uniform-fine 几乎追平 adaptive，不能把该差值解释成可部署 routing headroom。

## 13. 对十个必须问题的明确回答

1. **Checkpoint selection 是否是上一轮 NO-GO 的重要原因？** 是 stable StackCube 指标的重要原因之一：同 validation seeds 提高 5–7pp；但不是充分或唯一原因，上一轮多任务 gate 和 PushCube 健康度问题仍存在。
2. **StackCube 现在是否是健康 baseline？** 按本轮冻结四项 gate 是，且 3-seed 范围稳定；但仅窄幅通过，并存在强 stopping confound，所以只能用于 single-task diagnostic，不能泛称健康多任务 baseline。
3. **success_once 与 stable success 的差异由什么造成？** 主要由首次成功后继续执行 ACT 动作造成；neutral/termination 对照和 drift 重算共同支持。物理保持漂移有小的次要贡献。
4. **Physical action outcome boundary 是否存在，集中在哪些 phase？** 有观察到的 outcome change，集中于 placement/contact（42.14%）和 pre-contact（27.60%）；但 repeat agreement 和 overall density 未过 gate，不能称为确认性 boundary 证据。
5. **Visual refinement 是否真的改变物理结果？** 是 privileged oracle 证据：FC vs CC 在 35/192 行改变执行后的 physical signature，且每个 condition 都执行 simulator rollout；不是只比较 action vector。
6. **Joint FF 是否优于 FC 和 CF？** utility 上有小幅/局部优势，但预注册意义上的答案是否：跨 seed coupled state 为 0、recall 不增、regret 降幅不足，FF 没有形成稳定 joint coupling。
7. **Adaptive oracle 是否优于 matched random 和 phase heuristic？** 对 matched random-state 和 phase-state allocation 是，CI lower>0 且 3/3 seeds 同向；对 random/phase tile control 不显著，整体 joint gate 仍失败。
8. **结论是否依赖 post-success、某 seed 或 utility？** 不依赖 post-success，主统计完全排除；方向对三组 utility 稳健；效应大小明显依赖 seed，isolated coupled rows 在三个 seed 上不是同一 state。
9. **哪些内容完全没有测试？** learned predictor、boundary predictor、deployable selector/router、closed-loop adaptive episodes、OOD、LIBERO/其他任务 joint replication、真实 token/latency reduction、可部署 stopping、Diffusion Policy、DINO-WM、π0.5、real robot 均未测试。
10. **下一步是否有资格进入 predictor learnability？没有。** stopping confound 优先成立，action gate 和 joint gate 又都失败；本轮必须在 privileged oracle 后封账。

## 14. Call、budget 与 latency accounting

每个 visual condition（coarse/oracle-tile/random-tile/phase-tile）跨 192 atlas 都严格记录：

- 4,800 candidate opportunities、576 candidate repeats；
- 192 nominal policy rows；
- 288,000 rollout policy rows，实际 batched forward calls 3,840；
- 417,600 simulator-step rows，实际 batched simulator calls 5,568；
- invalid padding rows/steps：0；
- effect-model calls：0。

四 condition 的 atlas latency 分别约 582.35/583.96/583.45/583.70 秒；16-tile screens 共 3,072 policy rows、192 actual batched forward calls、0 simulator calls、3.77 秒。joint allocation/statistics 约 6.29 秒且不新增 simulator/effect calls。

这些数字证明 action candidate budget 和 abstract visual tile budget可核对，但视觉输入 tensor 始终为 224×224，因此明确拒绝 wall-clock compute matched 或真实 token-saving 声明。

## 15. 独立审计、测试与封账

- independent audit：`INDEPENDENT_STAGE25_AUDIT_PASS`；从 raw 独立重算 baseline、stopping、restoration、action、joint、Holm 和最终状态；
- trace terminal audit：`SUCCESS_TRACE_TERMINAL_AUDIT_PASS`；1,200 episodes / 219,679 trace rows；
- mechanism audit：`MECHANISM_REVERSE_ENGINEERING_AUDIT_PASS`；scope=`reverse_explanation_only_no_new_idea`；
- unit/static tests：正确冻结环境下 `26 passed`；
- Python compileall：通过；
- frozen scientific checksums：全部通过；
- source predecessor immutability：通过；
- post-oracle prohibited work：未执行；
- final status 在 formal summary、corrected summary 和 independent audit 中完全一致。

主要 hash：

- `FORMAL_COMPLETE.json`：`09e6dedaf0da6970764d99b7342b9e2bdb351fd171b9cc68063446947109f323`
- formal summary：`ab1976e633fbcc7ca2108224c6131922f6c15d0ca7d5a892f27b779e450ebbd5`
- trace-corrected summary：`94b7b795b9c1fe2f6ed84affd771a6f1b2ea29f6a627c38edacf7828c988c3ba`
- independent audit：`35f0ab027c7faca13d6779a87ab78b2679acc1db1e9ef9adfbf45e20d6b9adec`
- trace terminal audit：`e0d1af1fa0a11b79d5ee2a72574eda265e49d22d5a577d4c878cad1d9d116e40`
- mechanism audit：`b1d522ee2b2455274333bfaddc49d9d63e3c8978f2fe4e0a76c0dd59344e25ee`
- PAI monitor result：`0122338de7ced2015e3f8981b0c347fed8193dca5d7945df89ae4579f01b875b`
- full raw result archive：`d568b7e51fb55f7057840a4a838a6819f596a6675b00b2c53f0c7f514f91f79e`
- PAI log archive：`8203d68e3684f51882e48831d98bea15c8b3a9d603af3eb2db6fda902fe42f47`
- PAI registry archive：`7ffe6defa066575e95bbb6148226ac1399d599843f18ef0f32907313f761d54a`
- 384-file raw inventory：`11bb023cc2303e1992127aeb7b4ab52bf02d43a196bcbc03a84e5e701b151a9a`

## 16. 最终判定与停止范围

**最终判定：`REVISE_STOPPING_CONFOUND`。**

判定优先级为：baseline repair → stopping confound → restoration → action boundary → joint oracle。StackCube baseline 和 restoration 通过；stopping confound 首先触发，因此即使 action/joint gate 也失败，正式状态仍必须是 `REVISE_STOPPING_CONFOUND`，不能事后改成更下游的状态。

本结论只说明当前 StackCube ACT baseline 的成功后执行语义污染了稳定成功判定，并且本轮 privileged joint oracle 未满足 necessary conditions。它不表示 R16-P18 idea 被验证、接受、证伪或具备论文结论。实验在 oracle 后停止，没有创建 learned predictor、deployable selector、OOD、Stage-3、π0.5 或 real-robot 工作。
