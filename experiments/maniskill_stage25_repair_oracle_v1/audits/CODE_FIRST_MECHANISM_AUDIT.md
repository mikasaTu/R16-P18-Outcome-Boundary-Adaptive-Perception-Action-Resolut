# Step4 code-first 机制审计

Protocol: `R16-P18-MS4-STAGE25-REPAIR-ORACLE-V1`

本审计只反解本轮代码中导致指标升降的机制，不生成新 idea。最终报告必须把下列“代码语义”与 formal observation 分开；没有观测支持时，不把实现意图写成已证实机制。

## 证据标签

- **confirmed code semantics**：由冻结源码、manifest 和测试直接确认。
- **observed association**：formal 同状态配对结果中的关联。
- **privileged oracle evidence**：使用 simulator outcome 选 tile/action/state 的上限证据，不可部署。
- **bounded inference**：代码与配对观测共同支持、但仍受任务和状态银行限制的解释。
- **not tested**：本轮没有实验覆盖。

## 1. Checkpoint repair 实际改变了什么

**confirmed code semantics**

`select_checkpoint_closed_loop.py` 不再按 imitation validation loss 选 checkpoint。每个 task/model seed 先在 32 个固定 screen seeds 上筛 top-2，再在包含该 32-seed 前缀的 100 个 validation seeds 上按以下字典序选择：`success_hold5` 最大、`success_at_end` 最大、`post_success_loss` 最小、step 最早。旧 loss-selected checkpoint 和 final checkpoint 均保留为对照。全新 100 confirmatory seeds 只在选择完成后使用，因此不会反向调参。

这段代码能够区分“checkpoint selection 是否影响稳定闭环表现”，但不能单独证明上一轮所有任务失败都由 checkpoint selection 引起；该判断必须结合旧 checkpoint 与新 checkpoint 的同 seed 闭环差异。

## 2. `success_once` 与稳定成功为何可能分离

**confirmed code semantics**

四个 arm 使用同一 selected checkpoint 和同一 confirmatory seed 顺序：

- `fixed_horizon`：到 horizon 前一直执行策略；
- `terminate_first_success`：首次 simulator success 后停止；
- `terminate_hold5`：连续五步 success 后停止；
- `neutral_after_hold5`：连续五步 success 后停止调用 policy，arm delta 置零、gripper 保持最后合法命令。

因此 `fixed_horizon` 对比前三个 privileged diagnostics 可以定位成功后的继续动作与失稳是否关联；`neutral_after_hold5` 仍下降则说明“仅停止新策略动作”不足以保持状态，可能存在物理漂移或末端保持动力学。它们不是 deployable stopping policy，也不允许重新选择 checkpoint。

## 3. Action atlas 如何制造并检测 outcome boundary

**confirmed code semantics**

每个冻结 state/model seed 首先生成 native observation 下的 nominal ACT chunk。代码在 train split action chunks 中按标准化前 4-step arm action 距离稳定检索 256 个邻居，对 residual 做 SVD，固定符号后取前两个局部 PCA 方向。以 `[-1,-0.5,0,0.5,1]^2` 形成 5×5 nested fine grid；3×3 coarse grid严格取 fine grid 的 `(0,2,4)×(0,2,4)` 子集。越界 candidate 保留原值并标 invalid，不 clip。

所有 candidate 只扰动前三个 arm 坐标，并显式共享 state bank 的
`last_legal_gripper_command`。这个值是 ManiSkill 控制器实际生效的合法
归一化夹爪命令；若 metadata 自身越界，代码直接失败而不是裁 candidate。
raw ACT nominal 与替换后的 atlas center 分字段保存。该区别很重要：v21
观察到 raw gripper 仅高于上界约 0.12%–0.19%，旧实现却把这一共同的、
非 PCA 维度问题传播到 25/25 cells，制造了 0% validity。修复不会改变
任何 arm residual，因此后续 boundary 增减不能归因于扩大或收缩 PCA 网格。

每个 valid candidate 从完全相同 simulator state 恢复，执行 4-step candidate prefix、最多 20 次 base-policy follow-up、再 neutral hold 5 步，并重复 3 次。invalid cells 通过 padding 保持调用形状，但不进入科学 outcome。lattice 只使用 40 条 4-neighbor edge；edge 在 categorical tuple、stable success、recoverability 变化，或标准化 translation/rotation/progress effect distance 达到冻结阈值时算 boundary。汇总单位是 state，不是 edge。

Primary utility 明确由 stable success、clipped progress、support、intended/unintended contact、drop/slip、recoverability 和极小 action-residual penalty 组成。它是 privileged candidate ranking，不是 learned effect model。

## 4. Visual refinement 如何可能跨越物理结果

**confirmed code semantics**

`V_coarse` 把 128×128 RGB bilinear downsample 到 64×64 后再 upsample；`V_local_fine` 在 coarse 背景上恢复 4×4 grid 中一个 32×32 native tile，并使用冻结的两像素 cosine feather。所有图像之后仍走相同官方 ACT wrapper，因此这是 information-resolution oracle，不是 tensor/token/wall-clock compute reduction。

对每个 state，代码一次并行执行 16 个 local-fine tile 的 policy forward，把所得前 4-step chunks按冻结标准化坐标映射到 native physical atlas，并用 hidden simulator utility 选 `oracle_tile`。随后 coarse、oracle tile、hash-random tile、phase tile 各自重新生成完整 25-candidate physical atlas并真实 rollout；因此 formal outcome 差异不是仅比较 action 向量，而是比较同一恢复 state 下的 simulator 结果。

这里存在重要证据边界：tile screen 的 16 次 policy call 对所有 state 都发生，且 tile 由 hidden outcome 选择；它只量化 privileged headroom，不能声称 deployable visual router，也不能声称 visual wall-clock budget matched。

## 5. Joint 2×2 的精确定义

**confirmed code semantics**

四臂共享同一个 state：

- `CC`：coarse visual atlas，只允许 3×3 nested indices；
- `CF`：同一 coarse visual atlas，允许全部 5×5 candidates；
- `FC`：oracle local-fine atlas，只允许 3×3 nested indices；
- `FF`：同一 oracle local-fine atlas，允许全部 5×5 candidates。

每臂都用 simulator utility 选集合内最佳 valid candidate。代码计算 `I=FF-FC-CF+CC` 和 `J=FF-max(FC,CF)`。joint-coupled state 还必须满足 FF 的 `(stable_success, phase_outcome_rank, recoverable)` category 严格优于 FC、CF，`J` 达到 calibration 冻结阈值，非 post-success，且 budget flag 为真。

`full_native_upper` 是 native visual 的 25-candidate privileged 上限，不属于 matched arm。best-action recall 把各臂最佳 action 映射到同一冻结 native atlas 的最近 valid action后比较，避免直接把不同视觉条件下重新中心化的 cell index当成同一物理 action。

## 6. Matched allocation control 实际匹配了什么

**confirmed code semantics**

confirmatory bank 有 64 states。对每个 model seed 与每组 utility weights，adaptive 按 `U_FF-U_CC` 排序细化前 32 states；random 用 protocol/state/seed 的 SHA256 稳定排序取 32；phase heuristic 按 near-completion、in-hand、pre-contact、free-space 顺序取 32。三者细化 state 数完全相同。

`visual_only` 在 adaptive-selected states 用 FC，否则 CC；`action_only` 用 CF，否则 CC；`strongest_single_axis` 更严格地对每个被选 state 取 FC/CF 中 utility 更高者；joint adaptive 用 FF。random-state 与 phase-heuristic 各在其自己的 32-state allocation 使用 FF，random-tile 则在 adaptive 的 32 states 使用随机 tile 的 fine atlas。因此“最强单侧”是逐 state privileged control，不是先看 aggregate 后选一个固定轴。

匹配的是 abstract information budget（1 local tile）与 action candidate opportunities（coarse 9、fine 25），不是 wall-clock：所有视觉输入仍为 224×224 policy tensor，且 oracle tile screen 有额外 16 policy calls。报告不能把这组比较写成部署延迟等价。

## 7. 统计与敏感性为何能/不能支持机理

**confirmed code semantics**

primary unit 是 frozen state identity；三 model seeds 使用相同 64 identities。adaptive 与 controls 的差值按 state 做 10,000 次 paired percentile bootstrap，固定 bootstrap seed 16018。primary、success-dominant、progress-dominant 三组 utility weights均从相同 raw outcomes重算；calibration bank先冻结 PCA radius、effect threshold 和 J threshold，confirmatory outcomes不能调参。

Gate 同时要求 coupling density、recall 或 regret、adaptive-vs-random/phase 的 CI、至少 2/3 model seeds方向一致。独立 `audit_stage25.py` 从 raw JSONL重算 gate，不调用主 summarizer 的决策函数。最终状态按 baseline、stopping、restoration、action、joint 的预注册优先级判定；用户要求继续全部实验只改变执行范围，不改变最终状态优先级。

## 8. 预先限定的机理解释边界

**bounded inference**

只有在同 state、同 model seed 的 FF outcome/utility稳定超过 FC 与 CF，并同时胜过 matched random/phase，且对 utility weights与排除 post-success 稳健时，才可表述为“StackCube 冻结 state bank 上存在 privileged joint headroom”。即便满足，也不能推断 learned selector 可学、闭环长期收益、跨任务/OOD泛化或真实机器人效果。

**not tested**

本轮没有训练 effect/boundary predictor，没有 deployable state/tile/action selector，没有新 BC/Diffusion Policy/DINO-WM/π0.5，没有 OOD、跨任务联合机制、端到端 matched latency、真实机器人或论文 acceptance 验证。PushCube 只承担 baseline health diagnostic，不进入 joint oracle。
