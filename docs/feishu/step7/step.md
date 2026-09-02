<title>step7</title>

继续完成任务“按照这个计划继续推进验证，并且把结果放到对应step7的实验报告（首先在该idea的飞书云文档里实验规划下创建新的子云文档创建名叫“step7”云文档，然后把计划内容复制进step7文档中，再在step7下创建子文档“实验报告”）中（飞书云文档用skill连接）。针对有提升或者降低的那机制的代码，你需要反解（参考 code-first-ideation）什么原因导致带来的提升和降低，这里不生成新的idea，但是需要了解清楚机理。此外还需要把代码和测试结果等信息上传到github的main分支中，必须完成计划里面的全部实验，不能因为一个gate不到就停止验证其他实验。（使用 pai-vla-training、pai-spot-autoresume、pai-web-training-orchestrator）”

训练和推理可以使用pai 的4-16张 a800，使用闲时资源池（挂在robot池子下，现在单节点 cpu/mem 上限是 88 core/1.4TB）。能并行处理训练或者推理的任务就并行完成，但是每天北京时间9.30-9.40、19.30-19.40时间段内不能有你提交的任务；都需要在9.30和19.30先停掉你提交的pai任务，过了9.40和19.40后再重新resume提交。必须完成计划里面的全部实验，不能因为一个gate不到就停止验证其他实验。

# 任务：R16-P18 Stage-3 / step1 — S1 预算可行域重建（零 rollout）

## 背景（只读事实，不要重新推导，不要试图翻案）

- 仓库 mikasaTu/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut，main 分支
- Stage 2.6 终态 NO_GO，但该判定的每一组支撑数据都建立在不合格的 shared prefix 上：
  三个 seed 的 shared-prefix fidelity 全部失败，action 误差 0.237-0.391，
  平移 0.0152-0.0177 m，旋转 0.184-0.411 rad，observation hash 一致率 0%
- 纯数值 state 回放精确到 2.38e-7；恢复 observation / RNG / pending chunk /
  controller 后执行同一 action 仍差 0.46-0.65 mm。现有诊断：公开 state 不含
  PhysX contact solver warm-start cache
- 预算设计本身不成立：coarse/full 算力比 0.7388；50% 预算下所有自适应臂 refine
  0 个状态，joint 与 fixed 的成功率差恒为 0
- 正任务 screen 全失败：StackCube anchor 未过，PushCube 52.67% 低于 70% 负控门
- 唯一正信号在视觉轴：两个诊断任务、三套权重下一致为正
  （Peg balanced seed means 0.551/0.622/0.694），但从未在合格 prefix 上验证过
- 本步不重跑任何 rollout，不动 shared prefix，不碰核心机制，不做任何
  joint vs single-axis 的比较

## 本步目标

只回答一个问题：在当前模型与任务上，"固定预算下把分辨率向 outcome boundary
集中"是否存在非空可行域。拆成两问：
(a) 视觉轴上是否存在 (coarse, fine) 分辨率对，使某个预算 α 下可 refine 的状态
比例达到预注册门槛；
(b) 动作轴上是否存在同样的对。
(b) 的答案决定 joint 说法能否保留，必须在花任何 GPU rollout 之前给出。

## 硬约束

1. 禁止任何仿真 episode、任何 env.step() 循环、任何 PAI 作业。只允许 dev05
   单卡前向 profiling。profiling 的输入观测优先复用 Stage 2.6 已存 artifacts；
   确实没有存则允许 env.reset() 取观测，仍禁止 step
2. Stage 2.6 及以前的 artifacts 一律只读冻结。新产物写到 experiments/r16p18_stage3/；
   若仓库已有不同命名约定，沿用既有约定并在 INVENTORY 首行写明
3. 算力必须实测。FLOPs 解析估算不能作为唯一依据：wall-clock 中位数与 FLOPs
   两套数都要给，并单独列出二者结论不一致的格点
4. 不修改任何 frozen protocol / preregistration 文件；不改模型权重；不重训
5. 每个产出文件记录 SHA256，追加到 experiments/r16p18_stage3/SHA256SUMS
6. 任何复算与 Stage 2.6 已记录数值不一致，立即停止并报告，不要"修正"

## 执行顺序

### S1.0 底物盘点

定位并列出：

- 0.7388 这个比值的产生代码、原始计时数据、以及它的分子分母各自是什么
- 预算 α 的分母到底是什么（full 分辨率跑满所有状态？还是别的口径），
  写出代码里的确切定义位置
- perception 侧实际能跑通的分辨率格点（不是理论上支持的）
- action 侧的"分辨率"在代码里的确切定义（chunk 长度 / 插值步长 / token 数 /
  其他）及其可取值
- 预算约束的实现位置：具体哪一行决定了 refine 名额
- 正任务 screen 的任务清单、seed 数、成功率原始记录
  输出 S1_INVENTORY.md：路径、行号、字段名、SHA256。
  若 action 侧不存在独立于 perception 的可调分辨率参数，停在这里报告——
  那意味着"两条轴"在实现层面本来就不是两条轴，后续所有 joint 分析都无效。

### S1.1 复现 0.7388

用盘点到的代码与口径重测 coarse/full 比值。warmup 次数、重复次数、batch、
是否同步 CUDA，全部写死并记录。
输出 S1_COST_REPRO.json：新测值、原值、相对偏差、测量方差。
偏差超过 5% → 停止并报告，先解释偏差来源，不要继续。

### S1.2 两轴代价曲线

对 perception 与 action 两轴，分别实测每个可用分辨率格点的单状态代价
（wall-clock 中位数 + FLOPs），含误差棒。
输出 S1_COST_CURVE.json + 两张曲线图（横轴分辨率，纵轴归一化到该轴 full 的代价）。

### S1.3 可行域映射

定义 ρ = cost(coarse)/cost(fine)。两种口径都算，并说明代码实际实现的是哪种：

- 不复用：k/N ≤ α − ρ
- 复用 coarse 中间结果：k/N ≤ (α − ρ)/(1 − ρ)
  对每一个 (coarse, fine) 格点组合 × α ∈ {0.25, 0.50, 0.75}，输出 k/N 上界。
  输出 S1_FEASIBILITY.json + 两张热力图（视觉轴一张，动作轴一张）。
  单独列出：使 k/N ≥ 0.20 的全部 (coarse, fine, α) 组合，两轴分别列。
  0.20 是本步预注册门槛，同时给出把它换成 0.10 / 0.30 时可行组合数的变化，
  供后续按检验功效重新定档。

### S1.4 退守分支定价（只摆数字，不下结论）

若动作轴不存在任何满足门槛的组合，明确写出这一事实，并单独给出视觉轴的可行域。
不要在本步做任何"是否收窄论文范围"的判断或建议。

## G1 判定（写入 S1_DECISION.md，逐条记录 PASS/FAIL，不许跳过任何一条）

1. S1_COST_REPRO 与 0.7388 一致（或偏差已给出可验证解释）
2. 视觉轴存在 k/N ≥ 0.20 的 (coarse, fine, α) 组合
3. 动作轴存在 k/N ≥ 0.20 的 (coarse, fine, α) 组合
4. 选定组合中的 coarse 与 fine 都是模型原生支持的分辨率，不需要改权重或重训
   最后一行写唯一标签：
   PROCEED_JOINT（2、3 均过）/ PROCEED_VISION_ONLY（2 过 3 不过）/
   BLOCKED_BY_BUDGET（2 不过）/ BLOCKED_BY_SUBSTRATE（1 或 4 不过）

## 终止条件

G1 写完即停。不要启动 fresh-env 重执行，不要跑正任务 screen，不要提交任何
GPU 训练或 rollout 作业，不要写 Stage-3 最终报告，不要开始 S2 的任何准备工作。
交回 S1_DECISION.md 与候选 (coarse, fine, α) 表，等人工确认。
