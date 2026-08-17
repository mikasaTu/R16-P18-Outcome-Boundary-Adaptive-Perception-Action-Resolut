---
feishu_title: "step1"
feishu_url: "https://icnbwz7kd1ui.feishu.cn/wiki/SbJ3wiILHig6uzk6KXWc6wNqnWn"
feishu_wiki_token: "SbJ3wiILHig6uzk6KXWc6wNqnWn"
feishu_object_token: "S4LPdnINBoBKvSxLV7Bc7xRPno0"
feishu_revision: 7
---

<title>step1</title>

# 四、第一步的详细方案

## 名称

R16-P18 Stage 1：Small-BC Joint Outcome-Boundary Falsifier

中文：

小型行为克隆策略上的视觉—动作联合结果边界证伪实验。

## 第一阶段唯一目标

回答：

在不依赖大 VLA、不依赖完整世界模型的情况下，联合视觉—动作分辨率分配是否真的比单侧自适应更好？

# 五、第一步所用环境与任务

## 仿真器

固定：

ManiSkill3
版本：固定最新稳定 release / exact Git commit
robot：panda_wristcam
control_mode：pd_ee_delta_pose
obs_mode：rgb + state

## 任务

### T1：PushT-v1

作用：

• 非抓取接触；

• 接触点和推动方向决定结果；

• 动作有多种可行轨迹；

• 适合研究粗动作网格 aliasing。

### T2：PegInsertionSide-v1

作用：

• 窄孔对齐；

• 毫米级视觉和动作变化会导致成功/碰撞；

• 预期 outcome boundary 最密集。

### T3：PushCube-v1

作用：

• 较宽松和平滑；

• 作为 negative control；

• 验证方法不会处处细化。

# 六、小型 BC Baseline

## 模型名

BoundaryBC-S

## 模型结构

### 视觉编码器

ResNet-18
输入：128×128 RGB
取 layer3 的 8×8 feature map
1×1 projection 到 128 维
得到 64 个 micro visual tokens

### 本体状态

qpos / qvel / tcp pose
→ 2-layer MLP
→ 128 维 proprio token

### 动作头

2-layer Transformer
或 3-layer MLP aggregator
输出 H=8 的 action chunk

### 动作

pd_ee_delta_pose
translation + rotation + gripper

### 执行方式

预测 8 步
执行前 4 步
重新观测

## 为什么不用 π0.5 作为第一步

因为这里需要隔离：

• 视觉 token 分配；

• 动作分辨率；

• 两者耦合。

如果直接用 π0.5，任何结果都可能来自：

• VLM 表征；

• flow sampling；

• 大模型预训练；

• action expert；

• normalization；

• 多视角融合。

小型 BC 可以让失败归因更清楚。

# 七、数据准备

每个任务准备：

200 条成功 demonstration

优先使用 ManiSkill 官方 demonstration；不足时使用官方 motion-planning 或 teleoperation 工具补齐。

划分必须按 episode seed：

160 train
20 validation
20 test

不允许按帧随机拆分，否则同一轨迹会泄漏到训练和测试。

训练：

3 个 seed
相同数据
相同 optimizer steps
相同参数量

# 八、Baseline 健康门

在实现任何 adaptive resolution 之前，先训练并评测原始 BoundaryBC-S。

建议门槛：

任务

基础成功率要求

PushCube

≥80%

PushT

40%–90%

PegInsertion

25%–80%

原因：

• 太低：策略根本没学会任务；

• 太高：没有提升空间；

• 中等成功率更适合研究边界错误。

若不满足，先修数据和 BC，不得开始 R16-P18。

# 九、建立 Outcome-Boundary State Bank

每个任务保存 32 个状态，共 96 个状态。

阶段

每任务数量

free-space approach

8

pre-contact / pre-grasp

8

contact onset

8

near completion / post-contact

8

每个状态必须保存：

task_id
episode_seed
simulator state 或 deterministic replay prefix
RGB
proprioception
base BC action chunk
object pose
goal pose
contact state
task progress

其中 simulator privileged state 只能用于：

• 生成 boundary label；

• 计算真实 outcome；

• 评测。

不能作为 proposed 方法的输入。

首先验证：

restore same state
\+ execute same action
→ outcome 可重复

# 十、视觉分辨率的固定预算设计

ResNet feature map 为：

8×8 = 64 micro tokens

设最终视觉预算固定为：

16 tokens

## Uniform baseline

将每个 2×2 micro cell 平均：

8×8 → 4×4
= 16 uniform tokens

## Adaptive resolution

从相同 16-token 预算开始。

一次 refinement transaction：

将一个高价值 coarse token 拆成 4 个 micro tokens：+3
将一组低价值的 2×2 coarse tokens 合并成一个 macro token：-3

总 token 仍为：

16

允许每次决策最多进行：

2 次 split-merge transaction

所以 adaptive 方法不会获得额外 token。

# 十一、动作分辨率的固定预算设计

基础 BC 输出 action chunk：

A_0

从 demonstrations 中检索当前状态附近的 64 个动作，计算局部 PCA，得到两个主要动作变化方向：

d_1,d_2

## Full-fine oracle

构造：

7×7 = 49 个局部 action candidates

只用于生成隐藏真值和上限，不允许 budgeted 方法全部读取。

## 固定预算方法

先建立：

3×3 coarse action grid = 9 candidates

然后对一个候选 cell 做局部细分：

增加 4 个 sub-candidates

最终每种 budgeted 方法都使用：

13 action candidates

所有实验臂必须使用同样的：

• 13 个 action candidate slots；

• policy calls；

• local effect-model calls；

• simulator action opportunities。

# 十二、局部 Effect Model

第一步不训练完整世界模型，只训练一个小型局部后果预测器：

g\_\phi(z,a) \rightarrow \hat y

其中输出

y

包括：

object pose delta
goal progress
contact onset
unintended contact
grasp/release state
collision
short-horizon success probability
recoverability

它只预测未来 4–8 个控制步，不生成图像。

建议结构：

visual/proprio feature
\+ action chunk candidate
→ 3-layer MLP
→ outcome vector

这个模型的作用只是：

便宜地预测哪里可能跨越 outcome boundary。

它不是最终论文中的必需结构，但可以避免第一步完全依赖 simulator oracle。

# 十三、Joint Boundary Score

## Visual-only score

某个 visual cell 从 coarse 变为 fine 后：

V(v) = D[ g(z_v^{fine},A_0), g(z^{coarse},A_0) ]

表示视觉细节对基础动作后果预测的影响。

## Action-only score

某个 action cell 内相邻动作后果差异：

A(c) = \max\_{a_i,a_j\in c} D[g(z,a_i),g(z,a_j)]

表示动作 cell 是否包含多个不同结果。

## Joint score

关键是计算：

视觉 cell 的精细信息，是否改变了 action cell 内不同动作的相对结果。

可以定义：

J(v,c) = D \left[ g(z_v^{fine},a_c)-g(z_v^{fine},A_0), g(z^{coarse},a_c)-g(z^{coarse},A_0) \right]

若

J(v,c)

大，说明：

看清 visual region v
会改变对 action region c 的判断

这才是真正的 perception-action coupling。

# 十四、第一步实验臂

## B0：Uniform

16 个均匀视觉 token
13 个均匀动作候选

## B1：Random Adaptive

随机 split visual cell，随机 refine action cell。

用于检查自适应结构本身是否有益。

## B2：Visual-Only

视觉按：

• image gradient；

• feature saliency；

• action disagreement；

选择细化区域。

动作 grid 保持均匀。

## B3：Action-Only

视觉保持均匀。

动作根据局部 outcome curvature 细化。

## B4：Independent

视觉和动作都自适应，但分别独立选择：

argmax V(v)
argmax A(c)

## P：Joint Outcome-Boundary

选择：

(v^\*,c^\*) = \arg\max\_{v,c}J(v,c)

联合细化对应 visual/action region。

## A0：Shuffled Joint

保留所有分数和预算，但把：

当前 visual score
与另一个状态的 action boundary

随机配对。

它是最关键的机制消融。

## O：Full-Fine Oracle

64 visual micro tokens
49 action candidates

只作为非匹配计算的上限，不能进入主要公平比较。

# 十五、评测方式

## 离线 State-Bank 评测

主要指标：

### 1. Outcome Alias Rate

一个 coarse action cell 中同时包含成功和失败动作的比例。

越低越好。

### 2. Best-Action Recall

固定 13 个候选预算下，是否保留了 49-candidate oracle 中的最佳动作。

### 3. Outcome Regret

R\_{\text{oracle}} - R\_{\text{selected}}

越低越好。

### 4. Boundary F1 / AUROC

方法是否准确识别真正的视觉—动作结果边界。

### 5. Coupling Gain

G\_{\text{coupling}} = M\_{\text{joint}} - \max ( M\_{\text{visual}}, M\_{\text{action}}, M\_{\text{independent}} )

这是最重要的指标。

## 闭环评测

每个：

method × task × training seed

运行：

50 个 test episodes

总计至少：

3 tasks
× 3 training seeds
× 50 episodes

每个 method 为 450 episodes。

评测条件：

### ID

正常随机初始化。

### Geometry OOD

• object/target pose 偏移；

• PegInsertion clearance 改变；

• 接触角度变化。

### Visual nuisance OOD

• 光照；

• 背景；

• texture；

• 轻微 blur。

其中 visual nuisance OOD 是重要负控制：

它会改变图像，但不一定改变最优动作。

# 十六、Stage 1 的 GO / NO-GO 标准

## GO：进入 Diffusion Policy / DINO-WM

必须全部满足：

1. PushT 和 PegInsertion 中至少 25% sampled states 存在非平凡 outcome boundary；

2. Joint 相比 Uniform：

• Outcome Regret 相对下降 ≥15%；

• Alias Rate 相对下降 ≥20%；

3. Joint 相比最强 B2/B3/B4：

• Best-Action Recall 提升 ≥5 个百分点；

• 或 Outcome Regret 相对下降 ≥10%；

4. PushT 与 PegInsertion 都是正方向；

5. Shuffled Joint 至少消除完整增益的 50%；

6. PushCube 负控制成功率下降不超过 2 个百分点；

7. Joint 的 closed-loop：

• 成功率绝对提升 ≥5 个百分点；

• 或目标失败率相对下降 ≥15%；

8. 所有 token、candidate、model-call 和 latency 预算一致。

## NO-GO：停止当前版本

出现任意情况：

• Full-fine oracle 都没有明显收益；

• outcome boundary 极少；

• Visual-only 与 Joint 一样好；

• Action-only 与 Joint 一样好；

• Independent 与 Joint 一样好；

• Shuffled Joint 保留大部分收益；

• 只有 simulator privileged state 才能取得增益；

• 增益来自更多 token、candidate 或 effect calls。
