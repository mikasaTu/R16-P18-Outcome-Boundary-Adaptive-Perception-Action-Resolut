---
feishu_title: "实验报告"
feishu_url: "https://icnbwz7kd1ui.feishu.cn/wiki/UOaFwX6X7iAQ0Nk1zJ3cdce7nQg"
feishu_wiki_token: "UOaFwX6X7iAQ0Nk1zJ3cdce7nQg"
feishu_object_token: "LI2pdPfDOovCdrxbfc1cPlAenLb"
feishu_revision: 6
---

<title>实验报告</title>

# Step5 / Stage-2.6：Counterfactual Completion Gate

**形式化结论：NO_GO_SHARED_PREFIX_FIDELITY**

本轮完成了计划中的全部实验，但 shared-prefix fidelity 是预注册中优先级最高的 gate，三组模型 seed 均失败。因此后续 predictor 与闭环结果只能作为探索性机制证据，不能反转主结论，也不能据此进入 Stage-2.7。

## 1. 实验身份与可追溯信息

- 协议：R16-P18-MS5-STAGE26-COUNTERFACTUAL-COMPLETION-V1
- 任务：ManiSkill3 StackCube-v1，复用 Stage-2.5 已选 checkpoint，model seeds 16018 / 16019 / 16020
- 正式完成 run：r16p18-stage26-counterfactual-20260816-v13
- PAI 成功 Job：dlc1exqlfu0iiwaa，8×A800 idle/autoresume
- Git commit：75f513282f6002fd60a9827ef9247b9ca3c56fa6
- Git tree：67283b34f66578ee221b0a9a6ef220d17966c67b
- 协议冻结 SHA256：acab0372362d441a5b5b3a432cb4858289ea8a527cf1f1affc1f4586e0a8a220

训练、校准、confirmatory seed banks 分别为 512 / 128 / 200 episodes，三者互斥，三个模型使用相同顺序；confirmatory 未用于 checkpoint、结构、温度或阈值选择。

## 2. 实验覆盖

- 7,565 个 rollout capsules
- 22,695 条 counterfactual branch rows
- 120 个 completed collection shards
- 3 × 64 states × 10 steps = 1,920 条 shared-prefix comparisons
- 200 confirmatory seeds × 3 model seeds × 7 arms = 4,200 个闭环 episodes
- 10,000 次 source-episode clustered offline bootstrap
- 10,000 次 paired closed-loop bootstrap 与 sign-flip；secondary comparisons 使用 Holm correction

## 3. Shared-prefix fidelity

三个 seed 都远未达到 action ≤ 1e-6、translation ≤ 1e-5 m、rotation ≤ 1e-4 rad、categorical agreement 100% 的冻结阈值。

- 16018：action 0.323340；translation 0.016322 m；rotation 0.184156 rad；categorical 91.72%；observation hash 0%
- 16019：action 0.237250；translation 0.017701 m；rotation 0.411081 rad；categorical 83.75%；observation hash 0%
- 16020：action 0.391112；translation 0.015209 m；rotation 0.380347 rad；categorical 86.09%；observation hash 0%

开发探针显示公开 simulator state 的数值回放最大误差只有 2.38e-7，rerender 最多相差一个 RGB 灰度；但即使恢复 observation、RNG 顺序、pending ACT chunk 和 controller state，执行完全相同的首个 action 仍产生约 0.46–0.65 mm 平移与 0.016–0.021 rad 旋转误差。代码与探针共同支持的有限诊断是：公开状态没有包含 PhysX contact solver warm-start/cache。该诊断不放宽正式阈值。

## 4. Predictor 离线结果

按冻结的字典序选择了 3,495 参数 linear probe。三 seed 的 stop-beneficial AUPRC 为 0.392 / 0.449 / 0.609，ECE 为 0.055 / 0.037 / 0.040，NOT_DONE false-stop 为 2.79% / 4.10% / 4.10%，DONE_FRAGILE recall 为 28.57% / 34.78% / 71.43%。只有 seed 16020 通过全部条件，未达到至少 2/3 seeds 通过的 gate。

leave-one-model-seed-out AUPRC 仅为 0.283 / 0.133 / 0.101，说明 hold-vs-continue 的可学习信号对 base-policy checkpoint 非常敏感，不能视为稳定的跨 seed counterfactual advantage。

## 5. 闭环 paired evidence

- fixed horizon：end success 33.83%，success_once 47.83%，post-success loss 14.00%
- privileged terminate first success：end success 47.17%；相对 fixed +13.33pp，95% CI [+10.33, +16.50]；三个 seed 为 +13.5 / +16.0 / +10.5pp
- privileged neutral after hold5：+1.67pp，CI [-0.33, +3.67]，Holm 后不显著
- learned counterfactual gate：end success 28.83%；相对 fixed -5.00pp，CI [-8.00, -2.17]，sign-flip p=0.0013；三个 seed 为 +4 / -6 / -13pp
- learned success-only：-5.50pp，CI [-9.00, -2.17]
- fixed-time matched stop：-8.83pp，CI [-11.67, -6.17]
- random matched stop：-11.00pp，CI [-14.50, -7.67]

Learned gate 对 privileged terminate gain 的恢复比例为 -37.5%，不是正恢复。completion head latency 相对 fixed policy inference 为 1.84%，policy calls 不高于 fixed；这里只作调用与延迟核算，不宣称 token 或 wall-clock compute 节省。

## 6. 提升与降低的代码机理反解

**Privileged terminate 为何提升：**它在首次 simulator success 后立即终止，机械地消除了后续 policy action 将方块推离稳定堆叠的机会。因此 post-success loss 变为 0，三个 seed 都获得正向 end-success 增益。Neutral hold5 只小幅提升且不显著，说明“停止策略调用”和“让 episode 直接结束”不是同一机制。

**Learned gate 为何降低：**冻结规则要求 Q_hold 达到阈值且 Q_hold-Q_continue 达到 advantage 阈值连续两步。该规则在 seed 16020 的 stop rate 达到 74%，平均 stop step 84.95；它伤害 33 个原本 fixed 会成功的 episodes，只救回 7 个。三个 seed 的 rescued / harmed 分别为 15/7、7/19、7/33。过早 false stop 使 success_once 从 47.83% 降到 39.50%，下降 8.33pp，远超允许的 2pp。

**为何不能归因于稳定的反事实信号：**counterfactual gate 与 success-only stop 集合的 Jaccard 为 0.672 / 0.306 / 0.830；它虽比 stop-rate-matched fixed/random controls 少伤害一些，但自身仍显著为负。离线 labels 在 first-success 前后快速翻转：first-success-6 主要应 continue，而 first-success / hold5 更常应 hold；历史特征没有在三个 checkpoint 上稳定定位这个窄边界。

## 7. 独立审计与工程验证

- 独立审计：INDEPENDENT_STAGE26_AUDIT_PASS
- 从 raw 重新计算 7,565 条 branch label rows，problems 为空
- 独立重算 fidelity、causal gain、offline gate、learned gain 与最终优先级，均与 summarizer 一致
- 契约测试：13 passed
- SCIENTIFIC_SHA256SUMS 全量通过
- FORMAL_COMPLETE SHA256：7b73af8682523d803a897c57990ab5bf30b687d61cd5d4049a553cd0392fad88
- summary SHA256：9617b8e812e3e1b7e46d6eee50a7a9b68bc9e535cd36fb91215ec18d97755a44
- independent audit SHA256：b3aa40398393db28bf8c50a1235d325f0df7722eae0ff4e8ec82e393e8916ba0

## 8. 八个问题的直接回答

1. shared-prefix 是否一致？否，三 seed 全部正式失败。
2. stopping confound 是否仍存在？Privileged termination 的探索性闭环结果支持存在，但无效 shared-prefix 使本轮不能给出匹配前缀的因果确认。
3. hold 与 continue 差异能否学习？只有 1/3 seed 过 offline gate，不能稳定学习。
4. learned gate 恢复多少 oracle gain？-37.5%。
5. 是否只是成功检测？没有正向 gain 可归因；与 success-only 高度重叠且两者都降低表现。
6. false stop 是否伤害 success_once？是，下降 8.33pp。
7. 三个 model seed 是否一致？否，learned effect 为 +4 / -6 / -13pp。
8. 是否有资格进入 visual × temporal Stage-2.7？否。

## 9. 证据边界与停止点

- Confirmed code semantics：冻结 checkpoint/seed/threshold、非 privileged predictor inputs、七 arms、fail-on-overwrite/autoresume、测试与哈希通过。
- Observed paired evidence：4,200 闭环 episodes 与配对统计显示 learned gate 聚合为负且跨 seed 不一致。
- Privileged oracle evidence：terminate-first-success 三 seed 正向；neutral hold5 小且不显著。
- Learned deployable evidence：offline gate 失败，闭环 -5pp，success_once -8.33pp。
- Bounded inference：缺失 PhysX solver cache 是 restoration divergence 的最佳支持解释；early false stops 是 learned 降低的直接 trace 机理。
- Not tested：OOD、第二任务、spatial visual/action routing、token saving、wall-clock saving、Stage-2.7、pi0.5、真实机器人。

**停止：**本轮不创建也不运行 Stage-2.7，不宣称 idea 已验证或 accepted。

## 10. 运行与服务记录收尾

- v13 PAI Job **dlc1exqlfu0iiwaa** 终态为 Succeeded，FORMAL_COMPLETE 与独立审计已持久化。
- 在 v13 成功后，使用固定 CLI/OpenAPI 两阶段流程精确删除本工作流被取代的 9 条 Failed/Stopped 服务记录：v3、v4、v5、v6、v7、v8、v9、v10、v12；每条均返回 deleted=true 且 verified_absent=true。
- 只删除 PAI 服务行；所有 CPFS raw、log、manifest、checkpoint 与 Git 证据均保留。
- 最终 Git commit：75f513282f6002fd60a9827ef9247b9ca3c56fa6；tree：67283b34f66578ee221b0a9a6ef220d17966c67b。
- browser_not_used=fifo_not_applicable。
