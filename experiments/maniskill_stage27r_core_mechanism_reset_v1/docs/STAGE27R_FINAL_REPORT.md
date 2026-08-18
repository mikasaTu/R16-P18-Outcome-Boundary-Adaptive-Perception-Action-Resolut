# Stage-2.7R 最终报告：Outcome-Marginal-Value Core-Mechanism Reset

最终状态：**`NO_GO_CORE_MECHANISM`**。

本报告只裁决冻结的 Stage-2.7R policy/task/screen/protocol，不能外推为对整个
R16-P18 研究方向的普遍否证。由于 task screen 全失败，正式 Stack/Peg oracle
结果只能作为 diagnostic oracle evidence；它们不能升级 claim tier，也不能反转
上游失败。按冻结协议，所有已登记 oracle arms 仍执行完毕。

## 运行范围与证据身份

- Screen anchor `StackCube-v1` 及所有已执行的第二正任务候选均失败；`selected_positive=null`。
- `PushCube-v1` repaired `success_hold5` 聚合为 `0.5266666667`，低于 70% 负控门；
  `selected_negative=null`，没有正式负控制结果。`StackCubeHard` 与
  `PickCube/LiftPegUpright` fallback 未实现，属于已披露的协议限制。
- 正式诊断 oracle 使用 `StackCube-v1`、`PegInsertionSide-v1`，6 个 shard（3
  model seeds × 2 tasks），每 shard 16,320 rows，共 **97,920 rows**；每任务 96
  states、34 conditions（CC、CF、16 个 FC tile、16 个 FF tile）、每个
  state-condition 5 repeats。model seeds 为 `16018, 16019, 16020`，utility
  weights 为 balanced、success-dominant、progress-dominant。
- primary unit 是 `source_episode`；repeats 在 state-treatment 内聚合；paired
  bootstrap 为 10,000 replicates，secondary family 使用 Holm correction。
- `RESULT_VECTOR.json`、`statistics.json`、`ORACLE_VALIDATION.json`、独立审计和
  SHA256 manifest 已在本地 Git archive 中准备并校验通过；将随本次 commit 发布，
  raw shards 仍保留在 CPFS。关键 hash：RESULT_VECTOR
  `bbdd69a67f1cd5d3f760196524549ae7b442ac21afc5f95963194b4eb3b5a095`；statistics
  `4cc71eff2251f008bfb9910e8a4065459d998a9a635d8dc5d075b40896ce87bd`；oracle
  validation `be515e8f4fb12825a254ff046cffd0ad0939427d3a598ec7d293725439668bdb`。
- 六个 raw shard 的 SHA256（manifest 同时绑定 path、rows、bytes、hash）为：Peg
  16018 `798490f0514865cf63383cfd7238415150f0ffe9e618fcdb7e7fcced1a79e4c7`；Peg
  16019 `18d6c4d8c56d86a93334721ab5d2aa6ff430def9b82898b851ab8a1415287793`；Peg
  16020 `abed063078025d97dcc52f69ddb713c07bbb33ae6bea7e64eb6b3a37211a8186`；Stack
  16018 `9be2072119703fe1e8b2e1ecc16f3488cc49c75b10f606e44b1d93fc5c0581c2`；Stack
  16019 `86eb7d00e618486783f3ebb6c77ec6845ded1640172dace4e4f256b01f2a0981`；Stack
  16020 `e98c5e4a5d772d7f7375116b4d9e225fd987abbe333ec98e91d13c19222fba1f`。raw shards
  共 426,794,614 bytes，未进入 GitHub-safe archive，保留在 CPFS；本地归档选取
  37 files / 36,268,200 bytes，`SHA256SUMS` 为 38/38 校验通过。
- independent audit：`INDEPENDENT_AUDIT.json` 的 all checks PASS；raw outcome、compute
  accounting、paired statistics、split leakage、no-privileged-input、unit/compile/
  deterministic smoke、fail-on-overwrite、shard autoresume、scientific SHA256 与
  predecessor immutability 均 PASS。前序目录当前 hash 与冻结 hash 一致。

## confirmed code semantics

- 同一 unified multi-resolution policy 支持 CC/FC/CF/FF。每个相机先走共享
  ResNet18 的 112×112 global branch；fine 保留 global branch，并从原始 128×128
  RGB 加一个共享 crop branch，拼接后进入 transformer。正式 run 采用 4×4 grid，
  即 16 个 tile candidates、两路 RGB camera。
- 8-step treatment 对所有 arms 相同，随后切换相同 fine/native continuation，最多
  20 steps 或统一的 success hold5 terminate。CC 为 coarse visual + coarse action；
  FC 只加 visual crop；CF 只改 action schedule；FF 同时改变两者。
- coarse action 在 treatment step 0 及每 4 个已执行 step query，并复用 4-step
  chunk；fine action 每一步重新 query，只执行 fresh chunk 的第一步。代码确认了
  schedule，但 raw rows 没有保存逐步 action vectors。
- success semantics 是任意连续 5 步满足成功 predicate（`success_hold5`），首次
  hold5 后使用相同 privileged evaluator termination；stopping predictor 不是本阶段
  treatment。
- matched-prefix 使用 fresh-reset replay；deterministic smoke 的 broadcast action
  max abs=0、translation=0、RGB max difference=0、categorical agreement=100%，
  rotation `5.16e-8` rad，oracle fidelity pass rates 为 `[1.0, 1.0]`。这支持 reset-
  prefix replay 的可复现性；但正式条件是 **serial fresh-reset replay**，不是同时
  10-condition stepping。
- physical tile 选择由 repeated physical utility 的最大值确定；它是 offline、
  privileged oracle label，不是可部署的 crop selector。joint 定义是
  `FF - max(FC, CF)`，不是 `FF - CC`。
- 代码/审计确认了 shared global-plus-local branch、8-step treatment、4-step cached
  action 与 1-step re-query、common continuation、success_hold5 及 accounting
  公式；报告不把这些语义升级为 learned router 能力。

## observed paired evidence

### Screen、配对效应与跨 seed

screen `success_hold5`（seed 顺序 `[16018,16019,16020]`）为：

| task | CC | FF | screen |
|---|---:|---:|---|
| StackCube-v1 | [0.10, 0.15, 0.16] | [0.00, 0.02, 0.04] | fail |
| PegInsertionSide-v1 | [0.00, 0.00, 0.01] | [0.00, 0.00, 0.00] | fail |
| PlugCharger-v1 | [0.00, 0.00, 0.01] | [0.00, 0.00, 0.00] | fail |
| PullCubeTool-v1 | [0.00, 0.00, 0.00] | [0.00, 0.01, 0.01] | fail |
| PushT-v1 | [0.09, 0.04, 0.05] | [0.06, 0.03, 0.01] | fail |
| PushCube-v1（negative candidate） | [0.55, 0.43, 0.60] | — | fail，aggregate 0.5267 < 0.70 |

效应单位为 utility points；区间是 paired 95% bootstrap CI，p 为 Holm sign-flip p。

| weight | task | visual `FC-CC` | action `CF-CC` | joint `FF-max(FC,CF)` |
|---|---|---:|---:|---:|
| balanced | Peg | +0.6223 [0.3715, 0.9152] | +0.4553 [-1.5187, 2.8168] | -1.8188 [-3.4490, -0.5640] |
| balanced | Stack | +1.2934 [0.1799, 2.7355] | -4.3739 [-7.9486, -1.3700] | -5.2429 [-8.6949, -2.3045] |
| success-dominant | Peg | +0.3667 [0.1986, 0.5724] | +0.7329 [-1.4812, 3.3800] | -1.8851 [-3.6778, -0.5229] |
| success-dominant | Stack | +1.3794 [0.0915, 3.0737] | -5.0811 [-9.3367, -1.6001] | -5.9791 [-10.0047, -2.5661] |
| progress-dominant | Peg | +0.9849 [0.6073, 1.4287] | +0.1501 [-1.6006, 2.2857] | -1.8375 [-3.3141, -0.6780] |
| progress-dominant | Stack | +1.2479 [0.3037, 2.4585] | -3.7090 [-6.7430, -1.1565] | -4.5819 [-7.5145, -2.1042] |

因此，visual `FC-CC` 在两个诊断任务和三套 weights 均为正；Peg 的 action
aggregate CI 跨零，Stack 的 action 在三套 weights 均为负；joint 在两个任务、三套
weights 均为负。balanced utility 的每-seed means 也保留该边界：visual 为
Peg `[0.5506,0.6224,0.6940]`、Stack `[0.0874,2.3401,1.4529]`；action 为 Peg
`[-1.6832,3.3454,-0.2964]`、Stack `[-5.7781,-3.2466,-4.0971]`；joint 为
Peg `[-3.7267,-1.3525,-0.3773]`、Stack `[-5.6453,-5.5496,-4.5338]`。

balanced joint positive-state fraction（按 seed 16018/16019/16020）为 Peg
`[0.4167,0.3750,0.4063]`、Stack `[0.2917,0.2500,0.4479]`；六个 seed-task
组合均达到预注册的 10% fraction 门，但 task-level joint mean/CI 仍为负，因为
负 joint states 的损失更大，不能反转 joint gate。

### 预算与可比性

独立 accounting 重算给出 all-coarse `92,438,200,000,000` FLOPs、all-fine
`125,120,200,000,000` FLOPs，coarse/full=`0.7387951746`。所以 25% budget 和
50% budget 连 all-coarse 都不可行；50% 下所有 adaptive allocation arms（不含
all-fine baseline）refine 0 states，
paired joint-versus-fixed success difference 为 0，CI `[0,0]`。75% 仅为 exploratory：
joint 相对 strongest fixed 的 success gain `+0.0121528`（+1.2153pp），CI
`[0.0034722,0.0225694]`，不属于预注册的 50% gate，不能升级结论。

### Protocol/schema deviation、结果 provenance 与审计 hash

- preregistration YAML 的 literal treatment list 只写到 `FC_tile0..3` 与
  `FF_tile0..3`；冻结的 2×2 recovery rule 未通过，因此实际 formal grid 选择 4×4，
  产生 `tile0..15`。这是已知的 schema/name mismatch；实际 raw-row validators、
  统计和 34-condition 结果按 4×4 条件运行，不因名称不一致而改写结果。

- legacy producer：PAI job `dlc9nkd8q7u4szm3`，run
  `stage27r-formal-idle-v9`，idle resource alias
  `idle-a800-stablevla-native5-8gpu`、resource id `quotaewyznuc7b9l`，
  `AcceptQuotaOverSold`/idle quota；producer source commit
  `fa05c2ef52e5cce16f62397540162724bfd4a6b9`，tree
  `6fdb28764d002def6d10e5a9c4f41918fe7713d1`，launcher SHA256
  `e89d7793a336356c6cd6e6021fd305699551587f695dcf23efc4a18b760212c2`。
- clean posthoc continuation/verifier：PAI job `dlc1x9np56zy7gpm`，run
  `stage27r-formal-idle-v11`，同一 idle quota/resource；source commit
  `7554bca313d796fd0b4cdf3abbc817c6ecc7e9fd`，tree
  `837e29d5547c67c8045522164ab72e8fbcc2d5a0`，launcher SHA256
  `c6698debdebc73a0cec7fce5865bd5656bd9833ac703a052e0b87a8fb9f5aada`。
- final independent verifier commit `6893116948025785d0da7860e7ece94ea5497707`、
  tree `5326b8529ff5bc0bdc99ceb6f526117d0fb10e52`，audit script SHA256
  `e3e8bbb601fe1db6f048883c7d693f0fe6917c6bcdaa7a48f06ddb9f4dbbbaec`。
- malformed `CONTINUATION_V11_TERMINAL.json` 保留为失败证据，原因是 trailing literal
  `\\n`，SHA256 `bbf16df01c734e21e37c90d4f77b9291ea94a8ae203bc2c42841adeacf584dde`；
  `CONTINUATION_V11_TERMINAL_VALID.json` 是正式绑定的 corrected terminal，SHA256
  `73443d0cbb8277b38f3fb9468f2f29d75a3e8684ac60fbc9b3e596ed0cccd5ef`。修复是
  evidence-only；未覆盖 scientific output。v9 在 6/6 raw shards 已完成后，derived
  audit 因 `KeyError: preregistration_sha256` 失败；该失败发生在审计派生阶段，没有
  覆盖 scientific raw。随后 v11 continuation 和 final verifier 修复审计链并完成
  独立重算。

## privileged oracle evidence

- 每个 state 的 FC/FF 是 16 tile 中按重复 physical utility 取最大者；tile 不是
  independent statistical unit，也不是可部署 selector。因而 visual 正向、joint
  负向都应读作 privileged physical-oracle contrast，而非 learned routing result。
- formal Stack/Peg oracle 的 prefix fidelity、raw outcome recomputation 和 compute
  accounting 均通过；但由于 positive screen gate 失败，结果只能回答“在这两个
  diagnostic task bank 上，这个 oracle contrast 观察到什么”，不能回答核心机制在
  screened positive tasks 上已验证。
- phase/component decomposition（balanced）显示 visual 的提升主要来自 progress、
  recoverability，外加少量 Stack success transitions；未观察到 collision reduction。
  Peg action 在 in-hand phase 有局部正 pocket，但 task-level aggregate CI 跨零；
  Stack action 的损失主要集中在 contact/placement near completion。
- joint 是相对较优单轴的负 synergy：即使某些 phase 的 FF 仍高于 CC，FF 仍可能
  低于 FC 或 CF；这不等价于声称所有 state 都有 `FF<CC`。

## bounded inference

### 对计划十问的明确回答

1. **shared prefix 是否真的可信？** 在 fresh-reset replay 的已审计阈值内可信：
   broadcast action max abs=0、categorical agreement=100%，两组 fidelity pass rate
   均为 1.0；但这只支持可复现 prefix，不改变 serial fresh-reset、非同时 10-arm
   stepping 的限制。
2. **visual resolution 是否有稳定 marginal value？** 在两个 diagnostic tasks、三
   seeds 的 balanced 结果与三套 utility weights 的 aggregate 上为正；这是本轮最
   稳定的信号，但因 screen fail 只能称 diagnostic positive。
3. **action replanning resolution 是否有稳定 marginal value？** 没有：Peg task-level
   aggregate 跨零，Stack 在三 weights 均负；不能把 Peg 的局部 in-hand pocket 当作
   稳定 action gain。
4. **是否存在真正 joint synergy？** 没有。`FF-max(FC,CF)` 在两任务和三 weights
   均负；这是相对较优单轴的 negative synergy，不是简单的 FF-CC 结论。
5. **adaptive 的收益来自 state selection、axis selection，还是两者？** 本轮不能
   做正式归因：25/50% budget 不可行，75% joint oracle 仅 exploratory；因此不能
   声称 state selection 或 axis selection 在预注册 50% 下带来可部署收益。
6. **equal-cost 下是否优于 strongest single-axis？** 50% gate 不可评估（all-coarse
   已超预算、refined states=0）；75% 有 +1.2153pp exploratory success gain，不能
   作为 preregistered success claim。
7. **是否接近 all-fine，同时显著降低 compute？** 没有可裁决证据：all-fine 成本
   是 `1.251202e14`，all-coarse 已为 all-fine 的 0.738795，25/50% 不可行；未通过
   “50% compute 下接近 all-fine”的门。
8. **负对照是否会错误 refinement？** 无法回答正式问题：PushCube 52.67% 未过
   70% 门，且 fallback 未实现，所以没有健康负控制 formal result。
9. **结果是否跨任务、跨 model seed、跨 utility weights？** 诊断 oracle 的 visual
   正向和 joint 负向跨两个任务、三个 seeds、三套 weights；Peg action 仍不稳定、
   Stack action 为负。由于 screen fail，这不是核心机制的跨任务 confirmatory claim。
10. **当前 idea 应保留 full joint、shared router、visual-only，还是停止？** 按预注册
    final-status precedence 停止本轮 core-mechanism 主线，固定为
    `NO_GO_CORE_MECHANISM`；不创建或执行 Stage-2.8/learned router。

上述回答是 bounded inference，不把 privileged tile oracle、诊断任务或 75% 探索性
结果包装成 deployable causal mechanism。code-first audit 只做机制反向审计，未生成
新 idea，也未修改 raw evidence、threshold 或 result vector。

## not tested

- 这是明确的已知实现偏差：prereg YAML 期望四模式各 `0.25`，但实现中的
  `mode_probabilities` 未被消费；training 只随机 coarse/fine visual 与 tile，且
  `tile_grid=2`，没有 action-mode dropout。formal run 是 shared weights 上的
  runtime oracle treatments，不能声称执行了四路 training resolution-mode dropout。
- raw oracle rows 含 `source_episode`、`bank_id`、`model_seed`、`task`、`phase` 等字段，
  但未逐行嵌入 `branch_step`、`prefix_actions`、checkpoint hash 或 state-bank hash；
  这些通过 `bank_id`/`source_episode` 与 state-bank/lineage sidecar 关联。state bank
  本身 96/96 保存 `branch_step`、`prefix_actions`、`prefix_sha256`；因此不能把
  sidecar 关联误写成 raw row 内嵌绑定。
- raw rows 未记录 action vectors，不能声称存在已量化的 action discontinuity；代码只
  确认 coarse cache/fine re-query schedule。component pattern 是 bounded/descriptive
  evidence，不是 action-level mediation proof。
- 没有健康负对照 formal 结果；没有 OOD、closed-loop learned router、Stage-2.8、
  π0.5 或真机实验。没有创建 `DRAFT_PREREGISTRATION.md`，因为该文件只允许在
  GO/REVISE 状态下创建。
- 本报告不改变 Stage-1、Stage-2、Stage-2.5、Stage-2.6 的任何字节或既有结论；
  predecessor immutability audit 已通过。
