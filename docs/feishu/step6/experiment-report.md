---
feishu_title: "实验报告"
feishu_url: "https://icnbwz7kd1ui.feishu.cn/wiki/Rz6twhP00iucgmkmWTFcm8U6n6g"
feishu_wiki_token: "Rz6twhP00iucgmkmWTFcm8U6n6g"
feishu_object_token: "WTAadi9DgosSQjxH73rcXmT8npc"
feishu_revision: 6
---

<title>实验报告</title>

<callout emoji="⛔">
**Stage-2.7R 最终状态：`NO_GO_CORE_MECHANISM`**

本结论裁决冻结的 policy/task/screen/protocol，不是对整个 R16-P18 方向的
普遍否证。正任务 screen 全失败，因此 Stack/Peg oracle 结果为 diagnostic only。
</callout>

# 实验身份与证据范围

- Protocol：`R16-P18-MS6-STAGE27R-CORE-MECHANISM-RESET-V1`
- Branch：`stage27r-core-mechanism-reset-v1`
- 唯一可变目录：`experiments/maniskill_stage27r_core_mechanism_reset_v1/`
- 六 shard × 16,320 rows = **97,920 rows**；每任务 96 states、34 conditions、5 repeats；
  model seeds `16018,16019,16020`；utility weights 为 balanced、success-dominant、
  progress-dominant。
- `StackCube-v1` anchor 及所有已执行候选 screen 全失败；`StackCubeHard` fallback 未实现，
  `selected_positive=null`。
  `PushCube-v1` 仅 `52.6667%`，未过 70% 负控门；无正式负控结果。
- 结果、统计、审计和 manifest 位于 `experiments/.../artifacts/formal-run/`；本地 Git archive
  已准备并校验，待本次 commit 发布，raw shards 仍在 CPFS。
  RESULT_VECTOR SHA256=`bbdd69a67f1cd5d3f760196524549ae7b442ac21afc5f95963194b4eb3b5a095`；
  statistics SHA256=`4cc71eff2251f008bfb9910e8a4065459d998a9a635d8dc5d075b40896ce87bd`。

## confirmed code semantics

- 同一 policy 支持 CC/FC/CF/FF；fine 保留 112×112 global branch，并追加原始 128×128
  RGB 的 crop branch。正式 run 为 4×4、16 tile candidates；tile 取最大 repeated
  physical utility，是 privileged offline label，非可部署 selector。
- 所有 arm 共享 8-step treatment 和最多 20-step common native/fine continuation。
  coarse action 每 4 步 query/reuse 4-step chunk；fine action 每步 re-query，执行
  1-step receding horizon。success 是连续五步 predicate（`success_hold5`）。
- fresh-reset prefix smoke 与 oracle fidelity pass；但正式 conditions 是 serial fresh-reset
  replay，不是同时 10-condition stepping。raw rows 没有逐步 action vectors。

## observed paired evidence

visual `FC-CC` 在 Peg 与 Stack 两个 diagnostic tasks、三套 weights 均为正；Peg
action `CF-CC` 的 task-level CI 跨零，Stack action 三套 weights 均为负；joint
`FF-max(FC,CF)` 在两任务、三套 weights 均为负。balanced seed means 为：visual
Peg `[0.5506,0.6224,0.6940]`、Stack `[0.0874,2.3401,1.4529]`；action Peg
`[-1.6832,3.3454,-0.2964]`、Stack `[-5.7781,-3.2466,-4.0971]`；joint Peg
`[-3.7267,-1.3525,-0.3773]`、Stack `[-5.6453,-5.5496,-4.5338]`。
balanced joint positive-state fraction（seed 顺序 16018/16019/16020）为 Peg
`[0.4167,0.3750,0.4063]`、Stack `[0.2917,0.2500,0.4479]`；三 seed 均过 10% fraction
门，但负 state 的损失更大，task-level mean/CI 仍为负，不能反转 joint gate。

真实 compute：all-coarse=`9.24382e13` FLOPs，all-fine=`1.251202e14` FLOPs，
coarse/full=`0.7387951746`。25%/50% budget 不可行；50% 下所有 adaptive allocation
arms（不含 all-fine baseline）refine 0 states、joint
vs fixed success difference=`0`；75% 仅 exploratory（joint +1.2153pp，CI
`[0.3472,2.2569]`），不能作为 50% gate。

## privileged oracle evidence

Stack/Peg oracle 的 raw outcome、paired statistics、compute accounting 和 prefix
fidelity 均通过，但 positive screen failure 将其 claim tier 限定为 diagnostic。
visual signal 主要表现为 progress/recoverability 和少量 Stack success transitions；
Peg action 只在局部 in-hand pocket 为正，Stack action 损失主要在 contact/placement。
joint 的负值是相对更优单轴的 negative synergy，不等价于所有 state 都 `FF<CC`。

## bounded inference（含计划十问）

1. Prefix：在已审计阈值内可信且可复现，但不是 simultaneous stepping 证据。
2. Visual：有稳定的诊断性正 marginal signal，不能升级为 screened-task confirmatory claim。
3. Action：不稳定；Peg null、Stack negative。
4. Joint：无真正 joint synergy；相对 better single axis 为负。
5. Adaptive 收益：25/50% budget 不可行，不能归因是 state selection、axis selection 或两者。
6. Equal-cost：50% 无有效 comparison；75% 仅 exploratory，不能通过预注册 gate。
7. All-fine：未证明以显著更低 compute 接近 all-fine。
8. Negative control：PushCube 未过门，fallback 未形成健康正式负控，不能回答。
9. 跨任务/seed/weights：diagnostic visual/joint 方向跨两任务、三 seed、三 weights；
   这不是核心机制 confirmatory 通过。
10. 处理建议：按 precedence 停止本轮 core-mechanism 主线，固定 `NO_GO_CORE_MECHANISM`；
    不执行 Stage-2.8。

## not tested

- 已知实现偏差：prereg YAML 期望四模式各 `.25`，但 `mode_probabilities` 未被消费；
  training 仅随机 coarse/fine visual 与 tile（`tile_grid=2`），没有 action-mode dropout。
  formal 只是 shared weights 上的 runtime oracle treatments，不能声称执行了四路
  training dropout。
- raw oracle rows 含 `source_episode`、`bank_id`、`model_seed`、`task`、`phase` 等字段，
  但未逐行嵌入 `branch_step`、`prefix_actions`、checkpoint hash 或 state-bank hash；
  这些通过 `bank_id`/`source_episode` 与 state-bank/lineage sidecar 关联。state bank
  本身 96/96 保存 `branch_step`、`prefix_actions`、`prefix_sha256`；不能将 sidecar
  关联误写成 raw row 内嵌绑定。
- 未记录 action vectors，不能声称量化 action discontinuity；没有健康负控 formal result。
- 没有 OOD、learned router、Stage-2.8、pi0.5、真机实验，也没有 Stage-2.8 draft。

### Raw archive / provenance（observed paired evidence）

- 六 raw shard 共 426,794,614 bytes，未入 GitHub-safe archive，保留 CPFS；本地归档为
  37 files / 36,268,200 bytes，`SHA256SUMS` 38/38，manifest 绑定 path/rows/bytes/hash。
  shard hashes：Peg16018 `798490f0514865cf63383cfd7238415150f0ffe9e618fcdb7e7fcced1a79e4c7`；
  Peg16019 `18d6c4d8c56d86a93334721ab5d2aa6ff430def9b82898b851ab8a1415287793`；Peg16020
  `abed063078025d97dcc52f69ddb713c07bbb33ae6bea7e64eb6b3a37211a8186`；Stack16018
  `9be2072119703fe1e8b2e1ecc16f3488cc49c75b10f606e44b1d93fc5c0581c2`；Stack16019
  `86eb7d00e618486783f3ebb6c77ec6845ded1640172dace4e4f256b01f2a0981`；Stack16020
  `e98c5e4a5d772d7f7375116b4d9e225fd987abbe333ec98e91d13c19222fba1f`。
- Prereg literal treatment names only list tile0..3；2x2 recovery gate failed, so formal
  4x4 produced tile0..15. This is a known schema/name mismatch, not a row-validator change。
- v9 derived audit 在 6/6 raw shards 后因 `KeyError: preregistration_sha256` 失败，未覆盖
  scientific raw；v11/final verifier 随后修复并独立重算。

### PAI lineage / audit（observed paired evidence）

- v9 producer：job `dlc9nkd8q7u4szm3`，run `stage27r-formal-idle-v9`，idle resource
  `idle-a800-stablevla-native5-8gpu` / `quotaewyznuc7b9l`，source commit
  `fa05c2ef52e5cce16f62397540162724bfd4a6b9`。
- v11 verifier：job `dlc1x9np56zy7gpm`，run `stage27r-formal-idle-v11`，同一 idle
  resource，source commit `7554bca313d796fd0b4cdf3abbc817c6ecc7e9fd`。
- final independent verifier：commit `6893116948025785d0da7860e7ece94ea5497707`；
  independent audit all checks PASS。malformed terminal 保留为失败证据，corrected
  terminal 才是正式绑定；未覆盖 raw evidence 或旧实验目录。
