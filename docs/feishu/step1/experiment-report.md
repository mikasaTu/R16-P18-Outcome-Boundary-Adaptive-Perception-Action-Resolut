---
feishu_title: "实验报告"
feishu_url: "https://icnbwz7kd1ui.feishu.cn/wiki/MCkMwHrYpiQZa2kh3H2c7SWhnnd"
feishu_wiki_token: "MCkMwHrYpiQZa2kh3H2c7SWhnnd"
feishu_object_token: "B2ledyTWSo1XjzxCn0bcZAjGntg"
feishu_revision: 4
---

<title>实验报告</title>

# R16-P18 LIBERO Stage-1 初步验证（Baseline Gate）

**最终决策：NO_GO_BASELINE_GATE**

报告日期：2026-08-12（UTC）。协议：R16-P18-LIBERO-STAGE1-PILOT-V1。正式 PAI 作业 **dlcnouq6igkhfyub** 已以 Succeeded 终态完成。

本次结果只回答“当前 LIBERO 任务与 Small-BC 基线是否满足进入 R16-P18 自适应实验的必要条件”。它不是 R16-P18 联合分辨率方法的结果，不能据此称该 idea 已验证、已接受或已被否证。

## 1. 结论摘要

- 9 个 BoundaryBC-S 模型全部完成：3 个任务 × 3 个训练 seed × 3000 optimizer steps。
- 450 个闭环回合全部完成：每个任务、每个模型 seed 均为 50 回合；无缺失或冲突重复。
- Push 任务聚合成功率 80.0%，在预注册区间 40%–90% 内，PASS。
- Bottle 任务聚合成功率 98.0%，超过预注册上限 80%，FAIL；该任务对当前基线过易，缺少可辨识的提升空间。
- Bowl 负控聚合成功率 88.7%，达到预注册下限 80%，PASS。
- 由于三个任务必须同时通过，baseline gate 失败。依停止规则，没有实现 effect predictor、96-state bank 或任何 adaptive arm。
- 另有数据协议偏差：官方 LIBERO 对每个精确任务仅提供 50 条成功演示，且没有原始 episode seed 字段；因此即使 baseline gate 通过，本 pilot 也不允许返回 Stage-1 GO。

## 2. 对 idea 的理解

核心假设是：在视觉 token 数、动作候选数、policy call、effect-model call、动作机会与 timeout 都严格相同的预算下，围绕预测的任务结果边界，同时分配视觉分辨率与动作分辨率，应优于 uniform、random adaptive、visual-only、action-only 以及独立组合的 visual+action。

这里的“结果边界”不是普通图像边缘，而是局部动作结果发生离散变化或高敏感变化的区域，例如接触建立、碰撞、任务进度跃迁、短时成功和可恢复性变化。方法的必要机制是让视觉证据与动作候选共同聚焦这些边界，同时在测试时禁止使用 simulator privileged state。

该实验只是必要条件 falsifier，不是最终策略实验，也不是 world model、Diffusion Policy、DINO-WM 或 π0.5 实验。旧的“先完整读取高分辨率 outcome surface 再选择预算”的路径会泄漏隐藏标签，因此不允许使用。

## 3. LIBERO 适配与预注册任务

- **push_the_plate_to_the_front_of_the_stove**：boundary push；基线合格区间 40%–90%。
- **put_the_wine_bottle_on_the_rack**：constrained placement；基线合格区间 25%–80%。
- **put_the_bowl_on_the_plate**：smooth negative control；基线合格区间 80%–100%。

Gate 使用三个模型 seed 与每 seed 50 个固定 init state 的总体点估计；95% paired bootstrap CI 用于不确定性报告，不替代预注册点估计判据。三个任务必须全部通过，才允许实现自适应 Stage-1。

## 4. 数据协议与偏差

原计划要求每任务 200 条成功演示并按 episode seed 做 160/20/20 划分。官方 LIBERO 对上述每个精确任务实际只有 50 条成功演示，HDF5 中保留 init state，但不含原始 episode seed。

本 pilot 没有复制演示，也没有把数据增强轨迹伪装成独立 episode。每条演示以 init state 的 SHA256 作为稳定身份，按哈希字典序做 40/5/5 的 train/validation/test 划分。这是预先记录的重大协议偏差，也是本 pilot 永远不能返回 Stage-1 GO 的独立原因。

在观察任何训练结果之前，正式 runtime 从开发机 CUDA 13 环境改锁为已在 PAI carrier 上验证的 CUDA 12.4 环境；模型、数据、seed、预算和阈值均未改变。

## 5. 冻结实现与可复现性

### 5.1 BoundaryBC-S 基线

- 输入：128×128 agent-view RGB 与 15 维 proprio。
- 视觉骨干：random-init ResNet18 layer3，输出 8×8、128 维 feature map，共 64 个 microtoken；基线固定均匀使用 16 个视觉 token，不含 adaptive selector。
- 策略：2-layer、4-head、hidden size 128；action dim 7，action horizon 8，execute horizon 4；LIBERO OSC_POSE 控制。
- 训练：AdamW，batch size 64，BF16，3000 steps，seed 16018/16019/16020；每 250 steps 验证并写完整可恢复 checkpoint，只保留最新 3 个完整 checkpoint。
- 完整 checkpoint 包含 model、optimizer、scheduler、RNG、batch generator 与 global step；partial checkpoint 被忽略。

### 5.2 Runtime 与来源锁定

- 项目 source commit：20d65a9ab7ff42662496c093e14531904cf1fe31；tree：b87468210362a81a5d3754630f1ff804f79577e4。
- LIBERO commit：2319dbd914673f6ef34a00584536212b10e10395。
- 配置 SHA256：d4344c056e1c7682cec9deabd0c82888bd6ec29d6d473b9a1870a262aeaa64bf。
- 资产：585 files，422,313,213 bytes，tree manifest SHA256 39b115a50b590dfa061d36439702eddf99c178a0fcfeab90c5c60afcc6c05a35。
- 正式环境：Python 3.11.11，PyTorch 2.5.1+cu124，torchvision 0.20.1+cu124，CUDA 12.4，cuDNN 90100，MuJoCo 3.6.0，robosuite 1.4.0，NumPy 1.26.4。
- 演示文件 SHA256：Push 36b4e1bced49d2f4ff6b2fce6b1596a63978e14199e2513cd0df71e127bf47a6；Bottle f9092aa70734fc4083e97fc58c3ba25f87c614d18326182ddc7a455f0ab4da2e；Bowl e69528b0cf10dfc59b20698e12ec2affc03f3887309034d3eb74cac3ec929406。

### 5.3 开发机 smoke 与状态恢复

- 在开发机仅做 1×A800、batch 1 smoke；三个任务均完成 EGL-before-CUDA、policy step 与 forward/backward。
- 新鲜环境中的物理 state restore 与短时 replay 三任务误差为 0；GPU raster 对 Push/Bowl 完全一致，Bottle 最大 1 LSB、MAE 0.000061，低于预注册容差。
- proprio 对齐最大误差不超过 0.001827。
- smoke 明确记录 adaptive_components_implemented=false；未创建 PAI GPU probe。

### 5.4 PAI 执行合同

- 正式 JobId：dlcnouq6igkhfyub；run-id：r16-p18-libero-stage1-bc-gate-20260812-003；终态 Succeeded；PAI duration 1580 秒（26 分 20 秒）。
- 资源：1 worker、2×NVIDIA A800、12 CPU、200 GiB memory、200 GiB shared memory；两条独立单卡模型队列，不使用 data parallel，单模型 batch 仍为 64。
- AIMaster 关闭，PAI 自动容错关闭，maximum platform restarts=0；终态只观察到一个成功 PodUID。
- 五个新 CPFS 挂载、resource quota、UserCommand、secret 名称、W&B 实体与 JobId 均由 controller exact readback 验证。
- W&B 写入门禁在 CreateJob 前验证 entity=chen_jian-cj-workspace、membership role=admin、identity inference=false。

## 6. Baseline Gate 结果

### 6.1 Push：PASS

- seed 16018：48/50，96%。
- seed 16019：25/50，50%。
- seed 16020：47/50，94%。
- 聚合：120/150，80.0%；95% paired bootstrap CI：[52.0%, 98.0%]；要求：[40%, 90%]。

该任务点估计合格，但 seed 方差极大，是后续新协议需要单独审计的稳定性风险。

### 6.2 Bottle：FAIL

- seed 16018：48/50，96%。
- seed 16019：50/50，100%。
- seed 16020：49/50，98%。
- 聚合：147/150，98.0%；95% paired bootstrap CI：[94.0%, 100.0%]；要求：[25%, 80%]。

失败原因是任务过易，而不是基线失效。只有约 2 个百分点的失败余量，无法为联合 outcome-boundary adaptation 提供预注册要求的非平凡提升空间。

### 6.3 Bowl 负控：PASS

- seed 16018：47/50，94%。
- seed 16019：40/50，80%。
- seed 16020：46/50，92%。
- 聚合：133/150，88.7%；95% paired bootstrap CI：[78.0%, 97.3%]；要求：[80%, 100%]。

按预注册规则，gate 依据聚合点估计，因此该任务 PASS；CI 下界低于 80% 作为不确定性信息保留，不进行事后改判。

### 6.4 独立复算

独立读取 9 个 JSONL，验证每个文件恰好包含 episode_id 0–49，无缺失和冲突重复。重新执行 10,000 次 paired seed/episode bootstrap 后，三组 CI 与正式 baseline_gate.json 完全一致。全任务合计 400/450，88.9%；该总数不是 gate 判据，仅作完整性摘要。

## 7. 调用与时延核算

### 7.1 Push

- 150 episodes；6967 policy calls，平均 46.447 calls/episode。
- 27,673 executed environment steps，平均 184.49 steps/episode。
- 纯 inference 合计 24.480 秒，平均 3.514 ms/policy call。
- episode wall time 平均 3.935 秒，中位数 3.261 秒，P95 6.547 秒。

### 7.2 Bottle

- 150 episodes；6069 policy calls，平均 40.460 calls/episode。
- 24,023 executed environment steps，平均 160.15 steps/episode。
- 纯 inference 合计 21.655 秒，平均 3.568 ms/policy call。
- episode wall time 平均 3.373 秒，中位数 3.254 秒，P95 4.254 秒。

### 7.3 Bowl

- 150 episodes；4613 policy calls，平均 30.753 calls/episode。
- 18,228 executed environment steps，平均 121.52 steps/episode。
- 纯 inference 合计 16.789 秒，平均 3.639 ms/policy call。
- episode wall time 平均 2.667 秒，中位数 2.191 秒，P95 6.252 秒。

### 7.4 总计

- 450 episodes；17,649 policy calls；69,924 executed steps。
- 纯 inference 合计 62.923 秒，平均 3.565 ms/call。
- episode wall time 求和 1496.265 秒，平均 3.325 秒，P95 6.436 秒。该求和跨两张 GPU 的并行队列，不等同于 PAI 作业墙钟时间。
- Baseline 阶段没有 effect-model call 或 action-atlas call；这些组件依 gate 规则没有实现。

## 8. GO / NO-GO 与停止范围

**正式决策：NO_GO_BASELINE_GATE。**

1. Push 满足非平凡区间。
2. Bottle 为 98.0%，超过 80% 上限，baseline gate 失败。
3. Bowl 负控满足下限。
4. 由于 require_all_tasks=true，自适应实现授权为 false。

因此以下内容均未执行，也没有结果：96-state bank、adaptive visual selector、49-candidate hidden fine atlas、13-candidate budgeted atlas、local effect predictor、uniform/random/visual-only/action-only/independent/joint/shuffled/full-fine arms、state-bank alias rate、best-action recall、outcome regret、boundary F1/AUROC、coupling gain、ID/geometry OOD/visual-nuisance OOD。

没有启动 Diffusion Policy、DINO-WM 或 π0.5。当前报告不能声称 R16-P18 idea 已验证、已接受或已被拒绝；它只说明当前 LIBERO 三任务配置不具备继续检验该 idea 的 baseline 必要条件。

## 9. 执行异常与审计轨迹

- run-id 后缀 001 在 CreateJob 前被本地权限合同拒绝，没有创建 PAI 作业、没有 GPU 消耗。
- 首个正式 JobId dlc1eloj62mdzw2y 在任何训练步之前失败。根因是容器初始 cwd=/root，降权后 GNU find 无法恢复 cwd；这不是模型或指标结果。
- 修复仅在冻结 launcher 中先切换到锁定项目目录；数据、模型、seed、预算和阈值未变。修复后的 JobId dlcnouq6igkhfyub 成功完成。
- 只有在替代作业写出真实 global_step=1、loss=0.3117666 的持久证据后，才用官方 OpenAPI 两阶段协议删除旧 Failed 服务行；GetJob 与 ListJobs 均证明其已 absent，CPFS checkpoint、日志和 registry 诊断全部保留。

## 10. 核心证据与哈希

- baseline_gate.json SHA256：20d64ae8cec3e5405f79b678710189b0799e3ed4b87614e7e32ab6fd7295b5fd。
- run_manifest.json SHA256：d3ec8cb14033472fe89d3319ee108f82225bb272de1c1f7db9226d690d64d0cc。
- monitor-result.json SHA256：2237738c102184521467d34fae470a474f5ff07326ae47425a4f5cda899f12e0。
- artifact-evidence.json SHA256：c538d8231071fe5136033696791e6e75db5604dbfd6872ca3c4894ef45eb6b50。
- 旧作业清理 evidence SHA256：af7e9c47591cad08a1da6fd6c0902fde24301a5a3550eab230a2da700a217c14。
- 所有上述证据文件均为 2254:2254、mode 0600。

## 11. 后续建议（本次未执行）

1. 不要事后放宽 Bottle 的 80% 上限。应新建独立预注册，替换为更难且有足够失败余量的 constrained-placement 任务。
2. 用官方 pipeline 生成每任务 200 条互不重复、带 episode seed 的成功演示，或明确降低数据要求并重新预注册；不能把复制或增强样本当作独立 episode。
3. 在新协议中先复查 Push 的跨 seed 稳定性，再重新执行 baseline gate。
4. 只有新 gate 全部通过，才实现 state bank、effect predictor 与 matched adaptive arms，并继续使用原定 16 visual tokens、13 action candidates 和严格相等 call budgets。

**一句话结论：**当前 LIBERO pilot 的 Small-BC 基线可运行且证据完整，但 Bottle 任务 98% 造成 ceiling，故预注册结论为 NO_GO_BASELINE_GATE；Stage-1 联合 outcome-boundary 假设尚未被检验。
