Inventory convention: this directory follows the existing repository convention `experiments/r16p18_stage3/`; all Stage-2.7R and earlier paths below are read-only historical inputs.

# S1 Inventory

Protocol: `R16-P18-STAGE3-S1-BUDGET-FEASIBILITY-V1`  
Repository HEAD at inventory time: `ec3fbbd054f9218332122cc477912d3ddf0ad93b`  
Repository tree at inventory time: `25614ce430df07c9e3ac7f2e5fb8263c055c06ce`

## S1.0 substrate audit

### Historical 0.738795 ratio

The immutable Stage-2.7R record is `all_coarse_cost / all_fine_cost = 92438200000000 / 125120200000000 = 0.738795174560143`. The historical wall-clock ratio is `0.75025979`; it is not a fresh S1 measurement.

- Per-row FLOP accounting formula: `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut-stage3-s1/experiments/maniskill_stage27r_core_mechanism_reset_v1/scripts/stage27r_runtime.py:188` (`global_encoder_calls * 1.8e9 + fine_encoder_calls * 1.8e9 + policy_forward_calls * 0.7e9`).
- Budget numerator/denominator sums: `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut-stage3-s1/experiments/maniskill_stage27r_core_mechanism_reset_v1/scripts/analyze_stage27r.py:111-114` (`coarse = sum(CC.cost)`, `full = sum(FF.cost)`, `budget = alpha * full`).
- Refinement acceptance: `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut-stage3-s1/experiments/maniskill_stage27r_core_mechanism_reset_v1/scripts/analyze_stage27r.py:116-127`; a candidate is counted only when `du > 0` and `cost + dc <= budget`.
- Historical raw accounting: `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut-stage3-s1/experiments/maniskill_stage27r_core_mechanism_reset_v1/artifacts/formal-run/statistics.json:1` fields `aggregated_state_treatments[].accounting.estimated_flops`, `.gpu_latency_ms`, and `.cost`; this is old rollout accounting, not a new forward profile.
- Historical ratio sidecar: `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut-stage3-s1/experiments/maniskill_stage27r_core_mechanism_reset_v1/audits/mechanism_reverse_engineering_summary.json:230-237` fields `compute_budget.all_coarse_cost`, `all_fine_cost`, `coarse_to_fine_ratio`, and budget summaries.
- Archived pure-axis fallback (diagnostic only): code-derived fixed-window FLOP proxies are visual coarse/fine `8.6e9/15.8e9` and action coarse/fine `8.6e9/34.4e9`; these are derived from the native 8-output, interval-4-versus-1 schedule and are not fresh wall-clock measurements.

### Resolution definitions found in code

- Visual coarse path: `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut-stage3-s1/experiments/maniskill_stage27r_core_mechanism_reset_v1/scripts/multires_policy.py:91-119`; global image path is resized to 112x112, while fine adds a crop branch from the original tensor. Crop grid validation accepts 2 or 4 at `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut-stage3-s1/experiments/maniskill_stage27r_core_mechanism_reset_v1/scripts/multires_policy.py:30-39`. S1 treats these as model-native semantics and requires a fresh forward profile to prove they run on the selected checkpoint.
- Visual accounting: `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut-stage3-s1/experiments/maniskill_stage27r_core_mechanism_reset_v1/scripts/multires_policy.py:177-186`; global/fine encoder calls and tokens are reported separately.
- Action coarse/fine: `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut-stage3-s1/experiments/maniskill_stage27r_core_mechanism_reset_v1/scripts/stage27r_runtime.py:161-176`; fine queries every action opportunity, coarse reuses the cached 8-output chunk and queries at interval 4. The independent axis is therefore query interval 4 versus 1, not an action candidate-token grid.
- Fixed output chunk: `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut-stage3-s1/experiments/maniskill_stage27r_core_mechanism_reset_v1/scripts/train_multires_act.py:24-33` binds `num_queries=8`; S1 does not alter the weight or output shape.

### Budget denominator and screen

Budget alpha uses the all-fine cost sum as denominator at `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut-stage3-s1/experiments/maniskill_stage27r_core_mechanism_reset_v1/scripts/analyze_stage27r.py:111-114`. The old refine count is set by the `cost + dc > budget` guard at `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P18-Outcome-Boundary-Adaptive-Perception-Action-Resolut-stage3-s1/experiments/maniskill_stage27r_core_mechanism_reset_v1/scripts/analyze_stage27r.py:123-126`.

The frozen Stage-2.7R screen task list is StackCube-v1, PegInsertionSide-v1, PlugCharger-v1, PullCubeTool-v1, PushT-v1, and PushCube-v1; task/control metadata originates at `stage27r_runtime.py:20-27`. Raw selected screen fields are listed below from the immutable `TASK_SELECTION.json`.

| task/seed | selected step | 40-ep success_hold5 | 100-ep CC success_hold5 | 100-ep FF success_hold5 | success_at_end | post_success_loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PegInsertionSide-v1/seed_16018 | 25000 | 0.025 | 0.0 | 0.0 | 0.025 | 0.0 |
| PegInsertionSide-v1/seed_16019 | 20000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PegInsertionSide-v1/seed_16020 | 20000 | 0.0 | 0.01 | 0.0 | 0.0 | 0.0 |
| PlugCharger-v1/seed_16018 | 25000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PlugCharger-v1/seed_16019 | 25000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PlugCharger-v1/seed_16020 | 25000 | 0.0 | 0.01 | 0.0 | 0.0 | 0.0 |
| PullCubeTool-v1/seed_16018 | 25000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PullCubeTool-v1/seed_16019 | 25000 | 0.0 | 0.0 | 0.01 | 0.0 | 0.0 |
| PullCubeTool-v1/seed_16020 | 25000 | 0.0 | 0.0 | 0.01 | 0.0 | 0.0 |
| PushCube-v1/seed_16018 | 30000 | 0.525 | 0.85 | 0.55 | 0.525 | 0.0 |
| PushCube-v1/seed_16019 | 25000 | 0.375 | 0.8 | 0.43 | 0.375 | 0.0 |
| PushCube-v1/seed_16020 | 25000 | 0.525 | 0.78 | 0.6 | 0.525 | 0.0 |
| PushT-v1/seed_16018 | 20000 | 0.075 | 0.09 | 0.06 | 0.075 | 0.1 |
| PushT-v1/seed_16019 | 30000 | 0.025 | 0.04 | 0.03 | 0.025 | 0.1 |
| PushT-v1/seed_16020 | 20000 | 0.025 | 0.05 | 0.01 | 0.025 | 0.075 |
| StackCube-v1/seed_16018 | 25000 | 0.0 | 0.1 | 0.0 | 0.0 | 0.0 |
| StackCube-v1/seed_16019 | 30000 | 0.025 | 0.15 | 0.02 | 0.025 | 0.0 |
| StackCube-v1/seed_16020 | 15000 | 0.075 | 0.16 | 0.04 | 0.075 | 0.0 |

## Source file hashes

| path | exists | bytes | SHA256 | audited fields |
| --- | --- | ---: | --- | --- |
| `experiments/maniskill_stage27r_core_mechanism_reset_v1/scripts/stage27r_runtime.py` | True | 11793 | `900c448fe0bae99e26fdccad25d4ca2a1ded63173a84cc2b4ecb40eddcab5595` | accounting.estimated_flops; accounting.gpu_latency_ms; query schedule |
| `experiments/maniskill_stage27r_core_mechanism_reset_v1/scripts/analyze_stage27r.py` | True | 12407 | `8adc7cb3d909e5fce3e4d561daa0457e640cb42fd1c9dd315a2b24f67d3ecb86` | coarse sum; full sum; budget; dc; refine acceptance |
| `experiments/maniskill_stage27r_core_mechanism_reset_v1/scripts/multires_policy.py` | True | 9038 | `ab34c3e0c731c6523e4431bf2c6af27c3d4496c843b5bd7f5a2a4734243be75f` | visual_mode; action query accounting; num_queries adapter |
| `experiments/maniskill_stage27r_core_mechanism_reset_v1/artifacts/formal-run/statistics.json` | True | 29655763 | `4cc71eff2251f008bfb9910e8a4065459d998a9a635d8dc5d075b40896ce87bd` | aggregated_state_treatments[].accounting; cost; gpu_latency_ms |
| `experiments/maniskill_stage27r_core_mechanism_reset_v1/audits/mechanism_reverse_engineering_summary.json` | True | 20071 | `1193b7435bea426f195a50447739922ecfb2297f9ff54808b324a1052a82babb` | compute_budget.all_coarse_cost; compute_budget.all_fine_cost; recorded ratio |
| `experiments/maniskill_stage27r_core_mechanism_reset_v1/artifacts/formal-run/screen/TASK_SELECTION.json` | True | 82293 | `c9807d8786f75c334766636ffe892e81e4cb3d3f0da7344ac0499d3c044f0fbf` | groups[*].screened; groups[*].selected; seed; task |

## Fresh-profile boundary

No fresh S1 profiling result is asserted by this inventory. `prepare_s1_observation.py` reads one observation through the frozen replay-HDF5 preprocessing path, and `profile_s1_costs.py` reconstructs the frozen EMA model using dummy tensor spaces. Both fail closed when an input is absent; neither constructs, resets, or steps an environment. The archived Stage-2.7R accounting fallback may populate diagnostic FLOP feasibility rows, but it cannot pass G1/S1.1 because fresh wall-clock and FLOP measurements remain unproven.

`S1_COST_REPRO.json`, `S1_COST_CURVE.json`, `S1_FEASIBILITY.json`, plots, and `S1_DECISION.md` are generated only by the calculator after a supplied profile or explicitly selected archived fallback. They must not be hand-filled.
