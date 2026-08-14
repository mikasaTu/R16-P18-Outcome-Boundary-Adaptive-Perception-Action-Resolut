# CUDA atlas backend implementation repair

Protocol: `R16-P18-MS4-STAGE25-REPAIR-ORACLE-V1`

## Detection and failure boundary

After the serial restoration repair and before v18 reached action-atlas work, a
static downstream audit found that both `run_action_boundary_probe.py` and
`run_visual_resolution_probe.py` constructed their 75-way candidate rollout
environment with `sim_backend="physx_cpu"`.  ManiSkill 3.0.1 forbids multiple
environments on the CPU backend.  The frozen preregistration already specifies
`environment.formal_sim_backend: physx_cuda`; only the separate three-repeat
state-restoration audit is assigned `physx_cpu`.

Run v18 (`dlct241e2lcsom1a`) was therefore stopped before either of its first
two screen jobs completed.  It contains no evaluation summary, progress marker,
state bank, restoration row, or action/visual/joint output.  PAI records
`StoppedByUser`; its runtime log SHA-256 is
`069e1f79bff270681ded56bda91b1a76e73723b9ed54d41970efe7906f085426`.
No v18 output is eligible for or reused in the final run.

## Correction and verification contract

The two 75-way rollout environments and their one-way policy environments now
explicitly use `physx_cuda`.  A unit contract parses both scripts and compares
the padded rollout backend to the frozen preregistration.  The unit suite passed
11/11 after the correction.

All four development-server GPUs had pre-existing processes, so the agent did
not share or terminate them.  Instead, the next dedicated two-A800 PAI launcher
first runs `smoke_action_atlas_cuda.py` in its private artifact directory.  The
smoke creates 75 CUDA environments, loads a pinned verified StackCube
checkpoint, constructs one 5x5 atlas, executes all valid candidates with three
repeats and the full 4+20+5 rollout, and writes a fail-closed non-scientific
summary.  The formal pipeline starts only after that summary reports
`CUDA_ACTION_ATLAS_SMOKE_PASS`.

The repair changes no scientific threshold, candidate grid, repeat count,
action, state bank, phase, seed bank, checkpoint selection rule, utility, or
statistical decision rule.
