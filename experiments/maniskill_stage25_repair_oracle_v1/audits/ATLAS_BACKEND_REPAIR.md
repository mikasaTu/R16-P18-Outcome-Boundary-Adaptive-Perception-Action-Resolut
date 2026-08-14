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
smoke creates 75 CUDA environments, loads the selected seed-16018 StackCube
checkpoint, restores a hash-bound real calibration-bank state, constructs one
5x5 atlas, executes valid candidates with three repeats and the full 4+20+5
rollout, and writes a fail-closed non-scientific summary.  It requires at least
90% candidate validity and a real outcome for every valid candidate.  The
formal pipeline starts only after that summary reports
`CUDA_ACTION_ATLAS_SMOKE_PASS`.

An initial v19 smoke used an arbitrary 5k checkpoint/initial state and its
assertion only checked internal count equality; 0 valid candidates therefore
passed vacuously.  The agent caught this by inspecting the saved accounting,
stopped v19, rejected that smoke as evidence, and tightened the input and pass
condition above.  No v19 output is eligible for the final run.

The tightened v20 smoke then failed before the formal orchestrator began,
showing that the actual selected-checkpoint/calibration-state atlas did not meet
the frozen 90% validity requirement.  The next diagnostic revision persists
the nominal action, action-space bounds, per-step/dimension violation counts,
candidate extrema, and validity mask even on failure.  This is diagnosis only;
the threshold remains frozen.

Run v21 (`dlca742ruakkxqj7`) persisted that diagnostic and failed before the
formal orchestrator began.  Its 25/25 arm candidates were within `[-1, 1]` at
all four prefix steps.  The only violations were the unperturbed gripper
coordinates: raw ACT predictions were `1.00121` through `1.00191`, so all 25
candidates inherited the same small overshoot and were declared invalid.  The
failure JSON SHA-256 is
`7880685b1f92da8317afa70954756cfe524b6f601ed560ba288e6e1b0f52b99a`;
the runtime log SHA-256 is
`c0d4758074308ee89190e4a762010cf7ae0a7a946d9eea7654062a8881032409`.

This exposed a semantic implementation defect rather than a PCA/grid defect.
The frozen atlas varies `all_arm_dimensions_first_four_steps` and the frozen
state-bank contract stores `last_legal_gripper_command`.  ManiSkill 3.0.1's
normalized controllers execute `clip(raw, -1, 1)`, but the original state-bank
builder had stored the pre-controller raw prediction, and the atlas had copied
the raw nominal gripper into every non-gripper candidate.

The correction now records the controller-effective legal gripper command in
state-bank metadata and explicitly binds that single command into all 25
non-gripper candidates.  A stored out-of-range command raises an error; no
candidate coordinate is silently clipped.  The raw ACT nominal remains in the
output beside a separately named `atlas_center_action_first4`, so overshoot and
the exact replacement stay auditable.  Neutral holds likewise retain the
controller-effective final gripper value.  PCA neighbors, PCA directions,
candidate arm values, action-space validity checks, radii, thresholds, rollout
counts, seeds, utilities, and decision rules are unchanged.

The repairs change no scientific threshold, candidate arm grid, repeat count,
state identity/phase selection, seed bank, checkpoint selection rule, utility,
or statistical decision rule.  Fifteen unit/static tests pass, including
explicit rejection of an illegal stored gripper and a call-site contract that
every atlas binds the state-bank command.  All nine frozen scientific hashes
continue to verify.
