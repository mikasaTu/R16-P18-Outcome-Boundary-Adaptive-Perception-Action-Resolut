# ManiSkill3 RGB-ACT task screen and oracle boundary probe

This directory is the only mutable scope for the second R16-P18 validation
stage. The archived LIBERO baseline-gate tree and artifacts outside this
directory remain unchanged.

The stage answers four necessary-condition questions:

1. Are 300 independent, reproducible successful demonstrations available per
   formal task?
2. Does the official ManiSkill RGB ACT baseline leave useful performance
   headroom?
3. Do nontrivial local action-outcome boundaries exist?
4. Do visual distinctions and action distinctions interact?

It is an oracle existence probe, not an implementation of the R16-P18
adaptive selector. A learned effect predictor, budgeted selector, adaptive
token/action allocation, Diffusion Policy, DINO-WM, and pi0.5 are out of scope.

The preregistration is frozen in `preregistration.yaml`. Large demonstrations,
replayed RGB trajectories, checkpoints, and simulator state banks live under
the pinned CPFS paths recorded in `environment_lock.json`; Git contains their
manifests, SHA256 digests, summaries, and the code needed to reproduce them.

Implemented protocol components currently include deterministic 300-trajectory
selection and integrity verification, official trajectory-to-RGB replay,
official ACT architecture/data imports with split-aware validation-only
checkpoint selection, complete-state checkpoint/resume, fixed-seed closed-loop
evaluation with contact accounting, and a two-GPU PAI matrix launcher. The
oracle state-bank and boundary atlases remain gated on the ACT baseline result.

The SAPIEN PhysX GPU shared library is downloaded before PAI execution and
content-addressed in `locks/physx_gpu_library.json`; formal jobs reject a
missing or mismatched library instead of downloading simulator code at run
time. The launcher also enters the pinned project checkout before starting
official replay so CPU multiprocessing workers never inherit `/root` as their
working directory.

RGB replay follows the frozen data gate exactly: at least 95% of each locked
split must replay successfully. Every saved episode must still be successful,
seed-identifiable, unique, and an exact subset of the selected raw split;
missing episode seeds are reported explicitly. Incomplete multiprocessing
shards are quarantined before a retry and are never counted as demonstrations.
The frozen retry budget is nine for CPU control-mode conversion tasks and
three for PushT state replay; retries preserve the same selected trajectory,
episode seed, and initial simulator state.

`PegInsertionSide-v1` was retired after two official RGB replay attempts both
saved 188/200 training trajectories (94%), below the frozen 95% data gate.
The preregistered first fallback, `PlugCharger-v1`, was activated for the
allowed `data_availability` reason before any fallback replay result was
observed. The immutable decision record is `task_selection.json`; thresholds
were not changed.

The official PlugCharger archive labels its environment reward mode as
`dense`, while the pinned v3.0.1 task supports `none` and `sparse`. Replay does
not record or consume rewards, so a frozen metadata-only adapter changes this
field to `sparse` in selected subset JSON. It does not alter simulator states,
actions, seeds, control conversion, policy inputs, or dynamics; both values and
the zero-trajectory infrastructure failure that exposed the mismatch are
recorded in `task_selection.json`.

The adapted PlugCharger replay subsequently saved 163/200 training
trajectories (81.5%) with the frozen nine-retry budget. This is below the
unchanged 190/200 gate, so PlugCharger was retired. Once the gate was already
mathematically unreachable—and before any result from the next task—the
preregistered second fallback `PullCubeTool-v1` was activated. Its official
source contains 1,000 successful episodes and 937 eligible unique seed/state
identities; the fixed hash ranking selected 300 with complete 200/50/50 and
triple-uniqueness checks. Its `pd_ee_delta_pose` RGB replay uses the same
nine-retry CPU conversion protocol. The evaluation horizon was frozen at 300
before replay because the official successful-demo elapsed-step 95th
percentile is 283. The exact decision timeline, failed-output hashes, and PAI
job identity are recorded in `task_selection.json`.
