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
