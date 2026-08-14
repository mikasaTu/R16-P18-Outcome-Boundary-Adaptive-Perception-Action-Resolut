# R16-P18 ManiSkill Stage-2.5 repair and oracle

This directory is the independent implementation and evidence root for
`R16-P18-MS4-STAGE25-REPAIR-ORACLE-V1`.

The predecessor directory `../maniskill_act_boundary_screen_v1/` is immutable.
No script in this directory imports or executes predecessor downstream oracle
code.  Source ideas may be reimplemented here only after their predecessor
hashes and semantic differences are recorded by the Stage-0 audit.

The current user instruction explicitly overrides the archived early-stop
behavior: every planned checkpoint, success-semantics, contact, restoration,
action, visual, and joint-oracle experiment is executed.  Scientific gates and
thresholds remain unchanged and determine evidence tier and final status.  A
failed upstream gate cannot be repaired or relabeled by a downstream result.

Large artifacts are written below the following new CPFS roots:

- data: `/mnt/cpfs/zbl-cpfs-new/dataset/leon/r16-p18-maniskill-stage25-repair-oracle-v1`
- results/checkpoints: `/mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16-p18-maniskill-stage25-repair-oracle-v1`
- logs: `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16-p18-maniskill-stage25-repair-oracle-v1`
- cache: `/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r16-p18-maniskill-stage25-repair-oracle-v1`

The run stops after the privileged joint oracle.  Learned predictors,
deployable routers, OOD evaluation, pi0.5, and real-robot experiments are out
of scope.

