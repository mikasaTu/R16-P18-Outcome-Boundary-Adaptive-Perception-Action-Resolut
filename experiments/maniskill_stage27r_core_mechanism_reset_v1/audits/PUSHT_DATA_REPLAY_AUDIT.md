# PushT demonstration conversion audit

This audit explains a data-conversion failure found before training. It does
not change the preregistered success threshold or the formal causal backend.

## Observations

- PAI v12 replayed 400 official PushT action trajectories on PhysX CPU and
  retained 0 successes.
- PAI v13 replayed the same pinned official GPU-physics source on
  `physx_cuda`, with the first recorded state restored, and retained only
  149/400 successes.
- PAI v14 used ManiSkill 3.0.1's official `--use-env-states` replay mode and
  retained 389/400 successes. The first 300 retained trajectories were frozen
  as exact 200/50/50 train/validation/test splits.

No failed action replay was relabeled as successful, no threshold was lowered,
and the v12/v13 directories were preserved under explicit `.incomplete-*`
names.

## Code-path explanation

The pinned official replay implementation notes that GPU tasks such as PushT
can diverge when replayed with a different number of parallel environments.
The action-only path advances the contact solver again, so small GPU-physics
differences compound over a 100-step push. Restoring only the initial state
does not remove that accumulated drift. This explains the 149/400 result even
though the source trajectories were successful.

`--use-env-states` rebinds the rendering rollout to the physical state sequence
stored by the official successful demonstration. It is therefore appropriate
for converting the pinned `obs_mode=none` demonstration into RGB training
data. It is not used to create confirmatory branches or causal outcomes.

## Scope boundary

Formal Stage-2.7R evaluation remains `physx_cpu` with fresh reset and lockstep
prefix replay. Mid-state restoration is prohibited there. The env-state path
is limited to offline RGB demonstration materialization, where the scientific
object is the already recorded official successful trajectory.

Frozen PushT split SHA256 values are recorded in the task's
`DATA_COMPLETE.json` and copied into the final artifact manifest.
