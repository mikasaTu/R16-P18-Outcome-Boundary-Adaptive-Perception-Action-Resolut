# CPU lockstep smoke recovery

Formal run `stage27r-formal-v4` / `dlc1mehxqigs6afj` passed all 11 contract
tests and then failed before screening. The pinned ManiSkill 3.0.1 runtime
rejects `sim_backend="physx_cpu"` with `num_envs=2`; the original smoke tried
to represent two shadows as two slots in one CPU-vectorized environment.

The repair creates two independent `num_envs=1` CPU environments, resets both
with the same seed, applies identical actions, and compares object pose,
rotation, RGB, and categorical success after every step. This preserves the
preregistered deterministic fresh-reset lockstep claim and exercises a stricter
independent-shadow path. No task gate, threshold, seed, model, or formal budget
is changed. The failed run produced no screen or scientific result.
