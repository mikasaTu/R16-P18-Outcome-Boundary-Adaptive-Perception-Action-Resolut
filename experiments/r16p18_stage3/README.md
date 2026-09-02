# R16-P18 Stage 3 / S1: budget-feasibility reconstruction

This directory is the frozen, zero-rollout S1 implementation.  It answers only
whether the current model exposes a non-empty fixed-budget feasible region on
the visual and action axes.  It does not re-run an environment, create an
episode, submit a PAI job, train a model, or run policy inference in an
environment.

The preparation utility reads one observation from an immutable replay HDF5
through the frozen Stage-2.7R preprocessing code. The profiler then loads the
frozen Stage-2.7R EMA checkpoint with dummy tensor spaces—never an environment—
and measures batch-size-one forward passes. If any input is absent, it fails
closed. CUDA synchronization, warmups, repeats, seed, software versions,
device information, and operator FLOPs from PyTorch's counter are recorded in
the profile JSON.

The feasibility calculator reports both measured wall-clock and FLOP
accounts.  FLOPs are a secondary accounting signal, not a replacement for
wall-clock measurement.  It explicitly lists grid points at which the two
accounts disagree and emits no experimental conclusion when required profile
fields are missing.

The only inherited numerical facts in this directory are the previously
recorded Stage-2.7R values:

* estimated-FLOP numerator (all-coarse): `92438200000000`;
* estimated-FLOP denominator (all-fine): `125120200000000`;
* recorded ratio: `0.738795174560143`;
* previously recorded wall-clock ratio: `0.75025979` (historical reference,
  not a new measurement).

No new profiling result is claimed by the source tree.  Run
`profile_s1_costs.py` only after supplying the actual cached tensor and
checkpoint.  Every generated result is fail-on-overwrite and should be added
to `SHA256SUMS` with `update_sha256s.py`.

## Intended commands

```bash
python experiments/r16p18_stage3/scripts/prepare_s1_observation.py \
  --h5 /path/to/immutable/trajectory.h5 \
  --task StackCube-v1 \
  --output /tmp/r16p18-s1-stack-observation.pt

python experiments/r16p18_stage3/scripts/profile_s1_costs.py \
  --checkpoint /path/to/frozen/stage27r/checkpoint.pt \
  --checkpoint-format stage27r \
  --task StackCube-v1 --model-seed 16018 \
  --observations /tmp/r16p18-s1-stack-observation.pt \
  --stage27-statistics experiments/maniskill_stage27r_core_mechanism_reset_v1/artifacts/formal-run/statistics.json \
  --output experiments/r16p18_stage3/S1_PROFILE.json \
  --device cuda:0 --warmup 50 --repeats 200 --seed 2718001

python experiments/r16p18_stage3/scripts/compute_feasibility.py \
  --profile experiments/r16p18_stage3/S1_PROFILE.json \
  --output-dir experiments/r16p18_stage3

python experiments/r16p18_stage3/scripts/build_inventory.py \
  --repo-root . \
  --output experiments/r16p18_stage3/S1_INVENTORY.md

python experiments/r16p18_stage3/scripts/update_sha256s.py \
  --root experiments/r16p18_stage3
```

The profiler command is authorized only on dev05 with one owner-safe GPU.
These commands document the bounded workflow; they are not evidence that a
fresh profile has been performed.
