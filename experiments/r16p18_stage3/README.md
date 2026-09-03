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

The checked-in result now includes a fresh dev14 profile of the frozen
Stage-2.7R checkpoint. It used one cached replay observation, physical GPU 2,
batch size 1, 50 warmups, and 200 CUDA-synchronized repeats for each of five
native conditions. `S1_PROFILE.json` retains every timing sample; the immutable
input hashes and disclosed GPU co-tenancy are recorded in
`S1_DEV14_RUNTIME_AUDIT.json`. Because that co-tenancy violates the frozen
`one_owner_safe_cuda_gpu` requirement, the formal G1 label is
`BLOCKED_BY_SUBSTRATE`; `PROCEED_JOINT` is retained only as the diagnostic
budget-geometry result. Every generated result is covered by `SHA256SUMS`.

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

The original profiler location was dev05. The user explicitly amended the
location to the existing dev14 DSW environment on 2026-09-02; the amendment is
recorded separately so the frozen protocol is not silently rewritten. These
commands document the bounded workflow and a profile is evidence only when its
raw JSON, input hashes, runtime audit, and SHA256 manifest are all present.
