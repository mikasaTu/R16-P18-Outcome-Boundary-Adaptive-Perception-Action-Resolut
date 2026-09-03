# S1 independent final audit

Audit date: 2026-09-03 UTC  
Scope: read-only review of S1 protocol compliance, profile inputs, arithmetic,
plots, tests, and predecessor immutability.

## Formal disposition

The formal G1 result is **`BLOCKED_BY_SUBSTRATE`**. The frozen protocol requires
`one_owner_safe_cuda_gpu`, while the dev14 runtime audit and raw one-second GPU
samples disclose foreign-owner processes resident on physical GPU 2. The user
amended the authorized host from dev05 to dev14 but did not explicitly waive
the device-isolation clause. The measured numbers remain usable as diagnostic
evidence, not as a protocol-compliant formal profile.

Conditioned on accepting the measured cost geometry, the frozen G1 arithmetic
returns **`PROCEED_JOINT`** with
`VISUAL_GATE_REQUIRES_COARSE_REUSE`. The current visual evaluator is
`without_reuse`; its alpha=0.75 wall-clock bounds are only 0.175859 for grid2
and 0.118262 for grid4. Therefore the diagnostic label must not be presented as
a deployable current implementation or a mechanism result.

## Checks

- Cached observation, checkpoint, Stage-2.7R statistics, profile, and raw GPU
  sample hashes match their recorded SHA256 values.
- Five native conditions are present; batch size is 1; each condition has 50
  warmups, 200 CUDA-synchronized timed samples, and operator FLOPs.
- Independent arithmetic reproduces relative deviations of 0.3836% (FLOPs)
  and 0.6804% (wall clock), both numerically below 5%.
- Independent feasibility counts at thresholds 0.10/0.20/0.30 are visual
  4/2/2 and action 4/4/3 under the required both-metrics rule.
- The two recorded wall/FLOP disagreement cells match independent
  recomputation.
- `S1_COST_CURVE.json` now contains measured wall-clock standard-deviation
  error bars for every native grid point, and both curve SVGs render them.
- Static zero-rollout audit passes; no S1 script constructs/resets/steps a
  simulator or submits a PAI job.
- CPU test suite: 10 passed.
- Stage-2.5, Stage-2.6, and Stage-2.7R predecessor scopes are unchanged versus
  `origin/main`.

This audit does not validate outcome-boundary selection, joint synergy, task
success, or any paper claim. S1 stops at G1.
