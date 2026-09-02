# S1 Decision

This document is generated from the supplied profile. It records G1 only; it is not a rollout or training result.

## G1 gates

| Gate | Status |
| --- | --- |
| 1_cost_reproduction | **FAIL** |
| 2_visual_k_over_N_ge_0.20 | **FAIL** |
| 3_action_k_over_N_ge_0.20 | **FAIL** |
| 4_native_resolution_support | **PASS** |

**Unique label: `BLOCKED_BY_SUBSTRATE`**

G1.4 evidence tier: source-confirmed native paths; fresh runtime forward unverified.

### Candidate combinations at the preregistered 0.20 threshold

- `2_visual_k_over_N_ge_0.20`:
  - none under the conservative both-metrics rule
- `3_action_k_over_N_ge_0.20`:
  - none under the conservative both-metrics rule

### Diagnostic FLOP-only candidates

These do not pass G1 while fresh wall-clock is unavailable.

- `visual` threshold counts (0.10/0.20/0.30): 4/4/2
  - coarse -> fine_grid2, alpha=0.75, without_reuse, FLOPs=0.205696, native=True
  - coarse -> fine_grid2, alpha=0.75, with_coarse_reuse, FLOPs=0.451389, native=True
  - coarse -> fine_grid4, alpha=0.75, without_reuse, FLOPs=0.205696, native=True
  - coarse -> fine_grid4, alpha=0.75, with_coarse_reuse, FLOPs=0.451389, native=True
- `action` threshold counts (0.10/0.20/0.30): 4/4/3
  - coarse -> fine, alpha=0.50, without_reuse, FLOPs=0.250000, native=True
  - coarse -> fine, alpha=0.50, with_coarse_reuse, FLOPs=0.333333, native=True
  - coarse -> fine, alpha=0.75, without_reuse, FLOPs=0.500000, native=True
  - coarse -> fine, alpha=0.75, with_coarse_reuse, FLOPs=0.666667, native=True

## Accounting policy

a candidate must meet the threshold under both wall-clock and FLOP accounts; disagreements are reported separately

Wall-clock/FLOP disagreement cells are listed in `S1_FEASIBILITY.json`; they are not silently resolved.

G1 recorded; stop and await human confirmation; no S2 preparation

BLOCKED_BY_SUBSTRATE
