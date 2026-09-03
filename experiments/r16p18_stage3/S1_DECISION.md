# S1 Decision

This document is generated from the supplied profile. It records G1 only; it is not a rollout or training result.

## G1 gates

| Gate | Status |
| --- | --- |
| 1_cost_reproduction | **FAIL** |
| 2_visual_k_over_N_ge_0.20 | **PASS** |
| 3_action_k_over_N_ge_0.20 | **PASS** |
| 4_native_resolution_support | **PASS** |

**Unique label: `BLOCKED_BY_SUBSTRATE`**

Diagnostic budget-geometry label: `PROCEED_JOINT`

Conditional flags: `VISUAL_GATE_REQUIRES_COARSE_REUSE`

G1.4 evidence tier: source-confirmed native paths; fresh runtime forward verified.

G1.1 numeric reproduction within 5%: True; protocol device compliant: False.

### Candidate combinations at the preregistered 0.20 threshold

- `2_visual_k_over_N_ge_0.20`:
  - coarse -> fine_grid2, alpha=0.75, with_coarse_reuse, wall=0.4130, FLOPs=0.4946, native=True
  - coarse -> fine_grid4, alpha=0.75, with_coarse_reuse, wall=0.3211, FLOPs=0.4946, native=True
- `3_action_k_over_N_ge_0.20`:
  - coarse -> fine, alpha=0.50, without_reuse, wall=0.2527, FLOPs=0.2500, native=True
  - coarse -> fine, alpha=0.50, with_coarse_reuse, wall=0.3357, FLOPs=0.3333, native=True
  - coarse -> fine, alpha=0.75, without_reuse, wall=0.5027, FLOPs=0.5000, native=True
  - coarse -> fine, alpha=0.75, with_coarse_reuse, wall=0.6679, FLOPs=0.6667, native=True

### Per-metric FLOP candidates (sensitivity table)

These rows pass the FLOP account alone; G1 above uses the intersection with wall-clock.

- `visual` threshold counts (0.10/0.20/0.30): 4/4/2
  - coarse -> fine_grid2, alpha=0.75, without_reuse, FLOPs=0.244656, native=True
  - coarse -> fine_grid2, alpha=0.75, with_coarse_reuse, FLOPs=0.494599, native=True
  - coarse -> fine_grid4, alpha=0.75, without_reuse, FLOPs=0.244656, native=True
  - coarse -> fine_grid4, alpha=0.75, with_coarse_reuse, FLOPs=0.494599, native=True
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
