# Stage-2.7R Outcome-Marginal-Value Core-Mechanism Reset

Final status: **`NO_GO_CORE_MECHANISM`** (2026-08-19 report update).

This directory is the only mutable scientific scope for Stage-2.7R. It tests the
CC/FC/CF/FF outcome-marginal-value mechanism with one unified multi-resolution
policy, symmetric eight-step treatments, fresh-reset lockstep branching, real
compute accounting, and source-episode-paired statistics. The preregistration was
frozen before confirmatory output; all registered oracle arms ran after upstream
screen failure, but the failed gate caps the claim tier.

The anchor and every executed positive screen candidate failed; the `StackCubeHard`
fallback was not implemented. `PushCube-v1` reached only `0.5266666667` repaired
`success_hold5`, below the 70% negative-control gate, so there is no formal negative
control. Formal `StackCube-v1`/`PegInsertionSide-v1`
oracle rows are diagnostic only: six shards, 97,920 rows, 96 states per task, 34
conditions, five repeats, model seeds 16018/16019/16020, and three frozen utility
weight sets. The visual contrast is positive on both diagnostic tasks; Peg action is
task-level null, Stack action is negative, and `FF-max(FC,CF)` is negative for both
tasks and all weights. The physical tile choice is a privileged offline oracle, not
a deployable selector.

Measured all-coarse/all-fine cost is `0.7387951746` (coarse/full); 25% and 50%
budgets are infeasible, while 75% is exploratory only. Prefix fidelity and all
independent audits pass, but formal replay is serial fresh-reset rather than
simultaneous 10-condition stepping. See [`docs/STAGE27R_FINAL_REPORT.md`](docs/STAGE27R_FINAL_REPORT.md)
for the ten-question report and evidence boundaries.

The local Git archive is prepared and SHA256-checked for the upcoming commit; the
six raw shards (426,794,614 bytes) remain on CPFS and are not in the GitHub-safe
archive. The prereg literal treatment names list only tile0..3, while the failed
2x2 recovery gate selected the formal 4x4 grid (tile0..15); this known schema/name
mismatch does not change the row validators. The frozen four-way mode-dropout
probabilities were not consumed: training had visual/tile randomization but no
action-mode dropout, and formal results are runtime oracle treatments on shared
weights. Raw rows retain source_episode/bank_id/model_seed/task/phase, while
branch_step/prefix_actions/checkpoint hash/state-bank hash are linked through
bank/state-bank/lineage sidecars; the state banks themselves retain these fields.

No learned router, OOD study, Stage-2.8, pi0.5, or real-robot evaluation was
started. Existing Stage-1/2/2.5/2.6 directories and raw evidence remain
unchanged.
