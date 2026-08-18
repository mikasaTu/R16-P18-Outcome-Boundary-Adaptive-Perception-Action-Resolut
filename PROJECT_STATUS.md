# Project status

Updated 2026-08-19 for Step6 / Stage-2.7R.

## Current decision

**`NO_GO_CORE_MECHANISM`** is the fixed final status for
`stage27r-core-mechanism-reset-v1`. The positive task screen failed for the
anchor and every executed candidate; `StackCubeHard` fallback was not implemented.
`PushCube-v1` reached only 52.67% repaired
`success_hold5`, below the preregistered 70% negative-control gate, so no formal
negative control was available. Formal `StackCube-v1` and `PegInsertionSide-v1`
oracle output is diagnostic only.

## Evidence summary

- Six confirmatory shards, 97,920 rows total; 96 states per diagnostic task, 34
  conditions, five repeats; model seeds 16018/16019/16020.
- Visual `FC-CC` is positive on both diagnostic tasks and all three frozen utility
  weight sets. Peg action `CF-CC` is task-level null; Stack action is negative.
  Joint `FF-max(FC,CF)` is negative for both tasks and all weights. This is
  negative synergy relative to the better single axis, not a claim that FF is
  below CC in every state.
- Balanced joint positive-state fractions by seed are Peg
  `[0.4167,0.3750,0.4063]` and Stack `[0.2917,0.2500,0.4479]`; all clear 10%, but
  task-level means/CIs remain negative because negative states lose more, so the
  joint gate is not reversed.
- Physical FC/FF tile selection is a privileged offline outcome oracle, not a
  deployable selector. Raw rows do not contain action vectors, so no numerical
  action-discontinuity claim is made.
- Fresh-reset prefix fidelity, raw-outcome recomputation, compute accounting,
  paired statistics, split/leakage, unit/compile/smoke, fail-on-overwrite,
  autoresume, SHA256, and predecessor-immutability audits all pass. Formal
  conditions are serial fresh-reset replays, not simultaneous 10-condition
  stepping.
- All-coarse/all-fine measured cost ratio is `0.7387951746`; 25% and 50% budgets
  are infeasible and 75% is exploratory only.
- The prereg literal treatment names stop at tile3, while failed 2x2 recovery
  selected formal 4x4 tile0..15 (known schema/name mismatch). `mode_probabilities`
  was not consumed: no action-mode dropout was run; formal results are shared-weight
  runtime oracle treatments. The local archive is prepared and checked for the
  upcoming commit; raw shards remain on CPFS. Rows retain source_episode/bank_id/
  model_seed/task/phase; branch_step/prefix_actions/checkpoint/state-bank hashes are
  linked by sidecars, while the state banks retain the branch/prefix fields.

## Scope and next action

The report is in
[`experiments/maniskill_stage27r_core_mechanism_reset_v1/docs/STAGE27R_FINAL_REPORT.md`](experiments/maniskill_stage27r_core_mechanism_reset_v1/docs/STAGE27R_FINAL_REPORT.md).
No Stage-2.8/OOD/pi0.5/real-robot work was started and no Stage-2.8 draft is
created. Stage-1/2/2.5/2.6 directories, raw evidence, thresholds, and results
remain unchanged.
