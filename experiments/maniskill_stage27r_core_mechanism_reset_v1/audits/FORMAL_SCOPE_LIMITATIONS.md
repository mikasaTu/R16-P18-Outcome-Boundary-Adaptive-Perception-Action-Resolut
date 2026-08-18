# Formal scope and claim limitations

This audit records protocol coverage limitations without changing the frozen
thresholds, task order, treatments, or decision precedence. The remaining
oracle arms continue to run because the user explicitly required completion
after intermediate gate failures.

## No gate-qualified positive task

The frozen screen selected neither a second positive task nor a healthy
negative control. It also found that the StackCube anchor itself failed the
positive gate. The resumable launcher uses `PegInsertionSide-v1` as a
diagnostic fallback when `selected_positive` is null, so the executed
StackCube/Peg factorial is useful for mechanism diagnosis but is not a
gate-qualified two-positive-task confirmatory experiment. The frozen final
decision correctly gives the upstream task gate precedence and therefore
cannot return a positive status from these downstream outcomes.

## Unimplemented task fallbacks

The preregistration names `StackCubeHard-v1` as the last positive candidate and
`PickCube-v1`, then `LiftPegUpright-v1`, as negative fallbacks. This run did not
define/generate/train/screen `StackCubeHard-v1`, and it did not train or screen
the two negative fallbacks after PushCube missed its 70% gate. Those omissions
cannot be repaired post hoc after screen outcomes are known. The final report
must state that the candidate-order and negative-control search were
incomplete; it must not describe the diagnostic Peg run as the selected
positive task or claim that no healthy registry negative exists.

## Low-budget feasibility

All treatments share an eight-step intervention followed by a 20-step native
fine continuation. In observed calibration accounting, a representative CC
row costs 166.6 GFLOP while FF costs 221.2 GFLOP, so CC is about 75.3% of FF.
Consequently an absolute budget equal to 25% or 50% of all-fine cost cannot
even fund the all-CC baseline. The analysis correctly recomputes
`budget_compliant=false`, but still reports those arms for transparency. They
must not be interpreted as feasible matched-budget deployment comparisons.
The 75% budget is borderline and must be judged from its per-arm recomputed
flag rather than its label.

## Interpretation

These limitations do not invalidate matched-prefix physical contrasts on the
executed state banks. They do prevent a positive R16-P18 claim and narrow the
run to a completed diagnostic falsifier under the frozen
`NO_GO_CORE_MECHANISM` precedence.
