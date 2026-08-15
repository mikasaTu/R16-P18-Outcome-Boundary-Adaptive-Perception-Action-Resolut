# Stage-2.6 Counterfactual Completion

This directory implements preregistered shared-prefix causal stopping and a deployable observation/history counterfactual completion gate for the already-selected Stage-2.5 StackCube ACT checkpoints.

Stage-2.5 is immutable. The user execution override requires all planned Stage-2.6 experiments to run even after a failed gate; frozen thresholds and decision precedence still determine the final status, and downstream evidence cannot reverse an upstream failure.

Final status: `NO_GO_SHARED_PREFIX_FIDELITY`. All 4,200 planned closed-loop
episodes were nevertheless completed as explicitly requested. See
[`docs/STAGE26_FINAL_REPORT.md`](docs/STAGE26_FINAL_REPORT.md) for the bounded
interpretation and
[`artifacts/formal-run/r16p18-stage26-counterfactual-20260816-v13`](artifacts/formal-run/r16p18-stage26-counterfactual-20260816-v13)
for the compact, hash-verified result snapshot.
