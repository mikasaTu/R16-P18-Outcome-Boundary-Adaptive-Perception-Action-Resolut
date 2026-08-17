# Stage-2.7R frozen plan

The complete machine-readable protocol is in `../preregistration.yaml`. The
experiment uses StackCube-v1 as the anchor, screens the second positive task in
the frozen order PegInsertionSide, PlugCharger, PullCubeTool, PushT, then the
predeclared harder-StackCube fallback, and admits PushCube as negative control
only if its repaired `success_hold5` reaches 70%.

The causal comparison is CC/FC/CF/FF under an identical eight-step treatment
window followed by the same 20-step native/full continuation. Branches are
replayed from reset with identical prefix actions; no mid-state restore is used.
The formal backend is CPU PhysX. Confirmatory state/treatment repeats are
aggregated before source-episode-paired inference. Actual encoder calls, policy
calls, tokens, FLOPs, latency, memory and action opportunities determine the
25/50/75% budgets.

All registered arms run regardless of intermediate gate results. The result is
one of GO_FULL_JOINT, REVISE_SHARED_AXIS_ROUTER, REVISE_VISUAL_ONLY,
NO_GO_CORE_MECHANISM, or NO_GO_CAUSAL_BACKEND under the frozen precedence.
