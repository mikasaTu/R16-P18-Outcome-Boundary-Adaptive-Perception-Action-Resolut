# Stage-2.6 final report

Protocol: `R16-P18-MS5-STAGE26-COUNTERFACTUAL-COMPLETION-V1`

Formal result: **NO_GO_SHARED_PREFIX_FIDELITY**. The independent raw-evidence
audit passed, but the experiment did not establish a valid shared physical,
controller and policy prefix. The downstream predictor and closed-loop results
were completed because the user explicitly required every planned arm to run;
they are exploratory and cannot reverse the preregistered fidelity decision.
No Stage-2.7 draft was created.

## What was frozen and executed

The Stage-2.5 directory was not changed. The selected StackCube checkpoints for
model seeds 16018, 16019 and 16020 were reused without reselection. Deterministic,
pairwise-disjoint banks contain 512 train-source, 128 calibration and 200
confirmatory episode seeds, in the same order for all three checkpoints.

The formal evidence consists of 7,565 rollout capsules, 22,695 counterfactual
branch rows, 120 completed collection shards, 3 × 64 × 10 shared-prefix replay
comparisons, and 4,200 confirmatory closed-loop episodes (200 seeds × 3 model
seeds × 7 arms). Predictor architecture, temperatures and thresholds were
frozen before confirmatory evaluation. Confirmatory outcomes were not used for
model, checkpoint, threshold, or early-stopping selection.

The deployable predictor path uses frozen visual latents, proprioception,
executed-action history, the base policy's first five predicted actions,
gripper history, and temporal-consistency features. Static tests exclude
simulator success, phase truth, object pose, goal distance and privileged
contact from that feature construction. Privileged arms remain diagnostic.

## 1. Was the shared prefix actually identical?

No. All three checkpoint seeds failed every preregistered tolerance by a wide
margin:

| model seed | action max abs | translation | rotation | categorical agreement | observation hash agreement |
|---|---:|---:|---:|---:|---:|
| 16018 | 0.323340 | 0.016322 m | 0.184156 rad | 91.72% | 0% |
| 16019 | 0.237250 | 0.017701 m | 0.411081 rad | 83.75% | 0% |
| 16020 | 0.391112 | 0.015209 m | 0.380347 rad | 86.09% | 0% |

The frozen limits were action ≤ 1e-6, translation ≤ 1e-5 m, rotation ≤ 1e-4
rad, and 100% categorical agreement. Development probes showed that the public
ManiSkill state round-trips numerically (maximum serialized-state difference
2.38e-7), while a rerender can differ by one RGB gray level. Even when the
captured observation, RNG order, pending ACT chunk and controller state were
restored, an exact-reference first action still yielded roughly 0.46–0.65 mm
translation and 0.016–0.021 rad rotation error. The bounded implementation
diagnosis is that the public state does not serialize PhysX contact-solver
warm-start/cache state. This diagnosis does not relax or replace the failed
formal replay test.

## 2. Does the stopping confound still appear?

Exploratorily, yes for privileged *termination*, but not as a validated
shared-prefix causal result. `privileged_terminate_first_success` improved end
success over fixed horizon by **+13.33 percentage points**, paired 95% CI
**[+10.33, +16.50]**, with seed gains +13.5, +16.0 and +10.5 points.
`privileged_neutral_after_hold5` improved only +1.67 points, CI [-0.33, +3.67],
and was not significant after Holm correction. Terminating at observed success
therefore removes post-success loss in these rollouts, while neutral holding
does not reproduce most of that effect.

Because restored branches failed fidelity, this is privileged closed-loop
oracle evidence about these independently executed arms, not confirmation of
the requested matched-prefix counterfactual mechanism.

## 3. Can hold-versus-continue difference be learned?

Not reliably. The selected model is the 3,495-parameter linear probe. Its
calibration metrics by model seed were:

| seed | stop-beneficial AUPRC | ECE | NOT_DONE false-stop | DONE_FRAGILE recall |
|---|---:|---:|---:|---:|
| 16018 | 0.392 | 0.055 | 2.79% | 28.57% |
| 16019 | 0.449 | 0.037 | 4.10% | 34.78% |
| 16020 | 0.609 | 0.040 | 4.10% | 71.43% |

Only seed 16020 passed all offline thresholds; the preregistered requirement
was at least two of three with no catastrophic third seed. Source-episode
clustered 10,000-bootstrap intervals are stored in `extended_statistics.json`.
Leave-one-model-seed-out AUPRC was only 0.283, 0.133 and 0.101, further showing
that the learned distinction was checkpoint-specific rather than stable.

The label strata explain why the task is difficult. Six steps before first
success, continuing is overwhelmingly beneficial; at first success and
first-hold5, holding is more often beneficial. Visual/action history must
identify this narrow transition, and the learned boundary did not transfer
consistently across the three base-policy checkpoints.

## 4. How much privileged gain did the learned gate recover?

It recovered **-37.5%**: the privileged gain was +13.33 points, while the
learned counterfactual gate changed end success by **-5.00 points**, paired 95%
CI **[-8.00, -2.17]**, sign-flip p=0.0013. Per-seed changes were +4, -6 and -13
points, so only one of three seeds was nonnegative. End success fell from
33.83% to 28.83%.

## 5. Is any gain merely simple success detection?

There is no aggregate learned gain to attribute. The success-only classifier
also reduced end success by 5.5 points, CI [-9.0, -2.17]. The counterfactual
gate and success-only stop sets had Jaccard overlap 0.672, 0.306 and 0.830 by
seed. The counterfactual head was modestly less harmful than stop-rate-matched
fixed-time (-8.83 points) and random (-11.0 points) controls, but that ordering
does not turn its own negative effect into evidence for a useful
counterfactual advantage.

## 6. Did false stopping harm success-once?

Yes. Aggregate success-once fell from 47.83% to 39.50%, a loss of **8.33
points**, exceeding the allowed 2-point reduction. Paired rescued/harmed counts
for the learned gate were 15/7, 7/19 and 7/33 for seeds 16018, 16019 and 16020.
Seed 16020 stopped in 74% of episodes at mean step 84.95 and harmed 33 episodes
while rescuing 7. This is the direct code-and-trace mechanism behind the large
negative seed effect: the two-consecutive-step gate fires before recoverable
trajectories complete.

Post-success loss fell from 14.00% to 10.67%, a 23.81% relative reduction, but
this missed the 30% threshold and was outweighed by early false stops. The
completion head added 1.84% of fixed-policy inference wall latency and made no
more policy calls than fixed horizon; these accounting facts are not claims of
token or wall-clock compute savings.

## 7. Were the model seeds consistent?

No. The learned end-success effect was +4, -6 and -13 points. Offline
learnability passed only seed 16020, yet that seed had the worst closed-loop
effect because its frozen gate stopped most frequently. Thus calibration
quality at the state-row level did not imply safe episode-level stopping. In
contrast, the privileged terminate arm was positive for all three seeds.

## 8. Is the project eligible for visual × temporal Stage-2.7?

No. The highest-precedence shared-prefix gate failed, offline learnability
failed, and the deployable gate significantly reduced aggregate end success.
The protocol requires `GO_STOP_NORMALIZED_BASELINE` before creating a Stage-2.7
preregistration draft, so no draft or Stage-2.7 experiment was produced.

## Evidence classes and bounded conclusion

- **Confirmed code semantics:** frozen checkpoints and disjoint seed banks;
  complete capsule/controller/ACT/RNG serialization code; non-privileged
  deployable inputs; fixed thresholds; seven paired arms; fail-on-overwrite and
  autoresume contracts; 13 contract tests pass; scientific SHA256 manifest
  passes.
- **Observed paired evidence:** 4,200 closed-loop episodes and 10,000 paired
  bootstrap/sign-flip results. These show the learned gate is harmful on
  aggregate and inconsistent across seeds.
- **Privileged oracle evidence:** terminate-at-first-success is positive in all
  seeds. Neutral-after-hold5 is small and inconclusive.
- **Learned deployable evidence:** offline gate fails; closed-loop gate is -5
  points and reduces success-once by 8.33 points.
- **Bounded inference:** missing contact-solver cache is the best-supported
  explanation for restoration divergence; early false stops explain learned
  closed-loop harm.
- **Not tested:** OOD, another task, spatial visual/action routing, token
  saving, wall-clock compute saving, Stage-2.7, pi0.5, or real robots.

Independent audit status: `INDEPENDENT_STAGE26_AUDIT_PASS`, with 7,565 branch
label rows recomputed and no problems. This means the negative decision was
reproduced from raw rows; it does not make the failed fidelity gate pass.
