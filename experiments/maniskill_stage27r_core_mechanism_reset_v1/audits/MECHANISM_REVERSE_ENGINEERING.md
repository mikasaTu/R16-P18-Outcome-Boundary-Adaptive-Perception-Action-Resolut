# Stage-2.7R mechanism reverse engineering

This is a code-first audit of the executed Stage-2.7R oracle. It follows the
code/result tracing procedure in `code-first-ideation/SKILL.md`: read the real
implementation, map each treatment to its executed path, and then explain the
observed paired outcome differences. It does not propose a new idea, change a
threshold, or reinterpret an earlier stage.

The preregistered result remains `NO_GO_CORE_MECHANISM`. This is a bounded
conclusion about this formal policy/task/screen/protocol, not a claim that the
general research direction is universally disproved.

## Audit identity and evidence

- Protocol: `R16-P18-MS6-STAGE27R-CORE-MECHANISM-RESET-V1`.
- Final independent verifier source: commit
  `6893116948025785d0da7860e7ece94ea5497707`, tree
  `5326b8529ff5bc0bdc99ceb6f526117d0fb10e52`.
- Historical post-hoc verifier: commit
  `7554bca313d796fd0b4cdf3abbc817c6ecc7e9fd`, tree
  `837e29d5547c67c8045522164ab72e8fbcc2d5a0`.
- Oracle producer source: commit
  `fa05c2ef52e5cce16f62397540162724bfd4a6b9`, tree
  `6fdb28764d002def6d10e5a9c4f41918fe7713d1`.
- Pinned ManiSkill commit: `a4a4f9272ad64b1564035874b605ceb687b63ed8`.
- Raw evidence: six shards, 97,920 rows; each shard has 96 states, 34
  conditions (CC, CF, 16 FC tiles, 16 FF tiles), and five repeats.
- `ORACLE_VALIDATION.json`: PASS, SHA256
  `be515e8f4fb12825a254ff046cffd0ad0939427d3a598ec7d293725439668bdb`.
- `statistics.json`: SHA256
  `4cc71eff2251f008bfb9910e8a4065459d998a9a635d8dc5d075b40896ce87bd`.
- `RESULT_VECTOR.json`: SHA256
  `bbdd69a67f1cd5d3f760196524549ae7b442ac21afc5f95963194b4eb3b5a095`.
- Raw oracle shard hashes:
  - Peg 16018: `798490f0514865cf63383cfd7238415150f0ffe9e618fcdb7e7fcced1a79e4c7`
  - Peg 16019: `18d6c4d8c56d86a93334721ab5d2aa6ff430def9b82898b851ab8a1415287793`
  - Peg 16020: `abed063078025d97dcc52f69ddb713c07bbb33ae6bea7e64eb6b3a37211a8186`
  - Stack 16018: `9be2072119703fe1e8b2e1ecc16f3488cc49c75b10f606e44b1d93fc5c0581c2`
  - Stack 16019: `86eb7d00e618486783f3ebb6c77ec6845ded1640172dace4e4f256b01f2a0981`
  - Stack 16020: `e98c5e4a5d772d7f7375116b4d9e225fd987abbe333ec98e91d13c19222fba1f`

The implementation files traced here are unchanged at the final verifier
commit. Their SHA256 hashes are recorded in the companion JSON summary.

## What the code actually does

### Visual path

`MultiResolutionDETRVAE.forward` first resizes every camera image to 112x112
and runs the shared ResNet18 backbone. In `fine` mode it additionally crops
one tile from the original 128x128 image, resizes that crop to 112x112, and
runs the same backbone again. The global branch is retained; fine is therefore
global-plus-local, not local-only. The two feature maps are concatenated before
the transformer (`multires_policy.py:105-123`). The accounting code records one
global call per camera and, in fine mode, one extra fine call per camera
(`multires_policy.py:177-186`).

The formal run used the 4x4 calibration choice, hence 16 tile conditions. The
FC and FF rows used in `state_table` are not a deployable crop selector: for
each state and utility weight, the analysis selects the tile with the best
repeated physical outcome (`analyze_stage27r.py:87-106`). That is an offline
physical oracle and is useful for isolating the maximum possible crop effect,
but it must not be reported as a learned or deployable adaptive visual policy.

### Action path and time symmetry

During the eight-step treatment, coarse action queries at step 0 and every
fourth step, then reuses entries from the predicted eight-action chunk. Fine
action queries at every treatment step and executes the first action of each
fresh chunk. This is the exact branch at `stage27r_runtime.py:161-176`.
After the treatment, every arm switches to the same fine/native continuation
for up to 20 steps; this is why the treatment itself is symmetric in duration
and the continuation is common (`stage27r_runtime.py:162-165`). The evaluator
stops uniformly after a five-step success hold (`stage27r_runtime.py:179-183`).

Thus the intended contrasts are:

| contrast | code difference during the 8-step treatment |
| --- | --- |
| FC - CC | add the physical crop branch while retaining the coarse action schedule |
| CF - CC | re-query action every step while retaining the coarse visual branch |
| FF - max(FC, CF) | combine both branches, relative to the better single-axis arm |

The raw rows do not store the action vector at every step. Therefore the
existence of the re-query/cache schedule is confirmed by code and accounting,
but a numerical action-discontinuity magnitude is **not** directly measured.
The observed contact, drop, progress, and success patterns below are mediators
or outcome correlates; they are not proof of an unlogged action-level causal
path.

### Treatment and accounting

Each state-treatment has a fresh reset plus the frozen prefix replay, eight
treatment steps, and a common 20-step continuation (or uniform success hold
termination). Physical oracle simulator calls are offline labeling cost and
are excluded from deployment compute. All raw outcome, schedule, and compute
audits passed; no budget flag was hard-coded.

## Task screen and claim tier

The anchor and all candidate positive tasks failed the preregistered screen.
The per-seed `success_hold5` values are shown as `[16018,16019,16020]`:

| task | CC | FF | screen result |
| --- | --- | --- | --- |
| StackCube-v1 | [0.10, 0.15, 0.16] | [0.00, 0.02, 0.04] | fail |
| PegInsertionSide-v1 | [0.00, 0.00, 0.01] | [0.00, 0.00, 0.00] | fail |
| PlugCharger-v1 | [0.00, 0.00, 0.01] | [0.00, 0.00, 0.00] | fail |
| PullCubeTool-v1 | [0.00, 0.00, 0.00] | [0.00, 0.01, 0.01] | fail |
| PushT-v1 | [0.09, 0.04, 0.05] | [0.06, 0.03, 0.01] | fail |
| PushCube-v1 negative candidate | [0.55, 0.43, 0.60] | not used as formal arm | fail: aggregate is 0.5267 < 0.70 |

Consequently `selected_positive=null`, `selected_negative=null`, and the
formal StackCube/Peg oracle is diagnostic evidence under the preregistered
"run all oracle arms" rule. The missing StackCubeHard fallback and missing
PickCube/LiftPeg negative fallback are protocol limitations, not reasons to
upgrade the claim tier.

## Quantitative effects

All entries below are utility-point differences with 10,000 source-episode
cluster bootstrap replicates. The interval is the paired 95% bootstrap CI;
Holm-adjusted sign-flip p-values are included to show the correction family.

### Visual marginal value: FC - CC

| weight | PegInsertionSide-v1 | StackCube-v1 |
| --- | --- | --- |
| balanced | +0.6223 [0.3715, 0.9152], Holm p=0.0018 | +1.2934 [0.1799, 2.7355], Holm p=0.0018 |
| success-dominant | +0.3667 [0.1986, 0.5724], Holm p=0.0018 | +1.3794 [0.0915, 3.0737], Holm p=0.0018 |
| progress-dominant | +0.9849 [0.6073, 1.4287], Holm p=0.0018 | +1.2479 [0.3037, 2.4585], Holm p=0.0018 |

This is the most stable positive result in the run. It is not a collision
effect: collision was zero in the recorded rows. On the balanced utility,
the phase decomposition is:

| task / phase | visual utility delta | dominant observed components |
| --- | ---: | --- |
| Peg / free-space | +0.5272 | progress +0.5272 |
| Peg / pre-contact | +0.4991 | progress +0.3602; fewer drops +0.1389 |
| Peg / in-hand | +0.9994 | progress +0.7910; recoverability +0.2083 |
| Peg / contact-near-completion | +0.4637 | progress +0.2553; fewer drops +0.1389; recoverability +0.0694 |
| Stack / free-space | +0.4135 | progress +0.4135 |
| Stack / pre-contact | +0.2005 | progress +0.1311; recoverability +0.0694 |
| Stack / in-hand | +1.5188 | one 0->1 success transition contribution +1.3889; progress +0.1299 |
| Stack / contact-near-completion | +3.0410 | two 0->1 success transitions contribution +2.7778; progress +0.1937; recoverability +0.0694 |

The visual result is therefore mainly a small progress/recoverability shift,
with a few discrete success transitions in StackCube. It is not evidence that
fine vision reliably increases success probability in every state.

### Action marginal value: CF - CC

| weight | PegInsertionSide-v1 | StackCube-v1 |
| --- | --- | --- |
| balanced | +0.4553 [-1.5187, 2.8168], Holm p=1.0000 | -4.3739 [-7.9486, -1.3700], Holm p=0.0430 |
| success-dominant | +0.7329 [-1.4812, 3.3800], Holm p=1.0000 | -5.0811 [-9.3367, -1.6001], Holm p=0.0430 |
| progress-dominant | +0.1501 [-1.6006, 2.2857], Holm p=1.0000 | -3.7090 [-6.7430, -1.1565], Holm p=0.0408 |

Peg is not a stable positive action result. Its balanced phase pattern is
inconsistent: free-space -0.4119 (progress -0.4119), pre-contact -0.3338
(progress -0.2643 and recoverability -0.0694), in-hand +2.8822 (three 0->1
success transitions, +2.7778), and contact-near-completion -0.3154 (progress
-0.2459 and recoverability -0.2083, partly offset by fewer drops +0.1389).
The positive in-hand pocket does not survive task-level aggregation.

Stack action is robustly negative across all three weights. The main failure
is contact-near-completion: -16.0594 balanced utility, composed of a
-15.2778 success-hold contribution, -0.6428 progress, and -0.1389 from more
drops. The in-hand phase is -1.6916, mainly success -1.3889, progress
-0.3027, and recoverability -0.1389; fewer drops contribute +0.1389 there.
The free-space and pre-contact phases are near zero. No collision signal was
observed. This pattern is consistent with the fine-action schedule altering
contact/placement execution, but the raw evidence does not log action vectors,
so it cannot distinguish action discontinuity from model/action sensitivity or
other simulator-mediated effects.

### Joint effect: FF - max(FC, CF)

| weight | PegInsertionSide-v1 | StackCube-v1 |
| --- | --- | --- |
| balanced | -1.8188 [-3.4490, -0.5640], Holm p=0.0027 | -5.2429 [-8.6949, -2.3045], Holm p=0.0022 |
| success-dominant | -1.8851 [-3.6778, -0.5229], Holm p=0.0032 | -5.9791 [-10.0047, -2.5661], Holm p=0.0042 |
| progress-dominant | -1.8375 [-3.3141, -0.6780], Holm p=0.0022 | -4.5819 [-7.5145, -2.1042], Holm p=0.0018 |

This is a negative synergy result, not simply an FF-versus-CC result. On the
balanced utility, direct FF-CC is still positive in Peg free-space (+0.1874),
in-hand (+2.2082), and pre-contact (+0.3171), and in Stack free-space
(+0.1765) and pre-contact (+0.5095). Nevertheless FF falls below the better
single axis, particularly in Peg in-hand (-3.3996 relative to the better
single) and Stack contact-near-completion (-17.5979). The joint conclusion is
therefore that the two resolution changes did not combine super-additively;
they sometimes both improve over CC while still being worse than the best
single-axis treatment.

The balanced joint phase pattern makes the failure concrete:

| task / phase | joint utility delta | dominant observed components |
| --- | ---: | --- |
| Peg / free-space | -0.3577 | progress -0.3577 |
| Peg / pre-contact | -0.2543 | progress -0.1848; recoverability -0.0694 |
| Peg / in-hand | -3.3996 | success -2.7778; progress -0.6912 |
| Peg / contact-near-completion | -3.2638 | success -2.7778; progress -0.3471; recoverability -0.1389 |
| Stack / free-space | -0.5397 | progress -0.5397 |
| Stack / pre-contact | +0.2600 | only a small local positive, with progress -0.0178 and no success change |
| Stack / in-hand | -3.0941 | success -2.7778; progress -0.2469 |
| Stack / contact-near-completion | -17.5979 | success -16.6667; progress -0.8618 |

## Cross-seed and weight robustness

For balanced utility, the visual means by model seed were positive for every
task and seed: Stack `[0.0874, 2.3401, 1.4529]` and Peg
`[0.5506, 0.6224, 0.6940]`. Stack action was negative for all seeds
`[-5.7781, -3.2466, -4.0971]`; Peg action was mixed
`[-1.6832, +3.3454, -0.2964]` and all three aggregate CIs include zero.
Joint means were negative for all seeds: Stack `[-5.6453, -5.5496,
-4.5338]`, Peg `[-3.7267, -1.3525, -0.3773]`.

The fraction of state-by-seed rows with positive joint effect is not the same
as a positive mean effect. For balanced utility it is Peg
`[0.4167, 0.3750, 0.4063]` and Stack `[0.2917, 0.2500, 0.4479]`; all three
seeds clear the preregistered 0.10 fraction criterion, but negative states
have larger losses, so the task-level means and CIs remain negative. The same
qualitative fraction range holds across the other weights (aggregate fraction
about 0.3264--0.3993), while joint CIs remain below zero. Thus the local
positive-state fraction cannot reverse the task-level joint gate.

## Real compute budget

Across the positive diagnostic banks, measured estimated-FLOP cost was
92,438,200,000,000 for all-coarse and 125,120,200,000,000 for all-fine. The
coarse/full ratio is `0.7387951746`, not 0.50. Therefore:

- 25% budget = 31,280,050,000,000: even all-coarse is over budget;
- 50% budget = 62,560,100,000,000: even all-coarse is over budget;
- 75% budget = 93,840,150,000,000: all-coarse, equal-cost fixed-axis, and
  joint allocations can be compliant, but all-fine cannot.

At the 50% budget, all allocation arms refine zero states, are marked
noncompliant by the accounting recomputation, and the paired joint-versus-
fixed success difference is exactly 0.0 with CI `[0.0, 0.0]`. It is therefore
invalid to interpret a 50% budget gain. At 75%, the exploratory joint arm has
balanced success gain `+0.0121528` versus strongest fixed axis, CI
`[0.0034722, 0.0225694]`, sign-flip p `0.03330`; this is not the preregistered
50% gate and cannot upgrade the failed task screen or the final status.

## Prefix, oracle, and interpretation boundaries

The deterministic fresh-reset replay and categorical prefix agreement passed
(both reported fidelity rates are 1.0; `causal_fidelity_pass=true` on the
oracle rows). This supports the claim that the frozen reset-and-prefix replay
is reproducible under the audited backend. It does not turn the physical tile
choice into a deployable selector, and the formal conditions were evaluated by
serial fresh-reset replay rather than simultaneous condition stepping. The
prefix simulator latency is recorded separately from deployment episode
compute, as required by the protocol.

The evidence should be read in four layers:

1. **Confirmed code semantics.** Shared ResNet18 global-plus-crop branch;
   eight-step treatment; coarse cached four-step action chunks; fine per-step
   re-query; common fine continuation; repeated physical tile oracle; utility
   and budget formulas; uniform success-hold semantics.
2. **Observed paired evidence.** Visual FC-CC is positive on both diagnostic
   tasks and all three weight sets; Peg CF-CC is statistically null; Stack
   CF-CC is negative; FF-max(FC,CF) is negative on both tasks and all weights;
   the component patterns and call/latency differences above are present in
   the raw rows.
3. **Bounded mechanism inference.** Visual gains are most consistent with
   additional local crop features changing progress/recoverability and a few
   success transitions. Stack action loss is most concentrated in the
   fine-action contact/placement phase, consistent with a harmful action
   schedule/model response there. These are bounded explanations, not direct
   action-level mediation proofs.
4. **Not tested or not claimable.** A direct action-discontinuity trace,
   deployable learned crop/axis router, OOD behavior, closed-loop expansion,
   Stage-2.8, π0.5, and a healthy negative-control formal result were not
   established by this run. The failed task screen also prevents a positive
   core-mechanism claim.

## Final mechanism conclusion

The cleanest explanation of the observed result is: the added visual crop
branch supplies a small but repeatable visual marginal signal in these two
diagnostic tasks; per-step action replanning is not stable (null on Peg and
harmful on Stack); combining the two does not yield joint synergy and often
loses to the best single axis. The 50% compute claim is unavailable because
the implemented cost geometry makes it infeasible. The preregistered final
status is therefore `NO_GO_CORE_MECHANISM`, with no automatic Stage-2.8 or
closed-loop follow-up.
