# Training consistency code audit

This audit was completed before the first Stage-2.7R optimizer update. It is an
implementation-mechanism audit, not a new scientific hypothesis.

## Confound found

The original development implementation compared a fine-resolution ACT
prediction conditioned on the demonstration action sequence (a sampled
posterior latent) with a coarse-resolution inference prediction using the zero
latent. The resulting loss mixed two effects:

1. visual-resolution disagreement; and
2. the ACT posterior-versus-prior latent gap.

It also treated the first quarter of each shuffled minibatch as a free-space
surrogate. Minibatch position is unrelated to task phase, so that mask had no
valid physical interpretation. Had this version trained, it could have made a
coarse/fine gain or loss reflect latent shrinkage rather than visual
resolution.

No formal checkpoint used that implementation: PAI attempts through v12 had
not reached an optimizer update when the issue was found.

## Correction

Commit `dc7fcef880384a033ad41e045a8c0103be99de99` makes the two resolution
branches use the same action-conditioned posterior sample. Both branches
receive gradients from the consistency term. The dataset supplies an explicit
cross-task free-space stratum defined as the first 20% of each demonstration;
that boolean metadata is removed before policy forward and therefore cannot be
a learned or deployable input.

This temporal stratum is a task-independent operational proxy, not a claim that
contact onset always occurs at exactly 20% of an episode. The final mechanism
report must retain that limitation.

## Regression evidence

The Stage-2.7R test suite contains checks that:

- the first temporal quintile is the only marked free-space interval;
- the fine and coarse branches receive exactly the same latent-noise tensor;
- the consistency loss ignores non-free-space rows; and
- all pre-existing protocol, tiling, overwrite, clustering, and Holm tests
  still pass.

Result: 8 passed, 0 failed (six environment deprecation/runtime warnings).
