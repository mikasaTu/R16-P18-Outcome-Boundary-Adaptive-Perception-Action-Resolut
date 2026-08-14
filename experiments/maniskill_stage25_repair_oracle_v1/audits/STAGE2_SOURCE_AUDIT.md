# Stage-2 predecessor source and evidence audit

Audit target: `R16-P18-MS4-STAGE25-REPAIR-ORACLE-V1`

The predecessor evidence commit is `76e71f5eae9771b83906478f0c421183e38cdd9c`
with tree `088b74883a53f9577aacd742f4c9ac560704dad9`.  The
new branch began at exactly that commit.  The full machine-readable audit is
`../manifests/source_bindings.json`.

## Read-only predecessor evidence

The audit binds the previous final report, preregistration, README, task
selection, trainer, evaluator, baseline summarizer, independent baseline audit,
mechanism analysis, and baseline gate by size and SHA256.  It also binds every
pre-existing state-bank/oracle source file.  No predecessor file was modified.

The predecessor demo manifest SHA256 is
`d0e8a1b4ff2d26ea821d25eeb47b351baebc5176d578d7718afb7a6dbfc18d7b`.
It contains exactly 300 identities for each of PullCubeTool, PushCube, PushT,
and StackCube.  The old 100 closed-loop test seeds per task are copied into the
machine-readable audit as explicit exclusions for all new seed banks.

## Checkpoint inventory

There are 156 complete candidate payloads:

- StackCube: 6 checkpoints x 3 model seeds = 18;
- PushCube: 6 checkpoints x 3 model seeds = 18;
- PushT: 20 checkpoints x 3 model seeds = 60;
- PullCubeTool: 20 checkpoints x 3 model seeds = 60.

Every approximately 333 MB `checkpoint.pt` payload was re-hashed from CPFS in
this run, totaling about 49 GB of read evidence.  All 156 SHA256 values equal
the predecessor selection manifests.  Paths, validation losses, steps,
payload hashes, and completion markers are recorded individually in
`source_bindings.json` and the frozen `checkpoint_candidates.json`.

## Existing downstream code is not existing evidence

The old directory contains state-bank, action-atlas, visual-intervention, joint
matrix, and oracle summarization code.  The predecessor formal baseline gate
has `continue_to_oracle_probe=false`; the audited final report and CPFS audit
contain no formal state-bank/oracle result.  Accordingly, those files are
hashed as predecessor source only and are never imported or executed by this
protocol.

## Material semantic differences in Step-4

The new implementation selects checkpoints by closed-loop hold/end stability,
records four success-stopping arms, removes the ambiguous `collisions` alias,
uses disjoint expert/on-policy state sources, marks out-of-range actions invalid
instead of clipping, continues each action prefix with 20 policy and 5 neutral
steps, uses information-resolution tiles rather than destructive masks, and
implements the preregistered CC/FC/CF/FF matched oracle.

The current user instruction also changes execution control only: every planned
experiment runs even if a gate fails.  It does not change scientific thresholds
or permit a downstream diagnostic to reverse an upstream gate failure.

