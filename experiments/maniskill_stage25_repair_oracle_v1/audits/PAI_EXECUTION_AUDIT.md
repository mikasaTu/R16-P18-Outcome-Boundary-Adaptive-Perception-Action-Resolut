# PAI execution audit

This ledger separates infrastructure attempts from the one scientific run that
is eligible for the final Step-4 result.  Failed or preempted namespaces are
never merged into the final namespace.

| Attempt | JobId | Resource | Terminal/current state | Scientific eligibility |
|---|---|---|---|---|
| v1-v2 | none | preflight only | sealed before CreateJob | none |
| v3 | `dlc16bnpxmgav2jh` | exp-efficiency, 8 A800 | stopped while queued | none |
| v4 | `dlc1omzqbu6bme50` | exp-robot, 8 A800 | stopped while queued | none |
| v5 | `dlc111znd6ybbhst` | idle, 8 A800 | stopped while queued | none |
| v6-v7 | none | 2-A800 preflight | rejected by the then-missing task-specific whitelist/provenance contract | none |
| v8 | `dlc1j3ant64zlkyf` | idle, 8 A800 | failed before science: stale freeze-hash binding | none |
| v9 | `dlchcwowaaqdvehb` | idle, 8 A800 | failed before science: inaccessible SAPIEN working directory | none |
| v10 | none | preflight only | no CreateJob: required resume root absent | none |
| v11 | `dlc1w4x72ioid97u` | idle, 8 A800 | stopped after repeated platform preemptions | none; partial semantics archived read-only |
| v12 | `dlc12gwqh6fe336f` | exp-efficiency, 8 A800 | stopped by user while still queued; zero output files | none |
| v13 | none | exp-efficiency, 2 A800 | sealed preflight failure: required resume root absent | none |
| v14 | `dlc1mqa496zo1br4` | exp-efficiency, 2 A800 | failed before payload evidence was persisted | none |
| v15 | `dlc1275ki270m0dd` | exp-efficiency, 2 A800 | failed before science; diagnostic exposed a copied manifest-file hash error | none |
| v16 | `dlcec5w5mn3kk7ut` | exp-efficiency, 2 A800 | failed at the first two environment initializations because the safe work-directory change was absent | none; zero completed episodes |
| v17 | `dlc1nac45mnsov3m` | exp-efficiency, 2 A800 | failed after state-bank construction: restoration audit requested three CPU environments, which ManiSkill rejects | none; zero restoration rows and no downstream oracle output |
| v18 | `dlct241e2lcsom1a` | exp-efficiency, 2 A800 | intentionally stopped after static audit found the same CPU-vectorization error in the not-yet-reached atlas path | none; first two screen jobs incomplete and zero summaries |
| v19 | `dlc11nu9e2z0hrl7` | exp-efficiency, 2 A800 | intentionally stopped after the first CUDA atlas smoke was found to pass vacuously with 0/25 valid candidates | none; smoke rejected and formal screen partials not reused |
| v20 | `dlc16dq616en4mnc` | exp-efficiency, 2 A800 | failed closed in the tightened pre-formal atlas smoke because the real selected-checkpoint/state candidate validity was below 90% | none; formal orchestrator never started |
| v21 | `dlca742ruakkxqj7` | exp-efficiency, 2 A800 | failed closed after persisting the atlas diagnostic: 25/25 arm candidates legal, but all inherited the raw ACT gripper overshoot | none; diagnostic-only smoke, formal orchestrator never started |
| v22 | none | exp-efficiency, 2 A800 | sealed before CreateJob because the resume result root had not been pre-created | none |
| v23 | none | exp-efficiency, 2 A800 | sealed before CreateJob because the fresh artifact root had been pre-created instead of being registry-owned | none |
| v24 | `dlcugeneyssp0lsg` | exp-efficiency, 2 A800 | CUDA atlas smoke passed 25/25; intentionally stopped during checkpoint screen after static audit found a guaranteed later batched-action-space scalar conversion error | none; six screen summaries and all partials rejected, smoke retained as non-scientific implementation evidence |
| v25 | `dlctwi9sawp9cixz` | exp-efficiency, 2 A800 | CUDA atlas smoke passed 25/25; intentionally stopped during checkpoint screen when a complete pre-confirmatory audit found downstream radius-indexing and joint-control/statistical semantic defects | none; all screen partials rejected and no calibration, confirmatory, state-bank, or oracle evidence reused |

## v11 preserved partials

- Lease-2 archive manifest SHA-256:
  `1dee57e0df870a3b470cc4f8ed19e3778385014c4c6e0f723f0dc5e483c49e36`.
- Lease-4 archive manifest SHA-256:
  `7a79707051250ee508a13844103e1ae98b0dffa3ef589019ab9a7005cfc1ac9c`.
- Migration manifest SHA-256:
  `2416094e75bfaff4922d2a83026b9aec9463627f2dd4e0543bc75d2d82e74c9d`.
- No raw file from v11 is copied, merged, or used to skip work in v17.

## Dedicated two-GPU operational amendment

The frozen protocol declares an inclusive A800 hardware range of 2-8 GPUs, and
the frozen Python orchestrator accepts `--gpu-count` in that same range.  The
original shell entry point additionally asserted eight GPUs.  To backfill the
fragmented dedicated queue without modifying any scientific file, the PAI
registry uses a new operational entry point that:

1. verifies source commit `72d3606fcbe46b3356727ae45dc29a3005435af8`
   and tree `19d641ed9dd0aa097e322a8c61dec1f4c525c988`;
2. verifies the frozen launcher, protocol freeze, the scientific checksum file,
   and every checksum listed in that file;
3. requires exactly two NVIDIA A800 GPUs and UID/GID 2254:2254;
4. disables W&B and invokes the same frozen `run_stage25_formal.py` with the
   same manifests, demonstrations, thresholds, and output semantics;
5. writes only to a fresh per-run namespace.

The authoritative SHA-256 of the scientific checksum file is
`8a606d77d2e70ca6147943c01ba1dc7a8f713852f486698b7998c50c364b5e4f`.
The earlier copied value used in v15 was not the hash of the frozen file; v15's
fail-closed check caught it before any scientific work.

This amendment changes scheduling parallelism only.  It is not a threshold,
seed, task, checkpoint, metric, policy, or simulator change.

## v17 restoration implementation failure

PAI reports v17 as `Failed` (`JobFailed`) after 2,749 seconds, with running time
2026-08-14 08:53:32Z through 09:38:28Z.  The formal runtime log SHA-256 is
`9bd9feb964c29000615276ab2de43bae760cdb70a5a252c9135679d4951b4d4f`.
The direct restoration subprocess log SHA-256 is
`3ce3cd94576ff004d42aa2f81d08cdaa137447f87b99c7356f1f6e3ae71d1520`.
The failure occurred during environment construction, before a raw restoration
row or summary was written.  Consequently v17 is retained as failed audit
evidence and is not an eligible source for the final Step-4 result.

## v21 atlas diagnostic failure

PAI reports v21 as `Failed` (`JobFailed`) after 127 seconds, with running time
2026-08-14 10:15:55Z through 10:17:09Z.  The diagnostic JSON records
`scientific_evidence=false`, `formal_result_reuse_allowed=false`, 0/25 valid
candidates, and zero scientific simulator/policy calls.  Its per-dimension
evidence isolates the violation to the unperturbed gripper channel; the three
arm channels have zero bound violations.  The failure is retained as
implementation-diagnostic evidence only and is not merged into a subsequent
run namespace.

## v24 successful smoke and proactive stop

The corrected CUDA smoke passed with 25/25 valid candidates, 25/25 non-null
outcomes, 2,175 scientific simulator steps inside the smoke path, and 1,500
policy calls.  Its JSON SHA-256 is
`44dd945ab9bc8027272a5e1987d6da14df10e8e269e4c6b828a4cc719a8ef6ac`.
The artifact remains explicitly `scientific_evidence=false` and is used only to
validate the implementation path.

While v24 was still screening checkpoints, static audit of the newly added
state-bank gripper bookkeeping found that a 16-environment vector action space
would return a four-element row for `env.action_space.low[-1]`, causing a later
`float(...)` failure.  The job was stopped at 10:33:11Z rather than consuming
the remaining evaluation allocation.  It had produced six StackCube seed-16018
screen summaries only.  Those files are not copied or reused.  The correction
reads the scalar gripper bounds from `env.single_action_space`; the registry's
separate fresh artifact/result directory lifecycle is also now respected.

## v25 downstream semantic hardening stop

v25 repeated the corrected CUDA atlas smoke successfully, then entered the
checkpoint screen. Before any calibration or confirmatory result existed, a
full downstream code-path audit found that visual calibration indexed the
three-radius action JSONL before filtering its frozen radius; that the primary
metric discarded its already-computed per-state strongest FC/CF control in
favor of a weaker pooled fixed axis; that phase state allocation and phase tile
selection were combined in one control; and that the frozen Holm secondary
correction and explicit success-trace/call-latency fields were incomplete.

The job was stopped at 2026-08-14 10:44:04Z. Its partial checkpoint screens
are implementation progress only and are neither copied nor resumed. The
corrections are covered by 24 local unit/static tests, including adversarial
alternating-axis and multi-radius fixtures. A subsequent formal attempt must
use a new run ID, fresh result namespace, new source commit/tree binding, and a
fresh CUDA smoke before any result is scientifically eligible.

## v26 successful formal run and monitor termination

v26 is the sole scientific source for the final Step-4 result.

- Run ID: `r16p18-stage25-oracle-20260814-v26`
- JobId: `dlc1ptg07eqpdaxy`
- exact source commit: `7be99cd867d27372cfc75095d459b782c0f75a66`
- exact source tree: `31df45e1f28bbba57aebcd804e13c88fe14a0784`
- resource: dedicated `exp-efficiency`, one worker, 2 A800, 12 CPU,
  200Gi memory and 200Gi shared memory
- runtime identity: `2254:2254`
- AIMaster / automatic fault tolerance: disabled
- PAI running interval: 2026-08-14 11:05:53Z to 12:49:58Z
- PAI duration: 6,294 seconds
- terminal status: `Succeeded / JobSucceeded`
- final exact-job readback SHA-256:
  `e2fb6ce7f5f47c9d57901dbbf52785fda73ce4c889b5620f03a263d44e2a46b5`

The fresh v26 CUDA smoke passed 25/25 valid candidates and remained
`scientific_evidence=false`. The formal orchestrator then completed every
planned checkpoint, baseline, semantics, contact, restoration, action,
visual, post-success diagnostic and joint experiment. `FORMAL_COMPLETE.json`
reports `ALL_PREREGISTERED_STAGE25_EXPERIMENTS_COMPLETE`, user-mandated
downstream execution despite gates, no early stopping, and no prohibited
post-oracle work. Its SHA-256 is
`09e6dedaf0da6970764d99b7342b9e2bdb351fd171b9cc68063446947109f323`.

The independent raw-data audit passed and agreed on final status
`REVISE_STOPPING_CONFOUND`. After terminal GetJob, the registry sealed
`monitor-result.json` with `ROUTINE_MONITORING_DONE`. v26's requested and
resolved contracts contain no sealed superseded predecessor target set, so
cleanup is explicitly `EXPLICIT_ZERO_TARGET`; no historical service record
was inferred post hoc or deleted by wildcard. Exact-job CLI readback remained
available, so no browser lease or FIFO was used.
