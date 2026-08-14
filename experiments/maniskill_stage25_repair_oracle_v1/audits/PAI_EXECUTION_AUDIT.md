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
