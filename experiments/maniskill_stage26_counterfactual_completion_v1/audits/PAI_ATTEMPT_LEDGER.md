# Stage-2.6 PAI attempt ledger

Every operational attempt is preserved. Failed attempts are not pooled with
the formal scientific result.

| Attempt | Job ID | Outcome | Scientific evidence |
|---|---|---|---|
| v1 | not created | sealed preflight; two-card resource/profile mismatch | none |
| v2 | not created | sealed preflight; create-artifact contract mismatch | none |
| v3 | `dlc1wcx5vudgoi1m` | exit 126: non-executable frozen shell invoked directly | none |
| v4 | `dlc1g9m496pxklyd` | SAPIEN working-directory permission error | zero completed shards |
| v5 | `dlc1bjufbe1r5re2` | invalid StackCube predicate key in Stage-2.6 adapter | zero completed shards |
| v6 | `dlcumm03uyrlig0j` | frozen shell bound obsolete source clone | none |
| v7 | `dlcq6vb4a4gdaquu` | same obsolete-source binding, confirmed by preflight trace | none |
| v8 | `dlcqqvwgaf7b1oee` | strict bytewise rerender hash mismatch before shard commit | six uncommitted capsules |
| v9 | `dlcclp1calbj5zh3` | queued dedicated job; stopped before execution after idle contract validation | none |
| v10 | `dlc1co9nnuu5cx2f` | completed 120 collection shards and all fidelity audits; predictor freeze hit Python `false` typo | authoritative raw collection and fidelity |
| v11 | not created | run ID sealed after registry rejected a precreated output path | none |
| v12 | `dlcmcqsqyq6lc2kl` | completed predictor and all 21 arm/seed cells; audit JSON hit NumPy `bool_` serialization | authoritative predictor and closed-loop rows |
| v13 | `dlc1exqlfu0iiwaa` | succeeded; hash-bound resume of v10/v12 evidence and repaired independent audit | formal completion and final audit |

## Restoration diagnosis

For capsule `2109903571-first_near_completion-70-813ff30e6874c863cdbb0e54`,
direct state restore had maximum serialized-state difference 2.38e-7. GPU
rerender differed by at most one RGB gray level and 4.47e-7 in the observation
state vector. After restoring the captured observation, RNG ordering, pending
ACT chunk and controller state, an exact-reference first action still produced
object errors of 0.000649746 m / 0.0212094 rad in one environment and
0.000457480 m / 0.0161017 rad in 16 environments. The best-supported bounded
diagnosis is omitted PhysX contact-solver warm-start/cache state. The formal
64-state × 10-step audits remain authoritative and fail without relaxed limits.

## Final bindings

- Formal source commit entering v13: `06e729a0fe8e532ac0aa57cd5c242244f8ea2b7e`
- Source tree: `188596525b6e5f157fa2126947f30a9f686469f8`
- Protocol freeze SHA256: `acab0372362d441a5b5b3a432cb4858289ea8a527cf1f1affc1f4586e0a8a220`
- Scientific sums-file SHA256: `8622ec2f8b42a28a1e17cdf1dfd6dd34c27719a3142f8db85e310387666c8a86`
- v13 operational payload SHA256: `35533a6742bd500727361063d12cfd8dd8eb8fc284613b985ada7ed6d0fe10be`
