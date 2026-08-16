# PAI service-record cleanup

Replacement run: `r16p18-stage26-counterfactual-20260816-v13`

Replacement Job ID: `dlc1exqlfu0iiwaa`

Replacement terminal status: `Succeeded`

After persisted formal completion and independent-audit pass were verified,
the pinned two-phase OpenAPI helper prepared, deleted, and freshly verified
the absence of these exact superseded service records:

| Run | Job ID | Prior status | Result |
|---|---|---|---|
| v3 | `dlc1wcx5vudgoi1m` | Failed | deleted; absence verified |
| v4 | `dlc1g9m496pxklyd` | Failed | deleted; absence verified |
| v5 | `dlc1bjufbe1r5re2` | Stopped | deleted; absence verified |
| v6 | `dlcumm03uyrlig0j` | Failed | deleted; absence verified |
| v7 | `dlcq6vb4a4gdaquu` | Failed | deleted; absence verified |
| v8 | `dlcqqvwgaf7b1oee` | Failed | deleted; absence verified |
| v9 | `dlcclp1calbj5zh3` | Stopped | deleted; absence verified |
| v10 | `dlc1co9nnuu5cx2f` | Stopped | deleted; absence verified |
| v12 | `dlcmcqsqyq6lc2kl` | Stopped | deleted; absence verified |

Only PAI service rows were deleted. All CPFS data, source, registry manifests,
logs, checkpoints, and formal evidence remain preserved. No browser was used.
