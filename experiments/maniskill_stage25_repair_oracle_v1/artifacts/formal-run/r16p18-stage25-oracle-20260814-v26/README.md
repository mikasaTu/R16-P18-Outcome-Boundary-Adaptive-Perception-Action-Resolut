# Sealed PAI v26 evidence bundle

This directory contains the complete evidence package for PAI run
`r16p18-stage25-oracle-20260814-v26` (job `dlc1ptg07eqpdaxy`). The job reached
`Succeeded / JobSucceeded`; the scientific source was frozen at commit
`7be99cd867d27372cfc75095d459b782c0f75a66`, tree
`31df45e1f28bbba57aebcd804e13c88fe14a0784`.

Contents:

- `r16p18-stage25-v26-formal-results.tar.gz`: every formal result, raw row,
  summary, manifest, progress marker, and audit file (384 files; 529 archive
  entries including directories).
- `r16p18-stage25-v26-pai-logs.tar.gz`: runtime log and CUDA action-atlas smoke
  result; disposable XDG/Matplotlib cache is intentionally excluded.
- `r16p18-stage25-v26-pai-registry.tar.gz`: exact submission, resolved
  contract, source/resource readbacks, terminal GetJob record, payload, and
  monitoring result.
- `FORMAL_RESULTS_RAW_SHA256SUMS.txt`: per-file hashes relative to the extracted
  formal-result archive root.
- `key-results/`: browseable copies of the final summaries, freeze records, and
  independent audits.
- `SHA256SUMS`: hashes for every repository artifact in this bundle.

Verification:

```bash
sha256sum -c SHA256SUMS
gzip -t r16p18-stage25-v26-*.tar.gz
tmpdir="$(mktemp -d)"
tar -xzf r16p18-stage25-v26-formal-results.tar.gz -C "$tmpdir"
(cd "$tmpdir" && sha256sum -c "$OLDPWD/FORMAL_RESULTS_RAW_SHA256SUMS.txt")
```

The archive was rebuilt only after the PAI job was terminal and then extracted
in a fresh directory; all 384 raw-file hashes passed. A credential-pattern
scan found no secret values. PAI JSON contains only an empty
`ImageConfig.Password` field.

The scientific decision is `REVISE_STOPPING_CONFOUND`. This bundle does not
contain a learned predictor, deployable selector, OOD run, Stage-3 experiment,
Diffusion Policy, DINO-WM, pi0.5, or real-robot result.
