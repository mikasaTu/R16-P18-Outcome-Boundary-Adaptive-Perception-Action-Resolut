# Stage-2.7R formal evidence archive

This directory is a GitHub-safe, byte-checked archive of the final
Stage-2.7R oracle run and independent verification outputs.  The source
formal root is the CPFS path recorded in
`manifests/formal_evidence_manifest.json`.

The six raw confirmatory oracle shards are intentionally **not** copied into
GitHub: together they are more than 428 MiB and individual files exceed the
repository's 50 MiB archive policy.  Calibration raw shards are also kept on
CPFS.  This is not silent omission.  The evidence manifest records, for every
formal-root file, its exact relative path, byte count, SHA-256, role, and
whether it was archived here.  It also records the six oracle summaries (task,
model seed, rows, bytes, status, and hash), so the raw shards can be verified
against the CPFS source without using Git LFS.

The archived JSON outputs, screen/task-selection record, crop-grid freezes,
state-bank records, and final-verifier-v13 logs are covered by
`SHA256SUMS`.  The malformed predecessor
`CONTINUATION_V11_TERMINAL.json` is preserved only by hash and diagnosis in
the manifest; the repaired `CONTINUATION_V11_TERMINAL_VALID.json` is the
archived terminal attestation.

`statistics.json` is the complete paired-statistics output (about 29.7 MiB),
not a sampled or shortened result.  No checkpoint, demonstration, calibration
raw shard, or oracle raw shard is included in this GitHub archive.
