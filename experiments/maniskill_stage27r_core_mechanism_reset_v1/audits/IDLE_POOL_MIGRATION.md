# Formal evaluation idle-pool migration

## Scope

The formal Stage-2.7R evaluation was moved from the robot resource pool to the
idle A800 pool at the user's request. This is an infrastructure migration only:
the frozen source commit, training root, formal result root, protocol, seeds,
state banks, treatment definitions, and statistical rules are unchanged.

## Atomic handoff

- Previous job: `dlc2otzwqai79kw3` (`exp-robot`, 8 x A800).
- The previous job was stopped after all six 2x2 calibration shards had been
  atomically committed and before any 4x4 shard existed.
- `atomic_json` writes each shard only after it is complete. A filesystem audit
  found no temporary or partial shard at handoff.
- Replacement job: `dlc9nkd8q7u4szm3` (`stage27r-formal-idle-v9`).
- Result root is shared and resumable; completed shards are checked before work
  starts, so the replacement skips rather than overwrites them.

## Read-back contract

PAI `GetJob` reported the replacement job as `Running` with:

- resource id `quotaewyznuc7b9l`;
- 8 GPUs, 92 CPUs, and 1600 GiB memory;
- `OversoldType=AcceptQuotaOverSold` (idle-resource admission);
- a running AIMaster pod; and
- error-monitoring arguments
  `--job-execution-mode=Sync --enable-job-restart=True
  --max-num-of-job-restart=50 --fault-tolerant-policy=OnFailure`.

The API's separate `ElasticSpec.EnableAIMaster=false` field is not the
error-monitoring contract. The effective contract is visible in
`Settings.EnableErrorMonitoringInAIMaster=true`, `Settings.ErrorMonitoringArgs`,
and the running `aimaster` pod.

## Claim boundary

This migration does not add scientific evidence and cannot repair a failed
task gate. Its only purpose is to preserve and finish the preregistered oracle
evaluation using idle capacity without concurrent writers.
