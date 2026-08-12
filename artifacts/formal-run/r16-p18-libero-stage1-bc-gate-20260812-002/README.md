# No formal log artifacts

PAI JobId `dlc1eloj62mdzw2y` failed before the first optimizer step because the container
started in `/root` and the privilege-dropped process could not restore that working directory.
Consequently this run produced no experiment log artifacts in the formal log root. Its full
submission, failure, pod-log, polling, and deletion evidence is preserved under
`artifacts/pai-registry/runs/r16-p18-libero-stage1-bc-gate-20260812-002/`.
