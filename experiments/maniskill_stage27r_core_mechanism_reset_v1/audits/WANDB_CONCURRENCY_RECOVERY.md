# W&B concurrency recovery (v14 -> v15)

## Observed failure

Training run `stage27r-data-train-v14` launched eight independent single-GPU
workers. Six workers persisted a real optimizer step. The two PlugCharger
workers failed before step 1 while W&B's child service attempted to create its
port file under a Python-managed `/tmp/tmp*` directory that no longer existed.
The exception was `ServiceStartTimeoutError`, with the child traceback ending
in `FileNotFoundError` for `port-*.txt`.

The v14 job was stopped before any checkpoint was written because its two
missing workers made the required 108-checkpoint matrix unattainable. Its log
and first-step metrics remain immutable evidence; none are reused as formal
scientific results.

## Minimal recovery

The scientific protocol, demonstrations, optimizer, model, seeds, and training
budgets are unchanged. The launcher now:

1. assigns a persistent, per-GPU/per-task/per-seed `TMPDIR`;
2. assigns a per-run `WANDB_DIR` under the checkpoint directory; and
3. increases only W&B's service-start timeout from 30 to 300 seconds.

This changes tracking-process infrastructure only. It neither catches W&B
authentication errors nor permits training to be reported complete without all
required checkpoint and completion markers.
