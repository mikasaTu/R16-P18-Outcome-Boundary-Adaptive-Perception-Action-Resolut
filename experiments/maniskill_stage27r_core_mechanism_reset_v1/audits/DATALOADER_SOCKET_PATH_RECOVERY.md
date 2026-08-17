# DataLoader Unix-socket path recovery

## Scope

This is an infrastructure-only recovery for PAI run
`stage27r-data-train-v18` / `dlc2kxj4a3slahr2`. It does not change the dataset,
model, optimizer, seeds, iteration count, checkpoint schedule, or preregistered
decision rules.

## Failure evidence

After enabling eight DataLoader workers, every training process reached dataset
loading and then emitted `OSError: AF_UNIX path too long` from Python's
`multiprocessing.resource_sharer`. The per-run `TMPDIR` was nested below the
long CPFS artifact path. Python appends a random resource-sharer socket name,
which exceeded Linux's Unix-domain socket pathname limit.

## Repair

Each GPU worker now creates and retains a short job-local directory:
`/tmp/r27r-gpu-<gpu>`. Trainers scheduled sequentially on a GPU reuse that
directory. Concurrent GPU workers remain isolated. W&B files and all scientific
outputs continue to use their existing per-run CPFS directories.

## Acceptance checks

The replacement run must show all eight initial tracked losses matching the
accepted single-worker run, no resource-sharer errors, and materially lower
time to optimizer step 200 before the full training run is accepted.
