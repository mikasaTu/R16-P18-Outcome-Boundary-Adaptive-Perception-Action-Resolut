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

## Accepted replacement

PAI run `stage27r-data-train-v19` / `dlc12njk0ax98xec` passed the early
acceptance checks. All eight first-step losses exactly matched v17:

- StackCube seeds 16018/16019/16020: 80.3175049, 101.5326462, 65.9923096
- PegInsertionSide seeds 16018/16019/16020: 59.2976074, 78.5486679, 88.0450439
- PlugCharger seeds 16018/16019: 61.3166542, 79.2542191

The first observed StackCube step-200 wall time fell from 4444.99 seconds in
v17 to 42.57 seconds in v19 (104.4x faster), with eight separate W&B run files
and no resource-sharer, traceback, or W&B error at acceptance time. The full
30,000-step schedule remains unchanged and continues under the same JobId.
