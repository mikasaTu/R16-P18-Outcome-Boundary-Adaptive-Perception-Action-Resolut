# DataLoader throughput recovery (v17 -> v18)

## Evidence from v17

All eight v17 workers started successfully after the W&B isolation repair, and
their step-1 losses exactly matched the earlier attempt. Decoding the local W&B
event stream showed that StackCube seed 16018 reached only optimizer step 200
after 4,445 seconds (22.2 seconds/update). During the same interval A800 power
was close to idle. No 5,000-step complete-state checkpoint existed.

The pinned trainer's CLI default is `num_workers=0`, so every batch of 256
in-memory RGB samples was gathered and collated serially in the training
process. With eight independent models, the available 92 CPU cores were mostly
unused while the GPUs waited for host batches.

## Bounded recovery

Run v17 was stopped before any checkpoint. Run v18 changes only
`--num-workers` from 0 to 8. It retains the exact dataset tensors,
`DeterministicResumeBatchSampler`, batch size, model, optimizer, seeds, RNG
rules, update count, and checkpoint cadence. PyTorch DataLoader returns batches
in sampler order; this dataset has no random transform in `__getitem__`.

The recovery is accepted only if all eight step-1 losses reproduce v17 and the
decoded step-200 elapsed time improves materially. Otherwise the change is not
treated as validated and the scientific run is not advanced.
