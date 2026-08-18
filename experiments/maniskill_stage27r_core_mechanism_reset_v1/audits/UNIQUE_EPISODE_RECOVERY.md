# Unique-episode recovery

Formal v5 stopped before screening because the exact-dataset audit found
duplicate replay retries. The v1 splitter selected the first 300 successful
replay rows by episode id, but some official motion-planning seeds had multiple
successful retry rows with identical initial state.

Observed unique `(episode_seed, initial_state)` counts were 300 for StackCube,
PlugCharger, PushT, and PushCube; 223 for PegInsertionSide; and 290 seeds / 294
states for PullCubeTool. Success flags were 300/300 for every task, so this was
an independence failure rather than a replay-success failure.

The corrected splitter scans successful rows in frozen episode-id order and
keeps the first 300 whose episode seed and initial-state hash are both new.
PegInsertionSide expands its preregistered oversized source replay from 400 to
500 official rows to provide enough unique successes. PullCubeTool reuses its
330-row oversized pool. Only these two tasks are retrained; the four already
valid task datasets/checkpoints are retained bit-for-bit in a new composite
training root. No scientific threshold or evaluation budget is changed.
