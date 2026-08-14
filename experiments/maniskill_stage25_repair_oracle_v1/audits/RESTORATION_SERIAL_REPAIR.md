# State-restoration serial execution repair

Protocol: `R16-P18-MS4-STAGE25-REPAIR-ORACLE-V1`

## Failure boundary

Formal run v17 used source commit
`72d3606fcbe46b3356727ae45dc29a3005435af8` (tree
`19d641ed9dd0aa097e322a8c61dec1f4c525c988`).  Its restoration script SHA-256
was `13ce3f457b96171d240eb02d277ef37763d3aacf90d411f581bcc232a54a2dea`.
The script called `make_env(..., num_envs=3, sim_backend="physx_cpu")`.
ManiSkill 3.0.1 rejects CPU simulation with more than one environment, so the
subprocess failed before reading a state and before writing either
`state_restoration_raw.jsonl` or `state_restoration_audit.json`.

## Code-path correction

The corrected script SHA-256 is
`387af8b8d7a624b2fe3a15627d9fa48b5042735a22a52a464f843db6023743a0`.
It creates exactly one PhysX CPU environment.  For each frozen state it runs
three serial repeats, and each repeat independently performs reset, exact state
restore, the same four-step frozen action, and final state/categorical capture.
Only then are the three results compared.  This implements the already
preregistered phrase “CPU serial exact restore, repeats=3”; it does not alter
the state banks, source episode seeds, action, replay length, thresholds,
categorical definitions, or pass criteria.

Because scientific code changed, no v17 output will be resumed into the final
namespace.  The entire formal pipeline is rerun in a fresh namespace from a
clean commit containing this correction.

## Verification before resubmission

- Unit suite: 10/10 passed.
- Real simulator smoke: one state from each of the calibration,
  confirmatory, and post-success banks; all three serial repeats per state.
- Restoration pass rate: 3/3 = 1.0.
- Same-action categorical agreement: 3/3 = 1.0.
- Maximum initial restore error observed: `1.4901161193847656e-08`.
- Maximum four-step final-state difference between repeats: `0.0`.
- Smoke raw evidence SHA-256:
  `d76f864b3d3d931a000ff275cd7d30b191da0bbc1cc3a91b0c6ff636d1bd0f43`.
- Smoke summary SHA-256 from the generated temporary output:
  `1bf57bb6ebb8cdc32a7199a2364587847f57b440f6ebbab0fbb3cbddd63619de`.

This smoke only verifies the repaired path.  The formal gate remains determined
by all 112 frozen states in the new PAI run.
