---
feishu_title: "step2"
feishu_url: "https://icnbwz7kd1ui.feishu.cn/wiki/GbKbwE5eai5l72kje1NcJpn8n5f"
feishu_wiki_token: "GbKbwE5eai5l72kje1NcJpn8n5f"
feishu_object_token: "ASajdohnUoB7s9xwvBqcqkyRnEc"
feishu_revision: 4
---

# step2

Continue the R16-P18 project from the archived LIBERO baseline-gate repository.



The previous LIBERO run must remain immutable. It ended with

NO_GO_BASELINE_GATE because one task saturated at 98% success and the official

exact-task datasets provided only 50 successful demonstrations without the

required original episode-seed protocol.



The next step is NOT to implement the R16-P18 adaptive method yet.



Your sole objective in this stage is:



MANISKILL3 RGB-ACT TASK SCREEN

\+

ORACLE PERCEPTION–ACTION OUTCOME-BOUNDARY EXISTENCE PROBE



==================================================

1. SCIENTIFIC SCOPE

==================================================



Answer only these questions:



1. Can we obtain enough independent, reproducible demonstrations?
2. Does the official RGB ACT baseline have useful performance headroom?
3. Do nontrivial action-outcome boundaries exist?
4. Do visual distinctions and action distinctions interact, rather than acting

independently?

Do not claim that R16-P18 is validated, accepted, rejected, or paper-ready.



Do not implement:

- a learned effect predictor,
- the final budgeted selector,
- adaptive token splitting,
- adaptive action allocation,
- Diffusion Policy,
- DINO-WM,
- π0.5.

Stop after producing the task-screen and boundary-existence report.



==================================================

1. REPOSITORY AND EXECUTION RULES

==================================================



- Preserve all existing LIBERO artifacts byte-for-byte.
- Create a new branch:

maniskill-act-boundary-screen-v1

- Add new work only under:

experiments/maniskill_act_boundary_screen_v1/

- Pin ManiSkill3 v3.0.1 to its exact Git commit.
- Pin the official ManiSkill ACT implementation and all dependencies.
- Use at most two GPUs concurrently.
- Never kill, preempt, or share another user’s GPU process.
- Use normal Git, YAML/JSON/JSONL/Parquet, and SHA256.
- Do not build another large formal-activation or publication framework.
- Freeze thresholds before observing proposed-method results.
- No HTML report in this stage.

==================================================

1. TASKS

==================================================



Smoke-only:

- PickCube-v1

Formal candidates:

- PegInsertionSide-v1
- PushT-v1
- StackCube-v1
- PushCube-v1

Roles:

- PegInsertionSide: precision insertion
- PushT: contact-rich multimodal task
- StackCube: precise placement and release
- PushCube: smooth negative control

Frozen fallback order, used only if fewer than two positive tasks pass:

1. PlugCharger-v1
2. PullCubeTool-v1
3. PokeCube-v1

A fallback may be activated only from data availability, baseline performance,

seed stability, or boundary-density criteria. Never select a fallback after

seeing the final adaptive-method result.



==================================================

1. DATA GATE

==================================================



For each formal task, obtain exactly 300 unique successful trajectories:



- 200 train
- 50 validation
- 50 test

Prefer official demonstrations. If insufficient, use official ManiSkill

motion-planning or official RL demonstration generation with new independent

environment seeds.



Requirements:



- unique trajectory ID
- unique initial-state hash
- recorded environment seed
- recorded actions and environment states
- no duplicated or augmented trajectory counted as a new episode
- exact split by trajectory identity, never by frame
- replay success >= 95%

For high-precision or dynamical tasks, replay with saved environment states.



Pin:

- raw demo hashes
- replayed RGB demo hashes
- task assets
- control mode
- simulator backend
- camera configuration
- control frequency

Stop and mark the task BLOCKED if 300 unique successful trajectories cannot be

obtained without duplication or protocol substitution.



==================================================

1. OFFICIAL RGB ACT BASELINE

==================================================



Use ManiSkill’s official ACT RGB implementation, not the previous custom

BoundaryBC-S, as the primary scientific baseline.



Use the official task-specific control modes and recommended training horizons.



Train every formal task with three model seeds:



- 16018
- 16019
- 16020

Use the same train/validation/test identities for all seeds.



Checkpoint selection:

- choose the checkpoint using validation data only
- freeze it before test evaluation
- never choose a checkpoint using test success

Evaluation:

- 100 fixed test episodes per task per model seed
- identical test seeds across methods
- record success, episode length, contact events, collisions, and latency

Positive-task gate for PegInsertionSide, PushT, StackCube:



- aggregate success in [25%, 85%]
- max seed success minus min seed success <= 25 percentage points

Negative-control gate for PushCube:



- aggregate success in [70%, 98%]

At least two positive tasks must pass.



PickCube is only an integration smoke test and must not enter the scientific

task set.



==================================================

1. BUILD A REPLAY STATE BANK

==================================================



For each task that passes the baseline gate, freeze 64 replay states:



- 16 free-space
- 16 pre-contact or pre-grasp
- 16 contact / insertion / placement
- 16 near-completion

Use a deterministic rule based on held-out trajectory phase and task predicates.



Persist:



- exact simulator state
- task and trajectory ID
- environment seed
- RGB observation
- proprioception
- language/task label if present
- ACT base action chunk
- task phase
- contact state
- task-specific progress
- success/failure and recoverability labels

Simulator-only fields may be used to produce labels and stratify states, but

must never be fed to a future deployable selector.



Verify:



restore same state + execute same action

=> same short-horizon outcome within a frozen tolerance.



==================================================

1. LOCAL ACTION OUTCOME ATLAS

==================================================



For each replay state:



1. Obtain the frozen ACT base action chunk.
2. Use training action chunks to compute two local PCA directions over the

continuous motion dimensions.

1. Construct a 5x5 residual grid around the base chunk using levels:

{-1, -0.5, 0, 0.5, 1} times the local standard deviation.

1. Keep gripper fixed for the continuous atlas.
2. Evaluate gripper open/close separately as a binary control where relevant.
3. Restore the same simulator state for every candidate.
4. Execute only the first four control steps.
5. Record:

   - task-progress delta
   - intended contact
   - unintended contact
   - collision
   - object pose delta
   - articulation or insertion progress
   - short-horizon success
   - recoverability

Define adjacent action cells as crossing an outcome boundary when they change

task-relevant outcome class or exceed the preregistered normalized

task-effect-distance threshold.



Report:

- action boundary density
- outcome alias rate
- best-action identity
- local action sensitivity

==================================================

1. VISUAL INTERVENTION ATLAS

==================================================



Divide each RGB camera into a 4x4 tile grid.



For each tile, evaluate at least:



- local low-resolution replacement
- local blur
- conditional-mean or inpaint-like replacement

Do not use zero masking as the only intervention.



Use the same ACT inference seed and all other inputs unchanged.



Record:



- action-chunk change
- action-candidate ranking change
- short-horizon outcome change
- whether the identity of the best action candidate changes

A tile is not considered important merely because visual features change.

It is important only when it changes a task-relevant action decision or

realized short-horizon outcome.



==================================================

1. JOINT VISUAL–ACTION BOUNDARY PROBE

==================================================



For each replay state:



- select the four visual tiles with the largest preliminary action effect
- select the five most competitive action candidates
- evaluate all 4x5 visual-action combinations from the identical simulator state

A state contains a joint perception-action boundary only when changing the

visual resolution of a specific region changes which local action cell produces

the valid or best physical outcome.



Compute:



- visual boundary density
- action boundary density
- joint coupling density
- full-fine oracle best-action recall
- full-fine oracle outcome regret relative to coarse uniform

This is an oracle existence probe. It is not a deployable method.



==================================================

1. PREREGISTERED GO / NO-GO

==================================================



GO to the next R16-P18 implementation stage only if:



1. At least two positive tasks pass the ACT baseline gate.
2. At least two positive tasks have action-boundary density >= 20%.
3. At least two positive tasks have joint visual-action coupling density >= 15%.
4. Full-fine oracle improves best-action recall by >= 10 percentage points

or reduces outcome regret by >= 15% versus coarse uniform.

1. PushCube joint-boundary density is <= 10%.
2. The qualitative direction is reproduced by at least two of three ACT seeds.
3. Data, policy calls, action candidates, and simulator opportunities are

fully accounted.

Return NO-GO or REVISE if:



- the tasks remain saturated or nearly impossible,
- fewer than two tasks show nontrivial boundaries,
- visual-only effects explain all gains,
- action-only effects explain all gains,
- joint coupling is not reproducible,
- the result requires simulator privileged inputs,
- the result depends on extra action candidates or extra model calls.

==================================================

1. DELIVERABLES

==================================================



Create:



experiments/maniskill_act_boundary_screen_v1/

  preregistration.yaml

  environment_lock.json

  data_manifest.jsonl

  baseline/

  state_bank/

  action_atlas/

  visual_atlas/

  joint_probe/

  task_selection.json

  stage_report.md

  stage_summary.json



The final response must include:



1. exact commits and environment
2. data counts and replay success
3. baseline success by task and model seed
4. seed-stability analysis
5. state-bank composition
6. action, visual, and joint boundary-density tables
7. representative boundary and non-boundary examples
8. explicit GO / NO-GO / REVISE decision
9. selected tasks for the next stage
10. exact reason for every excluded task

Stop after this report.

Do not automatically implement the adaptive selector.
