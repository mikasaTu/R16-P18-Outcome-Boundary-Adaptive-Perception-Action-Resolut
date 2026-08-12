# Artifact inventory

This repository intentionally archives every R16-P18 artifact produced by the baseline-gate
work, including unsuccessful preflight/job attempts. Runtime caches and upstream input data
are not experiment outputs and are represented by immutable hashes instead.

## Source and protocol

- `boundarybc/`: complete experiment implementation.
- `configs/r16_p18_libero_stage1.yaml`: frozen formal configuration.
- `configs/dev14_smoke_evidence.yaml`: final smoke configuration.
- `scripts/pai_r16_p18_bootstrap.sh`: root-to-owner PAI bootstrap.
- `scripts/pai_r16_p18_leon.sh`: owner-side formal launcher.
- `tests/test_stage1_contract.py`: source contract tests.
- `provenance/source-commits/`: patches for original commits `2ecad26`, `80caf46`, and
  `20d65a9`.

The root import commit has tree `b87468210362a81a5d3754630f1ff804f79577e4`, identical
to the frozen source tree used on PAI.

## Development smoke

- `artifacts/dev-smoke/dev14-smoke-20260812-v4/smoke_result.json`
- `artifacts/dev-smoke/dev14-smoke-20260812-v5/smoke_result.json`

V5 is the final accepted smoke. V4 is retained because the request was to archive all testing
outcomes, including superseded development evidence.

## Formal run

`artifacts/formal-run/r16-p18-libero-stage1-bc-gate-20260812-003/` contains:

- 9 complete 3000-step training JSONL streams;
- 9 evaluation JSONL streams containing all 450 episode records;
- 9 first-rollout records and 9 evaluation summaries;
- `launcher_preflight.json` and `run_manifest.json`;
- `reports/baseline_gate.json` and `reports/baseline_gate.md`;
- raw W&B run bundles, logs, configs, metadata, and summaries under `wandb-raw/`.

The empty/superseded formal-log directory for run `...-002` is retained beside it.

## Checkpoints

`artifacts/checkpoints/r16-p18-libero-stage1-bc-gate-20260812-003/models/` contains
all 9 task/seed model directories. Each directory contains:

- `checkpoint_step_00002500.pt`;
- `checkpoint_step_00002750.pt`;
- `checkpoint_step_00003000.pt`;
- `final.pt`;
- completion markers, `FIRST_WORK.json`, and `training_summary.json`.

There are 36 `.pt` paths (approximately 1.4 GB in the working tree). They are tracked through
Git LFS. `final.pt` and step-3000 may share identical content; both original paths are retained
for an exact artifact archive.

## PAI registry and audit evidence

`artifacts/pai-registry/final-files/` contains the exact final versions of the generic
`pai-job` tool portions and R16-specific launcher/template/test required to describe the job.

`artifacts/pai-registry/runs/` contains the complete registry evidence for:

- `...-001`: local preflight only, no PAI job;
- `...-002`: failed PAI job `dlc1eloj62mdzw2y`, plus deletion audit;
- `...-003`: successful PAI job `dlcnouq6igkhfyub`, all polling snapshots, readbacks,
  stdout/stderr, pod log, contract checks, and artifact checks.

`provenance/registry-commits/` stores patches for registry commits `d708f0fe` and `0bf05ce0`.
The registry contained unrelated experiments, so only the R16-specific files, exact patches,
and R16 run directories were exported.

## Report source

- `docs/EXPERIMENT_REPORT.md`: readable complete report.
- `docs/feishu-report-source/part1.xml`, `part2.xml`, `part3.xml`: exact XML payloads used
  to populate the Feishu child document “实验报告”.
- Feishu URL: <https://icnbwz7kd1ui.feishu.cn/wiki/MCkMwHrYpiQZa2kh3H2c7SWhnnd>

## Deliberate exclusions

The following are not repository artifacts:

- Three upstream demonstration HDF5 files (about 2.1 GB total). Their exact byte counts and
  SHA256 values are recorded in the manifest and report.
- The upstream LIBERO asset tree (422 MB). Its 585-file tree-manifest SHA256 is recorded.
- Numba bytecode caches, NVIDIA GL caches, pip caches, and W&B cache directories. These are
  machine-generated caches, not experimental results.
- Credentials. PAI request evidence retains only redacted placeholders; no W&B API key,
  GitHub key, cloud secret, or access token is intentionally archived.

No adaptive-stage files are absent by accident: they were never implemented because the
baseline gate returned `NO_GO_BASELINE_GATE`.
