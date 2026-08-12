# Publication redactions

Before publication, 57 PAI OpenAPI readback/polling files contained the injected W&B API
key in the `EnvVars` response. Every value matching the provider-specific `wandb_v1_*`
credential form was replaced with the literal string `<redacted>`.

The affected files are confined to:

- `artifacts/pai-registry/runs/r16-p18-libero-stage1-bc-gate-20260812-002/`
- `artifacts/pai-registry/runs/r16-p18-libero-stage1-bc-gate-20260812-003/`

No structural fields, timestamps, job IDs, resources, status transitions, commands, metrics,
or experiment results were removed. The original files remain on the private CPFS evidence
store; this GitHub archive intentionally contains no usable credential.
