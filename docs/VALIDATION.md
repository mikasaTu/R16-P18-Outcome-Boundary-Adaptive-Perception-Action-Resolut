# Release validation

The following checks were executed before the GitHub publication.

## Archived-result recomputation

```bash
python scripts/verify_archived_results.py
```

Result: PASS. The script read all 9 evaluation JSONL files, required unique episode IDs
0–49 for each model, reran the fixed 10,000-replicate paired bootstrap, and exactly matched:

- 450 episodes and 400 successes;
- 17,649 policy calls and 69,924 executed steps;
- all per-seed and aggregate success rates;
- all three archived confidence intervals;
- decision `NO_GO_BASELINE_GATE`.

## Source contract

Executed with the pinned Python 3.11/CUDA 12.4 environment and runtime overlay:

```bash
PYTHONPATH=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/r22p10-libero-pai-overlay/site-packages:$PWD \
  /mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python \
  -m pytest -q tests/test_stage1_contract.py
```

Result: **5 passed**.

## PAI registry contract

The archived registry files retain their fail-closed absolute path contract. The original
registry checkout at commit `0bf05ce0` was tested with:

```bash
/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python \
  -m pytest -q tests/test_r16p18_libero_stage1_contract.py
```

Result: **6 passed, 4 subtests passed**. Running the copied test directly inside
`artifacts/pai-registry/final-files/` is expected to reject the relocated `command_file`;
the archive does not weaken or rewrite that absolute-path safety check.

## Credential sanitation

PAI OpenAPI responses were scanned before staging. Fifty-seven polling/readback files that
contained an injected W&B credential were mechanically sanitized to `<redacted>`. A second
scan found no provider-form W&B key, GitHub token, AWS/Alibaba key prefix, Hugging Face token,
OpenAI key, bearer token, Slack token, or PEM private key outside synthetic/upstream data.
See `provenance/REDACTIONS.md`.

## Checkpoint inventory

- 36 checkpoint paths.
- 27 unique SHA256 payloads; each `final.pt` is intentionally identical to its corresponding
  step-3000 checkpoint.
- Every `.pt` path is Git LFS managed.
- Exact payload hashes are in `artifacts/checkpoints/SHA256SUMS`.

## Whole-release integrity

`provenance/SHA256SUMS` covers every release file other than itself. After `git lfs pull`:

```bash
sha256sum -c provenance/SHA256SUMS
```

must complete without a mismatch.
