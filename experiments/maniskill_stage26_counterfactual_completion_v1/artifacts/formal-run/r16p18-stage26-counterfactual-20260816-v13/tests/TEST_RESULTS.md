# Verification results

- Runtime: pinned `libero_sft` Python environment plus the Stage-2.6
  ManiSkill/ACT overlay used by PAI.
- `pytest -q experiments/maniskill_stage26_counterfactual_completion_v1/tests`:
  **13 passed in 6.89s**.
- `sha256sum -c manifests/SCIENTIFIC_SHA256SUMS`: every frozen scientific
  source passed.
- Independent raw audit: `INDEPENDENT_STAGE26_AUDIT_PASS`.
- PAI formal completion: `ALL_PREREGISTERED_STAGE26_EXPERIMENTS_COMPLETE`.
