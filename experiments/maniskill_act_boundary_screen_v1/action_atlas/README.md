Local PCA action-atlas manifests and aggregate boundary evidence are stored
here. Large per-state rollouts remain on pinned CPFS paths.

`oracle_implementation_contract.json` freezes the 256-neighbor standardized
local PCA, 5x5 action grid, outcome utility, 4x4-by-three visual interventions,
4x5 joint probe, tie breaks, and paired bootstrap before formal baseline
results are available. `scripts/evaluate_oracle_atlas.py` is restart-safe at
the frozen state identity: every completed per-state surface is atomically
persisted and validated before it is skipped on resume. The oracle uses
simulator labels and is explicitly not a deployable selector.
