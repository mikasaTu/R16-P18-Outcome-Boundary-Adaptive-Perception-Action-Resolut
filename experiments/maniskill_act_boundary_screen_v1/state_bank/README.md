Replay-state manifests, deterministic restoration checks, and state-bank
summaries are stored here. Large simulator states remain on pinned CPFS paths.

`phase_contract.json` freezes the held-out phase predicates, SHA256 ordering,
fixed temporal fallback, repeat count, and unchanged restoration tolerances
before formal baseline results are observed. The formal builder writes 64
states per authorized task only after the baseline gate passes. A development
backend smoke found full-state drift under PhysX CUDA but exact repeatability
under pinned PhysX CPU, so all four-step state-bank and atlas counterfactuals
use PhysX CPU; RGB rendering and ACT inference remain GPU-backed. This backend
choice changes no gate threshold and is recorded in `smoke/`.
