#!/usr/bin/env python3
"""Non-scientific end-to-end smoke for the CUDA physical-action atlas path."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from action_runtime import PADDED_ENVS, generate_atlas, load_training_chunks
from common import PROTOCOL_ID, sha256_file, write_json
from stage25_runtime import (
    load_policy_from_checkpoint,
    make_env,
    state_to_numpy,
)
from state_bank_common import state_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--training-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def smoke_candidate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        row
        for row in payload["candidates"]
        if row["task_id"] == "StackCube-v1"
        and int(row["model_seed"]) == 16018
        and int(row["step"]) == 5000
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one pinned smoke checkpoint, found {len(matches)}")
    return matches[0]


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA atlas smoke requires a visible GPU")

    candidate = smoke_candidate(args.candidate_manifest)
    training_chunks = load_training_chunks(str(args.training_h5))
    device = torch.device("cuda")
    rollout_env = make_env(
        "StackCube-v1",
        PADDED_ENVS,
        sim_backend="physx_cuda",
        reconfiguration_freq=0,
    )
    policy_env = make_env(
        "StackCube-v1", 1, sim_backend="physx_cuda", reconfiguration_freq=0
    )
    started = time.time()
    try:
        agent, _ = load_policy_from_checkpoint(
            rollout_env,
            "StackCube-v1",
            16018,
            Path(candidate["checkpoint_path"]),
            device,
            candidate["checkpoint_sha256"],
        )
        policy_env.reset(seed=[160180005])
        state = state_index(
            state_to_numpy(policy_env.base_env.get_state_dict()), 0
        )
        atlas = generate_atlas(
            policy_env,
            rollout_env,
            agent,
            state,
            160180005,
            training_chunks,
            device,
            radius=0.5,
        )
    finally:
        policy_env.close()
        rollout_env.close()

    valid = np.asarray(atlas["valid"], dtype=bool)
    non_null_outcomes = sum(
        outcome is not None for outcome in atlas["outcomes"]
    )
    if len(valid) != 25 or int(valid.sum()) != non_null_outcomes:
        raise RuntimeError("CUDA atlas smoke produced inconsistent outcomes")
    write_json(
        args.output,
        {
            "protocol_id": PROTOCOL_ID,
            "status": "CUDA_ACTION_ATLAS_SMOKE_PASS",
            "scientific_evidence": False,
            "formal_result_reuse_allowed": False,
            "task_id": "StackCube-v1",
            "model_seed": 16018,
            "checkpoint_step": 5000,
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
            "training_h5_sha256": sha256_file(args.training_h5),
            "sim_backend": "physx_cuda",
            "rollout_envs": PADDED_ENVS,
            "candidate_opportunities": 25,
            "candidate_repeats": 3,
            "valid_candidates": int(valid.sum()),
            "non_null_outcomes": non_null_outcomes,
            "boundary_by_threshold": atlas["boundary_by_threshold"],
            "accounting": atlas["accounting"],
            "wall_seconds": time.time() - started,
        },
    )


if __name__ == "__main__":
    main()
