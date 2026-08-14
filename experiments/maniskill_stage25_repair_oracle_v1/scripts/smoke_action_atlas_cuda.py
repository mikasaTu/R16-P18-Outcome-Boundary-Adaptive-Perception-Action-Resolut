#!/usr/bin/env python3
"""Non-scientific end-to-end smoke for the CUDA physical-action atlas path."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from action_runtime import PADDED_ENVS, generate_atlas, load_training_chunks
from common import PROTOCOL_ID, sha256_file, write_json
from stage25_runtime import (
    load_policy_from_checkpoint,
    make_env,
)
from state_bank_common import h5_full


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-checkpoints", type=Path, required=True)
    parser.add_argument("--state-bank-manifest", type=Path, required=True)
    parser.add_argument("--training-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def smoke_candidate(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["groups"]["StackCube-v1/seed_16018"]["selected"]


def validate_smoke_atlas(atlas: dict) -> tuple[np.ndarray, int]:
    valid = np.asarray(atlas["valid"], dtype=bool)
    non_null_outcomes = sum(outcome is not None for outcome in atlas["outcomes"])
    if (
        len(valid) != 25
        or int(valid.sum()) != non_null_outcomes
        or float(valid.mean()) < 0.90
    ):
        raise RuntimeError("CUDA atlas smoke produced inconsistent outcomes")
    return valid, non_null_outcomes


def atlas_diagnostics(
    atlas: dict, action_low: np.ndarray, action_high: np.ndarray
) -> dict:
    candidates = np.asarray(atlas["candidates"], dtype=np.float64)
    low = np.broadcast_to(np.asarray(action_low), candidates.shape)
    high = np.broadcast_to(np.asarray(action_high), candidates.shape)
    nominal = np.asarray(atlas["nominal_action_first4"], dtype=np.float64)
    nominal_low = np.broadcast_to(np.asarray(action_low), nominal.shape)
    nominal_high = np.broadcast_to(np.asarray(action_high), nominal.shape)
    return {
        "nominal_action_first4": nominal.astype(float).tolist(),
        "atlas_center_action_first4": atlas["atlas_center_action_first4"],
        "atlas_gripper_command": atlas["atlas_gripper_command"],
        "action_space_low": np.asarray(action_low).astype(float).tolist(),
        "action_space_high": np.asarray(action_high).astype(float).tolist(),
        "nominal_below_low": (nominal < nominal_low).astype(int).tolist(),
        "nominal_above_high": (nominal > nominal_high).astype(int).tolist(),
        "candidate_below_low_count_by_step_dimension": np.sum(
            candidates < low, axis=0
        ).astype(int).tolist(),
        "candidate_above_high_count_by_step_dimension": np.sum(
            candidates > high, axis=0
        ).astype(int).tolist(),
        "candidate_min_by_step_dimension": np.min(candidates, axis=0)
        .astype(float)
        .tolist(),
        "candidate_max_by_step_dimension": np.max(candidates, axis=0)
        .astype(float)
        .tolist(),
        "valid_mask": list(atlas["valid"]),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA atlas smoke requires a visible GPU")

    candidate = smoke_candidate(args.selected_checkpoints)
    state_manifest = json.loads(
        args.state_bank_manifest.read_text(encoding="utf-8")
    )
    if state_manifest.get("bank") != "calibration":
        raise RuntimeError("smoke state must come from the calibration bank")
    h5_path = Path(state_manifest["state_bank_h5"])
    if sha256_file(h5_path) != state_manifest["state_bank_h5_sha256"]:
        raise RuntimeError("smoke state-bank HDF5 digest mismatch")
    metadata = state_manifest["states"][0]
    with h5py.File(h5_path, "r") as source:
        state = h5_full(source[f"{metadata['bank_id']}/env_state"])
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
    action_low = np.asarray(policy_env.action_space.low).copy()
    action_high = np.asarray(policy_env.action_space.high).copy()
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
        atlas = generate_atlas(
            policy_env,
            rollout_env,
            agent,
            state,
            int(metadata["source_episode_seed"]),
            training_chunks,
            device,
            radius=0.5,
            last_legal_gripper_command=float(
                metadata["last_legal_gripper_command"]
            ),
        )
    finally:
        policy_env.close()
        rollout_env.close()

    valid = np.asarray(atlas["valid"], dtype=bool)
    non_null_outcomes = sum(outcome is not None for outcome in atlas["outcomes"])
    result = {
        "protocol_id": PROTOCOL_ID,
        "scientific_evidence": False,
        "formal_result_reuse_allowed": False,
        "task_id": "StackCube-v1",
        "model_seed": 16018,
        "checkpoint_step": int(candidate["step"]),
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "selected_checkpoints_sha256": sha256_file(args.selected_checkpoints),
        "state_bank_manifest_sha256": sha256_file(args.state_bank_manifest),
        "state_bank_h5_sha256": state_manifest["state_bank_h5_sha256"],
        "bank_id": metadata["bank_id"],
        "state_sha256": metadata["state_sha256"],
        "training_h5_sha256": sha256_file(args.training_h5),
        "sim_backend": "physx_cuda",
        "rollout_envs": PADDED_ENVS,
        "candidate_opportunities": 25,
        "candidate_repeats": 3,
        "valid_candidates": int(valid.sum()),
        "candidate_validity": float(valid.mean()),
        "non_null_outcomes": non_null_outcomes,
        "boundary_by_threshold": atlas["boundary_by_threshold"],
        "accounting": atlas["accounting"],
        "atlas_latency_seconds": atlas["latency_seconds"],
        "diagnostics": atlas_diagnostics(atlas, action_low, action_high),
        "wall_seconds": time.time() - started,
    }
    try:
        validate_smoke_atlas(atlas)
    except RuntimeError as error:
        result["status"] = "CUDA_ACTION_ATLAS_SMOKE_FAIL"
        result["failure"] = str(error)
        write_json(args.output, result)
        raise
    result["status"] = "CUDA_ACTION_ATLAS_SMOKE_PASS"
    write_json(args.output, result)


if __name__ == "__main__":
    main()
