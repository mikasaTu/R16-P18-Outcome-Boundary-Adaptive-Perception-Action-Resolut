#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from action_runtime import PADDED_ENVS, generate_atlas, load_training_chunks
from common import MODEL_SEEDS, PHASES, PROTOCOL_ID, append_jsonl, sha256_file, write_json
from oracle_math import COARSE_INDICES, best_valid_index, transform_observation
from stage25_runtime import (
    load_policy_from_checkpoint,
    make_env,
    policy_chunk,
    reset_to_state,
)
from state_bank_common import h5_full

PHASE_TILE = {
    "free_space_approach": 5,
    "pre_grasp_or_pre_contact": 5,
    "object_in_hand_pre_placement": 10,
    "placement_contact_near_completion": 10,
    "post_success": 10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("calibration", "confirmatory", "post_success_diagnostic"),
        required=True,
    )
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--selected-checkpoints", type=Path, required=True)
    parser.add_argument("--state-bank-manifest", type=Path, required=True)
    parser.add_argument("--training-h5", type=Path, required=True)
    parser.add_argument("--native-action-jsonl", type=Path, required=True)
    parser.add_argument("--action-calibration-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-states", type=int)
    return parser.parse_args()


def selected_row(path: Path, seed: int) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value["groups"][f"StackCube-v1/seed_{seed}"]["selected"]


def batch_tiles(observation: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    transformed = [transform_observation(observation, "local_fine", tile) for tile in range(16)]
    return {
        key: torch.cat([row[key] for row in transformed], dim=0)
        for key in observation
    }


def map_chunks_to_native(
    chunks: np.ndarray, native: Mapping[str, Any]
) -> tuple[list[int], list[float | None]]:
    candidates = np.asarray(native["candidates"], dtype=np.float64)[:, :, :-1]
    valid = np.asarray(native["valid"], dtype=bool)
    coordinate_std = np.asarray(native["coordinate_std"], dtype=np.float64)
    candidate_vectors = candidates.reshape(25, -1)
    result_indices = []
    result_utilities = []
    for chunk in np.asarray(chunks)[:, :4, :-1].reshape(16, -1):
        distances = np.linalg.norm(
            (candidate_vectors - chunk[None]) / coordinate_std[None], axis=1
        )
        distances[~valid] = np.inf
        index = int(np.argmin(distances))
        result_indices.append(index)
        result_utilities.append(native["utilities"][index])
    return result_indices, result_utilities


def arm_summary(atlas: Mapping[str, Any], allowed: list[int]) -> dict[str, Any]:
    valid = [bool(flag) and index in set(allowed) for index, flag in enumerate(atlas["valid"])]
    best = best_valid_index(atlas["utilities"], valid)
    return {
        "best_index": best,
        "utility": float(atlas["utilities"][best]),
        "outcome": atlas["outcomes"][best],
        "accessible_candidate_count": len(allowed),
        "valid_accessible_candidate_count": int(sum(valid)),
    }


def main() -> None:
    args = parse_args()
    if args.model_seed not in MODEL_SEEDS:
        raise ValueError("model seed outside frozen set")
    freeze = json.loads(args.action_calibration_freeze.read_text(encoding="utf-8"))
    if freeze.get("status") != "ACTION_CALIBRATION_FROZEN":
        raise RuntimeError("action calibration is not frozen")
    radius = float(freeze["selected_radius"])
    state_manifest = json.loads(args.state_bank_manifest.read_text(encoding="utf-8"))
    if state_manifest["bank"] != args.stage:
        raise RuntimeError("visual stage/state bank mismatch")
    state_rows = state_manifest["states"]
    if args.max_states is not None:
        state_rows = state_rows[: args.max_states]
    native_rows = {
        row["bank_id"]: row
        for row in (
            json.loads(line)
            for line in args.native_action_jsonl.read_text(encoding="utf-8").splitlines()
        )
    }
    training_chunks = load_training_chunks(str(args.training_h5))
    h5_path = Path(state_manifest["state_bank_h5"])
    device = torch.device("cuda")
    rollout_env = make_env(
        "StackCube-v1", PADDED_ENVS, sim_backend="physx_cpu", reconfiguration_freq=0
    )
    policy_env = make_env(
        "StackCube-v1", 1, sim_backend="physx_cpu", reconfiguration_freq=0
    )
    selected = selected_row(args.selected_checkpoints, args.model_seed)
    agent, _ = load_policy_from_checkpoint(
        rollout_env,
        "StackCube-v1",
        args.model_seed,
        Path(selected["checkpoint_path"]),
        device,
        selected["checkpoint_sha256"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "states.jsonl"
    if raw_path.exists():
        raise FileExistsError(f"refusing to overwrite visual evidence: {raw_path}")
    started = time.time()
    try:
        with h5py.File(h5_path, "r") as source:
            for index, metadata in enumerate(state_rows):
                native_row = native_rows[metadata["bank_id"]]
                native = native_row["atlas"]
                if float(native["radius"]) != radius:
                    raise RuntimeError("native action row radius mismatch")
                state = h5_full(source[f"{metadata['bank_id']}/env_state"])
                obs, _ = reset_to_state(
                    policy_env, state, int(metadata["source_episode_seed"]), 1
                )
                tile_chunks = policy_chunk(agent, batch_tiles(obs), device).detach().cpu().numpy()
                mapped, hidden = map_chunks_to_native(tile_chunks, native)
                oracle_tile = max(
                    range(16),
                    key=lambda tile: (
                        -float("inf") if hidden[tile] is None else float(hidden[tile]),
                        -tile,
                    ),
                )
                random_tile = int.from_bytes(
                    hashlib.sha256(
                        f"{PROTOCOL_ID}|{metadata['bank_id']}|{args.model_seed}|random_tile".encode()
                    ).digest()[:8],
                    "big",
                ) % 16
                phase_tile = PHASE_TILE[metadata["phase"]]
                conditions = {
                    "coarse": generate_atlas(
                        policy_env,
                        rollout_env,
                        agent,
                        state,
                        int(metadata["source_episode_seed"]),
                        training_chunks,
                        device,
                        radius=radius,
                        visual_condition="coarse",
                    ),
                    "oracle_tile": generate_atlas(
                        policy_env,
                        rollout_env,
                        agent,
                        state,
                        int(metadata["source_episode_seed"]),
                        training_chunks,
                        device,
                        radius=radius,
                        visual_condition="local_fine",
                        tile_index=oracle_tile,
                    ),
                    "random_tile": generate_atlas(
                        policy_env,
                        rollout_env,
                        agent,
                        state,
                        int(metadata["source_episode_seed"]),
                        training_chunks,
                        device,
                        radius=radius,
                        visual_condition="local_fine",
                        tile_index=random_tile,
                    ),
                    "phase_tile": generate_atlas(
                        policy_env,
                        rollout_env,
                        agent,
                        state,
                        int(metadata["source_episode_seed"]),
                        training_chunks,
                        device,
                        radius=radius,
                        visual_condition="local_fine",
                        tile_index=phase_tile,
                    ),
                }
                arms = {
                    "CC": arm_summary(conditions["coarse"], list(COARSE_INDICES)),
                    "CF": arm_summary(conditions["coarse"], list(range(25))),
                    "FC": arm_summary(conditions["oracle_tile"], list(COARSE_INDICES)),
                    "FF": arm_summary(conditions["oracle_tile"], list(range(25))),
                    "random_FF": arm_summary(conditions["random_tile"], list(range(25))),
                    "phase_FF": arm_summary(conditions["phase_tile"], list(range(25))),
                    "full_native_upper": arm_summary(native, list(range(25))),
                }
                interaction_i = (
                    arms["FF"]["utility"]
                    - arms["FC"]["utility"]
                    - arms["CF"]["utility"]
                    + arms["CC"]["utility"]
                )
                interaction_j = arms["FF"]["utility"] - max(
                    arms["FC"]["utility"], arms["CF"]["utility"]
                )
                row = {
                    "protocol_id": PROTOCOL_ID,
                    "stage": args.stage,
                    "model_seed": args.model_seed,
                    "bank_id": metadata["bank_id"],
                    "phase": metadata["phase"],
                    "source": metadata["source"],
                    "state_sha256": metadata["state_sha256"],
                    "radius": radius,
                    "tile_screen": {
                        "mapped_native_candidate_indices": mapped,
                        "mapped_native_utilities": hidden,
                        "oracle_tile": oracle_tile,
                        "random_tile": random_tile,
                        "phase_tile": phase_tile,
                        "policy_calls": 16,
                        "simulator_calls": 0,
                        "deployable": False,
                    },
                    "native": native,
                    "conditions": conditions,
                    "arms": arms,
                    "interaction_I": interaction_i,
                    "interaction_J": interaction_j,
                    "abstract_budget": {
                        "CC": {"visual_tiles": 0, "action_candidates": 9},
                        "FC": {"visual_tiles": 1, "action_candidates": 9},
                        "CF": {"visual_tiles": 0, "action_candidates": 25},
                        "FF": {"visual_tiles": 1, "action_candidates": 25},
                    },
                }
                append_jsonl(raw_path, row)
                if index == 0:
                    write_json(
                        args.output_dir / "FIRST_REAL_STATE.json",
                        {
                            "protocol_id": PROTOCOL_ID,
                            "status": "FIRST_REAL_VISUAL_JOINT_STATE_COMPLETE",
                            "bank_id": metadata["bank_id"],
                            "model_seed": args.model_seed,
                            "raw_sha256": sha256_file(raw_path),
                        },
                    )
                print(
                    f"VISUAL_PROBE_PROGRESS stage={args.stage} model_seed={args.model_seed} "
                    f"states={index + 1}/{len(state_rows)}",
                    flush=True,
                )
    finally:
        policy_env.close()
        rollout_env.close()
    write_json(
        args.output_dir / "summary.json",
        {
            "protocol_id": PROTOCOL_ID,
            "status": "VISUAL_RESOLUTION_PROBE_COMPLETE",
            "stage": args.stage,
            "model_seed": args.model_seed,
            "states": len(state_rows),
            "radius": radius,
            "raw_path": str(raw_path),
            "raw_sha256": sha256_file(raw_path),
            "action_calibration_freeze_sha256": sha256_file(args.action_calibration_freeze),
            "wall_seconds": time.time() - started,
        },
    )


if __name__ == "__main__":
    main()
