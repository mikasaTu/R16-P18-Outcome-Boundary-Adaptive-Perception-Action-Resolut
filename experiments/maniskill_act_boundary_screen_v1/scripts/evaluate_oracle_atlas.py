#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from oracle_common import (  # noqa: E402
    COARSE_GRID_INDICES,
    GRID_INDICES,
    adjacent_grid_edges,
    best_index,
    candidate_recall_regret,
    continuous_effect_distance,
    is_alias,
    is_boundary,
    outcome_utility,
    paired_percentile_ci,
    pca_local_action_grid,
)
from oracle_runtime import (  # noqa: E402
    GRIPPER_DIMENSIONS,
    h5_full,
    load_policy,
    make_rgb_env,
    make_state_env,
    policy_chunk,
    reset_to_state,
    rollout_actions,
)
from oracle_visual import (  # noqa: E402
    select_joint_visual_pairs,
    visual_intervention_batch,
)
from protocol_common import MODEL_SEEDS, PROTOCOL_ID, sha256_file, write_json  # noqa: E402


IMPLEMENTATION_CONTRACT = (
    SCRIPT_DIR.parent / "action_atlas" / "oracle_implementation_contract.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-h5", type=Path, required=True)
    parser.add_argument("--state-bank-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-states", type=int)
    return parser.parse_args()


def load_training_chunks(path: Path) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with h5py.File(path, "r") as source:
        trajectory_ids = sorted(int(key.removeprefix("traj_")) for key in source)
        for trajectory_id in trajectory_ids:
            actions = np.asarray(source[f"traj_{trajectory_id}/actions"], dtype=np.float64)
            for timestep in range(len(actions)):
                chunk = actions[timestep : timestep + 4]
                if len(chunk) < 4:
                    chunk = np.concatenate(
                        [chunk, np.repeat(chunk[-1:], 4 - len(chunk), axis=0)], axis=0
                    )
                chunks.append(chunk)
    if len(chunks) < 256:
        raise RuntimeError(f"training action pool is too small: {len(chunks)}")
    return np.stack(chunks)


def clip_actions(actions: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(actions), low[None, None], high[None, None])


def rollout_candidate_set(
    env: Any,
    task_id: str,
    state: Mapping[str, Any],
    episode_seed: int,
    actions: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Evaluate candidates serially because pinned PhysX CPU forbids N>1."""

    outcomes: list[dict[str, Any]] = []
    accounting = {
        "simulator_restores": 0,
        "simulator_steps": 0,
        "action_opportunities": 0,
        "policy_calls": 0,
        "effect_model_calls": 0,
    }
    for candidate in np.asarray(actions):
        rows, counts, _, _ = rollout_actions(
            env,
            task_id,
            state,
            episode_seed,
            candidate[None],
        )
        outcomes.append(rows[0])
        for key in accounting:
            accounting[key] += int(counts[key])
    return outcomes, accounting


def logical_nearest_cell(
    chunk: np.ndarray,
    candidate_chunks: np.ndarray,
    coordinate_std: np.ndarray,
    non_gripper_dimensions: Sequence[int],
) -> int:
    dims = np.asarray(non_gripper_dimensions, dtype=np.int64)
    query = np.asarray(chunk)[:4, dims].reshape(-1)
    cells = np.asarray(candidate_chunks)[:, :4, :][:, :, dims].reshape(25, -1)
    distances = np.linalg.norm((cells - query[None]) / coordinate_std[None], axis=1)
    return int(np.argmin(distances))


def state_file_valid(
    path: Path,
    task_id: str,
    model_seed: int,
    bank_id: str,
    source_bindings: Mapping[str, Any],
    expected_sha256: str | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        return False
    return bool(
        row.get("protocol_id") == PROTOCOL_ID
        and row.get("status") == "ORACLE_STATE_COMPLETE"
        and row.get("task_id") == task_id
        and int(row.get("model_seed", -1)) == model_seed
        and row.get("bank_id") == bank_id
        and row.get("implementation_contract_sha256")
        == sha256_file(IMPLEMENTATION_CONTRACT)
        and row.get("source_bindings") == source_bindings
    )


def evaluate_state(
    *,
    task_id: str,
    model_seed: int,
    state_metadata: Mapping[str, Any],
    state: Mapping[str, Any],
    training_chunks: np.ndarray,
    agent: torch.nn.Module,
    env_policy: Any,
    env_rollout: Any,
    device: torch.device,
    source_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    episode_seed = int(state_metadata["episode_seed"])
    random.seed(16018)
    np.random.seed(16018)
    torch.manual_seed(16018)
    torch.cuda.manual_seed_all(16018)
    observation, _ = reset_to_state(env_policy, state, episode_seed, 1)
    base_chunk = policy_chunk(agent, observation, device)[0]
    action_dim = base_chunk.shape[-1]
    gripper = set(GRIPPER_DIMENSIONS[task_id])
    non_gripper = tuple(index for index in range(action_dim) if index not in gripper)
    atlas = pca_local_action_grid(base_chunk, training_chunks, non_gripper)
    low = np.asarray(env_rollout.single_action_space.low, dtype=np.float64)
    high = np.asarray(env_rollout.single_action_space.high, dtype=np.float64)
    action_candidates = clip_actions(atlas["candidates"], low, high)

    action_outcomes, action_accounting = rollout_candidate_set(
        env_rollout,
        task_id,
        state,
        episode_seed,
        action_candidates,
    )
    utilities = [
        outcome_utility(outcome, atlas["scaled_residual_norms"][index])
        for index, outcome in enumerate(action_outcomes)
    ]
    boundary_edges = [
        [left, right]
        for left, right in adjacent_grid_edges()
        if is_boundary(action_outcomes[left], action_outcomes[right])
    ]
    alias_edges = [
        [left, right]
        for left, right in adjacent_grid_edges()
        if is_alias(action_outcomes[left], action_outcomes[right])
    ]
    coarse_indices = [row * 5 + col for row, col in COARSE_GRID_INDICES]
    coarse_recall, coarse_regret = candidate_recall_regret(utilities, coarse_indices)
    ranked_action_indices = sorted(range(25), key=lambda index: (-utilities[index], index))
    selected_action_indices = ranked_action_indices[:5]
    oracle_recall, oracle_regret = candidate_recall_regret(
        utilities, selected_action_indices
    )
    original_best = best_index(utilities)

    visual_observation, visual_metadata = visual_intervention_batch(observation)
    visual_chunks = policy_chunk(agent, visual_observation, device)
    visual_first_four = clip_actions(visual_chunks[:, :4], low, high)
    visual_outcomes, visual_accounting = rollout_candidate_set(
        env_rollout,
        task_id,
        state,
        episode_seed,
        visual_first_four,
    )
    action_l2_changes = np.linalg.norm(
        visual_chunks[:, :4] - base_chunk[None, :4], axis=(1, 2)
    )
    nearest_cells = [
        logical_nearest_cell(
            visual_chunks[index],
            action_candidates,
            atlas["coordinate_std"],
            non_gripper,
        )
        for index in range(48)
    ]
    base_outcome = action_outcomes[12]
    important_pairs = [
        bool(
            nearest_cells[index] != 12
            or is_boundary(base_outcome, visual_outcomes[index])
        )
        for index in range(48)
    ]
    selected_visual_indices = select_joint_visual_pairs(
        visual_metadata, action_l2_changes
    )

    joint_actions: list[np.ndarray] = []
    joint_identity: list[tuple[int, int]] = []
    dims = np.asarray(non_gripper, dtype=np.int64)
    for visual_index in selected_visual_indices:
        for action_index in selected_action_indices:
            chunk = visual_chunks[visual_index, :4].copy()
            residual = atlas["physical_residuals"][action_index].reshape(4, len(dims))
            chunk[:, dims] += residual
            joint_actions.append(chunk)
            joint_identity.append((visual_index, action_index))
    joint_action_array = clip_actions(np.stack(joint_actions), low, high)
    joint_outcomes, joint_accounting = rollout_candidate_set(
        env_rollout,
        task_id,
        state,
        episode_seed,
        joint_action_array,
    )
    joint_utilities = [
        outcome_utility(
            outcome,
            atlas["scaled_residual_norms"][joint_identity[index][1]],
        )
        for index, outcome in enumerate(joint_outcomes)
    ]
    visual_best_cells: dict[int, int] = {}
    for visual_index in selected_visual_indices:
        positions = [
            index
            for index, identity in enumerate(joint_identity)
            if identity[0] == visual_index
        ]
        best_position = max(
            positions,
            key=lambda index: (joint_utilities[index], -joint_identity[index][1]),
        )
        visual_best_cells[visual_index] = joint_identity[best_position][1]
    joint_boundary = any(
        best_cell != original_best for best_cell in visual_best_cells.values()
    )

    action_cells = []
    for index, outcome in enumerate(action_outcomes):
        action_cells.append(
            {
                "flat_index": index,
                "grid_index": list(GRID_INDICES[index]),
                "grid_levels": [
                    float([-1.0, -0.5, 0.0, 0.5, 1.0][GRID_INDICES[index][0]]),
                    float([-1.0, -0.5, 0.0, 0.5, 1.0][GRID_INDICES[index][1]]),
                ],
                "scaled_residual_norm": float(
                    atlas["scaled_residual_norms"][index]
                ),
                "utility": float(utilities[index]),
                "first_four_actions": action_candidates[index].tolist(),
                "physical_non_gripper_residual": atlas["physical_residuals"][
                    index
                ].reshape(4, len(non_gripper)).tolist(),
                "outcome": outcome,
            }
        )
    visual_pairs = []
    for index, metadata in enumerate(visual_metadata):
        visual_pairs.append(
            {
                **metadata,
                "first_four_action_l2_change": float(action_l2_changes[index]),
                "nearest_action_cell": nearest_cells[index],
                "important": important_pairs[index],
                "first_four_actions": visual_first_four[index].tolist(),
                "outcome": visual_outcomes[index],
            }
        )
    joint_pairs = []
    for index, outcome in enumerate(joint_outcomes):
        visual_index, action_index = joint_identity[index]
        joint_pairs.append(
            {
                "visual_pair_index": visual_index,
                "action_cell_index": action_index,
                "utility": float(joint_utilities[index]),
                "first_four_actions": joint_action_array[index].tolist(),
                "outcome": outcome,
            }
        )

    accounting = {
        "logical_policy_inputs": 49,
        "batched_policy_invocations": 2,
        "effect_model_calls": 0,
        "action_atlas": action_accounting,
        "visual_atlas": visual_accounting,
        "joint_probe": joint_accounting,
        "total_simulator_restores": int(
            action_accounting["simulator_restores"]
            + visual_accounting["simulator_restores"]
            + joint_accounting["simulator_restores"]
        ),
        "total_simulator_steps": int(
            action_accounting["simulator_steps"]
            + visual_accounting["simulator_steps"]
            + joint_accounting["simulator_steps"]
        ),
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "ORACLE_STATE_COMPLETE",
        "task_id": task_id,
        "model_seed": model_seed,
        "bank_id": state_metadata["bank_id"],
        "phase": state_metadata["phase"],
        "episode_seed": episode_seed,
        "source_trajectory_id": int(state_metadata["source_trajectory_id"]),
        "source_timestep": int(state_metadata["timestep"]),
        "implementation_contract_sha256": sha256_file(IMPLEMENTATION_CONTRACT),
        "source_bindings": dict(source_bindings),
        "action_atlas": {
            "cells": action_cells,
            "neighbor_indices": [int(value) for value in atlas["neighbor_indices"]],
            "neighbor_distances": [float(value) for value in atlas["neighbor_distances"]],
            "coordinate_std": [float(value) for value in atlas["coordinate_std"]],
            "score_scales": [float(value) for value in atlas["score_scales"]],
            "standardized_pca_directions": atlas[
                "standardized_directions"
            ].tolist(),
            "boundary_edges": boundary_edges,
            "alias_edges": alias_edges,
            "action_boundary_density": len(boundary_edges) / 40.0,
            "adjacent_alias_rate": len(alias_edges) / 40.0,
            "full_fine_best_index": original_best,
            "coarse_uniform_indices": coarse_indices,
            "coarse_best_action_recall": coarse_recall,
            "coarse_outcome_regret": coarse_regret,
            "oracle_selected_indices": selected_action_indices,
            "oracle_best_action_recall": oracle_recall,
            "oracle_outcome_regret": oracle_regret,
        },
        "visual_atlas": {
            "pairs": visual_pairs,
            "important_pair_count": int(sum(important_pairs)),
            "visual_boundary_density": float(np.mean(important_pairs)),
            "selected_joint_visual_pair_indices": selected_visual_indices,
        },
        "joint_probe": {
            "pairs": joint_pairs,
            "best_cell_by_visual_pair": {
                str(key): int(value) for key, value in visual_best_cells.items()
            },
            "original_best_cell": original_best,
            "joint_boundary": joint_boundary,
        },
        "accounting": accounting,
    }


def main() -> None:
    args = parse_args()
    if args.model_seed not in MODEL_SEEDS:
        raise ValueError("model seed is outside the frozen set")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("oracle policy inference requires a visible CUDA GPU")
    state_manifest = json.loads(
        args.state_bank_manifest.read_text(encoding="utf-8")
    )
    if (
        state_manifest.get("protocol_id") != PROTOCOL_ID
        or state_manifest.get("status") != "STATE_BANK_COMPLETE"
        or state_manifest.get("task_id") != args.task_id
        or int(state_manifest.get("state_count", -1)) != 64
    ):
        raise RuntimeError("invalid state-bank manifest")
    state_h5_path = Path(state_manifest["state_bank_h5"])
    if sha256_file(state_h5_path) != state_manifest["state_bank_h5_sha256"]:
        raise RuntimeError("state-bank HDF5 digest mismatch")
    training_chunks = load_training_chunks(args.train_h5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    states_dir = args.output_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    env_policy = make_rgb_env(args.task_id, 1)
    env_rollout = make_state_env(args.task_id, 1)
    if training_chunks.shape[-1] != env_rollout.single_action_space.shape[-1]:
        raise RuntimeError("training/environment action dimension mismatch")
    agent, selection, checkpoint_path = load_policy(
        args.task_id, args.model_seed, args.run_dir, env_policy, device
    )
    source_bindings = {
        "oracle_evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "state_bank_manifest_sha256": sha256_file(args.state_bank_manifest),
        "state_bank_h5_sha256": state_manifest["state_bank_h5_sha256"],
        "train_h5_sha256": sha256_file(args.train_h5),
        "selected_checkpoint_step": int(selection["selected"]["step"]),
        "selected_checkpoint_sha256": sha256_file(checkpoint_path),
        "selected_checkpoint_path": str(checkpoint_path),
    }
    prior_surface_hashes: dict[str, str] = {}
    prior_summary_path = args.output_dir / "summary.json"
    if prior_summary_path.is_file():
        try:
            prior_summary = json.loads(
                prior_summary_path.read_text(encoding="utf-8")
            )
            if prior_summary.get("source_bindings") == source_bindings:
                prior_surface_hashes = {
                    str(item["bank_id"]): str(item["sha256"])
                    for item in prior_summary.get("surface_files", [])
                }
        except (KeyError, OSError, TypeError, ValueError):
            prior_surface_hashes = {}
    started = time.time()
    try:
        with h5py.File(state_h5_path, "r") as state_source:
            state_rows = list(state_manifest["states"])
            if args.max_states is not None:
                state_rows = state_rows[: args.max_states]
            for index, state_metadata in enumerate(state_rows):
                bank_id = state_metadata["bank_id"]
                output_path = states_dir / f"{bank_id}.json"
                if state_file_valid(
                    output_path,
                    args.task_id,
                    args.model_seed,
                    bank_id,
                    source_bindings,
                    prior_surface_hashes.get(bank_id),
                ):
                    print(
                        f"ORACLE_RESUME task={args.task_id} seed={args.model_seed} "
                        f"state={index + 1}/{len(state_rows)} bank_id={bank_id}",
                        flush=True,
                    )
                    continue
                state = h5_full(state_source[f"{bank_id}/env_state"])
                row = evaluate_state(
                    task_id=args.task_id,
                    model_seed=args.model_seed,
                    state_metadata=state_metadata,
                    state=state,
                    training_chunks=training_chunks,
                    agent=agent,
                    env_policy=env_policy,
                    env_rollout=env_rollout,
                    device=device,
                    source_bindings=source_bindings,
                )
                write_json(output_path, row)
                print(
                    f"ORACLE_PROGRESS task={args.task_id} seed={args.model_seed} "
                    f"state={index + 1}/{len(state_rows)} bank_id={bank_id}",
                    flush=True,
                )
    finally:
        env_policy.close()
        env_rollout.close()

    rows = [
        json.loads((states_dir / f"{state['bank_id']}.json").read_text(encoding="utf-8"))
        for state in state_rows
    ]
    if len(rows) != len(state_rows):
        raise RuntimeError("oracle state output count mismatch")
    action_density = [row["action_atlas"]["action_boundary_density"] for row in rows]
    visual_density = [row["visual_atlas"]["visual_boundary_density"] for row in rows]
    joint_boundary = [float(row["joint_probe"]["joint_boundary"]) for row in rows]
    coarse_recall = [row["action_atlas"]["coarse_best_action_recall"] for row in rows]
    oracle_recall = [row["action_atlas"]["oracle_best_action_recall"] for row in rows]
    coarse_regret = [row["action_atlas"]["coarse_outcome_regret"] for row in rows]
    oracle_regret = [row["action_atlas"]["oracle_outcome_regret"] for row in rows]
    index_rows = []
    for state in state_rows:
        path = states_dir / f"{state['bank_id']}.json"
        index_rows.append(
            {
                "bank_id": state["bank_id"],
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    mean_coarse_regret = float(np.mean(coarse_regret))
    mean_oracle_regret = float(np.mean(oracle_regret))
    regret_reduction = (
        (mean_coarse_regret - mean_oracle_regret) / mean_coarse_regret
        if mean_coarse_regret > 0
        else 0.0
    )
    summary = {
        "protocol_id": PROTOCOL_ID,
        "status": "ORACLE_ATLAS_COMPLETE",
        "scope": "privileged_stage2_oracle_not_deployable_selector",
        "task_id": args.task_id,
        "model_seed": args.model_seed,
        "states": len(rows),
        "state_bank_manifest": str(args.state_bank_manifest),
        "state_bank_manifest_sha256": sha256_file(args.state_bank_manifest),
        "train_h5": str(args.train_h5),
        "train_h5_sha256": sha256_file(args.train_h5),
        "training_action_chunks": int(len(training_chunks)),
        "selected_checkpoint_step": int(selection["selected"]["step"]),
        "selected_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_bindings": source_bindings,
        "surface_files": index_rows,
        "metrics": {
            "action_boundary_density": float(np.mean(action_density)),
            "action_boundary_density_paired_bootstrap_95_ci": paired_percentile_ci(
                action_density
            ),
            "visual_boundary_density": float(np.mean(visual_density)),
            "visual_boundary_density_paired_bootstrap_95_ci": paired_percentile_ci(
                visual_density
            ),
            "joint_coupling_density": float(np.mean(joint_boundary)),
            "joint_coupling_density_paired_bootstrap_95_ci": paired_percentile_ci(
                joint_boundary
            ),
            "coarse_best_action_recall": float(np.mean(coarse_recall)),
            "oracle_best_action_recall": float(np.mean(oracle_recall)),
            "best_action_recall_improvement_percentage_points": 100.0
            * (float(np.mean(oracle_recall)) - float(np.mean(coarse_recall))),
            "coarse_outcome_regret": mean_coarse_regret,
            "oracle_outcome_regret": mean_oracle_regret,
            "outcome_regret_reduction_fraction": regret_reduction,
            "adjacent_outcome_alias_rate": float(
                np.mean(
                    [row["action_atlas"]["adjacent_alias_rate"] for row in rows]
                )
            ),
        },
        "accounting": {
            "logical_policy_inputs": int(
                sum(row["accounting"]["logical_policy_inputs"] for row in rows)
            ),
            "batched_policy_invocations": int(
                sum(row["accounting"]["batched_policy_invocations"] for row in rows)
            ),
            "effect_model_calls": 0,
            "simulator_restores": int(
                sum(row["accounting"]["total_simulator_restores"] for row in rows)
            ),
            "simulator_steps": int(
                sum(row["accounting"]["total_simulator_steps"] for row in rows)
            ),
        },
        "short_horizon_sim_backend": "physx_cpu",
        "rgb_render_backend": "sapien_cuda",
        "policy_inference_device": str(device),
        "implementation_contract": "action_atlas/oracle_implementation_contract.json",
        "implementation_contract_sha256": sha256_file(IMPLEMENTATION_CONTRACT),
        "elapsed_seconds": time.time() - started,
        "completed_at_unix": time.time(),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
