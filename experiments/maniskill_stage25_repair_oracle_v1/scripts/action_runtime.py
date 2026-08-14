#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Mapping

import h5py
import numpy as np
import torch

from oracle_math import (
    PRIMARY_UTILITY,
    boundary_density,
    local_pca_grid,
    transform_observation,
    utility,
)
from stage25_runtime import (
    ContactTracker,
    neutral_from_last,
    policy_chunk,
    quaternion_distance_rad,
    reset_to_state,
    task_snapshot,
    temporal_action_for_indices,
)

REPEATS = 3
GRID_CANDIDATES = 25
PADDED_ENVS = REPEATS * GRID_CANDIDATES


def load_training_chunks(path: h5py.File | str) -> np.ndarray:
    close = False
    if isinstance(path, (str, bytes)):
        source = h5py.File(path, "r")
        close = True
    else:
        source = path
    chunks = []
    try:
        for key in sorted(source, key=lambda value: int(value.removeprefix("traj_"))):
            actions = np.asarray(source[f"{key}/actions"], dtype=np.float32)
            for start in range(max(0, len(actions) - 3)):
                chunks.append(actions[start : start + 4])
    finally:
        if close:
            source.close()
    result = np.asarray(chunks, dtype=np.float32)
    if result.ndim != 3 or result.shape[0] < 256:
        raise RuntimeError(f"insufficient training chunks: {result.shape}")
    return result


def phase_outcome(
    stable_success: bool,
    supported: bool,
    grasped: bool,
    progress_delta: float,
) -> tuple[str, int]:
    if stable_success:
        return "stable_success", 4
    if supported:
        return "supported_not_stable", 3
    if grasped:
        return "object_in_hand", 2
    if progress_delta >= 0:
        return "nonnegative_progress", 1
    return "regressed", 0


def aggregate_repeats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != REPEATS:
        raise ValueError(len(rows))
    categorical_fields = (
        "stable_success",
        "phase_outcome",
        "grasped",
        "supported",
        "dropped_or_slipped",
        "recoverable",
    )
    agreement = all(
        tuple(row[field] for field in categorical_fields)
        == tuple(rows[0][field] for field in categorical_fields)
        for row in rows[1:]
    )
    return {
        "stable_success": all(row["stable_success"] for row in rows),
        "phase_outcome": rows[0]["phase_outcome"] if agreement else "repeat_disagreement",
        "phase_outcome_rank": min(int(row["phase_outcome_rank"]) for row in rows),
        "grasped": all(row["grasped"] for row in rows),
        "supported": all(row["supported"] for row in rows),
        "dropped_or_slipped": any(row["dropped_or_slipped"] for row in rows),
        "recoverable": all(row["recoverable"] for row in rows),
        "intended_contact": any(row["intended_contact"] for row in rows),
        "unintended_contact": any(row["unintended_contact"] for row in rows),
        "object_delta_translation_m": np.mean(
            [row["object_delta_translation_m"] for row in rows], axis=0
        ).astype(float).tolist(),
        "object_delta_rotation_rad": float(
            np.mean([row["object_delta_rotation_rad"] for row in rows])
        ),
        "normalized_progress_delta": float(
            np.mean([row["normalized_progress_delta"] for row in rows])
        ),
        "categorical_repeat_agreement": agreement,
        "repeat_rows": rows,
    }


def rollout_grid(
    env: Any,
    agent: torch.nn.Module,
    state: Mapping[str, Any],
    episode_seed: int,
    grid: Mapping[str, Any],
    device: torch.device,
    *,
    visual_condition: str,
    tile_index: int | None,
) -> dict[str, Any]:
    if int(env.num_envs) != PADDED_ENVS:
        raise RuntimeError(f"rollout env must have {PADDED_ENVS} envs")
    candidates = np.asarray(grid["candidates"], dtype=np.float32)
    valid = np.asarray(grid["valid"], dtype=bool)
    actions = np.repeat(candidates, REPEATS, axis=0)
    valid_slots = np.repeat(valid, REPEATS)
    # Invalid slots are padded with a valid zero-arm action. Their physical work
    # is accounted separately and never enters a candidate outcome or utility.
    actions[~valid_slots] = 0.0
    obs, _ = reset_to_state(env, state, episode_seed, PADDED_ENVS)
    base = env.base_env
    initial = task_snapshot(base, "StackCube-v1")
    tracker = ContactTracker("StackCube-v1", base)
    success_once = torch.as_tensor(initial["success"], device=device, dtype=torch.bool)
    neutral_success = torch.ones(PADDED_ENVS, dtype=torch.bool, device=device)
    ever_grasped = torch.as_tensor(initial["grasped"], device=device, dtype=torch.bool)
    valid_mask = torch.as_tensor(valid_slots, device=device, dtype=torch.bool)
    last_action = torch.zeros(PADDED_ENVS, actions.shape[-1], device=device)
    for step in range(4):
        action = torch.as_tensor(actions[:, step], device=device)
        last_action = action
        obs, _, _, _, info = env.step(action)
        success_once |= info["success"].to(torch.bool) & valid_mask
        tracker.update(count_mask=valid_mask, success_seen=success_once)
        ever_grasped |= torch.as_tensor(
            task_snapshot(base, "StackCube-v1")["grasped"], device=device
        ) & valid_mask
    action_dim = actions.shape[-1]
    table = torch.zeros(
        PADDED_ENVS, 20, 50, action_dim, dtype=torch.float32, device=device
    )
    indices = torch.arange(PADDED_ENVS, device=device)
    for timestep in range(20):
        transformed = transform_observation(obs, visual_condition, tile_index)
        chunk = policy_chunk(agent, transformed, device)
        action = temporal_action_for_indices(table, chunk, timestep, indices)
        action[~valid_mask] = 0.0
        last_action = action
        obs, _, _, _, info = env.step(action)
        success_once |= info["success"].to(torch.bool) & valid_mask
        tracker.update(count_mask=valid_mask, success_seen=success_once)
        ever_grasped |= torch.as_tensor(
            task_snapshot(base, "StackCube-v1")["grasped"], device=device
        ) & valid_mask
    for _ in range(5):
        action = neutral_from_last(last_action)
        action[~valid_mask] = 0.0
        obs, _, _, _, info = env.step(action)
        success = info["success"].to(torch.bool)
        neutral_success &= success | ~valid_mask
        success_once |= success & valid_mask
        tracker.update(count_mask=valid_mask, success_seen=success_once)
    final = task_snapshot(base, "StackCube-v1")
    slot_rows: list[dict[str, Any] | None] = []
    for slot in range(PADDED_ENVS):
        if not valid_slots[slot]:
            slot_rows.append(None)
            continue
        translation = final["object_position"][slot] - initial["object_position"][slot]
        rotation = quaternion_distance_rad(
            initial["object_quaternion"][slot], final["object_quaternion"][slot]
        )
        progress_delta = float(
            final["normalized_progress"][slot] - initial["normalized_progress"][slot]
        )
        supported = bool(final["supported"][slot])
        grasped = bool(final["grasped"][slot])
        dropped = bool(ever_grasped[slot].item() and not grasped and not supported)
        contacts = tracker.episode_fields(slot)
        stable = bool(neutral_success[slot].item())
        phase, rank = phase_outcome(stable, supported, grasped, progress_delta)
        unintended = contacts["unintended_contact_onsets"] > 0
        recoverable = bool(stable or (not dropped and not unintended and progress_delta >= -0.05))
        slot_rows.append(
            {
                "stable_success": stable,
                "success_once": bool(success_once[slot].item()),
                "phase_outcome": phase,
                "phase_outcome_rank": rank,
                "grasped": grasped,
                "supported": supported,
                "dropped_or_slipped": dropped,
                "recoverable": recoverable,
                "intended_contact": contacts["intended_contact_onsets"] > 0,
                "unintended_contact": unintended,
                "object_delta_translation_m": translation.astype(float).tolist(),
                "object_delta_rotation_rad": float(rotation),
                "normalized_progress_delta": progress_delta,
                **contacts,
            }
        )
    aggregate: list[dict[str, Any] | None] = []
    utilities: list[float | None] = []
    for candidate in range(GRID_CANDIDATES):
        if not valid[candidate]:
            aggregate.append(None)
            utilities.append(None)
            continue
        repeats = [slot_rows[candidate * REPEATS + repeat] for repeat in range(REPEATS)]
        outcome = aggregate_repeats([row for row in repeats if row is not None])
        aggregate.append(outcome)
        utilities.append(
            utility(outcome, float(grid["scaled_residual_norms"][candidate]), PRIMARY_UTILITY)
        )
    return {
        "outcomes": aggregate,
        "utilities": utilities,
        "valid": valid.astype(bool).tolist(),
        "accounting": {
            "candidate_opportunities": GRID_CANDIDATES,
            "valid_candidates": int(valid.sum()),
            "candidate_repeats": REPEATS,
            "scientific_simulator_steps": int(valid.sum()) * REPEATS * 29,
            "scientific_policy_calls": int(valid.sum()) * REPEATS * 20,
            "padded_invalid_simulator_steps": int((~valid).sum()) * REPEATS * 29,
            "padded_invalid_policy_forward_rows": int((~valid).sum()) * REPEATS * 20,
            "effect_model_calls": 0,
        },
    }


def generate_atlas(
    policy_env: Any,
    rollout_env: Any,
    agent: torch.nn.Module,
    state: Mapping[str, Any],
    episode_seed: int,
    training_chunks: np.ndarray,
    device: torch.device,
    *,
    radius: float,
    visual_condition: str = "native",
    tile_index: int | None = None,
) -> dict[str, Any]:
    obs, _ = reset_to_state(policy_env, state, episode_seed, 1)
    transformed = transform_observation(obs, visual_condition, tile_index)
    nominal = policy_chunk(agent, transformed, device)[0].detach().cpu().numpy()
    action_low = np.broadcast_to(
        np.asarray(policy_env.action_space.low, dtype=np.float64), (4, nominal.shape[-1])
    )
    action_high = np.broadcast_to(
        np.asarray(policy_env.action_space.high, dtype=np.float64), (4, nominal.shape[-1])
    )
    grid = local_pca_grid(
        nominal,
        training_chunks,
        action_low,
        action_high,
        radius=radius,
    )
    rollout = rollout_grid(
        rollout_env,
        agent,
        state,
        episode_seed,
        grid,
        device,
        visual_condition=visual_condition,
        tile_index=tile_index,
    )
    result = {
        "visual_condition": visual_condition,
        "tile_index": tile_index,
        "radius": float(radius),
        "nominal_action_first4": nominal[:4].astype(float).tolist(),
        "candidates": np.asarray(grid["candidates"]).astype(float).tolist(),
        "valid": rollout["valid"],
        "scaled_residual_norms": grid["scaled_residual_norms"].astype(float).tolist(),
        "neighbor_indices": grid["neighbor_indices"].astype(int).tolist(),
        "neighbor_distances": grid["neighbor_distances"].astype(float).tolist(),
        "coordinate_std": grid["coordinate_std"].astype(float).tolist(),
        "directions": grid["directions"].astype(float).tolist(),
        "score_scales": grid["score_scales"].astype(float).tolist(),
        **rollout,
    }
    result["boundary_by_threshold"] = {
        str(threshold): boundary_density(
            result["outcomes"], result["valid"], threshold
        )
        for threshold in (0.5, 1.0, 1.5)
    }
    return result
