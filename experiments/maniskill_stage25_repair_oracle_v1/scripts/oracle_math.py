#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

FINE_LEVELS = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
FINE_INDICES = tuple((row, col) for row in range(5) for col in range(5))
COARSE_INDICES = tuple(row * 5 + col for row in (0, 2, 4) for col in (0, 2, 4))
RADIUS_CANDIDATES = (0.5, 1.0, 1.5)
EFFECT_THRESHOLD_CANDIDATES = (0.5, 1.0, 1.5)
J_THRESHOLD_CANDIDATES = (1.0, 2.0, 5.0)


def effective_gripper_command(
    raw_command: float,
    low: float = -1.0,
    high: float = 1.0,
) -> float:
    """Return the normalized gripper command that ManiSkill executes.

    ManiSkill's normalized controllers explicitly clip incoming actions before
    scaling them.  State-bank metadata records this controller-effective value
    as the last legal gripper command; it must not retain a raw ACT overshoot.
    """
    value = float(raw_command)
    lower = float(low)
    upper = float(high)
    if not np.isfinite(value) or not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("gripper command and bounds must be finite")
    if lower > upper:
        raise ValueError(f"invalid gripper bounds: {lower} > {upper}")
    return float(np.clip(value, lower, upper))


def atlas_center_with_frozen_gripper(
    nominal_chunk: np.ndarray,
    gripper_command: float,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    """Build the non-gripper atlas center without clipping a candidate.

    The frozen action atlas perturbs only arm dimensions.  Every candidate
    therefore shares the state bank's explicit last legal gripper command.
    Unlike controller preprocessing, this function refuses an illegal stored
    command rather than clipping it, preserving the candidate-bounds rule.
    """
    nominal = np.asarray(nominal_chunk, dtype=np.float64)
    low = np.broadcast_to(np.asarray(action_low, dtype=np.float64), nominal.shape)
    high = np.broadcast_to(np.asarray(action_high, dtype=np.float64), nominal.shape)
    if nominal.ndim != 2 or nominal.shape[1] < 2:
        raise ValueError(f"nominal chunk must be [H,A>=2], got {nominal.shape}")
    command = float(gripper_command)
    if not np.isfinite(command):
        raise ValueError("frozen gripper command must be finite")
    if np.any(command < low[:, -1]) or np.any(command > high[:, -1]):
        raise ValueError(
            f"frozen gripper command {command} is outside action-space bounds"
        )
    center = nominal.copy()
    center[:, -1] = command
    return center


def local_pca_grid(
    nominal_chunk: np.ndarray,
    training_chunks: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    radius: float,
    neighbor_count: int = 256,
    std_floor: float = 1e-3,
) -> dict[str, Any]:
    nominal = np.asarray(nominal_chunk, dtype=np.float64)
    chunks = np.asarray(training_chunks, dtype=np.float64)
    if nominal.ndim != 2 or nominal.shape[0] < 4:
        raise ValueError(f"nominal chunk must be [>=4,A], got {nominal.shape}")
    if chunks.ndim != 3 or chunks.shape[1:] != (4, nominal.shape[1]):
        raise ValueError(f"training chunks shape mismatch: {chunks.shape}")
    if chunks.shape[0] < neighbor_count:
        raise ValueError(f"need {neighbor_count} training chunks, got {len(chunks)}")
    arm_dims = np.arange(nominal.shape[1] - 1, dtype=np.int64)
    selected = chunks[:, :, arm_dims].reshape(len(chunks), -1)
    base = nominal[:4, arm_dims].reshape(-1)
    coordinate_std = np.maximum(selected.std(axis=0, ddof=1), std_floor)
    distances = np.linalg.norm((selected - base[None]) / coordinate_std[None], axis=1)
    neighbors = np.argsort(distances, kind="stable")[:neighbor_count]
    residual = (selected[neighbors] - base[None]) / coordinate_std[None]
    centered = residual - residual.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    directions = vh[:2].copy()
    for index in range(2):
        pivot = int(np.argmax(np.abs(directions[index])))
        if directions[index, pivot] < 0:
            directions[index] *= -1
    score_scales = np.maximum((centered @ directions.T).std(axis=0, ddof=1), std_floor)
    candidates = []
    residual_norms = []
    for row, col in FINE_INDICES:
        standardized = radius * (
            FINE_LEVELS[row] * score_scales[0] * directions[0]
            + FINE_LEVELS[col] * score_scales[1] * directions[1]
        )
        physical = standardized * coordinate_std
        candidate = nominal[:4].copy()
        candidate[:, arm_dims] += physical.reshape(4, len(arm_dims))
        candidates.append(candidate)
        residual_norms.append(float(np.linalg.norm(standardized)))
    candidate_array = np.stack(candidates)
    low = np.broadcast_to(np.asarray(action_low), candidate_array.shape)
    high = np.broadcast_to(np.asarray(action_high), candidate_array.shape)
    valid = np.all((candidate_array >= low) & (candidate_array <= high), axis=(1, 2))
    # No clipping is performed. Invalid candidates retain their out-of-range value
    # so the audit can verify why they were excluded.
    return {
        "candidates": candidate_array.astype(np.float32),
        "valid": valid,
        "scaled_residual_norms": np.asarray(residual_norms, dtype=np.float64),
        "neighbor_indices": neighbors,
        "neighbor_distances": distances[neighbors],
        "coordinate_std": coordinate_std,
        "directions": directions,
        "score_scales": score_scales,
        "radius": float(radius),
        "arm_dimensions": arm_dims,
    }


def adjacent_edges() -> tuple[tuple[int, int], ...]:
    edges = []
    for row in range(5):
        for col in range(5):
            index = row * 5 + col
            if row < 4:
                edges.append((index, index + 5))
            if col < 4:
                edges.append((index, index + 1))
    assert len(edges) == 40
    return tuple(edges)


def categorical_tuple(outcome: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        bool(outcome["stable_success"]),
        str(outcome["phase_outcome"]),
        bool(outcome["grasped"]),
        bool(outcome["supported"]),
        bool(outcome["dropped_or_slipped"]),
        bool(outcome["recoverable"]),
    )


def effect_distance(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    translation = np.linalg.norm(
        np.asarray(first["object_delta_translation_m"])
        - np.asarray(second["object_delta_translation_m"])
    ) / 0.01
    rotation = abs(
        float(first["object_delta_rotation_rad"])
        - float(second["object_delta_rotation_rad"])
    ) / math.radians(10)
    progress = abs(
        float(first["normalized_progress_delta"])
        - float(second["normalized_progress_delta"])
    ) / 0.05
    return float(np.sqrt(np.mean(np.square([translation, rotation, progress]))))


def edge_is_boundary(
    first: Mapping[str, Any], second: Mapping[str, Any], effect_threshold: float
) -> bool:
    return bool(
        categorical_tuple(first) != categorical_tuple(second)
        or bool(first["stable_success"]) != bool(second["stable_success"])
        or bool(first["recoverable"]) != bool(second["recoverable"])
        or effect_distance(first, second) >= effect_threshold
    )


def boundary_density(
    outcomes: Sequence[Mapping[str, Any] | None],
    valid: Sequence[bool],
    effect_threshold: float,
) -> dict[str, Any]:
    boundaries = 0
    eligible = 0
    rows = []
    for first, second in adjacent_edges():
        edge_valid = bool(valid[first] and valid[second])
        boundary = bool(
            edge_valid
            and outcomes[first] is not None
            and outcomes[second] is not None
            and edge_is_boundary(outcomes[first], outcomes[second], effect_threshold)
        )
        eligible += int(edge_valid)
        boundaries += int(boundary)
        rows.append({"first": first, "second": second, "valid": edge_valid, "boundary": boundary})
    return {
        "boundary_edges": boundaries,
        "eligible_edges": eligible,
        "boundary_density": boundaries / eligible if eligible else 0.0,
        "edges": rows,
    }


PRIMARY_UTILITY = {
    "stable_success": 100.0,
    "clipped_progress_delta": 20.0,
    "supported": 5.0,
    "intended_contact": 2.0,
    "unintended_contact": -5.0,
    "dropped_or_slipped": -10.0,
    "recoverable": 1.0,
    "scaled_action_residual_norm": -0.01,
}


def utility(
    outcome: Mapping[str, Any],
    scaled_action_residual_norm: float,
    weights: Mapping[str, float] = PRIMARY_UTILITY,
) -> float:
    return float(
        weights["stable_success"] * bool(outcome["stable_success"])
        + weights["clipped_progress_delta"]
        * np.clip(float(outcome["normalized_progress_delta"]), -1.0, 1.0)
        + weights["supported"] * bool(outcome["supported"])
        + weights["intended_contact"] * bool(outcome["intended_contact"])
        + weights["unintended_contact"] * bool(outcome["unintended_contact"])
        + weights["dropped_or_slipped"] * bool(outcome["dropped_or_slipped"])
        + weights["recoverable"] * bool(outcome["recoverable"])
        + weights["scaled_action_residual_norm"] * float(scaled_action_residual_norm)
    )


def best_valid_index(utilities: Sequence[float | None], valid: Sequence[bool]) -> int:
    indices = [index for index, flag in enumerate(valid) if flag and utilities[index] is not None]
    if not indices:
        raise ValueError("no valid action candidate")
    return max(indices, key=lambda index: (float(utilities[index]), -index))


def paired_percentile_ci(
    differences: Sequence[float], replicates: int = 10_000, seed: int = 16018
) -> list[float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired bootstrap requires a nonempty vector")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    # Chunked generation avoids allocating replicates x states for large banks.
    for start in range(0, replicates, 1000):
        stop = min(start + 1000, replicates)
        draw = rng.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = values[draw].mean(axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def coarse_rgb(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim != 5:
        raise ValueError(f"RGB must be [N,camera,C,H,W], got {rgb.shape}")
    n, cameras, channels, height, width = rgb.shape
    work = rgb.reshape(n * cameras, channels, height, width).to(torch.float32)
    down = F.interpolate(work, size=(64, 64), mode="bilinear", align_corners=False)
    up = F.interpolate(down, size=(height, width), mode="bilinear", align_corners=False)
    up = up.reshape(n, cameras, channels, height, width)
    return up.round().clamp(0, 255).to(rgb.dtype) if rgb.dtype == torch.uint8 else up.to(rgb.dtype)


def local_fine_rgb(rgb: torch.Tensor, tile_index: int) -> torch.Tensor:
    if tile_index not in range(16):
        raise ValueError(tile_index)
    coarse = coarse_rgb(rgb)
    _, _, _, height, width = rgb.shape
    tile_h, tile_w = height // 4, width // 4
    row, col = divmod(tile_index, 4)
    y0, x0 = row * tile_h, col * tile_w
    y1, x1 = y0 + tile_h, x0 + tile_w
    native = rgb[:, :, :, y0:y1, x0:x1].to(torch.float32)
    coarse_tile = coarse[:, :, :, y0:y1, x0:x1].to(torch.float32)
    y = torch.ones(tile_h, device=rgb.device, dtype=torch.float32)
    x = torch.ones(tile_w, device=rgb.device, dtype=torch.float32)
    # Frozen two-pixel cosine feather inside the tile.
    feather = torch.tensor([0.25, 0.75], device=rgb.device)
    y[:2], y[-2:] = feather, torch.flip(feather, dims=(0,))
    x[:2], x[-2:] = feather, torch.flip(feather, dims=(0,))
    alpha = (y[:, None] * x[None, :])[None, None, None]
    blended = alpha * native + (1.0 - alpha) * coarse_tile
    result = coarse.clone()
    if rgb.dtype == torch.uint8:
        blended = blended.round().clamp(0, 255)
    result[:, :, :, y0:y1, x0:x1] = blended.to(rgb.dtype)
    return result


def transform_observation(
    observation: Mapping[str, torch.Tensor], condition: str, tile_index: int | None = None
) -> dict[str, torch.Tensor]:
    result = {key: value.clone() for key, value in observation.items()}
    if condition == "native":
        return result
    if condition == "coarse":
        result["rgb"] = coarse_rgb(result["rgb"])
        return result
    if condition == "local_fine":
        if tile_index is None:
            raise ValueError("local_fine requires tile_index")
        result["rgb"] = local_fine_rgb(result["rgb"], tile_index)
        return result
    raise KeyError(condition)
