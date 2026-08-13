#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


GRID_LEVELS = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
GRID_INDICES = tuple((row, col) for row in range(5) for col in range(5))
COARSE_GRID_INDICES = tuple((row, col) for row in (0, 2, 4) for col in (0, 2, 4))
PHASES = (
    "free_space",
    "pre_contact_or_pre_grasp",
    "contact_insertion_or_placement",
    "near_completion",
)
CATEGORY_FIELDS = (
    "short_horizon_success",
    "intended_contact",
    "unintended_contact",
    "collision",
    "recoverable",
)


@dataclass(frozen=True)
class PhaseCandidate:
    task_id: str
    trajectory_id: int
    episode_seed: int
    timestep: int
    phase: str
    predicate_source: str

    @property
    def selection_sha256(self) -> str:
        payload = f"{self.task_id}{self.trajectory_id}{self.timestep}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _future_any(values: np.ndarray, timestep: int, width: int = 4) -> bool:
    stop = min(len(values), timestep + width + 1)
    return bool(np.any(values[timestep:stop]))


def phase_candidate_rows(
    *,
    task_id: str,
    trajectory_id: int,
    episode_seed: int,
    action_count: int,
    intended_contact_by_state: Sequence[bool],
    success_by_state: Sequence[bool],
) -> list[PhaseCandidate]:
    """Return frozen primary and temporal-fallback candidates for one trajectory.

    State index t has actions t..t+3 available, so the final eligible state is
    action_count-4.  Primary predicates are mutually exclusive by construction.
    """

    contact = np.asarray(intended_contact_by_state, dtype=bool)
    success = np.asarray(success_by_state, dtype=bool)
    expected = action_count + 1
    if contact.shape != (expected,) or success.shape != (expected,):
        raise ValueError(
            f"state-label length mismatch: expected={expected}, "
            f"contact={contact.shape}, success={success.shape}"
        )
    if action_count < 4:
        return []

    result: list[PhaseCandidate] = []
    for timestep in range(action_count - 3):
        fraction = timestep / max(action_count, 1)
        near = _future_any(success, timestep, 4)
        future_contact = _future_any(contact, timestep + 1, 3)
        current_contact = bool(contact[timestep])
        no_contact_window = not _future_any(contact, timestep, 4)

        primary_phase: str | None = None
        if near:
            primary_phase = "near_completion"
        elif current_contact:
            primary_phase = "contact_insertion_or_placement"
        elif future_contact:
            primary_phase = "pre_contact_or_pre_grasp"
        elif no_contact_window and fraction <= 0.50:
            primary_phase = "free_space"
        if primary_phase is not None:
            result.append(
                PhaseCandidate(
                    task_id,
                    trajectory_id,
                    episode_seed,
                    timestep,
                    primary_phase,
                    "primary",
                )
            )

        if fraction < 0.25:
            fallback_phase = "free_space"
        elif fraction < 0.50:
            fallback_phase = "pre_contact_or_pre_grasp"
        elif fraction < 0.80:
            fallback_phase = "contact_insertion_or_placement"
        else:
            fallback_phase = "near_completion"
        if fallback_phase != primary_phase:
            result.append(
                PhaseCandidate(
                    task_id,
                    trajectory_id,
                    episode_seed,
                    timestep,
                    fallback_phase,
                    "fixed_temporal_fallback",
                )
            )
    return result


def ordered_phase_candidates(
    candidates: Iterable[PhaseCandidate], phase: str
) -> list[PhaseCandidate]:
    if phase not in PHASES:
        raise KeyError(phase)
    source_priority = {"primary": 0, "fixed_temporal_fallback": 1}
    rows = [candidate for candidate in candidates if candidate.phase == phase]
    rows.sort(
        key=lambda candidate: (
            source_priority[candidate.predicate_source],
            candidate.selection_sha256,
        )
    )
    # The contract permits one state from a source trajectory per stratum.
    result: list[PhaseCandidate] = []
    used_trajectories: set[int] = set()
    for candidate in rows:
        if candidate.trajectory_id in used_trajectories:
            continue
        used_trajectories.add(candidate.trajectory_id)
        result.append(candidate)
    return result


def pca_local_action_grid(
    base_chunk: np.ndarray,
    training_chunks: np.ndarray,
    non_gripper_dimensions: Sequence[int],
    *,
    neighbor_count: int = 256,
    std_floor: float = 1e-3,
) -> dict[str, Any]:
    """Build the preregistered 5x5 first-four-action local PCA atlas."""

    base = np.asarray(base_chunk, dtype=np.float64)
    chunks = np.asarray(training_chunks, dtype=np.float64)
    dims = np.asarray(tuple(non_gripper_dimensions), dtype=np.int64)
    if base.ndim != 2 or base.shape[0] < 4:
        raise ValueError(f"base chunk must be [>=4, action_dim], got {base.shape}")
    if chunks.ndim != 3 or chunks.shape[1] != 4 or chunks.shape[2] != base.shape[1]:
        raise ValueError(
            f"training chunks must be [N,4,{base.shape[1]}], got {chunks.shape}"
        )
    if len(dims) < 1 or len(set(dims.tolist())) != len(dims):
        raise ValueError("non-gripper dimensions must be unique and non-empty")
    if chunks.shape[0] < min(neighbor_count, 2):
        raise ValueError("not enough training chunks")

    selected = chunks[:, :4, :][:, :, dims].reshape(chunks.shape[0], -1)
    base_vector = base[:4, dims].reshape(-1)
    coordinate_std = np.maximum(selected.std(axis=0, ddof=1), std_floor)
    standardized = selected / coordinate_std
    standardized_base = base_vector / coordinate_std
    distances = np.linalg.norm(standardized - standardized_base[None], axis=1)
    count = min(neighbor_count, chunks.shape[0])
    neighbor_indices = np.argsort(distances, kind="stable")[:count]
    residuals = standardized[neighbor_indices] - standardized_base[None]
    centered = residuals - residuals.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    if vh.shape[0] < 2:
        raise ValueError("local neighborhood has fewer than two PCA directions")
    directions = vh[:2].copy()
    # Canonical signs make identical inputs byte-stable across SVD sign choices.
    for index in range(2):
        pivot = int(np.argmax(np.abs(directions[index])))
        if directions[index, pivot] < 0:
            directions[index] *= -1
    scores = centered @ directions.T
    score_scales = scores.std(axis=0, ddof=1)
    score_scales = np.maximum(score_scales, std_floor)
    physical_directions = directions * coordinate_std[None]

    candidates: list[np.ndarray] = []
    residual_vectors: list[np.ndarray] = []
    scaled_norms: list[float] = []
    for row, col in GRID_INDICES:
        standardized_residual = (
            GRID_LEVELS[row] * score_scales[0] * directions[0]
            + GRID_LEVELS[col] * score_scales[1] * directions[1]
        )
        physical_residual = standardized_residual * coordinate_std
        candidate = base[:4].copy()
        candidate[:, dims] += physical_residual.reshape(4, len(dims))
        candidates.append(candidate)
        residual_vectors.append(physical_residual)
        scaled_norms.append(float(np.linalg.norm(standardized_residual)))
    return {
        "candidates": np.stack(candidates),
        "physical_residuals": np.stack(residual_vectors),
        "scaled_residual_norms": np.asarray(scaled_norms),
        "neighbor_indices": neighbor_indices,
        "neighbor_distances": distances[neighbor_indices],
        "coordinate_std": coordinate_std,
        "standardized_directions": directions,
        "physical_directions": physical_directions,
        "score_scales": score_scales,
    }


def quaternion_distance_rad(first: Sequence[float], second: Sequence[float]) -> float:
    q1 = np.asarray(first, dtype=np.float64)
    q2 = np.asarray(second, dtype=np.float64)
    q1 = q1 / max(np.linalg.norm(q1), np.finfo(np.float64).eps)
    q2 = q2 / max(np.linalg.norm(q2), np.finfo(np.float64).eps)
    cosine = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return 2.0 * math.acos(cosine)


def category_tuple(outcome: Mapping[str, Any]) -> tuple[bool, ...]:
    return tuple(bool(outcome[field]) for field in CATEGORY_FIELDS)


def continuous_effect_distance(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> float:
    p1 = np.asarray(first["object_delta_translation_m"], dtype=np.float64)
    p2 = np.asarray(second["object_delta_translation_m"], dtype=np.float64)
    translation = np.linalg.norm(p1 - p2) / 0.01
    rotation = abs(
        float(first["object_delta_rotation_rad"])
        - float(second["object_delta_rotation_rad"])
    ) / math.radians(10.0)
    progress = abs(
        float(first["normalized_progress_delta"])
        - float(second["normalized_progress_delta"])
    ) / 0.05
    return float(np.sqrt(np.mean(np.square([translation, rotation, progress]))))


def is_boundary(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return bool(
        category_tuple(first) != category_tuple(second)
        or continuous_effect_distance(first, second) >= 1.0
    )


def is_alias(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return bool(
        category_tuple(first) == category_tuple(second)
        and continuous_effect_distance(first, second) < 0.25
    )


def outcome_utility(outcome: Mapping[str, Any], scaled_residual_norm: float) -> float:
    return float(
        100.0 * bool(outcome["short_horizon_success"])
        + 10.0 * np.clip(float(outcome["normalized_progress_delta"]), -1.0, 1.0)
        + 2.0 * bool(outcome["intended_contact"])
        - 5.0 * bool(outcome["unintended_contact"])
        - 5.0 * bool(outcome["collision"])
        + 1.0 * bool(outcome["recoverable"])
        - 0.01 * float(scaled_residual_norm)
    )


def adjacent_grid_edges() -> tuple[tuple[int, int], ...]:
    edges: list[tuple[int, int]] = []
    for row in range(5):
        for col in range(5):
            index = row * 5 + col
            if row + 1 < 5:
                edges.append((index, (row + 1) * 5 + col))
            if col + 1 < 5:
                edges.append((index, row * 5 + col + 1))
    if len(edges) != 40:  # pragma: no cover - defensive invariant
        raise AssertionError(len(edges))
    return tuple(edges)


def best_index(utilities: Sequence[float], valid_indices: Sequence[int] | None = None) -> int:
    values = np.asarray(utilities, dtype=np.float64)
    indices = list(range(len(values))) if valid_indices is None else list(valid_indices)
    if not indices:
        raise ValueError("candidate set is empty")
    # max() preserves the first (lexicographically smallest flat grid index) tie.
    return max(indices, key=lambda index: (values[index], -index))


def candidate_recall_regret(
    utilities: Sequence[float], candidate_indices: Sequence[int]
) -> tuple[float, float]:
    full_best = best_index(utilities)
    candidate_best = best_index(utilities, candidate_indices)
    recall = float(full_best in set(candidate_indices))
    regret = float(np.asarray(utilities)[full_best] - np.asarray(utilities)[candidate_best])
    return recall, max(0.0, regret)


def paired_percentile_ci(
    values: Sequence[float], *, replicates: int = 10_000, seed: int = 16018
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("paired bootstrap expects one non-empty vector")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(replicates, array.size))
    estimates = array[draws].mean(axis=1)
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]
