from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from oracle_common import (  # noqa: E402
    COARSE_GRID_INDICES,
    adjacent_grid_edges,
    candidate_recall_regret,
    continuous_effect_distance,
    ordered_phase_candidates,
    pca_local_action_grid,
    phase_candidate_rows,
)


def test_phase_predicates_and_hash_order_are_deterministic() -> None:
    contact = [False] * 13
    contact[5:9] = [True] * 4
    success = [False] * 13
    success[11:] = [True, True]
    rows = phase_candidate_rows(
        task_id="Example-v1",
        trajectory_id=7,
        episode_seed=11,
        action_count=12,
        intended_contact_by_state=contact,
        success_by_state=success,
    )
    primary = {row.timestep: row.phase for row in rows if row.predicate_source == "primary"}
    assert primary[0] == "free_space"
    assert primary[1] == "pre_contact_or_pre_grasp"
    assert primary[5] == "contact_insertion_or_placement"
    assert primary[7] == "near_completion"
    assert len(ordered_phase_candidates(rows, "free_space")) == 1


def test_pca_grid_preserves_gripper_and_has_zero_center() -> None:
    rng = np.random.default_rng(16018)
    training = rng.normal(0, 0.2, size=(300, 4, 4))
    training[:, :, 3] = 0.75
    base = np.zeros((30, 4), dtype=np.float64)
    base[:, 3] = -0.25
    atlas = pca_local_action_grid(base, training, (0, 1, 2))
    assert atlas["candidates"].shape == (25, 4, 4)
    np.testing.assert_allclose(atlas["candidates"][:, :, 3], -0.25)
    np.testing.assert_allclose(atlas["candidates"][12], base[:4], atol=1e-12)
    assert len(set(atlas["neighbor_indices"].tolist())) == 256


def test_boundary_grid_and_recall_regret_contract() -> None:
    assert len(adjacent_grid_edges()) == 40
    coarse = [row * 5 + col for row, col in COARSE_GRID_INDICES]
    utilities = np.zeros(25)
    utilities[6] = 2.0
    utilities[0] = 1.0
    recall, regret = candidate_recall_regret(utilities, coarse)
    assert recall == 0.0
    assert regret == 1.0


def test_continuous_effect_scaling() -> None:
    base = {
        "object_delta_translation_m": [0.0, 0.0, 0.0],
        "object_delta_rotation_rad": 0.0,
        "normalized_progress_delta": 0.0,
    }
    shifted = dict(base)
    shifted["object_delta_translation_m"] = [0.01, 0.0, 0.0]
    assert np.isclose(continuous_effect_distance(base, shifted), 1 / np.sqrt(3))
