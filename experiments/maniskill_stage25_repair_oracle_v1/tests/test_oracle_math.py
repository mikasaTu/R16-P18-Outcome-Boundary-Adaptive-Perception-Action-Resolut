from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from oracle_math import (
    COARSE_INDICES,
    adjacent_edges,
    atlas_center_with_frozen_gripper,
    coarse_rgb,
    effective_gripper_command,
    holm_adjust,
    local_fine_rgb,
    local_pca_grid,
    paired_sign_flip_pvalue,
)
from stage25_runtime import neutral_from_last
from run_joint_factorial_oracle import metric_summary
from audit_stage25 import sign_flip_pvalue as independent_sign_flip_pvalue


def test_nested_grid_and_invalid_actions_are_not_clipped() -> None:
    rng = np.random.default_rng(7)
    nominal = np.zeros((30, 4), dtype=np.float32)
    chunks = rng.normal(0, 0.1, size=(300, 4, 4)).astype(np.float32)
    result = local_pca_grid(
        nominal,
        chunks,
        np.full((4, 4), -0.03),
        np.full((4, 4), 0.03),
        radius=1.5,
    )
    assert result["candidates"].shape == (25, 4, 4)
    assert len(COARSE_INDICES) == 9
    assert not result["valid"].all()
    assert np.max(result["candidates"]) > 0.03


def test_controller_effective_gripper_and_atlas_center_are_explicit() -> None:
    nominal = np.zeros((4, 4), dtype=np.float64)
    nominal[:, -1] = [1.001, 1.002, 1.003, 1.004]
    low = np.full((4, 4), -1.0)
    high = np.full((4, 4), 1.0)
    legal = effective_gripper_command(1.004, -1.0, 1.0)
    center = atlas_center_with_frozen_gripper(nominal, legal, low, high)
    assert legal == 1.0
    assert np.all(center[:, -1] == 1.0)
    assert np.all(nominal[:, -1] > 1.0)


def test_atlas_center_rejects_illegal_stored_gripper_without_clipping() -> None:
    nominal = np.zeros((4, 4), dtype=np.float64)
    with np.testing.assert_raises_regex(ValueError, "outside action-space"):
        atlas_center_with_frozen_gripper(
            nominal,
            1.001,
            np.full((4, 4), -1.0),
            np.full((4, 4), 1.0),
        )


def test_neutral_hold_retains_controller_effective_legal_gripper() -> None:
    raw = torch.tensor(
        [[0.2, -0.3, 0.4, 1.004], [-0.2, 0.3, -0.4, -1.006]],
        dtype=torch.float32,
    )
    neutral = neutral_from_last(raw)
    torch.testing.assert_close(
        neutral,
        torch.tensor(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]],
            dtype=torch.float32,
        ),
    )


def test_lattice_has_40_unique_edges() -> None:
    edges = adjacent_edges()
    assert len(edges) == 40
    assert len(set(edges)) == 40


def test_local_fine_restores_only_requested_tile_with_feather() -> None:
    yy, xx = torch.meshgrid(torch.arange(128), torch.arange(128), indexing="ij")
    rgb = (((yy + xx) % 2) * 255).to(torch.float32).reshape(1, 1, 1, 128, 128)
    coarse = coarse_rgb(rgb)
    fine = local_fine_rgb(rgb, 5)
    assert torch.equal(fine[:, :, :, :32, :], coarse[:, :, :, :32, :])
    # Interior of tile 5 is exactly native; its edge is blended.
    assert torch.equal(fine[:, :, :, 34:62, 34:62], rgb[:, :, :, 34:62, 34:62])
    assert not torch.equal(fine[:, :, :, 32, 32], rgb[:, :, :, 32, 32])


def test_sign_flip_and_holm_are_deterministic_and_monotone() -> None:
    assert paired_sign_flip_pvalue([1.0] * 16) < 0.001
    values = [1.0, -0.25, 0.5, 0.0] * 4
    assert independent_sign_flip_pvalue(values) == paired_sign_flip_pvalue(values)
    adjusted = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.5})
    assert adjusted == {"a": 0.03, "b": 0.04, "c": 0.5}


def test_joint_primary_uses_per_state_strongest_single_axis() -> None:
    rows = []
    for index, (visual, action) in enumerate(((9.0, 1.0), (1.0, 9.0))):
        rows.append(
            {
                "adaptive_selected": True,
                "categorical_strictly_better": False,
                "interaction_J": 0.0,
                "post_success": False,
                "budget_compliant": True,
                "allocation_utility": {
                    "joint_adaptive": 10.0,
                    "visual_only": visual,
                    "action_only": action,
                    "strongest_single_axis": max(visual, action),
                    "random_state": 0.0,
                    "phase_heuristic": 0.0,
                    "random_tile": 0.0,
                    "phase_tile": 0.0,
                    "uniform_coarse": 0.0,
                    "uniform_fine": 10.0,
                    "full_native_upper": 10.0,
                },
                "nearest_native_best_action_recall": {
                    "CC": 0,
                    "FC": int(index == 0),
                    "CF": int(index == 1),
                    "FF": 1,
                    "strongest_single_axis": 1,
                },
            }
        )
    result = metric_summary(rows, j_threshold=1.0)
    assert result["strongest_single_axis"] == "per_state_privileged_max_FC_CF"
    assert result["comparisons"]["strongest_single_axis"]["mean_gain"] == 1.0
    assert result["mean_outcome_regret_strongest_single"] == 1.0
    assert result["mean_outcome_regret_joint"] == 0.0
    for name in ("random_state", "phase_heuristic", "random_tile", "phase_tile"):
        assert "holm_adjusted_p" in result["comparisons"][name]
