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
    local_fine_rgb,
    local_pca_grid,
)
from stage25_runtime import neutral_from_last


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
