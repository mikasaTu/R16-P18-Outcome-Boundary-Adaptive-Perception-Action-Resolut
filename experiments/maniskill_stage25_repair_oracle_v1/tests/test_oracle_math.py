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
    coarse_rgb,
    local_fine_rgb,
    local_pca_grid,
)


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
