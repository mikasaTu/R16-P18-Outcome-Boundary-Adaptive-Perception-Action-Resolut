from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from oracle_visual import (  # noqa: E402
    INTERVENTIONS,
    select_joint_visual_pairs,
    visual_intervention_batch,
)


def test_visual_batch_shape_order_and_locality() -> None:
    rgb = torch.arange(8 * 8, dtype=torch.uint8).reshape(1, 1, 1, 8, 8)
    state = torch.tensor([[1.0, 2.0]])
    batch, metadata = visual_intervention_batch({"rgb": rgb, "state": state})
    assert batch["rgb"].shape == (48, 1, 1, 8, 8)
    assert batch["state"].shape == (48, 2)
    assert [row["intervention"] for row in metadata[:3]] == list(INTERVENTIONS)
    # First intervention changes only the first 2x2 tile.
    assert torch.equal(batch["rgb"][0, :, :, 2:, :], rgb[0, :, :, 2:, :])
    assert torch.equal(batch["rgb"][0, :, :, :2, 2:], rgb[0, :, :, :2, 2:])


def test_joint_visual_selection_uses_distinct_tiles_and_fixed_ties() -> None:
    rgb = torch.zeros((1, 1, 3, 8, 8), dtype=torch.uint8)
    batch, metadata = visual_intervention_batch({"rgb": rgb, "state": torch.zeros(1, 2)})
    changes = [0.0] * 48
    changes[3 * 3 + 2] = 4.0
    changes[8 * 3 + 1] = 3.0
    changes[1 * 3 + 0] = 2.0
    changes[14 * 3 + 0] = 1.0
    indices = select_joint_visual_pairs(metadata, changes)
    assert [metadata[index]["tile_index"] for index in indices] == [3, 8, 1, 14]
    assert batch["rgb"].shape[0] == 48
