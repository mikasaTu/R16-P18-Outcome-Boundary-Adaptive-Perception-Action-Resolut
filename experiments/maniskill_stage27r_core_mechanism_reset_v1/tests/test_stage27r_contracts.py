from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import atomic_json  # noqa: E402
from multires_policy import crop_tile  # noqa: E402


def test_crop_tiles_partition_exactly() -> None:
    image = torch.arange(128 * 128).reshape(1, 1, 128, 128)
    tiles = [crop_tile(image, tile, 2) for tile in range(4)]
    restored = torch.cat([torch.cat(tiles[:2], -1), torch.cat(tiles[2:], -1)], -2)
    assert torch.equal(restored, image)


def test_crop_validation() -> None:
    with pytest.raises(ValueError):
        crop_tile(torch.zeros(1, 1, 128, 128), 4, 2)
    with pytest.raises(ValueError):
        crop_tile(torch.zeros(1, 1, 128, 128), 0, 3)


def test_fail_on_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    atomic_json(path, {"a": 1})
    with pytest.raises(FileExistsError):
        atomic_json(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}


def test_protocol_freeze_has_required_status_order() -> None:
    import yaml
    value = yaml.safe_load((ROOT / "preregistration.yaml").read_text())
    assert value["final_status"]["precedence"] == [
        "NO_GO_CAUSAL_BACKEND", "NO_GO_CORE_MECHANISM", "REVISE_VISUAL_ONLY",
        "REVISE_SHARED_AXIS_ROUTER", "GO_FULL_JOINT",
    ]
    assert value["execution_contract"]["run_all_oracle_arms_after_any_gate_failure"] is True
