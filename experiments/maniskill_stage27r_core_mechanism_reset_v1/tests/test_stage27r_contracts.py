from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import atomic_json  # noqa: E402
from analyze_stage27r import holm, paired_summary  # noqa: E402
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


def test_paired_summary_clusters_model_seeds_by_source_episode() -> None:
    result = paired_summary([1.0, 3.0, -1.0, 1.0], [("task", "ep0"), ("task", "ep0"), ("task", "ep1"), ("task", "ep1")], n=1000)
    assert result["cluster_count"] == 2
    assert result["mean"] == pytest.approx(1.0)


def test_holm_is_monotone_in_sorted_pvalues() -> None:
    result = holm({"a": .01, "b": .03, "c": .2})
    assert result["a"] <= result["b"] <= result["c"]
