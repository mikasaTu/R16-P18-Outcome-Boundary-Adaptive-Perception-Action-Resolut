from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from smoke_action_atlas_cuda import validate_smoke_atlas


def padded_rollout_backends(script_name: str) -> list[str]:
    source = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
    tree = ast.parse(source)
    result: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "make_env":
            continue
        if len(node.args) < 2 or not isinstance(node.args[1], ast.Name):
            continue
        if node.args[1].id != "PADDED_ENVS":
            continue
        keyword = next(
            (item for item in node.keywords if item.arg == "sim_backend"), None
        )
        assert keyword is not None and isinstance(keyword.value, ast.Constant)
        result.append(str(keyword.value.value))
    return result


def test_parallel_atlas_rollouts_use_frozen_formal_cuda_backend() -> None:
    protocol = yaml.safe_load(
        (ROOT / "preregistration.yaml").read_text(encoding="utf-8")
    )
    assert protocol["environment"]["formal_sim_backend"] == "physx_cuda"
    assert padded_rollout_backends("run_action_boundary_probe.py") == ["physx_cuda"]
    assert padded_rollout_backends("run_visual_resolution_probe.py") == ["physx_cuda"]


def test_cuda_smoke_rejects_vacuous_zero_valid_candidates() -> None:
    with pytest.raises(RuntimeError, match="inconsistent outcomes"):
        validate_smoke_atlas(
            {"valid": [False] * 25, "outcomes": [None] * 25}
        )
