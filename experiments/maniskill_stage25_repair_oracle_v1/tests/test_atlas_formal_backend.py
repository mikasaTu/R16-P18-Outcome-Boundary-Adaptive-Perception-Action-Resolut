from __future__ import annotations

import ast
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


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
