from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from smoke_action_atlas_cuda import validate_smoke_atlas
from audit_success_trace_terminal import terminal_from_trace
from audit_mechanisms import comparison_summary, physical_signature


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


def test_every_atlas_call_binds_the_state_bank_legal_gripper() -> None:
    for script_name in (
        "run_action_boundary_probe.py",
        "run_visual_resolution_probe.py",
        "smoke_action_atlas_cuda.py",
    ):
        tree = ast.parse((ROOT / "scripts" / script_name).read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "generate_atlas"
        ]
        assert calls
        assert all(
            any(
                keyword.arg == "last_legal_gripper_command"
                for keyword in call.keywords
            )
            for call in calls
        )


def test_parallel_state_bank_reads_unbatched_gripper_bounds() -> None:
    source = (ROOT / "scripts" / "build_stackcube_state_bank.py").read_text(
        encoding="utf-8"
    )
    assert "env.single_action_space.low[-1]" in source
    assert "env.single_action_space.high[-1]" in source
    assert "env.action_space.low[-1]" not in source
    assert "env.action_space.high[-1]" not in source


def test_success_semantics_trace_keeps_policy_neutral_contact_and_drift_fields() -> None:
    source = (ROOT / "scripts" / "stage25_runtime.py").read_text(encoding="utf-8")
    for field in (
        '"success_predicate"',
        '"intended_contact_onset"',
        '"unintended_contact_onset"',
        '"post_success_object_drift"',
        '"executed_action"',
        '"policy_action"',
        '"neutral_action"',
    ):
        assert field in source


def test_terminal_trace_audit_uses_episode_terminal_not_vector_final_snapshot() -> None:
    row = {
        "episode_length": 3,
        "first_success_step": 2,
        "success_once": True,
        "trace": [
            {
                "step": 1,
                "success_predicate": False,
                "object_position": [0.0, 0.0, 0.0],
                "object_quaternion": [1.0, 0.0, 0.0, 0.0],
                "post_success_object_drift": None,
            },
            {
                "step": 2,
                "success_predicate": True,
                "object_position": [0.0, 0.0, 0.0],
                "object_quaternion": [1.0, 0.0, 0.0, 0.0],
                "post_success_object_drift": {
                    "translation_m": 0.0,
                    "rotation_rad": 0.0,
                },
            },
            {
                "step": 3,
                "success_predicate": True,
                "object_position": [0.003, 0.004, 0.0],
                "object_quaternion": [1.0, 0.0, 0.0, 0.0],
                "post_success_object_drift": {
                    "translation_m": 0.005,
                    "rotation_rad": 0.0,
                },
            },
        ],
    }
    result = terminal_from_trace(row)
    assert result["terminal_step"] == 3
    assert result["final_object_position"] == [0.003, 0.004, 0.0]
    assert result["drift"]["translation_m"] == pytest.approx(0.005)
    assert result["drift"]["to_step"] == 3


def test_mechanism_audit_counts_executed_physical_outcome_changes() -> None:
    base = {
        "stable_success": False,
        "phase_outcome": "regressed",
        "grasped": False,
        "supported": False,
        "dropped_or_slipped": False,
        "recoverable": True,
        "intended_contact": False,
        "unintended_contact": False,
    }
    refined = {**base, "phase_outcome": "progressed", "grasped": True}
    rows = [
        {
            "arms": {
                "CC": {"best_index": 12, "utility": 1.0, "outcome": base},
                "FC": {"best_index": 13, "utility": 2.5, "outcome": refined},
            }
        }
    ]
    result = comparison_summary(rows, "FC", "CC")
    assert physical_signature(base) != physical_signature(refined)
    assert result["physical_signature_change_count"] == 1
    assert result["phase_outcome_change_count"] == 1
    assert result["mean_utility_delta"] == pytest.approx(1.5)
