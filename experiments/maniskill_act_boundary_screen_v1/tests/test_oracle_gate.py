from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from summarize_oracle_gate import gate_from_task_aggregates  # noqa: E402


def task(action: bool, joint: bool, recall: float, regret: float, joint_density: float):
    return {
        "action_gate_pass": action,
        "joint_gate_pass": joint,
        "coarse_best_action_recall": 0.5,
        "oracle_best_action_recall": 0.5 + recall / 100.0,
        "coarse_outcome_regret": 1.0,
        "oracle_outcome_regret": 1.0 - regret,
        "joint_coupling_density": joint_density,
        "complete_call_and_opportunity_accounting": True,
        "per_seed": [
            {
                "joint_coupling_density": joint_density,
                "best_action_recall_improvement_percentage_points": recall,
                "outcome_regret_reduction_fraction": regret,
            }
            for _ in range(3)
        ],
    }


def test_oracle_gate_requires_two_positive_density_tasks_and_negative_control() -> None:
    tasks = {
        "A": task(True, True, 12.0, 0.0, 0.20),
        "B": task(True, True, 12.0, 0.0, 0.15),
        "PushCube-v1": task(False, False, 0.0, 0.0, 0.05),
    }
    result = gate_from_task_aggregates(["A", "B"], tasks)
    assert result["decision"] == "GO"
    tasks["B"]["joint_gate_pass"] = False
    result = gate_from_task_aggregates(["A", "B"], tasks)
    assert result["decision"] == "NO_GO"


def test_negative_control_blocks_go() -> None:
    tasks = {
        "A": task(True, True, 0.0, 0.20, 0.20),
        "B": task(True, True, 0.0, 0.20, 0.20),
        "PushCube-v1": task(False, False, 0.0, 0.0, 0.11),
    }
    result = gate_from_task_aggregates(["A", "B"], tasks)
    assert result["decision"] == "NO_GO"
