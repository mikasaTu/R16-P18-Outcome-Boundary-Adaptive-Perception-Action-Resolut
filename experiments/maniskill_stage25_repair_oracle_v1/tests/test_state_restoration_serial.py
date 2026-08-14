from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_state_restoration as restoration


class FakeEnv:
    def __init__(self) -> None:
        self.action_space = SimpleNamespace(shape=(4,))
        self.base_env = SimpleNamespace(
            device=torch.device("cpu"),
            get_state_dict=lambda: {"position": torch.tensor([[1.0, 2.0]])},
        )
        self.actions: list[torch.Tensor] = []

    def step(self, action: torch.Tensor):
        self.actions.append(action.clone())
        return None, None, None, None, {"success": torch.tensor([False])}


def test_restoration_uses_one_environment_and_three_serial_resets(monkeypatch) -> None:
    env = FakeEnv()
    reset_counts: list[int] = []

    def fake_reset(_env, _state, _seed, count):
        reset_counts.append(count)
        return {}, {}

    monkeypatch.setattr(restoration, "reset_to_state", fake_reset)
    monkeypatch.setattr(
        restoration,
        "task_snapshot",
        lambda _base, _task: {
            "success": np.array([False]),
            "grasped": np.array([False]),
            "supported": np.array([False]),
        },
    )
    monkeypatch.setattr(restoration, "stack_predicates", lambda _base: {})
    monkeypatch.setattr(restoration, "stack_phase", lambda _predicates, _index: "free")

    result = restoration.audit_serial_repeats(
        env,
        {"position": np.array([1.0, 2.0])},
        {"source_episode_seed": 17, "last_legal_gripper_command": -0.25},
    )

    assert reset_counts == [1, 1, 1]
    assert len(env.actions) == 12
    assert all(tuple(action.shape) == (1, 4) for action in env.actions)
    assert all(float(action[0, -1]) == -0.25 for action in env.actions)
    assert result["restore_errors"] == [0.0, 0.0, 0.0]
    assert result["final_difference"] == 0.0
    assert result["categorical_agreement"] is True
    assert len(result["categories"]) == 3
