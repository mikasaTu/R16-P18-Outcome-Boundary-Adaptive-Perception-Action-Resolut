from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from stage25_runtime import ContactTracker


def tracker_fixture(sequence: list[tuple[float, float]]) -> ContactTracker:
    tracker = object.__new__(ContactTracker)
    tracker.threshold = 1e-4
    tracker.base = type("Base", (), {"device": torch.device("cpu")})()
    tracker.previous_intended = torch.zeros(1, dtype=torch.bool)
    tracker.previous_unintended = torch.zeros(1, dtype=torch.bool)
    tracker.intended_onsets = torch.zeros(1, dtype=torch.int64)
    tracker.unintended_onsets = torch.zeros(1, dtype=torch.int64)
    tracker.intended_duration = torch.zeros(1, dtype=torch.int64)
    tracker.unintended_duration = torch.zeros(1, dtype=torch.int64)
    tracker.max_intended_force = torch.zeros(1)
    tracker.max_unintended_force = torch.zeros(1)
    tracker.post_success_onsets = torch.zeros(1, dtype=torch.int64)
    tracker.success_seen_before_step = torch.zeros(1, dtype=torch.bool)
    iterator = iter(sequence)
    tracker.forces = lambda: tuple(torch.tensor([value]) for value in next(iterator))
    return tracker


def test_contact_onsets_duration_force_and_post_success_are_distinct() -> None:
    tracker = tracker_fixture(
        [(0.0, 0.0), (2.0, 0.0), (3.0, 0.0), (0.0, 0.0), (0.0, 4.0)]
    )
    tracker.update(success_seen=torch.tensor([False]))
    tracker.update(success_seen=torch.tensor([False]))
    tracker.update(success_seen=torch.tensor([True]))
    tracker.update(success_seen=torch.tensor([True]))
    tracker.update(success_seen=torch.tensor([True]))
    fields = tracker.episode_fields(0)
    assert fields["intended_contact_onsets"] == 1
    assert fields["unintended_contact_onsets"] == 1
    assert fields["intended_contact_duration_steps"] == 2
    assert fields["unintended_contact_duration_steps"] == 1
    assert fields["max_intended_contact_force"] == 3.0
    assert fields["max_unintended_contact_force"] == 4.0
    assert fields["post_success_contact_onsets"] == 1
