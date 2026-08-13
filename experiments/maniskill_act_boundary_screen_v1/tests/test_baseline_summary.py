from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from summarize_baseline import paired_episode_bootstrap  # noqa: E402


def test_paired_bootstrap_is_deterministic_and_degenerate_for_constant_data() -> None:
    success = np.ones((3, 100), dtype=np.float64)
    assert paired_episode_bootstrap(success) == [1.0, 1.0]
    mixed = np.zeros((3, 100), dtype=np.float64)
    mixed[:, :50] = 1.0
    assert paired_episode_bootstrap(mixed) == paired_episode_bootstrap(mixed)
