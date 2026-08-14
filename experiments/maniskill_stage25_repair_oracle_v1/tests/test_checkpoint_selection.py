from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from select_checkpoint_closed_loop import selection_key, spearman


def row(step: int, hold: float, end: float, loss: float) -> dict:
    return {
        "step": step,
        "success_hold5": hold,
        "success_at_end": end,
        "post_success_loss": loss,
    }


def test_lexicographic_selection_rule() -> None:
    rows = [
        row(5000, 0.4, 0.3, 0.1),
        row(10000, 0.5, 0.2, 0.0),
        row(15000, 0.5, 0.4, 0.2),
        row(20000, 0.5, 0.4, 0.1),
        row(25000, 0.5, 0.4, 0.1),
    ]
    assert min(rows, key=selection_key)["step"] == 20000


def test_spearman_handles_ties_and_constant() -> None:
    assert spearman([1, 2, 3], [1, 2, 3]) == 1.0
    assert spearman([1, 1, 1], [1, 2, 3]) is None
