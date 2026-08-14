from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_visual_resolution_probe import native_rows_at_radius


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_native_action_rows_are_filtered_before_indexing(tmp_path: Path) -> None:
    path = tmp_path / "states.jsonl"
    write_rows(
        path,
        [
            {"bank_id": bank_id, "atlas": {"radius": radius}}
            for bank_id in ("a", "b")
            for radius in (0.5, 1.0, 1.5)
        ],
    )
    selected = native_rows_at_radius(path, 1.0, {"a", "b"})
    assert set(selected) == {"a", "b"}
    assert all(row["atlas"]["radius"] == 1.0 for row in selected.values())


def test_native_action_radius_requires_exact_complete_state_set(tmp_path: Path) -> None:
    path = tmp_path / "states.jsonl"
    write_rows(path, [{"bank_id": "a", "atlas": {"radius": 1.0}}])
    with pytest.raises(RuntimeError, match="incomplete"):
        native_rows_at_radius(path, 1.0, {"a", "b"})


def test_phase_state_and_phase_tile_controls_are_separate() -> None:
    source = (ROOT / "scripts" / "run_joint_factorial_oracle.py").read_text(
        encoding="utf-8"
    )
    assert '"phase_heuristic": arms["FF"]["utility"]' in source
    assert '"phase_tile": arms["phase_FF"]["utility"]' in source
