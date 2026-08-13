from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from protocol_common import SPLIT_COUNTS  # noqa: E402
from run_formal_matrix import TASKS, replay_jobs, validate_replay_gate  # noqa: E402


def write_summaries(root: Path, saved: dict[str, int] | None = None) -> None:
    summary_root = root / "replay_summaries"
    summary_root.mkdir(parents=True)
    saved = saved or {"train": 190, "validation": 48, "test": 48}
    for task_id in TASKS:
        records = [
            {
                "status": "PASS",
                "split": split,
                "episodes_attempted": expected,
                "episodes_saved_successful": saved[split],
                "replay_success_rate": saved[split] / expected,
            }
            for split, expected in SPLIT_COUNTS.items()
        ]
        attempted_total = sum(SPLIT_COUNTS.values())
        saved_total = sum(saved.values())
        (summary_root / f"{task_id}.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "task_id": task_id,
                    "splits": records,
                    "episodes_attempted": attempted_total,
                    "episodes_saved_successful": saved_total,
                    "replay_success_rate": saved_total / attempted_total,
                }
            ),
            encoding="utf-8",
        )


def test_matrix_gate_accepts_each_split_at_preregistered_minimum(
    tmp_path: Path,
) -> None:
    write_summaries(tmp_path)
    values = validate_replay_gate(argparse.Namespace(selected_raw_root=tmp_path))
    assert len(values) == len(TASKS)


def test_matrix_gate_rejects_one_split_below_preregistered_minimum(
    tmp_path: Path,
) -> None:
    write_summaries(tmp_path, {"train": 189, "validation": 48, "test": 48})
    with pytest.raises(RuntimeError, match="formal replay gate failed|split gate"):
        validate_replay_gate(argparse.Namespace(selected_raw_root=tmp_path))


def test_replay_retry_budget_is_task_pinned(tmp_path: Path) -> None:
    jobs = replay_jobs(
        argparse.Namespace(selected_raw_root=tmp_path, python=Path("/pinned/python"))
    )
    observed = {}
    for job in jobs:
        task_id = job.name.removeprefix("replay_")
        retry_index = job.command.index("--max-retry") + 1
        observed[task_id] = int(job.command[retry_index])
    assert observed == {
        "PullCubeTool-v1": 9,
        "PushT-v1": 3,
        "StackCube-v1": 9,
        "PushCube-v1": 9,
    }
