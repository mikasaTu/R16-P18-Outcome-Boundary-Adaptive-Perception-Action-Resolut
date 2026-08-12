#!/usr/bin/env python3
"""Read-only recomputation of the archived LIBERO baseline gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boundarybc.config import load_config


RUN_ID = "r16-p18-libero-stage1-bc-gate-20260812-003"
RUN_ROOT = ROOT / "artifacts" / "formal-run" / RUN_ID
REPORT_PATH = RUN_ROOT / "reports" / "baseline_gate.json"


def _read_records(path: Path) -> dict[int, dict[str, object]]:
    records: dict[int, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            episode_id = int(value["episode_id"])
            if episode_id in records and records[episode_id] != value:
                raise RuntimeError(f"conflicting duplicate episode {episode_id}: {path}")
            records[episode_id] = value
    if set(records) != set(range(50)):
        raise RuntimeError(f"expected episode IDs 0..49: {path}")
    return records


def _paired_seed_episode_bootstrap(
    matrix: np.ndarray, *, replicates: int, seed: int
) -> tuple[float, float]:
    if matrix.ndim != 2 or matrix.shape[1] != 50:
        raise ValueError(f"expected seed x 50 matrix, got {matrix.shape}")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        seed_indices = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        episode_indices = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        estimates[index] = matrix[np.ix_(seed_indices, episode_indices)].mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def main() -> None:
    config = load_config(ROOT / "configs" / "r16_p18_libero_stage1.yaml")
    expected = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    task_results: dict[str, object] = {}
    total_episodes = 0
    total_successes = 0
    total_calls = 0
    total_steps = 0
    total_inference_seconds = 0.0
    total_wall_seconds = 0.0

    for task_index, task in enumerate(config.tasks):
        matrix = np.zeros((len(config.training_seeds), 50), dtype=np.float64)
        records_for_task: list[dict[str, object]] = []
        seed_rates: dict[str, float] = {}
        for seed_index, seed in enumerate(config.training_seeds):
            path = RUN_ROOT / "evaluation" / task.key / f"seed_{seed}.jsonl"
            records = _read_records(path)
            ordered = [records[index] for index in range(50)]
            records_for_task.extend(ordered)
            values = np.asarray([bool(row["success"]) for row in ordered], dtype=np.float64)
            matrix[seed_index] = values
            seed_rates[str(seed)] = float(values.mean())

        rate = float(matrix.mean())
        low, high = _paired_seed_episode_bootstrap(
            matrix,
            replicates=int(config.raw["stage1_go_thresholds"]["paired_bootstrap_replicates"]),
            seed=1601800 + task_index,
        )
        archived = expected["task_results"][task.key]
        if not np.isclose(rate, archived["success_rate"], rtol=0.0, atol=1e-15):
            raise RuntimeError(f"success-rate mismatch for {task.key}")
        if not np.allclose(
            [low, high], archived["confidence_interval_95"], rtol=0.0, atol=1e-15
        ):
            raise RuntimeError(f"bootstrap-CI mismatch for {task.key}")
        if seed_rates != archived["seed_success_rates"]:
            raise RuntimeError(f"seed-rate mismatch for {task.key}")

        successes = sum(bool(row["success"]) for row in records_for_task)
        calls = sum(int(row["policy_calls"]) for row in records_for_task)
        steps = sum(int(row["executed_steps"]) for row in records_for_task)
        inference_seconds = sum(float(row["inference_seconds"]) for row in records_for_task)
        wall_seconds = sum(float(row["wall_seconds"]) for row in records_for_task)
        task_results[task.key] = {
            "episodes": len(records_for_task),
            "successes": successes,
            "success_rate": rate,
            "confidence_interval_95": [low, high],
            "seed_success_rates": seed_rates,
            "policy_calls": calls,
            "executed_steps": steps,
            "inference_seconds": inference_seconds,
            "milliseconds_per_policy_call": 1000.0 * inference_seconds / calls,
            "wall_seconds": wall_seconds,
            "gate_passed": bool(archived["passed"]),
        }
        total_episodes += len(records_for_task)
        total_successes += successes
        total_calls += calls
        total_steps += steps
        total_inference_seconds += inference_seconds
        total_wall_seconds += wall_seconds

    if total_episodes != 450:
        raise RuntimeError(f"expected 450 episodes, found {total_episodes}")
    if expected["decision"] != "NO_GO_BASELINE_GATE":
        raise RuntimeError(f"unexpected archived decision: {expected['decision']}")

    print(
        json.dumps(
            {
                "verified": True,
                "run_id": RUN_ID,
                "decision": expected["decision"],
                "tasks": task_results,
                "aggregate": {
                    "episodes": total_episodes,
                    "successes": total_successes,
                    "success_rate": total_successes / total_episodes,
                    "policy_calls": total_calls,
                    "executed_steps": total_steps,
                    "inference_seconds": total_inference_seconds,
                    "milliseconds_per_policy_call": 1000.0
                    * total_inference_seconds
                    / total_calls,
                    "wall_seconds_sum": total_wall_seconds,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
