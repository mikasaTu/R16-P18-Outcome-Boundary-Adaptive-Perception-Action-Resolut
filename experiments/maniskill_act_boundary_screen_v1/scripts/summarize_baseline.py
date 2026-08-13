#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from protocol_common import (  # noqa: E402
    FORMAL_TASKS,
    MODEL_SEEDS,
    PROTOCOL_ID,
    sha256_file,
    write_json,
)


POSITIVE_TASKS = tuple(task for task in FORMAL_TASKS if task != "PushCube-v1")
NEGATIVE_CONTROL = "PushCube-v1"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 16018


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_episodes(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 fixed episodes in {path}, found {len(rows)}")
    seeds = [int(row["episode_seed"]) for row in rows]
    if len(set(seeds)) != 100:
        raise RuntimeError(f"duplicate fixed episode seed in {path}")
    return rows


def paired_episode_bootstrap(success: np.ndarray) -> list[float]:
    """Resample the 100 shared episode identities, retaining all model seeds."""

    if success.shape != (len(MODEL_SEEDS), 100):
        raise ValueError(success.shape)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        episode_indices = rng.integers(0, success.shape[1], size=success.shape[1])
        estimates[index] = float(success[:, episode_indices].mean())
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def summarize_task(
    evaluation_root: Path,
    task_id: str,
    expected_episode_seeds: list[int],
    seed_manifest_sha256: str,
) -> dict[str, Any]:
    seed_rows: list[list[dict[str, Any]]] = []
    seed_summaries: list[dict[str, Any]] = []
    fixed_seed_order: list[int] | None = None
    for model_seed in MODEL_SEEDS:
        run_root = evaluation_root / task_id / f"seed_{model_seed}"
        summary_path = run_root / "summary.json"
        episodes_path = run_root / "episodes.jsonl"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("status") != "EVALUATION_COMPLETE"
            or summary.get("protocol_id") != PROTOCOL_ID
            or summary.get("task_id") != task_id
            or int(summary.get("model_seed", -1)) != model_seed
            or int(summary.get("episodes", -1)) != 100
            or summary.get("test_metrics_used_for_selection") is not False
            or summary.get("fixed_test_seed_manifest_sha256")
            != seed_manifest_sha256
            or summary.get("source_bindings", {}).get("seed_manifest_sha256")
            != seed_manifest_sha256
            or summary.get("episodes_jsonl_sha256") != sha256_file(episodes_path)
        ):
            raise RuntimeError(f"invalid evaluation summary: {summary_path}")
        rows = sorted(load_episodes(episodes_path), key=lambda row: row["episode_seed"])
        seed_order = [int(row["episode_seed"]) for row in rows]
        if seed_order != sorted(int(seed) for seed in expected_episode_seeds):
            raise RuntimeError(
                f"closed-loop identities differ from frozen seed bank: {task_id}"
            )
        if fixed_seed_order is None:
            fixed_seed_order = seed_order
        elif seed_order != fixed_seed_order:
            raise RuntimeError(f"test episode identities drifted across seeds: {task_id}")
        if any(int(row["model_seed"]) != model_seed for row in rows):
            raise RuntimeError(f"model seed drifted in {episodes_path}")
        seed_rows.append(rows)
        seed_summaries.append(
            {
                "model_seed": model_seed,
                "success_once": float(np.mean([row["success_once"] for row in rows])),
                "success_at_end": float(
                    np.mean([row["success_at_end"] for row in rows])
                ),
                "selected_checkpoint_step": int(rows[0]["selected_checkpoint_step"]),
            }
        )
    success = np.asarray(
        [[bool(row["success_once"]) for row in rows] for rows in seed_rows],
        dtype=np.float64,
    )
    aggregate = float(success.mean())
    seed_success = [item["success_once"] for item in seed_summaries]
    flat_rows = [row for rows in seed_rows for row in rows]
    result = {
        "task_id": task_id,
        "episodes_per_model_seed": 100,
        "model_seeds": list(MODEL_SEEDS),
        "per_seed": seed_summaries,
        "aggregate_success_once": aggregate,
        "paired_episode_bootstrap_95_ci": paired_episode_bootstrap(success),
        "seed_success_range_percentage_points": 100.0
        * (max(seed_success) - min(seed_success)),
        "aggregate_success_at_end": float(
            np.mean([row["success_at_end"] for row in flat_rows])
        ),
        "mean_episode_length": float(
            np.mean([row["episode_length"] for row in flat_rows])
        ),
        "mean_intended_contact_events": float(
            np.mean([row["intended_contact_events"] for row in flat_rows])
        ),
        "mean_unintended_contact_events": float(
            np.mean([row["unintended_contact_events"] for row in flat_rows])
        ),
        "mean_collisions": float(np.mean([row["collisions"] for row in flat_rows])),
        "total_policy_calls": int(sum(row["policy_calls"] for row in flat_rows)),
        "total_action_opportunities": int(
            sum(row["action_opportunities"] for row in flat_rows)
        ),
        "total_policy_latency_seconds": float(
            sum(row["policy_latency_seconds"] for row in flat_rows)
        ),
    }
    if result["total_policy_calls"] != result["total_action_opportunities"]:
        raise RuntimeError(f"policy/action accounting mismatch for {task_id}")
    return result


def main() -> None:
    args = parse_args()
    seed_manifest = json.loads(args.seed_manifest.read_text(encoding="utf-8"))
    if seed_manifest.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("fixed seed manifest protocol mismatch")
    seed_manifest_sha256 = sha256_file(args.seed_manifest)
    tasks = {
        task_id: summarize_task(
            args.evaluation_root,
            task_id,
            [
                int(seed)
                for seed in seed_manifest["formal_tasks"][task_id][
                    "closed_loop_test_seeds"
                ]
            ],
            seed_manifest_sha256,
        )
        for task_id in FORMAL_TASKS
    }
    positive_pass = {}
    for task_id in POSITIVE_TASKS:
        value = tasks[task_id]
        positive_pass[task_id] = bool(
            0.25 <= value["aggregate_success_once"] <= 0.85
            and value["seed_success_range_percentage_points"] <= 25.0 + 1e-12
        )
        value["baseline_gate_pass"] = positive_pass[task_id]
        value["baseline_gate"] = {
            "aggregate_success_inclusive": [0.25, 0.85],
            "maximum_seed_range_percentage_points": 25.0,
        }
    negative_value = tasks[NEGATIVE_CONTROL]
    negative_pass = bool(0.70 <= negative_value["aggregate_success_once"] <= 0.98)
    negative_value["baseline_gate_pass"] = negative_pass
    negative_value["baseline_gate"] = {
        "aggregate_success_inclusive": [0.70, 0.98]
    }
    passing_positive_tasks = [task for task, passed in positive_pass.items() if passed]
    continue_to_oracle = len(passing_positive_tasks) >= 2 and negative_pass
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": (
            "BASELINE_GATE_PASS_CONTINUE_TO_ORACLE_PROBE"
            if continue_to_oracle
            else "NO_GO_BASELINE_GATE"
        ),
        "tasks": tasks,
        "passing_positive_tasks": passing_positive_tasks,
        "passing_positive_task_count": len(passing_positive_tasks),
        "minimum_passing_positive_tasks": 2,
        "negative_control_pass": negative_pass,
        "continue_to_oracle_probe": continue_to_oracle,
        "bootstrap": {
            "method": "paired_percentile_resample_fixed_episode_identity",
            "replicates": BOOTSTRAP_REPLICATES,
            "confidence_level": 0.95,
            "seed": BOOTSTRAP_SEED,
        },
        "thresholds_changed_after_results": False,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
