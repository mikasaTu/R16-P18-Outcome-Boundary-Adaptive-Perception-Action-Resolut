#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
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
)


EXPECTED_UID = 2254
EXPECTED_GID = 2254
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 16018
POSITIVE_TASKS = tuple(task for task in FORMAL_TASKS if task != "PushCube-v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--verify-selected-checkpoint-payloads", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def require_regular_owned(path: Path) -> os.stat_result:
    if not path.is_file() or path.is_symlink() or path.resolve(strict=True) != path:
        raise RuntimeError(f"required canonical regular file is missing: {path}")
    stat = path.stat()
    if (stat.st_uid, stat.st_gid) != (EXPECTED_UID, EXPECTED_GID):
        raise RuntimeError(f"owner mismatch: {path}: {stat.st_uid}:{stat.st_gid}")
    return stat


def file_record(path: Path, *, digest: bool = True) -> dict[str, Any]:
    stat = require_regular_owned(path)
    result = {
        "path": str(path),
        "bytes": stat.st_size,
        "uid": stat.st_uid,
        "gid": stat.st_gid,
    }
    if digest:
        result["sha256"] = sha256_file(path)
    return result


def require_close(observed: Any, expected: Any, label: str) -> None:
    if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"{label} mismatch: observed={observed}, expected={expected}")


def paired_bootstrap(success: np.ndarray) -> list[float]:
    if success.shape != (len(MODEL_SEEDS), 100):
        raise RuntimeError(f"invalid paired bootstrap shape: {success.shape}")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        identities = rng.integers(0, 100, size=100)
        estimates[index] = float(success[:, identities].mean())
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def audit_run(
    *,
    evaluation_root: Path,
    checkpoint_root: Path,
    task_id: str,
    model_seed: int,
    expected_episode_seeds: list[int],
    seed_manifest_sha256: str,
    evaluator_sha256: str,
    verify_checkpoint_payload: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_root = evaluation_root / task_id / f"seed_{model_seed}"
    episodes_path = run_root / "episodes.jsonl"
    summary_path = run_root / "summary.json"
    first_batch_path = run_root / "FIRST_REAL_ROLLOUT_BATCH.json"
    selection_path = checkpoint_root / task_id / f"seed_{model_seed}" / "checkpoint_selection.json"
    for path in (episodes_path, summary_path, first_batch_path, selection_path):
        require_regular_owned(path)

    rows = load_jsonl(episodes_path)
    summary = load_json(summary_path)
    first_batch = load_json(first_batch_path)
    selection = load_json(selection_path)
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 episodes: {episodes_path}")
    observed_seeds = [int(row["episode_seed"]) for row in rows]
    if observed_seeds != expected_episode_seeds or len(set(observed_seeds)) != 100:
        raise RuntimeError(f"frozen episode identity/order mismatch: {run_root}")
    if any(
        row.get("protocol_id") != PROTOCOL_ID
        or row.get("task_id") != task_id
        or int(row.get("model_seed", -1)) != model_seed
        or int(row.get("policy_calls", -1))
        != int(row.get("action_opportunities", -2))
        for row in rows
    ):
        raise RuntimeError(f"episode binding/accounting mismatch: {episodes_path}")

    selected = selection["selected"]
    checkpoint_path = Path(selected["path"]) / "checkpoint.pt"
    checkpoint_stat = require_regular_owned(checkpoint_path)
    checkpoint_digest_verified = False
    if verify_checkpoint_payload:
        if sha256_file(checkpoint_path) != selected["checkpoint_sha256"]:
            raise RuntimeError(f"selected checkpoint digest mismatch: {checkpoint_path}")
        checkpoint_digest_verified = True

    if (
        summary.get("protocol_id") != PROTOCOL_ID
        or summary.get("status") != "EVALUATION_COMPLETE"
        or summary.get("task_id") != task_id
        or int(summary.get("model_seed", -1)) != model_seed
        or int(summary.get("episodes", -1)) != 100
        or summary.get("test_metrics_used_for_selection") is not False
        or summary.get("fixed_test_seed_manifest_sha256") != seed_manifest_sha256
        or summary.get("episodes_jsonl_sha256") != sha256_file(episodes_path)
        or summary.get("selected_checkpoint") != selected
    ):
        raise RuntimeError(f"summary contract mismatch: {summary_path}")
    bindings = summary.get("source_bindings", {})
    if (
        bindings.get("evaluator_sha256") != evaluator_sha256
        or bindings.get("seed_manifest_sha256") != seed_manifest_sha256
        or bindings.get("checkpoint_selection_sha256") != sha256_file(selection_path)
        or int(bindings.get("selected_checkpoint_step", -1)) != int(selected["step"])
        or bindings.get("selected_checkpoint_sha256") != selected["checkpoint_sha256"]
        or any(int(row["selected_checkpoint_step"]) != int(selected["step"]) for row in rows)
    ):
        raise RuntimeError(f"source binding mismatch: {summary_path}")

    if (
        first_batch.get("protocol_id") != PROTOCOL_ID
        or first_batch.get("status") != "FIRST_REAL_ROLLOUT_BATCH_COMPLETE"
        or first_batch.get("task_id") != task_id
        or int(first_batch.get("model_seed", -1)) != model_seed
        or int(first_batch.get("episode_count", -1)) != 20
        or first_batch.get("episode_seeds") != expected_episode_seeds[:20]
        or first_batch.get("evaluator_sha256") != evaluator_sha256
        or int(first_batch.get("selected_checkpoint_step", -1)) != int(selected["step"])
        or first_batch.get("selected_checkpoint_sha256") != selected["checkpoint_sha256"]
    ):
        raise RuntimeError(f"first rollout marker mismatch: {first_batch_path}")

    recomputed = {
        "success_once": float(np.mean([row["success_once"] for row in rows])),
        "success_at_end": float(np.mean([row["success_at_end"] for row in rows])),
        "mean_episode_length": float(np.mean([row["episode_length"] for row in rows])),
        "mean_intended_contact_events": float(
            np.mean([row["intended_contact_events"] for row in rows])
        ),
        "mean_unintended_contact_events": float(
            np.mean([row["unintended_contact_events"] for row in rows])
        ),
        "mean_collisions": float(np.mean([row["collisions"] for row in rows])),
        "total_policy_latency_seconds": float(
            sum(row["policy_latency_seconds"] for row in rows)
        ),
        "total_policy_calls": int(sum(row["policy_calls"] for row in rows)),
        "total_action_opportunities": int(
            sum(row["action_opportunities"] for row in rows)
        ),
    }
    for key, expected in recomputed.items():
        if isinstance(expected, float):
            require_close(summary[key], expected, f"{task_id}/{model_seed}/{key}")
        elif summary[key] != expected:
            raise RuntimeError(f"{task_id}/{model_seed}/{key} mismatch")

    record = {
        "task_id": task_id,
        "model_seed": model_seed,
        "episode_count": 100,
        "success_once": recomputed["success_once"],
        "success_at_end": recomputed["success_at_end"],
        "selected_checkpoint_step": int(selected["step"]),
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "checkpoint_payload_bytes": checkpoint_stat.st_size,
        "checkpoint_payload_sha256_verified": checkpoint_digest_verified,
        "files": {
            "episodes": file_record(episodes_path),
            "summary": file_record(summary_path),
            "first_real_rollout_batch": file_record(first_batch_path),
            "checkpoint_selection": file_record(selection_path),
        },
    }
    return record, rows


def main() -> None:
    args = parse_args()
    seed_manifest = load_json(args.seed_manifest)
    gate_path = args.evaluation_root / "baseline_gate.json"
    matrix_path = args.artifact_dir / "EVALUATION_MATRIX_COMPLETE.json"
    first_work_path = args.artifact_dir / "FIRST_REAL_WORK.json"
    for path in (args.seed_manifest, gate_path, matrix_path, first_work_path):
        require_regular_owned(path)
    if seed_manifest.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("seed manifest protocol mismatch")
    seed_manifest_sha256 = sha256_file(args.seed_manifest)
    evaluator_path = SCRIPT_DIR / "evaluate_official_act_protocol.py"
    evaluator_sha256 = sha256_file(evaluator_path)
    gate = load_json(gate_path)
    matrix = load_json(matrix_path)
    first_work = load_json(first_work_path)

    expected_names = {
        f"evaluate_{task_id}_seed{model_seed}"
        for task_id in FORMAL_TASKS
        for model_seed in MODEL_SEEDS
    }
    observed_names = {str(value.get("name")) for value in matrix.get("processes", [])}
    if (
        matrix.get("protocol_id") != PROTOCOL_ID
        or matrix.get("status") != "EVALUATION_MATRIX_COMPLETE"
        or matrix.get("baseline_gate_status") != "NO_GO_BASELINE_GATE"
        or matrix.get("continue_to_oracle_probe") is not False
        or observed_names != expected_names
        or any(int(value.get("exit_code", -1)) != 0 for value in matrix["processes"])
    ):
        raise RuntimeError("evaluation matrix completion contract mismatch")
    rollout_marker = Path(first_work["rollout_batch_marker"])
    if (
        first_work.get("protocol_id") != PROTOCOL_ID
        or first_work.get("status") != "FIRST_REAL_WORK"
        or first_work.get("evidence_scope") != "completed_closed_loop_rollout_batch"
        or int(first_work.get("episode_count", -1)) != 20
        or sha256_file(rollout_marker) != first_work.get("rollout_batch_marker_sha256")
    ):
        raise RuntimeError("first real work evidence mismatch")

    run_records: list[dict[str, Any]] = []
    rows_by_task: dict[str, list[list[dict[str, Any]]]] = {}
    for task_id in FORMAL_TASKS:
        expected_seeds = [
            int(value)
            for value in seed_manifest["formal_tasks"][task_id]["closed_loop_test_seeds"]
        ]
        if len(expected_seeds) != 100 or len(set(expected_seeds)) != 100:
            raise RuntimeError(f"invalid frozen seed bank: {task_id}")
        task_rows = []
        for model_seed in MODEL_SEEDS:
            record, rows = audit_run(
                evaluation_root=args.evaluation_root,
                checkpoint_root=args.checkpoint_root,
                task_id=task_id,
                model_seed=model_seed,
                expected_episode_seeds=expected_seeds,
                seed_manifest_sha256=seed_manifest_sha256,
                evaluator_sha256=evaluator_sha256,
                verify_checkpoint_payload=args.verify_selected_checkpoint_payloads,
            )
            run_records.append(record)
            task_rows.append(rows)
        rows_by_task[task_id] = task_rows

    passing_positive_tasks = []
    task_audit: dict[str, dict[str, Any]] = {}
    for task_id, seed_rows in rows_by_task.items():
        success = np.asarray(
            [
                [
                    bool(row["success_once"])
                    for row in sorted(rows, key=lambda value: int(value["episode_seed"]))
                ]
                for rows in seed_rows
            ],
            dtype=np.float64,
        )
        per_seed = [float(values.mean()) for values in success]
        aggregate = float(success.mean())
        seed_range_pp = 100.0 * (max(per_seed) - min(per_seed))
        ci = paired_bootstrap(success)
        observed = gate["tasks"][task_id]
        require_close(observed["aggregate_success_once"], aggregate, f"{task_id}/aggregate")
        require_close(
            observed["seed_success_range_percentage_points"],
            seed_range_pp,
            f"{task_id}/seed_range_pp",
        )
        for index in range(2):
            require_close(
                observed["paired_episode_bootstrap_95_ci"][index],
                ci[index],
                f"{task_id}/bootstrap/{index}",
            )
        if task_id in POSITIVE_TASKS:
            passed = 0.25 <= aggregate <= 0.85 and seed_range_pp <= 25.0 + 1e-12
            if passed:
                passing_positive_tasks.append(task_id)
        else:
            passed = 0.70 <= aggregate <= 0.98
        if observed.get("baseline_gate_pass") is not passed:
            raise RuntimeError(f"gate decision mismatch: {task_id}")
        task_audit[task_id] = {
            "aggregate_success_once": aggregate,
            "per_seed_success_once": per_seed,
            "seed_success_range_percentage_points": seed_range_pp,
            "paired_episode_bootstrap_95_ci": ci,
            "baseline_gate_pass": passed,
        }

    negative_pass = task_audit["PushCube-v1"]["baseline_gate_pass"]
    continue_to_oracle = len(passing_positive_tasks) >= 2 and negative_pass
    if (
        gate.get("protocol_id") != PROTOCOL_ID
        or gate.get("status") != "NO_GO_BASELINE_GATE"
        or gate.get("passing_positive_tasks") != passing_positive_tasks
        or int(gate.get("passing_positive_task_count", -1)) != len(passing_positive_tasks)
        or gate.get("negative_control_pass") is not negative_pass
        or gate.get("continue_to_oracle_probe") is not continue_to_oracle
        or gate.get("thresholds_changed_after_results") is not False
        or continue_to_oracle
    ):
        raise RuntimeError("top-level baseline gate mismatch")

    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "FORMAL_BASELINE_AUDIT_PASS",
        "scientific_decision": "NO_GO_BASELINE_GATE",
        "episode_count": 1200,
        "run_count": 12,
        "all_episode_identities_match_frozen_manifest": True,
        "all_selected_checkpoints_validation_only": True,
        "selected_checkpoint_payload_verification_scope": (
            "all_selected"
            if args.verify_selected_checkpoint_payloads
            else "metadata_only"
        ),
        "all_policy_calls_equal_action_opportunities": True,
        "paired_bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "gate": {
            "passing_positive_tasks": passing_positive_tasks,
            "passing_positive_task_count": len(passing_positive_tasks),
            "negative_control_pass": negative_pass,
            "continue_to_oracle_probe": continue_to_oracle,
        },
        "tasks": task_audit,
        "runs": run_records,
        "files": {
            "baseline_gate": file_record(gate_path),
            "evaluation_matrix_complete": file_record(matrix_path),
            "first_real_work": file_record(first_work_path),
            "seed_manifest": file_record(args.seed_manifest),
            "evaluator": file_record(evaluator_path),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
