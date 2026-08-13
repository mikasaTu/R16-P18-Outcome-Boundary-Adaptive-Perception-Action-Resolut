#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from protocol_common import (  # noqa: E402
    FORMAL_TASKS,
    MODEL_SEEDS,
    PROTOCOL_ID,
    sha256_file,
    write_json,
)


TOTAL_ITERATIONS = {
    "PullCubeTool-v1": 100_000,
    "PushT-v1": 100_000,
    "StackCube-v1": 30_000,
    "PushCube-v1": 30_000,
}
CHECKPOINT_INTERVAL = 5_000
EXPECTED_UID = 2254
EXPECTED_GID = 2254


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-all-candidate-payloads", action="store_true")
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
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"required regular file is missing: {path}")
    if path.resolve(strict=True) != path:
        raise RuntimeError(f"non-canonical evidence path: {path}")
    stat = path.stat()
    if (stat.st_uid, stat.st_gid) != (EXPECTED_UID, EXPECTED_GID):
        raise RuntimeError(
            f"evidence owner mismatch: {path}: {stat.st_uid}:{stat.st_gid}"
        )
    return stat


def file_record(path: Path) -> dict[str, Any]:
    stat = require_regular_owned(path)
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "sha256": sha256_file(path),
        "uid": stat.st_uid,
        "gid": stat.st_gid,
    }


def canonical_candidate(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(value["path"]),
        "step": int(value["step"]),
        "validation_loss": float(value["validation_loss"]),
        "checkpoint_sha256": str(value["checkpoint_sha256"]),
    }


def audit_run(
    checkpoint_root: Path,
    task_id: str,
    model_seed: int,
    *,
    verify_all_candidate_payloads: bool,
) -> dict[str, Any]:
    run_dir = checkpoint_root / task_id / f"seed_{model_seed}"
    selection_path = run_dir / "checkpoint_selection.json"
    completion_path = run_dir / "TRAINING_COMPLETE.json"
    validation_metrics_path = run_dir / "validation_metrics.jsonl"
    training_metrics_path = run_dir / "training_metrics.jsonl"
    for path in (
        selection_path,
        completion_path,
        validation_metrics_path,
        training_metrics_path,
    ):
        require_regular_owned(path)

    selection = load_json(selection_path)
    completion = load_json(completion_path)
    validation_rows = load_jsonl(validation_metrics_path)
    training_rows = load_jsonl(training_metrics_path)
    total_iterations = TOTAL_ITERATIONS[task_id]
    expected_steps = list(
        range(CHECKPOINT_INTERVAL, total_iterations + 1, CHECKPOINT_INTERVAL)
    )

    if (
        selection.get("protocol_id") != PROTOCOL_ID
        or selection.get("selection_metric")
        != "deterministic_mean_validation_imitation_loss"
        or selection.get("direction") != "minimize"
        or selection.get("tie_break") != "earliest_step"
        or selection.get("test_metrics_used") is not False
    ):
        raise RuntimeError(f"invalid checkpoint selection contract: {selection_path}")
    candidates = [canonical_candidate(value) for value in selection["candidates"]]
    if [value["step"] for value in candidates] != expected_steps:
        raise RuntimeError(f"checkpoint candidate cadence mismatch: {run_dir}")
    if any(not math.isfinite(value["validation_loss"]) for value in candidates):
        raise RuntimeError(f"non-finite validation loss: {run_dir}")
    selected = canonical_candidate(selection["selected"])
    expected_selected = min(
        candidates, key=lambda value: (value["validation_loss"], value["step"])
    )
    if selected != expected_selected:
        raise RuntimeError(f"selection is not validation-only argmin: {run_dir}")

    if len(validation_rows) != len(candidates):
        raise RuntimeError(f"validation candidate inventory mismatch: {run_dir}")
    for row, candidate in zip(validation_rows, candidates, strict=True):
        observed = {
            "path": str(row["checkpoint"]),
            "step": int(row["global_iteration"]),
            "validation_loss": float(row["validation_loss"]),
        }
        expected = {
            "path": candidate["path"],
            "step": candidate["step"],
            "validation_loss": candidate["validation_loss"],
        }
        if (
            observed != expected
            or row.get("protocol_id") != PROTOCOL_ID
            or row.get("task_id") != task_id
            or int(row.get("seed", -1)) != model_seed
        ):
            raise RuntimeError(f"validation metric/candidate mismatch: {run_dir}")

    if (
        completion.get("protocol_id") != PROTOCOL_ID
        or completion.get("status") != "TRAINING_COMPLETE"
        or completion.get("task_id") != task_id
        or int(completion.get("seed", -1)) != model_seed
        or int(completion.get("global_iteration", -1)) != total_iterations
        or canonical_candidate(completion["selected_checkpoint"]) != selected
        or completion.get("train_config_sha256")
        != selection.get("train_config_sha256")
    ):
        raise RuntimeError(f"invalid training completion marker: {completion_path}")
    if not training_rows or int(training_rows[-1].get("global_iteration", -1)) != total_iterations:
        raise RuntimeError(f"training metrics do not reach the terminal step: {run_dir}")
    for row in training_rows:
        if (
            row.get("protocol_id") != PROTOCOL_ID
            or row.get("task_id") != task_id
            or int(row.get("seed", -1)) != model_seed
            or not all(
                math.isfinite(float(row[key])) for key in ("loss", "l1", "kl")
            )
        ):
            raise RuntimeError(f"invalid or non-finite training metric: {run_dir}")

    candidate_records = []
    for candidate in candidates:
        candidate_dir = Path(candidate["path"])
        expected_dir = run_dir / "checkpoints" / f"step_{candidate['step']:09d}"
        if candidate_dir != expected_dir or candidate_dir.resolve(strict=True) != expected_dir:
            raise RuntimeError(f"candidate checkpoint escaped its run directory: {candidate_dir}")
        marker_path = candidate_dir / "COMPLETE.json"
        payload_path = candidate_dir / "checkpoint.pt"
        marker = load_json(marker_path)
        marker_record = file_record(marker_path)
        payload_stat = require_regular_owned(payload_path)
        if (
            marker.get("protocol_id") != PROTOCOL_ID
            or marker.get("complete") is not True
            or int(marker.get("global_iteration", -1)) != candidate["step"]
            or marker.get("train_config_sha256")
            != selection.get("train_config_sha256")
            or float(marker.get("validation_loss")) != candidate["validation_loss"]
            or marker.get("checkpoint_sha256") != candidate["checkpoint_sha256"]
        ):
            raise RuntimeError(f"invalid checkpoint COMPLETE marker: {marker_path}")
        verify_payload = verify_all_candidate_payloads or candidate == selected
        if verify_payload and sha256_file(payload_path) != candidate["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint payload digest mismatch: {payload_path}")
        candidate_records.append(
            {
                **candidate,
                "payload_bytes": payload_stat.st_size,
                "payload_uid": payload_stat.st_uid,
                "payload_gid": payload_stat.st_gid,
                "payload_sha256_verified": verify_payload,
                "complete_marker": marker_record,
            }
        )

    return {
        "task_id": task_id,
        "model_seed": model_seed,
        "total_iterations": total_iterations,
        "candidate_count": len(candidates),
        "selected": selected,
        "test_metrics_used_for_selection": False,
        "selection_is_validation_argmin_with_earliest_tie_break": True,
        "all_training_metrics_finite": True,
        "payload_verification_scope": (
            "all_candidates" if verify_all_candidate_payloads else "selected_only"
        ),
        "candidates": candidate_records,
        "files": {
            "selection": file_record(selection_path),
            "completion": file_record(completion_path),
            "validation_metrics": file_record(validation_metrics_path),
            "training_metrics": file_record(training_metrics_path),
        },
    }


def audit_matrix_artifacts(artifact_dir: Path) -> dict[str, Any]:
    paths = {
        "first_real_work": artifact_dir / "FIRST_REAL_WORK.json",
        "rgb_replay": artifact_dir / "RGB_REPLAY_MATRIX_COMPLETE.json",
        "training_matrix": artifact_dir / "TRAINING_MATRIX_COMPLETE.json",
        "formal_result": artifact_dir / "FORMAL_MATRIX_RESULT.json",
    }
    values = {key: load_json(path) for key, path in paths.items()}
    if (
        values["first_real_work"].get("protocol_id") != PROTOCOL_ID
        or values["first_real_work"].get("status")
        != "persisted_real_optimizer_step_and_loss"
        or values["rgb_replay"].get("status") != "PASS"
        or values["training_matrix"].get("status")
        != "TRAINING_MATRIX_COMPLETE"
        or int(values["training_matrix"].get("models", -1)) != 12
        or values["training_matrix"].get("tasks") != list(TOTAL_ITERATIONS)
        or values["training_matrix"].get("seeds") != list(MODEL_SEEDS)
        or values["formal_result"].get("protocol_id") != PROTOCOL_ID
        or values["formal_result"].get("status") != "PASS"
        or values["formal_result"].get("phase") != "replay-and-train"
        or len(values["formal_result"].get("training_gate", [])) != 12
    ):
        raise RuntimeError(f"invalid formal matrix artifact set: {artifact_dir}")
    return {
        key: {"value": values[key], "file": file_record(path)}
        for key, path in paths.items()
    }


def main() -> None:
    args = parse_args()
    matrix_artifacts = audit_matrix_artifacts(args.artifact_dir)
    runs = [
        audit_run(
            args.checkpoint_root,
            task_id,
            model_seed,
            verify_all_candidate_payloads=args.verify_all_candidate_payloads,
        )
        for task_id in FORMAL_TASKS
        for model_seed in MODEL_SEEDS
    ]
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "FORMAL_TRAINING_ARTIFACT_AUDIT_PASS",
        "checkpoint_root": str(args.checkpoint_root),
        "artifact_dir": str(args.artifact_dir),
        "model_count": len(runs),
        "tasks": list(FORMAL_TASKS),
        "model_seeds": list(MODEL_SEEDS),
        "checkpoint_selection_metric": (
            "deterministic_mean_validation_imitation_loss"
        ),
        "test_metrics_used_for_selection": False,
        "all_candidate_payloads_verified": args.verify_all_candidate_payloads,
        "expected_uid": EXPECTED_UID,
        "expected_gid": EXPECTED_GID,
        "matrix_artifacts": matrix_artifacts,
        "runs": runs,
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
        "completed_at_unix": time.time(),
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
