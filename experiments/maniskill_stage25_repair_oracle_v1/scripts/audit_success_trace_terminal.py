#!/usr/bin/env python3
"""Independently recompute terminal success-trace fields from immutable raw rows."""

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
from common import MODEL_SEEDS, PROTOCOL_ID, sha256_file, write_json


MODES = (
    "fixed_horizon",
    "terminate_first_success",
    "terminate_hold5",
    "neutral_after_hold5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def quaternion_distance(first: list[float], last: list[float]) -> float:
    first_array = np.asarray(first, dtype=np.float64)
    last_array = np.asarray(last, dtype=np.float64)
    first_array /= np.linalg.norm(first_array)
    last_array /= np.linalg.norm(last_array)
    cosine = float(np.clip(abs(np.dot(first_array, last_array)), -1.0, 1.0))
    return float(2.0 * math.acos(cosine))


def terminal_from_trace(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("trace")
    if not isinstance(trace, list) or not trace:
        raise RuntimeError("semantics row is missing its stepwise trace")
    terminal = trace[-1]
    if int(terminal["step"]) != int(row["episode_length"]):
        raise RuntimeError("terminal trace step differs from episode_length")
    final_position = np.asarray(terminal["object_position"], dtype=np.float64)
    if int(row["first_success_step"]) < 0:
        if row["success_once"]:
            raise RuntimeError("success_once row has no first_success_step")
        return {
            "terminal_step": int(terminal["step"]),
            "final_object_position": final_position.tolist(),
            "drift": None,
            "trace_reported_drift": None,
        }
    first_matches = [
        item for item in trace if int(item["step"]) == int(row["first_success_step"])
    ]
    if len(first_matches) != 1 or not bool(first_matches[0]["success_predicate"]):
        raise RuntimeError("first-success trace anchor is missing or invalid")
    first = first_matches[0]
    first_position = np.asarray(first["object_position"], dtype=np.float64)
    recomputed = {
        "translation_m": float(np.linalg.norm(final_position - first_position)),
        "rotation_rad": quaternion_distance(
            first["object_quaternion"], terminal["object_quaternion"]
        ),
        "from_step": int(row["first_success_step"]),
        "to_step": int(terminal["step"]),
    }
    trace_reported = terminal.get("post_success_object_drift")
    if trace_reported is None:
        raise RuntimeError("successful terminal trace is missing drift")
    return {
        "terminal_step": int(terminal["step"]),
        "final_object_position": final_position.tolist(),
        "drift": recomputed,
        "trace_reported_drift": {
            "translation_m": float(trace_reported["translation_m"]),
            "rotation_rad": float(trace_reported["rotation_rad"]),
        },
    }


def differs(first: dict[str, Any] | None, second: dict[str, Any] | None) -> bool:
    if first is None or second is None:
        return first is not second
    return bool(
        abs(float(first["translation_m"]) - float(second["translation_m"])) > 1e-9
        or abs(float(first["rotation_rad"]) - float(second["rotation_rad"])) > 1e-8
        or int(first.get("from_step", -1)) != int(second.get("from_step", -1))
        or int(first.get("to_step", -1)) != int(second.get("to_step", -1))
    )


def paired_ci(values: np.ndarray, seed: int) -> list[float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(10_000, len(values)))
    replicates = np.mean(values[indices], axis=1)
    return [
        float(np.quantile(replicates, 0.025)),
        float(np.quantile(replicates, 0.975)),
    ]


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    all_rows: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    mode_results: dict[str, Any] = {}
    raw_hashes: dict[str, str] = {}
    total_trace_rows = 0
    max_trace_pose_error = {"translation_m": 0.0, "rotation_rad": 0.0}

    for mode in MODES:
        indexed: dict[tuple[int, int], dict[str, Any]] = {}
        per_seed: dict[str, Any] = {}
        mode_top_drift_mismatches = 0
        mode_top_position_mismatches = 0
        corrected_drifts: list[dict[str, float]] = []
        hold5_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
        mode_rows: list[dict[str, Any]] = []
        for model_seed in MODEL_SEEDS:
            path = (
                args.result_root
                / "success_semantics"
                / mode
                / f"seed_{model_seed}"
                / "episodes.jsonl"
            )
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if len(rows) != 100:
                raise RuntimeError(f"incomplete semantics raw file: {path}")
            raw_hashes[str(path)] = sha256_file(path)
            seed_drifts: list[dict[str, float]] = []
            for row in rows:
                if row.get("protocol_id") != PROTOCOL_ID or row.get("mode") != mode:
                    raise RuntimeError(f"protocol or mode mismatch: {path}")
                key = (int(model_seed), int(row["episode_seed"]))
                if key in indexed:
                    raise RuntimeError(f"duplicate semantics identity: {key}")
                terminal = terminal_from_trace(row)
                total_trace_rows += len(row["trace"])
                derived = terminal["drift"]
                trace_reported = terminal["trace_reported_drift"]
                if derived is not None:
                    max_trace_pose_error["translation_m"] = max(
                        max_trace_pose_error["translation_m"],
                        abs(derived["translation_m"] - trace_reported["translation_m"]),
                    )
                    max_trace_pose_error["rotation_rad"] = max(
                        max_trace_pose_error["rotation_rad"],
                        abs(derived["rotation_rad"] - trace_reported["rotation_rad"]),
                    )
                    seed_drifts.append(derived)
                    corrected_drifts.append(derived)
                if differs(row.get("post_success_object_drift"), derived):
                    mode_top_drift_mismatches += 1
                position_error = float(
                    np.linalg.norm(
                        np.asarray(row["final_object_position"], dtype=np.float64)
                        - np.asarray(terminal["final_object_position"], dtype=np.float64)
                    )
                )
                if position_error > 1e-9:
                    mode_top_position_mismatches += 1
                if row["success_hold5"]:
                    if derived is None:
                        raise RuntimeError("hold5 row is missing a success drift")
                    hold5_rows.append((row, derived))
                indexed[key] = row
                mode_rows.append(row)
            per_seed[str(model_seed)] = {
                key: float(np.mean([row[key] for row in rows]))
                for key in (
                    "success_once",
                    "success_hold5",
                    "success_at_end",
                    "post_success_loss",
                )
            }
            per_seed[str(model_seed)]["terminal_drift_translation_m"] = float(
                np.mean([item["translation_m"] for item in seed_drifts])
            )
            per_seed[str(model_seed)]["terminal_drift_rotation_rad"] = float(
                np.mean([item["rotation_rad"] for item in seed_drifts])
            )
        aggregate = {
            key: float(np.mean([row[key] for row in mode_rows]))
            for key in (
                "success_once",
                "success_hold5",
                "success_at_end",
                "post_success_loss",
            )
        }
        aggregate.update(
            {
                "terminal_drift_translation_m": float(
                    np.mean([item["translation_m"] for item in corrected_drifts])
                ),
                "terminal_drift_rotation_rad": float(
                    np.mean([item["rotation_rad"] for item in corrected_drifts])
                ),
                "hold5_episode_count": len(hold5_rows),
                "hold5_terminal_drift_translation_m": float(
                    np.mean([item[1]["translation_m"] for item in hold5_rows])
                )
                if hold5_rows
                else None,
                "hold5_terminal_drift_rotation_rad": float(
                    np.mean([item[1]["rotation_rad"] for item in hold5_rows])
                )
                if hold5_rows
                else None,
                "success_at_end_given_hold5": float(
                    np.mean([item[0]["success_at_end"] for item in hold5_rows])
                )
                if hold5_rows
                else None,
            }
        )
        mode_results[mode] = {
            "per_model_seed": per_seed,
            "aggregate": aggregate,
            "redundant_top_level_drift_mismatch_rows": mode_top_drift_mismatches,
            "redundant_top_level_final_position_mismatch_rows": mode_top_position_mismatches,
        }
        all_rows[mode] = indexed

    identities = set(all_rows[MODES[0]])
    if any(set(all_rows[mode]) != identities for mode in MODES[1:]):
        raise RuntimeError("success-semantics identities differ across modes")
    fixed = all_rows["fixed_horizon"]
    for offset, mode in enumerate(MODES[1:]):
        differences = np.asarray(
            [
                float(all_rows[mode][key]["success_at_end"])
                - float(fixed[key]["success_at_end"])
                for key in sorted(identities)
            ],
            dtype=np.float64,
        )
        mode_results[mode]["aggregate"]["success_at_end_gain_vs_fixed"] = float(
            np.mean(differences)
        )
        mode_results[mode]["aggregate"]["paired_bootstrap_95_ci"] = paired_ci(
            differences, 16018 + offset
        )

    trace_pass = bool(
        max_trace_pose_error["translation_m"] <= 1e-9
        and max_trace_pose_error["rotation_rad"] <= 1e-8
    )
    write_json(
        args.output,
        {
            "protocol_id": PROTOCOL_ID,
            "status": "SUCCESS_TRACE_TERMINAL_AUDIT_PASS" if trace_pass else "SUCCESS_TRACE_TERMINAL_AUDIT_FAIL",
            "scientific_raw_files_modified": False,
            "decision_metrics_affected_by_redundant_field_defect": False,
            "raw_episode_rows": len(identities) * len(MODES),
            "raw_trace_rows": total_trace_rows,
            "max_abs_trace_vs_pose_recompute_error": max_trace_pose_error,
            "modes": mode_results,
            "raw_sha256": raw_hashes,
        },
    )


if __name__ == "__main__":
    main()
