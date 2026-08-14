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
from common import MODEL_SEEDS, PROTOCOL_ID, TASKS, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("screen", "final"), required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--screen-selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_candidates(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol_id") != PROTOCOL_ID or value.get("candidate_count") != 156:
        raise RuntimeError("candidate manifest binding mismatch")
    result: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in value["candidates"]:
        key = (row["task_id"], int(row["model_seed"]))
        if key[0] in TASKS:
            result.setdefault(key, []).append(row)
    for key, rows in result.items():
        rows.sort(key=lambda row: int(row["step"]))
        if len(rows) != 6:
            raise RuntimeError(f"expected six candidates for {key}, got {len(rows)}")
    return result


def summary_path(root: Path, split: str, task: str, seed: int, step: int) -> Path:
    return root / split / task / f"seed_{seed}" / f"step_{step:09d}" / "summary.json"


def read_metric(root: Path, split: str, candidate: dict[str, Any]) -> dict[str, Any]:
    path = summary_path(
        root,
        split,
        candidate["task_id"],
        int(candidate["model_seed"]),
        int(candidate["step"]),
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol_id": PROTOCOL_ID,
        "task_id": candidate["task_id"],
        "model_seed": int(candidate["model_seed"]),
        "checkpoint_step": int(candidate["step"]),
        "checkpoint_sha256": candidate["checkpoint_sha256"],
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise RuntimeError(f"evaluation binding mismatch: {path}")
    return {
        **candidate,
        "success_once": float(value["success_once"]),
        "success_hold5": float(value["success_hold5"]),
        "success_at_end": float(value["success_at_end"]),
        "post_success_loss": float(value["post_success_loss"]),
        "summary_path": str(path),
        "summary_sha256": sha256_file(path),
    }


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        -float(row["success_hold5"]),
        -float(row["success_at_end"]),
        float(row["post_success_loss"]),
        int(row["step"]),
    )


def rankdata(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def spearman(first: list[float], second: list[float]) -> float | None:
    x, y = rankdata(first), rankdata(second)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    args = parse_args()
    candidates = load_candidates(args.candidate_manifest)
    groups: dict[str, Any] = {}
    if args.stage == "screen":
        for task in TASKS:
            for seed in MODEL_SEEDS:
                key = (task, seed)
                rows = [read_metric(args.evaluation_root, "screen", row) for row in candidates[key]]
                ranked = sorted(rows, key=selection_key)
                old = min(rows, key=lambda row: (float(row["validation_loss"]), int(row["step"])))
                final = max(rows, key=lambda row: int(row["step"]))
                union = {int(row["step"]): row for row in (*ranked[:2], old, final)}
                groups[f"{task}/seed_{seed}"] = {
                    "screen_ranked": ranked,
                    "screen_top2": ranked[:2],
                    "predecessor_validation_loss_control": old,
                    "final_checkpoint_control": final,
                    "full_validation_candidates": [union[step] for step in sorted(union)],
                    "spearman_validation_loss_vs_success_hold5": spearman(
                        [float(row["validation_loss"]) for row in rows],
                        [float(row["success_hold5"]) for row in rows],
                    ),
                }
        status = "CHECKPOINT_SCREEN_SELECTION_COMPLETE"
    else:
        if args.screen_selection is None:
            raise ValueError("--screen-selection is required for final selection")
        screen = json.loads(args.screen_selection.read_text(encoding="utf-8"))
        if screen.get("status") != "CHECKPOINT_SCREEN_SELECTION_COMPLETE":
            raise RuntimeError("screen selection is incomplete")
        for task in TASKS:
            for seed in MODEL_SEEDS:
                name = f"{task}/seed_{seed}"
                screen_group = screen["groups"][name]
                top2_steps = {int(row["step"]) for row in screen_group["screen_top2"]}
                full_rows = [
                    read_metric(args.evaluation_root, "full_validation", row)
                    for row in screen_group["full_validation_candidates"]
                ]
                eligible = [row for row in full_rows if int(row["step"]) in top2_steps]
                selected = min(eligible, key=selection_key)
                old_step = int(screen_group["predecessor_validation_loss_control"]["step"])
                old = next(row for row in full_rows if int(row["step"]) == old_step)
                final_step = int(screen_group["final_checkpoint_control"]["step"])
                final = next(row for row in full_rows if int(row["step"]) == final_step)
                dominates_old = bool(
                    selected["success_hold5"] >= old["success_hold5"]
                    and selected["success_at_end"] >= old["success_at_end"]
                    and selected["post_success_loss"] <= old["post_success_loss"]
                    and (
                        selected["success_hold5"] > old["success_hold5"]
                        or selected["success_at_end"] > old["success_at_end"]
                        or selected["post_success_loss"] < old["post_success_loss"]
                    )
                )
                groups[name] = {
                    "selected": selected,
                    "full_validation_rows": sorted(full_rows, key=selection_key),
                    "predecessor_validation_loss_control": old,
                    "final_checkpoint_control": final,
                    "predecessor_checkpoint_pareto_dominated": dominates_old,
                    "screen_winner_equals_final_winner": int(
                        screen_group["screen_top2"][0]["step"]
                    )
                    == int(selected["step"]),
                    "rank_inversion_from_validation_loss": int(selected["step"]) != old_step,
                }
        status = "CHECKPOINT_FINAL_SELECTION_COMPLETE"
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "stage": args.stage,
        "selection_rule": [
            "maximize_success_hold5",
            "maximize_success_at_end",
            "minimize_post_success_loss",
            "earliest_step",
        ],
        "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
        "screen_selection_sha256": (
            sha256_file(args.screen_selection) if args.screen_selection else None
        ),
        "groups": groups,
    }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
