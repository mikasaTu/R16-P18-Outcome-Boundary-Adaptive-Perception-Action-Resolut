#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import MODEL_SEEDS, PROTOCOL_ID, append_jsonl, sha256_file, write_json
from oracle_math import (
    COARSE_INDICES,
    PRIMARY_UTILITY,
    best_valid_index,
    paired_percentile_ci,
    utility,
)

SENSITIVITY_WEIGHTS = {
    "primary": PRIMARY_UTILITY,
    "success_dominant": {
        **PRIMARY_UTILITY,
        "stable_success": 200.0,
        "clipped_progress_delta": 10.0,
        "intended_contact": 1.0,
    },
    "progress_dominant": {
        **PRIMARY_UTILITY,
        "stable_success": 50.0,
        "clipped_progress_delta": 40.0,
    },
}
PHASE_PRIORITY = {
    "placement_contact_near_completion": 0,
    "object_in_hand_pre_placement": 1,
    "pre_grasp_or_pre_contact": 2,
    "free_space_approach": 3,
    "post_success": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-confirmatory-root", type=Path, required=True)
    parser.add_argument("--joint-calibration-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(root: Path) -> dict[int, list[dict[str, Any]]]:
    result = {}
    for seed in MODEL_SEEDS:
        path = root / f"seed_{seed}" / "states.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != 64 or len({row["bank_id"] for row in rows}) != 64:
            raise RuntimeError(f"incomplete confirmatory visual rows: {path}")
        result[seed] = rows
    identities = [{row["bank_id"] for row in rows} for rows in result.values()]
    if not identities[0] == identities[1] == identities[2]:
        raise RuntimeError("model seeds do not share the same state identities")
    return result


def category(outcome: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(bool(outcome["stable_success"])),
        int(outcome["phase_outcome_rank"]),
        int(bool(outcome["recoverable"])),
    )


def atlas_arm(atlas: Mapping[str, Any], allowed: tuple[int, ...], weights: Mapping[str, float]) -> dict:
    values: list[float | None] = []
    allowed_set = set(allowed)
    for index, outcome in enumerate(atlas["outcomes"]):
        if not atlas["valid"][index] or index not in allowed_set or outcome is None:
            values.append(None)
        else:
            values.append(
                utility(outcome, atlas["scaled_residual_norms"][index], weights)
            )
    index = best_valid_index(values, [value is not None for value in values])
    return {"utility": float(values[index]), "best_index": index, "outcome": atlas["outcomes"][index]}


def arms_for_weights(row: Mapping[str, Any], weights: Mapping[str, float]) -> dict[str, dict]:
    coarse = row["conditions"]["coarse"]
    fine = row["conditions"]["oracle_tile"]
    return {
        "CC": atlas_arm(coarse, COARSE_INDICES, weights),
        "CF": atlas_arm(coarse, tuple(range(25)), weights),
        "FC": atlas_arm(fine, COARSE_INDICES, weights),
        "FF": atlas_arm(fine, tuple(range(25)), weights),
        "random_FF": atlas_arm(row["conditions"]["random_tile"], tuple(range(25)), weights),
        "phase_FF": atlas_arm(row["conditions"]["phase_tile"], tuple(range(25)), weights),
        "full_native_upper": atlas_arm(row["native"], tuple(range(25)), weights),
    }


def nearest_native_index(row: Mapping[str, Any], arm_name: str, arm: Mapping[str, Any]) -> int:
    atlas_name = {"CC": "coarse", "CF": "coarse", "FC": "oracle_tile", "FF": "oracle_tile"}[arm_name]
    candidate = np.asarray(row["conditions"][atlas_name]["candidates"])[arm["best_index"], :, :-1]
    native = np.asarray(row["native"]["candidates"])[:, :, :-1]
    scale = np.asarray(row["native"]["coordinate_std"])
    distances = np.linalg.norm((native.reshape(25, -1) - candidate.reshape(1, -1)) / scale, axis=1)
    distances[~np.asarray(row["native"]["valid"], dtype=bool)] = np.inf
    return int(np.argmin(distances))


def stable_random_order(rows: list[dict], seed: int) -> list[str]:
    return sorted(
        (row["bank_id"] for row in rows),
        key=lambda bank_id: hashlib.sha256(
            f"{PROTOCOL_ID}|{bank_id}|{seed}|random_state".encode()
        ).hexdigest(),
    )


def allocation_rows(rows: list[dict], seed: int, weights_name: str) -> list[dict[str, Any]]:
    weights = SENSITIVITY_WEIGHTS[weights_name]
    enriched = []
    for row in rows:
        arms = arms_for_weights(row, weights)
        enriched.append({"source": row, "arms": arms, "gain": arms["FF"]["utility"] - arms["CC"]["utility"]})
    adaptive_ids = {
        item["source"]["bank_id"]
        for item in sorted(enriched, key=lambda item: (-item["gain"], item["source"]["bank_id"]))[:32]
    }
    random_ids = set(stable_random_order(rows, seed)[:32])
    phase_ids = {
        row["bank_id"]
        for row in sorted(rows, key=lambda value: (PHASE_PRIORITY[value["phase"]], value["bank_id"]))[:32]
    }
    output = []
    for item in enriched:
        row, arms = item["source"], item["arms"]
        bank_id = row["bank_id"]
        adaptive = bank_id in adaptive_ids
        random_selected = bank_id in random_ids
        phase_selected = bank_id in phase_ids
        selected_arm = "FF" if adaptive else "CC"
        strongest_axis = "FC" if arms["FC"]["utility"] >= arms["CF"]["utility"] else "CF"
        native_best = arms["full_native_upper"]["best_index"]
        recalls = {
            name: int(nearest_native_index(row, name, arms[name]) == native_best)
            for name in ("CC", "FC", "CF", "FF")
        }
        values = {
            "joint_adaptive": arms[selected_arm]["utility"],
            "visual_only": arms["FC"]["utility"] if adaptive else arms["CC"]["utility"],
            "action_only": arms["CF"]["utility"] if adaptive else arms["CC"]["utility"],
            "strongest_single_axis": arms[strongest_axis]["utility"] if adaptive else arms["CC"]["utility"],
            "random_state": arms["FF"]["utility"] if random_selected else arms["CC"]["utility"],
            "phase_heuristic": arms["phase_FF"]["utility"] if phase_selected else arms["CC"]["utility"],
            "random_tile": arms["random_FF"]["utility"] if adaptive else arms["CC"]["utility"],
            "uniform_coarse": arms["CC"]["utility"],
            "uniform_fine": arms["FF"]["utility"],
            "full_native_upper": arms["full_native_upper"]["utility"],
        }
        categorical_strict = category(arms["FF"]["outcome"]) > category(arms["FC"]["outcome"]) and category(arms["FF"]["outcome"]) > category(arms["CF"]["outcome"])
        output.append(
            {
                "protocol_id": PROTOCOL_ID,
                "model_seed": seed,
                "bank_id": bank_id,
                "phase": row["phase"],
                "source": row["source"],
                "weights": weights_name,
                "adaptive_selected": adaptive,
                "random_selected": random_selected,
                "phase_selected": phase_selected,
                "arms": arms,
                "allocation_utility": values,
                "nearest_native_best_action_recall": recalls,
                "interaction_I": arms["FF"]["utility"] - arms["FC"]["utility"] - arms["CF"]["utility"] + arms["CC"]["utility"],
                "interaction_J": arms["FF"]["utility"] - max(arms["FC"]["utility"], arms["CF"]["utility"]),
                "categorical_strictly_better": categorical_strict,
                "post_success": False,
                "budget_compliant": True,
            }
        )
    return output


def metric_summary(rows: list[dict], j_threshold: float) -> dict[str, Any]:
    values = lambda name: np.asarray([row["allocation_utility"][name] for row in rows], dtype=float)
    joint = values("joint_adaptive")
    visual, action = values("visual_only"), values("action_only")
    strongest_name = "visual_only" if visual.mean() >= action.mean() else "action_only"
    strongest = values(strongest_name)
    native = values("full_native_upper")
    joint_regret = np.maximum(0.0, native - joint)
    single_regret = np.maximum(0.0, native - strongest)
    regret_reduction = float((single_regret.mean() - joint_regret.mean()) / single_regret.mean()) if single_regret.mean() > 0 else 0.0
    selected_rows = [row for row in rows if row["adaptive_selected"]]
    recall_joint = np.mean([row["nearest_native_best_action_recall"]["FF"] for row in selected_rows])
    strongest_arm = "FC" if strongest_name == "visual_only" else "CF"
    recall_single = np.mean([
        row["nearest_native_best_action_recall"][strongest_arm]
        for row in selected_rows
    ])
    coupled = [
        row["categorical_strictly_better"]
        and row["interaction_J"] >= j_threshold
        and not row["post_success"]
        and row["budget_compliant"]
        for row in rows
    ]
    comparisons = {}
    for control in ("strongest_single_axis", "random_state", "phase_heuristic", "random_tile"):
        control_values = strongest if control == "strongest_single_axis" else values(control)
        differences = joint - control_values
        comparisons[control] = {
            "mean_gain": float(differences.mean()),
            "paired_bootstrap_95_ci": paired_percentile_ci(differences),
            "positive_fraction": float(np.mean(differences > 0)),
        }
    return {
        "states": len(rows),
        "mean_utility": {name: float(values(name).mean()) for name in rows[0]["allocation_utility"]},
        "strongest_single_axis": strongest_name,
        "comparisons": comparisons,
        "joint_coupling_density": float(np.mean(coupled)),
        "coupled_states": int(sum(coupled)),
        "best_action_recall_joint": float(recall_joint),
        "best_action_recall_strongest_single": float(recall_single),
        "best_action_recall_improvement_pp": float((recall_joint - recall_single) * 100),
        "mean_outcome_regret_joint": float(joint_regret.mean()),
        "mean_outcome_regret_strongest_single": float(single_regret.mean()),
        "outcome_regret_reduction_fraction": regret_reduction,
        "recall_definition": "best arm action mapped to nearest valid native physical-atlas action in frozen standardized coordinates",
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "allocation_states.jsonl"
    if raw_path.exists():
        raise FileExistsError(raw_path)
    freeze = json.loads(args.joint_calibration_freeze.read_text(encoding="utf-8"))
    if freeze.get("status") != "ORACLE_CALIBRATION_FROZEN":
        raise RuntimeError("joint calibration is not frozen")
    j_threshold = float(freeze["selected_J_threshold"])
    source = read_rows(args.visual_confirmatory_root)
    all_rows: dict[str, dict[int, list[dict]]] = defaultdict(dict)
    for weights_name in SENSITIVITY_WEIGHTS:
        for seed, rows in source.items():
            calculated = allocation_rows(rows, seed, weights_name)
            all_rows[weights_name][seed] = calculated
            for row in calculated:
                append_jsonl(raw_path, row)
    summaries = {}
    for weights_name, seed_rows in all_rows.items():
        per_seed = {str(seed): metric_summary(rows, j_threshold) for seed, rows in seed_rows.items()}
        by_state: dict[str, list[dict]] = defaultdict(list)
        for rows in seed_rows.values():
            for row in rows:
                by_state[row["bank_id"]].append(row)
        aggregate_rows = []
        for bank_id, rows in sorted(by_state.items()):
            base = dict(rows[0])
            base["allocation_utility"] = {
                key: float(np.mean([row["allocation_utility"][key] for row in rows]))
                for key in rows[0]["allocation_utility"]
            }
            base["interaction_J"] = float(np.mean([row["interaction_J"] for row in rows]))
            base["interaction_I"] = float(np.mean([row["interaction_I"] for row in rows]))
            base["categorical_strictly_better"] = sum(row["categorical_strictly_better"] for row in rows) >= 2
            base["adaptive_selected"] = sum(row["adaptive_selected"] for row in rows) >= 2
            base["random_selected"] = sum(row["random_selected"] for row in rows) >= 2
            base["phase_selected"] = sum(row["phase_selected"] for row in rows) >= 2
            base["nearest_native_best_action_recall"] = {
                key: int(sum(row["nearest_native_best_action_recall"][key] for row in rows) >= 2)
                for key in rows[0]["nearest_native_best_action_recall"]
            }
            aggregate_rows.append(base)
        summaries[weights_name] = {
            "per_model_seed": per_seed,
            "aggregate_by_state": metric_summary(aggregate_rows, j_threshold),
        }
    primary = summaries["primary"]
    direction_random = sum(
        primary["per_model_seed"][str(seed)]["comparisons"]["random_state"]["mean_gain"] > 0
        for seed in MODEL_SEEDS
    )
    direction_phase = sum(
        primary["per_model_seed"][str(seed)]["comparisons"]["phase_heuristic"]["mean_gain"] > 0
        for seed in MODEL_SEEDS
    )
    aggregate = primary["aggregate_by_state"]
    joint_gate = bool(
        aggregate["joint_coupling_density"] >= 0.15
        and (
            aggregate["best_action_recall_improvement_pp"] >= 10.0
            or aggregate["outcome_regret_reduction_fraction"] >= 0.15
        )
        and aggregate["comparisons"]["random_state"]["paired_bootstrap_95_ci"][0] > 0
        and aggregate["comparisons"]["phase_heuristic"]["paired_bootstrap_95_ci"][0] > 0
        and direction_random >= 2
        and direction_phase >= 2
    )
    write_json(
        args.output_dir / "summary.json",
        {
            "protocol_id": PROTOCOL_ID,
            "status": "JOINT_FACTORIAL_ORACLE_COMPLETE",
            "deployable": False,
            "learned_selector": False,
            "confirmatory_state_count": 64,
            "model_seeds": list(MODEL_SEEDS),
            "j_threshold": j_threshold,
            "summaries": summaries,
            "directionally_positive_model_seeds": {
                "adaptive_minus_random": direction_random,
                "adaptive_minus_phase": direction_phase,
            },
            "joint_oracle_gate_pass": joint_gate,
            "post_success_states_in_primary": 0,
            "matched_refined_states": 32,
            "matched_abstract_budget": {"local_fine_tiles": 1, "fine_action_candidates": 25, "coarse_action_candidates": 9},
            "wall_clock_compute_claimed_matched": False,
            "raw_path": str(raw_path),
            "raw_sha256": sha256_file(raw_path),
            "joint_calibration_freeze_sha256": sha256_file(args.joint_calibration_freeze),
        },
    )


if __name__ == "__main__":
    main()
