#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import MODEL_SEEDS, PHASES, PROTOCOL_ID, sha256_file, write_json
from oracle_math import paired_percentile_ci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError(f"protocol mismatch: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def baseline(root: Path) -> dict[str, Any]:
    tasks = {}
    for task in ("StackCube-v1", "PushCube-v1"):
        per_seed = {}
        rows = []
        for seed in MODEL_SEEDS:
            path = root / "baseline_confirmatory" / task / f"seed_{seed}" / "episodes.jsonl"
            seed_rows = read_jsonl(path)
            if len(seed_rows) != 100:
                raise RuntimeError(f"incomplete baseline confirmatory: {path}")
            rows.extend(seed_rows)
            per_seed[str(seed)] = {
                key: float(np.mean([row[key] for row in seed_rows]))
                for key in ("success_once", "success_hold5", "success_at_end", "post_success_loss")
            }
        aggregate = {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("success_once", "success_hold5", "success_at_end", "post_success_loss")
        }
        aggregate["retention_success_at_end_over_once"] = (
            aggregate["success_at_end"] / aggregate["success_once"]
            if aggregate["success_once"] else 0.0
        )
        aggregate["success_hold5_model_seed_range_pp"] = 100 * (
            max(row["success_hold5"] for row in per_seed.values())
            - min(row["success_hold5"] for row in per_seed.values())
        )
        tasks[task] = {"per_model_seed": per_seed, "aggregate": aggregate}
    stack = tasks["StackCube-v1"]["aggregate"]
    stack_gate = bool(
        0.25 <= stack["success_once"] <= 0.85
        and stack["success_hold5"] >= 0.30
        and stack["success_hold5_model_seed_range_pp"] <= 20.0
        and stack["retention_success_at_end_over_once"] >= 0.60
    )
    return {
        "tasks": tasks,
        "stackcube_gate_pass": stack_gate,
        "pushcube_healthy_negative_control": tasks["PushCube-v1"]["aggregate"]["success_once"] >= 0.70,
    }


def checkpoint_repair(root: Path) -> dict[str, Any]:
    screen = read_json(root / "checkpoint" / "screen_selection.json")
    final = read_json(root / "checkpoint" / "final_selection.json")
    groups = {}
    for name, row in final["groups"].items():
        groups[name] = {
            "selected_step": row["selected"]["step"],
            "selected_hold5": row["selected"]["success_hold5"],
            "predecessor_step": row["predecessor_validation_loss_control"]["step"],
            "predecessor_hold5": row["predecessor_validation_loss_control"]["success_hold5"],
            "final_step": row["final_checkpoint_control"]["step"],
            "rank_inversion_from_validation_loss": row["rank_inversion_from_validation_loss"],
            "predecessor_checkpoint_pareto_dominated": row["predecessor_checkpoint_pareto_dominated"],
            "screen_winner_equals_final_winner": row["screen_winner_equals_final_winner"],
            "screen_spearman_loss_vs_hold5": screen["groups"][name]["spearman_validation_loss_vs_success_hold5"],
        }
    return {"groups": groups, "screen_selection_sha256": sha256_file(root / "checkpoint" / "screen_selection.json"), "final_selection_sha256": sha256_file(root / "checkpoint" / "final_selection.json")}


def stopping(root: Path) -> dict[str, Any]:
    modes = ("fixed_horizon", "terminate_first_success", "terminate_hold5", "neutral_after_hold5")
    by_mode_seed, indexed = {}, {}
    for mode in modes:
        by_mode_seed[mode] = {}
        indexed[mode] = {}
        for seed in MODEL_SEEDS:
            path = root / "success_semantics" / mode / f"seed_{seed}" / "episodes.jsonl"
            rows = read_jsonl(path)
            if len(rows) != 100:
                raise RuntimeError(f"incomplete stopping arm: {path}")
            by_mode_seed[mode][str(seed)] = {
                key: float(np.mean([row[key] for row in rows]))
                for key in ("success_once", "success_hold5", "success_at_end", "post_success_loss")
            }
            indexed[mode].update({(seed, row["episode_seed"]): row for row in rows})
    comparisons = {}
    for mode in modes[1:]:
        keys = sorted(indexed["fixed_horizon"])
        differences = [
            float(indexed[mode][key]["success_at_end"])
            - float(indexed["fixed_horizon"][key]["success_at_end"])
            for key in keys
        ]
        per_seed_gain = {
            str(seed): by_mode_seed[mode][str(seed)]["success_at_end"]
            - by_mode_seed["fixed_horizon"][str(seed)]["success_at_end"]
            for seed in MODEL_SEEDS
        }
        comparisons[mode] = {
            "success_at_end_gain": float(np.mean(differences)),
            "paired_bootstrap_95_ci": paired_percentile_ci(differences),
            "positive_model_seeds": sum(value > 0 for value in per_seed_gain.values()),
            "per_model_seed_gain": per_seed_gain,
        }
    confound = any(
        row["success_at_end_gain"] >= 0.10
        and row["paired_bootstrap_95_ci"][0] > 0
        and row["positive_model_seeds"] >= 2
        for row in comparisons.values()
    )
    return {"per_mode_model_seed": by_mode_seed, "comparisons": comparisons, "stopping_confound": confound}


def action_boundary(root: Path, freeze: dict) -> dict[str, Any]:
    threshold = str(float(freeze["selected_effect_threshold"]))
    per_seed, all_rows = {}, []
    for seed in MODEL_SEEDS:
        path = root / "action_boundary" / "confirmatory" / f"seed_{seed}" / "states.jsonl"
        rows = read_jsonl(path)
        if len(rows) != 64:
            raise RuntimeError(f"incomplete action boundary: {path}")
        all_rows.extend(rows)
        per_seed[str(seed)] = rows
    state_values = defaultdict(list)
    for rows in per_seed.values():
        for row in rows:
            density = row["atlas"]["boundary_by_threshold"][threshold]["boundary_density"]
            state_values[row["bank_id"]].append(
                {
                    "density": density,
                    "validity": float(np.mean(row["atlas"]["valid"])),
                    "repeat_agreement": float(np.mean([
                        outcome["categorical_repeat_agreement"]
                        for outcome in row["atlas"]["outcomes"] if outcome is not None
                    ])),
                    "phase": row["phase"],
                }
            )
    state_rows = []
    for bank_id, rows in state_values.items():
        state_rows.append({
            "bank_id": bank_id,
            "phase": rows[0]["phase"],
            "density": float(np.mean([row["density"] for row in rows])),
            "validity": float(np.mean([row["validity"] for row in rows])),
            "repeat_agreement": float(np.mean([row["repeat_agreement"] for row in rows])),
        })
    phase_density = {
        phase: float(np.mean([row["density"] for row in state_rows if row["phase"] == phase]))
        for phase in PHASES
    }
    overall = float(np.mean([row["density"] for row in state_rows]))
    validity = float(np.mean([row["validity"] for row in state_rows]))
    agreement = float(np.mean([row["repeat_agreement"] for row in state_rows]))
    contrast = phase_density["placement_contact_near_completion"] - phase_density["free_space_approach"]
    gate = bool(validity >= 0.90 and agreement >= 0.95 and overall >= 0.20 and contrast >= 0.10)
    post = {}
    for seed in MODEL_SEEDS:
        path = root / "action_boundary" / "post_success_diagnostic" / f"seed_{seed}" / "states.jsonl"
        rows = read_jsonl(path)
        post[str(seed)] = {
            "states": len(rows),
            "boundary_density": float(np.mean([row["atlas"]["boundary_by_threshold"][threshold]["boundary_density"] for row in rows])),
        }
    return {
        "selected_radius": freeze["selected_radius"],
        "selected_effect_threshold": freeze["selected_effect_threshold"],
        "state_mean_candidate_validity": validity,
        "same_action_categorical_repeat_agreement": agreement,
        "state_mean_boundary_density": overall,
        "phase_boundary_density": phase_density,
        "placement_minus_free_space_density": contrast,
        "action_boundary_gate_pass": gate,
        "post_success_diagnostic": post,
    }


def visual_summary(joint: dict) -> dict[str, Any]:
    primary = joint["summaries"]["primary"]["aggregate_by_state"]
    comparisons = primary["comparisons"]
    return {
        "joint_minus_strongest_single_axis": comparisons["strongest_single_axis"],
        "joint_minus_random_state": comparisons["random_state"],
        "joint_minus_phase_heuristic": comparisons["phase_heuristic"],
        "joint_minus_random_tile": comparisons["random_tile"],
        "mean_utility": primary["mean_utility"],
        "physical_outcome_evidence": "each condition executes the selected physical candidate in simulator; no conclusion is based on action-vector distance alone",
        "information_resolution_only": True,
        "wall_clock_compute_matched": False,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.result_root
    result: dict[str, Any] = {"protocol_id": PROTOCOL_ID, "status": "STAGE25_SUMMARY_COMPLETE"}
    result["checkpoint_repair"] = checkpoint_repair(root)
    result["baseline"] = baseline(root)
    result["success_semantics"] = stopping(root)
    result["contact_metrics"] = read_json(root / "contact" / "contact_metric_audit.json")
    result["state_restoration"] = read_json(root / "state_banks" / "state_restoration_audit.json")
    action_freeze = read_json(root / "action_boundary" / "ACTION_CALIBRATION_FREEZE.json")
    result["action_calibration"] = action_freeze
    result["action_boundary"] = action_boundary(root, action_freeze)
    joint = read_json(root / "joint_oracle" / "summary.json")
    result["joint_oracle"] = joint
    result["visual_resolution"] = visual_summary(joint)
    primary = joint["summaries"]["primary"]["aggregate_by_state"]
    sensitivity = joint["summaries"]
    sensitivity_robust = all(
        value["aggregate_by_state"]["comparisons"][control]["mean_gain"] > 0
        for value in sensitivity.values()
        for control in ("strongest_single_axis", "random_state", "phase_heuristic")
    )
    visual_gain = primary["mean_utility"]["visual_only"] - primary["mean_utility"]["uniform_coarse"]
    action_gain = primary["mean_utility"]["action_only"] - primary["mean_utility"]["uniform_coarse"]
    secondary = {
        "stopping_confound": result["success_semantics"]["stopping_confound"],
        "state_restoration_gate_pass": result["state_restoration"]["restoration_gate_pass"],
        "action_boundary_gate_pass": result["action_boundary"]["action_boundary_gate_pass"],
        "joint_oracle_gate_pass": joint["joint_oracle_gate_pass"],
        "utility_sensitivity_direction_robust": sensitivity_robust,
        "visual_axis_mean_gain_over_CC": visual_gain,
        "action_axis_mean_gain_over_CC": action_gain,
    }
    result["gate_ledger"] = secondary
    if not result["baseline"]["stackcube_gate_pass"]:
        final_status = "NO_GO_BASELINE_REPAIR"
    elif secondary["stopping_confound"]:
        final_status = "REVISE_STOPPING_CONFOUND"
    elif not secondary["state_restoration_gate_pass"]:
        final_status = "NO_GO_STATE_RESTORATION"
    elif not secondary["action_boundary_gate_pass"]:
        final_status = "NO_GO_NO_ACTION_BOUNDARY"
    elif joint["joint_oracle_gate_pass"] and not sensitivity_robust:
        final_status = "REVISE_UTILITY_DEPENDENT"
    elif not joint["joint_oracle_gate_pass"] and action_gain > 0 >= visual_gain:
        final_status = "REVISE_ACTION_ONLY"
    elif not joint["joint_oracle_gate_pass"] and visual_gain > 0 >= action_gain:
        final_status = "REVISE_VISUAL_ONLY"
    elif not joint["joint_oracle_gate_pass"]:
        final_status = "REVISE_NO_JOINT_COUPLING"
    else:
        final_status = "GO_SINGLE_TASK_JOINT_ORACLE"
    result["final_status"] = final_status
    result["claim_scope"] = "SINGLE_TASK_PRIVILEGED_ORACLE_ONLY"
    result["all_user_mandated_downstream_experiments_executed_despite_gates"] = True
    result["prohibited_followups_executed"] = []
    result["not_tested"] = [
        "learned effect or boundary predictor",
        "deployable visual/action selector",
        "OOD generalization",
        "multi-task oracle acceptance",
        "wall-clock or token-budget advantage",
        "Diffusion Policy, DINO-WM, pi0.5, or real robot",
    ]
    write_json(args.output, result)


if __name__ == "__main__":
    main()
