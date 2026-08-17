#!/usr/bin/env python3
"""Independent preregistered Stage-2.7R status decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import PROTOCOL_ID, atomic_json

WEIGHTS = ("balanced", "success_dominant", "progress_dominant")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--task-selection", type=Path, required=True)
    parser.add_argument("--state-banks", type=Path, nargs="+", required=True)
    parser.add_argument("--positive-tasks", nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text())
    task_selection = json.loads(args.task_selection.read_text())
    bank_payloads = [json.loads(path.read_text()) for path in args.state_banks]
    categorical_agreement = all(state["fidelity"]["categorical_agreement"] for bank in bank_payloads for state in bank["states"])
    task_gate = bool(task_selection["task_gates"]["StackCube-v1"]["pass"] and task_selection.get("selected_positive") and categorical_agreement)
    fidelity = [bank["fidelity_pass_rate"] for bank in bank_payloads]
    causal = min(fidelity) >= .95

    weight_gates = {}
    for weight in WEIGHTS:
        stats = analysis["statistics"][weight]
        visual = [stats[task]["visual"] for task in args.positive_tasks]
        action = [stats[task]["action"] for task in args.positive_tasks]
        joint = [stats[task]["joint"] for task in args.positive_tasks]
        budget = analysis["budgets"][weight]["0.5"]
        joint_arm, fixed_arm = budget["joint_oracle"], budget["strongest_equal_cost_fixed_axis"]
        success_pair = joint_arm["paired_vs_strongest_fixed_success"]
        success_gain = success_pair["mean"]
        compute_reduction = 1 - joint_arm["cost"] / max(fixed_arm["cost"], 1.0)
        weight_gates[weight] = {
            "visual": all(item["mean"] > 0 and item["ci95"][0] > 0 for item in visual),
            "action": sum(item["mean"] > 0 and item["ci95"][0] > 0 for item in action) >= 1 and all(item["mean"] >= 0 for item in action),
            "joint": all(item["ci95"][0] > 0 for item in joint),
            "joint_fraction": all(stats[task]["positive_joint_fraction_seeds_gte_0.10"] >= 2 for task in args.positive_tasks),
            "budget_joint": bool((success_gain >= .05 and success_pair["ci95"][0] > 0) or (abs(success_gain) <= .02 and compute_reduction >= .25)),
            "state_axis": budget["state_axis_oracle"]["utility"] > fixed_arm["utility"],
            "visual_budget": budget["visual_only_oracle"]["utility"] > max(budget["random_state"]["utility"], budget["all_coarse"]["utility"]),
            "budget_50_success_gain": success_gain,
            "budget_50_success_ci95": success_pair["ci95"],
            "budget_50_compute_reduction": compute_reduction,
        }

    stable = {gate: sum(values[gate] for values in weight_gates.values()) >= 2 for gate in ("visual", "action", "joint", "joint_fraction", "budget_joint", "state_axis", "visual_budget")}
    if not causal:
        status = "NO_GO_CAUSAL_BACKEND"
    elif not task_gate:
        status = "NO_GO_CORE_MECHANISM"
    elif not stable["visual"] and not stable["action"] and not stable["state_axis"]:
        status = "NO_GO_CORE_MECHANISM"
    elif stable["visual"] and not stable["action"] and not stable["joint"] and stable["visual_budget"]:
        status = "REVISE_VISUAL_ONLY"
    elif stable["visual"] and stable["action"] and not stable["joint"] and stable["state_axis"]:
        status = "REVISE_SHARED_AXIS_ROUTER"
    elif all(stable[name] for name in ("visual", "action", "joint", "joint_fraction", "budget_joint")):
        status = "GO_FULL_JOINT"
    else:
        status = "NO_GO_CORE_MECHANISM"
    result = {
        "protocol_id": PROTOCOL_ID, "final_status": status, "precedence_applied": True,
        "causal_backend_pass": causal, "fidelity_pass_rates": fidelity,
        "positive_task_screen_gate_pass": task_gate,
        "fresh_reset_categorical_agreement_100pct": categorical_agreement,
        "two_of_three_weight_set_gates": stable, "per_weight_gates": weight_gates,
        "downstream_cannot_reverse_upstream_failure": True,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
