#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from common import MODEL_SEEDS, PROTOCOL_ID, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def bootstrap(values: list[float]) -> list[float]:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(16018)
    means = np.empty(10_000)
    for start in range(0, 10_000, 1000):
        stop = min(10_000, start + 1000)
        indices = rng.integers(0, len(data), size=(stop - start, len(data)))
        means[start:stop] = data[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def sign_flip_pvalue(values: list[float]) -> float:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(16018)
    observed = float(data.mean())
    exceed = 0
    for start in range(0, 10_000, 1000):
        stop = min(10_000, start + 1000)
        signs = rng.integers(0, 2, size=(stop - start, len(data))) * 2 - 1
        exceed += int(np.sum((signs * data[None]).mean(axis=1) >= observed))
    return float((exceed + 1) / 10_001)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    result: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * value)
        result[name] = float(min(1.0, running))
    return result


def frozen_manifest_pass() -> tuple[bool, list[str]]:
    lines = (EXPERIMENT_ROOT / "manifests" / "SCIENTIFIC_SHA256SUMS").read_text().splitlines()
    failures = []
    for line in lines:
        expected, relative = line.split("  ", 1)
        if sha256_file(EXPERIMENT_ROOT / relative) != expected:
            failures.append(relative)
    return not failures, failures


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.result_root
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    problems = []
    # Independent baseline recomputation from the 600 raw confirmatory episodes.
    base = {}
    for task in ("StackCube-v1", "PushCube-v1"):
        per_seed, rows = {}, []
        for seed in MODEL_SEEDS:
            current = jsonl(root / "baseline_confirmatory" / task / f"seed_{seed}" / "episodes.jsonl")
            if len(current) != 100:
                problems.append(f"baseline_count:{task}:{seed}:{len(current)}")
            rows += current
            per_seed[seed] = float(np.mean([row["success_hold5"] for row in current]))
        once = float(np.mean([row["success_once"] for row in rows]))
        hold = float(np.mean([row["success_hold5"] for row in rows]))
        end = float(np.mean([row["success_at_end"] for row in rows]))
        base[task] = {"once": once, "hold5": hold, "end": end, "range_pp": 100 * (max(per_seed.values()) - min(per_seed.values())), "retention": end / once if once else 0.0}
    baseline_gate = bool(0.25 <= base["StackCube-v1"]["once"] <= 0.85 and base["StackCube-v1"]["hold5"] >= 0.30 and base["StackCube-v1"]["range_pp"] <= 20 and base["StackCube-v1"]["retention"] >= 0.60)
    # Independent stopping recomputation.
    stopping = False
    stopping_rows = {}
    fixed = {}
    for seed in MODEL_SEEDS:
        rows = jsonl(root / "success_semantics" / "fixed_horizon" / f"seed_{seed}" / "episodes.jsonl")
        fixed.update({(seed, row["episode_seed"]): row for row in rows})
    for mode in ("terminate_first_success", "terminate_hold5", "neutral_after_hold5"):
        current, per_seed = {}, {}
        for seed in MODEL_SEEDS:
            rows = jsonl(root / "success_semantics" / mode / f"seed_{seed}" / "episodes.jsonl")
            current.update({(seed, row["episode_seed"]): row for row in rows})
            per_seed[seed] = np.mean([row["success_at_end"] for row in rows]) - np.mean([fixed[(seed, row["episode_seed"])]["success_at_end"] for row in rows])
        diff = [float(current[key]["success_at_end"]) - float(fixed[key]["success_at_end"]) for key in sorted(fixed)]
        ci = bootstrap(diff)
        finding = bool(np.mean(diff) >= 0.10 and ci[0] > 0 and sum(value > 0 for value in per_seed.values()) >= 2)
        stopping |= finding
        stopping_rows[mode] = {"gain": float(np.mean(diff)), "ci": ci, "finding": finding}
    # Independent restoration raw audit.
    restoration_rows = jsonl(root / "state_banks" / "state_restoration_raw.jsonl")
    restoration_gate = bool(len(restoration_rows) == 112 and all(row["restoration_pass"] for row in restoration_rows) and np.mean([row["categorical_agreement"] for row in restoration_rows]) >= 0.95)
    # Independent action gate from candidate rows, using the frozen threshold.
    freeze = json.loads((root / "action_boundary" / "ACTION_CALIBRATION_FREEZE.json").read_text())
    threshold = str(float(freeze["selected_effect_threshold"]))
    action_by_state = defaultdict(list)
    for seed in MODEL_SEEDS:
        rows = jsonl(root / "action_boundary" / "confirmatory" / f"seed_{seed}" / "states.jsonl")
        if len(rows) != 64:
            problems.append(f"action_count:{seed}:{len(rows)}")
        for row in rows:
            outcomes = [value for value in row["atlas"]["outcomes"] if value is not None]
            action_by_state[row["bank_id"]].append((row["phase"], np.mean(row["atlas"]["valid"]), row["atlas"]["boundary_by_threshold"][threshold]["boundary_density"], np.mean([value["categorical_repeat_agreement"] for value in outcomes])))
    reduced = []
    for bank_id, rows in action_by_state.items():
        reduced.append((rows[0][0], np.mean([row[1] for row in rows]), np.mean([row[2] for row in rows]), np.mean([row[3] for row in rows])))
    phase = {name: np.mean([row[2] for row in reduced if row[0] == name]) for name in {row[0] for row in reduced}}
    action_gate = bool(np.mean([row[1] for row in reduced]) >= 0.90 and np.mean([row[3] for row in reduced]) >= 0.95 and np.mean([row[2] for row in reduced]) >= 0.20 and phase["placement_contact_near_completion"] - phase["free_space_approach"] >= 0.10)
    # Independent joint allocation and coupling recomputation from raw allocation rows.
    all_allocation = jsonl(root / "joint_oracle" / "allocation_states.jsonl")
    if len(all_allocation) != 576:
        problems.append(f"joint_all_weights_count:{len(all_allocation)}")
    allocation = [row for row in all_allocation if row["weights"] == "primary"]
    if len(allocation) != 192:
        problems.append(f"joint_primary_count:{len(allocation)}")
    state_groups = defaultdict(list)
    for row in allocation:
        state_groups[row["bank_id"]].append(row)
    differences_random, differences_phase, differences_single, coupled = [], [], [], []
    recall_joint, recall_single, regret_joint, regret_single = [], [], [], []
    joint_summary = json.loads((root / "joint_oracle" / "summary.json").read_text())
    j_threshold = float(joint_summary["j_threshold"])
    prepared = [
        (
            rows,
            {
                key: np.mean([row["allocation_utility"][key] for row in rows])
                for key in rows[0]["allocation_utility"]
            },
        )
        for rows in state_groups.values()
    ]
    for rows, means in prepared:
        differences_random.append(means["joint_adaptive"] - means["random_state"])
        differences_phase.append(means["joint_adaptive"] - means["phase_heuristic"])
        differences_single.append(means["joint_adaptive"] - means["strongest_single_axis"])
        coupled.append(
            sum(row["categorical_strictly_better"] for row in rows) >= 2
            and np.mean([row["interaction_J"] for row in rows]) >= j_threshold
            and not any(row["post_success"] for row in rows)
            and all(row["budget_compliant"] for row in rows)
        )
        if sum(row["adaptive_selected"] for row in rows) >= 2:
            recall_joint.append(int(sum(row["nearest_native_best_action_recall"]["FF"] for row in rows) >= 2))
            recall_single.append(int(sum(row["nearest_native_best_action_recall"]["strongest_single_axis"] for row in rows) >= 2))
        regret_joint.append(max(0.0, means["full_native_upper"] - means["joint_adaptive"]))
        regret_single.append(max(0.0, means["full_native_upper"] - means["strongest_single_axis"]))
    recall_pp = 100 * (np.mean(recall_joint) - np.mean(recall_single)) if recall_joint else 0.0
    regret_reduction = (np.mean(regret_single) - np.mean(regret_joint)) / np.mean(regret_single) if np.mean(regret_single) > 0 else 0.0
    positive_random = sum(np.mean([row["allocation_utility"]["joint_adaptive"] - row["allocation_utility"]["random_state"] for row in allocation if row["model_seed"] == seed]) > 0 for seed in MODEL_SEEDS)
    positive_phase = sum(np.mean([row["allocation_utility"]["joint_adaptive"] - row["allocation_utility"]["phase_heuristic"] for row in allocation if row["model_seed"] == seed]) > 0 for seed in MODEL_SEEDS)
    joint_gate = bool(np.mean(coupled) >= 0.15 and (recall_pp >= 10 or regret_reduction >= 0.15) and bootstrap(differences_random)[0] > 0 and bootstrap(differences_phase)[0] > 0 and positive_random >= 2 and positive_phase >= 2)
    sensitivity_robust = all(
        np.mean([
            row["allocation_utility"]["joint_adaptive"]
            - row["allocation_utility"][control]
            for row in all_allocation
            if row["weights"] == weights
        ]) > 0
        for weights in {row["weights"] for row in all_allocation}
        for control in ("strongest_single_axis", "random_state", "phase_heuristic")
    )
    visual_gain = float(np.mean([
        row["allocation_utility"]["visual_only"]
        - row["allocation_utility"]["uniform_coarse"]
        for row in allocation
    ]))
    action_gain = float(np.mean([
        row["allocation_utility"]["action_only"]
        - row["allocation_utility"]["uniform_coarse"]
        for row in allocation
    ]))
    # Independently reconstruct the preregistered Holm family from raw states.
    secondary_controls = ("random_state", "phase_heuristic", "random_tile", "phase_tile")
    holm_checks = {}
    for weights in sorted({row["weights"] for row in all_allocation}):
        groups = defaultdict(list)
        for row in all_allocation:
            if row["weights"] == weights:
                groups[row["bank_id"]].append(row)
        state_differences = {
            control: [
                float(np.mean([
                    row["allocation_utility"]["joint_adaptive"]
                    - row["allocation_utility"][control]
                    for row in rows
                ]))
                for bank_id in sorted(groups)
                for rows in (groups[bank_id],)
            ]
            for control in secondary_controls
        }
        adjusted = holm({
            control: sign_flip_pvalue(values)
            for control, values in state_differences.items()
        })
        reported = joint_summary["summaries"][weights]["aggregate_by_state"]["comparisons"]
        checks = {
            control: bool(
                "holm_adjusted_p" in reported[control]
                and abs(float(reported[control]["holm_adjusted_p"]) - adjusted[control])
                <= 1e-12
            )
            for control in secondary_controls
        }
        holm_checks[weights] = checks
        for control, passed in checks.items():
            if not passed:
                problems.append(f"holm_mismatch:{weights}:{control}")
    if not baseline_gate:
        final = "NO_GO_BASELINE_REPAIR"
    elif stopping:
        final = "REVISE_STOPPING_CONFOUND"
    elif not restoration_gate:
        final = "NO_GO_STATE_RESTORATION"
    elif not action_gate:
        final = "NO_GO_NO_ACTION_BOUNDARY"
    elif joint_gate and not sensitivity_robust:
        final = "REVISE_UTILITY_DEPENDENT"
    elif not joint_gate and action_gain > 0 >= visual_gain:
        final = "REVISE_ACTION_ONLY"
    elif not joint_gate and visual_gain > 0 >= action_gain:
        final = "REVISE_VISUAL_ONLY"
    elif not joint_gate:
        final = "REVISE_NO_JOINT_COUPLING"
    else:
        final = "GO_SINGLE_TASK_JOINT_ORACLE"
    frozen_pass, frozen_failures = frozen_manifest_pass()
    immutable_old = subprocess.run(["git", "-C", str(REPO_ROOT), "diff", "--quiet", "76e71f5eae9771b83906478f0c421183e38cdd9c", "--", "experiments/maniskill_act_boundary_screen_v1"], check=False).returncode == 0
    comparisons = {
        "baseline_gate": baseline_gate == summary["baseline"]["stackcube_gate_pass"],
        "stopping_confound": stopping == summary["success_semantics"]["stopping_confound"],
        "restoration_gate": restoration_gate == summary["state_restoration"]["restoration_gate_pass"],
        "action_gate": action_gate == summary["action_boundary"]["action_boundary_gate_pass"],
        "joint_gate": joint_gate == summary["joint_oracle"]["joint_oracle_gate_pass"],
        "utility_sensitivity": sensitivity_robust == summary["gate_ledger"]["utility_sensitivity_direction_robust"],
        "visual_axis_gain": bool(np.isclose(visual_gain, summary["gate_ledger"]["visual_axis_mean_gain_over_CC"])),
        "action_axis_gain": bool(np.isclose(action_gain, summary["gate_ledger"]["action_axis_mean_gain_over_CC"])),
        "final_status": final == summary["final_status"],
    }
    audit_pass = bool(all(comparisons.values()) and frozen_pass and immutable_old and not problems)
    write_json(args.output, {
        "protocol_id": PROTOCOL_ID,
        "status": "INDEPENDENT_STAGE25_AUDIT_PASS" if audit_pass else "INDEPENDENT_STAGE25_AUDIT_FAIL",
        "audit_pass": audit_pass,
        "independently_recomputed": {"baseline": base, "baseline_gate": baseline_gate, "stopping": stopping_rows, "stopping_confound": stopping, "restoration_rows": len(restoration_rows), "restoration_gate": restoration_gate, "action_states": len(reduced), "action_gate": action_gate, "joint_states": len(state_groups), "joint_coupling_density": float(np.mean(coupled)), "joint_minus_per_state_strongest_single_axis": float(np.mean(differences_single)), "best_action_recall_improvement_pp": float(recall_pp), "outcome_regret_reduction_fraction": float(regret_reduction), "joint_gate": joint_gate, "utility_sensitivity_direction_robust": sensitivity_robust, "visual_axis_mean_gain_over_CC": visual_gain, "action_axis_mean_gain_over_CC": action_gain, "holm_secondary_checks": holm_checks, "final_status": final},
        "summary_agreement": comparisons,
        "frozen_scientific_manifest_pass": frozen_pass,
        "frozen_manifest_failures": frozen_failures,
        "predecessor_directory_unchanged_vs_audited_commit": immutable_old,
        "problems": problems,
        "summary_sha256": sha256_file(args.summary),
        "decision_logic_source_independent_from_summarizer": True,
    })


if __name__ == "__main__":
    main()
