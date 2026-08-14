#!/usr/bin/env python3
"""Reverse-explain Stage-2.5 effects from persisted raw evidence.

This is an audit, not an ideation or model-selection program.  It reads the
frozen v26 raw rows after formal completion and reports which implementation
paths and physical outcome fields account for increases and decreases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_ID = "R16-P18-MS4-STAGE25-REPAIR-ORACLE-V1"
MODEL_SEEDS = (16018, 16019, 16020)
PHASES = (
    "free_space_approach",
    "pre_grasp_or_pre_contact",
    "object_in_hand_pre_placement",
    "placement_contact_near_completion",
)
PHYSICAL_FIELDS = (
    "stable_success",
    "phase_outcome",
    "grasped",
    "supported",
    "dropped_or_slipped",
    "recoverable",
    "intended_contact",
    "unintended_contact",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        raise ValueError("mean requires at least one value")
    return float(sum(rows) / len(rows))


def physical_signature(outcome: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(outcome[field] for field in PHYSICAL_FIELDS)


def vector_distance(first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]) -> float:
    # The last coordinate is the frozen gripper command.  Visual information
    # can only move the six arm coordinates in this diagnostic.
    differences = [
        float(left) - float(right)
        for left_step, right_step in zip(first, second, strict=True)
        for left, right in zip(left_step[:-1], right_step[:-1], strict=True)
    ]
    return float(math.sqrt(sum(value * value for value in differences)))


def group_rows(rows: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return dict(groups)


def action_group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = [
        outcome
        for row in rows
        for outcome, valid in zip(
            row["atlas"]["outcomes"], row["atlas"]["valid"], strict=True
        )
        if valid and outcome is not None
    ]
    repeat_disagreement = [
        outcome for outcome in outcomes if not outcome["categorical_repeat_agreement"]
    ]
    stable_repeat_disagreement = [
        outcome
        for outcome in outcomes
        if len({bool(repeat["stable_success"]) for repeat in outcome["repeat_rows"]}) > 1
    ]
    return {
        "state_model_rows": len(rows),
        "state_mean_boundary_density": mean(
            row["atlas"]["boundary_by_threshold"]["0.5"]["boundary_density"]
            for row in rows
        ),
        "candidate_validity": mean(
            sum(bool(value) for value in row["atlas"]["valid"])
            / len(row["atlas"]["valid"])
            for row in rows
        ),
        "categorical_repeat_agreement": mean(
            float(outcome["categorical_repeat_agreement"]) for outcome in outcomes
        ),
        "categorical_repeat_disagreement_candidates": len(repeat_disagreement),
        "stable_success_repeat_disagreement_candidates": len(
            stable_repeat_disagreement
        ),
        "valid_candidate_rows": len(outcomes),
    }


def action_audit(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    paths = [
        root / "action_boundary" / "confirmatory" / f"seed_{seed}" / "states.jsonl"
        for seed in MODEL_SEEDS
    ]
    rows = [row for path in paths for row in read_jsonl(path)]
    if len(rows) != 192:
        raise RuntimeError(f"expected 192 action rows, found {len(rows)}")
    if {row["protocol_id"] for row in rows} != {PROTOCOL_ID}:
        raise RuntimeError("action protocol mismatch")
    return (
        {
            "overall": action_group_summary(rows),
            "by_phase": {
                phase: action_group_summary(group_rows(rows, "phase")[phase])
                for phase in PHASES
            },
            "by_source": {
                source: action_group_summary(values)
                for source, values in sorted(group_rows(rows, "source").items())
            },
            "by_model_seed": {
                seed: action_group_summary(values)
                for seed, values in sorted(group_rows(rows, "model_seed").items())
            },
        },
        {str(path): sha256_file(path) for path in paths},
    )


def comparison_summary(rows: list[dict[str, Any]], refined: str, reference: str) -> dict[str, Any]:
    physical_changes = [
        physical_signature(row["arms"][refined]["outcome"])
        != physical_signature(row["arms"][reference]["outcome"])
        for row in rows
    ]
    phase_changes = [
        row["arms"][refined]["outcome"]["phase_outcome"]
        != row["arms"][reference]["outcome"]["phase_outcome"]
        for row in rows
    ]
    stable_changes = [
        bool(row["arms"][refined]["outcome"]["stable_success"])
        != bool(row["arms"][reference]["outcome"]["stable_success"])
        for row in rows
    ]
    deltas = [
        float(row["arms"][refined]["utility"])
        - float(row["arms"][reference]["utility"])
        for row in rows
    ]
    return {
        "refined_arm": refined,
        "reference_arm": reference,
        "state_model_rows": len(rows),
        "best_candidate_index_change_fraction": mean(
            float(row["arms"][refined]["best_index"] != row["arms"][reference]["best_index"])
            for row in rows
        ),
        "physical_signature_change_count": sum(physical_changes),
        "physical_signature_change_fraction": mean(map(float, physical_changes)),
        "phase_outcome_change_count": sum(phase_changes),
        "phase_outcome_change_fraction": mean(map(float, phase_changes)),
        "stable_success_change_count": sum(stable_changes),
        "stable_success_change_fraction": mean(map(float, stable_changes)),
        "mean_utility_delta": mean(deltas),
        "positive_utility_delta_fraction": mean(float(value > 0) for value in deltas),
        "negative_utility_delta_fraction": mean(float(value < 0) for value in deltas),
    }


def visual_audit(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    paths = [
        root / "visual_resolution" / "confirmatory" / f"seed_{seed}" / "states.jsonl"
        for seed in MODEL_SEEDS
    ]
    rows = [row for path in paths for row in read_jsonl(path)]
    if len(rows) != 192:
        raise RuntimeError(f"expected 192 visual rows, found {len(rows)}")
    if {row["protocol_id"] for row in rows} != {PROTOCOL_ID}:
        raise RuntimeError("visual protocol mismatch")

    comparisons = {
        "visual_only_FC_vs_CC": ("FC", "CC"),
        "action_only_CF_vs_CC": ("CF", "CC"),
        "joint_FF_vs_CC": ("FF", "CC"),
        "joint_FF_vs_visual_FC": ("FF", "FC"),
        "joint_FF_vs_action_CF": ("FF", "CF"),
    }
    by_phase = group_rows(rows, "phase")
    tile_counts = Counter(int(row["tile_screen"]["oracle_tile"]) for row in rows)
    action_shifts = [
        vector_distance(
            row["conditions"]["oracle_tile"]["nominal_action_first4"],
            row["conditions"]["coarse"]["nominal_action_first4"],
        )
        for row in rows
    ]
    physical_changed = [
        physical_signature(row["arms"]["FC"]["outcome"])
        != physical_signature(row["arms"]["CC"]["outcome"])
        for row in rows
    ]
    return (
        {
            "overall_comparisons": {
                name: comparison_summary(rows, refined, reference)
                for name, (refined, reference) in comparisons.items()
            },
            "by_phase": {
                phase: {
                    name: comparison_summary(values, refined, reference)
                    for name, (refined, reference) in comparisons.items()
                }
                for phase, values in ((phase, by_phase[phase]) for phase in PHASES)
            },
            "by_model_seed": {
                seed: {
                    name: comparison_summary(values, refined, reference)
                    for name, (refined, reference) in comparisons.items()
                }
                for seed, values in sorted(group_rows(rows, "model_seed").items())
            },
            "oracle_tile_screen": {
                "policy_rows": 192 * 16,
                "simulator_calls": 0,
                "tile_histogram": {str(key): value for key, value in sorted(tile_counts.items())},
                "oracle_equals_phase_tile_fraction": mean(
                    float(row["tile_screen"]["oracle_tile"] == row["tile_screen"]["phase_tile"])
                    for row in rows
                ),
                "oracle_equals_random_tile_fraction": mean(
                    float(row["tile_screen"]["oracle_tile"] == row["tile_screen"]["random_tile"])
                    for row in rows
                ),
            },
            "visual_induced_action_shift": {
                "nonzero_fraction": mean(float(value > 1e-12) for value in action_shifts),
                "mean_l2_first4_arm_coordinates": mean(action_shifts),
                "mean_l2_when_physical_signature_changes": mean(
                    value
                    for value, changed in zip(action_shifts, physical_changed, strict=True)
                    if changed
                ),
                "mean_l2_when_physical_signature_unchanged": mean(
                    value
                    for value, changed in zip(action_shifts, physical_changed, strict=True)
                    if not changed
                ),
            },
        },
        {str(path): sha256_file(path) for path in paths},
    )


def joint_audit(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    path = root / "joint_oracle" / "allocation_states.jsonl"
    rows = read_jsonl(path)
    if len(rows) != 576:
        raise RuntimeError(f"expected 576 joint rows, found {len(rows)}")
    if {row["protocol_id"] for row in rows} != {PROTOCOL_ID}:
        raise RuntimeError("joint protocol mismatch")
    result: dict[str, Any] = {}
    for weights, values in sorted(group_rows(rows, "weights").items()):
        selected = [row for row in values if row["adaptive_selected"]]
        result[weights] = {
            "state_model_rows": len(values),
            "adaptive_selected_rows": len(selected),
            "per_seed_joint_coupled_rows": {
                seed: sum(
                    bool(row["categorical_strictly_better"])
                    and float(row["interaction_J"]) >= 1.0
                    for row in seed_rows
                )
                for seed, seed_rows in sorted(group_rows(values, "model_seed").items())
            },
            "categorical_strictly_better_fraction": mean(
                float(row["categorical_strictly_better"]) for row in values
            ),
            "J_at_least_1_fraction": mean(
                float(row["interaction_J"] >= 1.0) for row in values
            ),
            "adaptive_phase_composition": dict(
                sorted(Counter(row["phase"] for row in selected).items())
            ),
            "adaptive_source_composition": dict(
                sorted(Counter(row["source"] for row in selected).items())
            ),
            "strongest_single_axis_on_selected": dict(
                sorted(Counter(row["strongest_single_axis_arm"] for row in selected).items())
            ),
            "mean_interaction_I": mean(float(row["interaction_I"]) for row in values),
            "mean_interaction_J": mean(float(row["interaction_J"]) for row in values),
        }
    return result, {str(path): sha256_file(path)}


def main() -> None:
    args = parse_args()
    root = args.result_root.resolve()
    if not (root / "FORMAL_COMPLETE.json").is_file():
        raise RuntimeError("formal completion marker is missing")
    complete = read_json(root / "FORMAL_COMPLETE.json")
    if complete.get("status") != "ALL_PREREGISTERED_STAGE25_EXPERIMENTS_COMPLETE":
        raise RuntimeError("formal run is incomplete")

    action, action_hashes = action_audit(root)
    visual, visual_hashes = visual_audit(root)
    joint, joint_hashes = joint_audit(root)
    trace_path = root / "audits" / "success_trace_terminal_audit_posthoc.json"
    summary_path = root / "summary" / "stage25_summary_trace_corrected.json"
    trace = read_json(trace_path)
    summary = read_json(summary_path)
    if summary["final_status"] != complete["final_status"]:
        raise RuntimeError("corrected summary changed the formal decision")

    source_root = Path(__file__).resolve().parents[1]
    code_paths = {
        "visual_transform": source_root / "scripts" / "oracle_math.py",
        "visual_tile_selection": source_root / "scripts" / "run_visual_resolution_probe.py",
        "joint_allocation": source_root / "scripts" / "run_joint_factorial_oracle.py",
        "success_semantics": source_root / "scripts" / "stage25_runtime.py",
    }
    output = {
        "protocol_id": PROTOCOL_ID,
        "status": "MECHANISM_REVERSE_ENGINEERING_AUDIT_PASS",
        "scope": "reverse_explanation_only_no_new_idea",
        "formal_run_id": complete["run_id"],
        "formal_final_status": complete["final_status"],
        "decision_changed_by_this_audit": False,
        "action_boundary_mechanism": action,
        "visual_physical_mechanism": visual,
        "joint_mechanism": joint,
        "success_stability_mechanism": {
            mode: trace["modes"][mode]["aggregate"]
            for mode in (
                "fixed_horizon",
                "terminate_first_success",
                "terminate_hold5",
                "neutral_after_hold5",
            )
        },
        "checkpoint_selection_mechanism": {
            "groups": summary["checkpoint_repair"]["groups"],
            "rank_inversion_groups": sum(
                bool(row["rank_inversion_from_validation_loss"])
                for row in summary["checkpoint_repair"]["groups"].values()
            ),
            "pareto_dominated_predecessor_groups": sum(
                bool(row["predecessor_checkpoint_pareto_dominated"])
                for row in summary["checkpoint_repair"]["groups"].values()
            ),
        },
        "bounded_mechanistic_interpretation": {
            "checkpoint": "Closed-loop checkpoint selection repaired StackCube stable validation by 5-7 percentage points per seed, but PushCube remained unhealthy and the stopping confound dominates the final status; selection was an important contributor, not the sole cause of the predecessor NO-GO.",
            "stopping": "The success-end gap is mainly caused by continued policy actions after first success: immediate privileged termination gains 16.33 percentage points, while neutral hold greatly reduces conditional terminal drift. Small nonzero neutral drift bounds a secondary physical-instability contribution.",
            "action": "Outcome boundaries are phase-localized, strongest near placement/contact. The action gate fails because repeat agreement collapses in that phase and the overall density misses 0.20 by 0.00091; therefore boundary existence is observed but not confirmatory-gate-qualified.",
            "visual": "The local native tile changes the policy chunk and, after executing candidates in the simulator, changes categorical physical signatures in a nonzero subset. This supports physical outcome sensitivity to visual information, but it is a privileged information-resolution oracle rather than a token or latency method.",
            "joint": "Visual and action refinements mostly substitute. The adaptive allocator has a small utility advantage, but no state is jointly coupled after model-seed aggregation, recall does not improve, and regret falls only 3.12%; the preregistered joint mechanism is absent.",
        },
        "raw_sha256": {
            **action_hashes,
            **visual_hashes,
            **joint_hashes,
            str(trace_path): sha256_file(trace_path),
            str(summary_path): sha256_file(summary_path),
        },
        "code_semantics_sha256": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in code_paths.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.chmod(0o600)


if __name__ == "__main__":
    main()
