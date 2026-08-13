#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from protocol_common import MODEL_SEEDS, PROTOCOL_ID, sha256_file, write_json  # noqa: E402


NEGATIVE_CONTROL = "PushCube-v1"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 16018
EXPECTED_STATES = 64
EXPECTED_PER_STATE_ACCOUNTING = {
    "logical_policy_inputs": 49,
    "batched_policy_invocations": 2,
    "simulator_restores": 93,
    "simulator_steps": 372,
}
ORACLE_CONTRACT = (
    SCRIPT_DIR.parent / "action_atlas" / "oracle_implementation_contract.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-gate", type=Path, required=True)
    parser.add_argument("--state-bank-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def paired_state_ci(values: np.ndarray) -> list[float]:
    """Resample shared state identities and retain all model seeds."""

    if values.ndim != 2 or values.shape[0] != len(MODEL_SEEDS):
        raise ValueError(values.shape)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        state_indices = rng.integers(0, values.shape[1], size=values.shape[1])
        estimates[index] = float(values[:, state_indices].mean())
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def load_state_rows(
    summary: Mapping[str, Any], source_bindings: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for item in summary["surface_files"]:
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"] or path.stat().st_size != int(
            item["bytes"]
        ):
            raise RuntimeError(f"oracle state digest/size mismatch: {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") != "ORACLE_STATE_COMPLETE":
            raise RuntimeError(f"incomplete oracle state: {path}")
        if row.get("source_bindings") != source_bindings:
            raise RuntimeError(f"oracle state source binding mismatch: {path}")
        if row.get("implementation_contract_sha256") != sha256_file(
            ORACLE_CONTRACT
        ):
            raise RuntimeError(f"oracle state implementation binding mismatch: {path}")
        rows.append(row)
    rows.sort(key=lambda row: row["bank_id"])
    if len(rows) != EXPECTED_STATES:
        raise RuntimeError(f"expected {EXPECTED_STATES} oracle states")
    return rows


def summarize_task(oracle_root: Path, task_id: str) -> dict[str, Any]:
    summaries = []
    state_rows_by_seed = []
    bank_order: list[str] | None = None
    for model_seed in MODEL_SEEDS:
        path = oracle_root / task_id / f"seed_{model_seed}" / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        if (
            summary.get("protocol_id") != PROTOCOL_ID
            or summary.get("status") != "ORACLE_ATLAS_COMPLETE"
            or summary.get("task_id") != task_id
            or int(summary.get("model_seed", -1)) != model_seed
            or int(summary.get("states", -1)) != EXPECTED_STATES
        ):
            raise RuntimeError(f"invalid oracle summary: {path}")
        state_manifest_path = Path(summary["state_bank_manifest"])
        state_manifest = json.loads(
            state_manifest_path.read_text(encoding="utf-8")
        )
        state_h5 = Path(state_manifest["state_bank_h5"])
        train_h5 = Path(summary["train_h5"])
        checkpoint = Path(summary["source_bindings"]["selected_checkpoint_path"])
        expected_bindings = {
            "oracle_evaluator_sha256": sha256_file(
                SCRIPT_DIR / "evaluate_oracle_atlas.py"
            ),
            "state_bank_manifest_sha256": sha256_file(state_manifest_path),
            "state_bank_h5_sha256": sha256_file(state_h5),
            "train_h5_sha256": sha256_file(train_h5),
            "selected_checkpoint_step": int(summary["selected_checkpoint_step"]),
            "selected_checkpoint_sha256": sha256_file(checkpoint),
            "selected_checkpoint_path": str(checkpoint),
        }
        if (
            summary.get("source_bindings") != expected_bindings
            or summary.get("state_bank_manifest_sha256")
            != expected_bindings["state_bank_manifest_sha256"]
            or state_manifest.get("state_bank_h5_sha256")
            != expected_bindings["state_bank_h5_sha256"]
            or summary.get("train_h5_sha256")
            != expected_bindings["train_h5_sha256"]
            or summary.get("selected_checkpoint_sha256")
            != expected_bindings["selected_checkpoint_sha256"]
            or summary.get("implementation_contract_sha256")
            != sha256_file(ORACLE_CONTRACT)
        ):
            raise RuntimeError(f"oracle source binding mismatch: {path}")
        for key, per_state in EXPECTED_PER_STATE_ACCOUNTING.items():
            observed = int(summary["accounting"][key])
            expected = EXPECTED_STATES * per_state
            if observed != expected:
                raise RuntimeError(
                    f"oracle accounting mismatch {task_id}/{model_seed}/{key}: "
                    f"{observed} != {expected}"
                )
        rows = load_state_rows(summary, expected_bindings)
        order = [row["bank_id"] for row in rows]
        manifest_order = sorted(row["bank_id"] for row in state_manifest["states"])
        if order != manifest_order:
            raise RuntimeError(f"oracle/state-bank identities differ: {task_id}")
        if bank_order is None:
            bank_order = order
        elif order != bank_order:
            raise RuntimeError(f"state identities drifted across model seeds: {task_id}")
        summaries.append(summary)
        state_rows_by_seed.append(rows)

    def matrix(extractor: Any) -> np.ndarray:
        return np.asarray(
            [[float(extractor(row)) for row in rows] for rows in state_rows_by_seed],
            dtype=np.float64,
        )

    action = matrix(lambda row: row["action_atlas"]["action_boundary_density"])
    visual = matrix(lambda row: row["visual_atlas"]["visual_boundary_density"])
    joint = matrix(lambda row: row["joint_probe"]["joint_boundary"])
    coarse_recall = matrix(
        lambda row: row["action_atlas"]["coarse_best_action_recall"]
    )
    oracle_recall = matrix(
        lambda row: row["action_atlas"]["oracle_best_action_recall"]
    )
    coarse_regret = matrix(lambda row: row["action_atlas"]["coarse_outcome_regret"])
    oracle_regret = matrix(lambda row: row["action_atlas"]["oracle_outcome_regret"])
    seed_action = action.mean(axis=1)
    seed_joint = joint.mean(axis=1)
    seed_recall_gain_pp = 100.0 * (
        oracle_recall.mean(axis=1) - coarse_recall.mean(axis=1)
    )
    seed_regret_reduction = []
    for index in range(len(MODEL_SEEDS)):
        coarse = float(coarse_regret[index].mean())
        oracle = float(oracle_regret[index].mean())
        seed_regret_reduction.append((coarse - oracle) / coarse if coarse > 0 else 0.0)
    mean_coarse_regret = float(coarse_regret.mean())
    mean_oracle_regret = float(oracle_regret.mean())
    regret_reduction = (
        (mean_coarse_regret - mean_oracle_regret) / mean_coarse_regret
        if mean_coarse_regret > 0
        else 0.0
    )
    per_seed = []
    for index, model_seed in enumerate(MODEL_SEEDS):
        per_seed.append(
            {
                "model_seed": model_seed,
                "action_boundary_density": float(seed_action[index]),
                "joint_coupling_density": float(seed_joint[index]),
                "best_action_recall_improvement_percentage_points": float(
                    seed_recall_gain_pp[index]
                ),
                "outcome_regret_reduction_fraction": float(
                    seed_regret_reduction[index]
                ),
                "oracle_condition": bool(
                    seed_recall_gain_pp[index] >= 10.0
                    or seed_regret_reduction[index] >= 0.15
                ),
            }
        )
    result = {
        "task_id": task_id,
        "states_per_model_seed": EXPECTED_STATES,
        "model_seeds": list(MODEL_SEEDS),
        "per_seed": per_seed,
        "action_boundary_density": float(action.mean()),
        "action_boundary_density_paired_state_bootstrap_95_ci": paired_state_ci(action),
        "visual_boundary_density": float(visual.mean()),
        "visual_boundary_density_paired_state_bootstrap_95_ci": paired_state_ci(visual),
        "joint_coupling_density": float(joint.mean()),
        "joint_coupling_density_paired_state_bootstrap_95_ci": paired_state_ci(joint),
        "coarse_best_action_recall": float(coarse_recall.mean()),
        "oracle_best_action_recall": float(oracle_recall.mean()),
        "best_action_recall_improvement_percentage_points": float(
            100.0 * (oracle_recall.mean() - coarse_recall.mean())
        ),
        "coarse_outcome_regret": mean_coarse_regret,
        "oracle_outcome_regret": mean_oracle_regret,
        "outcome_regret_reduction_fraction": regret_reduction,
        "action_direction_reproducing_seeds": int(np.sum(seed_action >= 0.20)),
        "joint_direction_reproducing_seeds": int(np.sum(seed_joint >= 0.15)),
        "oracle_direction_reproducing_seeds": int(
            sum(row["oracle_condition"] for row in per_seed)
        ),
        "complete_call_and_opportunity_accounting": True,
    }
    result["action_gate_pass"] = bool(
        result["action_boundary_density"] >= 0.20
        and result["action_direction_reproducing_seeds"] >= 2
    )
    result["joint_gate_pass"] = bool(
        result["joint_coupling_density"] >= 0.15
        and result["joint_direction_reproducing_seeds"] >= 2
    )
    result["oracle_condition_pass"] = bool(
        (
            result["best_action_recall_improvement_percentage_points"] >= 10.0
            or result["outcome_regret_reduction_fraction"] >= 0.15
        )
        and result["oracle_direction_reproducing_seeds"] >= 2
    )
    return result


def gate_from_task_aggregates(
    positive_tasks: Sequence[str], tasks: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    action_pass = [task for task in positive_tasks if tasks[task]["action_gate_pass"]]
    joint_pass = [task for task in positive_tasks if tasks[task]["joint_gate_pass"]]
    pooled_coarse_recall = float(
        np.mean([tasks[task]["coarse_best_action_recall"] for task in positive_tasks])
    )
    pooled_oracle_recall = float(
        np.mean([tasks[task]["oracle_best_action_recall"] for task in positive_tasks])
    )
    pooled_coarse_regret = float(
        np.mean([tasks[task]["coarse_outcome_regret"] for task in positive_tasks])
    )
    pooled_oracle_regret = float(
        np.mean([tasks[task]["oracle_outcome_regret"] for task in positive_tasks])
    )
    pooled_recall_gain_pp = 100.0 * (pooled_oracle_recall - pooled_coarse_recall)
    pooled_regret_reduction = (
        (pooled_coarse_regret - pooled_oracle_regret) / pooled_coarse_regret
        if pooled_coarse_regret > 0
        else 0.0
    )
    pooled_seed_directions = 0
    for seed_index in range(len(MODEL_SEEDS)):
        recall = float(
            np.mean(
                [
                    tasks[task]["per_seed"][seed_index][
                        "best_action_recall_improvement_percentage_points"
                    ]
                    for task in positive_tasks
                ]
            )
        )
        # Per-task per-seed reduction is the only stored seed-level regret statistic.
        reduction = float(
            np.mean(
                [
                    tasks[task]["per_seed"][seed_index][
                        "outcome_regret_reduction_fraction"
                    ]
                    for task in positive_tasks
                ]
            )
        )
        pooled_seed_directions += int(recall >= 10.0 or reduction >= 0.15)
    pooled_oracle_pass = bool(
        (pooled_recall_gain_pp >= 10.0 or pooled_regret_reduction >= 0.15)
        and pooled_seed_directions >= 2
    )
    negative = tasks[NEGATIVE_CONTROL]
    negative_seed_pass = sum(
        row["joint_coupling_density"] <= 0.10 for row in negative["per_seed"]
    )
    negative_pass = bool(
        negative["joint_coupling_density"] <= 0.10 and negative_seed_pass >= 2
    )
    accounting_pass = all(
        tasks[task]["complete_call_and_opportunity_accounting"]
        for task in [*positive_tasks, NEGATIVE_CONTROL]
    )
    overall = bool(
        len(action_pass) >= 2
        and len(joint_pass) >= 2
        and pooled_oracle_pass
        and negative_pass
        and accounting_pass
    )
    return {
        "decision": "GO" if overall else "NO_GO",
        "continue_to_stage3": overall,
        "positive_action_gate_tasks": action_pass,
        "positive_joint_gate_tasks": joint_pass,
        "minimum_positive_tasks_per_density_gate": 2,
        "pooled_positive_best_action_recall_improvement_percentage_points": pooled_recall_gain_pp,
        "pooled_positive_outcome_regret_reduction_fraction": pooled_regret_reduction,
        "pooled_positive_oracle_direction_reproducing_seeds": pooled_seed_directions,
        "pooled_positive_oracle_condition_pass": pooled_oracle_pass,
        "negative_control_joint_coupling_density": negative["joint_coupling_density"],
        "negative_control_seeds_lte_0_10": int(negative_seed_pass),
        "negative_control_pass": negative_pass,
        "complete_call_and_opportunity_accounting": accounting_pass,
    }


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline_gate.read_text(encoding="utf-8"))
    if (
        baseline.get("protocol_id") != PROTOCOL_ID
        or baseline.get("continue_to_oracle_probe") is not True
    ):
        raise RuntimeError("baseline gate did not authorize the oracle probe")
    positives = list(baseline["passing_positive_tasks"])
    active_tasks = [*positives, NEGATIVE_CONTROL]
    state_bank_failures = []
    for task_id in active_tasks:
        path = args.state_bank_root / task_id / "state_bank_manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "STATE_BANK_COMPLETE":
            state_bank_failures.append(
                {"task_id": task_id, "manifest": str(path), "status": value.get("status")}
            )
    if state_bank_failures:
        result = {
            "protocol_id": PROTOCOL_ID,
            "status": "REVISE_STATE_RESTORATION_GATE",
            "decision": "REVISE",
            "continue_to_stage3": False,
            "state_bank_failures": state_bank_failures,
            "thresholds_changed_after_results": False,
        }
        write_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    tasks = {task_id: summarize_task(args.oracle_root, task_id) for task_id in active_tasks}
    gate = gate_from_task_aggregates(positives, tasks)
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "GO_STAGE2_ORACLE_GATE" if gate["continue_to_stage3"] else "NO_GO_STAGE2_ORACLE_GATE",
        **gate,
        "baseline_passing_positive_tasks": positives,
        "tasks": tasks,
        "thresholds": {
            "positive_action_boundary_density_gte": 0.20,
            "positive_joint_coupling_density_gte": 0.15,
            "best_action_recall_improvement_percentage_points_gte": 10.0,
            "outcome_regret_reduction_fraction_gte": 0.15,
            "PushCube-v1_joint_coupling_density_lte": 0.10,
            "qualitative_direction_minimum_reproducing_model_seeds": 2,
        },
        "bootstrap": {
            "method": "paired_percentile_resample_frozen_state_identity",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
        },
        "oracle_is_privileged_and_not_deployable": True,
        "thresholds_changed_after_results": False,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
