#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from protocol_common import FORMAL_TASKS, MODEL_SEEDS, PROTOCOL_ID, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(sum(float(row[key]) for row in rows) / len(rows))


def group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "mean_intended_contact_events": mean(rows, "intended_contact_events"),
        "mean_unintended_contact_events": mean(rows, "unintended_contact_events"),
        "mean_collisions": mean(rows, "collisions"),
        "mean_policy_latency_seconds": mean(rows, "policy_latency_seconds"),
    }


def task_diagnostics(
    evaluation_root: Path, checkpoint_root: Path, task_id: str
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    per_seed = []
    for model_seed in MODEL_SEEDS:
        run_root = evaluation_root / task_id / f"seed_{model_seed}"
        seed_rows = load_jsonl(run_root / "episodes.jsonl")
        selection = load_json(
            checkpoint_root / task_id / f"seed_{model_seed}" / "checkpoint_selection.json"
        )
        selected = selection["selected"]
        per_seed.append(
            {
                "model_seed": model_seed,
                "success_once": mean(seed_rows, "success_once"),
                "success_at_end": mean(seed_rows, "success_at_end"),
                "selected_checkpoint_step": int(selected["step"]),
                "selected_validation_loss": float(selected["validation_loss"]),
            }
        )
        rows.extend(seed_rows)

    successful_once = [row for row in rows if bool(row["success_once"])]
    successful_at_end = [row for row in rows if bool(row["success_at_end"])]
    transient_success = [
        row
        for row in rows
        if bool(row["success_once"]) and not bool(row["success_at_end"])
    ]
    never_successful = [row for row in rows if not bool(row["success_once"])]
    first_steps = [int(row["first_success_step"]) for row in successful_once]
    return {
        "task_id": task_id,
        "episodes": len(rows),
        "horizon": int(rows[0]["episode_length"]),
        "success_once_count": len(successful_once),
        "success_at_end_count": len(successful_at_end),
        "transient_success_count": len(transient_success),
        "post_success_retention": (
            len(successful_at_end) / len(successful_once) if successful_once else None
        ),
        "first_success_step": {
            "mean": mean(successful_once, "first_success_step"),
            "minimum": min(first_steps) if first_steps else None,
            "maximum": max(first_steps) if first_steps else None,
        },
        "per_seed": per_seed,
        "groups": {
            "successful_once": group_metrics(successful_once),
            "successful_at_end": group_metrics(successful_at_end),
            "transient_success": group_metrics(transient_success),
            "never_successful": group_metrics(never_successful),
            "all": group_metrics(rows),
        },
    }


def main() -> None:
    args = parse_args()
    gate = load_json(args.evaluation_root / "baseline_gate.json")
    if gate.get("protocol_id") != PROTOCOL_ID or gate.get("status") != "NO_GO_BASELINE_GATE":
        raise RuntimeError("this diagnostic is only valid for the frozen NO-GO baseline")
    diagnostics = {
        task_id: task_diagnostics(args.evaluation_root, args.checkpoint_root, task_id)
        for task_id in FORMAL_TASKS
    }
    evaluator = SCRIPT_DIR / "evaluate_official_act_protocol.py"
    trainer = SCRIPT_DIR / "train_official_act_protocol.py"
    summarizer = SCRIPT_DIR / "summarize_baseline.py"
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "BASELINE_FAILURE_MECHANISM_DIAGNOSTIC_COMPLETE",
        "decision": "NO_GO_BASELINE_GATE",
        "analysis_method": "code_first_observed_execution_path_plus_episode_stratification",
        "claim_taxonomy": {
            "confirmed_code_semantics": (
                "directly established by the frozen evaluator implementation"
            ),
            "observed_association": (
                "computed from the 1200 formal episodes; not a causal estimate"
            ),
            "bounded_inference": (
                "consistent with code and observations but not isolated by an ablation"
            ),
            "not_tested": "no data were generated for this claim",
        },
        "source_bindings": {
            "evaluator": {"path": str(evaluator), "sha256": sha256_file(evaluator)},
            "trainer": {"path": str(trainer), "sha256": sha256_file(trainer)},
            "summarizer": {"path": str(summarizer), "sha256": sha256_file(summarizer)},
        },
        "confirmed_code_semantics": [
            {
                "mechanism": "fixed_horizon_execution_after_success",
                "evidence": (
                    "ManiSkillVectorEnv is configured with ignore_terminations=True; "
                    "evaluate_batch executes every timestep in the frozen horizon, ORs "
                    "success into success_once, and reads success_at_end only after the loop."
                ),
                "symbols": ["make_env", "evaluate_batch"],
                "effect_on_metrics": (
                    "A policy can count as success_once and later lose the terminal success "
                    "predicate because policy actions continue after the first success."
                ),
            },
            {
                "mechanism": "per_step_temporal_action_aggregation",
                "evidence": (
                    "The official ACT emits 30-query chunks; temporal_action blends all "
                    "available predictions for the current timestep with exp(-0.01*k) weights, "
                    "and the policy is called at every environment step."
                ),
                "symbols": ["temporal_action", "make_official_args"],
                "effect_on_metrics": (
                    "All arms in this baseline share the same action smoothing and one "
                    "policy call per action opportunity; this is not an adaptive-resolution arm."
                ),
            },
            {
                "mechanism": "validation_only_checkpoint_selection",
                "evidence": (
                    "Each checkpoint is selected by minimum deterministic validation imitation "
                    "loss with earliest-step tie break; test outcomes are not inputs."
                ),
                "effect_on_metrics": (
                    "Closed-loop test performance cannot be repaired by choosing a different "
                    "checkpoint after observing these results."
                ),
            },
        ],
        "observed_and_bounded_findings": [
            {
                "finding": "floor_failure_is_not_single_seed_instability",
                "level": "observed_association",
                "evidence": (
                    "PullCubeTool success_once is 1% for all three model seeds; PushT is "
                    "2%, 7%, and 2%. Their seed ranges are 0pp and 5pp, both below 25pp."
                ),
                "interpretation": (
                    "The baseline failure is replicated across seeds rather than being caused "
                    "by one divergent training seed."
                ),
            },
            {
                "finding": "continued_actions_realize_post_success_loss",
                "level": "bounded_inference",
                "evidence": (
                    "StackCube has 164 success_once episodes but only 108 successful at the "
                    "horizon (56 transient; 65.85% retention). PushT has 11 success_once but "
                    "only 1 at the horizon (10 transient; 9.09% retention)."
                ),
                "interpretation": (
                    "The fixed-horizon code path directly permits the drop and the drop is "
                    "observed. Which physical post-success motion causes each loss was not "
                    "isolated, so no stronger causal claim is made."
                ),
            },
            {
                "finding": "collision_burden_separates_pushcube_failures",
                "level": "observed_association",
                "evidence": (
                    "PushCube never-success episodes average 7.398 collisions versus 0.535 "
                    "among success_once episodes; success is 57.33%, below the frozen 70% floor."
                ),
                "interpretation": (
                    "Excess unintended robot-table contact is a strong diagnostic marker of "
                    "failure in this run, but the observational split does not establish that "
                    "collisions are the sole cause."
                ),
            },
            {
                "finding": "stackcube_success_is_contact_cleaner",
                "level": "observed_association",
                "evidence": (
                    "StackCube success_once episodes average 0.250 collisions versus 1.566 in "
                    "never-success episodes; this is the only positive task that passes."
                ),
                "interpretation": (
                    "The passing task is associated with cleaner contact execution, consistent "
                    "with the intended outcome-boundary setting but not evidence for an adaptive "
                    "selector."
                ),
            },
            {
                "finding": "pusht_intended_contact_counter_is_noninformative",
                "level": "observed_association",
                "evidence": (
                    "The PushT intended-contact event count is zero in all 300 episodes, including "
                    "11 success_once episodes, under the frozen pair-contact predicate."
                ),
                "interpretation": (
                    "This contact channel cannot explain PushT success or failure in this run; "
                    "whether the cause is the predicate, actor representation, or force threshold "
                    "was not isolated."
                ),
            },
            {
                "finding": "early_pusht_checkpoint_is_not_sufficient_explanation",
                "level": "observed_association",
                "evidence": (
                    "PushT seed 16020 selected 5k by validation loss and scores 2%; seed 16018 "
                    "selected 80k and also scores 2%, while seed 16019 selected 85k and scores 7%."
                ),
                "interpretation": (
                    "The 5k selection alone is insufficient to explain the task-level floor."
                ),
            },
        ],
        "not_tested": [
            "action_boundary_density",
            "visual_boundary_density",
            "joint_coupling_density",
            "best_action_recall",
            "outcome_regret",
            "learned_effect_predictor",
            "boundary_predictor",
            "budgeted_joint_selector",
            "outcome_flip_margin",
            "boundary_normal_anisotropic_refinement",
            "marginal_value_of_compute_router",
            "contact_phase_hysteresis",
            "action_commutator_proxy",
        ],
        "tasks": diagnostics,
        "scope_conclusion": (
            "The baseline is not healthy enough to authorize the privileged oracle or Stage-3. "
            "These diagnostics explain observed baseline behavior; they do not validate, reject, "
            "or generate a new R16-P18 mechanism idea."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
