#!/usr/bin/env python3
"""Code-first reverse engineering of positive and negative oracle effects.

This diagnostic does not propose a new method. It traces measured gains and
regressions back to the implemented visual/action execution paths and outcome
components.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyze_stage27r import aggregate, state_table
from common import PROTOCOL_ID, atomic_json

COMPONENT_WEIGHTS = {"success_hold5": 100.0, "normalized_progress": 20.0, "recoverability_probability": 5.0, "drop_probability": -10.0, "collision_probability": -5.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = []
    for path in args.inputs:
        raw.extend(json.loads(path.read_text())["rows"])
    rows = aggregate(raw)
    states = state_table(rows, "balanced")
    records = []
    for state in states:
        best_single = "FC" if state["FC"]["utility"]["balanced"] >= state["CF"]["utility"]["balanced"] else "CF"
        for effect, after, before in (("visual", "FC", "CC"), ("action", "CF", "CC"), ("joint", "FF", best_single)):
            contributions = {}
            for field, weight in COMPONENT_WEIGHTS.items():
                contributions[field] = weight * (float(state[after][field]) - float(state[before][field]))
            records.append({
                "task": state["task"], "model_seed": state["seed"], "source_episode": state["source_episode"],
                "phase": state["phase"], "bank": state["bank"], "effect": effect,
                "before": before, "after": after, "utility_delta": float(sum(contributions.values())),
                "component_contributions": contributions,
                "success_transition": f"{int(state[before]['success_hold5'] > .5)}->{int(state[after]['success_hold5'] > .5)}",
                "gpu_latency_delta_ms": state[after]["accounting"]["gpu_latency_ms"] - state[before]["accounting"]["gpu_latency_ms"],
                "policy_call_delta": state[after]["accounting"]["policy_forward_calls"] - state[before]["accounting"]["policy_forward_calls"],
                "fine_encoder_call_delta": state[after]["accounting"]["fine_encoder_calls"] - state[before]["accounting"]["fine_encoder_calls"],
            })
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["task"], record["effect"], record["phase"])].append(record)
    phase_summary = {}
    for key, values in sorted(grouped.items()):
        task, effect, phase = key
        contributions = {field: float(np.mean([row["component_contributions"][field] for row in values])) for field in COMPONENT_WEIGHTS}
        phase_summary[f"{task}/{effect}/{phase}"] = {
            "states_x_seeds": len(values), "mean_utility_delta": float(np.mean([row["utility_delta"] for row in values])),
            "positive_fraction": float(np.mean([row["utility_delta"] > 0 for row in values])),
            "success_transitions": dict(Counter(row["success_transition"] for row in values)),
            "mean_component_contribution": contributions,
            "dominant_absolute_component": max(contributions, key=lambda field: abs(contributions[field])),
            "mean_gpu_latency_delta_ms": float(np.mean([row["gpu_latency_delta_ms"] for row in values])),
            "mean_policy_call_delta": float(np.mean([row["policy_call_delta"] for row in values])),
            "mean_fine_encoder_call_delta": float(np.mean([row["fine_encoder_call_delta"] for row in values])),
        }
    task_summary = {}
    for task in sorted({record["task"] for record in records}):
        task_summary[task] = {}
        for effect in ("visual", "action", "joint"):
            values = [row for row in records if row["task"] == task and row["effect"] == effect]
            by_phase = defaultdict(list)
            for row in values:
                by_phase[row["phase"]].append(row["utility_delta"])
            task_summary[task][effect] = {
                "mean_utility_delta": float(np.mean([row["utility_delta"] for row in values])),
                "positive_fraction": float(np.mean([row["utility_delta"] > 0 for row in values])),
                "best_phase": max(by_phase, key=lambda phase: np.mean(by_phase[phase])),
                "worst_phase": min(by_phase, key=lambda phase: np.mean(by_phase[phase])),
                "improved_success_count": sum(row["success_transition"] == "0->1" for row in values),
                "regressed_success_count": sum(row["success_transition"] == "1->0" for row in values),
            }
    atomic_json(args.output, {
        "protocol_id": PROTOCOL_ID,
        "scope": "mechanism_reverse_engineering_without_new_idea_generation",
        "confirmed_code_semantics": {
            "visual": "fine adds one selected 112x112 crop through the shared ResNet18 while retaining the 112x112 global branch",
            "action": "fine re-observes and queries every step; coarse queries once per four executed steps",
            "symmetry": "both interventions last exactly eight steps before a common fine continuation",
            "oracle": "FC and FF choose a tile by repeated physical outcome, so tile choice is privileged and not deployable",
        },
        "task_summary": task_summary,
        "phase_summary": phase_summary,
        "state_effect_records": records,
        "interpretation_boundary": "Component decomposition is descriptive of the executed code path; only matched-prefix treatment contrasts support bounded causal claims.",
    })


if __name__ == "__main__":
    main()
