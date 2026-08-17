#!/usr/bin/env python3
"""Repeat-aware, source-episode-clustered Stage-2.7R analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import PROTOCOL_ID, atomic_json

ACCOUNTING_FIELDS = (
    "global_encoder_calls", "fine_encoder_calls", "policy_forward_calls",
    "policy_forward_rows", "visual_tokens", "action_opportunities",
    "executed_steps", "gpu_latency_ms", "simulator_latency_ms",
    "estimated_flops", "peak_memory_bytes", "selector_latency_ms",
    "episode_total_compute",
)


def _cluster_means(values, clusters):
    grouped = defaultdict(list)
    for value, cluster in zip(values, clusters):
        key = tuple(cluster) if isinstance(cluster, (list, tuple)) else cluster
        grouped[key].append(float(value))
    return np.asarray([np.mean(rows) for rows in grouped.values()], dtype=float)


def paired_summary(values, clusters, seed=2718001, n=10000):
    x = _cluster_means(values, clusters)
    if not len(x):
        raise ValueError("empty paired statistic")
    rng = np.random.default_rng(seed)
    draws = np.mean(x[rng.integers(0, len(x), size=(n, len(x)))], axis=1)
    observed = abs(float(np.mean(x)))
    flips = rng.choice((-1.0, 1.0), size=(n, len(x)))
    pvalue = (int(np.sum(np.abs(np.mean(flips * x, axis=1)) >= observed)) + 1) / (n + 1)
    return {
        "mean": float(np.mean(x)),
        "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
        "signflip_p": float(pvalue),
        "cluster_count": int(len(x)),
    }


def holm(pairs):
    ordered = sorted(pairs.items(), key=lambda item: item[1])
    adjusted, running, total = {}, 0.0, len(ordered)
    for rank, (name, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * float(pvalue)))
        adjusted[name] = running
    return adjusted


def aggregate(raw):
    grouped = defaultdict(list)
    for row in raw:
        key = (row["task"], row["model_seed"], row["bank"], row["bank_id"], row["source_episode"], row["phase"], row["condition"])
        grouped[key].append(row)
    result = []
    for key, rows in grouped.items():
        base = dict(zip(("task", "model_seed", "bank", "bank_id", "source_episode", "phase", "condition"), key))
        progress = np.asarray([r["normalized_progress"] for r in rows], float)
        success = np.asarray([r["success_hold5"] for r in rows], float)
        utilities = {name: np.asarray([r["utilities"][name] for r in rows], float) for name in rows[0]["utilities"]}
        base.update(
            repeat_count=len(rows), success_probability=float(success.mean()), success_hold5=float(success.mean()),
            mean_progress=float(progress.mean()), normalized_progress=float(progress.mean()),
            progress_variance=float(progress.var(ddof=1)) if len(progress) > 1 else 0.0,
            drop_probability=float(np.mean([r["dropped_or_slipped"] for r in rows])),
            collision_probability=float(np.mean([r["collision"] for r in rows])),
            recoverability_probability=float(np.mean([r["recoverable"] for r in rows])),
            utility={name: float(values.mean()) for name, values in utilities.items()},
            utility_standard_error={name: float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0 for name, values in utilities.items()},
            accounting={name: float(np.mean([r["accounting"][name] for r in rows])) for name in ACCOUNTING_FIELDS},
            cost=float(np.mean([r["accounting"]["estimated_flops"] for r in rows])),
            causal=all(r["causal_fidelity_pass"] for r in rows),
        )
        result.append(base)
    return result


def state_table(rows, weight):
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["task"], row["model_seed"], row["bank"], row["bank_id"], row["source_episode"], row["phase"])
        grouped[key][row["condition"]] = row
    out = []
    for key, conditions in grouped.items():
        fc_rows = [value for name, value in conditions.items() if name.startswith("FC_tile")]
        ff_rows = [value for name, value in conditions.items() if name.startswith("FF_tile")]
        if not fc_rows or len(fc_rows) != len(ff_rows) or "CC" not in conditions or "CF" not in conditions:
            raise RuntimeError(f"incomplete factorial {key}: {len(conditions)}")
        fc = max(fc_rows, key=lambda row: (row["utility"][weight], -int(row["condition"].split("tile")[-1])))
        ff = max(ff_rows, key=lambda row: (row["utility"][weight], -int(row["condition"].split("tile")[-1])))
        cc, cf = conditions["CC"], conditions["CF"]
        out.append({
            "key": key, "cluster": (key[0], key[4]), "task": key[0], "seed": key[1], "bank": key[2],
            "source_episode": key[4], "phase": key[5], "CC": cc, "CF": cf, "FC": fc, "FF": ff,
            "dv": fc["utility"][weight] - cc["utility"][weight],
            "da": cf["utility"][weight] - cc["utility"][weight],
            "dj": ff["utility"][weight] - max(fc["utility"][weight], cf["utility"][weight]),
        })
    return out


def arm_allocate(states, budget_fraction, weight):
    coarse = sum(state["CC"]["cost"] for state in states)
    full = sum(state["FF"]["cost"] for state in states)
    budget = budget_fraction * full

    def allocate(options):
        chosen, cost, upgrades = ["CC"] * len(states), coarse, []
        for index, state in enumerate(states):
            for mode in options:
                dc = state[mode]["cost"] - state["CC"]["cost"]
                du = state[mode]["utility"][weight] - state["CC"]["utility"][weight]
                upgrades.append((du / max(dc, 1.0), du, -dc, -index, index, mode, dc))
        for _, du, _, _, index, mode, dc in sorted(upgrades, reverse=True):
            if chosen[index] != "CC" or du <= 0 or cost + dc > budget:
                continue
            chosen[index], cost = mode, cost + dc
        return chosen, cost

    fixed_axis = "FC" if np.mean([s["dv"] for s in states]) >= np.mean([s["da"] for s in states]) else "CF"
    arms = {
        "all_coarse": (["CC"] * len(states), coarse), "all_fine": (["FF"] * len(states), full),
        "visual_only_oracle": allocate(("FC",)), "action_only_oracle": allocate(("CF",)),
        "strongest_equal_cost_fixed_axis": allocate((fixed_axis,)),
        "state_axis_oracle": allocate(("FC", "CF")), "joint_oracle": allocate(("FC", "CF", "FF")),
    }
    order = sorted(range(len(states)), key=lambda i: hashlib.sha256(str(states[i]["key"]).encode()).hexdigest())
    random_modes, random_cost = ["CC"] * len(states), coarse
    for index in order:
        dc = states[index]["FF"]["cost"] - states[index]["CC"]["cost"]
        if random_cost + dc <= budget:
            random_modes[index], random_cost = "FF", random_cost + dc
    arms["random_state"] = random_modes, random_cost
    phase_order = sorted(range(len(states)), key=lambda i: (states[i]["phase"] != "contact_placement_near_completion", states[i]["phase"] != "object_in_hand_pre_placement", i))
    heuristic_modes, heuristic_cost = ["CC"] * len(states), coarse
    for index in phase_order:
        dc = states[index]["FF"]["cost"] - states[index]["CC"]["cost"]
        if heuristic_cost + dc <= budget:
            heuristic_modes[index], heuristic_cost = "FF", heuristic_cost + dc
    arms["phase_heuristic"] = heuristic_modes, heuristic_cost

    result, selected = {}, {}
    for name, (modes, cost) in arms.items():
        selected[name] = modes
        chosen = [state[mode] for state, mode in zip(states, modes)]
        result[name] = {
            "success_hold5": float(np.mean([row["success_hold5"] for row in chosen])),
            "utility": float(np.mean([row["utility"][weight] for row in chosen])),
            "estimated_flops": float(cost), "cost": float(cost), "budget": float(budget),
            "budget_compliant": bool(cost <= budget + 1e-6), "refined_states": sum(mode != "CC" for mode in modes),
            "mean_episode_total_compute": float(np.mean([row["accounting"]["episode_total_compute"] for row in chosen])),
            "mean_gpu_latency_ms": float(np.mean([row["accounting"]["gpu_latency_ms"] for row in chosen])),
            "mean_peak_memory_bytes": float(np.mean([row["accounting"]["peak_memory_bytes"] for row in chosen])),
        }
    fixed_modes, joint_modes = selected["strongest_equal_cost_fixed_axis"], selected["joint_oracle"]
    clusters = [state["cluster"] for state in states]
    result["joint_oracle"]["paired_vs_strongest_fixed_success"] = paired_summary(
        [state[j]["success_hold5"] - state[f]["success_hold5"] for state, j, f in zip(states, joint_modes, fixed_modes)], clusters, seed=2718101)
    result["joint_oracle"]["paired_vs_strongest_fixed_utility"] = paired_summary(
        [state[j]["utility"][weight] - state[f]["utility"][weight] for state, j, f in zip(states, joint_modes, fixed_modes)], clusters, seed=2718102)
    result["fixed_axis"] = fixed_axis
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = []
    for path in args.inputs:
        raw.extend(json.loads(path.read_text())["rows"])
    rows = aggregate(raw)
    statistics, budgets, negative_budgets, holm_input = {}, {}, {}, {}
    for weight in ("balanced", "success_dominant", "progress_dominant"):
        states = state_table(rows, weight)
        statistics[weight] = {}
        for task in sorted({state["task"] for state in states}):
            task_states = [state for state in states if state["task"] == task]
            task_stats = {"per_model_seed": {}}
            for seed in sorted({state["seed"] for state in task_states}):
                seed_states = [state for state in task_states if state["seed"] == seed]
                task_stats["per_model_seed"][str(seed)] = {
                    effect: paired_summary([state[field] for state in seed_states], [state["cluster"] for state in seed_states], seed=2718200 + seed % 100)
                    for effect, field in (("visual", "dv"), ("action", "da"), ("joint", "dj"))
                }
            for effect, field in (("visual", "dv"), ("action", "da"), ("joint", "dj")):
                summary = paired_summary([state[field] for state in task_states], [state["cluster"] for state in task_states], seed=2718300 + len(statistics[weight]))
                task_stats[effect] = summary
                holm_input[f"{weight}/{task}/{effect}"] = summary["signflip_p"]
            seed_fractions = {
                str(seed): float(np.mean([state["dj"] > 0 for state in task_states if state["seed"] == seed]))
                for seed in sorted({state["seed"] for state in task_states})
            }
            task_stats["positive_joint_state_fraction_by_seed"] = seed_fractions
            task_stats["positive_joint_state_fraction"] = float(np.mean(list(seed_fractions.values())))
            task_stats["positive_joint_fraction_seeds_gte_0.10"] = sum(value >= .10 for value in seed_fractions.values())
            statistics[weight][task] = task_stats
        positive_states = [state for state in states if state["bank"] == "confirmatory"]
        budgets[weight] = {str(fraction): arm_allocate(positive_states, fraction, weight) for fraction in (.25, .50, .75)}
        negative_states = [state for state in states if state["bank"] == "negative"]
        negative_budgets[weight] = ({str(fraction): arm_allocate(negative_states, fraction, weight) for fraction in (.25, .50, .75)} if negative_states else {})
    for key, value in holm(holm_input).items():
        weight, task, effect = key.split("/")
        statistics[weight][task][effect]["holm_signflip_p"] = value
    atomic_json(args.output, {
        "protocol_id": PROTOCOL_ID, "aggregated_state_treatments": rows, "statistics": statistics,
        "budgets": budgets, "negative_control_budgets": negative_budgets,
        "bootstrap_replicates": 10000, "primary_unit": "source_episode",
        "offline_physical_oracle_labeling": {"simulator_rows": len(raw), "included_in_deployment_cost": False},
        "holm_family_size": len(holm_input),
    })


if __name__ == "__main__":
    main()
