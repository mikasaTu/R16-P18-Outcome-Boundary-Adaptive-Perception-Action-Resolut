#!/usr/bin/env python3
"""Independent raw-trace and accounting audit for Stage-2.7R.

This file intentionally does not import ``analyze_stage27r``.  It recomputes
episode outcomes from the stored traces, checks the mode-specific call
schedule, and computes its own source-episode clustered bootstrap/sign-flip
summaries.  It is an audit of the persisted oracle evidence, not a second
scientific analysis with tunable thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import PROTOCOL_ID, atomic_json, sha256_file

WEIGHTS = {
    "balanced": (100.0, 20.0, 5.0, -10.0, -5.0),
    "success_dominant": (120.0, 10.0, 3.0, -12.0, -6.0),
    "progress_dominant": (80.0, 35.0, 5.0, -10.0, -5.0),
}
ACCOUNTING = (
    "global_encoder_calls", "fine_encoder_calls", "policy_forward_calls",
    "policy_forward_rows", "visual_tokens", "action_opportunities",
    "executed_steps", "gpu_latency_ms", "simulator_latency_ms",
    "prefix_replay_simulator_latency_ms", "estimated_flops",
    "peak_memory_bytes", "selector_latency_ms", "episode_total_compute",
)


def _longest(values):
    best = run = 0
    for value in values:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def recompute_outcome(row):
    success = [bool(x) for x in row.get("success_trace", [])]
    reward = [float(x) for x in row.get("reward_trace", [])]
    contact = [bool(x) for x in row.get("intended_contact_trace", [])]
    grasp = [bool(x) for x in row.get("grasp_trace", [])]
    catastrophic = [bool(x) for x in row.get("catastrophic_trace", [])]
    streak = _longest(success)
    first_grasp = next((i for i, value in enumerate(grasp) if value), None)
    dropped = False
    if first_grasp is not None:
        dropped = any(not value for value in grasp[first_grasp + 1:]) and streak < 5
    return {
        "success_once": any(success),
        "success_hold5": streak >= 5,
        "success_at_end": bool(success[-1]) if success else False,
        "first_success_step": next((i + 1 for i, value in enumerate(success) if value), None),
        "longest_success_streak": streak,
        "post_success_loss": bool(any(success) and not (bool(success[-1]) if success else False)),
        "normalized_progress": (reward[-1] - reward[0]) if len(reward) > 1 else 0.0,
        "intended_contact": any(contact),
        "unintended_contact": any(catastrophic),
        "collision": any(catastrophic),
        "dropped_or_slipped": bool(dropped),
        "recoverable": bool(streak >= 5 or len(reward) < 2 or reward[-1] >= reward[0] - 0.05),
    }


def expected_schedule(row):
    n = len(row.get("success_trace", []))
    condition = str(row["condition"])
    action_fine = condition == "CF" or condition.startswith("FF_")
    visual_fine = condition.startswith("FC_") or condition.startswith("FF_")
    treatment = min(8, n)
    treatment_queries = treatment if action_fine else ((treatment + 3) // 4)
    continuation = max(0, n - 8)
    calls = treatment_queries + continuation
    cameras = 1  # all formal tasks use one RGB camera in the frozen env
    return {
        "executed_steps": n,
        "action_opportunities": n,
        "policy_forward_calls": calls,
        "policy_forward_rows": calls,
        "global_encoder_calls": calls * cameras,
        "fine_encoder_calls": (treatment_queries if visual_fine else 0) * cameras + continuation * cameras,
        "visual_tokens": (treatment_queries * (2 if visual_fine else 1) + continuation * 2) * cameras * 16,
    }


def clustered_summary(values, clusters, seed, reps=10000):
    grouped = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[tuple(cluster)].append(float(value))
    x = np.asarray([np.mean(v) for v in grouped.values()], dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.mean(x[rng.integers(0, len(x), size=(reps, len(x)))], axis=1)
    observed = abs(float(np.mean(x)))
    flips = rng.choice((-1.0, 1.0), size=(reps, len(x)))
    p = (int(np.sum(np.abs(np.mean(flips * x, axis=1)) >= observed)) + 1) / (reps + 1)
    return {"mean": float(np.mean(x)), "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))], "signflip_p": float(p), "clusters": len(x)}


def holm(pvalues):
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    out, running = {}, 0.0
    for i, (name, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * float(pvalue)))
        out[name] = running
    return out


def utility(row, weights):
    return (weights[0] * row["success_hold5"] + weights[1] * row["normalized_progress"] + weights[2] * row["recoverable"] + weights[3] * row["dropped_or_slipped"] + weights[4] * row["collision"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    files = sorted((args.formal_root / "oracle").glob("*.json"))
    raw, outcome_mismatches, schedule_mismatches, accounting_mismatches = [], [], [], []
    for path in files:
        payload = json.loads(path.read_text())
        for row in payload.get("rows", []):
            raw.append(row)
            recomputed = recompute_outcome(row)
            for field, expected in recomputed.items():
                observed = row.get(field)
                if isinstance(expected, float):
                    equal = abs(float(observed) - expected) <= 1e-9
                else:
                    equal = observed == expected
                if not equal:
                    outcome_mismatches.append({"file": path.name, "episode_seed": row.get("episode_seed"), "condition": row.get("condition"), "field": field, "expected": expected, "observed": observed})
            schedule = expected_schedule(row)
            accounting = row.get("accounting", {})
            for field, expected in schedule.items():
                if int(accounting.get(field, -1)) != int(expected):
                    schedule_mismatches.append({"file": path.name, "episode_seed": row.get("episode_seed"), "condition": row.get("condition"), "field": field, "expected": expected, "observed": accounting.get(field)})
            if set(accounting) < set(ACCOUNTING):
                accounting_mismatches.append({"file": path.name, "missing": sorted(set(ACCOUNTING) - set(accounting))})
            expected_flops = accounting.get("global_encoder_calls", 0) * 1.8e9 + accounting.get("fine_encoder_calls", 0) * 1.8e9 + accounting.get("policy_forward_calls", 0) * 0.7e9
            if abs(float(accounting.get("estimated_flops", 0)) - expected_flops) > 1e-6:
                accounting_mismatches.append({"file": path.name, "field": "estimated_flops", "expected": expected_flops, "observed": accounting.get("estimated_flops")})
            expected_latency = accounting.get("gpu_latency_ms", 0) + accounting.get("simulator_latency_ms", 0) + accounting.get("selector_latency_ms", 0)
            if abs(float(accounting.get("episode_total_compute", 0)) - expected_latency) > 1e-6:
                accounting_mismatches.append({"file": path.name, "field": "episode_total_compute", "expected": expected_latency, "observed": accounting.get("episode_total_compute")})

    # Aggregate repeats independently, then compute matched effects per source episode.
    grouped = defaultdict(list)
    for row in raw:
        key = (row["task"], int(row["model_seed"]), row["bank"], row["bank_id"], row["source_episode"], row["phase"], row["condition"])
        grouped[key].append(row)
    means = {}
    for key, rows in grouped.items():
        means[key] = {"task": key[0], "seed": key[1], "bank": key[2], "bank_id": key[3], "source_episode": key[4], "phase": key[5], "condition": key[6], **{field: float(np.mean([float(r[field]) for r in rows])) for field in ("success_hold5", "normalized_progress", "recoverable", "dropped_or_slipped", "collision")}}
        means[key]["utility"] = {name: float(np.mean([utility(r, w) for r in rows])) for name, w in WEIGHTS.items()}
        means[key]["accounting"] = {field: float(np.mean([r["accounting"][field] for r in rows])) for field in ACCOUNTING}
    effects = defaultdict(list)
    for key in sorted({k[:6] for k in means}):
        cond = {k[6]: v for k, v in means.items() if k[:6] == key}
        for weight in WEIGHTS:
            fc = max((cond[k] for k in cond if k.startswith("FC_tile")), key=lambda r: (r["utility"][weight], r["condition"]))
            ff = max((cond[k] for k in cond if k.startswith("FF_tile")), key=lambda r: (r["utility"][weight], r["condition"]))
            cc, cf = cond["CC"], cond["CF"]
            cluster = (key[0], key[4])
            effects[(weight, key[0], "visual")].append((fc["utility"][weight] - cc["utility"][weight], cluster))
            effects[(weight, key[0], "action")].append((cf["utility"][weight] - cc["utility"][weight], cluster))
            effects[(weight, key[0], "joint")].append((ff["utility"][weight] - max(fc["utility"][weight], cf["utility"][weight]), cluster))
    summaries, pvalues = {}, {}
    for index, (key, pairs) in enumerate(sorted(effects.items())):
        values, clusters = zip(*pairs)
        summary = clustered_summary(values, clusters, 730000 + index, args.bootstrap_replicates)
        name = "/".join(map(str, key)); summaries[name] = summary; pvalues[name] = summary["signflip_p"]
    adjusted = holm(pvalues)

    # Prefix replay latency is persisted but not included in episode_total_compute.
    prefix_total = float(sum(float(r.get("accounting", {}).get("prefix_replay_simulator_latency_ms", 0.0)) for r in raw))
    episode_total = float(sum(float(r.get("accounting", {}).get("episode_total_compute", 0.0)) for r in raw))
    result = {
        "protocol_id": PROTOCOL_ID,
        "independence": {"does_not_import_analyze_stage27r": True, "script_sha256": sha256_file(Path(__file__))},
        "raw_files": [{"path": str(p.relative_to(args.formal_root)), "sha256": sha256_file(p), "bytes": p.stat().st_size} for p in files],
        "raw_row_count": len(raw),
        "outcome_recompute": {"pass": not outcome_mismatches, "mismatches": outcome_mismatches[:100], "mismatch_count": len(outcome_mismatches)},
        "schedule_recompute": {"pass": not schedule_mismatches, "mismatches": schedule_mismatches[:100], "mismatch_count": len(schedule_mismatches), "schedule_definition": "trace length, treatment 8-step query cadence, common fine continuation"},
        "accounting_recompute": {"pass": not accounting_mismatches, "mismatches": accounting_mismatches[:100], "mismatch_count": len(accounting_mismatches), "flop_formula": "global*1.8e9 + fine*1.8e9 + policy*0.7e9"},
        "prefix_latency_disclosure": {"prefix_replay_simulator_latency_ms_sum": prefix_total, "episode_total_compute_ms_sum": episode_total, "prefix_included_in_episode_total": False, "interpretation": "prefix replay cost is persisted separately and omitted from episode_total_compute; all arms share the prefix, but reported total compute is deployment-only treatment/continuation cost"},
        "paired_effects_independent": {"bootstrap_replicates": args.bootstrap_replicates, "summaries": summaries, "holm_adjusted_signflip_p": adjusted, "unit": "source_episode cluster"},
        "status": "PASS" if not outcome_mismatches and not schedule_mismatches and not accounting_mismatches else "FAIL_WITH_DISCLOSED_MISMATCHES",
    }
    atomic_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("raw_row_count", "outcome_recompute", "schedule_recompute", "accounting_recompute", "prefix_latency_disclosure", "status")}, indent=2))


if __name__ == "__main__":
    main()
