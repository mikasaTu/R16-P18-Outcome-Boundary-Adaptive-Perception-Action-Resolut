#!/usr/bin/env python3
"""Reverse-analyze observed stopping gains/losses without proposing a new idea."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import MODEL_SEEDS, PROTOCOL_ID, read_jsonl, write_json_new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    strata: dict[str, Counter] = defaultdict(Counter)
    for seed in MODEL_SEEDS:
        for bank in ("train_source", "calibration"):
            for row in read_jsonl(args.data_root / f"seed_{seed}" / bank / "capsules.jsonl"):
                key = f"{bank}/{row['capture_type']}/{row['phase']}"
                hold, cont = bool(row["hold_success_20"]), bool(row["continue_success_20"])
                strata[key]["n"] += 1
                strata[key]["hold_success"] += int(hold)
                strata[key]["continue_success"] += int(cont)
                strata[key]["hold_beneficial"] += int(hold and not cont)
                strata[key]["continue_beneficial"] += int(cont and not hold)
    closed = {}
    for seed in MODEL_SEEDS:
        modes = {}
        for mode in ("fixed_horizon", "fixed_time_matched_stop", "random_matched_stop", "learned_success_only_classifier", "learned_counterfactual_completion_gate", "privileged_neutral_after_hold5", "privileged_terminate_first_success"):
            modes[mode] = {int(row["episode_seed"]): row for row in read_jsonl(args.result_root / f"seed_{seed}" / mode / "episodes.jsonl")}
        fixed = modes["fixed_horizon"]
        seed_result = {}
        for mode, rows in modes.items():
            gained = sum(not fixed[e]["success_at_end"] and rows[e]["success_at_end"] for e in fixed)
            lost = sum(fixed[e]["success_at_end"] and not rows[e]["success_at_end"] for e in fixed)
            stops = [row["stop_step"] for row in rows.values() if row["stop_step"] is not None]
            seed_result[mode] = {"net_end_success_gain": (gained - lost) / len(fixed), "rescued_episodes": gained, "harmed_episodes": lost, "stop_rate": len(stops) / len(fixed), "mean_stop_step": float(np.mean(stops)) if stops else None, "success_once": float(np.mean([row["success_once"] for row in rows.values()])), "post_success_loss": float(np.mean([row["post_success_loss"] for row in rows.values()]))}
        learned_stops = {e for e, row in modes["learned_counterfactual_completion_gate"].items() if row["stop_step"] is not None}
        success_stops = {e for e, row in modes["learned_success_only_classifier"].items() if row["stop_step"] is not None}
        seed_result["gate_overlap"] = {"intersection": len(learned_stops & success_stops), "union": len(learned_stops | success_stops), "jaccard": len(learned_stops & success_stops) / len(learned_stops | success_stops) if learned_stops | success_stops else 1.0}
        closed[str(seed)] = seed_result
    write_json_new(args.output, {"protocol_id": PROTOCOL_ID, "status": "MECHANISM_ANALYSIS_COMPLETE", "analysis_only_no_new_idea": True, "counterfactual_label_strata": {key: dict(value) for key, value in sorted(strata.items())}, "closed_loop_decomposition": closed, "interpretation_contract": ["compare counterfactual gate against success-only classifier", "compare against fixed-time and random matched stop controls", "attribute net gain as rescued minus harmed paired episodes", "do not interpret invalid restored branches as causal if shared-prefix gate fails"]})


if __name__ == "__main__":
    main()
