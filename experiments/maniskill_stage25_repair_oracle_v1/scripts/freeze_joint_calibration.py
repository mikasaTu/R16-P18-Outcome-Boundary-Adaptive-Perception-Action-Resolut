#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import MODEL_SEEDS, PROTOCOL_ID, sha256_file, write_json
from oracle_math import J_THRESHOLD_CANDIDATES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-calibration-root", type=Path, required=True)
    parser.add_argument("--action-calibration-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace joint calibration freeze: {args.output}")
    action = json.loads(args.action_calibration_freeze.read_text(encoding="utf-8"))
    if action.get("status") != "ACTION_CALIBRATION_FROZEN":
        raise RuntimeError("action calibration freeze is invalid")
    rows = []
    hashes = {}
    for seed in MODEL_SEEDS:
        path = args.visual_calibration_root / f"seed_{seed}" / "states.jsonl"
        hashes[str(path)] = sha256_file(path)
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    if len(rows) != 32 * len(MODEL_SEEDS):
        raise RuntimeError(f"incomplete visual calibration rows: {len(rows)}")
    positive = [float(row["interaction_J"]) for row in rows if row["interaction_J"] > 0]
    if positive:
        quartile = float(np.quantile(positive, 0.25))
        eligible = [value for value in J_THRESHOLD_CANDIDATES if value >= quartile]
        selected = min(eligible) if eligible else max(J_THRESHOLD_CANDIDATES)
    else:
        quartile = None
        selected = 5.0
    write_json(
        args.output,
        {
            "protocol_id": PROTOCOL_ID,
            "status": "ORACLE_CALIBRATION_FROZEN",
            "confirmatory_results_observed_before_freeze": False,
            "selected_radius": action["selected_radius"],
            "selected_effect_threshold": action["selected_effect_threshold"],
            "selected_J_threshold": selected,
            "positive_J_first_quartile": quartile,
            "positive_J_rows": len(positive),
            "J_threshold_rule": "smallest_candidate_not_below_positive_J_first_quartile_else_5.0",
            "action_calibration_freeze_sha256": sha256_file(args.action_calibration_freeze),
            "visual_calibration_source_sha256": hashes,
        },
    )


if __name__ == "__main__":
    main()
