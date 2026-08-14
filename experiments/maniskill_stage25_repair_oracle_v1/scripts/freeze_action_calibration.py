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
from oracle_math import EFFECT_THRESHOLD_CANDIDATES, RADIUS_CANDIDATES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace calibration freeze: {args.output}")
    rows = []
    source_hashes = {}
    for seed in MODEL_SEEDS:
        path = args.calibration_root / f"seed_{seed}" / "states.jsonl"
        source_hashes[str(path)] = sha256_file(path)
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    by_radius = {}
    for radius in RADIUS_CANDIDATES:
        selected = [row for row in rows if float(row["atlas"]["radius"]) == radius]
        if len(selected) != 32 * len(MODEL_SEEDS):
            raise RuntimeError(f"incomplete radius calibration {radius}: {len(selected)}")
        validity = float(np.mean([np.mean(row["atlas"]["valid"]) for row in selected]))
        density = {
            str(threshold): float(
                np.mean(
                    [
                        row["atlas"]["boundary_by_threshold"][str(threshold)][
                            "boundary_density"
                        ]
                        for row in selected
                    ]
                )
            )
            for threshold in EFFECT_THRESHOLD_CANDIDATES
        }
        by_radius[str(radius)] = {
            "state_model_rows": len(selected),
            "candidate_validity": validity,
            "boundary_density_by_effect_threshold": density,
        }
    eligible = [
        radius
        for radius in RADIUS_CANDIDATES
        if by_radius[str(radius)]["candidate_validity"] >= 0.90
    ]
    if not eligible:
        # The preregistered failure remains visible; the smallest radius is
        # frozen only to permit the user-mandated downstream diagnostics.
        selected_radius = min(RADIUS_CANDIDATES)
        radius_gate_pass = False
    else:
        selected_radius = max(
            eligible,
            key=lambda radius: (
                by_radius[str(radius)]["boundary_density_by_effect_threshold"]["1.0"],
                -radius,
            ),
        )
        radius_gate_pass = True
    selected_threshold = min(
        EFFECT_THRESHOLD_CANDIDATES,
        key=lambda threshold: (
            abs(
                by_radius[str(selected_radius)]["boundary_density_by_effect_threshold"][
                    str(threshold)
                ]
                - 0.30
            ),
            threshold,
        ),
    )
    write_json(
        args.output,
        {
            "protocol_id": PROTOCOL_ID,
            "status": "ACTION_CALIBRATION_FROZEN",
            "confirmatory_results_observed_before_freeze": False,
            "selected_radius": selected_radius,
            "selected_effect_threshold": selected_threshold,
            "radius_validity_gate_pass": radius_gate_pass,
            "selection_rules": {
                "radius": "validity>=0.90_then_max_boundary_density_at_threshold_1.0_then_smallest",
                "effect_threshold": "closest_boundary_density_to_0.30_then_smallest",
            },
            "calibration": by_radius,
            "source_sha256": source_hashes,
        },
    )


if __name__ == "__main__":
    main()
