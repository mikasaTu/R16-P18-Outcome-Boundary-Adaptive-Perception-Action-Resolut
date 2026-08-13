#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from build_data_splits import SMOKE_SOURCE, build_subset, load_candidates  # noqa: E402
from protocol_common import PROTOCOL_ID, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--selected-raw-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_h5 = args.official_root / SMOKE_SOURCE
    metadata, candidates = load_candidates("PickCube-v1", source_h5)
    if len(candidates) < args.count:
        raise RuntimeError(f"only {len(candidates)} eligible smoke trajectories")
    selected = candidates[: args.count]
    for index, item in enumerate(selected):
        item["split"] = "smoke"
        item["split_index"] = index
        item["selection_rank"] = index
    output_h5 = args.selected_raw_root / "PickCube-v1/smoke/trajectory.h5"
    output = build_subset(source_h5, metadata, selected, output_h5)
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "scope": "integration_smoke_only",
        "task_id": "PickCube-v1",
        "count": args.count,
        "source_h5": str(source_h5),
        "selected": selected,
        "output": output,
    }
    write_json(args.output_manifest, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
