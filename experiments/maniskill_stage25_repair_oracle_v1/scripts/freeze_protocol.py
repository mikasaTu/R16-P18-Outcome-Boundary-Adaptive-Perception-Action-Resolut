#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import PROTOCOL_ID, sha256_file, write_json

SCIENTIFIC_FILES = (
    "preregistration.yaml",
    "manifests/source_bindings.json",
    "manifests/checkpoint_screen_seed_bank.json",
    "manifests/checkpoint_final_val_seed_bank.json",
    "manifests/confirmatory_test_seed_bank.json",
    "manifests/oracle_source_seed_bank.json",
    "manifests/seed_bank_audit.json",
    "manifests/checkpoint_candidates.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.experiment_root.resolve()
    source_path = root / "manifests/source_bindings.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") != "STAGE2_SOURCE_AUDIT_PASS":
        raise RuntimeError("source audit did not pass")
    if source.get("checkpoint_candidate_count") != 156:
        raise RuntimeError("checkpoint inventory is incomplete")
    if source.get("checkpoint_payloads_verified_now") is not True:
        raise RuntimeError("all checkpoint payloads must be verified before freeze")
    candidates = {
        "protocol_id": PROTOCOL_ID,
        "predecessor_checkpoint_root": str(
            Path(source["checkpoint_candidates"][0]["checkpoint_path"]).parents[4]
        ),
        "candidate_count": len(source["checkpoint_candidates"]),
        "payloads_verified_before_freeze": True,
        "candidates": source["checkpoint_candidates"],
    }
    candidate_path = root / "manifests/checkpoint_candidates.json"
    if candidate_path.exists():
        raise FileExistsError(f"refusing to replace frozen candidate manifest: {candidate_path}")
    write_json(candidate_path, candidates)

    missing = [relative for relative in SCIENTIFIC_FILES if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    bindings = {relative: sha256_file(root / relative) for relative in SCIENTIFIC_FILES}
    frozen = {
        "protocol_id": PROTOCOL_ID,
        "status": "PROTOCOL_FROZEN_BEFORE_CONFIRMATORY_RESULTS",
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "confirmatory_results_observed_before_freeze": False,
        "current_user_override": {
            "run_all_planned_experiments": True,
            "gate_failures_constrain_claims_but_do_not_stop_execution": True,
            "a800_gpu_count_inclusive": [2, 8],
        },
        "threshold_mutation_after_freeze_allowed": False,
        "scientific_file_sha256": bindings,
        "checkpoint_payload_count_verified": 156,
        "predecessor_path_mutation_allowed": False,
    }
    freeze_path = root / "PROTOCOL_FREEZE.json"
    if freeze_path.exists():
        raise FileExistsError(f"refusing to replace existing protocol freeze: {freeze_path}")
    write_json(freeze_path, frozen)
    lines = [f"{digest}  {relative}" for relative, digest in sorted(bindings.items())]
    lines.append(f"{sha256_file(freeze_path)}  PROTOCOL_FREEZE.json")
    sums_path = root / "manifests/SCIENTIFIC_SHA256SUMS"
    if sums_path.exists():
        raise FileExistsError(f"refusing to replace existing SHA manifest: {sums_path}")
    from common import atomic_write_bytes

    atomic_write_bytes(sums_path, ("\n".join(lines) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()

