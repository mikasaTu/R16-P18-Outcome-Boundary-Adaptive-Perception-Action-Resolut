#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from common import PROTOCOL_ID, sha256_file, write_json

REQUIRED_OLD_FILES = (
    "docs/MANISKILL_STAGE2_FINAL_REPORT.md",
    "experiments/maniskill_act_boundary_screen_v1/preregistration.yaml",
    "experiments/maniskill_act_boundary_screen_v1/README.md",
    "experiments/maniskill_act_boundary_screen_v1/task_selection.json",
    "experiments/maniskill_act_boundary_screen_v1/scripts/train_official_act_protocol.py",
    "experiments/maniskill_act_boundary_screen_v1/scripts/evaluate_official_act_protocol.py",
    "experiments/maniskill_act_boundary_screen_v1/scripts/summarize_baseline.py",
    "experiments/maniskill_act_boundary_screen_v1/scripts/audit_formal_baseline.py",
    "experiments/maniskill_act_boundary_screen_v1/baseline/baseline_failure_mechanism_analysis_20260814.json",
    "experiments/maniskill_act_boundary_screen_v1/baseline/baseline_gate_20260814.json",
)
OLD_ORACLE_FILES = (
    "experiments/maniskill_act_boundary_screen_v1/scripts/build_state_bank.py",
    "experiments/maniskill_act_boundary_screen_v1/scripts/evaluate_oracle_atlas.py",
    "experiments/maniskill_act_boundary_screen_v1/scripts/oracle_common.py",
    "experiments/maniskill_act_boundary_screen_v1/scripts/oracle_runtime.py",
    "experiments/maniskill_act_boundary_screen_v1/scripts/oracle_visual.py",
    "experiments/maniskill_act_boundary_screen_v1/scripts/run_oracle_matrix.py",
    "experiments/maniskill_act_boundary_screen_v1/scripts/summarize_oracle_gate.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--old-data-manifest", type=Path, required=True)
    parser.add_argument("--old-seed-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-checkpoint-payloads", action="store_true")
    return parser.parse_args()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def candidate_rows(root: Path, verify: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for selection_path in sorted(root.glob("*-v1/seed_*/checkpoint_selection.json")):
        value = json.loads(selection_path.read_text(encoding="utf-8"))
        task_id = selection_path.parent.parent.name
        model_seed = int(selection_path.parent.name.removeprefix("seed_"))
        candidates = value.get("candidates")
        if not isinstance(candidates, list):
            raise RuntimeError(f"invalid candidates: {selection_path}")
        for candidate in candidates:
            checkpoint = Path(candidate["path"]) / "checkpoint.pt"
            expected = candidate["checkpoint_sha256"]
            actual = sha256_file(checkpoint) if verify else None
            if verify and actual != expected:
                raise RuntimeError(f"checkpoint digest mismatch: {checkpoint}")
            rows.append(
                {
                    "task_id": task_id,
                    "model_seed": model_seed,
                    "step": int(candidate["step"]),
                    "validation_loss": float(candidate["validation_loss"]),
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": expected,
                    "payload_digest_verified_now": bool(verify),
                    "complete_marker_path": str(Path(candidate["path"]) / "COMPLETE.json"),
                }
            )
    if len(rows) != 156:
        raise RuntimeError(f"expected 156 candidate checkpoints, got {len(rows)}")
    return rows


def main() -> None:
    args = parse_args()
    old_bindings = {}
    for relative in (*REQUIRED_OLD_FILES, *OLD_ORACLE_FILES):
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        old_bindings[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    rows = candidate_rows(args.checkpoint_root, args.verify_checkpoint_payloads)
    old_seed_summary = json.loads(args.old_seed_summary.read_text(encoding="utf-8"))
    old_test = {
        task: value["closed_loop_test_seeds"]
        for task, value in old_seed_summary["formal_tasks"].items()
    }
    demo_counts: dict[str, int] = {}
    with args.old_data_manifest.open(encoding="utf-8") as handle:
        for line in handle:
            task = json.loads(line)["task_id"]
            demo_counts[task] = demo_counts.get(task, 0) + 1
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "STAGE2_SOURCE_AUDIT_PASS",
        "current_head": git("rev-parse", "HEAD"),
        "current_tree": git("rev-parse", "HEAD^{tree}"),
        "audited_predecessor_commit": "76e71f5eae9771b83906478f0c421183e38cdd9c",
        "audited_predecessor_tree": "088b74883a53f9577aacd742f4c9ac560704dad9",
        "predecessor_files": old_bindings,
        "checkpoint_candidates": rows,
        "checkpoint_candidate_count": len(rows),
        "checkpoint_payloads_verified_now": bool(args.verify_checkpoint_payloads),
        "old_closed_loop_test_seeds": old_test,
        "old_demo_identity_binding": {
            "path": str(args.old_data_manifest),
            "sha256": sha256_file(args.old_data_manifest),
            "counts": demo_counts,
        },
        "preexisting_downstream_code": {
            "exists": True,
            "formal_oracle_evidence_exists": False,
            "predecessor_continue_to_oracle_probe": False,
            "execution_policy": "hash_for_audit_only_never_import_or_execute",
            "files": list(OLD_ORACLE_FILES),
        },
        "semantic_differences": [
            "new closed-loop checkpoint selection uses success_hold5/end/post-loss instead of validation loss",
            "new success evaluator records full streak semantics and four stopping diagnostics",
            "contact collision alias is removed and onset/duration/force channels are explicit",
            "state sources are independent unused expert episodes plus new on-policy seeds",
            "candidate actions outside action bounds are invalid rather than clipped",
            "candidate rollouts add 20 base-policy and 5 neutral-hold steps after the four-step prefix",
            "visual primary arm restores native tiles on a low-resolution background instead of destructive masking",
            "joint oracle uses preregistered CC/FC/CF/FF arms and matched state allocation",
            "current user instruction runs every downstream experiment while preserving failed-gate claim restrictions",
        ],
    }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
