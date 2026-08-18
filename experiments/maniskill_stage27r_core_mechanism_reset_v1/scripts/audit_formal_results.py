#!/usr/bin/env python3
"""Independent recomputation and provenance audit for the formal Stage-2.7R run."""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import tempfile
from pathlib import Path

from common import PROTOCOL_ID, atomic_json, sha256_file

PREDECESSORS = ("experiments/maniskill_act_boundary_screen_v1", "experiments/maniskill_stage25_repair_oracle_v1", "experiments/maniskill_stage26_counterfactual_completion_v1")
WEIGHTS = {"balanced": (100, 20, 5, -10, -5), "success_dominant": (120, 10, 3, -12, -6), "progress_dominant": (80, 35, 5, -10, -5)}
ACCOUNTING = {"global_encoder_calls", "fine_encoder_calls", "policy_forward_calls", "policy_forward_rows", "visual_tokens", "action_opportunities", "executed_steps", "gpu_latency_ms", "simulator_latency_ms", "estimated_flops", "peak_memory_bytes", "selector_latency_ms", "episode_total_compute"}


def frozen_preregistration_digest(repo: Path, experiment: Path) -> dict:
    """Verify preregistration against the frozen Git snapshot, without editing it.

    PROTOCOL_FREEZE.json was intentionally kept minimal and may not contain a
    digest field.  In that case the protocol's predecessor-freeze manifest
    supplies the immutable Git head from which the preregistration bytes are
    derived.  A missing/invalid Git snapshot is a failed check, never a pass.
    """
    protocol = json.loads((experiment / "PROTOCOL_FREEZE.json").read_text())
    current = sha256_file(experiment / "preregistration.yaml")
    expected = protocol.get("preregistration_sha256")
    if expected is not None:
        return {"pass": expected == current, "digest_field_present": True, "expected": expected, "observed": current}
    manifest_path = experiment / "manifests/predecessor_tree_freeze.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        frozen_head = str(manifest["head"])
        prereg_path = str(protocol["preregistration_path"])
        frozen_bytes = subprocess.check_output(["git", "show", f"{frozen_head}:{prereg_path}"], cwd=repo)
        frozen_digest = __import__("hashlib").sha256(frozen_bytes).hexdigest()
        return {
            "pass": frozen_digest == current,
            "digest_field_present": False,
            "expected": frozen_digest,
            "observed": current,
            "derived_from_frozen_git_commit": frozen_head,
            "derived_path": prereg_path,
        }
    except (KeyError, OSError, subprocess.CalledProcessError, ValueError) as exc:
        return {
            "pass": False,
            "digest_field_present": False,
            "expected": None,
            "observed": current,
            "derived_from_frozen_git_commit": None,
            "error": f"cannot derive preregistration from frozen Git snapshot: {exc}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = {}
    experiment = args.repo / "experiments/maniskill_stage27r_core_mechanism_reset_v1"
    freeze = json.loads((experiment / "manifests/predecessor_tree_freeze.json").read_text())["trees"]
    current = {path: subprocess.check_output(["git", "rev-parse", f"HEAD:{path}"], cwd=args.repo, text=True).strip() for path in PREDECESSORS}
    checks["predecessor_immutability"] = {"pass": current == freeze, "frozen": freeze, "current": current}
    checks["clean_source_commit"] = {"pass": not subprocess.check_output(["git", "status", "--porcelain"], cwd=args.repo, text=True).strip(), "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()}
    checks["protocol_freeze"] = frozen_preregistration_digest(args.repo, experiment)

    model_text = (experiment / "scripts/multires_policy.py").read_text()
    tree, obs_keys = ast.parse(model_text), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in {"obs", "data"} and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            obs_keys.add(node.slice.value)
    allowed = {"state", "rgb", "_visual_mode", "_tile_id", "_tile_grid"}
    checks["no_privileged_model_input"] = {"pass": not bool(obs_keys - allowed), "observed_keys": sorted(obs_keys)}

    exact = json.loads((args.formal_root / "EXACT_DATASET_AUDIT.json").read_text())
    checks["split_leakage"] = {"pass": exact["status"] == "PASS" and all(not row["split_leakage"] for row in exact["task_checks"].values()), "task_checks": exact["task_checks"]}
    preflight = json.loads((args.formal_root / "PRECHECKS.json").read_text())
    checks["unit_compile_smoke_overwrite"] = {"pass": bool(preflight.get("all_pass")), "evidence": preflight}
    training_complete = (args.training_root / "DATA_AND_TRAINING_COMPLETE.json").is_file()
    checkpoints = list(args.training_root.glob("training/*/seed_*/checkpoints/step_*/COMPLETE.json"))
    checks["shard_autoresume"] = {"pass": training_complete and len(checkpoints) >= 18 * 6, "training_complete": training_complete, "complete_checkpoint_markers": len(checkpoints)}

    bank_files = sorted((args.formal_root / "state_banks").glob("*.json"))
    pass_rates = [json.loads(path.read_text())["fidelity_pass_rate"] for path in bank_files]
    checks["fresh_reset_prefix_fidelity"] = {"pass": bool(pass_rates) and min(pass_rates) >= .95, "pass_rates": pass_rates}
    raw_files = sorted((args.formal_root / "oracle").glob("*.json"))
    row_count = utility_mismatch = accounting_missing = accounting_mismatch = 0
    for path in raw_files:
        for row in json.loads(path.read_text()).get("rows", []):
            row_count += 1
            accounting_missing += not ACCOUNTING.issubset(row["accounting"])
            accounting = row["accounting"]
            expected_flops = accounting["global_encoder_calls"] * 1.8e9 + accounting["fine_encoder_calls"] * 1.8e9 + accounting["policy_forward_calls"] * 0.7e9
            expected_latency = accounting["gpu_latency_ms"] + accounting["simulator_latency_ms"] + accounting["selector_latency_ms"]
            accounting_mismatch += abs(accounting["estimated_flops"] - expected_flops) > 1e-6
            accounting_mismatch += abs(accounting["episode_total_compute"] - expected_latency) > 1e-6
            for name, weights in WEIGHTS.items():
                expected = weights[0] * row["success_hold5"] + weights[1] * row["normalized_progress"] + weights[2] * row["recoverable"] + weights[3] * row["dropped_or_slipped"] + weights[4] * row["collision"]
                utility_mismatch += abs(expected - row["utilities"][name]) > 1e-9
    checks["raw_outcome_recompute"] = {"pass": row_count > 0 and utility_mismatch == 0, "rows": row_count, "utility_mismatches": utility_mismatch}
    statistics = json.loads((args.formal_root / "statistics.json").read_text())
    budget_mismatches = []
    for family in ("budgets", "negative_control_budgets"):
        for weight, fractions in statistics[family].items():
            for fraction, arms in fractions.items():
                expected_budget = float(fraction) * arms["all_fine"]["cost"]
                for arm, values in arms.items():
                    if not isinstance(values, dict) or "cost" not in values:
                        continue
                    observed = bool(values["cost"] <= expected_budget + 1e-6)
                    if abs(values["budget"] - expected_budget) > 1e-6 or values["budget_compliant"] != observed:
                        budget_mismatches.append(f"{family}/{weight}/{fraction}/{arm}")
    checks["compute_accounting_recompute"] = {
        "pass": row_count > 0 and accounting_missing == 0 and accounting_mismatch == 0 and not budget_mismatches,
        "missing_rows": accounting_missing,
        "formula_mismatches": accounting_mismatch,
        "budget_mismatches": budget_mismatches,
    }

    with tempfile.TemporaryDirectory(prefix="stage27r-audit-") as directory:
        recomputed = Path(directory) / "statistics.json"
        command = [str(Path(__import__("sys").executable)), str(experiment / "scripts/analyze_stage27r.py"), "--inputs", *map(str, raw_files), "--output", str(recomputed)]
        subprocess.run(command, check=True, cwd=experiment / "scripts")
        expected_stats = args.formal_root / "statistics.json"
        checks["paired_statistics_recompute"] = {"pass": sha256_file(recomputed) == sha256_file(expected_stats), "recomputed_sha256": sha256_file(recomputed), "reported_sha256": sha256_file(expected_stats)}

    # Keep the manifest stable when a resume-safe audit is recomputed to a
    # sibling temporary path after the canonical audit file already exists.
    # The canonical audit outputs are metadata about the root, not scientific
    # inputs, and the original run excluded its own output before writing it.
    audit_outputs = {
        args.formal_root / "INDEPENDENT_AUDIT.json",
        args.formal_root / "POSTHOC_INDEPENDENT_AUDIT.json",
        args.output,
    }
    files = sorted(path for path in args.formal_root.rglob("*") if path.is_file() and path not in audit_outputs)
    manifest = [{"path": str(path.relative_to(args.formal_root)), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in files]
    checks["scientific_sha256_manifest"] = {"pass": bool(manifest), "files": len(manifest)}
    checks["all_pass"] = all(value["pass"] for value in checks.values() if isinstance(value, dict) and "pass" in value)
    atomic_json(args.output, {"protocol_id": PROTOCOL_ID, "checks": checks, "manifest": manifest})
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
