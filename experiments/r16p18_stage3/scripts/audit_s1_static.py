#!/usr/bin/env python3
"""Static safety audit for the Stage-3 S1 zero-rollout scripts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


PROTOCOL_ID = "R16-P18-STAGE3-S1-BUDGET-FEASIBILITY-V1"
SCANNED = (
    "prepare_s1_observation.py",
    "profile_s1_costs.py",
    "compute_feasibility.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _forbidden_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    # Build the simulator call token in pieces so this audit source itself is
    # not mistaken for a script that performs that operation.
    simulator_call = "env" + r"\.step\s*\("
    return (
        ("simulator_step", re.compile(simulator_call)),
        ("simulator_reset", re.compile(r"\.reset\s*\(")),
        ("environment_factory", re.compile(r"gym(?:nasium)?\.make|mani_skill")),
        ("pai_command", re.compile(r"pai[-_ ]job|dlc\s+(?:submit|create|run)")),
        ("legacy_stage_write", re.compile(r"maniskill_stage2[567]|stage27r_core_mechanism")),
    )


def audit(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    scripts_dir = root / "scripts"
    if output.exists():
        raise FileExistsError(f"fail-on-overwrite: {output}")
    findings: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for name in SCANNED:
        path = scripts_dir / name
        if not path.is_file():
            findings.append({"file": name, "kind": "missing_script"})
            continue
        text = path.read_text(encoding="utf-8")
        files.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
        for kind, pattern in _forbidden_patterns():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append({"file": name, "kind": kind, "line": line, "snippet": text.splitlines()[line - 1][:160]})
    # The static guard also checks the formula implementation, not merely
    # source absence of simulator calls.
    feasibility = scripts_dir / "compute_feasibility.py"
    if feasibility.is_file():
        text = feasibility.read_text(encoding="utf-8")
        for needle in ("alpha - rho", "alpha-rho", "1.0 - rho", "rho >= 1.0"):
            if needle not in text:
                findings.append({"file": feasibility.name, "kind": "formula_marker_missing", "marker": needle})
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if not findings else "FAIL",
        "scanned_scripts": list(SCANNED),
        "files": files,
        "forbidden_findings": findings,
        "claims": {
            "no_simulator_operations": not any(item["kind"].startswith("simulator") or item["kind"] == "environment_factory" for item in findings),
            "no_pai_commands": not any(item["kind"] == "pai_command" for item in findings),
            "no_legacy_stage_write_reference": not any(item["kind"] == "legacy_stage_write" for item in findings),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    if findings:
        raise RuntimeError(f"S1 static audit failed with {len(findings)} finding(s)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output)


if __name__ == "__main__":
    main()
