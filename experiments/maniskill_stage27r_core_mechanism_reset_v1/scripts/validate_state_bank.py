#!/usr/bin/env python3
"""Validate an existing lockstep state bank before resume skip."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common import PROTOCOL_ID


def validate(path: Path, task: str, bank: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("task") != task or payload.get("bank") != bank:
        raise RuntimeError(f"state-bank protocol/task/bank mismatch: {path}")
    expected = {"calibration": 48, "confirmatory": 96, "negative": 48}[bank]
    rows = payload.get("states")
    if not isinstance(rows, list) or len(rows) != expected or int(payload.get("count", -1)) != expected:
        raise RuntimeError(f"state-bank row count mismatch: {path}")
    if float(payload.get("fidelity_pass_rate", -1.0)) < 0.95:
        raise RuntimeError(f"state-bank fidelity below threshold: {path}")
    ids, source = set(), []
    phases, source_types = Counter(), Counter()
    for row in rows:
        bank_id = str(row.get("bank_id", ""))
        if not bank_id or bank_id in ids:
            raise RuntimeError(f"duplicate/empty state-bank id: {path}")
        ids.add(bank_id)
        phase, source_type = row.get("phase"), row.get("source_type")
        if phase not in {"free_space_approach", "pre_contact_or_pre_grasp", "object_in_hand_pre_placement", "contact_placement_near_completion"}:
            raise RuntimeError(f"invalid state-bank phase: {phase}")
        if source_type not in {"expert", "on_policy"}:
            raise RuntimeError(f"invalid state-bank source type: {source_type}")
        fidelity = row.get("fidelity")
        if not isinstance(fidelity, dict) or fidelity.get("pass") is not True or fidelity.get("categorical_agreement") is not True:
            raise RuntimeError(f"state-bank fidelity flag failed: {bank_id}")
        if row.get("branch_success") is True or fidelity.get("branch_success") is True:
            raise RuntimeError(f"post-success state admitted: {bank_id}")
        phases[phase] += 1
        source_types[source_type] += 1
        source.append(str(row.get("source_episode")))
    expected_phase = expected // 4
    if set(phases.values()) != {expected_phase} or source_types["expert"] != expected // 2 or source_types["on_policy"] != expected // 2:
        raise RuntimeError(f"state-bank phase/source balance mismatch: {path}")
    if len(source) != len(set(source)):
        raise RuntimeError(f"source episode reused in state-bank: {path}")
    return {"status": "PASS", "task": task, "bank": bank, "count": expected, "fidelity_pass_rate": float(payload["fidelity_pass_rate"])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--bank", choices=("calibration", "confirmatory", "negative"), required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.path, args.task, args.bank), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
