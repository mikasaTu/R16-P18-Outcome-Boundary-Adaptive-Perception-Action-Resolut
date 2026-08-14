#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "R16-P18-MS5-STAGE26-COUNTERFACTUAL-COMPLETION-V1"
COUNTS = {"train_source": 512, "calibration": 128, "confirmatory": 200}


def candidate(bank: str, index: int, nonce: int) -> int:
    raw = hashlib.sha256(f"{PROTOCOL}|{bank}|{index}|{nonce}".encode()).digest()
    return int.from_bytes(raw[:8], "big") & 0x7FFFFFFF


def inherited_exclusions() -> set[int]:
    result: set[int] = set()
    predecessor = ROOT.parent / "maniskill_act_boundary_screen_v1" / "manifests"
    stage25 = ROOT.parent / "maniskill_stage25_repair_oracle_v1" / "manifests"
    for base in (predecessor, stage25):
        if not base.exists():
            continue
        for path in sorted(base.glob("*seed*.json")):
            payload = json.loads(path.read_text())
            stack = payload.get("tasks", {}).get("StackCube-v1", payload)
            todo = [stack]
            while todo:
                value = todo.pop()
                if isinstance(value, dict):
                    todo.extend(value.values())
                elif isinstance(value, list):
                    todo.extend(value)
                elif isinstance(value, int):
                    result.add(value)
    return result


def main() -> None:
    used = inherited_exclusions()
    banks = {}
    for bank, count in COUNTS.items():
        values = []
        for index in range(count):
            nonce = 0
            while True:
                value = candidate(bank, index, nonce)
                nonce += 1
                if value not in used:
                    break
            used.add(value)
            values.append(value)
        banks[bank] = values
    payload = {"protocol_id": PROTOCOL, "algorithm": "sha256_low31_disjoint_v1", "same_order_across_model_seeds": True, "banks": banks}
    target = ROOT / "manifests" / "seed_banks.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
