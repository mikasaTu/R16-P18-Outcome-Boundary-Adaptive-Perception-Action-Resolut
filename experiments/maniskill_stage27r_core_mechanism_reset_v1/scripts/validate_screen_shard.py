#!/usr/bin/env python3
"""Validate an existing task/seed screen shard before resume skip."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import PROTOCOL_ID


def validate(path: Path, task: str, seed: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError(f"screen protocol mismatch: {path}")
    if payload.get("task") != task or int(payload.get("seed", -1)) != int(seed):
        raise RuntimeError(f"screen task/seed mismatch: {path}")
    screened = payload.get("screened")
    top2 = payload.get("validated_top2")
    selected = payload.get("selected")
    # The frozen screen contract requires the selected top-2 comparison.  A
    # completed shard may contain exactly two candidates (the formal screen
    # used that minimum), so do not invent a third candidate requirement here.
    if not isinstance(screened, list) or len(screened) < 2:
        raise RuntimeError(f"screen candidate rows incomplete: {path}")
    if not isinstance(top2, list) or len(top2) < 2:
        raise RuntimeError(f"screen validation rows incomplete: {path}")
    if not isinstance(selected, dict):
        raise RuntimeError(f"screen selected row missing: {path}")
    for row in screened:
        for field in ("path", "sha256", "step", "screen"):
            if field not in row:
                raise RuntimeError(f"screen candidate field missing: {field}: {path}")
        if not set(row["screen"]) >= {"success_hold5", "success_at_end", "post_success_loss"}:
            raise RuntimeError(f"screen metrics incomplete: {path}")
    for row in top2:
        if not set(row) >= {"path", "sha256", "step", "validation", "screen"}:
            raise RuntimeError(f"screen top2 fields incomplete: {path}")
        if set(row["validation"]) != {"CC", "FF"}:
            raise RuntimeError(f"screen validation modes incomplete: {path}")
    if not set(selected) >= {"path", "sha256", "step", "validation", "screen"}:
        raise RuntimeError(f"screen selected fields incomplete: {path}")
    return {"status": "PASS", "task": task, "seed": int(seed), "screened": len(screened), "validated_top2": len(top2)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.path, args.task, args.seed), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
