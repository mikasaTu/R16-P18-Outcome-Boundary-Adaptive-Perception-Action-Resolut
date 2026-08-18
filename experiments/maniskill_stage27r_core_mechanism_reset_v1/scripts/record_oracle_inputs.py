#!/usr/bin/env python3
"""Snapshot oracle files before a continuation starts producing new shards.

This sidecar separates legacy shards already present in the formal root from
shards created by the clean continuation source.  It never edits an oracle
file and installs its own evidence atomically with same-filesystem link
semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from common import PROTOCOL_ID, canonical_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive_json(path: Path, value: dict) -> str:
    payload = canonical_json(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.complete-{os.getpid()}-{os.urandom(8).hex()}.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise RuntimeError(f"oracle input snapshot mismatch; refusing overwrite: {path}")
            return "validated_existing"
        return "installed_missing"
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def snapshot(formal_root: Path, tasks: list[str], seeds: list[int]) -> dict:
    entries = []
    for task in tasks:
        for seed in seeds:
            path = Path(formal_root) / "oracle" / f"{task}-seed{seed}-confirmatory.json"
            if path.is_file():
                entries.append({
                    "task": task,
                    "model_seed": int(seed),
                    "path": str(path),
                    "preexisting": True,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                })
            else:
                entries.append({
                    "task": task,
                    "model_seed": int(seed),
                    "path": str(path),
                    "preexisting": False,
                    "sha256": None,
                    "bytes": None,
                })
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "scope": "pre_oracle_continuation_input_snapshot",
        "oracle_files": entries,
        "legacy_rows_have_no_intrinsic_checkpoint_hash": True,
    }


def validate_existing(path: Path, formal_root: Path, tasks: list[str], seeds: list[int]) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol_id") != PROTOCOL_ID or value.get("status") != "PASS" or value.get("scope") != "pre_oracle_continuation_input_snapshot" or value.get("legacy_rows_have_no_intrinsic_checkpoint_hash") is not True:
        raise RuntimeError(f"invalid oracle input snapshot: {path}")
    expected = {(task, int(seed)) for task in tasks for seed in seeds}
    rows = value.get("oracle_files", [])
    observed = {(str(row.get("task")), int(row.get("model_seed", -1))) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise RuntimeError(f"oracle input snapshot task/seed set mismatch: {path}")
    for row in rows:
        oracle = Path(row["path"])
        if bool(row.get("preexisting")):
            if not oracle.is_file() or row.get("sha256") != sha256_file(oracle) or int(row.get("bytes", -1)) != oracle.stat().st_size:
                raise RuntimeError(f"preexisting oracle changed after input snapshot: {oracle}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-task", action="append", required=True)
    parser.add_argument("--model-seed", type=int, action="append", required=True)
    args = parser.parse_args()
    if args.output.is_file():
        value = validate_existing(args.output, args.formal_root, args.expected_task, args.model_seed)
        status = "validated_existing"
    else:
        value = snapshot(args.formal_root, args.expected_task, args.model_seed)
        status = exclusive_json(args.output, value)
    print(json.dumps({"status": status, **value}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
