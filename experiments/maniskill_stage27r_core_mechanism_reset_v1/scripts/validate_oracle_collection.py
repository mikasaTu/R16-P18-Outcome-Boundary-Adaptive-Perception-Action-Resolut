#!/usr/bin/env python3
"""Validate the complete confirmatory oracle shard collection."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from common import PROTOCOL_ID, canonical_json, sha256_file
from validate_oracle_shard import validate_shard


def exclusive_json(path: Path, payload: dict) -> str:
    data = canonical_json(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.complete-{os.getpid()}-{os.urandom(8).hex()}.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise RuntimeError(f"oracle validation mismatch; refusing overwrite: {path}")
            return "validated_existing"
        return "installed_missing"
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def validate_collection(
    formal_root: Path, state_bank_dir: Path, expected_tasks: set[str]
) -> dict:
    all_files = sorted((formal_root / "oracle").glob("*.json"))
    # The confirmatory collection is the six preregistered task×seed shards.
    # A diagnostic negative shard, if explicitly requested by a future run,
    # must not silently become a seventh gate arm; it is disclosed separately
    # and remains outside this gate-qualified collection.
    files = []
    diagnostic_files = []
    for path in all_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows") or []
        bank = str(rows[0].get("bank", "")) if rows else ""
        if bank == "confirmatory":
            files.append(path)
        else:
            diagnostic_files.append({"path": str(path), "bank": bank})
    if len(files) != 6:
        raise RuntimeError(
            f"expected six confirmatory oracle shards, got {len(files)}; "
            f"diagnostic={diagnostic_files}"
        )
    records = []
    seen_pairs = set()
    row_count = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows") or []
        if not rows:
            raise RuntimeError(f"empty oracle shard: {path}")
        first = rows[0]
        task = str(first.get("task"))
        seed = int(first.get("model_seed", -1))
        bank = str(first.get("bank"))
        if bank != "confirmatory":
            raise RuntimeError(f"non-confirmatory shard in collection: {path}")
        pair = (task, seed)
        if pair in seen_pairs:
            raise RuntimeError(f"duplicate task/model shard: {pair}")
        seen_pairs.add(pair)
        result = validate_shard(
            path,
            task=task,
            model_seed=seed,
            bank=bank,
            grid=int(payload.get("tile_grid", -1)),
            state_bank=state_bank_dir / f"{task}-{bank}.json",
        )
        if int(result["row_count"]) != 16320:
            raise RuntimeError(f"confirmatory shard does not contain 16320 rows: {path}")
        records.append({**result, "sha256": sha256_file(path), "bytes": path.stat().st_size})
        row_count += int(result["row_count"])
    tasks = {task for task, _ in seen_pairs}
    if expected_tasks and tasks != expected_tasks:
        raise RuntimeError(f"oracle task set mismatch: {tasks} != {expected_tasks}")
    model_seeds = {seed for _, seed in seen_pairs}
    if len(tasks) != 2 or len(seen_pairs) != 6 or len(model_seeds) != 3:
        raise RuntimeError(f"oracle collection must be 2 tasks x 3 seeds: {seen_pairs}")
    if row_count != 97920:
        raise RuntimeError(f"oracle collection row count mismatch: {row_count}")
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "validated_shards": len(records),
        "tasks": sorted(tasks),
        "model_seeds": sorted(model_seeds),
        "row_count": row_count,
        "shards": sorted(records, key=lambda row: (row["task"], row["model_seed"])),
        "diagnostic_oracle_files_excluded": diagnostic_files,
        "lineage_note": "legacy rows are validated by visible fields; checkpoint/state-bank lineage is carried in ORACLE_LINEAGE_MANIFEST.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--state-bank-dir", type=Path, required=True)
    parser.add_argument("--expected-task", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = validate_collection(
        args.formal_root, args.state_bank_dir, set(args.expected_task)
    )
    print(json.dumps({"status": exclusive_json(args.output, payload), **payload}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
