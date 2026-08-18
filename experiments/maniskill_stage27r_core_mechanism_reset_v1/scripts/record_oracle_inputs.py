#!/usr/bin/env python3
"""Freeze legacy oracle inputs and attest continuation-created shards.

``ORACLE_INPUT_SNAPSHOT.json`` is a one-time, immutable six-shard snapshot.
It intentionally records both present and absent inputs.  A shard that was
absent at that instant is never silently reclassified as legacy on restart:
once it appears, a separate immutable record under ``ORACLE_INPUT_RECORDS``
binds its path, hash, size, and semantic validation result.  The initial
snapshot itself is never rewritten, so changing an old shard or a record is a
hard failure.
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
from pathlib import Path
from typing import Any

from common import PROTOCOL_ID, canonical_json, sha256_file
from validate_continuation_evidence import validate_continuation_registry, validate_old_producer_terminal


SNAPSHOT_SCOPE = "pre_oracle_continuation_input_snapshot"
RECORD_DIR_NAME = "ORACLE_INPUT_RECORDS"


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_timestamp(value: Any, label: str) -> _datetime.datetime:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{label} must be a timezone-aware UTC timestamp")
    text = value.strip()
    try:
        parsed = _datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"{label} is not ISO-8601: {text}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != _datetime.timedelta(0):
        raise RuntimeError(f"{label} must be UTC: {text}")
    return parsed.astimezone(_datetime.timezone.utc)


def _aware_timestamp(value: Any, label: str) -> _datetime.datetime:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{label} must be a timezone-aware ISO-8601 timestamp")
    text = value.strip()
    try:
        parsed = _datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"{label} is not ISO-8601: {text}") from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"{label} must include a timezone: {text}")
    return parsed.astimezone(_datetime.timezone.utc)


def _snapshot_terminal_binding(old_terminal: dict[str, Any] | None) -> dict[str, Any] | None:
    if old_terminal is None:
        return None
    evidence = old_terminal.get("evidence", {})
    evidence_hash = evidence.get("sha256")
    if not isinstance(evidence_hash, str) or not evidence_hash:
        raise RuntimeError("validated old producer terminal lacks evidence hash")
    return {
        "job_id": old_terminal.get("job_id"),
        "run_id": old_terminal.get("run_id"),
        "observed_at": old_terminal.get("observed_at"),
        "evidence_sha256": evidence_hash,
    }


def _validate_snapshot_terminal(value: dict[str, Any], old_terminal: dict[str, Any] | None) -> None:
    created_at = _utc_timestamp(value.get("created_at"), "oracle snapshot created_at")
    binding = value.get("old_producer_terminal")
    if binding is None:
        if old_terminal is not None:
            raise RuntimeError("oracle snapshot is missing old producer terminal binding")
        return
    if not isinstance(binding, dict):
        raise RuntimeError("oracle snapshot old producer terminal binding is invalid")
    observed_at = _aware_timestamp(binding.get("observed_at"), "oracle snapshot terminal observed_at")
    if observed_at > created_at:
        raise RuntimeError("old producer terminal observed_at is after oracle snapshot created_at")
    if old_terminal is not None:
        expected = _snapshot_terminal_binding(old_terminal)
        if binding != expected:
            raise RuntimeError("oracle snapshot old producer terminal binding/hash mismatch")


validate_snapshot_metadata = _validate_snapshot_terminal


def exclusive_json(path: Path, value: dict[str, Any]) -> str:
    """Install canonical evidence once, with no-replace race semantics."""
    path = Path(path)
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
                raise RuntimeError(f"immutable oracle evidence mismatch; refusing overwrite: {path}")
            return "validated_existing"
        return "installed_missing"
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _oracle_path(formal_root: Path, task: str, seed: int) -> Path:
    return Path(formal_root) / "oracle" / f"{task}-seed{int(seed)}-confirmatory.json"


def _record_path(formal_root: Path, task: str, seed: int) -> Path:
    return Path(formal_root) / RECORD_DIR_NAME / f"{task}-seed{int(seed)}-confirmatory.json"


def snapshot(
    formal_root: Path,
    tasks: list[str],
    seeds: list[int],
    *,
    old_terminal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not tasks or not seeds or len(set(tasks)) != len(tasks) or len(set(seeds)) != len(seeds):
        raise RuntimeError("oracle input snapshot task/seed arguments must be unique and non-empty")
    entries = []
    for task in tasks:
        for seed in seeds:
            path = _oracle_path(formal_root, task, seed)
            if path.is_symlink():
                raise RuntimeError(f"oracle shard must not be a symlink: {path}")
            if path.is_file():
                entries.append(
                    {
                        "task": task,
                        "model_seed": int(seed),
                        "path": str(path),
                        "preexisting": True,
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                )
            else:
                entries.append(
                    {
                        "task": task,
                        "model_seed": int(seed),
                        "path": str(path),
                        "preexisting": False,
                        "sha256": None,
                        "bytes": None,
                    }
                )
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "schema_version": 2,
        "scope": SNAPSHOT_SCOPE,
        "created_at": _utc_now(),
        "old_producer_terminal": _snapshot_terminal_binding(old_terminal),
        "oracle_files": entries,
        "continuation_record_dir": str(Path(formal_root) / RECORD_DIR_NAME),
        "legacy_rows_have_no_intrinsic_checkpoint_hash": True,
        "continuation_rows_require_immutable_semantic_record": True,
    }


def _load_snapshot(
    path: Path,
    formal_root: Path,
    tasks: list[str],
    seeds: list[int],
    *,
    old_terminal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"invalid oracle input snapshot: {path}: {exc}") from exc
    if (
        value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "PASS"
        or value.get("scope") != SNAPSHOT_SCOPE
        or value.get("legacy_rows_have_no_intrinsic_checkpoint_hash") is not True
        or value.get("continuation_rows_require_immutable_semantic_record") is not True
    ):
        raise RuntimeError(f"invalid oracle input snapshot: {path}")
    _validate_snapshot_terminal(value, old_terminal)
    expected = {(str(task), int(seed)) for task in tasks for seed in seeds}
    rows = value.get("oracle_files")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise RuntimeError(f"oracle input snapshot task/seed set mismatch: {path}")
    observed: set[tuple[str, int]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError(f"oracle input snapshot contains a non-object row: {path}")
        key = (str(row.get("task", "")), int(row.get("model_seed", -1)))
        if key in observed or key not in expected:
            raise RuntimeError(f"oracle input snapshot task/seed set mismatch: {path}")
        observed.add(key)
        oracle = _oracle_path(formal_root, key[0], key[1])
        if Path(row.get("path", "")) != oracle:
            raise RuntimeError(f"oracle input snapshot path drift: {row.get('path')} != {oracle}")
        preexisting = row.get("preexisting")
        if not isinstance(preexisting, bool):
            raise RuntimeError(f"oracle input snapshot preexisting flag is invalid: {oracle}")
        if preexisting:
            recorded_hash = row.get("sha256")
            recorded_bytes = row.get("bytes")
            if not oracle.is_file() or oracle.is_symlink() or not isinstance(recorded_hash, str):
                raise RuntimeError(f"preexisting oracle disappeared or has invalid hash: {oracle}")
            if recorded_hash != sha256_file(oracle) or int(recorded_bytes) != oracle.stat().st_size:
                raise RuntimeError(f"preexisting oracle changed after input snapshot: {oracle}")
        elif row.get("sha256") is not None or row.get("bytes") is not None:
            raise RuntimeError(f"absent oracle has fabricated hash/size evidence: {oracle}")
    if observed != expected:
        raise RuntimeError(f"oracle input snapshot task/seed set mismatch: {path}")
    expected_dir = Path(formal_root) / RECORD_DIR_NAME
    if Path(value.get("continuation_record_dir", "")) != expected_dir:
        raise RuntimeError(f"oracle continuation record directory drift: {path}")
    return value


def _semantic_validation(
    oracle: Path,
    *,
    task: str,
    seed: int,
    state_bank_dir: Path | None,
) -> dict[str, Any]:
    """Validate a continuation shard, or disclose why binding is unavailable."""
    try:
        payload = json.loads(oracle.read_text(encoding="utf-8"))
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            raise RuntimeError("oracle rows are missing/empty")
        bank = str(rows[0].get("bank", ""))
        grid = int(payload.get("tile_grid", -1))
    except Exception as exc:
        raise RuntimeError(f"continuation oracle is not parseable: {oracle}: {exc}") from exc
    if state_bank_dir is None:
        return {
            "status": "LIMITATION",
            "validator": "validate_oracle_shard",
            "limitations": ["state-bank directory was not supplied; row/state binding cannot be established"],
            "task": task,
            "model_seed": int(seed),
            "bank": bank,
            "tile_grid": grid,
        }
    state_bank = Path(state_bank_dir) / f"{task}-{bank}.json"
    if not state_bank.is_file():
        return {
            "status": "LIMITATION",
            "validator": "validate_oracle_shard",
            "limitations": [f"state-bank is missing; row/state binding cannot be established: {state_bank}"],
            "task": task,
            "model_seed": int(seed),
            "bank": bank,
            "tile_grid": grid,
        }
    from validate_oracle_shard import validate_shard

    result = validate_shard(
        oracle,
        task=task,
        model_seed=int(seed),
        bank=bank,
        grid=grid,
        state_bank=state_bank,
    )
    return {"status": "PASS", "validator": "validate_oracle_shard", "result": result}


def _read_record(
    path: Path,
    snapshot_path: Path,
    row: dict[str, Any],
    formal_root: Path,
    expected_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing immutable continuation record: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"invalid immutable continuation record: {path}: {exc}") from exc
    if record.get("protocol_id") != PROTOCOL_ID or record.get("status") != "PASS":
        raise RuntimeError(f"invalid immutable continuation record status: {path}")
    snapshot_hash = sha256_file(snapshot_path)
    if record.get("snapshot_sha256") != snapshot_hash or int(record.get("snapshot_bytes", -1)) != snapshot_path.stat().st_size:
        raise RuntimeError(f"continuation record is bound to a different snapshot: {path}")
    task, seed = str(row["task"]), int(row["model_seed"])
    oracle = _oracle_path(formal_root, task, seed)
    if record.get("task") != task or int(record.get("model_seed", -1)) != seed or record.get("path") != str(oracle):
        raise RuntimeError(f"continuation record key/path mismatch: {path}")
    if not oracle.is_file() or oracle.is_symlink():
        raise RuntimeError(f"continuation oracle disappeared: {oracle}")
    if record.get("sha256") != sha256_file(oracle) or int(record.get("bytes", -1)) != oracle.stat().st_size:
        raise RuntimeError(f"continuation oracle changed after immutable record: {oracle}")
    semantic = record.get("semantic_validation")
    if not isinstance(semantic, dict) or semantic.get("status") not in {"PASS", "LIMITATION"}:
        raise RuntimeError(f"continuation semantic validation record missing: {path}")
    if expected_provenance is not None and record.get("continuation_provenance") != expected_provenance:
        raise RuntimeError(f"continuation provenance mismatch: {path}")
    return record


def continuation_records(
    formal_root: Path,
    snapshot_path: Path,
    snapshot_value: dict[str, Any],
    *,
    state_bank_dir: Path | None = None,
    create: bool = False,
    continuation_provenance: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Validate existing records and optionally create records for new shards."""
    root = Path(formal_root)
    records: list[dict[str, Any]] = []
    for row in sorted(snapshot_value["oracle_files"], key=lambda item: (str(item["task"]), int(item["model_seed"]))):
        if bool(row["preexisting"]):
            continue
        task, seed = str(row["task"]), int(row["model_seed"])
        oracle = _oracle_path(root, task, seed)
        record_path = _record_path(root, task, seed)
        if record_path.exists():
            record = _read_record(record_path, snapshot_path, row, root, continuation_provenance)
            records.append(record)
            continue
        if not oracle.is_file():
            continue
        if not create:
            raise RuntimeError(f"continuation oracle exists without immutable record: {oracle}")
        semantic = _semantic_validation(oracle, task=task, seed=seed, state_bank_dir=state_bank_dir)
        record = {
            "protocol_id": PROTOCOL_ID,
            "status": "PASS",
            "schema_version": 1,
            "task": task,
            "model_seed": seed,
            "path": str(oracle),
            "sha256": sha256_file(oracle),
            "bytes": oracle.stat().st_size,
            "snapshot_sha256": sha256_file(snapshot_path),
            "snapshot_bytes": snapshot_path.stat().st_size,
            "semantic_validation": semantic,
            "origin": "created_after_external_old_producer_terminal_no_overlap",
        }
        if continuation_provenance is not None:
            record["continuation_provenance"] = continuation_provenance
        exclusive_json(record_path, record)
        records.append(_read_record(record_path, snapshot_path, row, root, continuation_provenance))
    return records


def validate_existing(
    path: Path,
    formal_root: Path,
    tasks: list[str],
    seeds: list[int],
    *,
    state_bank_dir: Path | None = None,
    create_continuation_records: bool = True,
    continuation_provenance: dict[str, Any] | None = None,
    old_terminal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = _load_snapshot(path, formal_root, tasks, seeds, old_terminal=old_terminal)
    records = continuation_records(
        formal_root,
        path,
        value,
        state_bank_dir=state_bank_dir,
        create=create_continuation_records,
        continuation_provenance=continuation_provenance,
    )
    return {**value, "continuation_records": records}


validate_snapshot = validate_existing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-task", action="append", required=True)
    parser.add_argument("--model-seed", type=int, action="append", required=True)
    parser.add_argument("--state-bank-dir", type=Path, default=None)
    parser.add_argument("--continuation-registry-run", type=Path, default=None)
    parser.add_argument("--continuation-registry-evidence", type=Path, default=None)
    parser.add_argument("--continuation-run-id", default=None)
    parser.add_argument("--continuation-job-id", default=None)
    parser.add_argument("--continuation-source-commit", default=None)
    parser.add_argument("--continuation-source-tree", default=None)
    parser.add_argument("--continuation-launcher", type=Path, default=None)
    parser.add_argument("--continuation-source-manifest", type=Path, default=None)
    parser.add_argument("--old-producer-terminal", type=Path, default=None)
    parser.add_argument("--old-producer-job-id", default=None)
    parser.add_argument("--old-producer-run-id", default=None)
    parser.add_argument("--old-producer-registry-run", type=Path, default=None)
    parser.add_argument("--old-producer-registry-evidence", type=Path, default=None)
    args = parser.parse_args()
    provenance = None
    registry_args = (
        args.continuation_registry_run,
        args.continuation_registry_evidence,
        args.continuation_run_id,
        args.continuation_job_id,
        args.continuation_source_commit,
        args.continuation_source_tree,
        args.continuation_launcher,
        args.continuation_source_manifest,
        args.old_producer_terminal,
        args.old_producer_job_id,
        args.old_producer_run_id,
        args.old_producer_registry_run,
        args.old_producer_registry_evidence,
    )
    if any(value is not None for value in registry_args):
        if any(value is None for value in registry_args):
            parser.error("all continuation registry/source provenance arguments are required together")
        registry = validate_continuation_registry(
            args.continuation_registry_run,
            registry_evidence=args.continuation_registry_evidence,
            expected_run_id=args.continuation_run_id,
            expected_source_commit=args.continuation_source_commit,
            expected_source_tree=args.continuation_source_tree,
            expected_launcher=args.continuation_launcher,
            expected_job_id=args.continuation_job_id,
            expected_source_manifest=args.continuation_source_manifest,
        )
        if args.old_producer_terminal is None or args.old_producer_job_id is None or args.old_producer_run_id is None:
            parser.error("old producer terminal/job/run evidence is required with continuation provenance")
        old_terminal = validate_old_producer_terminal(
            args.old_producer_terminal,
            expected_job_id=args.old_producer_job_id,
            expected_run_id=args.old_producer_run_id,
            producer_registry_run=args.old_producer_registry_run,
            producer_registry_evidence=args.old_producer_registry_evidence,
        )
        provenance = {
            "run_id": registry["run_id"],
            "job_id": registry["job_id"],
            "use_oversold_resource": registry["use_oversold_resource"],
            "uid_gid": registry["uid_gid"],
            "source_commit": registry["source_commit"],
            "source_tree": registry["source_tree"],
            "launcher_sha256": registry["launcher_sha256"],
            "registry_files": registry["files"],
            "old_producer_terminal": old_terminal,
        }
    if args.output.is_file():
        value = validate_existing(
            args.output,
            args.formal_root,
            args.expected_task,
            args.model_seed,
            state_bank_dir=args.state_bank_dir,
            create_continuation_records=True,
            continuation_provenance=provenance,
            old_terminal=old_terminal if provenance is not None else None,
        )
        status = "validated_existing"
    else:
        value = snapshot(
            args.formal_root,
            args.expected_task,
            args.model_seed,
            old_terminal=old_terminal if provenance is not None else None,
        )
        status = exclusive_json(args.output, value)
        value = _load_snapshot(
            args.output,
            args.formal_root,
            args.expected_task,
            args.model_seed,
            old_terminal=old_terminal if provenance is not None else None,
        )
        value["continuation_records"] = continuation_records(
            args.formal_root,
            args.output,
            value,
            state_bank_dir=args.state_bank_dir,
            create=True,
            continuation_provenance=provenance,
        )
    print(json.dumps({"status": status, **value}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
