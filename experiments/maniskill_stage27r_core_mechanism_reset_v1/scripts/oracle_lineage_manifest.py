#!/usr/bin/env python3
"""Create/validate process-level lineage for legacy oracle rows.

Rows written by the first Stage-2.7R producer predate checkpoint/state-bank
hash fields.  This sidecar is the honest replacement: it binds each shard to
the clean source commit, selected screen checkpoint, state-bank bytes, screen
selection, launcher and PAI source provenance without pretending those hashes
were intrinsic to the old rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from common import PROTOCOL_ID, canonical_json
from record_oracle_inputs import continuation_records
from validate_continuation_evidence import (
    validate_continuation_registry,
    validate_old_producer_terminal,
)
from validate_oracle_shard import validate_shard

PRODUCER_SOURCE_COMMIT = "fa05c2ef52e5cce16f62397540162724bfd4a6b9"
PRODUCER_SOURCE_TREE = "6fdb28764d002def6d10e5a9c4f41918fe7713d1"
PRODUCER_JOB_ID = "dlc9nkd8q7u4szm3"
PRODUCER_RUN_ID = "stage27r-formal-idle-v9"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def source_record(repo: Path, launcher: Path, *, role: str, pai_run_id: str, pai_job_id: str) -> dict:
    if not pai_run_id or pai_run_id in {"unknown", "UNKNOWN"}:
        raise RuntimeError("continuation source run_id must be non-empty")
    if not pai_job_id or pai_job_id in {"unknown", "UNKNOWN"}:
        raise RuntimeError("continuation source exact JobId must come from external registry evidence")
    return {
        "role": role,
        "repository": str(repo),
        "commit": git_value(repo, "rev-parse", "HEAD"),
        "tree": git_value(repo, "rev-parse", "HEAD^{tree}"),
        "launcher": str(launcher),
        "launcher_sha256": sha256_file(launcher),
        "pai_run_id": pai_run_id,
        "pai_job_id": pai_job_id,
        "job_id_binding": "external registry resolved.json exact JobId",
    }


def producer_registry_record(
    *,
    evidence_path: Path,
    registry_run: Path,
    producer_source_root: Path,
    producer_launcher: Path,
    producer_job_id: str,
    old_producer_terminal: Path | None = None,
    continuation_registry_run: Path | None = None,
    continuation_registry_evidence: Path | None = None,
    continuation_job_id: str | None = None,
) -> dict:
    if producer_job_id != PRODUCER_JOB_ID:
        raise RuntimeError(f"unexpected producer JobId: {producer_job_id}")
    resolved = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    placement_path = Path(registry_run) / "placement-evidence.external.json"
    payload_files = sorted((Path(registry_run) / "payload").glob("*.sh"))
    if not placement_path.is_file() or len(payload_files) != 1:
        raise RuntimeError("producer registry evidence is incomplete")
    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    run_id = str(resolved.get("run_id", ""))
    if run_id != PRODUCER_RUN_ID or str(placement.get("run_id")) != PRODUCER_RUN_ID:
        raise RuntimeError(f"producer registry run mismatch: {run_id}/{placement.get('run_id')}")
    if str(placement.get("job_id")) != producer_job_id:
        raise RuntimeError("producer placement JobId mismatch")
    if placement.get("complete") is not True or placement.get("use_oversold_resource") is not True:
        raise RuntimeError("producer placement evidence is not complete/exact oversold")
    evidence = resolved.get("evidence", {})
    if evidence.get("source_commit") != PRODUCER_SOURCE_COMMIT or evidence.get("source_tree") != PRODUCER_SOURCE_TREE:
        raise RuntimeError("producer registry source commit/tree mismatch")
    if not producer_source_root.is_dir() or not producer_launcher.is_file():
        raise RuntimeError("producer source/launcher evidence path missing")
    if git_value(producer_source_root, "rev-parse", "HEAD") != PRODUCER_SOURCE_COMMIT or git_value(producer_source_root, "rev-parse", "HEAD^{tree}") != PRODUCER_SOURCE_TREE:
        raise RuntimeError("producer checkout is not the frozen formal-source-v8 snapshot")
    payload = payload_files[0]
    payload_sha = sha256_file(payload)
    if evidence.get("validated_payload_sha256") != payload_sha:
        raise RuntimeError("producer payload hash mismatch")
    return {
        "role": "legacy_oracle_producer",
        "job_id": producer_job_id,
        "run_id": run_id,
        "source_repository": str(producer_source_root),
        "source_commit": PRODUCER_SOURCE_COMMIT,
        "source_tree": PRODUCER_SOURCE_TREE,
        "scientific_launcher": str(producer_launcher),
        "scientific_launcher_sha256": sha256_file(producer_launcher),
        "registry_evidence": {
            "resolved": {"path": str(evidence_path), "sha256": sha256_file(evidence_path), "bytes": Path(evidence_path).stat().st_size},
            "placement": {"path": str(placement_path), "sha256": sha256_file(placement_path), "bytes": placement_path.stat().st_size},
            "payload": {"path": str(payload), "sha256": payload_sha, "bytes": payload.stat().st_size},
        },
        "exact_placement": {"complete": True, "use_oversold_resource": True, "job_id": producer_job_id, "run_id": run_id},
    }


def exclusive_json(path: Path, value: dict) -> str:
    """Install a canonical sidecar exactly once, or verify an existing one."""
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
            existing = path.read_bytes()
            if existing != payload:
                raise RuntimeError(
                    f"lineage sidecar mismatch; refusing overwrite: {path}"
                )
            return "validated_existing"
        return "installed_missing"
    except Exception:
        raise
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def selected_checkpoint(selection: dict, task: str, seed: int) -> dict:
    key = f"{task}/seed_{seed}"
    try:
        row = selection["groups"][key]["selected"]
    except KeyError as exc:
        raise RuntimeError(f"missing selected checkpoint for {key}") from exc
    required = ("path", "sha256", "step")
    if any(field not in row for field in required):
        raise RuntimeError(f"selected checkpoint fields incomplete for {key}")
    checkpoint = Path(row["path"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != row["sha256"]:
        raise RuntimeError(f"selected checkpoint hash mismatch for {key}")
    return {
        "path": str(checkpoint),
        "sha256": row["sha256"],
        "step": int(row["step"]),
        "validation_loss": float(row.get("validation_loss", 0.0)),
        "selected_screen_sha256": sha256_file(selection["_path"]),
    }


def build_manifest(
    *,
    formal_root: Path,
    repo: Path,
    launcher: Path,
    state_bank_dir: Path,
    pai_run_id: str,
    pai_job_id: str,
    pai_source_manifest: Path | None,
    oracle_input_snapshot: Path,
    producer_registry_evidence: Path,
    producer_registry_run: Path,
    producer_source_root: Path,
    producer_launcher: Path,
    producer_job_id: str,
    old_producer_terminal: Path | None = None,
    continuation_registry_run: Path | None = None,
    continuation_registry_evidence: Path | None = None,
    continuation_job_id: str | None = None,
) -> dict:
    selection_path = formal_root / "screen" / "TASK_SELECTION.json"
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["_path"] = selection_path
    if old_producer_terminal is None or continuation_registry_run is None or continuation_registry_evidence is None:
        raise RuntimeError("continuation terminal and registry evidence are required for lineage")
    if not pai_source_manifest or not Path(pai_source_manifest).is_file():
        raise RuntimeError("continuation source manifest is required for lineage")
    snapshot = json.loads(Path(oracle_input_snapshot).read_text(encoding="utf-8"))
    if snapshot.get("protocol_id") != PROTOCOL_ID or snapshot.get("status") != "PASS" or snapshot.get("scope") != "pre_oracle_continuation_input_snapshot":
        raise RuntimeError("oracle input snapshot is invalid")
    snapshot_by_key = {
        (str(row["task"]), int(row["model_seed"])): row
        for row in snapshot.get("oracle_files", [])
    }
    snapshot_tasks = sorted({str(row["task"]) for row in snapshot.get("oracle_files", [])})
    snapshot_seeds = sorted({int(row["model_seed"]) for row in snapshot.get("oracle_files", [])})
    if len(snapshot_by_key) != len(snapshot.get("oracle_files", [])):
        raise RuntimeError("oracle input snapshot has duplicate task/seed rows")
    continuation_record_rows = continuation_records(
        formal_root,
        Path(oracle_input_snapshot),
        snapshot,
        create=False,
    )
    continuation_record_by_key = {
        (str(row["task"]), int(row["model_seed"])): row for row in continuation_record_rows
    }
    old_terminal = validate_old_producer_terminal(
        Path(old_producer_terminal), expected_job_id=producer_job_id, expected_run_id=PRODUCER_RUN_ID
    )
    expected_commit = git_value(repo, "rev-parse", "HEAD")
    expected_tree = git_value(repo, "rev-parse", "HEAD^{tree}")
    continuation_registry = validate_continuation_registry(
        Path(continuation_registry_run),
        registry_evidence=Path(continuation_registry_evidence),
        expected_run_id=pai_run_id,
        expected_source_commit=expected_commit,
        expected_source_tree=expected_tree,
        expected_launcher=Path(launcher),
        expected_job_id=continuation_job_id or pai_job_id,
        expected_source_manifest=Path(pai_source_manifest),
    )
    if continuation_registry["job_id"] != pai_job_id:
        raise RuntimeError("lineage pai_job_id does not match external continuation registry JobId")
    legacy_producer = producer_registry_record(
        evidence_path=producer_registry_evidence,
        registry_run=producer_registry_run,
        producer_source_root=producer_source_root,
        producer_launcher=producer_launcher,
        producer_job_id=producer_job_id,
    )
    verifier = source_record(
        repo,
        launcher,
        role="clean_posthoc_verifier_and_continuation_source",
        pai_run_id=pai_run_id,
        pai_job_id=pai_job_id,
    )
    oracle_files = sorted((formal_root / "oracle").glob("*.json"))
    if not oracle_files:
        raise RuntimeError("no oracle shards for lineage manifest")
    entries = []
    for path in oracle_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows") or []
        if not rows:
            raise RuntimeError(f"empty oracle shard: {path}")
        first = rows[0]
        # Negative-control diagnostics are intentionally outside the six
        # confirmatory inputs frozen by ORACLE_INPUT_SNAPSHOT.
        if str(first.get("bank", "")) != "confirmatory":
            continue
        task, seed, bank, grid = (
            str(first["task"]),
            int(first["model_seed"]),
            str(first["bank"]),
            int(payload.get("tile_grid", -1)),
        )
        state_bank = state_bank_dir / f"{task}-{bank}.json"
        shard_validation = validate_shard(
            path,
            task=task,
            model_seed=seed,
            bank=bank,
            grid=grid,
            state_bank=state_bank,
        )
        checkpoint = selected_checkpoint(selection, task, seed)
        snapshot_row = snapshot_by_key.get((task, seed))
        if snapshot_row is None:
            raise RuntimeError(f"oracle shard is outside frozen input snapshot: {path}")
        preexisting = bool(snapshot_row and snapshot_row.get("preexisting"))
        if preexisting:
            if snapshot_row.get("sha256") != sha256_file(path) or int(snapshot_row.get("bytes", -1)) != path.stat().st_size:
                raise RuntimeError(f"legacy oracle changed after input snapshot: {path}")
            entry_producer = legacy_producer
            producer_origin = "preexisting_legacy_formal_job"
            continuation_record = None
        else:
            entry_producer = {
                **verifier,
                "role": "continuation_oracle_producer",
                "source_lineage_note": "this shard was absent in the pre-oracle snapshot and was created by the clean continuation launcher",
            }
            producer_origin = "created_after_clean_continuation_start"
            continuation_record = continuation_record_by_key.get((task, seed))
            if continuation_record is None:
                raise RuntimeError(f"continuation shard has no immutable input record: {path}")
            semantic = continuation_record.get("semantic_validation", {})
            if semantic.get("status") != "PASS":
                raise RuntimeError(f"continuation shard semantic validation is limited/failed: {path}")
            provenance = continuation_record.get("continuation_provenance", {})
            if provenance.get("run_id") != continuation_registry["run_id"] or provenance.get("job_id") != continuation_registry["job_id"]:
                raise RuntimeError(f"continuation shard provenance JobId/run_id mismatch: {path}")
            if provenance.get("source_commit") != continuation_registry["source_commit"] or provenance.get("source_tree") != continuation_registry["source_tree"]:
                raise RuntimeError(f"continuation shard provenance source mismatch: {path}")
            if provenance.get("old_producer_terminal", {}).get("evidence", {}).get("sha256") != old_terminal.get("evidence", {}).get("sha256"):
                raise RuntimeError(f"continuation shard provenance old terminal mismatch: {path}")
        entries.append(
            {
                "task": task,
                "model_seed": seed,
                "bank": bank,
                "tile_grid": grid,
                "oracle": {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                },
                "state_bank": {
                    "path": str(state_bank),
                    "sha256": sha256_file(state_bank),
                    "bytes": state_bank.stat().st_size,
                },
                "selected_checkpoint": checkpoint,
                "shard_validation": shard_validation,
                "producer_origin": producer_origin,
                "producer_source": entry_producer,
                "posthoc_verifier_source": verifier,
                "continuation_record": continuation_record,
            }
        )
    if not entries:
        raise RuntimeError("no confirmatory oracle shards for lineage manifest")
    source_manifest = None
    if pai_source_manifest is not None:
        source_manifest = {
            "path": str(pai_source_manifest),
            "sha256": sha256_file(pai_source_manifest),
            "bytes": pai_source_manifest.stat().st_size,
        }
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "lineage_scope": "process_level_sidecar_for_legacy_rows",
        "rows_embed_checkpoint_or_state_bank_hash": False,
        "lineage_limitation": "legacy rows lack intrinsic checkpoint/state-bank hash fields; this sidecar binds clean launcher inputs/outputs and discloses that limitation",
        "producer_source": legacy_producer,
        "old_producer_terminal": old_terminal,
        "continuation_registry": continuation_registry,
        "posthoc_verifier_source": {
            **verifier,
            "screen_selection": {
                "path": str(selection_path),
                "sha256": sha256_file(selection_path),
                "bytes": selection_path.stat().st_size,
            },
            "pai_source_manifest": source_manifest,
        },
        "oracle_input_snapshot": {
            "path": str(oracle_input_snapshot),
            "sha256": sha256_file(oracle_input_snapshot),
            "bytes": Path(oracle_input_snapshot).stat().st_size,
        },
        "oracle_input_continuation_records": {
            "directory": str(formal_root / "ORACLE_INPUT_RECORDS"),
            "records": continuation_record_rows,
            "record_count": len(continuation_record_rows),
        },
        "oracle_files": entries,
        "manifest_sha256_excludes_self": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--state-bank-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pai-run-id", default="")
    parser.add_argument("--pai-job-id", default="")
    parser.add_argument("--pai-source-manifest", type=Path)
    parser.add_argument("--oracle-input-snapshot", type=Path, required=True)
    parser.add_argument("--producer-registry-evidence", type=Path, required=True)
    parser.add_argument("--producer-registry-run", type=Path, required=True)
    parser.add_argument("--producer-source-root", type=Path, required=True)
    parser.add_argument("--producer-launcher", type=Path, required=True)
    parser.add_argument("--producer-job-id", required=True)
    parser.add_argument("--old-producer-terminal", type=Path, required=True)
    parser.add_argument("--continuation-registry-run", type=Path, required=True)
    parser.add_argument("--continuation-registry-evidence", type=Path, required=True)
    parser.add_argument("--continuation-job-id", required=True)
    args = parser.parse_args()
    result = build_manifest(
        formal_root=args.formal_root,
        repo=args.repo,
        launcher=args.launcher,
        state_bank_dir=args.state_bank_dir,
        pai_run_id=args.pai_run_id,
        pai_job_id=args.pai_job_id,
        pai_source_manifest=args.pai_source_manifest,
        oracle_input_snapshot=args.oracle_input_snapshot,
        producer_registry_evidence=args.producer_registry_evidence,
        producer_registry_run=args.producer_registry_run,
        producer_source_root=args.producer_source_root,
        producer_launcher=args.producer_launcher,
        producer_job_id=args.producer_job_id,
        old_producer_terminal=args.old_producer_terminal,
        continuation_registry_run=args.continuation_registry_run,
        continuation_registry_evidence=args.continuation_registry_evidence,
        continuation_job_id=args.continuation_job_id,
    )
    print(json.dumps({"status": exclusive_json(args.output, result), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
