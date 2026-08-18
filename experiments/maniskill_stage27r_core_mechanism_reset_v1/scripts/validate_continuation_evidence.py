#!/usr/bin/env python3
"""Fail-closed validation for a clean Stage-2.7R continuation.

The continuation is deliberately not allowed to infer predecessor state from
PAI environment variables.  The orchestrator must first leave two immutable
sets of read-only evidence on disk:

* an ``OLD_PRODUCER_TERMINAL`` attestation, and
* the registry records for the *new* run (``resolved.json``, placement
  evidence, exactly one payload, template, and source manifest).

This module only reads those files.  In particular it never creates a
terminal attestation and never falls back to ``PAI_CANARY_JOB_ID``.  The
launcher uses the returned registry JobId, which is learned from the external
readback after PAI creates the job.
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


TERMINAL_STATUSES = {
    "SUCCEEDED",
    "SUCCESS",
    "COMPLETED",
    "COMPLETE",
    "FAILED",
    "FAILURE",
    "STOPPED",
    "TERMINATED",
    "CANCELLED",
    "CANCELED",
}
NON_TERMINAL_STATUSES = {
    "RUNNING",
    "PENDING",
    "QUEUED",
    "CREATING",
    "SUBMITTED",
    "WAITING",
    "UNKNOWN",
    "",
}
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} must be a regular immutable file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - message is the useful part
        raise RuntimeError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return value


def _values_for_keys(value: Any, keys: set[str]) -> list[Any]:
    """Collect exact key matches without treating ``*_source_job_id`` as JobId."""
    found: list[Any] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in keys:
                found.append(child)
            found.extend(_values_for_keys(child, keys))
    elif isinstance(value, list):
        for child in value:
            found.extend(_values_for_keys(child, keys))
    return found


def _nonempty_strings(values: Iterable[Any], label: str) -> list[str]:
    result = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"{label} contains an empty/non-string value")
        result.append(value.strip())
    return result


def _one_consistent(values: Iterable[Any], label: str) -> str:
    rows = _nonempty_strings(values, label)
    if not rows or len(set(rows)) != 1:
        raise RuntimeError(f"{label} is missing or inconsistent: {rows}")
    return rows[0]


def _timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{label} is required and must be an ISO-8601 timestamp")
    text = value.strip()
    try:
        parsed = _datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"{label} is not ISO-8601: {text}") from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"{label} must include a timezone: {text}")
    # Do not compare against the worker clock: the registry and worker may be
    # in different time zones or have a deliberate clock skew.  Presence of a
    # timezone-aware, parseable observation is the auditable requirement.
    return text


def _bool_true(value: Any, label: str) -> None:
    if value is not True:
        raise RuntimeError(f"{label} must be the JSON boolean true")


def validate_old_producer_terminal(
    path: Path,
    *,
    expected_job_id: str,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    """Validate an externally-created no-overlap predecessor attestation."""
    payload = _read_json(path, "OLD_PRODUCER_TERMINAL")
    if not expected_job_id or expected_job_id in {"unknown", "UNKNOWN"}:
        raise RuntimeError("exact old producer JobId is required")

    job_values = _values_for_keys(payload, {"job_id", "JobId", "old_job_id", "producer_job_id"})
    job_id = _one_consistent(job_values, "OLD_PRODUCER_TERMINAL JobId")
    if job_id != expected_job_id:
        raise RuntimeError(f"old producer JobId mismatch: {job_id} != {expected_job_id}")

    if expected_run_id:
        run_values = _values_for_keys(payload, {"run_id", "old_run_id", "producer_run_id"})
        run_id = _one_consistent(run_values, "OLD_PRODUCER_TERMINAL run_id")
        if run_id != expected_run_id:
            raise RuntimeError(f"old producer run_id mismatch: {run_id} != {expected_run_id}")
    else:
        run_id = _one_consistent(
            _values_for_keys(payload, {"run_id", "old_run_id", "producer_run_id"}),
            "OLD_PRODUCER_TERMINAL run_id",
        )

    status_values = _nonempty_strings(
        _values_for_keys(payload, {"status", "state", "terminal_status", "job_status"}),
        "OLD_PRODUCER_TERMINAL status",
    )
    if len(set(status_values)) != 1:
        raise RuntimeError(f"old producer status is inconsistent: {status_values}")
    status = status_values[0]
    normalized_status = status.upper().replace(" ", "_")
    if normalized_status in NON_TERMINAL_STATUSES or normalized_status not in TERMINAL_STATUSES:
        raise RuntimeError(f"old producer is not terminal: {status}")

    terminal_values = _values_for_keys(payload, {"terminal", "is_terminal", "terminal_confirmed"})
    if not terminal_values or any(value is not True for value in terminal_values):
        raise RuntimeError("OLD_PRODUCER_TERMINAL lacks terminal=true attestation")

    no_overlap_values = _values_for_keys(
        payload,
        {"no_overlap", "no_overlap_confirmed", "overlap_prohibited", "old_job_not_running"},
    )
    if not no_overlap_values or any(value is not True for value in no_overlap_values):
        raise RuntimeError("OLD_PRODUCER_TERMINAL lacks no_overlap=true attestation")

    observed_values = _values_for_keys(payload, {"observed_at", "observed_at_utc", "terminal_observed_at"})
    observed_at = _timestamp(
        _one_consistent(observed_values, "OLD_PRODUCER_TERMINAL observed_at"),
        "OLD_PRODUCER_TERMINAL observed_at",
    )
    return {
        "status": "PASS",
        "job_id": job_id,
        "run_id": run_id,
        "terminal_status": status,
        "observed_at": observed_at,
        "no_overlap": True,
        "evidence": {
            "path": str(Path(path)),
            "sha256": sha256_file(Path(path)),
            "bytes": Path(path).stat().st_size,
        },
    }


def _regular_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    found: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file() and not path.is_symlink():
                found.add(path)
    return sorted(found)


def _unique_named_file(root: Path, label: str, patterns: Iterable[str]) -> Path:
    paths = _regular_files(root, patterns)
    if len(paths) != 1:
        raise RuntimeError(f"continuation registry requires exactly one {label}; got {paths}")
    return paths[0]


def _uid_gid(payloads: Iterable[dict[str, Any]]) -> str:
    pairs: list[str] = []
    for payload in payloads:
        for value in _values_for_keys(payload, {"uid_gid", "UID:GID", "runtime_uid_gid"}):
            if isinstance(value, str) and value.strip():
                pairs.append(value.strip())
        uids = _values_for_keys(payload, {"uid", "UID", "recorded_by_uid", "runtime_uid", "expected_first_work_uid"})
        gids = _values_for_keys(payload, {"gid", "GID", "recorded_by_gid", "runtime_gid", "expected_first_work_gid"})
        if uids and gids:
            for uid in uids:
                for gid in gids:
                    if isinstance(uid, (int, str)) and isinstance(gid, (int, str)):
                        pairs.append(f"{uid}:{gid}")
    normalized = {str(pair) for pair in pairs if str(pair) != ":"}
    if "2254:2254" not in normalized:
        raise RuntimeError(f"continuation registry is not bound to UID:GID 2254:2254: {sorted(normalized)}")
    if any(pair != "2254:2254" for pair in normalized):
        raise RuntimeError(f"continuation registry has inconsistent UID:GID evidence: {sorted(normalized)}")
    return "2254:2254"


def _hash_value(payloads: Iterable[dict[str, Any]], keys: set[str], label: str) -> str:
    values = _nonempty_strings(
        [item for payload in payloads for item in _values_for_keys(payload, keys)],
        label,
    )
    hashes = {value.lower() for value in values if HEX64.fullmatch(value)}
    if not hashes:
        raise RuntimeError(f"continuation registry has no valid {label}")
    if len(hashes) != 1:
        raise RuntimeError(f"continuation registry has inconsistent {label}: {sorted(hashes)}")
    return next(iter(hashes))


def _required_hash(payload: dict[str, Any], key: str, label: str) -> str:
    """Require one exact, consistently recorded SHA-256 field in one record."""
    values = _nonempty_strings(_values_for_keys(payload, {key}), label)
    hashes = {value.lower() for value in values if HEX64.fullmatch(value)}
    if not hashes:
        raise RuntimeError(f"{label} is missing or not a valid SHA-256: {values}")
    if len(hashes) != 1:
        raise RuntimeError(f"{label} is inconsistent: {sorted(hashes)}")
    return next(iter(hashes))


def _source_template_binding(
    root: Path,
    resolved: dict[str, Any],
    source_manifest_payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate an optional source-template path and its independent hash.

    A registry may point at a template copied into the run, or at a template
    outside the registry directory.  For the latter, the source path and its
    hash must be repeated in both the resolved/request evidence and the source
    manifest.  If the path is locally readable, its bytes are checked too;
    an unreadable external path is rejected because a path-only assertion is
    not provenance.
    """
    values = _values_for_keys(resolved, {"source_template"})
    if not values:
        return None
    source_template = _one_consistent(values, "resolved source_template")
    manifest_paths = _values_for_keys(source_manifest_payload, {"source_template"})
    if not manifest_paths:
        raise RuntimeError("source manifest must bind resolved source_template")
    manifest_template = _one_consistent(manifest_paths, "source manifest source_template")
    if manifest_template != source_template:
        raise RuntimeError(
            f"source_template path mismatch: {source_template} != {manifest_template}"
        )

    candidate = Path(source_template)
    if not candidate.is_absolute():
        candidate = root / candidate
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve(strict=False)
    is_external = candidate_resolved != root_resolved and root_resolved not in candidate_resolved.parents
    if not is_external:
        # A registry-relative source_template is the copied template itself;
        # template_sha256 below already binds it to the immutable copy.
        return {"path": source_template, "external": False}

    source_hash = _required_hash(
        resolved,
        "source_template_sha256",
        "resolved source_template SHA-256",
    )
    manifest_hash = _required_hash(
        source_manifest_payload,
        "source_template_sha256",
        "source manifest source_template SHA-256",
    )
    if source_hash != manifest_hash:
        raise RuntimeError(
            f"source_template SHA-256 mismatch: {source_hash} != {manifest_hash}"
        )

    if not candidate.is_file() or candidate.is_symlink():
        raise RuntimeError(
            f"external source_template must be a readable regular file for hash binding: {candidate}"
        )
    observed = sha256_file(candidate)
    if observed != source_hash:
        raise RuntimeError(
            f"external source_template SHA-256 mismatch: {source_hash} != {observed}"
        )
    return {
        "path": source_template,
        "external": True,
        "sha256": source_hash,
        "bytes": candidate.stat().st_size,
    }


def validate_continuation_registry(
    registry_run: Path,
    *,
    registry_evidence: Path | None = None,
    expected_run_id: str,
    expected_source_commit: str,
    expected_source_tree: str,
    expected_launcher: Path,
    expected_job_id: str | None = None,
    expected_source_manifest: Path | None = None,
) -> dict[str, Any]:
    """Validate external readback for the newly-created continuation job."""
    if not expected_run_id or expected_run_id in {"unknown", "UNKNOWN"}:
        raise RuntimeError("continuation run_id cannot be unknown")
    if not expected_source_commit or expected_source_commit in {"unknown", "UNKNOWN"}:
        raise RuntimeError("continuation source commit cannot be unknown")
    if not expected_source_tree or expected_source_tree in {"unknown", "UNKNOWN"}:
        raise RuntimeError("continuation source tree cannot be unknown")
    root = Path(registry_run)
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"continuation registry run is missing/not a directory: {root}")
    resolved_path = Path(registry_evidence) if registry_evidence is not None else root / "resolved.json"
    if resolved_path != root / "resolved.json":
        raise RuntimeError("registry evidence must be the run's exact resolved.json")
    placement_path = root / "placement-evidence.external.json"
    resolved = _read_json(resolved_path, "continuation resolved.json")
    placement = _read_json(placement_path, "placement-evidence.external.json")

    run_values = _values_for_keys(resolved, {"run_id"}) + _values_for_keys(placement, {"run_id"})
    run_id = _one_consistent(run_values, "continuation registry run_id")
    if run_id != expected_run_id:
        raise RuntimeError(f"continuation registry run_id mismatch: {run_id} != {expected_run_id}")

    job_values = _values_for_keys(
        resolved,
        {"job_id", "JobId", "pai_job_id", "continuation_job_id"},
    ) + _values_for_keys(placement, {"job_id", "JobId", "pai_job_id", "continuation_job_id"})
    job_id = _one_consistent(job_values, "continuation registry exact JobId")
    if job_id in {"unknown", "UNKNOWN", "null", "None"}:
        raise RuntimeError("continuation registry JobId cannot be unknown")
    if expected_job_id not in (None, "") and job_id != expected_job_id:
        raise RuntimeError(f"continuation JobId mismatch: {job_id} != {expected_job_id}")

    _bool_true(placement.get("complete"), "placement evidence complete")
    _bool_true(placement.get("use_oversold_resource"), "placement evidence use_oversold_resource")
    # A string "true" is intentionally not accepted: this is an attestation,
    # not a best-effort parse of an untrusted shell environment.
    if "use_oversold_resource" in resolved:
        _bool_true(resolved["use_oversold_resource"], "resolved use_oversold_resource")
    elif isinstance(resolved.get("evidence"), dict) and "use_oversold_resource" in resolved["evidence"]:
        _bool_true(resolved["evidence"]["use_oversold_resource"], "resolved evidence use_oversold_resource")
    uid_gid = _uid_gid((resolved, placement))

    payload = _unique_named_file(root, "payload", ("payload", "payload/*", "payload/**/*", "payload.*"))
    template = _unique_named_file(
        root,
        "template",
        (
            "template",
            "template.*",
            "templates/template",
            "templates/template.*",
            "**/template",
            "**/template.*",
        ),
    )
    source_manifest = _unique_named_file(
        root,
        "source manifest",
        (
            "source-manifest",
            "source-manifest.*",
            "source_manifest",
            "source_manifest.*",
            "**/source-manifest",
            "**/source-manifest.*",
            "**/source_manifest",
            "**/source_manifest.*",
        ),
    )
    if expected_source_manifest is not None and Path(expected_source_manifest) != source_manifest:
        raise RuntimeError(
            f"source manifest path mismatch: {source_manifest} != {expected_source_manifest}"
        )

    source_manifest_payload = _read_json(source_manifest, "continuation source manifest")
    # A template placeholder is not provenance.  The source manifest and the
    # external resolved/request evidence must each carry the exact hash of the
    # immutable template copy, and that hash must match the bytes on disk.
    template_sha = sha256_file(template)
    manifest_template_sha = _required_hash(
        source_manifest_payload,
        "template_sha256",
        "source manifest template SHA-256",
    )
    resolved_template_sha = _required_hash(
        resolved,
        "template_sha256",
        "resolved/request template SHA-256",
    )
    if manifest_template_sha != resolved_template_sha:
        raise RuntimeError(
            f"template SHA-256 mismatch between source manifest and resolved/request evidence: "
            f"{manifest_template_sha} != {resolved_template_sha}"
        )
    if template_sha != manifest_template_sha:
        raise RuntimeError(
            f"template SHA-256 mismatch: recorded {manifest_template_sha} != actual {template_sha}"
        )
    source_template_binding = _source_template_binding(
        root,
        resolved,
        source_manifest_payload,
    )
    source_payloads = (resolved, placement, source_manifest_payload)
    source_commit = _hash_or_text(source_payloads, {"source_commit", "commit", "git_commit"}, "source commit")
    source_tree = _hash_or_text(source_payloads, {"source_tree", "tree", "git_tree"}, "source tree")
    if source_commit != expected_source_commit:
        raise RuntimeError(f"continuation source commit mismatch: {source_commit} != {expected_source_commit}")
    if source_tree != expected_source_tree:
        raise RuntimeError(f"continuation source tree mismatch: {source_tree} != {expected_source_tree}")
    launcher_sha = sha256_file(Path(expected_launcher))
    recorded_launcher_sha = _hash_value(
        source_payloads,
        {"launcher_sha256", "source_launcher_sha256", "scientific_launcher_sha256"},
        "launcher SHA-256",
    )
    if recorded_launcher_sha != launcher_sha:
        raise RuntimeError(f"continuation launcher hash mismatch: {recorded_launcher_sha} != {launcher_sha}")
    payload_sha = sha256_file(payload)
    recorded_payload_sha = _hash_value(
        source_payloads,
        {"payload_sha256", "validated_payload_sha256", "user_command_sha256"},
        "payload SHA-256",
    )
    if recorded_payload_sha != payload_sha:
        raise RuntimeError(f"continuation payload hash mismatch: {recorded_payload_sha} != {payload_sha}")

    return {
        "status": "PASS",
        "run_id": run_id,
        "job_id": job_id,
        "use_oversold_resource": True,
        "uid_gid": uid_gid,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "template_sha256": template_sha,
        "source_template": source_template_binding,
        "files": {
            "resolved": {"path": str(resolved_path), "sha256": sha256_file(resolved_path), "bytes": resolved_path.stat().st_size},
            "placement": {"path": str(placement_path), "sha256": sha256_file(placement_path), "bytes": placement_path.stat().st_size},
            "payload": {"path": str(payload), "sha256": payload_sha, "bytes": payload.stat().st_size},
            "template": {"path": str(template), "sha256": sha256_file(template), "bytes": template.stat().st_size},
            "source_manifest": {"path": str(source_manifest), "sha256": sha256_file(source_manifest), "bytes": source_manifest.stat().st_size},
        },
        "launcher_sha256": launcher_sha,
    }


def _hash_or_text(payloads: Iterable[dict[str, Any]], keys: set[str], label: str) -> str:
    values = _nonempty_strings([item for payload in payloads for item in _values_for_keys(payload, keys)], label)
    if not values or len(set(values)) != 1:
        raise RuntimeError(f"continuation registry has missing/inconsistent {label}: {values}")
    return values[0]


# Short aliases make the read-only gates convenient for independent audits
# without exposing a second implementation or a mutation-capable API.
validate_terminal = validate_old_producer_terminal
validate_registry = validate_continuation_registry


def validate_all(
    *,
    old_terminal: Path,
    old_job_id: str,
    old_run_id: str,
    registry_run: Path,
    registry_evidence: Path,
    expected_run_id: str,
    expected_source_commit: str,
    expected_source_tree: str,
    expected_launcher: Path,
    expected_job_id: str | None = None,
    expected_source_manifest: Path | None = None,
) -> dict[str, Any]:
    terminal = validate_old_producer_terminal(
        old_terminal, expected_job_id=old_job_id, expected_run_id=old_run_id
    )
    registry = validate_continuation_registry(
        registry_run,
        registry_evidence=registry_evidence,
        expected_run_id=expected_run_id,
        expected_source_commit=expected_source_commit,
        expected_source_tree=expected_source_tree,
        expected_launcher=expected_launcher,
        expected_job_id=expected_job_id,
        expected_source_manifest=expected_source_manifest,
    )
    return {"status": "PASS", "old_producer_terminal": terminal, "continuation_registry": registry}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-terminal", type=Path, required=True)
    parser.add_argument("--old-job-id", required=True)
    parser.add_argument("--old-run-id", required=True)
    parser.add_argument("--registry-run", type=Path, required=True)
    parser.add_argument("--registry-evidence", type=Path, required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--expected-launcher", type=Path, required=True)
    parser.add_argument("--expected-job-id", default=None)
    parser.add_argument("--expected-source-manifest", type=Path, default=None)
    parser.add_argument("--print-job-id", action="store_true")
    args = parser.parse_args()
    result = validate_all(
        old_terminal=args.old_terminal,
        old_job_id=args.old_job_id,
        old_run_id=args.old_run_id,
        registry_run=args.registry_run,
        registry_evidence=args.registry_evidence,
        expected_run_id=args.expected_run_id,
        expected_source_commit=args.expected_source_commit,
        expected_source_tree=args.expected_source_tree,
        expected_launcher=args.expected_launcher,
        expected_job_id=args.expected_job_id,
        expected_source_manifest=args.expected_source_manifest,
    )
    if args.print_job_id:
        print(result["continuation_registry"]["job_id"])
    else:
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
