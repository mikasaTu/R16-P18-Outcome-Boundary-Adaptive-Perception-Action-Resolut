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
EXPECTED_RESOURCE_ID = "quotaewyznuc7b9l"
EXPECTED_OVERSOLD_TYPE = "AcceptQuotaOverSold"
EXPECTED_WORKER_COUNT = 1
EXPECTED_GPU_COUNT = 8
EXPECTED_CPU_COUNT = 92
EXPECTED_MEMORY_GI = "1600Gi"
EXPECTED_UID_GID = "2254:2254"
EXPECTED_TEMPLATE_SCHEMA_VERSION = 2
EXPECTED_TEMPLATE_KIND = "pytorchjob"
EXPECTED_WORKSPACE_ID = 179169
EXPECTED_RESOURCE_ALIAS = "idle-a800-stablevla-native5-8gpu"


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


def _top_or_recursive_values(payload: dict[str, Any], keys: set[str]) -> list[Any]:
    """Prefer manifest top-level fields, then accept a nested contract field."""
    direct = [payload[key] for key in keys if key in payload]
    return direct or _values_for_keys(payload, keys)


def _required_consistent(
    payloads: Iterable[dict[str, Any]],
    keys: set[str],
    label: str,
    normalizer=None,
) -> Any:
    values: list[Any] = []
    for payload in payloads:
        found = _values_for_keys(payload, keys)
        if not found:
            raise RuntimeError(f"{label} is missing from one placement evidence record")
        values.extend(found)
    if not values:
        raise RuntimeError(f"{label} is missing")
    converted = [normalizer(value) if normalizer is not None else value for value in values]
    if len(set(converted)) != 1:
        raise RuntimeError(f"{label} is inconsistent: {converted}")
    return converted[0]


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise RuntimeError(f"{label} must be an integer: {value!r}")


def _memory_gi(value: Any, label: str) -> str:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be a GiB quantity")
    text = str(value).strip().replace(" ", "")
    lower = text.lower()
    if lower.endswith("gi"):
        number = lower[:-2]
    else:
        number = lower
    try:
        amount = float(number)
    except ValueError as exc:
        raise RuntimeError(f"{label} must be a GiB quantity: {value!r}") from exc
    if not amount.is_integer() or amount < 0:
        raise RuntimeError(f"{label} must be a non-negative integer GiB quantity: {value!r}")
    return f"{int(amount)}Gi"


def _bound_json_file(
    metadata: Any,
    label: str,
    *,
    expected_path: Path | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    if not isinstance(metadata, dict):
        raise RuntimeError(f"{label} path/hash/bytes binding is missing")
    raw_path = metadata.get("path")
    raw_hash = metadata.get("sha256")
    raw_bytes = metadata.get("bytes")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise RuntimeError(f"{label} path is required")
    if not isinstance(raw_hash, str) or not HEX64.fullmatch(raw_hash.strip()):
        raise RuntimeError(f"{label} sha256 is required and must be valid")
    if isinstance(raw_bytes, bool) or not isinstance(raw_bytes, int) or raw_bytes < 0:
        raise RuntimeError(f"{label} bytes is required and must be a non-negative integer")
    path = Path(raw_path)
    if expected_path is not None and path != Path(expected_path):
        raise RuntimeError(f"{label} path mismatch: {path} != {expected_path}")
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} must be a regular immutable JSON file: {path}")
    observed_hash = sha256_file(path)
    observed_bytes = path.stat().st_size
    if observed_hash.lower() != raw_hash.lower() or observed_bytes != raw_bytes:
        raise RuntimeError(f"{label} hash/bytes mismatch: {path}")
    payload = _read_json(path, label)
    return path, payload, {"path": str(path), "sha256": observed_hash, "bytes": observed_bytes}


def _metadata_object(payload: dict[str, Any], names: set[str], label: str) -> dict[str, Any]:
    values = _values_for_keys(payload, names)
    objects = [value for value in values if isinstance(value, dict)]
    if len(objects) != 1:
        raise RuntimeError(f"{label} must contain exactly one object binding")
    return objects[0]


def _status_value(payload: dict[str, Any], label: str) -> tuple[str, str]:
    values = _nonempty_strings(
        _top_or_recursive_values(
            payload,
            {"status", "Status", "state", "State", "terminal_status", "job_status", "JobStatus"},
        ),
        label,
    )
    if len(set(values)) != 1:
        raise RuntimeError(f"{label} is inconsistent: {values}")
    status = values[0]
    normalized = status.upper().replace(" ", "_")
    if normalized in NON_TERMINAL_STATUSES or normalized not in TERMINAL_STATUSES:
        raise RuntimeError(f"{label} is not terminal: {status}")
    return status, normalized


def _job_status_value(payload: dict[str, Any], label: str) -> tuple[str, str]:
    """Read only the top-level status of a raw GetJob object.

    A GetJob response legitimately contains independent pod/container status
    fields.  Those must not be mixed with the job-level terminal status.
    """
    values = [payload[key] for key in ("Status", "status", "JobStatus", "job_status") if key in payload]
    status = _one_consistent(values, label)
    normalized = status.upper().replace(" ", "_")
    if normalized in NON_TERMINAL_STATUSES or normalized not in TERMINAL_STATUSES:
        raise RuntimeError(f"{label} is not terminal: {status}")
    return status, normalized


def _registry_record_fields(payload: dict[str, Any], label: str, expected_run_id: str, expected_job_id: str) -> None:
    run_id = _one_consistent(_values_for_keys(payload, {"run_id"}), f"{label} run_id")
    job_id = _one_consistent(
        _values_for_keys(payload, {"job_id", "JobId", "pai_job_id", "continuation_job_id"}),
        f"{label} JobId",
    )
    if run_id != expected_run_id or job_id != expected_job_id:
        raise RuntimeError(
            f"{label} run/job mismatch: {run_id}/{job_id} != {expected_run_id}/{expected_job_id}"
        )


def _registry_run_only(payload: dict[str, Any], label: str, expected_run_id: str) -> None:
    run_id = _one_consistent(_values_for_keys(payload, {"run_id"}), f"{label} run_id")
    if run_id != expected_run_id:
        raise RuntimeError(f"{label} run_id mismatch: {run_id} != {expected_run_id}")


def validate_old_producer_terminal(
    path: Path,
    *,
    expected_job_id: str,
    expected_run_id: str | None = None,
    producer_registry_run: Path | None = None,
    producer_registry_evidence: Path | None = None,
) -> dict[str, Any]:
    """Validate an externally-created no-overlap predecessor attestation."""
    payload = _read_json(path, "OLD_PRODUCER_TERMINAL")
    if not expected_job_id or expected_job_id in {"unknown", "UNKNOWN"}:
        raise RuntimeError("exact old producer JobId is required")
    if producer_registry_run is None or producer_registry_evidence is None:
        raise RuntimeError("OLD_PRODUCER_TERMINAL must bind the old producer registry")

    job_values = _top_or_recursive_values(payload, {"job_id", "JobId", "old_job_id", "producer_job_id"})
    job_id = _one_consistent(job_values, "OLD_PRODUCER_TERMINAL JobId")
    if job_id != expected_job_id:
        raise RuntimeError(f"old producer JobId mismatch: {job_id} != {expected_job_id}")

    if expected_run_id:
        run_values = _top_or_recursive_values(payload, {"run_id", "old_run_id", "producer_run_id"})
        run_id = _one_consistent(run_values, "OLD_PRODUCER_TERMINAL run_id")
        if run_id != expected_run_id:
            raise RuntimeError(f"old producer run_id mismatch: {run_id} != {expected_run_id}")
    else:
        run_id = _one_consistent(
            _top_or_recursive_values(payload, {"run_id", "old_run_id", "producer_run_id"}),
            "OLD_PRODUCER_TERMINAL run_id",
        )

    status, normalized_status = _status_value(payload, "OLD_PRODUCER_TERMINAL status")

    terminal_values = _values_for_keys(payload, {"terminal", "is_terminal", "terminal_confirmed"})
    if not terminal_values or any(value is not True for value in terminal_values):
        raise RuntimeError("OLD_PRODUCER_TERMINAL lacks terminal=true attestation")

    no_overlap_values = _values_for_keys(
        payload,
        {"no_overlap", "no_overlap_confirmed", "overlap_prohibited", "old_job_not_running"},
    )
    if not no_overlap_values or any(value is not True for value in no_overlap_values):
        raise RuntimeError("OLD_PRODUCER_TERMINAL lacks no_overlap=true attestation")

    observed_values = _top_or_recursive_values(payload, {"observed_at", "observed_at_utc", "terminal_observed_at"})
    observed_at = _timestamp(
        _one_consistent(observed_values, "OLD_PRODUCER_TERMINAL observed_at"),
        "OLD_PRODUCER_TERMINAL observed_at",
    )
    getjob_binding = _metadata_object(
        payload,
        {"getjob", "raw_getjob", "raw_get_job", "get_job"},
        "OLD_PRODUCER_TERMINAL raw GetJob",
    )
    getjob_path, getjob_payload, getjob_file = _bound_json_file(getjob_binding, "raw GetJob")
    raw_job_id = _one_consistent(
        [getjob_payload[key] for key in ("JobId", "job_id", "pai_job_id") if key in getjob_payload],
        "raw GetJob JobId",
    )
    if raw_job_id != expected_job_id:
        raise RuntimeError(f"raw GetJob JobId mismatch: {raw_job_id} != {expected_job_id}")
    raw_status, raw_normalized_status = _job_status_value(getjob_payload, "raw GetJob status")
    if raw_normalized_status != normalized_status:
        raise RuntimeError(f"OLD_PRODUCER_TERMINAL summary/raw status mismatch: {status} != {raw_status}")

    registry_binding = _metadata_object(
        payload,
        {"producer_registry", "old_producer_registry", "registry"},
        "OLD_PRODUCER_TERMINAL producer registry",
    )
    registry_run_values = _values_for_keys(registry_binding, {"run_id"})
    registry_run_id = _one_consistent(registry_run_values, "OLD_PRODUCER_TERMINAL producer registry run_id")
    if registry_run_id != run_id:
        raise RuntimeError(f"OLD_PRODUCER_TERMINAL producer registry run_id mismatch: {registry_run_id} != {run_id}")
    registry_job_id = _one_consistent(
        _values_for_keys(registry_binding, {"job_id", "JobId", "producer_job_id"}),
        "OLD_PRODUCER_TERMINAL producer registry JobId",
    )
    if registry_job_id != expected_job_id:
        raise RuntimeError(f"OLD_PRODUCER_TERMINAL producer registry JobId mismatch: {registry_job_id} != {expected_job_id}")

    expected_registry_run = Path(producer_registry_run)
    expected_resolved = Path(producer_registry_evidence)
    if expected_resolved != expected_registry_run / "resolved.json":
        raise RuntimeError("old producer registry evidence must be the exact run resolved.json")
    expected_placement = expected_registry_run / "placement-evidence.external.json"
    resolved_binding = registry_binding.get("resolved")
    placement_binding = registry_binding.get("placement")
    resolved_path, resolved_payload, resolved_file = _bound_json_file(
        resolved_binding,
        "old producer resolved registry",
        expected_path=expected_resolved,
    )
    placement_path, placement_payload, placement_file = _bound_json_file(
        placement_binding,
        "old producer placement registry",
        expected_path=expected_placement,
    )
    _registry_run_only(resolved_payload, "old producer resolved registry", run_id)
    _registry_record_fields(placement_payload, "old producer placement registry", run_id, expected_job_id)
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
        "raw_getjob": {"path": str(getjob_path), **getjob_file},
        "producer_registry": {
            "run_id": run_id,
            "job_id": expected_job_id,
            "resolved": {"path": str(resolved_path), **resolved_file},
            "placement": {"path": str(placement_path), **placement_file},
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


def _uid_gid_each(payloads: Iterable[dict[str, Any]], label: str) -> str:
    """Require every sealed placement/template record to state UID:GID."""
    observed: list[str] = []
    for index, payload in enumerate(payloads):
        pairs: set[str] = set()
        for value in _values_for_keys(payload, {"uid_gid", "UID:GID", "runtime_uid_gid"}):
            if isinstance(value, str) and value.strip():
                pairs.add(value.strip())
        uids = _values_for_keys(payload, {"uid", "UID", "recorded_by_uid", "runtime_uid", "expected_first_work_uid"})
        gids = _values_for_keys(payload, {"gid", "GID", "recorded_by_gid", "runtime_gid", "expected_first_work_gid"})
        for uid in uids:
            for gid in gids:
                if isinstance(uid, (int, str)) and isinstance(gid, (int, str)):
                    pairs.add(f"{uid}:{gid}")
        if pairs != {EXPECTED_UID_GID}:
            raise RuntimeError(f"{label} record {index} UID:GID is missing/inconsistent: {sorted(pairs)}")
        observed.extend(pairs)
    if not observed:
        raise RuntimeError(f"{label} UID:GID evidence is missing")
    return EXPECTED_UID_GID


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
    template: Path,
    resolved: dict[str, Any],
    source_manifest_payload: dict[str, Any],
) -> dict[str, Any]:
    """Validate source-template bytes and their immutable copied counterpart.

    The source path and hash are required even when the source is registry
    relative.  A path-only assertion is not provenance: the source bytes and
    copied template bytes must hash identically, and both hashes are repeated
    in the resolved/request evidence and source manifest.
    """
    values = _values_for_keys(resolved, {"source_template"})
    if not values:
        raise RuntimeError("resolved/request source_template path is required")
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
    copied_hash = sha256_file(template)
    if copied_hash != source_hash:
        raise RuntimeError(
            f"copied/source template SHA-256 mismatch: {copied_hash} != {source_hash}"
        )
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve(strict=False)
    is_external = candidate_resolved != root_resolved and root_resolved not in candidate_resolved.parents
    return {
        "path": source_template,
        "external": is_external,
        "sha256": source_hash,
        "bytes": candidate.stat().st_size,
        "copied_sha256": copied_hash,
    }


def _exact_path(payload: dict[str, Any], path: tuple[str, ...], label: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            joined = ".".join(path)
            raise RuntimeError(f"{label} missing exact path {joined}")
        current = current[key]
    return current


def _sealed_placement_source(
    placement: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Read the external placement's immutable sealed raw response binding."""
    raw_path = _exact_path(placement, ("sealed_source_path",), "placement sealed source")
    raw_hash = _exact_path(placement, ("sealed_source_sha256",), "placement sealed source")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise RuntimeError("placement sealed source path is required")
    if not isinstance(raw_hash, str) or not HEX64.fullmatch(raw_hash.strip()):
        raise RuntimeError("placement sealed source sha256 is required and must be valid")
    path = Path(raw_path)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"placement sealed source must be a regular immutable JSON file: {path}")
    observed_hash = sha256_file(path)
    observed_bytes = path.stat().st_size
    if observed_hash.lower() != raw_hash.strip().lower():
        raise RuntimeError(f"placement sealed source hash mismatch: {path}")
    payload = _read_json(path, "sealed raw placement response")
    return path, payload, {"path": str(path), "sha256": observed_hash, "bytes": observed_bytes}


def _sealed_placement_job(
    payload: dict[str, Any],
    *,
    expected_run_id: str,
    expected_job_id: str,
) -> dict[str, Any]:
    """Parse the exact PAI GetJobs response shape used by the sealed readback."""
    response = _exact_path(payload, ("response",), "sealed raw placement response")
    if not isinstance(response, dict):
        raise RuntimeError("sealed raw placement response.response must be an object")
    jobs = _exact_path(response, ("Jobs",), "sealed raw placement response")
    if not isinstance(jobs, list):
        raise RuntimeError("sealed raw placement response.response.Jobs must be a list")
    matches = [job for job in jobs if isinstance(job, dict) and job.get("JobId") == expected_job_id]
    if len(matches) != 1:
        raise RuntimeError(
            f"sealed raw placement response must contain exactly one exact JobId {expected_job_id}; "
            f"found {len(matches)}"
        )
    job = matches[0]
    settings = _exact_path(job, ("Settings",), "sealed raw placement job")
    if not isinstance(settings, dict):
        raise RuntimeError("sealed raw placement job Settings must be an object")
    tags = _exact_path(settings, ("Tags",), "sealed raw placement job")
    if not isinstance(tags, dict) or tags.get("run_id") != expected_run_id:
        raise RuntimeError("sealed raw placement job Settings.Tags.run_id mismatch")

    use_values = []
    if "UseOversoldResource" in job:
        use_values.append(job["UseOversoldResource"])
    if "UseOversoldResource" in settings:
        use_values.append(settings["UseOversoldResource"])
    if not use_values or len(set(use_values)) != 1 or use_values[0] is not True:
        raise RuntimeError("sealed raw placement job UseOversoldResource must be true")

    specs = _exact_path(job, ("JobSpecs",), "sealed raw placement job")
    if not isinstance(specs, list):
        raise RuntimeError("sealed raw placement job JobSpecs must be a list")
    workers = [spec for spec in specs if isinstance(spec, dict) and spec.get("Type") == "Worker"]
    if len(workers) != 1:
        raise RuntimeError(f"sealed raw placement job must contain exactly one Worker JobSpecs entry; found {len(workers)}")
    worker = workers[0]
    resource_config = _exact_path(worker, ("ResourceConfig",), "sealed raw placement Worker")
    if not isinstance(resource_config, dict):
        raise RuntimeError("sealed raw placement Worker ResourceConfig must be an object")
    return {
        "resource_id": _exact_path(job, ("ResourceId",), "sealed raw placement job"),
        "oversold_type": _exact_path(settings, ("OversoldType",), "sealed raw placement Settings"),
        "use_oversold_resource": True,
        "worker_count": _strict_int(
            _exact_path(worker, ("PodCount",), "sealed raw placement Worker"),
            "sealed raw placement worker count",
        ),
        "gpu_count": _strict_int(
            _exact_path(resource_config, ("GPU",), "sealed raw placement ResourceConfig"),
            "sealed raw placement GPU count",
        ),
        "cpu_count": _strict_int(
            _exact_path(resource_config, ("CPU",), "sealed raw placement ResourceConfig"),
            "sealed raw placement CPU count",
        ),
        "memory": _memory_gi(
            _exact_path(resource_config, ("Memory",), "sealed raw placement ResourceConfig"),
            "sealed raw placement memory",
        ),
    }


def _placement_contract(
    root: Path,
    resolved: dict[str, Any],
    placement: dict[str, Any],
    *,
    expected_run_id: str,
    expected_job_id: str,
    expected_resource_id: str,
    expected_oversold_type: str,
    expected_worker_count: int,
    expected_gpu_count: int,
    expected_cpu_count: int,
    expected_memory_gi: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if placement.get("complete") is not True:
        raise RuntimeError("placement evidence complete must be true")
    if not expected_job_id or expected_job_id in {"unknown", "UNKNOWN"}:
        raise RuntimeError("placement exact JobId is required")
    _registry_record_fields(placement, "placement evidence", expected_run_id, expected_job_id)
    raw_path, raw_payload, raw_file = _sealed_placement_source(placement)
    raw_contract = _sealed_placement_job(
        raw_payload,
        expected_run_id=expected_run_id,
        expected_job_id=expected_job_id,
    )

    resource_id = str(_exact_path(resolved, ("resource", "resource_id"), "resolved resource")).strip()
    resolved_oversold = str(_exact_path(resolved, ("resource", "oversold_type"), "resolved resource")).strip()
    placement_resource_id = str(_exact_path(placement, ("resource_id",), "placement evidence")).strip()
    placement_oversold = str(_exact_path(placement, ("oversold_type",), "placement evidence")).strip()
    placement_use = _exact_path(placement, ("use_oversold_resource",), "placement evidence")
    _bool_true(placement_use, "placement use_oversold_resource")
    if resource_id != placement_resource_id or resource_id != str(raw_contract["resource_id"]).strip():
        raise RuntimeError(
            f"placement resource_id is inconsistent: {resource_id}, {placement_resource_id}, "
            f"{raw_contract['resource_id']}"
        )
    if resolved_oversold != placement_oversold or resolved_oversold != str(raw_contract["oversold_type"]).strip():
        raise RuntimeError(
            f"placement oversold_type is inconsistent: {resolved_oversold}, {placement_oversold}, "
            f"{raw_contract['oversold_type']}"
        )

    resolved_worker = _exact_path(resolved, ("worker",), "resolved worker")
    if not isinstance(resolved_worker, dict):
        raise RuntimeError("resolved worker must be an object")
    worker_count = _strict_int(_exact_path(resolved_worker, ("count",), "resolved worker"), "placement worker count")
    gpu_count = _strict_int(_exact_path(resolved_worker, ("gpu",), "resolved worker"), "placement GPU count")
    cpu_count = _strict_int(_exact_path(resolved_worker, ("cpu",), "resolved worker"), "placement CPU count")
    memory_gi = _memory_gi(_exact_path(resolved_worker, ("memory",), "resolved worker"), "placement memory")
    observed_resources = {
        "worker_count": worker_count,
        "gpu_count": gpu_count,
        "cpu_count": cpu_count,
        "memory": memory_gi,
    }
    raw_resources = {key: raw_contract[key] for key in observed_resources}
    if observed_resources != raw_resources:
        raise RuntimeError(f"placement worker resource mismatch: {observed_resources} != {raw_resources}")

    runtime = _exact_path(resolved, ("runtime",), "resolved runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError("resolved runtime must be an object")
    resolved_uid = _strict_int(_exact_path(runtime, ("uid",), "resolved runtime"), "resolved runtime uid")
    resolved_gid = _strict_int(_exact_path(runtime, ("gid",), "resolved runtime"), "resolved runtime gid")
    placement_uid = _strict_int(
        _exact_path(placement, ("recorded_by_uid",), "placement evidence"),
        "placement recorded_by_uid",
    )
    placement_gid = _strict_int(
        _exact_path(placement, ("recorded_by_gid",), "placement evidence"),
        "placement recorded_by_gid",
    )
    if f"{resolved_uid}:{resolved_gid}" != EXPECTED_UID_GID or f"{placement_uid}:{placement_gid}" != EXPECTED_UID_GID:
        raise RuntimeError("placement UID:GID evidence must be exactly 2254:2254")
    uid_gid = EXPECTED_UID_GID
    if resource_id != expected_resource_id:
        raise RuntimeError(f"placement resource_id mismatch: {resource_id} != {expected_resource_id}")
    if resolved_oversold != expected_oversold_type:
        raise RuntimeError(f"placement oversold_type mismatch: {resolved_oversold} != {expected_oversold_type}")
    expected = {
        "worker_count": expected_worker_count,
        "gpu_count": expected_gpu_count,
        "cpu_count": expected_cpu_count,
        "memory": expected_memory_gi,
    }
    observed = {
        "worker_count": worker_count,
        "gpu_count": gpu_count,
        "cpu_count": cpu_count,
        "memory": memory_gi,
    }
    if observed != expected:
        raise RuntimeError(f"placement worker resource mismatch: {observed} != {expected}")
    return (
        {
            "resource_id": resource_id,
            "oversold_type": resolved_oversold,
            "use_oversold_resource": True,
            "worker_count": worker_count,
            "gpu_count": gpu_count,
            "cpu_count": cpu_count,
            "memory": memory_gi,
            "uid_gid": uid_gid,
        },
        {"path": str(raw_path), **raw_file},
    )


def _template_contract(template: dict[str, Any]) -> dict[str, Any]:
    schema_version = _strict_int(
        _exact_path(template, ("schema_version",), "template schema/contract"),
        "template schema_version",
    )
    kind = _exact_path(template, ("kind",), "template")
    if not isinstance(kind, str) or not kind.strip():
        raise RuntimeError("template kind must be a non-empty string")
    kind = kind.strip()
    workspace_id = _strict_int(_exact_path(template, ("workspace_id",), "template"), "template workspace_id")
    resource_alias = _exact_path(template, ("resource_alias",), "template")
    if not isinstance(resource_alias, str) or not resource_alias.strip():
        raise RuntimeError("template resource_alias must be a non-empty string")
    resource_alias = resource_alias.strip()
    worker = _exact_path(template, ("worker",), "template")
    if not isinstance(worker, dict):
        raise RuntimeError("template worker must be an object")
    worker_count = _strict_int(_exact_path(worker, ("count",), "template worker"), "template worker.count")
    gpu_count = _strict_int(_exact_path(worker, ("gpu",), "template worker"), "template worker.gpu")
    cpu_count = _strict_int(_exact_path(worker, ("cpu",), "template worker"), "template worker.cpu")
    memory_gi = _memory_gi(_exact_path(worker, ("memory",), "template worker"), "template worker.memory")
    runtime = _exact_path(template, ("runtime",), "template")
    if not isinstance(runtime, dict):
        raise RuntimeError("template runtime must be an object")
    uid = _strict_int(_exact_path(runtime, ("uid",), "template runtime"), "template runtime.uid")
    gid = _strict_int(_exact_path(runtime, ("gid",), "template runtime"), "template runtime.gid")
    output_mode = _exact_path(runtime, ("output_mode",), "template runtime")
    if not isinstance(output_mode, str) or not output_mode.strip():
        raise RuntimeError("template runtime.output_mode must be a non-empty string")
    output_mode = output_mode.strip()
    fault_tolerance = _exact_path(template, ("fault_tolerance",), "template")
    if not isinstance(fault_tolerance, dict):
        raise RuntimeError("template fault_tolerance must be an object")
    autoresume = _exact_path(fault_tolerance, ("application_auto_resume",), "template fault_tolerance")
    pai_fault_tolerance = _exact_path(
        fault_tolerance,
        ("pai_automatic_fault_tolerance",),
        "template fault_tolerance",
    )
    _bool_true(autoresume, "template fault_tolerance.application_auto_resume")
    _bool_true(pai_fault_tolerance, "template fault_tolerance.pai_automatic_fault_tolerance")
    evidence = _exact_path(template, ("evidence",), "template")
    if not isinstance(evidence, dict):
        raise RuntimeError("template evidence must be an object")
    require_idle = _exact_path(evidence, ("require_actual_idle",), "template evidence")
    _bool_true(require_idle, "template evidence.require_actual_idle")
    submission = _exact_path(template, ("submission",), "template")
    if not isinstance(submission, dict):
        raise RuntimeError("template submission must be an object")
    priority = _strict_int(_exact_path(submission, ("priority",), "template submission"), "template submission.priority")
    disable_stock_check = _exact_path(submission, ("disable_ecs_stock_check",), "template submission")
    _bool_true(disable_stock_check, "template submission.disable_ecs_stock_check")
    observed = {
        "schema_version": schema_version,
        "kind": kind,
        "workspace_id": workspace_id,
        "resource_alias": resource_alias,
        "worker_count": worker_count,
        "gpu_count": gpu_count,
        "cpu_count": cpu_count,
        "memory": memory_gi,
        "uid_gid": f"{uid}:{gid}",
        "output_mode": output_mode,
        "autoresume": autoresume,
        "pai_automatic_fault_tolerance": pai_fault_tolerance,
        "require_actual_idle": require_idle,
        "priority": priority,
        "disable_ecs_stock_check": disable_stock_check,
    }
    expected = {
        "schema_version": EXPECTED_TEMPLATE_SCHEMA_VERSION,
        "kind": EXPECTED_TEMPLATE_KIND,
        "workspace_id": EXPECTED_WORKSPACE_ID,
        "resource_alias": EXPECTED_RESOURCE_ALIAS,
        "worker_count": EXPECTED_WORKER_COUNT,
        "gpu_count": EXPECTED_GPU_COUNT,
        "cpu_count": EXPECTED_CPU_COUNT,
        "memory": EXPECTED_MEMORY_GI,
        "uid_gid": EXPECTED_UID_GID,
        "output_mode": "resume",
        "autoresume": True,
        "pai_automatic_fault_tolerance": True,
        "require_actual_idle": True,
        "priority": 9,
        "disable_ecs_stock_check": True,
    }
    if observed != expected:
        raise RuntimeError(f"template schema/contract mismatch: {observed} != {expected}")
    return observed


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
    expected_resource_id: str = EXPECTED_RESOURCE_ID,
    expected_oversold_type: str = EXPECTED_OVERSOLD_TYPE,
    expected_worker_count: int = EXPECTED_WORKER_COUNT,
    expected_gpu_count: int = EXPECTED_GPU_COUNT,
    expected_cpu_count: int = EXPECTED_CPU_COUNT,
    expected_memory_gi: str = EXPECTED_MEMORY_GI,
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
    placement_contract, raw_placement_file = _placement_contract(
        root,
        resolved,
        placement,
        expected_run_id=run_id,
        expected_job_id=job_id,
        expected_resource_id=expected_resource_id,
        expected_oversold_type=expected_oversold_type,
        expected_worker_count=expected_worker_count,
        expected_gpu_count=expected_gpu_count,
        expected_cpu_count=expected_cpu_count,
        expected_memory_gi=expected_memory_gi,
    )
    template_payload = _read_json(template, "continuation template")
    template_contract = _template_contract(template_payload)
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
        template,
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
        "uid_gid": placement_contract["uid_gid"],
        "placement": placement_contract,
        "template_contract": template_contract,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "template_sha256": template_sha,
        "source_template": source_template_binding,
        "files": {
            "resolved": {"path": str(resolved_path), "sha256": sha256_file(resolved_path), "bytes": resolved_path.stat().st_size},
            "placement": {"path": str(placement_path), "sha256": sha256_file(placement_path), "bytes": placement_path.stat().st_size},
            "placement_readback": raw_placement_file,
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
    old_registry_run: Path,
    old_registry_evidence: Path,
    registry_run: Path,
    registry_evidence: Path,
    expected_run_id: str,
    expected_source_commit: str,
    expected_source_tree: str,
    expected_launcher: Path,
    expected_job_id: str | None = None,
    expected_source_manifest: Path | None = None,
    expected_resource_id: str = EXPECTED_RESOURCE_ID,
    expected_oversold_type: str = EXPECTED_OVERSOLD_TYPE,
    expected_worker_count: int = EXPECTED_WORKER_COUNT,
    expected_gpu_count: int = EXPECTED_GPU_COUNT,
    expected_cpu_count: int = EXPECTED_CPU_COUNT,
    expected_memory_gi: str = EXPECTED_MEMORY_GI,
) -> dict[str, Any]:
    terminal = validate_old_producer_terminal(
        old_terminal,
        expected_job_id=old_job_id,
        expected_run_id=old_run_id,
        producer_registry_run=old_registry_run,
        producer_registry_evidence=old_registry_evidence,
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
        expected_resource_id=expected_resource_id,
        expected_oversold_type=expected_oversold_type,
        expected_worker_count=expected_worker_count,
        expected_gpu_count=expected_gpu_count,
        expected_cpu_count=expected_cpu_count,
        expected_memory_gi=expected_memory_gi,
    )
    return {"status": "PASS", "old_producer_terminal": terminal, "continuation_registry": registry}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-terminal", type=Path, required=True)
    parser.add_argument("--old-job-id", required=True)
    parser.add_argument("--old-run-id", required=True)
    parser.add_argument("--old-registry-run", type=Path, required=True)
    parser.add_argument("--old-registry-evidence", type=Path, required=True)
    parser.add_argument("--registry-run", type=Path, required=True)
    parser.add_argument("--registry-evidence", type=Path, required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--expected-launcher", type=Path, required=True)
    parser.add_argument("--expected-job-id", default=None)
    parser.add_argument("--expected-source-manifest", type=Path, default=None)
    parser.add_argument("--expected-resource-id", default=EXPECTED_RESOURCE_ID)
    parser.add_argument("--expected-oversold-type", default=EXPECTED_OVERSOLD_TYPE)
    parser.add_argument("--expected-worker-count", type=int, default=EXPECTED_WORKER_COUNT)
    parser.add_argument("--expected-gpu-count", type=int, default=EXPECTED_GPU_COUNT)
    parser.add_argument("--expected-cpu-count", type=int, default=EXPECTED_CPU_COUNT)
    parser.add_argument("--expected-memory-gi", default=EXPECTED_MEMORY_GI)
    parser.add_argument("--print-job-id", action="store_true")
    args = parser.parse_args()
    result = validate_all(
        old_terminal=args.old_terminal,
        old_job_id=args.old_job_id,
        old_run_id=args.old_run_id,
        old_registry_run=args.old_registry_run,
        old_registry_evidence=args.old_registry_evidence,
        registry_run=args.registry_run,
        registry_evidence=args.registry_evidence,
        expected_run_id=args.expected_run_id,
        expected_source_commit=args.expected_source_commit,
        expected_source_tree=args.expected_source_tree,
        expected_launcher=args.expected_launcher,
        expected_job_id=args.expected_job_id,
        expected_source_manifest=args.expected_source_manifest,
        expected_resource_id=args.expected_resource_id,
        expected_oversold_type=args.expected_oversold_type,
        expected_worker_count=args.expected_worker_count,
        expected_gpu_count=args.expected_gpu_count,
        expected_cpu_count=args.expected_cpu_count,
        expected_memory_gi=args.expected_memory_gi,
    )
    if args.print_job_id:
        print(result["continuation_registry"]["job_id"])
    else:
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
