#!/usr/bin/env python3
"""Race-safe exclusive installation of the Stage-2.7R terminal marker."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from common import PROTOCOL_ID
from validate_derived_output import validate_payload

MARKER = {"protocol_id": PROTOCOL_ID, "status": "FORMAL_COMPLETE"}


def validate_prerequisites(
    *,
    official_audit: Path,
    posthoc_audit: Path,
    result_vector: Path,
    oracle_validation: Path,
) -> dict:
    official = validate_payload(official_audit, "official_audit")
    posthoc = validate_payload(posthoc_audit, "posthoc_audit")
    result = validate_payload(result_vector, "result")
    oracle = json.loads(oracle_validation.read_text(encoding="utf-8"))
    if oracle.get("protocol_id") != PROTOCOL_ID or oracle.get("status") != "PASS":
        raise RuntimeError("oracle validation is not PASS")
    if int(oracle.get("validated_shards", 0)) != 6:
        raise RuntimeError(f"oracle shard count is not six: {oracle.get('validated_shards')}")
    if int(oracle.get("row_count", -1)) != 97920:
        raise RuntimeError(f"oracle row count is not 97920: {oracle.get('row_count')}")
    return {"official_audit": official, "posthoc_audit": posthoc, "result_vector": result, "oracle_validation": oracle}


def install_or_validate(marker: Path, prerequisites: dict) -> str:
    marker = Path(marker)
    payload = json.dumps(MARKER, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.parent / f".{marker.name}.complete-{os.getpid()}-{os.urandom(8).hex()}.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, marker)
        except FileExistsError:
            existing = marker.read_bytes()
            if existing != payload:
                raise RuntimeError("existing FORMAL_COMPLETE marker mismatch")
            # Existing marker is not enough by itself: every prerequisite is
            # revalidated on a resumed launcher before this return.
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--official-audit", type=Path, required=True)
    parser.add_argument("--posthoc-audit", type=Path, required=True)
    parser.add_argument("--result-vector", type=Path, required=True)
    parser.add_argument("--oracle-validation", type=Path, required=True)
    args = parser.parse_args()
    prerequisites = validate_prerequisites(
        official_audit=args.official_audit,
        posthoc_audit=args.posthoc_audit,
        result_vector=args.result_vector,
        oracle_validation=args.oracle_validation,
    )
    print(json.dumps({"status": install_or_validate(args.marker, prerequisites), "marker": str(args.marker)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
