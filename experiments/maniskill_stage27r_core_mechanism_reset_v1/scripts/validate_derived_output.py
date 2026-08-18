#!/usr/bin/env python3
"""Validate existing derived Stage-2.7R outputs before safe recomputation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import PROTOCOL_ID

FINAL_STATUSES = {
    "GO_FULL_JOINT",
    "REVISE_SHARED_AXIS_ROUTER",
    "REVISE_VISUAL_ONLY",
    "NO_GO_CORE_MECHANISM",
    "NO_GO_CAUSAL_BACKEND",
}


def validate_payload(path: Path, kind: str) -> dict:
    if not Path(path).is_file():
        raise RuntimeError(f"missing derived output: {path}")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"invalid JSON derived output {path}: {exc}") from exc
    if kind != "marker" and payload.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError(f"derived protocol mismatch for {path}")
    if kind == "statistics":
        if not isinstance(payload.get("statistics"), dict) or not payload.get("budgets"):
            raise RuntimeError(f"statistics schema incomplete: {path}")
        if int(payload.get("bootstrap_replicates", -1)) != 10000:
            raise RuntimeError(f"statistics bootstrap count mismatch: {path}")
    elif kind == "mechanism":
        if payload.get("scope") != "mechanism_reverse_engineering_without_new_idea_generation":
            raise RuntimeError(f"mechanism scope mismatch: {path}")
        if not isinstance(payload.get("task_summary"), dict):
            raise RuntimeError(f"mechanism task summary missing: {path}")
    elif kind == "result":
        status = payload.get("final_status")
        if status not in FINAL_STATUSES:
            raise RuntimeError(f"invalid result vector status: {path}: {status}")
        if payload.get("precedence_applied") is not True:
            raise RuntimeError(f"result vector precedence not applied: {path}")
    elif kind == "official_audit":
        checks = payload.get("checks")
        if not isinstance(checks, dict) or payload.get("checks", {}).get("all_pass") is not True:
            raise RuntimeError(f"official audit is not all_pass: {path}")
        if not isinstance(payload.get("manifest"), list) or not payload["manifest"]:
            raise RuntimeError(f"official audit manifest missing: {path}")
    elif kind == "posthoc_audit":
        if payload.get("status") != "PASS":
            raise RuntimeError(f"posthoc audit status is not PASS: {path}")
        camera = payload.get("camera_evidence", {})
        if camera.get("status") != "PASS" or camera.get("posthoc_evidence_only") is not True:
            raise RuntimeError(f"posthoc camera evidence missing/invalid: {path}")
        camera_checks = payload.get("camera_checks", {})
        if not camera_checks or not all(value.get("pass") is True for value in camera_checks.values()):
            raise RuntimeError(f"posthoc camera checks are not all PASS: {path}")
        for check_name in ("outcome_recompute", "schedule_recompute", "accounting_recompute"):
            if payload.get(check_name, {}).get("pass") is not True:
                raise RuntimeError(f"posthoc {check_name} failed: {path}")
    elif kind == "marker":
        if payload != {"protocol_id": PROTOCOL_ID, "status": "FORMAL_COMPLETE"}:
            raise RuntimeError(f"terminal marker mismatch: {path}")
    elif kind == "lineage":
        if payload.get("status") != "PASS" or payload.get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError(f"lineage manifest status mismatch: {path}")
        if payload.get("lineage_scope") != "process_level_sidecar_for_legacy_rows":
            raise RuntimeError(f"lineage scope missing: {path}")
        if payload.get("rows_embed_checkpoint_or_state_bank_hash") is not False:
            raise RuntimeError(f"lineage limitation was not disclosed: {path}")
        producer = payload.get("producer_source", {})
        verifier = payload.get("posthoc_verifier_source", {})
        terminal = payload.get("old_producer_terminal", {})
        continuation = payload.get("continuation_registry", {})
        if terminal.get("status") != "PASS" or terminal.get("no_overlap") is not True:
            raise RuntimeError(f"old producer terminal/no-overlap evidence missing: {path}")
        if continuation.get("status") != "PASS" or not continuation.get("job_id") or continuation.get("job_id") in {"unknown", "UNKNOWN"}:
            raise RuntimeError(f"continuation registry exact JobId evidence missing: {path}")
        if continuation.get("use_oversold_resource") is not True or continuation.get("uid_gid") != "2254:2254":
            raise RuntimeError(f"continuation placement identity/oversold evidence missing: {path}")
        files = continuation.get("files", {})
        for name in ("resolved", "placement", "payload", "template", "source_manifest"):
            if not isinstance(files.get(name), dict) or not files[name].get("sha256"):
                raise RuntimeError(f"continuation registry {name} hash missing: {path}")
        if not continuation.get("source_commit") or not continuation.get("source_tree") or not continuation.get("launcher_sha256"):
            raise RuntimeError(f"continuation source hashes missing: {path}")
        if producer.get("role") != "legacy_oracle_producer" or not producer.get("registry_evidence"):
            raise RuntimeError(f"legacy producer provenance missing: {path}")
        if verifier.get("role") != "clean_posthoc_verifier_and_continuation_source":
            raise RuntimeError(f"posthoc verifier provenance missing: {path}")
        if not payload.get("oracle_input_snapshot", {}).get("sha256"):
            raise RuntimeError(f"oracle input snapshot missing: {path}")
        if not isinstance(payload.get("oracle_files"), list) or len(payload["oracle_files"]) < 6:
            raise RuntimeError(f"lineage manifest has no shards: {path}")
        for entry in payload["oracle_files"]:
            if not all(field in entry for field in ("task", "model_seed", "bank", "oracle", "state_bank", "selected_checkpoint", "shard_validation", "producer_origin", "producer_source", "posthoc_verifier_source", "continuation_record")):
                raise RuntimeError(f"lineage shard entry incomplete: {path}")
            if entry["shard_validation"].get("status") != "PASS":
                raise RuntimeError(f"lineage shard validation failed: {path}")
            if entry.get("producer_origin") == "created_after_clean_continuation_start":
                record = entry.get("continuation_record") or {}
                if record.get("semantic_validation", {}).get("status") != "PASS":
                    raise RuntimeError(f"continuation shard immutable semantic record missing: {path}")
                provenance = record.get("continuation_provenance", {})
                if provenance.get("run_id") != continuation.get("run_id") or provenance.get("job_id") != continuation.get("job_id"):
                    raise RuntimeError(f"continuation shard provenance run/JobId mismatch: {path}")
                if provenance.get("source_commit") != continuation.get("source_commit") or provenance.get("source_tree") != continuation.get("source_tree"):
                    raise RuntimeError(f"continuation shard provenance source mismatch: {path}")
                if provenance.get("old_producer_terminal", {}).get("evidence", {}).get("sha256") != terminal.get("evidence", {}).get("sha256"):
                    raise RuntimeError(f"continuation shard provenance old terminal mismatch: {path}")
        records = payload.get("oracle_input_continuation_records", {})
        if not isinstance(records.get("records"), list) or int(records.get("record_count", -1)) != len(records["records"]):
            raise RuntimeError(f"oracle continuation record summary missing: {path}")
    elif kind == "oracle_validation":
        if payload.get("status") != "PASS" or int(payload.get("validated_shards", 0)) != 6:
            raise RuntimeError(f"oracle collection validation is not complete: {path}")
        if int(payload.get("row_count", -1)) != 97920:
            raise RuntimeError(f"oracle collection row count mismatch: {path}")
    else:
        raise ValueError(f"unknown derived output kind: {kind}")
    return {
        "status": "PASS",
        "kind": kind,
        "path": str(path),
        "protocol_id": payload.get("protocol_id", PROTOCOL_ID),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=("statistics", "mechanism", "result", "official_audit", "posthoc_audit", "marker", "lineage", "oracle_validation"),
        required=True,
    )
    args = parser.parse_args()
    print(json.dumps(validate_payload(args.path, args.kind), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
