from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import PROTOCOL_ID, sha256_file  # noqa: E402
from record_oracle_inputs import exclusive_json, snapshot, validate_existing  # noqa: E402
from resume_derived_output import run_or_validate  # noqa: E402
from validate_derived_output import validate_payload  # noqa: E402
from validate_continuation_evidence import (  # noqa: E402
    validate_continuation_registry,
    validate_old_producer_terminal,
)


def _registry(root: Path, launcher: Path) -> tuple[Path, Path]:
    run = root / "registry"
    (run / "payload").mkdir(parents=True)
    (run / "payload" / "payload.sh").write_text("payload\n", encoding="utf-8")
    template = run / "template.json"
    template_payload = {
        "schema_version": 2,
        "kind": "pytorchjob",
        "workspace_id": 179169,
        "resource_alias": "idle-a800-stablevla-native5-8gpu",
        "worker_count": 1,
        "gpu_count": 8,
        "cpu_count": 92,
        "memory": "1600Gi",
        "runtime": {"uid": 2254, "gid": 2254, "output_mode": "resume"},
        "fault": {"autoresume": True},
        "evidence": {"require_actual_idle": True},
        "submission": {"priority": 9, "disable_ecs_stock_check": True},
    }
    template.write_text(json.dumps(template_payload, sort_keys=True) + "\n", encoding="utf-8")
    source_template = root / "source-template.json"
    source_template.write_bytes(template.read_bytes())
    source = run / "source-manifest.json"
    payload_hash = sha256_file(run / "payload" / "payload.sh")
    template_hash = sha256_file(template)
    source_template_hash = sha256_file(source_template)
    launcher_hash = sha256_file(launcher)
    source.write_text(
        json.dumps(
            {
                "source_commit": "c" * 40,
                "source_tree": "t" * 40,
                "launcher_sha256": launcher_hash,
                "payload_sha256": payload_hash,
                "template_sha256": template_hash,
                "source_template": str(source_template),
                "source_template_sha256": source_template_hash,
            }
        ),
        encoding="utf-8",
    )
    (run / "resolved.json").write_text(
        json.dumps(
            {
                "run_id": "continuation-run",
                "job_id": "new-job",
                "evidence": {
                    "source_commit": "c" * 40,
                    "source_tree": "t" * 40,
                    "launcher_sha256": launcher_hash,
                    "payload_sha256": payload_hash,
                    "template_sha256": template_hash,
                    "source_template": str(source_template),
                    "source_template_sha256": source_template_hash,
                    "resource_id": "quotaewyznuc7b9l",
                    "oversold_type": "AcceptQuotaOverSold",
                    "use_oversold_resource": True,
                    "worker_count": 1,
                    "gpu_count": 8,
                    "cpu_count": 92,
                    "memory": "1600Gi",
                    "uid": 2254,
                    "gid": 2254,
                },
            }
        ),
        encoding="utf-8",
    )
    raw_placement = root / "raw-placement.json"
    raw_placement.write_text(
        json.dumps(
            {
                "run_id": "continuation-run",
                "job_id": "new-job",
                "resource_id": "quotaewyznuc7b9l",
                "oversold_type": "AcceptQuotaOverSold",
                "use_oversold_resource": True,
                "worker_count": 1,
                "gpu_count": 8,
                "cpu_count": 92,
                "memory": "1600Gi",
                "recorded_by_uid": 2254,
                "recorded_by_gid": 2254,
            }
        ),
        encoding="utf-8",
    )
    (run / "placement-evidence.external.json").write_text(
        json.dumps(
            {
                "run_id": "continuation-run",
                "job_id": "new-job",
                "complete": True,
                "use_oversold_resource": True,
                "resource_id": "quotaewyznuc7b9l",
                "oversold_type": "AcceptQuotaOverSold",
                "worker_count": 1,
                "gpu_count": 8,
                "cpu_count": 92,
                "memory": "1600Gi",
                "recorded_by_uid": 2254,
                "recorded_by_gid": 2254,
                "raw_placement_readback": {
                    "sealed": True,
                    "path": str(raw_placement),
                    "sha256": sha256_file(raw_placement),
                    "bytes": raw_placement.stat().st_size,
                },
            }
        ),
        encoding="utf-8",
    )
    return run, source


def _old_registry(root: Path) -> tuple[Path, Path, Path]:
    run = root / "old-registry"
    run.mkdir()
    resolved = run / "resolved.json"
    placement = run / "placement-evidence.external.json"
    resolved.write_text(json.dumps({"run_id": "old-run", "job_id": "old-job"}), encoding="utf-8")
    placement.write_text(json.dumps({"run_id": "old-run", "job_id": "old-job"}), encoding="utf-8")
    return run, resolved, placement


def _terminal(root: Path, registry_run: Path, resolved: Path, placement: Path, *, status: str = "Succeeded", observed_at: str = "2026-08-18T00:00:00Z") -> Path:
    getjob = root / "raw-getjob.json"
    getjob.write_text(json.dumps({"JobId": "old-job", "status": status}), encoding="utf-8")
    binding = lambda path: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    terminal = root / "OLD_PRODUCER_TERMINAL.json"
    terminal.write_text(
        json.dumps(
            {
                "old_job_id": "old-job",
                "run_id": "old-run",
                "status": status,
                "terminal": status != "Running",
                "no_overlap": status != "Running",
                "observed_at": observed_at,
                "getjob": binding(getjob),
                "producer_registry": {
                    "run_id": "old-run",
                    "job_id": "old-job",
                    "resolved": binding(resolved),
                    "placement": binding(placement),
                },
            }
        ),
        encoding="utf-8",
    )
    return terminal


def test_external_terminal_and_registry_are_fail_closed(tmp_path: Path) -> None:
    old_run, old_resolved, old_placement = _old_registry(tmp_path)
    terminal = _terminal(tmp_path, old_run, old_resolved, old_placement, status="Running")
    with pytest.raises(RuntimeError, match="not terminal"):
        validate_old_producer_terminal(
            terminal,
            expected_job_id="old-job",
            expected_run_id="old-run",
            producer_registry_run=old_run,
            producer_registry_evidence=old_resolved,
        )
    terminal = _terminal(tmp_path, old_run, old_resolved, old_placement)
    validated_terminal = validate_old_producer_terminal(
        terminal,
        expected_job_id="old-job",
        expected_run_id="old-run",
        producer_registry_run=old_run,
        producer_registry_evidence=old_resolved,
    )
    assert validated_terminal["raw_getjob"]["sha256"]
    run, source = _registry(tmp_path, ROOT / "launchers/run_stage27r_formal_pai.sh")
    template = run / "template.json"
    result = validate_continuation_registry(
        run,
        registry_evidence=run / "resolved.json",
        expected_run_id="continuation-run",
        expected_source_commit="c" * 40,
        expected_source_tree="t" * 40,
        expected_launcher=ROOT / "launchers/run_stage27r_formal_pai.sh",
        expected_job_id="new-job",
        expected_source_manifest=source,
    )
    assert result["uid_gid"] == "2254:2254"
    template.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="template (SHA-256 mismatch|schema)"):
        validate_continuation_registry(
            run,
            registry_evidence=run / "resolved.json",
            expected_run_id="continuation-run",
            expected_source_commit="c" * 40,
            expected_source_tree="t" * 40,
            expected_launcher=ROOT / "launchers/run_stage27r_formal_pai.sh",
            expected_job_id="new-job",
            expected_source_manifest=source,
        )
    template.write_bytes((run.parent / "source-template.json").read_bytes())

    source_payload = json.loads(source.read_text(encoding="utf-8"))
    source_payload.pop("template_sha256")
    source.write_text(json.dumps(source_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source manifest template SHA-256"):
        validate_continuation_registry(
            run,
            registry_evidence=run / "resolved.json",
            expected_run_id="continuation-run",
            expected_source_commit="c" * 40,
            expected_source_tree="t" * 40,
            expected_launcher=ROOT / "launchers/run_stage27r_formal_pai.sh",
            expected_job_id="new-job",
            expected_source_manifest=source,
        )
    source_payload["template_sha256"] = sha256_file(template)
    source.write_text(json.dumps(source_payload), encoding="utf-8")
    resolved_payload = json.loads((run / "resolved.json").read_text(encoding="utf-8"))
    resolved_payload["evidence"].pop("template_sha256")
    (run / "resolved.json").write_text(json.dumps(resolved_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="resolved/request template SHA-256"):
        validate_continuation_registry(
            run,
            registry_evidence=run / "resolved.json",
            expected_run_id="continuation-run",
            expected_source_commit="c" * 40,
            expected_source_tree="t" * 40,
            expected_launcher=ROOT / "launchers/run_stage27r_formal_pai.sh",
            expected_job_id="new-job",
            expected_source_manifest=source,
        )
    source.unlink()
    with pytest.raises(RuntimeError, match="source manifest"):
        validate_continuation_registry(
            run,
            registry_evidence=run / "resolved.json",
            expected_run_id="continuation-run",
            expected_source_commit="c" * 40,
            expected_source_tree="t" * 40,
            expected_launcher=ROOT / "launchers/run_stage27r_formal_pai.sh",
            expected_job_id="new-job",
        )


def test_placement_resource_id_and_raw_terminal_evidence_are_exact(tmp_path: Path) -> None:
    run, source = _registry(tmp_path, ROOT / "launchers/run_stage27r_formal_pai.sh")
    resolved = run / "resolved.json"
    resolved_payload = json.loads(resolved.read_text(encoding="utf-8"))
    resolved_payload["evidence"]["resource_id"] = "wrong-resource"
    resolved.write_text(json.dumps(resolved_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="resource_id (is inconsistent|mismatch)"):
        validate_continuation_registry(
            run,
            registry_evidence=resolved,
            expected_run_id="continuation-run",
            expected_source_commit="c" * 40,
            expected_source_tree="t" * 40,
            expected_launcher=ROOT / "launchers/run_stage27r_formal_pai.sh",
            expected_job_id="new-job",
            expected_source_manifest=source,
        )

    old_run, old_resolved, old_placement = _old_registry(tmp_path)
    terminal = _terminal(tmp_path, old_run, old_resolved, old_placement)
    raw_getjob = tmp_path / "raw-getjob.json"
    raw_getjob.write_text('{"JobId":"old-job","status":"Succeeded","tampered":true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="raw GetJob hash/bytes mismatch"):
        validate_old_producer_terminal(
            terminal,
            expected_job_id="old-job",
            expected_run_id="old-run",
            producer_registry_run=old_run,
            producer_registry_evidence=old_resolved,
        )

    terminal = _terminal(tmp_path, old_run, old_resolved, old_placement)
    terminal_payload = json.loads(terminal.read_text(encoding="utf-8"))
    terminal_payload["status"] = "Failed"
    terminal.write_text(json.dumps(terminal_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="summary/raw status mismatch"):
        validate_old_producer_terminal(
            terminal,
            expected_job_id="old-job",
            expected_run_id="old-run",
            producer_registry_run=old_run,
            producer_registry_evidence=old_resolved,
        )


def test_external_source_template_requires_path_and_source_hash_binding(tmp_path: Path) -> None:
    run, source = _registry(tmp_path, ROOT / "launchers/run_stage27r_formal_pai.sh")
    external = tmp_path / "external-template.json"
    external.write_bytes((run / "template.json").read_bytes())
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    source_payload.update(
        {
            "source_template": str(external),
            "source_template_sha256": sha256_file(external),
        }
    )
    source.write_text(json.dumps(source_payload), encoding="utf-8")
    resolved = run / "resolved.json"
    resolved_payload = json.loads(resolved.read_text(encoding="utf-8"))
    resolved_payload["evidence"].update(
        {
            "source_template": str(external),
            "source_template_sha256": sha256_file(external),
        }
    )
    resolved.write_text(json.dumps(resolved_payload), encoding="utf-8")
    kwargs = {
        "registry_evidence": resolved,
        "expected_run_id": "continuation-run",
        "expected_source_commit": "c" * 40,
        "expected_source_tree": "t" * 40,
        "expected_launcher": ROOT / "launchers/run_stage27r_formal_pai.sh",
        "expected_job_id": "new-job",
        "expected_source_manifest": source,
    }
    assert validate_continuation_registry(run, **kwargs)["source_template"]["external"] is True
    external.write_text("mutated-external-template\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="external source_template SHA-256 mismatch"):
        validate_continuation_registry(run, **kwargs)


def test_oracle_snapshot_adds_immutable_record_and_rejects_mutation(tmp_path: Path) -> None:
    formal = tmp_path
    (formal / "oracle").mkdir()
    tasks, seeds = ["A", "B"], [1, 2, 3]
    for task, seed in (("A", 1), ("A", 2), ("A", 3)):
        (formal / "oracle" / f"{task}-seed{seed}-confirmatory.json").write_text(
            '{"rows":[{"bank":"confirmatory"}],"tile_grid":2}', encoding="utf-8"
        )
    snapshot_path = formal / "ORACLE_INPUT_SNAPSHOT.json"
    exclusive_json(snapshot_path, snapshot(formal, tasks, seeds))
    assert validate_existing(snapshot_path, formal, tasks, seeds)["continuation_records"] == []
    new_shard = formal / "oracle/B-seed1-confirmatory.json"
    new_shard.write_text('{"rows":[{"bank":"confirmatory"}],"tile_grid":2}', encoding="utf-8")
    records = validate_existing(snapshot_path, formal, tasks, seeds)["continuation_records"]
    assert len(records) == 1 and records[0]["semantic_validation"]["status"] == "LIMITATION"
    new_shard.write_text('{"rows":[{"bank":"confirmatory","mutated":true}],"tile_grid":2}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        validate_existing(snapshot_path, formal, tasks, seeds)


def test_snapshot_cli_records_terminal_hash_and_rejects_reverse_time(tmp_path: Path) -> None:
    formal = tmp_path / "formal"
    (formal / "oracle").mkdir(parents=True)
    run, source = _registry(tmp_path, ROOT / "launchers/run_stage27r_formal_pai.sh")
    old_run, old_resolved, old_placement = _old_registry(tmp_path)
    terminal = _terminal(tmp_path, old_run, old_resolved, old_placement)
    snapshot_path = formal / "ORACLE_INPUT_SNAPSHOT.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/record_oracle_inputs.py"),
        "--formal-root", str(formal),
        "--output", str(snapshot_path),
        "--expected-task", "A",
        "--model-seed", "1",
        "--continuation-registry-run", str(run),
        "--continuation-registry-evidence", str(run / "resolved.json"),
        "--continuation-run-id", "continuation-run",
        "--continuation-job-id", "new-job",
        "--continuation-source-commit", "c" * 40,
        "--continuation-source-tree", "t" * 40,
        "--continuation-launcher", str(ROOT / "launchers/run_stage27r_formal_pai.sh"),
        "--continuation-source-manifest", str(source),
        "--old-producer-terminal", str(terminal),
        "--old-producer-job-id", "old-job",
        "--old-producer-run-id", "old-run",
        "--old-producer-registry-run", str(old_run),
        "--old-producer-registry-evidence", str(old_resolved),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    output = json.loads(completed.stdout)
    snapshot_value = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert output["status"] == "PASS"
    assert snapshot_value["created_at"].endswith("Z")
    assert snapshot_value["old_producer_terminal"]["evidence_sha256"] == sha256_file(terminal)
    validated_terminal = validate_old_producer_terminal(
        terminal,
        expected_job_id="old-job",
        expected_run_id="old-run",
        producer_registry_run=old_run,
        producer_registry_evidence=old_resolved,
    )
    snapshot_value["created_at"] = "2026-08-17T00:00:00Z"
    snapshot_path.write_text(json.dumps(snapshot_value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="after oracle snapshot"):
        validate_existing(snapshot_path, formal, ["A"], [1], old_terminal=validated_terminal)


def test_template_empty_object_is_rejected_by_schema_contract(tmp_path: Path) -> None:
    run, source = _registry(tmp_path, ROOT / "launchers/run_stage27r_formal_pai.sh")
    template = run / "template.json"
    template.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="template schema"):
        validate_continuation_registry(
            run,
            registry_evidence=run / "resolved.json",
            expected_run_id="continuation-run",
            expected_source_commit="c" * 40,
            expected_source_tree="t" * 40,
            expected_launcher=ROOT / "launchers/run_stage27r_formal_pai.sh",
            expected_job_id="new-job",
            expected_source_manifest=source,
        )


def test_official_audit_manifest_requires_current_scientific_outputs(tmp_path: Path) -> None:
    required = (
        "statistics.json",
        "MECHANISM_AUDIT.json",
        "RESULT_VECTOR.json",
        "ORACLE_VALIDATION.json",
        "ORACLE_LINEAGE_MANIFEST.json",
    )
    manifest = []
    for name in required:
        path = tmp_path / name
        path.write_text(name + "\n", encoding="utf-8")
        manifest.append({"path": name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    audit = tmp_path / "INDEPENDENT_AUDIT.json"
    audit.write_text(json.dumps({"protocol_id": PROTOCOL_ID, "checks": {"all_pass": True}, "manifest": manifest}), encoding="utf-8")
    assert validate_payload(audit, "official_audit", tmp_path)["status"] == "PASS"
    (tmp_path / "MECHANISM_AUDIT.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest hash/bytes mismatch"):
        validate_payload(audit, "official_audit", tmp_path)
    (tmp_path / "MECHANISM_AUDIT.json").write_text("MECHANISM_AUDIT.json\n", encoding="utf-8")
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["manifest"] = [entry for entry in payload["manifest"] if entry["path"] != "MECHANISM_AUDIT.json"]
    audit.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing required files"):
        validate_payload(audit, "official_audit", tmp_path)


def test_resume_rejects_stale_or_malformed_candidate(tmp_path: Path) -> None:
    target = tmp_path / "statistics.json"
    stale = tmp_path / (".statistics.json.resume-1-" + "0" * 32 + ".json")
    stale.write_text("stale", encoding="utf-8")
    old = time.time() - 7200
    os.utime(stale, (old, old))
    with pytest.raises(RuntimeError, match="stale"):
        run_or_validate(target, [sys.executable, "-c", "open('__OUTPUT__','w').write('x')", "--output", "__OUTPUT__"])


def _race_worker(target: str) -> str:
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; import time; Path('__OUTPUT__').write_text('same'); time.sleep(.05)",
        "--output",
        "__OUTPUT__",
    ]
    return run_or_validate(Path(target), command)


def test_resume_real_process_race_is_no_replace(tmp_path: Path) -> None:
    target = tmp_path / "statistics.json"
    context = multiprocessing.get_context("fork")
    with context.Pool(2) as pool:
        statuses = pool.map(_race_worker, [str(target), str(target)])
    assert set(statuses) <= {"installed_missing", "validated_existing"}
    assert target.read_text(encoding="utf-8") == "same"
