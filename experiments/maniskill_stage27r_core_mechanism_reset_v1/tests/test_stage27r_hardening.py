from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import PROTOCOL_ID, sha256_file  # noqa: E402
from record_oracle_inputs import exclusive_json, snapshot, validate_existing  # noqa: E402
from resume_derived_output import run_or_validate  # noqa: E402
from validate_continuation_evidence import (  # noqa: E402
    validate_continuation_registry,
    validate_old_producer_terminal,
)


def _registry(root: Path, launcher: Path) -> tuple[Path, Path]:
    run = root / "registry"
    (run / "payload").mkdir(parents=True)
    (run / "payload" / "payload.sh").write_text("payload\n", encoding="utf-8")
    template = run / "template.json"
    template.write_text("{}\n", encoding="utf-8")
    source = run / "source-manifest.json"
    payload_hash = sha256_file(run / "payload" / "payload.sh")
    template_hash = sha256_file(template)
    launcher_hash = sha256_file(launcher)
    source.write_text(
        json.dumps(
            {
                "source_commit": "c" * 40,
                "source_tree": "t" * 40,
                "launcher_sha256": launcher_hash,
                "payload_sha256": payload_hash,
                "template_sha256": template_hash,
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
                    "uid": 2254,
                    "gid": 2254,
                },
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
                "recorded_by_uid": 2254,
                "recorded_by_gid": 2254,
            }
        ),
        encoding="utf-8",
    )
    return run, source


def test_external_terminal_and_registry_are_fail_closed(tmp_path: Path) -> None:
    terminal = tmp_path / "OLD_PRODUCER_TERMINAL.json"
    terminal.write_text(
        json.dumps(
            {
                "old_job_id": "old-job",
                "run_id": "old-run",
                "status": "Running",
                "terminal": True,
                "no_overlap": True,
                "observed_at": "2026-08-18T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="not terminal"):
        validate_old_producer_terminal(terminal, expected_job_id="old-job", expected_run_id="old-run")
    terminal.write_text(
        json.dumps(
            {
                "old_job_id": "old-job",
                "run_id": "old-run",
                "status": "Succeeded",
                "terminal": True,
                "no_overlap": True,
                "observed_at": "2026-08-18T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
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
    with pytest.raises(RuntimeError, match="template SHA-256 mismatch"):
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
    template.write_text("{}\n", encoding="utf-8")

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


def test_external_source_template_requires_path_and_source_hash_binding(tmp_path: Path) -> None:
    run, source = _registry(tmp_path, ROOT / "launchers/run_stage27r_formal_pai.sh")
    external = tmp_path / "external-template.json"
    external.write_text("external-template\n", encoding="utf-8")
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
