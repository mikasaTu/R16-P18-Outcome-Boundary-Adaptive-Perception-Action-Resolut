from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_SCRIPT = EXPERIMENT_ROOT / "scripts" / "archive_preempted_semantics_partial.py"
PROTOCOL_ID = "R16-P18-MS4-STAGE25-REPAIR-ORACLE-V1"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_preempted_partial_is_archived_without_mutation(tmp_path: Path) -> None:
    root = tmp_path / "formal-run"
    root.mkdir()
    write_json(root / "FORMAL_RUN_MANIFEST.json", {
        "protocol_id": PROTOCOL_ID,
        "run_id": "fixture-run",
    })
    seed_bank = tmp_path / "confirmatory.json"
    write_json(seed_bank, {"tasks": {"StackCube-v1": [101, 102]}})
    seed_bank_sha256 = hashlib.sha256(seed_bank.read_bytes()).hexdigest()

    output = root / "success_semantics" / "fixed_horizon" / "seed_16018"
    record = {
        "protocol_id": PROTOCOL_ID,
        "task_id": "StackCube-v1",
        "mode": "fixed_horizon",
        "model_seed": 16018,
        "episode_seed": 101,
        "seed_bank_sha256": seed_bank_sha256,
    }
    write_json(output / "episodes.jsonl", record)
    write_json(output / "FIRST_REAL_ROLLOUT.json", {
        "protocol_id": PROTOCOL_ID,
        "status": "FIRST_REAL_ROLLOUT_BATCH_COMPLETE",
    })
    original_sha256 = hashlib.sha256((output / "episodes.jsonl").read_bytes()).hexdigest()

    subprocess.run([
        sys.executable,
        str(RECOVERY_SCRIPT),
        "--result-root", str(root),
        "--confirmatory-seed-bank", str(seed_bank),
        "--archive-id", "fixture-lease",
        "--pai-job-id", "fixture-job",
        "--expected-total", "2",
        "--expected-batch", "1",
    ], check=True, capture_output=True, text=True)

    assert not (root / "success_semantics").exists()
    archive = root / "recovery" / "preempted_partials" / "fixture-lease"
    archived_jsonl = archive / "success_semantics" / "fixed_horizon" / "seed_16018" / "episodes.jsonl"
    assert hashlib.sha256(archived_jsonl.read_bytes()).hexdigest() == original_sha256
    manifest = json.loads((archive / "RECOVERY_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["scientific_outputs_used"] is False
    assert manifest["partial_count"] == 1
    assert manifest["partials"][0]["episodes_jsonl_sha256"] == original_sha256
