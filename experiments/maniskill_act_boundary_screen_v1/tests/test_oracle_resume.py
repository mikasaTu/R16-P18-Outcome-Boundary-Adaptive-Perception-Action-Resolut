from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from protocol_common import PROTOCOL_ID, sha256_file  # noqa: E402
from run_oracle_matrix import oracle_complete, state_bank_terminal  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_state_bank_resume_is_bound_to_source_builder_and_contract(
    tmp_path: Path,
) -> None:
    task_id = "PushCube-v1"
    test_h5 = tmp_path / "trajectory.rgb.test.h5"
    test_h5.write_bytes(b"test states")
    test_h5.with_suffix(".json").write_bytes(b"test metadata")
    bank_h5 = tmp_path / "state_bank.h5"
    bank_h5.write_bytes(b"state bank")
    manifest_path = tmp_path / "state_bank_manifest.json"
    write_json(
        manifest_path,
        {
            "protocol_id": PROTOCOL_ID,
            "status": "STATE_BANK_COMPLETE",
            "task_id": task_id,
            "source_test_h5": str(test_h5),
            "source_test_h5_sha256": sha256_file(test_h5),
            "source_test_json_sha256": sha256_file(test_h5.with_suffix(".json")),
            "builder_sha256": sha256_file(SCRIPT_DIR / "build_state_bank.py"),
            "phase_contract_sha256": sha256_file(
                EXPERIMENT_ROOT / "state_bank" / "phase_contract.json"
            ),
            "state_bank_h5": str(bank_h5),
            "state_bank_h5_sha256": sha256_file(bank_h5),
            "state_count": 64,
            "phase_counts": {
                "free_space": 16,
                "pre_contact_or_pre_grasp": 16,
                "contact_insertion_or_placement": 16,
                "near_completion": 16,
            },
            "restoration_repeats": 3,
            "rollout_steps": 4,
        },
    )

    assert state_bank_terminal(manifest_path, task_id, test_h5)
    test_h5.write_bytes(b"changed test states")
    assert not state_bank_terminal(manifest_path, task_id, test_h5)


def test_oracle_resume_is_bound_to_all_scientific_inputs(tmp_path: Path) -> None:
    task_id = "PushCube-v1"
    model_seed = 16018
    state_h5 = tmp_path / "state_bank.h5"
    state_h5.write_bytes(b"bank")
    state_manifest_path = tmp_path / "state_bank_manifest.json"
    write_json(
        state_manifest_path,
        {
            "state_bank_h5": str(state_h5),
            "state_bank_h5_sha256": sha256_file(state_h5),
        },
    )
    train_h5 = tmp_path / "trajectory.rgb.train.h5"
    train_h5.write_bytes(b"training actions")
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints" / "step_000000001"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    selection = {
        "test_metrics_used": False,
        "selected": {
            "path": str(checkpoint_dir),
            "step": 1,
            "checkpoint_sha256": sha256_file(checkpoint),
        },
    }
    write_json(run_dir / "checkpoint_selection.json", selection)
    bindings = {
        "oracle_evaluator_sha256": sha256_file(
            SCRIPT_DIR / "evaluate_oracle_atlas.py"
        ),
        "state_bank_manifest_sha256": sha256_file(state_manifest_path),
        "state_bank_h5_sha256": sha256_file(state_h5),
        "train_h5_sha256": sha256_file(train_h5),
        "selected_checkpoint_step": 1,
        "selected_checkpoint_sha256": sha256_file(checkpoint),
        "selected_checkpoint_path": str(checkpoint),
    }
    surface_files = []
    for index in range(64):
        surface = tmp_path / f"state-{index:02d}.json"
        surface.write_bytes(f"state {index}".encode("ascii"))
        surface_files.append(
            {
                "path": str(surface),
                "sha256": sha256_file(surface),
                "bytes": surface.stat().st_size,
            }
        )
    summary_path = tmp_path / "summary.json"
    write_json(
        summary_path,
        {
            "protocol_id": PROTOCOL_ID,
            "status": "ORACLE_ATLAS_COMPLETE",
            "task_id": task_id,
            "model_seed": model_seed,
            "states": 64,
            "source_bindings": bindings,
            "state_bank_manifest_sha256": bindings[
                "state_bank_manifest_sha256"
            ],
            "train_h5_sha256": bindings["train_h5_sha256"],
            "selected_checkpoint_sha256": bindings[
                "selected_checkpoint_sha256"
            ],
            "implementation_contract_sha256": sha256_file(
                EXPERIMENT_ROOT
                / "action_atlas"
                / "oracle_implementation_contract.json"
            ),
            "surface_files": surface_files,
        },
    )

    assert oracle_complete(
        summary_path,
        task_id,
        model_seed,
        state_manifest_path,
        train_h5,
        run_dir,
    )
    checkpoint.write_bytes(b"changed checkpoint")
    assert not oracle_complete(
        summary_path,
        task_id,
        model_seed,
        state_manifest_path,
        train_h5,
        run_dir,
    )
