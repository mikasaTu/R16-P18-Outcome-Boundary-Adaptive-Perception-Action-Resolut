from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_launcher_pins_cwd_and_physx_gpu_library() -> None:
    launcher = (ROOT / "pai" / "formal_replay_train_launcher.sh").read_text(
        encoding="utf-8"
    )
    lock = json.loads(
        (ROOT / "locks" / "physx_gpu_library.json").read_text(encoding="utf-8")
    )
    environment = json.loads(
        (ROOT / "environment_lock.json").read_text(encoding="utf-8")
    )

    assert launcher.index('cd "$PROJECT_ROOT"') < launcher.index(
        '"$PYTHON" "$EXPERIMENT_ROOT/scripts/run_formal_matrix.py"'
    )
    assert (
        f"EXPECTED_PHYSX_GPU_LIBRARY_SHA256={lock['library']['sha256']}"
        in launcher
    )
    assert (
        f"EXPECTED_PHYSX_GPU_LIBRARY_BYTES={lock['library']['bytes']}"
        in launcher
    )
    assert environment["simulator"]["physx_gpu_library_sha256"] == lock[
        "library"
    ]["sha256"]
    assert "WANDB_API_KEY" not in launcher
    assert "--track" not in launcher


def test_evaluation_launcher_is_two_gpu_secret_free_and_fixed_seed() -> None:
    launcher = (ROOT / "pai" / "formal_baseline_evaluation_launcher.sh").read_text(
        encoding="utf-8"
    )
    assert 'cd "$PROJECT_ROOT"' in launcher
    assert 'test "${PAI_CANARY_EXPECTED_GPUS:-}" = 2' in launcher
    assert '--gpu-count 2' in launcher
    assert '--num-envs 20' in launcher
    assert 'SEED_MANIFEST="$EXPERIMENT_ROOT/manifests/data_selection_summary.json"' in launcher
    assert "WANDB_API_KEY" not in launcher
    assert "--track" not in launcher


def test_oracle_launcher_is_two_gpu_resumable_and_uses_fixed_cpfs_roots() -> None:
    launcher = (ROOT / "pai" / "formal_stage2_oracle_launcher.sh").read_text(
        encoding="utf-8"
    )
    assert 'cd "$PROJECT_ROOT"' in launcher
    assert 'test "${PAI_CANARY_EXPECTED_GPUS:-}" = 2' in launcher
    assert '--gpu-count 2' in launcher
    assert 'STATE_BANK_ROOT="$DATA_ROOT/state_bank"' in launcher
    assert 'ORACLE_ROOT="$DATA_ROOT/oracle_atlas"' in launcher
    assert 'BASELINE_GATE="$CHECKPOINT_ROOT/baseline_evaluation/baseline_gate.json"' in launcher
    assert 'test -s "$ARTIFACT_DIR/FIRST_REAL_WORK.json"' in launcher
    assert "WANDB_API_KEY" not in launcher
    assert "--track" not in launcher
