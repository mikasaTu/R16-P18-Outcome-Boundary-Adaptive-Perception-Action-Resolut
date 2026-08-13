from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from summarize_baseline import paired_episode_bootstrap  # noqa: E402
from protocol_common import PROTOCOL_ID, sha256_file  # noqa: E402
from run_closed_loop_matrix import valid_completion  # noqa: E402


def test_paired_bootstrap_is_deterministic_and_degenerate_for_constant_data() -> None:
    success = np.ones((3, 100), dtype=np.float64)
    assert paired_episode_bootstrap(success) == [1.0, 1.0]
    mixed = np.zeros((3, 100), dtype=np.float64)
    mixed[:, :50] = 1.0
    assert paired_episode_bootstrap(mixed) == paired_episode_bootstrap(mixed)


def test_completed_evaluation_is_bound_to_checkpoint_seed_bank_and_episodes(
    tmp_path: Path,
) -> None:
    task_id = "PushCube-v1"
    model_seed = 16018
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "output"
    checkpoint_dir = run_dir / "checkpoints" / "step_000000001"
    checkpoint_dir.mkdir(parents=True)
    output_dir.mkdir()
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint fixture")
    selected = {
        "path": str(checkpoint_dir),
        "step": 1,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_loss": 0.5,
    }
    selection_path = run_dir / "checkpoint_selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "selection_metric": "validation_imitation_loss",
                "test_metrics_used": False,
                "selected": selected,
            }
        ),
        encoding="utf-8",
    )
    episode_seeds = list(range(1000, 1100))
    seed_manifest = tmp_path / "seed_manifest.json"
    seed_manifest.write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "formal_tasks": {
                    task_id: {"closed_loop_test_seeds": episode_seeds}
                },
            }
        ),
        encoding="utf-8",
    )
    episodes_path = output_dir / "episodes.jsonl"
    episodes_path.write_text(
        "".join(
            json.dumps({"episode_seed": seed, "model_seed": model_seed}) + "\n"
            for seed in episode_seeds
        ),
        encoding="utf-8",
    )
    bindings = {
        "evaluator_sha256": sha256_file(
            SCRIPT_DIR / "evaluate_official_act_protocol.py"
        ),
        "seed_manifest_sha256": sha256_file(seed_manifest),
        "checkpoint_selection_sha256": sha256_file(selection_path),
        "selected_checkpoint_step": 1,
        "selected_checkpoint_sha256": sha256_file(checkpoint_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "EVALUATION_COMPLETE",
                "protocol_id": PROTOCOL_ID,
                "task_id": task_id,
                "model_seed": model_seed,
                "episodes": 100,
                "test_metrics_used_for_selection": False,
                "selected_checkpoint": selected,
                "source_bindings": bindings,
                "episodes_jsonl_sha256": sha256_file(episodes_path),
            }
        ),
        encoding="utf-8",
    )

    assert valid_completion(
        output_dir, run_dir, seed_manifest, task_id, model_seed
    )

    episodes_path.write_text(
        episodes_path.read_text(encoding="utf-8").replace("1000", "9999", 1),
        encoding="utf-8",
    )
    assert not valid_completion(
        output_dir, run_dir, seed_manifest, task_id, model_seed
    )
