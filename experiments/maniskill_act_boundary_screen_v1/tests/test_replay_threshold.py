from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from replay_rgb_datasets import (  # noqa: E402
    TASK_CONFIGS,
    quarantine_output_family,
    validate_output,
)


def make_replay_pair(root: Path, *, expected: int, saved: int) -> tuple[Path, Path]:
    input_h5 = root / "trajectory.h5"
    output_h5 = root / "trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5"
    with h5py.File(input_h5, "w") as handle:
        handle.create_dataset("identity", data=np.arange(expected, dtype=np.int32))
    input_h5.with_suffix(".json").write_text(
        json.dumps(
            {
                "episodes": [
                    {"episode_id": index, "episode_seed": index}
                    for index in range(expected)
                ]
            }
        ),
        encoding="utf-8",
    )
    with h5py.File(output_h5, "w") as handle:
        for index in range(saved):
            trajectory = handle.create_group(f"traj_{index}")
            trajectory.create_dataset(
                "actions", data=np.array([[index]], dtype=np.float32)
            )
            obs = trajectory.create_group("obs")
            camera = obs.create_group("sensor_param")
            camera.create_dataset("intrinsic", data=np.eye(3, dtype=np.float32))
            sensor_data = obs.create_group("sensor_data")
            base_camera = sensor_data.create_group("base_camera")
            base_camera.create_dataset(
                "rgb", data=np.full((1, 2, 2, 3), index, dtype=np.uint8)
            )
    output_h5.with_suffix(".json").write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "episode_id": index,
                        "episode_seed": index,
                        "success": True,
                    }
                    for index in range(saved)
                ],
                "env_info": {
                    "env_kwargs": {
                        "control_mode": "pd_ee_delta_pose",
                        "obs_mode": "rgb",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return input_h5, output_h5


def test_preregistered_95_percent_replay_gate_passes_and_records_missing(
    tmp_path: Path,
) -> None:
    input_h5, output_h5 = make_replay_pair(tmp_path, expected=20, saved=19)
    record = validate_output(
        "PushT-v1",
        "train",
        input_h5,
        output_h5,
        20,
        TASK_CONFIGS["PushT-v1"],
    )
    assert record["status"] == "PASS"
    assert record["minimum_episodes_required"] == 19
    assert record["replay_success_rate"] == 0.95
    assert record["missing_episode_seeds"] == [19]


def test_replay_below_preregistered_gate_fails(tmp_path: Path) -> None:
    input_h5, output_h5 = make_replay_pair(tmp_path, expected=20, saved=18)
    with pytest.raises(RuntimeError, match="preregistered minimum is 19"):
        validate_output(
            "PushT-v1",
            "train",
            input_h5,
            output_h5,
            20,
            TASK_CONFIGS["PushT-v1"],
        )


def test_incomplete_multiprocessing_shards_are_quarantined(tmp_path: Path) -> None:
    target = tmp_path / "trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5"
    shard_h5 = tmp_path / f"{target.stem}.0.h5"
    shard_json = tmp_path / f"{target.stem}.0.json"
    shard_h5.write_bytes(b"partial")
    shard_json.write_text("{}", encoding="utf-8")

    quarantine_output_family(target)

    assert not shard_h5.exists()
    assert not shard_json.exists()
    assert len(list(tmp_path.glob(f"{shard_h5.name}.incomplete-*"))) == 1
    assert len(list(tmp_path.glob(f"{shard_json.name}.incomplete-*"))) == 1
