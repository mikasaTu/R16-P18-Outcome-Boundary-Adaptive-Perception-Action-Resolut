#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import h5py

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from protocol_common import (  # noqa: E402
    PROTOCOL_ID,
    SPLIT_COUNTS,
    atomic_write_text,
    canonical_json,
    sha256_file,
    sha256_hdf5_group,
)


TASK_CONFIGS = {
    "PickCube-v1": {
        "control_mode": "pd_ee_delta_pos",
        "sim_backend": "physx_cpu",
        "state_flag": "--use-first-env-state",
        "cpu_processes": 2,
    },
    "PegInsertionSide-v1": {
        "control_mode": "pd_ee_delta_pose",
        "sim_backend": "physx_cpu",
        "state_flag": "--use-first-env-state",
        "cpu_processes": 4,
    },
    "PushT-v1": {
        "control_mode": "pd_ee_delta_pose",
        "sim_backend": "physx_cuda",
        "state_flag": "--use-env-states",
        "gpu_num_envs": 1024,
    },
    "StackCube-v1": {
        "control_mode": "pd_ee_delta_pos",
        "sim_backend": "physx_cpu",
        "state_flag": "--use-first-env-state",
        "cpu_processes": 4,
    },
    "PushCube-v1": {
        "control_mode": "pd_ee_delta_pos",
        "sim_backend": "physx_cpu",
        "state_flag": "--use-first-env-state",
        "cpu_processes": 4,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", choices=sorted(TASK_CONFIGS), required=True)
    parser.add_argument("--selected-raw-root", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--max-retry", type=int, default=3)
    parser.add_argument("--smoke-count", type=int)
    return parser.parse_args()


def output_path(input_h5: Path, config: dict[str, Any]) -> Path:
    return input_h5.with_name(
        f"trajectory.rgb.{config['control_mode']}.{config['sim_backend']}.h5"
    )


def marker_path(output_h5: Path) -> Path:
    return output_h5.with_name(output_h5.stem + ".COMPLETE.json")


def atomic_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def quarantine_existing(path: Path) -> None:
    if not path.exists():
        return
    timestamp = int(time.time())
    target = path.with_name(f"{path.name}.incomplete-{timestamp}-{os.getpid()}")
    os.replace(path, target)


def observation_inventory(group: h5py.Group) -> dict[str, Any]:
    datasets: dict[str, Any] = {}

    def visitor(name: str, value: h5py.Group | h5py.Dataset) -> None:
        if isinstance(value, h5py.Dataset):
            datasets[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}

    group.visititems(visitor)
    rgb_paths = [name for name in datasets if name.endswith("/rgb")]
    if not rgb_paths:
        raise RuntimeError(f"no RGB observation dataset under {group.name}")
    return {"datasets": datasets, "rgb_paths": sorted(rgb_paths)}


def validate_output(
    task_id: str,
    split: str,
    input_h5: Path,
    output_h5: Path,
    expected_count: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    output_json = output_h5.with_suffix(".json")
    if not output_h5.is_file() or not output_json.is_file():
        raise FileNotFoundError(f"replay output missing: {output_h5}")
    metadata = json.loads(output_json.read_text(encoding="utf-8"))
    episodes = sorted(metadata["episodes"], key=lambda item: int(item["episode_id"]))
    if len(episodes) != expected_count:
        raise RuntimeError(
            f"{task_id}/{split}: replay saved {len(episodes)} of {expected_count}; exact split required"
        )
    if any(not bool(episode.get("success", False)) for episode in episodes):
        raise RuntimeError(f"{task_id}/{split}: replay output contains a failed trajectory")
    with h5py.File(output_h5, "r") as handle:
        expected_groups = {f"traj_{index}" for index in range(expected_count)}
        if set(handle) != expected_groups:
            raise RuntimeError(f"{task_id}/{split}: output trajectory groups are not contiguous")
        inventory = observation_inventory(handle["traj_0/obs"])
        trajectory_hashes = [
            sha256_hdf5_group(handle[f"traj_{index}"]) for index in range(expected_count)
        ]
        if len(set(trajectory_hashes)) != expected_count:
            raise RuntimeError(f"{task_id}/{split}: replayed trajectories are not unique")
        camera_group = handle["traj_0/obs/sensor_param"]
        camera_sha = sha256_hdf5_group(camera_group)
    input_metadata = json.loads(input_h5.with_suffix(".json").read_text(encoding="utf-8"))
    input_seeds = [int(item["episode_seed"]) for item in input_metadata["episodes"]]
    output_seeds = [int(item["episode_seed"]) for item in episodes]
    if set(input_seeds[:expected_count]) != set(output_seeds):
        raise RuntimeError(f"{task_id}/{split}: episode identities changed during replay")
    env_kwargs = metadata["env_info"]["env_kwargs"]
    if env_kwargs["control_mode"] != config["control_mode"]:
        raise RuntimeError(f"{task_id}/{split}: control mode mismatch")
    if env_kwargs["obs_mode"] != "rgb":
        raise RuntimeError(f"{task_id}/{split}: observation mode mismatch")
    return {
        "protocol_id": PROTOCOL_ID,
        "task_id": task_id,
        "split": split,
        "status": "PASS",
        "input_h5": str(input_h5),
        "input_h5_sha256": sha256_file(input_h5),
        "input_json_sha256": sha256_file(input_h5.with_suffix(".json")),
        "output_h5": str(output_h5),
        "output_h5_sha256": sha256_file(output_h5),
        "output_json": str(output_json),
        "output_json_sha256": sha256_file(output_json),
        "episodes_attempted": expected_count,
        "episodes_saved_successful": len(episodes),
        "replay_success_rate": len(episodes) / expected_count,
        "unique_episode_seeds": len(set(output_seeds)),
        "unique_replayed_trajectory_hashes": len(set(trajectory_hashes)),
        "camera_parameter_sha256": camera_sha,
        "camera_inventory_sha256": hashlib.sha256(canonical_json(inventory)).hexdigest(),
        "camera_inventory": inventory,
        "control_mode": config["control_mode"],
        "sim_backend": config["sim_backend"],
        "state_replay_flag": config["state_flag"],
    }


def existing_complete(
    task_id: str,
    split: str,
    input_h5: Path,
    output_h5: Path,
    expected_count: int,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    marker = marker_path(output_h5)
    if not marker.is_file():
        return None
    record = json.loads(marker.read_text(encoding="utf-8"))
    if record != validate_output(
        task_id, split, input_h5, output_h5, expected_count, config
    ):
        raise RuntimeError(f"sealed replay marker disagrees with output: {marker}")
    return record


def run_replay(
    args: argparse.Namespace,
    split: str,
    input_h5: Path,
    expected_count: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    target_h5 = output_path(input_h5, config)
    prior = existing_complete(
        args.task_id, split, input_h5, target_h5, expected_count, config
    )
    if prior is not None:
        print(f"RGB_REPLAY_ALREADY_COMPLETE task={args.task_id} split={split}", flush=True)
        return prior

    target_json = target_h5.with_suffix(".json")
    quarantine_existing(target_h5)
    quarantine_existing(target_json)
    command = [
        str(args.python),
        "-m",
        "mani_skill.trajectory.replay_trajectory",
        "--traj-path",
        str(input_h5),
        "--sim-backend",
        config["sim_backend"],
        "--obs-mode",
        "rgb",
        "--target-control-mode",
        config["control_mode"],
        "--save-traj",
        config["state_flag"],
        "--max-retry",
        str(args.max_retry),
    ]
    if config["sim_backend"] == "physx_cpu":
        command.extend(["--num-envs", str(config["cpu_processes"])])
    else:
        command.extend(["--num-envs", str(config["gpu_num_envs"])])
    if args.smoke_count is not None:
        command.extend(["--count", str(args.smoke_count)])
    log_path = target_h5.with_name(target_h5.stem + ".replay.log")
    print("RGB_REPLAY_COMMAND " + " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.flush()
        os.fsync(log.fileno())
    if completed.returncode != 0:
        raise RuntimeError(
            f"official replay failed with exit {completed.returncode}; see {log_path}"
        )
    record = validate_output(
        args.task_id, split, input_h5, target_h5, expected_count, config
    )
    atomic_json(marker_path(target_h5), record)
    print(
        f"RGB_REPLAY_COMPLETE task={args.task_id} split={split} episodes={expected_count}",
        flush=True,
    )
    return record


def main() -> None:
    args = parse_args()
    config = TASK_CONFIGS[args.task_id]
    records = []
    if args.smoke_count is not None:
        split_names = ("smoke",)
        expected = {"smoke": args.smoke_count}
    else:
        split_names = tuple(SPLIT_COUNTS)
        expected = SPLIT_COUNTS
    for split in split_names:
        input_h5 = args.selected_raw_root / args.task_id / split / "trajectory.h5"
        if not input_h5.is_file():
            raise FileNotFoundError(input_h5)
        records.append(
            run_replay(args, split, input_h5, int(expected[split]), config)
        )
    summary = {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "task_id": args.task_id,
        "splits": records,
        "episodes_attempted": sum(item["episodes_attempted"] for item in records),
        "episodes_saved_successful": sum(
            item["episodes_saved_successful"] for item in records
        ),
        "replay_success_rate": sum(
            item["episodes_saved_successful"] for item in records
        )
        / sum(item["episodes_attempted"] for item in records),
    }
    atomic_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
