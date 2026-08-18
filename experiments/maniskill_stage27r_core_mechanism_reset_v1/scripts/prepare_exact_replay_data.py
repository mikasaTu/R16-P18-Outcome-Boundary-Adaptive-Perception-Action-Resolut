#!/usr/bin/env python3
"""Create exact 200/50/50 RGB splits from an oversized official source pool."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np

from common import PROTOCOL_ID, atomic_json, sha256_file

CONFIG = {
    "StackCube-v1": ("pd_ee_delta_pos", 340, "physx_cpu"),
    "PegInsertionSide-v1": ("pd_ee_delta_pose", 500, "physx_cpu"),
    "PlugCharger-v1": ("pd_ee_delta_pose", 450, "physx_cpu"),
    "PullCubeTool-v1": ("pd_ee_delta_pose", 330, "physx_cpu"),
    # The official PushT policy/data are GPU-physics artifacts. Exact action
    # replay on CPU is empirically 0/400 successful, so data conversion stays
    # on the pinned source backend; formal evaluation remains PhysX CPU.
    "PushT-v1": ("pd_ee_delta_pose", 400, "physx_cuda"),
    "PushCube-v1": ("pd_ee_delta_pos", 340, "physx_cpu"),
}
SPLITS = (("train", 200), ("validation", 50), ("test", 50))


def replay_state_flags(task_id: str) -> list[str]:
    """Bind GPU-sensitive PushT rendering to the recorded successful states."""
    return ["--use-env-states"] if task_id == "PushT-v1" else ["--use-first-env-state"]


def initial_hash(group: h5py.Group) -> str:
    digest = hashlib.sha256()
    datasets = []
    group.visititems(lambda name, obj: datasets.append((name, obj)) if isinstance(obj, h5py.Dataset) else None)
    for name, dataset in sorted(datasets):
        value = np.ascontiguousarray(dataset[0])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def write_subset(source_h5: Path, target_h5: Path, count: int) -> None:
    if target_h5.exists() or target_h5.with_suffix(".json").exists():
        raise FileExistsError(f"fail-on-overwrite: {target_h5}")
    metadata = json.loads(source_h5.with_suffix(".json").read_text())
    # Some v1 official metadata (notably PlugCharger) names the retired
    # ``dense`` mode. Replay does not record reward, so bind both original and
    # target environments to the universally supported ``none`` mode.
    metadata["env_info"]["env_kwargs"]["reward_mode"] = "none"
    episodes = sorted(metadata["episodes"], key=lambda row: int(row["episode_id"]))[:count]
    target_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_h5, "r") as source, h5py.File(target_h5, "x") as target:
        for output_id, episode in enumerate(episodes):
            source.copy(source[f"traj_{episode['episode_id']}"], target, name=f"traj_{output_id}")
            episode["episode_id"] = output_id
    metadata["episodes"] = episodes
    target_h5.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def split_replay(source_h5: Path, output_root: Path) -> list[dict]:
    metadata = json.loads(source_h5.with_suffix(".json").read_text())
    episodes = sorted(metadata["episodes"], key=lambda row: int(row["episode_id"]))
    chosen_unique, seen_seeds, seen_hashes = [], set(), set()
    with h5py.File(source_h5, "r") as source:
        for row in episodes:
            if not bool(row.get("success")):
                continue
            seed = int(row["episode_seed"])
            state_hash = initial_hash(source[f"traj_{row['episode_id']}"]["env_states"])
            if seed in seen_seeds or state_hash in seen_hashes:
                continue
            seen_seeds.add(seed); seen_hashes.add(state_hash)
            chosen_unique.append(row)
            if len(chosen_unique) == 300:
                break
    if len(chosen_unique) < 300:
        raise RuntimeError(
            f"replay yielded only {len(chosen_unique)} unique successful trajectories "
            f"from {len(episodes)} successful replay rows"
        )
    cursor, records = 0, []
    with h5py.File(source_h5, "r") as source:
        for split, count in SPLITS:
            target = output_root / split / "trajectory.h5"
            if target.exists() or target.with_suffix(".json").exists():
                raise FileExistsError(f"fail-on-overwrite: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            chosen = chosen_unique[cursor : cursor + count]
            with h5py.File(target, "x") as sink:
                for new_id, row in enumerate(chosen):
                    source.copy(source[f"traj_{row['episode_id']}"], sink, name=f"traj_{new_id}")
            split_rows = []
            for new_id, row in enumerate(chosen):
                row = dict(row); row["episode_id"] = new_id; row["split"] = split
                split_rows.append(row)
            split_meta = dict(metadata); split_meta["episodes"] = split_rows
            target.with_suffix(".json").write_text(json.dumps(split_meta, indent=2) + "\n")
            records.append({"split": split, "count": count, "h5": str(target), "h5_sha256": sha256_file(target), "json_sha256": sha256_file(target.with_suffix('.json')), "selection": "first_300_successful_unique_seed_and_initial_state"})
            cursor += count
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", choices=sorted(CONFIG), required=True)
    parser.add_argument("--official-h5", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    control, pool_count, replay_backend = CONFIG[args.task_id]
    task_root = args.output_root / args.task_id
    complete = task_root / "DATA_COMPLETE.json"
    if complete.exists():
        print(complete.read_text()); return
    raw = task_root / "oversized_source" / "trajectory.h5"
    write_subset(args.official_h5, raw, pool_count)
    command = [str(args.python), "-m", "mani_skill.trajectory.replay_trajectory", "--traj-path", str(raw), "--sim-backend", replay_backend, "--obs-mode", "rgb", "--reward-mode", "none", "--target-control-mode", control, "--save-traj", *replay_state_flags(args.task_id), "--max-retry", "9", "--num-envs", "8"]
    log = task_root / "replay.log"
    with log.open("x") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"replay failed {result.returncode}: {log}")
    replayed = raw.with_name(f"trajectory.rgb.{control}.{replay_backend}.h5")
    records = split_replay(replayed, task_root / "splits")
    atomic_json(complete, {"protocol_id": PROTOCOL_ID, "status": "PASS", "task": args.task_id, "source_pool": pool_count, "control_mode": control, "replay_backend": replay_backend, "formal_backend": "physx_cpu", "splits": records})
    print(complete.read_text())


if __name__ == "__main__":
    main()
