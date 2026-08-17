#!/usr/bin/env python3
"""Verify exact successful episode splits and derive immutable initial-state identities."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from common import PROTOCOL_ID, atomic_json, sha256_file

TASKS = ("StackCube-v1", "PegInsertionSide-v1", "PlugCharger-v1", "PullCubeTool-v1", "PushT-v1", "PushCube-v1")
SPLITS = {"train": 200, "validation": 50, "test": 50}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, task_checks = [], {}
    for task in TASKS:
        source_ids, seeds, hashes = set(), set(), set()
        split_seed_sets = {}
        for split, expected in SPLITS.items():
            h5_path = args.dataset_root / task / "splits" / split / "trajectory.h5"
            json_path = h5_path.with_suffix(".json")
            metadata = json.loads(json_path.read_text())["episodes"]
            if len(metadata) != expected:
                raise RuntimeError(f"{task}/{split}: {len(metadata)} != {expected}")
            split_seed_sets[split] = set()
            with h5py.File(h5_path, "r") as handle:
                for row in metadata:
                    episode_id, seed = int(row["episode_id"]), int(row["episode_seed"])
                    group = handle[f"traj_{episode_id}"]
                    succeeded = bool(row.get("success")) and bool(np.asarray(group["success"]).any())
                    state_hash = initial_hash(group["env_states"])
                    source_id = f"official-seed-{seed}"
                    if not succeeded or source_id in source_ids or seed in seeds or state_hash in hashes:
                        raise RuntimeError(f"identity/success gate failed: {task}/{split}/{episode_id}")
                    source_ids.add(source_id); seeds.add(seed); hashes.add(state_hash); split_seed_sets[split].add(seed)
                    rows.append({"task": task, "split": split, "source_trajectory_id": source_id, "episode_seed": seed, "initial_state_sha256": state_hash})
            rows[-expected]["split_h5_sha256"] = sha256_file(h5_path)
            rows[-expected]["split_json_sha256"] = sha256_file(json_path)
        leakage = any(split_seed_sets[a] & split_seed_sets[b] for a in SPLITS for b in SPLITS if a < b)
        task_checks[task] = {"episodes": len(source_ids), "unique_sources": len(source_ids), "unique_seeds": len(seeds), "unique_initial_states": len(hashes), "split_leakage": leakage, "pass": len(source_ids) == 300 and not leakage}
    atomic_json(args.output, {"protocol_id": PROTOCOL_ID, "status": "PASS" if all(v["pass"] for v in task_checks.values()) else "FAIL", "task_checks": task_checks, "rows": rows})


if __name__ == "__main__":
    main()
