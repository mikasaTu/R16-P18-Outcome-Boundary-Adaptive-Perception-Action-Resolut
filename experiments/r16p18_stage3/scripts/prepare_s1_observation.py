#!/usr/bin/env python3
"""Materialize one batch-1 observation from an immutable replay HDF5.

This utility never constructs, resets, or steps an environment. It uses the
same Stage-2.7R Native128Dataset preprocessing as training and writes one
fail-on-overwrite tensor payload for the dev05 forward-only profiler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


PROTOCOL_ID = "R16-P18-STAGE3-S1-BUDGET-FEASIBILITY-V1"
TASKS = {
    "StackCube-v1": ("pd_ee_delta_pos", 200),
    "PegInsertionSide-v1": ("pd_ee_delta_pose", 200),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.h5.is_file():
        raise FileNotFoundError(args.h5)
    if args.output.exists():
        raise FileExistsError(f"fail-on-overwrite: {args.output}")

    import train_rgbd as official
    from multires_policy import Native128Dataset

    control_mode, horizon = TASKS[args.task]
    official.args = official.Args(
        seed=16018,
        env_id=args.task,
        include_depth=False,
        backbone="resnet18",
        lr_backbone=1e-5,
        num_queries=8,
        control_mode=control_mode,
        max_episode_steps=horizon,
        temporal_agg=False,
        sim_backend="physx_cpu",
        num_eval_envs=1,
        capture_video=False,
    )
    dataset = Native128Dataset(str(args.h5), 8, num_traj=1, include_depth=False)
    item = dataset[args.index]
    observation = {
        key: value.unsqueeze(0).contiguous()
        for key, value in item["observations"].items()
        if not key.startswith("_")
    }
    payload = {
        "protocol_id": PROTOCOL_ID,
        "task": args.task,
        "source_h5": str(args.h5),
        "source_h5_sha256": sha256_file(args.h5),
        "source_index": args.index,
        "observation": observation,
        "no_environment_operations": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": "PASS",
        "output": str(args.output),
        "sha256": sha256_file(args.output),
        "state_shape": list(observation["state"].shape),
        "rgb_shape": list(observation["rgb"].shape),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
