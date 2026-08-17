#!/usr/bin/env python3
"""Small real-simulator fresh-reset lockstep smoke before formal screening."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import PROTOCOL_ID, atomic_json
from stage27r_runtime import make_env, object_pose, quat_distance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2_709_001)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    env = make_env("StackCube-v1", 2)
    try:
        obs, info = env.reset(seed=[args.seed, args.seed])
        max_action = max_translation = max_rotation = 0.0
        max_rgb = 0
        categorical = True
        action = torch.zeros((2, env.single_action_space.shape[0]), device=env.base_env.device)
        for _ in range(args.steps):
            max_action = max(max_action, float(torch.max(torch.abs(action[0] - action[1])).item()))
            obs, _, _, _, info = env.step(action)
            position, quaternion = object_pose(env.base_env, "StackCube-v1")
            max_translation = max(max_translation, float(np.linalg.norm(position[0] - position[1])))
            max_rotation = max(max_rotation, quat_distance(quaternion[0], quaternion[1]))
            rgb = obs["rgb"].detach().cpu().numpy().astype(np.int16)
            max_rgb = max(max_rgb, int(np.max(np.abs(rgb[0] - rgb[1]))))
            categorical = categorical and bool(info["success"][0] == info["success"][1])
        passed = (
            max_action == 0.0
            and max_translation <= 1e-5
            and max_rotation <= 1e-4
            and max_rgb <= 1
            and categorical
        )
        atomic_json(args.output, {
            "protocol_id": PROTOCOL_ID,
            "kind": "fresh_reset_deterministic_lockstep_smoke",
            "task": "StackCube-v1",
            "seed": args.seed,
            "steps": args.steps,
            "broadcast_action_max_abs": max_action,
            "translation_m": max_translation,
            "rotation_rad": max_rotation,
            "rgb_max_lsb": max_rgb,
            "categorical_agreement": categorical,
            "pass": passed,
        })
        if not passed:
            raise RuntimeError("deterministic lockstep smoke failed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
