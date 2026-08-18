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
    # The pinned ManiSkill CPU backend intentionally rejects num_envs > 1.
    # Use two independent single-environment instances: this is also the
    # stronger fresh-reset/shadow contract that the smoke is meant to test.
    left = make_env("StackCube-v1", 1)
    right = make_env("StackCube-v1", 1)
    try:
        left_obs, left_info = left.reset(seed=[args.seed])
        right_obs, right_info = right.reset(seed=[args.seed])
        max_action = max_translation = max_rotation = 0.0
        max_rgb = 0
        categorical = True
        left_action = torch.zeros((1, left.single_action_space.shape[0]), device=left.base_env.device)
        right_action = torch.zeros((1, right.single_action_space.shape[0]), device=right.base_env.device)
        for _ in range(args.steps):
            max_action = max(max_action, float(torch.max(torch.abs(left_action - right_action)).item()))
            left_obs, _, _, _, left_info = left.step(left_action)
            right_obs, _, _, _, right_info = right.step(right_action)
            left_position, left_quaternion = object_pose(left.base_env, "StackCube-v1")
            right_position, right_quaternion = object_pose(right.base_env, "StackCube-v1")
            max_translation = max(max_translation, float(np.linalg.norm(left_position[0] - right_position[0])))
            max_rotation = max(max_rotation, quat_distance(left_quaternion[0], right_quaternion[0]))
            left_rgb = left_obs["rgb"].detach().cpu().numpy().astype(np.int16)
            right_rgb = right_obs["rgb"].detach().cpu().numpy().astype(np.int16)
            max_rgb = max(max_rgb, int(np.max(np.abs(left_rgb[0] - right_rgb[0]))))
            categorical = categorical and bool(left_info["success"][0] == right_info["success"][0])
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
        left.close()
        right.close()


if __name__ == "__main__":
    main()
