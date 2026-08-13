#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import gymnasium as gym
import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from evaluate_official_act_protocol import (  # noqa: E402
    ContactTracker,
    TASK_CONFIGS,
    policy_args,
    selected_checkpoint,
)
from oracle_common import quaternion_distance_rad  # noqa: E402

import train_rgbd as official_act  # noqa: E402


OBJECT_ATTRIBUTES = {
    "PullCubeTool-v1": "cube",
    "PushT-v1": "tee",
    "StackCube-v1": "cubeA",
    "PushCube-v1": "obj",
}
GRIPPER_DIMENSIONS = {
    "PullCubeTool-v1": (6,),
    "PushT-v1": (),
    "StackCube-v1": (3,),
    "PushCube-v1": (3,),
}


def episode_metadata(h5_path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(h5_path.with_suffix(".json").read_text(encoding="utf-8"))
    return {int(row["episode_id"]): row for row in payload["episodes"]}


def h5_timestep(value: h5py.Group | h5py.Dataset, timestep: int) -> Any:
    if isinstance(value, h5py.Dataset):
        return np.asarray(value[timestep])
    return {key: h5_timestep(value[key], timestep) for key in value.keys()}


def h5_full(value: h5py.Group | h5py.Dataset) -> Any:
    if isinstance(value, h5py.Dataset):
        return np.asarray(value[()])
    return {key: h5_full(value[key]) for key in value.keys()}


def repeat_state(state: Mapping[str, Any], count: int) -> dict[str, Any]:
    def repeat(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: repeat(child) for key, child in value.items()}
        array = np.asarray(value)
        return np.repeat(array[None], count, axis=0)

    return repeat(state)


def clone_observation(obs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in obs.items()}


def reset_to_state(
    env: Any,
    state: Mapping[str, Any],
    episode_seed: int,
    count: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    seeds = [int(episode_seed)] * count
    obs, info = env.reset(
        seed=seeds,
        options={"reset_to_env_states": {"env_states": repeat_state(state, count)}},
    )
    # reset() already resets every controller after applying the exact state.  This
    # explicit invariant catches accidental partial reset behavior.
    if int(env.base_env._elapsed_steps.max().item()) != 0:
        raise RuntimeError("restored environment did not reset elapsed steps")
    return obs, info


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def task_snapshot(base: Any, task_id: str) -> dict[str, np.ndarray]:
    actor = getattr(base, OBJECT_ATTRIBUTES[task_id])
    position = _as_numpy(actor.pose.p)
    quaternion = _as_numpy(actor.pose.q)
    if task_id == "PullCubeTool-v1":
        robot_base = base.agent.robot.get_links()[0].pose.p
        workspace = robot_base.clone()
        workspace[:, 0] += float(base.arm_reach) * 0.1
        distance = torch.linalg.norm(actor.pose.p - workspace, dim=1)
        progress = 1.0 - torch.tanh(3.0 * distance)
    elif task_id == "PushT-v1":
        progress = base.pseudo_render_intersection()
    elif task_id == "StackCube-v1":
        cube_b = base.cubeB.pose.p
        goal = torch.hstack(
            [
                cube_b[:, :2],
                (cube_b[:, 2] + base.cube_half_size[2] * 2)[:, None],
            ]
        )
        distance = torch.linalg.norm(goal - actor.pose.p, dim=1)
        progress = 1.0 - torch.tanh(5.0 * distance)
    elif task_id == "PushCube-v1":
        distance = torch.linalg.norm(
            actor.pose.p[:, :2] - base.goal_region.pose.p[:, :2], dim=1
        )
        progress = 1.0 - torch.tanh(5.0 * distance)
    else:  # pragma: no cover
        raise KeyError(task_id)
    success = base.evaluate()["success"].to(dtype=torch.bool)
    return {
        "object_position": position,
        "object_quaternion": quaternion,
        "normalized_progress": _as_numpy(progress),
        "success": _as_numpy(success),
    }


def rollout_actions(
    env: Any,
    task_id: str,
    state: Mapping[str, Any],
    episode_seed: int,
    actions: np.ndarray,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Roll out N four-step candidates from N identical exact restores."""

    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.ndim != 3 or action_array.shape[1] != 4:
        raise ValueError(f"actions must be [N,4,A], got {action_array.shape}")
    count = action_array.shape[0]
    obs, _ = reset_to_state(env, state, episode_seed, count)
    base = env.base_env
    restored_state = state_to_numpy(base.get_state_dict())
    initial = task_snapshot(base, task_id)
    tracker = ContactTracker(task_id, base)
    success_once = initial["success"].astype(bool).copy()
    final_info: Mapping[str, Any] | None = None
    for step in range(4):
        action = torch.as_tensor(action_array[:, step], device=base.device)
        obs, _, _, _, final_info = env.step(action)
        tracker.update()
        success_once |= _as_numpy(final_info["success"].to(dtype=torch.bool))
    final = task_snapshot(base, task_id)
    intended = _as_numpy(tracker.intended_events > 0).astype(bool)
    unintended = _as_numpy(tracker.unintended_events > 0).astype(bool)
    outcomes: list[dict[str, Any]] = []
    for index in range(count):
        translation = final["object_position"][index] - initial["object_position"][index]
        rotation = quaternion_distance_rad(
            initial["object_quaternion"][index], final["object_quaternion"][index]
        )
        progress_delta = float(
            final["normalized_progress"][index]
            - initial["normalized_progress"][index]
        )
        collision = bool(unintended[index])
        recoverable = bool(
            success_once[index]
            or (not collision and not unintended[index] and progress_delta >= -0.05)
        )
        outcomes.append(
            {
                "short_horizon_success": bool(success_once[index]),
                "success_at_end": bool(final["success"][index]),
                "intended_contact": bool(intended[index]),
                "unintended_contact": bool(unintended[index]),
                "collision": collision,
                "recoverable": recoverable,
                "object_delta_translation_m": [float(value) for value in translation],
                "object_delta_rotation_rad": float(rotation),
                "normalized_progress_before": float(
                    initial["normalized_progress"][index]
                ),
                "normalized_progress_after": float(final["normalized_progress"][index]),
                "normalized_progress_delta": progress_delta,
            }
        )
    accounting = {
        "simulator_restores": count,
        "simulator_steps": count * 4,
        "action_opportunities": count,
        "policy_calls": 0,
        "effect_model_calls": 0,
    }
    final_state = state_to_numpy(base.get_state_dict())
    return outcomes, accounting, final_state, restored_state


def state_to_numpy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: state_to_numpy(child) for key, child in value.items()}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def flatten_expected_state(value: Mapping[str, Any], prefix: str = "") -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key in sorted(value):
        path = f"{prefix}/{key}" if prefix else key
        child = value[key]
        if isinstance(child, Mapping):
            result.update(flatten_expected_state(child, path))
        else:
            result[path] = np.asarray(child)
    return result


def state_restore_max_abs(
    expected: Mapping[str, Any], actual: Mapping[str, Any], env_index: int = 0
) -> tuple[float, dict[str, float]]:
    expected_leaves = flatten_expected_state(expected)
    actual_leaves = flatten_expected_state(actual)
    missing = sorted(set(expected_leaves) - set(actual_leaves))
    if missing:
        raise RuntimeError(f"restored state is missing fields: {missing}")
    per_field: dict[str, float] = {}
    for path, expected_array in expected_leaves.items():
        observed = np.asarray(actual_leaves[path][env_index])
        if observed.shape != expected_array.shape:
            raise RuntimeError(
                f"restored state shape mismatch for {path}: "
                f"{observed.shape} != {expected_array.shape}"
            )
        per_field[path] = float(np.max(np.abs(observed - expected_array), initial=0.0))
    return max(per_field.values(), default=0.0), per_field


def policy_chunk(
    agent: torch.nn.Module,
    observation: Mapping[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    policy_obs = {
        key: value.to(device, non_blocking=True).clone()
        for key, value in observation.items()
    }
    with torch.no_grad():
        chunk = agent.get_action(policy_obs)
    return _as_numpy(chunk)


def load_policy(
    task_id: str,
    model_seed: int,
    run_dir: Path,
    env: Any,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], Path]:
    checkpoint_path, selection = selected_checkpoint(run_dir)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    official_args = policy_args(task_id, model_seed)
    official_act.args = official_args
    agent = official_act.Agent(env, official_args).to(device)
    agent.load_state_dict(payload["ema_model"])
    agent.eval()
    return agent, selection, checkpoint_path


def make_rgb_env(task_id: str, num_envs: int) -> Any:
    task = TASK_CONFIGS[task_id]
    raw = gym.make(
        task_id,
        num_envs=num_envs,
        sim_backend="physx_cpu",
        render_backend="sapien_cuda",
        reconfiguration_freq=0,
        control_mode=task["control_mode"],
        reward_mode="sparse",
        obs_mode="rgb",
        render_mode="rgb_array",
        max_episode_steps=task["horizon"],
    )
    raw = official_act.FlattenRGBDObservationWrapper(raw, depth=False)
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    return ManiSkillVectorEnv(
        raw,
        auto_reset=False,
        ignore_terminations=True,
        record_metrics=False,
    )


def make_state_env(task_id: str, num_envs: int = 1) -> Any:
    task = TASK_CONFIGS[task_id]
    raw = gym.make(
        task_id,
        num_envs=num_envs,
        sim_backend="physx_cpu",
        render_backend="sapien_cuda",
        reconfiguration_freq=0,
        control_mode=task["control_mode"],
        reward_mode="sparse",
        obs_mode="state",
        render_mode="rgb_array",
        max_episode_steps=task["horizon"],
    )
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    return ManiSkillVectorEnv(
        raw,
        auto_reset=False,
        ignore_terminations=True,
        record_metrics=False,
    )
