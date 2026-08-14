#!/usr/bin/env python3
from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
import torch

from common import PROTOCOL_ID, sha256_file

TASK_CONFIGS = {
    "StackCube-v1": {"control_mode": "pd_ee_delta_pos", "horizon": 200},
    "PushCube-v1": {"control_mode": "pd_ee_delta_pos", "horizon": 100},
}


def official_module() -> Any:
    import train_rgbd as official_act

    return official_act


def policy_args(task_id: str, model_seed: int, num_envs: int = 20) -> Any:
    official_act = official_module()
    task = TASK_CONFIGS[task_id]
    return official_act.Args(
        seed=model_seed,
        env_id=task_id,
        include_depth=False,
        backbone="resnet18",
        lr_backbone=1e-5,
        num_queries=30,
        control_mode=task["control_mode"],
        max_episode_steps=task["horizon"],
        temporal_agg=True,
        sim_backend="physx_cuda",
        num_eval_envs=num_envs,
        capture_video=False,
    )


def make_env(
    task_id: str,
    num_envs: int,
    *,
    obs_mode: str = "rgb",
    sim_backend: str = "physx_cuda",
    reconfiguration_freq: int = 1,
) -> Any:
    # Importing the task package is the explicit Gym registration side effect.
    import mani_skill.envs  # noqa: F401

    task = TASK_CONFIGS[task_id]
    render_backend = "sapien_cuda"
    raw = gym.make(
        task_id,
        num_envs=num_envs,
        sim_backend=sim_backend,
        render_backend=render_backend,
        reconfiguration_freq=reconfiguration_freq,
        control_mode=task["control_mode"],
        reward_mode="sparse",
        obs_mode=obs_mode,
        render_mode="rgb_array",
        max_episode_steps=task["horizon"],
    )
    if obs_mode == "rgb":
        raw = official_module().FlattenRGBDObservationWrapper(raw, depth=False)
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    return ManiSkillVectorEnv(
        raw,
        auto_reset=False,
        ignore_terminations=True,
        record_metrics=False,
    )


def load_policy_from_checkpoint(
    env: Any,
    task_id: str,
    model_seed: int,
    checkpoint_path: Path,
    device: torch.device,
    expected_sha256: str | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if expected_sha256 is not None and sha256_file(checkpoint_path) != expected_sha256:
        raise RuntimeError(f"checkpoint digest mismatch: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("protocol_id") != "R16-P18-MS3-ACT-BOUNDARY-SCREEN-V1":
        raise RuntimeError("existing checkpoint protocol mismatch")
    config = payload["train_config"]
    if config["task_id"] != task_id or int(config["seed"]) != model_seed:
        raise RuntimeError("checkpoint task/model-seed mismatch")
    official_act = official_module()
    args = policy_args(task_id, model_seed, int(env.num_envs))
    official_act.args = args
    agent = official_act.Agent(env, args).to(device)
    agent.load_state_dict(payload["ema_model"])
    agent.eval()
    return agent, payload


def tensor_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def pair_force_norm(base: Any, links: Sequence[Any], actor: Any) -> torch.Tensor:
    result = torch.zeros(base.num_envs, dtype=torch.float32, device=base.device)
    for link in links:
        force = base.scene.get_pairwise_contact_forces(link, actor)
        result = torch.maximum(result, torch.linalg.norm(force, dim=-1))
    return result


class ContactTracker:
    """Explicit onset, duration, force, and post-success contact accounting."""

    def __init__(self, task_id: str, base: Any, threshold: float = 1e-4) -> None:
        self.task_id = task_id
        self.base = base
        self.threshold = threshold
        self.all_links = list(base.agent.robot.get_links())
        agent = base.agent
        self.tool_links = (
            [agent.finger1_link, agent.finger2_link]
            if hasattr(agent, "finger1_link")
            else [agent.tcp]
        )
        zeros_bool = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
        zeros_long = torch.zeros(base.num_envs, dtype=torch.int64, device=base.device)
        zeros_float = torch.zeros(base.num_envs, dtype=torch.float32, device=base.device)
        self.previous_intended = zeros_bool.clone()
        self.previous_unintended = zeros_bool.clone()
        self.intended_onsets = zeros_long.clone()
        self.unintended_onsets = zeros_long.clone()
        self.intended_duration = zeros_long.clone()
        self.unintended_duration = zeros_long.clone()
        self.max_intended_force = zeros_float.clone()
        self.max_unintended_force = zeros_float.clone()
        self.post_success_onsets = zeros_long.clone()
        self.success_seen_before_step = zeros_bool.clone()

    def forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.base
        table_builder = getattr(base, "table_scene", None)
        if table_builder is None:
            table_builder = getattr(base, "scene_builder", None)
        if table_builder is None or not hasattr(table_builder, "table"):
            raise RuntimeError(f"unable to resolve table for {self.task_id}")
        robot_table = pair_force_norm(base, self.all_links, table_builder.table)
        if self.task_id == "StackCube-v1":
            tool_a = pair_force_norm(base, self.tool_links, base.cubeA)
            a_b = torch.linalg.norm(
                base.scene.get_pairwise_contact_forces(base.cubeA, base.cubeB), dim=-1
            )
            intended = torch.maximum(tool_a, a_b)
            unintended = torch.maximum(
                robot_table, pair_force_norm(base, self.tool_links, base.cubeB)
            )
        elif self.task_id == "PushCube-v1":
            intended = pair_force_norm(base, self.tool_links, base.obj)
            unintended = robot_table
        else:
            raise KeyError(self.task_id)
        return intended, unintended

    def update(
        self,
        *,
        count_mask: torch.Tensor | None = None,
        success_seen: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        intended_force, unintended_force = self.forces()
        intended = intended_force > self.threshold
        unintended = unintended_force > self.threshold
        mask = (
            torch.ones_like(intended)
            if count_mask is None
            else count_mask.to(device=intended.device, dtype=torch.bool)
        )
        intended_onset = intended & ~self.previous_intended & mask
        unintended_onset = unintended & ~self.previous_unintended & mask
        self.intended_onsets += intended_onset.to(torch.int64)
        self.unintended_onsets += unintended_onset.to(torch.int64)
        self.intended_duration += (intended & mask).to(torch.int64)
        self.unintended_duration += (unintended & mask).to(torch.int64)
        self.max_intended_force = torch.maximum(
            self.max_intended_force, torch.where(mask, intended_force, torch.zeros_like(intended_force))
        )
        self.max_unintended_force = torch.maximum(
            self.max_unintended_force, torch.where(mask, unintended_force, torch.zeros_like(unintended_force))
        )
        if success_seen is not None:
            current_success_seen = success_seen.to(
                device=intended.device, dtype=torch.bool
            )
            self.post_success_onsets += (
                (intended_onset | unintended_onset)
                & self.success_seen_before_step
                & mask
            ).to(torch.int64)
            self.success_seen_before_step |= current_success_seen & mask
        self.previous_intended = torch.where(mask, intended, self.previous_intended)
        self.previous_unintended = torch.where(mask, unintended, self.previous_unintended)
        return intended, unintended, intended_onset, unintended_onset

    def episode_fields(self, index: int) -> dict[str, Any]:
        return {
            "intended_contact_onsets": int(self.intended_onsets[index].item()),
            "unintended_contact_onsets": int(self.unintended_onsets[index].item()),
            "intended_contact_duration_steps": int(self.intended_duration[index].item()),
            "unintended_contact_duration_steps": int(self.unintended_duration[index].item()),
            "max_intended_contact_force": float(self.max_intended_force[index].item()),
            "max_unintended_contact_force": float(self.max_unintended_force[index].item()),
            "post_success_contact_onsets": int(self.post_success_onsets[index].item()),
        }


def quaternion_distance_rad(first: Sequence[float], second: Sequence[float]) -> float:
    q1 = np.asarray(first, dtype=np.float64)
    q2 = np.asarray(second, dtype=np.float64)
    q1 /= max(np.linalg.norm(q1), np.finfo(np.float64).eps)
    q2 /= max(np.linalg.norm(q2), np.finfo(np.float64).eps)
    return 2.0 * math.acos(float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0)))


def object_pose_drift(
    first_position: Sequence[float],
    first_quaternion: Sequence[float],
    second_position: Sequence[float],
    second_quaternion: Sequence[float],
) -> dict[str, float]:
    return {
        "translation_m": float(
            np.linalg.norm(
                np.asarray(second_position, dtype=np.float64)
                - np.asarray(first_position, dtype=np.float64)
            )
        ),
        "rotation_rad": quaternion_distance_rad(first_quaternion, second_quaternion),
    }


def task_snapshot(base: Any, task_id: str) -> dict[str, np.ndarray]:
    if task_id == "StackCube-v1":
        actor = base.cubeA
        cube_b = base.cubeB.pose.p
        goal = torch.hstack(
            [cube_b[:, :2], (cube_b[:, 2] + base.cube_half_size[2] * 2)[:, None]]
        )
        distance = torch.linalg.norm(goal - actor.pose.p, dim=1)
        progress = 1.0 - torch.tanh(5.0 * distance)
        grasped = base.agent.is_grasping(base.cubeA)
        support_gap = torch.linalg.norm(actor.pose.p[:, :2] - cube_b[:, :2], dim=1)
        supported = (support_gap < 0.025) & (actor.pose.p[:, 2] > cube_b[:, 2])
    elif task_id == "PushCube-v1":
        actor = base.obj
        distance = torch.linalg.norm(
            actor.pose.p[:, :2] - base.goal_region.pose.p[:, :2], dim=1
        )
        progress = 1.0 - torch.tanh(5.0 * distance)
        grasped = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
        supported = distance < float(base.goal_radius)
    else:
        raise KeyError(task_id)
    success = base.evaluate()["success"].to(dtype=torch.bool)
    return {
        "object_position": tensor_numpy(actor.pose.p),
        "object_quaternion": tensor_numpy(actor.pose.q),
        "normalized_progress": tensor_numpy(progress),
        "success": tensor_numpy(success),
        "grasped": tensor_numpy(grasped.to(torch.bool)),
        "supported": tensor_numpy(supported.to(torch.bool)),
    }


def policy_chunk(
    agent: torch.nn.Module,
    observation: Mapping[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    policy_obs = {key: value.to(device, non_blocking=True) for key, value in observation.items()}
    with torch.no_grad():
        return agent.get_action(policy_obs)


def temporal_action_for_indices(
    table: torch.Tensor,
    chunk: torch.Tensor,
    timestep: int,
    indices: torch.Tensor,
) -> torch.Tensor:
    table[indices, timestep, timestep : timestep + chunk.shape[1]] = chunk
    start = max(0, timestep + 1 - chunk.shape[1])
    actions = table[indices, start : timestep + 1, timestep]
    weights = torch.exp(-0.01 * torch.arange(actions.shape[1], device=actions.device))
    weights /= weights.sum()
    return (actions * weights[None, :, None]).sum(dim=1)


def neutral_from_last(last_action: torch.Tensor) -> torch.Tensor:
    neutral = torch.zeros_like(last_action)
    # Normalized ManiSkill controllers execute clip(raw, -1, 1).  Retain that
    # controller-effective legal command when policy calls stop.
    neutral[:, -1] = torch.clamp(last_action[:, -1], -1.0, 1.0)
    return neutral


def evaluate_policy_batch(
    env: Any,
    agent: torch.nn.Module,
    seeds: Sequence[int],
    task_id: str,
    device: torch.device,
    *,
    mode: str = "fixed_horizon",
    record_trace: bool = False,
) -> list[dict[str, Any]]:
    if mode not in {
        "fixed_horizon",
        "terminate_first_success",
        "terminate_hold5",
        "neutral_after_hold5",
    }:
        raise ValueError(mode)
    horizon = int(TASK_CONFIGS[task_id]["horizon"])
    count = len(seeds)
    random.seed(16018)
    np.random.seed(16018)
    torch.manual_seed(16018)
    torch.cuda.manual_seed_all(16018)
    obs, _ = env.reset(seed=[int(seed) for seed in seeds])
    base = env.base_env
    tracker = ContactTracker(task_id, base)
    action_dim = int(env.action_space.shape[-1])
    table = torch.zeros(
        count, horizon, horizon + 30, action_dim, dtype=torch.float32, device=device
    )
    last_action = torch.zeros(count, action_dim, dtype=torch.float32, device=device)
    success_once = torch.zeros(count, dtype=torch.bool, device=device)
    success_hold5 = torch.zeros_like(success_once)
    streak = torch.zeros(count, dtype=torch.int64, device=device)
    longest = torch.zeros_like(streak)
    first_success = torch.full((count,), -1, dtype=torch.int64, device=device)
    active_policy = torch.ones(count, dtype=torch.bool, device=device)
    metric_active = torch.ones_like(active_policy)
    episode_length = torch.full((count,), horizon, dtype=torch.int64, device=device)
    terminal_success = torch.zeros_like(success_once)
    policy_calls = torch.zeros(count, dtype=torch.int64, device=device)
    policy_latency = torch.zeros(count, dtype=torch.float64, device=device)
    traces: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    first_success_poses: list[tuple[np.ndarray, np.ndarray] | None] = [None] * count
    agent.eval()
    info: Mapping[str, Any] = {"success": torch.zeros_like(success_once)}
    for timestep in range(horizon):
        policy_active_for_step = active_policy.clone()
        policy_indices = torch.nonzero(active_policy, as_tuple=False).flatten()
        action = neutral_from_last(last_action)
        if policy_indices.numel():
            subset = {key: value[policy_indices] for key, value in obs.items()}
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            chunk = policy_chunk(agent, subset, device)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            chosen = temporal_action_for_indices(table, chunk, timestep, policy_indices)
            action[policy_indices] = chosen
            last_action[policy_indices] = chosen
            policy_calls[policy_indices] += 1
            policy_latency[policy_indices] += elapsed / int(policy_indices.numel())
        obs, _, _, _, info = env.step(action)
        success = info["success"].to(device=device, dtype=torch.bool)
        current_mask = metric_active.clone()
        newly = success & ~success_once & current_mask
        first_success[newly] = timestep + 1
        success_once |= success & current_mask
        streak = torch.where(success & current_mask, streak + 1, torch.where(current_mask, 0, streak))
        longest = torch.maximum(longest, streak)
        success_hold5 |= streak >= 5
        intended, unintended, intended_onset, unintended_onset = tracker.update(
            count_mask=current_mask, success_seen=success_once
        )
        if record_trace:
            snapshot = task_snapshot(base, task_id)
            for index in torch.nonzero(newly, as_tuple=False).flatten().tolist():
                first_success_poses[index] = (
                    snapshot["object_position"][index].copy(),
                    snapshot["object_quaternion"][index].copy(),
                )
            for index in range(count):
                if not bool(current_mask[index].item()) and mode.startswith("terminate_"):
                    continue
                first_pose = first_success_poses[index]
                drift = (
                    None
                    if first_pose is None
                    else object_pose_drift(
                        first_pose[0],
                        first_pose[1],
                        snapshot["object_position"][index],
                        snapshot["object_quaternion"][index],
                    )
                )
                executed = action[index].detach().cpu().float().tolist()
                used_policy = bool(policy_active_for_step[index].item())
                traces[index].append(
                    {
                        "step": timestep + 1,
                        "success_predicate": bool(success[index].item()),
                        "success": bool(success[index].item()),
                        "success_streak": int(streak[index].item()),
                        "policy_active": used_policy,
                        "object_position": snapshot["object_position"][index].astype(float).tolist(),
                        "object_quaternion": snapshot["object_quaternion"][index].astype(float).tolist(),
                        "normalized_progress": float(snapshot["normalized_progress"][index]),
                        "intended_contact": bool((intended[index] > 1e-4).item()),
                        "unintended_contact": bool((unintended[index] > 1e-4).item()),
                        "intended_contact_onset": bool(intended_onset[index].item()),
                        "unintended_contact_onset": bool(unintended_onset[index].item()),
                        "post_success_object_drift": drift,
                        "executed_action": executed,
                        "policy_action": executed if used_policy else None,
                        "neutral_action": None if used_policy else executed,
                        "policy_or_neutral_action": executed,
                    }
                )
        if mode == "terminate_first_success":
            stop = success & metric_active
        elif mode in {"terminate_hold5", "neutral_after_hold5"}:
            stop = (streak >= 5) & metric_active
        else:
            stop = torch.zeros_like(metric_active)
        if mode.startswith("terminate_"):
            terminal_success[stop] = success[stop]
            episode_length[stop] = timestep + 1
            metric_active &= ~stop
            active_policy &= ~stop
            if not metric_active.any():
                break
        elif mode == "neutral_after_hold5":
            active_policy &= ~stop
    still = metric_active
    terminal_success[still] = info["success"].to(device=device, dtype=torch.bool)[still]
    final_snapshot = task_snapshot(base, task_id)
    records: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        record = {
            "protocol_id": PROTOCOL_ID,
            "task_id": task_id,
            "episode_seed": int(seed),
            "mode": mode,
            "success_once": bool(success_once[index].item()),
            "success_hold5": bool(success_hold5[index].item()),
            "success_at_end": bool(terminal_success[index].item()),
            "post_success_loss": bool(
                success_once[index].item() and not terminal_success[index].item()
            ),
            "longest_success_streak": int(longest[index].item()),
            "first_success_step": int(first_success[index].item()),
            "episode_length": int(episode_length[index].item()),
            "policy_calls": int(policy_calls[index].item()),
            "action_opportunities": int(episode_length[index].item()),
            "policy_latency_seconds": float(policy_latency[index].item()),
            "final_object_position": final_snapshot["object_position"][index].astype(float).tolist(),
            **tracker.episode_fields(index),
        }
        if record_trace:
            record["trace"] = traces[index]
            first_pose = first_success_poses[index]
            terminal_trace = traces[index][-1]
            # A terminated vector slot remains in the shared simulator while
            # other slots continue.  The vector-wide final snapshot therefore
            # is not this episode's terminal state.  Bind descriptive terminal
            # fields to the last persisted per-episode trace row instead.
            record["final_object_position"] = terminal_trace[
                "object_position"
            ]
            record["post_success_object_drift"] = (
                None
                if first_pose is None
                else {
                    **object_pose_drift(
                        first_pose[0],
                        first_pose[1],
                        np.asarray(terminal_trace["object_position"]),
                        np.asarray(terminal_trace["object_quaternion"]),
                    ),
                    "from_step": int(first_success[index].item()),
                    "to_step": int(terminal_trace["step"]),
                }
            )
        records.append(record)
    return records


def repeat_state(state: Mapping[str, Any], count: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in state.items():
        if isinstance(child, Mapping):
            result[key] = repeat_state(child, count)
        else:
            result[key] = np.repeat(np.asarray(child)[None], count, axis=0)
    return result


def state_to_numpy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: state_to_numpy(child) for key, child in value.items()}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def reset_to_state(
    env: Any, state: Mapping[str, Any], episode_seed: int, count: int
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, Any]]:
    obs, info = env.reset(
        seed=[int(episode_seed)] * count,
        options={"reset_to_env_states": {"env_states": repeat_state(state, count)}},
    )
    if int(env.base_env._elapsed_steps.max().item()) != 0:
        raise RuntimeError("restored environment did not reset elapsed steps")
    return obs, info


def flatten_state(value: Mapping[str, Any], prefix: str = "") -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key in sorted(value):
        path = f"{prefix}/{key}" if prefix else key
        child = value[key]
        if isinstance(child, Mapping):
            result.update(flatten_state(child, path))
        else:
            result[path] = np.asarray(child)
    return result


def state_restore_max_abs(
    expected: Mapping[str, Any], actual: Mapping[str, Any], env_index: int = 0
) -> tuple[float, dict[str, float]]:
    expected_leaves = flatten_state(expected)
    actual_leaves = flatten_state(actual)
    missing = sorted(set(expected_leaves) - set(actual_leaves))
    if missing:
        raise RuntimeError(f"restored state missing fields: {missing}")
    errors: dict[str, float] = {}
    for path, expected_array in expected_leaves.items():
        observed = np.asarray(actual_leaves[path][env_index])
        if observed.shape != expected_array.shape:
            raise RuntimeError(f"state shape mismatch {path}: {observed.shape} != {expected_array.shape}")
        errors[path] = float(np.max(np.abs(observed - expected_array), initial=0.0))
    return max(errors.values(), default=0.0), errors
