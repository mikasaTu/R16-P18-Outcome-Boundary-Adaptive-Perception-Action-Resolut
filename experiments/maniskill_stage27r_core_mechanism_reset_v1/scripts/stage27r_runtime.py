from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms as T

from common import PROTOCOL_ID, sha256_file
from multires_policy import MultiResolutionAgent

TASKS = {
    "StackCube-v1": ("pd_ee_delta_pos", 200, "cubeA"),
    "PegInsertionSide-v1": ("pd_ee_delta_pose", 200, "peg"),
    "PlugCharger-v1": ("pd_ee_delta_pose", 300, "charger"),
    "PullCubeTool-v1": ("pd_ee_delta_pose", 200, "cube"),
    "PushT-v1": ("pd_ee_delta_pose", 200, "tee"),
    "PushCube-v1": ("pd_ee_delta_pos", 100, "obj"),
}

GRASP_ACTORS = {
    "StackCube-v1": "cubeA",
    "PegInsertionSide-v1": "peg",
    "PlugCharger-v1": "charger",
    "PullCubeTool-v1": "l_shape_tool",
}

INTENDED_CONTACT_PAIRS = {
    "StackCube-v1": (("cubeA", "tcp"), ("cubeA", "cubeB")),
    "PegInsertionSide-v1": (("peg", "tcp"), ("peg", "box")),
    "PlugCharger-v1": (("charger", "tcp"), ("charger", "receptacle")),
    "PullCubeTool-v1": (("l_shape_tool", "tcp"), ("l_shape_tool", "cube")),
    "PushT-v1": (("tee", "tcp"),),
    "PushCube-v1": (("obj", "tcp"),),
}


def policy_args(task: str, seed: int):
    import train_rgbd as official

    control, horizon, _ = TASKS[task]
    return official.Args(seed=seed, env_id=task, include_depth=False, backbone="resnet18", lr_backbone=1e-5, num_queries=8, control_mode=control, max_episode_steps=horizon, temporal_agg=False, sim_backend="physx_cpu", num_eval_envs=1, capture_video=False)


def make_env(task: str, num_envs: int = 1):
    import mani_skill.envs  # noqa: F401
    import train_rgbd as official
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    control, horizon, _ = TASKS[task]
    reward_mode = "sparse" if task == "PlugCharger-v1" else "normalized_dense"
    raw = gym.make(task, num_envs=num_envs, sim_backend="physx_cpu", render_backend="sapien_cuda", reconfiguration_freq=1, control_mode=control, reward_mode=reward_mode, obs_mode="rgb", render_mode="rgb_array", max_episode_steps=horizon)
    wrapped = official.FlattenRGBDObservationWrapper(raw, depth=False)
    wrapped.transforms = T.Resize((128, 128), antialias=True)
    return ManiSkillVectorEnv(wrapped, auto_reset=False, ignore_terminations=True, record_metrics=False)


def load_agent(env, task: str, seed: int, checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("checkpoint protocol mismatch")
    config = payload["train_config"]
    if config["task_id"] != task or int(config["seed"]) != seed or int(config["num_queries"]) != 8:
        raise RuntimeError("checkpoint binding mismatch")
    args = policy_args(task, seed)
    import train_rgbd as official
    official.args = args
    agent = MultiResolutionAgent(env, args).to(device)
    agent.load_state_dict(payload["ema_model"])
    agent.eval()
    return agent, payload


def choose_tile(obs: Mapping[str, torch.Tensor], grid: int = 2) -> int:
    image = obs["rgb"].float()
    scores = []
    h, w = image.shape[-2:]
    for tile in range(grid * grid):
        row, col = divmod(tile, grid)
        crop = image[..., row*h//grid:(row+1)*h//grid, col*w//grid:(col+1)*w//grid]
        scores.append(float(crop.var().item()))
    return max(range(len(scores)), key=lambda index: (scores[index], -index))


def object_pose(base: Any, task: str) -> tuple[np.ndarray, np.ndarray]:
    actor = getattr(base, TASKS[task][2])
    return actor.pose.p.detach().cpu().numpy(), actor.pose.q.detach().cpu().numpy()


def quat_distance(q1, q2) -> float:
    first, second = np.asarray(q1, float), np.asarray(q2, float)
    first /= max(np.linalg.norm(first), 1e-12); second /= max(np.linalg.norm(second), 1e-12)
    return 2 * math.acos(float(np.clip(abs(np.dot(first, second)), 0, 1)))


def _actor(base: Any, name: str) -> Any:
    return base.agent.tcp if name == "tcp" else getattr(base, name, None)


def physical_events(base: Any, task: str, info: Mapping[str, Any]) -> dict[str, bool]:
    """Task-agnostic physical diagnostics derived only after simulator steps."""
    intended = False
    for left_name, right_name in INTENDED_CONTACT_PAIRS.get(task, ()):
        left, right = _actor(base, left_name), _actor(base, right_name)
        if left is None or right is None:
            continue
        try:
            force = base.scene.get_pairwise_contact_forces(left, right)
            intended = intended or bool(torch.linalg.norm(force, dim=-1)[0].item() > 1e-4)
        except (AttributeError, RuntimeError, TypeError):
            continue
    grasped = False
    grasp_actor = GRASP_ACTORS.get(task)
    if grasp_actor is not None:
        try:
            grasped = bool(base.agent.is_grasping(getattr(base, grasp_actor))[0].item())
        except (AttributeError, RuntimeError, TypeError):
            grasped = False
    target_p, _ = object_pose(base, task)
    catastrophic = bool(target_p[0, 2] < -0.02)
    for key in ("fail", "collision"):
        if key in info:
            value = info[key]
            catastrophic = catastrophic or bool(value[0].item() if hasattr(value, "shape") else value)
    return {"intended_contact": intended or grasped, "grasped": grasped, "catastrophic": catastrophic}


def query(agent, obs, device, visual: str, tile: int, accounting: dict, tile_grid: int = 2) -> torch.Tensor:
    moved = {key: value.to(device) for key, value in obs.items()}
    if device.type == "cuda": torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        chunk, counts = agent.get_action_with_accounting(moved, visual, tile, tile_grid)
    if device.type == "cuda": torch.cuda.synchronize(device)
    accounting["gpu_latency_ms"] += (time.perf_counter() - started) * 1000
    for key, value in counts.items(): accounting[key] += value
    return chunk


def evaluate_episode(env, agent, seed: int, device: torch.device, visual: str, action_mode: str, tile: int | None = None, horizon: int | None = None, prefix_actions: list | None = None, treatment_steps: int | None = None, continuation_steps: int | None = None, tile_grid: int = 2) -> dict:
    obs, _ = env.reset(seed=[int(seed)])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    accounting = {key: 0 for key in ["global_encoder_calls","fine_encoder_calls","visual_tokens","policy_forward_calls","policy_forward_rows"]}
    accounting.update(gpu_latency_ms=0.0, simulator_latency_ms=0.0, prefix_replay_simulator_latency_ms=0.0, selector_latency_ms=0.0, action_opportunities=0, executed_steps=0, peak_memory_bytes=0)
    success_trace, reward_trace, contact_trace, grasp_trace, catastrophic_trace, cached = [], [], [], [], [], None
    last = None
    if prefix_actions:
        for raw_action in prefix_actions:
            action = torch.as_tensor(raw_action, dtype=torch.float32, device=env.base_env.device).reshape(1, -1)
            started = time.perf_counter(); obs, reward, _, _, info = env.step(action); accounting["prefix_replay_simulator_latency_ms"] += (time.perf_counter()-started)*1000
    initial_p, initial_q = object_pose(env.base_env, env.base_env.spec.id)
    total = horizon if horizon is not None else TASKS[env.base_env.spec.id][1]
    for step in range(total):
        in_treatment = treatment_steps is None or step < treatment_steps
        current_visual = visual if in_treatment else "fine"
        current_action = action_mode if in_treatment else "fine"
        current_tile = 0
        if current_visual == "fine":
            if tile is None or not in_treatment:
                selector_started = time.perf_counter()
                current_tile = choose_tile(obs)
                accounting["selector_latency_ms"] += (time.perf_counter() - selector_started) * 1000
            else:
                current_tile = tile
        if current_action == "fine" or step % 4 == 0 or cached is None:
            cached = query(agent, obs, device, current_visual, current_tile, accounting, tile_grid if in_treatment else 2)
        action = cached[:, 0 if current_action == "fine" else step % 4]
        accounting["action_opportunities"] += 1
        started = time.perf_counter(); obs, reward, _, _, info = env.step(action); accounting["simulator_latency_ms"] += (time.perf_counter()-started)*1000
        accounting["executed_steps"] += 1
        success = bool(info["success"][0].item())
        events = physical_events(env.base_env, env.base_env.spec.id, info)
        success_trace.append(success); reward_trace.append(float(reward[0].item()))
        contact_trace.append(events["intended_contact"]); grasp_trace.append(events["grasped"]); catastrophic_trace.append(events["catastrophic"])
        if len(success_trace) >= 5 and all(success_trace[-5:]): break
    final_p, final_q = object_pose(env.base_env, env.base_env.spec.id)
    longest = max((len(list(group)) for value, group in __import__('itertools').groupby(success_trace) if value), default=0)
    first = next((i + 1 for i, value in enumerate(success_trace) if value), None)
    if device.type == "cuda": accounting["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    accounting["estimated_flops"] = int(accounting["global_encoder_calls"] * 1.8e9 + accounting["fine_encoder_calls"] * 1.8e9 + accounting["policy_forward_calls"] * 0.7e9)
    accounting["episode_total_latency_ms"] = accounting["gpu_latency_ms"] + accounting["simulator_latency_ms"] + accounting["selector_latency_ms"]
    accounting["episode_total_compute"] = accounting["episode_total_latency_ms"]
    lost_after_grasp = any(grasp_trace) and any(not value for value in grasp_trace[next(i for i, value in enumerate(grasp_trace) if value) + 1 :])
    collision = any(catastrophic_trace)
    return {
        "episode_seed": seed, "success_once": any(success_trace), "success_hold5": longest >= 5,
        "success_at_end": bool(success_trace[-1]) if success_trace else False,
        "first_success_step": first, "longest_success_streak": longest,
        "post_success_loss": any(success_trace) and not bool(success_trace[-1]),
        "normalized_progress": float((reward_trace[-1] - reward_trace[0]) if len(reward_trace) > 1 else 0.0),
        "intended_contact": any(contact_trace), "unintended_contact": collision, "collision": collision,
        "dropped_or_slipped": bool(lost_after_grasp and not (longest >= 5)),
        "recoverable": bool((longest >= 5) or (len(reward_trace) < 2 or reward_trace[-1] >= reward_trace[0] - 0.05)),
        "object_translation_drift": float(np.linalg.norm(final_p[0]-initial_p[0])),
        "object_rotation_drift": quat_distance(initial_q[0], final_q[0]),
        "success_trace": success_trace, "reward_trace": reward_trace, "intended_contact_trace": contact_trace,
        "grasp_trace": grasp_trace, "catastrophic_trace": catastrophic_trace, "accounting": accounting,
    }


def checkpoint_candidates(run_root: Path, task: str, seed: int) -> list[dict]:
    root = run_root / "training" / task / f"seed_{seed}" / "checkpoints"
    rows = []
    for marker in sorted(root.glob("step_*/COMPLETE.json")):
        data = json.loads(marker.read_text()); rows.append({"step": int(data["global_iteration"]), "validation_loss": float(data["validation_loss"]), "path": str(marker.parent / "checkpoint.pt"), "sha256": data["checkpoint_sha256"]})
    if not rows: raise RuntimeError(f"no checkpoints: {task}/{seed}")
    top = sorted(rows, key=lambda row: (row["validation_loss"], row["step"]))[:2]
    final = max(rows, key=lambda row: row["step"])
    return list({row["step"]: row for row in [*top, final]}.values())
