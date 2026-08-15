from __future__ import annotations

import base64
import copy
import hashlib
import io
import pickle
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
STAGE25_DIR = THIS_DIR.parents[1] / "maniskill_stage25_repair_oracle_v1" / "scripts"
sys.path.insert(0, str(STAGE25_DIR))
from oracle_math import effective_gripper_command  # type: ignore
from stage25_runtime import (  # type: ignore
    ContactTracker,
    load_policy_from_checkpoint,
    make_env,
    neutral_from_last,
    object_pose_drift,
    policy_chunk,
    quaternion_distance_rad,
    task_snapshot,
    temporal_action_for_indices,
)
from state_bank_common import stack_phase, stack_predicates, state_index  # type: ignore

from common import PROTOCOL_ID, canonical_json, sha256_bytes


def array_hash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return sha256_bytes(str(array.dtype).encode() + repr(array.shape).encode() + array.tobytes())


def observation_hash(obs: Mapping[str, torch.Tensor], index: int = 0) -> str:
    digest = hashlib.sha256()
    for key in sorted(obs):
        value = obs[key][index].detach().cpu().numpy()
        digest.update(key.encode())
        digest.update(array_hash(value).encode())
    return digest.hexdigest()


def encode_pickle(value: Any) -> str:
    return base64.b64encode(pickle.dumps(value, protocol=5)).decode("ascii")


def decode_pickle(value: str) -> Any:
    return pickle.loads(base64.b64decode(value.encode("ascii")))


def capture_rng() -> dict[str, Any]:
    return {
        "python": encode_pickle(random.getstate()),
        "numpy": encode_pickle(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().cpu().tolist(),
        "torch_cuda": [state.cpu().tolist() for state in torch.cuda.get_rng_state_all()],
    }


def restore_rng(value: Mapping[str, Any]) -> None:
    random.setstate(decode_pickle(value["python"]))
    np.random.set_state(decode_pickle(value["numpy"]))
    torch.set_rng_state(torch.tensor(value["torch_cpu"], dtype=torch.uint8))
    if torch.cuda.is_available() and value.get("torch_cuda"):
        states = [torch.tensor(item, dtype=torch.uint8) for item in value["torch_cuda"]]
        torch.cuda.set_rng_state_all(states)


def numpy_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: numpy_tree(child) for key, child in value.items()}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def repeat_tree(value: Any, count: int) -> Any:
    if isinstance(value, Mapping):
        return {key: repeat_tree(child, count) for key, child in value.items()}
    return np.repeat(np.asarray(value)[None], count, axis=0)


def tree_max_abs(left: Any, right: Any) -> float:
    """Return the largest numeric difference between matching state trees."""
    if isinstance(left, Mapping):
        if not isinstance(right, Mapping) or set(left) != set(right):
            raise RuntimeError("restored simulator-state tree keys do not match capsule")
        return max((tree_max_abs(left[key], right[key]) for key in left), default=0.0)
    lhs = np.asarray(left.detach().cpu() if isinstance(left, torch.Tensor) else left)
    rhs = np.asarray(right.detach().cpu() if isinstance(right, torch.Tensor) else right)
    # The capsule is unbatched while ManiSkill's live state remains batched.
    if rhs.ndim == lhs.ndim + 1 and rhs.shape[0] == 1:
        rhs = rhs[0]
    if lhs.shape != rhs.shape:
        raise RuntimeError(f"restored simulator-state shape mismatch: {lhs.shape} != {rhs.shape}")
    if lhs.size == 0:
        return 0.0
    return float(np.max(np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))))


def frozen_observation(capsule: "Capsule", device: torch.device) -> dict[str, torch.Tensor]:
    """Materialize the exact captured observation used at the branch point."""
    return {key: torch.as_tensor(value, device=device)[None] for key, value in capsule.observation.items()}


def visual_latent(agent: torch.nn.Module, obs: Mapping[str, torch.Tensor], indices: torch.Tensor) -> torch.Tensor:
    rgb = obs["rgb"][indices].to(next(agent.parameters()).device, non_blocking=True).float() / 255.0
    rgb = agent.normalize(rgb)
    cameras = []
    for camera in range(rgb.shape[1]):
        features, _ = agent.model.backbones[0](rgb[:, camera])
        projected = agent.model.input_proj(features[0])
        cameras.append(projected.mean(dim=(-2, -1)))
    return torch.stack(cameras, dim=1).mean(dim=1)


def public_snapshot(base: Any) -> dict[str, np.ndarray]:
    raw = task_snapshot(base, "StackCube-v1")
    intended_force, _ = ContactTracker("StackCube-v1", base).forces()
    return {
        **raw,
        "contact": (intended_force > 1e-4).detach().cpu().numpy(),
    }


@dataclass
class Capsule:
    capsule_id: str
    capture_type: str
    task_id: str
    model_seed: int
    episode_seed: int
    checkpoint_path: str
    checkpoint_sha256: str
    source_step: int
    phase: str
    full_simulator_state: Any
    elapsed_step: int
    observation: dict[str, np.ndarray]
    observation_sha256: str
    recent_visual_latents: list[list[float]]
    recent_proprio: list[list[float]]
    recent_actions: list[list[float]]
    predicted_first5_actions: list[list[float]]
    temporal_table_prefix: np.ndarray
    last_executed_action: list[float]
    last_legal_gripper_command: float
    success_once: bool
    success_streak: int
    longest_success_streak: int
    rng_states: dict[str, Any]
    trace_prefix_sha256: str
    reference_future: list[dict[str, Any]] = field(default_factory=list)
    pending_policy_chunk: list[list[float]] = field(default_factory=list)

    def feature_dict(self) -> dict[str, np.ndarray]:
        latent = pad_history(self.recent_visual_latents, 4)
        proprio = pad_history(self.recent_proprio, 4)
        actions = pad_history(self.recent_actions, 4)
        predicted = np.asarray(self.predicted_first5_actions, dtype=np.float32).reshape(-1)
        action_array = np.asarray(actions, dtype=np.float32)
        consistency = np.asarray([
            float(np.mean(np.linalg.norm(np.diff(action_array, axis=0), axis=1))) if len(action_array) > 1 else 0.0,
            float(np.std(action_array)),
            float(np.linalg.norm(action_array[-1] - action_array[-2])) if len(action_array) > 1 else 0.0,
            self.last_legal_gripper_command,
        ], dtype=np.float32)
        return {
            "visual": np.asarray(latent, dtype=np.float32),
            "proprio": np.asarray(proprio, dtype=np.float32),
            "actions": action_array,
            "predicted": predicted,
            "consistency": consistency,
        }


def pad_history(rows: Sequence[Sequence[float]], count: int) -> list[list[float]]:
    values = [list(map(float, row)) for row in rows]
    if not values:
        raise ValueError("history cannot be empty")
    while len(values) < count:
        values.insert(0, values[0].copy())
    return values[-count:]


def capsule_bytes(capsule: Capsule) -> bytes:
    buffer = io.BytesIO()
    torch.save(capsule, buffer)
    return buffer.getvalue()


def save_capsule_new(path: Path, capsule: Capsule) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = capsule_bytes(capsule)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
    return sha256_bytes(data)


def load_capsule(path: Path) -> Capsule:
    return torch.load(path, map_location="cpu", weights_only=False)


def make_capsule(
    *, capture_type: str, env: Any, obs: Mapping[str, torch.Tensor], index: int,
    table: torch.Tensor, last_action: torch.Tensor, recent_latents: Sequence[deque],
    recent_proprio: Sequence[deque], recent_actions: Sequence[deque], chunk: torch.Tensor,
    episode_seed: int, model_seed: int, checkpoint_path: str, checkpoint_sha256: str,
    step: int, success_once: bool, streak: int, longest: int, trace_prefix: Sequence[dict[str, Any]],
) -> Capsule:
    predicates = stack_predicates(env.base_env)
    phase = stack_phase(predicates, index)
    state = state_index(env.base_env.get_state_dict(), index)
    one_obs = {key: value[index].detach().cpu().numpy().copy() for key, value in obs.items()}
    identifier = hashlib.sha256(f"{PROTOCOL_ID}|{model_seed}|{episode_seed}|{capture_type}|{step}".encode()).hexdigest()[:24]
    gripper = effective_gripper_command(float(last_action[index, -1].item()), float(env.single_action_space.low[-1]), float(env.single_action_space.high[-1]))
    return Capsule(
        capsule_id=identifier, capture_type=capture_type, task_id="StackCube-v1", model_seed=model_seed,
        episode_seed=int(episode_seed), checkpoint_path=checkpoint_path, checkpoint_sha256=checkpoint_sha256,
        source_step=int(step), phase=phase, full_simulator_state=state, elapsed_step=int(step),
        observation=one_obs, observation_sha256=observation_hash(obs, index),
        recent_visual_latents=[list(map(float, row)) for row in recent_latents[index]],
        recent_proprio=[list(map(float, row)) for row in recent_proprio[index]],
        recent_actions=(
            [list(map(float, row)) for row in recent_actions[index]]
            or [last_action[index].detach().cpu().float().tolist()]
        ),
        predicted_first5_actions=chunk[index, :5].detach().cpu().float().tolist(),
        temporal_table_prefix=table[index, : step + 1, : step + 30].detach().cpu().numpy().copy(),
        last_executed_action=last_action[index].detach().cpu().float().tolist(), last_legal_gripper_command=float(gripper),
        success_once=bool(success_once), success_streak=int(streak), longest_success_streak=int(longest),
        rng_states=capture_rng(), trace_prefix_sha256=sha256_bytes(canonical_json(list(trace_prefix)).encode()),
        pending_policy_chunk=chunk[index].detach().cpu().float().tolist(),
    )


def restore_capsule(
    env: Any, capsule: Capsule, device: torch.device, diagnostics: dict[str, Any] | None = None,
) -> Mapping[str, torch.Tensor]:
    # Reset first: reset itself may consume process RNG.  The captured RNG state
    # belongs exactly at the branch point and therefore must be restored after it.
    restored_state = repeat_tree(capsule.full_simulator_state, 1)
    rerendered, _ = env.reset(seed=[capsule.episode_seed], options={"reset_to_env_states": {"env_states": restored_state}})
    # ManiSkill resets controllers *after* applying reset_to_env_states.  Since
    # get_state_dict includes controller state, apply the complete state once
    # more after reset so the ACT continuation sees the captured controller
    # target instead of a newly initialized target.
    env.base_env.set_state_dict(restored_state)
    env.base_env._elapsed_steps[:] = int(capsule.elapsed_step)
    state_error = tree_max_abs(capsule.full_simulator_state, env.base_env.get_state_dict())
    if state_error > 1e-6:
        raise RuntimeError(f"restored simulator-state mismatch for {capsule.capsule_id}: {state_error}")
    obs = frozen_observation(capsule, device)
    if observation_hash(obs, 0) != capsule.observation_sha256:
        raise RuntimeError(f"serialized capsule observation hash mismatch for {capsule.capsule_id}")
    restore_rng(capsule.rng_states)
    if diagnostics is not None:
        diagnostics.update({
            "simulator_state_max_abs": state_error,
            "captured_observation_sha256": capsule.observation_sha256,
            "policy_input_observation_sha256": observation_hash(obs, 0),
            "rerendered_observation_sha256": observation_hash(rerendered, 0),
            "rerendered_observation_exact": observation_hash(rerendered, 0) == capsule.observation_sha256,
        })
    return obs


def branch_rollout(
    env: Any, agent: torch.nn.Module, capsule: Capsule, device: torch.device, branch: str,
    horizon: int = 20, original_horizon: bool = False,
) -> dict[str, Any]:
    if branch not in {"continue_policy", "neutral_hold", "hold_then_reobserve", "terminate_oracle"}:
        raise ValueError(branch)
    restore_diagnostics: dict[str, Any] = {}
    obs = restore_capsule(env, capsule, device, restore_diagnostics)
    action_dim = int(env.action_space.shape[-1])
    remaining = 200 - capsule.source_step if original_horizon else horizon
    table = torch.zeros(1, 230, 260, action_dim, device=device)
    prefix = torch.as_tensor(capsule.temporal_table_prefix, device=device)
    table[0, : prefix.shape[0], : prefix.shape[1]] = prefix
    last_action = torch.tensor(capsule.last_executed_action, device=device)[None]
    success_once = bool(capsule.success_once)
    streak = int(capsule.success_streak)
    hold5 = streak >= 5
    first_success_step = capsule.source_step if success_once else -1
    traces = []
    policy_calls = 0
    policy_latency = 0.0
    tracker = ContactTracker("StackCube-v1", env.base_env)
    if branch == "terminate_oracle":
        remaining = 0
    for offset in range(remaining):
        absolute = capsule.source_step + offset
        use_policy = branch == "continue_policy" or (branch == "hold_then_reobserve" and offset >= 2)
        if branch == "hold_then_reobserve" and offset == 2:
            table.zero_()
        absolute_for_table = offset - 2 if branch == "hold_then_reobserve" and offset >= 2 else absolute
        if use_policy and branch == "continue_policy" and offset == 0:
            if not capsule.pending_policy_chunk:
                raise RuntimeError(f"capsule {capsule.capsule_id} lacks the pending ACT policy chunk")
            chunk = torch.as_tensor(capsule.pending_policy_chunk, device=device)[None]
            chosen = temporal_action_for_indices(table, chunk, absolute_for_table, torch.tensor([0], device=device))
            action = chosen
            last_action = chosen
        elif use_policy:
            started = time.perf_counter()
            chunk = policy_chunk(agent, obs, device)
            if device.type == "cuda": torch.cuda.synchronize(device)
            policy_latency += time.perf_counter() - started
            chosen = temporal_action_for_indices(table, chunk, absolute_for_table, torch.tensor([0], device=device))
            action = chosen
            last_action = chosen
            policy_calls += 1
        else:
            action = neutral_from_last(last_action)
        obs, _, _, _, info = env.step(action)
        success = bool(info["success"][0].item())
        if success and not success_once:
            first_success_step = capsule.source_step + offset + 1
        success_once |= success
        streak = streak + 1 if success else 0
        hold5 |= streak >= 5
        intended, unintended, _, _ = tracker.update(success_seen=torch.tensor([success_once], device=device))
        snap = public_snapshot(env.base_env)
        traces.append({
            "step": capsule.source_step + offset + 1, "success": success, "success_streak": streak,
            "object_position": snap["object_position"][0].astype(float).tolist(),
            "object_quaternion": snap["object_quaternion"][0].astype(float).tolist(),
            "normalized_progress": float(snap["normalized_progress"][0]),
            "grasped": bool(snap["grasped"][0]), "supported": bool(snap["supported"][0]),
            "contact": bool(snap["contact"][0]), "intended_contact": bool(intended[0].item()),
            "unintended_contact": bool(unintended[0].item()), "executed_action": action[0].detach().cpu().tolist(),
            "observation_sha256": observation_hash(obs, 0), "policy_called": use_policy,
        })
    terminal_success = bool(traces[-1]["success"]) if traces else bool(capsule.success_streak > 0)
    return {
        "protocol_id": PROTOCOL_ID, "capsule_id": capsule.capsule_id, "capture_type": capsule.capture_type,
        "model_seed": capsule.model_seed, "episode_seed": capsule.episode_seed, "source_step": capsule.source_step,
        "phase": capsule.phase, "branch": branch, "horizon": remaining, "success_once": success_once,
        "success_hold5": hold5, "success_at_horizon": terminal_success,
        "first_success_step": first_success_step, "policy_calls": policy_calls,
        "policy_latency_seconds": policy_latency, "restore_diagnostics": restore_diagnostics, "trace": traces,
    }
