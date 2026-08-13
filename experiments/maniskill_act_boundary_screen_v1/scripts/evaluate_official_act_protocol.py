#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from protocol_common import PROTOCOL_ID, atomic_write_text, sha256_file  # noqa: E402

import train_rgbd as official_act  # noqa: E402
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # noqa: E402


TASK_CONFIGS = {
    "PegInsertionSide-v1": {"control_mode": "pd_ee_delta_pose", "horizon": 300},
    "PushT-v1": {"control_mode": "pd_ee_delta_pose", "horizon": 150},
    "StackCube-v1": {"control_mode": "pd_ee_delta_pos", "horizon": 200},
    "PushCube-v1": {"control_mode": "pd_ee_delta_pos", "horizon": 100},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", choices=sorted(TASK_CONFIGS), required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, value: Any) -> None:
    line = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def selected_checkpoint(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    selection_path = run_dir / "checkpoint_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["test_metrics_used"] is not False:
        raise RuntimeError("checkpoint selection was contaminated by test metrics")
    checkpoint = Path(selection["selected"]["path"]) / "checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if sha256_file(checkpoint) != selection["selected"]["checkpoint_sha256"]:
        raise RuntimeError("selected checkpoint digest mismatch")
    return checkpoint, selection


def policy_args(task_id: str, model_seed: int) -> official_act.Args:
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
        num_eval_envs=20,
        capture_video=False,
    )


def make_env(task_id: str, num_envs: int) -> ManiSkillVectorEnv:
    task = TASK_CONFIGS[task_id]
    env = gym.make(
        task_id,
        num_envs=num_envs,
        sim_backend="physx_cuda",
        render_backend="sapien_cuda",
        reconfiguration_freq=1,
        control_mode=task["control_mode"],
        reward_mode="sparse",
        obs_mode="rgb",
        render_mode="rgb_array",
        max_episode_steps=task["horizon"],
    )
    env = official_act.FlattenRGBDObservationWrapper(env, depth=False)
    return ManiSkillVectorEnv(
        env,
        auto_reset=False,
        ignore_terminations=True,
        record_metrics=False,
    )


def pair_contact(base: Any, links: list[Any], actor: Any, threshold: float) -> torch.Tensor:
    contact = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    for link in links:
        force = base.scene.get_pairwise_contact_forces(link, actor)
        contact |= torch.linalg.norm(force, dim=-1) > threshold
    return contact


class ContactTracker:
    def __init__(self, task_id: str, base: Any, threshold: float = 1e-4) -> None:
        self.task_id = task_id
        self.base = base
        self.threshold = threshold
        agent = base.agent
        self.all_links = list(agent.robot.get_links())
        if hasattr(agent, "finger1_link"):
            self.tool_links = [agent.finger1_link, agent.finger2_link]
        else:
            self.tool_links = [agent.tcp]
        self.previous_intended = torch.zeros(
            base.num_envs, dtype=torch.bool, device=base.device
        )
        self.previous_unintended = torch.zeros_like(self.previous_intended)
        self.intended_events = torch.zeros(
            base.num_envs, dtype=torch.int64, device=base.device
        )
        self.unintended_events = torch.zeros_like(self.intended_events)

    def predicates(self) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.base
        table = base.table_scene.table
        robot_table = pair_contact(base, self.all_links, table, self.threshold)
        if self.task_id == "PegInsertionSide-v1":
            intended = pair_contact(base, self.tool_links, base.peg, self.threshold)
            intended |= torch.linalg.norm(
                base.scene.get_pairwise_contact_forces(base.peg, base.box), dim=-1
            ) > self.threshold
            unintended = robot_table | pair_contact(
                base, self.tool_links, base.box, self.threshold
            )
        elif self.task_id == "PushT-v1":
            intended = pair_contact(base, self.tool_links, base.tee, self.threshold)
            unintended = robot_table
        elif self.task_id == "StackCube-v1":
            intended = pair_contact(base, self.tool_links, base.cubeA, self.threshold)
            intended |= torch.linalg.norm(
                base.scene.get_pairwise_contact_forces(base.cubeA, base.cubeB), dim=-1
            ) > self.threshold
            unintended = robot_table | pair_contact(
                base, self.tool_links, base.cubeB, self.threshold
            )
        elif self.task_id == "PushCube-v1":
            intended = pair_contact(base, self.tool_links, base.obj, self.threshold)
            unintended = robot_table
        else:  # pragma: no cover
            raise KeyError(self.task_id)
        return intended, unintended

    def update(self) -> None:
        intended, unintended = self.predicates()
        self.intended_events += intended & ~self.previous_intended
        self.unintended_events += unintended & ~self.previous_unintended
        self.previous_intended = intended
        self.previous_unintended = unintended


def temporal_action(
    action_table: torch.Tensor,
    action_chunk: torch.Tensor,
    timestep: int,
) -> torch.Tensor:
    num_queries = action_chunk.shape[1]
    action_table[:, timestep, timestep : timestep + num_queries] = action_chunk
    start = max(0, timestep + 1 - num_queries)
    actions = action_table[:, start : timestep + 1, timestep]
    weights = torch.exp(
        -0.01 * torch.arange(actions.shape[1], device=actions.device)
    )
    weights = weights / weights.sum()
    return (actions * weights[None, :, None]).sum(dim=1)


def evaluate_batch(
    env: ManiSkillVectorEnv,
    agent: torch.nn.Module,
    seeds: list[int],
    task_id: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    horizon = TASK_CONFIGS[task_id]["horizon"]
    num_envs = len(seeds)
    random.seed(16018)
    np.random.seed(16018)
    torch.manual_seed(16018)
    torch.cuda.manual_seed_all(16018)
    obs, info = env.reset(seed=seeds)
    tracker = ContactTracker(task_id, env.base_env)
    action_dim = env.action_space.shape[-1]
    action_table = torch.zeros(
        num_envs,
        horizon,
        horizon + 30,
        action_dim,
        dtype=torch.float32,
        device=device,
    )
    success_once = torch.zeros(num_envs, dtype=torch.bool, device=device)
    first_success_step = torch.full(
        (num_envs,), -1, dtype=torch.int64, device=device
    )
    policy_seconds = 0.0
    agent.eval()
    with torch.no_grad():
        for timestep in range(horizon):
            policy_obs = {
                key: value.to(device, non_blocking=True) for key, value in obs.items()
            }
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            action_chunk = agent.get_action(policy_obs)
            torch.cuda.synchronize(device)
            policy_seconds += time.perf_counter() - started
            action = temporal_action(action_table, action_chunk, timestep)
            obs, reward, terminated, truncated, info = env.step(action)
            tracker.update()
            success = info["success"].to(device=device, dtype=torch.bool)
            newly_successful = success & ~success_once
            first_success_step[newly_successful] = timestep + 1
            success_once |= success
    success_at_end = info["success"].to(device=device, dtype=torch.bool)
    records = []
    for index, seed in enumerate(seeds):
        records.append(
            {
                "protocol_id": PROTOCOL_ID,
                "task_id": task_id,
                "episode_seed": int(seed),
                "success_once": bool(success_once[index].item()),
                "success_at_end": bool(success_at_end[index].item()),
                "episode_length": horizon,
                "first_success_step": int(first_success_step[index].item()),
                "intended_contact_events": int(tracker.intended_events[index].item()),
                "unintended_contact_events": int(
                    tracker.unintended_events[index].item()
                ),
                "collisions": int(tracker.unintended_events[index].item()),
                "policy_latency_seconds": policy_seconds / num_envs,
                "policy_calls": horizon,
                "action_opportunities": horizon,
            }
        )
    return records


def main() -> None:
    args = parse_args()
    if args.model_seed not in (16018, 16019, 16020):
        raise ValueError("model seed is outside the frozen set")
    if 100 % args.num_envs:
        raise ValueError("num-envs must divide the fixed 100 episodes")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal RGB closed-loop evaluation requires CUDA")
    checkpoint_path, selection = selected_checkpoint(args.run_dir)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError("checkpoint protocol mismatch")
    if payload["train_config"]["task_id"] != args.task_id:
        raise RuntimeError("checkpoint task mismatch")
    if int(payload["train_config"]["seed"]) != args.model_seed:
        raise RuntimeError("checkpoint seed mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = args.output_dir / "episodes.jsonl"
    if episodes_path.exists():
        episodes_path.unlink()
    seed_manifest = json.loads(args.seed_manifest.read_text(encoding="utf-8"))
    seeds = seed_manifest["formal_tasks"][args.task_id]["closed_loop_test_seeds"]
    if len(seeds) != 100 or len(set(seeds)) != 100:
        raise RuntimeError("fixed test seed bank is invalid")

    official_args = policy_args(args.task_id, args.model_seed)
    official_act.args = official_args
    env = make_env(args.task_id, args.num_envs)
    agent = official_act.Agent(env, official_args).to(device)
    agent.load_state_dict(payload["ema_model"])
    all_records: list[dict[str, Any]] = []
    try:
        for start in range(0, 100, args.num_envs):
            batch = evaluate_batch(
                env,
                agent,
                [int(value) for value in seeds[start : start + args.num_envs]],
                args.task_id,
                device,
            )
            for record in batch:
                record["model_seed"] = args.model_seed
                record["selected_checkpoint_step"] = selection["selected"]["step"]
                append_jsonl(episodes_path, record)
            all_records.extend(batch)
            print(
                f"EVAL_PROGRESS task={args.task_id} model_seed={args.model_seed} episodes={len(all_records)}/100",
                flush=True,
            )
    finally:
        env.close()

    summary = {
        "protocol_id": PROTOCOL_ID,
        "status": "EVALUATION_COMPLETE",
        "task_id": args.task_id,
        "model_seed": args.model_seed,
        "episodes": len(all_records),
        "success_once": float(np.mean([item["success_once"] for item in all_records])),
        "success_at_end": float(
            np.mean([item["success_at_end"] for item in all_records])
        ),
        "mean_episode_length": float(
            np.mean([item["episode_length"] for item in all_records])
        ),
        "mean_intended_contact_events": float(
            np.mean([item["intended_contact_events"] for item in all_records])
        ),
        "mean_unintended_contact_events": float(
            np.mean([item["unintended_contact_events"] for item in all_records])
        ),
        "mean_collisions": float(np.mean([item["collisions"] for item in all_records])),
        "total_policy_latency_seconds": float(
            sum(item["policy_latency_seconds"] for item in all_records)
        ),
        "total_policy_calls": int(sum(item["policy_calls"] for item in all_records)),
        "total_action_opportunities": int(
            sum(item["action_opportunities"] for item in all_records)
        ),
        "selected_checkpoint": selection["selected"],
        "checkpoint_selection_metric": selection["selection_metric"],
        "test_metrics_used_for_selection": False,
        "fixed_test_seed_manifest": str(args.seed_manifest),
        "fixed_test_seed_manifest_sha256": sha256_file(args.seed_manifest),
        "contact_metric_schema": "baseline/contact_metric_schema.json",
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
