#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import PROTOCOL_ID, append_jsonl, sha256_file, write_json_new
from stage26_runtime import (
    branch_rollout, load_policy_from_checkpoint, make_capsule, make_env,
    observation_hash, policy_chunk, public_snapshot, save_capsule_new,
    temporal_action_for_indices, visual_latent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--seed-bank", type=Path, required=True)
    parser.add_argument("--bank", choices=("train_source", "calibration"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--max-episodes", type=int)
    return parser.parse_args()


def reset_rng() -> None:
    random.seed(16018); np.random.seed(16018); torch.manual_seed(16018); torch.cuda.manual_seed_all(16018)


def scan_batch(env: Any, agent: torch.nn.Module, seeds: list[int], device: torch.device) -> list[dict[str, list[int]]]:
    reset_rng()
    obs, _ = env.reset(seed=seeds)
    count, horizon, queries = len(seeds), 200, 30
    dim = int(env.action_space.shape[-1])
    table = torch.zeros(count, horizon + queries, horizon + 2 * queries, dim, device=device)
    indices = torch.arange(count, device=device)
    phases, progress, successes = [[] for _ in seeds], [[] for _ in seeds], [[] for _ in seeds]
    for step in range(horizon):
        chunk = policy_chunk(agent, obs, device)
        action = temporal_action_for_indices(table, chunk, step, indices)
        obs, _, _, _, info = env.step(action)
        snap = public_snapshot(env.base_env)
        from stage26_runtime import stack_phase, stack_predicates
        predicates = stack_predicates(env.base_env)
        for index in range(count):
            phases[index].append(stack_phase(predicates, index))
            progress[index].append(float(snap["normalized_progress"][index]))
            successes[index].append(bool(info["success"][index].item()))
    targets = []
    for index in range(count):
        current: dict[str, list[int]] = {}
        success_steps = [i + 1 for i, value in enumerate(successes[index]) if value]
        first_success = success_steps[0] if success_steps else None
        streak = 0; first_hold5 = None
        for i, value in enumerate(successes[index]):
            streak = streak + 1 if value else 0
            if streak >= 5:
                first_hold5 = i + 1; break
        near = [i + 1 for i, phase in enumerate(phases[index]) if phase == "placement_contact_near_completion"]
        if near: current["first_near_completion"] = [near[0]]
        if first_success is not None:
            for name, delta in (("first_success_minus_6", -6), ("first_success_minus_3", -3), ("first_success", 0), ("first_success_plus_3", 3)):
                step = first_success + delta
                if 0 <= step < horizon: current[name] = [step]
            if first_hold5 is not None: current["first_hold5"] = [first_hold5]
        else:
            if near: current["first_near_completion_false_positive"] = [near[0]]
            current["maximum_progress"] = [int(np.argmax(progress[index])) + 1]
        targets.append(current)
    return targets


def collect_batch(env: Any, agent: torch.nn.Module, seeds: list[int], targets: list[dict[str, list[int]]], device: torch.device, args: argparse.Namespace) -> list[tuple[Any, str]]:
    reset_rng()
    obs, _ = env.reset(seed=seeds)
    count, horizon, queries = len(seeds), 200, 30
    dim = int(env.action_space.shape[-1])
    table = torch.zeros(count, horizon + queries, horizon + 2 * queries, dim, device=device)
    indices = torch.arange(count, device=device)
    last_action = torch.zeros(count, dim, device=device)
    histories_latent = [deque(maxlen=4) for _ in seeds]
    histories_proprio = [deque(maxlen=4) for _ in seeds]
    histories_action = [deque(maxlen=4) for _ in seeds]
    traces = [[] for _ in seeds]
    capsules: list[Any] = []
    success_once = [False] * count; streak = [0] * count; longest = [0] * count
    pending: dict[int, list[Any]] = {i: [] for i in range(count)}
    for step in range(horizon):
        latent = visual_latent(agent, obs, indices).detach().cpu().numpy()
        for index in range(count):
            histories_latent[index].append(latent[index].astype(float).tolist())
            histories_proprio[index].append(obs["state"][index].detach().cpu().float().tolist())
        chunk = policy_chunk(agent, obs, device)
        for index, mapping in enumerate(targets):
            for capture_type, capture_steps in mapping.items():
                if step in capture_steps:
                    capsule = make_capsule(
                        capture_type=capture_type, env=env, obs=obs, index=index, table=table,
                        last_action=last_action, recent_latents=histories_latent, recent_proprio=histories_proprio,
                        recent_actions=histories_action, chunk=chunk, episode_seed=seeds[index], model_seed=args.model_seed,
                        checkpoint_path=str(args.checkpoint), checkpoint_sha256=args.checkpoint_sha256, step=step,
                        success_once=success_once[index], streak=streak[index], longest=longest[index], trace_prefix=traces[index],
                    )
                    capsules.append(capsule); pending[index].append(capsule)
        action = temporal_action_for_indices(table, chunk, step, indices)
        last_action = action
        for index in range(count): histories_action[index].append(action[index].detach().cpu().float().tolist())
        obs, _, _, _, info = env.step(action)
        snap = public_snapshot(env.base_env)
        for index in range(count):
            success = bool(info["success"][index].item()); success_once[index] |= success
            streak[index] = streak[index] + 1 if success else 0; longest[index] = max(longest[index], streak[index])
            row = {
                "step": step + 1, "success": success, "object_position": snap["object_position"][index].astype(float).tolist(),
                "object_quaternion": snap["object_quaternion"][index].astype(float).tolist(),
                "contact": bool(snap["contact"][index]), "grasped": bool(snap["grasped"][index]),
                "supported": bool(snap["supported"][index]), "executed_action": action[index].detach().cpu().float().tolist(),
                "observation_sha256": observation_hash(obs, index),
            }
            traces[index].append(row)
            for capsule in pending[index]:
                if len(capsule.reference_future) < 10: capsule.reference_future.append(row)
            pending[index] = [item for item in pending[index] if len(item.reference_future) < 10]
    # Preserve capture types whose first valid state is the terminal fixed-horizon
    # observation (for example first success exactly at step 200). These capsules
    # are not eligible for the ten-step shared-prefix audit, but remain valid
    # counterfactual training states when restored under ignore_terminations.
    terminal_step = horizon
    terminal_indices = [index for index, mapping in enumerate(targets) if any(terminal_step in steps for steps in mapping.values())]
    if terminal_indices:
        index_tensor = torch.arange(count, device=device)
        latent = visual_latent(agent, obs, index_tensor).detach().cpu().numpy()
        chunk = policy_chunk(agent, obs, device)
        for index in range(count):
            histories_latent[index].append(latent[index].astype(float).tolist())
            histories_proprio[index].append(obs["state"][index].detach().cpu().float().tolist())
        for index in terminal_indices:
            for capture_type, capture_steps in targets[index].items():
                if terminal_step not in capture_steps:
                    continue
                capsules.append(make_capsule(capture_type=capture_type, env=env, obs=obs, index=index, table=table, last_action=last_action, recent_latents=histories_latent, recent_proprio=histories_proprio, recent_actions=histories_action, chunk=chunk, episode_seed=seeds[index], model_seed=args.model_seed, checkpoint_path=str(args.checkpoint), checkpoint_sha256=args.checkpoint_sha256, step=terminal_step, success_once=success_once[index], streak=streak[index], longest=longest[index], trace_prefix=traces[index]))
    result = []
    for capsule in capsules:
        name = f"{capsule.episode_seed}-{capsule.capture_type}-{capsule.source_step}-{capsule.capsule_id}.pt"
        result.append((capsule, name))
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("collection requires CUDA")
    payload = json.loads(args.seed_bank.read_text())
    seeds = [int(x) for x in payload["banks"][args.bank]]
    if args.max_episodes is not None: seeds = seeds[:args.max_episodes]
    if len(seeds) % args.num_envs: raise ValueError("num-envs must divide episode count")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    marker = args.output_dir / "COLLECTION_COMPLETE.json"
    if marker.exists(): print("COLLECTION_ALREADY_COMPLETE"); return
    device = torch.device("cuda")
    env = make_env("StackCube-v1", args.num_envs, sim_backend="physx_cuda")
    agent, _ = load_policy_from_checkpoint(env, "StackCube-v1", args.model_seed, args.checkpoint, device, args.checkpoint_sha256)
    manifest = args.output_dir / "capsules.jsonl"; branches = args.output_dir / "branches.jsonl"
    if manifest.exists() or branches.exists(): raise FileExistsError("top-level output exists without completion marker")
    count = 0; started = time.time()
    branch_env = make_env("StackCube-v1", 1, sim_backend="physx_cuda", reconfiguration_freq=0)
    try:
        for offset in range(0, len(seeds), args.num_envs):
            batch = seeds[offset:offset + args.num_envs]
            shard = args.output_dir / "shards" / f"batch_{offset:06d}"
            shard_marker = shard / "SHARD_COMPLETE.json"
            if shard_marker.exists():
                value = json.loads(shard_marker.read_text())
                if value.get("protocol_id") != PROTOCOL_ID or value.get("episode_seeds") != batch:
                    raise RuntimeError(f"invalid resume shard: {shard_marker}")
                print(f"STAGE26_COLLECTION_RESUME_SKIP bank={args.bank} model_seed={args.model_seed} offset={offset}", flush=True)
                continue
            if shard.exists():
                # A preempted attempt is evidence, not a resumable unit: only a
                # SHARD_COMPLETE marker commits a shard. Preserve the partial
                # directory under a unique name and restart this deterministic
                # batch without overwriting any bytes.
                preserved = shard.with_name(f"{shard.name}.partial-preserved-{time.time_ns()}")
                shard.rename(preserved)
                print(f"STAGE26_COLLECTION_PRESERVED_PARTIAL source={shard} target={preserved}", flush=True)
            shard.mkdir(parents=True)
            shard_manifest = shard / "capsules.jsonl"; shard_branches = shard / "branches.jsonl"
            targets = scan_batch(env, agent, batch, device)
            captured = collect_batch(env, agent, batch, targets, device, args)
            for capsule, name in captured:
                path = shard / "capsules" / name
                digest = save_capsule_new(path, capsule)
                row = {"protocol_id": PROTOCOL_ID, "bank": args.bank, "capsule_id": capsule.capsule_id, "capture_type": capsule.capture_type, "episode_seed": capsule.episode_seed, "model_seed": args.model_seed, "source_step": capsule.source_step, "phase": capsule.phase, "path": str(path), "sha256": digest, "reference_future_steps": len(capsule.reference_future)}
                branch_rows = {}
                for branch in ("continue_policy", "neutral_hold", "hold_then_reobserve"):
                    branch_rows[branch] = branch_rollout(branch_env, agent, capsule, device, branch, horizon=20)
                    append_jsonl(shard_branches, branch_rows[branch])
                row.update({"hold_success_20": branch_rows["neutral_hold"]["success_at_horizon"], "continue_success_20": branch_rows["continue_policy"]["success_at_horizon"], "reobserve_success_20": branch_rows["hold_then_reobserve"]["success_at_horizon"]})
                append_jsonl(shard_manifest, row); count += 1
            write_json_new(shard_marker, {"protocol_id": PROTOCOL_ID, "status": "COLLECTION_SHARD_COMPLETE", "episode_seeds": batch, "capsules": len(captured), "capsules_jsonl_sha256": sha256_file(shard_manifest), "branches_jsonl_sha256": sha256_file(shard_branches)})
            print(f"STAGE26_COLLECTION_PROGRESS bank={args.bank} model_seed={args.model_seed} episodes={offset + len(batch)}/{len(seeds)} capsules={count}", flush=True)
    finally:
        branch_env.close()
        env.close()
    shard_markers = sorted((args.output_dir / "shards").glob("batch_*/SHARD_COMPLETE.json"))
    if len(shard_markers) != len(seeds) // args.num_envs: raise RuntimeError("not all collection shards completed")
    with manifest.open("xb") as target_manifest, branches.open("xb") as target_branches:
        for shard_marker in shard_markers:
            shard = shard_marker.parent
            target_manifest.write((shard / "capsules.jsonl").read_bytes())
            target_branches.write((shard / "branches.jsonl").read_bytes())
    count = sum(1 for line in manifest.read_text().splitlines() if line.strip())
    write_json_new(marker, {"protocol_id": PROTOCOL_ID, "status": "COUNTERFACTUAL_COLLECTION_COMPLETE", "bank": args.bank, "model_seed": args.model_seed, "episodes": len(seeds), "capsules": count, "shards": len(shard_markers), "capsules_jsonl_sha256": sha256_file(manifest), "branches_jsonl_sha256": sha256_file(branches), "wall_seconds": time.time() - started})


if __name__ == "__main__": main()
