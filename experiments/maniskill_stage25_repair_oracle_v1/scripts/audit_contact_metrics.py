#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import PROTOCOL_ID, sha256_file, write_json
from stage25_runtime import ContactTracker, make_env, reset_to_state
from state_bank_common import h5_timestep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converted-training-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    return parser.parse_args()


def replay_positive(env, trajectory: h5py.Group) -> dict:
    actions = np.asarray(trajectory["actions"], dtype=np.float32)
    initial = h5_timestep(trajectory["env_states"], 0)
    reset_to_state(env, initial, 0, 1)
    tracker = ContactTracker("StackCube-v1", env.base_env)
    success_seen = torch.zeros(1, dtype=torch.bool, device=env.base_env.device)
    for action in actions:
        _, _, _, _, info = env.step(torch.as_tensor(action[None], device=env.base_env.device))
        success_seen |= info["success"].to(torch.bool)
        tracker.update(success_seen=success_seen)
    return {
        **tracker.episode_fields(0),
        "steps": int(len(actions)),
        "success_seen": bool(success_seen[0].item()),
    }


def replay_negative(env, trajectory: h5py.Group) -> dict:
    actions = np.asarray(trajectory["actions"], dtype=np.float32)
    initial = h5_timestep(trajectory["env_states"], 0)
    reset_to_state(env, initial, 0, 1)
    tracker = ContactTracker("StackCube-v1", env.base_env)
    action = torch.zeros(1, actions.shape[-1], device=env.base_env.device)
    action[:, -1] = float(actions[0, -1])
    for _ in range(10):
        env.step(action)
        tracker.update()
    return {**tracker.episode_fields(0), "steps": 10}


def forbidden_aliases() -> list[str]:
    tree = ast.parse((SCRIPT_DIR / "stage25_runtime.py").read_text(encoding="utf-8"))
    return sorted(
        {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id == "collisions"
        }
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    env = make_env(
        "StackCube-v1", 1, sim_backend="physx_cpu", reconfiguration_freq=0
    )
    positive, negative = [], []
    try:
        with h5py.File(args.converted_training_h5, "r") as source:
            keys = sorted(source, key=lambda value: int(value.removeprefix("traj_")))
            for key in keys[: args.episodes]:
                positive.append(replay_positive(env, source[key]))
                negative.append(replay_negative(env, source[key]))
    finally:
        env.close()
    finite_forces = all(
        math.isfinite(row[field])
        for row in positive + negative
        for field in ("max_intended_contact_force", "max_unintended_contact_force")
    )
    positive_pass = all(row["intended_contact_onsets"] >= 1 for row in positive)
    negative_pass = all(row["intended_contact_onsets"] == 0 for row in negative)
    aliases = forbidden_aliases()
    write_json(
        args.output,
        {
            "protocol_id": PROTOCOL_ID,
            "status": "CONTACT_METRIC_AUDIT_COMPLETE",
            "positive_replay_rows": positive,
            "negative_neutral_rows": negative,
            "positive_intended_contact_pass": positive_pass,
            "negative_intended_contact_pass": negative_pass,
            "finite_force_channels": finite_forces,
            "forbidden_collisions_alias_identifiers": aliases,
            "contact_metric_gate_pass": bool(
                positive_pass and negative_pass and finite_forces and not aliases
            ),
            "force_threshold_newtons": 1e-4,
            "converted_training_h5": str(args.converted_training_h5),
            "converted_training_h5_sha256": sha256_file(args.converted_training_h5),
            "scope": "scripted StackCube positive replay and initial-state neutral negative control",
            "task_outcome_replacement": False,
        },
    )


if __name__ == "__main__":
    main()
