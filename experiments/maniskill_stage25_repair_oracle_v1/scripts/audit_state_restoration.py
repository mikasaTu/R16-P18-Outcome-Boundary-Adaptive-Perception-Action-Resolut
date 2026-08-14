#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import PROTOCOL_ID, append_jsonl, sha256_file, write_json
from stage25_runtime import (
    make_env,
    reset_to_state,
    state_restore_max_abs,
    state_to_numpy,
    task_snapshot,
)
from state_bank_common import h5_full, stack_phase, stack_predicates, state_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-bank-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-states-per-bank", type=int)
    return parser.parse_args()


def flatten_difference(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    from stage25_runtime import flatten_state

    left, right = flatten_state(first), flatten_state(second)
    if set(left) != set(right):
        raise RuntimeError("final-state field mismatch")
    return max(
        float(np.max(np.abs(np.asarray(left[key]) - np.asarray(right[key])), initial=0.0))
        for key in left
    )


def audit_serial_repeats(
    env: Any,
    state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    repeats: int = 3,
    short_replay_steps: int = 4,
) -> dict[str, Any]:
    """Restore and replay one state repeatedly in a single CPU environment.

    ManiSkill's PhysX CPU backend only supports one environment per process.  The
    protocol calls for serial exact restoration, so every repeat begins with a
    fresh reset of the same one-environment simulator before the frozen action is
    replayed.
    """
    if repeats < 2:
        raise ValueError("restoration audit requires at least two repeats")

    restore_errors: list[float] = []
    finals: list[Mapping[str, Any]] = []
    categories: list[dict[str, Any]] = []
    for _repeat_index in range(repeats):
        reset_to_state(env, state, int(metadata["source_episode_seed"]), 1)
        restored = state_to_numpy(env.base_env.get_state_dict())
        restore_errors.append(state_restore_max_abs(state, restored, 0)[0])

        action = torch.zeros(
            1, env.action_space.shape[-1], device=env.base_env.device
        )
        action[:, -1] = float(metadata["last_legal_gripper_command"])
        success_once = torch.zeros(
            1, dtype=torch.bool, device=env.base_env.device
        )
        for _ in range(short_replay_steps):
            _, _, _, _, info = env.step(action)
            success_once |= info["success"].to(torch.bool)

        final_states = state_to_numpy(env.base_env.get_state_dict())
        finals.append(state_index(final_states, 0))
        snapshot = task_snapshot(env.base_env, "StackCube-v1")
        predicates = stack_predicates(env.base_env)
        categories.append(
            {
                "success_once": bool(success_once[0].item()),
                "success_at_end": bool(snapshot["success"][0]),
                "grasped": bool(snapshot["grasped"][0]),
                "supported": bool(snapshot["supported"][0]),
                "phase": stack_phase(predicates, 0),
            }
        )

    final_difference = max(
        flatten_difference(finals[0], final) for final in finals[1:]
    )
    categorical_agreement = all(
        category == categories[0] for category in categories[1:]
    )
    return {
        "restore_errors": restore_errors,
        "final_difference": final_difference,
        "categories": categories,
        "categorical_agreement": categorical_agreement,
    }


def main() -> None:
    args = parse_args()
    env = make_env(
        "StackCube-v1",
        1,
        obs_mode="state",
        sim_backend="physx_cpu",
        reconfiguration_freq=0,
    )
    raw_path = args.output.with_name("state_restoration_raw.jsonl")
    if args.output.exists() or raw_path.exists():
        raise FileExistsError("refusing to overwrite restoration evidence")
    rows = []
    try:
        for bank in ("calibration", "confirmatory", "post_success_diagnostic"):
            manifest_path = args.state_bank_root / bank / "state_bank_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            h5_path = Path(manifest["state_bank_h5"])
            if sha256_file(h5_path) != manifest["state_bank_h5_sha256"]:
                raise RuntimeError(f"state HDF5 digest mismatch: {bank}")
            state_rows = manifest["states"]
            if args.max_states_per_bank is not None:
                state_rows = state_rows[: args.max_states_per_bank]
            with h5py.File(h5_path, "r") as source:
                for metadata in state_rows:
                    state = h5_full(source[f"{metadata['bank_id']}/env_state"])
                    audit = audit_serial_repeats(env, state, metadata)
                    restore_errors = audit["restore_errors"]
                    final_difference = audit["final_difference"]
                    categories = audit["categories"]
                    categorical_agreement = audit["categorical_agreement"]
                    row = {
                        "protocol_id": PROTOCOL_ID,
                        "bank": bank,
                        "bank_id": metadata["bank_id"],
                        "state_sha256": metadata["state_sha256"],
                        "initial_restore_max_abs": max(restore_errors),
                        "short_replay_final_state_repeat_max_abs": final_difference,
                        "categorical_agreement": categorical_agreement,
                        "categories": categories,
                        "restoration_pass": bool(
                            max(restore_errors) <= 1e-4
                            and final_difference <= 1e-4
                            and categorical_agreement
                        ),
                    }
                    append_jsonl(raw_path, row)
                    rows.append(row)
    finally:
        env.close()
    restoration_rate = float(np.mean([row["restoration_pass"] for row in rows]))
    agreement_rate = float(np.mean([row["categorical_agreement"] for row in rows]))
    summary = {
        "protocol_id": PROTOCOL_ID,
        "status": "STATE_RESTORATION_AUDIT_COMPLETE",
        "states": len(rows),
        "restoration_pass_rate": restoration_rate,
        "same_action_categorical_agreement": agreement_rate,
        "restoration_gate_pass": restoration_rate == 1.0 and agreement_rate >= 0.95,
        "thresholds": {
            "restoration_pass_rate": 1.0,
            "same_action_categorical_agreement_gte": 0.95,
            "max_absolute_state": 1e-4,
        },
        "raw_path": str(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "backend": "physx_cpu",
        "execution": "single_environment_serial",
        "num_envs": 1,
        "repeats": 3,
        "short_replay_steps": 4,
    }
    write_json(args.output, summary)


if __name__ == "__main__":
    main()
