#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import PHASES, PROTOCOL_ID, canonical_json, sha256_file, write_json
from stage25_runtime import (
    load_policy_from_checkpoint,
    make_env,
    policy_chunk,
    task_snapshot,
    temporal_action_for_indices,
)
from state_bank_common import (
    hash_rgb,
    hash_state,
    h5_timestep,
    public_predicates,
    stack_phase,
    stack_predicates,
    state_index,
    write_nested,
)

BANK_CONTRACT = {
    "calibration": {"per_phase_expert": 4, "per_phase_on_policy": 4},
    "confirmatory": {"per_phase_expert": 8, "per_phase_on_policy": 8},
}
HARD_PHASE_ORDER = (
    "placement_contact_near_completion",
    "object_in_hand_pre_placement",
    "pre_grasp_or_pre_contact",
    "free_space_approach",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-checkpoints", type=Path, required=True)
    parser.add_argument("--oracle-seed-bank", type=Path, required=True)
    parser.add_argument("--official-h5", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--max-on-policy-seeds", type=int)
    return parser.parse_args()


def selection_row(path: Path, model_seed: int = 16018) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "CHECKPOINT_FINAL_SELECTION_COMPLETE":
        raise RuntimeError("closed-loop selection is not complete")
    return value["groups"][f"StackCube-v1/seed_{model_seed}"]["selected"]


def selection_hash(bank: str, source: str, episode_seed: int, step: int) -> str:
    return hashlib.sha256(
        f"{PROTOCOL_ID}|{bank}|{source}|{episode_seed}|{step}".encode("utf-8")
    ).hexdigest()


def state_row(
    *,
    bank: str,
    source: str,
    episode_seed: int,
    source_episode_id: int | None,
    step: int,
    phase: str,
    state: Mapping[str, Any],
    rgb_sha256: str,
    predicates: Mapping[str, Any],
    last_gripper: float,
    predicate_source: str,
) -> dict[str, Any]:
    state_sha = hash_state(state)
    return {
        "bank": bank,
        "source": source,
        "source_episode_seed": int(episode_seed),
        "source_episode_id": source_episode_id,
        "source_step": int(step),
        "phase": phase,
        "state_sha256": state_sha,
        "rgb_sha256": rgb_sha256,
        "task_predicates": dict(predicates),
        "last_legal_gripper_command": float(last_gripper),
        "predicate_source": predicate_source,
        "selection_sha256": selection_hash(bank, source, episode_seed, step),
        "_state": state,
    }


def expert_candidates(
    env: Any,
    official_h5: Path,
    expert_rows: list[dict[str, Any]],
    bank: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with h5py.File(official_h5, "r") as source:
        for source_row in expert_rows:
            episode_id = int(source_row["episode_id"])
            episode_seed = int(source_row["episode_seed"])
            trajectory = source[f"traj_{episode_id}"]
            states = trajectory["env_states"]
            action_count = int(trajectory["actions"].shape[0])
            # A deterministic coarse scan is sufficient because the selected
            # state is reclassified from simulator geometry after restoration.
            scan_steps = sorted(set(np.linspace(0, action_count, 18, dtype=int).tolist()))
            found: set[str] = set()
            for step in scan_steps:
                state = h5_timestep(states, min(step, action_count))
                obs, _ = env.reset(
                    seed=[episode_seed],
                    options={"reset_to_env_states": {"env_states": _repeat(state, 1)}},
                )
                predicates = stack_predicates(env.base_env)
                phase = stack_phase(predicates, 0)
                if phase in found:
                    continue
                found.add(phase)
                result.append(
                    state_row(
                        bank=bank,
                        source="expert",
                        episode_seed=episode_seed,
                        source_episode_id=episode_id,
                        step=step,
                        phase=phase,
                        state=state,
                        rgb_sha256=hash_rgb(obs, 0),
                        predicates=public_predicates(predicates, 0),
                        last_gripper=0.0,
                        predicate_source="simulator_geometry",
                    )
                )
    return result


def _repeat(value: Any, count: int) -> Any:
    if isinstance(value, Mapping):
        return {key: _repeat(child, count) for key, child in value.items()}
    return np.repeat(np.asarray(value)[None], count, axis=0)


def collect_on_policy(
    env: Any,
    agent: torch.nn.Module,
    seeds: list[int],
    bank: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result: list[dict[str, Any]] = []
    post_success: list[dict[str, Any]] = []
    horizon = 200
    for start in range(0, len(seeds), int(env.num_envs)):
        batch_seeds = seeds[start : start + int(env.num_envs)]
        if len(batch_seeds) != int(env.num_envs):
            break
        obs, _ = env.reset(seed=batch_seeds)
        action_dim = int(env.action_space.shape[-1])
        table = torch.zeros(
            len(batch_seeds), horizon, horizon + 30, action_dim, device=device
        )
        indices = torch.arange(len(batch_seeds), device=device)
        last_action = torch.zeros(len(batch_seeds), action_dim, device=device)
        streak = torch.zeros(len(batch_seeds), dtype=torch.int64, device=device)
        seen_phase: list[set[str]] = [set() for _ in batch_seeds]
        post_seen = [False] * len(batch_seeds)

        def capture(step: int) -> None:
            states = env.base_env.get_state_dict()
            predicates = stack_predicates(env.base_env)
            for index, episode_seed in enumerate(batch_seeds):
                phase = stack_phase(predicates, index)
                if phase not in seen_phase[index]:
                    seen_phase[index].add(phase)
                    state = state_index(states, index)
                    result.append(
                        state_row(
                            bank=bank,
                            source="on_policy",
                            episode_seed=episode_seed,
                            source_episode_id=None,
                            step=step,
                            phase=phase,
                            state=state,
                            rgb_sha256=hash_rgb(obs, index),
                            predicates=public_predicates(predicates, index),
                            last_gripper=float(last_action[index, -1].item()),
                            predicate_source="simulator_geometry",
                        )
                    )
                if int(streak[index].item()) >= 5 and not post_seen[index]:
                    post_seen[index] = True
                    state = state_index(states, index)
                    post_success.append(
                        state_row(
                            bank="post_success_diagnostic",
                            source="on_policy",
                            episode_seed=episode_seed,
                            source_episode_id=None,
                            step=step,
                            phase="post_success",
                            state=state,
                            rgb_sha256=hash_rgb(obs, index),
                            predicates=public_predicates(predicates, index),
                            last_gripper=float(last_action[index, -1].item()),
                            predicate_source="official_success_hold5",
                        )
                    )

        capture(0)
        for timestep in range(horizon):
            chunk = policy_chunk(agent, obs, device)
            action = temporal_action_for_indices(table, chunk, timestep, indices)
            last_action = action
            obs, _, _, _, info = env.step(action)
            success = info["success"].to(device=device, dtype=torch.bool)
            streak = torch.where(success, streak + 1, torch.zeros_like(streak))
            capture(timestep + 1)
        print(
            f"STATE_SOURCE_PROGRESS bank={bank} seeds={min(start + len(batch_seeds), len(seeds))}/{len(seeds)}",
            flush=True,
        )
    return result, post_success


def choose(
    candidates: list[dict[str, Any]],
    per_phase: int,
    *,
    source: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_seeds: set[int] = set()
    for phase in HARD_PHASE_ORDER:
        rows = sorted(
            (
                row
                for row in candidates
                if row["phase"] == phase and row["source"] == source
            ),
            key=lambda row: row["selection_sha256"],
        )
        phase_rows = []
        for row in rows:
            seed = int(row["source_episode_seed"])
            if seed in used_seeds:
                continue
            used_seeds.add(seed)
            phase_rows.append(row)
            if len(phase_rows) == per_phase:
                break
        if len(phase_rows) != per_phase:
            raise RuntimeError(
                f"insufficient {source} states for {phase}: {len(phase_rows)}/{per_phase}"
            )
        selected.extend(phase_rows)
    return selected


def persist_bank(output_root: Path, bank: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = output_root / bank
    output.mkdir(parents=True, exist_ok=True)
    h5_path = output / "state_bank.h5"
    if h5_path.exists():
        raise FileExistsError(f"refusing to overwrite state bank: {h5_path}")
    public_rows = []
    with h5py.File(h5_path, "w") as target:
        target.attrs["protocol_id"] = PROTOCOL_ID
        target.attrs["bank"] = bank
        for index, row in enumerate(rows):
            bank_id = f"{bank}-{index:03d}"
            group = target.create_group(bank_id)
            write_nested(group.create_group("env_state"), row["_state"])
            public = {key: value for key, value in row.items() if key != "_state"}
            public["bank_id"] = bank_id
            public_rows.append(public)
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "status": "STATE_BANK_BUILT_AWAITING_RESTORATION_AUDIT",
        "task_id": "StackCube-v1",
        "bank": bank,
        "state_count": len(rows),
        "state_bank_h5": str(h5_path),
        "state_bank_h5_sha256": sha256_file(h5_path),
        "states": public_rows,
        "phase_counts": {
            phase: sum(row["phase"] == phase for row in public_rows) for phase in PHASES
        },
        "source_counts": {
            source: sum(row["source"] == source for row in public_rows)
            for source in ("expert", "on_policy")
        },
    }
    write_json(output / "state_bank_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("on-policy state collection requires CUDA")
    seed_manifest = json.loads(args.oracle_seed_bank.read_text(encoding="utf-8"))
    stack = seed_manifest["tasks"]["StackCube-v1"]
    simulator_seeds = [int(value) for value in stack["simulator_seeds"]]
    if args.max_on_policy_seeds is not None:
        simulator_seeds = simulator_seeds[: args.max_on_policy_seeds]
    expert_rows = stack["expert_source_episodes"]
    if len(expert_rows) != 96:
        raise RuntimeError("expert source bank is not frozen at 96 episodes")
    selected = selection_row(args.selected_checkpoints)
    checkpoint = Path(selected["checkpoint_path"])
    device = torch.device("cuda")
    env = make_env(
        "StackCube-v1",
        args.num_envs,
        sim_backend="physx_cuda",
        reconfiguration_freq=1,
    )
    agent, _ = load_policy_from_checkpoint(
        env,
        "StackCube-v1",
        16018,
        checkpoint,
        device,
        selected["checkpoint_sha256"],
    )
    expert_env = make_env(
        "StackCube-v1", 1, sim_backend="physx_cpu", reconfiguration_freq=0
    )
    manifests = {}
    try:
        expert_slices = {"calibration": expert_rows[:32], "confirmatory": expert_rows[32:]}
        seed_slices = {
            "calibration": simulator_seeds[:128],
            "confirmatory": simulator_seeds[128:384],
            "post_success_diagnostic": simulator_seeds[384:512],
        }
        post_candidates: list[dict[str, Any]] = []
        for bank in ("calibration", "confirmatory"):
            experts = expert_candidates(expert_env, args.official_h5, expert_slices[bank], bank)
            on_policy, _ = collect_on_policy(env, agent, seed_slices[bank], bank, device)
            contract = BANK_CONTRACT[bank]
            rows = choose(
                experts, contract["per_phase_expert"], source="expert"
            ) + choose(
                on_policy, contract["per_phase_on_policy"], source="on_policy"
            )
            rows.sort(key=lambda row: (PHASES.index(row["phase"]), row["source"], row["selection_sha256"]))
            manifests[bank] = persist_bank(args.output_root, bank, rows)
        _, post = collect_on_policy(
            env,
            agent,
            seed_slices["post_success_diagnostic"],
            "post_success_diagnostic",
            device,
        )
        post_candidates.extend(post)
        unique_post = []
        used = set()
        for row in sorted(post_candidates, key=lambda item: item["selection_sha256"]):
            seed = int(row["source_episode_seed"])
            if seed in used:
                continue
            used.add(seed)
            unique_post.append(row)
            if len(unique_post) == 16:
                break
        if len(unique_post) != 16:
            raise RuntimeError(f"insufficient post-success states: {len(unique_post)}/16")
        manifests["post_success_diagnostic"] = persist_bank(
            args.output_root, "post_success_diagnostic", unique_post
        )
    finally:
        env.close()
        expert_env.close()
    source_seeds = {
        bank: {int(row["source_episode_seed"]) for row in manifest["states"]}
        for bank, manifest in manifests.items()
    }
    for first, second in (
        ("calibration", "confirmatory"),
        ("calibration", "post_success_diagnostic"),
        ("confirmatory", "post_success_diagnostic"),
    ):
        if source_seeds[first] & source_seeds[second]:
            raise RuntimeError(f"state source overlap: {first}/{second}")
    write_json(
        args.output_root / "STATE_BANK_BUILD_COMPLETE.json",
        {
            "protocol_id": PROTOCOL_ID,
            "status": "STATE_BANK_BUILD_COMPLETE_AWAITING_RESTORATION_AUDIT",
            "selected_checkpoint": selected,
            "oracle_seed_bank_sha256": sha256_file(args.oracle_seed_bank),
            "official_h5_sha256": sha256_file(args.official_h5),
            "manifests": {
                bank: sha256_file(args.output_root / bank / "state_bank_manifest.json")
                for bank in manifests
            },
            "source_seed_sets_pairwise_disjoint": True,
        },
    )


if __name__ == "__main__":
    main()
