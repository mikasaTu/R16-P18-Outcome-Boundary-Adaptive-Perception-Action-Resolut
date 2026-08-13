#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from oracle_common import (  # noqa: E402
    PHASES,
    category_tuple,
    ordered_phase_candidates,
    phase_candidate_rows,
)
from oracle_runtime import (  # noqa: E402
    ContactTracker,
    episode_metadata,
    flatten_expected_state,
    h5_timestep,
    make_state_env,
    repeat_state,
    reset_to_state,
    rollout_actions,
    state_restore_max_abs,
    state_to_numpy,
)
from protocol_common import PROTOCOL_ID, sha256_file, write_json  # noqa: E402


STATES_PER_PHASE = 16
RESTORE_REPEATS = 3
ROLLOUT_STEPS = 4
TOLERANCES = {
    "max_absolute_state": 1e-4,
    "object_translation_m": 1e-4,
    "object_rotation_rad": 1e-3,
    "normalized_progress": 1e-4,
}
PHASE_CONTRACT = SCRIPT_DIR.parent / "state_bank" / "phase_contract.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--test-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-episodes", type=int)
    return parser.parse_args()


def reset_controller(base: Any) -> None:
    controller = base.agent.controller
    if isinstance(controller, dict):
        for value in controller.values():
            value.reset()
    else:
        controller.reset()


def state_repeat_difference(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    left = flatten_expected_state(first)
    right = flatten_expected_state(second)
    if set(left) != set(right):
        raise RuntimeError("repeat final-state fields differ")
    maximum = 0.0
    for path in left:
        first_array = np.asarray(left[path])
        second_array = np.asarray(right[path])
        if first_array.shape != second_array.shape:
            raise RuntimeError(f"repeat final-state shape differs for {path}")
        maximum = max(
            maximum,
            float(np.max(np.abs(first_array - second_array), initial=0.0)),
        )
    return maximum


def scan_trajectory_labels(
    env: Any,
    task_id: str,
    trajectory: h5py.Group,
    episode_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    actions = np.asarray(trajectory["actions"], dtype=np.float32)
    action_count = len(actions)
    success = np.asarray(trajectory["success"], dtype=bool)
    if success.shape != (action_count,):
        raise RuntimeError("source success/action length mismatch")
    intended_by_state = np.zeros(action_count + 1, dtype=bool)
    success_by_state = np.zeros(action_count + 1, dtype=bool)
    success_by_state[1:] = success
    initial_state = h5_timestep(trajectory["env_states"], 0)
    reset_to_state(env, initial_state, episode_seed, 1)
    base = env.base_env
    tracker = ContactTracker(task_id, base)
    divergence: list[float] = []
    for timestep in range(action_count):
        action = torch.as_tensor(actions[timestep][None], device=base.device)
        env.step(action)
        intended, _ = tracker.predicates()
        intended_by_state[timestep + 1] = bool(intended[0].item())
        next_state = h5_timestep(trajectory["env_states"], timestep + 1)
        actual = state_to_numpy(base.get_state_dict())
        maximum, _ = state_restore_max_abs(next_state, actual)
        divergence.append(maximum)
        base.set_state_dict(repeat_state(next_state, 1))
        reset_controller(base)
    return intended_by_state, success_by_state, {
        "action_count": action_count,
        "intended_contact_state_count": int(intended_by_state.sum()),
        "success_state_count": int(success_by_state.sum()),
        "one_step_replay_state_divergence_max_abs": float(max(divergence, default=0.0)),
        "one_step_replay_state_divergence_mean_abs": float(np.mean(divergence))
        if divergence
        else 0.0,
    }


def audit_candidate(
    env: Any,
    task_id: str,
    state: Mapping[str, Any],
    actions: np.ndarray,
    episode_seed: int,
) -> dict[str, Any]:
    outcomes: list[dict[str, Any]] = []
    finals: list[dict[str, Any]] = []
    restore_maxima: list[float] = []
    for _ in range(RESTORE_REPEATS):
        repeat_outcomes, _, final_state, restored = rollout_actions(
            env,
            task_id,
            state,
            episode_seed,
            np.asarray(actions, dtype=np.float32)[None],
        )
        restore_maximum, _ = state_restore_max_abs(state, restored)
        restore_maxima.append(restore_maximum)
        outcomes.append(repeat_outcomes[0])
        finals.append(final_state)

    reference = outcomes[0]
    maximum_final_state = max(
        state_repeat_difference(finals[0], value) for value in finals[1:]
    )
    maximum_translation = max(
        float(
            np.linalg.norm(
                np.asarray(reference["object_delta_translation_m"])
                - np.asarray(value["object_delta_translation_m"])
            )
        )
        for value in outcomes[1:]
    )
    maximum_rotation = max(
        abs(
            float(reference["object_delta_rotation_rad"])
            - float(value["object_delta_rotation_rad"])
        )
        for value in outcomes[1:]
    )
    maximum_progress = max(
        abs(
            float(reference["normalized_progress_delta"])
            - float(value["normalized_progress_delta"])
        )
        for value in outcomes[1:]
    )
    categorical_exact = all(
        category_tuple(reference) == category_tuple(value)
        and bool(reference["success_at_end"]) == bool(value["success_at_end"])
        for value in outcomes[1:]
    )
    stable = bool(
        max(restore_maxima) <= TOLERANCES["max_absolute_state"]
        and maximum_final_state <= TOLERANCES["max_absolute_state"]
        and maximum_translation <= TOLERANCES["object_translation_m"]
        and maximum_rotation <= TOLERANCES["object_rotation_rad"]
        and maximum_progress <= TOLERANCES["normalized_progress"]
        and categorical_exact
    )
    reasons = []
    if max(restore_maxima) > TOLERANCES["max_absolute_state"]:
        reasons.append("initial_state_restore_tolerance")
    if maximum_final_state > TOLERANCES["max_absolute_state"]:
        reasons.append("short_replay_state_tolerance")
    if maximum_translation > TOLERANCES["object_translation_m"]:
        reasons.append("object_translation_tolerance")
    if maximum_rotation > TOLERANCES["object_rotation_rad"]:
        reasons.append("object_rotation_tolerance")
    if maximum_progress > TOLERANCES["normalized_progress"]:
        reasons.append("normalized_progress_tolerance")
    if not categorical_exact:
        reasons.append("categorical_outcome_mismatch")
    return {
        "stable": stable,
        "exclusion_reasons": reasons,
        "initial_restore_max_abs": float(max(restore_maxima)),
        "short_replay_final_state_max_abs": maximum_final_state,
        "object_translation_repeat_max_m": maximum_translation,
        "object_rotation_repeat_max_rad": maximum_rotation,
        "normalized_progress_repeat_max": maximum_progress,
        "categorical_outcomes_exact": categorical_exact,
        "reference_outcome": reference,
    }


def write_nested(group: h5py.Group, value: Mapping[str, Any]) -> None:
    for key, child in value.items():
        if isinstance(child, Mapping):
            write_nested(group.create_group(key), child)
        else:
            group.create_dataset(key, data=np.asarray(child), compression="gzip")


def main() -> None:
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("state-bank restoration gate requires a visible CUDA GPU")
    if not args.test_h5.is_file() or not args.test_h5.with_suffix(".json").is_file():
        raise FileNotFoundError(args.test_h5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = episode_metadata(args.test_h5)
    env = make_state_env(args.task_id, 1)
    all_candidates = []
    scan_records: list[dict[str, Any]] = []
    try:
        with h5py.File(args.test_h5, "r") as source:
            trajectory_ids = sorted(int(key.removeprefix("traj_")) for key in source)
            if args.max_episodes is not None:
                trajectory_ids = trajectory_ids[: args.max_episodes]
            for index, trajectory_id in enumerate(trajectory_ids):
                row = metadata[trajectory_id]
                trajectory = source[f"traj_{trajectory_id}"]
                intended, success, scan = scan_trajectory_labels(
                    env,
                    args.task_id,
                    trajectory,
                    int(row["episode_seed"]),
                )
                candidates = phase_candidate_rows(
                    task_id=args.task_id,
                    trajectory_id=trajectory_id,
                    episode_seed=int(row["episode_seed"]),
                    action_count=len(trajectory["actions"]),
                    intended_contact_by_state=intended,
                    success_by_state=success,
                )
                all_candidates.extend(candidates)
                scan_records.append(
                    {
                        "trajectory_id": trajectory_id,
                        "episode_seed": int(row["episode_seed"]),
                        **scan,
                    }
                )
                print(
                    f"STATE_SCAN task={args.task_id} trajectories={index + 1}/{len(trajectory_ids)}",
                    flush=True,
                )

            primary_keys = {
                (candidate.trajectory_id, candidate.timestep)
                for candidate in all_candidates
                if candidate.predicate_source == "primary"
            }
            accepted: list[dict[str, Any]] = []
            exclusions: list[dict[str, Any]] = []
            used_state_keys: set[tuple[int, int]] = set()
            for phase in PHASES:
                phase_count = 0
                for candidate in ordered_phase_candidates(all_candidates, phase):
                    key = (candidate.trajectory_id, candidate.timestep)
                    if key in used_state_keys:
                        continue
                    if (
                        candidate.predicate_source == "fixed_temporal_fallback"
                        and key in primary_keys
                    ):
                        continue
                    trajectory = source[f"traj_{candidate.trajectory_id}"]
                    state = h5_timestep(trajectory["env_states"], candidate.timestep)
                    actions = np.asarray(
                        trajectory["actions"][
                            candidate.timestep : candidate.timestep + ROLLOUT_STEPS
                        ],
                        dtype=np.float32,
                    )
                    audit = audit_candidate(
                        env, args.task_id, state, actions, candidate.episode_seed
                    )
                    common = {
                        "task_id": candidate.task_id,
                        "source_trajectory_id": candidate.trajectory_id,
                        "episode_seed": candidate.episode_seed,
                        "timestep": candidate.timestep,
                        "phase": candidate.phase,
                        "predicate_source": candidate.predicate_source,
                        "selection_sha256": candidate.selection_sha256,
                    }
                    if not audit["stable"]:
                        exclusions.append({**common, "restore_audit": audit})
                        continue
                    bank_id = f"{phase}-{phase_count:02d}"
                    accepted.append(
                        {
                            **common,
                            "bank_id": bank_id,
                            "restore_audit": audit,
                            "_state": state,
                            "_actions": actions,
                        }
                    )
                    used_state_keys.add(key)
                    phase_count += 1
                    if phase_count == STATES_PER_PHASE:
                        break
                if phase_count != STATES_PER_PHASE:
                    failure = {
                        "protocol_id": PROTOCOL_ID,
                        "status": "STATE_BANK_GATE_FAIL",
                        "task_id": args.task_id,
                        "failed_phase": phase,
                        "stable_states_observed": phase_count,
                        "stable_states_required": STATES_PER_PHASE,
                        "source_test_h5": str(args.test_h5),
                        "source_test_h5_sha256": sha256_file(args.test_h5),
                        "source_test_json_sha256": sha256_file(
                            args.test_h5.with_suffix(".json")
                        ),
                        "builder_sha256": sha256_file(Path(__file__).resolve()),
                        "phase_contract_sha256": sha256_file(PHASE_CONTRACT),
                        "candidate_predicates_or_tolerances_changed": False,
                        "accepted_before_failure": [
                            {
                                key: value
                                for key, value in row.items()
                                if not key.startswith("_")
                            }
                            for row in accepted
                        ],
                        "excluded_candidates": exclusions,
                        "phase_scan": scan_records,
                        "phase_contract": "state_bank/phase_contract.json",
                        "completed_at_unix": time.time(),
                    }
                    write_json(args.output_dir / "state_bank_manifest.json", failure)
                    print(json.dumps(failure, indent=2, sort_keys=True), flush=True)
                    return

            h5_path = args.output_dir / "state_bank.h5"
            temporary_h5 = h5_path.with_name(f".{h5_path.name}.tmp-{os.getpid()}")
            with h5py.File(temporary_h5, "w") as target:
                target.attrs["protocol_id"] = PROTOCOL_ID
                target.attrs["task_id"] = args.task_id
                target.attrs["states"] = len(accepted)
                for row in accepted:
                    group = target.create_group(row["bank_id"])
                    group.attrs["source_trajectory_id"] = row["source_trajectory_id"]
                    group.attrs["episode_seed"] = row["episode_seed"]
                    group.attrs["timestep"] = row["timestep"]
                    group.attrs["phase"] = row["phase"]
                    group.attrs["predicate_source"] = row["predicate_source"]
                    write_nested(group.create_group("env_state"), row["_state"])
                    group.create_dataset("source_actions", data=row["_actions"])
                target.flush()
            os.replace(temporary_h5, h5_path)

        public_rows = [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in accepted
        ]
        phase_counts = Counter(row["phase"] for row in public_rows)
        fallback_counts = Counter(
            row["phase"]
            for row in public_rows
            if row["predicate_source"] == "fixed_temporal_fallback"
        )
        result = {
            "protocol_id": PROTOCOL_ID,
            "status": "STATE_BANK_COMPLETE",
            "task_id": args.task_id,
            "source_test_h5": str(args.test_h5),
            "source_test_h5_sha256": sha256_file(args.test_h5),
            "source_test_json_sha256": sha256_file(args.test_h5.with_suffix(".json")),
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "phase_contract_sha256": sha256_file(PHASE_CONTRACT),
            "state_bank_h5": str(h5_path),
            "state_bank_h5_sha256": sha256_file(h5_path),
            "state_count": len(public_rows),
            "phase_counts": dict(phase_counts),
            "temporal_fallback_counts": {
                phase: int(fallback_counts.get(phase, 0)) for phase in PHASES
            },
            "restoration_repeats": RESTORE_REPEATS,
            "rollout_steps": ROLLOUT_STEPS,
            "tolerances": TOLERANCES,
            "states": public_rows,
            "excluded_candidates": exclusions,
            "excluded_candidate_count": len(exclusions),
            "phase_scan": scan_records,
            "phase_contract": "state_bank/phase_contract.json",
            "simulator_privileged_fields_used_as_labels_only": True,
            "completed_at_unix": time.time(),
        }
        write_json(args.output_dir / "state_bank_manifest.json", result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
