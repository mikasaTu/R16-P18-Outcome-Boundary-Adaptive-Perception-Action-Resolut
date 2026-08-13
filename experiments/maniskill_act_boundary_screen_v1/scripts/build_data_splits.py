#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import h5py

from protocol_common import (
    FORMAL_TASKS,
    PROTOCOL_ID,
    SPLIT_COUNTS,
    atomic_write_text,
    closed_loop_seeds,
    selection_key,
    sha256_file,
    sha256_hdf5_group,
    sha256_initial_state,
    write_json,
)


EXPERIMENT_RELATIVE = Path("experiments/maniskill_act_boundary_screen_v1")
TASK_SOURCES = {
    "PegInsertionSide-v1": Path("PegInsertionSide-v1/motionplanning/trajectory.h5"),
    "PushT-v1": Path(
        "PushT-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5"
    ),
    "StackCube-v1": Path("StackCube-v1/motionplanning/trajectory.h5"),
    "PushCube-v1": Path("PushCube-v1/motionplanning/trajectory.h5"),
}
SMOKE_SOURCE = Path("PickCube-v1/motionplanning/trajectory.h5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    return parser.parse_args()


def load_candidates(task_id: str, h5_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    json_path = h5_path.with_suffix(".json")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    episodes = sorted(metadata["episodes"], key=lambda item: int(item["episode_id"]))
    candidates: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    seen_initial_states: set[str] = set()
    with h5py.File(h5_path, "r") as source:
        for episode in episodes:
            if not bool(episode.get("success", False)):
                continue
            source_episode_id = int(episode["episode_id"])
            episode_seed = int(episode["episode_seed"])
            group_name = f"traj_{source_episode_id}"
            if group_name not in source:
                raise KeyError(f"{group_name} missing from {h5_path}")
            trajectory = source[group_name]
            if "env_states" not in trajectory or "actions" not in trajectory:
                raise KeyError(f"{group_name} lacks actions or env_states")
            initial_hash = sha256_initial_state(trajectory["env_states"])
            if episode_seed in seen_seeds or initial_hash in seen_initial_states:
                continue
            seen_seeds.add(episode_seed)
            seen_initial_states.add(initial_hash)
            candidates.append(
                {
                    "task_id": task_id,
                    "source_episode_id": source_episode_id,
                    "episode_seed": episode_seed,
                    "initial_state_sha256": initial_hash,
                    "source_trajectory_sha256": sha256_hdf5_group(trajectory),
                    "selection_key": selection_key(
                        task_id, source_episode_id, episode_seed, initial_hash
                    ),
                    "elapsed_steps": int(episode["elapsed_steps"]),
                    "control_mode": episode["control_mode"],
                    "reset_kwargs": copy.deepcopy(episode["reset_kwargs"]),
                    "source_success": True,
                }
            )
    candidates.sort(key=lambda item: item["selection_key"])
    return metadata, candidates


def assign_splits(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = sum(SPLIT_COUNTS.values())
    if len(candidates) < required:
        raise RuntimeError(f"only {len(candidates)} unique candidates; need {required}")
    selected = copy.deepcopy(candidates[:required])
    offset = 0
    for split_name, count in SPLIT_COUNTS.items():
        for split_index, item in enumerate(selected[offset : offset + count]):
            item["split"] = split_name
            item["split_index"] = split_index
            item["selection_rank"] = offset + split_index
        offset += count
    return selected


def build_subset(
    source_h5_path: Path,
    source_metadata: dict[str, Any],
    selected: list[dict[str, Any]],
    output_h5_path: Path,
) -> dict[str, Any]:
    output_json_path = output_h5_path.with_suffix(".json")
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_h5 = output_h5_path.with_name(f".{output_h5_path.name}.tmp-{os.getpid()}")
    output_episodes: list[dict[str, Any]] = []
    with h5py.File(source_h5_path, "r") as source, h5py.File(temporary_h5, "w") as target:
        for subset_id, item in enumerate(selected):
            source_name = f"traj_{item['source_episode_id']}"
            target_name = f"traj_{subset_id}"
            source.copy(source[source_name], target, name=target_name)
            episode = copy.deepcopy(
                next(
                    value
                    for value in source_metadata["episodes"]
                    if int(value["episode_id"]) == item["source_episode_id"]
                )
            )
            episode["episode_id"] = subset_id
            episode["source_episode_id"] = item["source_episode_id"]
            episode["source_initial_state_sha256"] = item["initial_state_sha256"]
            episode["selection_key"] = item["selection_key"]
            episode["split"] = item["split"]
            output_episodes.append(episode)
    os.replace(temporary_h5, output_h5_path)
    output_metadata = {
        key: copy.deepcopy(value)
        for key, value in source_metadata.items()
        if key != "episodes"
    }
    output_metadata.update(
        {
            "episodes": output_episodes,
            "protocol_id": PROTOCOL_ID,
            "selection_algorithm": "sha256_rank_v1",
            "split": selected[0]["split"],
            "source_h5_sha256": sha256_file(source_h5_path),
        }
    )
    write_json(output_json_path, output_metadata)
    return {
        "h5_path": str(output_h5_path),
        "h5_sha256": sha256_file(output_h5_path),
        "json_path": str(output_json_path),
        "json_sha256": sha256_file(output_json_path),
        "episodes": len(output_episodes),
    }


def main() -> None:
    args = parse_args()
    experiment_root = args.repo_root / EXPERIMENT_RELATIVE
    selected_raw_root = args.data_root / "selected_raw"
    manifest_rows: list[dict[str, Any]] = []
    task_summary: dict[str, Any] = {}
    subset_files: list[dict[str, Any]] = []

    for task_id in FORMAL_TASKS:
        source_h5 = args.official_root / TASK_SOURCES[task_id]
        source_metadata, candidates = load_candidates(task_id, source_h5)
        selected = assign_splits(candidates)
        source_json = source_h5.with_suffix(".json")
        source_record = {
            "h5_path": str(source_h5),
            "h5_sha256": sha256_file(source_h5),
            "json_path": str(source_json),
            "json_sha256": sha256_file(source_json),
        }
        for item in selected:
            manifest_rows.append(
                {
                    **item,
                    "protocol_id": PROTOCOL_ID,
                    "source_h5_path": str(source_h5),
                    "source_h5_sha256": source_record["h5_sha256"],
                    "replayed_rgb_sha256": None,
                    "replay_success": None,
                }
            )
        for split_name in SPLIT_COUNTS:
            split_items = [item for item in selected if item["split"] == split_name]
            output_h5 = selected_raw_root / task_id / split_name / "trajectory.h5"
            subset_files.append(
                {
                    "task_id": task_id,
                    "split": split_name,
                    **build_subset(source_h5, source_metadata, split_items, output_h5),
                }
            )
        selected_seeds = [item["episode_seed"] for item in selected]
        task_summary[task_id] = {
            "official_successful_episodes": sum(
                bool(value.get("success", False))
                for value in source_metadata["episodes"]
            ),
            "eligible_unique_seed_and_initial_state": len(candidates),
            "selected": len(selected),
            "split_counts": {
                split: sum(item["split"] == split for item in selected)
                for split in SPLIT_COUNTS
            },
            "selected_unique_seeds": len(set(selected_seeds)),
            "selected_unique_initial_states": len(
                {item["initial_state_sha256"] for item in selected}
            ),
            "closed_loop_test_seeds": closed_loop_seeds(task_id, selected_seeds),
            "source": source_record,
        }

    manifest_rows.sort(
        key=lambda item: (
            FORMAL_TASKS.index(item["task_id"]),
            item["selection_rank"],
        )
    )
    manifest_text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in manifest_rows
    )
    atomic_write_text(experiment_root / "data_manifest.jsonl", manifest_text)
    write_json(
        experiment_root / "manifests/data_selection_summary.json",
        {
            "protocol_id": PROTOCOL_ID,
            "status": "selected_raw_pending_replay",
            "formal_tasks": task_summary,
            "total_selected": len(manifest_rows),
        },
    )
    write_json(
        experiment_root / "manifests/raw_subset_files.json",
        {"protocol_id": PROTOCOL_ID, "files": subset_files},
    )
    print(json.dumps(task_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
