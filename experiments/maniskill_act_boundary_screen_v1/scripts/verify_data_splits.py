#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import h5py

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from protocol_common import (  # noqa: E402
    FORMAL_TASKS,
    PROTOCOL_ID,
    SPLIT_COUNTS,
    sha256_file,
    sha256_hdf5_group,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--verify-source-files",
        action="store_true",
        help="also re-hash the larger upstream source HDF5 files",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(experiment_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = experiment_root / "data_manifest.jsonl"
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(len(rows) == len(FORMAL_TASKS) * sum(SPLIT_COUNTS.values()), "row count")
    require({row["protocol_id"] for row in rows} == {PROTOCOL_ID}, "protocol id")
    require({row["task_id"] for row in rows} == set(FORMAL_TASKS), "formal task set")

    results: dict[str, Any] = {}
    for task_id in FORMAL_TASKS:
        task_rows = [row for row in rows if row["task_id"] == task_id]
        require(len(task_rows) == sum(SPLIT_COUNTS.values()), f"{task_id}: selected count")
        require(
            Counter(row["split"] for row in task_rows) == Counter(SPLIT_COUNTS),
            f"{task_id}: split counts",
        )
        for key in (
            "episode_seed",
            "initial_state_sha256",
            "source_episode_id",
            "source_trajectory_sha256",
            "selection_key",
        ):
            require(
                len({row[key] for row in task_rows}) == len(task_rows),
                f"{task_id}: {key} is not unique",
            )
        require(all(row["source_success"] is True for row in task_rows), f"{task_id}: success")
        require(
            [row["selection_rank"] for row in task_rows] == list(range(len(task_rows))),
            f"{task_id}: selection ranks",
        )
        for split, count in SPLIT_COUNTS.items():
            split_rows = [row for row in task_rows if row["split"] == split]
            require(
                [row["split_index"] for row in split_rows] == list(range(count)),
                f"{task_id}/{split}: split indices",
            )
        results[task_id] = {
            "selected": len(task_rows),
            "splits": dict(Counter(row["split"] for row in task_rows)),
            "unique_episode_seeds": len({row["episode_seed"] for row in task_rows}),
            "unique_initial_states": len({row["initial_state_sha256"] for row in task_rows}),
            "unique_trajectories": len({row["source_trajectory_sha256"] for row in task_rows}),
        }
    return rows, results


def verify_subset_files(
    experiment_root: Path,
    manifest_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lock = load_json(experiment_root / "manifests/raw_subset_files.json")
    require(lock["protocol_id"] == PROTOCOL_ID, "subset lock protocol id")
    require(len(lock["files"]) == len(FORMAL_TASKS) * len(SPLIT_COUNTS), "subset file count")
    verified: list[dict[str, Any]] = []
    for record in lock["files"]:
        h5_path = Path(record["h5_path"])
        json_path = Path(record["json_path"])
        require(h5_path.is_file() and json_path.is_file(), f"missing subset: {h5_path}")
        require(sha256_file(h5_path) == record["h5_sha256"], f"HDF5 digest: {h5_path}")
        require(sha256_file(json_path) == record["json_sha256"], f"JSON digest: {json_path}")
        metadata = load_json(json_path)
        require(metadata["protocol_id"] == PROTOCOL_ID, f"protocol id: {json_path}")
        require(metadata["split"] == record["split"], f"split: {json_path}")
        if record["task_id"] == "PlugCharger-v1":
            require(
                metadata["env_info"]["env_kwargs"]["reward_mode"] == "sparse",
                f"PlugCharger reward-mode adapter: {json_path}",
            )
            require(
                metadata.get("protocol_metadata_adapter", {}).get("version")
                == "plug_charger_reward_mode_v1",
                f"PlugCharger metadata adapter version: {json_path}",
            )
        episodes = sorted(metadata["episodes"], key=lambda item: int(item["episode_id"]))
        expected_rows = [
            row
            for row in manifest_rows
            if row["task_id"] == record["task_id"] and row["split"] == record["split"]
        ]
        require(len(episodes) == record["episodes"] == len(expected_rows), f"episode count: {h5_path}")
        with h5py.File(h5_path, "r") as handle:
            require(set(handle) == {f"traj_{index}" for index in range(len(episodes))}, f"groups: {h5_path}")
            for index, (episode, row) in enumerate(zip(episodes, expected_rows, strict=True)):
                require(int(episode["episode_id"]) == index, f"episode id: {h5_path}/{index}")
                require(int(episode["source_episode_id"]) == row["source_episode_id"], f"source id: {h5_path}/{index}")
                require(episode["selection_key"] == row["selection_key"], f"selection key: {h5_path}/{index}")
                require(
                    sha256_hdf5_group(handle[f"traj_{index}"]) == row["source_trajectory_sha256"],
                    f"trajectory digest: {h5_path}/{index}",
                )
        verified.append(
            {
                "task_id": record["task_id"],
                "split": record["split"],
                "episodes": len(episodes),
                "h5_sha256": record["h5_sha256"],
                "json_sha256": record["json_sha256"],
            }
        )
    return verified


def verify_sources(experiment_root: Path) -> list[dict[str, str]]:
    summary = load_json(experiment_root / "manifests/data_selection_summary.json")
    verified: list[dict[str, str]] = []
    for task_id in FORMAL_TASKS:
        source = summary["formal_tasks"][task_id]["source"]
        for kind in ("h5", "json"):
            path = Path(source[f"{kind}_path"])
            require(path.is_file(), f"missing source: {path}")
            require(sha256_file(path) == source[f"{kind}_sha256"], f"source digest: {path}")
        verified.append({"task_id": task_id, "h5_sha256": source["h5_sha256"]})
    return verified


def main() -> None:
    args = parse_args()
    rows, manifest = verify_manifest(args.experiment_root)
    result: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "manifest": manifest,
        "subset_files": verify_subset_files(args.experiment_root, rows),
    }
    if args.verify_source_files:
        result["source_files"] = verify_sources(args.experiment_root)
    if args.output is not None:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
