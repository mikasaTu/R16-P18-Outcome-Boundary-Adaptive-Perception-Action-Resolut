#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from common import PROTOCOL_ID, atomic_json, sha256_file

TASKS = ["StackCube-v1", "PegInsertionSide-v1", "PlugCharger-v1", "PullCubeTool-v1", "PushT-v1", "PushCube-v1"]
SPLITS = {"train": 200, "validation": 50, "test": 50}
PREDECESSORS = [
    "experiments/maniskill_act_boundary_screen_v1",
    "experiments/maniskill_stage25_repair_oracle_v1",
    "experiments/maniskill_stage26_counterfactual_completion_v1",
]


def git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rows, all_ids, all_seeds, all_hashes = [], set(), set(), set()
    for task in TASKS:
        task_ids, task_seeds, task_hashes = set(), set(), set()
        for split, expected in SPLITS.items():
            h5 = args.dataset_root / "selected_raw" / task / split / "trajectory.h5"
            meta = h5.with_suffix(".json")
            data = json.loads(meta.read_text(encoding="utf-8"))
            episodes = data["episodes"]
            if len(episodes) != expected:
                raise RuntimeError(f"{task}/{split}: {len(episodes)} != {expected}")
            h5_sha, meta_sha = sha256_file(h5), sha256_file(meta)
            for episode in episodes:
                source_id = f"{task}:{episode['source_episode_id']}"
                seed_id = f"{task}:{episode['episode_seed']}"
                state_id = f"{task}:{episode['source_initial_state_sha256']}"
                if source_id in task_ids or seed_id in task_seeds or state_id in task_hashes:
                    raise RuntimeError(f"within-task identity collision: {task}/{split}")
                task_ids.add(source_id); task_seeds.add(seed_id); task_hashes.add(state_id)
                rows.append({
                    "task": task, "split": split,
                    "source_trajectory_id": source_id,
                    "episode_seed": int(episode["episode_seed"]),
                    "initial_state_sha256": episode["source_initial_state_sha256"],
                    "success": bool(episode["success"]),
                    "h5": str(h5), "h5_sha256": h5_sha,
                    "json": str(meta), "json_sha256": meta_sha,
                })
            if not all(bool(row["success"]) for row in rows if row["task"] == task and row["split"] == split):
                raise RuntimeError(f"unsuccessful source admitted: {task}/{split}")
        if len(task_ids) != 300 or len(task_seeds) != 300 or len(task_hashes) != 300:
            raise RuntimeError(f"{task}: uniqueness gate failed")
        all_ids.update(task_ids); all_seeds.update(task_seeds); all_hashes.update(task_hashes)
    trees = {}
    for path in PREDECESSORS:
        trees[path] = git("rev-parse", f"HEAD:{path}", cwd=args.repo)
    output = {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "task_count": len(TASKS),
        "episodes_per_task": 300,
        "total_rows": len(rows),
        "split_counts": SPLITS,
        "unique_source_ids": len(all_ids),
        "unique_task_seed_pairs": len(all_seeds),
        "unique_task_initial_state_hashes": len(all_hashes),
        "predecessor_tree_hashes": trees,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "data_identity_manifest.json", output, args.overwrite)
    atomic_json(args.output_dir / "predecessor_tree_freeze.json", {"protocol_id": PROTOCOL_ID, "head": git("rev-parse", "HEAD", cwd=args.repo), "trees": trees}, args.overwrite)
    print(json.dumps({k: output[k] for k in output if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
