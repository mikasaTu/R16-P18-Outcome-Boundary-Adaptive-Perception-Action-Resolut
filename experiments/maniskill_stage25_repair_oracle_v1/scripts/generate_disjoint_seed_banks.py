#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import PROTOCOL_ID, TASKS, sha256_file, unique_hash_seeds, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-data-manifest", type=Path, required=True)
    parser.add_argument("--old-selection-summary", type=Path, required=True)
    parser.add_argument("--official-demo-json-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_demo_seeds(path: Path) -> dict[str, set[int]]:
    result = {task: set() for task in TASKS}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            task = row.get("task_id")
            if task in result:
                result[task].add(int(row["episode_seed"]))
    for task, values in result.items():
        if len(values) != 300:
            raise RuntimeError(f"expected 300 predecessor demo seeds for {task}, got {len(values)}")
    return result


def official_episodes(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    episodes = value.get("episodes")
    if not isinstance(episodes, list):
        raise RuntimeError(f"missing episodes in {path}")
    return episodes


def main() -> None:
    args = parse_args()
    old_demo = load_demo_seeds(args.old_data_manifest)
    old_summary = json.loads(args.old_selection_summary.read_text(encoding="utf-8"))
    globally_used: set[int] = set()
    for task in TASKS:
        globally_used.update(old_demo[task])
        old_test = old_summary["formal_tasks"][task]["closed_loop_test_seeds"]
        if len(old_test) != 100:
            raise RuntimeError(f"invalid predecessor test bank for {task}")
        globally_used.update(int(value) for value in old_test)

    banks: dict[str, dict[str, Any]] = {
        "checkpoint_screen_seed_bank.json": {},
        "checkpoint_final_val_seed_bank.json": {},
        "confirmatory_test_seed_bank.json": {},
        "oracle_source_seed_bank.json": {},
    }
    for task in TASKS:
        final_validation = unique_hash_seeds(
            f"{task}:checkpoint_final_val", 100, globally_used
        )
        confirmatory = unique_hash_seeds(
            f"{task}:confirmatory_test", 100, globally_used
        )
        oracle = unique_hash_seeds(f"{task}:oracle_source", 512, globally_used)
        banks["checkpoint_screen_seed_bank.json"][task] = final_validation[:32]
        banks["checkpoint_final_val_seed_bank.json"][task] = final_validation
        banks["confirmatory_test_seed_bank.json"][task] = confirmatory
        banks["oracle_source_seed_bank.json"][task] = {
            "simulator_seeds": oracle,
            "expert_source_episodes": [],
        }

    stack_json = args.official_demo_json_root / "StackCube-v1" / "motionplanning" / "trajectory.json"
    candidates = []
    for episode in official_episodes(stack_json):
        seed = int(episode["episode_seed"])
        episode_id = int(episode["episode_id"])
        if seed in globally_used or seed in old_demo["StackCube-v1"]:
            continue
        key = __import__("hashlib").sha256(
            f"{PROTOCOL_ID}|StackCube-v1|expert|{episode_id}|{seed}".encode()
        ).hexdigest()
        candidates.append((key, episode_id, seed))
    candidates.sort()
    if len(candidates) < 96:
        raise RuntimeError(f"only {len(candidates)} eligible expert source episodes")
    selected = candidates[:96]
    expert_seeds = [seed for _, _, seed in selected]
    if any(seed in globally_used for seed in expert_seeds):
        raise AssertionError("expert source collision")
    globally_used.update(expert_seeds)
    banks["oracle_source_seed_bank.json"]["StackCube-v1"]["expert_source_episodes"] = [
        {"episode_id": episode_id, "episode_seed": seed, "selection_sha256": key}
        for key, episode_id, seed in selected
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bindings = {
        "protocol_id": PROTOCOL_ID,
        "algorithm": "sha256_low31_disjoint_v1",
        "predecessor_data_manifest": str(args.old_data_manifest),
        "predecessor_data_manifest_sha256": sha256_file(args.old_data_manifest),
        "predecessor_selection_summary": str(args.old_selection_summary),
        "predecessor_selection_summary_sha256": sha256_file(args.old_selection_summary),
        "official_stackcube_json": str(stack_json),
        "official_stackcube_json_sha256": sha256_file(stack_json),
        "screen_is_first_32_of_final_validation": True,
        "globally_disjoint_except_declared_screen_prefix": True,
        "model_seed_order_reused": True,
    }
    for filename, task_values in banks.items():
        payload = {"protocol_id": PROTOCOL_ID, "bindings": bindings, "tasks": task_values}
        write_json(args.output_dir / filename, payload)
    write_json(
        args.output_dir / "seed_bank_audit.json",
        {
            "protocol_id": PROTOCOL_ID,
            "status": "SEED_BANK_AUDIT_PASS",
            "globally_reserved_seed_count": len(globally_used),
            "declared_exception": "checkpoint_screen_is_prefix_of_checkpoint_final_val",
            "files": {
                filename: sha256_file(args.output_dir / filename) for filename in sorted(banks)
            },
        },
    )


if __name__ == "__main__":
    main()

