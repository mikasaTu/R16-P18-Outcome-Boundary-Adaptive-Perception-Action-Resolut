#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import PROTOCOL_ID, append_jsonl, sha256_bytes, write_json_new
from stage26_runtime import branch_rollout, load_capsule, load_policy_from_checkpoint, make_env, quaternion_distance_rad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--capsule-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--states", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    marker = args.output_dir / "SHARED_PREFIX_FIDELITY.json"
    if marker.exists(): print("FIDELITY_ALREADY_COMPLETE"); return
    rows = [json.loads(line) for line in args.capsule_manifest.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: (row["episode_seed"], row["capture_type"], row["source_step"], row["capsule_id"]))
    rows = [row for row in rows if row["reference_future_steps"] >= 10][:args.states]
    if len(rows) != args.states: raise RuntimeError(f"need {args.states} capsules with ten reference steps, got {len(rows)}")
    device = torch.device("cuda")
    env = make_env("StackCube-v1", 1, sim_backend="physx_cuda", reconfiguration_freq=0)
    agent, _ = load_policy_from_checkpoint(env, "StackCube-v1", args.model_seed, args.checkpoint, device, args.checkpoint_sha256)
    raw = args.output_dir / "shared_prefix_fidelity_raw.jsonl"
    maxima = {"action": 0.0, "translation": 0.0, "rotation": 0.0}; categorical_ok = 0; categorical_total = 0; obs_ok = 0; obs_total = 0
    try:
        for row in rows:
            capsule = load_capsule(Path(row["path"]))
            result = branch_rollout(env, agent, capsule, device, "continue_policy", horizon=10)
            reference = capsule.reference_future[:10]
            for expected, actual in zip(reference, result["trace"], strict=True):
                action_error = float(np.max(np.abs(np.asarray(expected["executed_action"]) - np.asarray(actual["executed_action"]))))
                translation = float(np.linalg.norm(np.asarray(expected["object_position"]) - np.asarray(actual["object_position"])))
                rotation = quaternion_distance_rad(expected["object_quaternion"], actual["object_quaternion"])
                agreement = all(bool(expected[key]) == bool(actual[key]) for key in ("success", "contact", "grasped", "supported"))
                obs_agreement = expected["observation_sha256"] == actual["observation_sha256"]
                maxima["action"] = max(maxima["action"], action_error); maxima["translation"] = max(maxima["translation"], translation); maxima["rotation"] = max(maxima["rotation"], rotation)
                categorical_ok += int(agreement); categorical_total += 1; obs_ok += int(obs_agreement); obs_total += 1
                append_jsonl(raw, {"protocol_id": PROTOCOL_ID, "capsule_id": capsule.capsule_id, "step": actual["step"], "action_max_abs": action_error, "object_translation_m": translation, "object_rotation_rad": rotation, "categorical_agreement": agreement, "observation_hash_agreement": obs_agreement, "act_table_prefix_sha256": sha256_bytes(np.ascontiguousarray(capsule.temporal_table_prefix).tobytes()), "rng_fields_present": sorted(capsule.rng_states)})
    finally: env.close()
    passed = bool(maxima["action"] <= 1e-6 and maxima["translation"] <= 1e-5 and maxima["rotation"] <= 1e-4 and categorical_ok == categorical_total)
    write_json_new(marker, {"protocol_id": PROTOCOL_ID, "status": "SHARED_PREFIX_FIDELITY_PASS" if passed else "NO_GO_SHARED_PREFIX_FIDELITY", "pass": passed, "states": len(rows), "steps_per_state": 10, "max_executed_action_abs": maxima["action"], "max_object_translation_m": maxima["translation"], "max_object_rotation_rad": maxima["rotation"], "categorical_agreement": categorical_ok / categorical_total, "observation_hash_agreement": obs_ok / obs_total, "thresholds": {"action": 1e-6, "translation": 1e-5, "rotation": 1e-4, "categorical": 1.0}})


if __name__ == "__main__": main()
