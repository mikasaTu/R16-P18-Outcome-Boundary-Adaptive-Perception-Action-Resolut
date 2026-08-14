#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from action_runtime import PADDED_ENVS, generate_atlas, load_training_chunks
from common import MODEL_SEEDS, PROTOCOL_ID, append_jsonl, sha256_file, write_json
from oracle_math import RADIUS_CANDIDATES
from stage25_runtime import load_policy_from_checkpoint, make_env
from state_bank_common import h5_full


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("calibration", "confirmatory", "post_success_diagnostic"),
        required=True,
    )
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--selected-checkpoints", type=Path, required=True)
    parser.add_argument("--state-bank-manifest", type=Path, required=True)
    parser.add_argument("--training-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-calibration-freeze", type=Path)
    parser.add_argument("--max-states", type=int)
    return parser.parse_args()


def selected_row(path: Path, seed: int) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value["groups"][f"StackCube-v1/seed_{seed}"]["selected"]


def main() -> None:
    args = parse_args()
    if args.model_seed not in MODEL_SEEDS:
        raise ValueError("model seed outside frozen set")
    if not torch.cuda.is_available():
        raise RuntimeError("formal action probe requires CUDA")
    manifest = json.loads(args.state_bank_manifest.read_text(encoding="utf-8"))
    if manifest["bank"] != args.stage or manifest["task_id"] != "StackCube-v1":
        raise RuntimeError("state bank/stage mismatch")
    h5_path = Path(manifest["state_bank_h5"])
    if sha256_file(h5_path) != manifest["state_bank_h5_sha256"]:
        raise RuntimeError("state-bank HDF5 digest mismatch")
    radii = RADIUS_CANDIDATES
    freeze_sha = None
    if args.stage in {"confirmatory", "post_success_diagnostic"}:
        if args.action_calibration_freeze is None:
            raise ValueError("frozen-radius probe requires action calibration freeze")
        freeze = json.loads(args.action_calibration_freeze.read_text(encoding="utf-8"))
        if freeze.get("status") != "ACTION_CALIBRATION_FROZEN":
            raise RuntimeError("action calibration is not frozen")
        radii = (float(freeze["selected_radius"]),)
        freeze_sha = sha256_file(args.action_calibration_freeze)
    rows = manifest["states"]
    if args.max_states is not None:
        rows = rows[: args.max_states]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "states.jsonl"
    if raw_path.exists():
        raise FileExistsError(f"refusing to overwrite formal raw output: {raw_path}")
    training_chunks = load_training_chunks(str(args.training_h5))
    device = torch.device("cuda")
    rollout_env = make_env(
        "StackCube-v1", PADDED_ENVS, sim_backend="physx_cpu", reconfiguration_freq=0
    )
    policy_env = make_env(
        "StackCube-v1", 1, sim_backend="physx_cpu", reconfiguration_freq=0
    )
    selected = selected_row(args.selected_checkpoints, args.model_seed)
    agent, _ = load_policy_from_checkpoint(
        rollout_env,
        "StackCube-v1",
        args.model_seed,
        Path(selected["checkpoint_path"]),
        device,
        selected["checkpoint_sha256"],
    )
    started = time.time()
    written = 0
    try:
        with h5py.File(h5_path, "r") as source:
            for metadata in rows:
                state = h5_full(source[f"{metadata['bank_id']}/env_state"])
                for radius in radii:
                    atlas = generate_atlas(
                        policy_env,
                        rollout_env,
                        agent,
                        state,
                        int(metadata["source_episode_seed"]),
                        training_chunks,
                        device,
                        radius=float(radius),
                    )
                    row = {
                        "protocol_id": PROTOCOL_ID,
                        "stage": args.stage,
                        "model_seed": args.model_seed,
                        "bank_id": metadata["bank_id"],
                        "phase": metadata["phase"],
                        "source": metadata["source"],
                        "post_success": metadata["phase"] == "post_success",
                        "state_sha256": metadata["state_sha256"],
                        "selected_checkpoint_step": int(selected["step"]),
                        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
                        "action_calibration_freeze_sha256": freeze_sha,
                        "atlas": atlas,
                    }
                    append_jsonl(raw_path, row)
                    written += 1
                    if written == 1:
                        write_json(
                            args.output_dir / "FIRST_REAL_STATE.json",
                            {
                                "protocol_id": PROTOCOL_ID,
                                "status": "FIRST_REAL_ACTION_ATLAS_STATE_COMPLETE",
                                "bank_id": metadata["bank_id"],
                                "model_seed": args.model_seed,
                                "raw_sha256": sha256_file(raw_path),
                            },
                        )
                print(
                    f"ACTION_PROBE_PROGRESS stage={args.stage} model_seed={args.model_seed} "
                    f"states={written}/{len(rows) * len(radii)}",
                    flush=True,
                )
    finally:
        policy_env.close()
        rollout_env.close()
    values = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    summary = {
        "protocol_id": PROTOCOL_ID,
        "status": "ACTION_BOUNDARY_PROBE_COMPLETE",
        "stage": args.stage,
        "model_seed": args.model_seed,
        "state_radius_rows": len(values),
        "radii": list(radii),
        "mean_candidate_validity": float(
            np.mean([np.mean(row["atlas"]["valid"]) for row in values])
        ),
        "raw_path": str(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "state_bank_manifest_sha256": sha256_file(args.state_bank_manifest),
        "training_h5_sha256": sha256_file(args.training_h5),
        "selected_checkpoint": selected,
        "wall_seconds": time.time() - started,
    }
    write_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
