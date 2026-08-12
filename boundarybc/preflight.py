from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from boundarybc.checkpoint import atomic_write_json, discover_complete_checkpoints
from boundarybc.config import load_config


LEON_UID = 2254
LEON_GID = 2254


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the PAI run identity and resume inventory")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-run-dir", required=True)
    parser.add_argument("--log-run-dir", required=True)
    parser.add_argument("--cache-run-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if (os.getuid(), os.getgid()) != (LEON_UID, LEON_GID):
        raise RuntimeError(
            f"workload identity must be {LEON_UID}:{LEON_GID}, got {os.getuid()}:{os.getgid()}"
        )
    write_paths = tuple(
        Path(value).resolve()
        for value in (args.checkpoint_run_dir, args.log_run_dir, args.cache_run_dir)
    )
    for directory in write_paths:
        _ownership_probe(directory)

    resume_inventory: dict[str, list[int]] = {}
    model_root = write_paths[0] / "models"
    for task in config.tasks:
        for seed in config.training_seeds:
            key = f"{task.key}/seed_{seed}"
            checkpoints = discover_complete_checkpoints(model_root / task.key / f"seed_{seed}")
            resume_inventory[key] = [step for step, _ in checkpoints]
    auto_resume = any(resume_inventory.values())
    artifact = write_paths[1] / "launcher_preflight.json"
    atomic_write_json(
        artifact,
        {
            "schema_version": 1,
            "event": "PAI_LAUNCHER_PREFLIGHT_COMPLETE",
            "run_id": args.run_id,
            "protocol_id": config.protocol_id,
            "config_sha256": config.sha256,
            "uid": os.getuid(),
            "gid": os.getgid(),
            "write_paths": [str(path) for path in write_paths],
            "auto_resume": auto_resume,
            "resume_inventory": resume_inventory,
            "pai_automatic_fault_tolerance": False,
        },
    )
    _require_owner(artifact)
    print(f"RUN_ID={args.run_id}", flush=True)
    print(f"AUTO_RESUME={int(auto_resume)}", flush=True)
    print(f"RESUME_STEP_MAP={json.dumps(resume_inventory, sort_keys=True)}", flush=True)
    print(f"RESUME_DIR={model_root}", flush=True)
    print("PAI_AUTO_FAULT_TOLERANCE=0", flush=True)
    print(f"FIRST_ARTIFACT={artifact}", flush=True)
    print(f"FIRST_ARTIFACT_OWNER={LEON_UID}:{LEON_GID}", flush=True)


def _ownership_probe(directory: Path) -> None:
    if not directory.is_dir() or not os.access(directory, os.W_OK):
        raise RuntimeError(f"persistent run directory is not writable: {directory}")
    probe = directory / f".leon_owner_probe.{os.getpid()}"
    descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    try:
        _require_owner(probe)
    finally:
        probe.unlink()


def _require_owner(path: Path) -> None:
    stat = path.lstat()
    if (stat.st_uid, stat.st_gid) != (LEON_UID, LEON_GID):
        raise RuntimeError(
            f"artifact ownership mismatch for {path}: {stat.st_uid}:{stat.st_gid}"
        )


if __name__ == "__main__":
    main()
