from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from boundarybc.checkpoint import atomic_write_json
from boundarybc.config import load_config
from boundarybc.libero_runtime import configure_headless_runtime
from boundarybc.provenance import verify_locked_inputs
from boundarybc.reporting import build_baseline_gate_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the locked BoundaryBC-S health gate on PAI")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--devices",
        help="Comma-separated independent worker devices; overrides --device",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    project_root = config.path.parents[1]
    configure_headless_runtime(project_root)
    _validate_paths(args, project_root)
    devices = _parse_devices(args)
    locked_inputs = verify_locked_inputs(config, dataset_root=args.dataset_root)
    manifest_path = Path(args.log_root) / args.run_id / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "protocol_id": config.protocol_id,
            "config_sha256": config.sha256,
            "run_id": args.run_id,
            "project_root": str(project_root),
            "git_commit": _git(project_root, "rev-parse", "HEAD"),
            "git_head_tree": _git(project_root, "rev-parse", "HEAD^{tree}"),
            "git_status_porcelain": _git(project_root, "status", "--porcelain"),
            "python": platform.python_version(),
            "torch": config.raw["runtime"]["torch"],
            "cuda": config.raw["runtime"]["cuda_runtime"],
            "requested_devices": list(devices),
            "worker_parallelism": len(devices),
            "per_model_batch_size": int(config.raw["training"]["batch_size"]),
            "data_parallel": False,
            "uid": os.getuid(),
            "gid": os.getgid(),
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "checkpoint_root": str(Path(args.checkpoint_root).resolve()),
            "log_root": str(Path(args.log_root).resolve()),
            "pai_probe_created": False,
            "locked_inputs": locked_inputs,
        },
    )
    _require_leon_owner(manifest_path)
    print(json.dumps({"event": "RUN_MANIFEST", "path": str(manifest_path)}), flush=True)
    units = [
        (task.key, seed)
        for task in config.tasks
        for seed in config.training_seeds
    ]
    _run_phase(args, phase="train", units=units, devices=devices)
    _run_phase(args, phase="evaluate", units=units, devices=devices)
    report_path, report = build_baseline_gate_report(
        config,
        run_id=args.run_id,
        log_root=args.log_root,
    )
    print(
        json.dumps(
            {
                "event": "BASELINE_GATE_COMPLETE",
                "decision": report["decision"],
                "path": str(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _validate_paths(args: argparse.Namespace, project_root: Path) -> None:
    new_root = Path("/mnt/cpfs/zbl-cpfs-new")
    dataset_root = Path(args.dataset_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    log_root = Path(args.log_root).resolve()
    if not project_root.is_relative_to(new_root / "USERS" / "leon" / "code"):
        raise RuntimeError(f"project root violates the new storage contract: {project_root}")
    if not dataset_root.is_relative_to(new_root / "dataset" / "leon"):
        raise RuntimeError(f"dataset root violates the new storage contract: {dataset_root}")
    if not checkpoint_root.is_relative_to(new_root / "CKPT" / "leon"):
        raise RuntimeError(f"checkpoint root violates the new storage contract: {checkpoint_root}")
    if not log_root.is_relative_to(new_root / "USERS" / "leon" / "logs"):
        raise RuntimeError(f"log root violates the new storage contract: {log_root}")
    for path in (dataset_root, checkpoint_root, log_root):
        if not path.exists():
            raise FileNotFoundError(path)


def _git(project_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project_root), *args],
        text=True,
    ).strip()


def _run_worker(
    args: argparse.Namespace,
    *,
    phase: str,
    task_key: str,
    model_seed: int,
    device: str,
) -> None:
    command = [
        sys.executable,
        "-m",
        "boundarybc.worker",
        phase,
        "--config",
        str(Path(args.config).resolve()),
        "--run-id",
        args.run_id,
        "--task-key",
        task_key,
        "--model-seed",
        str(model_seed),
        "--dataset-root",
        str(Path(args.dataset_root).resolve()),
        "--checkpoint-root",
        str(Path(args.checkpoint_root).resolve()),
        "--log-root",
        str(Path(args.log_root).resolve()),
        "--device",
        device,
    ]
    print(
        json.dumps(
            {
                "event": "WORKER_START",
                "phase": phase,
                "task_key": task_key,
                "model_seed": model_seed,
                "device": device,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    subprocess.run(command, check=True)


def _run_phase(
    args: argparse.Namespace,
    *,
    phase: str,
    units: list[tuple[str, int]],
    devices: tuple[str, ...],
) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        assignments = _assign_units(units, devices)
        futures = [
            executor.submit(
                _run_device_queue,
                args,
                phase=phase,
                device=device,
                units=device_units,
            )
            for device, device_units in assignments
        ]
        for future in futures:
            future.result()


def _run_device_queue(
    args: argparse.Namespace,
    *,
    phase: str,
    device: str,
    units: tuple[tuple[str, int], ...],
) -> None:
    for task_key, model_seed in units:
        _run_worker(
            args,
            phase=phase,
            task_key=task_key,
            model_seed=model_seed,
            device=device,
        )


def _assign_units(
    units: list[tuple[str, int]],
    devices: tuple[str, ...],
) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
    return tuple(
        (device, tuple(units[index:: len(devices)]))
        for index, device in enumerate(devices)
    )


def _parse_devices(args: argparse.Namespace) -> tuple[str, ...]:
    raw = args.devices if args.devices is not None else args.device
    devices = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not devices:
        raise ValueError("at least one worker device is required")
    if len(set(devices)) != len(devices):
        raise ValueError(f"duplicate worker devices are forbidden: {devices}")
    return devices


def _require_leon_owner(path: Path) -> None:
    stat = path.lstat()
    if (stat.st_uid, stat.st_gid) != (2254, 2254):
        raise RuntimeError(
            f"persistent artifact is not Leon-owned: {path} -> {stat.st_uid}:{stat.st_gid}"
        )


if __name__ == "__main__":
    main()
