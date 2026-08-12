from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from boundarybc.checkpoint import atomic_write_json
from boundarybc.config import load_config
from boundarybc.data import (
    discover_episodes,
    fit_proprio_normalizer,
    load_task_arrays,
    make_batch,
    official_demo_path,
    split_episode_records,
)
from boundarybc.libero_runtime import configure_headless_runtime
from boundarybc.model import masked_action_mse
from boundarybc.provenance import verify_locked_inputs
from boundarybc.training import make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-GPU, batch-one R16-P18 dev14 smoke")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_headless_runtime(config.path.parents[1])
    device = torch.device(args.device)
    provenance = verify_locked_inputs(config, dataset_root=args.dataset_root)
    task_results: dict[str, dict[str, object]] = {}
    suite = str(config.raw["benchmark"]["suite"])
    # Establish every EGL context before any CUDA compute context is created.
    for task in config.tasks:
        demo_path = official_demo_path(args.dataset_root, suite, task.name)
        records = discover_episodes(demo_path)
        split = split_episode_records(records)
        replay_worker = _run_replay_worker(
            config_path=config.path,
            dataset_root=args.dataset_root,
            task_key=task.key,
            device=args.device,
        )
        replay = replay_worker["replay"]
        if float(replay["state_repeat_max_abs"]) > 1e-9:
            raise RuntimeError(f"non-deterministic state replay: {replay}")
        if int(replay["image_repeat_max_abs"]) > 1 or float(replay["image_repeat_mae"]) > 0.001:
            raise RuntimeError(f"non-deterministic image replay: {replay}")
        if float(replay["alignment_proprio_max_abs"]) > 0.003:
            raise RuntimeError(f"state/observation alignment mismatch: {replay}")
        if float(replay["alignment_image_mae"]) > 2.0:
            raise RuntimeError(f"state/observation image mismatch: {replay}")
        task_results[task.key] = {
            "task_name": task.name,
            "episodes": len(records),
            "split_counts": {key: len(value) for key, value in split.items()},
            "replay": replay,
            "egl_before_cuda_policy_step": replay_worker["egl_before_cuda_policy_step"],
        }
        print(json.dumps({"event": "SMOKE_REPLAY_COMPLETE", "task": task.key}, sort_keys=True), flush=True)

    for task in config.tasks:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA smoke requested but CUDA is unavailable")
        demo_path = official_demo_path(args.dataset_root, suite, task.name)
        records = discover_episodes(demo_path)
        arrays = load_task_arrays(demo_path, records[:1], action_horizon=8)
        normalizer = fit_proprio_normalizer(arrays.proprio)
        batch = make_batch(
            arrays,
            torch.tensor([0]),
            normalizer,
            device=device,
            augment=True,
        )
        model = make_model(config).to(device)
        model.train()
        use_bf16 = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            microtokens = model.encode_microtokens(batch["images"])
            prediction = model(batch["images"], batch["proprio"])
            loss = masked_action_mse(prediction, batch["actions"], batch["mask"])
        loss.backward()
        if microtokens.shape != (1, 64, 128) or prediction.shape != (1, 8, 7):
            raise RuntimeError(
                f"model contract mismatch: microtokens={microtokens.shape}, prediction={prediction.shape}"
            )
        task_results[task.key].update({
            "first_episode_frames": len(arrays),
            "microtokens": list(microtokens.shape),
            "prediction": list(prediction.shape),
            "loss": float(loss.detach().cpu()),
        })
        del model, arrays, batch, prediction, loss, microtokens
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"event": "SMOKE_TASK_COMPLETE", "task": task.key}, sort_keys=True), flush=True)
    result = {
        "schema_version": 1,
        "event": "DEV14_ONE_GPU_SMOKE_COMPLETE",
        "protocol_id": config.protocol_id,
        "config_sha256": config.sha256,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "batch_size": 1,
        "gpu_count": 1 if device.type == "cuda" else 0,
        "provenance": provenance,
        "tasks": task_results,
        "adaptive_components_implemented": False,
    }
    output = Path(args.output)
    atomic_write_json(output, result)
    print(json.dumps({"event": result["event"], "output": str(output)}, sort_keys=True), flush=True)


def _run_replay_worker(
    *,
    config_path: Path,
    dataset_root: str,
    task_key: str,
    device: str,
) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "boundarybc.replay_worker",
        "--config",
        str(config_path),
        "--dataset-root",
        str(Path(dataset_root).resolve()),
        "--task-key",
        task_key,
        "--device",
        device,
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"replay worker emitted no result for {task_key}")
    result = json.loads(lines[-1])
    if result.get("event") != "REPLAY_WORKER_COMPLETE":
        raise RuntimeError(f"unexpected replay worker result for {task_key}: {result}")
    return result


if __name__ == "__main__":
    main()
