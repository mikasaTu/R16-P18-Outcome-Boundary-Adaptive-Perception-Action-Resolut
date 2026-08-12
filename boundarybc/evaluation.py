from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from boundarybc.checkpoint import atomic_write_json, validate_final_model
from boundarybc.config import ExperimentConfig
from boundarybc.data import ProprioNormalizer
from boundarybc.libero_runtime import (
    load_task_init_states,
    make_offscreen_env,
    observation_tensors,
)
from boundarybc.training import final_model_path, make_model
from boundarybc.training import _init_wandb_run


def evaluation_result_path(log_root: str | Path, run_id: str, task_key: str, seed: int) -> Path:
    return Path(log_root) / run_id / "evaluation" / task_key / f"seed_{seed}.jsonl"


def evaluate_one_model(
    config: ExperimentConfig,
    *,
    task_key: str,
    model_seed: int,
    run_id: str,
    checkpoint_root: str | Path,
    log_root: str | Path,
    device: torch.device,
    gpu_device_id: int = 0,
) -> Path:
    task = config.task(task_key)
    model_path = final_model_path(checkpoint_root, run_id, task_key, model_seed)
    if not validate_final_model(model_path):
        raise RuntimeError(f"final model is absent or incomplete: {model_path}")
    # Load and validate on CPU first. LIBERO's EGL context must be created
    # before this process initializes a CUDA compute context.
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    for key, expected in {
        "run_id": run_id,
        "task_key": task_key,
        "model_seed": model_seed,
        "config_sha256": config.sha256,
    }.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"evaluation checkpoint contract mismatch: {key}")
    model = make_model(config)
    model.load_state_dict(checkpoint["model"])
    normalizer = ProprioNormalizer.from_dict(checkpoint["normalizer"])
    result_path = evaluation_result_path(log_root, run_id, task_key, model_seed)
    existing = _read_episode_records(result_path)
    init_states = load_task_init_states(config, task)
    env = make_offscreen_env(config, task, gpu_device_id=gpu_device_id)
    try:
        model = model.to(device)
        model.eval()
        use_bf16 = config.raw["training"].get("precision") == "bf16" and device.type == "cuda"
        for episode_id, init_state in enumerate(init_states):
            if episode_id in existing:
                continue
            env.seed(episode_id)
            env.reset()
            observation = env.set_init_state(init_state)
            for _ in range(int(config.raw["benchmark"]["wait_steps"])):
                observation, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
            success = bool(env.check_success())
            executed_steps = 0
            policy_calls = 0
            inference_seconds = 0.0
            start = time.perf_counter()
            while not success and executed_steps < int(config.raw["benchmark"]["task_horizon"]):
                image, proprio = observation_tensors(observation, normalizer, device=device)
                call_start = time.perf_counter()
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_bf16,
                ):
                    chunk = model(image, proprio)[0].float().cpu().numpy()
                inference_seconds += time.perf_counter() - call_start
                policy_calls += 1
                execute_count = min(
                    int(config.raw["model"]["execute_horizon"]),
                    int(config.raw["benchmark"]["task_horizon"]) - executed_steps,
                )
                for action in chunk[:execute_count]:
                    observation, _, _, _ = env.step(np.clip(action, -1.0, 1.0))
                    executed_steps += 1
                    success = bool(env.check_success())
                    if success:
                        break
            record = {
                "schema_version": 1,
                "event": "COMPLETED_EVALUATION_EPISODE",
                "run_id": run_id,
                "config_sha256": config.sha256,
                "task_key": task_key,
                "task_name": task.name,
                "model_seed": model_seed,
                "episode_id": episode_id,
                "success": success,
                "executed_steps": executed_steps,
                "policy_calls": policy_calls,
                "inference_seconds": inference_seconds,
                "wall_seconds": time.perf_counter() - start,
            }
            _append_jsonl(result_path, record)
            if episode_id == 0:
                atomic_write_json(
                    result_path.with_suffix(".first_rollout.json"),
                    {
                        **record,
                        "event": "PERSISTED_FIRST_COMPLETED_ROLLOUT",
                        "result_jsonl": str(result_path),
                        "uid": os.getuid(),
                        "gid": os.getgid(),
                    },
                )
            existing[episode_id] = record
            print(json.dumps(record, sort_keys=True), flush=True)
    finally:
        env.close()
    records = _read_episode_records(result_path)
    if set(records) != set(range(50)):
        raise RuntimeError(f"evaluation is incomplete for {task_key}/seed {model_seed}")
    summary_path = result_path.with_suffix(".summary.json")
    atomic_write_json(
        summary_path,
        {
            "schema_version": 1,
            "event": "PERSISTED_COMPLETED_EVALUATION_RESULT",
            "run_id": run_id,
            "config_sha256": config.sha256,
            "task_key": task_key,
            "task_name": task.name,
            "model_seed": model_seed,
            "episodes": 50,
            "successes": sum(bool(record["success"]) for record in records.values()),
            "success_rate": float(np.mean([record["success"] for record in records.values()])),
            "result_jsonl": str(result_path),
        },
    )
    wandb_run = _init_wandb_run(
        config,
        run_id=run_id,
        task_key=task_key,
        task_name=task.name,
        model_seed=model_seed,
        job_type="evaluation",
    )
    if wandb_run is not None:
        successes = sum(bool(record["success"]) for record in records.values())
        wandb_run.log(
            {
                "evaluation/episodes": 50,
                "evaluation/successes": successes,
                "evaluation/success_rate": successes / 50.0,
            },
            step=int(checkpoint["global_step"]) + 1,
        )
        wandb_run.summary["evaluation/complete"] = True
        wandb_run.finish(exit_code=0)
    print(
        json.dumps(
            {
                "event": "PERSISTED_COMPLETED_EVALUATION_RESULT",
                "path": str(summary_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return summary_path


def _read_episode_records(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            episode_id = int(record["episode_id"])
            if episode_id in records and records[episode_id] != record:
                raise RuntimeError(f"conflicting duplicate episode {episode_id}: {path}")
            records[episode_id] = record
    return records


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
