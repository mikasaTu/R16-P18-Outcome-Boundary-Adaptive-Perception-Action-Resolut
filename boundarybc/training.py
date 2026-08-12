from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from boundarybc.checkpoint import (
    atomic_write_json,
    capture_rng_state,
    latest_complete_checkpoint,
    load_complete_checkpoint,
    restore_rng_state,
    save_complete_checkpoint,
    save_final_model,
    validate_final_model,
)
from boundarybc.config import ExperimentConfig
from boundarybc.data import (
    ProprioNormalizer,
    discover_episodes,
    fit_proprio_normalizer,
    load_task_arrays,
    make_batch,
    official_demo_path,
    split_episode_records,
)
from boundarybc.model import BoundaryBCS, masked_action_mse


def model_directory(checkpoint_root: str | Path, run_id: str, task_key: str, seed: int) -> Path:
    return Path(checkpoint_root) / run_id / "models" / task_key / f"seed_{seed}"


def final_model_path(checkpoint_root: str | Path, run_id: str, task_key: str, seed: int) -> Path:
    return model_directory(checkpoint_root, run_id, task_key, seed) / "final.pt"


def train_one_model(
    config: ExperimentConfig,
    *,
    task_key: str,
    model_seed: int,
    run_id: str,
    dataset_root: str | Path,
    checkpoint_root: str | Path,
    log_root: str | Path,
    device: torch.device,
) -> Path:
    if model_seed not in config.training_seeds:
        raise ValueError(f"model seed {model_seed} is not preregistered")
    task = config.task(task_key)
    directory = model_directory(checkpoint_root, run_id, task_key, model_seed)
    final_path = final_model_path(checkpoint_root, run_id, task_key, model_seed)
    if validate_final_model(final_path):
        print(json.dumps({"event": "MODEL_ALREADY_COMPLETE", "path": str(final_path)}), flush=True)
        return final_path
    directory.mkdir(parents=True, exist_ok=True)
    log_directory = Path(log_root) / run_id
    log_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = log_directory / f"train_{task_key}_seed_{model_seed}.jsonl"
    wandb_run = _init_wandb_run(
        config,
        run_id=run_id,
        task_key=task_key,
        task_name=task.name,
        model_seed=model_seed,
        job_type="train",
    )

    hdf5_path = official_demo_path(dataset_root, config.raw["benchmark"]["suite"], task.name)
    records = discover_episodes(hdf5_path)
    split = split_episode_records(records)
    train_arrays = load_task_arrays(
        hdf5_path,
        split["train"],
        action_horizon=int(config.raw["model"]["action_horizon"]),
    )
    validation_arrays = load_task_arrays(
        hdf5_path,
        split["validation"],
        action_horizon=int(config.raw["model"]["action_horizon"]),
    )
    normalizer = fit_proprio_normalizer(train_arrays.proprio)
    _set_seed(model_seed)
    model = make_model(config).to(device)
    training = config.raw["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    optimizer_steps = int(training["optimizer_steps"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=optimizer_steps,
        eta_min=float(training["learning_rate"]) * 0.05,
    )
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(model_seed ^ 0x5A17)
    global_step = 0
    latest = latest_complete_checkpoint(directory)
    split_identities = {
        name: [record.identity for record in subset]
        for name, subset in split.items()
    }
    if latest is not None:
        checkpoint = load_complete_checkpoint(latest, map_location=device)
        _validate_resume(
            checkpoint,
            config=config,
            run_id=run_id,
            task_key=task_key,
            model_seed=model_seed,
            split_identities=split_identities,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng_state(checkpoint["rng"])
        batch_generator.set_state(checkpoint["batch_generator_state"])
        global_step = int(checkpoint["global_step"])
        if global_step >= optimizer_steps:
            raise RuntimeError("complete final model is missing for an already-finished checkpoint")
        print(
            json.dumps(
                {
                    "event": "AUTO_RESUME",
                    "run_id": run_id,
                    "task_key": task_key,
                    "model_seed": model_seed,
                    "checkpoint": str(latest),
                    "restored_global_step": global_step,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        print(
            json.dumps(
                {
                    "event": "AUTO_RESUME",
                    "run_id": run_id,
                    "task_key": task_key,
                    "model_seed": model_seed,
                    "checkpoint": None,
                    "restored_global_step": 0,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    batch_size = int(training["batch_size"])
    validation_interval = int(training["validation_interval"])
    checkpoint_interval = int(training["complete_checkpoint_interval"])
    use_bf16 = training.get("precision") == "bf16" and device.type == "cuda"
    start_time = time.perf_counter()
    model.train()
    while global_step < optimizer_steps:
        indices = torch.randint(
            len(train_arrays),
            (batch_size,),
            generator=batch_generator,
        )
        batch = make_batch(
            train_arrays,
            indices,
            normalizer,
            device=device,
            augment=True,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            prediction = model(batch["images"], batch["proprio"])
            loss = masked_action_mse(prediction, batch["actions"], batch["mask"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {global_step + 1}: {loss}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
        global_step += 1
        record: dict[str, Any] = {
            "event": "TRAIN_STEP",
            "run_id": run_id,
            "task_key": task_key,
            "model_seed": model_seed,
            "global_step": global_step,
            "loss": float(loss.detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        if global_step == 1 or global_step % 10 == 0:
            print(json.dumps(record, sort_keys=True), flush=True)
        if global_step == 1 or global_step % 50 == 0:
            _append_jsonl(metrics_path, record)
        if global_step == 1:
            first_work_path = directory / "FIRST_WORK.json"
            atomic_write_json(
                first_work_path,
                {
                    "schema_version": 1,
                    "event": "PERSISTED_OPTIMIZER_STEP_AND_LOSS",
                    "run_id": run_id,
                    "config_sha256": config.sha256,
                    "task_key": task_key,
                    "model_seed": model_seed,
                    "global_step": global_step,
                    "loss": record["loss"],
                    "learning_rate": record["learning_rate"],
                    "uid": os.getuid(),
                    "gid": os.getgid(),
                },
            )
        if wandb_run is not None and (global_step == 1 or global_step % 10 == 0):
            wandb_run.log(
                {
                    "train/global_step": global_step,
                    "train/loss": record["loss"],
                    "train/gradient_norm": record["gradient_norm"],
                    "train/learning_rate": record["learning_rate"],
                },
                step=global_step,
            )
        if global_step % validation_interval == 0 or global_step == optimizer_steps:
            validation_loss = evaluate_validation_loss(
                model,
                validation_arrays,
                normalizer,
                device=device,
                use_bf16=use_bf16,
            )
            validation_record = {
                "event": "VALIDATION",
                "run_id": run_id,
                "task_key": task_key,
                "model_seed": model_seed,
                "global_step": global_step,
                "validation_loss": validation_loss,
            }
            print(json.dumps(validation_record, sort_keys=True), flush=True)
            _append_jsonl(metrics_path, validation_record)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "validation/global_step": global_step,
                        "validation/loss": validation_loss,
                    },
                    step=global_step,
                )
            model.train()
        if global_step % checkpoint_interval == 0 or global_step == optimizer_steps:
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                normalizer=normalizer,
                config=config,
                run_id=run_id,
                task_key=task_key,
                task_name=task.name,
                model_seed=model_seed,
                global_step=global_step,
                batch_generator=batch_generator,
                split_identities=split_identities,
            )
            save_complete_checkpoint(
                directory,
                step=global_step,
                payload=payload,
                keep_last=3,
            )

    payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        normalizer=normalizer,
        config=config,
        run_id=run_id,
        task_key=task_key,
        task_name=task.name,
        model_seed=model_seed,
        global_step=global_step,
        batch_generator=batch_generator,
        split_identities=split_identities,
    )
    save_final_model(final_path, payload)
    atomic_write_json(
        directory / "training_summary.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "task_key": task_key,
            "task_name": task.name,
            "model_seed": model_seed,
            "global_step": global_step,
            "config_sha256": config.sha256,
            "train_frames": len(train_arrays),
            "validation_frames": len(validation_arrays),
            "final_model": str(final_path),
        },
    )
    print(json.dumps({"event": "MODEL_COMPLETE", "path": str(final_path)}), flush=True)
    if wandb_run is not None:
        wandb_run.summary["train/final_global_step"] = global_step
        wandb_run.finish(exit_code=0)
    return final_path


@torch.inference_mode()
def evaluate_validation_loss(
    model: BoundaryBCS,
    arrays: Any,
    normalizer: ProprioNormalizer,
    *,
    device: torch.device,
    use_bf16: bool,
    batch_size: int = 128,
) -> float:
    model.eval()
    numerator = 0.0
    denominator = 0.0
    for start in range(0, len(arrays), batch_size):
        stop = min(len(arrays), start + batch_size)
        indices = torch.arange(start, stop, dtype=torch.int64)
        batch = make_batch(
            arrays,
            indices,
            normalizer,
            device=device,
            augment=False,
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            prediction = model(batch["images"], batch["proprio"])
            squared = (prediction - batch["actions"]).square().mean(dim=-1)
        numerator += float((squared * batch["mask"]).sum().cpu())
        denominator += float(batch["mask"].sum().cpu())
    return numerator / max(denominator, 1.0)


def make_model(config: ExperimentConfig) -> BoundaryBCS:
    model = config.raw["model"]
    return BoundaryBCS(
        proprio_dim=int(model["proprio_dim"]),
        hidden_dim=int(model["hidden_dim"]),
        action_dim=int(config.raw["benchmark"]["action_dim"]),
        action_horizon=int(model["action_horizon"]),
        transformer_layers=int(model["transformer_layers"]),
        transformer_heads=int(model["transformer_heads"]),
    )


def _checkpoint_payload(
    *,
    model: BoundaryBCS,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    normalizer: ProprioNormalizer,
    config: ExperimentConfig,
    run_id: str,
    task_key: str,
    task_name: str,
    model_seed: int,
    global_step: int,
    batch_generator: torch.Generator,
    split_identities: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "task_key": task_key,
        "task_name": task_name,
        "model_seed": model_seed,
        "global_step": global_step,
        "config_sha256": config.sha256,
        "config_canonical_json": config.canonical_json(),
        "split_identities": split_identities,
        "normalizer": normalizer.as_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": capture_rng_state(),
        "batch_generator_state": batch_generator.get_state(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _validate_resume(
    checkpoint: dict[str, Any],
    *,
    config: ExperimentConfig,
    run_id: str,
    task_key: str,
    model_seed: int,
    split_identities: dict[str, list[str]],
) -> None:
    expected = {
        "run_id": run_id,
        "task_key": task_key,
        "model_seed": model_seed,
        "config_sha256": config.sha256,
        "split_identities": split_identities,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"resume contract mismatch for {key}")
    required = (
        "model",
        "optimizer",
        "scheduler",
        "rng",
        "batch_generator_state",
        "global_step",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise RuntimeError(f"incomplete checkpoint state: {missing}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, encoded.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _init_wandb_run(
    config: ExperimentConfig,
    *,
    run_id: str,
    task_key: str,
    task_name: str,
    model_seed: int,
    job_type: str,
) -> Any | None:
    entity = os.environ.get("WANDB_ENTITY")
    api_key_present = bool(os.environ.get("WANDB_API_KEY"))
    if entity is None and not api_key_present:
        return None
    if entity != "chen_jian-cj-workspace" or not api_key_present:
        raise RuntimeError("W&B requires the exact entity contract and a nonempty injected key")
    import wandb

    return wandb.init(
        entity=entity,
        project="r16-p18-libero-stage1",
        id=f"{run_id}-{task_key}-seed-{model_seed}-{job_type}",
        name=f"{task_key}-seed-{model_seed}-{job_type}",
        group=run_id,
        job_type=job_type,
        resume="allow",
        config={
            "protocol_id": config.protocol_id,
            "config_sha256": config.sha256,
            "task_key": task_key,
            "task_name": task_name,
            "model_seed": model_seed,
            "training": config.raw["training"],
            "model": config.raw["model"],
        },
    )
