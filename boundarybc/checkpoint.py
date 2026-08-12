from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


_STEP_RE = re.compile(r"checkpoint_step_(\d{8})\.pt$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def checkpoint_path(directory: str | Path, step: int) -> Path:
    return Path(directory) / f"checkpoint_step_{step:08d}.pt"


def marker_path(checkpoint: str | Path) -> Path:
    checkpoint = Path(checkpoint)
    return checkpoint.with_suffix(checkpoint.suffix + ".complete.json")


def save_complete_checkpoint(
    directory: str | Path,
    *,
    step: int,
    payload: dict[str, Any],
    keep_last: int = 3,
) -> Path:
    _validate_complete_payload(payload, expected_step=step)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = checkpoint_path(directory, step)
    temporary = directory / f".{target.name}.{os.getpid()}.tmp"
    if temporary.exists():
        raise FileExistsError(f"temporary checkpoint already exists: {temporary}")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(directory)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    marker = {
        "schema_version": 1,
        "step": int(step),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "run_id": payload["run_id"],
        "task_key": payload["task_key"],
        "model_seed": int(payload["model_seed"]),
        "complete_state": [
            "model",
            "optimizer",
            "scheduler",
            "rng",
            "batch_generator_state",
            "global_step",
        ],
    }
    atomic_write_json(marker_path(target), marker)
    retained = prune_complete_checkpoints(directory, keep_last=keep_last)
    print(
        json.dumps(
            {
                "event": "CHECKPOINT_RETENTION_OK",
                "saved_step": int(step),
                "retained_steps": [item[0] for item in retained],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return target


def discover_complete_checkpoints(directory: str | Path) -> list[tuple[int, Path]]:
    directory = Path(directory)
    if not directory.exists():
        return []
    complete: list[tuple[int, Path]] = []
    for checkpoint in directory.glob("checkpoint_step_*.pt"):
        match = _STEP_RE.search(checkpoint.name)
        if match is None:
            continue
        marker = marker_path(checkpoint)
        if not marker.is_file():
            continue
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        step = int(match.group(1))
        if metadata.get("step") != step:
            raise RuntimeError(f"checkpoint marker step mismatch: {marker}")
        if checkpoint.stat().st_size != int(metadata.get("bytes", -1)):
            raise RuntimeError(f"checkpoint size mismatch: {checkpoint}")
        if sha256_file(checkpoint) != metadata.get("sha256"):
            raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
        complete.append((step, checkpoint))
    return sorted(complete)


def latest_complete_checkpoint(directory: str | Path) -> Path | None:
    checkpoints = discover_complete_checkpoints(directory)
    return checkpoints[-1][1] if checkpoints else None


def prune_complete_checkpoints(
    directory: str | Path,
    *,
    keep_last: int,
) -> list[tuple[int, Path]]:
    if keep_last < 1:
        raise ValueError("keep_last must be positive")
    complete = discover_complete_checkpoints(directory)
    for _, checkpoint in complete[:-keep_last]:
        marker = marker_path(checkpoint)
        marker.unlink()
        checkpoint.unlink()
        _fsync_directory(Path(directory))
    retained = discover_complete_checkpoints(directory)
    if retained != complete[-keep_last:]:
        raise RuntimeError("checkpoint retention inventory did not converge")
    return retained


def load_complete_checkpoint(path: str | Path, *, map_location: str | torch.device) -> dict[str, Any]:
    path = Path(path)
    valid = {candidate for _, candidate in discover_complete_checkpoints(path.parent)}
    if path not in valid:
        raise RuntimeError(f"checkpoint is not complete and validated: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def save_final_model(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    atomic_write_json(
        marker_path(path),
        {
            "schema_version": 1,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "run_id": payload["run_id"],
            "task_key": payload["task_key"],
            "model_seed": int(payload["model_seed"]),
            "global_step": int(payload["global_step"]),
        },
    )
    return path


def validate_final_model(path: str | Path) -> bool:
    path = Path(path)
    marker = marker_path(path)
    if not path.is_file() or not marker.is_file():
        return False
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    return (
        path.stat().st_size == int(metadata.get("bytes", -1))
        and sha256_file(path) == metadata.get("sha256")
    )


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_complete_payload(payload: dict[str, Any], *, expected_step: int) -> None:
    required = {
        "run_id",
        "task_key",
        "model_seed",
        "model",
        "optimizer",
        "scheduler",
        "rng",
        "batch_generator_state",
        "global_step",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"checkpoint payload is incomplete: {missing}")
    if int(payload["global_step"]) != int(expected_step):
        raise ValueError(
            f"checkpoint step mismatch: filename step={expected_step}, payload={payload['global_step']}"
        )
