from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


PROTOCOL_ID = "R16-P18-MS3-ACT-BOUNDARY-SCREEN-V1"
FORMAL_TASKS = (
    "PlugCharger-v1",
    "PushT-v1",
    "StackCube-v1",
    "PushCube-v1",
)
RETIRED_FORMAL_TASKS = ("PegInsertionSide-v1",)
MODEL_SEEDS = (16018, 16019, 16020)
SPLIT_COUNTS = {"train": 200, "validation": 50, "test": 50}


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_dataset(digest: "hashlib._Hash", name: str, dataset: h5py.Dataset) -> None:
    array = np.ascontiguousarray(dataset[()])
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json(list(array.shape)))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))


def sha256_hdf5_group(group: h5py.Group) -> str:
    digest = hashlib.sha256()

    def visitor(name: str, value: h5py.Group | h5py.Dataset) -> None:
        if isinstance(value, h5py.Dataset):
            _hash_dataset(digest, name, value)

    group.visititems(visitor)
    return digest.hexdigest()


def sha256_initial_state(env_states: h5py.Group | h5py.Dataset) -> str:
    """Hash the first timestep of every state dataset in stable path order."""

    digest = hashlib.sha256()

    def hash_first(name: str, dataset: h5py.Dataset) -> None:
        if dataset.ndim == 0:
            array = np.ascontiguousarray(dataset[()])
        else:
            if dataset.shape[0] < 1:
                raise ValueError(f"empty state dataset: {dataset.name}")
            array = np.ascontiguousarray(dataset[0])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json(list(array.shape)))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))

    if isinstance(env_states, h5py.Dataset):
        hash_first("env_states", env_states)
    else:
        def visitor(name: str, value: h5py.Group | h5py.Dataset) -> None:
            if isinstance(value, h5py.Dataset):
                hash_first(name, value)

        env_states.visititems(visitor)
    return digest.hexdigest()


def selection_key(
    task_id: str,
    source_episode_id: int,
    episode_seed: int,
    initial_state_sha256: str,
) -> str:
    value = (
        f"{PROTOCOL_ID}{task_id}{source_episode_id}"
        f"{episode_seed}{initial_state_sha256}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def closed_loop_seeds(
    task_id: str,
    demonstration_seeds: Iterable[int],
    count: int = 100,
) -> list[int]:
    forbidden = {int(seed) for seed in demonstration_seeds}
    result: list[int] = []
    suffix = 0
    while len(result) < count:
        index = len(result)
        payload = f"{PROTOCOL_ID}{task_id}closed_loop{index}:{suffix}"
        seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")
        seed &= 0x7FFFFFFF
        if seed not in forbidden and seed not in result:
            result.append(seed)
        else:
            suffix += 1
    return result


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")
