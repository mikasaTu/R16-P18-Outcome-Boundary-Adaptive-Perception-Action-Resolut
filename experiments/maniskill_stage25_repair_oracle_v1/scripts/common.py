#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

PROTOCOL_ID = "R16-P18-MS4-STAGE25-REPAIR-ORACLE-V1"
MODEL_SEEDS = (16018, 16019, 16020)
TASKS = ("StackCube-v1", "PushCube-v1")
PHASES = (
    "free_space_approach",
    "pre_grasp_or_pre_contact",
    "object_in_hand_pre_placement",
    "placement_contact_near_completion",
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def low31_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def unique_hash_seeds(namespace: str, count: int, excluded: set[int]) -> list[int]:
    result: list[int] = []
    index = 0
    while len(result) < count:
        candidate = low31_seed(PROTOCOL_ID, namespace, index)
        index += 1
        if candidate in excluded or candidate in result:
            continue
        excluded.add(candidate)
        result.append(candidate)
    return result


def require_disjoint(groups: Iterable[tuple[str, Iterable[int]]]) -> None:
    owner: dict[int, str] = {}
    for name, values in groups:
        for value in values:
            previous = owner.get(int(value))
            if previous is not None:
                raise RuntimeError(f"seed overlap: {value} in {previous} and {name}")
            owner[int(value)] = name

