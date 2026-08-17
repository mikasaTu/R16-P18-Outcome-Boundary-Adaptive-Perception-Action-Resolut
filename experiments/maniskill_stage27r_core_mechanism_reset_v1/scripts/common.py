from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

PROTOCOL_ID = "R16-P18-MS6-STAGE27R-CORE-MECHANISM-RESET-V1"
ROOT = Path(__file__).resolve().parents[1]


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"fail-on-overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stable_seed(*parts: object) -> int:
    raw = ":".join(map(str, (PROTOCOL_ID, *parts))).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big") & 0x7FFFFFFF


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
