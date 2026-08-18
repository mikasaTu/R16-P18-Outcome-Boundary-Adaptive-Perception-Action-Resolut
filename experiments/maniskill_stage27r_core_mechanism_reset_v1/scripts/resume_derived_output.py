#!/usr/bin/env python3
"""Fail-closed resume helper for deterministic derived Stage-2.7R outputs.

The formal launcher may be resumed after an eviction at any boundary between
oracle shards and derived analysis files.  A derived file is never overwritten:
the requested command writes to a unique sibling temporary path; an existing
target must byte-for-byte match that recomputation, otherwise the helper fails
closed and leaves the target untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import uuid
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_or_validate(target: Path, command: list[str]) -> str:
    """Run a deterministic producer and install/validate its output safely."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.resume-{os.getpid()}-{uuid.uuid4().hex}.json"
    rendered = [str(part).replace("__OUTPUT__", str(temporary)) for part in command]
    try:
        subprocess.run(rendered, check=True)
        if not temporary.is_file():
            raise RuntimeError(f"derived producer did not create {temporary}")
        candidate_sha = sha256_file(temporary)
        if target.exists():
            target_sha = sha256_file(target)
            if target_sha != candidate_sha:
                raise RuntimeError(
                    f"derived output mismatch; refusing overwrite: {target} "
                    f"existing={target_sha} recomputed={candidate_sha}"
                )
            temporary.unlink()
            return "validated_existing"
        os.replace(temporary, target)
        return "installed_missing"
    except Exception:
        # Keep a failed candidate for post-mortem evidence.  It is outside the
        # scientific filename and can never be mistaken for a complete shard.
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command or "__OUTPUT__" not in command:
        parser.error("command must contain __OUTPUT__ exactly as the producer output")
    status = run_or_validate(args.target, command)
    print(f"RESUME_DERIVED_OUTPUT {status} {args.target}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
