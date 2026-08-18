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
import time
import uuid
import re
from pathlib import Path


DEFAULT_STALE_RESUME_SECONDS = 3600.0
_CANDIDATE_NAME = re.compile(r"^\.[^/]+\.resume-[0-9]+-[0-9a-f]{32}\.json$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resume_candidates(target: Path) -> list[Path]:
    """Return only regular sibling candidates; symlinks are ambiguous evidence."""
    candidates = sorted(target.parent.glob(f".{target.name}.resume-*") )
    for candidate in candidates:
        if not _CANDIDATE_NAME.fullmatch(candidate.name):
            raise RuntimeError(f"ambiguous/stale resume candidate name is invalid: {candidate}")
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError(f"ambiguous/stale resume candidate is not a regular file: {candidate}")
    return candidates


def _check_resume_candidates(
    target: Path,
    *,
    stale_after_seconds: float = DEFAULT_STALE_RESUME_SECONDS,
) -> list[Path]:
    """Fail closed on old/ambiguous candidates without breaking a fresh crash resume.

    A just-created candidate may be the only evidence left by a process that
    was interrupted between producer write and install; it is retained for
    post-mortem inspection and can safely be ignored while a deterministic
    recomputation runs.  Candidates older than the bounded grace period, or
    more than one candidate, are ambiguous and must be manually resolved.
    """
    candidates = _resume_candidates(target)
    if not candidates:
        return []
    if len(candidates) > 1:
        raise RuntimeError(f"ambiguous resume candidates; refusing to guess: {candidates}")
    age = max(0.0, time.time() - candidates[0].stat().st_mtime)
    if age > float(stale_after_seconds):
        raise RuntimeError(
            f"stale resume candidate; refusing to guess: {candidates[0]} age={age:.1f}s"
        )
    return candidates


def run_or_validate(
    target: Path,
    command: list[str],
    validator: list[str] | None = None,
    *,
    stale_after_seconds: float = DEFAULT_STALE_RESUME_SECONDS,
) -> str:
    """Run a deterministic producer and install/validate its output safely.

    If ``validator`` is supplied it is run against an existing target before
    any producer work, and against the temporary recomputation before the
    byte/hash comparison.  Thus stale/invalid evidence fails closed rather
    than being silently accepted or replaced.
    """
    target = Path(target)
    if target.is_symlink():
        raise RuntimeError(f"derived target must not be a symlink: {target}")
    candidates = _check_resume_candidates(target, stale_after_seconds=stale_after_seconds)
    if target.exists() and candidates:
        raise RuntimeError(f"resume candidate conflicts with existing target; refusing ambiguity: {candidates[0]}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.resume-{os.getpid()}-{uuid.uuid4().hex}.json"
    rendered = [str(part).replace("__OUTPUT__", str(temporary)) for part in command]
    try:
        if target.exists() and validator:
            check_target = [
                str(part).replace("__TARGET__", str(target)) for part in validator
            ]
            subprocess.run(check_target, check=True)
        subprocess.run(rendered, check=True)
        if not temporary.is_file():
            raise RuntimeError(f"derived producer did not create {temporary}")
        if validator:
            check_temporary = [
                str(part).replace("__TARGET__", str(temporary)) for part in validator
            ]
            subprocess.run(check_temporary, check=True)
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

        # Do not copy into the target and do not use os.replace here: a crash
        # during either operation could leave partial formal evidence, and a
        # concurrent resume could overwrite a winner.  A same-filesystem hard
        # link installs the already-complete candidate atomically with
        # no-replace semantics.  A concurrent equal candidate is validated,
        # while a conflicting candidate fails closed and remains available for
        # audit.
        try:
            os.link(temporary, target)
        except FileExistsError:
            target_sha = sha256_file(target)
            if target_sha != candidate_sha:
                raise RuntimeError(
                    f"derived output race mismatch; refusing overwrite: {target} "
                    f"existing={target_sha} recomputed={candidate_sha}"
                )
            temporary.unlink()
            return "validated_existing"
        temporary.unlink()
        return "installed_missing"
    except Exception:
        # Keep a failed candidate for post-mortem evidence.  It is outside the
        # scientific filename and can never be mistaken for a complete shard.
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument(
        "--validator",
        action="append",
        default=None,
        help=(
            "one validator argv token; repeat this option once per token. "
            "Each token may contain __TARGET__; validation runs before and "
            "after recomputation"
        ),
    )
    parser.add_argument(
        "--stale-after-seconds",
        type=float,
        default=DEFAULT_STALE_RESUME_SECONDS,
        help="age after which an orphan .resume-* candidate is rejected",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command or "__OUTPUT__" not in command:
        parser.error("command must contain __OUTPUT__ exactly as the producer output")
    status = run_or_validate(
        args.target,
        command,
        args.validator,
        stale_after_seconds=args.stale_after_seconds,
    )
    print(f"RESUME_DERIVED_OUTPUT {status} {args.target}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
