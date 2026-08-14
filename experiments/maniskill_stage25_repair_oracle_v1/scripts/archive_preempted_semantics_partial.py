#!/usr/bin/env python3
"""Archive an incomplete success-semantics tree after a PAI preemption.

This is an operational recovery tool, not part of the frozen scientific
pipeline.  It never edits or merges observations.  After validating every
partial JSONL against the frozen confirmatory seed prefix, it atomically moves
the complete ``success_semantics`` directory into a recovery namespace.  A
later formal lease therefore reruns each incomplete arm from episode zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path


PROTOCOL_ID = "R16-P18-MS4-STAGE25-REPAIR-ORACLE-V1"
ARCHIVE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,95}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--confirmatory-seed-bank", type=Path, required=True)
    parser.add_argument("--archive-id", required=True)
    parser.add_argument("--pai-job-id", required=True)
    parser.add_argument("--expected-total", type=int, default=100)
    parser.add_argument("--expected-batch", type=int, default=20)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise RuntimeError(f"non-object JSONL record at {path}:{line_number}")
            records.append(value)
    return records


def main() -> None:
    args = parse_args()
    if not ARCHIVE_ID_PATTERN.fullmatch(args.archive_id):
        raise ValueError("invalid archive id")
    if args.expected_total <= args.expected_batch or args.expected_batch <= 0:
        raise ValueError("invalid expected episode counts")

    root = args.result_root.resolve(strict=True)
    run_manifest_path = root / "FORMAL_RUN_MANIFEST.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("result root protocol mismatch")
    if (root / "FORMAL_COMPLETE.json").exists():
        raise RuntimeError("refusing recovery on a completed formal run")

    semantics_root = root / "success_semantics"
    if not semantics_root.is_dir() or semantics_root.is_symlink():
        raise RuntimeError("missing or unsafe success_semantics directory")
    seed_payload = json.loads(args.confirmatory_seed_bank.read_text(encoding="utf-8"))
    expected_seed_prefix = [
        int(value)
        for value in seed_payload["tasks"]["StackCube-v1"][: args.expected_batch]
    ]
    expected_seed_bank_sha256 = sha256_file(args.confirmatory_seed_bank)

    partials: list[dict[str, object]] = []
    for episodes_path in sorted(semantics_root.glob("*/seed_*/episodes.jsonl")):
        output_dir = episodes_path.parent
        if (output_dir / "summary.json").exists():
            raise RuntimeError(f"refusing to archive completed output: {output_dir}")
        marker_path = output_dir / "FIRST_REAL_ROLLOUT.json"
        if not marker_path.is_file():
            raise RuntimeError(f"partial lacks first-rollout marker: {output_dir}")
        records = load_jsonl(episodes_path)
        if len(records) != args.expected_batch:
            raise RuntimeError(
                f"unexpected partial length {len(records)} at {episodes_path}; "
                f"expected exactly {args.expected_batch}"
            )
        mode = output_dir.parent.name
        model_seed = int(output_dir.name.removeprefix("seed_"))
        observed_episode_seeds = [int(record["episode_seed"]) for record in records]
        if observed_episode_seeds != expected_seed_prefix:
            raise RuntimeError(f"partial seed prefix mismatch: {episodes_path}")
        for record in records:
            required = {
                "protocol_id": PROTOCOL_ID,
                "task_id": "StackCube-v1",
                "mode": mode,
                "model_seed": model_seed,
                "seed_bank_sha256": expected_seed_bank_sha256,
            }
            if any(record.get(key) != value for key, value in required.items()):
                raise RuntimeError(f"partial record binding mismatch: {episodes_path}")
        partials.append(
            {
                "relative_output_dir": str(output_dir.relative_to(root)),
                "mode": mode,
                "model_seed": model_seed,
                "episodes": len(records),
                "episode_seeds": observed_episode_seeds,
                "episodes_jsonl_sha256": sha256_file(episodes_path),
                "episodes_jsonl_size": episodes_path.stat().st_size,
                "first_real_rollout_sha256": sha256_file(marker_path),
            }
        )

    if not partials:
        raise RuntimeError("no incomplete semantics outputs found")
    unexpected_files = [
        path
        for path in semantics_root.rglob("*")
        if path.is_file()
        and path.name not in {"episodes.jsonl", "FIRST_REAL_ROLLOUT.json"}
    ]
    if unexpected_files:
        raise RuntimeError(f"unexpected file(s) in partial tree: {unexpected_files}")

    archive_root = root / "recovery" / "preempted_partials" / args.archive_id
    archived_semantics = archive_root / "success_semantics"
    if archive_root.exists():
        raise FileExistsError(f"archive already exists: {archive_root}")
    archive_root.mkdir(parents=True, exist_ok=False)

    # Same-filesystem rename: either the entire visible tree moves or none of it.
    os.replace(semantics_root, archived_semantics)
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "status": "PREEMPTED_PARTIALS_ARCHIVED_FOR_FULL_DETERMINISTIC_RERUN",
        "scientific_outputs_used": False,
        "recovery_policy": "archive_unmodified_then_rerun_same_frozen_seed_bank_from_zero",
        "pai_job_id": args.pai_job_id,
        "run_id": run_manifest.get("run_id"),
        "archive_id": args.archive_id,
        "expected_total_episodes_per_arm": args.expected_total,
        "expected_partial_episodes_per_arm": args.expected_batch,
        "confirmatory_seed_bank_path": str(args.confirmatory_seed_bank),
        "confirmatory_seed_bank_sha256": expected_seed_bank_sha256,
        "partial_count": len(partials),
        "partials": partials,
        "archived_semantics_path": str(archived_semantics),
        "archived_at_unix": time.time(),
    }
    manifest_path = archive_root / "RECOVERY_MANIFEST.json"
    atomic_write_json(manifest_path, manifest)
    os.sync()
    print(json.dumps({
        "status": manifest["status"],
        "manifest": str(manifest_path),
        "partial_count": len(partials),
        "archived_semantics": str(archived_semantics),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
