#!/usr/bin/env python3
"""Strictly validate an existing Stage-2.7R oracle shard before resume.

The old producer did not embed checkpoint/state-bank hashes in each row.  This
validator therefore checks every visible protocol/task/model/bank/grid/repeat
field and the complete key set, while deliberately *not* manufacturing an
in-row lineage claim.  The launcher writes a separate input/output SHA
sidecar for that process-level lineage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from common import PROTOCOL_ID

REQUIRED_ACCOUNTING = {
    "global_encoder_calls",
    "fine_encoder_calls",
    "policy_forward_calls",
    "policy_forward_rows",
    "visual_tokens",
    "action_opportunities",
    "executed_steps",
    "gpu_latency_ms",
    "simulator_latency_ms",
    "prefix_replay_simulator_latency_ms",
    "estimated_flops",
    "peak_memory_bytes",
    "selector_latency_ms",
    "episode_total_compute",
}
WEIGHTS = {"balanced", "success_dominant", "progress_dominant"}
INTEGER_ACCOUNTING = {
    "global_encoder_calls",
    "fine_encoder_calls",
    "policy_forward_calls",
    "policy_forward_rows",
    "visual_tokens",
    "action_opportunities",
    "executed_steps",
    "estimated_flops",
    "peak_memory_bytes",
}


def _finite_nonnegative(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"accounting field is not numeric: {label}={value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise RuntimeError(f"accounting field is not finite/non-negative: {label}={value!r}")
    return number


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"numeric field is not numeric: {label}={value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"numeric field is not finite: {label}={value!r}")
    return number


def _prefix_hash(prefix: Any) -> str:
    array = np.asarray(prefix, dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _numeric_prefix_equal(left: Any, right: Any) -> bool:
    try:
        lhs, rhs = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return lhs.shape == rhs.shape and bool(np.allclose(lhs, rhs, rtol=1e-6, atol=1e-6, equal_nan=False))


def _optional_state_binding(
    row: dict[str, Any], state: dict[str, Any], bank_id: str
) -> tuple[list[str], list[str]]:
    """Bind fields when both records expose them; disclose legacy gaps."""
    limitations: list[str] = []
    bound: list[str] = []

    # Old producer rows do not contain these fields.  Their absence is not
    # turned into a fabricated match; it is carried as an explicit limitation.
    state_seed = state.get("episode_seed")
    row_seed = row.get("episode_seed")
    if state_seed is None or row_seed is None:
        limitations.append("episode_seed unavailable in row or state bank; binding not established")
    else:
        try:
            if int(row_seed) != int(state_seed):
                raise RuntimeError(f"row/state-bank episode_seed mismatch for {bank_id}")
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"row/state-bank episode_seed is not an integer for {bank_id}") from exc
        bound.append("episode_seed")

    state_branch = state.get("branch_step")
    row_branch = row.get("branch_step")
    if state_branch is None or row_branch is None:
        limitations.append("branch_step unavailable in row or state bank; binding not established")
    else:
        try:
            if int(row_branch) != int(state_branch):
                raise RuntimeError(f"row/state-bank branch_step mismatch for {bank_id}")
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"row/state-bank branch_step is not an integer for {bank_id}") from exc
        bound.append("branch_step")

    state_prefix = state.get("prefix_actions")
    row_prefix = row.get("prefix_actions", row.get("prefix"))
    if state_prefix is None or row_prefix is None:
        limitations.append("prefix_actions unavailable in row or state bank; binding not established")
    else:
        if not _numeric_prefix_equal(row_prefix, state_prefix):
            raise RuntimeError(f"row/state-bank prefix_actions mismatch for {bank_id}")
        bound.append("prefix_actions")

    row_prefix_hash = row.get("prefix_sha256")
    if row_prefix_hash is not None:
        if state_prefix is None:
            limitations.append("row prefix_sha256 cannot be checked because state prefix_actions is absent")
        elif str(row_prefix_hash) != _prefix_hash(state_prefix):
            raise RuntimeError(f"row/state-bank prefix_sha256 mismatch for {bank_id}")
        else:
            bound.append("prefix_sha256")
    return bound, limitations


def _validate_accounting(accounting: dict[str, Any], bank_id: str, condition: str) -> dict[str, Any]:
    if not REQUIRED_ACCOUNTING.issubset(accounting):
        raise RuntimeError(f"accounting schema incomplete for {bank_id}/{condition}")
    for name, value in accounting.items():
        _finite_nonnegative(value, label=f"{bank_id}/{condition}/{name}")
        if name in INTEGER_ACCOUNTING:
            if isinstance(value, bool) or int(value) != value:
                raise RuntimeError(f"accounting integer field is not integral: {bank_id}/{condition}/{name}={value!r}")
    if accounting["policy_forward_rows"] < accounting["policy_forward_calls"]:
        raise RuntimeError(f"policy_forward_rows is smaller than policy_forward_calls for {bank_id}/{condition}")
    if accounting["executed_steps"] > accounting["action_opportunities"]:
        raise RuntimeError(f"executed_steps exceeds action_opportunities for {bank_id}/{condition}")
    return {
        "status": "PASS",
        "required_fields": sorted(REQUIRED_ACCOUNTING),
        "finite_nonnegative": True,
        "integer_fields": sorted(INTEGER_ACCOUNTING),
        "semantic_relations": {
            "policy_forward_rows_ge_policy_forward_calls": True,
            "executed_steps_le_action_opportunities": True,
        },
    }


def expected_conditions(grid: int) -> tuple[str, ...]:
    if int(grid) not in (2, 4):
        raise ValueError(f"unsupported tile grid: {grid}")
    tiles = int(grid) * int(grid)
    return (
        "CC",
        "CF",
        *(f"FC_tile{i}" for i in range(tiles)),
        *(f"FF_tile{i}" for i in range(tiles)),
    )


def _state_rows(path: Path, task: str, bank: str) -> tuple[dict, dict[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError(f"state-bank protocol mismatch: {path}")
    if payload.get("task") != task or payload.get("bank") != bank:
        raise RuntimeError(
            f"state-bank binding mismatch: task={payload.get('task')} "
            f"bank={payload.get('bank')}"
        )
    rows = payload.get("states")
    if not isinstance(rows, list) or len(rows) != int(payload.get("count", -1)):
        raise RuntimeError(f"state-bank count/schema mismatch: {path}")
    by_id = {}
    for row in rows:
        bank_id = str(row.get("bank_id", ""))
        if not bank_id or bank_id in by_id:
            raise RuntimeError(f"duplicate/empty state bank id: {bank_id}")
        if row.get("task", task) != task:
            raise RuntimeError(f"state task mismatch for {bank_id}")
        by_id[bank_id] = row
    return payload, by_id


def validate_shard(
    path: Path,
    *,
    task: str,
    model_seed: int,
    bank: str,
    grid: int,
    state_bank: Path,
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError(f"oracle protocol mismatch: {path}")
    if int(payload.get("tile_grid", -1)) != int(grid):
        raise RuntimeError(f"oracle tile-grid mismatch: {path}")
    conditions = tuple(payload.get("conditions", ()))
    expected = expected_conditions(int(grid))
    if conditions != expected:
        raise RuntimeError(
            f"oracle conditions mismatch: got={conditions} expected={expected}"
        )
    repeats = 3 if bank == "calibration" else 5
    if int(payload.get("repeats", -1)) != repeats:
        raise RuntimeError(f"oracle repeat count mismatch: {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"oracle rows is not a list: {path}")
    bank_payload, state_by_id = _state_rows(state_bank, task, bank)
    expected_count = len(state_by_id) * len(expected) * repeats
    if int(payload.get("row_count", -1)) != expected_count or len(rows) != expected_count:
        raise RuntimeError(
            f"oracle row count mismatch: {path}: visible={len(rows)}/"
            f"{payload.get('row_count')} expected={expected_count}"
        )
    keys = []
    lineage_bound_fields: set[str] = set()
    lineage_limitations: set[str] = set()
    accounting_checks: list[dict[str, Any]] = []
    for row in rows:
        if row.get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError(f"row protocol mismatch in {path}")
        if row.get("task") != task or int(row.get("model_seed", -1)) != int(model_seed):
            raise RuntimeError(f"row task/model binding mismatch in {path}")
        if row.get("bank") != bank:
            raise RuntimeError(f"row bank mismatch in {path}")
        bank_id = str(row.get("bank_id", ""))
        condition = str(row.get("condition", ""))
        repeat = int(row.get("repeat", -1))
        if bank_id not in state_by_id:
            raise RuntimeError(f"row references unknown state-bank id {bank_id}")
        if condition not in expected or not 0 <= repeat < repeats:
            raise RuntimeError(f"row condition/repeat invalid: {condition}/{repeat}")
        state = state_by_id[bank_id]
        for field in ("source_episode", "phase", "source_type"):
            if row.get(field) != state.get(field):
                raise RuntimeError(f"row/state-bank {field} mismatch for {bank_id}")
        bound, limitations = _optional_state_binding(row, state, bank_id)
        lineage_bound_fields.update(bound)
        lineage_limitations.update(limitations)
        expected_causal = bool(state.get("fidelity", {}).get("pass", False))
        if row.get("causal_fidelity_pass") is not expected_causal:
            raise RuntimeError(f"causal flag mismatch for {bank_id}")
        accounting = row.get("accounting")
        if not isinstance(accounting, dict):
            raise RuntimeError(f"accounting schema incomplete for {bank_id}/{condition}")
        accounting_checks.append(_validate_accounting(accounting, bank_id, condition))
        utilities = row.get("utilities")
        if not isinstance(utilities, dict) or set(utilities) != WEIGHTS:
            raise RuntimeError(f"utility accounting schema mismatch for {bank_id}/{condition}")
        for name, value in utilities.items():
            _finite_number(value, label=f"{bank_id}/{condition}/utilities/{name}")
        keys.append((bank_id, condition, repeat))
    expected_keys = {
        (bank_id, condition, repeat)
        for bank_id in state_by_id
        for condition in expected
        for repeat in range(repeats)
    }
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise RuntimeError(f"oracle key uniqueness/completeness mismatch: {path}")
    return {
        "status": "PASS",
        "path": str(path),
        "protocol_id": PROTOCOL_ID,
        "task": task,
        "model_seed": int(model_seed),
        "bank": bank,
        "tile_grid": int(grid),
        "state_bank": str(state_bank),
        "state_bank_count": len(state_by_id),
        "row_count": len(rows),
        "repeats": repeats,
        "conditions": list(expected),
        "unique_keys": len(set(keys)),
        "causal_flags_all_visible": all(
            bool(row.get("causal_fidelity_pass")) for row in rows
        ),
        "checkpoint_hash_embedded": any(
            "checkpoint_sha256" in row or "checkpoint_path" in row for row in rows
        ),
        "lineage_bindings": {
            "bound_fields": sorted(lineage_bound_fields),
            "limitations": sorted(lineage_limitations),
            "episode_seed": "bound when present in both row and state bank; legacy absence is disclosed",
            "branch_step": "bound when present in both row and state bank; legacy absence is disclosed",
            "prefix_actions": "bound when present in both row and state bank; legacy absence is disclosed",
        },
        "field_bindings": sorted(lineage_bound_fields),
        "limitations": sorted(lineage_limitations),
        "accounting_validation": {
            "status": "PASS",
            "rows_checked": len(accounting_checks),
            "finite_nonnegative": True,
            "integer_fields": sorted(INTEGER_ACCOUNTING),
            "semantic_relations": [
                "policy_forward_rows >= policy_forward_calls",
                "executed_steps <= action_opportunities",
            ],
        },
        "lineage_note": "checkpoint/state-bank hashes are process-level sidecar evidence, not embedded in legacy rows; missing row fields are disclosed as limitations",
        "source_state_bank_protocol": bank_payload.get("protocol_id"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--bank", choices=("calibration", "confirmatory", "negative"), required=True)
    parser.add_argument("--grid", type=int, choices=(2, 4), required=True)
    parser.add_argument("--state-bank", type=Path, required=True)
    args = parser.parse_args()
    result = validate_shard(
        args.oracle,
        task=args.task,
        model_seed=args.model_seed,
        bank=args.bank,
        grid=args.grid,
        state_bank=args.state_bank,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
