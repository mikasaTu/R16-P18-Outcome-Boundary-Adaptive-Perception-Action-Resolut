#!/usr/bin/env python3
"""Build a source-and-artifact inventory for the zero-rollout S1 audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


PROTOCOL_ID = "R16-P18-STAGE3-S1-BUDGET-FEASIBILITY-V1"
OLD_EXPERIMENT = "experiments/maniskill_stage27r_core_mechanism_reset_v1"
RATIO_NUMERATOR = 92438200000000.0
RATIO_DENOMINATOR = 125120200000000.0
RATIO = 0.738795174560143
WALL_RATIO = 0.75025979


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _line(path: Path, start: int, end: int | None = None) -> str:
    suffix = f"-{end}" if end is not None else ""
    return f"`{path.as_posix()}:{start}{suffix}`"


def _git_value(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _record(repo_root: Path, relative: str, fields: str) -> dict[str, Any]:
    path = repo_root / relative
    result: dict[str, Any] = {"path": relative, "fields": fields, "exists": path.is_file()}
    if path.is_file():
        result["sha256"] = sha256_file(path)
        result["bytes"] = path.stat().st_size
    return result


def build(repo_root: Path, output: Path) -> str:
    repo_root = repo_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"fail-on-overwrite: {output}")
    source_rows = [
        _record(
            repo_root,
            f"{OLD_EXPERIMENT}/scripts/stage27r_runtime.py",
            "accounting.estimated_flops; accounting.gpu_latency_ms; query schedule",
        ),
        _record(
            repo_root,
            f"{OLD_EXPERIMENT}/scripts/analyze_stage27r.py",
            "coarse sum; full sum; budget; dc; refine acceptance",
        ),
        _record(
            repo_root,
            f"{OLD_EXPERIMENT}/scripts/multires_policy.py",
            "visual_mode; action query accounting; num_queries adapter",
        ),
        _record(
            repo_root,
            f"{OLD_EXPERIMENT}/artifacts/formal-run/statistics.json",
            "aggregated_state_treatments[].accounting; cost; gpu_latency_ms",
        ),
        _record(
            repo_root,
            f"{OLD_EXPERIMENT}/audits/mechanism_reverse_engineering_summary.json",
            "compute_budget.all_coarse_cost; compute_budget.all_fine_cost; recorded ratio",
        ),
        _record(
            repo_root,
            f"{OLD_EXPERIMENT}/artifacts/formal-run/screen/TASK_SELECTION.json",
            "groups[*].screened; groups[*].selected; seed; task",
        ),
        _record(
            repo_root,
            "experiments/r16p18_stage3/S1_PROFILE.json",
            "measurement; immutable input hashes; native support; raw timing samples; operator FLOPs",
        ),
        _record(
            repo_root,
            "experiments/r16p18_stage3/S1_DEV14_RUNTIME_AUDIT.json",
            "host; physical GPU; co-tenancy disclosure; sampling summary; no-rollout assertions",
        ),
        _record(
            repo_root,
            "experiments/r16p18_stage3/S1_PROTOCOL_AMENDMENT_DEV14.json",
            "user-authorized profiling-host amendment; unchanged constraints",
        ),
    ]
    screen_path = repo_root / f"{OLD_EXPERIMENT}/artifacts/formal-run/screen/TASK_SELECTION.json"
    screen_summary: dict[str, Any] = {}
    if screen_path.is_file():
        try:
            payload = json.loads(screen_path.read_text(encoding="utf-8"))
            groups = payload.get("groups", {}) if isinstance(payload, dict) else {}
            for key, group in groups.items():
                if isinstance(group, dict):
                    selected = group.get("selected", {})
                    screen = selected.get("screen", {}) if isinstance(selected, dict) else {}
                    validation = selected.get("validation", {}) if isinstance(selected, dict) else {}
                    screen_summary[key] = {
                        "task": group.get("task", key.split("/", 1)[0]),
                        "seed": group.get("seed", key.rsplit("_", 1)[-1]),
                        "selected_step": selected.get("step") if isinstance(selected, dict) else None,
                        "success_hold5": screen.get("success_hold5"),
                        "success_at_end": screen.get("success_at_end"),
                        "post_success_loss": screen.get("post_success_loss"),
                        "validation_cc_success_hold5": validation.get("CC", {}).get("success_hold5"),
                        "validation_ff_success_hold5": validation.get("FF", {}).get("success_hold5"),
                    }
        except (OSError, json.JSONDecodeError):
            screen_summary = {"parse_error": True}

    tree_ref = "HEAD^{tree}"
    lines = [
        "Inventory convention: this directory follows the existing repository convention `experiments/r16p18_stage3/`; all Stage-2.7R and earlier paths below are read-only historical inputs.",
        "",
        "# S1 Inventory",
        "",
        f"Protocol: `{PROTOCOL_ID}`",
        f"Repository HEAD at inventory time: `{_git_value(repo_root, 'rev-parse', 'HEAD')}`",
        f"Repository tree at inventory time: `{_git_value(repo_root, 'rev-parse', tree_ref)}`",
        "",
        "## S1.0 substrate audit",
        "",
        "### Historical 0.738795 ratio",
        "",
        f"The immutable Stage-2.7R record is `all_coarse_cost / all_fine_cost = {RATIO_NUMERATOR:.0f} / {RATIO_DENOMINATOR:.0f} = {RATIO:.15f}`. The historical wall-clock ratio is `{WALL_RATIO}`; it is not a fresh S1 measurement.",
        "",
        f"- Per-row FLOP accounting formula: {_line(repo_root / f'{OLD_EXPERIMENT}/scripts/stage27r_runtime.py', 188)} (`global_encoder_calls * 1.8e9 + fine_encoder_calls * 1.8e9 + policy_forward_calls * 0.7e9`).",
        f"- Budget numerator/denominator sums: {_line(repo_root / f'{OLD_EXPERIMENT}/scripts/analyze_stage27r.py', 111, 114)} (`coarse = sum(CC.cost)`, `full = sum(FF.cost)`, `budget = alpha * full`).",
        f"- Refinement acceptance: {_line(repo_root / f'{OLD_EXPERIMENT}/scripts/analyze_stage27r.py', 116, 127)}; a candidate is counted only when `du > 0` and `cost + dc <= budget`.",
        f"- Historical raw accounting: {_line(repo_root / f'{OLD_EXPERIMENT}/artifacts/formal-run/statistics.json', 1)} fields `aggregated_state_treatments[].accounting.estimated_flops`, `.gpu_latency_ms`, and `.cost`; this is old rollout accounting, not a new forward profile.",
        f"- Historical ratio sidecar: {_line(repo_root / f'{OLD_EXPERIMENT}/audits/mechanism_reverse_engineering_summary.json', 230, 237)} fields `compute_budget.all_coarse_cost`, `all_fine_cost`, `coarse_to_fine_ratio`, and budget summaries.",
        "- Archived pure-axis fallback (diagnostic only): code-derived fixed-window FLOP proxies are visual coarse/fine `8.6e9/15.8e9` and action coarse/fine `8.6e9/34.4e9`; these are derived from the native 8-output, interval-4-versus-1 schedule and are not fresh wall-clock measurements.",
        "",
        "### Resolution definitions found in code",
        "",
        f"- Visual coarse path: {_line(repo_root / f'{OLD_EXPERIMENT}/scripts/multires_policy.py', 91, 119)}; global image path is resized to 112x112, while fine adds a crop branch from the original tensor. Crop grid validation accepts 2 or 4 at {_line(repo_root / f'{OLD_EXPERIMENT}/scripts/multires_policy.py', 30, 39)}. S1 treats these as model-native semantics and requires a fresh forward profile to prove they run on the selected checkpoint.",
        f"- Visual accounting: {_line(repo_root / f'{OLD_EXPERIMENT}/scripts/multires_policy.py', 177, 186)}; global/fine encoder calls and tokens are reported separately.",
        f"- Action coarse/fine: {_line(repo_root / f'{OLD_EXPERIMENT}/scripts/stage27r_runtime.py', 161, 176)}; fine queries every action opportunity, coarse reuses the cached 8-output chunk and queries at interval 4. The independent axis is therefore query interval 4 versus 1, not an action candidate-token grid.",
        f"- Fixed output chunk: {_line(repo_root / f'{OLD_EXPERIMENT}/scripts/train_multires_act.py', 24, 33)} binds `num_queries=8`; S1 does not alter the weight or output shape.",
        "",
        "### Budget denominator and screen",
        "",
        f"Budget alpha uses the all-fine cost sum as denominator at {_line(repo_root / f'{OLD_EXPERIMENT}/scripts/analyze_stage27r.py', 111, 114)}. The old refine count is set by the `cost + dc > budget` guard at {_line(repo_root / f'{OLD_EXPERIMENT}/scripts/analyze_stage27r.py', 123, 126)}.",
        "",
        "The frozen Stage-2.7R screen task list is StackCube-v1, PegInsertionSide-v1, PlugCharger-v1, PullCubeTool-v1, PushT-v1, and PushCube-v1; task/control metadata originates at `stage27r_runtime.py:20-27`. Raw selected screen fields are listed below from the immutable `TASK_SELECTION.json`.",
        "",
        "| task/seed | selected step | 40-ep success_hold5 | 100-ep CC success_hold5 | 100-ep FF success_hold5 | success_at_end | post_success_loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(screen_summary):
        row = screen_summary[key]
        lines.append(
            f"| {key} | {row.get('selected_step')} | {row.get('success_hold5')} | {row.get('validation_cc_success_hold5')} | {row.get('validation_ff_success_hold5')} | {row.get('success_at_end')} | {row.get('post_success_loss')} |"
        )
    if not screen_summary:
        lines.append("| unavailable (immutable screen file absent or unparseable) | — | — | — | — | — | — |")
    profile_exists = (repo_root / "experiments/r16p18_stage3/S1_PROFILE.json").is_file()
    profile_boundary = (
        "Fresh S1 profiling is asserted by `S1_PROFILE.json`: batch size 1, 50 warmups, "
        "200 CUDA-synchronized repeats, all five native resolution conditions, raw timing "
        "samples, and PyTorch operator FLOPs. The dev14 host amendment and disclosed GPU "
        "co-tenancy are recorded in `S1_PROTOCOL_AMENDMENT_DEV14.json` and "
        "`S1_DEV14_RUNTIME_AUDIT.json`. No environment was created, reset, or stepped."
        if profile_exists
        else
        "No fresh S1 profiling result is asserted by this inventory. "
        "`prepare_s1_observation.py` and `profile_s1_costs.py` fail closed when an input is absent."
    )
    lines += [
        "",
        "## Source file hashes",
        "",
        "| path | exists | bytes | SHA256 | audited fields |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in source_rows:
        lines.append(
            f"| `{row['path']}` | {row['exists']} | {row.get('bytes', '—')} | `{row.get('sha256', '—')}` | {row['fields']} |"
        )
    lines += [
        "",
        "## Fresh-profile boundary",
        "",
        profile_boundary,
        "",
        "`S1_COST_REPRO.json`, `S1_COST_CURVE.json`, `S1_FEASIBILITY.json`, plots, and `S1_DECISION.md` are generated only by the calculator after a supplied profile or explicitly selected archived fallback. They must not be hand-filled.",
        "",
    ]
    text = "\n".join(lines)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.repo_root, args.output)


if __name__ == "__main__":
    main()
