from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from boundarybc.checkpoint import atomic_write_json
from boundarybc.config import ExperimentConfig
from boundarybc.evaluation import evaluation_result_path


def build_baseline_gate_report(
    config: ExperimentConfig,
    *,
    run_id: str,
    log_root: str | Path,
) -> tuple[Path, dict[str, Any]]:
    task_reports: dict[str, Any] = {}
    all_pass = True
    bootstrap_replicates = int(config.raw["stage1_go_thresholds"]["paired_bootstrap_replicates"])
    for task in config.tasks:
        matrix = np.zeros((len(config.training_seeds), 50), dtype=np.float64)
        seed_rates: dict[str, float] = {}
        for seed_index, seed in enumerate(config.training_seeds):
            path = evaluation_result_path(log_root, run_id, task.key, seed)
            records = _read_records(path)
            if set(records) != set(range(50)):
                raise RuntimeError(f"missing evaluation episodes: {path}")
            values = np.asarray([bool(records[index]["success"]) for index in range(50)], dtype=np.float64)
            matrix[seed_index] = values
            seed_rates[str(seed)] = float(values.mean())
        rate = float(matrix.mean())
        low, high = paired_seed_episode_bootstrap(
            matrix,
            replicates=bootstrap_replicates,
            seed=1601800 + len(task_reports),
        )
        passed = task.baseline_success_min <= rate <= task.baseline_success_max
        all_pass = all_pass and passed
        task_reports[task.key] = {
            "task_name": task.name,
            "role": task.role,
            "success_rate": rate,
            "confidence_interval_95": [low, high],
            "seed_success_rates": seed_rates,
            "required_range": [task.baseline_success_min, task.baseline_success_max],
            "passed": passed,
        }
    protocol_deviation = bool(config.raw["data"]["protocol_deviation"])
    decision = "BASELINE_GATE_PASS_CONTINUE_STAGE1" if all_pass else "NO_GO_BASELINE_GATE"
    report = {
        "schema_version": 1,
        "protocol_id": config.protocol_id,
        "config_sha256": config.sha256,
        "run_id": run_id,
        "decision": decision,
        "baseline_gate_passed": all_pass,
        "task_results": task_reports,
        "data_protocol_deviation": protocol_deviation,
        "stage1_go_permitted_from_this_pilot": False,
        "stage1_go_block_reason": (
            "The official 200-demo/episode-seed protocol is unresolved."
            if protocol_deviation
            else "Stage-1 adaptive arms have not yet been evaluated."
        ),
        "adaptive_implementation_authorized_by_gate": all_pass,
    }
    output_directory = Path(log_root) / run_id / "reports"
    json_path = output_directory / "baseline_gate.json"
    atomic_write_json(json_path, report)
    markdown_path = output_directory / "baseline_gate.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_baseline_markdown(report), encoding="utf-8")
    return json_path, report


def paired_seed_episode_bootstrap(
    matrix: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if matrix.ndim != 2 or matrix.shape[1] != 50:
        raise ValueError(f"expected seed x 50 matrix, got {matrix.shape}")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        seed_indices = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        episode_indices = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        estimates[index] = matrix[np.ix_(seed_indices, episode_indices)].mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _read_records(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                episode_id = int(value["episode_id"])
                if episode_id in records and records[episode_id] != value:
                    raise RuntimeError(f"conflicting duplicate episode {episode_id}: {path}")
                records[episode_id] = value
    return records


def _baseline_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# R16-P18 LIBERO Stage-1 baseline gate",
        "",
        f"Decision: **{report['decision']}**",
        "",
        "| Task | Success | 95% paired bootstrap CI | Required | Gate |",
        "|---|---:|---:|---:|---|",
    ]
    for task in report["task_results"].values():
        low, high = task["confidence_interval_95"]
        minimum, maximum = task["required_range"]
        lines.append(
            f"| {task['task_name']} | {task['success_rate']:.1%} | "
            f"[{low:.1%}, {high:.1%}] | [{minimum:.0%}, {maximum:.0%}] | "
            f"{'PASS' if task['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "This is a baseline health gate, not an R16-P18 result. The adaptive selector and effect model must remain unimplemented until this gate passes.",
            "",
            "The pilot cannot return Stage-1 GO because official LIBERO provides only 50 demonstrations per exact task and no original episode-seed field; the requested 200-demo protocol remains unresolved.",
            "",
        ]
    )
    return "\n".join(lines)

