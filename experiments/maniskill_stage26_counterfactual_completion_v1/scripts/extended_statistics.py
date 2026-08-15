#!/usr/bin/env python3
"""Analysis-only clustered intervals and Holm-corrected secondary tests.

This script never selects a model or threshold.  It consumes the frozen
calibration logits and confirmatory episode rows after the formal run.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import MODEL_SEEDS, PROTOCOL_ID, read_jsonl, write_json_new
from predictor import average_precision, ece


def clustered_bootstrap(rows: list[dict], metric: Callable[[list[dict]], float], seed: int) -> dict:
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[int(row["episode_seed"])].append(row)
    keys = np.asarray(sorted(groups))
    rng = np.random.default_rng(seed)
    values = np.empty(10_000, dtype=np.float64)
    for start in range(0, 10_000, 250):
        draws = rng.integers(0, len(keys), size=(250, len(keys)))
        for local, indices in enumerate(draws):
            sample = [row for index in indices for row in groups[int(keys[index])]]
            values[start + local] = metric(sample)
    return {
        "estimate": float(metric(rows)),
        "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))],
        "bootstrap_replicates": 10_000,
        "cluster_unit": "source_episode",
    }


def offline_metrics(rows: list[dict], calibration: dict) -> dict[str, float]:
    logits = np.asarray([row["logits"] for row in rows], dtype=np.float64)
    labels = np.asarray([[row["hold_label"], row["continue_label"]] for row in rows], dtype=bool)
    probs = 1 / (1 + np.exp(-logits[:, :2] / float(calibration["temperature"])))
    beneficial = labels[:, 0] & ~labels[:, 1]
    not_done = ~labels[:, 0] & ~labels[:, 1]
    score = probs[:, 0] - probs[:, 1]
    stop = (probs[:, 0] >= calibration["tau_hold"]) & (score >= calibration["tau_advantage"])
    return {
        "stop_beneficial_auprc": average_precision(beneficial, score),
        "ece": (ece(labels[:, 0], probs[:, 0]) + ece(labels[:, 1], probs[:, 1])) / 2,
        "not_done_false_stop": float(stop[not_done].mean()) if not_done.any() else 0.0,
        "done_fragile_recall": float(stop[beneficial].mean()) if beneficial.any() else 0.0,
    }


def paired_values(root: Path, arm: str, seed: int | None = None) -> list[float]:
    by_episode: dict[int, list[float]] = defaultdict(list)
    seeds = MODEL_SEEDS if seed is None else (seed,)
    for model_seed in seeds:
        fixed = {r["episode_seed"]: r for r in read_jsonl(root / f"seed_{model_seed}" / "fixed_horizon" / "episodes.jsonl")}
        other = {r["episode_seed"]: r for r in read_jsonl(root / f"seed_{model_seed}" / arm / "episodes.jsonl")}
        if set(fixed) != set(other):
            raise RuntimeError(f"unpaired arm {arm} seed {model_seed}")
        for episode in fixed:
            by_episode[int(episode)].append(float(other[episode]["success_at_end"]) - float(fixed[episode]["success_at_end"]))
    return [float(np.mean(by_episode[key])) for key in sorted(by_episode)]


def paired_bootstrap(values: list[float], seed: int) -> dict:
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(10_000, dtype=np.float64)
    for start in range(0, 10_000, 500):
        indices = rng.integers(0, len(x), size=(500, len(x)))
        draws[start:start + 500] = x[indices].mean(1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(10_000, len(x)))
    null_means = (signs * x[None]).mean(1)
    p = float((1 + np.sum(np.abs(null_means) >= abs(x.mean()))) / 10_001)
    return {"gain": float(x.mean()), "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))], "two_sided_signflip_p": p, "bootstrap_replicates": 10_000, "signflip_replicates": 10_000, "cluster_unit": "episode_seed"}


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, (total - rank) * p_values[name])
        adjusted[name] = min(1.0, running)
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--predictor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads((args.predictor_root / "PREDICTOR_CALIBRATION_FREEZE.json").read_text())
    logits = read_jsonl(args.predictor_root / "calibration_logits.jsonl")
    selected = freeze["selected_architecture"]
    offline = {}
    for seed in MODEL_SEEDS:
        rows = [row for row in logits if row["model_seed"] == seed and row["architecture"] == selected]
        calibration = freeze["selected_checkpoints"][str(seed)]["calibration"]
        metrics = {}
        for index, name in enumerate(("stop_beneficial_auprc", "ece", "not_done_false_stop", "done_fragile_recall")):
            metrics[name] = clustered_bootstrap(rows, lambda sample, n=name, c=calibration: offline_metrics(sample, c)[n], 16018 + seed + index)
        offline[str(seed)] = metrics
    arms = (
        "fixed_time_matched_stop", "random_matched_stop", "learned_success_only_classifier",
        "learned_counterfactual_completion_gate", "privileged_neutral_after_hold5",
        "privileged_terminate_first_success",
    )
    paired = {arm: {"aggregate": paired_bootstrap(paired_values(args.result_root, arm), 16018 + i), "per_model_seed": {str(seed): paired_bootstrap(paired_values(args.result_root, arm, seed), 17018 + i + seed) for seed in MODEL_SEEDS}} for i, arm in enumerate(arms)}
    secondary = [arm for arm in arms if arm not in {"learned_counterfactual_completion_gate", "privileged_terminate_first_success"}]
    adjusted = holm({arm: paired[arm]["aggregate"]["two_sided_signflip_p"] for arm in secondary})
    for arm in secondary:
        paired[arm]["aggregate"]["holm_adjusted_p"] = adjusted[arm]
    write_json_new(args.output, {"protocol_id": PROTOCOL_ID, "status": "EXTENDED_STATISTICS_COMPLETE", "analysis_only_no_refit": True, "offline_cluster_bootstrap": offline, "paired_confirmatory": paired, "holm_family": secondary})


if __name__ == "__main__":
    main()
