#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import MODEL_SEEDS, PROTOCOL_ID, append_jsonl, sha256_file, write_json_new
from predictor import CompletionModel, average_precision, ece, flatten_feature, shape_from_feature
from stage26_runtime import load_capsule

ARCHITECTURES = ("linear_probe", "two_layer_mlp", "one_layer_small_gru")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--epochs", type=int, default=50); return parser.parse_args()


def dataset(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    features, labels = [], []
    shape = None
    for row in rows:
        capsule = load_capsule(Path(row["path"])); feature = capsule.feature_dict(); shape = shape or shape_from_feature(feature)
        features.append(flatten_feature(feature)); labels.append([float(row["hold_success_20"]), float(row["continue_success_20"]), float(capsule.success_streak > 0)])
    return np.stack(features), np.asarray(labels, dtype=np.float32), rows, shape


def train_model(architecture: str, shape: Any, x: np.ndarray, y: np.ndarray, seed: int, epochs: int, device: torch.device) -> CompletionModel:
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = CompletionModel(architecture, shape).to(device); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed)
    tx = torch.from_numpy(x); ty = torch.from_numpy(y)
    for _ in range(epochs):
        order = torch.randperm(len(tx), generator=generator)
        for start in range(0, len(tx), 64):
            index = order[start:start+64]; logits = model(tx[index].to(device)); loss = F.binary_cross_entropy_with_logits(logits, ty[index].to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return model


def calibrate(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    best_temp, best_ece = None, float("inf")
    for temperature in (0.5, 0.75, 1.0, 1.5, 2.0):
        probabilities = 1 / (1 + np.exp(-logits / temperature)); value = (ece(labels[:,0], probabilities[:,0]) + ece(labels[:,1], probabilities[:,1])) / 2
        if value < best_ece: best_temp, best_ece = temperature, value
    probabilities = 1 / (1 + np.exp(-logits / best_temp)); beneficial = (labels[:,0] == 1) & (labels[:,1] == 0); not_done = (labels[:,0] == 0) & (labels[:,1] == 0)
    candidates = []
    for tau_hold in (0.5, 0.6, 0.7, 0.8, 0.9):
        for tau_advantage in (0.0, 0.1, 0.2, 0.3, 0.4):
            stop = (probabilities[:,0] >= tau_hold) & ((probabilities[:,0] - probabilities[:,1]) >= tau_advantage)
            false_stop = float(stop[not_done].mean()) if not_done.any() else 0.0; recall = float(stop[beneficial].mean()) if beneficial.any() else 0.0
            candidates.append((false_stop > 0.05, -recall, false_stop, tau_hold, tau_advantage))
    chosen = min(candidates); score = probabilities[:,0] - probabilities[:,1]
    success_candidates = []
    current_success = labels[:,2].astype(bool)
    for threshold in (0.5,0.6,0.7,0.8,0.9):
        pred = probabilities[:,2] >= threshold; false = float(pred[~current_success].mean()) if (~current_success).any() else 0.0; recall = float(pred[current_success].mean()) if current_success.any() else 0.0
        success_candidates.append((false > .05, -recall, false, threshold))
    success_choice = min(success_candidates)
    return {"temperature": best_temp, "ece": best_ece, "tau_hold": chosen[3], "tau_advantage": chosen[4], "not_done_false_stop": chosen[2], "done_fragile_recall": -chosen[1], "stop_beneficial_auprc": average_precision(beneficial, score), "success_only_threshold": success_choice[3], "success_only_false_stop": success_choice[2], "success_only_recall": -success_choice[1]}


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True); marker = args.output_dir / "PREDICTOR_CALIBRATION_FREEZE.json"
    if marker.exists(): print("PREDICTOR_ALREADY_FROZEN"); return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); all_metrics = {}; checkpoints = {}
    for seed in MODEL_SEEDS:
        train_x, train_y, _, shape = dataset(args.data_root / f"seed_{seed}" / "train_source" / "capsules.jsonl")
        cal_x, cal_y, cal_rows, _ = dataset(args.data_root / f"seed_{seed}" / "calibration" / "capsules.jsonl")
        for architecture in ARCHITECTURES:
            model = train_model(architecture, shape, train_x, train_y, seed, args.epochs, device)
            with torch.no_grad(): logits = model(torch.from_numpy(cal_x).to(device)).cpu().numpy()
            metrics = calibrate(logits, cal_y); metrics.update({"parameters": sum(p.numel() for p in model.parameters()), "calibration_rows": len(cal_x)})
            all_metrics[f"{seed}/{architecture}"] = metrics
            path = args.output_dir / "candidates" / f"seed_{seed}" / f"{architecture}.pt"; path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"protocol_id": PROTOCOL_ID, "architecture": architecture, "shape": shape.__dict__, "model_seed": seed, "state_dict": model.state_dict()}, path)
            checkpoints[f"{seed}/{architecture}"] = {"path": str(path), "sha256": sha256_file(path)}
            for row, values in zip(cal_rows, logits, strict=True): append_jsonl(args.output_dir / "calibration_logits.jsonl", {"protocol_id": PROTOCOL_ID, "model_seed": seed, "architecture": architecture, "capsule_id": row["capsule_id"], "episode_seed": row["episode_seed"], "hold_label": row["hold_success_20"], "continue_label": row["continue_success_20"], "logits": values.astype(float).tolist()})
    architecture_rows = []
    for architecture in ARCHITECTURES:
        values = [all_metrics[f"{seed}/{architecture}"] for seed in MODEL_SEEDS]
        feasible = sum(value["not_done_false_stop"] <= .05 for value in values) >= 2 and max(value["not_done_false_stop"] for value in values) <= .20
        architecture_rows.append((not feasible, -float(np.mean([v["stop_beneficial_auprc"] for v in values])), float(np.mean([v["ece"] for v in values])), values[0]["parameters"], architecture))
    selected = min(architecture_rows)[-1]
    selected_checkpoints = {}
    for seed in MODEL_SEEDS:
        source = Path(checkpoints[f"{seed}/{selected}"]["path"]); target = args.output_dir / f"seed_{seed}" / "predictor.pt"; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            output.write(source.read_bytes())
        selected_checkpoints[str(seed)] = {"path": str(target), "sha256": sha256_file(target), "calibration": all_metrics[f"{seed}/{selected}"]}
    # Secondary cross-model-seed generalization: train on two source banks and
    # evaluate logits on the held-out seed's calibration episodes. It is never
    # used to choose the primary threshold or confirmatory arm.
    leave_one_seed_out = {}
    for held_out in MODEL_SEEDS:
        train_parts, label_parts = [], []
        for source_seed in MODEL_SEEDS:
            if source_seed == held_out:
                continue
            sx, sy, _, _ = dataset(args.data_root / f"seed_{source_seed}" / "train_source" / "capsules.jsonl")
            train_parts.append(sx); label_parts.append(sy)
        test_x, test_y, _, held_shape = dataset(args.data_root / f"seed_{held_out}" / "calibration" / "capsules.jsonl")
        loso_model = train_model(selected, held_shape, np.concatenate(train_parts), np.concatenate(label_parts), 100000 + held_out, args.epochs, device)
        with torch.no_grad(): held_logits = loso_model(torch.from_numpy(test_x).to(device)).cpu().numpy()
        leave_one_seed_out[str(held_out)] = calibrate(held_logits, test_y)
    learnability_passes = [m["stop_beneficial_auprc"] >= .60 and m["ece"] <= .05 and m["not_done_false_stop"] <= .05 and m["done_fragile_recall"] >= .60 for m in (selected_checkpoints[str(seed)]["calibration"] for seed in MODEL_SEEDS)]
    write_json_new(marker, {"protocol_id": PROTOCOL_ID, "status": "PREDICTOR_CALIBRATION_FROZEN", "selected_architecture": selected, "architecture_selection": architecture_rows, "candidate_metrics": all_metrics, "selected_checkpoints": selected_checkpoints, "leave_one_model_seed_out": leave_one_seed_out, "offline_learnability_pass_per_seed": learnability_passes, "offline_learnability_gate_pass": sum(learnability_passes) >= 2 and all(selected_checkpoints[str(seed)]["calibration"]["not_done_false_stop"] <= .20 for seed in MODEL_SEEDS), "confirmatory_data_used": false, "epochs": args.epochs})


if __name__ == "__main__": main()
