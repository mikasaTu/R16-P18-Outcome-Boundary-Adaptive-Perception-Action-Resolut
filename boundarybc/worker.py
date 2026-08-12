from __future__ import annotations

import argparse

import torch

from boundarybc.config import load_config
from boundarybc.evaluation import evaluate_one_model
from boundarybc.libero_runtime import configure_headless_runtime
from boundarybc.training import train_one_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated BoundaryBC-S train/eval worker")
    parser.add_argument("phase", choices=("train", "evaluate"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-key", required=True)
    parser.add_argument("--model-seed", required=True, type=int)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    gpu_device_id = int(device.index or 0) if device.type == "cuda" else 0
    configure_headless_runtime(config.path.parents[1], gpu_device_id=gpu_device_id)
    if args.phase == "train":
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but CUDA is unavailable")
        train_one_model(
            config,
            task_key=args.task_key,
            model_seed=args.model_seed,
            run_id=args.run_id,
            dataset_root=args.dataset_root,
            checkpoint_root=args.checkpoint_root,
            log_root=args.log_root,
            device=device,
        )
    else:
        evaluate_one_model(
            config,
            task_key=args.task_key,
            model_seed=args.model_seed,
            run_id=args.run_id,
            checkpoint_root=args.checkpoint_root,
            log_root=args.log_root,
            device=device,
            gpu_device_id=gpu_device_id,
        )


if __name__ == "__main__":
    main()
