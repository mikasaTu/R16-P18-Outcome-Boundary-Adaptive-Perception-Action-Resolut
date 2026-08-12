from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from boundarybc.config import load_config
from boundarybc.data import ProprioNormalizer, official_demo_path
from boundarybc.libero_runtime import (
    configure_headless_runtime,
    deterministic_short_replay_check,
    load_task_init_states,
    make_offscreen_env,
    observation_tensors,
)
from boundarybc.training import make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fresh-process LIBERO replay and EGL/CUDA smoke")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--task-key", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_headless_runtime(config.path.parents[1])
    task = config.task(args.task_key)
    demo_path = official_demo_path(
        args.dataset_root,
        str(config.raw["benchmark"]["suite"]),
        task.name,
    )
    replay = deterministic_short_replay_check(
        config,
        task,
        demo_path,
        demo_key="demo_0",
        state_index=30,
        horizon=4,
        gpu_device_id=0,
    )
    eval_order = _egl_before_cuda_policy_step(config, task, torch.device(args.device))
    print(
        json.dumps(
            {
                "event": "REPLAY_WORKER_COMPLETE",
                "task_key": task.key,
                "replay": replay,
                "egl_before_cuda_policy_step": eval_order,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _egl_before_cuda_policy_step(config, task, device: torch.device) -> dict[str, object]:
    model = make_model(config)
    normalizer = ProprioNormalizer(
        mean=np.zeros(15, dtype=np.float32),
        std=np.ones(15, dtype=np.float32),
    )
    init_state = load_task_init_states(config, task)[0]
    env = make_offscreen_env(config, task, gpu_device_id=0)
    try:
        env.seed(0)
        env.reset()
        observation = env.set_init_state(init_state)
        # This is intentionally the first CUDA compute-context initialization
        # in the worker, after EGL is alive.
        model = model.to(device).eval()
        image, proprio = observation_tensors(observation, normalizer, device=device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            chunk = model(image, proprio)
        observation, _, _, _ = env.step(chunk[0, 0].float().cpu().numpy())
        return {
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "prediction_shape": list(chunk.shape),
            "post_step_image_shape": list(np.asarray(observation["agentview_image"]).shape),
        }
    finally:
        env.close()


if __name__ == "__main__":
    main()
