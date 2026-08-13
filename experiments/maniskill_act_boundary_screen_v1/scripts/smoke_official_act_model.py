#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from protocol_common import PROTOCOL_ID, write_json  # noqa: E402
from ema_compat import NonDeepSpeedEMAModel  # noqa: E402

import train_rgbd as official_act  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    device = torch.device(cli.device)
    args = official_act.Args(
        seed=16018,
        env_id="PickCube-v1",
        include_depth=False,
        backbone="resnet18",
        lr_backbone=1e-5,
        num_queries=30,
        control_mode="pd_ee_delta_pos",
        total_iters=1,
        batch_size=1,
    )
    official_act.args = args
    holder = SimpleNamespace(
        single_observation_space=gym.spaces.Dict(
            {
                "state": gym.spaces.Box(-np.inf, np.inf, shape=(42,), dtype=np.float32),
                "rgb": gym.spaces.Box(0, 255, shape=(1, 3, 224, 224), dtype=np.uint8),
            }
        ),
        single_action_space=gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
    )
    torch.manual_seed(16018)
    started = time.monotonic()
    agent = official_act.Agent(holder, args).to(device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-4, weight_decay=1e-4)
    ema = NonDeepSpeedEMAModel(parameters=agent.parameters(), power=0.75)
    observations = {
        "state": torch.zeros((1, 42), device=device),
        "rgb": torch.zeros((1, 1, 3, 224, 224), dtype=torch.uint8, device=device),
    }
    actions = torch.zeros((1, 30, 4), device=device)
    losses = agent.compute_loss(observations, actions)
    optimizer.zero_grad(set_to_none=True)
    losses["loss"].backward()
    optimizer.step()
    ema.step(agent.parameters())
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "scope": "official_architecture_cpu_or_gpu_smoke_only",
        "device": str(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "upstream_module": str(Path(official_act.__file__).resolve()),
        "backbone": "resnet18",
        "rgb_shape": [1, 1, 3, 224, 224],
        "action_shape": [1, 30, 4],
        "parameter_count": sum(parameter.numel() for parameter in agent.parameters()),
        "optimizer_step": 1,
        "ema_implementation": "NonDeepSpeedEMAModel",
        "ema_optimization_step": ema.optimization_step,
        "ema_state_fields": sorted(ema.state_dict()),
        "loss": float(losses["loss"].item()),
        "l1": float(losses["l1"].item()),
        "kl": float(losses["kl"].item()),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(cli.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
