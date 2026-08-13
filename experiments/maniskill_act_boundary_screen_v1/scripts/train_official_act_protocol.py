#!/usr/bin/env python3
"""Protocol adapter around ManiSkill 3.0.1's official RGB ACT baseline.

The architecture and in-memory demonstration dataset are imported from the
pinned upstream checkout.  This adapter adds only the preregistered identity
split, deterministic validation loss, complete-state checkpoints, and resume.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader, Sampler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from protocol_common import PROTOCOL_ID, atomic_write_text, canonical_json, sha256_file  # noqa: E402

try:
    import train_rgbd as official_act
except ImportError as exc:  # pragma: no cover - exercised by launcher preflight
    raise RuntimeError(
        "The pinned ManiSkill examples/baselines/act directory must be on PYTHONPATH"
    ) from exc


UPSTREAM_COMMIT = "a4a4f9272ad64b1564035874b605ceb687b63ed8"
CANDIDATE_INTERVAL = 5_000


@dataclass(frozen=True)
class TrainConfig:
    protocol_id: str
    upstream_commit: str
    task_id: str
    seed: int
    control_mode: str
    train_h5: str
    validation_h5: str
    train_h5_sha256: str
    train_json_sha256: str
    validation_h5_sha256: str
    validation_json_sha256: str
    total_iterations: int
    batch_size: int
    validation_batch_size: int
    checkpoint_interval: int
    num_queries: int
    learning_rate: float
    backbone_learning_rate: float
    weight_decay: float
    kl_weight: float
    ema_power: float
    validation_seed: int
    sampler_version: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--control-mode", required=True)
    parser.add_argument("--train-h5", type=Path, required=True)
    parser.add_argument("--validation-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-iterations", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-interval", type=int, default=CANDIDATE_INTERVAL)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project", default="R16-P18-ManiSkill-ACT-Screen")
    parser.add_argument("--run-name")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required regular file is missing: {path}")


def dataset_digests(path: Path) -> tuple[str, str]:
    require_file(path)
    metadata_path = path.with_suffix(".json")
    require_file(metadata_path)
    return sha256_file(path), sha256_file(metadata_path)


def make_official_args(args: argparse.Namespace) -> official_act.Args:
    return official_act.Args(
        exp_name=args.run_name,
        seed=args.seed,
        torch_deterministic=True,
        cuda=True,
        track=args.track,
        wandb_project_name=args.wandb_project,
        capture_video=False,
        env_id=args.task_id,
        demo_path=str(args.train_h5),
        num_demos=None,
        total_iters=args.total_iterations,
        batch_size=args.batch_size,
        lr=1e-4,
        kl_weight=10,
        temporal_agg=True,
        position_embedding="sine",
        backbone="resnet18",
        lr_backbone=1e-5,
        masks=False,
        dilation=False,
        include_depth=False,
        enc_layers=2,
        dec_layers=4,
        dim_feedforward=512,
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        num_queries=30,
        pre_norm=False,
        log_freq=args.log_interval,
        eval_freq=CANDIDATE_INTERVAL,
        save_freq=CANDIDATE_INTERVAL,
        num_eval_episodes=100,
        num_eval_envs=100,
        sim_backend="physx_cuda",
        num_dataload_workers=args.num_workers,
        control_mode=args.control_mode,
    )


def preflight_summary(args: argparse.Namespace, config: TrainConfig) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "DRY_RUN_PASS",
        "task_id": args.task_id,
        "seed": args.seed,
        "device": args.device,
        "train_config": asdict(config),
        "train_config_sha256": config_sha256(config),
        "train_json_episodes": len(
            json.loads(args.train_h5.with_suffix(".json").read_text(encoding="utf-8"))["episodes"]
        ),
        "validation_json_episodes": len(
            json.loads(args.validation_h5.with_suffix(".json").read_text(encoding="utf-8"))["episodes"]
        ),
        "resume_checkpoint": None,
    }


def make_train_config(args: argparse.Namespace) -> TrainConfig:
    train_h5_sha, train_json_sha = dataset_digests(args.train_h5)
    val_h5_sha, val_json_sha = dataset_digests(args.validation_h5)
    return TrainConfig(
        protocol_id=PROTOCOL_ID,
        upstream_commit=UPSTREAM_COMMIT,
        task_id=args.task_id,
        seed=args.seed,
        control_mode=args.control_mode,
        train_h5=str(args.train_h5),
        validation_h5=str(args.validation_h5),
        train_h5_sha256=train_h5_sha,
        train_json_sha256=train_json_sha,
        validation_h5_sha256=val_h5_sha,
        validation_json_sha256=val_json_sha,
        total_iterations=args.total_iterations,
        batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        checkpoint_interval=args.checkpoint_interval,
        num_queries=30,
        learning_rate=1e-4,
        backbone_learning_rate=1e-5,
        weight_decay=1e-4,
        kl_weight=10,
        ema_power=0.75,
        validation_seed=16018,
        sampler_version="independent_epoch_sha256_v1",
    )


def config_sha256(config: TrainConfig) -> str:
    return hashlib.sha256(canonical_json(asdict(config))).hexdigest()


class DeterministicResumeBatchSampler(Sampler[list[int]]):
    """Without-replacement epochs addressable by completed optimizer step."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        seed: int,
        start_step: int,
        total_steps: int,
    ) -> None:
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.start_step = start_step
        self.total_steps = total_steps
        self.batches_per_epoch = dataset_size // batch_size
        if self.batches_per_epoch < 1:
            raise ValueError("dataset must contain at least one full batch")

    def _epoch_seed(self, epoch: int) -> int:
        payload = f"{PROTOCOL_ID}:{self.seed}:train-epoch:{epoch}"
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")

    def __iter__(self) -> Iterator[list[int]]:
        completed = self.start_step
        while completed < self.total_steps:
            epoch = completed // self.batches_per_epoch
            offset = completed % self.batches_per_epoch
            generator = torch.Generator().manual_seed(self._epoch_seed(epoch))
            permutation = torch.randperm(self.dataset_size, generator=generator).tolist()
            usable = permutation[: self.batches_per_epoch * self.batch_size]
            for batch_index in range(offset, self.batches_per_epoch):
                start = batch_index * self.batch_size
                yield usable[start : start + self.batch_size]
                completed += 1
                if completed >= self.total_steps:
                    return

    def __len__(self) -> int:
        return self.total_steps - self.start_step


def make_space_holder(dataset: Any) -> SimpleNamespace:
    sample = dataset[0]
    state = sample["observations"]["state"]
    rgb = sample["observations"]["rgb"]
    actions = sample["actions"]
    observation_space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(-np.inf, np.inf, shape=tuple(state.shape), dtype=np.float32),
            "rgb": gym.spaces.Box(0, 255, shape=tuple(rgb.shape), dtype=np.uint8),
        }
    )
    action_space = gym.spaces.Box(
        -1.0, 1.0, shape=(int(actions.shape[-1]),), dtype=np.float32
    )
    return SimpleNamespace(
        single_observation_space=observation_space,
        single_action_space=action_space,
    )


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@contextlib.contextmanager
def preserved_rng(seed: int) -> Iterator[None]:
    state = rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        restore_rng_state(state)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def checkpoint_dirs(checkpoint_root: Path) -> list[Path]:
    result: list[tuple[int, Path]] = []
    if checkpoint_root.is_dir():
        for path in checkpoint_root.glob("step_*"):
            try:
                step = int(path.name.removeprefix("step_"))
            except ValueError:
                continue
            if (path / "COMPLETE.json").is_file() and (path / "checkpoint.pt").is_file():
                result.append((step, path))
    return [path for _, path in sorted(result)]


def validate_checkpoint(path: Path, expected_config_sha256: str) -> dict[str, Any]:
    marker = json.loads((path / "COMPLETE.json").read_text(encoding="utf-8"))
    if marker["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError(f"checkpoint protocol mismatch: {path}")
    if marker["train_config_sha256"] != expected_config_sha256:
        raise RuntimeError(f"checkpoint config mismatch: {path}")
    if marker["checkpoint_sha256"] != sha256_file(path / "checkpoint.pt"):
        raise RuntimeError(f"checkpoint digest mismatch: {path}")
    return marker


def discover_resume(checkpoint_root: Path, expected_config_sha256: str) -> Path | None:
    valid: list[Path] = []
    for path in checkpoint_dirs(checkpoint_root):
        validate_checkpoint(path, expected_config_sha256)
        valid.append(path)
    return valid[-1] if valid else None


def save_checkpoint(
    checkpoint_root: Path,
    step: int,
    validation_loss: float,
    train_config_sha: str,
    train_config: TrainConfig,
    agent: torch.nn.Module,
    ema_agent: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    ema: EMAModel,
) -> Path:
    final_dir = checkpoint_root / f"step_{step:09d}"
    if final_dir.exists():
        marker = validate_checkpoint(final_dir, train_config_sha)
        if int(marker["global_iteration"]) != step:
            raise RuntimeError(f"existing checkpoint has wrong step: {final_dir}")
        return final_dir
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_root / f".step_{step:09d}.tmp-{os.getpid()}"
    temporary.mkdir(mode=0o700)
    checkpoint_path = temporary / "checkpoint.pt"
    payload = {
        "protocol_id": PROTOCOL_ID,
        "upstream_commit": UPSTREAM_COMMIT,
        "train_config": asdict(train_config),
        "train_config_sha256": train_config_sha,
        "global_iteration": step,
        "validation_loss": validation_loss,
        "model": agent.state_dict(),
        "ema_model": ema_agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema_state": ema.state_dict(),
        "rng_state": rng_state(),
        "saved_at_unix": time.time(),
    }
    torch.save(payload, checkpoint_path)
    with checkpoint_path.open("rb") as handle:
        os.fsync(handle.fileno())
    marker = {
        "protocol_id": PROTOCOL_ID,
        "upstream_commit": UPSTREAM_COMMIT,
        "global_iteration": step,
        "validation_loss": validation_loss,
        "train_config_sha256": train_config_sha,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "complete": True,
    }
    atomic_json(temporary / "COMPLETE.json", marker)
    os.replace(temporary, final_dir)
    directory_fd = os.open(checkpoint_root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return final_dir


def validation_loss(
    agent: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    seed: int,
) -> float:
    total = 0.0
    count = 0
    agent.eval()
    with preserved_rng(seed), torch.no_grad():
        for data_batch in dataloader:
            obs = {key: value.to(device, non_blocking=True) for key, value in data_batch["observations"].items()}
            actions = data_batch["actions"].to(device, non_blocking=True)
            loss = agent.compute_loss(obs, actions)["loss"]
            batch = int(actions.shape[0])
            total += float(loss.item()) * batch
            count += batch
    if count == 0:
        raise RuntimeError("validation split yielded no samples")
    return total / count


def select_checkpoint(checkpoint_root: Path, train_config_sha: str) -> dict[str, Any]:
    candidates = []
    for path in checkpoint_dirs(checkpoint_root):
        marker = validate_checkpoint(path, train_config_sha)
        candidates.append(
            {
                "step": int(marker["global_iteration"]),
                "validation_loss": float(marker["validation_loss"]),
                "path": str(path),
                "checkpoint_sha256": marker["checkpoint_sha256"],
            }
        )
    if not candidates:
        raise RuntimeError("no complete candidate checkpoints")
    selected = min(candidates, key=lambda item: (item["validation_loss"], item["step"]))
    return {
        "protocol_id": PROTOCOL_ID,
        "selection_metric": "deterministic_mean_validation_imitation_loss",
        "direction": "minimize",
        "tie_break": "earliest_step",
        "train_config_sha256": train_config_sha,
        "candidates": candidates,
        "selected": selected,
        "test_metrics_used": False,
    }


def main() -> None:
    cli = parse_args()
    if cli.seed not in (16018, 16019, 16020):
        raise ValueError("model seed is outside the frozen set")
    if cli.checkpoint_interval != CANDIDATE_INTERVAL:
        raise ValueError("checkpoint interval is frozen at 5000 updates")
    if cli.total_iterations % cli.checkpoint_interval:
        raise ValueError("total iterations must end on a checkpoint candidate")
    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    if cli.output_dir.is_symlink():
        raise RuntimeError("output directory must not be a symlink")

    run_name = cli.run_name or f"{cli.task_id}-seed{cli.seed}"
    cli.run_name = run_name
    official_args = make_official_args(cli)
    official_act.args = official_args
    config = make_train_config(cli)
    config_sha = config_sha256(config)
    if cli.dry_run:
        summary = preflight_summary(cli, config)
        summary["resume_checkpoint"] = (
            str(discover_resume(cli.output_dir / "checkpoints", config_sha))
            if (cli.output_dir / "checkpoints").exists()
            else None
        )
        atomic_json(cli.output_dir / "dry_run.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    atomic_json(cli.output_dir / "train_config.json", {**asdict(config), "sha256": config_sha})

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset = official_act.SmallDemoDataset_ACTPolicy(
        str(cli.train_h5), official_args.num_queries, num_traj=None, include_depth=False
    )
    validation_dataset = official_act.SmallDemoDataset_ACTPolicy(
        str(cli.validation_h5), official_args.num_queries, num_traj=None, include_depth=False
    )
    if train_dataset.num_traj != 200 or validation_dataset.num_traj != 50:
        raise RuntimeError(
            f"identity split mismatch: train={train_dataset.num_traj}, validation={validation_dataset.num_traj}"
        )
    space_holder = make_space_holder(train_dataset)
    agent = official_act.Agent(space_holder, official_args).to(device)
    ema_agent = official_act.Agent(space_holder, official_args).to(device)
    param_dicts = [
        {
            "params": [
                parameter
                for name, parameter in agent.named_parameters()
                if "backbone" not in name and parameter.requires_grad
            ]
        },
        {
            "params": [
                parameter
                for name, parameter in agent.named_parameters()
                if "backbone" in name and parameter.requires_grad
            ],
            "lr": config.backbone_learning_rate,
        },
    ]
    optimizer = optim.AdamW(
        param_dicts, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=int((2 / 3) * config.total_iterations), gamma=0.1
    )
    ema = EMAModel(parameters=agent.parameters(), power=config.ema_power)

    checkpoint_root = cli.output_dir / "checkpoints"
    resume_path = discover_resume(checkpoint_root, config_sha)
    completed_steps = 0
    if resume_path is not None:
        payload = torch.load(resume_path / "checkpoint.pt", map_location=device, weights_only=False)
        agent.load_state_dict(payload["model"])
        ema_agent.load_state_dict(payload["ema_model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        ema.load_state_dict(payload["ema_state"])
        restore_rng_state(payload["rng_state"])
        completed_steps = int(payload["global_iteration"])
        print(f"AUTO_RESUME=1 RESUME_STEP={completed_steps} RESUME_DIR={resume_path}", flush=True)
    else:
        print("AUTO_RESUME=0 RESUME_STEP=0 RESUME_DIR=none", flush=True)

    batch_sampler = DeterministicResumeBatchSampler(
        len(train_dataset), config.batch_size, cli.seed, completed_steps, config.total_iterations
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=cli.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=lambda worker_id: official_act.worker_init_fn(worker_id, base_seed=cli.seed),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.validation_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cli.num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics_path = cli.output_dir / "training_metrics.jsonl"

    wandb_run = None
    if cli.track:
        import wandb

        wandb_run = wandb.init(
            project=cli.wandb_project,
            entity=os.environ.get("WANDB_ENTITY"),
            name=run_name,
            id=f"{PROTOCOL_ID}-{cli.task_id}-{cli.seed}".replace("/", "-"),
            resume="allow",
            config=asdict(config),
            tags=["official-act", "stage2", "baseline-screen"],
        )

    agent.train()
    start_time = time.monotonic()
    for offset, data_batch in enumerate(train_loader, start=1):
        step = completed_steps + offset
        obs = {
            key: value.to(device, non_blocking=True)
            for key, value in data_batch["observations"].items()
        }
        actions = data_batch["actions"].to(device, non_blocking=True)
        loss_dict = agent.compute_loss(obs, actions)
        optimizer.zero_grad(set_to_none=True)
        loss_dict["loss"].backward()
        optimizer.step()
        scheduler.step()
        ema.step(agent.parameters())

        if step == 1 or step % cli.log_interval == 0:
            record = {
                "protocol_id": PROTOCOL_ID,
                "task_id": cli.task_id,
                "seed": cli.seed,
                "global_iteration": step,
                "loss": float(loss_dict["loss"].item()),
                "l1": float(loss_dict["l1"].item()),
                "kl": float(loss_dict["kl"].item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "backbone_learning_rate": float(optimizer.param_groups[1]["lr"]),
                "elapsed_seconds": time.monotonic() - start_time,
                "timestamp_unix": time.time(),
            }
            append_jsonl(metrics_path, record)
            print(
                f"TRAIN_STEP task={cli.task_id} seed={cli.seed} step={step} loss={record['loss']:.8f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(record, step=step)

        if step % config.checkpoint_interval == 0:
            ema.copy_to(ema_agent.parameters())
            val_loss = validation_loss(
                ema_agent, validation_loader, device, config.validation_seed
            )
            path = save_checkpoint(
                checkpoint_root,
                step,
                val_loss,
                config_sha,
                config,
                agent,
                ema_agent,
                optimizer,
                scheduler,
                ema,
            )
            candidate = {
                "protocol_id": PROTOCOL_ID,
                "task_id": cli.task_id,
                "seed": cli.seed,
                "global_iteration": step,
                "validation_loss": val_loss,
                "checkpoint": str(path),
                "timestamp_unix": time.time(),
            }
            append_jsonl(cli.output_dir / "validation_metrics.jsonl", candidate)
            print(
                f"VALIDATION_CANDIDATE task={cli.task_id} seed={cli.seed} step={step} loss={val_loss:.8f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log({"validation/loss": val_loss}, step=step)
            agent.train()

    selection = select_checkpoint(checkpoint_root, config_sha)
    atomic_json(cli.output_dir / "checkpoint_selection.json", selection)
    completion = {
        "protocol_id": PROTOCOL_ID,
        "status": "TRAINING_COMPLETE",
        "task_id": cli.task_id,
        "seed": cli.seed,
        "global_iteration": config.total_iterations,
        "train_config_sha256": config_sha,
        "selected_checkpoint": selection["selected"],
        "completed_at_unix": time.time(),
    }
    atomic_json(cli.output_dir / "TRAINING_COMPLETE.json", completion)
    if wandb_run is not None:
        wandb_run.finish()
    print(
        f"TRAINING_COMPLETE task={cli.task_id} seed={cli.seed} selected_step={selection['selected']['step']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
