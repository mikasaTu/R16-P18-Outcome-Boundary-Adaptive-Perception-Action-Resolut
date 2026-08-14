#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import MODEL_SEEDS, PROTOCOL_ID, append_jsonl, sha256_file, write_json
from stage25_runtime import (
    TASK_CONFIGS,
    evaluate_policy_batch,
    load_policy_from_checkpoint,
    make_env,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", choices=sorted(TASK_CONFIGS), required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--seed-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument(
        "--mode",
        choices=(
            "fixed_horizon",
            "terminate_first_success",
            "terminate_hold5",
            "neutral_after_hold5",
        ),
        default="fixed_horizon",
    )
    parser.add_argument("--record-trace", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_seed not in MODEL_SEEDS:
        raise ValueError("model seed outside frozen set")
    if not torch.cuda.is_available():
        raise RuntimeError("formal evaluation requires CUDA")
    seed_payload = json.loads(args.seed_bank.read_text(encoding="utf-8"))
    seeds = [int(value) for value in seed_payload["tasks"][args.task_id]]
    if args.max_episodes is not None:
        seeds = seeds[: args.max_episodes]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise RuntimeError("invalid seed bank")
    if len(seeds) % args.num_envs:
        raise ValueError("num-envs must divide selected episode count")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = args.output_dir / "episodes.jsonl"
    if episodes_path.exists():
        raise FileExistsError(f"refusing to overwrite formal raw output: {episodes_path}")
    device = torch.device("cuda")
    env = make_env(args.task_id, args.num_envs)
    agent, payload = load_policy_from_checkpoint(
        env,
        args.task_id,
        args.model_seed,
        args.checkpoint,
        device,
        args.checkpoint_sha256,
    )
    records = []
    started = time.time()
    try:
        for offset in range(0, len(seeds), args.num_envs):
            batch = evaluate_policy_batch(
                env,
                agent,
                seeds[offset : offset + args.num_envs],
                args.task_id,
                device,
                mode=args.mode,
                record_trace=args.record_trace,
            )
            for record in batch:
                record.update(
                    {
                        "model_seed": args.model_seed,
                        "checkpoint_step": args.checkpoint_step,
                        "checkpoint_sha256": args.checkpoint_sha256,
                        "seed_bank_sha256": sha256_file(args.seed_bank),
                    }
                )
                append_jsonl(episodes_path, record)
            records.extend(batch)
            if offset == 0:
                write_json(
                    args.output_dir / "FIRST_REAL_ROLLOUT.json",
                    {
                        "protocol_id": PROTOCOL_ID,
                        "status": "FIRST_REAL_ROLLOUT_BATCH_COMPLETE",
                        "task_id": args.task_id,
                        "model_seed": args.model_seed,
                        "checkpoint_step": args.checkpoint_step,
                        "episode_seeds": seeds[: args.num_envs],
                        "episodes_jsonl_sha256": sha256_file(episodes_path),
                        "completed_at_unix": time.time(),
                    },
                )
            print(
                f"STAGE25_EVAL_PROGRESS task={args.task_id} seed={args.model_seed} "
                f"step={args.checkpoint_step} mode={args.mode} episodes={len(records)}/{len(seeds)}",
                flush=True,
            )
    finally:
        env.close()
    metrics = {
        key: float(np.mean([bool(row[key]) for row in records]))
        for key in ("success_once", "success_hold5", "success_at_end", "post_success_loss")
    }
    summary = {
        "protocol_id": PROTOCOL_ID,
        "status": "CHECKPOINT_EVALUATION_COMPLETE",
        "task_id": args.task_id,
        "model_seed": args.model_seed,
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "validation_loss": float(payload.get("validation_loss", float("nan"))),
        "mode": args.mode,
        "episodes": len(records),
        **metrics,
        "longest_success_streak_mean": float(
            np.mean([row["longest_success_streak"] for row in records])
        ),
        "first_success_step_mean_successes": float(
            np.mean(
                [row["first_success_step"] for row in records if row["first_success_step"] >= 0]
                or [float("nan")]
            )
        ),
        "total_policy_calls": int(sum(row["policy_calls"] for row in records)),
        "total_action_opportunities": int(
            sum(row["action_opportunities"] for row in records)
        ),
        "total_policy_latency_seconds": float(
            sum(row["policy_latency_seconds"] for row in records)
        ),
        "seed_bank_path": str(args.seed_bank),
        "seed_bank_sha256": sha256_file(args.seed_bank),
        "episodes_jsonl_sha256": sha256_file(episodes_path),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "wall_seconds": time.time() - started,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

