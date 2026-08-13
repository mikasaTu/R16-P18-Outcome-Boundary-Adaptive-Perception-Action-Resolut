#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from protocol_common import PROTOCOL_ID, SPLIT_COUNTS, atomic_write_text  # noqa: E402


TASKS = {
    "PlugCharger-v1": {
        "control_mode": "pd_ee_delta_pose",
        "total_iterations": 100_000,
        "replay_max_retry": 9,
    },
    "PushT-v1": {
        "control_mode": "pd_ee_delta_pose",
        "total_iterations": 100_000,
        "replay_max_retry": 3,
    },
    "StackCube-v1": {
        "control_mode": "pd_ee_delta_pos",
        "total_iterations": 30_000,
        "replay_max_retry": 9,
    },
    "PushCube-v1": {
        "control_mode": "pd_ee_delta_pos",
        "total_iterations": 30_000,
        "replay_max_retry": 9,
    },
}
SEEDS = (16018, 16019, 16020)


@dataclass(frozen=True)
class Job:
    name: str
    command: list[str]
    gpu: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("replay", "train", "replay-and-train"), required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--selected-raw-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, default=2)
    parser.add_argument("--track", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_id() -> str:
    return os.environ.get("PAI_CANARY_RUN_ID", "dev14-local")


def nonce() -> str:
    return os.environ.get("PAI_CANARY_NONCE", "local-no-pai-nonce")


def make_first_work(artifact_dir: Path, source: dict[str, Any]) -> None:
    path = artifact_dir / "FIRST_REAL_WORK.json"
    if path.exists():
        return
    value = {
        "protocol_id": PROTOCOL_ID,
        "status": "persisted_real_optimizer_step_and_loss",
        "run_id": run_id(),
        "nonce": nonce(),
        "uid": os.getuid(),
        "gid": os.getgid(),
        "source_metric": source,
        "observed_at_unix": time.time(),
    }
    atomic_json(path, value)


def metric_key(task_id: str, seed: int) -> str:
    return f"{task_id}/seed_{seed}"


def latest_metrics(checkpoint_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for task_id in TASKS:
        for seed in SEEDS:
            path = checkpoint_root / task_id / f"seed_{seed}" / "training_metrics.jsonl"
            if not path.is_file():
                continue
            last = ""
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        last = line
            if not last:
                continue
            value = json.loads(last)
            if int(value.get("global_iteration", 0)) >= 1 and "loss" in value:
                result[metric_key(task_id, seed)] = value
    return result


def read_new_metric(
    checkpoint_root: Path,
    baseline_steps: dict[str, int],
) -> dict[str, Any] | None:
    for key, value in latest_metrics(checkpoint_root).items():
        if int(value["global_iteration"]) > baseline_steps.get(key, 0):
            return value
    return None


def execute_jobs(
    jobs: list[Job],
    artifact_dir: Path,
    gpu_count: int,
    checkpoint_root: Path | None = None,
    baseline_steps: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    if gpu_count < 1 or gpu_count > 2:
        raise ValueError("this protocol permits one or two GPUs only")
    pending = deque(jobs)
    free_gpus = deque(range(gpu_count))
    active: dict[int, tuple[Job, subprocess.Popen[Any], Any, int, float]] = {}
    results: list[dict[str, Any]] = []
    logs = artifact_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    while pending or active:
        while pending and free_gpus:
            gpu = free_gpus.popleft()
            job = pending.popleft()
            log_path = logs / f"{job.name}.log"
            handle = log_path.open("a", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["PYTHONUNBUFFERED"] = "1"
            started = time.time()
            process = subprocess.Popen(
                job.command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment,
                text=True,
            )
            active[process.pid] = (job, process, handle, gpu, started)
            print(f"MATRIX_START name={job.name} gpu={gpu} pid={process.pid}", flush=True)
        if checkpoint_root is not None:
            metric = read_new_metric(checkpoint_root, baseline_steps or {})
            if metric is not None:
                make_first_work(artifact_dir, metric)
        finished = []
        for pid, (job, process, handle, gpu, started) in active.items():
            status = process.poll()
            if status is None:
                continue
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            record = {
                "name": job.name,
                "gpu": gpu,
                "exit_code": status,
                "started_at_unix": started,
                "finished_at_unix": time.time(),
                "log": str(logs / f"{job.name}.log"),
            }
            results.append(record)
            finished.append(pid)
            free_gpus.append(gpu)
            print(
                f"MATRIX_FINISH name={job.name} gpu={gpu} exit_code={status}",
                flush=True,
            )
            if status != 0:
                for other_pid, (_, other, other_handle, _, _) in active.items():
                    if other_pid != pid and other.poll() is None:
                        other.terminate()
                        try:
                            other.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            other.kill()
                            other.wait(timeout=30)
                    if other_pid != pid and not other_handle.closed:
                        other_handle.flush()
                        os.fsync(other_handle.fileno())
                        other_handle.close()
                atomic_json(
                    artifact_dir / "MATRIX_FAILURE.json",
                    {
                        "protocol_id": PROTOCOL_ID,
                        "status": "FAIL",
                        "failed_job": record,
                        "completed_jobs": results,
                    },
                )
                raise RuntimeError(f"matrix job failed: {job.name}, exit={status}")
        for pid in finished:
            active.pop(pid)
        if active and not finished:
            time.sleep(5)
    return results


def replay_jobs(args: argparse.Namespace) -> list[Job]:
    summaries = args.selected_raw_root / "replay_summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    jobs = []
    for task_id in TASKS:
        summary = summaries / f"{task_id}.json"
        jobs.append(
            Job(
                name=f"replay_{task_id}",
                command=[
                    str(args.python),
                    str(SCRIPT_DIR / "replay_rgb_datasets.py"),
                    "--task-id",
                    task_id,
                    "--selected-raw-root",
                    str(args.selected_raw_root),
                    "--summary-output",
                    str(summary),
                    "--python",
                    str(args.python),
                    "--max-retry",
                    str(TASKS[task_id]["replay_max_retry"]),
                ],
            )
        )
    return jobs


def train_jobs(args: argparse.Namespace) -> list[Job]:
    jobs = []
    for task_id in ("PushCube-v1", "StackCube-v1", "PlugCharger-v1", "PushT-v1"):
        task = TASKS[task_id]
        train_h5 = args.selected_raw_root / task_id / "train" / (
            f"trajectory.rgb.{task['control_mode']}."
            + ("physx_cuda.h5" if task_id == "PushT-v1" else "physx_cpu.h5")
        )
        validation_h5 = args.selected_raw_root / task_id / "validation" / train_h5.name
        for seed in SEEDS:
            output = args.checkpoint_root / task_id / f"seed_{seed}"
            completion = output / "TRAINING_COMPLETE.json"
            if completion.is_file():
                value = json.loads(completion.read_text(encoding="utf-8"))
                if (
                    value.get("status") == "TRAINING_COMPLETE"
                    and int(value.get("global_iteration", -1)) == task["total_iterations"]
                ):
                    continue
            command = [
                str(args.python),
                str(SCRIPT_DIR / "train_official_act_protocol.py"),
                "--task-id",
                task_id,
                "--seed",
                str(seed),
                "--control-mode",
                task["control_mode"],
                "--train-h5",
                str(train_h5),
                "--validation-h5",
                str(validation_h5),
                "--output-dir",
                str(output),
                "--total-iterations",
                str(task["total_iterations"]),
                "--batch-size",
                "256",
                "--validation-batch-size",
                "256",
                "--checkpoint-interval",
                "5000",
                "--log-interval",
                "100",
                "--run-name",
                f"r16p18-ms3-act-{task_id}-seed{seed}",
            ]
            if args.track:
                command.append("--track")
            jobs.append(Job(name=f"train_{task_id}_seed{seed}", command=command))
    return jobs


def validate_replay_gate(args: argparse.Namespace) -> list[dict[str, Any]]:
    summaries = []
    for task_id in TASKS:
        path = args.selected_raw_root / "replay_summaries" / f"{task_id}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        expected_total = sum(SPLIT_COUNTS.values())
        saved_total = int(value.get("episodes_saved_successful", -1))
        if (
            value.get("status") != "PASS"
            or value.get("episodes_attempted") != expected_total
            or saved_total < math.ceil(expected_total * 0.95)
            or saved_total > expected_total
            or float(value.get("replay_success_rate", -1.0)) < 0.95
        ):
            raise RuntimeError(f"formal replay gate failed: {task_id}")
        split_records = value.get("splits")
        if not isinstance(split_records, list) or len(split_records) != len(SPLIT_COUNTS):
            raise RuntimeError(f"formal replay split inventory failed: {task_id}")
        by_name = {record.get("split"): record for record in split_records}
        if set(by_name) != set(SPLIT_COUNTS):
            raise RuntimeError(f"formal replay split identities failed: {task_id}")
        for split, expected_count in SPLIT_COUNTS.items():
            record = by_name[split]
            saved_count = int(record.get("episodes_saved_successful", -1))
            if (
                record.get("status") != "PASS"
                or record.get("episodes_attempted") != expected_count
                or saved_count < math.ceil(expected_count * 0.95)
                or saved_count > expected_count
                or float(record.get("replay_success_rate", -1.0)) < 0.95
            ):
                raise RuntimeError(f"formal replay split gate failed: {task_id}/{split}")
        summaries.append(value)
    return summaries


def validate_training_gate(args: argparse.Namespace) -> list[dict[str, Any]]:
    completions = []
    for task_id, task in TASKS.items():
        for seed in SEEDS:
            path = args.checkpoint_root / task_id / f"seed_{seed}" / "TRAINING_COMPLETE.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                value.get("status") != "TRAINING_COMPLETE"
                or int(value.get("global_iteration", -1)) != task["total_iterations"]
            ):
                raise RuntimeError(f"training completion gate failed: {task_id}/{seed}")
            completions.append(value)
    return completions


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    for path in (args.artifact_dir, args.checkpoint_root, args.selected_raw_root):
        if path.is_symlink():
            raise RuntimeError(f"symlinked persistent path is forbidden: {path}")
        if not os.access(path, os.W_OK):
            raise PermissionError(path)
    if os.getuid() != 2254 or os.getgid() != 2254:
        raise RuntimeError(f"expected workload identity 2254:2254, got {os.getuid()}:{os.getgid()}")
    if torch_gpu_count() != args.gpu_count:
        raise RuntimeError("visible GPU count differs from frozen matrix shape")

    phase_results: dict[str, Any] = {}
    if args.phase in ("replay", "replay-and-train"):
        phase_results["replay_processes"] = execute_jobs(
            replay_jobs(args), args.artifact_dir, args.gpu_count
        )
        phase_results["replay_gate"] = validate_replay_gate(args)
        atomic_json(
            args.artifact_dir / "RGB_REPLAY_MATRIX_COMPLETE.json",
            {
                "protocol_id": PROTOCOL_ID,
                "status": "PASS",
                "tasks": [item["task_id"] for item in phase_results["replay_gate"]],
                "episodes_attempted": sum(
                    item["episodes_attempted"] for item in phase_results["replay_gate"]
                ),
                "episodes_saved_successful": sum(
                    item["episodes_saved_successful"]
                    for item in phase_results["replay_gate"]
                ),
            },
        )
    if args.phase in ("train", "replay-and-train"):
        validate_replay_gate(args)
        baseline_steps = {
            key: int(value["global_iteration"])
            for key, value in latest_metrics(args.checkpoint_root).items()
        }
        phase_results["training_processes"] = execute_jobs(
            train_jobs(args),
            args.artifact_dir,
            args.gpu_count,
            checkpoint_root=args.checkpoint_root,
            baseline_steps=baseline_steps,
        )
        phase_results["training_gate"] = validate_training_gate(args)
        metric = read_new_metric(args.checkpoint_root, baseline_steps)
        if metric is None:
            if not (args.artifact_dir / "FIRST_REAL_WORK.json").is_file():
                raise RuntimeError("no optimizer step and loss newer than this launch")
            metric = json.loads(
                (args.artifact_dir / "FIRST_REAL_WORK.json").read_text(encoding="utf-8")
            )["source_metric"]
        make_first_work(args.artifact_dir, metric)
        atomic_json(
            args.artifact_dir / "TRAINING_MATRIX_COMPLETE.json",
            {
                "protocol_id": PROTOCOL_ID,
                "status": "TRAINING_MATRIX_COMPLETE",
                "run_id": run_id(),
                "nonce": nonce(),
                "models": 12,
                "tasks": list(TASKS),
                "seeds": list(SEEDS),
                "completed_at_unix": time.time(),
            },
        )
    atomic_json(
        args.artifact_dir / "FORMAL_MATRIX_RESULT.json",
        {"protocol_id": PROTOCOL_ID, "status": "PASS", "phase": args.phase, **phase_results},
    )
    print(f"FORMAL_MATRIX_COMPLETE phase={args.phase}", flush=True)


def torch_gpu_count() -> int:
    import torch

    return torch.cuda.device_count()


if __name__ == "__main__":
    main()
