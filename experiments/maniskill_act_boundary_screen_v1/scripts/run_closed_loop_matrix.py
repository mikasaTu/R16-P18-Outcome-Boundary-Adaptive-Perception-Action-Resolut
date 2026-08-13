#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from protocol_common import (  # noqa: E402
    FORMAL_TASKS,
    MODEL_SEEDS,
    PROTOCOL_ID,
    sha256_file,
    write_json,
)


@dataclass(frozen=True)
class Job:
    name: str
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, default=2)
    parser.add_argument("--num-envs", type=int, default=20)
    return parser.parse_args()


def valid_completion(
    output_dir: Path,
    run_dir: Path,
    seed_manifest: Path,
    task_id: str,
    model_seed: int,
) -> bool:
    summary_path = output_dir / "summary.json"
    episodes_path = output_dir / "episodes.jsonl"
    selection_path = run_dir / "checkpoint_selection.json"
    if not summary_path.is_file() or not episodes_path.is_file():
        return False
    try:
        value = json.loads(summary_path.read_text(encoding="utf-8"))
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        seed_bank = json.loads(seed_manifest.read_text(encoding="utf-8"))[
            "formal_tasks"
        ][task_id]["closed_loop_test_seeds"]
        rows = [
            json.loads(line)
            for line in episodes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        selected = selection["selected"]
        checkpoint_path = Path(selected["path"]) / "checkpoint.pt"
        expected_bindings = {
            "evaluator_sha256": sha256_file(
                SCRIPT_DIR / "evaluate_official_act_protocol.py"
            ),
            "seed_manifest_sha256": sha256_file(seed_manifest),
            "checkpoint_selection_sha256": sha256_file(selection_path),
            "selected_checkpoint_step": int(selected["step"]),
            "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        }
        row_seeds = [int(row["episode_seed"]) for row in rows]
        return bool(
            value.get("status") == "EVALUATION_COMPLETE"
            and value.get("protocol_id") == PROTOCOL_ID
            and value.get("task_id") == task_id
            and int(value.get("model_seed", -1)) == model_seed
            and int(value.get("episodes", -1)) == 100
            and value.get("test_metrics_used_for_selection") is False
            and selection.get("test_metrics_used") is False
            and value.get("selected_checkpoint") == selected
            and value.get("source_bindings") == expected_bindings
            and value.get("episodes_jsonl_sha256") == sha256_file(episodes_path)
            and checkpoint_path.is_file()
            and sha256_file(checkpoint_path) == selected["checkpoint_sha256"]
            and len(rows) == 100
            and row_seeds == [int(seed) for seed in seed_bank]
            and len(set(row_seeds)) == 100
            and all(int(row["model_seed"]) == model_seed for row in rows)
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def jobs(args: argparse.Namespace) -> list[Job]:
    result = []
    for task_id in FORMAL_TASKS:
        for model_seed in MODEL_SEEDS:
            run_dir = args.checkpoint_root / task_id / f"seed_{model_seed}"
            output_dir = args.evaluation_root / task_id / f"seed_{model_seed}"
            if valid_completion(
                output_dir,
                run_dir,
                args.seed_manifest,
                task_id,
                model_seed,
            ):
                continue
            result.append(
                Job(
                    name=f"evaluate_{task_id}_seed{model_seed}",
                    command=[
                        str(args.python),
                        str(SCRIPT_DIR / "evaluate_official_act_protocol.py"),
                        "--task-id",
                        task_id,
                        "--model-seed",
                        str(model_seed),
                        "--run-dir",
                        str(run_dir),
                        "--seed-manifest",
                        str(args.seed_manifest),
                        "--output-dir",
                        str(output_dir),
                        "--num-envs",
                        str(args.num_envs),
                        "--device",
                        "cuda",
                    ],
                )
            )
    return result


def execute(all_jobs: list[Job], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not 1 <= args.gpu_count <= 2:
        raise ValueError("this protocol permits one or two GPUs only")
    pending = deque(all_jobs)
    free_gpus = deque(range(args.gpu_count))
    active: dict[int, tuple[Job, subprocess.Popen[Any], Any, int, float]] = {}
    records: list[dict[str, Any]] = []
    logs = args.artifact_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    while pending or active:
        while pending and free_gpus:
            gpu = free_gpus.popleft()
            job = pending.popleft()
            handle = (logs / f"{job.name}.log").open("a", encoding="utf-8")
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
            print(f"EVAL_START name={job.name} gpu={gpu} pid={process.pid}", flush=True)
        finished: list[int] = []
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
            records.append(record)
            finished.append(pid)
            free_gpus.append(gpu)
            print(f"EVAL_FINISH name={job.name} gpu={gpu} exit={status}", flush=True)
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
                write_json(
                    args.artifact_dir / "EVALUATION_MATRIX_FAILURE.json",
                    {"protocol_id": PROTOCOL_ID, "status": "FAIL", "failed": record},
                )
                raise RuntimeError(f"evaluation failed: {job.name}")
        for pid in finished:
            active.pop(pid)
        if active and not finished:
            time.sleep(5)
    return records


def main() -> None:
    args = parse_args()
    if os.getuid() != 2254 or os.getgid() != 2254:
        raise RuntimeError("formal evaluation requires uid:gid 2254:2254")
    import torch

    if torch.cuda.device_count() != args.gpu_count:
        raise RuntimeError("visible GPU count differs from the fixed evaluation matrix")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.evaluation_root.mkdir(parents=True, exist_ok=True)
    processes = execute(jobs(args), args)
    baseline_output = args.evaluation_root / "baseline_gate.json"
    subprocess.run(
        [
            str(args.python),
            str(SCRIPT_DIR / "summarize_baseline.py"),
            "--evaluation-root",
            str(args.evaluation_root),
            "--seed-manifest",
            str(args.seed_manifest),
            "--output",
            str(baseline_output),
        ],
        check=True,
    )
    gate = json.loads(baseline_output.read_text(encoding="utf-8"))
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "EVALUATION_MATRIX_COMPLETE",
        "processes": processes,
        "tasks": list(FORMAL_TASKS),
        "model_seeds": list(MODEL_SEEDS),
        "baseline_gate_status": gate["status"],
        "continue_to_oracle_probe": gate["continue_to_oracle_probe"],
        "completed_at_unix": time.time(),
    }
    write_json(args.artifact_dir / "EVALUATION_MATRIX_COMPLETE.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
