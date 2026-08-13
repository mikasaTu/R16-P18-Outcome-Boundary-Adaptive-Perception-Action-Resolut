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
from protocol_common import MODEL_SEEDS, PROTOCOL_ID, write_json  # noqa: E402


NEGATIVE_CONTROL = "PushCube-v1"


@dataclass(frozen=True)
class Job:
    name: str
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--baseline-gate", type=Path, required=True)
    parser.add_argument("--selected-raw-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--state-bank-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, default=2)
    return parser.parse_args()


def selected_h5(root: Path, task_id: str, split: str) -> Path:
    values = sorted((root / task_id / split).glob("trajectory.rgb.*.h5"))
    if len(values) != 1:
        raise RuntimeError(
            f"expected exactly one replayed RGB HDF5 for {task_id}/{split}, got {values}"
        )
    return values[0]


def state_bank_terminal(path: Path, task_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        value.get("protocol_id") == PROTOCOL_ID
        and value.get("task_id") == task_id
        and value.get("status") in {"STATE_BANK_COMPLETE", "STATE_BANK_GATE_FAIL"}
    )


def oracle_complete(path: Path, task_id: str, model_seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        value.get("protocol_id") == PROTOCOL_ID
        and value.get("status") == "ORACLE_ATLAS_COMPLETE"
        and value.get("task_id") == task_id
        and int(value.get("model_seed", -1)) == model_seed
        and int(value.get("states", -1)) == 64
    )


def execute(jobs: list[Job], args: argparse.Namespace, phase: str) -> list[dict[str, Any]]:
    if not 1 <= args.gpu_count <= 2:
        raise ValueError("the frozen protocol permits one or two GPUs only")
    pending = deque(jobs)
    free_gpus = deque(range(args.gpu_count))
    active: dict[int, tuple[Job, subprocess.Popen[Any], Any, int, float]] = {}
    records: list[dict[str, Any]] = []
    logs = args.artifact_dir / "logs"
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
            print(
                f"ORACLE_START phase={phase} name={job.name} gpu={gpu} pid={process.pid}",
                flush=True,
            )
        finished: list[int] = []
        for pid, (job, process, handle, gpu, started) in list(active.items()):
            exit_code = process.poll()
            if exit_code is None:
                continue
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            record = {
                "phase": phase,
                "name": job.name,
                "gpu": gpu,
                "exit_code": exit_code,
                "started_at_unix": started,
                "finished_at_unix": time.time(),
                "log": str(logs / f"{job.name}.log"),
            }
            records.append(record)
            finished.append(pid)
            free_gpus.append(gpu)
            print(
                f"ORACLE_FINISH phase={phase} name={job.name} gpu={gpu} exit={exit_code}",
                flush=True,
            )
            first_real_work = args.artifact_dir / "FIRST_REAL_WORK.json"
            if exit_code == 0 and not first_real_work.exists():
                write_json(
                    first_real_work,
                    {
                        "protocol_id": PROTOCOL_ID,
                        "status": "FIRST_REAL_WORK",
                        "evidence_scope": "completed_simulator_subtask",
                        "phase": phase,
                        "job": job.name,
                        "gpu": gpu,
                        "started_at_unix": started,
                        "completed_at_unix": record["finished_at_unix"],
                        "log": record["log"],
                    },
                )
            if exit_code != 0:
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
                    args.artifact_dir / "ORACLE_MATRIX_FAILURE.json",
                    {
                        "protocol_id": PROTOCOL_ID,
                        "status": "INFRASTRUCTURE_OR_CODE_FAILURE",
                        "failed": record,
                    },
                )
                raise RuntimeError(f"oracle matrix job failed: {job.name}")
        for pid in finished:
            active.pop(pid)
        if active and not finished:
            time.sleep(5)
    return records


def state_bank_jobs(
    args: argparse.Namespace, active_tasks: list[str]
) -> list[Job]:
    jobs = []
    for task_id in active_tasks:
        output_dir = args.state_bank_root / task_id
        if state_bank_terminal(output_dir / "state_bank_manifest.json", task_id):
            continue
        jobs.append(
            Job(
                name=f"state_bank_{task_id}",
                command=[
                    str(args.python),
                    str(SCRIPT_DIR / "build_state_bank.py"),
                    "--task-id",
                    task_id,
                    "--test-h5",
                    str(selected_h5(args.selected_raw_root, task_id, "test")),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cuda",
                ],
            )
        )
    return jobs


def oracle_jobs(args: argparse.Namespace, active_tasks: list[str]) -> list[Job]:
    jobs = []
    for task_id in active_tasks:
        state_manifest = args.state_bank_root / task_id / "state_bank_manifest.json"
        for model_seed in MODEL_SEEDS:
            output_dir = args.oracle_root / task_id / f"seed_{model_seed}"
            if oracle_complete(output_dir / "summary.json", task_id, model_seed):
                continue
            jobs.append(
                Job(
                    name=f"oracle_{task_id}_seed{model_seed}",
                    command=[
                        str(args.python),
                        str(SCRIPT_DIR / "evaluate_oracle_atlas.py"),
                        "--task-id",
                        task_id,
                        "--model-seed",
                        str(model_seed),
                        "--run-dir",
                        str(args.checkpoint_root / task_id / f"seed_{model_seed}"),
                        "--train-h5",
                        str(selected_h5(args.selected_raw_root, task_id, "train")),
                        "--state-bank-manifest",
                        str(state_manifest),
                        "--output-dir",
                        str(output_dir),
                        "--device",
                        "cuda",
                    ],
                )
            )
    return jobs


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    output = args.oracle_root / "oracle_gate.json"
    subprocess.run(
        [
            str(args.python),
            str(SCRIPT_DIR / "summarize_oracle_gate.py"),
            "--baseline-gate",
            str(args.baseline_gate),
            "--state-bank-root",
            str(args.state_bank_root),
            "--oracle-root",
            str(args.oracle_root),
            "--output",
            str(output),
        ],
        check=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if os.getuid() != 2254 or os.getgid() != 2254:
        raise RuntimeError("formal oracle matrix requires uid:gid 2254:2254")
    import torch

    if torch.cuda.device_count() != args.gpu_count:
        raise RuntimeError("visible GPU count differs from the frozen matrix")
    baseline = json.loads(args.baseline_gate.read_text(encoding="utf-8"))
    if (
        baseline.get("protocol_id") != PROTOCOL_ID
        or baseline.get("continue_to_oracle_probe") is not True
    ):
        result = {
            "protocol_id": PROTOCOL_ID,
            "status": "STOPPED_BY_BASELINE_GATE",
            "continue_to_stage3": False,
            "baseline_gate": str(args.baseline_gate),
        }
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.artifact_dir / "ORACLE_MATRIX_COMPLETE.json", result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return
    positives = list(baseline["passing_positive_tasks"])
    active_tasks = [*positives, NEGATIVE_CONTROL]
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.state_bank_root.mkdir(parents=True, exist_ok=True)
    args.oracle_root.mkdir(parents=True, exist_ok=True)
    state_processes = execute(state_bank_jobs(args, active_tasks), args, "state_bank")
    state_status = {
        task_id: json.loads(
            (args.state_bank_root / task_id / "state_bank_manifest.json").read_text(
                encoding="utf-8"
            )
        )["status"]
        for task_id in active_tasks
    }
    first_real_work = args.artifact_dir / "FIRST_REAL_WORK.json"
    if not first_real_work.exists() and all(
        status in {"STATE_BANK_COMPLETE", "STATE_BANK_GATE_FAIL"}
        for status in state_status.values()
    ):
        write_json(
            first_real_work,
            {
                "protocol_id": PROTOCOL_ID,
                "status": "FIRST_REAL_WORK",
                "evidence_scope": "validated_resumed_state_bank_artifacts",
                "active_tasks": active_tasks,
                "state_bank_status": state_status,
                "validated_at_unix": time.time(),
            },
        )
    oracle_processes: list[dict[str, Any]] = []
    if all(status == "STATE_BANK_COMPLETE" for status in state_status.values()):
        oracle_processes = execute(oracle_jobs(args, active_tasks), args, "oracle_atlas")
    gate = summarize(args)
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "ORACLE_MATRIX_COMPLETE",
        "active_tasks": active_tasks,
        "model_seeds": list(MODEL_SEEDS),
        "state_bank_status": state_status,
        "state_bank_processes": state_processes,
        "oracle_processes": oracle_processes,
        "decision": gate["decision"],
        "continue_to_stage3": gate["continue_to_stage3"],
        "oracle_gate": str(args.oracle_root / "oracle_gate.json"),
        "completed_at_unix": time.time(),
    }
    write_json(args.artifact_dir / "ORACLE_MATRIX_COMPLETE.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
