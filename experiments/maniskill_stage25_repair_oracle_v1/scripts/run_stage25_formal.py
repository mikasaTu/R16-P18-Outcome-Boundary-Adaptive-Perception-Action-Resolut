#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import MODEL_SEEDS, PROTOCOL_ID, sha256_file, write_json


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]
    marker: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--screen-seed-bank", type=Path, required=True)
    parser.add_argument("--final-val-seed-bank", type=Path, required=True)
    parser.add_argument("--confirmatory-seed-bank", type=Path, required=True)
    parser.add_argument("--oracle-seed-bank", type=Path, required=True)
    parser.add_argument("--official-stack-h5", type=Path, required=True)
    parser.add_argument("--training-stack-h5", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, choices=range(2, 9), required=True)
    return parser.parse_args()


def command(*parts: object) -> tuple[str, ...]:
    return tuple(str(part) for part in parts)


def valid_marker(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    status = str(value.get("status", ""))
    return value.get("protocol_id") == PROTOCOL_ID and any(
        terminal in status for terminal in ("COMPLETE", "FROZEN", "_PASS")
    )


def run_job(job: Job, gpu: int, logs: Path) -> None:
    if valid_marker(job.marker):
        print(f"FORMAL_RESUME_SKIP gpu={gpu} job={job.name}", flush=True)
        return
    if job.marker.exists():
        raise RuntimeError(f"invalid preexisting marker for {job.name}: {job.marker}")
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{job.name}.log"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["WANDB_MODE"] = "disabled"
    environment["WANDB_DISABLED"] = "true"
    started = time.time()
    print(f"FORMAL_JOB_START gpu={gpu} job={job.name}", flush=True)
    with log_path.open("ab", buffering=0) as handle:
        process = subprocess.run(job.command, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"job failed rc={process.returncode}: {job.name}; log={log_path}")
    if not valid_marker(job.marker):
        raise RuntimeError(f"job returned without valid marker: {job.name}; marker={job.marker}")
    print(f"FORMAL_JOB_COMPLETE gpu={gpu} job={job.name} wall={time.time() - started:.1f}", flush=True)


def run_parallel(jobs: list[Job], gpu_count: int, logs: Path) -> None:
    pending: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        pending.put(job)
    failures: list[BaseException] = []
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            with lock:
                if failures:
                    return
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                run_job(job, gpu, logs)
            except BaseException as error:
                with lock:
                    failures.append(error)
            finally:
                pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in range(gpu_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise failures[0]


def run_cpu(name: str, parts: tuple[str, ...], marker: Path, logs: Path) -> None:
    run_job(Job(name, parts, marker), 0, logs)


def eval_job(
    *,
    name: str,
    script: str,
    task: str,
    seed: int,
    candidate: dict[str, Any],
    seed_bank: Path,
    output: Path,
    num_envs: int,
    mode: str = "fixed_horizon",
    trace: bool = False,
) -> Job:
    parts = [
        sys.executable,
        str(SCRIPT_DIR / script),
        "--task-id", task,
        "--model-seed", str(seed),
        "--checkpoint", candidate["checkpoint_path"],
        "--checkpoint-sha256", candidate["checkpoint_sha256"],
        "--checkpoint-step", str(candidate["step"]),
        "--seed-bank", str(seed_bank),
        "--output-dir", str(output),
        "--num-envs", str(num_envs),
        "--mode", mode,
    ]
    if trace:
        parts.append("--record-trace")
    return Job(name, tuple(parts), output / "summary.json")


def candidate_groups(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for task in ("StackCube-v1", "PushCube-v1"):
        for seed in MODEL_SEEDS:
            rows = [row for row in value["candidates"] if row["task_id"] == task and int(row["model_seed"]) == seed]
            rows.sort(key=lambda row: int(row["step"]))
            if len(rows) != 6:
                raise RuntimeError(f"candidate group {task}/{seed} has {len(rows)} rows")
            result[(task, seed)] = rows
    return result


def stage_marker(root: Path, stage: str, evidence: list[Path]) -> None:
    write_json(root / "progress" / f"{stage}.json", {
        "protocol_id": PROTOCOL_ID,
        "status": f"{stage.upper()}_COMPLETE",
        "evidence": [{"path": str(path), "sha256": sha256_file(path)} for path in evidence],
        "completed_at_unix": time.time(),
    })


def main() -> None:
    args = parse_args()
    root = args.result_root
    root.mkdir(parents=True, exist_ok=True)
    if (root / "FORMAL_COMPLETE.json").exists():
        value = json.loads((root / "FORMAL_COMPLETE.json").read_text())
        if value.get("protocol_id") == PROTOCOL_ID and value.get("run_id") == args.run_id:
            print("FORMAL_ALREADY_COMPLETE", flush=True)
            return
        raise RuntimeError("result root belongs to another or invalid completed run")
    logs = root / "logs"
    input_paths = {
        "candidate_manifest": args.candidate_manifest,
        "screen_seed_bank": args.screen_seed_bank,
        "final_val_seed_bank": args.final_val_seed_bank,
        "confirmatory_seed_bank": args.confirmatory_seed_bank,
        "oracle_seed_bank": args.oracle_seed_bank,
        "official_stack_h5": args.official_stack_h5,
        "training_stack_h5": args.training_stack_h5,
        "protocol_freeze": EXPERIMENT_ROOT / "PROTOCOL_FREEZE.json",
    }
    manifest_path = root / "FORMAL_RUN_MANIFEST.json"
    if not manifest_path.exists():
        write_json(manifest_path, {
            "protocol_id": PROTOCOL_ID,
            "status": "FORMAL_RUN_INPUTS_FROZEN",
            "run_id": args.run_id,
            "gpu_count": args.gpu_count,
            "runtime_python": sys.executable,
            "runtime_uid": os.getuid(),
            "runtime_gid": os.getgid(),
            "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in input_paths.items()},
            "orchestrator_sha256": sha256_file(Path(__file__).resolve()),
            "started_at_unix": time.time(),
        })
    candidates = candidate_groups(args.candidate_manifest)
    checkpoint = root / "checkpoint"
    screen_jobs = []
    for (task, seed), rows in candidates.items():
        for row in rows:
            output = checkpoint / "screen" / task / f"seed_{seed}" / f"step_{int(row['step']):09d}"
            screen_jobs.append(eval_job(name=f"screen-{task}-s{seed}-k{row['step']}", script="evaluate_checkpoint_grid.py", task=task, seed=seed, candidate=row, seed_bank=args.screen_seed_bank, output=output, num_envs=32))
    run_parallel(screen_jobs, args.gpu_count, logs / "checkpoint_screen")
    if not (root / "FIRST_REAL_WORK.json").exists():
        first = screen_jobs[0].marker
        write_json(root / "FIRST_REAL_WORK.json", {"protocol_id": PROTOCOL_ID, "status": "FIRST_REAL_SIMULATOR_WORK_COMPLETE", "run_id": args.run_id, "evidence": str(first), "evidence_sha256": sha256_file(first)})
    screen_selection = checkpoint / "screen_selection.json"
    run_cpu("checkpoint-screen-selection", command(sys.executable, SCRIPT_DIR / "select_checkpoint_closed_loop.py", "--stage", "screen", "--candidate-manifest", args.candidate_manifest, "--evaluation-root", checkpoint, "--output", screen_selection), screen_selection, logs / "selection")
    selected_screen = json.loads(screen_selection.read_text())
    full_jobs = []
    for name, group in selected_screen["groups"].items():
        task, seed_text = name.split("/seed_")
        seed = int(seed_text)
        for row in group["full_validation_candidates"]:
            output = checkpoint / "full_validation" / task / f"seed_{seed}" / f"step_{int(row['step']):09d}"
            full_jobs.append(eval_job(name=f"fullval-{task}-s{seed}-k{row['step']}", script="evaluate_checkpoint_grid.py", task=task, seed=seed, candidate=row, seed_bank=args.final_val_seed_bank, output=output, num_envs=20))
    run_parallel(full_jobs, args.gpu_count, logs / "checkpoint_full_validation")
    final_selection = checkpoint / "final_selection.json"
    run_cpu("checkpoint-final-selection", command(sys.executable, SCRIPT_DIR / "select_checkpoint_closed_loop.py", "--stage", "final", "--candidate-manifest", args.candidate_manifest, "--evaluation-root", checkpoint, "--screen-selection", screen_selection, "--output", final_selection), final_selection, logs / "selection")
    stage_marker(root, "checkpoint_repair", [screen_selection, final_selection])
    selected = json.loads(final_selection.read_text())
    confirm_jobs = []
    for task in ("StackCube-v1", "PushCube-v1"):
        for seed in MODEL_SEEDS:
            row = selected["groups"][f"{task}/seed_{seed}"]["selected"]
            output = root / "baseline_confirmatory" / task / f"seed_{seed}"
            confirm_jobs.append(eval_job(name=f"confirm-{task}-s{seed}", script="evaluate_checkpoint_grid.py", task=task, seed=seed, candidate=row, seed_bank=args.confirmatory_seed_bank, output=output, num_envs=20))
    run_parallel(confirm_jobs, args.gpu_count, logs / "baseline_confirmatory")
    stage_marker(root, "baseline_confirmatory", [job.marker for job in confirm_jobs])
    semantics_jobs = []
    for mode in ("fixed_horizon", "terminate_first_success", "terminate_hold5", "neutral_after_hold5"):
        for seed in MODEL_SEEDS:
            row = selected["groups"][f"StackCube-v1/seed_{seed}"]["selected"]
            output = root / "success_semantics" / mode / f"seed_{seed}"
            semantics_jobs.append(eval_job(name=f"semantics-{mode}-s{seed}", script="evaluate_success_semantics.py", task="StackCube-v1", seed=seed, candidate=row, seed_bank=args.confirmatory_seed_bank, output=output, num_envs=20, mode=mode, trace=True))
    run_parallel(semantics_jobs, args.gpu_count, logs / "success_semantics")
    stage_marker(root, "success_semantics", [job.marker for job in semantics_jobs])
    contact = root / "contact" / "contact_metric_audit.json"
    run_cpu("contact-metric-audit", command(sys.executable, SCRIPT_DIR / "audit_contact_metrics.py", "--converted-training-h5", args.training_stack_h5, "--output", contact), contact, logs / "contact")
    state_root = root / "state_banks"
    build_marker = state_root / "STATE_BANK_BUILD_COMPLETE.json"
    canonical_stack = selected["groups"]["StackCube-v1/seed_16018"]["selected"]
    run_cpu("state-bank-build", command(sys.executable, SCRIPT_DIR / "build_stackcube_state_bank.py", "--selected-checkpoints", final_selection, "--oracle-seed-bank", args.oracle_seed_bank, "--official-h5", args.official_stack_h5, "--output-root", state_root, "--num-envs", "16"), build_marker, logs / "state_bank")
    restoration = state_root / "state_restoration_audit.json"
    run_cpu("state-restoration-audit", command(sys.executable, SCRIPT_DIR / "audit_state_restoration.py", "--state-bank-root", state_root, "--output", restoration), restoration, logs / "state_bank")
    stage_marker(root, "state_banks", [build_marker, restoration, contact])
    action_root = root / "action_boundary"
    action_cal_jobs = []
    for seed in MODEL_SEEDS:
        output = action_root / "calibration" / f"seed_{seed}"
        action_cal_jobs.append(Job(f"action-cal-s{seed}", command(sys.executable, SCRIPT_DIR / "build_local_action_atlas.py", "--stage", "calibration", "--model-seed", seed, "--selected-checkpoints", final_selection, "--state-bank-manifest", state_root / "calibration" / "state_bank_manifest.json", "--training-h5", args.training_stack_h5, "--output-dir", output), output / "summary.json"))
    run_parallel(action_cal_jobs, args.gpu_count, logs / "action_calibration")
    action_freeze = action_root / "ACTION_CALIBRATION_FREEZE.json"
    run_cpu("action-calibration-freeze", command(sys.executable, SCRIPT_DIR / "freeze_action_calibration.py", "--calibration-root", action_root / "calibration", "--output", action_freeze), action_freeze, logs / "action_calibration")
    action_formal_jobs = []
    for bank in ("confirmatory", "post_success_diagnostic"):
        for seed in MODEL_SEEDS:
            output = action_root / bank / f"seed_{seed}"
            action_formal_jobs.append(Job(f"action-{bank}-s{seed}", command(sys.executable, SCRIPT_DIR / "build_local_action_atlas.py", "--stage", bank, "--model-seed", seed, "--selected-checkpoints", final_selection, "--state-bank-manifest", state_root / bank / "state_bank_manifest.json", "--training-h5", args.training_stack_h5, "--action-calibration-freeze", action_freeze, "--output-dir", output), output / "summary.json"))
    run_parallel(action_formal_jobs, args.gpu_count, logs / "action_formal")
    stage_marker(root, "action_boundary", [action_freeze] + [job.marker for job in action_formal_jobs])
    visual_root = root / "visual_resolution"
    visual_cal_jobs = []
    for seed in MODEL_SEEDS:
        output = visual_root / "calibration" / f"seed_{seed}"
        visual_cal_jobs.append(Job(f"visual-cal-s{seed}", command(sys.executable, SCRIPT_DIR / "run_visual_resolution_probe.py", "--stage", "calibration", "--model-seed", seed, "--selected-checkpoints", final_selection, "--state-bank-manifest", state_root / "calibration" / "state_bank_manifest.json", "--training-h5", args.training_stack_h5, "--native-action-jsonl", action_root / "calibration" / f"seed_{seed}" / "states.jsonl", "--action-calibration-freeze", action_freeze, "--output-dir", output), output / "summary.json"))
    run_parallel(visual_cal_jobs, args.gpu_count, logs / "visual_calibration")
    joint_freeze = root / "joint_oracle" / "ORACLE_CALIBRATION_FREEZE.json"
    run_cpu("joint-calibration-freeze", command(sys.executable, SCRIPT_DIR / "freeze_joint_calibration.py", "--visual-calibration-root", visual_root / "calibration", "--action-calibration-freeze", action_freeze, "--output", joint_freeze), joint_freeze, logs / "visual_calibration")
    visual_formal_jobs = []
    for bank in ("confirmatory", "post_success_diagnostic"):
        for seed in MODEL_SEEDS:
            output = visual_root / bank / f"seed_{seed}"
            visual_formal_jobs.append(Job(f"visual-{bank}-s{seed}", command(sys.executable, SCRIPT_DIR / "run_visual_resolution_probe.py", "--stage", bank, "--model-seed", seed, "--selected-checkpoints", final_selection, "--state-bank-manifest", state_root / bank / "state_bank_manifest.json", "--training-h5", args.training_stack_h5, "--native-action-jsonl", action_root / bank / f"seed_{seed}" / "states.jsonl", "--action-calibration-freeze", action_freeze, "--output-dir", output), output / "summary.json"))
    run_parallel(visual_formal_jobs, args.gpu_count, logs / "visual_formal")
    stage_marker(root, "visual_resolution", [joint_freeze] + [job.marker for job in visual_formal_jobs])
    joint_summary = root / "joint_oracle" / "summary.json"
    run_cpu("joint-factorial-oracle", command(sys.executable, SCRIPT_DIR / "run_joint_factorial_oracle.py", "--visual-confirmatory-root", visual_root / "confirmatory", "--joint-calibration-freeze", joint_freeze, "--output-dir", root / "joint_oracle"), joint_summary, logs / "joint")
    summary = root / "summary" / "stage25_summary.json"
    run_cpu("stage25-summarize", command(sys.executable, SCRIPT_DIR / "summarize_stage25.py", "--result-root", root, "--output", summary), summary, logs / "summary")
    audit = root / "audits" / "independent_stage25_audit.json"
    run_cpu("stage25-independent-audit", command(sys.executable, SCRIPT_DIR / "audit_stage25.py", "--result-root", root, "--summary", summary, "--output", audit), audit, logs / "summary")
    audit_value = json.loads(audit.read_text())
    if audit_value.get("status") != "INDEPENDENT_STAGE25_AUDIT_PASS":
        raise RuntimeError(f"independent scientific audit failed: {audit}")
    write_json(root / "FORMAL_COMPLETE.json", {
        "protocol_id": PROTOCOL_ID,
        "status": "ALL_PREREGISTERED_STAGE25_EXPERIMENTS_COMPLETE",
        "run_id": args.run_id,
        "final_status": json.loads(summary.read_text())["final_status"],
        "summary_path": str(summary),
        "summary_sha256": sha256_file(summary),
        "audit_path": str(audit),
        "audit_sha256": sha256_file(audit),
        "all_gates_observed_without_early_stopping": True,
        "prohibited_post_oracle_work_executed": False,
        "completed_at_unix": time.time(),
    })
    print(f"FORMAL_ALL_COMPLETE run_id={args.run_id}", flush=True)


if __name__ == "__main__":
    main()
