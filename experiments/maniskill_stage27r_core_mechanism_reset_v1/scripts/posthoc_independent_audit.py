#!/usr/bin/env python3
"""Independent raw-trace and accounting audit for Stage-2.7R.

This file intentionally does not import ``analyze_stage27r``.  It recomputes
episode outcomes from the stored traces, checks the mode-specific call
schedule, and computes its own source-episode clustered bootstrap/sign-flip
summaries.  It is an audit of the persisted oracle evidence, not a second
scientific analysis with tunable thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from collections.abc import Mapping

import h5py
import numpy as np

from common import PROTOCOL_ID, atomic_json, sha256_file

WEIGHTS = {
    "balanced": (100.0, 20.0, 5.0, -10.0, -5.0),
    "success_dominant": (120.0, 10.0, 3.0, -12.0, -6.0),
    "progress_dominant": (80.0, 35.0, 5.0, -10.0, -5.0),
}
ACCOUNTING = (
    "global_encoder_calls", "fine_encoder_calls", "policy_forward_calls",
    "policy_forward_rows", "visual_tokens", "action_opportunities",
    "executed_steps", "gpu_latency_ms", "simulator_latency_ms",
    "prefix_replay_simulator_latency_ms", "estimated_flops",
    "peak_memory_bytes", "selector_latency_ms", "episode_total_compute",
)

# This is deliberately duplicated here rather than imported from the runtime
# or preregistration.  The posthoc audit must remain independent of the
# confirmatory producer and must prove which pinned source it inspected.
PINNED_MANISKILL_COMMIT = "a4a4f9272ad64b1564035874b605ceb687b63ed8"
CONTROL_MODES = {
    "StackCube-v1": "pd_ee_delta_pos",
    "PegInsertionSide-v1": "pd_ee_delta_pose",
    "PlugCharger-v1": "pd_ee_delta_pose",
    "PullCubeTool-v1": "pd_ee_delta_pose",
    "PushT-v1": "pd_ee_delta_pose",
    "PushCube-v1": "pd_ee_delta_pos",
}
HORIZONS = {
    "StackCube-v1": 200,
    "PegInsertionSide-v1": 200,
    "PlugCharger-v1": 300,
    "PullCubeTool-v1": 200,
    "PushT-v1": 200,
    "PushCube-v1": 100,
}


def _longest(values):
    best = run = 0
    for value in values:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _first_traj_key(handle: h5py.File) -> str:
    keys = [key for key in handle.keys() if str(key).startswith("traj_")]
    if not keys:
        raise RuntimeError("replay H5 has no trajectory groups")
    return min(keys, key=lambda key: int(str(key).removeprefix("traj_")))


def replay_rgb_camera_keys(path: Path) -> dict:
    """Derive RGB camera keys and native shape from the frozen replay H5.

    This only reads the replay source.  It never looks at an oracle row's
    accounting, so an incorrect producer cannot make the audit agree with it.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        traj = _first_traj_key(handle)
        sensor_data = handle[f"{traj}/obs/sensor_data"]
        keys = []
        shapes = {}
        for key in sensor_data.keys():
            if "rgb" not in sensor_data[key]:
                continue
            dataset = sensor_data[f"{key}/rgb"]
            if len(dataset.shape) != 4 or tuple(dataset.shape[-1:]) != (3,):
                raise RuntimeError(
                    f"{path}: {key}/rgb is not [T,H,W,3]: {dataset.shape}"
                )
            keys.append(str(key))
            shapes[str(key)] = [int(value) for value in dataset.shape]
    if not keys:
        raise RuntimeError(f"{path}: no RGB camera keys in frozen replay H5")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "trajectory_key": traj,
        "rgb_camera_keys": keys,
        "rgb_camera_count": len(keys),
        "rgb_shapes": shapes,
    }


def _pinned_commit(maniskill_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(maniskill_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot read pinned ManiSkill commit: {exc}") from exc


def environment_rgb_camera_evidence(
    task: str, maniskill_root: Path, seed: int = 2704000
) -> dict:
    """Derive sensor keys and post-wrapper ``rgb.shape[1]`` from a reset.

    The official ACT wrapper is intentionally imported from the pinned source
    tree.  ``sensor_data`` is captured before flattening and the postprocess
    shape is read after the wrapper, matching the exact two independent facts
    requested by the audit protocol.
    """
    if task not in CONTROL_MODES:
        raise ValueError(f"unsupported task for camera evidence: {task}")
    maniskill_root = Path(maniskill_root)
    observed_commit = _pinned_commit(maniskill_root)
    if observed_commit != PINNED_MANISKILL_COMMIT:
        raise RuntimeError(
            f"pinned ManiSkill commit mismatch: {observed_commit} != "
            f"{PINNED_MANISKILL_COMMIT}"
        )
    # The launcher normally supplies these paths in PYTHONPATH.  Inserting the
    # pinned tree here makes the source binding explicit for direct audit use.
    act_root = maniskill_root / "examples" / "baselines" / "act"
    for path in (maniskill_root, act_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
        import train_rgbd as official_act
    except Exception as exc:  # pragma: no cover - exercised in formal image
        raise RuntimeError(f"cannot import pinned ManiSkill/ACT source: {exc}") from exc

    raw = gym.make(
        task,
        num_envs=1,
        sim_backend="physx_cpu",
        render_backend="sapien_cuda",
        reconfiguration_freq=1,
        control_mode=CONTROL_MODES[task],
        reward_mode="normalized_dense",
        obs_mode="rgb",
        render_mode="rgb_array",
        max_episode_steps=HORIZONS[task],
    )
    try:
        # Capture the deterministic reset sensor keys before the official
        # wrapper pops/flatten the sensor_data mapping.
        raw_obs, _ = raw.reset(seed=[int(seed)])
        sensor_data = raw_obs.get("sensor_data", {})
        sensor_keys = [
            str(name)
            for name, value in sensor_data.items()
            if isinstance(value, Mapping) and "rgb" in value
        ]
        sensor_shapes = {
            str(name): [int(v) for v in value["rgb"].shape]
            for name, value in sensor_data.items()
            if isinstance(value, Mapping) and "rgb" in value
        }
        if not sensor_keys:
            raise RuntimeError(f"{task}: deterministic reset returned no RGB sensors")
        wrapped = official_act.FlattenRGBDObservationWrapper(raw, depth=False)
        env = ManiSkillVectorEnv(
            wrapped,
            auto_reset=False,
            ignore_terminations=True,
            record_metrics=False,
        )
        try:
            processed, _ = env.reset(seed=[int(seed)])
            shape = [int(v) for v in processed["rgb"].shape]
        finally:
            env.close()
    finally:
        # ``env.close`` closes the wrapped raw env, but closing again is safe
        # and protects the exception path before wrapper construction.
        try:
            raw.close()
        except Exception:
            pass
    if len(shape) != 5 or shape[1] != len(sensor_keys):
        raise RuntimeError(
            f"{task}: official preprocess camera shape disagrees with reset "
            f"keys: keys={sensor_keys} shape={shape}"
        )
    return {
        "task": task,
        "pinned_maniskill_root": str(maniskill_root),
        "pinned_maniskill_commit": observed_commit,
        "official_act_preprocess": str(act_root / "train_rgbd.py"),
        "official_act_preprocess_sha256": _sha256(act_root / "train_rgbd.py"),
        "deterministic_reset_seed": int(seed),
        "env_rgb_sensor_keys": sensor_keys,
        "env_rgb_sensor_count": len(sensor_keys),
        "env_rgb_native_shapes": sensor_shapes,
        "official_preprocess_rgb_shape": shape,
        "official_preprocess_camera_count": int(shape[1]),
    }


def derive_camera_evidence(
    tasks: list[str], dataset_root: Path, maniskill_root: Path
) -> dict[str, dict]:
    evidence = {}
    for task in sorted(set(tasks)):
        candidates = sorted(
            (Path(dataset_root) / task / "oversized_source").glob(
                "trajectory.rgb.*.h5"
            )
        )
        if len(candidates) != 1:
            raise RuntimeError(
                f"{task}: expected exactly one frozen replay RGB H5, got "
                f"{[str(path) for path in candidates]}"
            )
        env = environment_rgb_camera_evidence(task, maniskill_root)
        replay = replay_rgb_camera_keys(candidates[0])
        env_keys = list(env["env_rgb_sensor_keys"])
        replay_keys = list(replay["rgb_camera_keys"])
        checks = {
            "camera_key_set_equal": set(env_keys) == set(replay_keys),
            "camera_key_order_equal": env_keys == replay_keys,
            "camera_count_equal": env["official_preprocess_camera_count"]
            == replay["rgb_camera_count"],
            "native_rgb_shape_128x128x3": all(
                tuple(shape[1:]) == (128, 128, 3)
                for shape in replay["rgb_shapes"].values()
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"{task}: environment/replay camera evidence mismatch: {checks}"
            )
        evidence[task] = {
            **env,
            "replay_h5": replay,
            "cross_validation": checks,
            "derived_camera_count": int(replay["rgb_camera_count"]),
            "posthoc_evidence_only": True,
            "preregistration_unchanged": True,
        }
    return evidence


def recompute_outcome(row):
    success = [bool(x) for x in row.get("success_trace", [])]
    reward = [float(x) for x in row.get("reward_trace", [])]
    contact = [bool(x) for x in row.get("intended_contact_trace", [])]
    grasp = [bool(x) for x in row.get("grasp_trace", [])]
    catastrophic = [bool(x) for x in row.get("catastrophic_trace", [])]
    streak = _longest(success)
    first_grasp = next((i for i, value in enumerate(grasp) if value), None)
    dropped = False
    if first_grasp is not None:
        dropped = any(not value for value in grasp[first_grasp + 1:]) and streak < 5
    return {
        "success_once": any(success),
        "success_hold5": streak >= 5,
        "success_at_end": bool(success[-1]) if success else False,
        "first_success_step": next((i + 1 for i, value in enumerate(success) if value), None),
        "longest_success_streak": streak,
        "post_success_loss": bool(any(success) and not (bool(success[-1]) if success else False)),
        "normalized_progress": (reward[-1] - reward[0]) if len(reward) > 1 else 0.0,
        "intended_contact": any(contact),
        "unintended_contact": any(catastrophic),
        "collision": any(catastrophic),
        "dropped_or_slipped": bool(dropped),
        "recoverable": bool(streak >= 5 or len(reward) < 2 or reward[-1] >= reward[0] - 0.05),
    }


def expected_schedule(row, camera_count: int | None = None):
    """Recompute calls from treatment semantics and independent camera proof.

    ``camera_count`` is required by the formal audit.  The optional default is
    retained only for small pure-function unit tests that do not contain a
    task/replay binding; production ``main`` never uses the default.
    """
    n = len(row.get("success_trace", []))
    condition = str(row["condition"])
    action_fine = condition == "CF" or condition.startswith("FF_")
    visual_fine = condition.startswith("FC_") or condition.startswith("FF_")
    treatment = min(8, n)
    treatment_queries = treatment if action_fine else ((treatment + 3) // 4)
    continuation = max(0, n - 8)
    calls = treatment_queries + continuation
    if camera_count is None:
        cameras = 1
    else:
        cameras = int(camera_count)
        if cameras < 1:
            raise ValueError(f"invalid independently derived camera count: {cameras}")
    return {
        "executed_steps": n,
        "action_opportunities": n,
        "policy_forward_calls": calls,
        "policy_forward_rows": calls,
        "global_encoder_calls": calls * cameras,
        "fine_encoder_calls": (treatment_queries if visual_fine else 0) * cameras + continuation * cameras,
        "visual_tokens": (treatment_queries * (2 if visual_fine else 1) + continuation * 2) * cameras * 16,
    }


def clustered_summary(values, clusters, seed, reps=10000):
    grouped = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[tuple(cluster)].append(float(value))
    x = np.asarray([np.mean(v) for v in grouped.values()], dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.mean(x[rng.integers(0, len(x), size=(reps, len(x)))], axis=1)
    observed = abs(float(np.mean(x)))
    flips = rng.choice((-1.0, 1.0), size=(reps, len(x)))
    p = (int(np.sum(np.abs(np.mean(flips * x, axis=1)) >= observed)) + 1) / (reps + 1)
    return {"mean": float(np.mean(x)), "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))], "signflip_p": float(p), "clusters": len(x)}


def holm(pvalues):
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    out, running = {}, 0.0
    for i, (name, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * float(pvalue)))
        out[name] = running
    return out


def utility(row, weights):
    return (weights[0] * row["success_hold5"] + weights[1] * row["normalized_progress"] + weights[2] * row["recoverable"] + weights[3] * row["dropped_or_slipped"] + weights[4] * row["collision"])


def lower_tile_tiebreak(row):
    """Match analyze_stage27r.state_table: ties select the lower tile id."""
    return (float(row["utility_value"]), -int(str(row["condition"]).split("tile")[-1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="frozen replay dataset root used for independent camera evidence",
    )
    parser.add_argument(
        "--maniskill-root",
        type=Path,
        required=True,
        help="pinned ManiSkill checkout (commit is checked independently)",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    files = sorted((args.formal_root / "oracle").glob("*.json"))
    if not files:
        raise RuntimeError(f"no oracle shards found under {args.formal_root / 'oracle'}")
    raw, outcome_mismatches, schedule_mismatches, accounting_mismatches = [], [], [], []
    for path in files:
        payload = json.loads(path.read_text())
        for row in payload.get("rows", []):
            raw.append(row)
            recomputed = recompute_outcome(row)
            for field, expected in recomputed.items():
                observed = row.get(field)
                if isinstance(expected, float):
                    equal = abs(float(observed) - expected) <= 1e-9
                else:
                    equal = observed == expected
                if not equal:
                    outcome_mismatches.append({"file": path.name, "episode_seed": row.get("episode_seed"), "condition": row.get("condition"), "field": field, "expected": expected, "observed": observed})
            accounting = row.get("accounting", {})
            if set(accounting) < set(ACCOUNTING):
                accounting_mismatches.append({"file": path.name, "missing": sorted(set(ACCOUNTING) - set(accounting))})
            expected_flops = accounting.get("global_encoder_calls", 0) * 1.8e9 + accounting.get("fine_encoder_calls", 0) * 1.8e9 + accounting.get("policy_forward_calls", 0) * 0.7e9
            if abs(float(accounting.get("estimated_flops", 0)) - expected_flops) > 1e-6:
                accounting_mismatches.append({"file": path.name, "field": "estimated_flops", "expected": expected_flops, "observed": accounting.get("estimated_flops")})
            expected_latency = accounting.get("gpu_latency_ms", 0) + accounting.get("simulator_latency_ms", 0) + accounting.get("selector_latency_ms", 0)
            if abs(float(accounting.get("episode_total_compute", 0)) - expected_latency) > 1e-6:
                accounting_mismatches.append({"file": path.name, "field": "episode_total_compute", "expected": expected_latency, "observed": accounting.get("episode_total_compute")})

    tasks = sorted({str(row["task"]) for row in raw})
    camera_evidence = derive_camera_evidence(
        tasks, args.dataset_root, args.maniskill_root
    )
    camera_checks = {
        task: {
            "derived_camera_count": int(value["derived_camera_count"]),
            "env_rgb_sensor_keys": value["env_rgb_sensor_keys"],
            "official_preprocess_rgb_shape": value["official_preprocess_rgb_shape"],
            "replay_rgb_camera_keys": value["replay_h5"]["rgb_camera_keys"],
            "cross_validation": value["cross_validation"],
            "pass": all(value["cross_validation"].values()),
        }
        for task, value in camera_evidence.items()
    }
    if not all(value["pass"] for value in camera_checks.values()):
        raise RuntimeError(f"independent camera evidence failed: {camera_checks}")
    for row in raw:
        # Re-run schedule checks with the camera count derived from the pinned
        # environment/H5 pair.  The earlier field checks intentionally remain
        # visible in the mismatch list rather than being silently replaced.
        task = str(row["task"])
        schedule = expected_schedule(
            row, camera_evidence[task]["derived_camera_count"]
        )
        accounting = row.get("accounting", {})
        for field, expected in schedule.items():
            if int(accounting.get(field, -1)) != int(expected):
                mismatch = {
                    "file": "camera_derived_schedule",
                    "episode_seed": row.get("episode_seed"),
                    "condition": row.get("condition"),
                    "field": field,
                    "expected": expected,
                    "observed": accounting.get(field),
                    "derived_camera_count": camera_evidence[task]["derived_camera_count"],
                }
                if mismatch not in schedule_mismatches:
                    schedule_mismatches.append(mismatch)

    # Aggregate repeats independently, then compute matched effects per source episode.
    grouped = defaultdict(list)
    for row in raw:
        key = (row["task"], int(row["model_seed"]), row["bank"], row["bank_id"], row["source_episode"], row["phase"], row["condition"])
        grouped[key].append(row)
    means = {}
    for key, rows in grouped.items():
        means[key] = {"task": key[0], "seed": key[1], "bank": key[2], "bank_id": key[3], "source_episode": key[4], "phase": key[5], "condition": key[6], **{field: float(np.mean([float(r[field]) for r in rows])) for field in ("success_hold5", "normalized_progress", "recoverable", "dropped_or_slipped", "collision")}}
        means[key]["utility"] = {name: float(np.mean([utility(r, w) for r in rows])) for name, w in WEIGHTS.items()}
        means[key]["accounting"] = {field: float(np.mean([r["accounting"][field] for r in rows])) for field in ACCOUNTING}
    effects = defaultdict(list)
    for key in sorted({k[:6] for k in means}):
        cond = {k[6]: v for k, v in means.items() if k[:6] == key}
        for weight in WEIGHTS:
            fc_candidates = []
            ff_candidates = []
            for candidate in (cond[k] for k in cond if k.startswith("FC_tile")):
                candidate = dict(candidate, utility_value=candidate["utility"][weight])
                fc_candidates.append(candidate)
            for candidate in (cond[k] for k in cond if k.startswith("FF_tile")):
                candidate = dict(candidate, utility_value=candidate["utility"][weight])
                ff_candidates.append(candidate)
            fc = max(fc_candidates, key=lower_tile_tiebreak)
            ff = max(ff_candidates, key=lower_tile_tiebreak)
            cc, cf = cond["CC"], cond["CF"]
            cluster = (key[0], key[4])
            effects[(weight, key[0], "visual")].append((fc["utility"][weight] - cc["utility"][weight], cluster))
            effects[(weight, key[0], "action")].append((cf["utility"][weight] - cc["utility"][weight], cluster))
            effects[(weight, key[0], "joint")].append((ff["utility"][weight] - max(fc["utility"][weight], cf["utility"][weight]), cluster))
    summaries, pvalues = {}, {}
    for index, (key, pairs) in enumerate(sorted(effects.items())):
        values, clusters = zip(*pairs)
        summary = clustered_summary(values, clusters, 730000 + index, args.bootstrap_replicates)
        name = "/".join(map(str, key)); summaries[name] = summary; pvalues[name] = summary["signflip_p"]
    adjusted = holm(pvalues)

    # Prefix replay latency is persisted but not included in episode_total_compute.
    prefix_total = float(sum(float(r.get("accounting", {}).get("prefix_replay_simulator_latency_ms", 0.0)) for r in raw))
    episode_total = float(sum(float(r.get("accounting", {}).get("episode_total_compute", 0.0)) for r in raw))
    result = {
        "protocol_id": PROTOCOL_ID,
        "independence": {"does_not_import_analyze_stage27r": True, "script_sha256": sha256_file(Path(__file__))},
        "raw_files": [{"path": str(p.relative_to(args.formal_root)), "sha256": sha256_file(p), "bytes": p.stat().st_size} for p in files],
        "raw_row_count": len(raw),
        "camera_evidence": {
            "status": "PASS" if all(value["pass"] for value in camera_checks.values()) else "FAIL",
            "source": "pinned_ManiSkill_reset_plus_official_ACT_preprocess_plus_frozen_replay_H5",
            "posthoc_evidence_only": True,
            "does_not_modify_preregistration": True,
            "by_task": camera_evidence,
        },
        "camera_checks": camera_checks,
        "outcome_recompute": {"pass": not outcome_mismatches, "mismatches": outcome_mismatches[:100], "mismatch_count": len(outcome_mismatches)},
        "schedule_recompute": {"pass": not schedule_mismatches, "mismatches": schedule_mismatches[:100], "mismatch_count": len(schedule_mismatches), "schedule_definition": "trace length, treatment 8-step query cadence, common fine continuation"},
        "accounting_recompute": {"pass": not accounting_mismatches, "mismatches": accounting_mismatches[:100], "mismatch_count": len(accounting_mismatches), "flop_formula": "global*1.8e9 + fine*1.8e9 + policy*0.7e9"},
        "prefix_latency_disclosure": {"prefix_replay_simulator_latency_ms_sum": prefix_total, "episode_total_compute_ms_sum": episode_total, "prefix_included_in_episode_total": False, "interpretation": "prefix replay cost is persisted separately and omitted from episode_total_compute; all arms share the prefix, but reported total compute is deployment-only treatment/continuation cost"},
        "paired_effects_independent": {"bootstrap_replicates": args.bootstrap_replicates, "summaries": summaries, "holm_adjusted_signflip_p": adjusted, "unit": "source_episode cluster"},
        "status": "PASS" if camera_checks and all(value["pass"] for value in camera_checks.values()) and not outcome_mismatches and not schedule_mismatches and not accounting_mismatches else "FAIL_WITH_DISCLOSED_MISMATCHES",
    }
    atomic_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("raw_row_count", "outcome_recompute", "schedule_recompute", "accounting_recompute", "prefix_latency_disclosure", "status")}, indent=2))


if __name__ == "__main__":
    main()
