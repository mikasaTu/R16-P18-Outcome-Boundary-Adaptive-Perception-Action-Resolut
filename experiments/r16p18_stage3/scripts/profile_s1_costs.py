#!/usr/bin/env python3
"""Profile Stage-3 S1 costs from cached tensors only.

This module intentionally has no simulator dependency.  A successful run
loads a previously serialized observation tensor and a callable Torch module,
then measures batch-size-one forward passes.  Missing inputs are fatal.  The
module does not create an environment or obtain observations itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
from types import SimpleNamespace
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:  # Keep --help and static audits usable without the training environment.
    import torch
except ImportError:  # pragma: no cover - exercised only on minimal machines.
    torch = None  # type: ignore[assignment]


PROTOCOL_ID = "R16-P18-STAGE3-S1-BUDGET-FEASIBILITY-V1"
ACTION_WINDOW = 8
ACTION_INTERVALS = {"coarse": 4, "fine": 1}
VISUAL_MODES = ("coarse", "fine_grid2", "fine_grid4")
REFERENCE = {
    "flops_numerator": 92438200000000.0,
    "flops_denominator": 125120200000000.0,
    "flops_ratio": 0.738795174560143,
    "wall_clock_ratio": 0.7502597918865904,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Write one result and refuse to overwrite an existing result."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"fail-on-overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for profiling; no simulator fallback is permitted"
        )
    return torch


def _map_device(value: Any, device: Any) -> Any:
    """Move nested cached tensors while preserving the observation structure."""
    t = _require_torch()
    if t.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return type(value)((key, _map_device(item, device)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_map_device(item, device) for item in value)
    if isinstance(value, list):
        return [_map_device(item, device) for item in value]
    return value


def _find_batch_size(value: Any) -> int | None:
    t = _require_torch()
    if t.is_tensor(value):
        return int(value.shape[0]) if value.ndim else None
    if isinstance(value, Mapping):
        for item in value.values():
            found = _find_batch_size(item)
            if found is not None:
                return found
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            found = _find_batch_size(item)
            if found is not None:
                return found
    return None


def _load_cached_observation(path: Path) -> Any:
    """Load a pre-existing tensor file; never synthesize an observation."""
    t = _require_torch()
    if not path.is_file():
        raise FileNotFoundError(
            f"cached observation tensor is required and was not found: {path}"
        )
    try:
        value = t.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Older PyTorch has no weights_only keyword.
        value = t.load(path, map_location="cpu")
    if isinstance(value, Mapping) and "observation" in value:
        if value.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("cached observation protocol_id mismatch")
        value = value["observation"]
    batch = _find_batch_size(value)
    if batch != 1:
        raise ValueError(
            f"cached observation must be batch-size one; detected batch={batch}: {path}"
        )
    return value


def _load_callable(
    path: Path,
    device: Any,
    checkpoint_format: str,
    task: str | None,
    model_seed: int | None,
    observation: Any,
) -> tuple[Any, dict[str, Any]]:
    """Load TorchScript or a serialized ``nn.Module`` without reconstructing it."""
    t = _require_torch()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint/module was not found: {path}")
    if checkpoint_format == "stage27r":
        if task is None or model_seed is None:
            raise ValueError("stage27r format requires --task and --model-seed")
        if not isinstance(observation, Mapping) or not all(
            key in observation for key in ("state", "rgb")
        ):
            raise ValueError("stage27r observation requires state and rgb tensors")
        payload = t.load(path, map_location=device, weights_only=False)
        if payload.get("protocol_id") != "R16-P18-MS6-STAGE27R-CORE-MECHANISM-RESET-V1":
            raise RuntimeError("Stage-2.7R checkpoint protocol mismatch")
        config = payload.get("train_config", {})
        if config.get("task_id") != task or int(config.get("seed", -1)) != model_seed:
            raise RuntimeError("Stage-2.7R checkpoint task/seed binding mismatch")
        import train_rgbd as official
        from multires_policy import MultiResolutionAgent

        task_table = {
            "StackCube-v1": ("pd_ee_delta_pos", 200),
            "PegInsertionSide-v1": ("pd_ee_delta_pose", 200),
        }
        if task not in task_table:
            raise ValueError(f"unsupported S1 task: {task}")
        control_mode, horizon = task_table[task]
        args = official.Args(
            seed=model_seed,
            env_id=task,
            include_depth=False,
            backbone="resnet18",
            lr_backbone=1e-5,
            num_queries=8,
            control_mode=control_mode,
            max_episode_steps=horizon,
            temporal_agg=False,
            sim_backend="physx_cpu",
            num_eval_envs=1,
            capture_video=False,
        )
        official.args = args
        state_dim = int(observation["state"].shape[-1])
        action_dim = int(payload["ema_model"]["model.action_head.weight"].shape[0])
        dummy = SimpleNamespace(
            single_observation_space={"state": SimpleNamespace(shape=(state_dim,))},
            single_action_space=SimpleNamespace(shape=(action_dim,)),
        )
        model = MultiResolutionAgent(dummy, args).to(device)
        model.load_state_dict(payload["ema_model"])
        model.eval()
        return model, dict(payload)

    model: Any = None
    payload: Any = None
    try:
        model = t.jit.load(str(path), map_location=device)
        payload = {"loader": "torch.jit.load"}
    except (RuntimeError, ValueError, TypeError):
        try:
            payload = t.load(path, map_location=device, weights_only=False)
        except TypeError:
            payload = t.load(path, map_location=device)
        if isinstance(payload, t.nn.Module):
            model = payload
        elif isinstance(payload, Mapping):
            for key in ("model", "module", "policy"):
                candidate = payload.get(key)
                if isinstance(candidate, t.nn.Module):
                    model = candidate
                    break
        if model is None:
            raise TypeError(
                "checkpoint contains no callable serialized module; S1 does not "
                "reconstruct models from weights or run training"
            )
    model = model.to(device)
    model.eval()
    metadata = payload if isinstance(payload, Mapping) else {}
    return model, dict(metadata)


def _flops_value(value: Any, axis: str, resolution: str) -> float | None:
    """Resolve per-forward FLOPs from explicit checkpoint/config metadata."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping):
        for key in (
            f"{axis}.{resolution}",
            resolution,
            "per_forward",
            "default",
        ):
            if key in value and isinstance(value[key], (int, float)):
                return float(value[key])
        nested = value.get(axis)
        if isinstance(nested, Mapping):
            return _flops_value(nested.get(resolution), axis, resolution)
    return None


def _load_flops_config(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"FLOP metadata file was not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("FLOP metadata must be a JSON object")
    return value


def _sync(device: Any) -> None:
    t = _require_torch()
    if getattr(device, "type", "cpu") == "cuda":
        t.cuda.synchronize(device)


def _forward(model: Any, observation: Any, visual_mode: str, action_mode: str) -> Any:
    """Call a native resolution-aware module, retaining compatibility adapters."""
    # The first form is the S1 contract.  The fallbacks support serialized
    # modules that expose positional or visual-only APIs without changing their
    # weights.  Query interval is represented by repeated calls in the caller.
    native_visual_mode = "fine" if visual_mode.startswith("fine_grid") else visual_mode
    tile_grid = int(visual_mode.removeprefix("fine_grid")) if visual_mode.startswith("fine_grid") else 4
    if hasattr(model, "get_action"):
        return model.get_action(
            observation,
            visual_mode=native_visual_mode,
            tile_id=0,
            tile_grid=tile_grid,
        )
    try:
        return model(
            observation,
            visual_mode=native_visual_mode,
            action_mode=action_mode,
            query_interval=ACTION_INTERVALS[action_mode],
        )
    except TypeError as first_error:
        try:
            return model(observation, visual_mode=native_visual_mode)
        except TypeError:
            try:
                return model(observation, native_visual_mode)
            except TypeError:
                raise first_error


def _measure_condition(
    model: Any,
    observation: Any,
    device: Any,
    axis: str,
    resolution: str,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    t = _require_torch()
    if axis == "visual":
        visual_mode = resolution
        action_mode = "coarse"
    elif axis == "action":
        visual_mode = "coarse"
        action_mode = resolution
    else:
        raise ValueError(f"unknown profiling axis: {axis}")
    query_count = (ACTION_WINDOW + ACTION_INTERVALS[action_mode] - 1) // ACTION_INTERVALS[action_mode]
    moved = _map_device(observation, device)
    # Warmup is never included in samples.  Input is the same frozen cached
    # tensor for every condition; only the native mode arguments differ.
    with t.inference_mode():
        for _ in range(warmup):
            for _ in range(query_count):
                _forward(model, moved, visual_mode, action_mode)
        _sync(device)
        samples: list[float] = []
        for repeat in range(repeats):
            random.seed(seed + repeat)
            t.manual_seed(seed + repeat)
            if getattr(device, "type", "cpu") == "cuda":
                t.cuda.manual_seed_all(seed + repeat)
            _sync(device)
            started = time.perf_counter()
            for _ in range(query_count):
                _forward(model, moved, visual_mode, action_mode)
            _sync(device)
            samples.append((time.perf_counter() - started) * 1000.0)
    median_ms = statistics.median(samples)
    stdev_ms = statistics.stdev(samples) if len(samples) > 1 else 0.0
    total_flops = _measure_flops(
        model, moved, visual_mode, action_mode, query_count
    )
    return {
        "axis": axis,
        "resolution": resolution,
        "visual_mode": visual_mode,
        "action_mode": action_mode,
        "query_interval": ACTION_INTERVALS[action_mode],
        "action_window": ACTION_WINDOW,
        "query_count": query_count,
        "native_supported": True,
        "wall_clock_ms_samples": samples,
        "wall_clock_ms_median": median_ms,
        "wall_clock_ms_stdev": stdev_ms,
        "wall_clock_ms_p95": _percentile(samples, 0.95),
        "flops_per_forward": total_flops / query_count,
        "flops_per_state": total_flops,
        "flops_source": "torch.utils.flop_counter.FlopCounterMode",
    }


def _measure_flops(
    model: Any,
    observation: Any,
    visual_mode: str,
    action_mode: str,
    query_count: int,
) -> float:
    """Measure operator FLOPs on the exact forward graph, fail-closed."""
    t = _require_torch()
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as exc:  # pragma: no cover - pinned torch provides it.
        raise RuntimeError("PyTorch FlopCounterMode is required") from exc
    # PyTorch 2.5's ModuleTracker (used by FlopCounterMode) registers graph
    # hooks and is incompatible with tensors created inside inference_mode.
    # The model is in eval mode and this call performs no optimizer/backward;
    # leave autograd enabled solely so the official counter can trace every
    # native operator without changing the forward graph or weights.
    with FlopCounterMode(display=False) as counter:
        for _ in range(query_count):
            _forward(model, observation, visual_mode, action_mode)
    value = float(counter.get_total_flops())
    if not value > 0:
        raise RuntimeError("FLOP counter returned a non-positive value")
    return value


def _schedule_reproduction(
    statistics_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reweight fresh single-forwards by immutable Stage-2.7R query counts."""
    payload = json.loads(statistics_path.read_text(encoding="utf-8"))
    records = payload["aggregated_state_treatments"]
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for record in records:
        if record.get("bank") != "confirmatory":
            continue
        key = (record["task"], record["model_seed"], record["bank_id"])
        grouped.setdefault(key, []).append(record)
    coarse_queries = fine_continuation_queries = full_queries = 0.0
    state_count = 0
    for state_rows in grouped.values():
        cc = next(record for record in state_rows if record["condition"] == "CC")
        ff = max(
            (record for record in state_rows if record["condition"].startswith("FF_tile")),
            key=lambda record: (
                record["utility"]["balanced"],
                -int(record["condition"].split("tile")[-1]),
            ),
        )
        cc_accounting = cc["accounting"]
        ff_accounting = ff["accounting"]
        cameras = 2.0
        continuation = float(cc_accounting["fine_encoder_calls"]) / cameras
        coarse = (
            float(cc_accounting["global_encoder_calls"])
            - float(cc_accounting["fine_encoder_calls"])
        ) / cameras
        full = float(ff_accounting["global_encoder_calls"]) / cameras
        coarse_queries += coarse
        fine_continuation_queries += continuation
        full_queries += full
        state_count += 1
    by_key = {(row["axis"], row["resolution"]): row for row in rows}
    coarse_row = by_key[("visual", "coarse")]
    # The formal S1 grid names the two native crop settings explicitly.  Keep
    # accepting the legacy generic name so this pure accounting helper can be
    # exercised with archived/minimal fixtures that predate that split.
    fine_row = by_key.get(("visual", "fine_grid4"), by_key.get(("visual", "fine")))
    if fine_row is None:
        raise KeyError("visual fine_grid4 (or legacy fine) profiling row is required")
    cq = float(coarse_row["query_count"])
    fq = float(fine_row["query_count"])
    coarse_ms = float(coarse_row["wall_clock_ms_median"]) / cq
    fine_ms = float(fine_row["wall_clock_ms_median"]) / fq
    coarse_flops = float(coarse_row["flops_per_state"]) / cq
    fine_flops = float(fine_row["flops_per_state"]) / fq
    wall_num = coarse_queries * coarse_ms + fine_continuation_queries * fine_ms
    wall_den = full_queries * fine_ms
    flop_num = coarse_queries * coarse_flops + fine_continuation_queries * fine_flops
    flop_den = full_queries * fine_flops
    coarse_samples = [float(value) / cq for value in coarse_row["wall_clock_ms_samples"]]
    fine_samples = [float(value) / fq for value in fine_row["wall_clock_ms_samples"]]
    paired_ratios = [
        (coarse_queries * coarse_sample + fine_continuation_queries * fine_sample)
        / (full_queries * fine_sample)
        for coarse_sample, fine_sample in zip(coarse_samples, fine_samples)
    ]
    return {
        "source": "fresh_forwards_reweighted_by_immutable_stage27r_query_counts",
        "statistics_path": str(statistics_path),
        "statistics_sha256": sha256_file(statistics_path),
        "state_count": state_count,
        "coarse_query_count": coarse_queries,
        "fine_continuation_query_count": fine_continuation_queries,
        "full_query_count": full_queries,
        "measured_wall_clock_ratio": wall_num / wall_den,
        "measured_flops_ratio": flop_num / flop_den,
        "measurement_variance": {
            "wall_clock_ratio": statistics.variance(paired_ratios)
            if len(paired_ratios) > 1
            else None,
            "flops_ratio": None,
        },
    }


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of empty samples")
    ordered = sorted(float(v) for v in values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _device_metadata(device: Any) -> dict[str, Any]:
    t = _require_torch()
    value: dict[str, Any] = {
        "requested": str(device),
        "type": str(getattr(device, "type", "unknown")),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": getattr(t, "__version__", "unknown"),
        "cuda_available": bool(t.cuda.is_available()),
        "cuda_version": getattr(getattr(t, "version", None), "cuda", None),
    }
    if getattr(device, "type", "cpu") == "cuda" and t.cuda.is_available():
        value["device_name"] = t.cuda.get_device_name(device)
        value["device_capability"] = list(t.cuda.get_device_capability(device))
    return value


def profile(args: argparse.Namespace) -> dict[str, Any]:
    t = _require_torch()
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"fail-on-overwrite: {output}")
    checkpoint = Path(args.checkpoint)
    observation_path = Path(args.observations)
    observation = _load_cached_observation(observation_path)
    device = t.device(args.device)
    if device.type == "cuda" and not t.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    model, checkpoint_metadata = _load_callable(
        checkpoint,
        device,
        args.checkpoint_format,
        args.task,
        args.model_seed,
        observation,
    )
    rows = []
    # Visual costs use the native coarse action-query schedule.  Action costs
    # use the native coarse visual path, so each axis is isolated.
    for axis, resolutions in (("visual", VISUAL_MODES), ("action", ("coarse", "fine"))):
        for resolution in resolutions:
            rows.append(
                _measure_condition(
                    model=model,
                    observation=observation,
                    device=device,
                    axis=axis,
                    resolution=resolution,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    seed=args.seed,
                )
            )
    reproduction = _schedule_reproduction(Path(args.stage27_statistics), rows)
    output_value = {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "measurement": {
            "batch_size": 1,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "cuda_synchronize": True,
            "action_window": ACTION_WINDOW,
            "device": _device_metadata(device),
            "no_environment_operations": True,
        },
        "inputs": {
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
            },
            "cached_observations": {
                "path": str(observation_path),
                "sha256": sha256_file(observation_path),
            },
            "stage27_statistics": {
                "path": str(args.stage27_statistics),
                "sha256": sha256_file(Path(args.stage27_statistics)),
            },
        },
        "native_support": {
            "visual": {"coarse": True, "fine_grid2": True, "fine_grid4": True},
            "action": {"coarse": True, "fine": True},
            "action_semantics": "fixed 8-query output chunk; native query interval is 4 versus 1",
        },
        "reference_stage27r": REFERENCE,
        "resolution_pairs": {
            "visual": [["coarse", "fine_grid2"], ["coarse", "fine_grid4"]],
            "action": [["coarse", "fine"]]
        },
        "reproduction": reproduction,
        "samples": rows,
        "status_note": "These are new forward measurements only; no rollout or budget conclusion is produced here.",
    }
    atomic_json(output, output_value)
    return output_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-format", choices=("stage27r", "serialized_module"), default="stage27r")
    parser.add_argument("--task", choices=("StackCube-v1", "PegInsertionSide-v1"))
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--stage27-statistics", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2718001)
    args = parser.parse_args()
    profile(args)


if __name__ == "__main__":
    main()
