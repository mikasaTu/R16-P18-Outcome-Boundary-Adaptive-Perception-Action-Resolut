from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import h5py
import mujoco
import numpy as np
import robosuite
import scipy
import torch
import torchvision
import yaml

from boundarybc.checkpoint import sha256_file
from boundarybc.config import ExperimentConfig
from boundarybc.data import official_demo_path
from boundarybc.libero_runtime import task_bddl_path, task_init_states_path


def verify_locked_inputs(
    config: ExperimentConfig,
    *,
    dataset_root: str | Path,
) -> dict[str, Any]:
    """Hash and version-check every mutable external input before training."""

    expected_runtime = config.raw["runtime"]
    actual_runtime: dict[str, Any] = {
        "python": _python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cudnn": torch.backends.cudnn.version(),
        "numpy": str(np.__version__),
        "h5py": str(h5py.__version__),
        "mujoco": str(mujoco.__version__),
        "robosuite": str(robosuite.__version__),
        "scipy": str(scipy.__version__),
        "pyyaml": str(yaml.__version__),
    }
    if actual_runtime != expected_runtime:
        raise RuntimeError(
            f"runtime version lock mismatch: expected={expected_runtime}, actual={actual_runtime}"
        )

    task_inputs: dict[str, Any] = {}
    suite = str(config.raw["benchmark"]["suite"])
    for task in config.tasks:
        raw_task = config.raw["tasks"][task.key]
        demo = official_demo_path(dataset_root, suite, task.name).resolve()
        bddl = task_bddl_path(config, task).resolve()
        init_states = task_init_states_path(config, task).resolve()
        demo_record = _verify_file(
            demo,
            expected_bytes=int(raw_task["demo_file_bytes"]),
            expected_sha256=str(raw_task["demo_file_sha256"]),
        )
        bddl_record = _verify_file(bddl, expected_sha256=str(raw_task["bddl_sha256"]))
        init_record = _verify_file(
            init_states,
            expected_sha256=str(raw_task["init_states_sha256"]),
        )
        task_inputs[task.key] = {
            "demo": demo_record,
            "bddl": bddl_record,
            "init_states": init_record,
        }

    asset_expected = config.raw["assets"]
    asset_root = Path(asset_expected["root"]).resolve()
    asset_actual = hash_asset_tree(asset_root)
    for key in ("files", "bytes", "tree_manifest_sha256"):
        if asset_actual[key] != asset_expected[key]:
            raise RuntimeError(
                f"asset tree lock mismatch for {key}: "
                f"expected={asset_expected[key]!r}, actual={asset_actual[key]!r}"
            )
    return {
        "runtime": actual_runtime,
        "tasks": task_inputs,
        "assets": {"root": str(asset_root), **asset_actual},
    }


def hash_asset_tree(root: str | Path) -> dict[str, int | str]:
    """Match `find | sort | sha256sum | sha256sum` with absolute filenames."""

    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and not path.is_symlink()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    manifest_digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        size = path.stat().st_size
        total_bytes += size
        line = f"{sha256_file(path)}  {path}\n"
        manifest_digest.update(line.encode("utf-8"))
    return {
        "files": len(files),
        "bytes": total_bytes,
        "tree_manifest_sha256": manifest_digest.hexdigest(),
    }


def _verify_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None = None,
) -> dict[str, int | str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"file size lock mismatch: {path}; expected={expected_bytes}, actual={path.stat().st_size}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"file hash lock mismatch: {path}; expected={expected_sha256}, actual={actual_sha256}"
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual_sha256,
    }


def _python_version() -> str:
    import platform

    return platform.python_version()
