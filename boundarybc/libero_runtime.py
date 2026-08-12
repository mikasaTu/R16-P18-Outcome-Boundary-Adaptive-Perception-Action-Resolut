from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from boundarybc.config import ExperimentConfig, TaskConfig
from boundarybc.data import ProprioNormalizer


def configure_headless_runtime(project_root: str | Path, *, gpu_device_id: int = 0) -> None:
    project_root = Path(project_root).resolve()
    vendor = project_root / "configs" / "egl" / "10_nvidia.json"
    libero_config = project_root / "configs" / "libero"
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(vendor))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(gpu_device_id))
    os.environ.setdefault("EGL_DEVICE_ID", str(gpu_device_id))
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(libero_config))


def task_bddl_path(config: ExperimentConfig, task: TaskConfig) -> Path:
    project_root = config.path.parents[1]
    return (
        project_root
        / "libero"
        / "libero"
        / "bddl_files"
        / config.raw["benchmark"]["suite"]
        / f"{task.name}.bddl"
    )


def task_init_states_path(config: ExperimentConfig, task: TaskConfig) -> Path:
    project_root = config.path.parents[1]
    return (
        project_root
        / "libero"
        / "libero"
        / "init_files"
        / config.raw["benchmark"]["suite"]
        / f"{task.name}.pruned_init"
    )


def make_offscreen_env(config: ExperimentConfig, task: TaskConfig, *, gpu_device_id: int = 0) -> Any:
    from libero.libero.envs import OffScreenRenderEnv

    image_height, image_width = config.raw["benchmark"]["image_size"]
    return OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_path(config, task)),
        camera_heights=int(image_height),
        camera_widths=int(image_width),
        camera_names=["agentview", "robot0_eye_in_hand"],
        horizon=int(config.raw["benchmark"]["task_horizon"]) + int(config.raw["benchmark"]["wait_steps"]),
        render_gpu_device_id=gpu_device_id,
    )


def load_task_init_states(config: ExperimentConfig, task: TaskConfig) -> np.ndarray:
    states = torch.load(task_init_states_path(config, task), map_location="cpu", weights_only=False)
    states = np.asarray(states, dtype=np.float64)
    if len(states) != 50:
        raise ValueError(f"expected 50 official init states for {task.key}, got {len(states)}")
    return states


def runtime_proprio(observation: dict[str, np.ndarray]) -> np.ndarray:
    from robosuite.utils import transform_utils as transform

    quaternion = np.asarray(observation["robot0_eef_quat"], dtype=np.float64)
    axis_angle = np.asarray(transform.quat2axisangle(quaternion), dtype=np.float32)
    value = np.concatenate(
        (
            np.asarray(observation["robot0_joint_pos"], dtype=np.float32),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
            axis_angle,
        )
    )
    if value.shape != (15,):
        raise ValueError(f"unexpected runtime proprio shape: {value.shape}")
    return value


def observation_tensors(
    observation: dict[str, np.ndarray],
    normalizer: ProprioNormalizer,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = np.asarray(observation["agentview_image"], dtype=np.uint8)
    if image.shape != (128, 128, 3):
        raise ValueError(f"unexpected agentview image shape: {image.shape}")
    image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device=device, dtype=torch.float32).div_(255.0).sub_(0.5).div_(0.5)
    proprio = normalizer.normalize_numpy(runtime_proprio(observation)).astype(np.float32, copy=False)
    proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).to(device=device)
    return image_tensor, proprio_tensor


def rewrite_demo_model_xml(xml_string: str) -> str:
    from libero.libero import get_libero_path
    from libero.libero.utils.utils import postprocess_model_xml

    rewritten = postprocess_model_xml(xml_string, {})
    root = ET.fromstring(rewritten)
    asset_root = Path(get_libero_path("assets")).resolve()
    asset_element = root.find("asset")
    if asset_element is None:
        raise ValueError("demo model XML is missing its asset section")
    elements = list(asset_element.findall("mesh")) + list(asset_element.findall("texture"))
    unresolved: list[str] = []
    for element in elements:
        old_path = element.get("file")
        if not old_path or Path(old_path).exists():
            continue
        marker = "/assets/"
        if marker not in old_path:
            unresolved.append(old_path)
            continue
        candidate = asset_root / old_path.split(marker, 1)[1]
        if not candidate.is_file():
            unresolved.append(old_path)
            continue
        element.set("file", str(candidate))
    if unresolved:
        raise FileNotFoundError(f"unresolved demo XML assets: {unresolved[:3]}")
    return ET.tostring(root, encoding="unicode")


def deterministic_short_replay_check(
    config: ExperimentConfig,
    task: TaskConfig,
    demo_path: str | Path,
    *,
    demo_key: str = "demo_0",
    state_index: int = 30,
    horizon: int = 4,
    gpu_device_id: int = 0,
) -> dict[str, float | int | str]:
    with h5py.File(demo_path, "r") as handle:
        group = handle[f"data/{demo_key}"]
        model_xml = rewrite_demo_model_xml(group.attrs["model_file"])
        states = np.asarray(group["states"], dtype=np.float64)
        actions = np.asarray(group["actions"], dtype=np.float64)
        if state_index < 1:
            raise ValueError("LIBERO states[t] aligns to observations[t-1], so state_index must be positive")
        expected_proprio = np.concatenate(
            (
                np.asarray(group["obs/joint_states"][state_index - 1], dtype=np.float32),
                np.asarray(group["obs/gripper_states"][state_index - 1], dtype=np.float32),
                np.asarray(group["obs/ee_pos"][state_index - 1], dtype=np.float32),
                np.asarray(group["obs/ee_ori"][state_index - 1], dtype=np.float32),
            )
        )
        expected_image = np.asarray(
            group["obs/agentview_rgb"][state_index - 1],
            dtype=np.uint8,
        )
    if state_index + horizon > len(actions):
        raise ValueError("short replay exceeds demonstration length")
    final_states: list[np.ndarray] = []
    final_images: list[np.ndarray] = []
    alignment_proprio_error: float | None = None
    alignment_image_mae: float | None = None
    # A flattened LIBERO MuJoCo state does not include every warm-start and
    # controller cache. Reusing one Python env therefore is not a valid replay
    # isolation test. Each repetition starts from a freshly constructed env,
    # then loads the exact demonstration XML and flattened state.
    for _ in range(2):
        env = make_offscreen_env(config, task, gpu_device_id=gpu_device_id)
        try:
            env.reset()
            env.reset_from_xml_string(model_xml)
            observation = env.set_init_state(states[state_index])
            if alignment_proprio_error is None:
                alignment_proprio_error = float(
                    np.max(np.abs(runtime_proprio(observation) - expected_proprio))
                )
                restored_image = np.asarray(observation["agentview_image"], dtype=np.uint8)
                alignment_image_mae = float(
                    np.mean(
                        np.abs(
                            restored_image.astype(np.int16) - expected_image.astype(np.int16)
                        )
                    )
                )
            env.robots[0].controller.reset_goal()
            env.env.timestep = 0
            env.env.done = False
            for action in actions[state_index : state_index + horizon]:
                observation, _, _, _ = env.step(action)
            final_states.append(env.get_sim_state().copy())
            final_images.append(np.asarray(observation["agentview_image"]).copy())
        finally:
            env.close()
    state_error = float(np.max(np.abs(final_states[0] - final_states[1])))
    image_error = int(
        np.max(
            np.abs(
                final_images[0].astype(np.int16) - final_images[1].astype(np.int16)
            )
        )
    )
    image_mae = float(
        np.mean(
            np.abs(
                final_images[0].astype(np.int16) - final_images[1].astype(np.int16)
            )
        )
    )
    return {
        "task_key": task.key,
        "demo_key": demo_key,
        "state_index": state_index,
        "horizon": horizon,
        "isolation": "fresh_env_per_replay",
        "state_observation_alignment": "states[t]_to_obs[t-1]",
        "alignment_proprio_max_abs": float(alignment_proprio_error),
        "alignment_image_mae": float(alignment_image_mae),
        "state_repeat_max_abs": state_error,
        "image_repeat_max_abs": image_error,
        "image_repeat_mae": image_mae,
    }
