#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from typing import Any, Mapping

import h5py
import numpy as np
import torch

from common import canonical_json


def h5_timestep(value: h5py.Group | h5py.Dataset, timestep: int) -> Any:
    if isinstance(value, h5py.Dataset):
        return np.asarray(value[timestep])
    return {key: h5_timestep(value[key], timestep) for key in value.keys()}


def h5_full(value: h5py.Group | h5py.Dataset) -> Any:
    if isinstance(value, h5py.Dataset):
        return np.asarray(value[()])
    return {key: h5_full(value[key]) for key in value.keys()}


def state_index(value: Any, index: int) -> Any:
    if isinstance(value, Mapping):
        return {key: state_index(child, index) for key, child in value.items()}
    if isinstance(value, torch.Tensor):
        return value[index].detach().cpu().numpy().copy()
    return np.asarray(value)[index].copy()


def hash_state(value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()

    def visit(child: Any, path: str) -> None:
        if isinstance(child, Mapping):
            for key in sorted(child):
                visit(child[key], f"{path}/{key}")
            return
        array = np.ascontiguousarray(np.asarray(child))
        digest.update(path.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(canonical_json(list(array.shape)))
        digest.update(array.tobytes())

    visit(value, "")
    return digest.hexdigest()


def hash_rgb(observation: Mapping[str, torch.Tensor], index: int) -> str:
    rgb = observation["rgb"][index].detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(rgb.dtype).encode("ascii"))
    digest.update(canonical_json(list(rgb.shape)))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def write_nested(group: h5py.Group, value: Mapping[str, Any]) -> None:
    for key, child in value.items():
        if isinstance(child, Mapping):
            write_nested(group.create_group(key), child)
        else:
            group.create_dataset(key, data=np.asarray(child), compression="gzip")


def stack_predicates(base: Any) -> dict[str, np.ndarray]:
    cube_a = base.cubeA.pose.p
    cube_b = base.cubeB.pose.p
    tcp = base.agent.tcp.pose.p
    tcp_distance = torch.linalg.norm(tcp - cube_a, dim=1)
    xy_distance = torch.linalg.norm(cube_a[:, :2] - cube_b[:, :2], dim=1)
    target_z = cube_b[:, 2] + base.cube_half_size[2] * 2
    vertical_error = torch.abs(cube_a[:, 2] - target_z)
    lifted = cube_a[:, 2] > cube_b[:, 2] + base.cube_half_size[2] * 1.5
    geometric_grasp = lifted & (tcp_distance < 0.09)
    near_placement = (xy_distance < 0.05) & (vertical_error < 0.06)
    official_success = base.evaluate()["success"].to(torch.bool)
    progress_distance = torch.sqrt(xy_distance**2 + vertical_error**2)
    progress = 1.0 - torch.tanh(5.0 * progress_distance)
    return {
        "tcp_cube_distance_m": tcp_distance.detach().cpu().numpy(),
        "cube_xy_distance_m": xy_distance.detach().cpu().numpy(),
        "vertical_error_m": vertical_error.detach().cpu().numpy(),
        "geometric_grasp": geometric_grasp.detach().cpu().numpy(),
        "near_placement": near_placement.detach().cpu().numpy(),
        "official_success": official_success.detach().cpu().numpy(),
        "normalized_progress": progress.detach().cpu().numpy(),
    }


def stack_phase(predicates: Mapping[str, np.ndarray], index: int) -> str:
    if bool(predicates["official_success"][index]) or bool(
        predicates["near_placement"][index]
    ):
        return "placement_contact_near_completion"
    if bool(predicates["geometric_grasp"][index]):
        return "object_in_hand_pre_placement"
    if float(predicates["tcp_cube_distance_m"][index]) <= 0.10:
        return "pre_grasp_or_pre_contact"
    return "free_space_approach"


def public_predicates(predicates: Mapping[str, np.ndarray], index: int) -> dict[str, Any]:
    return {
        key: (
            bool(np.asarray(value)[index])
            if np.asarray(value).dtype == np.bool_
            else float(np.asarray(value)[index])
        )
        for key, value in predicates.items()
    }
