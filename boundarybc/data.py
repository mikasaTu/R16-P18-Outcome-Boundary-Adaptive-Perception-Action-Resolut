from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class EpisodeRecord:
    demo_key: str
    identity: str
    length: int


@dataclasses.dataclass
class TaskArrays:
    images: np.ndarray
    proprio: np.ndarray
    action_chunks: np.ndarray
    action_mask: np.ndarray
    episode_ids: np.ndarray
    timesteps: np.ndarray

    def __len__(self) -> int:
        return int(self.images.shape[0])


@dataclasses.dataclass(frozen=True)
class ProprioNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def normalize_numpy(self, value: np.ndarray) -> np.ndarray:
        return (value - self.mean) / self.std

    def as_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, value: dict[str, list[float]]) -> "ProprioNormalizer":
        return cls(
            mean=np.asarray(value["mean"], dtype=np.float32),
            std=np.asarray(value["std"], dtype=np.float32),
        )


def official_demo_path(dataset_root: str | Path, suite: str, task_name: str) -> Path:
    return Path(dataset_root) / suite / f"{task_name}_demo.hdf5"


def discover_episodes(hdf5_path: str | Path) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    with h5py.File(hdf5_path, "r") as handle:
        data = handle["data"]
        for demo_key in data.keys():
            group = data[demo_key]
            initial_state = np.asarray(group.attrs["init_state"], dtype=np.float64)
            identity = hashlib.sha256(initial_state.tobytes(order="C")).hexdigest()
            records.append(
                EpisodeRecord(
                    demo_key=demo_key,
                    identity=identity,
                    length=int(group["actions"].shape[0]),
                )
            )
    records.sort(key=lambda item: item.identity)
    if len(records) != 50:
        raise ValueError(f"expected exactly 50 official demos, found {len(records)}")
    if len({record.identity for record in records}) != len(records):
        raise ValueError("duplicate initial-state episode identities")
    return records


def split_episode_records(
    records: Iterable[EpisodeRecord],
) -> dict[str, tuple[EpisodeRecord, ...]]:
    ordered = tuple(records)
    if len(ordered) != 50:
        raise ValueError("the locked LIBERO pilot split requires 50 episodes")
    return {
        "train": ordered[:40],
        "validation": ordered[40:45],
        "test": ordered[45:50],
    }


def _build_action_chunks(actions: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    chunks = np.zeros((len(actions), horizon, actions.shape[-1]), dtype=np.float32)
    masks = np.zeros((len(actions), horizon), dtype=np.float32)
    for t in range(len(actions)):
        count = min(horizon, len(actions) - t)
        chunks[t, :count] = actions[t : t + count]
        masks[t, :count] = 1.0
        if count < horizon:
            chunks[t, count:] = actions[-1]
    return chunks, masks


def load_task_arrays(
    hdf5_path: str | Path,
    records: Iterable[EpisodeRecord],
    *,
    action_horizon: int = 8,
) -> TaskArrays:
    image_parts: list[np.ndarray] = []
    proprio_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    timestep_parts: list[np.ndarray] = []
    records = tuple(records)
    with h5py.File(hdf5_path, "r") as handle:
        data = handle["data"]
        for episode_index, record in enumerate(records):
            group = data[record.demo_key]
            images = np.asarray(group["obs/agentview_rgb"], dtype=np.uint8)
            joints = np.asarray(group["obs/joint_states"], dtype=np.float32)
            gripper = np.asarray(group["obs/gripper_states"], dtype=np.float32)
            ee_pos = np.asarray(group["obs/ee_pos"], dtype=np.float32)
            ee_ori = np.asarray(group["obs/ee_ori"], dtype=np.float32)
            actions = np.asarray(group["actions"], dtype=np.float32)
            if not (len(images) == len(joints) == len(gripper) == len(ee_pos) == len(ee_ori) == len(actions)):
                raise ValueError(f"unaligned arrays in {record.demo_key}")
            proprio = np.concatenate((joints, gripper, ee_pos, ee_ori), axis=-1)
            if proprio.shape[1] != 15 or actions.shape[1] != 7:
                raise ValueError(
                    f"unexpected dimensions in {record.demo_key}: proprio={proprio.shape}, actions={actions.shape}"
                )
            chunks, masks = _build_action_chunks(actions, action_horizon)
            image_parts.append(images)
            proprio_parts.append(proprio)
            action_parts.append(chunks)
            mask_parts.append(masks)
            episode_parts.append(np.full(len(actions), episode_index, dtype=np.int32))
            timestep_parts.append(np.arange(len(actions), dtype=np.int32))
    return TaskArrays(
        images=np.concatenate(image_parts, axis=0),
        proprio=np.concatenate(proprio_parts, axis=0),
        action_chunks=np.concatenate(action_parts, axis=0),
        action_mask=np.concatenate(mask_parts, axis=0),
        episode_ids=np.concatenate(episode_parts, axis=0),
        timesteps=np.concatenate(timestep_parts, axis=0),
    )


def fit_proprio_normalizer(proprio: np.ndarray) -> ProprioNormalizer:
    mean = proprio.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = proprio.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(1e-4))
    return ProprioNormalizer(mean=mean, std=std)


def make_batch(
    arrays: TaskArrays,
    indices: torch.Tensor,
    normalizer: ProprioNormalizer,
    *,
    device: torch.device,
    augment: bool,
) -> dict[str, torch.Tensor]:
    cpu_indices = indices.detach().cpu().numpy()
    images = torch.from_numpy(arrays.images[cpu_indices]).to(device=device, non_blocking=True)
    images = images.permute(0, 3, 1, 2).to(dtype=torch.float32).div_(255.0)
    images = images.sub_(0.5).div_(0.5)
    if augment:
        images = random_shift(images, pad=4)
        brightness = 1.0 + 0.10 * (2.0 * torch.rand((len(images), 1, 1, 1), device=device) - 1.0)
        images = (images * brightness).clamp_(-1.0, 1.0)
    proprio = normalizer.normalize_numpy(arrays.proprio[cpu_indices]).astype(np.float32, copy=False)
    return {
        "images": images,
        "proprio": torch.from_numpy(proprio).to(device=device, non_blocking=True),
        "actions": torch.from_numpy(arrays.action_chunks[cpu_indices]).to(device=device, non_blocking=True),
        "mask": torch.from_numpy(arrays.action_mask[cpu_indices]).to(device=device, non_blocking=True),
    }


def random_shift(images: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return images
    batch, channels, height, width = images.shape
    padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
    max_offset = 2 * pad + 1
    offsets_y = torch.randint(max_offset, (batch,), device=images.device)
    offsets_x = torch.randint(max_offset, (batch,), device=images.device)
    result = torch.empty_like(images)
    for index in range(batch):
        y = int(offsets_y[index])
        x = int(offsets_x[index])
        result[index] = padded[index, :, y : y + height, x : x + width]
    return result

