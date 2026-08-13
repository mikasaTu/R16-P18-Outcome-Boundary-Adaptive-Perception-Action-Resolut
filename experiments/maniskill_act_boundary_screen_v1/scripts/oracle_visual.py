#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


INTERVENTIONS = (
    "local_low_resolution",
    "local_blur",
    "conditional_mean",
)


def _cast_image(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.uint8:
        return value.round().clamp(0, 255).to(dtype)
    return value.to(dtype)


def _gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coordinate = torch.arange(size, device=device, dtype=torch.float32)
    coordinate -= (size - 1) / 2
    one_dimensional = torch.exp(-(coordinate**2) / (2 * sigma**2))
    one_dimensional /= one_dimensional.sum()
    return one_dimensional[:, None] * one_dimensional[None, :]


def gaussian_blur(images: torch.Tensor, kernel_size: int = 9, sigma: float = 3.0) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"images must be [N,C,H,W], got {images.shape}")
    source_dtype = images.dtype
    work = images.to(torch.float32)
    kernel = _gaussian_kernel(kernel_size, sigma, images.device)
    weight = kernel[None, None].repeat(work.shape[1], 1, 1, 1)
    padding = kernel_size // 2
    work = F.pad(work, (padding, padding, padding, padding), mode="reflect")
    blurred = F.conv2d(work, weight, groups=work.shape[1])
    return _cast_image(blurred, source_dtype)


def conditional_ring_mean(
    image: torch.Tensor, y0: int, y1: int, x0: int, x1: int
) -> torch.Tensor:
    """Per-camera, per-channel mean of available one-pixel outside-ring pixels."""

    if image.ndim != 4:
        raise ValueError(f"image must be [camera,C,H,W], got {image.shape}")
    _, _, height, width = image.shape
    pieces: list[torch.Tensor] = []
    if y0 > 0:
        pieces.append(image[:, :, y0 - 1, x0:x1].reshape(image.shape[0], image.shape[1], -1))
    if y1 < height:
        pieces.append(image[:, :, y1, x0:x1].reshape(image.shape[0], image.shape[1], -1))
    if x0 > 0:
        pieces.append(image[:, :, y0:y1, x0 - 1].reshape(image.shape[0], image.shape[1], -1))
    if x1 < width:
        pieces.append(image[:, :, y0:y1, x1].reshape(image.shape[0], image.shape[1], -1))
    if not pieces:  # pragma: no cover - impossible for a nonempty four-tile grid
        raise RuntimeError("tile has no outside ring")
    ring = torch.cat(pieces, dim=-1).to(torch.float32)
    return ring.mean(dim=-1, keepdim=True).unsqueeze(-1)


def visual_intervention_batch(
    observation: Mapping[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Create the frozen 16-tile x 3-intervention observation batch."""

    rgb = observation["rgb"]
    if rgb.ndim != 5 or rgb.shape[0] != 1:
        raise ValueError(f"expected one RGB observation [1,camera,C,H,W], got {rgb.shape}")
    _, cameras, channels, height, width = rgb.shape
    if height % 4 or width % 4:
        raise ValueError("RGB height and width must be divisible by four")
    tile_height = height // 4
    tile_width = width // 4
    source = rgb[0]
    blurred = gaussian_blur(source)
    images: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    for tile_index in range(16):
        tile_row, tile_col = divmod(tile_index, 4)
        y0, y1 = tile_row * tile_height, (tile_row + 1) * tile_height
        x0, x1 = tile_col * tile_width, (tile_col + 1) * tile_width
        for intervention_index, intervention in enumerate(INTERVENTIONS):
            value = source.clone()
            if intervention == "local_low_resolution":
                tile = source[:, :, y0:y1, x0:x1].to(torch.float32)
                down = F.interpolate(tile, size=(2, 2), mode="bilinear", align_corners=False)
                up = F.interpolate(
                    down,
                    size=(tile_height, tile_width),
                    mode="bilinear",
                    align_corners=False,
                )
                value[:, :, y0:y1, x0:x1] = _cast_image(up, source.dtype)
            elif intervention == "local_blur":
                value[:, :, y0:y1, x0:x1] = blurred[:, :, y0:y1, x0:x1]
            elif intervention == "conditional_mean":
                mean = conditional_ring_mean(source, y0, y1, x0, x1)
                fill = mean.expand(cameras, channels, tile_height, tile_width)
                value[:, :, y0:y1, x0:x1] = _cast_image(fill, source.dtype)
            else:  # pragma: no cover
                raise KeyError(intervention)
            images.append(value)
            metadata.append(
                {
                    "pair_index": len(metadata),
                    "tile_index": tile_index,
                    "tile_row": tile_row,
                    "tile_col": tile_col,
                    "intervention": intervention,
                    "intervention_index": intervention_index,
                    "pixel_bounds_yx": [y0, y1, x0, x1],
                }
            )
    batch = {
        key: (torch.cat([value] * 48, dim=0) if key != "rgb" else torch.stack(images))
        for key, value in observation.items()
    }
    return batch, metadata


def strongest_pair_per_tile(
    metadata: Sequence[Mapping[str, Any]], action_l2_changes: Sequence[float]
) -> list[int]:
    if len(metadata) != 48 or len(action_l2_changes) != 48:
        raise ValueError("visual atlas must contain exactly 48 pairs")
    selected: list[int] = []
    for tile_index in range(16):
        indices = [
            index
            for index, row in enumerate(metadata)
            if int(row["tile_index"]) == tile_index
        ]
        # Earlier intervention order wins an exact tie.
        selected.append(max(indices, key=lambda index: (float(action_l2_changes[index]), -index)))
    return selected


def select_joint_visual_pairs(
    metadata: Sequence[Mapping[str, Any]], action_l2_changes: Sequence[float]
) -> list[int]:
    per_tile = strongest_pair_per_tile(metadata, action_l2_changes)
    return sorted(
        per_tile,
        key=lambda index: (
            -float(action_l2_changes[index]),
            int(metadata[index]["tile_index"]),
        ),
    )[:4]
