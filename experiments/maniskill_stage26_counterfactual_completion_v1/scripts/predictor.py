from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class FeatureShape:
    visual: int
    proprio: int
    action: int
    predicted: int
    consistency: int

    @property
    def flat(self) -> int:
        return 4 * (self.visual + self.proprio + self.action) + self.predicted + self.consistency


def flatten_feature(feature: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([feature["visual"].reshape(-1), feature["proprio"].reshape(-1), feature["actions"].reshape(-1), feature["predicted"].reshape(-1), feature["consistency"].reshape(-1)]).astype(np.float32)


class CompletionModel(nn.Module):
    def __init__(self, architecture: str, shape: FeatureShape) -> None:
        super().__init__(); self.architecture = architecture; self.shape = shape
        if architecture == "linear_probe":
            self.body = nn.Identity(); width = shape.flat
        elif architecture == "two_layer_mlp":
            self.body = nn.Sequential(nn.Linear(shape.flat, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()); width = 128
        elif architecture == "one_layer_small_gru":
            self.gru = nn.GRU(shape.visual + shape.proprio + shape.action, 128, batch_first=True)
            self.tail = nn.Sequential(nn.Linear(128 + shape.predicted + shape.consistency, 128), nn.ReLU()); width = 128
        else: raise ValueError(architecture)
        self.heads = nn.Linear(width, 3)

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        if self.architecture != "one_layer_small_gru":
            return self.heads(self.body(flat))
        offset = 0; batch = flat.shape[0]
        v = flat[:, offset:offset + 4*self.shape.visual].reshape(batch, 4, self.shape.visual); offset += 4*self.shape.visual
        p = flat[:, offset:offset + 4*self.shape.proprio].reshape(batch, 4, self.shape.proprio); offset += 4*self.shape.proprio
        a = flat[:, offset:offset + 4*self.shape.action].reshape(batch, 4, self.shape.action); offset += 4*self.shape.action
        pred = flat[:, offset:offset + self.shape.predicted]; offset += self.shape.predicted
        consistency = flat[:, offset:offset + self.shape.consistency]
        _, hidden = self.gru(torch.cat([v, p, a], dim=-1))
        return self.heads(self.tail(torch.cat([hidden[-1], pred, consistency], dim=-1)))


def shape_from_feature(feature: dict[str, np.ndarray]) -> FeatureShape:
    return FeatureShape(feature["visual"].shape[-1], feature["proprio"].shape[-1], feature["actions"].shape[-1], feature["predicted"].size, feature["consistency"].size)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool); positives = int(labels.sum())
    if positives == 0: return 0.0
    order = np.argsort(-scores, kind="stable"); ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    for lower in np.linspace(0, 1, bins, endpoint=False):
        upper = lower + 1 / bins; mask = (probabilities >= lower) & (probabilities < upper if upper < 1 else probabilities <= upper)
        if mask.any(): result += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
    return result
