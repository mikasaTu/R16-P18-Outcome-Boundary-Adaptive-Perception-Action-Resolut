from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class BoundaryBCS(nn.Module):
    """BoundaryBC-S health-gate policy with a fixed 16-token visual budget."""

    def __init__(
        self,
        *,
        proprio_dim: int = 15,
        hidden_dim: int = 128,
        action_dim: int = 7,
        action_horizon: int = 8,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
    ) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        self.visual_stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )
        self.visual_projection = nn.Conv2d(256, hidden_dim, kernel_size=1)
        self.visual_position = nn.Parameter(torch.zeros(1, 16, hidden_dim))
        self.proprio_position = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.aggregator = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, action_horizon * action_dim),
        )
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.visual_position, std=0.02)
        nn.init.trunc_normal_(self.proprio_position, std=0.02)

    def encode_microtokens(self, images: torch.Tensor) -> torch.Tensor:
        features = self.visual_projection(self.visual_stem(images))
        if features.shape[-2:] != (8, 8):
            raise RuntimeError(f"expected an 8x8 layer3 feature map, got {features.shape[-2:]}")
        return features.flatten(2).transpose(1, 2)

    @staticmethod
    def uniform_tokens(microtokens: torch.Tensor) -> torch.Tensor:
        batch, count, channels = microtokens.shape
        if count != 64:
            raise ValueError(f"expected 64 microtokens, got {count}")
        grid = microtokens.reshape(batch, 8, 8, channels)
        pooled = grid.reshape(batch, 4, 2, 4, 2, channels).mean(dim=(2, 4))
        return pooled.reshape(batch, 16, channels)

    def forward(self, images: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        microtokens = self.encode_microtokens(images)
        visual_tokens = self.uniform_tokens(microtokens) + self.visual_position
        proprio_token = self.proprio_mlp(proprio).unsqueeze(1) + self.proprio_position
        tokens = torch.cat((proprio_token, visual_tokens), dim=1)
        encoded = self.aggregator(tokens)
        action = torch.tanh(self.action_head(encoded[:, 0]))
        return action.reshape(-1, self.action_horizon, self.action_dim)


def masked_action_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    squared = (prediction - target).square().mean(dim=-1)
    denominator = mask.sum().clamp_min(1.0)
    return (squared * mask).sum() / denominator

