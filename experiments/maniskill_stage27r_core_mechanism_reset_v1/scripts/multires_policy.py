"""One ACT policy whose shared ResNet18 supports coarse and local-fine vision."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

try:
    import train_rgbd as official_act
    from act.detr.backbone import build_backbone
    from act.detr.detr_vae import DETRVAE, build_encoder
    from act.detr.transformer import build_transformer
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("pinned ManiSkill ACT directory must be on PYTHONPATH") from exc


@dataclass(frozen=True)
class ResolutionSpec:
    visual: str = "coarse"
    action: str = "coarse"
    tile_id: int = 0
    tile_grid: int = 2


def crop_tile(images: torch.Tensor, tile_id: int, grid: int) -> torch.Tensor:
    if grid not in (2, 4):
        raise ValueError("tile grid must be 2 or 4")
    if not 0 <= tile_id < grid * grid:
        raise ValueError("tile id outside grid")
    h, w = images.shape[-2:]
    row, col = divmod(tile_id, grid)
    y0, y1 = (row * h) // grid, ((row + 1) * h) // grid
    x0, x1 = (col * w) // grid, ((col + 1) * w) // grid
    return images[..., y0:y1, x0:x1]


class Native128Dataset(official_act.SmallDemoDataset_ACTPolicy):
    """Official loader with original 128px RGB retained instead of 224px."""

    def process_obs(self, obs_dict):
        from mani_skill.utils import common

        sensor_data = obs_dict.pop("sensor_data")
        del obs_dict["sensor_param"]
        images = []
        for cam_data in sensor_data.values():
            rgb = torch.from_numpy(cam_data["rgb"]).permute(0, 3, 1, 2)
            images.append(F.interpolate(rgb.float(), size=(128, 128), mode="bilinear", align_corners=False).round().to(torch.uint8))
        rgb = torch.stack(images, dim=1)
        obs_dict["extra"] = {k: v[:, None] if len(v.shape) == 1 else v for k, v in obs_dict["extra"].items()}
        state = common.flatten_state_dict(obs_dict, use_torch=True)
        return {"state": state, "rgb": rgb}


class MultiResolutionDETRVAE(DETRVAE):
    """DETRVAE with global tokens plus optional crop tokens from one backbone."""

    def forward(self, obs, actions=None):
        mode = str(obs.get("_visual_mode", "coarse"))
        tile_id = int(obs.get("_tile_id", 0))
        tile_grid = int(obs.get("_tile_grid", 2))
        state = obs["state"]
        bs = state.shape[0]
        is_training = actions is not None
        if is_training:
            cls = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
            state_embed = self.encoder_state_proj(state).unsqueeze(1)
            action_embed = self.encoder_action_proj(actions)
            encoder_input = torch.cat([cls, state_embed, action_embed], dim=1).permute(1, 0, 2)
            is_pad = torch.zeros((bs, encoder_input.shape[0]), dtype=torch.bool, device=state.device)
            pos_embed = self.pos_table.detach().permute(1, 0, 2)
            encoded = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)[0]
            latent_info = self.latent_proj(encoded)
            mu, logvar = latent_info[:, : self.latent_dim], latent_info[:, self.latent_dim :]
            std = (logvar / 2).exp()
            latent = mu + torch.randn_like(std) * std
            latent_input = self.latent_out_proj(latent)
        else:
            mu = logvar = None
            latent_input = self.latent_out_proj(torch.zeros(bs, self.latent_dim, device=state.device))

        images = obs["rgb"]
        all_features, all_pos = [], []
        for camera in range(images.shape[1]):
            native = images[:, camera]
            global_image = F.interpolate(native, size=(112, 112), mode="bilinear", align_corners=False)
            features, pos = self.backbones[0](global_image)
            all_features.append(self.input_proj(features[0]))
            all_pos.append(pos[0])
            if mode == "fine":
                crop = crop_tile(native, tile_id, tile_grid)
                crop = F.interpolate(crop, size=(112, 112), mode="bilinear", align_corners=False)
                fine_features, fine_pos = self.backbones[0](crop)
                all_features.append(self.input_proj(fine_features[0]))
                all_pos.append(fine_pos[0])
        src = torch.cat(all_features, dim=3)
        pos = torch.cat(all_pos, dim=3)
        proprio = self.input_proj_robot_state(state)
        hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio, self.additional_pos_embed.weight)[0]
        return self.action_head(hs), [mu, logvar]


class MultiResolutionAgent(nn.Module):
    def __init__(self, env: Any, args: Any):
        super().__init__()
        self.state_dim = env.single_observation_space["state"].shape[0]
        self.act_dim = env.single_action_space.shape[0]
        self.kl_weight = args.kl_weight
        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        backbone = build_backbone(args)
        self.model = MultiResolutionDETRVAE(
            [backbone], build_transformer(args), build_encoder(args), self.state_dim, self.act_dim, args.num_queries
        )
        self.mode_probabilities = (0.25, 0.25, 0.25, 0.25)
        self.consistency_weight = 0.1

    def _normalized(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = dict(obs)
        result["rgb"] = self.normalize(obs["rgb"].float() / 255.0)
        return result

    @staticmethod
    def _kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean()

    def compute_loss(self, obs, action_seq):
        data = self._normalized(obs)
        mode_index = int(torch.randint(0, 4, (), device=action_seq.device).item())
        data["_visual_mode"] = "fine" if mode_index in (1, 3) else "coarse"
        data["_tile_id"] = int(torch.randint(0, 4, ()).item())
        data["_tile_grid"] = 2
        predicted, (mu, logvar) = self.model(data, action_seq)
        l1 = F.l1_loss(predicted, action_seq)
        kl = self._kl(mu, logvar)
        consistency = torch.zeros((), device=action_seq.device)
        # Free-space is not privileged in the model. A deterministic 1/4 batch
        # surrogate prevents unconditional coarse collapse without phase input.
        if mode_index in (1, 3) and action_seq.shape[0] >= 4:
            coarse = dict(data, _visual_mode="coarse")
            with torch.no_grad():
                coarse_pred, _ = self.model(coarse, None)
            consistency = F.smooth_l1_loss(predicted[: action_seq.shape[0] // 4], coarse_pred[: action_seq.shape[0] // 4])
        loss = l1 + self.kl_weight * kl + self.consistency_weight * consistency
        return {"loss": loss, "l1": l1, "kl": kl, "consistency": consistency}

    def get_action(self, obs, visual_mode="coarse", tile_id=0, tile_grid=2):
        data = self._normalized(obs)
        data.update(_visual_mode=visual_mode, _tile_id=tile_id, _tile_grid=tile_grid)
        return self.model(data, None)[0]

    def get_action_with_accounting(self, obs, visual_mode="coarse", tile_id=0, tile_grid=2):
        action = self.get_action(obs, visual_mode, tile_id, tile_grid)
        cameras = int(obs["rgb"].shape[1])
        feature_tokens_per_call = 16  # ResNet18 at 112px: 4x4
        return action, {
            "global_encoder_calls": cameras,
            "fine_encoder_calls": cameras if visual_mode == "fine" else 0,
            "visual_tokens": cameras * feature_tokens_per_call * (2 if visual_mode == "fine" else 1),
            "policy_forward_calls": 1,
            "policy_forward_rows": int(obs["rgb"].shape[0]),
        }


def install_into_official_trainer(num_queries: int = 8) -> None:
    official_act.Agent = MultiResolutionAgent
    official_act.SmallDemoDataset_ACTPolicy = Native128Dataset
