"""Narrow Diffusers EMA compatibility shim for a non-DeepSpeed workload.

Diffusers 0.26.3 probes ``transformers.deepspeed`` on every EMA step.  Newer
Transformers releases no longer expose that attribute.  The formal protocol
does not use DeepSpeed, so this subclass preserves Diffusers' update schedule,
state dictionary, and copy semantics while omitting only that unavailable
zero-3 probe.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
from diffusers.training_utils import EMAModel


class NonDeepSpeedEMAModel(EMAModel):
    """EMAModel.step with the upstream non-ZeRO update path made explicit."""

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter] | torch.nn.Module) -> None:
        if isinstance(parameters, torch.nn.Module):
            parameters = parameters.parameters()
        materialized = list(parameters)
        if len(materialized) != len(self.shadow_params):
            raise RuntimeError(
                "EMA parameter inventory changed: "
                f"observed={len(materialized)}, expected={len(self.shadow_params)}"
            )

        self.optimization_step += 1
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay
        for shadow, parameter in zip(self.shadow_params, materialized, strict=True):
            if parameter.requires_grad:
                shadow.sub_(one_minus_decay * (shadow - parameter))
            else:
                shadow.copy_(parameter)
