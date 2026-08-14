# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPU adaptations for the upstream Kimi-K3 MoonViT implementation."""

from __future__ import annotations

import torch
from torch import nn
from torch_tpu._internal import sync
from vllm.model_executor.models.kimi_k25_vit import MoonViT3dPretrainedModel


def _materialize_vision_qkv(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: tuple[torch.Tensor, torch.Tensor | None],
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """Bound the lazy graph before K3 reshapes and rotates projected QKV."""
    del module, inputs
    qkv = output[0]
    if qkv.device.type != "tpu":
        return None

    materialized_qkv = torch.empty_like(qkv).copy_(qkv)
    sync.synchronize(materialized_qkv, wait=True)
    return materialized_qkv, output[1]


class KimiK3MoonViT3dPretrainedModel(MoonViT3dPretrainedModel):
    """Upstream K3 MoonViT with TPU-safe compilation boundaries."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        for block in self.encoder.blocks:
            block.wqkv.register_forward_hook(_materialize_vision_qkv)
