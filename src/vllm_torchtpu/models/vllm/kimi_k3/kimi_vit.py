# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPU adaptations for the upstream Kimi-K3 MoonViT implementation."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from vllm.model_executor.models.kimi_k25_vit import MoonViT3dPretrainedModel

from vllm_torchtpu.utils import synchronize_tensors


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
    synchronize_tensors(materialized_qkv)
    return materialized_qkv, output[1]


class KimiK3MoonViT3dPretrainedModel(MoonViT3dPretrainedModel):
    """Upstream K3 MoonViT with TPU-safe compilation boundaries."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for block in self.encoder.blocks:
            block.wqkv.register_forward_hook(_materialize_vision_qkv)
