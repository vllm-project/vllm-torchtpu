# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPU adaptations for the upstream Kimi-K3 MoonViT implementation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_tpu._internal import sync

from vllm.model_executor.models.kimi_k25_vit import (
    Learnable2DInterpPosEmbDivided_fixed, MoonViT3dPretrainedModel)


class KimiK3Learnable2DInterpPosEmb(Learnable2DInterpPosEmbDivided_fixed):
    """Run K3 positional interpolation without the CUDA-oriented compiler."""

    def forward(
        self,
        x: torch.Tensor,
        grid_thws: torch.Tensor | list[list[int]],
    ) -> torch.Tensor:
        pos_embs = []
        grid_thw_list = (grid_thws if isinstance(grid_thws, list) else
                         grid_thws.tolist())
        for t, h, w in grid_thw_list:
            if t > self.num_frames:
                raise ValueError(f"t={t} exceeds num_frames={self.num_frames}")
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = F.interpolate(
                    self.weight.permute(2, 0, 1).unsqueeze(0),
                    size=(h, w),
                    mode=self.interpolation_mode,
                ).squeeze(0).permute(1, 2, 0).flatten(end_dim=1)

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = (pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) +
                              self.time_weight[:t])
            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        return x + torch.cat(pos_embs)


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
        config = self.config
        self.patch_embed.pos_emb = KimiK3Learnable2DInterpPosEmb(
            height=config.init_pos_emb_height,
            width=config.init_pos_emb_width,
            num_frames=config.init_pos_emb_time,
            dim=config.hidden_size,
            interpolation_mode=getattr(config, "pos_emb_interpolation_mode",
                                       "bicubic"),
        )
        for block in self.encoder.blocks:
            block.wqkv.register_forward_hook(_materialize_vision_qkv)
