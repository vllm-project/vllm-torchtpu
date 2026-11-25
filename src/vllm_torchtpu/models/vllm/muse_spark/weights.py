# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU reference decoding of rank-local Muse NVFP4 expert weights."""

import torch

from vllm_torchtpu.layers.core.quantization import unpack_uint8_to_fp4


def dequantize_nvfp4(
    packed: torch.Tensor, block_scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    """Linear-layout E2M1 * block-16 E4M3 * global scale, in FP32."""
    if packed.dtype != torch.uint8:
        raise ValueError("NVFP4 weights must contain packed uint8 bytes")
    if packed.shape[-1] != block_scale.shape[-1] * 8:
        raise ValueError("NVFP4 block scales must cover 16 unpacked values")
    if packed.shape[:-1] != block_scale.shape[:-1]:
        raise ValueError("NVFP4 scale and weight row shapes disagree")
    values = unpack_uint8_to_fp4(packed)
    scales = block_scale.float() * global_scale.float()
    return (values.reshape(*block_scale.shape, 16) * scales.unsqueeze(-1)).flatten(-2)


def selected_packed_expert_outputs(
    x: torch.Tensor,
    ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_global: torch.Tensor | None = None,
    w2_global: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference decoder for selected experts; TPU serving uses tiled GMMs."""
    tokens, top_k = ids.shape
    flat_ids = ids.flatten()

    def selected(weight, scale, global_scale):
        scale = scale.index_select(0, flat_ids)
        if global_scale is not None:
            scale = scale.view(torch.float8_e4m3fn).float()
            global_scale = global_scale.index_select(0, flat_ids).reshape(
                flat_ids.numel(), -1
            )
            global_scale = global_scale.repeat_interleave(
                scale.shape[1] // global_scale.shape[1], dim=-1
            )
            scale = scale * global_scale[:, :, None]
        return (
            dequantize_nvfp4(
                weight.index_select(0, flat_ids), scale, scale.new_ones(())
            )
            .to(x.dtype)
            .transpose(1, 2)
        )

    inputs = x[:, None, :].expand(-1, top_k, -1).reshape(-1, 1, x.shape[-1])
    gate, up = torch.bmm(inputs, selected(w13, w13_scale, w13_global)).chunk(2, dim=-1)
    activated = (torch.nn.functional.silu(gate.float()) * up.float()).to(x.dtype)
    rows = torch.bmm(activated, selected(w2, w2_scale, w2_global))
    return rows.reshape(tokens, top_k, x.shape[-1])
