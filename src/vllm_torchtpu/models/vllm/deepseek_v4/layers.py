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
"""DeepSeek-V4 feed-forward and Manifold-Harmonized Coupling helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_torchtpu import envs


# Pure PyTorch head projection for MHC manifold collapse;
# upstream vLLM only provides Triton/TileLang GPU kernels for hc_head.
def mhc_collapse_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Collapse multi-stream residual manifold into a single vector before LM head."""
    residual_flat = hidden_states.flatten(-2).float()
    residual_norm = residual_flat * torch.rsqrt(
        residual_flat.square().mean(dim=-1, keepdim=True) + rms_norm_eps
    )
    pre_mix = torch.nn.functional.linear(residual_norm, hc_fn)
    pre_mix = torch.sigmoid(pre_mix * hc_scale + hc_base) + hc_eps
    return torch.sum(pre_mix.unsqueeze(-1) * hidden_states.float(), dim=-2).to(
        hidden_states.dtype
    )


class DeepseekV4MLP(nn.Module):
    """DeepSeek-V4 feed-forward layer for dense/shared expert projections."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if envs.TPU_DEEPSEEK_V41_FUSED_SHARED_EXPERT:
            return self._fused_forward(x)
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x

    def _fused_forward(self, x: torch.Tensor) -> torch.Tensor:
        """The whole MLP as one op: what the three modules above compute."""
        from vllm_torchtpu.layers.adapter.linear_common import (
            WEIGHT_FLIPPED_ATTR,
            fp8_swiglu_mlp,
        )

        act = self.act_fn
        for layer in (self.gate_up_proj, self.down_proj):
            scale = getattr(layer, "weight_scale", None)
            if not (
                getattr(layer, WEIGHT_FLIPPED_ATTR, False)
                and layer.weight.dtype == torch.float8_e4m3fn
                and scale is not None
                and scale.dtype == torch.float32
                and scale.shape == (layer.weight.shape[1],)
                and getattr(layer, "bias", None) is None
                and getattr(layer, "tp_size", 1) == 1
            ):
                raise NotImplementedError(
                    "TPU_DEEPSEEK_V41_FUSED_SHARED_EXPERT needs unsharded, "
                    "bias-free, per-channel FP8 projections in (k, n) layout"
                )
        if not (
            isinstance(act, SiluAndMulWithClamp)
            and act.alpha == 1.0
            and act.beta == 0.0
        ):
            raise NotImplementedError(
                "TPU_DEEPSEEK_V41_FUSED_SHARED_EXPERT needs SiluAndMulWithClamp "
                "with alpha 1 and beta 0"
            )
        return fp8_swiglu_mlp(
            x,
            self.gate_up_proj.weight,
            self.gate_up_proj.weight_scale,
            self.down_proj.weight,
            self.down_proj.weight_scale,
            act.swiglu_limit,
        )
