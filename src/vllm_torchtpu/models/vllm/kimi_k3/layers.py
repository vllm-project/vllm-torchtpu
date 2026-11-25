# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Small Kimi layers that are not attention or MoE."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, cast

import jax
import torch
from torch import nn
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.gated_mlp import GatedMLP
from vllm_torchtpu.utils import fresh_buffer

from .collective_ops import TokenShardCollectives

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator


@cache
def _build_attention_residual_op(eps: float):
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.kimi_k3.attention_residual import attention_residual

    def core(prefix: jax.Array, history: jax.Array, weight: jax.Array) -> jax.Array:
        return attention_residual(prefix, history, weight, eps=eps)

    suffix = float(eps).hex().replace(".", "_").replace("-", "m").replace("+", "p")
    op = pallas.jax_op(f"pallas::kimi_attention_residual_{suffix}", core)
    op.register_fake(lambda prefix, history, weight: torch.empty_like(prefix))
    return op


class KimiMLP(GatedMLP):
    """Gated MLP with Kimi K3's token collectives under TPU_K3_SP_PREFILL."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        reduce_results: bool = True,
        situ_beta: float | None = None,
        situ_linear_beta: float | None = None,
        tp_group: GroupCoordinator | None = None,
    ) -> None:
        super().__init__(
            hidden_size,
            intermediate_size,
            hidden_act,
            quant_config=quant_config,
            prefix=prefix,
            reduce_results=reduce_results,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            tp_group=tp_group,
            sp_group=(
                TokenShardCollectives(tp_group or get_tp_group())
                if envs.TPU_K3_SP_PREFILL
                else None
            ),
        )


class AttentionResidual(nn.Module):
    """K3 attention residual with a fused kernel for tiled prefill."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        prefix: str,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.norm = RMSNorm(hidden_size, eps)
        self.proj = ReplicatedLinear(
            hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=prefix,
        )

    def process_weights_after_loading(self) -> None:
        """Fold the norm and projection weights into one [hidden] vector."""
        folded = (
            self.norm.weight.float() * cast(torch.Tensor, self.proj.weight)[0].float()
        )
        out = fresh_buffer(folded)
        self.folded_weight = out
        self._prefill_op = (
            _build_attention_residual_op(self.eps) if out.device.type == "tpu" else None
        )

    def forward(
        self,
        prefix_sum: torch.Tensor,
        block_residuals: torch.Tensor,
    ) -> torch.Tensor:
        if block_residuals.shape[1] == 0:
            return prefix_sum
        if self._prefill_op is not None and prefix_sum.shape[0] % 16 == 0:
            return self._prefill_op(prefix_sum, block_residuals, self.folded_weight)
        w = self.folded_weight
        if prefix_sum.shape[0] == 1:
            # With one token the per-slot statistics below are scalars that
            # XLA leaves unfused; reducing over the concatenated slots keeps
            # them one vector. A shape branch, so bucket 1 gets its own trace
            # (see compilation.shape_variants).
            values = torch.cat(
                (block_residuals, prefix_sum.unsqueeze(-2)), dim=-2
            ).float()  # [1, K, hidden]
            inv = torch.rsqrt(values.pow(2).mean(-1) + self.eps)
            probabilities = ((values * w).sum(-1) * inv).softmax(dim=-1)
            out = (probabilities.unsqueeze(-1) * values).sum(dim=-2)
            return out.to(block_residuals.dtype)
        v_blocks = block_residuals.float()
        v_prefix = prefix_sum.float()
        inv_blocks = torch.rsqrt(v_blocks.pow(2).mean(-1) + self.eps)
        inv_prefix = torch.rsqrt(v_prefix.pow(2).mean(-1) + self.eps)
        scores = torch.cat(
            (
                (v_blocks * w).sum(-1) * inv_blocks,
                ((v_prefix * w).sum(-1) * inv_prefix).unsqueeze(-1),
            ),
            dim=-1,
        )  # [T, K]
        probabilities = scores.softmax(dim=-1)
        out = (probabilities[..., :-1].unsqueeze(-1) * v_blocks).sum(dim=-2)
        out = out + probabilities[..., -1].unsqueeze(-1) * v_prefix
        return out.to(block_residuals.dtype)
