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
"""MiniMax M3 feed-forward blocks."""

import torch
from torch import nn
from transformers import PreTrainedConfig
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_torchtpu.layers.adapter.dense_fp4_dual import dual_operands
from vllm_torchtpu.layers.adapter.dense_fp4_shared import (
    SHARED_ROWS_ATTR,
    SHARED_ROWS_PIECES,
    model_side_shared,
    shared_operands,
    shared_rows_enabled,
)
from vllm_torchtpu.models.minimax_m3.ops.rows_join import join_rows_op
from vllm_torchtpu.models.minimax_m3.quant import (
    _KERNEL_SWIGLU,
    MiniMaxM3Mxfp8LinearMethod,
    _e4m3_matmul,
    _materialized_bf16,
)
from vllm_torchtpu.utils import fresh_buffer

Rows = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


def cat_rows(pieces: "Rows | list[Rows]") -> Rows:
    """One row block from an SP step's gathered pieces (bf16 rows, or e4m3
    rows with their f32 scales); a single block passes through."""
    if not isinstance(pieces, list):
        return pieces
    if isinstance(pieces[0], tuple):
        return tuple(torch.cat(parts) for parts in zip(*pieces))
    return torch.cat(pieces)


def _dequant(x: Rows) -> torch.Tensor:
    return (x[0].float() * x[1]).to(torch.bfloat16) if isinstance(x, tuple) else x


# The checkpoint config may omit swiglu_beta; SwiGLU-OAI's up-bias is 1.
_DEFAULT_SWIGLU_BETA = 1.0


def swiglu_params(config: PreTrainedConfig) -> tuple[float, float, float]:
    """``(limit, alpha, beta)`` of the SwiGLU-OAI activation."""
    if config.hidden_act != "swigluoai":
        raise ValueError(f"Unsupported activation {config.hidden_act!r}")
    return (
        config.swiglu_limit,
        config.swiglu_alpha,
        getattr(config, "swiglu_beta", _DEFAULT_SWIGLU_BETA),
    )


class MiniMaxM3MLP(nn.Module):
    """Dense SwiGLU-OAI MLP: ``g * sigmoid(alpha * g) * (u + beta)``."""

    def __init__(
        self,
        config: PreTrainedConfig,
        intermediate_size: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        limit, alpha, beta = swiglu_params(config)
        self.act_fn = SiluAndMulWithClamp(swiglu_limit=limit, alpha=alpha, beta=beta)

    def forward(
        self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """``x`` may come as e4m3 rows with per-row f32 scales (an SP step's
        e4m3 all-gather), which gate_up_proj contracts as e4m3."""
        x = cat_rows(x)
        e4m3_apply = getattr(self.gate_up_proj.quant_method, "apply_e4m3_rows", None)
        if isinstance(x, tuple) and e4m3_apply is None:
            x = (x[0].float() * x[1]).to(torch.bfloat16)
        if isinstance(x, tuple):
            # gate_up_proj is bias-free and keeps the linear's forward = apply.
            gate_up = e4m3_apply(self.gate_up_proj, *x)
        else:
            gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(gate_up))
        return x


def e4m3_mlp_operands(mlp: MiniMaxM3MLP) -> tuple[torch.Tensor, ...] | None:
    """``(w_gu, s_gu, w_d, s_d)`` of a bias-free SwiGLU-OAI MLP (the kernels'
    `_KERNEL_SWIGLU`) on per-channel e4m3 linears, for kernels that evaluate it
    whole: e4m3 ``w_gu [h, 2i]`` ([gate | up]) and ``w_d [i, h]``, f32 scales
    ``[1, 2i]`` / ``[1, h]``. None for any other MLP."""
    act = mlp.act_fn
    if not isinstance(act, SiluAndMulWithClamp) or (
        (act.alpha, act.swiglu_limit, act.beta) != _KERNEL_SWIGLU
    ):
        return None
    linears = (mlp.gate_up_proj, mlp.down_proj)
    for linear in linears:
        if (
            not isinstance(linear.quant_method, MiniMaxM3Mxfp8LinearMethod)
            or linear.weight.dtype != torch.float8_e4m3fn
            or linear.weight.dim() != 2
            or linear.weight_scale.dim() != 1
            or getattr(linear, "bias", None) is not None
        ):
            return None
    (w1, s1), (w2, s2) = ((m.weight.data, m.weight_scale.data) for m in linears)
    h, two_i = w1.shape
    if two_i % 2 or tuple(w2.shape) != (two_i // 2, h):
        return None
    # Weights as the parameters themselves (a view would be re-materialized on
    # every call); the f32 [1, n] scales are buffers of their own.
    return (
        w1,
        fresh_buffer(s1.float().reshape(1, -1)),
        w2,
        fresh_buffer(s2.float().reshape(1, -1)),
    )


class MiniMaxM3MoE(nn.Module):
    """Sigmoid-routed experts plus one always-on shared expert.

    The routing bias only picks the experts; the picked sigmoid scores are
    renormalized and scaled by ``routed_scaling_factor``.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        num_experts = config.num_local_experts
        self.gate = GateLinear(
            config.hidden_size,
            num_experts,
            bias=False,
            params_dtype=torch.float32,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        self.shared_experts = MiniMaxM3MLP(
            config,
            config.shared_intermediate_size * config.n_shared_experts,
            quant_config,
            prefix=f"{prefix}.shared_experts",
            reduce_results=False,
        )
        limit, alpha, beta = swiglu_params(config)
        self.experts = FusedMoEFactory(
            num_experts=num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            renormalize=True,
            # w13 is [all gates; all ups], not gpt-oss's interleaved layout.
            activation="swigluoai_uninterleave",
            swiglu_limit=limit,
            swiglu_alpha=alpha,
            swiglu_beta=beta,
            routed_scaling_factor=config.routed_scaling_factor,
            router_logits_dtype=torch.float32,
            shared_experts=self.shared_experts,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def _join_pieces(self, pieces: list[Rows]) -> torch.Tensor:
        """The gathered pieces' bf16 rows in one block: e4m3 pieces in one
        dequantizing pass (`kernels.rows_join`)."""
        join = join_rows_op(len(pieces))
        if join is not None and all(isinstance(p, tuple) for p in pieces):
            return join(*(t for p in pieces for t in p))
        return torch.cat([_dequant(p) for p in pieces])

    @staticmethod
    def _shared_rows(
        routed_layer: nn.Module, pieces: list[Rows]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...] | None:
        """Gathered e4m3 pieces for the routed op's shared expert to contract
        as e4m3 (fused_moe_gmm.shared_activation_e4m3), else None."""
        if (
            not shared_rows_enabled()
            or shared_operands(routed_layer) is None
            or dual_operands(routed_layer) is not None
            or len(pieces) != SHARED_ROWS_PIECES
            or not all(isinstance(p, tuple) for p in pieces)
        ):
            return None
        return tuple(pieces)

    def forward(
        self, hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """``hidden_states`` may come as e4m3 rows with per-row f32 scales (an
        SP step's e4m3 all-gather), or as a list of such gathered pieces: the
        router and routed experts take them dequantized, a model-side shared
        expert contracts them as e4m3."""
        pieces = hidden_states if isinstance(hidden_states, list) else None
        if pieces is not None:
            # Gathered pieces: dequantize and route each as it lands.
            router_logits = torch.cat([self.gate(_dequant(p))[0] for p in pieces])
            x = self._join_pieces(pieces)
            hidden_states = cat_rows(pieces)
        else:
            x = _dequant(hidden_states)
            router_logits = None
        e4m3 = isinstance(hidden_states, tuple)
        routed_layer = getattr(self.experts, "routed_experts", None)
        shared = shared_rows = None
        if routed_layer is not None and model_side_shared(routed_layer, x.shape[0]):
            shared = shared_operands(routed_layer)
        elif routed_layer is not None and pieces is not None:
            shared_rows = self._shared_rows(routed_layer, pieces)
        if router_logits is None:
            router_logits, _ = self.gate(x)
        if shared_rows is not None:
            setattr(routed_layer, SHARED_ROWS_ATTR, shared_rows)
        out = self.experts(hidden_states=x, router_logits=router_logits)
        if shared_rows is not None:
            setattr(routed_layer, SHARED_ROWS_ATTR, None)
        if shared is None:
            return out
        act_fn = self.shared_experts.act_fn
        return out + _shared_expert(act_fn, hidden_states if e4m3 else x, *shared)


def _shared_expert(
    act_fn: nn.Module,
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    w1: torch.Tensor,
    s1: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
) -> torch.Tensor:
    """The armed shared expert (dense_fp4_shared: e4m3 [H, 2I] / [I, H]
    weights, f32 [1, n] channel scales) as `fused_moe_gmm.shared_expert_mlp`
    rounds it; e4m3 rows ``x`` contract as e4m3 x e4m3."""
    if isinstance(x, tuple):
        gate_up = _e4m3_matmul(x[0], x[1], w1, s1)
    else:
        gate_up = (torch.matmul(x, _materialized_bf16(w1)).float() * s1).to(x.dtype)
    act = act_fn(gate_up)
    return (torch.matmul(act, _materialized_bf16(w2)).float() * s2).to(act.dtype)
