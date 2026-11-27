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

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.megablox import tc_combine
from vllm_torchtpu.layers.adapter import moe_permute

# The checkpoint config may omit swiglu_beta; SwiGLU-OAI's up-bias is 1.
_DEFAULT_SWIGLU_BETA = 1.0

# An SP step's gathered rows: bf16, or e4m3 with per-row f32 scales.
Rows = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


def cat_rows(pieces: "Rows | list[Rows]") -> Rows:
    """One row block from an SP step's gathered row pieces
    (TPU_MINIMAX_M3_SP_FFN_PIECES); a single block passes through."""
    if not isinstance(pieces, list):
        return pieces
    if isinstance(pieces[0], tuple):
        return (torch.cat([p[0] for p in pieces]), torch.cat([p[1] for p in pieces]))
    return torch.cat(pieces)


def _dequant(x: Rows) -> torch.Tensor:
    return (x[0].float() * x[1]).to(torch.bfloat16) if isinstance(x, tuple) else x


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

    def forward(self, x: Rows | list[Rows]) -> torch.Tensor:
        """``x`` may come as e4m3 rows with per-row f32 scales (an SP step's
        e4m3 all-gather), which gate_up_proj contracts as e4m3, or as a list
        of such gathered row pieces, joined in order."""
        x, _ = self.down_proj(self.activation(x))
        return x

    def activation(self, x: Rows | list[Rows]) -> torch.Tensor:
        """``act_fn(gate_up_proj(x))``: the down projection's input."""
        x = cat_rows(x)
        if isinstance(x, tuple):
            e4m3_apply = getattr(self.gate_up_proj.quant_method, "apply_e4m3_rows", None)
            if e4m3_apply is None:
                gate_up, _ = self.gate_up_proj((x[0].float() * x[1]).to(torch.bfloat16))
            else:
                # gate_up_proj is bias-free and keeps the linear's forward = apply.
                gate_up = e4m3_apply(self.gate_up_proj, *x)
        else:
            rows = self.__dict__.get("_e4m3_rows")
            if rows is not None and rows[0].shape[0] == x.shape[0]:
                # Shared expert of an SP step (TPU_MINIMAX_M3_SP_E4M3_SHARED):
                # x is the bf16 dequantization of these e4m3 rows.
                return self.activation(rows)
            gate_up, _ = self.gate_up_proj(x)
        return self.act_fn(gate_up)


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
        # VLLM_TPU_MOE_EXPERT_PERMUTATION: the router emits physical expert ids.
        moe_permute.permute_router(
            self.experts,
            gate_weight=self.gate.weight,
            e_score_correction_bias=self.e_score_correction_bias,
        )

    def forward(self, hidden_states: Rows | list[Rows]) -> torch.Tensor:
        """``hidden_states`` may come as e4m3 rows with per-row f32 scales (an
        SP step's e4m3 all-gather): the router and the experts (shared expert
        included) take them dequantized to bf16. An SP step with row pieces
        (TPU_MINIMAX_M3_SP_FFN_PIECES) hands the gathered pieces as a list
        (`_join_pieces`)."""
        rows = None
        router_logits = None
        if isinstance(hidden_states, list):
            hidden_states, rows, router_logits = self._join_pieces(hidden_states)
        elif isinstance(hidden_states, tuple):
            q8, row_scale = hidden_states
            hidden_states = (q8.float() * row_scale).to(torch.bfloat16)
            if envs.TPU_MINIMAX_M3_SP_E4M3_SHARED:
                rows = (q8, row_scale)
        if router_logits is None:
            router_logits, _ = self.gate(hidden_states)
        if self._shared_in_combine(hidden_states.shape[0]):
            return self._forward_shared_in_combine(hidden_states, router_logits, rows)
        if rows is None:
            return self.experts(hidden_states=hidden_states, router_logits=router_logits)
        # The shared expert runs inside the MoE runner on hidden_states; hand
        # it the e4m3 rows for the duration of this call.
        self.shared_experts.__dict__["_e4m3_rows"] = rows
        try:
            return self.experts(hidden_states=hidden_states, router_logits=router_logits)
        finally:
            self.shared_experts.__dict__.pop("_e4m3_rows", None)

    def _join_pieces(
        self, pieces: list[Rows]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor]:
        """An SP step's gathered row pieces (piece-major rows; the layer sets
        their padding mask, model.py `_ffn_sp`) as (bf16 rows, the e4m3 rows
        for TPU_MINIMAX_M3_SP_E4M3_SHARED or None, router logits). Each piece
        is dequantized and routed on its own, so that work can start on a
        piece while the next one lands; every step is row-local (same bits
        as one block). This is the hook where the whole MoE sees the pieces:
        an op that wants them separately can take ``pieces`` here."""
        rows_bf16 = [_dequant(p) for p in pieces]
        router_logits = torch.cat([self.gate(x)[0] for x in rows_bf16])
        hidden_states = torch.cat(rows_bf16)
        rows = None
        if envs.TPU_MINIMAX_M3_SP_E4M3_SHARED and isinstance(pieces[0], tuple):
            rows = cat_rows(pieces)
        return hidden_states, rows, router_logits

    def _shared_in_combine(self, num_tokens: int) -> bool:
        """TPU_MINIMAX_M3_SHARED_IN_COMBINE on a step whose layer reduces the
        MoE output itself (SP steps: the runner would return shared + routed
        unreduced), with the plain GMM path, routing weights carrying the
        routed scale, an e4m3 per-channel shared down_proj, and a token count
        the TensorCore combine tiles (other steps keep the runner's graph)."""
        if not envs.TPU_MINIMAX_M3_SHARED_IN_COMBINE or not envs.TPU_MOE_TC_COMBINE:
            return False
        if (
            num_tokens % tc_combine.TOKEN_BLOCK
            or num_tokens < envs.TPU_MINIMAX_M3_SHARED_IN_COMBINE_MIN_TOKENS
        ):
            return False
        runner = self.experts
        if not runner.moe_config.skip_final_all_reduce:
            return False
        down = self.shared_experts.down_proj
        qm = runner.routed_experts.quant_method
        return (
            getattr(qm, "shared_in_combine_supported", None) is not None
            and qm.shared_in_combine_supported()
            and runner.routed_scaling_factor == 1.0
            and down.weight.dtype == torch.float8_e4m3fn
            and getattr(down, "weight_scale", None) is not None
            and down.weight_scale.dim() == 1
            and getattr(down, "bias", None) is None
        )

    def _forward_shared_in_combine(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        rows: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """The runner's unreduced shared + routed sum, with the shared
        down_proj run inside the TensorCore combine (same roundings).
        ``rows``: the e4m3 rows of TPU_MINIMAX_M3_SP_E4M3_SHARED, which the
        shared gate_up contracts as the runner's shared expert would."""
        down = self.shared_experts.down_proj
        act = self.shared_experts.activation(hidden_states if rows is None else rows)
        # Consume the shared down_proj weight and scale here, right after the
        # shared gate_up, as the runner's shared expert does. vLLM's
        # split_graph orders a subgraph's inputs by first use; left to the
        # combine op, these two would come after the routed experts' weights,
        # and the warmup's shape-variant traces (this path only exists in the
        # >= SHARED_IN_COMBINE_MIN_TOKENS trace) would take their inputs in a
        # different order than the small-bucket trace, which refuses the
        # bucket hand-over (ShapeSpecializationError at startup). The views
        # are free.
        shared_w = down.weight.view(down.weight.shape)
        shared_scale = down.weight_scale.view(down.weight_scale.shape)
        routed = self.experts.routed_experts
        return routed.quant_method.apply_monolithic_shared(
            routed,
            hidden_states,
            router_logits,
            act,
            shared_w,
            shared_scale,
        )
