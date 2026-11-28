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
"""TPU-portable Inkling 2 / Inkling 2 Loop reference model.

Implements the prelude -> looped core (with explicit per-visit KV and 4-stream
short-conv caches plus rank-r inter-pass message controllers) -> coda schedule
with zero CUDA or Triton dependencies. Serves as the end-to-end numerical
reference for ``ModelAdapter`` and Pallas kernel integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn
from torch.nn import functional

from vllm_torchtpu.models.inkling2.experimental.config import Inkling2LoopConfig
from vllm_torchtpu.models.inkling2.experimental.ops import (
    attn_res,
    compute_log_scaling_tau,
    inkling2_moe_forward,
    l2_clip,
    message_controller_blend,
    relative_bias_attention_dense,
    relative_bias_attention_split_tpu,
    rms_norm,
    short_conv_dynamic,
    short_conv_static,
    value_norm_no_gamma,
)

AttnImpl = Literal["dense", "split_tpu"]


@dataclass
class PassStateCache:
    """Per-visit KV and 4-stream short-conv cache for one ``(layer_id, pass_idx)``."""

    k_cache: torch.Tensor | None = None
    v_cache: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    k_conv_hist: torch.Tensor | None = None
    v_conv_hist: torch.Tensor | None = None
    attn_conv_hist: torch.Tensor | None = None
    mlp_conv_hist: torch.Tensor | None = None


@dataclass
class Inkling2LoopCache:
    """Explicit per-visit cache keyed by ``(physical_layer_id, pass_idx)``."""

    visits: dict[tuple[int, int], PassStateCache] = field(default_factory=dict)

    def get(self, layer_id: int, pass_idx: int) -> PassStateCache:
        """Return or initialize the state cache for ``(layer_id, pass_idx)``."""
        key = (layer_id, pass_idx)
        if key not in self.visits:
            self.visits[key] = PassStateCache()
        return self.visits[key]


class Inkling2DecoderLayer(nn.Module):
    """Single physical Inkling 2 decoder layer (pure PyTorch, TPU-portable)."""

    def __init__(
        self, config: Inkling2LoopConfig, layer_id: int, *, is_core: bool
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.is_local = layer_id in config.local_layer_set
        self.is_core = is_core

        hidden = config.hidden_size
        num_q = config.num_attention_heads
        num_kv = config.num_key_value_heads
        head_dim = config.head_dim
        d_rel = config.d_rel
        kernel_size = config.sconv_kernel_size
        rel_width = config.sliding_window_size if self.is_local else config.rel_extent

        self.num_attn_inputs = 1 if self.is_local else 3
        self.attn_norm_weight = nn.Parameter(torch.ones(self.num_attn_inputs, hidden))
        self.mlp_norm_weight = nn.Parameter(torch.ones(1, hidden))

        self.wq = nn.Parameter(torch.randn(num_q * head_dim, hidden) * 0.02)
        self.wr = nn.Parameter(torch.randn(num_q * d_rel, hidden) * 0.02)
        self.w_gate = nn.Parameter(torch.randn(num_q * head_dim, hidden) * 0.02)
        self.w_sink = nn.Parameter(torch.randn(num_q, hidden) * 0.02)
        self.wk = nn.Parameter(torch.randn(num_kv * head_dim, hidden) * 0.02)
        self.wv = nn.Parameter(torch.randn(num_kv * head_dim, hidden) * 0.02)
        self.wo = nn.Parameter(torch.randn(hidden, num_q * head_dim) * 0.02)

        self.q_scale = nn.Parameter(torch.ones(num_q, head_dim))
        self.r_scale = nn.Parameter(torch.ones(num_q, d_rel))
        self.rel_logits_proj = nn.Parameter(torch.randn(d_rel, rel_width) * 0.02)

        kv_dim = num_kv * head_dim
        self.k_sconv_proj: nn.Parameter | None
        self.v_sconv_proj: nn.Parameter | None
        if self.is_local:
            self.k_sconv_w = nn.Parameter(torch.randn(kv_dim, kernel_size) * 0.05)
            self.v_sconv_w = nn.Parameter(torch.randn(kv_dim, kernel_size) * 0.05)
            self.k_sconv_proj = None
            self.v_sconv_proj = None
        else:
            self.k_sconv_w = nn.Parameter(torch.randn(kernel_size, kv_dim) * 0.05)
            self.v_sconv_w = nn.Parameter(torch.randn(kernel_size, kv_dim) * 0.05)
            self.k_sconv_proj = nn.Parameter(
                torch.randn(num_kv * kernel_size, hidden) * 0.02
            )
            self.v_sconv_proj = nn.Parameter(
                torch.randn(num_kv * kernel_size, hidden) * 0.02
            )

        self.attn_sconv_w = nn.Parameter(torch.randn(hidden, kernel_size) * 0.05)
        self.mlp_sconv_w = nn.Parameter(torch.randn(hidden, kernel_size) * 0.05)

        num_experts = config.n_routed_experts
        inter = config.intermediate_size
        self.gate_weight = nn.Parameter(torch.randn(num_experts, hidden) * 0.02)
        self.gate_bias = nn.Parameter(torch.zeros(num_experts, dtype=torch.float32))
        self.w13_weight = nn.Parameter(
            torch.randn(num_experts, 2 * inter, hidden) * 0.02
        )
        self.w2_weight = nn.Parameter(torch.randn(num_experts, hidden, inter) * 0.02)

        self.ctrl_down: nn.Parameter | None = None
        self.ctrl_gain: nn.Parameter | None = None
        self.ctrl_forget: nn.Parameter | None = None
        if is_core:
            rank = config.gate_rank
            self.ctrl_down = nn.Parameter(torch.randn(4, rank, 2 * hidden) * 0.02)
            self.ctrl_gain = nn.Parameter(torch.randn(4, hidden, rank) * 0.02)
            self.ctrl_forget = nn.Parameter(torch.randn(4, hidden, rank) * 0.02)

    def run_attention(
        self,
        positions: torch.Tensor,
        attn_inputs: torch.Tensor,
        tau: torch.Tensor | None,
        state: PassStateCache,
        attn_impl: AttnImpl,
    ) -> torch.Tensor:
        """Project Q/K/V/R/gate/sink, update per-visit KV + sconv state, and attend."""
        cfg = self.config
        num_tokens = positions.shape[0]
        num_q = cfg.num_attention_heads
        num_kv = cfg.num_key_value_heads
        head_dim = cfg.head_dim
        d_rel = cfg.d_rel

        if attn_inputs.shape[0] == 1:
            x_q = x_k = x_v = attn_inputs[0]
        else:
            x_q, x_k, x_v = attn_inputs[0], attn_inputs[1], attn_inputs[2]

        q = functional.linear(x_q, self.wq).view(num_tokens, num_q, head_dim)
        r = functional.linear(x_q, self.wr).view(num_tokens, num_q, d_rel)
        gate = functional.linear(x_q, self.w_gate).view(num_tokens, num_q, head_dim)
        sinks = functional.linear(x_q, self.w_sink)
        k_raw = functional.linear(x_k, self.wk)
        v_raw = functional.linear(x_v, self.wv)

        if self.is_local:
            k_conv, state.k_conv_hist = short_conv_static(
                k_raw, self.k_sconv_w, state.k_conv_hist
            )
            v_conv, state.v_conv_hist = short_conv_static(
                v_raw, self.v_sconv_w, state.v_conv_hist
            )
        else:
            if self.k_sconv_proj is None or self.v_sconv_proj is None:
                raise RuntimeError("Full-attention layer missing dynamic sconv proj")
            k_dyn = functional.linear(x_k, self.k_sconv_proj)
            v_dyn = functional.linear(x_v, self.v_sconv_proj)
            k_conv, state.k_conv_hist = short_conv_dynamic(
                k_raw, self.k_sconv_w, k_dyn, num_kv, state.k_conv_hist
            )
            v_conv, state.v_conv_hist = short_conv_dynamic(
                v_raw, self.v_sconv_w, v_dyn, num_kv, state.v_conv_hist
            )

        k_cur = l2_clip(
            k_conv.view(num_tokens, num_kv, head_dim),
            None,
            cfg.qk_l2_clip_threshold,
            cfg.rms_norm_eps,
        )
        v_cur = value_norm_no_gamma(
            v_conv.view(num_tokens, num_kv, head_dim), cfg.rms_norm_eps
        )
        tau_view = (
            None if (self.is_local or tau is None) else tau.view(num_tokens, 1, 1)
        )
        q_cur = l2_clip(
            q,
            self.q_scale.unsqueeze(0),
            cfg.qk_l2_clip_threshold,
            cfg.rms_norm_eps,
            tau_view,
        )
        r_cur = l2_clip(
            r,
            self.r_scale.unsqueeze(0),
            cfg.rel_l2_clip_threshold,
            cfg.rms_norm_eps,
            tau_view,
        )

        if state.k_cache is None or state.v_cache is None or state.positions is None:
            k_all, v_all, pos_all = k_cur, v_cur, positions
        else:
            k_all = torch.cat((state.k_cache, k_cur), dim=0)
            v_all = torch.cat((state.v_cache, v_cur), dim=0)
            pos_all = torch.cat((state.positions, positions), dim=0)
        state.k_cache, state.v_cache, state.positions = k_all, v_all, pos_all

        attn_fn = (
            relative_bias_attention_split_tpu
            if attn_impl == "split_tpu"
            else relative_bias_attention_dense
        )
        attn_out = attn_fn(
            q_cur,
            k_all,
            v_all,
            r_cur,
            self.rel_logits_proj,
            gate,
            sinks,
            positions,
            pos_all,
            sliding_window=cfg.sliding_window_size if self.is_local else None,
        )
        return functional.linear(
            attn_out.reshape(num_tokens, num_q * head_dim), self.wo
        )


class Inkling2LoopForCausalLM(nn.Module):
    """End-to-end TPU-portable Inkling 2 / Inkling 2 Loop causal language model."""

    def __init__(self, config: Inkling2LoopConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        num_layers = config.num_hidden_layers
        has_loop = config.num_passes > 1 and config.core_start < config.core_end
        num_core = (config.core_end - config.core_start) if has_loop else 0

        self.embed = nn.Embedding(config.vocab_size, hidden)
        self.embed_norm_weight = nn.Parameter(torch.ones(hidden))
        self.layers = nn.ModuleList(
            [
                Inkling2DecoderLayer(
                    config,
                    layer_id,
                    is_core=has_loop
                    and (config.core_start <= layer_id < config.core_end),
                )
                for layer_id in range(num_layers)
            ]
        )

        scale = hidden**-0.5
        self.attn_queries = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        (1, hidden)
                        if layer_id in config.local_layer_set
                        else (3, hidden)
                    )
                    * 0.02
                    * scale
                )
                for layer_id in range(1, num_layers)
            ]
        )
        self.mlp_queries = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, hidden) * 0.02 * scale)
                for _ in range(num_layers)
            ]
        )
        self.output_query = nn.Parameter(torch.randn(1, hidden) * 0.02 * scale)
        self.message_queries: nn.Parameter | None = (
            nn.Parameter(torch.randn(num_core, 4, hidden) * 0.02 * scale)
            if num_core > 0
            else None
        )
        self.final_norm_weight = nn.Parameter(torch.ones(hidden))
        self.unembed = nn.Parameter(torch.randn(config.vocab_size, hidden) * 0.02)

    def _max_loop_blocks(self) -> int:
        cfg = self.config
        num_blocks = 0
        for layer_id in range(cfg.num_hidden_layers):
            num_blocks += int(layer_id % cfg.n_layers_per_attn_res_block == 0)
        return num_blocks

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cache: Inkling2LoopCache | None = None,
        *,
        attn_impl: AttnImpl = "split_tpu",
    ) -> torch.Tensor:
        """Compute FP32 softcapped logits ``[T, unpadded_vocab_size]``."""
        cfg = self.config
        active_cache = Inkling2LoopCache() if cache is None else cache
        num_tokens = input_ids.shape[0]
        eps = cfg.rms_norm_eps
        block_size = cfg.n_layers_per_attn_res_block

        h0 = rms_norm(self.embed(input_ids), self.embed_norm_weight, eps)
        dtype = h0.dtype
        stack = torch.zeros(
            (self._max_loop_blocks() + 1, num_tokens, cfg.hidden_size),
            dtype=dtype,
            device=h0.device,
        )
        stack[0] = h0

        tau = compute_log_scaling_tau(
            positions, cfg.log_scaling_n_floor, cfg.log_scaling_alpha
        )
        visits = cfg.visits

        num_blocks = 0
        inputs: torch.Tensor | None = None
        bank: torch.Tensor | None = None
        prelude_blocks: int | None = None
        prelude_partial: torch.Tensor | None = None
        prelude_dests: torch.Tensor | None = None

        for visit_idx, (layer_id, pass_idx) in enumerate(visits):
            looped = pass_idx > 0
            if layer_id == cfg.core_start and not looped and cfg.num_passes > 1:
                prelude_blocks = num_blocks
                prelude_partial = stack[num_blocks].clone()
            elif layer_id == cfg.core_start and looped:
                if prelude_blocks is None or prelude_partial is None:
                    raise RuntimeError("Looped core pass reached without prelude state")
                bank = stack[prelude_blocks + 1 : num_blocks + 1].clone()
                num_blocks = prelude_blocks
                stack[num_blocks] = prelude_partial.clone()
                if prelude_dests is None:
                    prelude_dests = attn_res(
                        stack[:num_blocks],
                        prelude_partial,
                        None,
                        self.attn_queries[layer_id - 1],
                        None,
                        eps,
                        cfg.attn_res_normalize_values,
                    )
                inputs = prelude_dests

            layer = self.layers[layer_id]
            if not isinstance(layer, Inkling2DecoderLayer):
                raise TypeError(f"Unexpected layer type: {type(layer)}")
            state = active_cache.get(layer_id, pass_idx)
            partial = stack[num_blocks]
            messages: torch.Tensor | None = None

            if looped and bank is not None:
                if (
                    inputs is None
                    or self.message_queries is None
                    or layer.ctrl_down is None
                    or layer.ctrl_gain is None
                    or layer.ctrl_forget is None
                ):
                    raise RuntimeError(
                        "Missing loop controller parameters on core layer"
                    )
                messages = attn_res(
                    bank[:1],
                    bank[1],
                    None,
                    self.message_queries[layer_id - cfg.core_start],
                    None,
                    eps,
                    cfg.attn_res_normalize_values,
                )
                attn_inputs = message_controller_blend(
                    inputs,
                    messages[:3],
                    layer.ctrl_down[:3],
                    layer.ctrl_gain[:3],
                    layer.ctrl_forget[:3],
                    layer.attn_norm_weight,
                    eps,
                    out_dtype=dtype,
                )
            elif inputs is None:
                attn_inputs = torch.stack(
                    [
                        rms_norm(partial, layer.attn_norm_weight[j], eps)
                        for j in range(layer.num_attn_inputs)
                    ],
                    dim=0,
                )
            else:
                attn_inputs = inputs

            attn_raw = layer.run_attention(
                positions, attn_inputs, tau, state, attn_impl
            )
            attn_out, state.attn_conv_hist = short_conv_static(
                attn_raw, layer.attn_sconv_w, state.attn_conv_hist
            )

            first = layer_id % block_size == 0
            mlp_norm = None if messages is not None else layer.mlp_norm_weight
            if first:
                mlp_in = attn_res(
                    stack[: num_blocks + 1],
                    attn_out,
                    None,
                    self.mlp_queries[layer_id],
                    mlp_norm,
                    eps,
                    cfg.attn_res_normalize_values,
                    out_dtype=dtype,
                )
            else:
                mlp_in = attn_res(
                    stack[:num_blocks],
                    partial,
                    attn_out,
                    self.mlp_queries[layer_id],
                    mlp_norm,
                    eps,
                    cfg.attn_res_normalize_values,
                    out_dtype=dtype,
                )
            if messages is not None:
                if (
                    layer.ctrl_down is None
                    or layer.ctrl_gain is None
                    or layer.ctrl_forget is None
                ):
                    raise RuntimeError(
                        "Missing loop controller parameters on core layer"
                    )
                mlp_in = message_controller_blend(
                    mlp_in,
                    messages[3:4],
                    layer.ctrl_down[3:4],
                    layer.ctrl_gain[3:4],
                    layer.ctrl_forget[3:4],
                    layer.mlp_norm_weight,
                    eps,
                    out_dtype=dtype,
                )

            residual = None if first else partial
            if residual is None:
                num_blocks += 1
            mlp_raw = inkling2_moe_forward(
                mlp_in[0],
                layer.gate_weight,
                layer.gate_bias,
                layer.w13_weight,
                layer.w2_weight,
                topk=cfg.num_experts_per_tok,
                route_scale=cfg.route_scale,
                gate_cap=cfg.mlp_gate_softcap,
                up_cap=cfg.mlp_up_softcap,
            )
            mlp_conv, state.mlp_conv_hist = short_conv_static(
                mlp_raw, layer.mlp_sconv_w, state.mlp_conv_hist
            )
            acc = (
                attn_out.to(torch.float32)
                if residual is None
                else residual.to(torch.float32) + attn_out.to(torch.float32)
            )
            stack[num_blocks] = (acc + mlp_conv.to(torch.float32)).to(dtype)

            if visit_idx + 1 == len(visits):
                next_query = self.output_query
                next_norm: torch.Tensor | None = self.final_norm_weight
            else:
                next_layer_id, next_pass = visits[visit_idx + 1]
                next_layer = self.layers[next_layer_id]
                if not isinstance(next_layer, Inkling2DecoderLayer):
                    raise TypeError(f"Unexpected layer type: {type(next_layer)}")
                next_query = self.attn_queries[next_layer_id - 1]
                next_norm = None if next_pass else next_layer.attn_norm_weight

            inputs = attn_res(
                stack[:num_blocks],
                stack[num_blocks],
                None,
                next_query,
                next_norm,
                eps,
                cfg.attn_res_normalize_values,
                out_dtype=dtype,
            )

        if inputs is None:
            raise RuntimeError("Model executed zero layer visits")
        logits = functional.linear(
            inputs[0].to(torch.float32), self.unembed.to(torch.float32)
        )
        if cfg.final_logit_softcapping > 0:
            cap = cfg.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)
        return logits[:, : cfg.unpadded_vocab_size]
