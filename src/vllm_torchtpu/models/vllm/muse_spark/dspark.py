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
"""TPU implementation of the Muse Spark DSpark draft model."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    get_draft_quant_config,
    maybe_prefix,
)

from vllm_torchtpu.models.vllm.muse_spark.attention_compat import (
    install_muse_hd64_attention,
)
from vllm_torchtpu.models.vllm.muse_spark.batch_invariant import det_projection
from vllm_torchtpu.models.vllm.muse_spark.configuration import MuseSparkDSparkConfig
from vllm_torchtpu.models.vllm.muse_spark.det_collectives import det_all_reduce
from vllm_torchtpu.models.vllm.muse_spark.text_model import (
    MuseSparkCenteredRMSNorm,
    MuseSparkInputScaledRMSNorm,
    MuseSparkResidualGate,
    MuseSparkRMSNorm,
)


def _remap_dspark_name(name: str) -> str | None:
    # Muse Spark DSpark has no confidence head on TPU (see
    # MuseSparkDSparkTextModel.confidence_head); drop its weights.
    if "confidence_head" in name:
        return None
    if name == "fc.weight":
        return "model.fc.weight"
    if name == "hidden_norm.weight":
        return "model.hidden_norm.weight"
    if name == "norm.weight":
        return "model.norm.weight"
    if name == "d2t":
        return "model.d2t"
    if name.startswith("markov_head."):
        return f"model.{name}"
    if name == "lm_head.weight":
        return "lm_head.weight"

    m = re.match(r"^layers\.(\d+)\.(.*)$", name)
    if m:
        layer_idx = m.group(1)
        rest = m.group(2)
        if rest == "self_attn.pre_qkv_proj_norm.weight":
            return f"model.layers.{layer_idx}.input_layernorm.weight"
        if rest == "add_attn.gate":
            return f"model.layers.{layer_idx}.post_attention_residual_gate.gate"
        if rest == "pre_ffn_norm.weight":
            return f"model.layers.{layer_idx}.pre_feedforward_layernorm.weight"
        if rest == "post_ffn_norm.weight":
            return f"model.layers.{layer_idx}.post_feedforward_layernorm.weight"
        if rest == "add_ffn.gate":
            return f"model.layers.{layer_idx}.post_feedforward_residual_gate.gate"
        return f"model.layers.{layer_idx}.{rest}"
    return name


class MuseSparkDSparkAttention(nn.Module):
    def __init__(
        self,
        config: MuseSparkDSparkConfig,
        cache_config: CacheConfig | None = None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = int(config.hidden_size)
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(config.num_key_value_heads)
        # q/k/v are plain column-parallel projections with no KV-head
        # replication, so both head counts must split evenly across TP.
        if self.total_num_heads % tp_size != 0:
            raise ValueError(
                f"DSpark draft num_attention_heads ({self.total_num_heads}) "
                f"must be divisible by tensor_parallel_size ({tp_size})"
            )
        if self.total_num_kv_heads % tp_size != 0:
            raise ValueError(
                f"DSpark draft num_key_value_heads ({self.total_num_kv_heads}) "
                f"must be divisible by tensor_parallel_size ({tp_size})"
            )
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = int(config.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = float(self.head_dim**-0.5)
        self.qk_scale_factor = float(config.qk_scale_factor)
        self.sliding_window = (
            int(config.sliding_window) if config.sliding_window else 2048
        )
        self._muse_rope_permuted: bool = False

        self.qk_norm = MuseSparkRMSNorm(eps=config.rms_norm_eps, with_scale=False)
        self.o_norm = MuseSparkRMSNorm(eps=config.rms_norm_eps, with_scale=False)

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.o_gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_gate_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.o_proj",
        )

        rope_theta = float(config.rope_theta)
        rope_params = dict(config.rope_parameters) if config.rope_parameters else {}
        rope_params["rope_theta"] = rope_theta
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_params,
            is_neox_style=True,
            dtype=torch.float32,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=None,
            per_layer_sliding_window=self.sliding_window,
            prefix=f"{prefix}.attn",
        )
        install_muse_hd64_attention(self.attn)
        # impl keeps the 2048 window for masking; a full attention KV spec
        # stops the sliding window manager from evicting warm prefix blocks.
        self.attn.sliding_window = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q = det_projection(self.q_proj, hidden_states)
        k = det_projection(self.k_proj, hidden_states)
        v = det_projection(self.v_proj, hidden_states)

        q = (
            self.qk_norm(q.reshape(-1, self.head_dim)).reshape(-1, self.q_size)
            * self.qk_scale_factor
        ).to(v.dtype)
        k = (self.qk_norm(k.reshape(-1, self.head_dim)).reshape(-1, self.kv_size)).to(
            v.dtype
        )

        if self.rotary_emb is not None:
            q_f32, k_f32 = self.rotary_emb(positions, q.float(), k.float())
            q, k = q_f32.to(v.dtype), k_f32.to(v.dtype)

        attn_output = self.attn(q, k, v)
        gate = det_projection(self.o_gate_proj, hidden_states)
        attn_output = self.o_norm(attn_output.reshape(-1, self.head_dim)).reshape(
            -1, self.q_size
        )
        attn_output = attn_output * torch.sigmoid(gate)
        output = det_projection(self.o_proj, attn_output)
        output = det_all_reduce(output)
        return output


class MuseSparkDSparkMLP(nn.Module):
    def __init__(
        self,
        config: MuseSparkDSparkConfig,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = det_projection(self.gate_proj, x)
        up = det_projection(self.up_proj, x)
        act = F.silu(gate.float()) * up.float()
        out = det_projection(self.down_proj, act.to(x.dtype))
        return det_all_reduce(out)


class MuseSparkDSparkDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MuseSparkDSparkConfig,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = int(config.hidden_size)

        self.input_layernorm = MuseSparkCenteredRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = MuseSparkDSparkAttention(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.post_attention_layernorm = MuseSparkRMSNorm(
            self.hidden_size, eps=config.rms_post_norm_eps, with_scale=False
        )
        self.post_attention_residual_gate = MuseSparkResidualGate(
            config=config, residual_gate_eps=1e-3
        )
        self.pre_feedforward_layernorm = MuseSparkCenteredRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.feed_forward = MuseSparkDSparkMLP(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.feed_forward",
        )
        self.post_feedforward_layernorm = MuseSparkInputScaledRMSNorm(
            self.hidden_size, eps=config.rms_post_norm_eps
        )
        self.post_feedforward_residual_gate = MuseSparkResidualGate(
            config=config, residual_gate_eps=1e-3
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        act_dtype = hidden_states.dtype
        if act_dtype == torch.float32:
            act_dtype = torch.bfloat16

        residual = hidden_states
        h = self.input_layernorm(residual, out_dtype=act_dtype)
        attn_out = self.self_attn(positions=positions, hidden_states=h)
        attn_out = self.post_attention_layernorm(attn_out)
        hidden_states = self.post_attention_residual_gate(residual, attn_out)

        residual = hidden_states
        h = self.pre_feedforward_layernorm(residual, out_dtype=act_dtype)
        ff_out = self.feed_forward(h)
        ff_out = self.post_feedforward_layernorm(ff_out, round_dtype=act_dtype)
        hidden_states = self.post_feedforward_residual_gate(residual, ff_out)
        return hidden_states


def _safe_draft_quant_config(vllm_config: VllmConfig):
    spec_cfg = vllm_config.speculative_config
    if spec_cfg is None or spec_cfg.draft_model_config is None:
        return None
    return get_draft_quant_config(vllm_config)


class MuseSparkDSparkTextModel(nn.Module):
    def __init__(
        self,
        config: MuseSparkDSparkConfig,
        vllm_config: VllmConfig,
        start_layer_id: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        quant_config = _safe_draft_quant_config(vllm_config)
        cache_config = vllm_config.cache_config
        self.residual_fp32 = config.residual_dtype == "fp32"

        self.embed_tokens: nn.Module | None = None
        self.embed_norm = MuseSparkRMSNorm(eps=config.rms_norm_eps, with_scale=False)

        target_len = (
            len(config.target_layer_ids)
            if config.target_layer_ids
            else config.num_target_layers
        )
        self.fc = ReplicatedLinear(
            config.target_hidden_size * target_len,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "fc"),
        )
        self.hidden_norm = MuseSparkCenteredRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # Draft layers are numbered after the target's last layer so their
        # attention (KV cache) layer names never collide with the target's.
        self.layers = nn.ModuleList(
            [
                MuseSparkDSparkDecoderLayer(
                    config=config,
                    layer_idx=i,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, f"layers.{start_layer_id + i}"),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = MuseSparkCenteredRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            config.draft_vocab_size,
            config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        # The DSpark proposer reads this attribute; Muse Spark DSpark
        # checkpoints ship no confidence head, so adaptive verification
        # verifies every draft.
        self.confidence_head: DSparkConfidenceHead | None = None
        self.register_buffer(
            "d2t", torch.zeros(config.draft_vocab_size, dtype=torch.int64)
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None, (
            "embed_tokens must be set from target model"
        )
        embeds = self.embed_tokens(input_ids)
        if self.residual_fp32:
            return self.embed_norm._norm(embeds.float())
        return self.embed_norm(embeds)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            hidden_states = self.embed_input_ids(input_ids)
        else:
            hidden_states = inputs_embeds
        if self.residual_fp32:
            hidden_states = hidden_states.float()
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return hidden_states


class MuseSparkDSparkModel(nn.Module):
    """Muse Spark DSpark draft: a causal sliding-window decoder over fused
    target hidden states with a draft-vocab LM head and a low-rank Markov
    transition bias."""

    owns_context_kv = True
    has_own_embed_tokens = False
    has_own_lm_head = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        spec_config = vllm_config.speculative_config
        assert spec_config is not None, (
            "speculative_config is required for MuseSparkDSparkModel"
        )
        hf_config = spec_config.draft_model_config.hf_config
        if not isinstance(hf_config, MuseSparkDSparkConfig):
            hf_config = MuseSparkDSparkConfig(**hf_config.to_dict())
        self.config = hf_config
        self.vllm_config = vllm_config

        self.model = MuseSparkDSparkTextModel(
            config=self.config,
            vllm_config=vllm_config,
            start_layer_id=vllm_config.model_config.get_num_layers(
                vllm_config.parallel_config
            ),
            prefix=maybe_prefix(prefix, "model"),
        )
        draft_quant_config = _safe_draft_quant_config(vllm_config)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            bias=False,
            quant_config=draft_quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        # LogitsProcessor soft caps before scaling; Muse scales first, so the
        # processor stays raw and compute_draft_logits applies both in fp32.
        self.logits_processor = LogitsProcessor(self.config.draft_vocab_size)
        self.output_multiplier = float(self.config.final_logit_scale)
        soft_cap = self.config.final_logit_softcapping
        self.final_logit_softcapping = float(soft_cap) if soft_cap else None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(
        self,
        target_hidden: (torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]),
    ) -> torch.Tensor:
        if isinstance(target_hidden, (list, tuple)):
            target_hidden = (
                target_hidden[0]
                if len(target_hidden) == 1
                else torch.cat(target_hidden, dim=-1)
            )
        context = self.model.fc(target_hidden)
        if isinstance(context, tuple):
            context = context[0]
        return self.model.hidden_norm(context)

    def get_draft_attn_causal(self) -> list[bool]:
        return [True] * len(self.model.layers)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [
            cast(MuseSparkDSparkDecoderLayer, layer).self_attn.attn.layer_name
            for layer in self.model.layers
        ]

    def precompute_and_store_context_kv(
        self,
        target_hidden_states: torch.Tensor,
        positions: torch.Tensor,
        metadata: tuple,
    ) -> torch.Tensor:
        context = self.combine_hidden_states(target_hidden_states)
        for i, raw_layer in enumerate(self.model.layers):
            layer = cast(MuseSparkDSparkDecoderLayer, raw_layer)
            x = layer.input_layernorm(context)
            k = det_projection(layer.self_attn.k_proj, x)
            v = det_projection(layer.self_attn.v_proj, x)
            k = layer.self_attn.qk_norm(
                k.reshape(-1, layer.self_attn.head_dim)
            ).reshape(-1, layer.self_attn.kv_size)
            dummy_q = torch.zeros(
                (k.shape[0], layer.self_attn.num_heads, layer.self_attn.head_dim),
                dtype=k.dtype,
                device=k.device,
            )
            if layer.self_attn.rotary_emb is not None:
                dummy_q_f32, k_f32 = layer.self_attn.rotary_emb(
                    positions, dummy_q.float(), k.float()
                )
                dummy_q = dummy_q_f32.to(v.dtype)
                k = k_f32.to(v.dtype)

            attn_obj: Any = layer.self_attn.attn
            attn_obj.impl.forward(
                layer=attn_obj,
                query=dummy_q,
                key=k,
                value=v,
                kv_cache=attn_obj.kv_cache,
                attn_metadata=metadata[i],
            )
        return target_hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden = self.model.norm(hidden_states)
        logits = self.logits_processor(self.lm_head, hidden).float()
        logits = logits * self.output_multiplier
        cap = self.final_logit_softcapping
        if cap is not None:
            logits = cap * torch.tanh(logits / cap)
        return logits

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compute_draft_logits(hidden_states)

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        bias = self.model.markov_head.bias(markov_embed, self.logits_processor)
        return bias.float() * self.output_multiplier

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids + self.model.d2t[draft_ids]

    def process_weights_after_loading(self) -> None:
        for raw_layer in self.model.layers:
            layer = cast(MuseSparkDSparkDecoderLayer, raw_layer)
            layer.post_attention_residual_gate.process_weights_after_loading()
            layer.post_feedforward_residual_gate.process_weights_after_loading()
            attn = layer.self_attn
            if attn._muse_rope_permuted:
                continue
            # The DSpark export keeps interleaved RoPE pairs per head; the
            # target converter permutes to NEOX halves. Match the target here.
            for proj in (attn.q_proj, attn.k_proj):
                proj_w = proj.weight
                assert isinstance(proj_w, torch.Tensor)
                w = proj_w.data
                half = int(attn.head_dim) // 2
                perm = w.view(-1, half, 2, w.shape[-1]).transpose(1, 2)
                w.copy_(perm.reshape_as(w).contiguous())
            attn._muse_rope_permuted = True

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        def filtered_and_remapped_weights():
            for name, weight in weights:
                if "embed_tokens" in name:
                    continue
                new_name = _remap_dspark_name(name)
                if new_name is not None:
                    yield new_name, weight

        loaded = AutoWeightsLoader(self).load_weights(filtered_and_remapped_weights())
        self.process_weights_after_loading()
        return loaded
