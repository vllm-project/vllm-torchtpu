# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPU implementation of the dense MLA Kimi-K3 DSpark drafter."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    maybe_prefix,
)

from vllm_torchtpu.layers.core.quantization import quantize_kv

from .attention import MultiHeadLatentAttention
from .layers import KimiMLP


def _duplicate_context_kv_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    num_layers: int,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Also load every layer's KV-A projection into one fused projection."""
    for name, weight in weights:
        yield name, weight
        layer_prefix, marker, param_name = name.partition(
            ".self_attn.kv_a_proj_with_mqa."
        )
        if not marker:
            continue
        layer_idx_text = layer_prefix.rsplit(".", 1)[-1]
        if not layer_idx_text.isdecimal():
            continue
        layer_idx = int(layer_idx_text)
        if layer_idx >= num_layers:
            continue
        duplicate = weight.detach()
        duplicate.shard_id = layer_idx
        yield f"context_kv_proj.{param_name}", duplicate


class K3DSparkDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        config,
        vllm_config: VllmConfig,
        layer_idx: int,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        layer_prefix = maybe_prefix(prefix, f"layers.{start_layer_id + layer_idx}")
        self.self_attn = MultiHeadLatentAttention(
            config,
            vllm_config,
            prefix=f"{layer_prefix}.self_attn",
            use_rope=True,
            non_causal_multi_token_decode=True,
        )
        quant_config = get_draft_quant_config(vllm_config)
        self.mlp = KimiMLP(
            config.hidden_size,
            config.intermediate_size,
            getattr(config, "hidden_act", "silu"),
            quant_config=quant_config,
            prefix=f"{layer_prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class K3DSparkModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        quant_config = get_draft_quant_config(vllm_config)

        # Aliased to the target embedding after the draft checkpoint loads.
        self.embed_tokens: nn.Module | None = None
        self.context_proj = ReplicatedLinear(
            self.config.target_hidden_size * self.config.num_target_layers,
            self.config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "context_proj"),
        )
        self.context_norm = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [
                K3DSparkDecoderLayer(
                    config=self.config,
                    vllm_config=vllm_config,
                    layer_idx=layer_idx,
                    start_layer_id=start_layer_id,
                    prefix=prefix,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )

        kv_width = self.config.kv_lora_rank + self.config.qk_rope_head_dim
        self.context_kv_proj = MergedColumnParallelLinear(
            self.config.hidden_size,
            [kv_width] * self.config.num_hidden_layers,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "context_kv_proj"),
            disable_tp=True,
        )
        self.final_norm = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(
            self.config.vocab_size,
            self.config.draft_vocab_size,
            self.config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.context_proj(hidden_states)
        return self.context_norm(hidden_states)

    def get_draft_attn_causal(self) -> list[bool]:
        return [False] * len(self.layers)

    def precompute_and_store_context_kv(
        self,
        target_hidden_states: torch.Tensor,
        positions: torch.Tensor,
        metadata: tuple,
    ) -> torch.Tensor:
        """Project target aux states and insert five latent context caches."""
        context = self.combine_hidden_states(target_hidden_states)
        all_kv, _ = self.context_kv_proj(context)
        num_tokens = context.shape[0]
        num_layers = len(self.layers)
        kv_width = self.config.kv_lora_rank + self.config.qk_rope_head_dim
        all_kv = all_kv.view(num_tokens, num_layers, kv_width)

        for layer_idx, layer in enumerate(self.layers):
            wrapper = layer.self_attn.mla_attn
            core = wrapper.mla_attn
            kv_lora = all_kv[:, layer_idx]
            kv_c, k_pe = kv_lora.split(
                [self.config.kv_lora_rank, self.config.qk_rope_head_dim],
                dim=-1,
            )
            kv_c = wrapper.kv_a_layernorm(kv_c)
            k_pe = k_pe.unsqueeze(1)
            assert wrapper.rotary_emb is not None
            dummy_q_pe = torch.zeros(
                (num_tokens, wrapper.num_heads, self.config.qk_rope_head_dim),
                dtype=context.dtype,
                device=context.device,
            )
            _, k_pe = wrapper.rotary_emb(positions, dummy_q_pe, k_pe)
            # This context path bypasses the regular MLA forward, so mirror
            # its cache quantization here.  RoPE may also promote k_pe to
            # fp32; both latent components must reach the cache-update kernel
            # in the cache dtype and with the layer's KV scale applied.
            cache_dtype = getattr(core, "kv_cache_quantized_dtype", None)
            if cache_dtype is not None:
                k_scale = getattr(core, "_k_scale_float", None) or 1.0
                kv_c, _ = quantize_kv(cache_dtype, kv_c, value=None, k_scale=k_scale)
                k_pe, _ = quantize_kv(cache_dtype, k_pe, value=None, k_scale=k_scale)
            else:
                kv_c = kv_c.to(dtype=core.kv_cache.dtype)
                k_pe = k_pe.to(dtype=core.kv_cache.dtype)
            ql_nope = torch.zeros(
                (num_tokens, wrapper.num_heads, self.config.kv_lora_rank),
                dtype=context.dtype,
                device=context.device,
            )
            core.mla_op(
                core.kv_cache,
                ql_nope,
                dummy_q_pe,
                kv_c,
                k_pe.squeeze(1),
                metadata[layer_idx].seq_lens,
                metadata[layer_idx].block_tables,
                metadata[layer_idx].query_start_loc,
                metadata[layer_idx].request_distribution,
            )
        return context

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        )
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.final_norm(hidden_states)


class K3DSparkForCausalLM(nn.Module):
    """vLLM model-registry entry used by ``architectures=K3DSparkModel``."""

    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"": "model."},
        orig_to_new_stacked={
            ".gate_proj.": (".gate_up_proj.", 0),
            ".up_proj.": (".gate_up_proj.", 1),
            ".q_a_proj.": (".fused_qkv_a_proj.", 0),
            ".kv_a_proj_with_mqa.": (".fused_qkv_a_proj.", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        target_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = K3DSparkModel(
            vllm_config=vllm_config,
            start_layer_id=target_layers,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head: nn.Module | None = None
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size,
            scale=getattr(self.config, "logit_scale", 1.0),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def get_draft_attn_causal(self) -> list[bool]:
        return self.model.get_draft_attn_causal()

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [
            layer.self_attn.mla_attn.mla_attn.layer_name for layer in self.model.layers
        ]

    def tpu_precompute_and_store_context_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        metadata: tuple,
    ) -> torch.Tensor:
        return self.model.precompute_and_store_context_kv(
            hidden_states, positions, metadata
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.lm_head is not None
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compute_draft_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        def filtered_weights():
            for name, weight in weights:
                if any(
                    part in name
                    for part in ("confidence_head", "embed_tokens", "lm_head")
                ):
                    continue
                yield name, weight

        duplicated = _duplicate_context_kv_weights(
            filtered_weights(), len(self.model.layers)
        )
        mapped = self.hf_to_vllm_mapper.apply(duplicated)
        return AutoWeightsLoader(self).load_weights(mapped)
