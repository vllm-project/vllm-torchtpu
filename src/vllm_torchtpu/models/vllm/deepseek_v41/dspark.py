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
"""DSpark speculative draft model for DeepSeek-V4.1 on TPU."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import copy
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.deepseek_v41.nvidia import dspark as upstream_dspark
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

from .attention import DeepseekV41Attention
from .cache import write_cache
from .metadata import active_cache, cache_slots, token_requests
from .model import DeepseekV41DecoderLayer, DeepseekV41Model, collapse, mhc_post

ContextSlotMeta = Tensor | AttentionMetadata | None


def _insert_context_kv(
    attn: DeepseekV41Attention,
    kv_latent: Tensor,
    positions: Tensor,
    slot_meta: ContextSlotMeta,
) -> None:
    """Rotate and scatter trailing-window target context KV into ``attn``'s SWA cache.

    Args:
        attn: Draft attention layer owning ``swa_cache_layer`` (Changed in place).
        kv_latent: RMS-normalized context KV latents ``[num_tokens, head_dim]`` (Used).
        positions: Token positions ``[num_tokens]`` (Used).
        slot_meta: Precomputed ``int32`` slot tensor, ``AttentionMetadata`` for
            window-tail filtering, or ``None`` during warmup/profiling (Used).
    """
    swa_cache = active_cache(attn.swa_cache_layer)
    if slot_meta is None or not swa_cache.numel():
        return
    if isinstance(slot_meta, Tensor):
        slots = slot_meta
        valid = slots >= 0
    else:
        slots = cache_slots(swa_cache, slot_meta, positions)
        valid = slots >= 0
        seq_lens = cast(Tensor, slot_meta.seq_lens)
        if seq_lens.numel() > 0:
            pos_i32 = positions.int()
            seq_lens_i32 = seq_lens.int()
            req_indices = token_requests(slot_meta, pos_i32).clamp_max(
                seq_lens_i32.shape[0] - 1
            )
            window_offset = (
                pos_i32 - seq_lens_i32.index_select(0, req_indices) + attn.window_size
            )
            valid = valid & (window_offset >= 0) & (window_offset < attn.window_size)
    write_cache(swa_cache, slots, attn.rotate(kv_latent, positions), valid)


class DSparkDeepseekV41Model(upstream_dspark.DSparkDeepseekV4Model):
    """Three-layer block-SWA speculative draft backbone for DeepSeek-V4.1."""

    finalize_mhc_broadcast_weights = DeepseekV41Model.finalize_mhc_broadcast_weights
    mhc_post_op = DeepseekV41Model.mhc_post_op

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Initialize the DSpark three-layer draft backbone.

        Args:
            vllm_config: Global vLLM engine configuration.
            prefix: Model parameter prefix.

        Raises:
            ValueError: If the draft query block size exceeds the sliding window.
        """
        # TODO(upstream): vllm/models/deepseek_v41/nvidia/dspark.py:130,175 hooks.
        nn.Module.__init__(self)
        spec_config = vllm_config.speculative_config
        assert spec_config is not None and spec_config.draft_model_config is not None
        hf_config = spec_config.draft_model_config.hf_config
        quant_config = vllm_config.quant_config
        hidden_size: int = hf_config.hidden_size
        num_backbone_layers: int = hf_config.num_hidden_layers
        num_mtp_layers: int = hf_config.num_nextn_predict_layers

        self.config = hf_config
        self.hidden_size = hidden_size
        self.hc_mult: int = hf_config.hc_mult
        self.hc_eps: float = hf_config.hc_eps
        self.rms_norm_eps: float = hf_config.rms_norm_eps
        self.num_hidden_layers = num_backbone_layers
        self.start_layer = 0
        self.end_layer = num_mtp_layers
        self.target_layer_ids: tuple[int, ...] = tuple(
            hf_config.dspark_target_layer_ids
        )
        self.num_dspark_layers = num_mtp_layers
        # pyrefly: ignore[bad-override-mutable-attribute]
        self.embed_tokens: nn.Module | None = None

        # NOTE(cross-boundary-contract): Upstream SpeculativeConfig leaves target
        # n_routed_experts=384 on draft_model_config.hf_config; copy and override
        # n_routed_experts and num_experts_per_tok from dspark_* fields for draft
        # layer construction (paired with model_loader_patches.py:126-133).
        draft_hf = copy(hf_config)
        draft_model_config = copy(vllm_config.model_config)
        draft_vllm_config = copy(vllm_config)
        draft_hf.n_routed_experts = hf_config.dspark_n_routed_experts
        draft_hf.num_experts_per_tok = hf_config.dspark_num_experts_per_tok
        draft_model_config.hf_config = draft_hf
        draft_vllm_config.model_config = draft_model_config

        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        topk_buffer = torch.empty(max_tokens, hf_config.index_topk, dtype=torch.int32)
        self.topk_indices_buffer = topk_buffer

        with (
            set_default_torch_dtype(cast(torch.dtype, vllm_config.model_config.dtype)),
            set_current_vllm_config(draft_vllm_config),
        ):
            proj_in_dim = hidden_size * len(self.target_layer_ids)
            self.main_proj = ReplicatedLinear(
                proj_in_dim,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                return_bias=False,
                prefix=maybe_prefix(prefix, "main_proj"),
            )
            self.main_norm = RMSNorm(hidden_size, eps=hf_config.rms_norm_eps)
            layer_prefixes = [
                maybe_prefix(prefix, f"layers.{num_backbone_layers + idx}")
                for idx in range(num_mtp_layers)
            ]
            draft_layers = [
                DeepseekV41DecoderLayer(draft_vllm_config, layer_prefix, topk_buffer)
                for layer_prefix in layer_prefixes
            ]
            self.layers = nn.ModuleList(draft_layers)
            self.norm = RMSNorm(hidden_size, hf_config.rms_norm_eps)
            vocab_size: int = hf_config.vocab_size
            markov_rank: int = hf_config.dspark_markov_rank
            self.markov_head = DSparkMarkovHead(
                vocab_size,
                vocab_size,
                markov_rank,
                prefix=maybe_prefix(prefix, "markov_head"),
            )
            self.confidence_head: DSparkConfidenceHead | None = DSparkConfidenceHead(
                hidden_size + markov_rank,
                prefix=maybe_prefix(prefix, "confidence_head"),
            )

        num_spec = int(spec_config.num_speculative_tokens)
        query_block_size = int(vllm_config.num_lookahead_tokens)
        self.query_block_size = query_block_size
        for layer in cast(Iterable[DeepseekV41DecoderLayer], self.layers):
            if (
                query_block_size > layer.attn.window_size
                or num_spec + 1 > layer.attn.window_size
            ):
                raise ValueError("DSpark block exceeds sliding window")
            layer.use_sequence_parallel = False
            layer.attn.non_causal_multi_token_decode = True
            layer.attn.dspark_block_size = query_block_size

    # pyrefly: ignore[bad-override-param-name]
    def precompute_and_store_context_kv(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        kv_caches: Sequence[ContextSlotMeta] | None = None,
    ) -> Tensor:
        """Project fused target aux hidden states and populate draft SWA context KV.

        Args:
            hidden_states: Concatenated target auxiliary hidden states or
                already-projected ``[num_tokens, hidden_size]`` context hidden states.
            positions: Token positions ``int32[num_tokens]``.
            kv_caches: Optional per-layer slot metadata or ``AttentionMetadata``.

        Returns:
            Projected context hidden states ``[num_tokens, hidden_size]``.
        """
        needs_combine = hidden_states.shape[-1] != self.config.hidden_size
        main_hidden = (
            self.combine_hidden_states(hidden_states)
            if needs_combine
            else hidden_states
        )
        for idx, layer in enumerate(
            cast(Iterable[DeepseekV41DecoderLayer], self.layers)
        ):
            kv_latent = layer.attn.project_context_kv(main_hidden)
            slot_meta = None if kv_caches is None else kv_caches[idx]
            _insert_context_kv(layer.attn, kv_latent, positions, slot_meta)
        return main_hidden

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        """Run non-causal block-parallel draft layers and collapse mHC streams.

        Args:
            input_ids: Draft block token IDs ``int32[num_tokens]``.
            positions: Draft block token positions ``int32[num_tokens]``.
            inputs_embeds: Optional pre-computed input embeddings.

        Returns:
            Collapsed draft hidden states ``[num_tokens, hidden_size]``.
        """
        embeds = (
            self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        )
        first_layer = cast(DeepseekV41DecoderLayer, self.layers[0])
        has_broadcast = first_layer.hc_attn_fn_broadcast is not None
        hidden = (
            embeds if has_broadcast else embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        )
        residual: Tensor | None = None
        post_mix: Tensor | None = None
        res_mix: Tensor | None = None
        pre_mix: Tensor | None = None
        for raw_layer in self.layers:
            layer = cast(DeepseekV41DecoderLayer, raw_layer)
            hidden, residual, post_mix, res_mix, pre_mix, _ = layer(
                hidden, positions, input_ids, pre_mix, post_mix, res_mix, residual
            )
        assert (
            residual is not None
            and post_mix is not None
            and res_mix is not None
            and pre_mix is not None
        )
        return collapse(mhc_post(hidden, residual, post_mix, res_mix), pre_mix)


class DSparkDeepseekV41ForCausalLM(upstream_dspark.DSparkDeepseekV4ForCausalLM):
    """Registry entry for ``architectures=["DSparkV41DraftModel"]`` on TPU."""

    requires_dp_collective_pairing = True
    owns_context_kv = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Initialize the DSpark draft causal language model.

        Args:
            vllm_config: Global vLLM engine configuration.
            prefix: Model parameter prefix.
        """
        # TODO(upstream): vllm/models/deepseek_v41/nvidia/dspark.py:303 model_cls.
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        draft_model_config = vllm_config.speculative_config.draft_model_config
        assert draft_model_config is not None
        self.draft_model_config = draft_model_config
        self.config = draft_model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.linear_scale_name = "weight_scale_inv"
        self.pad_shared_expert = False
        # pyrefly: ignore[bad-override-mutable-attribute]
        self.model: DSparkDeepseekV41Model = DSparkDeepseekV41Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # pyrefly: ignore[bad-override-mutable-attribute]
        self.lm_head: nn.Module | None = None
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def get_draft_attn_causal(self) -> list[bool]:
        """Declare non-causal block-parallel attention across all draft layers.

        Returns:
            List of ``False`` flags with length equal to ``len(self.model.layers)``.
        """
        return [False] * len(self.model.layers)

    # pyrefly: ignore[bad-override,bad-override-param-name]
    def precompute_and_store_context_kv(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        kv_caches: Sequence[ContextSlotMeta] | None = None,
    ) -> Tensor:
        """Delegate context KV projection and SWA scatter to ``self.model``.

        Args:
            hidden_states: Target auxiliary hidden states.
            positions: Token positions ``int32[num_tokens]``.
            kv_caches: Optional per-layer slot metadata or ``AttentionMetadata``.

        Returns:
            Projected context hidden states ``[num_tokens, hidden_size]``.
        """
        return self.model.precompute_and_store_context_kv(
            hidden_states, positions, kv_caches
        )

    def map_draft_to_target(self, draft_ids: Tensor) -> Tensor:
        """Return full-vocab draft token IDs cast to ``int32``.

        Args:
            draft_ids: Sampled draft token IDs.

        Returns:
            ``int32`` target-vocab token IDs.
        """
        return draft_ids.to(torch.int32)

    def _remap_dspark_name(self, name: str) -> str | None:
        """Skip vision-language router biases before upstream MTP name remapping.

        Args:
            name: Raw checkpoint tensor name.

        Returns:
            Remapped parameter name, or ``None`` if the weight should be skipped.
        """
        if name.endswith("ffn.gate.bias_vl"):
            return None
        return super()._remap_dspark_name(name)

    def process_weights_after_loading(self) -> None:
        """Precompute first-layer broadcast hyper-connection weights."""
        # pyrefly: ignore[bad-argument-type]
        self.model.finalize_mhc_broadcast_weights()

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        """Load MTP draft checkpoint weights under ``torch.no_grad()``.

        Args:
            weights: Iterable of ``(name, tensor)`` checkpoint pairs.

        Returns:
            Set of loaded parameter names.
        """
        with torch.no_grad():
            return super().load_weights(weights)
