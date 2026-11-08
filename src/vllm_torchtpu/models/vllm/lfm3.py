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
"""LFM3 (``Lfm3ForCausalLM``, e.g. LiquidAI/LFM3-1.7B) on TPU.

LFM2's hybrid of gated short convolutions and GQA, plus: sliding-window
attention layers with their own RoPE, zero-tail partial RoPE on the
full-attention layers, a per-head sigmoid output gate, sandwich norms on both
block outputs, and a width-7 convolution. The convolution and its
per-sequence state run in the XLA ragged scan shared with Kimi KDA.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import islice
from typing import Any, cast

import jax
import torch
from torch import nn
from torch_tpu._internal import pallas
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    make_layers,
    maybe_prefix,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_torchtpu.layers.adapter.custom_ops.kda_attention_op import (
    kimi_short_conv_scan,
)
from vllm_torchtpu.layers.adapter.rotary_embedding import TpuRotaryEmbedding
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.models.vllm.kimi_k3.attention import CausalDepthwiseConv1d
from vllm_torchtpu.platforms.tpu_block_size_utils import unified_kv_layout_enabled
from vllm_torchtpu.utils import fresh_buffer


class Lfm3ZeroTailRotaryEmbedding(TpuRotaryEmbedding):
    """FP32 neox RoPE over the whole head with LFM3's inverse frequencies.

    HF LFM3 keeps the full-head progression ``theta^(-2i/head_size)`` for the
    first ``partial_rotary_factor * head_size / 2`` frequencies and zeroes the
    rest, so those dims stay unrotated. vLLM's ``partial_rotary_factor``
    instead rotates a prefix with ``theta^(-2i/rotary_dim)``, a different map.
    """

    def __init__(
        self,
        head_size: int,
        max_position: int,
        base: float,
        partial_rotary_factor: float,
        dtype: torch.dtype,
    ) -> None:
        self.partial_rotary_factor = partial_rotary_factor
        super().__init__(head_size, head_size, max_position, base, True, dtype)

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        live = int(self.partial_rotary_factor * self.head_size) // 2
        exponents = torch.arange(0, 2 * live, 2, dtype=torch.float) / self.head_size
        inv_freq = 1.0 / (base**exponents)
        return torch.cat([inv_freq, inv_freq.new_zeros(self.head_size // 2 - live)])


# One instance per (theta, factor): each holds a max_position x head_size
# cos/sin table (64 MiB in bf16 at 256K), so layers of a kind share it.
_ROPE_CACHE: dict[tuple, Lfm3ZeroTailRotaryEmbedding] = {}


def _get_rope(
    head_size: int, max_position: int, rope: dict[str, Any], dtype: torch.dtype
) -> Lfm3ZeroTailRotaryEmbedding:
    if rope.get("rope_type", "default") != "default":
        raise NotImplementedError(f"LFM3 on TPU supports default RoPE only: {rope}")
    factor = float(rope.get("partial_rotary_factor", 1.0))
    key = (head_size, max_position, float(rope["rope_theta"]), factor, dtype)
    if key not in _ROPE_CACHE:
        _ROPE_CACHE[key] = Lfm3ZeroTailRotaryEmbedding(
            head_size, max_position, float(rope["rope_theta"]), factor, dtype
        )
    return _ROPE_CACHE[key]


def _short_conv_core(
    bx: jax.Array,
    conv_state: jax.Array,
    conv_weight: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    seq_lens: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Ragged causal depthwise conv; returns (output, updated state pool)."""
    return kimi_short_conv_scan(
        bx, conv_state, conv_weight, query_start_loc, state_indices, seq_lens
    )


# Registered at import, never from forward: the backbone forward is traced
# fullgraph by Dynamo, which cannot trace torch.library registration. One op
# serves every conv layer (same graph, one compile).
_SHORT_CONV_OP = pallas.jax_op(
    "pallas::lfm3_short_conv", _short_conv_core, donate_argnums=(1,)
)


@_SHORT_CONV_OP.register_fake
def _short_conv_fake(bx, conv_state, *args, **kwargs):
    return torch.empty_like(bx), torch.empty_like(conv_state)


class Lfm3ShortConv(nn.Module, MambaBase):
    """Gated short convolution ``out_proj(C * conv(B * x))`` with TPU state.

    The state of each request is its last ``K - 1`` values of ``B * x``,
    oldest first: one ``(K - 1, dim)`` slot per request, addressed by
    ``AttentionMetadata.mamba_state_indices``.
    """

    supports_dcp = False

    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "LFM3 short conv on TPU has no spec-decode window"
            )
        if unified_kv_layout_enabled(vllm_config):
            # One state tensor per layer is assumed below; the unified pool
            # would hand this layer an attention-shaped buffer instead.
            raise NotImplementedError("LFM3 on TPU needs the per-layer KV layout")
        self.prefix = prefix
        self.vllm_config = vllm_config
        dim = config.hidden_size
        self.kernel_size = config.conv_L_cache
        self.local_dim = dim // get_tensor_model_parallel_world_size()
        self.in_proj = MergedColumnParallelLinear(
            dim,
            [dim] * 3,
            bias=config.conv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )
        self.out_proj = RowParallelLinear(
            dim,
            dim,
            bias=config.conv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        if config.conv_bias:
            raise NotImplementedError("LFM3 short conv on TPU assumes conv_bias=False")
        self.conv = CausalDepthwiseConv1d(dim, self.kernel_size)
        # The scan takes taps-first [K, dim]; built once in
        # `process_weights_after_loading`, not a checkpoint entry.
        self.register_buffer(
            "conv_weight_t",
            torch.empty(self.kernel_size, self.local_dim),
            persistent=False,
        )
        self.kv_cache: tuple[torch.Tensor, ...] | None = None
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return Lfm3ForCausalLM.get_mamba_state_dtype_from_config(self.vllm_config)

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return Lfm3ForCausalLM.get_mamba_state_shape_from_config(self.vllm_config)

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        # fresh_buffer: a transposed view would be re-materialized every step.
        self.conv_weight_t = fresh_buffer(self.conv.weight[:, 0, :].t())

    def _metadata(self) -> AttentionMetadata | None:
        if not is_forward_context_available():
            return None
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return None
        return cast(AttentionMetadata | None, metadata.get(self.prefix))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bcx, _ = self.in_proj(hidden_states)
        b, c, x = bcx.chunk(3, dim=-1)
        bx = b * x
        metadata = self._metadata()
        if metadata is None or self.kv_cache is None or self.kv_cache[0].numel() == 0:
            # Profile / dummy run without a state pool: shapes only.
            conv_out = bx
        else:
            if metadata.mamba_state_indices is None:
                raise RuntimeError("LFM3 short conv needs mamba_state_indices")
            conv_state = self.kv_cache[0]
            conv_out, new_state = _SHORT_CONV_OP(
                bx,
                conv_state,
                self.conv_weight_t,
                cast(torch.Tensor, metadata.query_start_loc),
                cast(torch.Tensor, metadata.mamba_state_indices),
                cast(torch.Tensor, metadata.seq_lens),
            )
            conv_state.copy_(new_state)
        out, _ = self.out_proj(c * conv_out)
        return out


class Lfm3Attention(nn.Module):
    """GQA with QK RMSNorm, per-layer-type RoPE and window, and a sigmoid
    output gate per query head."""

    def __init__(
        self,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        # One gate logit per query head, sharded with the heads.
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        layer_type = config.layer_types[layer_idx]
        self.rotary_emb = _get_rope(
            self.head_dim,
            config.max_position_embeddings,
            config.rope_parameters[layer_type],
            torch.get_default_dtype(),
        )
        # HF window semantics (key kept iff k > q - W) match RPA's q < k + W.
        sliding_window = (
            config.sliding_window if layer_type == "sliding_attention" else None
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        n = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(n, self.num_heads, self.head_dim)).view(n, self.q_size)
        k = self.k_norm(k.view(n, self.num_kv_heads, self.head_dim)).view(
            n, self.kv_size
        )
        q, k = self.rotary_emb(positions, q, k)
        attn_out = self.attn(q, k, v)
        gate, _ = self.gate_proj(hidden_states)
        attn_out = attn_out.view(n, self.num_heads, self.head_dim) * torch.sigmoid(
            gate
        ).unsqueeze(-1)
        out, _ = self.o_proj(attn_out.view(n, self.q_size))
        return out


class Lfm3MLP(nn.Module):
    def __init__(self, config, quant_config: QuantizationConfig | None, prefix: str):
        super().__init__()
        self.w13 = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w13",
        )
        self.w2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w2",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.w2(self.act_fn(self.w13(x)[0]))
        return x


class Lfm3DecoderLayer(nn.Module):
    """``h += post_op_norm(op(op_norm(h))); h += post_ffn_norm(mlp(ffn_norm(h)))``.

    The residual rides alongside as in vLLM's fused add-RMSNorm: each layer
    returns its post-normed FFN output and the residual stream before it.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        layer_idx = extract_layer_index(prefix)
        self.is_attention = config.layer_types[layer_idx] != "conv"
        if not getattr(config, "use_sandwich_norm", True):
            raise NotImplementedError("LFM3 on TPU assumes use_sandwich_norm=True")
        if self.is_attention:
            self.self_attn = Lfm3Attention(
                config,
                layer_idx,
                vllm_config.cache_config,
                quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.short_conv = Lfm3ShortConv(
                config, vllm_config, quant_config, prefix=f"{prefix}.short_conv"
            )
        self.feed_forward = Lfm3MLP(
            config, quant_config, prefix=f"{prefix}.feed_forward"
        )
        eps = config.norm_eps
        self.operator_norm = RMSNorm(config.hidden_size, eps=eps)
        self.post_operator_norm = RMSNorm(config.hidden_size, eps=eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=eps)
        self.post_ffn_norm = RMSNorm(config.hidden_size, eps=eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.operator_norm(hidden_states)
        else:
            hidden_states, residual = self.operator_norm(hidden_states, residual)
        if self.is_attention:
            hidden_states = self.self_attn(positions, hidden_states)
        else:
            hidden_states = self.short_conv(hidden_states)
        hidden_states = self.post_operator_norm(hidden_states)
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        return self.post_ffn_norm(self.feed_forward(hidden_states)), residual


@support_torch_compile(
    dynamic_arg_dims={"input_ids": 0, "positions": 0, "inputs_embeds": 0}
)
class Lfm3Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Lfm3DecoderLayer(vllm_config, prefix),
            prefix=f"{prefix}.layers",
        )
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_input_ids(input_ids)
        )
        residual = None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.embedding_norm(hidden_states, residual)
        return hidden_states


class Lfm3ForCausalLM(nn.Module, HasInnerState, IsHybrid):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "w13": ["w1", "w3"],
        "in_proj": ["in_proj"],
    }
    # HF `.conv.` holds a child `.conv`; vLLM names the layer `.short_conv.`
    # (as upstream LFM2 does) so the two never alias.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={".conv.": ".short_conv."},
        orig_to_new_stacked={
            ".q_proj.": (".qkv_proj.", "q"),
            ".k_proj.": (".qkv_proj.", "k"),
            ".v_proj.": (".qkv_proj.", "v"),
            ".w1.": (".w13.", 0),
            ".w3.": (".w13.", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.model = Lfm3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...], ...]:
        # Taps-first (K - 1, dim) per slot: the layout the ragged scan reads,
        # independent of vLLM's conv-state layout switch.
        config = vllm_config.model_config.hf_text_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        return ((config.conv_L_cache - 1, config.hidden_size // tp_size),)

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, ...]:
        return MambaStateCopyFuncCalculator.short_conv_state_copy_func()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: object | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise ValueError("LFM3 on TPU does not support pipeline parallelism")
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The checkpoint has no lm_head tensor when tied; AutoWeightsLoader
        # resolves the tied alias itself.
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
