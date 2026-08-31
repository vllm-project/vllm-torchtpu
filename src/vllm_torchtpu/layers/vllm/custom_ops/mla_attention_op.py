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
"""Out-of-tree (OOT) custom operator layers and wrappers for Multi-Head Latent Attention (MLA)."""

from typing import Any

import jax
import jax.numpy as jnp
import torch
from torch.nn import Parameter
from torch_tpu._internal import pallas
from vllm.config import CacheConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import mla_attention
from vllm.model_executor.layers.attention.attention import \
    get_attention_context
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.mla import (MLAModules,
                                            MultiHeadLatentAttentionWrapper)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.common import (rotate_gptj,
                                                                rotate_neox)
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import \
    DeepseekScalingRotaryEmbedding
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.models.deepseek_v2 import (DeepseekV32IndexerCache,
                                                    Indexer)
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.v1.attention.backends.mla.prefill import selector

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk
from vllm_torchtpu.layers.common.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.common.quantization import quantize_tensor
from vllm_torchtpu.layers.vllm.attention import (
    TPU_STR_DTYPE_TO_TORCH_DTYPE, VllmTPUDeepseekV32IndexerBackend)
from vllm_torchtpu.layers.vllm.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.utils import synchronize_tensors

# Skip indexer scoring when every sequence in the batch is shorter than
# `index_topk`, where the top-k is the identity and the scores decide nothing.
# The guard is batch-wide, so this fires during warmup and on short-context
# traffic and not at all once any request in the batch is past `index_topk`
# tokens.
STREAMIDX_ENABLE_EARLY_EXIT = True


@SparseAttnIndexer.register_oot
class VllmTPUSparseAttnIndexer(SparseAttnIndexer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build eagerly: pallas.jax_op runs inspect.signature(), which Dynamo
        # cannot trace if the op is first built inside a compiled forward.
        self.topk_op = self._build_streamidx_op()

    def _build_streamidx_op(self):
        topk = self.topk_tokens

        def _streamidx_topk_jax(
                q_bytes: jax.Array,  # uint8 view of fp8 q [num_tokens, H, D]
                weights: jax.Array,  # [num_tokens, H]
                cache_kv: jax.
            Array,  # uint8 [num_blocks, block_size_per_kv_packing, kv_packing, lkv_dim] where lkv_dim = D + 1 padded to a 128 multiple
                k_packed: jax.
            Array,  # uint8 [num_tokens, D + 1] fp8 k + e8m0 scale
                seq_lens: jax.Array,  # i32 [num_seqs]
                page_indices: jax.Array,  # i32 [num_seqs * pages_per_seq]
                cu_q_lens: jax.Array,  # i32 [num_seqs + 1]
                distribution: jax.Array,  # i32 [3]
        ) -> tuple[jax.Array, jax.Array]:
            _, block_size_per_kv_packing, kv_packing, lkv_dim = cache_kv.shape
            block_size = block_size_per_kv_packing * kv_packing
            # Cache rows are padded to the 128-lane HBM width; pad the packed
            # fp8+scale rows to match before scattering.
            k_packed = jnp.pad(k_packed,
                               ((0, 0), (0, lkv_dim - k_packed.shape[-1])))

            # Insert this step's K rows into the KV cache.
            tok = jnp.arange(k_packed.shape[0], dtype=jnp.int32)
            seq_id = jnp.searchsorted(cu_q_lens[1:], tok,
                                      side="right").astype(jnp.int32)
            q_len = cu_q_lens[seq_id + 1] - cu_q_lens[seq_id]
            pos = seq_lens[seq_id] - q_len + (tok - cu_q_lens[seq_id])
            page = jnp.where(
                tok < cu_q_lens[-1],
                page_indices.reshape(seq_lens.shape[0], -1)[seq_id,
                                                            pos // block_size],
                jnp.int32(2**30))
            slot = pos % block_size
            cache_kv = cache_kv.at[page, slot // kv_packing,
                                   slot % kv_packing].set(k_packed)

            q = jax.lax.bitcast_convert_type(q_bytes, jnp.float8_e4m3fn)
            topk_indices = streamindex_topk(
                q,
                weights,
                cache_kv,
                seq_lens,
                page_indices,
                cu_q_lens,
                distribution,
                k=topk,
                compression_ratio=1,
                # The following parameters are tuned based on microbenchmark
                # results.
                num_kv_pages_per_block=(2, 2, 2),
                num_queries_per_block=(1, 128, 128),
                enable_early_exit=STREAMIDX_ENABLE_EARLY_EXIT,
            )
            return cache_kv, topk_indices

        op_name = f"pallas::streamidx_topk_{self.k_cache.prefix.replace('.', '_')}"
        jax_op = pallas.jax_op(op_name,
                               _streamidx_topk_jax,
                               donate_argnums=(2, ))

        def _fake(q_bytes, weights, cache_kv, *args, **kwargs):
            return torch.empty_like(cache_kv), torch.empty(
                (q_bytes.shape[0], topk),
                dtype=torch.int32,
                device=q_bytes.device)

        jax_op.register_fake(_fake)

        def op(*args):
            new_cache, topk_indices = jax_op(*args)
            args[2].copy_(new_cache)
            return topk_indices

        self._streamidx_op = op
        return op

    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        q_values: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        assert q_values.dtype == torch.float8_e4m3fn, (
            f"streamindex_topk expects fp8 q, got {q_values.dtype}")

        k_quant, k_scale = quantize_tensor(
            k,
            torch.float8_e4m3fn,
            axis=-1,
            use_ue8m0=True,
        )
        # Pack per-token fp8 values + 1-byte e8m0 scale into the uint8 row
        # layout the TPU kernel unpacks (head_dim value bytes + 1 scale byte).
        # cat requires a single dtype, so view the fp8 values as raw bytes.
        k_quant_packed = torch.cat([k_quant.view(torch.uint8), k_scale],
                                   dim=-1)

        attn_metadata = get_forward_context().attn_metadata
        metadata = None
        if isinstance(attn_metadata, dict):
            metadata = attn_metadata.get(self.k_cache.prefix)
        if not isinstance(metadata, AttentionMetadata):
            return self.topk_indices_buffer

        kv_cache = self.k_cache.kv_cache  # uint8 [num_blocks, block_size_per_kv_packing, kv_packing, lkv_dim], lkv_dim = head_dim + 1 padded to a 128 multiple
        if kv_cache.numel() == 0:
            return self.topk_indices_buffer

        seq_lens = metadata.seq_lens
        page_indices = metadata.block_tables
        cu_q_lens = metadata.query_start_loc
        rd = metadata.request_distribution
        distribution = torch.stack([rd[0], rd[0], rd[2]])

        topk_indices = self.topk_op(
            q_values.view(torch.uint8),
            weights,
            kv_cache,
            k_quant_packed,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )

        self.topk_indices_buffer[:hidden_states.shape[0]] = -1
        num_out = min(topk_indices.shape[0], self.topk_indices_buffer.shape[0])
        self.topk_indices_buffer[:num_out, :self.
                                 topk_tokens] = topk_indices[:num_out]
        return self.topk_indices_buffer


class VllmTPUIndexerCache(DeepseekV32IndexerCache):
    """Indexer K cache retyped onto the TPU backend. See `VllmTPUIndexer`."""

    def get_attn_backend(self) -> type[DeepseekV32IndexerBackend]:
        return VllmTPUDeepseekV32IndexerBackend


class VllmTPUIndexer(Indexer):
    """TPU-optimized out-of-tree wrapper for Indexer.

    Never constructed directly. `Indexer` is neither a `CustomOp` nor a
    `PluggableLayer`, so it has no `register_oot` hook, and
    `DeepseekV2MLAAttention` builds it before this plugin sees it. `rebind`
    retypes the finished instance in place: state is preserved, method lookup
    moves to this class.
    """

    @classmethod
    def rebind(cls, indexer: Indexer) -> "VllmTPUIndexer":
        """Retype an in-tree `Indexer`, and its K cache, onto the TPU path."""
        if isinstance(indexer, cls):
            return indexer
        assert indexer.quant_block_size == indexer.head_dim, (
            "streamindex_topk requires quant_block_size == head_dim, got "
            f"{indexer.quant_block_size} != {indexer.head_dim}")
        indexer.__class__ = cls
        indexer.k_cache.__class__ = VllmTPUIndexerCache
        # ue8m0 keeps one scale byte per quant block; upstream sizes the row for
        # a 4-byte fp32 scale. Resize in place rather than rebuilding: the cache
        # is already registered in `static_forward_context` under its prefix and
        # a second instance would be rejected as a duplicate. `get_kv_cache_spec`
        # reads `head_dim` lazily, so this lands before the runner collects specs.
        indexer.k_cache.head_dim = (
            indexer.head_dim + indexer.head_dim // indexer.quant_block_size)
        return indexer

    def forward(self, hidden_states: torch.Tensor, qr: torch.Tensor, positions,
                rotary_emb) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        # Fused wk + weights_proj: one GEMM, then split
        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, :self.head_dim]
        weights = kw[:, self.head_dim:]

        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # Note: RoPE (NeoX) can introduce extra leading dimensions during
        # compilation so we need to reshape back to token-flattened shapes
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        # `rotary_emb` is shape-preserving; `q_pe` is already
        # [num_tokens, n_head, rope_dim].
        q = torch.cat([q_pe, q_nope], dim=-1)
        # `k_pe` is [num_tokens, 1, rope_dim] (MQA).
        k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)

        # we only quant q here since k quant is fused with cache insertion
        q = q.view(-1, self.head_dim)
        q_fp8, q_scale = quantize_tensor(q,
                                         torch.float8_e4m3fn,
                                         axis=-1,
                                         use_ue8m0=False)
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)

        # q_scale is fp32, so fold it in and cast back: the kernel expects the
        # weights in the activation dtype.
        weights_dtype = weights.dtype
        weights = (weights.unsqueeze(-1) * q_scale * self.softmax_scale *
                   self.n_head**-0.5)
        weights = weights.squeeze(-1).to(weights_dtype)

        return self.indexer_op(hidden_states, q_fp8, k, weights)


def _fresh(t: torch.Tensor) -> torch.Tensor:
    """Allocate a fresh contiguous device buffer (breaks view/stride chains so
    the torch_tpu pallas boundary ships a plain row-major buffer)."""
    return torch.empty(t.shape, dtype=t.dtype, device=t.device).copy_(t)


class TPUDummyMLAPrefillBackend:

    def __init__(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


class VllmTPUMLAAttention(MLAAttention):
    """TPU-optimized out-of-tree wrapper for Multi-Head Latent Attention."""

    def __init__(
        self,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        kv_b_proj: ColumnParallelLinear,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_backend: Any | None = None,
        use_sparse: bool = False,
        indexer: object | None = None,
        **extra_impl_args,
    ):
        torch.nn.Module.__init__(self)

        original_mla_get_backend = getattr(mla_attention,
                                           "get_mla_prefill_backend", None)
        original_selector_get_backend = getattr(selector,
                                                "get_mla_prefill_backend",
                                                None)

        if original_mla_get_backend is not None:
            mla_attention.get_mla_prefill_backend = lambda config: TPUDummyMLAPrefillBackend
        if original_selector_get_backend is not None:
            selector.get_mla_prefill_backend = lambda config: TPUDummyMLAPrefillBackend

        try:
            # Keyword-only: vLLM inserts new positional params into
            # MLAAttention.__init__ (e.g. dcp_q_replicate in v0.26.1rc0),
            # which silently shifts positional args into the wrong slots.
            super().__init__(num_heads=num_heads,
                             scale=scale,
                             qk_nope_head_dim=qk_nope_head_dim,
                             qk_rope_head_dim=qk_rope_head_dim,
                             v_head_dim=v_head_dim,
                             q_lora_rank=q_lora_rank,
                             kv_lora_rank=kv_lora_rank,
                             kv_b_proj=kv_b_proj,
                             cache_config=cache_config,
                             quant_config=quant_config,
                             prefix=prefix,
                             attn_backend=attn_backend,
                             use_sparse=use_sparse,
                             indexer=indexer,
                             **extra_impl_args)
        finally:
            if original_mla_get_backend is not None:
                mla_attention.get_mla_prefill_backend = original_mla_get_backend
            if original_selector_get_backend is not None:
                selector.get_mla_prefill_backend = original_selector_get_backend

        # For compatibility reasons.
        self.kv_sharing_target_layer_name = None
        self.attn_type = AttentionType.DECODER
        self.sliding_window = None

        self.kv_cache_quantized_dtype = None
        if self.kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE.get(
                self.kv_cache_dtype.lower().strip())

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # Upstream `MLAAttention.process_weights_after_loading` reads this
        # weight raw -- `get_and_maybe_dequant_weights(kv_b_proj).T` -- and
        # asserts the [n_out, n_in] shape. The TPU linear method has already
        # flipped it to (k, n), so present the layout upstream expects for the
        # duration of that call. `kv_b_proj`'s parameters are cleared a few
        # lines below once `W_UK_T`/`W_UV` are extracted, and it never runs a
        # forward pass, so this view is the only consumer either way.
        kv_b_weight = getattr(self.kv_b_proj, "weight", None)
        # The weight's own type says whether it was flipped, so no bookkeeping
        # flag is needed. Upstream below wants the [n_out, n_in] view.
        flipped = (kv_b_weight is not None
                   and getattr(self.kv_b_proj, WEIGHT_FLIPPED_ATTR, False))
        if flipped:
            self.kv_b_proj.weight = Parameter(kv_b_weight.transpose(
                0, 1).contiguous(),
                                              requires_grad=False)
        try:
            super().process_weights_after_loading(act_dtype)
        finally:
            if flipped and getattr(self.kv_b_proj, "weight", None) is not None:
                self.kv_b_proj.weight = Parameter(kv_b_weight,
                                                  requires_grad=False)

        device = torch.device("tpu")

        # move W_UK_T and W_UV into fresh contiguous buffers
        self.W_UK_T = Parameter(_fresh(self.W_UK_T), requires_grad=False)
        self.W_UV = Parameter(_fresh(self.W_UV), requires_grad=False)

        # Keep `W_UK_T`/`W_UV` in the activation dtype, matching upstream vLLM's
        # `MLAAttention.process_weights_after_loading`: there is no quantized bmm
        # for these two, so `forward_mla` upcasts them back to the activation
        # dtype before `torch.bmm` anyway. Quantizing here only cost precision.
        # `kv_cache_quantized_dtype` still governs the KV cache itself.
        self.W_UK_T = Parameter(self.W_UK_T.to(device), requires_grad=False)
        self.W_UV = Parameter(self.W_UV.to(device), requires_grad=False)

        # Safely detach and clear kv_b_proj parameter buffers without breaking PyTorch attribute integrity
        kv_b_proj_params = dict(self.kv_b_proj.named_parameters())
        for key in kv_b_proj_params.keys():
            if key in self.kv_b_proj._parameters:
                self.kv_b_proj._parameters[key] = None
            elif hasattr(self.kv_b_proj, key):
                delattr(self.kv_b_proj, key)

        if self.W_UK_T.device.type == "tpu":
            synchronize_tensors([self.W_UK_T, self.W_UV])

        q_scale, k_scale, v_scale = self.impl._get_kv_scales(self)
        self.mla_op = self.impl._build_mla_op(self,
                                              q_scale=q_scale,
                                              k_scale=k_scale,
                                              v_scale=v_scale)
        self.sparse_mla_op = self.impl._build_sparse_mla_op(self)

    def forward(self,
                q: tuple[torch.Tensor, torch.Tensor],
                kv_c_normed: torch.Tensor,
                k_pe: torch.Tensor,
                output: torch.Tensor | None = None,
                **kwargs) -> torch.Tensor:
        if getattr(self, "calculate_kv_scales", False):
            torch.ops.vllm.maybe_calc_kv_scales(q, kv_c_normed, k_pe,
                                                self.layer_name)

        attn_metadata, _, kv_cache, _ = get_attention_context(self.layer_name)

        return self.impl.forward(
            layer=self,
            q=q,
            kv_c_normed=kv_c_normed,
            k_pe=k_pe,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            topk_indices=kwargs.get("topk_indices"),
        )


# Backward compatibility alias right in case legacy external imports reference old name
VllmMLAAttention = VllmTPUMLAAttention


@MultiHeadLatentAttentionWrapper.register_oot
class VllmTPUMultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        skip_topk: bool = False,
        non_causal_multi_token_decode: bool = False,
        allow_short_prefill_indexer_scoring_skip: bool = False,
    ) -> None:
        torch.nn.Module.__init__(self)

        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.g_proj = getattr(mla_modules, "g_proj", None)
        self.gate_is_fused = getattr(mla_modules, "gate_is_fused", False)
        # `Indexer` has no `register_oot` hook and is fully built by
        # `DeepseekV2MLAAttention` before this wrapper runs, so retype it in
        # place. The mutation is visible through every reference to the object,
        # including `DeepseekV2MLAAttention.indexer`.
        indexer = mla_modules.indexer
        if indexer is not None:
            indexer = VllmTPUIndexer.rebind(indexer)
        self.indexer = indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse
        self.skip_topk = skip_topk
        self.topk_indices_buffer = mla_modules.topk_indices_buffer

        if self.indexer is not None and not self.skip_topk:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens

        self.mla_attn = VllmTPUMLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
            non_causal_multi_token_decode=non_causal_multi_token_decode,
        )

        self.prefix = prefix

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if hasattr(super(), "process_weights_after_loading"):
            super().process_weights_after_loading(act_dtype)
        if self.rotary_emb is not None and hasattr(self.rotary_emb,
                                                   "cos_sin_cache"):
            device = torch.device("tpu")
            self.rotary_emb.cos_sin_cache = self.rotary_emb.cos_sin_cache.to(
                device)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None
        output_gate = None

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None")
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None")
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None")

            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            if self.gate_is_fused:
                q_c, kv_lora, output_gate = qkv_lora.split(
                    [
                        self.q_lora_rank,
                        self.kv_lora_rank + self.qk_rope_head_dim,
                        self.num_heads * self.v_head_dim,
                    ],
                    dim=-1,
                )
            else:
                q_c, kv_lora = qkv_lora.split(
                    [
                        self.q_lora_rank,
                        self.kv_lora_rank + self.qk_rope_head_dim
                    ],
                    dim=-1,
                )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None")
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None")
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim],
                                   dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],
                               dim=-1)

        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        if self.rotary_emb is not None:
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        topk_indices = None
        if self.is_sparse:
            if self.indexer is not None and not self.skip_topk:
                topk_index = self.indexer(hidden_states, q_c, positions,
                                          self.indexer_rope_emb)
            else:
                # reuse the indices an earlier layer wrote into the shared buffer.
                topk_index = self.topk_indices_buffer
            topk_indices = topk_index[:hidden_states.shape[0]]

            # print(f"top_k_indices.shape: {topk_indices.shape}, topk_indices: {topk_indices[0]}")

        if llama_4_scaling is not None:
            q_nope *= llama_4_scaling
            q_pe *= llama_4_scaling

        attn_out = self.mla_attn(
            (q_nope, q_pe),
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0],
                          self.num_heads * self.v_head_dim),
            topk_indices=topk_indices,
        )

        if output_gate is not None:
            attn_out *= output_gate.sigmoid()
        elif self.g_proj is not None:
            attn_out *= self.g_proj(hidden_states)[0].sigmoid()
        return self.o_proj(attn_out)[0]


VllmMultiHeadLatentAttentionWrapper = VllmTPUMultiHeadLatentAttentionWrapper


@DeepseekScalingRotaryEmbedding.register_oot
class VllmDeepseekScalingRotaryEmbedding(DeepseekScalingRotaryEmbedding):
    """TPU-compatible DeepSeek Scaling RoPE implementation."""

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert key is not None
        query_rot = query[..., :self.rotary_dim]
        key_rot = key[..., :self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim:]
            key_pass = key[..., self.rotary_dim:]

        pos = torch.add(positions,
                        offsets) if offsets is not None else positions
        cos_sin = self.cos_sin_cache.index_select(0, pos)
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
            sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key
