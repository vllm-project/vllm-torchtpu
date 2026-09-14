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

import functools
from typing import Optional

import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4 import attention as dsv4_attention
from vllm.models.deepseek_v4.attention import (DeepseekV4Indexer,
                                               DeepseekV4IndexerCache)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_torchtpu.kernels.deepseek_v4.rope import rope_quant
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_compressor import \
    VllmDeepseekCompressor
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context
from vllm_torchtpu.utils import align_to

logger = init_logger(__name__)

_indexer_op_cache = {}

BATCH_AXIS = None


# Module-level so `pallas.jax_op` can trace and register it as a torch op.
def _indexer_jax(
    q: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    indexer_weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    k: int,
    compression_ratio: int,
    softmax_scale: float,
    n_head: int,
) -> jax.Array:
    if cache_kv.shape[0] == 0:
        # Profiling-shape trace: the kernel is skipped. Unused inputs stay in
        # the compiled signature because torch_tpu jits with keep_unused=True.
        return jnp.zeros((q.shape[0], k), dtype=jnp.int32)

    def _indexer_local(q, positions, cos_sin_cache, indexer_weights, cache_kv,
                       seq_lens, page_indices, cu_q_lens, distribution):
        # One kernel rotates the queries and quantizes each row on the way out.
        # Note: vLLM's implementation rounds the scale factors up to the next
        # power of 2, but the plain `abs_max / dtype_max` the kernel returns is
        # sufficient here.
        q_quant, q_scales = rope_quant(q,
                                       positions,
                                       cos_sin_cache,
                                       quant_dtype=jnp.float8_e4m3fn)

        # Fold the query quantization scales into the weights.
        weights = (indexer_weights * softmax_scale * (n_head**-0.5) * q_scales)

        return streamindex_topk(
            q=q_quant,
            indexer_weights=weights,
            cache_kv=cache_kv,
            seq_lens=seq_lens,
            page_indices=page_indices.flatten(),
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            k=k,
            compression_ratio=compression_ratio,
            # The following parameters are tuned based on microbenchmark results.
            num_kv_pages_per_block=(3, 2, 2),
            num_queries_per_block=(1, 128, 128),
        )

    return _indexer_local(q, positions, cos_sin_cache, indexer_weights,
                          cache_kv, seq_lens, page_indices, cu_q_lens,
                          distribution)


class VllmDeepseekV4IndexerCache(DeepseekV4IndexerCache):
    """Indexer K cache as a packed uint8 record on TPU."""

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size = self.cache_config.block_size
        return MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=align_to(128 + 1, 128),
            dtype=torch.uint8,
            compress_ratio=min(self.compress_ratio, block_size),
            alignment=None,
        )


class VllmDeepseekV4Indexer(DeepseekV4Indexer):
    """TPU indexer: ranks compressed KV rows via the streamindex top-k kernel."""

    def __init__(self, *args, **kwargs) -> None:
        orig_compressor = dsv4_attention.DeepseekCompressor
        orig_cache = dsv4_attention.DeepseekV4IndexerCache
        dsv4_attention.DeepseekCompressor = VllmDeepseekCompressor
        dsv4_attention.DeepseekV4IndexerCache = VllmDeepseekV4IndexerCache
        try:
            super().__init__(*args, **kwargs)
        finally:
            dsv4_attention.DeepseekCompressor = orig_compressor
            dsv4_attention.DeepseekV4IndexerCache = orig_cache
        object.__setattr__(self, "indexer_op", self._build_indexer_op())
        object.__setattr__(self.compressor, "k_cache", self.k_cache)

    def _build_indexer_op(self):
        vllm_context = get_vllm_model_wrapper_context()
        mesh = vllm_context.mesh

        wrapped_fn = functools.partial(
            _indexer_jax,
            k=self.topk_tokens,
            compression_ratio=self.compress_ratio,
            softmax_scale=self.softmax_scale,
            n_head=self.n_head,
        )

        _scale = f"{self.softmax_scale:.6g}".replace(".", "p").replace(
            "-", "m").replace("+", "")
        op_name = (f"pallas::deepseek_v4_indexer_k{self.topk_tokens}"
                   f"_c{self.compress_ratio}_h{self.n_head}_s{_scale}")
        global _indexer_op_cache
        if op_name in _indexer_op_cache:
            return _indexer_op_cache[op_name]

        attn_data_axis = None
        attn_head_axis = "model"
        batch_axis = BATCH_AXIS
        indexer_jax_op = pallas.jax_op(
            op_name,
            wrapped_fn,
            mesh=mesh,
            input_partition_specs=(
                P(attn_data_axis, attn_head_axis, None),  # q
                P(attn_data_axis),  # positions
                P(),  # cos_sin_cache
                P(attn_data_axis, attn_head_axis),  # indexer_weights
                P(),  # cache_kv
                P(batch_axis),  # seq_lens
                P(batch_axis),  # page_indices
                P(),  # cu_q_lens
                P(),  # distribution
            ),
        )

        def _fake_indexer(q, *args, **kwargs):
            return torch.empty((q.shape[0], self.topk_tokens),
                               dtype=torch.int32,
                               device=q.device)

        indexer_jax_op.register_fake(_fake_indexer)

        _indexer_op_cache[op_name] = indexer_jax_op
        return indexer_jax_op

    def forward(
        self,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
        slot_mapping: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, _ = self.wq_b(query)
        q = q.view(-1, self.n_head, self.head_dim)

        self.compressor(hidden_states, positions, rotary_emb)

        # No KV cache is bound during the profiling forward; the values are
        # unused, so return correctly-shaped dummies. vLLM binds a numel==0
        # placeholder rather than None, so test emptiness.
        kv_cache = getattr(self.k_cache, "kv_cache", None)
        if kv_cache is None or kv_cache.numel() == 0:
            return torch.zeros((q.shape[0], self.topk_tokens),
                               dtype=torch.int32,
                               device=q.device)

        attn_ctx = get_forward_context().attn_metadata
        if isinstance(attn_ctx, dict):
            prefix_key = getattr(getattr(self, "k_cache", None), "prefix",
                                 None)
            if prefix_key not in attn_ctx:
                raise KeyError(
                    f"DeepSeek-V4 indexer prefix {prefix_key!r} has no "
                    f"attention metadata. Known: {sorted(attn_ctx)}")
            attn_metadata = attn_ctx[prefix_key]
        else:
            attn_metadata = attn_ctx

        idx_seq_lens = attn_metadata.seq_lens
        idx_block_tables = attn_metadata.block_tables
        idx_q_start_loc = attn_metadata.query_start_loc
        idx_req_dist = attn_metadata.request_distribution

        return self.indexer_op(
            q,
            positions,
            rotary_emb.cos_sin_cache,
            indexer_weights,
            kv_cache,
            idx_seq_lens,
            idx_block_tables,
            idx_q_start_loc,
            idx_req_dist,
        )
