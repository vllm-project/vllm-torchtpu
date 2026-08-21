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

import jax
import jax.numpy as jnp
import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention_layer_base import \
    AttentionBackend  # isort: skip
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend, DeepseekV4SWACache)
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowMLASpec

from vllm_torchtpu.kernels.deepseek_v4.mla import mla_ragged_paged_attention
from vllm_torchtpu.kernels.deepseek_v4.mla_swa import \
    mla_sliding_window_ragged_paged_attention
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import align_to

logger = init_logger(__name__)

BATCH_AXIS = None


# Top-level function required by pallas.jax_op for JAX tracing and
# registration as a custom PyTorch TPU operator.
def _attention_jax(
    q: jax.Array,
    new_kv: jax.Array,
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    main_cache_kv: jax.Array,
    main_kv_lens: jax.Array,
    extra: jax.Array,
    main_page_indices: jax.Array,
    main_cu_q_lens: jax.Array,
    main_distribution: jax.Array,
    attention_sinks: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    swa_only: bool,
    is_csa: bool,
    two_caches_same_buffer: bool,
) -> tuple[jax.Array, jax.Array]:
    # Skip kernel execution during vLLM dummy/profiling passes with numel==0 caches;
    # TorchTPU traces unused inputs into the graph signature via keep_unused=True.
    if main_cache_kv.shape[0] == 0 or (swa_only and sw_cache.shape[0] == 0):
        return jnp.zeros(q.shape, dtype=q.dtype), sw_cache

    def _attention_local(q, new_kv, sw_cache, swa_kv_lens, swa_page_indices,
                         swa_cu_q_lens, swa_distribution, main_cache_kv,
                         main_kv_lens, extra, main_page_indices,
                         main_cu_q_lens, main_distribution, attention_sinks):
        if swa_page_indices is not None:
            swa_page_indices = swa_page_indices.flatten()
        if main_page_indices is not None:
            main_page_indices = main_page_indices.flatten()

        if sw_cache.shape[0] == 0:
            swa_output = jnp.zeros_like(q)
            updated_sw_cache = sw_cache
            swa_l = jnp.zeros((q.shape[0], q.shape[1], 1), dtype=jnp.float32)
            swa_m = jnp.full((q.shape[0], q.shape[1], 1),
                             -1e9,
                             dtype=jnp.float32)
        else:
            swa_output, updated_sw_cache, swa_l, swa_m = (
                mla_sliding_window_ragged_paged_attention(
                    q=q,
                    new_kv=new_kv,
                    cache_kv=sw_cache,
                    kv_lens=swa_kv_lens,
                    page_indices=swa_page_indices,
                    cu_q_lens=swa_cu_q_lens,
                    distribution=swa_distribution,
                    attention_sinks=attention_sinks,
                    sm_scale=sm_scale,
                    sliding_window=sliding_window,
                    logical_page_size=logical_page_size,
                    num_kv_pages_per_block=1,
                    num_queries_per_block=1,
                    unnormalized_output=False if swa_only else True,
                ))

        if swa_only:
            return swa_output, updated_sw_cache

        # When SWA and main caches share physical memory, forward the SWA kernel's
        # updated buffer to main attention so it reads newly written tokens.
        if two_caches_same_buffer:
            main_cache_kv = updated_sw_cache

        if swa_l is not None and swa_l.ndim == 3:
            swa_l = jnp.squeeze(swa_l, axis=-1)
        if swa_m is not None and swa_m.ndim == 3:
            swa_m = jnp.squeeze(swa_m, axis=-1)

        output = mla_ragged_paged_attention(
            q=q,
            cache_kv=main_cache_kv,
            kv_lens=main_kv_lens,
            kv_lens_to_attend=None if is_csa else extra,
            topk_indices=extra if is_csa else None,
            page_indices=main_page_indices,
            cu_q_lens=main_cu_q_lens,
            distribution=main_distribution,
            attention_sinks=attention_sinks,
            swa_accumulation=swa_output,
            swa_l=swa_l,
            swa_m=swa_m,
            sm_scale=sm_scale,
            num_kv_pages_per_block=1,
            num_queries_per_block=1,
        )

        return output, updated_sw_cache

    return _attention_local(q, new_kv, sw_cache, swa_kv_lens, swa_page_indices,
                            swa_cu_q_lens, swa_distribution, main_cache_kv,
                            main_kv_lens, extra, main_page_indices,
                            main_cu_q_lens, main_distribution, attention_sinks)


def get_packed_mla_head_size(hf_config) -> int:
    """Calculate TPU 128-byte aligned packed width for DSv4 FP8 cache entries."""
    kv_bytes = getattr(hf_config, "kv_lora_rank", 448)
    rope_bytes = getattr(hf_config, "qk_rope_head_dim", 64) * 2
    scale_bytes = kv_bytes // 64
    return align_to(kv_bytes + rope_bytes + scale_bytes, 128)


class VllmDeepseekSparseSWABackend(DeepseekSparseSWABackend):
    """TPU attention backend for DeepSeek-V4 SWA cache."""

    @classmethod
    def get_min_page_size(cls, vllm_config: VllmConfig) -> int:
        return 1


class VllmDeepseekV4SWACache(DeepseekV4SWACache):
    """Sliding-window KV cache, paged to fit inside a CSA main page."""

    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config,
    ) -> None:
        super().__init__(head_dim, window_size, dtype, prefix, cache_config)
        # Initialize with default block size at construction time;
        # get_kv_cache_spec recomputes self.block_size from finalized runtime config.
        self.block_size = self._swa_block_size(cache_config.block_size,
                                               window_size)

    @staticmethod
    def _swa_block_size(compressed_kv_cache_bz: int,
                        window_size: int | None) -> int:
        """Derive SWA page size clamped to window size to fit inside host CSA pages."""
        csa_compression_ratio = 4
        block_size = compressed_kv_cache_bz // csa_compression_ratio
        if window_size is None:
            return block_size
        return min(block_size, window_size)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # Update self.block_size so the returned spec and the Pallas kernel's
        # logical_page_size in attention.py agree on the final runtime block size.
        self.block_size = self._swa_block_size(
            vllm_config.cache_config.block_size, self.window_size)
        hf_config = vllm_config.model_config.hf_config
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=get_packed_mla_head_size(hf_config),
            dtype=torch.uint8,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            alignment=None,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return VllmDeepseekSparseSWABackend
