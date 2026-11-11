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
"""Head dim 64 attention compatibility for Muse Spark Open on TPU.

The hd64 KV pool packs K and V along the last axis (last dim 128).
PallasAttentionBackendImpl.forward reads that 128 as the pool head dim and
zero pads q/k/v from 64 to 128. attention_interface then dispatches on
q.shape[-1] == 64 and picks the head 128 kernel, which rejects the hd64 pool.
This impl slices the padding back off before the kernel call. The slice is
exact because the base only appended zeros, and forward already slices the
output to head_size.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import torch
import torch.nn.functional as F

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel_hd64
from vllm_torchtpu.kernels.ragged_paged_attention.v3.tuned_block_sizes_hd64 import (
    get_tuned_block_sizes,
)
from vllm_torchtpu.layers.adapter import attention as attention_adapter
from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl

MUSE_HD64 = 64
_MIN_RPA_TOKENS = 32


def _slice_hd64(t: torch.Tensor) -> torch.Tensor:
    return t[..., :MUSE_HD64] if t.shape[-1] > MUSE_HD64 else t


def _as_mixed_distribution(dist: torch.Tensor) -> torch.Tensor:
    """Route every sequence through the bq_sz=32 mixed stage in kernel_hd64."""
    return F.pad(dist[2:3], (2, 0)) if dist.shape == (3,) else dist


def wrap_hd64_rpa_kernel(inner: Callable[..., Any]) -> Callable[..., Any]:
    """Return a kernel that drops the base impl's head dim padding on q/k/v."""

    def rpa_kernel_hd64(kv_cache, query, key, value, *rest, **kwargs):
        q = _slice_hd64(query)
        k = _slice_hd64(key)
        v = _slice_hd64(value)
        if not bool(envs.VLLM_TPU_MUSE_BATCH_INVARIANT):
            return inner(kv_cache, q, k, v, *rest, **kwargs)
        if len(rest) >= 4 and isinstance(rest[3], torch.Tensor):
            rest = (*rest[:3], _as_mixed_distribution(rest[3]), *rest[4:])
        m = q.shape[0]
        if q.device.type == "tpu" and m < _MIN_RPA_TOKENS:
            # tuned_block_sizes_hd64 clamps bq_sz to min(max_num_tokens, 32), so
            # a 16-token step would compile a bq_sz=16 kernel. Padding to 32
            # keeps bq_sz=32; cu_q_lens stops at the real tokens.
            pad = _MIN_RPA_TOKENS - m
            q = F.pad(q, (0, 0, 0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))
            return inner(kv_cache, q, k, v, *rest, **kwargs)[:m]
        return inner(kv_cache, q, k, v, *rest, **kwargs)

    rpa_kernel_hd64.muse_hd64_inner = inner  # pyrefly: ignore[missing-attribute]
    return rpa_kernel_hd64


def prefill_queries_per_block(num_tokens: int, tuned: int) -> int:
    """hd64 query block for a step of num_tokens (padded) tokens.

    The query block only changes how query rows are batched; each row's
    softmax and PV accumulation run over the same KV blocks in the same order,
    so outputs are bit identical across block sizes (unlike the KV block).
    """
    if num_tokens >= 4096:
        return max(tuned, 128)
    if num_tokens >= 2048:
        return max(tuned, 64)
    return tuned


def _hd64_with_prefill_block(
    queries, keys, values, kv_cache, kv_lens, page_indices, *args, **kwargs
):
    """kernel_hd64 call with the tuned KV block and prefill query block."""
    if (
        kwargs.get("num_kv_pages_per_block") is None
        and kwargs.get("num_queries_per_block") is None
    ):
        bkv, bq = get_tuned_block_sizes(
            queries.dtype,
            kv_cache.dtype,
            queries.shape[1],
            keys.shape[1],
            queries.shape[2],
            kv_cache.shape[1],
            queries.shape[0],
            page_indices.shape[0] // kv_lens.shape[0],
        )
        kwargs["num_kv_pages_per_block"] = bkv
        kwargs["num_queries_per_block"] = prefill_queries_per_block(
            queries.shape[0], bq
        )
    return kernel_hd64.ragged_paged_attention_hd64(
        queries, keys, values, kv_cache, kv_lens, page_indices, *args, **kwargs
    )


def _muse_rpa_kernel_entry(
    kv_cache: jax.Array,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    sinks: jax.Array | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    *,
    mesh: jax.sharding.Mesh,
    sliding_window: int | None,
    sm_scale: float | None = None,
    soft_cap: float | None = None,
    skip_kv_update: bool,
    use_causal_mask: bool = True,
    block_major: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Run the default RPA entry with the Muse hd64 prefill query block table."""
    return attention_adapter._pallas_rpa_kernel_impl(
        kv_cache,
        query,
        key,
        value,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        sinks,
        q_scale,
        k_scale,
        v_scale,
        mesh=mesh,
        sliding_window=sliding_window,
        skip_kv_update=skip_kv_update,
        rpa_func=attention_adapter.ragged_paged_attention,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        use_causal_mask=use_causal_mask,
        block_major=block_major,
        hd64_rpa_func=_hd64_with_prefill_block,
    )


class MuseHd64PallasAttentionImpl(PallasAttentionBackendImpl):
    """PallasAttentionBackendImpl that keeps head dim 64 on the hd64 kernel."""

    _kernel_entry = staticmethod(_muse_rpa_kernel_entry)
    _kernel_op_prefix = "pallas::muse_rpa_hd64"

    def _build_rpa_kernel(self, *args: Any, **kwargs: Any):
        inner = super()._build_rpa_kernel(*args, **kwargs)
        if self.head_size != MUSE_HD64:
            return inner
        cache = self.__dict__.setdefault("_muse_hd64_wrappers", {})
        wrapped = cache.get(id(inner))
        if wrapped is None or wrapped.muse_hd64_inner is not inner:
            wrapped = wrap_hd64_rpa_kernel(inner)
            cache[id(inner)] = wrapped
        return wrapped

    def setup_bundled(self, *args: Any, **kwargs: Any) -> None:
        if self.head_size == MUSE_HD64:
            raise NotImplementedError(
                "Muse Spark head_dim 64 attention does not support the bundled "
                "block major KV path (VLLM_TPU_BLOCK_MAJOR_KV=1)."
            )
        super().setup_bundled(*args, **kwargs)


def install_muse_hd64_attention(attn: torch.nn.Module) -> bool:
    """Swap a head dim 64 Pallas attention impl to the Muse subclass in place."""
    impl = getattr(attn, "impl", None)
    if impl is None or getattr(impl, "head_size", None) != MUSE_HD64:
        return False
    if isinstance(impl, MuseHd64PallasAttentionImpl):
        return True
    if type(impl) is not PallasAttentionBackendImpl:
        return False
    impl.__class__ = MuseHd64PallasAttentionImpl
    return True
