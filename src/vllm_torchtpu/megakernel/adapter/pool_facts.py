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
"""The framework's recurrent pool slot, as `megakernel.core.runtime.PoolLayoutFacts`.

Built only from what vllm-torchtpu itself uses to lay out the pool: the GDN
layer's declared state dtypes, resolved through `gdn_pool_layout` exactly as
the block-size derivation and the stock kernel's state plan resolve them, the
layer's TP head geometry, the cache block size, the resolved KV cache layout,
the raw pool tensor, and the conv-order switch.
"""

from __future__ import annotations

import math
from typing import Any

import vllm_torchtpu.envs as envs
from vllm_torchtpu import gdn_pool_layout as gpl
from vllm_torchtpu.megakernel.core.runtime.contract import (
    POOL_LAYOUT_MISMATCH,
    POOL_LAYOUT_UNKNOWN,
    EligibilityResult,
    LayoutDecline,
)
from vllm_torchtpu.megakernel.core.runtime.layout_facts import (
    CONV_ORDER_TAPS_QKV,
    PoolLayoutFacts,
    dtype_name,
)

# Conv element order with Q/K head pairs interleaved (`TPU_GDN_CONV_QK_PAIR_LAYOUT`).
CONV_ORDER_QK_PAIR = "qk_pair"

_PAGE_RANK = 4


def _page_token_axis(kv_layout: Any) -> int | None:
    """The page axis that holds a pool page's tokens under vLLM's resolved
    `KVCacheLayout` (`cache_config.get_resolved_kv_cache_layout()`), or None.

    HND (`LBHNC`, SEQ_ALONG_LANE, the megakernel's) pages are
    `(2 * kvh, hd / pack, pack, page_size)`; NHD (`LBNHC`, HEAD_ALONG_SUBLANE,
    `PallasAttentionBackend.get_kv_cache_shape`) pages are
    `(page_size, 2 * kvh / pack, pack, hd)`. Any other layout is declined.
    """
    # Imported here, as the adapter's other vLLM imports are, so that the
    # module imports without vLLM.
    from vllm.v1.kv_cache_layout import KVCacheLayout

    return {KVCacheLayout.LBHNC: -1, KVCacheLayout.LBNHC: 0}.get(kv_layout)


def _mismatch(detail: str) -> LayoutDecline:
    msg = f"Megakernel can't use the framework's pool: {detail}; skipping."
    return LayoutDecline(EligibilityResult(False, POOL_LAYOUT_MISMATCH, msg))


def _unknown(detail: str) -> LayoutDecline:
    msg = (
        f"Megakernel can't describe the framework's recurrent pool slot: {detail}; "
        "skipping."
    )
    return LayoutDecline(EligibilityResult(False, POOL_LAYOUT_UNKNOWN, msg))


def conv_qk_pair_layout_enabled() -> bool:
    return envs.TPU_GDN_CONV_QK_PAIR_LAYOUT


def framework_pool_layout_facts(
    *,
    pool: Any,
    linear_attn: Any,
    block_size: int,
    kv_layout: Any,
    pcp_size: int,
    conv_qk_pair: bool,
) -> PoolLayoutFacts:
    """Describe one rank's slot the way the framework lays it out.

    `pool` is the per-rank raw pool `(num_pages, *page_shape)`; `linear_attn`
    is one of the GDN layers that alias it. `block_size` is the cache block
    size (`RunnerView.cache_block_size`), which the stock pooled GDN kernel
    also uses as one rank's slot (`pool_block_tokens`). `kv_layout` is the
    runner's `cache_config.get_resolved_kv_cache_layout()` and `pcp_size` its
    `parallel_config.prefill_context_parallel_size`.

    Raises `LayoutDecline` (`pool_layout_unknown`) under PCP or for a KV
    layout it doesn't know, and (`pool_layout_mismatch`) for a pool whose
    pages it can't read as whole-granule token rows.
    """
    if pcp_size != 1:
        raise _unknown(
            f"prefill context parallel size {pcp_size} shards the TP-local GDN "
            "heads again per PCP rank, which these facts don't describe"
        )
    axis = _page_token_axis(kv_layout)
    if axis is None:
        raise _unknown(f"KV cache layout {kv_layout.name} isn't one of HND or NHD")
    page_shape = tuple(int(d) for d in pool.shape[1:])
    if len(page_shape) != _PAGE_RANK or min(page_shape) <= 0:
        raise _mismatch(f"pool page shape {page_shape} isn't a {_PAGE_RANK}-D page")
    page_size = page_shape[axis]
    token_bytes = math.prod(page_shape) * pool.dtype.itemsize // page_size
    if token_bytes % gpl.GRANULE_BYTES != 0:
        raise _mismatch(
            f"a {kv_layout.name} page {page_shape} of {dtype_name(pool.dtype)} has "
            f"{token_bytes}-byte token rows, not whole {gpl.GRANULE_BYTES}-byte "
            "granules"
        )

    conv_dt, ssm_dt = gpl.pooled_gdn_state_dtypes(tuple(linear_attn.get_state_dtype()))
    geom = linear_attn.tp_head_geometry
    n_kq, n_v = geom.local_num_kq_heads, geom.local_num_v_heads
    d_k, d_v = linear_attn.head_k_dim, linear_attn.head_v_dim
    kernel_size = linear_attn.conv_kernel_size
    conv_dim = geom.local_conv_dim(d_k, d_v)
    ssm_bytes = gpl.pooled_gdn_ssm_state_bytes(
        num_v_heads=n_v, head_k_dim=d_k, head_v_dim=d_v, dtype=ssm_dt
    )
    conv_bytes = gpl.pooled_gdn_conv_state_bytes(
        kernel_size=kernel_size, conv_dim=conv_dim
    )
    state = gpl.derive_pooled_gdn_state_layout(
        ssm_bytes=ssm_bytes, conv_bytes=conv_bytes, token_bytes=token_bytes
    )
    return PoolLayoutFacts(
        page_dtype=dtype_name(pool.dtype),
        page_shape=page_shape,
        page_size=page_size,
        token_bytes=token_bytes,
        slot_tokens=block_size,
        ssm_dtype=dtype_name(ssm_dt),
        ssm_bytes=ssm_bytes,
        ssm_offset_bytes=0,
        conv_dtype=dtype_name(conv_dt),
        conv_bytes=conv_bytes,
        conv_offset_bytes=state.ssm_tokens * token_bytes,
        conv_order=CONV_ORDER_QK_PAIR if conv_qk_pair else CONV_ORDER_TAPS_QKV,
        local_kh=n_kq,
        local_vh=n_v,
        d_k=d_k,
        d_v=d_v,
        conv_taps=kernel_size - 1,
        conv_dim=conv_dim,
    )


__all__ = [
    "CONV_ORDER_QK_PAIR",
    "conv_qk_pair_layout_enabled",
    "framework_pool_layout_facts",
]
