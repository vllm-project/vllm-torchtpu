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
"""The framework's recurrent pool slot, as `megakernel.runtime.PoolLayoutFacts`.

Built only from what vllm-torchtpu itself uses to lay out the pool: the GDN
layer's declared state dtypes, resolved through `gdn_pool_layout` exactly as
the block-size derivation and the stock kernel's state plan resolve them, the
cache block size, the raw pool tensor, and the conv-order switch.
"""

from __future__ import annotations

import math
from typing import Any

from vllm_torchtpu.megakernel.runtime.layout_facts import (
    CONV_ORDER_TAPS_QKV,
    PoolLayoutFacts,
    dtype_name,
)

# Conv element order with Q/K head pairs interleaved (`TPU_GDN_CONV_QK_PAIR_LAYOUT`).
CONV_ORDER_QK_PAIR = "qk_pair"


def _itemsize(dt: Any) -> int:
    size = getattr(dt, "itemsize", None)
    if isinstance(size, int):
        return size
    import numpy as np

    return int(np.dtype(dtype_name(dt)).itemsize)


def runner_block_size(runner: Any) -> int | None:
    """The cache block size the framework sizes one recurrent slot with."""
    for cfg in (
        getattr(getattr(runner, "vllm_config", None), "cache_config", None),
        getattr(runner, "cache_config", None),
    ):
        bs = getattr(cfg, "block_size", None)
        if bs:
            return int(bs)
    return None


def conv_qk_pair_layout_enabled() -> bool:
    from vllm_torchtpu import envs as tpu_envs

    return bool(getattr(tpu_envs, "TPU_GDN_CONV_QK_PAIR_LAYOUT", False))


def framework_pool_layout_facts(
    *,
    pool: Any,
    linear_attn: Any,
    block_size: int,
    conv_qk_pair: bool,
) -> PoolLayoutFacts:
    """Describe one rank's slot the way the framework lays it out.

    `pool` is the per-rank raw pool `(num_pages, *page_shape)`; `linear_attn`
    is one of the GDN layers that alias it.
    """
    from vllm_torchtpu import gdn_pool_layout as gpl

    conv_dt, ssm_dt = gpl.pooled_gdn_state_dtypes(tuple(linear_attn.get_state_dtype()))
    geom = linear_attn.tp_head_geometry
    n_kq = int(geom.local_num_kq_heads)
    n_v = int(linear_attn.num_v_heads) // int(linear_attn.tp_size)
    d_k, d_v = int(linear_attn.head_k_dim), int(linear_attn.head_v_dim)
    kernel_size = int(linear_attn.conv_kernel_size)
    conv_dim = 2 * n_kq * d_k + n_v * d_v

    page_shape = tuple(int(d) for d in pool.shape[1:])
    page_size = page_shape[-1]
    token_bytes = math.prod(page_shape) * _itemsize(pool.dtype) // page_size
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
        slot_tokens=int(block_size),
        ssm_dtype=dtype_name(ssm_dt),
        ssm_bytes=int(ssm_bytes),
        ssm_offset_bytes=0,
        conv_dtype=dtype_name(conv_dt),
        conv_bytes=int(conv_bytes),
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
    "runner_block_size",
]
