# SPDX-License-Identifier: Apache-2.0
"""Persistent SAL Indexer cache writer with independent FP32 compressor state.

Only this step's new boundary records are staged. The persistent cache is never
converted or rebuilt, including across prefill, mixed batches and decode.
"""

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    compressor_v1,
    kernel,
    proj_and_save_state,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    KVLayout,
    scatter_rows,
)
from vllm_torchtpu.utils import get_device_name


def compressor_forward(
    hidden_states: jax.Array,
    wkv_wgate: jax.Array,
    ape: jax.Array,
    norm_weight: jax.Array,
    cos_sin_cache: jax.Array,
    positions: jax.Array,
    block_table: jax.Array,
    query_start_loc: jax.Array,
    kv_block_table: jax.Array,
    cache: jax.Array,
    rope_cache: jax.Array | None,
    distribution: jax.Array,
    state_block_size: int,
    head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
    state_cache: jax.Array,
) -> tuple[jax.Array, None, jax.Array]:
    """Compress new boundaries, then append them with the native SAL writer."""
    if (head_dim, compress_ratio, overlap) != (128, 4, True):
        raise ValueError("SAL compressor requires the DSV4 H128/R4 indexer")
    if cache.ndim != 4 or cache.shape[1:3] != (33, 4):
        raise ValueError("SAL Indexer cache must have shape [pages, 33, 4, page_size]")
    if state_cache is None or rope_cache is not None:
        raise ValueError("SAL requires independent FP32 state, without a RoPE cache")
    tokens = positions.size
    reqs = query_start_loc.size - 1
    fast_decode = (
        tokens in (1, 64)
        and hidden_states.shape[1] == 4096
        and hidden_states.dtype == wkv_wgate.dtype == jnp.bfloat16
        and get_device_name() == "TPU v7"
    )
    # Measured with runtime metadata and the adapter's 256 request slots.
    # Count leading empty requests, including the all-empty padded case.
    fast_metadata = fast_decode and tokens == 1 and reqs == 256
    token_to_req_indices = None
    if fast_metadata:
        token_to_req_indices = (
            jnp.minimum(jnp.sum(query_start_loc[1:] <= 0), reqs - 1)
            .astype(jnp.int32)
            .reshape(1)
        )
    # Derive logical compressed row addresses with the same geometry as HAS.
    # The shape-only object is never allocated, read or converted.
    geometry = jax.ShapeDtypeStruct(
        (cache.shape[0], cache.shape[-1] // 4, 4, 256), jnp.uint8
    )
    req, state_slots, dst = compressor_v1.derive_metadata(
        positions,
        block_table,
        query_start_loc,
        kv_block_table,
        geometry,
        compress_ratio,
        state_block_size,
        head_dim,
        overlap,
        cos_sin_cache,
        state_cache=state_cache,
        token_to_req_indices=token_to_req_indices,
    )
    real = jnp.arange(tokens) < query_start_loc[distribution[2]]
    state_slots = jnp.where(real, state_slots, -1)
    state_cache = proj_and_save_state.proj_and_save_state(
        hidden_states,
        wkv_wgate,
        ape,
        positions,
        state_slots,
        state_cache,
        compress_ratio,
    )
    boundary = real & (dst >= 0)
    # Stable request/token order preserves exclusive-page scatter ownership.
    capacity = (tokens + 3) // 4 * 4
    if fast_metadata:
        index = jnp.zeros((capacity,), jnp.int32)
        valid = (jnp.arange(capacity) == 0) & boundary[0]
    else:
        index = jnp.nonzero(boundary, size=capacity, fill_value=0)[0]
        valid = jnp.arange(capacity) < jnp.sum(boundary)
    boundary_pos = jnp.where(valid, positions[index], 0)
    boundary_req = jnp.where(valid, req[index], 0)
    target = jnp.where(valid, dst[index], -1)

    # Each temporary quad belongs to one compression tile, so the baseline
    # pack/merge operation is exact even for mixed requests and partial rows.
    def write_records(cache):
        scratch = jnp.zeros(((capacity + 255) // 256, 64, 4, 256), jnp.uint8)
        scratch, _ = kernel.compress_norm_rope_store(
            scratch,
            boundary_pos,
            block_table,
            boundary_req,
            jnp.where(valid, jnp.arange(capacity), -1),
            norm_weight,
            block_table_stride=compressor_v1.block_table_row_stride(block_table, reqs),
            state_block_size=state_block_size,
            state_cache=state_cache,
            cos_sin_cache=cos_sin_cache,
            compress_ratio=compress_ratio,
            overlap=overlap,
            quant_block=quant_block,
            rms_eps=rms_eps,
            name="indexer_compress_new_records",
        )
        records = scratch.reshape(-1, 256)[:capacity, :129]
        # The existing writer bounds its scalar run table; keep long prefill valid.
        for start in range(0, capacity, 48 * 1024):
            cache = scatter_rows(
                cache,
                records[start : start + 48 * 1024],
                target[start : start + 48 * 1024],
                kv_layout=KVLayout.SEQ_ALONG_LANE,
            )
        return cache

    # The conditional costs more on staggered B4/B16 decode; keep those
    # workloads on the original unconditional writer.
    if fast_decode:
        cache = jax.lax.cond(
            jnp.any(boundary), write_records, lambda cache: cache, cache
        )
    else:
        cache = write_records(cache)
    state_cache, cache = jax.lax.optimization_barrier((state_cache, cache))
    return cache, None, state_cache
