# SPDX-License-Identifier: Apache-2.0
"""SC-interleaved selected KV and grouped-query prefill MXU tiles.

Transport preserves native packed rows. Each query owns its KV list; batched
QK/PV products keep query softmaxes independent. ``selected`` gives each
query's live leading rows of its gathered list (``slab_gather`` fields).
``kv_head_offset`` reads a paired cache row's second layer; ``return_lse``
returns the FP32 normalized, ungated output and its log-sum-exp (the
context-parallel partial form).
QK, probabilities, PV and online state retain the original prefill FP32
arithmetic; only the original output/gate boundaries round to BF16.
"""

import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import utils
from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16
from vllm_torchtpu.kernels.sparse_gqa_prefill import slab_gather
from vllm_torchtpu.kernels.sparse_gqa_prefill.tuned_block_sizes import (
    SC_GATHER_MAX_CACHE_BYTES,
)


def _supports_whole_slab(blocks, table, cache, *, query_group, tile_k, padding_rows):
    source_rows = cache.shape[0] * cache.shape[1] * cache.shape[2]
    if (
        blocks.shape[0] % query_group
        or source_rows > jnp.iinfo(jnp.int32).max
        or not 0 < padding_rows <= source_rows
    ):
        return False
    info = pltpu.get_tpu_info().sparse_core
    if info is None:
        return False
    descriptor_bytes = info.num_lanes * jnp.dtype(jnp.int32).itemsize
    return (
        tile_k * cache.shape[2] % (4 * info.num_lanes) == 0
        and slab_gather.buffer_bytes(blocks.shape[1], table.shape[1], cache.shape[-1])[
            "total"
        ]
        + descriptor_bytes
        <= info.vmem_capacity_bytes
        and blocks.shape[0]
        * slab_gather.field_layout(blocks.shape[1], table.shape[1])["stride_words"]
        * jnp.dtype(jnp.int32).itemsize
        <= SC_GATHER_MAX_CACHE_BYTES
    )


def _attention_block(
    block,
    lengths_ref,
    q_ref,
    gate_ref,
    compact_ref,
    out_ref,
    lse_ref,
    kv_ref,
    sem_ref,
    m_ref,
    l_ref,
    acc_ref,
    *,
    query_group,
    source_group,
    head_group,
    kv_heads,
    dim,
    tile_k,
    capacity,
    budget,
    ratio,
    k_scale,
    v_scale,
    kv_head_offset=0,
    sm_scale=None,
):
    rows = query_group * head_group
    if sm_scale is None:
        sm_scale = dim**-0.5
    width = query_group * tile_k
    packing = 32 // jax.dtypes.itemsize_bits(compact_ref.dtype)
    head_axes = compact_ref.shape[1]
    del budget, ratio  # the caller's selection fields give the live rows
    selected = [lengths_ref[block * query_group + i] for i in range(query_group)]
    query_owner = jnp.arange(rows)[:, None] % query_group
    row_lengths = jnp.zeros((rows, 128), jnp.int32)
    max_length = selected[0]
    for i in range(query_group):
        row_lengths = jnp.where(query_owner == i, selected[i], row_lengths)
        max_length = jnp.maximum(max_length, selected[i])
    row_lengths = row_lengths[:, :1]
    tile_count = jnp.maximum(1, pl.cdiv(max_length, tile_k))

    def start(tile, buf):
        groups_per_source = source_group // query_group
        source_block = block // groups_per_source
        subgroup = block % groups_per_source
        offset = (
            source_block * (capacity // tile_k) + tile
        ) * source_group * tile_k + subgroup * width
        pltpu.make_async_copy(
            compact_ref.at[pl.ds(offset, width)], kv_ref.at[buf], sem_ref.at[buf]
        ).start()

    m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, jnp.float32)
    l_ref[...] = jnp.zeros(l_ref.shape, jnp.float32)
    acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)
    start(0, 0)

    def step(tile, _):
        buf = tile % 2
        dst = kv_ref.at[buf]
        pltpu.make_async_copy(dst, dst, sem_ref.at[buf]).wait()

        @pl.when(tile + 1 < tile_count)
        def prefetch():
            start(tile + 1, 1 - buf)

        token = lax.broadcasted_iota(jnp.int32, (width, dim), 0)
        live = jnp.zeros((width, dim), jnp.bool_)
        for i in range(query_group):
            live |= (token // tile_k == i) & (
                tile * tile_k + token % tile_k < selected[i]
            )
        words = dst.bitcast(jnp.uint32).reshape(-1, dim)
        for head in range(kv_heads):
            pair = head + kv_head_offset
            packed = utils.strided_load(
                words, pair // (packing // 2), width * head_axes, head_axes
            )
            key, value = utils.convert_to_target_bitwidth(
                packed, target_bitwidth=32 // packing, kv_dtype=compact_ref.dtype
            )[pair % (packing // 2)]
            key = key.astype(jnp.float32)
            value = value.astype(jnp.float32)
            if compact_ref.dtype == jnp.float8_e4m3fn:
                key = jnp.where(live & jnp.isfinite(key), key, 0)
                value = jnp.where(live & jnp.isfinite(value), value, 0)
            else:
                key = jnp.where(live, key, 0)
                value = jnp.where(live, value, 0)
            # Batch by the actual KV-list owner; no cross-query scores.
            query = (
                q_ref[head]
                .astype(jnp.float32)
                .reshape(head_group, query_group, dim)
                .transpose(1, 0, 2)
            )
            key = key.reshape(query_group, tile_k, dim)
            value = value.reshape(query_group, tile_k, dim)
            score = lax.dot_general(
                query,
                key,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            ) * (sm_scale * k_scale)
            lengths = row_lengths.reshape(head_group, query_group, 1).transpose(1, 0, 2)
            valid_batch = tile * tile_k + jnp.arange(tile_k)[None, None, :] < lengths
            score = jnp.where(valid_batch, score, -jnp.inf)
            old_m = (
                m_ref[head, :, :1]
                .reshape(head_group, query_group, 1)
                .transpose(1, 0, 2)
            )
            old_l = (
                l_ref[head, :, :1]
                .reshape(head_group, query_group, 1)
                .transpose(1, 0, 2)
            )
            next_m = jnp.maximum(old_m, jnp.max(score, axis=2, keepdims=True))
            safe_m = jnp.where(jnp.isfinite(next_m), next_m, 0)
            prob = jnp.where(valid_batch, jnp.exp(score - safe_m), 0)
            alpha = jnp.exp(old_m - safe_m)
            weighted = lax.dot_general(
                prob,
                value.astype(jnp.float32),
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            weighted = weighted.transpose(1, 0, 2).reshape(rows, dim)
            alpha = alpha.transpose(1, 0, 2).reshape(rows, 1)
            next_m = next_m.transpose(1, 0, 2).reshape(rows, 1)
            next_l = old_l * alpha.reshape(head_group, query_group, 1).transpose(
                1, 0, 2
            ) + jnp.sum(prob, axis=2, keepdims=True)
            acc_ref[head] = acc_ref[head] * alpha + weighted * v_scale
            m_ref[head] = jnp.broadcast_to(next_m, (rows, 128))
            l_ref[head] = jnp.broadcast_to(
                next_l.transpose(1, 0, 2).reshape(rows, 1), (rows, 128)
            )
        return None

    lax.fori_loop(0, tile_count, step, None)
    for head in range(kv_heads):
        mass = l_ref[head, :, :1]
        result = acc_ref[head] / jnp.where(mass > 0, mass, 1)
        if lse_ref is not None:
            live = (row_lengths > 0) & (mass > 0)
            out_ref[head] = jnp.where(live, result, 0)
            lse = m_ref[head] + jnp.log(jnp.where(l_ref[head] > 0, l_ref[head], 1))
            lse_ref[head] = jnp.where(live, lse, -jnp.inf)
            continue
        gated = bf16(
            bf16(result).astype(jnp.float32)
            * bf16(jax.nn.sigmoid(gate_ref[head].astype(jnp.float32))).astype(
                jnp.float32
            )
        )
        out_ref[head] = jnp.where(row_lengths > 0, gated, 0)


def _attend_compact_ref(
    q: jax.Array,
    gate: jax.Array,
    lengths: jax.Array,
    destination,
    *,
    kv_heads: int,
    ratio: int,
    query_group: int,
    source_group: int,
    tile_k: int,
    capacity: int,
    budget: int,
    scheduling_group_id: int | None = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    head_axes: int | None = None,
    kv_head_offset: int = 0,
    return_lse: bool = False,
    sm_scale: float | None = None,
):
    """Attend each query to its gathered rows; ``lengths`` holds each query's
    live row count. Returns the gated BF16 output, or with ``return_lse`` the
    FP32 normalized output and lse [batch, heads]."""
    packing = 32 // jax.dtypes.itemsize_bits(destination.dtype)
    if head_axes is None:
        head_axes = pl.cdiv(2 * kv_heads, packing)
    compact = jax.ShapeDtypeStruct(
        (destination.shape[0] // head_axes, head_axes, *destination.shape[1:]),
        destination.dtype,
    )
    batch, heads, dim = q.shape
    real_heads = heads // kv_heads
    head_group = pl.cdiv(real_heads, 8) * 8

    def pack(x):
        x = x.reshape(batch // query_group, query_group, kv_heads, real_heads, dim)
        x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, head_group - real_heads), (0, 0)))
        return x.transpose(0, 2, 3, 1, 4).reshape(
            batch // query_group, kv_heads, head_group * query_group, dim
        )

    spec = pl.BlockSpec(
        (None, kv_heads, head_group * query_group, dim), lambda b, *_: (b, 0, 0, 0)
    )
    state = (kv_heads, head_group * query_group, 128)
    packed_query, packed_gate = pack(q), pack(gate)
    score_elements = batch * kv_heads * head_group * capacity
    alpha_elements = batch * kv_heads * head_group * (capacity // tile_k)
    output_elements = batch * kv_heads * head_group * dim
    block_ids = _block_id_data(batch, query_group)

    def reader(lengths_ref, block_id_ref, q_ref, gate_ref, *refs):
        if return_lse:
            out_ref, lse_ref, *refs = refs
        else:
            (out_ref, *refs), lse_ref = refs, None
        kv_ref, sem_ref, m_ref, l_ref, acc_ref = refs
        _attention_block(
            block_id_ref[0, 0],
            lengths_ref,
            q_ref,
            gate_ref,
            destination.reshape(*compact.shape),
            out_ref,
            lse_ref,
            kv_ref,
            sem_ref,
            m_ref,
            l_ref,
            acc_ref,
            query_group=query_group,
            source_group=source_group,
            head_group=head_group,
            kv_heads=kv_heads,
            dim=dim,
            tile_k=tile_k,
            capacity=capacity,
            budget=budget,
            ratio=ratio,
            k_scale=k_scale,
            v_scale=v_scale,
            kv_head_offset=kv_head_offset,
            sm_scale=sm_scale,
        )

    out_dtype = jnp.float32 if return_lse else jnp.bfloat16
    out_shape = jax.ShapeDtypeStruct(
        (batch // query_group, kv_heads, head_group * query_group, dim), out_dtype
    )
    lse_spec = pl.BlockSpec(
        (None, kv_heads, head_group * query_group, 128), lambda b, *_: (b, 0, 0, 0)
    )
    lse_shape = jax.ShapeDtypeStruct(
        (batch // query_group, kv_heads, head_group * query_group, 128), jnp.float32
    )
    y = pl.pallas_call(
        reader,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(batch // query_group,),
            in_specs=(_block_id_spec(), spec, spec),
            out_specs=(spec, lse_spec) if return_lse else spec,
            scratch_shapes=(
                pltpu.VMEM(
                    (2, query_group * tile_k, *compact.shape[1:]), destination.dtype
                ),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.VMEM(state, jnp.float32),
                pltpu.VMEM(state, jnp.float32),
                pltpu.VMEM((kv_heads, head_group * query_group, dim), jnp.float32),
            ),
        ),
        out_shape=(out_shape, lse_shape) if return_lse else out_shape,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        cost_estimate=pl.CostEstimate(
            flops=4 * score_elements * dim,
            transcendentals=score_elements + alpha_elements + output_elements,
            bytes_accessed=packed_query.nbytes
            + packed_gate.nbytes
            + output_elements * jnp.dtype(out_dtype).itemsize
            + math.prod(destination.shape) * jnp.dtype(destination.dtype).itemsize
            + lengths.nbytes
            + block_ids.nbytes,
        ),
        name="destination_ref_block_id_tc_attention",
    )(lengths, block_ids, packed_query, packed_gate)
    if scheduling_group_id is not None:
        y = xla_metadata.set_xla_metadata(y, _scheduling_group_id=scheduling_group_id)

    def unpack(x, width):
        x = x.reshape(batch // query_group, kv_heads, head_group, query_group, width)
        x = x.transpose(0, 3, 1, 2, 4)[:, :, :, :real_heads]
        return x.reshape(batch, heads, width)

    if return_lse:
        out, lse = y
        return unpack(out, dim), unpack(lse[..., :1], 1)[..., 0]
    return unpack(y, dim)


def _block_id_spec():
    info = pltpu.get_tpu_info()
    return pl.BlockSpec(
        (None, info.num_sublanes, info.num_lanes), lambda b, *_: (b, 0, 0)
    )


def _block_id_data(batch, query_group):
    """Each parallel program receives its own ID in one native TC tile."""
    info = pltpu.get_tpu_info()
    tile = (info.num_sublanes, info.num_lanes)
    blocks = batch // query_group
    return jnp.broadcast_to(
        jnp.arange(blocks, dtype=jnp.int32)[:, None, None], (blocks, *tile)
    )
