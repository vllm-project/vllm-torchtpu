# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request unions and reusable index keys for candidate-filtered decode.

The caller owns cache lifetime and falls back when any union overflows. IDs are
ascending logical blocks; membership is applied before selection for every Q.
"""

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41 import indexer
from vllm_torchtpu.kernels.deepseek_v41.index_selection import decode, tuned_block_sizes


class UnionMetadata(NamedTuple):
    ids: jax.Array
    membership: jax.Array
    actual_count: jax.Array
    overflow: jax.Array


def _sorted_union_metadata(
    candidates: jax.Array, logical_blocks: int, capacity_blocks: int
) -> UnionMetadata:
    requests, queries, _ = candidates.shape
    valid = (candidates >= 0) & (candidates < logical_blocks)
    # Include query ownership in the sort key so duplicate IDs within one
    # query contribute its membership bit only once.
    encoded = jnp.where(
        valid,
        candidates * queries + jnp.arange(queries)[None, :, None],
        logical_blocks * queries,
    )
    encoded = jnp.sort(encoded.reshape(requests, -1), axis=1)
    blocks = encoded // queries
    valid = blocks < logical_blocks
    previous = jnp.pad(encoded[:, :-1], ((0, 0), (1, 0)), constant_values=-1)
    first = valid & (blocks != previous // queries)
    count = first.sum(1).astype(jnp.int32)
    slots = jnp.where(valid, jnp.cumsum(first, axis=1) - 1, capacity_blocks)
    rows = jnp.arange(requests)[:, None]
    ids = jnp.full((requests, capacity_blocks), -1, jnp.int32)
    ids = ids.at[rows, slots].max(jnp.where(valid, blocks, -1), mode="drop")
    bits = jnp.where(valid & (encoded != previous), 1 << (encoded % queries), 0)
    membership = jnp.zeros_like(ids).at[rows, slots].add(bits, mode="drop")
    return UnionMetadata(ids, membership, count, count > capacity_blocks)


def union_metadata(
    candidates: jax.Array, *, logical_blocks: int, capacity_blocks: int
) -> UnionMetadata:
    """Compact [request, query, candidate] IDs without losing overflow evidence."""
    requests, queries, _ = candidates.shape
    if not 1 <= queries <= 8 or min(logical_blocks, capacity_blocks) < 1:
        raise ValueError("Union requires 1..8 queries and positive block capacities")
    # Preserve the calibrated dense route through 128K index records. For
    # longer contexts, bound metadata work by candidate count, not ID space.
    if logical_blocks > 131072 // 8:
        return _sorted_union_metadata(candidates, logical_blocks, capacity_blocks)
    valid = (candidates >= 0) & (candidates < logical_blocks)
    safe = jnp.where(valid, candidates, logical_blocks)
    presence = jnp.zeros((requests, queries, logical_blocks + 1), jnp.int32)
    presence = presence.at[
        jnp.arange(requests)[:, None, None], jnp.arange(queries)[None, :, None], safe
    ].max(valid.astype(jnp.int32))
    bits = (presence[:, :, :-1] << jnp.arange(queries)[None, :, None]).sum(1)
    count = (bits != 0).sum(1).astype(jnp.int32)
    ids = jax.vmap(
        lambda row: jnp.nonzero(row, size=capacity_blocks, fill_value=-1)[0]
    )(bits)
    membership = jnp.take_along_axis(bits, jnp.maximum(ids, 0), axis=1)
    return UnionMetadata(
        ids, jnp.where(ids >= 0, membership, 0), count, count > capacity_blocks
    )


def union_record_starts(
    metadata: UnionMetadata,
    records: jax.Array,
    *,
    block_size: int,
    pool_records: int,
) -> tuple[jax.Array, jax.Array]:
    """Resolve logical blocks to aligned physical runs; invalidate bad pages."""
    logical = jnp.maximum(metadata.ids, 0) * block_size
    starts = jnp.take_along_axis(records, jnp.minimum(logical, records.shape[1] - 1), 1)
    valid = (metadata.ids >= 0) & (logical < records.shape[1])
    valid &= (starts >= 0) & (starts + block_size <= pool_records)
    # D128 BF16 packs two records into one pool word row. Logical blocks
    # need not start at a multiple of block_size in the physical allocation.
    valid &= starts % 2 == 0
    return jnp.where(valid, starts, -1), jnp.where(valid, metadata.membership, 0)


def _unpack_words(words, scratch):
    rows = words.shape[0]
    scratch[pl.ds(0, rows, stride=2), :] = words
    scratch[pl.ds(1, rows, stride=2), :] = pltpu.roll(words, 64, 1)
    words = scratch[...]
    lo = jax.lax.bitcast_convert_type(words << 16, jnp.float32).astype(jnp.bfloat16)
    hi = jax.lax.bitcast_convert_type(words & jnp.int32(-65536), jnp.float32).astype(
        jnp.bfloat16
    )
    return jnp.where(
        jax.lax.broadcasted_iota(jnp.int32, words.shape, 1) < 64,
        lo,
        pltpu.roll(hi, 64, 1),
    )


def _unpack_kernel(words_ref, out_ref, *scratch, halfplane=False):
    out_ref[0] = (
        pltpu.bitcast(words_ref[0], jnp.bfloat16)
        if halfplane
        else _unpack_words(words_ref[0], scratch[0])
    )


@functools.partial(jax.jit, static_argnames=("interpret", "halfplane"))
def unpack_candidate_words(
    words: jax.Array, *, interpret: bool = False, halfplane: bool = False
) -> jax.Array:
    """Unpack native word pairs once into split-order BF16 [request, key, 128]."""
    groups, rows, dim = words.shape
    if dim != 128 or words.dtype != jnp.int32 or rows < 1:
        raise ValueError("Expected nonempty int32 [requests, word rows, 128]")
    tile = math.gcd(rows, 1024)
    return pl.pallas_call(
        functools.partial(_unpack_kernel, halfplane=halfplane),
        out_shape=jax.ShapeDtypeStruct((groups, rows * 2, dim), jnp.bfloat16),
        grid=(groups, rows // tile),
        in_specs=[pl.BlockSpec((1, tile, dim), lambda g, t: (g, t, 0))],
        out_specs=pl.BlockSpec((1, tile * 2, dim), lambda g, t: (g, t, 0)),
        scratch_shapes=[] if halfplane else [pltpu.VMEM((tile * 2, dim), jnp.int32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",) * 2),
        interpret=interpret,
        name="candidate_unpack_cache",
    )(words)


def _score_kernel(
    q,
    weights,
    keys,
    ids,
    membership,
    lengths,
    out,
    *scratch,
    packed=False,
    halfplane=False,
):
    queries, heads, dim = q.shape[1:]
    kk = (
        (
            pltpu.bitcast(keys[0], jnp.bfloat16)
            if halfplane
            else _unpack_words(keys[0], scratch[0])
        )
        if packed
        else keys[0]
    )
    logits = jax.lax.dot_general(
        q[0].reshape(queries * heads, dim),
        kk,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).reshape(queries, heads, kk.shape[0])
    values = (jnp.maximum(logits, 0) * weights[0][:, :, None]).sum(1)
    logical = ids[0, 0]
    allow = ((membership[0, 0][None, :] >> jnp.arange(queries)[:, None]) & 1) != 0
    allow &= (logical[None, :] >= 0) & (logical[None, :] < lengths[0, 0][:, None])
    out[0] = jnp.where(allow, values, -jnp.inf)


@functools.partial(
    jax.jit, static_argnames=("block_size", "key_tile", "interpret", "halfplane")
)
def candidate_scores(
    q: jax.Array,
    weights: jax.Array,
    keys: jax.Array,
    ids: jax.Array,
    membership: jax.Array,
    lengths: jax.Array,
    *,
    block_size: int,
    key_tile: int = 4096,
    interpret: bool = False,
    halfplane: bool = False,
) -> jax.Array:
    """Per-head dot/ReLU/weight/sum with per-query membership and causal masks."""
    groups, queries, heads, dim = q.shape
    width = ids.shape[1] * block_size
    packed = keys.dtype == jnp.int32
    key_rows = width // 2 if packed else width
    if (
        dim != 128
        or q.dtype != jnp.bfloat16
        or keys.dtype not in (jnp.bfloat16, jnp.int32)
        or weights.dtype != jnp.float32
        or weights.shape != (groups, queries, heads)
        or keys.shape != (groups, key_rows, dim)
        or membership.shape != ids.shape
        or lengths.shape != (groups, queries)
        or key_tile < 1
        or width % key_tile
        or (packed and key_tile % 2)
    ):
        raise ValueError("Expected aligned BF16 or packed int32 candidate inputs")
    logical = jnp.repeat(ids, block_size, 1) * block_size
    logical += jnp.arange(width)[None, :] % block_size
    logical = jnp.where(jnp.repeat(ids >= 0, block_size, 1), logical, -1)
    members = jnp.repeat(membership, block_size, 1)
    return pl.pallas_call(
        functools.partial(_score_kernel, packed=packed, halfplane=halfplane),
        out_shape=jax.ShapeDtypeStruct((groups, queries, width), jnp.float32),
        grid=(groups, width // key_tile),
        in_specs=[
            pl.BlockSpec((1, queries, heads, dim), lambda g, t: (g, 0, 0, 0)),
            pl.BlockSpec((1, queries, heads), lambda g, t: (g, 0, 0)),
            pl.BlockSpec(
                (1, key_tile // 2 if packed else key_tile, dim),
                lambda g, t: (g, t, 0),
            ),
            pl.BlockSpec((1, 1, key_tile), lambda g, t: (g, 0, t)),
            pl.BlockSpec((1, 1, key_tile), lambda g, t: (g, 0, t)),
            pl.BlockSpec((1, 1, queries), lambda g, t: (g, 0, 0)),
        ],
        out_specs=pl.BlockSpec((1, queries, key_tile), lambda g, t: (g, 0, t)),
        scratch_shapes=[pltpu.VMEM((key_tile, dim), jnp.int32)]
        if packed and not halfplane
        else [],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",) * 2),
        interpret=interpret,
        name="candidate_union_packed_scores" if packed else "candidate_union_scores",
    )(q, weights, keys, logical[:, None], members[:, None], lengths[:, None])


@functools.partial(
    jax.jit,
    static_argnames=(
        "block_size",
        "key_tile",
        "request_tile",
        "k",
        "overlap",
        "halfplane",
    ),
)
def candidate_indexer(
    q: jax.Array,
    weights: jax.Array,
    keys: jax.Array,
    ids: jax.Array,
    membership: jax.Array,
    lengths: jax.Array,
    *,
    block_size: int,
    key_tile: int = 4096,
    request_tile: int | None = None,
    k: int = 512,
    overlap: bool = True,
    halfplane: bool = False,
) -> jax.Array:
    """Pipeline whole-request score groups and SC selection over a reusable union."""
    requests, queries = q.shape[:2]
    width = ids.shape[1] * block_size
    if request_tile is None:
        request_tile = tuned_block_sizes.get_candidate_request_tile(
            tuned_block_sizes.CandidateTuningKey(
                jax.devices()[0].device_kind,
                requests,
                queries,
                q.shape[2],
                width,
                key_tile,
                k,
                keys.dtype == jnp.int32,
            )
        )
    if request_tile < 1 or requests % request_tile:
        raise ValueError("request_tile must divide the request-major batch")

    def prepare(i):
        return jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, i * request_tile, request_tile),
            (q, weights, keys, ids, membership, lengths),
        )

    def score(args):
        return candidate_scores(
            *args, block_size=block_size, key_tile=key_tile, halfplane=halfplane
        ).reshape(request_tile * queries, width)

    def select(i, values):
        group_ids = prepare(i)[3]
        selected = decode._select(
            values,
            jnp.full((request_tile * queries,), width, jnp.int32),
            k,
            8192,
            8192,
            True,
        ).reshape(request_tile, queries, k)
        blocks = jnp.take_along_axis(
            group_ids[:, None, :], jnp.maximum(selected, 0) // block_size, axis=2
        )
        logical = blocks * block_size + selected % block_size
        return jnp.where(selected >= 0, logical, -1).reshape(-1, k)

    def write(i, selected, out):
        return jax.lax.dynamic_update_slice_in_dim(
            out, selected, i * request_tile * queries, axis=0
        )

    out = jnp.full((requests * queries, k), -1, jnp.int32)
    count = requests // request_tile
    if overlap:
        return indexer.pipeline_score_topk(count, prepare, score, select, write, out)

    def serial(i, out):
        args, out = jax.lax.optimization_barrier((prepare(i), out))
        return write(i, select(i, score(args)), out)

    return jax.lax.fori_loop(0, count, serial, out)


def cache_capacity(
    query_shape: tuple[int, ...],
    record_shape: tuple[int, ...],
    candidate_count: int,
    block_size: int,
    page_records: int,
) -> int | None:
    """Static envelope without specializing Torch's dynamic token dimension.

    Request/query alignment is checked after JAX lowering, where bucket shapes
    are concrete. Python guards on ``rows == requests * 6`` break Dynamo's
    explicitly dynamic input contract.
    """
    _, width = record_shape
    _, heads, dim = query_shape
    capacity = tuned_block_sizes.CANDIDATE_KEY_CAPACITY
    if (
        (heads, dim) != (32, 128)
        or (candidate_count, block_size) != (2048, 8)
        or not capacity < width <= 1048576
        or page_records < 16
        or page_records % block_size
        or width % page_records
    ):
        return None
    return capacity
