# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-grouped paged scoring and score/selection overlap for decode.

Callers supply request-major queries. Candidate publication is optional;
serving routing and candidate filtering remain outside this API.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41 import indexer
from vllm_torchtpu.kernels.deepseek_v41.index_selection import scores, streaming_topk
from vllm_torchtpu.kernels.deepseek_v41.index_selection.tuned_block_sizes import (
    DecodeTuningKey,
    get_decode_request_tile,
)
from vllm_torchtpu.kernels.deepseek_v41.index_word_scores import DIM, ROW_WORDS, SLOTS


def supports_decode(
    q: jax.Array,
    pool: jax.Array | tuple[jax.Array, jax.Array],
    records: jax.Array,
    page_records: int | None,
) -> bool:
    """Static envelope; the caller must also prove request-major live rows."""
    banks = pool if isinstance(pool, tuple) else (pool,)
    requests = records.shape[0]
    return (
        requests > 1
        and q.shape[0] % requests == 0
        and 1 < q.shape[0] // requests <= 8
        and q.shape[1:] == (32, DIM)
        and q.dtype == jnp.bfloat16
        and 512 <= records.shape[1] <= 131072
        and records.dtype == jnp.int32
        and page_records is not None
        and 0 < page_records <= 4096
        and page_records % 16 == 0
        and 4096 % page_records == 0
        and records.shape[1] % page_records == 0
        and all(
            bank.dtype == jnp.int32
            and bank.ndim == 2
            and bank.shape[1] == ROW_WORDS
            and bank.shape[0] % (page_records // SLOTS) == 0
            for bank in banks
        )
    )


def _request_kernel(
    pages,
    limits,
    query_limits,
    q,
    weights,
    pool,
    out,
    words,
    *scratch,
    page_records,
    tile,
    coalesce,
    prefetch,
    profile_stages,
    fuse_mask,
    halfplane=False,
):
    request = pl.program_id(0)
    # Keep the scalar-prefetch address space flat: a 2D SMEM row subview
    # misaddresses long page tables after the first 128 columns on TPU.
    page_count = pages.shape[0] // limits.shape[0]
    scores._kernel(
        pages.at[pl.ds(request * page_count, page_count)],
        limits.at[pl.ds(request, 1)],
        q,
        weights,
        pool,
        None,
        out.at[:, 0, :],
        words,
        *scratch,
        page_records=page_records,
        tile=tile,
        coalesce=coalesce,
        prefetch=prefetch,
        profile_stages=profile_stages,
        row_lengths=query_limits if fuse_mask else None,
        row_offset=request * q.shape[0],
        halfplane=halfplane,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "page_records",
        "key_tile",
        "prefetch",
        "profile_stages",
        "interpret",
        "score_window",
        "fuse_mask",
        "halfplane",
    ),
)
def grouped_decode_scores(
    q: jax.Array,
    weights: jax.Array,
    pool: jax.Array | tuple[jax.Array, jax.Array],
    pages: jax.Array,
    lengths: jax.Array,
    *,
    page_records: int,
    key_tile: int,
    prefetch: bool = True,
    profile_stages: bool = False,
    interpret: bool = False,
    score_window: int = 32768,
    fuse_mask: bool = False,
    halfplane: bool = False,
) -> jax.Array:
    """Score split-order Q [requests, queries, heads, 128] against paged keys.

    Pages contain physical first-record addresses; -1 invalidates a whole page.
    Lengths [requests, queries] apply independently, including zero-length rows.
    """
    requests, queries, heads, dim = q.shape
    width = pages.shape[1] * page_records
    banks = pool if isinstance(pool, tuple) else (pool,)
    if (
        min(requests, queries, heads, page_records, key_tile, score_window) < 1
        or dim != DIM
        or q.dtype != jnp.bfloat16
        or weights.dtype != jnp.float32
        or weights.shape != (requests, queries, heads, 1)
        or pages.shape[0] != requests
        or pages.dtype != jnp.int32
        or lengths.shape != (requests, queries)
        or lengths.dtype != jnp.int32
        or page_records % 16
        or key_tile % page_records
        or width % key_tile
        or any(
            bank.dtype != jnp.int32
            or bank.shape[1] != ROW_WORDS
            or bank.shape[0] % (page_records // SLOTS)
            for bank in banks
        )
    ):
        raise ValueError("Expected aligned paged BF16 request-major decode inputs")
    window = math.gcd(width, max(key_tile, score_window))
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    result = pl.pallas_call(
        functools.partial(
            _request_kernel,
            page_records=page_records,
            tile=key_tile,
            coalesce=True,
            prefetch=prefetch,
            profile_stages=profile_stages,
            fuse_mask=fuse_mask,
            halfplane=halfplane,
        ),
        out_shape=jax.ShapeDtypeStruct((requests * queries, 1, width), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(requests, width // window),
            in_specs=(
                pl.BlockSpec((queries, heads, DIM), lambda i, j, *_: (i, 0, 0)),
                pl.BlockSpec((queries, heads, 1), lambda i, j, *_: (i, 0, 0)),
                (hbm, hbm) if isinstance(pool, tuple) else hbm,
            ),
            out_specs=pl.BlockSpec((queries, 1, window), lambda i, j, *_: (i, 0, j)),
            scratch_shapes=(
                pltpu.VMEM((2, key_tile // SLOTS, ROW_WORDS), jnp.int32),
                *(
                    (pltpu.VMEM((key_tile, ROW_WORDS), jnp.int32),)
                    if not halfplane
                    else ()
                ),
                pltpu.SemaphoreType.DMA,
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
        ),
        interpret=interpret,
        name="decode_grouped_paged_index_scores",
    )(
        pages.reshape(-1),
        jnp.max(lengths, axis=1),
        lengths.reshape(-1),
        q.reshape(-1, heads, DIM),
        weights.reshape(-1, heads, 1),
        pool,
    ).reshape(requests, queries, width)
    if fuse_mask:
        return result.reshape(requests * queries, width)
    valid = jnp.arange(width)[None, None] < lengths[:, :, None]
    valid &= jnp.repeat(pages >= 0, page_records, axis=1)[:, None]
    return jnp.where(valid, result, -jnp.inf).reshape(requests * queries, width)


def _select(scores, lengths, k, block_size, threshold, fuse_validity):
    selected = streaming_topk.streaming_topk(
        scores,
        valid_lengths=lengths,
        k=k,
        block_size=block_size,
        reconstruct_threshold=threshold,
        exclude_neg_inf=fuse_validity,
    )
    valid = selected >= 0
    if not fuse_validity:
        values = jnp.take_along_axis(scores, jnp.maximum(selected, 0), axis=1)
        valid &= values > -jnp.inf
    ordered = jnp.sort(jnp.where(valid, selected, scores.shape[1]), axis=1)
    return jnp.where(ordered < scores.shape[1], ordered, -1)


@functools.partial(
    jax.jit,
    static_argnames=(
        "page_records",
        "key_tile",
        "request_tile",
        "k",
        "overlap",
        "prefetch",
        "profile_stages",
        "score_window",
        "fuse_mask",
        "halfplane",
        "fuse_validity",
        "topk_block",
        "topk_threshold",
        "candidate_count",
        "candidate_block_size",
    ),
)
def decode_indexer(
    q: jax.Array,
    weights: jax.Array,
    pool: jax.Array | tuple[jax.Array, jax.Array],
    pages: jax.Array,
    lengths: jax.Array,
    *,
    page_records: int,
    key_tile: int,
    request_tile: int | None = None,
    k: int = 512,
    overlap: bool = True,
    prefetch: bool = True,
    profile_stages: bool = False,
    score_window: int = 131072,
    fuse_mask: bool = True,
    halfplane: bool = False,
    fuse_validity: bool = True,
    topk_block: int = 8192,
    topk_threshold: int = 8192,
    candidate_count: int = 0,
    candidate_block_size: int = 1,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Return ascending indices, plus candidate blocks when requested.

    None selects a shape-tuned request tile.

    Tables target the default score/top-k options; ablations should pass an
    explicit request_tile. Selection happens at tracing time, never on device.
    """
    requests, queries = q.shape[:2]
    width = pages.shape[1] * page_records
    if candidate_count and (
        candidate_block_size < 1
        or width % candidate_block_size
        or candidate_count >= width // candidate_block_size
    ):
        raise ValueError("Candidate publication requires fewer candidates than blocks")
    if request_tile is None:
        request_tile = get_decode_request_tile(
            DecodeTuningKey(
                device_kind=jax.devices()[0].device_kind,
                requests=requests,
                queries=queries,
                heads=q.shape[2],
                score_capacity=pages.shape[1] * page_records,
                page_records=page_records,
                key_tile=key_tile,
                k=k,
                banks=len(pool) if isinstance(pool, tuple) else 1,
            )
        )
    if type(request_tile) is not int or request_tile < 1 or requests % request_tile:
        raise ValueError("request_tile must divide the request-major batch")

    def prepare(i):
        return jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, i * request_tile, request_tile),
            (q, weights, pages, lengths),
        )

    def score(args):
        qr, wr, pr, lr = args
        return grouped_decode_scores(
            qr,
            wr,
            pool,
            pr,
            lr,
            page_records=page_records,
            key_tile=key_tile,
            prefetch=prefetch,
            profile_stages=profile_stages,
            score_window=score_window,
            fuse_mask=fuse_mask,
            halfplane=halfplane,
        )

    def select(i, values):
        row_lengths = prepare(i)[-1].reshape(-1)
        selected = _select(
            values,
            row_lengths,
            k,
            topk_block,
            topk_threshold,
            fuse_validity,
        )
        if not candidate_count:
            return selected
        blocks = width // candidate_block_size
        block_scores = values.reshape(-1, blocks, candidate_block_size).max(-1)
        last = (row_lengths - 1) // candidate_block_size
        block_scores = jnp.where(
            jnp.arange(blocks)[None] == last[:, None], jnp.inf, block_scores
        )
        published = _select(
            block_scores,
            pl.cdiv(row_lengths, candidate_block_size),
            candidate_count,
            topk_block,
            topk_threshold,
            fuse_validity,
        )
        return selected, published

    def write(i, selected, out):
        return jax.tree.map(
            lambda target, value: jax.lax.dynamic_update_slice_in_dim(
                target, value, i * request_tile * queries, axis=0
            ),
            out,
            selected,
        )

    out = jnp.full((requests * queries, k), -1, jnp.int32)
    if candidate_count:
        out = (out, jnp.full((requests * queries, candidate_count), -1, jnp.int32))
    count = requests // request_tile
    if overlap:
        return indexer.pipeline_score_topk(count, prepare, score, select, write, out)

    def serial(i, out):
        args, out = jax.lax.optimization_barrier((prepare(i), out))
        selected = select(i, score(args))
        return write(i, selected, out)

    return jax.lax.fori_loop(0, count, serial, out)
