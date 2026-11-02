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

from vllm_torchtpu.kernels.deepseek_v41.index_key_layout import DIM, ROW_WORDS, SLOTS
from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
    decode_scores,
    publication,
    streaming_topk,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.tuned_block_sizes import (
    DecodeTuningKey,
    PersistentScoreConfig,
    get_decode_request_tile,
    get_persistent_score_config,
)


def supports_decode(
    q: jax.Array,
    pool: jax.Array | tuple[jax.Array, jax.Array],
    records: jax.Array,
    page_records: int | None,
) -> bool:
    """Static envelope; the caller must also prove request-major live rows."""
    banks = pool if isinstance(pool, tuple) else (pool,)
    requests = records.shape[0]
    if not (
        requests > 0
        and q.shape[0] % requests == 0
        and 1 < q.shape[0] // requests <= 8
        and q.shape[1:] == (32, DIM)
        and q.dtype == jnp.bfloat16
        and 512 <= records.shape[1] <= 131072
        and records.dtype == jnp.int32
        and page_records is not None
        and page_records >= SLOTS
        and page_records % SLOTS == 0
        and 1 <= len(banks) <= 2
        and records.shape[1] % page_records == 0
        and all(
            bank.dtype == jnp.int32
            and bank.ndim == 2
            and bank.shape[1] == ROW_WORDS
            and bank.shape[0] % (page_records // SLOTS) == 0
            for bank in banks
        )
    ):
        return False
    assert page_records is not None
    width = 1 << (records.shape[1] - 1).bit_length()
    config = get_persistent_score_config(
        (q.shape[0] // requests, *q.shape[1:], width, len(banks))
    )
    return (
        page_records % (SLOTS * pltpu.get_tpu_info().num_sublanes) == 0
        and width % page_records == 0
        and config.compute_tile % page_records == 0
    )


def _select(scores, lengths, k, block_size, threshold):
    return streaming_topk.streaming_topk(
        scores,
        valid_lengths=lengths,
        k=k,
        block_size=block_size,
        reconstruct_threshold=threshold,
        exclude_neg_inf=True,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "page_records",
        "request_tile",
        "score_config",
        "k",
        "overlap",
        "topk_block",
        "topk_threshold",
        "candidate_count",
        "candidate_block_size",
    ),
)
def decode_indexer(
    q: jax.Array,
    weights: jax.Array,
    pool: jax.Array | tuple[jax.Array, ...],
    pages: jax.Array,
    lengths: jax.Array,
    *,
    page_records: int,
    request_tile: int | None = None,
    score_config: PersistentScoreConfig | None = None,
    k: int = 512,
    overlap: bool = True,
    topk_block: int = 8192,
    topk_threshold: int = 8192,
    candidate_count: int = 0,
    candidate_block_size: int = 1,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Return ascending indices, plus candidate blocks when requested.

    None selects a shape-tuned request tile.

    Tables target the default score/top-k options; ablations should pass an
    explicit request_tile. score_config overrides scorer tiles for benchmarking.
    Selection happens at tracing time, never on device.
    """
    if score_config is not None and request_tile is None:
        raise ValueError("score_config overrides require an explicit request_tile")
    requests, queries = q.shape[:2]
    width = pages.shape[1] * page_records
    if candidate_count and (
        candidate_block_size < 1
        or width % candidate_block_size
        or candidate_count >= width // candidate_block_size
    ):
        raise ValueError("Candidate publication requires fewer candidates than blocks")
    if page_records < 1:
        raise ValueError("page_records must be positive")
    pool = pool if isinstance(pool, tuple) else (pool,)
    if not 1 <= len(pool) <= 2:
        raise ValueError("Persistent decode requires one or two packed cache banks")
    fused_blocks = candidate_count > 0 and candidate_block_size == 8
    if request_tile is None:
        request_tile = get_decode_request_tile(
            DecodeTuningKey(
                device_kind=jax.devices()[0].device_kind,
                queries=queries,
                heads=q.shape[2],
                score_capacity=pages.shape[1] * page_records,
                page_records=page_records,
                k=k,
                banks=len(pool),
            ),
            requests,
        )
    elif type(request_tile) is not int or request_tile < 1:
        raise ValueError("request_tile must be a positive integer")
    request_tile = min(request_tile, requests)

    def prepare(i, size=request_tile):
        return jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, i * request_tile, size),
            (q, weights, pages, lengths),
        )

    def score(args):
        qr, wr, pr, lr = args
        return decode_scores.grouped_decode_scores(
            qr,
            wr,
            pool,
            pr,
            lr,
            page_records=page_records,
            score_config=score_config,
            block_size=candidate_block_size if fused_blocks else 0,
            disable_bounds_checks=True,
        )

    def select(i, values):
        block_scores = None
        if fused_blocks:
            values, block_scores = values
        group_size = math.prod(values.shape[:-1]) // queries
        row_lengths = jax.lax.dynamic_slice_in_dim(
            lengths, i * request_tile, group_size
        ).reshape(-1)
        selected = _select(
            values,
            row_lengths,
            k,
            topk_block,
            topk_threshold,
        )
        if not candidate_count:
            return selected
        blocks = width // candidate_block_size
        if not fused_blocks:
            block_scores = values.reshape(
                group_size * queries, blocks, candidate_block_size
            ).max(-1)
        assert block_scores is not None
        if publication.is_tuned_for_publish_candidates(
            block_scores, candidate_count, candidate_block_size
        ):
            return selected, publication.publish_candidates(
                block_scores,
                row_lengths,
                k=candidate_count,
                block_size=candidate_block_size,
            )
        last = (row_lengths - 1) // candidate_block_size
        block_scores = jnp.where(
            jnp.arange(blocks)[None] == last[:, None],
            jnp.inf,
            block_scores,
        )
        published = _select(
            block_scores,
            pl.cdiv(row_lengths, candidate_block_size),
            candidate_count,
            topk_block,
            topk_threshold,
        )
        return selected, published

    count = pl.cdiv(requests, request_tile)
    selected_groups = []
    previous = None
    if overlap:
        current = score(prepare(jnp.int32(0)))
        for i in range(count - 1):
            prepared = prepare(
                jnp.int32(i + 1),
                min(request_tile, requests - (i + 1) * request_tile),
            )
            if previous is None:
                prepared, current = jax.lax.optimization_barrier((prepared, current))
            else:
                prepared, current, previous = jax.lax.optimization_barrier(
                    (prepared, current, previous)
                )
                selected_groups.append(previous)
            following = score(prepared)
            previous = select(jnp.int32(i), current)
            current = following
        if previous is not None:
            previous, current = jax.lax.optimization_barrier((previous, current))
            selected_groups.append(previous)
        selected_groups.append(select(jnp.int32(count - 1), current))
    else:
        for i in range(count):
            prepared = prepare(
                jnp.int32(i), min(request_tile, requests - i * request_tile)
            )
            if previous is not None:
                prepared, previous = jax.lax.optimization_barrier((prepared, previous))
            previous = select(jnp.int32(i), score(prepared))
            selected_groups.append(previous)
    if count == 1:
        return selected_groups[0]
    return jax.tree.map(
        lambda *groups: jnp.concatenate(groups, axis=0), *selected_groups
    )
