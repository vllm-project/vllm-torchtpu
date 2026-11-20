# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate selection pairs the queries of odd-sized requests and padding rows.

`prepare_candidate_metadata` pairs consecutive queries inside each run of equal
request IDs, so a step has at most `candidate_group_capacity` query groups and
the gathered candidate K holds one region per group. The first test checks the
grouping on the host. The second runs the candidate selection on ragged
requests (odd and even sizes, then padding rows) and checks that every query
selects the top scores of its own candidate tokens, against dense scores of the
whole context. TPU only for the second test (SparseCore gather).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8 import candidates
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import (
    PAGE_ROWS,
    PackedFP8Layout,
    quantize_index_rows,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.dense_logits import (
    fp8_paged_mqa_logits,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_fp8 import (
    select_fp8_indexer_jax,
)

# Ragged requests (odd and even), then padding rows; 4 page-table slots.
COUNTS, PADDING, SLOTS = [255, 101, 150], 6, 4
TOKENS = sum(COUNTS) + PADDING
WIDTH, TOPK, PAGE, BLOCK, CANDIDATES = 16384, 512, 256, 8, 256


def test_group_capacity_bounds_ragged_runs():
    assert candidates.candidate_group_capacity(8, 3) == 7
    assert candidates.candidate_group_capacity(2048, 8) == 1032
    assert candidates.candidate_group_capacity(4, 8) == 4
    requests = np.concatenate(
        (np.repeat(np.arange(len(COUNTS)), COUNTS), [-1] * PADDING)
    )
    rows = np.arange(TOKENS)
    first = np.array([np.flatnonzero(requests == requests[r])[0] for r in rows])
    length = np.array([np.sum(requests == requests[r]) for r in rows])
    groups = sum((n + 1) // 2 for n in COUNTS) + (PADDING + 1) // 2
    assert groups <= candidates.candidate_group_capacity(TOKENS, SLOTS)
    # The pairing rule of prepare_candidate_metadata, on the host.
    paired = (rows - first) < length - length % 2
    starts = ~paired | ((rows - first) % 2 == 0)
    assert starts.sum() == groups


def _pack(keys, layout):
    """Feature-major pages: field f of record r at row-word f * page_size + r."""
    values, scales, _ = quantize_index_rows(keys)
    bits = np.asarray(jax.lax.bitcast_convert_type(values, jnp.uint8))
    bits = bits.astype(np.uint32).reshape(-1, 32, 4)
    words = bits[..., 0] | bits[..., 1] << 8 | bits[..., 2] << 16 | bits[..., 3] << 24
    fields = np.concatenate(
        (words.view(np.int32), np.asarray(scales).view(np.int32)[:, None]), axis=1
    )
    records = np.arange(len(keys))
    words_at = (records // PAGE * layout.page_stride * 128)[:, None]
    words_at = words_at + np.arange(PAGE_ROWS)[None, :] * PAGE
    words_at = words_at + (records % PAGE)[:, None]
    pool = np.zeros(layout.pages * layout.page_stride * 128, np.int32)
    pool[words_at] = fields
    return jnp.asarray(pool.reshape(-1, 128))


def test_ragged_candidate_selection_selects_top_candidate_scores():
    if jax.default_backend() != "tpu":
        pytest.skip("Candidate selection requires TPU")
    rng = np.random.default_rng(5)
    requests = np.concatenate(
        (np.repeat(np.arange(len(COUNTS)), COUNTS), [-1] * PADDING)
    ).astype(np.int32)
    real = sum(COUNTS)
    lengths = np.where(
        requests >= 0, rng.integers(WIDTH // 2, WIDTH + 1, TOKENS), 0
    ).astype(np.int32)
    layout = PackedFP8Layout(WIDTH // PAGE * SLOTS + 8, page_size=PAGE)
    keys = jax.random.normal(
        jax.random.key(1), (layout.pages * PAGE, 128), jnp.bfloat16
    )
    pool = _pack(keys, layout)
    table = np.stack(
        [rng.permutation(layout.pages)[: WIDTH // PAGE] for _ in range(SLOTS)]
    ).astype(np.int32)
    order = np.argsort(rng.random((TOKENS, WIDTH // BLOCK)), axis=1)
    cands = np.sort(order[:, :CANDIDATES], axis=1).astype(np.int32)
    q = jnp.asarray(rng.normal(size=(TOKENS, 32, 128)), jnp.float8_e4m3fn)
    weights = jnp.asarray(rng.uniform(0.01, 0.1, (TOKENS, 32)), jnp.float32)
    offsets = np.concatenate(([0], np.cumsum(COUNTS), [real] * (SLOTS - len(COUNTS))))
    offsets = jnp.asarray(offsets, jnp.int32)
    layout_args = (layout.pages, layout.page_stride, layout.page_offset, PAGE)
    selected = np.asarray(
        jax.jit(
            lambda q, w, n, c: select_fp8_indexer_jax(
                q,
                (pool,),
                jnp.asarray(table),
                w,
                n,
                c,
                jnp.asarray(requests),
                n.max().reshape(1),
                offsets,
                layout_args,
                TOPK,
                BLOCK,
                False,
                True,
                True,
                WIDTH,
            )[0]
        )(q, weights, jnp.asarray(lengths), jnp.asarray(cands))
    )
    assert (selected[real:] == -1).all()
    scores = np.asarray(
        fp8_paged_mqa_logits(
            q[:real],
            pool,
            weights[:real],
            jnp.asarray(lengths[:real]),
            jnp.asarray(table),
            query_start_loc=offsets,
            max_model_len=WIDTH,
            q_scale=jnp.ones_like(weights[:real]),
            query_size=6,
            cache_layout=layout,
        )
    )
    for row in range(real):
        ids = selected[row][selected[row] >= 0]
        tokens = (cands[row][:, None] * BLOCK + np.arange(BLOCK)).reshape(-1)
        tokens = tokens[tokens < lengths[row]]
        assert np.unique(ids).size == ids.size == min(TOPK, tokens.size)
        assert np.isin(ids, tokens).all()
        kth = np.sort(scores[row, tokens])[-ids.size]
        tolerance = 2e-2 * np.abs(scores[row, tokens]).max()
        assert scores[row, ids].min() >= kth - tolerance
