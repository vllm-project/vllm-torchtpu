# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Packed FP8 decode selection: ragged groups, padding rows, short batches."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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

TOKENS, WIDTH, TOPK, PAGE = 288, 16384, 512, 256


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


def _select(rng, counts, padding, short):
    """Request-major rows, then `padding` rows of request -1; returns inputs too."""
    requests = np.concatenate(
        (np.repeat(np.arange(len(counts)), counts), [-1] * padding)
    )
    low, high = (0, TOPK) if short else (TOPK + 1, WIDTH)
    lengths = np.where(requests >= 0, rng.integers(low, high + 1, TOKENS), 0)
    lengths = jnp.asarray(lengths, jnp.int32)
    layout = PackedFP8Layout(512, page_size=PAGE)
    keys = jax.random.normal(
        jax.random.key(3), (layout.pages * PAGE, 128), jnp.bfloat16
    )
    pool = _pack(keys, layout)
    table = rng.integers(0, layout.pages, (len(counts), WIDTH // PAGE)).astype(np.int32)
    q = jnp.asarray(rng.normal(size=(TOKENS, 32, 128)), jnp.float8_e4m3fn)
    weights = jnp.asarray(rng.uniform(0.01, 0.1, (TOKENS, 32)), jnp.float32)
    # As query_start_loc: trailing padding rows are uncounted.
    offsets = jnp.asarray(np.concatenate(([0], np.cumsum(counts))), jnp.int32)
    layout_args = (layout.pages, layout.page_stride, layout.page_offset, PAGE)
    select = jax.jit(
        lambda q, w, n: select_fp8_indexer_jax(
            q,
            (pool,),
            jnp.asarray(table),
            w,
            n,
            jnp.empty((TOKENS, 0), jnp.int32),
            jnp.asarray(requests),
            n.max().reshape(1),
            offsets,
            layout_args,
            TOPK,
            8,
            False,
            False,
            True,
            WIDTH,
        )[0]
    )
    selected = np.asarray(select(q, weights, lengths))
    return selected, (q, pool, weights, lengths, jnp.asarray(table), offsets, layout)


@pytest.mark.parametrize("short", [False, True], ids=["long", "short_batch"])
def test_ragged_requests_select_top_scores(short):
    """Requests that are not whole six-query groups, then padding rows."""
    if jax.default_backend() != "tpu":
        pytest.skip("Packed FP8 selection requires TPU")
    rng, padding = np.random.default_rng(11), 6
    counts = rng.integers(1, 10, 64)
    counts = counts[np.cumsum(counts) <= TOKENS - padding]
    counts[-1] += TOKENS - padding - counts.sum()
    selected, (q, pool, weights, lengths, table, offsets, layout) = _select(
        rng, counts, padding, short
    )
    real, lengths = int(counts.sum()), np.asarray(lengths)
    assert (selected[real:] == -1).all()
    if short:
        # Every visible key, as in the reference short-context bypass.
        for row in range(real):
            ids = np.sort(selected[row][selected[row] >= 0])
            np.testing.assert_array_equal(ids, np.arange(lengths[row]))
        return
    scores = np.asarray(
        fp8_paged_mqa_logits(
            q[:real],
            pool,
            weights[:real],
            jnp.asarray(lengths[:real]),
            table,
            query_start_loc=offsets,
            max_model_len=WIDTH,
            q_scale=jnp.ones_like(weights[:real]),
            query_size=6,
            cache_layout=layout,
        )
    )
    for row in range(real):
        ids = selected[row][selected[row] >= 0]
        assert np.unique(ids).size == ids.size == TOPK
        assert (ids < lengths[row]).all()
        # Split groups may round differently from the reference tiling.
        kth = np.sort(scores[row, : lengths[row]])[-TOPK]
        tolerance = 2e-2 * np.abs(scores[row, : lengths[row]]).max()
        assert scores[row, ids].min() >= kth - tolerance
