# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 candidate metadata: SparseCore bitmaps against a NumPy reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8 import candidates
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import (
    PackedFP8Layout,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_select import (
    _publish_candidate_blocks,
)

BLOCK, PER_QUERY = 8, 2048
# (request sizes, invalid-request rows, half-width page table). Odd-sized
# requests leave single queries unpaired; the narrow table invalidates IDs
# past its last column.
CASES = {
    "edge": ([2, 1, 3, 4, 2, 5, 2, 6, 1, 2, 2, 4], 3, True),
    "decode": ([6] * 48, 0, False),
    "prefill": ([256], 0, False),
}
# Score widths with their page sizes; 1M-token contexts at ratios 2 and 1, and
# one block per page.
WIDTHS = [(98816, 256), (524288, 128), (1048576, 256), (16384, 8)]


def _rows(rng, requests, blocks):
    """Ascending -1-suffixed rows; queries of a request share most blocks."""
    rows = []
    for index, request in enumerate(requests):
        shared = np.random.default_rng(max(request, 0)).choice(blocks, 1434, False)
        full = np.unique(np.concatenate((shared, rng.choice(blocks, 614, False))))
        row = (
            full,
            full[: rng.integers(1, PER_QUERY)],
            full[:0],
            np.sort(rng.choice(shared[:1000], PER_QUERY)),  # adjacent repeats
            np.unique(rng.integers(blocks - 1000, blocks + 1000, PER_QUERY)),
            np.arange(blocks - rng.integers(PER_QUERY // 2, PER_QUERY), blocks),
        )[index % 6]
        rows.append(np.pad(row, (0, PER_QUERY - row.size), constant_values=-1))
    return np.asarray(rows, np.int32)


def _reference(candidates, table, requests, offsets, blocks, per_page, cache_pages):
    """Per-query windows, per-pair physical unions and spans in NumPy."""
    tokens, per_query = candidates.shape
    columns = table.shape[1]
    ids = np.full((tokens, per_query), -1, np.int32)
    union = np.full((tokens, 2 * per_query), -1, np.int32)
    spans = np.zeros((tokens, 2), np.int32)

    def valid(row, request):
        if not 0 <= request < len(table):
            return row[:0]
        row = np.unique(row[(row >= 0) & (row < blocks)])
        column = row // per_page
        page = np.where(
            column < columns, table[request, np.minimum(column, columns - 1)], -1
        )
        return row[(page >= 0) & (page < cache_pages)]

    for pair, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        if start == end:
            continue
        request = requests[start]
        a = valid(candidates[start], request)
        b = valid(candidates[start + 1], request) if end - start == 2 else a[:0]
        a_only, shared, b_only = (
            np.setdiff1d(a, b),
            np.intersect1d(a, b),
            np.setdiff1d(b, a),
        )
        windows = (np.concatenate((a_only, shared)), np.concatenate((shared, b_only)))
        for member in range(end - start):
            ids[start + member, : windows[member].size] = windows[member]
        spans[start] = 0, a_only.size
        if end - start == 2:
            spans[start + 1] = a_only.size, shared.size
        blocks_of_pair = np.concatenate((a_only, shared, b_only))
        page = table[request, blocks_of_pair // per_page]
        union[pair, : blocks_of_pair.size] = page * per_page + blocks_of_pair % per_page
    return ids, union, spans


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("width,page_size", WIDTHS)
@pytest.mark.parametrize("case", CASES)
def test_metadata_matches_reference(case, width, page_size, ascending):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore candidate metadata requires TPU")
    sizes, invalid, narrow = CASES[case]
    rng = np.random.default_rng(7)
    requests = np.repeat(np.arange(-1, len(sizes)), [invalid, *sizes])
    requests = np.roll(requests, -invalid).astype(np.int32)
    columns = -(-width // page_size) // (2 if narrow else 1)
    layout = PackedFP8Layout(len(sizes) * columns + 8, page_size=page_size)
    table = rng.permutation(layout.pages)[: len(sizes) * columns]
    table = table.reshape(len(sizes), columns).astype(np.int32)
    bad = rng.random(table.shape) < 0.02
    table[bad] = np.where(rng.random(bad.sum()) < 0.5, -1, layout.pages + 3)
    if narrow:
        table[-2] = -1  # a request whose pages are all invalid
    rows = _rows(rng, requests, width // BLOCK)
    # Unordered rows, padding included, take the sorting path.
    inputs = rows if ascending else rng.permuted(rows, axis=1)

    got = candidates.prepare_candidate_metadata(
        jnp.asarray(table),
        jnp.asarray(inputs),
        jnp.asarray(requests),
        layout,
        BLOCK,
        width,
        ascending=ascending,
    )
    want = _reference(
        rows,
        table,
        requests,
        np.asarray(got.offsets),
        width // BLOCK,
        page_size // BLOCK,
        layout.pages,
    )
    for name, value, expected in zip(
        ("candidate_block_ids", "physical_block_ids", "block_spans"),
        (got.candidate_block_ids, got.physical_block_ids, got.block_spans),
        want,
    ):
        np.testing.assert_array_equal(np.asarray(value), expected, err_msg=name)


# Single- and two-stage SparseCore top-k, and a count past its 4096 limit, which
# takes `lax.top_k`.
@pytest.mark.parametrize(
    "rows,width,count",
    [(288, 131072, PER_QUERY), (16, 131072, PER_QUERY), (2, 67584, 4224)],
)
def test_publication_ascends(rows, width, count):
    """Candidate metadata relies on ascending, -1-suffixed published rows."""
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore publication requires TPU")
    rng = np.random.default_rng(11)
    scores = jnp.asarray(rng.normal(size=(rows, width)), jnp.float32)
    lengths = jnp.asarray(rng.integers(0, width + 1, rows), jnp.int32)
    published = _publish_candidate_blocks(scores, lengths, count, BLOCK, True)
    for row in np.asarray(published):
        live = row[row >= 0]
        assert np.all(row[live.size :] == -1) and np.all(np.diff(live) > 0)
