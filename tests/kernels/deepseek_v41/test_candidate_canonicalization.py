# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XLA candidate canonicalization against a set-based reference."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.candidates import (
    _canonical_candidate_blocks_ref,
)


def _expected(candidates, table, block, width, cache_pages, page_size):
    rows, _, per_query = candidates.shape
    parts, fragment = max(1, block // page_size), min(block, page_size)
    per_page = page_size // fragment

    def fragments(row, logical):
        out = []
        for part in range(parts):
            unit = logical * parts + part
            column = unit // per_page
            page = table[row, column] if 0 <= column < table.shape[1] else -1
            ok = 0 <= page < cache_pages
            out.append(page * per_page + unit % per_page if ok else -1)
        return out

    def valid(row, logical):
        if not 0 <= logical < -(-width // block):
            return False
        return all(
            fragment_id >= 0 or logical * block + part * fragment >= width
            for part, fragment_id in enumerate(fragments(row, logical))
        )

    ids = np.full((rows, 2 * per_query), -1, np.int32)
    physical = np.full((rows, 2 * per_query * parts), -1, np.int32)
    counts = np.zeros((rows, 3), np.int32)
    for row in range(rows):
        a, b = ({int(x) for x in q if valid(row, int(x))} for q in candidates[row])
        segments = (sorted(a - b), sorted(a & b), sorted(b - a))
        union = [x for segment in segments for x in segment]
        ids[row, : len(union)] = union
        physical[row, : len(union) * parts] = [
            f for x in union for f in fragments(row, x)
        ]
        counts[row] = [len(segment) for segment in segments]
    return ids, physical, counts


@pytest.mark.parametrize(
    "per_query,block,page_size", [(16, 8, 256), (64, 64, 64), (33, 512, 128)]
)
def test_canonical_candidate_blocks(per_query, block, page_size):
    rng = np.random.default_rng(per_query)
    rows, width, cache_pages = 9, 4000, 50
    blocks = -(-width // block)
    table = rng.integers(-1, cache_pages + 2, (rows, width // page_size + 2))
    candidates = rng.integers(-2, min(blocks + 2, 3 * per_query), (rows, 2, per_query))
    # Overlapping pairs, repeats within a query and an empty first query.
    shuffled = candidates[:, 0][:, rng.permutation(per_query)]
    overlap = rng.random((rows, per_query)) < 0.5
    candidates[:, 1] = np.where(overlap, shuffled, candidates[:, 1])
    candidates[0, 0] = -1
    table, candidates = table.astype(np.int32), candidates.astype(np.int32)

    args = (block, width, cache_pages, page_size)
    result = _canonical_candidate_blocks_ref(
        jnp.asarray(candidates), jnp.asarray(table), *args
    )
    for got, want in zip(result, _expected(candidates, table, *args)):
        np.testing.assert_array_equal(np.asarray(got), want)
