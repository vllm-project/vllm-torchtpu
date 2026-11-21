# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SparseCore candidate-K gather: exact output rows, partial last tile, banks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import (
    PAGE_RECORDS,
    PAGE_ROWS,
    PackedFP8Layout,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.candidates import (
    gather_candidate_keys,
)

BLOCK = 8
PER_PAGE = PAGE_RECORDS // BLOCK


def _reference(bank, ids, layout):
    """[groups, blocks] ids -> [groups * blocks // PER_PAGE * PAGE_ROWS, 128]."""
    source_per_page = layout.page_size // BLOCK
    safe = np.maximum(ids, 0)
    row = layout.page_offset + safe // source_per_page * layout.page_stride
    base = row * PER_PAGE + safe % source_per_page
    words = bank.reshape(-1, BLOCK)[
        base[..., None] + np.arange(PAGE_ROWS) * source_per_page
    ]  # [groups, blocks, fields, BLOCK]
    groups, blocks = ids.shape
    pages = words.reshape(groups, blocks // PER_PAGE, PER_PAGE, PAGE_ROWS, BLOCK)
    return pages.transpose(0, 1, 3, 2, 4).reshape(-1, PAGE_RECORDS)


# (groups, blocks, banks, live groups): the decode shape leaves a partial last
# tile with one bank; two banks tile it exactly.
@pytest.mark.parametrize(
    "groups,blocks,banks,live",
    [(16, 4096, 1, 16), (16, 4096, 1, 3), (16, 4096, 2, 16), (8, 1024, 1, 8)],
)
def test_gather_matches_reference(groups, blocks, banks, live):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore gather requires TPU")
    rng = np.random.default_rng(7)
    pages = 512
    layout = PackedFP8Layout(pages)
    rows = pages * layout.page_stride
    split = (pages // 2) * layout.page_stride
    bank = rng.integers(-(2**31), 2**31 - 1, (rows, PAGE_RECORDS), np.int64)
    bank = bank.astype(np.int32)
    ids = rng.integers(0, pages * PER_PAGE, (groups, blocks)).astype(np.int32)
    ids[rng.random(ids.shape) < 0.2] = -1
    ids[live:] = -1
    offsets = np.minimum(np.arange(groups + 1), live).astype(np.int32)
    bank_arrays = (
        (jnp.asarray(bank),)
        if banks == 1
        else (jnp.asarray(bank[:split]), jnp.asarray(bank[split:]))
    )
    keys = np.asarray(
        gather_candidate_keys(
            bank_arrays,
            jnp.asarray(ids),
            jnp.asarray(offsets),
            block=BLOCK,
            layout=layout,
        )
    )
    assert keys.shape == (groups * blocks // PER_PAGE * PAGE_ROWS, PAGE_RECORDS)
    expected = _reference(bank, ids, layout)
    # Only valid ids are defined; their words sit at the reference positions.
    valid = np.broadcast_to(
        (ids >= 0).reshape(groups, blocks // PER_PAGE, 1, PER_PAGE, 1),
        (groups, blocks // PER_PAGE, PAGE_ROWS, PER_PAGE, BLOCK),
    ).reshape(-1, PAGE_RECORDS)
    np.testing.assert_array_equal(keys[valid], expected[valid])
