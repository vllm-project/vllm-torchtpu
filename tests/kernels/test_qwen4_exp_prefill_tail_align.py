# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for default-off QSA prefill tail-align and active-prefix helpers."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as bsp


def test_prefill_tail_align_flags_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_QSA_PREFILL_TAIL_ALIGN", raising=False)
    monkeypatch.delenv("VLLM_TPU_QSA_PREFILL_ACTIVE_PREFIX", raising=False)
    assert envs.VLLM_TPU_QSA_PREFILL_TAIL_ALIGN is False
    assert envs.VLLM_TPU_QSA_PREFILL_ACTIVE_PREFIX is False


def test_block_page_table_accepts_token_and_block_shapes():
    nblocks, qb, pages = 4, 32, 8
    per_block = jnp.arange(nblocks * pages, dtype=jnp.int32).reshape(nblocks, pages)
    per_token = jnp.repeat(per_block, qb, axis=0)
    np.testing.assert_array_equal(
        np.asarray(bsp._block_page_table(per_token, nblocks, qb)),
        np.asarray(per_block),
    )
    np.testing.assert_array_equal(
        np.asarray(bsp._block_page_table(per_block, nblocks, qb)),
        np.asarray(per_block),
    )


@pytest.mark.parametrize(
    "starts_list,rows,qb",
    [
        ([0, 35, 40, 40], 128, 32),
        ([0, 0, 5, 69], 128, 32),
        ([0, 300, 300], 512, 256),
        ([0, 256, 300, 300], 512, 256),
    ],
)
def test_aligned_block_sources_matches_scatter(starts_list, rows, qb):
    starts = jnp.asarray(starts_list, dtype=jnp.int32)
    real = starts[1:] - starts[:-1]
    padded = ((real + qb - 1) // qb) * qb
    offsets = jnp.concatenate((jnp.zeros((1,), jnp.int32), jnp.cumsum(padded)))
    max_tokens = max(rows, int(offsets[-1]))
    max_tokens = ((max_tokens + qb - 1) // qb) * qb
    pages = 6
    page_table = jnp.arange(rows * pages, dtype=jnp.int32).reshape(rows, pages) + 1

    # Reference full-token scatter followed by block-head pick.
    req = jnp.minimum(
        jnp.searchsorted(starts[1:], jnp.arange(rows), side="right"), len(real) - 1
    )
    positions = jnp.arange(rows) - starts[req]
    valid = jnp.arange(rows) < starts[-1]
    slots = jnp.where(valid, offsets[req] + positions, max_tokens)
    ref_full = jnp.zeros((max_tokens, pages), jnp.int32).at[slots].set(
        page_table, mode="drop"
    )
    ref_blocks = ref_full.reshape(max_tokens // qb, qb, pages)[:, 0]

    b_src, b_fill = bsp._aligned_block_sources(
        starts, offsets, real, rows, max_tokens, qb
    )
    got_blocks = jnp.where(
        b_fill[:, None], jnp.take(page_table, b_src, axis=0), jnp.int32(0)
    )
    np.testing.assert_array_equal(np.asarray(got_blocks), np.asarray(ref_blocks))


def test_selection_words_gates_dead_row_stale_blocks(monkeypatch):
    # Live row selects block 3; dead padding row (length == 0) carries stale id 5.
    blocks = jnp.asarray([[3, -1], [5, 7]], dtype=jnp.int32)
    lengths = jnp.asarray([16, 0], dtype=jnp.int32)
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_TAIL_ALIGN", "1")
    word, bit = bsp._selection_words(blocks, lengths, ratio=4, groups=16)
    assert np.all(np.asarray(word[1]) == -1)
    assert np.all(np.asarray(bit[1]) == 0)
    assert int(word[0, 0]) == 0 and int(bit[0, 0]) == (1 << 3)
