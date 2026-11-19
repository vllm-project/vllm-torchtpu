# SPDX-License-Identifier: Apache-2.0
"""The bounded (lax.map) QSA prefill metadata equals the scatter reference.

``_prepare_metadata_bounded`` serves VLLM_TPU_QSA_PREFILL_BITSET_KERNEL=0 at
long-context shapes (paired pages, Q256, T x words x 4 >= 16 MiB). It must
produce the same page table rows, active KV tile lists, counts and word-major
bitset as the unbounded scatter path. Pure JAX: runs on CPU.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as bsp


def _reference(blocks, lengths, page_table, *, ratio, query_block, kv_tile, page):
    """The scatter branch of block_sparse_gqa_prefill (non-DCP, paired pages)."""
    groups = page_table.shape[1] * page // ratio
    num_kv_tiles = (groups * ratio + kv_tile - 1) // kv_tile
    words_per_tile = kv_tile // (ratio * 32)
    words = ((num_kv_tiles * words_per_tile + 127) // 128) * 128
    bitset = bsp._scatter_bitset(blocks, lengths, ratio=ratio, groups=groups, words=words)
    nblocks = blocks.shape[0] // query_block
    block_bits = bitset.reshape(nblocks, query_block, words)
    active = jnp.any(
        block_bits[:, :, : words_per_tile * num_kv_tiles].reshape(
            nblocks, query_block, num_kv_tiles, words_per_tile
        )
        != 0,
        axis=(1, 3),
    )
    width = ((query_block + 127) // 128) * 128
    native = jnp.pad(
        block_bits.transpose(0, 2, 1), ((0, 0), (0, 0), (0, width - query_block))
    )
    table = page_table.reshape(nblocks, query_block, -1)[:, 0]
    return (
        table,
        bsp._compact_active(active),
        jnp.sum(active, axis=1, dtype=jnp.int32),
        native,
    )


def _case(seed, *, tokens, context, page, pages, topk=512, ratio=4):
    rng = np.random.default_rng(seed)
    lengths = (context + np.arange(1, tokens + 1)).astype(np.int32)
    lengths[-37:] = 0  # padded rows of a bucket
    blocks = np.full((tokens, topk), -1, np.int32)
    for row, length in enumerate(lengths):
        complete = int(length) // ratio
        count = min(complete, topk)
        if count:
            # Local window plus scattered picks: some tiles shared, some not.
            local = np.arange(complete - count // 2, complete)
            far = rng.choice(max(complete - count // 2, 1), count - len(local), replace=False)
            blocks[row, :count] = np.sort(np.concatenate((far, local)))[:count]
    table = np.broadcast_to(rng.permutation(pages), (tokens, pages)).astype(np.int32)
    return jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table)


@pytest.mark.parametrize("kv_tile", [256, 512, 1024])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_bounded_metadata_matches_scatter(kv_tile, batch_size):
    page, pages, query_block = 3200, 6, 256
    blocks, lengths, table = _case(
        kv_tile + batch_size, tokens=1024, context=12800, page=page, pages=pages
    )
    got = bsp._prepare_metadata_bounded(
        blocks,
        lengths,
        table,
        ratio=4,
        query_block=query_block,
        kv_tile=kv_tile,
        physical_page_size=page,
        batch_size=batch_size,
    )
    want = _reference(
        blocks, lengths, table, ratio=4, query_block=query_block, kv_tile=kv_tile, page=page
    )
    for name, g, w in zip(("table", "tiles", "counts", "bits"), got, want):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w), err_msg=name)
    # Not vacuous: some blocks skip tiles, and every block has work.
    counts = np.asarray(got[2])
    assert counts.min() > 0
    assert counts.min() < (pages * page) // kv_tile
