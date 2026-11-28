# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_PREFILL_PREFIX_PAGES: sparse prefill on a certified page-table
prefix picks the right tier and stays bit-identical (the kernel check is TPU
only: the Pallas interpreter cannot run the bitcast FP8 cache loads)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.qwen38_flash.experimental.kernels import (
    block_sparse_gqa_prefill as bsp,
)


def _tier(blocks, lengths, full_pages, *, tokens=32, page=128, target=None):
    table = jnp.zeros((tokens, full_pages), jnp.int32)

    def run(t):
        return jnp.full((1,), t.shape[1], jnp.int32)

    if target is not None:
        orig = bsp._metadata_resident_pages

        def patched(*args, **kwargs):
            return orig(*args, target_bytes=target, **kwargs)

        bsp._metadata_resident_pages, saved = patched, orig
    try:
        out = bsp._metadata_prefix_dispatch(
            run,
            jnp.asarray(blocks, jnp.int32),
            jnp.asarray(lengths, jnp.int32),
            table,
            tokens=tokens,
            query_block=32,
            ratio=4,
            page_size=page,
            kv_tile=128,
        )
    finally:
        if target is not None:
            bsp._metadata_resident_pages = saved
    return int(out[0])


def test_small_prefix_when_everything_fits():
    blocks = np.full((32, 12), -1, np.int32)
    blocks[:, :3] = [0, 5, 20]
    lengths = np.arange(97, 129, dtype=np.int32)
    assert _tier(blocks, lengths, 64) == 1


def test_full_table_when_a_length_exceeds_the_prefix():
    blocks = np.full((32, 12), -1, np.int32)
    lengths = np.full((32,), 4000, np.int32)
    assert _tier(blocks, lengths, 64) == 64


def test_resident_tier_and_selection_outside_prefix():
    blocks = np.full((32, 12), -1, np.int32)
    lengths = np.full((32,), 1000, np.int32)
    # 128 bitset words (the native minimum) of one 128-row block: 128 tiles.
    target = 128 * 128 * 4
    assert _tier(blocks, lengths, 512, target=target) == 128
    blocks[0, 0] = 128 * 128 // 4  # first group past the resident prefix
    assert _tier(blocks, lengths, 512, target=target) == 512
    # Ids past the whole table are invalid and ignored by both runs.
    blocks[0, 0] = 512 * 128 // 4
    assert _tier(blocks, lengths, 512, target=target) == 128


@pytest.mark.tpu_test
@pytest.mark.parametrize("seed", [0, 1])
def test_prefix_pages_bit_identical(monkeypatch, seed):
    tokens, pages, page, ratio, topk = 32, 8, 128, 4, 12
    rng = np.random.default_rng(seed)
    query = jnp.asarray(rng.normal(size=(tokens, 2, 256)), jnp.bfloat16)
    lengths = np.arange(128 - tokens + 1, 129, dtype=np.int32)
    blocks = np.full((tokens, topk), -1, np.int32)
    for r, n in enumerate(lengths):
        pick = rng.choice(n // ratio, size=topk - 2, replace=False)
        blocks[r, : topk - 2] = np.sort(pick)
    table = rng.permutation(pages).astype(np.int32)
    cache = rng.normal(size=(pages, page, 1, 4, 256)).astype(np.float32)
    inputs = (
        query,
        jnp.asarray(blocks),
        jnp.asarray(lengths),
        jnp.asarray(np.broadcast_to(table, (tokens, pages))),
        jnp.asarray(cache, jnp.float8_e4m3fn),
    )

    def run(flag):
        monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_PREFIX_PAGES", flag)
        jax.clear_caches()
        out = bsp.block_sparse_gqa_prefill(
            *inputs, ratio=ratio, kv_heads=1, query_block=32, kv_tile=128
        )
        return np.asarray(jax.block_until_ready(out), np.float32)

    np.testing.assert_array_equal(run("1"), run("0"))
