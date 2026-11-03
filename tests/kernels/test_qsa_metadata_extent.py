# Copyright 2026 Google LLC
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
"""Working-extent certificates preserve selected sets and full DMA bounds."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as tc


@pytest.fixture(autouse=True)
def cpu_native_layout(monkeypatch):
    if jax.default_backend() == "cpu":
        info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
        monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


def native_bytes(tokens, query_block, pages, page, ratio, tile, requests=1):
    nblocks = min(tokens, (tokens + query_block - 1) // query_block + requests - 1)
    width = (query_block + 127) // 128 * 128
    tiles = (pages * page + tile - 1) // tile
    words = (tiles * (tile // (ratio * 32)) + 127) // 128 * 128
    return nblocks * width * words * 4


def test_resident_rounding_respects_policy_and_uses_largest_fitting_page_prefix():
    for tokens, qb, page, ratio, tile, requests in itertools.product(
        [1, 257, 8192, 32768],
        [32, 128, 256],
        [128, 640, 1664, 2048],
        [4, 8],
        [128, 256, 512],
        [1, 8],
    ):
        full = 1000
        pages = tc._metadata_resident_pages(
            tokens, qb, ratio, page, full, tile, requests
        )
        if tile % (ratio * 32):
            assert pages == full
        elif pages < full:
            assert pages > 0
            assert (
                native_bytes(tokens, qb, pages, page, ratio, tile, requests)
                <= tc.TC_METADATA_BITSET_TARGET_BYTES
            )
            assert (
                native_bytes(tokens, qb, pages + 1, page, ratio, tile, requests)
                > tc.TC_METADATA_BITSET_TARGET_BYTES
            )


@pytest.mark.parametrize("field", range(7))
@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_planner_geometry_disables_resident_prefix(field, bad):
    args = [8192, 256, 4, 1664, 158, 256, 1]
    args[field] = bad
    assert tc._metadata_resident_pages(*args) == args[4]


def test_resident_extent_main_geometry_and_ragged_padding():
    args = (8192, 256, 4, 1664, 158, 256)
    pages = tc._metadata_resident_pages(*args)
    assert pages > tc._metadata_prefix_pages(8192, 512, 4, 1664, 158, 256)
    assert (
        native_bytes(8192, 256, pages, 1664, 4, 256)
        <= tc.TC_METADATA_BITSET_TARGET_BYTES
    )
    ragged = tc._metadata_resident_pages(*args, request_capacity=8)
    assert ragged <= pages
    assert (
        native_bytes(8192, 256, ragged, 1664, 4, 256, 8)
        <= tc.TC_METADATA_BITSET_TARGET_BYTES
    )
    assert tc._metadata_resident_pages(8192, 256, 4, 2**31, 158, 256) == 158


@pytest.mark.parametrize(
    "kind",
    [
        "small",
        "resident",
        "equal",
        "past_dma",
        "valid_high",
        "invalid_high",
        "negative",
        "oversized",
        "empty",
    ],
)
def test_dispatch_certifies_small_resident_and_full_fallback(kind):
    tokens, qb, page, full, ratio, tile = 8192, 256, 1664, 158, 4, 256
    resident = tc._metadata_resident_pages(tokens, qb, ratio, page, full, tile)
    small = tc._metadata_prefix_pages(tokens, 512, ratio, page, full, tile)
    safe = resident * page // tile * tile
    blocks = jnp.full((2, 512), -1, jnp.int32).at[:, 0].set(0)
    lengths = jnp.full(2, 16384, jnp.int32)
    expected = resident
    if kind == "small":
        lengths = jnp.full(2, tokens, jnp.int32)
        expected = small
    elif kind == "equal":
        lengths = lengths.at[:].set(safe)
    elif kind == "past_dma":
        lengths = lengths.at[:].set(safe + 1)
        expected = full
    elif kind == "valid_high":
        blocks = blocks.at[0, -1].set(safe // ratio)
        expected = full
    elif kind == "invalid_high":
        blocks = (
            blocks.at[:, 1]
            .set(full * page // ratio)
            .at[:, 2]
            .set(np.iinfo(np.int32).max)
        )
    elif kind == "negative":
        lengths = lengths.at[0].set(-1)
        expected = full
    elif kind == "oversized":
        lengths = lengths.at[0].set(full * page + 1)
        expected = full
    elif kind == "empty":
        blocks = blocks[:0]
        lengths = lengths[:0]
        expected = small
    actual = jax.jit(
        lambda selected, live: tc._metadata_prefix_dispatch(
            lambda t: jnp.int32(t.shape[1]),
            selected,
            live,
            jnp.zeros((2, full), jnp.int32),
            tokens=tokens,
            query_block=qb,
            ratio=ratio,
            page_size=page,
            kv_tile=tile,
        )
    )(blocks, lengths)
    assert int(actual) == expected


def test_resident_bits_match_full_and_independent_causal_sets():
    rows, ratio, page, tile, full, budget = 256, 4, 1664, 256, 158, 16
    pages = tc._metadata_resident_pages(8192, rows, ratio, page, full, tile)
    safe = pages * page // tile * tile
    rng = np.random.default_rng(1964)
    lengths = rng.integers(0, safe + 1, rows, dtype=np.int32)
    lengths[:8] = [0, 1, 3, 4, 1663, 1664, 1665, safe]
    blocks = rng.integers(-1, safe // ratio, (rows, budget), dtype=np.int32)
    blocks[:, 1] = blocks[:, 0]
    blocks[::3, 0] = np.iinfo(np.int32).max
    blocks[::5, 2] = full * page // ratio
    table = jnp.broadcast_to(jnp.arange(full, dtype=jnp.int32), (rows, full))
    operands = (jnp.asarray(blocks), jnp.asarray(lengths))
    kwargs = dict(query_block=rows, ratio=ratio, kv_tile=tile, physical_page_size=page)
    _, _, counts, actual = tc._prepare_metadata_bounded(
        *operands, table[:, :pages], **kwargs
    )
    _, _, full_counts, original = tc._prepare_metadata_bounded(
        *operands, table, **kwargs
    )
    bits = np.asarray(actual).view(np.uint32)[0].T
    full_bits = np.asarray(original).view(np.uint32)[0].T
    np.testing.assert_array_equal(full_bits[:, : bits.shape[1]], bits)
    np.testing.assert_array_equal(full_bits[:, bits.shape[1] :], 0)
    np.testing.assert_array_equal(counts, full_counts)
    wanted = np.zeros_like(bits)
    for row, length in enumerate(lengths):
        complete = int(length) // ratio
        selected = {
            int(x) for x in blocks[row, : min(complete, budget)] if 0 <= x < complete
        }
        if length % ratio:
            selected.add(complete)
        for group in selected:
            wanted[row, group // 32] |= np.uint32(1) << np.uint32(group % 32)
    np.testing.assert_array_equal(bits, wanted)


@pytest.mark.parametrize("history", [0, 8192, 131072])
def test_ragged_prefix_preserves_request_capacity_and_mixed_history(
    monkeypatch, history
):
    def core(
        query,
        gate,
        blocks,
        lengths,
        table,
        cache,
        starts,
        output,
        offset,
        size,
        **kwargs,
    ):
        return jnp.full_like(output, table.shape[1])

    monkeypatch.setattr(tc, "_ragged_prefill_full", core)
    query = jnp.zeros((257, 1, 1), jnp.bfloat16)
    table = jnp.zeros((257, 158), jnp.int32)
    cache = jnp.zeros((1, 1664, 1, 4, 1), jnp.float8_e4m3fn)
    lengths = jnp.arange(1, 258, dtype=jnp.int32) + history
    blocks = jnp.full((257, 512), -1, jnp.int32)
    starts = jnp.array([0, 1, 17, 257], jnp.int32)
    actual = jax.jit(
        lambda: tc._ragged_prefill(
            query,
            query,
            blocks,
            lengths,
            table,
            cache,
            starts,
            query,
            0,
            257,
            ratio=4,
            kv_heads=1,
            query_block=256,
            kv_tile=256,
            k_scale=1.0,
            v_scale=1.0,
        )
    )()
    small = tc._metadata_prefix_pages(257, 512, 4, 1664, 158, 256)
    resident = tc._metadata_resident_pages(257, 256, 4, 1664, 158, 256, 3)
    # This tiny request set can already keep the full table under the policy;
    # no additional prefix is invented when the resident domain equals full.
    expected = (
        small
        if history == 0
        else (
            resident
            if small < resident < 158 and history + 257 <= resident * 1664 // 256 * 256
            else 158
        )
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("target", [0, 1, 4 * 1024**2, 16 * 1024**2])
def test_target_is_explicit_soft_policy(target):
    pages = tc._metadata_resident_pages(
        8192, 256, 4, 1664, 158, 256, target_bytes=target
    )
    if pages < 158:
        assert native_bytes(8192, 256, pages, 1664, 4, 256) <= target
    else:
        assert target < native_bytes(8192, 256, 1, 1664, 4, 256)
