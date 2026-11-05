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
"""Kimi-K3.1's sparse MLA prefill attention by per-row token gathers (CPU,
Pallas TPU interpret mode): rows with fewer selected keys than slots must
not read unwritten gather-table rows (0 x NaN in P V)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill
from vllm_torchtpu.megakernel.kimi_k3.kernels import mla_sparse_prefill as sp

pytestmark = pytest.mark.cpu_test

CAPACITY, WINDOW, HEADS = 512, 64, 4


def _window_only_case(seed=0):
    rng = np.random.default_rng(seed)
    latent = jnp.asarray(rng.standard_normal((WINDOW, 576), np.float32) * 0.7).astype(
        jnp.bfloat16
    )
    q = rng.standard_normal((WINDOW, HEADS, 256), np.float32)
    q[..., 192:] = 0
    query = jnp.asarray(q * 0.1).astype(jnp.bfloat16)
    up = jnp.asarray(
        rng.standard_normal((512, HEADS * 256), np.float32) / np.sqrt(512)
    ).astype(jnp.bfloat16)
    # Every arena page unwritten (NaN bytes); no cached prefix, so every arena page
    # maps to slot 0.
    pages = jnp.full((2, 128, 4, 640), jnp.nan, jnp.float32).astype(jnp.float8_e4m3fn)
    order = jnp.zeros((1 + CAPACITY // 512,), jnp.int32)
    segments = jnp.asarray([0, WINDOW, 0, 0, 0], jnp.int32)
    valid = dsa_prefill.candidate_valid(
        segments,
        jnp.asarray([WINDOW], jnp.int32),
        count=1,
        tokens=WINDOW,
        capacity=CAPACITY,
    )
    return query, latent, up, pages, order, dsa_prefill.pack_bits(valid)


def _reference(query, latent, up, row):
    """f64 attention of window row ``row`` over window tokens ``0..row``
    (FP8-rounded latents)."""
    lat = np.asarray(latent.astype(jnp.float8_e4m3fn).astype(jnp.float32), np.float64)[
        : row + 1
    ]
    q = np.asarray(query.astype(jnp.float32), np.float64)[row]
    w = np.asarray(up.astype(jnp.float32), np.float64).reshape(512, HEADS, 256)
    out = np.zeros((HEADS, 128))
    for h in range(HEADS):
        logits = (
            lat[:, :512] @ (w[:, h, :128] @ q[h, :128])
            + lat[:, 512:576] @ q[h, 128:192]
        )
        p = np.exp2(logits - logits.max())
        out[h] = (p @ (lat[:, :512] @ w[:, h, 128:])) / p.sum()
    return out


def test_rows_with_fewer_keys_than_slots_ignore_unwritten_pages():
    query, latent, up, pages, order, words = _window_only_case()
    rows = np.asarray([0, 1, 30, 63])
    with pltpu.force_tpu_interpret_mode():
        out = sp.attend_rows(
            query[rows],
            latent,
            up,
            pages,
            order,
            words[rows],
            capacity=CAPACITY,
            gather=lambda table, index, end: table[index],
            topk=128,
        )
    out = np.asarray(out.astype(jnp.float32))
    assert np.isfinite(out).all()
    for i, row in enumerate(rows):
        want = _reference(query, latent, up, int(row))
        np.testing.assert_allclose(out[i], want, atol=5e-3 * np.abs(want).max())


def test_rows_without_keys_skip_their_gather():
    query, latent, up, pages, order, words = _window_only_case()
    rows = np.asarray([0, 5, 1, 30, 63, 7, 9, 11])
    live = np.asarray([True, False, True, True, True, False, False, False])
    words = jnp.where(jnp.asarray(live)[:, None], words[rows], 0)
    ends = []

    def gather(table, index, end):
        ends.append(int(end))
        written = (jnp.arange(index.shape[0]) < end)[:, None]
        return jnp.where(written, table[index], jnp.int8(0x7F))  # FP8 NaN past end

    with pltpu.force_tpu_interpret_mode():
        out = sp.attend_rows(
            query[rows],
            latent,
            up,
            pages,
            order,
            words,
            capacity=CAPACITY,
            gather=gather,
            topk=128,
        )
    out = np.asarray(out.astype(jnp.float32))
    assert ends == [5 * 128]
    assert np.isfinite(out).all()
    for i, row in enumerate(rows):
        if not live[i]:
            assert not out[i].any()
            continue
        want = _reference(query, latent, up, int(row))
        np.testing.assert_allclose(out[i], want, atol=5e-3 * np.abs(want).max())


def test_selected_columns_match_the_bits():
    rng = np.random.default_rng(1)
    bits = rng.random((8, 1024)) < 0.05
    words = dsa_prefill.pack_bits(jnp.asarray(bits))
    columns, counts = jax.jit(lambda w: sp.selected_columns(w, topk=128))(words)
    columns, counts = np.asarray(columns), np.asarray(counts)
    for r in range(8):
        want = np.flatnonzero(bits[r])
        assert counts[r] == want.size
        np.testing.assert_array_equal(columns[r, : counts[r]], want)
        assert not columns[r, counts[r] :].any()


@pytest.mark.parametrize("scattered", [False, True])
def test_token_rows_follow_the_arena_order(scattered):
    # A serving order is a few runs (a sequence's pages on one rank); a scattered
    # one has more runs than the comparison path holds and must take the gather.
    rng = np.random.default_rng(2)
    pages, capacity = 224, 256 * 512
    if scattered:
        slots = 1 + rng.permutation(pages)
    else:
        slots = np.concatenate([r * 40 + 1 + np.arange(28) for r in range(8)])
    order = np.zeros(1 + capacity // 512, np.int32)
    order[0], order[1 : 1 + pages] = pages, slots
    columns = rng.integers(0, pages * 512, (4, 300)).astype(np.int32)
    got = sp.token_rows(
        jnp.asarray(columns), jnp.asarray(order), capacity=capacity, runs=64
    )
    want = order[1 + columns // 512] * 512 + columns % 512
    np.testing.assert_array_equal(np.asarray(got), want)


@pytest.mark.parametrize("rows,block,tile", [(16, 8, 128), (128, 16, 256)])
def test_block_unions_match_each_rows_keys(rows, block, tile):
    # A scattered arena order, NaN bytes in unwritten slots and stale spare lanes,
    # rows with fewer keys than topk, and (128 rows) two groups of membership lanes.
    rng = np.random.default_rng(rows)
    capacity, slots, window, cached, topk, heads = (
        8 * 512,
        10,
        64,
        5 * 512 - 100,
        128,
        4,
    )
    latent = rng.standard_normal((capacity, 576)).astype(np.float32) * 0.7
    used = -(-cached // 512)
    perm = rng.permutation(slots)[:used]
    order = np.zeros(1 + capacity // 512, np.int32)
    order[0], order[1 : 1 + used] = used, perm
    pages = np.full((slots, 512, 640), np.nan, np.float32)
    for j in range(used):
        pages[perm[j], :, :576] = latent[j * 512 : (j + 1) * 512]
        pages[perm[j], :, 576:] = rng.standard_normal((512, 64))
    pages = jnp.asarray(pages).astype(jnp.float8_e4m3fn).reshape(slots, 128, 4, 640)
    window_latent = jnp.asarray(
        rng.standard_normal((window, 576)) * 0.7, jnp.float32
    ).astype(jnp.bfloat16)
    positions = np.arange(rows) % window
    bits = np.zeros((rows, capacity + window), bool)
    for i, w in enumerate(positions):
        candidates = np.concatenate((np.arange(cached), capacity + np.arange(w + 1)))
        keep = topk if i % 5 else topk // 3
        bits[i, rng.choice(candidates, keep, replace=False)] = True
    q = rng.standard_normal((rows, heads, 256)).astype(np.float32) * 0.1
    q[..., 192:] = 0
    query = jnp.asarray(q).astype(jnp.bfloat16)
    up = jnp.asarray(
        rng.standard_normal((512, heads * 256)) / np.sqrt(512), jnp.float32
    ).astype(jnp.bfloat16)
    with pltpu.force_tpu_interpret_mode():
        out = sp.attend_blocks(
            query,
            window_latent,
            up,
            pages,
            jnp.asarray(order),
            dsa_prefill.pack_bits(jnp.asarray(bits)),
            capacity=capacity,
            gather=lambda table, index, end: table[index],
            topk=topk,
            block=block,
            tile=tile,
        )
    out = np.asarray(out.astype(jnp.float32))
    assert np.isfinite(out).all()

    def fp8(x):
        return np.asarray(
            jnp.asarray(x).astype(jnp.float8_e4m3fn).astype(jnp.float32), np.float64
        )

    arena, window64 = fp8(latent), fp8(np.asarray(window_latent.astype(jnp.float32)))
    w = np.asarray(up.astype(jnp.float32), np.float64).reshape(512, heads, 256)
    for i in range(rows):
        cols = np.flatnonzero(bits[i])
        lat = np.concatenate(
            (arena[cols[cols < capacity]], window64[cols[cols >= capacity] - capacity])
        )
        qi = np.asarray(query[i].astype(jnp.float32), np.float64)
        want = np.zeros((heads, 128))
        for h in range(heads):
            logits = (
                lat[:, :512] @ (w[:, h, :128] @ qi[h, :128])
                + lat[:, 512:576] @ qi[h, 128:192]
            )
            p = np.exp2(logits - logits.max())
            want[h] = ((p @ lat[:, :512]) @ w[:, h, 128:]) / p.sum()
        np.testing.assert_allclose(out[i], want, atol=8e-3 * np.abs(want).max())


def test_split_queries_keep_sharp_softmaxes_exact():
    # Large logits (a sharp softmax, as layer 3's): the absorbed queries' BF16
    # rounding dominates the error unless they are split in two BF16 parts.
    rng = np.random.default_rng(11)
    capacity, window, heads, rows, topk = 1024, 64, 8, 8, 256
    latent = rng.standard_normal((capacity, 576)).astype(np.float32)
    pages = np.zeros((3, 512, 640), np.float32)
    order = np.zeros(1 + capacity // 512, np.int32)
    order[0], order[1:3] = 2, [2, 0]
    pages[2, :, :576], pages[0, :, :576] = latent[:512], latent[512:]
    pages = jnp.asarray(pages).astype(jnp.float8_e4m3fn).reshape(3, 128, 4, 640)
    window_latent = jnp.asarray(rng.standard_normal((window, 576)), jnp.float32)
    window_latent = window_latent.astype(jnp.bfloat16)
    bits = np.zeros((rows, capacity + window), bool)
    for i in range(rows):
        bits[i, rng.choice(1000, topk, replace=False)] = True
    up = jnp.asarray(
        rng.standard_normal((512, heads * 256)) / np.sqrt(512), jnp.float32
    )
    up = up.astype(jnp.bfloat16)
    q = rng.standard_normal((rows, heads, 256)).astype(np.float32)
    q[..., 192:] = 0
    query = jnp.asarray(q).astype(jnp.bfloat16)
    with pltpu.force_tpu_interpret_mode():
        out = np.asarray(
            sp.attend_rows(
                query,
                window_latent,
                up,
                pages,
                jnp.asarray(order),
                dsa_prefill.pack_bits(jnp.asarray(bits)),
                capacity=capacity,
                gather=lambda table, index, end: table[index],
                topk=topk,
                split=True,
            ).astype(jnp.float32)
        )

    def fp8(x):
        return np.asarray(
            jnp.asarray(x).astype(jnp.float8_e4m3fn).astype(jnp.float32), np.float64
        )

    arena, window64 = fp8(latent), fp8(np.asarray(window_latent.astype(jnp.float32)))
    w = np.asarray(up.astype(jnp.float32), np.float64).reshape(512, heads, 256)
    for i in range(rows):
        cols = np.flatnonzero(bits[i])
        lat = np.concatenate(
            (arena[cols[cols < capacity]], window64[cols[cols >= capacity] - capacity])
        )
        qi = np.asarray(query[i].astype(jnp.float32), np.float64)
        want = np.zeros((heads, 128))
        for h in range(heads):
            logits = lat[:, :512] @ (w[:, h, :128] @ qi[h, :128])
            logits += lat[:, 512:576] @ qi[h, 128:192]
            p = np.exp2(logits - logits.max())
            want[h] = ((p @ lat[:, :512]) @ w[:, h, 128:]) / p.sum()
        np.testing.assert_allclose(out[i], want, atol=5e-3 * np.abs(want).max())
