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
"""Per-rank TC geometry and request-aware bounded adaptive tails."""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
    block_sparse_gqa_attention,
)
from vllm_torchtpu.kernels.qwen4_exp.prefill_tiles import get_sparse_prefill_tiles
from vllm_torchtpu.kernels.sparse_gqa_prefill.adaptive import adaptive_prefill


@pytest.mark.parametrize("tokens", [17, 256, 512, 4096, 8192, 8448, 16640])
@pytest.mark.parametrize(
    "heads,kv_heads,dim",
    [(24, 2, 256), (12, 1, 256), (16, 4, 128), (8, 4, 256), (24, 3, 256), (48, 4, 256)],
)
@pytest.mark.parametrize("fp8", [False, True])
def test_tiles_follow_local_footprint_not_tensor_parallel_degree(
    monkeypatch, tokens, heads, kv_heads, dim, fp8
):
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    dtype, packing = (jnp.float8_e4m3fn, 4) if fp8 else (jnp.bfloat16, 2)
    q = jax.ShapeDtypeStruct((tokens, heads, dim), jnp.bfloat16)
    cache = jax.ShapeDtypeStruct(
        (3, 1664, math.ceil(2 * kv_heads / packing), packing, dim), dtype
    )
    query_block, kv_tile = get_sparse_prefill_tiles(
        q, cache, kv_heads=kv_heads, ratio=4
    )
    expected = (256, 256)
    if heads == 48:
        expected = (128, 256 if fp8 else 128)
    elif heads == 24 and kv_heads == 3 and not fp8:
        expected = (128, 256)
    assert (query_block, kv_tile) == expected


def inputs(tokens, heads, kv_heads, dim, fp8, *, ragged=False):
    rng = np.random.default_rng(9721)
    packing = 4 if fp8 else 2
    dtype = jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    # Separate shuffled physical page sets, nonzero context, odd page1664.
    page_size, pages = 1664, 3
    boundaries = [0, 129, min(tokens - 1, 385)] if ragged else [0]
    boundaries += [tokens - 1]
    starts = np.array(boundaries + [tokens - 1] * (9 - len(boundaries)), np.int32)
    owner = np.minimum(
        np.searchsorted(starts[1:], np.arange(tokens), side="right"), len(starts) - 2
    )
    context = np.arange(len(starts) - 1, dtype=np.int32) * 37 + 2049
    lengths = context[owner] + np.arange(tokens) - starts[owner] + 1
    lengths[-1] = 0
    tables = np.stack(
        [rng.permutation(pages) + i * pages for i in range(len(context))]
    ).astype(np.int32)
    table = tables[owner]
    query = jnp.asarray(rng.normal(0, 0.3, (tokens, heads, dim)), jnp.bfloat16)
    gate = jnp.asarray(rng.normal(0, 2, query.shape), jnp.bfloat16).at[-1].set(jnp.nan)
    query = query.at[-1].set(jnp.nan)
    blocks = np.full((tokens, 512), -1, np.int32)
    for row, n in enumerate(lengths):
        count = min(n // 4, 512)
        blocks[row, :count] = rng.choice(n // 4, count, replace=False)
    shape = (
        len(context) * pages,
        page_size,
        math.ceil(2 * kv_heads / packing),
        packing,
        dim,
    )
    cache = jnp.asarray(rng.normal(0, 0.3, shape), dtype)
    return (
        query,
        gate,
        jnp.asarray(blocks),
        jnp.asarray(lengths, jnp.int32),
        jnp.asarray(table),
        cache,
        jnp.asarray(starts),
    ), (owner, context)


def check_reference(args, output, *, kv_heads, rows, ratio=4, page_size=1664):
    q, gate, blocks, lengths, table, cache, _ = map(np.asarray, args)
    result = np.asarray(output, np.float32)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[lengths == 0], 0)
    for row in rows:
        n = int(lengths[row])
        selected = (
            blocks[row, : min(n // ratio, 512), None] * ratio + np.arange(ratio)
        ).reshape(-1)
        selected = np.concatenate((selected, np.arange(n // ratio * ratio, n)))
        values = (
            cache[table[row, selected // page_size], selected % page_size]
            .reshape(len(selected), -1, q.shape[-1])
            .astype(np.float64)
        )
        wanted = []
        per_head = q.shape[1] // kv_heads
        for h in range(kv_heads):
            score = (
                q[row, h * per_head : (h + 1) * per_head].astype(np.float64)
                @ values[:, 2 * h].T
                / math.sqrt(q.shape[-1])
            )
            mass = np.exp(score - score.max(axis=1, keepdims=True))
            wanted.append(
                (mass / mass.sum(axis=1, keepdims=True)) @ values[:, 2 * h + 1]
            )
        expected = np.concatenate(wanted)
        # Match the existing two BF16 boundaries and original 1% threshold.
        weight = jnp.asarray(
            1 / (1 + np.exp(-gate[row].astype(np.float64))), jnp.bfloat16
        )
        expected = np.asarray(
            jnp.asarray(
                jnp.asarray(expected, jnp.bfloat16).astype(jnp.float32) * weight,
                jnp.bfloat16,
            ),
            np.float32,
        )
        assert np.linalg.norm(result[row] - expected) / np.linalg.norm(expected) < 0.01


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "tokens,heads,kv_heads,dim,fp8",
    [
        (256, 12, 1, 256, False),
        (256, 8, 2, 384, False),
        (256, 24, 3, 256, False),
        (512, 8, 4, 256, True),
        (4096, 16, 4, 128, False),
        (8448, 24, 3, 256, True),
        (16640, 12, 1, 256, True),
    ],
)
def test_general_paired_geometry(tokens, heads, kv_heads, dim, fp8):
    args, _ = inputs(tokens, heads, kv_heads, dim, fp8)
    # Long prefill uses context/table capacity large enough for every query.
    # This test keeps each selected length bounded while repeating legal rows.
    args = list(args)
    args[3] = jnp.where(args[3] > 0, jnp.minimum(args[3], 4991), 0)
    args[2] = jnp.where(args[2] < args[3][:, None] // 4, args[2], -1)
    # Rebuild valid unique selection prefix after length bounding.
    count = jnp.minimum(args[3] // 4, 512)
    args[2] = jnp.where(
        jnp.arange(512)[None, :] < count[:, None], jnp.arange(512)[None, :], -1
    ).astype(jnp.int32)
    query_block, kv_tile = get_sparse_prefill_tiles(
        args[0], args[5], kv_heads=kv_heads, ratio=4
    )
    run = jax.jit(
        functools.partial(
            block_sparse_gqa_attention,
            kv_heads=kv_heads,
            ratio=4,
            query_block=query_block,
            kv_tile=kv_tile,
        )
    )
    compiled = run.lower(*args).compile()
    assert "prefill_paired_pages" in compiled.as_text()
    before = [np.asarray(x).copy() for x in args]
    out, returned = compiled(*args)
    check_reference(
        args, out, kv_heads=kv_heads, rows=[0, min(127, tokens - 2), tokens - 2]
    )
    again, _ = compiled(*args)
    np.testing.assert_array_equal(again, out)
    for x, old in zip(args, before):
        np.testing.assert_array_equal(
            np.asarray(x).copy().view(np.uint8), old.view(np.uint8)
        )
    np.testing.assert_array_equal(
        np.asarray(returned).copy().view(np.uint8), before[5].view(np.uint8)
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "tokens,ragged,fp8",
    [(17, False, False), (257, False, True), (512, True, False), (512, True, True)],
)
def test_adaptive_short_tail_and_padded_request_boundaries(tokens, ragged, fp8):
    args, _ = inputs(tokens, 24, 2, 256, fp8, ragged=ragged)
    before = [np.asarray(x).copy() for x in args]
    compiled = jax.jit(adaptive_prefill).lower(*args).compile()
    assert "prefill_paired_pages" in compiled.as_text()
    out, returned = compiled(*args)
    rows = [0, tokens - 2] + ([128, 129, 384, 385] if ragged else [])
    check_reference(args, out, kv_heads=2, rows=rows)
    again, _ = compiled(*args)
    np.testing.assert_array_equal(again, out)
    for x, old in zip(args, before):
        np.testing.assert_array_equal(
            np.asarray(x).copy().view(np.uint8), old.view(np.uint8)
        )
    np.testing.assert_array_equal(
        np.asarray(returned).copy().view(np.uint8), before[5].view(np.uint8)
    )


@pytest.mark.tpu_test
def test_other_compression_ratio_uses_general_tc():
    args, _ = inputs(256, 8, 4, 256, False)
    args = list(args)
    args[5] = args[5][:, :128]
    args[3] = jnp.where(args[3] > 0, 255, 0)
    count = args[3] // 8
    args[2] = jnp.where(
        jnp.arange(512)[None, :] < count[:, None], jnp.arange(512)[None, :], -1
    ).astype(jnp.int32)
    qb, kt = get_sparse_prefill_tiles(args[0], args[5], kv_heads=4, ratio=8)
    assert kt == 8 * 32
    compiled = (
        jax.jit(
            functools.partial(
                block_sparse_gqa_attention,
                kv_heads=4,
                ratio=8,
                query_block=qb,
                kv_tile=kt,
            )
        )
        .lower(*args)
        .compile()
    )
    assert "prefill_paired_pages" not in compiled.as_text()
    output, _ = compiled(*args)
    check_reference(
        args, output, kv_heads=4, rows=[0, 128, 254], ratio=8, page_size=128
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
def test_aligned_requests_keep_sc_gathers_local_with_padded_slots(fp8):
    from tests.kernels.test_sparse_gqa_adaptive_prefill import _make_inputs
    from vllm_torchtpu.kernels.sparse_gqa_prefill.selector import select_sc

    args = list(_make_inputs(512, 131072, fp8, True))
    # Distinct request page permutations in one shared pool. An unused NaN
    # page must not enter either request's gather or the implicit partial tail.
    args[4] = args[4].at[256:].set(args[4][256:, ::-1])
    # Reversing the mapping makes the old logical tail reachable. Initialize
    # those rows as valid data before poisoning a wholly unreferenced page.
    args[5] = jnp.concatenate(
        (jnp.nan_to_num(args[5]), jnp.full_like(args[5][:1], jnp.nan))
    )
    args[6] = jnp.array([0, 256, 512, 512, 512], jnp.int32)
    choose = jax.jit(
        functools.partial(
            select_sc,
            num_tiles=math.ceil(args[4].shape[1] * 1664 / 256),
            dtype=args[5].dtype,
        )
    )
    assert bool(choose(args[2], args[3]))
    compiled = jax.jit(adaptive_prefill).lower(*args).compile()
    before = [np.asarray(x).copy() for x in args]
    output, returned = compiled(*args)
    check_reference(args, output, kv_heads=2, rows=[1, 255, 256, 510])
    np.testing.assert_array_equal(compiled(*args)[0], output)
    for x, old in zip(args, before):
        np.testing.assert_array_equal(
            np.asarray(x).copy().view(np.uint8), old.view(np.uint8)
        )
    np.testing.assert_array_equal(
        np.asarray(returned).copy().view(np.uint8), before[5].view(np.uint8)
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
def test_live_tail_smaller_than_sc_microchunk(fp8):
    from vllm_torchtpu.kernels.sparse_gqa_prefill.selector import select_sc
    from vllm_torchtpu.kernels.sparse_gqa_prefill.tuned_block_sizes import SC_CHUNK_SIZE

    args, _ = inputs(17, 24, 2, 256, fp8)
    args = list(args)
    rng = np.random.default_rng(2018)
    lengths = np.arange(131072 - 16, 131073, dtype=np.int32)
    lengths[-1] = 0
    blocks = np.full((17, 512), -1, np.int32)
    for row in range(16):
        blocks[row] = rng.choice(lengths[row] // 4, 512, replace=False)
    pages = math.ceil(131072 / 1664)
    args[2] = jnp.asarray(blocks)
    args[3] = jnp.asarray(lengths)
    args[4] = jnp.broadcast_to(
        jnp.asarray(rng.permutation(pages), jnp.int32), (17, pages)
    )
    args[5] = jax.jit(
        lambda key: jax.random.normal(key, (pages, *args[5].shape[1:])).astype(
            args[5].dtype
        )
    )(jax.random.key(391))
    choose = jax.jit(
        functools.partial(
            select_sc,
            num_tiles=math.ceil(pages * 1664 / 256),
            dtype=args[5].dtype,
            sc_queries=SC_CHUNK_SIZE[jnp.dtype(args[5].dtype)],
        )
    )
    assert bool(
        choose(
            jnp.pad(args[2], ((0, 239), (0, 0)), constant_values=-1),
            jnp.pad(args[3], (0, 239)),
        )
    )
    before = [np.asarray(x).copy() for x in args]
    compiled = jax.jit(adaptive_prefill).lower(*args).compile()
    output, returned = compiled(*args)
    check_reference(args, output, kv_heads=2, rows=[0, 15])
    np.testing.assert_array_equal(compiled(*args)[0], output)
    for x, old in zip(args, before):
        np.testing.assert_array_equal(
            np.asarray(x).copy().view(np.uint8), old.view(np.uint8)
        )
    np.testing.assert_array_equal(
        np.asarray(returned).copy().view(np.uint8), before[5].view(np.uint8)
    )
