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
"""Two logical KV pages may have unrelated physical addresses."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
    block_sparse_gqa_attention,
    block_sparse_gqa_prefill,
)


def pack_rows(key, value, *, fp8, k_scale=1.0, v_scale=1.0):
    """Independent fixture for interleaved [K0,V0,K1,V1] cache heads."""
    key, value = np.asarray(key, np.float32), np.asarray(value, np.float32)
    if fp8:
        key = np.clip(key / k_scale, -448, 448).astype(jnp.float8_e4m3fn)
        value = np.clip(value / v_scale, -448, 448).astype(jnp.float8_e4m3fn)
    else:
        key, value = key.astype(jnp.bfloat16), value.astype(jnp.bfloat16)
    packing = 4 if fp8 else 2
    rows = np.stack((key, value), axis=2).reshape(len(key), -1, key.shape[-1])
    rows = np.pad(rows, ((0, 0), (0, (-rows.shape[1]) % packing), (0, 0)))
    return jnp.asarray(rows.reshape(len(key), -1, packing, key.shape[-1]))


@pytest.mark.parametrize("page_size", [128, 1664])
def test_paired_page_abstract_shape(monkeypatch, page_size):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    shape = jax.ShapeDtypeStruct
    query = shape((256, 24, 256), jnp.bfloat16)
    output = jax.eval_shape(
        functools.partial(
            block_sparse_gqa_prefill,
            ratio=4,
            kv_heads=2,
            query_block=256,
            kv_tile=256,
            bf16_io=True,
        ),
        query,
        shape((256, 512), jnp.int32),
        shape((256,), jnp.int32),
        shape((256, 3), jnp.int32),
        shape((3, page_size, 1, 4, 256), jnp.float8_e4m3fn),
    )
    assert output == query


@pytest.mark.parametrize("page_size", [128, 1664])
def test_paired_tile_selection_for_shared_pool(monkeypatch, page_size):
    from vllm_torchtpu.kernels.qwen4_exp.prefill_tiles import (
        get_sparse_prefill_tiles,
    )

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    query = jax.ShapeDtypeStruct((8192, 24, 256), jnp.bfloat16)
    cache = jax.ShapeDtypeStruct((11, page_size, 1, 4, 256), jnp.float8_e4m3fn)
    assert get_sparse_prefill_tiles(query, cache, kv_heads=2, ratio=4) == (256, 256)


@pytest.mark.tpu_test
@pytest.mark.parametrize("starts", [(0, 256), (0, 129, 129, 255)])
@pytest.mark.parametrize("page_size", [128, 1664])
def test_permuted_odd_pages_and_partial_groups(starts, page_size):
    rng = np.random.default_rng(75)
    tokens, pages = 256, 3
    table = np.array([2, 0, 1], np.int32)
    lengths = np.resize(
        np.array(
            [
                0,
                1,
                3,
                4,
                127,
                128,
                129,
                page_size - 1,
                page_size,
                page_size + 1,
                2 * page_size,
                2 * page_size + 1,
            ],
            np.int32,
        ),
        tokens,
    )
    lengths[starts[-1] :] = 0
    query = jnp.asarray(rng.normal(size=(tokens, 24, 256)), jnp.bfloat16)
    gate = jnp.asarray(rng.uniform(-12, 12, query.shape), jnp.bfloat16)
    key = rng.normal(size=(pages * page_size, 2, 256)).astype(np.float32)
    value = rng.normal(size=key.shape).astype(np.float32)
    for logical in range(2 * page_size + 1, pages * page_size):
        physical = table[logical // page_size] * page_size + logical % page_size
        key[physical] = np.nan
        value[physical] = np.nan
    cache = pack_rows(
        jnp.asarray(key), jnp.asarray(value), fp8=True, k_scale=0.375, v_scale=2.5
    ).reshape(pages, page_size, 1, 4, 256)
    blocks = np.full((tokens, 512), -1, np.int32)
    for row, length in enumerate(lengths):
        selected = rng.permutation(length // 4)[:512]
        blocks[row, : len(selected)] = selected
    args = (
        query,
        gate,
        jnp.asarray(blocks),
        jnp.asarray(lengths),
        jnp.asarray(np.broadcast_to(table, (tokens, pages)).copy()),
        cache,
        jnp.asarray(starts, jnp.int32),
    )
    outputs = []
    for kv_tile in (128, 256):
        run = jax.jit(
            functools.partial(
                block_sparse_gqa_attention,
                kv_heads=2,
                ratio=4,
                query_block=256,
                kv_tile=kv_tile,
                k_scale=0.375,
                v_scale=2.5,
            )
        )
        output, returned_cache = jax.block_until_ready(run(*args))
        np.testing.assert_array_equal(
            np.asarray(returned_cache).view(np.uint8), np.asarray(cache).view(np.uint8)
        )
        output = np.asarray(output, np.float32)
        assert np.isfinite(output).all()
        np.testing.assert_array_equal(output[lengths == 0], 0)
        outputs.append(output)
    expected, actual = outputs
    difference = actual.astype(np.float64) - expected
    assert np.linalg.norm(difference) / np.linalg.norm(expected) < 0.01
    row_norm = np.maximum(np.linalg.norm(expected.reshape(tokens, -1), axis=1), 1e-12)
    assert (
        np.max(np.linalg.norm(difference.reshape(tokens, -1), axis=1) / row_norm) < 0.01
    )
    assert (
        np.vdot(actual.ravel(), expected.ravel())
        / (np.linalg.norm(actual) * np.linalg.norm(expected))
        > 0.999
    )


@pytest.mark.tpu_test
def test_production_shape_torch_bridge():
    import torch
    from torch_tpu._internal import compile as tpu_compile
    from torch_tpu._internal import pallas, sync

    from vllm_torchtpu.kernels.qwen4_exp.prefill_tiles import (
        get_sparse_prefill_tiles,
    )

    def core(
        query: jax.Array,
        gate: jax.Array,
        blocks: jax.Array,
        lengths: jax.Array,
        table: jax.Array,
        cache: jax.Array,
        starts: jax.Array,
    ) -> jax.Array:
        query_block, kv_tile = get_sparse_prefill_tiles(
            query, cache, kv_heads=2, ratio=4
        )
        assert kv_tile == 256, "Production geometry must exercise paired pages"
        return block_sparse_gqa_attention(
            query,
            gate,
            blocks,
            lengths,
            table,
            cache,
            starts,
            kv_heads=2,
            ratio=4,
            query_block=query_block,
            kv_tile=kv_tile,
        )[0]

    op = pallas.jax_op("vllm_tpu::test_qsa_paired_pages_bridge", core)

    @op.register_fake
    def fake(query, gate, blocks, lengths, table, cache, starts):
        return query.new_empty(query.shape)

    tokens, pages, page_size = 8192, 11, 1664
    query = torch.randn(tokens, 24, 256, device="cpu").bfloat16()
    gate = torch.zeros_like(query)
    lengths = torch.arange(1, tokens + 1, dtype=torch.int32, device="cpu")
    groups = torch.arange(512, dtype=torch.int32, device="cpu")[None, :]
    blocks = torch.where(groups < lengths[:, None] // 4, groups, -1)
    table = torch.arange(pages - 1, -1, -1, dtype=torch.int32, device="cpu")
    table = table[None, :].repeat(tokens, 1)
    cache = torch.zeros(pages, page_size, 1, 4, 256, device="cpu")
    cache[:, :, :, 1] = 1
    cache[:, :, :, 3] = -2
    cache = cache.to(torch.float8_e4m3fn)
    starts = torch.tensor([0, tokens], dtype=torch.int32, device="cpu")
    run = torch.compile(
        op, backend=tpu_compile.TpuBackend(), fullgraph=True, dynamic=False
    )
    actual = run(
        *(
            value.to("tpu")
            for value in (query, gate, blocks, lengths, table, cache, starts)
        )
    )
    sync.synchronize([actual], wait=True)
    expected = torch.empty_like(query)
    expected[:, :12] = 0.5
    expected[:, 12:] = -1
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def _dense_selected_reference(query, gate, key, value, blocks, lengths, table, ratio):
    """NumPy attention over explicit token sets, independent of Pallas metadata."""
    query, gate = np.asarray(query, np.float32), np.asarray(gate, np.float32)
    key, value = np.asarray(key, np.float32), np.asarray(value, np.float32)
    out = np.zeros_like(query)
    per_kv = query.shape[1] // key.shape[2]
    for row, length in enumerate(lengths):
        if not length:
            continue
        selected = {int(g) for g in blocks[row] if 0 <= g < length // ratio}
        positions = [
            p
            for p in range(int(length))
            if p // ratio in selected or p >= length // ratio * ratio
        ]
        if not positions:
            continue
        physical = table[row, np.asarray(positions) // key.shape[1]]
        offsets = np.asarray(positions) % key.shape[1]
        keys, values = key[physical, offsets], value[physical, offsets]
        for head in range(query.shape[1]):
            scores = (
                keys[:, head // per_kv] @ query[row, head] / np.sqrt(query.shape[-1])
            )
            probability = np.exp(scores - scores.max())
            probability /= probability.sum()
            out[row, head] = probability @ values[:, head // per_kv]
    out = out.astype(jnp.bfloat16).astype(np.float32)
    sigmoid = (1 / (1 + np.exp(-gate))).astype(jnp.bfloat16).astype(np.float32)
    return (out * sigmoid).astype(jnp.bfloat16).astype(np.float32)


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("ratio,page_size", [(4, 128), (8, 128), (4, 1664)])
def test_prefill_matches_independent_dense_selected_attention(fp8, ratio, page_size):
    rng = np.random.default_rng(1964)
    tokens, pages, heads, kv_heads, dim = 32, 6, 4, 1, 128
    query = jnp.asarray(rng.normal(0, 0.3, (tokens, heads, dim)), jnp.bfloat16)
    gate = jnp.asarray(rng.uniform(-3, 3, query.shape), jnp.bfloat16)
    key = rng.normal(0, 0.3, (pages * page_size, kv_heads, dim)).astype(np.float32)
    value = rng.normal(0, 0.3, key.shape).astype(np.float32)
    k_scale, v_scale = (0.375, 2.5) if fp8 else (1.0, 1.0)
    cache = pack_rows(key, value, fp8=fp8, k_scale=k_scale, v_scale=v_scale)
    cache = cache.reshape(pages, page_size, 1, 4 if fp8 else 2, dim)
    # Recover the actually stored values, not the pre-quantized input.
    stored = np.asarray(cache, np.float32).reshape(pages, page_size, -1, dim)
    key, value = stored[:, :, 0:1] * k_scale, stored[:, :, 1:2] * v_scale
    starts = np.array([0, 13, 13, 31], np.int32)
    owners = np.array([[4, 0, 2], [4, 0, 2], [1, 5, 3]], np.int32)
    request = np.minimum(
        np.searchsorted(starts[1:], np.arange(tokens), side="right"), 2
    )
    table = owners[request]
    lengths = np.concatenate(
        (np.arange(1, 14), page_size + np.arange(1, 19), [0])
    ).astype(np.int32)
    blocks = np.full((tokens, 16), -1, np.int32)
    for row, length in enumerate(lengths):
        selected = rng.permutation(length // ratio)[:16]
        blocks[row, : len(selected)] = selected
    expected = _dense_selected_reference(
        query, gate, key, value, blocks, lengths, table, ratio
    )
    actual, returned = jax.block_until_ready(
        jax.jit(
            functools.partial(
                block_sparse_gqa_attention,
                kv_heads=kv_heads,
                ratio=ratio,
                query_block=32,
                kv_tile=ratio * 32,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        )(
            query,
            gate,
            jnp.asarray(blocks),
            jnp.asarray(lengths),
            jnp.asarray(table),
            cache,
            jnp.asarray(starts),
        )
    )
    actual = np.asarray(actual, np.float32)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[lengths == 0], 0)
    np.testing.assert_array_equal(
        np.asarray(returned).view(np.uint8), np.asarray(cache).view(np.uint8)
    )
    error = np.linalg.norm((actual - expected).reshape(tokens, -1), axis=1)
    norm = np.linalg.norm(expected.reshape(tokens, -1), axis=1)
    assert np.max(error / np.maximum(norm, 1e-12)) < 0.01
