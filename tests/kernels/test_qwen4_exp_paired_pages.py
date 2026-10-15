# SPDX-License-Identifier: Apache-2.0
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
