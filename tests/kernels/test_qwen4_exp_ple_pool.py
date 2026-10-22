# SPDX-License-Identifier: Apache-2.0
"""Pooled state contracts: exact continuation and preservation of other owners."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.ple import short_conv
from vllm_torchtpu.kernels.qwen4_exp.pool import (
    pooled_short_conv,
    read_region,
    write_region,
)
from vllm_torchtpu.kernels.qwen4_exp.qsa_state import (
    read_ring_and_tags,
    write_ring_and_tags,
)


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("batch", [7, 8, 128])
def test_qsa_ring_tag_dma_preserves_exact_bytes_and_inactive_owners(fp8, batch):
    rng = np.random.default_rng(73)
    dtype = jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    page_size, packing = (1664, 4) if fp8 else (896, 2)
    shape = (batch + 5, page_size, 1, packing, 256)
    host = (
        rng.integers(
            0, 256, size=np.prod(shape) * np.dtype(dtype).itemsize, dtype=np.uint8
        )
        .view(np.dtype(dtype))
        .reshape(shape)
    )
    ids = np.arange(1, batch + 1, dtype=np.int32)[::-1].copy()
    ids[0], ids[-1] = 0, -1
    ids = jnp.asarray(ids)
    ring = jnp.asarray(rng.normal(size=(batch, 1024)), jnp.bfloat16)
    tags = jnp.asarray(rng.integers(0, 10000, size=(batch, 16)), jnp.int32)
    pool = jax.device_put(host)
    actual_ring, actual_tags = jax.jit(
        lambda owner: read_ring_and_tags(owner, ids, 1024, 16)
    )(pool)
    expected_ring = read_region(pool, ids, elements=1024, dtype=jnp.bfloat16)
    expected_tags = read_region(pool, ids, elements=16, dtype=jnp.int32, tok0=2)
    for actual, expected in (
        (actual_ring, expected_ring),
        (actual_tags, expected_tags),
    ):
        np.testing.assert_array_equal(
            np.ascontiguousarray(actual).view(np.uint8),
            np.ascontiguousarray(expected).view(np.uint8),
        )

    def reference(owner):
        owner = write_region(owner, ring, ids)
        return write_region(owner, tags, ids, tok0=2)

    expected = jax.jit(reference, donate_argnums=(0,))(jax.device_put(host))
    actual = jax.jit(
        lambda owner: write_ring_and_tags(owner, ring, tags, ids),
        donate_argnums=(0,),
    )(pool)
    actual_bytes = np.ascontiguousarray(actual).view(np.uint8)
    np.testing.assert_array_equal(
        actual_bytes, np.ascontiguousarray(expected).view(np.uint8)
    )
    untouched = [0, batch + 1, batch + 2, batch + 3, batch + 4]
    np.testing.assert_array_equal(
        actual_bytes[untouched], host.view(np.uint8)[untouched]
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("dtype", ["bfloat16", "float8_e4m3fn"])
@pytest.mark.parametrize("channels", [128, 10240])
def test_pooled_ple_matches_private_state_and_preserves_other_pages(dtype, channels):
    rng = np.random.default_rng(48)
    carrier = getattr(jnp, dtype)
    packing = 4 if dtype == "float8_e4m3fn" else 2
    pool = jnp.zeros((7, 1664, 1, packing, 256), carrier)
    ids = jnp.array([3, 0, 5], jnp.int32)
    starts = jnp.array([0, 3, 3, 5], jnp.int32)
    initial = jnp.array([True, False, False])
    history = jnp.asarray(rng.normal(size=(3, 9, channels)), jnp.bfloat16)
    x = jnp.asarray(rng.normal(size=(8, channels)), jnp.bfloat16)
    weight = jnp.asarray(rng.normal(size=(channels, 4)), jnp.bfloat16)
    pool = jax.jit(write_region, donate_argnums=(0,))(pool, history.reshape(3, -1), ids)
    before = np.asarray(pool).view(np.uint8).copy()
    expected, next_history = short_conv(
        x,
        weight,
        history,
        starts,
        jnp.array([0, -1, 2], jnp.int32),
        initial,
        dilation=3,
    )
    run = jax.jit(
        pooled_short_conv,
        donate_argnums=(2,),
        static_argnames=("dilation", "state_len"),
    )
    pool, actual = run(x, weight, pool, starts, ids, initial, dilation=3, state_len=9)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    restored = jax.jit(read_region, static_argnames=("elements", "dtype"))(
        pool, ids, elements=9 * channels, dtype=jnp.bfloat16
    ).reshape(3, 9, channels)
    np.testing.assert_array_equal(
        np.asarray(restored)[[0, 2]], np.asarray(next_history)[[0, 2]]
    )
    # Includes the null page: idle rows must never write it, even if their
    # transient values contain nonzero data.
    after = np.asarray(pool).view(np.uint8)
    np.testing.assert_array_equal(after[[0, 1, 2, 4, 6]], before[[0, 1, 2, 4, 6]])


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
def test_pooled_qsa_continuation_across_manager_page_and_partial_pairs(fp8):
    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import pooled_qsa_step
    from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import qsa_step

    rng = np.random.default_rng(8192)

    def rand(shape):
        return jnp.asarray(rng.normal(0, 0.3, shape), jnp.bfloat16)

    packing = 4 if fp8 else 2
    carrier = jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    pool = jnp.zeros((12, 1664, 4 // packing, packing, 256), carrier)
    native = pool.reshape(-1, 128, *pool.shape[2:]).copy()
    compressed = jnp.zeros((12 * 13, 32, 1, 128), jnp.bfloat16)
    ring = rand((3, 8, 128))
    tags = jnp.zeros((3, 8, 2), jnp.int32)
    context = jnp.array([1661, 1663, 0], jnp.int32)
    for row in (0, 1):
        tags = tags.at[row, :4, 0].set(
            context[row] - 1 - (context[row] - 1 - jnp.arange(4)) % 4
        )
    ring_ids = jnp.array([3, 6, 0], jnp.int32)
    write = jax.jit(write_region, donate_argnums=(0,), static_argnames=("tok0",))
    pool = write(pool, ring.reshape(3, -1), ring_ids)
    token_bytes = np.prod(pool.shape[2:]) * pool.dtype.itemsize
    tag_start = 8 * 128 * 2 // token_bytes
    pool = write(pool, tags.reshape(3, -1), ring_ids, tok0=tag_start)
    before = np.asarray(pool).view(np.uint8).copy()
    main = jnp.array([[2, 7], [5, 9], [0, 0]], jnp.int32)
    comp = jnp.array([[1, 8], [4, 10], [0, 0]], jnp.int32)

    def expand(t):
        return (t[..., None] * 13 + jnp.arange(13)).reshape(3, -1)

    starts = jnp.array([0, 3, 7, 7], jnp.int32)
    slots = jnp.array([0, 1, -1], jnp.int32)
    nq, nk, niq, nik = rand((256,)), rand((256,)), rand((128,)), rand((128,))
    angle = jnp.arange(4096)[:, None] * jnp.arange(32)[None, :] / 8192
    trig = jnp.pad(
        jnp.concatenate([jnp.cos(angle), jnp.sin(angle)], 1), ((0, 0), (0, 64))
    ).astype(jnp.bfloat16)
    options = dict(
        compress_ratio=4,
        token_topk=2048,
        norm_eps=1e-6,
        rotary_dim=64,
        k_scale=0.375,
        v_scale=2.5,
    )
    run = jax.jit(pooled_qsa_step, donate_argnums=(10,), static_argnames=tuple(options))
    reference = jax.jit(
        qsa_step, donate_argnums=(10, 11, 12, 13), static_argnames=tuple(options)
    )
    for step in range(3):
        positions = jnp.concatenate(
            [
                context[0] + jnp.arange(3),
                context[1] + jnp.arange(4),
                jnp.zeros(1, jnp.int32),
            ]
        )
        args = (
            rand((8, 24 * 256 * 2)),
            rand((8, 2 * 256)),
            rand((8, 2 * 256)),
            rand((8, 5 * 128)),
            nq,
            nk,
            niq,
            nik,
            positions,
            trig,
        )
        native, compressed, ring, tags, wanted, selected = reference(
            *args,
            native,
            compressed,
            ring,
            tags,
            starts,
            context,
            slots,
            expand(main),
            expand(comp),
            **options,
        )
        pool, actual, blocks = run(
            *args,
            pool,
            starts,
            context,
            # Native ring rows cover one block, padded to 128 / ratio columns.
            jnp.pad(ring_ids[:, None], ((0, 0), (0, 31))).reshape(-1),
            main,
            comp,
            **options,
        )
        np.testing.assert_allclose(
            np.asarray(actual, np.float32),
            np.asarray(wanted, np.float32),
            rtol=0.05,
            atol=0.03,
        )
        # Check full main-cache contents under both scheduler pages.
        expected_main = np.asarray(native).reshape(pool.shape)[[2, 5, 7, 9]]
        np.testing.assert_array_equal(
            np.asarray(pool)[[2, 5, 7, 9]].view(np.uint8), expected_main.view(np.uint8)
        )
        read = jax.jit(read_region, static_argnames=("elements", "dtype", "tok0"))
        restored_ring = read(pool, ring_ids, elements=8 * 128, dtype=jnp.bfloat16)
        np.testing.assert_array_equal(
            np.asarray(restored_ring)[:2], np.asarray(ring).reshape(3, -1)[:2]
        )
        restored_tags = read(
            pool, ring_ids, elements=8 * 2, dtype=jnp.int32, tok0=tag_start
        )
        np.testing.assert_array_equal(
            np.asarray(restored_tags)[:2], np.asarray(tags).reshape(3, -1)[:2]
        )
        # The compressed pool region pads the lane width to 256; compare
        # every stored BF16 key, not just the subset selected by attention.
        compressed_ids = jnp.array([1, 8, 4, 10], jnp.int32)
        restored_keys = read(
            pool, compressed_ids, elements=416 * 256, dtype=jnp.bfloat16
        ).reshape(4, 416, 256)
        private_keys = np.asarray(compressed).reshape(12, 416, 128)[[1, 8, 4, 10]]
        np.testing.assert_array_equal(
            np.asarray(restored_keys)[..., :128], private_keys
        )
        np.testing.assert_array_equal(np.asarray(restored_keys)[..., 128:], 0)
        np.testing.assert_array_equal(
            np.asarray(pool)[[0, 11]].view(np.uint8), before[[0, 11]]
        )
        context = context + jnp.array([3, 4, 0], jnp.int32)


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("starts", [[0, 128, 256], [0, 96, 256]])
def test_qsa_main_cache_universal_tiled_writer_matches_scatter(fp8, starts):
    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import write_main_cache
    from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import pack_rows

    rng = np.random.default_rng(88421)
    tokens, page_size = 256, 256
    dtype = jnp.float32 if fp8 else jnp.bfloat16
    key = jnp.asarray(
        rng.standard_normal((tokens, 2, 256), dtype=np.float32) * 0.1, dtype
    )
    value = jnp.asarray(
        rng.standard_normal((tokens, 2, 256), dtype=np.float32) * 0.1, dtype
    )
    rows = pack_rows(key, value, fp8=fp8)
    pool = jnp.zeros((5, page_size, *rows.shape[1:]), rows.dtype)
    starts = jnp.asarray(starts, jnp.int32)
    request = jnp.searchsorted(starts[1:], jnp.arange(tokens), side="right")
    logical = jnp.arange(tokens) - starts[request]
    page = 1 + request
    valid = jnp.ones((tokens,), bool)
    expected = pool.at[page, logical].set(rows)
    actual = jax.jit(write_main_cache)(pool, rows, page, logical, valid)
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint8), np.asarray(expected).view(np.uint8)
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("query_block", [32, 64, 128, 256])
def test_qsa_block_sparse_matches_exact_mask_with_partial_tail_and_nan(
    fp8, query_block
):
    from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
        block_sparse_gqa_attention,
        block_sparse_gqa_prefill,
    )
    from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import pack_rows

    rng = np.random.default_rng(731904)
    tokens, context, page_size = query_block, 2304, 128
    query = jnp.asarray(
        rng.standard_normal((tokens, 24, 256), dtype=np.float32) * 0.1,
        dtype=jnp.bfloat16,
    )
    keys = rng.standard_normal((context, 2, 256), dtype=np.float32) * 0.1
    values = rng.standard_normal((context, 2, 256), dtype=np.float32) * 0.1
    keys[2052:] = np.nan if fp8 else 0
    values[2052:] = np.nan if fp8 else 0
    cache_dtype = jnp.float32 if fp8 else jnp.bfloat16
    cache = pack_rows(
        jnp.asarray(keys, cache_dtype), jnp.asarray(values, cache_dtype), fp8=fp8
    ).reshape(
        context // page_size, page_size, 2 if not fp8 else 1, 2 if not fp8 else 4, 256
    )
    probe = np.array([0, 1, 3, 4, 2047, 2048, 2049, 2051, 2052], np.int32)
    lengths = jnp.asarray(np.resize(probe, tokens))
    complete = np.minimum(np.asarray(lengths) // 4, 512)
    groups = np.arange(512, dtype=np.int32)[None, :]
    blocks = jnp.asarray(np.where(groups < complete[:, None], groups, -1))
    table = jnp.broadcast_to(
        jnp.arange(context // page_size, dtype=jnp.int32),
        (tokens, context // page_size),
    )
    native = np.asarray(cache, np.float32).reshape(context, 4, 256)
    native = np.where(np.isfinite(native), native, 0)
    expected = np.zeros(query.shape, np.float32)
    for row, length in enumerate(np.asarray(lengths)):
        completed = min(length // 4, 512)
        ids = np.concatenate(
            (np.arange(completed * 4), np.arange(length // 4 * 4, length))
        )
        if not len(ids):
            continue
        for head in range(24):
            kv_head = head // 12
            scores = (
                native[ids, 2 * kv_head] @ np.asarray(query[row, head], np.float32) / 16
            )
            probs = np.exp(scores - scores.max())
            expected[row, head] = (probs / probs.sum()) @ native[ids, 2 * kv_head + 1]
    actual = block_sparse_gqa_prefill(
        query,
        blocks,
        lengths,
        table,
        cache,
        ratio=4,
        kv_heads=2,
        query_block=query_block,
        bf16_io=query_block > 128,
    )
    actual = np.asarray(actual, np.float32)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(
        actual,
        np.asarray(expected, np.float32),
        rtol=0.05,
        atol=0.03,
    )
    wrapped, _ = block_sparse_gqa_attention(
        query,
        jnp.zeros((tokens, 24, 1), jnp.float32),
        blocks,
        lengths,
        table,
        cache,
        jnp.asarray([0, tokens], jnp.int32),
        ratio=4,
        kv_heads=2,
        query_block=query_block,
    )
    gated_expected = (
        jnp.asarray(expected, jnp.bfloat16).astype(jnp.float32) * 0.5
    ).astype(jnp.bfloat16)
    np.testing.assert_allclose(
        np.asarray(wrapped, np.float32),
        np.asarray(gated_expected, np.float32),
        rtol=0.05,
        atol=0.03,
    )


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("dilation", [1, 3])
def test_pooled_ple_zero_history_traces_without_pool_access(fp8, dilation):
    # eval_shape traces the real region path; no backend or DMA is required.
    carrier = jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    args = (
        jax.ShapeDtypeStruct((8, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((128, 1), jnp.bfloat16),
        jax.ShapeDtypeStruct((7, 64, 1, 4 if fp8 else 2, 256), carrier),
        jax.ShapeDtypeStruct((4,), jnp.int32),
        jax.ShapeDtypeStruct((3,), jnp.int32),
        jax.ShapeDtypeStruct((3,), jnp.bool_),
    )

    def run(*values):
        return pooled_short_conv(*values, dilation=dilation, state_len=0)

    pool, output = jax.eval_shape(run, *args)
    assert pool == args[2] and output == args[0]
    graph = jax.make_jaxpr(run)(*args).jaxpr
    assert graph.outvars[0] is graph.invars[2]
    assert not any("pallas" in eq.primitive.name for eq in graph.eqns)


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("kernel_size", [1, 4])
def test_pooled_ple_history_matches_token_reference(fp8, kernel_size):
    """Independent causal oracle with explicit BF16 pre/post-SiLU boundaries."""
    rng = np.random.default_rng(1966)
    channels, dilation = 128, 3
    state_len = (kernel_size - 1) * dilation
    carrier = jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    packing = 4 if fp8 else 2
    host_pool = np.arange(7 * 64 * packing * 256, dtype=np.uint8).reshape(
        7, 64, 1, packing, 256
    )
    # Repeat byte patterns to fill BF16 carriers without changing their bits.
    host_pool = np.resize(host_pool, host_pool.size * np.dtype(carrier).itemsize)
    host_pool = host_pool.view(np.dtype(carrier)).reshape(7, 64, 1, packing, 256)
    pool = jax.device_put(host_pool)
    ids = jnp.asarray([3, 0, 5], jnp.int32)
    history = np.asarray(rng.normal(0, 0.2, (3, state_len, channels)), jnp.bfloat16)
    weight = np.asarray(rng.normal(0, 0.2, (channels, kernel_size)), jnp.bfloat16)
    if state_len:
        pool = jax.jit(write_region, donate_argnums=(0,))(
            pool, jnp.asarray(history.reshape(3, -1)), ids
        )
    run = jax.jit(
        pooled_short_conv,
        donate_argnums=(2,),
        static_argnames=("dilation", "state_len"),
    )
    for starts, initial in [
        ([0, 3, 4, 6], [True, False, False]),
        ([0, 2, 2, 4], [True, False, True]),
    ]:
        x = np.asarray(rng.normal(0, 0.2, (8, channels)), jnp.bfloat16)
        before = np.asarray(pool).view(np.uint8).copy()
        expected = np.zeros(x.shape, jnp.bfloat16)
        next_history = history.copy()
        for r, (start, stop) in enumerate(zip(starts[:-1], starts[1:])):
            if r == 1 or stop == start:
                continue
            prefix = history[r] if initial[r] else np.zeros_like(history[r])
            for t in range(start, stop):
                accum = np.zeros(channels, np.float32)
                for tap in range(kernel_size):
                    relative = t - start - state_len + tap * dilation
                    value = (
                        x[start + relative]
                        if relative >= 0
                        else prefix[state_len + relative]
                    )
                    accum += value.astype(np.float32) * weight[:, tap].astype(
                        np.float32
                    )
                pre = accum.astype(jnp.bfloat16).astype(np.float32)
                expected[t] = (pre / (1 + np.exp(-pre))).astype(jnp.bfloat16)
            if state_len:
                next_history[r] = np.concatenate((prefix, x[start:stop]))[-state_len:]
        pool, output = run(
            jnp.asarray(x),
            jnp.asarray(weight),
            pool,
            jnp.asarray(starts, jnp.int32),
            ids,
            jnp.asarray(initial),
            dilation=dilation,
            state_len=state_len,
        )
        np.testing.assert_allclose(
            np.asarray(output, np.float32),
            expected.astype(np.float32),
            rtol=0.01,
            atol=0.001,
        )
        after = np.asarray(pool).view(np.uint8)
        if state_len:
            restored = jax.jit(read_region, static_argnames=("elements", "dtype"))(
                pool, ids, elements=state_len * channels, dtype=jnp.bfloat16
            ).reshape(3, state_len, channels)
            np.testing.assert_array_equal(
                np.asarray(restored)[[0, 2]], next_history[[0, 2]]
            )
            np.testing.assert_array_equal(
                after[[0, 1, 2, 4, 6]], before[[0, 1, 2, 4, 6]]
            )
            token_bytes = np.prod(pool.shape[2:]) * pool.dtype.itemsize
            used = (state_len * channels * 2 + token_bytes - 1) // token_bytes
            np.testing.assert_array_equal(after[:, used:], before[:, used:])
        else:
            np.testing.assert_array_equal(after, before)
        history = next_history
