# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ragged tile coverage, visibility, tail writes and automatic DMA lowering."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.deepseek_v41.csa2.prefill import (
    build_query_tiles,
    csa2_prefill,
)
from vllm_torchtpu.kernels.deepseek_v41.csa2.reference import prefill_reference


@pytest.mark.cpu_test
@pytest.mark.parametrize("bq", [1, 6, 16])
def test_tile_plan_covers_each_output_once(bq):
    rng = np.random.default_rng(51)
    for lengths in ([0, 0, 0], [1, 0, 17, 0, 5], rng.integers(0, 35, 13)):
        starts = np.r_[0, np.cumsum(lengths)].astype(np.int32)
        tokens = int(starts[-1]) + 3
        tiles, count = build_query_tiles(jnp.asarray(starts), tokens, bq)
        tiles = np.asarray(tiles)[: int(count)]
        coverage = np.zeros(tokens, np.int32)
        boundaries = np.r_[starts, tokens]
        for request, start, size in tiles:
            assert 0 < size <= bq
            assert (
                boundaries[request] <= start < start + size <= boundaries[request + 1]
            )
            coverage[start : start + size] += 1
        np.testing.assert_array_equal(coverage, 1)


def prefill_inputs(banked=False, non_causal=False, heads=8, page_size=64):
    rng = np.random.default_rng(103)
    tokens = 27
    starts = np.array([0, 0, 17, 17, 25, 25], np.int32)
    positions = np.array([*range(120, 137), *range(250, 258), 0, 0], np.int32)

    def random(shape):
        return jnp.asarray(rng.normal(size=shape) * 0.2, jnp.bfloat16)

    main = random((tokens, 512, 4, 128)).astype(jnp.float8_e4m3fn)
    bits = jax.lax.bitcast_convert_type(main, jnp.uint8).astype(jnp.uint32)
    words = jnp.bitwise_or.reduce(
        bits << (8 * jnp.arange(4, dtype=jnp.uint32))[None, None, :, None], axis=2
    ).view(jnp.int32)
    base = np.array([0, 0, 0, 64, 0], np.int32)
    visible = np.array([0, 20, 0, 130, 0], np.int32)
    columns = 384 // page_size
    table = np.arange(1, 5 * columns + 1, dtype=np.int32).reshape(5, columns)
    metadata = jnp.asarray(np.column_stack((base, visible, table)))
    swa = random((table.size + 1, page_size, 512)).at[0].set(0)
    bounds = np.array([[0, 0], [0, 137], [0, 0], [123, 258], [0, 0]], np.int32)
    for request in (1, 3):
        absolute = base[request] + np.arange(384).reshape(columns, page_size)
        keep = (absolute >= visible[request]) & (absolute < bounds[request, 1])
        swa = swa.at[table[request]].set(
            jnp.where(keep[..., None], swa[table[request]], jnp.nan)
        )
    if banked:
        split = 3 * columns + 1
        swa = (swa[:split], swa[split:])
    angles = jnp.repeat(jnp.asarray(rng.normal(size=(tokens, 64)), jnp.float32), 2, -1)
    rope = jnp.stack(
        (jnp.cos(angles), jnp.sin(angles) * jnp.tile(jnp.array([1, -1]), 64)), 1
    )
    lengths = jnp.resize(jnp.array([0, 1, 127, 511, 512], jnp.int32), tokens)
    args = (
        random((tokens, heads, 512)),
        words,
        jnp.asarray(rng.uniform(0.1, 0.3, (tokens, 512)), jnp.float32),
        swa,
        jnp.asarray(starts),
        jnp.asarray(positions),
        jnp.linspace(-1, 1, heads, dtype=jnp.float32),
        rope,
    )
    kwargs = dict(
        swa_metadata=metadata,
        main_lengths=lengths,
        swa_bounds=jnp.asarray(bounds) if non_causal else None,
    )
    return args, kwargs


def numpy_reference(args, kwargs):
    q, words, scales, swa, starts, positions, sink, rope = args
    q, words, scales, starts, positions, sink, rope = map(
        np.asarray, (q, words, scales, starts, positions, sink, rope)
    )
    banks = (
        tuple(map(np.asarray, swa)) if isinstance(swa, tuple) else (np.asarray(swa),)
    )
    page_size = banks[0].shape[1]
    pages = np.asarray(kwargs["swa_metadata"])
    counts = np.asarray(kwargs["main_lengths"])
    bounds = kwargs["swa_bounds"]
    result = np.zeros(q.shape, np.float32)
    for request, (begin, end) in enumerate(zip(starts[:-1], starts[1:], strict=True)):
        for token in range(begin, end):
            bits = np.stack(
                [(words[token] >> (8 * g)).astype(np.uint8) for g in range(4)], 1
            )
            main = (
                bits.view(np.dtype(jnp.float8_e4m3fn))
                .astype(np.float32)
                .reshape(512, 512)
            )
            main = main[: counts[token]] * scales[token, : counts[token], None]
            left, right = (max(0, positions[token] - 127), positions[token] + 1)
            if bounds is not None:
                left, right = map(int, bounds[request])
            keys = range(max(left, int(pages[request, 1])), right)
            values = []
            for position in keys:
                page = pages[request, 2 + (position - pages[request, 0]) // page_size]
                bank = int(page >= banks[0].shape[0]) if len(banks) == 2 else 0
                page -= banks[0].shape[0] if bank else 0
                values.append(
                    banks[bank][page, position % page_size].astype(np.float32)
                )
            kv = np.concatenate((main, np.array(values).reshape(-1, 512)))
            logits = q[token].astype(np.float32) @ kv.T / np.sqrt(512)
            maximum = np.maximum(logits.max(-1), sink)
            p = np.exp(logits - maximum[:, None])
            p /= (p.sum(-1) + np.exp(sink - maximum))[:, None]
            output = p @ kv
            tail = output[:, 384:].copy()
            output[:, 384:] = (
                tail * rope[token, 0] + tail[:, np.arange(128) ^ 1] * rope[token, 1]
            )
            result[token] = output
    return result


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "bq,banked,non_causal,heads,page_size",
    [
        (6, False, False, 8, 64),
        (16, True, True, 8, 64),
        (16, True, False, 128, 64),
        (6, False, False, 64, 32),
        (6, True, True, 128, 32),
    ],
)
def test_prefill_visibility_and_tail_rows(bq, banked, non_causal, heads, page_size):
    args, kwargs = prefill_inputs(banked, non_causal, heads, page_size)
    # CPU validates reference math; TPU exercises the buffered kernel itself.
    actual = (
        csa2_prefill(*args, **kwargs, bq=bq)
        if jax.default_backend() == "tpu"
        else prefill_reference(*args, **kwargs)
    )
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        numpy_reference(args, kwargs),
        atol=3e-3,
        rtol=3e-2,
    )
    np.testing.assert_array_equal(actual[25:], 0)


@pytest.mark.cpu_test
@pytest.mark.parametrize("bq,banked,non_causal", [(6, False, False), (16, True, True)])
def test_prefill_traces_dynamic_tile_grid(bq, banked, non_causal):
    args, kwargs = prefill_inputs(banked, non_causal)
    mesh = jax.sharding.AbstractMesh(
        (1,), ("core",), abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    )
    with jax.sharding.use_abstract_mesh(mesh):
        traced = jax.make_jaxpr(lambda *x: csa2_prefill(*x, **kwargs, bq=bq))(*args)
    assert traced.out_avals[0].shape == args[0].shape


@pytest.mark.cpu_test
@pytest.mark.parametrize("direct_output", [False, True])
def test_buffered_index_maps_allow_epilogue_lookahead(monkeypatch, direct_output):
    args, kwargs = prefill_inputs()
    if direct_output:
        args = (*args[:4], jnp.array([0, 25], jnp.int32), *args[5:])
        kwargs["swa_metadata"] = kwargs["swa_metadata"][:1]

    def inspect_call(kernel, *, out_shape, grid_spec, **unused):
        def call(*operands):
            # NumPy rejects SMEM reads beyond the logical tile table instead
            # of silently clamping them as JAX array indexing would.
            scalars = tuple(map(np.asarray, operands[: grid_spec.num_scalar_prefetch]))
            capacity = scalars[0].shape[0]
            for spec in jax.tree.leaves(grid_spec.in_specs):
                expected = spec.index_map(capacity - 1, *scalars)
                for tile in (capacity, capacity + 1):
                    np.testing.assert_array_equal(
                        spec.index_map(tile, *scalars), expected
                    )
            return jnp.zeros(out_shape.shape, out_shape.dtype)

        return call

    monkeypatch.setattr(pl, "pallas_call", inspect_call)
    with jax.disable_jit():
        csa2_prefill(*args, **kwargs, bq=9 if direct_output else 6)


@pytest.mark.cpu_test
@pytest.mark.parametrize("tokens", [128, 288])
@pytest.mark.parametrize("requests", [1, 8])
def test_prefill_warmup_with_mostly_padding(tokens, requests):
    # Runner warmup schedules one token per request and pads the whole bucket.
    args, _ = prefill_inputs(heads=64, page_size=32)
    q, _, _, swa, _, _, sink, _ = args
    args = (
        jnp.resize(q, (tokens, 64, 512)),
        jnp.zeros((tokens, 512, 128), jnp.int32),
        jnp.zeros((tokens, 512), jnp.float32),
        swa,
        jnp.arange(requests + 1, dtype=jnp.int32),
        jnp.zeros(tokens, jnp.int32),
        sink,
        jnp.broadcast_to(jnp.array([1.0, 0.0])[None, :, None], (tokens, 2, 128)),
    )
    kwargs = dict(
        swa_metadata=jnp.concatenate(
            (jnp.zeros((requests, 2), jnp.int32), jnp.ones((requests, 9), jnp.int32)),
            axis=1,
        ),
        main_lengths=jnp.zeros(tokens, jnp.int32),
    )
    expected = prefill_reference(*args, **kwargs)
    if jax.default_backend() == "tpu":
        actual = csa2_prefill(*args, **kwargs, bq=6)
        np.testing.assert_allclose(actual, expected, atol=3e-3, rtol=3e-2)
        np.testing.assert_array_equal(actual[requests:], 0)
    else:
        mesh = jax.sharding.AbstractMesh(
            (1,),
            ("core",),
            abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"),
        )
        with jax.sharding.use_abstract_mesh(mesh):
            traced = jax.make_jaxpr(lambda *x: csa2_prefill(*x, **kwargs, bq=6))(*args)
        assert traced.out_avals[0].shape == expected.shape


@pytest.mark.cpu_test
@pytest.mark.parametrize("non_causal", [False, True])
@pytest.mark.parametrize("valid_tokens", [0, 17])
def test_single_request_direct_output_preserves_padding(non_causal, valid_tokens):
    args, kwargs = prefill_inputs(non_causal=non_causal)
    q, words, scales, swa, _, positions, sink, rope = args

    def padded(x):
        return jnp.pad(x[:17], ((0, 15), *[(0, 0)] * (x.ndim - 1)))

    args = (
        padded(q),
        padded(words),
        padded(scales),
        swa,
        jnp.array([0, valid_tokens], jnp.int32),
        padded(positions),
        sink,
        padded(rope),
    )
    kwargs = dict(
        swa_metadata=kwargs["swa_metadata"][1:2],
        main_lengths=padded(kwargs["main_lengths"]),
        swa_bounds=kwargs["swa_bounds"][1:2] if non_causal else None,
    )
    actual = (
        csa2_prefill(*args, **kwargs, bq=16)
        if jax.default_backend() == "tpu"
        else prefill_reference(*args, **kwargs)
    )
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        numpy_reference(args, kwargs),
        atol=3e-3,
        rtol=3e-2,
    )
    np.testing.assert_array_equal(actual[valid_tokens:], 0)
