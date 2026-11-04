# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check async accumulator reuse against synchronous attention math."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.csa2.cache import encode_csa2_kv
from vllm_torchtpu.kernels.deepseek_v41.csa2.csa2 import csa2
from vllm_torchtpu.kernels.deepseek_v41.csa2.pipeline import (
    csa2_pipeline,
)
from vllm_torchtpu.kernels.deepseek_v41.csa2.reference import prefill_reference


def _decode_inputs(batch, heads, page_size):
    rng = np.random.default_rng(83)
    tokens = batch * 6

    def random(shape):
        return jnp.asarray(rng.normal(size=shape) * 0.2, jnp.bfloat16)

    first = jnp.array([190, 1024, 0][:batch], jnp.int32)
    positions = (first[:, None] + jnp.arange(6)).reshape(-1)
    starts = jnp.arange(batch + 1, dtype=jnp.int32) * 6
    words, scales = encode_csa2_kv(random((tokens * 512, 512)))
    swa = random((1 + batch * 4, page_size, 512)).at[0].set(0)
    base = jnp.maximum(0, first - 127) // page_size * page_size
    minimum = jnp.maximum(0, first - 80)
    metadata = jnp.concatenate(
        (
            base[:, None],
            minimum[:, None],
            jnp.arange(1, batch * 4 + 1, dtype=jnp.int32).reshape(batch, 4),
        ),
        axis=1,
    )
    angles = jnp.asarray(rng.normal(size=(tokens, 64)), jnp.float32)
    rope = jnp.stack(
        (
            jnp.repeat(jnp.cos(angles), 2, axis=1),
            jnp.repeat(jnp.sin(angles), 2, axis=1),
        ),
        axis=1,
    )
    counts = jnp.resize(jnp.array([0, 1, 17, 127, 511, 512], jnp.int32), tokens)
    args = (
        random((tokens, heads, 512)),
        words.reshape(tokens, 512, 128),
        scales.reshape(tokens, 512),
        swa,
        starts,
        positions,
        jnp.linspace(-1, 1, heads, dtype=jnp.float32),
        rope,
    )
    return args, dict(
        swa_metadata=metadata,
        swa_bounds=None,
        main_lengths=counts,
        scale=512**-0.5,
        compress_ratio=2,
        bq=6,
    )


@pytest.mark.cpu_test
def test_decode_traces_on_pinned_jax():
    args, kwargs = _decode_inputs(2, 8, 64)
    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("core",),
        abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"),
    )
    with jax.sharding.use_abstract_mesh(mesh):
        traced = jax.make_jaxpr(lambda *x: csa2(*x, **kwargs))(*args)
        assert traced.out_avals[0].shape == args[0].shape


@pytest.mark.parametrize(
    "batch,heads,page_size", [(1, 8, 64), (2, 64, 64), (3, 24, 128), (2, 128, 256)]
)
def test_decode_matches_ragged_reference_with_authoritative_counts(
    batch, heads, page_size
):
    if jax.default_backend() != "tpu":
        pytest.skip("Compiled decode accumulator operations require TPU")
    args, kwargs = _decode_inputs(batch, heads, page_size)
    expected = prefill_reference(
        *args,
        swa_metadata=kwargs["swa_metadata"],
        main_lengths=kwargs["main_lengths"],
        swa_bounds=kwargs["swa_bounds"],
        scale=kwargs["scale"],
    )
    actual = csa2(*args, **kwargs)
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        np.asarray(expected, np.float32),
        atol=3e-3,
        rtol=2e-2,
    )


@pytest.mark.parametrize("chunk_size", [4, 12])
def test_decode_gather_and_reuse_preserve_valid_counts(chunk_size):
    if jax.default_backend() != "tpu":
        pytest.skip("Decode gather requires TPU SparseCore")
    args, kwargs = _decode_inputs(2, 8, 64)
    q, words, scales, swa, starts, positions, sink, rope = args
    # Uniform logits expose scale rounding in PV rather than hiding it behind
    # a nearly one-hot softmax. A chunk may split the six-query requests.
    q = jnp.zeros_like(q)
    scales = jnp.linspace(0.01, 0.8, scales.size, dtype=jnp.float32).reshape(
        scales.shape
    )
    args = (q, words, scales, swa, starts, positions, sink, rope)
    counts = kwargs["main_lengths"]
    records = jnp.arange(q.shape[0] * 512, dtype=jnp.int32).reshape(-1, 512)
    records = jnp.where(jnp.arange(512) < counts[:, None], records, -1)
    output, gathered, gathered_scales = csa2_pipeline(
        q,
        words.reshape(-1, 128),
        scales.reshape(-1),
        records,
        swa,
        starts,
        positions,
        sink,
        rope,
        kwargs["swa_metadata"],
        None,
        counts,
        chunk_size=chunk_size,
        bq=6,
        scale=512**-0.5,
        compress_ratio=2,
    )
    expected_words = jnp.where(records[..., None] >= 0, words, 0)
    expected_scales = jnp.where(records >= 0, scales, 0)
    np.testing.assert_array_equal(gathered, expected_words)
    np.testing.assert_array_equal(gathered_scales, expected_scales)
    expected = csa2(*args, **kwargs)
    np.testing.assert_allclose(output, expected, atol=3e-3, rtol=2e-2)
    reused = csa2(q, gathered, gathered_scales, *args[3:], **kwargs)
    np.testing.assert_array_equal(reused, output)


@pytest.mark.parametrize("heads", [8, 16, 32, 64, 96, 104, 128])
def test_csa2_accumulator_pipeline_matches_reference(heads: int) -> None:
    """Cover both accumulator layouts and every one-to-six-query tail."""
    if jax.default_backend() != "tpu":
        pytest.skip("Compiled accumulator operations require TPU")
    rng = np.random.default_rng(107)
    starts = np.array([0, 1, 3, 6, 10, 15, 21], np.int32)
    tokens = 23
    positions = np.zeros(tokens, np.int32)
    for start, end in zip(starts[:-1], starts[1:], strict=True):
        positions[start:end] = np.arange(20, 20 + end - start)

    def random(shape: tuple[int, ...]) -> jax.Array:
        return jnp.asarray(rng.normal(size=shape) * 0.2, jnp.bfloat16)

    q = random((tokens, heads, 512))
    main = random((tokens, 512, 4, 128)).astype(jnp.float8_e4m3fn)
    bits = jax.lax.bitcast_convert_type(main, jnp.uint8).astype(jnp.uint32)
    main = jnp.bitwise_or.reduce(
        bits << (8 * jnp.arange(4, dtype=jnp.uint32))[None, None, :, None], axis=2
    ).view(jnp.int32)
    scales = jnp.ones((tokens, 512), jnp.float32)
    swa = jnp.concatenate((jnp.zeros((1, 64, 512), jnp.bfloat16), random((2, 64, 512))))
    metadata = jnp.tile(jnp.array([[0, 0, 1, 2]], jnp.int32), (len(starts) - 1, 1))
    angles = jnp.repeat(jnp.asarray(rng.normal(size=(tokens, 32)) * 0.1), 2, -1)
    signed_sin = jnp.sin(angles) * jnp.tile(jnp.array([1, -1]), 32)
    rope = jnp.stack(
        (
            jnp.concatenate((jnp.ones_like(angles), jnp.cos(angles)), -1),
            jnp.concatenate((jnp.zeros_like(angles), signed_sin), -1),
        ),
        1,
    )
    lengths = jnp.resize(jnp.array([0, 1, 127, 511, 512], jnp.int32), tokens)
    args = (
        q,
        main,
        scales,
        swa,
        jnp.asarray(starts),
        jnp.asarray(positions),
        jnp.linspace(-1, 1, heads, dtype=jnp.float32),
        rope,
    )
    expected = prefill_reference(*args, main_lengths=lengths, swa_metadata=metadata)
    actual = csa2(*args, bq=6, main_lengths=lengths, swa_metadata=metadata)
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        np.asarray(expected, np.float32),
        atol=3e-3,
        rtol=2e-2,
    )
    np.testing.assert_array_equal(actual[21:], 0)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "bq,round_scales", [(1, False), (6, None), (8, False), (8, True)]
)
@pytest.mark.parametrize("batch", [1, 2])
def test_unified_attention_traces(bq, round_scales, batch):
    args, kwargs = _decode_inputs(batch, 8, 64)
    kwargs["bq"] = bq
    mesh = jax.sharding.AbstractMesh(
        (1,), ("core",), abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    )
    with jax.sharding.use_abstract_mesh(mesh):
        traced = jax.make_jaxpr(
            lambda *x: csa2(*x, **kwargs, round_scales=round_scales)
        )(*args)
    assert traced.out_avals[0].shape == args[0].shape


@pytest.mark.parametrize("round_scales", [False, True])
@pytest.mark.parametrize("explicit_bounds", [False, True])
def test_unified_precision_and_visibility_are_independent(
    round_scales, explicit_bounds
):
    if jax.default_backend() != "tpu":
        pytest.skip("Async MXU operations require TPU")
    args, kwargs = _decode_inputs(2, 8, 64)
    if explicit_bounds:
        kwargs["swa_bounds"] = jnp.array([[128, 196], [960, 1030]], jnp.int32)
    expected = prefill_reference(
        *args,
        swa_metadata=kwargs["swa_metadata"],
        main_lengths=kwargs["main_lengths"],
        swa_bounds=kwargs["swa_bounds"],
        round_scales=round_scales,
    )
    actual = csa2(*args, **kwargs, round_scales=round_scales)
    np.testing.assert_allclose(actual, expected, atol=3e-3, rtol=3e-2)


@pytest.mark.parametrize("regular", [False, True])
def test_auto_scale_rounding_checks_request_boundaries(regular):
    if jax.default_backend() != "tpu":
        pytest.skip("Async MXU operations require TPU")
    args, kwargs = _decode_inputs(2, 8, 64)
    if not regular:
        args = (
            *args[:4],
            jnp.array([0, 3, 12], jnp.int32),
            jnp.concatenate((jnp.arange(190, 193), jnp.arange(1024, 1033))),
            *args[6:],
        )
    automatic = csa2(*args, **kwargs)
    explicit = csa2(*args, **kwargs, round_scales=regular)
    np.testing.assert_array_equal(automatic, explicit)


@pytest.mark.parametrize("heads", [8, 64])
def test_long_prefill_uses_common_interface_across_request_tiles(heads):
    if jax.default_backend() != "tpu":
        pytest.skip("Async MXU operations require TPU")
    args, kwargs = _decode_inputs(2, heads, 256)
    q, words, scales, swa, _, positions, sink, rope = args
    queries = 128
    indices = (jnp.arange(2)[:, None] * 6 + jnp.arange(queries) % 6).reshape(-1)
    args = (
        q[indices],
        words[indices],
        scales[indices],
        swa,
        jnp.arange(3, dtype=jnp.int32) * queries,
        (positions[::6, None] + jnp.arange(queries)).reshape(-1),
        sink,
        rope[indices],
    )
    counts = kwargs["main_lengths"][indices]
    expected = prefill_reference(
        *args,
        swa_metadata=kwargs["swa_metadata"],
        main_lengths=counts,
    )
    actual = csa2(
        *args,
        swa_metadata=kwargs["swa_metadata"],
        main_lengths=counts,
        round_scales=False,
        bq=8,
    )
    np.testing.assert_allclose(actual, expected, atol=3e-3, rtol=3e-2)


@pytest.mark.parametrize("valid_tokens", [0, 1])
def test_single_query_or_padding_drains_pipeline(valid_tokens):
    if jax.default_backend() != "tpu":
        pytest.skip("Async MXU operations require TPU")
    args, kwargs = _decode_inputs(1, 8, 64)
    q, words, scales, swa, _, positions, sink, rope = args
    args = (
        q[:1],
        words[:1],
        scales[:1],
        swa,
        jnp.array([0, valid_tokens], jnp.int32),
        positions[:1],
        sink,
        rope[:1],
    )
    kwargs["main_lengths"] = kwargs["main_lengths"][:1]
    expected = prefill_reference(
        *args, swa_metadata=kwargs["swa_metadata"], main_lengths=kwargs["main_lengths"]
    )
    actual = csa2(*args, **kwargs)
    np.testing.assert_allclose(actual, expected, atol=3e-3, rtol=3e-2)


@pytest.mark.parametrize("tiles", [1, 2, 3, 4, 5, 7, 8, 9])
def test_shared_tile_loop_preserves_buffer_wraparound(tiles):
    """Exercise startup, both buffer rings, and every final input-buffer slot."""
    if jax.default_backend() != "tpu":
        pytest.skip("Async MXU operations require TPU")
    args, kwargs = _decode_inputs(1, 8, 64)
    q, words, scales, swa, _, positions, sink, rope = args
    tokens = tiles * 2
    indices = jnp.arange(tokens) % q.shape[0]
    args = (
        q[indices],
        words[indices],
        scales[indices],
        swa,
        jnp.array([0, 0, tokens], jnp.int32),
        positions[0] + jnp.arange(tokens),
        sink,
        rope[indices],
    )
    metadata = jnp.repeat(kwargs["swa_metadata"], 2, axis=0)
    lengths = kwargs["main_lengths"][indices]
    expected = prefill_reference(*args, swa_metadata=metadata, main_lengths=lengths)

    def run():
        return csa2(
            *args,
            swa_metadata=metadata,
            main_lengths=lengths,
            bq=2,
            round_scales=False,
        )

    actual = run()
    np.testing.assert_allclose(actual, expected, atol=3e-3, rtol=3e-2)
    # A staged RHS left unconsumed at return can corrupt the next invocation.
    for _ in range(3):
        np.testing.assert_array_equal(run(), actual)


@pytest.mark.cpu_test
@pytest.mark.parametrize("valid_tokens", [0, 1])
def test_single_tile_attention_traces_through_final_drain(valid_tokens):
    args, kwargs = _decode_inputs(1, 8, 64)
    q, words, scales, swa, _, positions, sink, rope = args
    args = (
        q[:1],
        words[:1],
        scales[:1],
        swa,
        jnp.array([0, valid_tokens], jnp.int32),
        positions[:1],
        sink,
        rope[:1],
    )
    kwargs["main_lengths"] = kwargs["main_lengths"][:1]
    mesh = jax.sharding.AbstractMesh(
        (1,), ("core",), abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    )
    with jax.sharding.use_abstract_mesh(mesh):
        traced = jax.make_jaxpr(lambda *x: csa2(*x, **kwargs))(*args)
    assert traced.out_avals[0].shape == (1, 8, 512)
    assert traced.out_avals[0].dtype == jnp.bfloat16
