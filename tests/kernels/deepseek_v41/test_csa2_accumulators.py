# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check async accumulator reuse against synchronous attention math."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend import core

from vllm_torchtpu.kernels.deepseek_v41.csa2 import csa2


@pytest.fixture
def cpu_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


@pytest.mark.cpu_test
@pytest.mark.parametrize("fp8", [False, True])
def test_csa_mxu_operand_precision(fp8: bool) -> None:
    """The asynchronous path must retain FP8 operands through MXU submission."""
    shape = jax.ShapeDtypeStruct
    args = (
        shape((6, 64, 512), jnp.bfloat16),
        shape((6, 512, 128), jnp.int32),
        shape((6, 512), jnp.float32),
        shape((6, 128, 512), jnp.bfloat16),
        shape((2,), jnp.int32),
        shape((6,), jnp.int32),
        shape((64,), jnp.float32),
        shape((6, 2, 128), jnp.float32),
    )
    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("x",),
        abstract_device=jax.sharding.AbstractDevice(
            device_kind="TPU7x", num_cores=1, platform="tpu"
        ),
    )
    with jax.sharding.use_abstract_mesh(mesh):
        traced = jax.make_jaxpr(functools.partial(csa2, fp8=fp8))(*args)
    operands = {"matmul_push_rhs": set(), "matmul_acc_lhs": set()}

    def collect(value):
        if isinstance(value, core.Jaxpr):
            for equation in value.eqns:
                name = equation.primitive.name
                if name in operands:
                    index = 1 if name == "matmul_acc_lhs" else 0
                    operands[name].add(equation.invars[index].aval.dtype)
                collect(equation.params)
        elif isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)

    collect(traced)
    expected = {jnp.dtype(jnp.bfloat16)}
    if fp8:
        expected.add(jnp.dtype(jnp.float8_e4m3fn))
    assert operands["matmul_push_rhs"] == operands["matmul_acc_lhs"] == expected


@pytest.mark.cpu_test
@pytest.mark.parametrize("packed", [False, True])
def test_fp8_csa_matches_scaled_attention(cpu_device, packed: bool) -> None:
    """Check both scale placements, zero rows and masks against dense math."""
    rng = np.random.default_rng(43)
    tokens, heads = 3, 8
    q = jnp.asarray(rng.normal(size=(tokens, heads, 512)), jnp.bfloat16)
    q = q.at[0].set(0)
    kv = jnp.asarray(rng.normal(size=(tokens, 512, 512)), jnp.float8_e4m3fn)
    scales = jnp.asarray(rng.uniform(0.01, 1.5, (tokens, 512)), jnp.float32)
    scales = scales.at[0].set(0)
    swa = jnp.asarray(rng.normal(size=(1, 128, 512)) * 0.3, jnp.bfloat16)
    swa = jnp.broadcast_to(swa, (tokens, 128, 512))
    counts = jnp.array([0, 17, 512], jnp.int32)
    sink = jnp.linspace(-1, 1, heads, dtype=jnp.float32)
    rope = jnp.stack((jnp.ones((tokens, 128)), jnp.zeros((tokens, 128))), 1)
    main = kv.reshape(tokens, 512, 4, 128)
    if packed:
        bits = jax.lax.bitcast_convert_type(main, jnp.uint8).astype(jnp.uint32)
        words = sum(bits[:, :, group] << (8 * group) for group in range(4))
        main = jax.lax.bitcast_convert_type(words, jnp.int32)
    args = (
        q,
        main,
        scales,
        swa,
        jnp.array([0, tokens], jnp.int32),
        jnp.arange(20, 20 + tokens, dtype=jnp.int32),
        sink,
        rope,
    )
    actual = csa2(*args, main_lengths=counts, fp8=True, interpret=True, bq=3)

    def quantized(values):
        values = np.asarray(values, np.float32)
        row_scale = np.max(np.abs(values), axis=-1, keepdims=True) / 448
        row_scale = np.where(row_scale > 0, row_scale, 1)
        rounded = np.asarray(
            jnp.asarray(values / row_scale, jnp.float8_e4m3fn), np.float32
        )
        return rounded, row_scale

    query, q_scale = quantized(q)
    raw = np.asarray(kv, np.float32)
    main_logits = (
        (query @ raw.swapaxes(-1, -2)) * q_scale * np.asarray(scales)[:, None, :]
    )
    main_logits = np.where(
        np.arange(512) < np.asarray(counts)[:, None, None],
        main_logits / np.sqrt(512),
        -np.inf,
    )
    swa_logits = (
        np.asarray(q, np.float32)
        @ np.asarray(swa, np.float32).swapaxes(-1, -2)
        / np.sqrt(512)
    )
    swa_logits = np.where(
        np.arange(128) <= np.arange(20, 23)[:, None, None], swa_logits, -np.inf
    )
    logits = np.concatenate((main_logits, swa_logits), -1)
    maximum = np.maximum(
        np.max(logits, axis=-1, keepdims=True), np.asarray(sink)[None, :, None]
    )
    exp = np.exp(logits - maximum)
    probabilities = exp / (
        np.sum(exp, axis=-1, keepdims=True)
        + np.exp(np.asarray(sink)[None, :, None] - maximum)
    )
    weighted, p_scale = quantized(
        probabilities[:, :, :512] * np.asarray(scales)[:, None, :]
    )
    swa_p = np.asarray(jnp.asarray(probabilities[:, :, 512:], jnp.bfloat16), np.float32)
    expected = (weighted @ raw) * p_scale + swa_p @ np.asarray(swa, np.float32)
    np.testing.assert_allclose(
        np.asarray(actual, np.float32), expected, atol=3e-3, rtol=2e-2
    )
    baseline = csa2(*args, main_lengths=counts, interpret=True, bq=3)
    relative_error = np.linalg.norm(
        np.asarray(actual, np.float32) - np.asarray(baseline, np.float32)
    ) / np.linalg.norm(np.asarray(baseline, np.float32))
    assert relative_error < 0.06


@pytest.mark.parametrize("heads", [8, 16, 32, 64, 96, 104, 128])
@pytest.mark.parametrize("fp8", [False, True])
def test_csa2_accumulator_pipeline_matches_interpret(heads: int, fp8: bool) -> None:
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
    scales = jnp.asarray(rng.uniform(0.1, 2, (tokens, 512)), jnp.float32)
    # At these positions all windows start at zero. Reuse the same KV rows
    # within each request, as required by the grouped SWA DMA contract.
    swa = jnp.broadcast_to(random((1, 128, 512)), (tokens, 128, 512))
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
    expected = csa2(*args, bq=6, main_lengths=lengths, interpret=True, fp8=fp8)
    actual = csa2(*args, bq=6, main_lengths=lengths, fp8=fp8)
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        np.asarray(expected, np.float32),
        atol=3e-3,
        rtol=2e-2,
    )
    np.testing.assert_array_equal(actual[21:], 0)
