# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Admission and interpreted arithmetic for shapes beyond the original caps.

Run with JAX_PLATFORMS=cpu. Interpretation checks routing/DMA indexing and
arithmetic, not TPU resource capacity or compiled device performance.
"""

import functools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental import moe_dense_expert as kernel


@pytest.mark.parametrize(
    "capacity_mib, expected", [(128, 3), (64, 3), (48, 2), (32, 1)]
)
def test_qwen_buffers_follow_vmem_capacity(capacity_mib, expected):
    # The performance UT's local kernel sees all 256 tokens and 64 experts.
    assert (
        kernel._select_expert_buffers(256, 4096, 1024, 64, capacity_mib * 1024**2)
        == expected
    )


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((256, 4096, 2048, 64), 1),
        ((1024, 4096, 1024, 64), 1),
        ((256, 4096, 1024, 8192), 2),
        ((16, 512, 512, 4), 3),
    ],
)
def test_buffer_budget_includes_weights_activations_and_routes(shape, expected):
    assert kernel._select_expert_buffers(*shape, 64 * 1024**2) == expected


def test_buffer_selection_leaves_resource_failure_to_compiler():
    assert kernel._select_expert_buffers(1024, 8192, 4096, 512, 16 * 1024**2) == 1


def _shapes(m, h, i, e):
    return (
        jax.ShapeDtypeStruct((m, h), jnp.bfloat16),
        jax.ShapeDtypeStruct((e, h, 2 * i), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((e, i, h), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((e, 1, 1, 2 * i), jnp.float32),
        jax.ShapeDtypeStruct((e, 1, 1, h), jnp.float32),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (272, 512, 512, 4),
        (16, 1536, 1536, 2),
        (16, 512, 512, 129),
        (1024, 8192, 4096, 512),
    ],
)
@pytest.mark.parametrize("skip_padded_tokens", [False, True])
def test_aligned_shapes_have_no_resource_admission_cap(shape, skip_padded_tokens):
    # Includes a deliberately enormous shape: resource admission belongs to
    # the compiler, so this predicate must neither allocate nor query a TPU.
    assert kernel.can_use_dense_expert(
        *_shapes(*shape),
        None,
        None,
        activation="silu",
        rhs_quant_dtype=None,
        skip_padded_tokens=skip_padded_tokens,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (15, 512, 512, 4),
        (16, 768, 512, 4),
        (16, 512, 768, 4),
        (0, 512, 512, 4),
        (16, 0, 512, 4),
        (16, 512, 0, 4),
        (16, 512, 512, 0),
    ],
)
def test_alignment_and_positive_dimensions_remain_required(shape):
    assert not kernel.can_use_dense_expert(
        *_shapes(*shape),
        None,
        None,
        activation="silu",
        rhs_quant_dtype=None,
        skip_padded_tokens=False,
    )


@pytest.mark.parametrize(
    "rhs_dtype, supported", [(None, True), (jnp.float8_e4m3fn, True), (jnp.int4, False)]
)
def test_only_unpacked_fp8_arithmetic_is_supported(rhs_dtype, supported):
    assert (
        kernel.can_use_dense_expert(
            *_shapes(16, 512, 512, 4),
            None,
            None,
            activation="silu",
            rhs_quant_dtype=rhs_dtype,
            skip_padded_tokens=False,
        )
        == supported
    )


def _reference_expert(x, w1, w2, s1, s2):
    def matmul(lhs, rhs, scale):
        # Independent full-output contraction per quantization block, with
        # the documented BF16 intermediate rounding and accumulation.
        result = jnp.zeros((lhs.shape[0], rhs.shape[1]), jnp.bfloat16)
        for start in range(0, lhs.shape[1], 512):
            block = lhs[:, start : start + 512]
            xs = jnp.max(jnp.abs(block), axis=1, keepdims=True) / 448.0
            inv = jnp.where(xs == 0, 0, 1 / xs)
            q = (block * inv).astype(jnp.float8_e4m3fn)
            part = jnp.matmul(
                q, rhs[start : start + 512], preferred_element_type=jnp.float32
            ).astype(jnp.bfloat16)
            result += (part * xs) * scale.astype(jnp.bfloat16)
        return result

    gate, up = jnp.split(matmul(x, w1, s1), 2, axis=1)
    return matmul(jax.nn.silu(gate) * up, w2, s2)


@pytest.mark.parametrize(
    "shape", [(272, 512, 512, 2), (16, 1536, 1536, 2), (16, 512, 512, 129)]
)
def test_expanded_shapes_match_routed_reference(shape, monkeypatch):
    if jax.default_backend() != "cpu":
        pytest.skip("CPU Pallas interpreter; run with JAX_PLATFORMS=cpu")
    m, h, i, e = shape
    rng = np.random.default_rng(20260922)
    x = jnp.asarray(rng.normal(size=(m, h)), jnp.bfloat16)
    w1 = jnp.asarray(rng.integers(-4, 5, size=(e, h, 2 * i)), jnp.float8_e4m3fn)
    w2 = jnp.asarray(rng.integers(-4, 5, size=(e, i, h)), jnp.float8_e4m3fn)
    s1 = jnp.full((e, 1, 1, 2 * i), 0.01, jnp.float32)
    s2 = jnp.full((e, 1, 1, h), 0.01, jnp.float32)
    # Exercise both sides of the 128-column boundary, duplicate IDs, an
    # invalid local ID and a padded row whose otherwise-valid weights are 0.
    route = [0, e - 2, e - 1, e - 1, -1]
    ids = jnp.asarray(np.tile(route, (m, 1)), jnp.int32)
    weights = (
        jnp.asarray(np.tile([0.125, 0.25, 0.125, 0.125, 0.0], (m, 1)), jnp.bfloat16)
        .at[-1]
        .set(0)
    )
    expected = jnp.zeros((m, h), jnp.float32)
    for expert in sorted(set(route) - {-1}):
        output = _reference_expert(
            x, w1[expert], w2[expert], s1[expert, 0, 0], s2[expert, 0, 0]
        )
        coeff = jnp.sum(
            jnp.where(ids == expert, weights.astype(jnp.float32), 0), axis=1
        )
        expected += jnp.where(
            coeff[:, None] != 0, output.astype(jnp.float32) * coeff[:, None], 0
        )
    expected = np.asarray(expected.astype(jnp.bfloat16), dtype=np.float32)
    monkeypatch.setattr(
        kernel.pl,
        "pallas_call",
        functools.partial(kernel.pl.pallas_call, interpret=True),
    )
    monkeypatch.setattr(
        kernel.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(vmem_capacity_bytes=64 * 1024**2),
    )
    kernel.dense_expert_moe.clear_cache()
    try:
        actual = np.asarray(
            kernel.dense_expert_moe(x, w1, w2, s1, s2, weights, ids), dtype=np.float32
        )
    finally:
        kernel.dense_expert_moe.clear_cache()
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[-1], 0)
    relative_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert relative_l2 < 0.008, relative_l2


@pytest.mark.parametrize(
    "m, threshold, dense",
    [
        (256, 0, False),
        (256, 1024, True),
        (1024, 1024, True),
        (1040, 1024, False),
        (16384, 1024, False),
        (256, 128, False),
        (272, 272, True),
        (15, 1024, False),
    ],
)
def test_token_threshold_dispatch_and_resource_failure_propagation(
    m, threshold, dense, monkeypatch
):
    from vllm_torchtpu.layers.core import fused_moe_gmm as wrapper

    class ResourceFailure(RuntimeError):
        pass

    class StandardSelected(RuntimeError):
        pass

    def resource_failure(*args, **kwargs):
        raise ResourceFailure("simulated compiler resource exhaustion")

    def standard_selected(*args, **kwargs):
        raise StandardSelected("original GMM path")

    monkeypatch.setattr(wrapper, "_DENSE_EXPERT_THRESHOLD", threshold)
    monkeypatch.setattr(wrapper, "dense_expert_moe", resource_failure)
    monkeypatch.setattr(wrapper, "prepare_routed_gmm_inputs", standard_selected)
    wrapper.fused_moe_func.clear_cache()
    x, w1, w2, s1, s2 = _shapes(m, 512, 512, 4)
    args = (
        x,
        w1,
        w2,
        s1,
        s2,
        None,
        None,
        jax.ShapeDtypeStruct((m, 10), jnp.bfloat16),
        jax.ShapeDtypeStruct((m, 10), jnp.int32),
    )
    error = ResourceFailure if dense else StandardSelected
    try:
        with pytest.raises(error):
            jax.eval_shape(wrapper.fused_moe_func, *args, topk=10)
    finally:
        wrapper.fused_moe_func.clear_cache()


@pytest.mark.parametrize(
    "value, expected", [(None, 0), ("", 0), ("0", 0), ("1024", 1024), ("256", 256)]
)
def test_dense_expert_threshold_environment(monkeypatch, value, expected):
    from vllm_torchtpu import envs

    name = "TPU_MOE_DENSE_EXPERT_THRESHOLD"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    assert envs.environment_variables[name]() == expected


@pytest.mark.parametrize("value", ["-1", "dense_expert", "1.5"])
def test_dense_expert_threshold_rejects_invalid_value(monkeypatch, value):
    from vllm_torchtpu import envs

    name = "TPU_MOE_DENSE_EXPERT_THRESHOLD"
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        envs.environment_variables[name]()
