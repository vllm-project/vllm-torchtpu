# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check sequence-local tile indexing, page mapping, and decode reuse."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.csa2.decode import csa2_decode


@pytest.fixture(scope="module")
def benchmark():
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "long_benchmark", root / "scripts/dsv41_bench/bench_csa2_long.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.cpu_test
def test_long_decode_traces(benchmark):
    host = benchmark.make_inputs(2, 128, 8, varied_counts=True)
    inputs = jax.tree.map(jnp.asarray, host)
    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("core",),
        abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"),
    )
    with jax.sharding.use_abstract_mesh(mesh):
        traced = jax.make_jaxpr(lambda data: csa2_decode(**data, num_tokens=128))(
            inputs
        )
    assert traced.out_avals[0].shape == inputs["q"].shape


def test_long_decode_matches_reference(benchmark):
    if jax.default_backend() != "tpu":
        pytest.skip("Async MXU operations require TPU")
    host = benchmark.make_inputs(4, 128, 8, varied_counts=True)
    expected = benchmark.reference(host, 128, range(512))
    actual = csa2_decode(**jax.device_put(host), num_tokens=128)
    output = np.asarray(actual, np.float32)
    assert np.isfinite(output).all()
    np.testing.assert_allclose(output, expected, atol=3e-3, rtol=3e-2)


@pytest.mark.parametrize("heads", [8, 64])
def test_decoder_prefill_reuses_decode_with_fp32_scales(benchmark, heads):
    if jax.default_backend() != "tpu":
        pytest.skip("Async MXU operations require TPU")
    from vllm_torchtpu.kernels.deepseek_v41.csa2.prefill import csa2_prefill

    host = benchmark.make_inputs(2, 128, heads, varied_counts=True)
    inputs = jax.tree.map(jnp.asarray, host)
    positions = (inputs["positions"][:, None] + jnp.arange(128)).reshape(-1)
    expected = csa2_prefill(
        inputs["q"],
        inputs["main_kv"],
        inputs["main_kv_scale"],
        inputs["swa_kv"],
        jnp.array([0, 128, 256], jnp.int32),
        positions,
        inputs["sink"],
        inputs["rope_cache"],
        swa_metadata=inputs["swa_page"],
        main_lengths=inputs["main_lengths"],
    )
    actual = csa2_decode(**inputs, num_tokens=128, decoder_prefill=True)
    np.testing.assert_allclose(actual, expected, atol=3e-3, rtol=3e-2)
