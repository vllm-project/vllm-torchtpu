# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check sequence-local tile indexing, page mapping, and decode reuse."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.csa2.csa2 import csa2


@pytest.fixture(scope="module")
def benchmark():
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "long_benchmark", root / "scripts/dsv41_bench/bench_csa2.py"
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
        traced = jax.make_jaxpr(lambda data: csa2(**data, bq=16))(inputs)
    assert traced.out_avals[0].shape == inputs["q"].shape


def test_long_decode_matches_reference(benchmark):
    if jax.default_backend() != "tpu":
        pytest.skip("Async MXU operations require TPU")
    host = benchmark.make_inputs(4, 128, 8, varied_counts=True)
    expected = benchmark.reference(host, range(512))
    actual = csa2(**jax.device_put(host), bq=16)
    output = np.asarray(actual, np.float32)
    assert np.isfinite(output).all()
    np.testing.assert_allclose(output, expected, atol=3e-3, rtol=3e-2)


@pytest.mark.parametrize("batch", [8, 9, 11, 12, 13, 48])
@pytest.mark.parametrize("page_size", [32, 64])
def test_decode_preserves_page_mapping_and_fp32_scales(benchmark, batch, page_size):
    if jax.default_backend() != "tpu":
        pytest.skip("Async MXU operations require TPU")
    host = benchmark.make_inputs(batch, 6, 64, varied_counts=True, page_size=page_size)
    expected = benchmark.reference(host, range(batch * 6))
    actual = csa2(**jax.device_put(host), bq=6)
    np.testing.assert_allclose(actual, expected, atol=3e-3, rtol=3e-2)
    assert np.isfinite(actual).all()
