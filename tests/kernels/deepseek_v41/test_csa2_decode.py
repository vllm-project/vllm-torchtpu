# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check sequence-local tile indexing, page mapping, and decode reuse."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.deepseek_v41.csa2.decode import csa2_decode


@pytest.mark.cpu_test
@pytest.mark.parametrize("batch,tokens", [(3, 6), (4, 6), (2, 16)])
def test_decode_buffered_index_maps_allow_epilogue_lookahead(
    monkeypatch, benchmark, batch, tokens
):
    inputs = jax.tree.map(jnp.asarray, benchmark.make_inputs(batch, tokens, 8))

    def inspect_call(kernel, *, out_shape, grid_spec, **unused):
        def call(*operands):
            # NumPy exposes out-of-range SMEM reads that JAX would clamp.
            scalars = tuple(map(np.asarray, operands[: grid_spec.num_scalar_prefetch]))
            last = [n - 1 for n in grid_spec.grid]
            specs = [
                *jax.tree.leaves((grid_spec.in_specs, grid_spec.out_specs)),
            ]
            for axis, capacity in enumerate(grid_spec.grid):
                for extra in (0, 1):
                    lookahead = last.copy()
                    lookahead[axis] = capacity + extra
                    for spec in specs:
                        if spec.index_map is None:
                            continue
                        np.testing.assert_array_equal(
                            spec.index_map(*lookahead, *scalars),
                            spec.index_map(*last, *scalars),
                        )
            return jnp.zeros(out_shape.shape, out_shape.dtype)

        return call

    monkeypatch.setattr(pl, "pallas_call", inspect_call)
    with jax.disable_jit():
        csa2_decode(**inputs, num_tokens=tokens)


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
def test_tiled_decode_matches_prefill(benchmark, heads):
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
    actual = csa2_decode(**inputs, num_tokens=128)
    np.testing.assert_allclose(actual, expected, atol=3e-3, rtol=3e-2)
