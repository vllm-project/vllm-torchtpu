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


def _serving_verify_inputs(benchmark, batch, page_size, context, page_stride):
    """DSpark verify inputs shaped like DP8 serving: 6 queries per request.

    SWA metadata comes from prepare_swa_metadata over a compact block table:
    only each request's live window owns pages, IDs are shuffled across
    requests, older columns hold the null block, and replay starts vary.
    """
    from vllm_torchtpu.kernels.deepseek_v41.csa2.cache import prepare_swa_metadata

    host = benchmark.make_inputs(batch, 6, 64, varied_counts=True)
    rng = np.random.default_rng(batch * 1000 + page_size)
    first = context + (np.arange(batch, dtype=np.int32) * 37) % 97
    positions = (first[:, None] + np.arange(6)).reshape(-1).astype(np.int32)
    left = np.maximum(0, first - 127)
    width = (int(first.max()) + 6) // page_size + 1
    live = [
        range(lo // page_size, (hi + 5) // page_size + 1) for lo, hi in zip(left, first)
    ]
    ids = rng.permutation(np.arange(1, 1 + sum(map(len, live)), dtype=np.int32))
    table = np.zeros((batch, width), np.int32)
    for request, columns in enumerate(live):
        table[request, list(columns)], ids = ids[: len(columns)], ids[len(columns) :]
    # Replay requests hide part of the window: none, a partial tail, or all
    # history before the verified tokens.
    minimum = np.where(
        np.arange(batch) % 3 == 0,
        0,
        np.where(np.arange(batch) % 3 == 1, first - 80, first),
    ).astype(np.int32)
    metadata, _, _ = prepare_swa_metadata(
        jnp.asarray(table),
        jnp.arange(batch + 1, dtype=jnp.int32) * 6,
        jnp.asarray(positions),
        jnp.asarray(minimum),
        page_size=page_size,
        page_stride=page_stride,
        row_offset=page_stride - page_size,
    )
    pages = int(table.max() + 1) * (page_stride // page_size)
    swa = (rng.standard_normal((pages, page_size, 512), dtype=np.float32) * 0.2).astype(
        host["q"].dtype
    )
    swa[0] = 0
    # Reused pages keep stale rows outside each query's window: older history,
    # rejected drafts, a previous owner. Poison them; no kernel may read them.
    for request, columns in enumerate(live):
        lo, hi = max(int(left[request]), int(minimum[request])), int(first[request]) + 6
        for column in columns:
            page = table[request, column] * (page_stride // page_size) + (
                page_stride // page_size - 1
            )
            keys = column * page_size + np.arange(page_size)
            swa[page, (keys < lo) | (keys >= hi)] = np.nan
    host.update(positions=positions, swa_kv=swa, swa_metadata=np.asarray(metadata))
    return host


@pytest.mark.parametrize("context", [8192, 98304])
@pytest.mark.parametrize("page_stride_pages", [1, 2])
@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("batch", [4, 48])
def test_serving_verify_ignores_stale_swa_rows(
    benchmark, batch, page_size, page_stride_pages, context
):
    """C384 DSpark verify with non-finite stale rows outside every window.

    Additive -inf score masks alone let a non-finite stale row reach softmax
    and PV; the kernel must zero rows outside [left, right) before QK.
    """
    if jax.default_backend() != "tpu":
        pytest.skip("Async MXU operations require TPU")
    host = _serving_verify_inputs(
        benchmark, batch, page_size, context, page_size * page_stride_pages
    )
    expected = benchmark.reference(host, range(batch * 6))
    assert np.isfinite(expected).all()
    actual = np.asarray(csa2(**jax.device_put(host), bq=6), np.float32)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=3e-3, rtol=3e-2)


@pytest.mark.parametrize("batch", [8, 48])
def test_swa_only_matches_reference(benchmark, batch):
    """main_kv=None is attention over the SWA window alone (compress ratio 0)."""
    host = benchmark.make_inputs(batch, 6, 64, varied_counts=False)
    host["main_lengths"] = np.zeros_like(host["main_lengths"])
    inputs = dict(host, main_kv=None, main_kv_scale=None)
    del inputs["main_lengths"]
    if jax.default_backend() != "tpu":
        arrays = jax.tree.map(jnp.asarray, inputs)
        mesh = jax.sharding.AbstractMesh(
            (1,),
            ("core",),
            abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"),
        )
        with jax.sharding.use_abstract_mesh(mesh):
            traced = jax.make_jaxpr(lambda data: csa2(**data, bq=6))(arrays)
        assert traced.out_avals[0].shape == host["q"].shape
        return
    expected = benchmark.reference(host, range(batch * 6))
    actual = csa2(**jax.device_put(inputs), bq=6)
    np.testing.assert_allclose(actual, expected, atol=3e-3, rtol=3e-2)


@pytest.mark.parametrize("page_size", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("banked", [False, True])
def test_row_packed_swa_matches_native(benchmark, page_size, banked):
    if jax.default_backend() != "tpu":
        pytest.skip("Async MXU operations require TPU")
    host = _serving_verify_inputs(benchmark, 2, page_size, 8192, page_size * 2)
    expected = benchmark.reference(host, range(12))
    pages = host["swa_kv"].shape[0]
    words = host["swa_kv"].view(np.uint16).astype(np.uint32)
    packed = (words[:, 0::2] | (words[:, 1::2] << 16)).view(np.int32)
    packed = packed.reshape(pages, 2 * page_size, 128)
    inputs = jax.device_put(host)
    packed = jax.device_put(packed)
    if banked:
        cut = packed.shape[0] // 2
        inputs["swa_kv"] = (inputs["swa_kv"][:cut], inputs["swa_kv"][cut:])
        packed = (packed[:cut], packed[cut:])
    native = csa2(**inputs, bq=6)
    actual = csa2(**dict(inputs, swa_kv=packed), bq=6)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, native)
    np.testing.assert_allclose(actual, expected, atol=3e-3, rtol=3e-2)


@pytest.mark.parametrize("banked", [False, True])
def test_published_word_pool_attention(benchmark, banked):
    if jax.default_backend() != "tpu":
        pytest.skip("Async MXU and SparseCore operations require TPU")
    from vllm_torchtpu.kernels.mla.paged_swa import swa_word_pages, write_swa_word_pool

    page_size = 32
    host = _serving_verify_inputs(benchmark, 2, page_size, 8192, page_size * 2)
    expected = benchmark.reference(host, range(12))
    values = host.pop("swa_kv").reshape(-1, 512)
    records = np.arange(values.shape[0], dtype=np.int32)
    raw = jnp.zeros((16384, 128), jnp.int32)
    if banked:
        cut = values.shape[0] // (2 * page_size) * (2 * page_size)
        raw = (raw[:cut], raw[cut:])

    def publish_attend(raw, records, values, inputs):
        raw = write_swa_word_pool(raw, records, values, page_size=page_size)
        output = csa2(**inputs, swa_kv=swa_word_pages(raw, page_size), bq=6)
        return raw, output

    inputs = jax.device_put(host)
    records, values = jax.device_put((records, values))
    compiled = (
        jax.jit(publish_attend, donate_argnums=(0,))
        .lower(raw, records, values, inputs)
        .compile()
    )
    memory = compiled.memory_analysis()
    pool_bytes = sum(x.nbytes for x in jax.tree.leaves(raw))
    assert memory.alias_size_in_bytes == pool_bytes
    assert memory.temp_size_in_bytes < pool_bytes
    _, actual = compiled(raw, records, values, inputs)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=3e-3, rtol=3e-2)
