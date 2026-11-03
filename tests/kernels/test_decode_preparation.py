# SPDX-License-Identifier: Apache-2.0
"""Production-entry correctness gates for retained DSv4.1 decode regions."""

import os

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from scripts.dsv41_bench import peripheral_references as peripheral_ref
from scripts.dsv41_bench import region_references as region_ref
from vllm_torchtpu.kernels.experimental import decode_preparation as prep


@pytest.fixture(scope="module", autouse=True)
def _require_tpu():
    # Initialize only during execution; collection must not claim the TPU.
    if (
        "FALCON_JAX_PROCESS_COUNT" in os.environ
        and not jax.distributed.is_initialized()
    ):
        jax.distributed.initialize(
            coordinator_address=os.environ["FALCON_JAX_COORDINATOR_ADDRESS"],
            num_processes=int(os.environ["FALCON_JAX_PROCESS_COUNT"]),
            process_id=int(os.environ["FALCON_JAX_PROCESS_ID"]),
        )
    if jax.default_backend() != "tpu":
        pytest.skip("TPU kernel")


def _check(actual, expected):
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(np.asarray(got), want)


@pytest.mark.parametrize("width", [1280, 2304, 5120, 6144])
def test_quantize_entry(width):
    from vllm_torchtpu.kernels.quantized_matmul.util import (
        quantize_block,
        quantize_tensor,
    )

    dtype = np.float32 if width == 6144 else ml_dtypes.bfloat16
    x = np.random.default_rng(17).normal(size=(288, width)).astype(dtype)
    x[0] = 0
    x[1] = np.float32(1e-12)
    actual = jax.jit(lambda v: quantize_tensor(v, jnp.float8_e4m3fn))(jnp.asarray(x))
    expected = jax.jit(
        lambda v: quantize_block(v, axis=-1, target_dtype=jnp.float8_e4m3fn)
    )(jnp.asarray(x))
    _check(actual, expected)
    peripheral_ref.check_quantization((x,), actual)


@pytest.mark.parametrize("tb", [32, 144])
def test_compact_entry(tb):
    from vllm_torchtpu.kernels.attention_preprocessing.compress_rope import _compact

    records = np.random.default_rng(19).integers(-1000, 35000, 288, dtype=np.int32)
    records[:6] = [-1, 0, 32767, 32768, 7, 7]
    valid = (records >= 0) & (records < 32768)
    pairs = np.zeros((288, 2), np.int32)
    indices = np.flatnonzero(valid)
    pairs[: len(indices)] = np.stack((indices, records[indices]), axis=1)
    offsets = np.array([valid[:i].sum() for i in range(0, 289, tb)], np.int32)
    actual = jax.jit(lambda r: _compact(r, 32768, 288 // tb, tb))(records)
    _check(actual, (pairs.reshape(-1), offsets))


@pytest.mark.parametrize("columns", [386, 772])
def test_main_records_entry(columns):
    from vllm_torchtpu.kernels.deepseek_v41.csa2.cache import prepare_main_records

    rng = np.random.default_rng(23)
    selected = rng.integers(0, columns * 64, (288, 512), dtype=np.int32)
    selected[:, 400:] = -1
    pages = rng.integers(1, 700000, (48, columns), dtype=np.int32)
    starts = np.arange(49, dtype=np.int32) * 6
    starts[-1] = 285
    actual = jax.jit(lambda *xs: prepare_main_records(*xs, page_size=64))(
        selected, pages, starts
    )
    _check(
        actual,
        peripheral_ref.reference(
            "main_records", (selected, pages, starts), page_size=64
        ),
    )


@pytest.mark.parametrize("tokens", [12288, 36864])
def test_cache_unpack_shared_and_direct(tokens):
    rng = np.random.default_rng(29)
    values = rng.normal(size=(2, 2, tokens, 128)).astype(ml_dtypes.bfloat16)
    bits = values.view(np.uint16).astype(np.uint32)
    words = (bits[:, 0] | (bits[:, 1] << 16)).view(np.int32)
    actual = jax.jit(prep.unpack_cache_words)(words)
    _check(actual, values.reshape(4, tokens, 128))


@pytest.mark.parametrize("rank", [0, 7, 15])
def test_dispatch_tables(rank):
    rng = np.random.default_rng(31)
    counts = rng.integers(0, 10, (16, 384), dtype=np.int32)
    counts[:, ::13] = 0
    starts = np.cumsum(counts[rank], dtype=np.int32) - counts[rank]
    xs = counts, starts, np.array([rank], np.int32)
    actual = jax.jit(lambda *a: prep.dispatch_layout_call(a)(*a))(*xs)
    _check(actual, region_ref.dispatch(*xs))


@pytest.mark.parametrize("bank_count", [1, 2])
def test_swa_prepare_and_cached_metadata(bank_count):
    from vllm_torchtpu.kernels.mla.paged_swa import write_swa_pages

    rng = np.random.default_rng(37)
    bank_pages = 512 // bank_count
    bank = np.full((bank_pages, 64, 512), -0.25, dtype=ml_dtypes.bfloat16)
    table = (
        np.broadcast_to((np.arange(48) * 9 + 1)[:, None], (48, 1539))
        .astype(np.int32)
        .copy()
    )
    table[1] = 0
    table[2] = 900
    table[3] = 257
    starts = np.arange(49, dtype=np.int32) * 6
    positions = np.tile(98304 + np.arange(6, dtype=np.int32), 48)
    minimum = np.zeros(48, np.int32)
    values = rng.normal(size=(288, 512)).astype(ml_dtypes.bfloat16)
    expected = region_ref.swa(bank, bank, table, starts, positions, minimum, values)
    expected_owners = expected[:bank_count]
    expected_metadata = expected[2:]
    banks = tuple(jnp.asarray(bank.copy()) for _ in range(bank_count))
    actual = jax.jit(prep.swa_prepare_publish)(
        banks, table, starts, positions, minimum, values
    )
    _check(actual, (*expected_owners, *expected_metadata))
    banks = tuple(jnp.asarray(bank.copy()) for _ in range(bank_count))
    pages = banks if bank_count == 2 else banks[0]
    packed_only = write_swa_pages(pages, jnp.asarray(expected[-1]), jnp.asarray(values))
    _check(packed_only, expected_owners if bank_count == 2 else expected_owners[0])


def test_mhc_pre_entry():
    from vllm_torchtpu.models.vllm.deepseek_v41.common.mhc import delayed_pre_flat_jax

    rng = np.random.default_rng(41)
    xs = (
        rng.normal(size=(288, 20480)).astype(ml_dtypes.bfloat16),
        (rng.normal(size=(20480, 72)) * 0.002).astype(ml_dtypes.bfloat16),
        np.ones(3, np.float32),
        np.zeros(24, np.float32),
        rng.uniform(0, 1, (288, 4)).astype(np.float32),
    )
    actual = jax.jit(lambda *a: delayed_pre_flat_jax(*a, 1e-6, 1e-6, 1e-6, 2.0, 20))(
        *xs
    )
    expected = peripheral_ref.reference("mhc_pre", xs)
    for i, (got, want) in enumerate(zip(actual, expected)):
        if i == 2:
            np.testing.assert_array_equal(np.asarray(got), want)
        else:
            np.testing.assert_allclose(np.asarray(got), want, rtol=2e-5, atol=2e-6)


def test_csa2_bridge_registration(monkeypatch):
    from types import SimpleNamespace

    from vllm_torchtpu.models.vllm import vllm_model_wrapper_context as context
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import csa2_adapter

    mesh = jax.sharding.Mesh(np.array(jax.local_devices()[:1]), ("data",))
    monkeypatch.setattr(
        context,
        "get_vllm_model_wrapper_context",
        lambda: SimpleNamespace(mesh=mesh),
    )
    monkeypatch.setattr(context, "get_pallas_op_mesh", lambda: mesh)
    csa2_adapter.initialize_csa2_ops()
    assert "prepare_publish_swa" in csa2_adapter._ops
    assert "prepare_publish_swa_banked" in csa2_adapter._ops
