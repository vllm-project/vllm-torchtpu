# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_PREFILL_VPU: the reduced-VPU sparse prefill inner loop.

``mask`` must match the default loop bitwise; ``qk16`` and ``pv16`` match it
bitwise on TPU at a power-of-two score scale (Mosaic runs FP32 dots as one
BF16 pass) and stay inside the dense reference tolerance otherwise. Runs natively on TPU
and
in the Pallas interpreter elsewhere.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tests.models.qwen38_flash.experimental.kernels.test_paired_pages import (
    _dense_selected_reference,
    pack_rows,
)
from vllm_torchtpu.models.qwen38_flash.experimental.kernels.block_sparse_gqa_prefill import (  # noqa: E501
    block_sparse_gqa_attention,
    vpu_features,
)


@pytest.fixture
def backend(monkeypatch):
    if jax.default_backend() == "tpu":
        return
    original = pl.pallas_call
    monkeypatch.setattr(
        pl,
        "pallas_call",
        lambda *args, **kwargs: original(*args, **{**kwargs, "interpret": True}),
    )
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


def test_feature_parsing(monkeypatch):
    for raw, expected in (
        ("", ()),
        ("0", ()),
        ("1", ("mask", "qk16", "pv16")),
        ("mask", ("mask",)),
        ("qk16", ("mask", "qk16")),
        ("pv16", ("mask", "pv16")),
    ):
        monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_VPU", raw)
        assert vpu_features() == expected
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_VPU", "bogus")
    with pytest.raises(ValueError):
        vpu_features()


def _case(seed, query_heads, kv_tile, k_scale):
    rng = np.random.default_rng(seed)
    ratio, dim = 4, 256
    page_size, pages = (512, 4) if kv_tile == 1024 else (128, 12)
    tokens = 1024
    # Two requests: a 600-token prefill on a 700-token context and a short one.
    starts = np.array([0, 600, 700], np.int32)
    contexts = np.array([700, 50], np.int32)
    request = np.searchsorted(starts[1:], np.arange(tokens), side="right")
    lengths = np.zeros(tokens, np.int32)
    live = request < 2
    lengths[live] = (
        contexts[request[live]] + np.arange(tokens)[live] - starts[request[live]] + 1
    )
    owners = np.array([rng.permutation(pages), rng.permutation(pages)])
    table = owners[np.minimum(request, 1)].astype(np.int32)
    query = jnp.asarray(rng.normal(0, 1.0, (tokens, query_heads, dim)), jnp.bfloat16)
    gate = jnp.asarray(rng.uniform(-3, 3, query.shape), jnp.bfloat16)
    key = rng.normal(0, 1.0, (pages * page_size, 2, dim)).astype(np.float32)
    value = rng.normal(0, 0.3, key.shape).astype(np.float32)
    cache = pack_rows(key, value, fp8=True, k_scale=k_scale, v_scale=2.5)
    cache = cache.reshape(pages, page_size, 1, 4, dim)
    stored = np.asarray(cache, np.float32).reshape(pages, page_size, -1, dim)
    key = stored[:, :, 0:1] * k_scale
    value = stored[:, :, 1:2] * 2.5
    blocks = np.full((tokens, 64), -1, np.int32)
    for row, length in enumerate(lengths):
        selected = rng.permutation(length // ratio)[:64]
        blocks[row, : len(selected)] = selected
    expected = _dense_selected_reference(
        query, gate, key, value, blocks, lengths, table, ratio
    )
    args = (
        query,
        gate,
        jnp.asarray(blocks),
        jnp.asarray(lengths),
        jnp.asarray(table),
        cache,
        jnp.asarray(starts),
    )
    return args, expected, lengths, ratio


def _run(monkeypatch, features, args, *, query_heads, kv_tile, k_scale, ratio):
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_VPU", features)
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_LEAN", "1")
    jax.clear_caches()
    query_block, query_tile = (512, 256) if kv_tile == 1024 else (256, None)
    run = jax.jit(
        functools.partial(
            block_sparse_gqa_attention,
            kv_heads=1,
            ratio=ratio,
            query_block=query_block,
            kv_tile=kv_tile,
            k_scale=k_scale,
            v_scale=2.5,
            query_tile=query_tile,
        )
    )
    out, _ = run(*args)
    return np.asarray(jax.block_until_ready(out), np.float32)


def _row_error(actual, expected):
    tokens = len(actual)
    error = np.linalg.norm((actual - expected).reshape(tokens, -1), axis=1)
    norm = np.linalg.norm(expected.reshape(tokens, -1), axis=1)
    return float(np.max(error / np.maximum(norm, 1e-12)))


@pytest.mark.parametrize(
    "query_heads,kv_tile,k_scale",
    [(3, 1024, 1.0), (3, 1024, 0.375), (3, 256, 0.5)],
)
@pytest.mark.parametrize(
    "features,bound",
    [("mask", 1e-6), ("mask,qk16", 1e-5), ("1", 1e-2)],
)
def test_vpu_path(backend, monkeypatch, query_heads, kv_tile, k_scale, features, bound):
    args, expected, lengths, ratio = _case(7, query_heads, kv_tile, k_scale)
    common = dict(
        query_heads=query_heads, kv_tile=kv_tile, k_scale=k_scale, ratio=ratio
    )
    base = _run(monkeypatch, "", args, **common)
    actual = _run(monkeypatch, features, args, **common)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[lengths == 0], 0)
    vs_base = _row_error(actual, base)
    vs_dense = _row_error(actual, expected)
    print(
        f"VPUTEST features={features} kv_tile={kv_tile} k_scale={k_scale} "
        f"max_abs_vs_base={float(np.max(np.abs(actual - base))):.3e} "
        f"row_rel_vs_base={vs_base:.3e} row_rel_vs_dense={vs_dense:.3e} "
        f"base_vs_dense={_row_error(base, expected):.3e}"
    )
    # Exact features: the outputs are BF16, so f32-rounding differences can
    # flip at most a final BF16 bit of a few elements.
    assert vs_base < max(bound, 1e-2 if "16" in features or features == "1" else 4e-3)
    assert vs_dense < 0.01 + bound


@pytest.mark.parametrize("features", ["", "1"])
def test_page_copies_bitwise(backend, monkeypatch, features):
    """VLLM_TPU_QSA_PREFILL_PAGE_COPIES moves the same rows in fewer DMAs."""
    args, _, _, ratio = _case(7, 3, 1024, 1.0)
    common = dict(query_heads=3, kv_tile=1024, k_scale=1.0, ratio=ratio)
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_PAGE_COPIES", "0")
    base = _run(monkeypatch, features, args, **common)
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_PAGE_COPIES", "1")
    actual = _run(monkeypatch, features, args, **common)
    np.testing.assert_array_equal(actual, base)
