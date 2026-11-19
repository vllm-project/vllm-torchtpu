# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata-only checks for sorted FP8 preparation eligibility."""

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8 import prepare


def descriptors(tokens=288, width=128, key_dtype=jnp.bfloat16):
    banks = (jax.ShapeDtypeStruct((512, 128), jnp.int32),) * 2
    records = jax.ShapeDtypeStruct((tokens,), jnp.int32)
    keys = jax.ShapeDtypeStruct((tokens, width), key_dtype)
    return banks, records, keys


@pytest.mark.parametrize("page", [128, 256])
@pytest.mark.parametrize("tokens", [128, 256, 288, 384, 512])
def test_verified_shapes_supported(monkeypatch, tokens, page):
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    assert prepare.supports_sorted_fp8(*descriptors(tokens), page)


@pytest.mark.parametrize("tokens", [0, 1, 127, 129, 255, 287, 289, 513, 1024])
def test_unverified_shapes_fallback(monkeypatch, tokens):
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    assert not prepare.supports_sorted_fp8(*descriptors(tokens), 256)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_non_tpu_fallback(monkeypatch, backend):
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    assert not prepare.supports_sorted_fp8(*descriptors(), 256)


@pytest.mark.parametrize(
    "change", ["width", "dtype", "records", "page", "banks", "bank_dtype"]
)
def test_layout_contract_fallback(monkeypatch, change):
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    banks, records, keys = descriptors()
    page = 256
    if change == "width":
        keys = jax.ShapeDtypeStruct((288, 64), jnp.bfloat16)
    elif change == "dtype":
        keys = jax.ShapeDtypeStruct((288, 128), jnp.float32)
    elif change == "records":
        records = jax.ShapeDtypeStruct((287,), jnp.int32)
    elif change == "page":
        page = 64
    elif change == "banks":
        banks = banks[:1]
    elif change == "bank_dtype":
        banks = (jax.ShapeDtypeStruct((512, 128), jnp.float32),) * 2
    assert not prepare.supports_sorted_fp8(banks, records, keys, page)


@pytest.mark.parametrize("page_size", [128, 256])
@pytest.mark.parametrize(("tokens", "invalid"), [(128, False), (288, True)])
def test_sorted_write_matches_native(monkeypatch, page_size, tokens, invalid):
    """The fused writer (interpreted) is bitwise the native writer."""
    import functools

    import numpy as np
    from jax.experimental import pallas as pl

    from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8 import cache

    monkeypatch.setattr(
        prepare.pl, "pallas_call", functools.partial(pl.pallas_call, interpret=True)
    )
    write = cache._write_sorted_packed_pages
    monkeypatch.setattr(
        cache,
        "_write_sorted_packed_pages",
        lambda *a, **k: write(*a, **{**k, "interpret": True}),
    )
    monkeypatch.setattr(cache, "supports_sorted_fp8", lambda *a: True)
    rng = np.random.default_rng(page_size + tokens)
    rows = (33 * page_size + 127) // 128
    pages, stride, offset = 8, rows + 1, 3
    split = offset + 4 * stride
    banks = tuple(
        jnp.asarray(rng.integers(-(2**31), 2**31, (n, 128)).astype(np.int32))
        for n in (split, offset + pages * stride + 5 - split)
    )
    layout = cache.PackedFP8Layout(pages, stride, offset, page_size)
    records = rng.choice(pages * page_size, tokens, replace=False)
    if invalid:
        records[: tokens // 4] = -1
    records = jnp.asarray(records, jnp.int32)
    keys = jnp.asarray(rng.standard_normal((tokens, 128)), jnp.bfloat16)
    native = cache._write_fp8_records_native(banks, records, keys, layout)
    fused = cache.write_fp8_records(banks, records, keys, layout)
    for got, want in zip(fused, native):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
