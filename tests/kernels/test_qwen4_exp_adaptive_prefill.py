# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen dispatch preserves fallback paths outside adaptive prefill's envelope."""

import functools
import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import (
    block_sparse_gqa_prefill,
    prefill_tiles,
    qsa_rpa,
)
from vllm_torchtpu.kernels.sparse_gqa_decode import decode
from vllm_torchtpu.kernels.sparse_gqa_prefill import adaptive
from vllm_torchtpu.kernels.sparse_gqa_prefill.tuned_block_sizes import (
    SC_GATHER_MAX_CACHE_BYTES,
)


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "supported",
        "decode",
        "slots",
        "tail",
        "padded",
        "scale",
        "page",
        "cache",
        "hardware",
    ],
)
def test_qwen_dispatch_preserves_existing_paths(monkeypatch, fp8, case):
    dtype = jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    packing = 4 if fp8 else 2
    tokens = 1 if case == "decode" else 8448 if case == "tail" else 256
    page = 128 if case == "page" else 1664
    # "cache" is oversized for SparseCore's 16 GiB per-tensor cap and must
    # take the block-sparse TC path instead of failing XLA compilation.
    cache_page = (page, 4 // packing, packing, 256)
    page_bytes = math.prod(cache_page) * jnp.dtype(dtype).itemsize
    pages = SC_GATHER_MAX_CACHE_BYTES // page_bytes + 1 if case == "cache" else 79
    shapes = [
        ((tokens, 24, 256), jnp.bfloat16),
        ((tokens, 24, 256), jnp.bfloat16),
        ((tokens, 512), jnp.int32),
        ((tokens,), jnp.int32),
        ((tokens, 79), jnp.int32),
        ((pages, *cache_page), dtype),
        (((9,) if case == "padded" else (3,) if case == "slots" else (2,)), jnp.int32),
    ]
    args: list[jax.Array | jax.ShapeDtypeStruct] = [
        jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in shapes
    ]
    args[3] = jnp.ones(tokens, jnp.int32)
    args[6] = jnp.array([0] + [tokens] * (shapes[6][0][0] - 1), jnp.int32)
    inputs = tuple(args)
    info = pltpu.get_tpu_info_for_chip(
        pltpu.ChipVersion.TPU_7X,
        1,
    )
    if case == "hardware":
        info = replace(info, sparse_core=None)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setattr(
        prefill_tiles, "get_sparse_prefill_tiles", lambda *a, **k: (256, 256)
    )
    calls = []

    def capture(path, *inputs, **kwargs):
        calls.append((path, inputs, kwargs))
        return inputs[0], inputs[5]

    monkeypatch.setattr(
        adaptive, "adaptive_prefill", functools.partial(capture, "adaptive")
    )
    monkeypatch.setattr(
        block_sparse_gqa_prefill,
        "block_sparse_gqa_attention",
        functools.partial(capture, "prefill"),
    )
    monkeypatch.setattr(
        decode,
        "execute_decode",
        functools.partial(capture, "fused_decode"),
    )
    scale = 2.0 if case == "scale" else 1.0
    # Eligibility belongs to the algorithm leaves. Request partition routing
    # is exercised separately with real row metadata in test_qsa_review_guards.
    output, cache = (
        qsa_rpa.rpa_attention(*args[:-1], kv_heads=2, ratio=4, v_scale=scale)
        if case == "decode"
        else qsa_rpa._prefill_attention(*args, kv_heads=2, ratio=4, v_scale=scale)
    )
    expected = (
        "adaptive"
        if case in ("supported", "slots", "tail", "padded", "page")
        else "fused_decode"
        if case == "decode"
        else "prefill"
    )
    assert len(calls) == 1 and calls[0][0] == expected
    forwarded = inputs[:-1] if case == "decode" else inputs
    for index, (actual, expected_input) in enumerate(
        zip(calls[0][1], forwarded, strict=True)
    ):
        if index in (3, 6):
            np.testing.assert_array_equal(actual, expected_input)
        else:
            assert actual is expected_input
    assert output is inputs[0] and cache is inputs[5]
    if expected != "adaptive":
        assert calls[0][2]["v_scale"] == scale


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("ragged_tail", [False, True])
def test_qwen_adaptive_entry_masks_padding_and_matches_paired_prefill(fp8, ragged_tail):
    if pltpu.get_tpu_info().chip_version != pltpu.ChipVersion.TPU_7X:
        pytest.skip("Adaptive SparseCore schedule is measured on TPU7x")
    tokens, context = (8448, 16384) if ragged_tail else (256, 131072)
    page = 1664
    pages = (context + page - 1) // page
    dtype = jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    packing = 4 if fp8 else 2
    rng = np.random.default_rng(761)
    query = jnp.asarray(rng.normal(0, 0.4, (tokens, 24, 256)), jnp.bfloat16)
    gate = jnp.asarray(rng.normal(0, 2, query.shape), jnp.bfloat16)
    lengths = np.arange(context - tokens + 1, context + 1, dtype=np.int32)
    blocks = np.stack([rng.choice(n // 4, 512, replace=False) for n in lengths]).astype(
        np.int32
    )
    inactive = [31, tokens - 3, tokens - 2, tokens - 1] if ragged_tail else [0, 31, 255]
    lengths[inactive] = 0
    blocks[inactive] = -1
    gate = gate.at[jnp.array(inactive)].set(jnp.nan)
    permutation = rng.permutation(pages).astype(np.int32)
    table = jnp.broadcast_to(jnp.asarray(permutation), (tokens, pages))
    shape = (pages, page, 4 // packing, packing, 256)
    cache = jax.jit(lambda key: jax.random.normal(key, shape).astype(dtype))(
        jax.random.key(1071)
    )
    cache = cache.at[int(permutation[-1]), context % page :].set(jnp.nan)
    before = np.asarray(cache).view(np.uint8).copy()
    args = (
        query,
        gate,
        jnp.asarray(blocks),
        jnp.asarray(lengths),
        table,
        cache,
        jnp.array([0, 1] + [tokens - 3] * 7 if ragged_tail else [0, tokens], jnp.int32),
    )
    input_bytes = [np.asarray(value).tobytes() for value in args]
    distribution = jnp.array([1, 1, 2] if ragged_tail else [0, 0, 1], jnp.int32)
    compiled = (
        jax.jit(functools.partial(qsa_rpa.qsa_attention, kv_heads=2, ratio=4))
        .lower(*args, request_distribution=distribution)
        .compile()
    )
    hlo = compiled.as_text()
    assert isinstance(hlo, str)
    assert "fused_qsa_decode" in hlo
    assert "conditional" in hlo
    if not ragged_tail:
        assert "sc_compact_grouped_prefill" in hlo
    output, returned = compiled(*args, request_distribution=distribution)
    reference, _ = jax.jit(
        functools.partial(
            block_sparse_gqa_prefill.block_sparse_gqa_attention,
            kv_heads=2,
            ratio=4,
            query_block=256,
            kv_tile=256,
        )
    )(*args)
    actual, expected = np.asarray(output, np.float32), np.asarray(reference, np.float32)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[lengths == 0], 0)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.01
    np.testing.assert_array_equal(np.asarray(returned).view(np.uint8), before)
    assert [np.asarray(value).tobytes() for value in args] == input_bytes
    assert (
        np.asarray(compiled(*args, request_distribution=distribution)[0]).tobytes()
        == np.asarray(output).tobytes()
    )
