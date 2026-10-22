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
"""Adaptive QSA entry point, selection boundaries and output/cache ownership."""

import functools
import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
    block_sparse_gqa_attention,
)
from vllm_torchtpu.kernels.sparse_gqa_prefill.adaptive import (
    adaptive_prefill,
    is_supported_by_adaptive_prefill,
)
from vllm_torchtpu.kernels.sparse_gqa_prefill.selector import select_sc


def _abstract_args(dtype=jnp.bfloat16):
    packing = 4 if dtype == jnp.float8_e4m3fn else 2
    specs = [
        ((8192, 24, 256), jnp.bfloat16),
        ((8192, 24, 256), jnp.bfloat16),
        ((8192, 512), jnp.int32),
        ((8192,), jnp.int32),
        ((8192, 158), jnp.int32),
        ((158, 1664, 4 // packing, packing, 256), dtype),
        ((2,), jnp.int32),
    ]
    return [jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in specs]


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "supported",
        "decode",
        "ragged",
        "page",
        "heads",
        "budget",
        "scale",
        "dynamic_scale",
        "packing",
        "cache_size",
        "context",
        "dtype",
        "hardware",
    ],
)
def test_adaptive_envelope_preserves_existing_fallbacks(monkeypatch, fp8, case):
    info = pltpu.get_tpu_info_for_chip(
        pltpu.ChipVersion.TPU_V5P if case == "hardware" else pltpu.ChipVersion.TPU_7X,
        1,
    )
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    args = _abstract_args(jnp.float8_e4m3fn if fp8 else jnp.bfloat16)
    kwargs = dict(kv_heads=2, ratio=4, k_scale=1.0, v_scale=1.0)
    replacement = {
        "decode": (0, (1, 24, 256)),
        "ragged": (6, (3,)),
        "page": (5, (158, 128, *args[5].shape[2:])),
        "heads": (0, (8192, 12, 256)),
        "budget": (2, (8192, 256)),
        "packing": (5, (158, 1664, 1, 2, 256)),
        "cache_size": (5, (20000, *args[5].shape[1:])),
        "context": (4, (8192, 159)),
    }
    if case in replacement:
        index, shape = replacement[case]
        args[index] = jax.ShapeDtypeStruct(shape, args[index].dtype)
    if case == "dtype":
        args[0] = jax.ShapeDtypeStruct(args[0].shape, jnp.float32)
    if case == "scale":
        kwargs["v_scale"] = 2.0
    if case == "dynamic_scale":
        kwargs["k_scale"] = jax.ShapeDtypeStruct((), jnp.float32)
    assert is_supported_by_adaptive_prefill(*args, **kwargs) == (
        case in ("supported", "ragged")
    )


def _make_inputs(tokens, context, fp8, sc_first):
    rng = np.random.default_rng(1071)
    dtype = jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    packing = 4 if fp8 else 2
    query = jnp.asarray(rng.normal(0, 0.4, (tokens, 24, 256)), jnp.bfloat16)
    gate = jnp.asarray(rng.normal(0, 2, query.shape), jnp.bfloat16)
    lengths = np.arange(context - tokens + 1, context + 1, dtype=np.int32)
    blocks = np.full((tokens, 512), -1, np.int32)
    for row, length in enumerate(lengths):
        count = min(length // 4, 512)
        uniform = context > 8192 and (tokens <= 8192 or (row < 8192) == sc_first)
        blocks[row, :count] = (
            rng.choice(length // 4, count, replace=False)
            if uniform
            else np.arange(length // 4 - count, length // 4)
        )
    inactive = [0, 31, 63, min(8191, tokens - 1), tokens - 1]
    if tokens > 8192:
        inactive.append(8192)
    gate = gate.at[jnp.asarray(inactive)].set(jnp.nan)
    lengths[inactive] = 0
    blocks[inactive] = -1
    pages = math.ceil(context / 1664)
    permutation = rng.permutation(pages).astype(np.int32)
    table = jnp.broadcast_to(jnp.asarray(permutation), (tokens, pages))
    shape = (pages, 1664, 4 // packing, packing, 256)
    cache = jax.jit(lambda key: jax.random.normal(key, shape).astype(dtype))(
        jax.random.key(2117)
    )
    cache = cache.at[int(permutation[-1]), context % 1664 :].set(jnp.nan)
    return (
        query,
        gate,
        jnp.asarray(blocks),
        jnp.asarray(lengths),
        table,
        cache,
        jnp.array([0, tokens], jnp.int32),
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize(
    "tokens,context,sc_first",
    [
        (512, 8192, False),
        (512, 262144, True),
        (12288, 262144, True),
        (8448, 262144, True),
        (8192, 8192, False),
        (8192, 262144, True),
        (16384, 131072, False),
        (16384, 131072, True),
        (16640, 131072, True),
    ],
)
def test_adaptive_prefill_preserves_mixed_outputs_with_bounded_memory(
    tokens, context, sc_first, fp8, record_property
):
    if pltpu.get_tpu_info().chip_version != pltpu.ChipVersion.TPU_7X:
        pytest.skip("Adaptive SparseCore schedule is measured on TPU7x")
    args = _make_inputs(tokens, context, fp8, sc_first)
    cache = np.asarray(args[5]).copy()
    compiled = jax.jit(adaptive_prefill).lower(*args).compile()
    hlo = compiled.as_text()
    assert ("sc_compact_grouped_prefill" in hlo) == (context > 8192)
    # Empirical compiler-plan regression budget, not a device capacity or
    # measured runtime peak. Keep the original mixed-output budget; unused
    # physical capacity is checked separately with fixed logical inputs.
    temporary = compiled.memory_analysis().temp_size_in_bytes
    record_property("compiler_temporary_bytes", temporary)
    assert temporary < 400 * 1024**2
    output, returned = compiled(*args)
    baseline, _ = jax.jit(
        functools.partial(
            block_sparse_gqa_attention,
            kv_heads=2,
            ratio=4,
            query_block=256,
            kv_tile=256,
        )
    )(*args)
    actual, expected = np.asarray(output, np.float32), np.asarray(baseline, np.float32)
    assert np.isfinite(actual).all()
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.01
    np.testing.assert_array_equal(actual[np.asarray(args[3]) == 0], 0)
    np.testing.assert_array_equal(
        np.asarray(returned).view(np.uint8), cache.view(np.uint8)
    )
    if tokens > 8192:
        choices = [
            bool(
                jax.jit(
                    functools.partial(
                        select_sc,
                        num_tiles=math.ceil(args[4].shape[1] * 1664 / 256),
                        dtype=args[5].dtype,
                    )
                )(args[2][i : i + 8192], args[3][i : i + 8192])
            )
            for i in range(0, tokens, 8192)
        ]
        np.testing.assert_array_equal(
            choices, [(i < 8192) == sc_first for i in range(0, tokens, 8192)]
        )
    # FP64 independent reference includes partial groups and physical page tails.
    for row in (1, 32, min(8190, tokens - 2), tokens - 2):
        length = int(args[3][row])
        count = min(length // 4, 512)
        selected = (np.asarray(args[2][row, :count, None]) * 4 + np.arange(4)).reshape(
            -1
        )
        selected = np.concatenate((selected, np.arange(length // 4 * 4, length)))
        values = cache[np.asarray(args[4][row, selected // 1664]), selected % 1664]
        values = values.reshape(-1, 4, 256).astype(np.float64)
        wanted = np.empty((24, 256), np.float64)
        for head in range(24):
            logits = (
                values[:, 2 * (head // 12)]
                @ np.asarray(args[0][row, head], np.float64)
                / 16
            )
            mass = np.exp(logits - logits.max())
            wanted[head] = (mass / mass.sum()) @ values[:, 2 * (head // 12) + 1]
        gate = np.asarray(args[1][row], np.float64)
        wanted *= 1 / (1 + np.exp(-gate))
        assert np.linalg.norm(actual[row] - wanted) / np.linalg.norm(wanted) < 0.015


@pytest.mark.parametrize(
    "tokens,supported",
    [
        (256, True),
        (512, True),
        (12288, True),
        (262144, True),
        (128, True),
        (257, True),
        (262400, False),
    ],
)
def test_query_bucket_envelope(monkeypatch, tokens, supported):
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    args = _abstract_args()
    for i in range(5):
        args[i] = jax.ShapeDtypeStruct((tokens, *args[i].shape[1:]), args[i].dtype)
    assert (
        is_supported_by_adaptive_prefill(
            *args, kv_heads=2, ratio=4, k_scale=1.0, v_scale=1.0
        )
        == supported
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
def test_full_256k_compiler_workspace_growth_is_bounded_by_metadata(
    fp8, record_property
):
    if pltpu.get_tpu_info().chip_version != pltpu.ChipVersion.TPU_7X:
        pytest.skip("Adaptive SparseCore schedule is measured on TPU7x")
    temporary, metadata = [], []
    for tokens in (16384, 262144):
        args = _abstract_args(jnp.float8_e4m3fn if fp8 else jnp.bfloat16)
        for i in range(5):
            args[i] = jax.ShapeDtypeStruct((tokens, *args[i].shape[1:]), args[i].dtype)
        compiled = jax.jit(adaptive_prefill, donate_argnums=(5,)).lower(*args).compile()
        temporary.append(compiled.memory_analysis().temp_size_in_bytes)
        metadata.append(sum(x.size * np.dtype(x.dtype).itemsize for x in args[2:5]))
        record_property(f"compiler_temporary_bytes_{tokens}", temporary[-1])
    # Query/selected-KV workspaces stay chunk-local. Increasing the token
    # bucket may add source-shaped selection metadata, but must not add a
    # whole query/gate matrix or context-sized selected-KV workspace. The
    # cache geometry and donation match in both compilations. No inference
    # or measured runtime peak is implied by this compiler-plan regression.
    assert temporary[1] - temporary[0] <= metadata[1] - metadata[0]


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
def test_output_only_attention_does_not_copy_unused_physical_cache(
    fp8, record_property
):
    args = _abstract_args(jnp.float8_e4m3fn if fp8 else jnp.bfloat16)
    for i in range(5):
        shape = (8448, 79) if i == 4 else (8448, *args[i].shape[1:])
        args[i] = jax.ShapeDtypeStruct(shape, args[i].dtype)
    for capacity in (79, 463):
        args[5] = jax.ShapeDtypeStruct((capacity, *args[5].shape[1:]), args[5].dtype)
        compiled = jax.jit(lambda *xs: adaptive_prefill(*xs)[0]).lower(*args).compile()
        record_property(
            f"compiler_temporary_bytes_{capacity}_pages",
            compiled.memory_analysis().temp_size_in_bytes,
        )
        # Allocation/layout choices can change a little with physical capacity.
        # Test the unwanted whole-cache copy itself, not exact scratch bytes.
        # Synchronized capacity sweeps and device traces validate execution.
        dtype = "f8e4m3fn" if fp8 else "bf16"
        for line in compiled.as_text().splitlines():
            if not re.search(r"\bcopy(?:-start|-done)?\(", line):
                continue
            result = line.split("=", 1)[1].split(" copy", 1)[0]
            for shape in re.findall(rf"{dtype}\[([0-9,]+)\]", result):
                assert math.prod(map(int, shape.split(","))) < args[5].size
