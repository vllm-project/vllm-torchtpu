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

"""Decode dispatch, arithmetic, tail masking and cache ownership."""

import dataclasses
import functools
import hashlib
import math
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import (
    _rpa_attention_in_chunks as rpa_attention,
)
from vllm_torchtpu.kernels.sparse_gqa_decode import (
    is_supported_by_fused_decode,
    qsa_decode_attention,
)


def _case(
    batch,
    context,
    page,
    fp8,
    heads=24,
    kv_heads=2,
    dim=256,
    ratio=4,
    groups=512,
    *,
    poison_packed_padding=False,
):
    rng = np.random.default_rng(6317 + batch + page)
    dtype = jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    packing = 4 if fp8 else 2
    columns = max(1, math.ceil(max(context, 1) / page))
    table = rng.permutation(batch * columns).reshape(batch, columns).astype(np.int32)
    lengths = np.maximum(context - np.arange(batch) % 4, 0).astype(np.int32)
    if batch > 1:
        lengths[batch // 2] = 0
    blocks = np.full((batch, groups), -1, np.int32)
    for row, length in enumerate(lengths):
        n = min(length // ratio, groups)
        blocks[row, :n] = rng.choice(length // ratio, n, replace=False)
    q = jnp.asarray(rng.normal(0, 0.5, (batch, heads, dim)), jnp.bfloat16)
    gate = jnp.asarray(rng.normal(0, 3, q.shape), jnp.bfloat16)
    q = jnp.where(jnp.asarray(lengths)[:, None, None] > 0, q, jnp.nan)
    gate = jnp.where(jnp.asarray(lengths)[:, None, None] > 0, gate, jnp.nan)
    cache = jnp.asarray(
        rng.normal(
            0,
            0.4,
            (batch * columns, page, math.ceil(2 * kv_heads / packing), packing, dim),
        ),
        dtype,
    )
    if poison_packed_padding:
        flat = cache.reshape(*cache.shape[:2], -1, dim)
        cache = flat.at[:, :, 2 * kv_heads :].set(jnp.nan).reshape(cache.shape)
    for row, length in enumerate(lengths):
        # All cache values beyond the actual context are poisoned, including
        # tokens in the final partially selected compression group.
        for logical in range(columns):
            start = max(0, int(length) - logical * page)
            if start < page:
                cache = cache.at[int(table[row, logical]), start:].set(jnp.nan)
    return q, gate, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table), cache


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize(
    "batch,context,page",
    [
        (1, 1, 128),
        (3, 4, 128),
        (3, 8193, 1664),
        (3, 2051, 128),
        (65, 2051, 128),
        (3, 2051, 1664),
        (65, 2051, 1664),
        (3, 0, 1664),
        (257, 2051, 256),
        (513, 7, 4),
        (3, 2051, 132),
        (1, 524289, 256),
    ],
)
def test_qsa_decode_entry_matches_rpa_and_reference_without_cache_writes(
    batch, context, page, fp8
):
    _check_case(batch, context, page, fp8)


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize(
    "batch,context,page,heads,kv_heads,dim,ratio,groups",
    [
        (3, 1027, 132, 12, 2, 128, 4, 128),
        (3, 4099, 136, 16, 4, 256, 8, 256),
        (3, 2051, 128, 8, 1, 128, 16, 64),
        (3, 8195, 128, 24, 2, 512, 4, 1024),
        (129, 7, 128, 12, 2, 128, 8, 1024),
    ],
)
def test_generalized_decode_geometry(
    batch, context, page, heads, kv_heads, dim, ratio, groups, fp8
):
    _check_case(batch, context, page, fp8, heads, kv_heads, dim, ratio, groups)


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
def test_decode_packed_padding_and_odd_budget(fp8):
    _check_case(3, 155, 136, fp8, 8, 1, 128, 8, 17, poison_packed_padding=True)


def _check_case(
    batch,
    context,
    page,
    fp8,
    heads=24,
    kv_heads=2,
    dim=256,
    ratio=4,
    groups=512,
    *,
    poison_packed_padding=False,
):
    if pltpu.get_tpu_info().chip_version != pltpu.ChipVersion.TPU_7X:
        pytest.skip("Measured fused decode schedule requires TPU7x")
    args = _case(
        batch,
        context,
        page,
        fp8,
        heads,
        kv_heads,
        dim,
        ratio,
        groups,
        poison_packed_padding=poison_packed_padding,
    )
    ks, vs = (0.375, 2.5) if fp8 else (1.0, 1.0)
    opts = dict(kv_heads=kv_heads, ratio=ratio, k_scale=ks, v_scale=vs)
    host = tuple(np.asarray(x).copy() for x in args)
    reference_compiled = (
        jax.jit(functools.partial(rpa_attention, **opts)).lower(*args).compile()
    )
    reference, _ = reference_compiled(*args)
    compiled = (
        jax.jit(
            functools.partial(
                qsa_decode_attention,
                **opts,
            ),
            donate_argnums=(5,),
        )
        .lower(*args)
        .compile()
    )
    text = compiled.as_text()
    assert text is not None and "fused_qsa_decode" in text
    out, cache = compiled(*args)
    actual, expected = np.asarray(out, np.float32), np.asarray(reference, np.float32)
    assert np.isfinite(actual).all()
    # Keep the original numerical contract. Bitwise identity is not required
    # between distinct compiled programs at very small output magnitudes.
    print(
        "NUMERIC",
        batch,
        context,
        page,
        fp8,
        "max_abs",
        float(np.max(np.abs(actual - expected))),
        "relative_l2",
        float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12)),
        flush=True,
    )
    np.testing.assert_array_equal(actual[host[3] == 0], 0)
    assert (
        np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12) < 0.01
    )
    np.testing.assert_array_equal(
        np.asarray(cache).view(np.uint8), host[5].view(np.uint8)
    )
    live = np.flatnonzero(host[3] > 0)
    for row in live[
        np.linspace(0, max(0, len(live) - 1), min(3, len(live)), dtype=int)
    ]:
        length = int(host[3][row])
        selected = (
            host[2][row, : min(length // ratio, groups), None] * ratio
            + np.arange(ratio)
        ).reshape(-1)
        selected = np.r_[selected, np.arange(length // ratio * ratio, length)]
        values = (
            host[5][host[4][row, selected // page], selected % page]
            .reshape(len(selected), -1, dim)[:, : 2 * kv_heads]
            .astype(np.float64)
        )
        wanted = np.empty((heads, dim), np.float64)
        per_kv = heads // kv_heads
        for head in range(kv_heads):
            logits = (
                host[0][row, head * per_kv : (head + 1) * per_kv].astype(np.float64)
                @ values[:, head * 2].T
                * (ks / math.sqrt(dim))
            )
            mass = np.exp(logits - logits.max(axis=1, keepdims=True))
            wanted[head * per_kv : (head + 1) * per_kv] = (
                (mass / mass.sum(axis=1, keepdims=True)) @ values[:, head * 2 + 1] * vs
            )
        wanted *= 1 / (1 + np.exp(-host[1][row].astype(np.float64)))
        assert (
            np.linalg.norm(actual[row] - wanted) / max(np.linalg.norm(wanted), 1e-12)
            < 0.01
        )

    repeated, repeated_cache = jax.block_until_ready(compiled(*args[:-1], cache))
    np.testing.assert_array_equal(
        np.asarray(repeated).view(np.uint8), np.asarray(out).view(np.uint8)
    )
    np.testing.assert_array_equal(
        np.asarray(repeated_cache).view(np.uint8), host[5].view(np.uint8)
    )
    for value, before in zip(args[:-1], host[:-1]):
        assert np.asarray(value).tobytes() == before.tobytes()
    memory = compiled.memory_analysis()
    reference_memory = reference_compiled.memory_analysis()
    assert memory is not None and reference_memory is not None
    return dict(
        input_sha256=[hashlib.sha256(value.tobytes()).hexdigest() for value in host],
        output_sha256=hashlib.sha256(np.asarray(out).tobytes()).hexdigest(),
        input_cache_repeat_exact=True,
        relative_l2=float(
            np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12)
        ),
        memory=dict(
            arguments=memory.argument_size_in_bytes,
            outputs=memory.output_size_in_bytes,
            temporary=memory.temp_size_in_bytes,
            alias=memory.alias_size_in_bytes,
        ),
        reference_memory=dict(
            arguments=reference_memory.argument_size_in_bytes,
            outputs=reference_memory.output_size_in_bytes,
            temporary=reference_memory.temp_size_in_bytes,
            alias=reference_memory.alias_size_in_bytes,
        ),
    )


def _specs(fp8=False):
    packing = 4 if fp8 else 2
    return [
        jax.ShapeDtypeStruct((8, 24, 256), jnp.bfloat16),
        jax.ShapeDtypeStruct((8, 24, 256), jnp.bfloat16),
        jax.ShapeDtypeStruct((8, 512), jnp.int32),
        jax.ShapeDtypeStruct((8,), jnp.int32),
        jax.ShapeDtypeStruct((8, 158), jnp.int32),
        jax.ShapeDtypeStruct(
            (158, 1664, 4 // packing, packing, 256),
            jnp.float8_e4m3fn if fp8 else jnp.bfloat16,
        ),
    ]


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "supported",
        "divisible_heads",
        "heads",
        "page",
        "budget",
        "batch",
        "empty",
        "dtype",
        "table",
        "scale",
        "hardware",
    ],
)
def test_decode_eligibility_rejects_invalid_contracts(monkeypatch, fp8, case):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    if case == "hardware":
        info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V5P, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    args = _specs(fp8)
    changes = {
        "heads": (0, (8, 3, 256)),
        "divisible_heads": (0, (8, 12, 256)),
        "batch": (0, (257, 24, 256)),
        "page": (5, (158, 130, *args[5].shape[2:])),
        "budget": (2, (8, 0)),
        "table": (4, (8, 0)),
    }
    if case in changes:
        i, shape = changes[case]
        args[i] = jax.ShapeDtypeStruct(shape, args[i].dtype)
        if case in ("heads", "divisible_heads", "batch"):
            args[1] = jax.ShapeDtypeStruct(shape, args[1].dtype)
    if case == "empty":
        args[:5] = [
            jax.ShapeDtypeStruct((0, *arg.shape[1:]), arg.dtype) for arg in args[:5]
        ]
    if case == "dtype":
        args[0] = jax.ShapeDtypeStruct(args[0].shape, jnp.float32)
    scale = jax.ShapeDtypeStruct((), jnp.float32) if case == "scale" else 1.0
    assert is_supported_by_fused_decode(
        *args, kv_heads=2, ratio=4, k_scale=scale, v_scale=1.0
    ) == (case in ("supported", "divisible_heads"))
    if case not in ("supported", "divisible_heads"):
        with pytest.raises(NotImplementedError, match="Fused decode requires"):
            concrete = [
                jnp.zeros((1, *arg.shape[1:]) if i == 5 else arg.shape, arg.dtype)
                for i, arg in enumerate(args)
            ]
            qsa_decode_attention(
                *concrete, kv_heads=2, ratio=4, k_scale=cast(float, scale)
            )


@pytest.mark.parametrize("supported", [False, True])
def test_qsa_decode_dispatch_preserves_rpa_fallback(monkeypatch, supported):
    from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa

    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    args = _specs()
    if not supported:
        args[0] = jax.ShapeDtypeStruct((8, 25, 256), jnp.bfloat16)
        args[1] = args[0]
    used = []

    def fused(*inputs, **options):
        used.append("fused")
        return inputs[0], inputs[5]

    def baseline(*inputs, **options):
        used.append("rpa")
        return inputs[0], inputs[5]

    from vllm_torchtpu.kernels.sparse_gqa_decode import decode

    monkeypatch.setattr(decode, "execute_decode", fused)
    monkeypatch.setattr(qsa_rpa, "_rpa_attention_in_chunks", baseline)
    out, cache = qsa_rpa.rpa_attention(*args, kv_heads=2, ratio=4)
    assert used == (["fused"] if supported else ["rpa"])
    assert out is args[0]
    assert cache is args[5]


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize(
    "batch,page,columns",
    [(257, 256, 2049), (1025, 4, 131073), (3, 132, 4096)],
)
def test_decode_eligibility_uses_geometry_not_benchmark_limits(
    monkeypatch, fp8, batch, page, columns
):
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    args = _specs(fp8)
    shapes = [
        (batch, 24, 256),
        (batch, 24, 256),
        (batch, 512),
        (batch,),
        (batch, columns),
        (1, page, *args[5].shape[2:]),
    ]
    args = [jax.ShapeDtypeStruct(shape, arg.dtype) for shape, arg in zip(shapes, args)]
    assert is_supported_by_fused_decode(
        *args, kv_heads=2, ratio=4, k_scale=1.0, v_scale=1.0
    )


@pytest.mark.parametrize(
    "case",
    [
        "empty_batch",
        "empty_page",
        "physical_overflow",
        "logical_overflow",
        "table_overflow",
    ],
)
def test_decode_eligibility_rejects_empty_or_overflowing_metadata(monkeypatch, case):
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    args = _specs()
    if case == "empty_batch":
        for i in range(5):
            args[i] = jax.ShapeDtypeStruct((0, *args[i].shape[1:]), args[i].dtype)
    else:
        i, shape = {
            "empty_page": (5, (1, 0, 2, 2, 256)),
            "physical_overflow": (5, (2**24, 128, 2, 2, 256)),
            "logical_overflow": (4, (8, 2**31 // args[5].shape[1] + 2)),
            "table_overflow": (4, (8, 2**28)),
        }[case]
        args[i] = jax.ShapeDtypeStruct(shape, args[i].dtype)
        if case == "table_overflow":
            args[5] = jax.ShapeDtypeStruct((1, 4, 2, 2, 256), args[5].dtype)
    assert not is_supported_by_fused_decode(
        *args, kv_heads=2, ratio=4, k_scale=1.0, v_scale=1.0
    )


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize(
    "heads,kv_heads,dim,ratio,groups",
    [
        (12, 2, 128, 4, 128),
        (16, 4, 256, 8, 256),
        (8, 1, 128, 16, 64),
        (24, 2, 512, 4, 1024),
    ],
)
def test_geometry_replaces_fixed_model_values(
    monkeypatch, fp8, heads, kv_heads, dim, ratio, groups
):
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    args = _specs(fp8)
    packing = 32 // jax.dtypes.itemsize_bits(args[5].dtype)
    shapes = [
        (8, heads, dim),
        (8, heads, dim),
        (8, groups),
        (8,),
        (8, 158),
        (158, 1664, math.ceil(2 * kv_heads / packing), packing, dim),
    ]
    args = [jax.ShapeDtypeStruct(shape, arg.dtype) for shape, arg in zip(shapes, args)]
    assert is_supported_by_fused_decode(
        *args, kv_heads=kv_heads, ratio=ratio, k_scale=1.0, v_scale=1.0
    )


@pytest.mark.parametrize(
    "case",
    [
        "unaligned_dim",
        "nondividing_heads",
        "ratio_tile",
        "zero_ratio",
        "small_ratio_smem",
        "wide_kv_vmem",
        "zero_kv_heads",
        "kv_packing",
        "smem",
        "vmem",
        "selected_overflow",
    ],
)
def test_geometry_and_resource_limits(monkeypatch, case):
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    args = _specs()
    ratio, kv_heads = 4, 2
    if case in ("unaligned_dim", "nondividing_heads", "vmem"):
        shape = {
            "unaligned_dim": (8, 24, 64),
            "nondividing_heads": (8, 25, 256),
            "vmem": (8, 8192, 256),
        }[case]
        args[0] = args[1] = jax.ShapeDtypeStruct(shape, jnp.bfloat16)
        args[5] = jax.ShapeDtypeStruct((*args[5].shape[:-1], shape[-1]), args[5].dtype)
    elif case == "ratio_tile":
        ratio = 3
        args[5] = jax.ShapeDtypeStruct((158, 129, *args[5].shape[2:]), args[5].dtype)
    elif case == "small_ratio_smem":
        ratio = 2
    elif case == "wide_kv_vmem":
        kv_heads = 4
        args[0] = args[1] = jax.ShapeDtypeStruct((8, 32, 512), jnp.bfloat16)
        args[5] = jax.ShapeDtypeStruct((158, 1664, 4, 2, 512), jnp.bfloat16)
    elif case == "zero_ratio":
        ratio = 0
    elif case == "zero_kv_heads":
        kv_heads = 0
    elif case == "kv_packing":
        args[5] = jax.ShapeDtypeStruct((158, 1664, 3, 2, 256), args[5].dtype)
    else:
        groups = 2**30 if case == "selected_overflow" else 2**18
        args[2] = jax.ShapeDtypeStruct((8, groups), jnp.int32)
    assert not is_supported_by_fused_decode(
        *args, kv_heads=kv_heads, ratio=ratio, k_scale=1.0, v_scale=1.0
    )


def test_launch_bound_tracks_selected_tiles_and_hardware(monkeypatch):
    from vllm_torchtpu.kernels.sparse_gqa_decode.planning import query_batch_size

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    normal = query_batch_size(512, 24, 256, 2, 4, 512, jnp.dtype(jnp.bfloat16))
    large = query_batch_size(512, 24, 256, 2, 4, 2048, jnp.dtype(jnp.bfloat16))
    assert normal == 64
    assert 0 < large < normal
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: dataclasses.replace(
            info, smem_capacity_bytes=info.smem_capacity_bytes // 4
        ),
    )
    assert query_batch_size(512, 24, 256, 2, 4, 512, jnp.dtype(jnp.bfloat16)) < normal
