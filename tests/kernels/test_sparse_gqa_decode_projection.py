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

"""Generic projection planning, real arithmetic, wiring and native ownership."""

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from test_sparse_gqa_decode import _case

from vllm_torchtpu.kernels.sparse_gqa_decode import projection
from vllm_torchtpu.kernels.sparse_gqa_decode.eligibility import plan_decode
from vllm_torchtpu.kernels.sparse_gqa_decode.projection_kernel import project_tile
from vllm_torchtpu.kernels.sparse_gqa_decode.projection_planning import (
    plan_projected_decode,
    projection_fits,
    projection_resources,
    projection_width,
)

# B,H,D,KV,ratio,groups,page,N,HC branches; cover both N and batch tails.
GEOMETRIES = [
    (1, 4, 128, 1, 4, 7, 128, 193, 1),
    (3, 6, 128, 2, 4, 17, 132, 640, 3),
    (7, 12, 256, 3, 8, 33, 136, 1025, 2),
    (16, 24, 256, 2, 4, 512, 1664, 2560, 4),
]


def replace_spec(value, *, shape=None, dtype=None):
    return jax.ShapeDtypeStruct(
        value.shape if shape is None else shape, value.dtype if dtype is None else dtype
    )


def specs(geometry, fp8):
    batch, heads, dim, kv, ratio, groups, page, width, branches = geometry
    packing = 4 if fp8 else 2
    q = jax.ShapeDtypeStruct((batch, heads, dim), jnp.bfloat16)
    args = (
        q,
        q,
        jax.ShapeDtypeStruct((batch, groups), jnp.int32),
        jax.ShapeDtypeStruct((batch,), jnp.int32),
        jax.ShapeDtypeStruct((batch, 8), jnp.int32),
        jax.ShapeDtypeStruct(
            (16, page, math.ceil(2 * kv / packing), packing, dim),
            jnp.float8_e4m3fn if fp8 else jnp.bfloat16,
        ),
    )
    weight = jax.ShapeDtypeStruct(
        (math.ceil(width / 512), heads * dim, 512), jnp.bfloat16
    )
    residual = jax.ShapeDtypeStruct((batch, branches * width), jnp.bfloat16)
    gate = jax.ShapeDtypeStruct((batch, branches), jnp.float32)
    return (
        args,
        weight,
        residual,
        gate,
        dict(kv_heads=kv, ratio=ratio, k_scale=0.375, v_scale=2.5),
    )


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("chip", [pltpu.ChipVersion.TPU_7X, pltpu.ChipVersion.TPU_V5P])
def test_generic_projection_admission(geometry, fp8, chip, monkeypatch):
    info = dataclasses.replace(
        pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1), chip_version=chip
    )
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    args, weight, residual, hc, options = specs(geometry, fp8)
    assert projection_width(args[0], weight, residual, hc) == geometry[-2]
    launch = plan_decode(*args, **options)
    assert launch is not None
    projected = plan_projected_decode(*args, weight, hc, **options)
    assert projected is not None
    assert projection_fits(projected, args[0], weight, hc, info)


def test_projection_resource_boundary_counts_resident_weight_and_padding(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    args, weight, residual, hc, options = specs(GEOMETRIES[-1], True)
    launch = plan_decode(*args, **options)
    assert launch is not None
    usage = projection_resources(args[0], weight, hc, info)
    assert usage.weight_bytes == 30 * 2**20
    assert usage.attended_bytes >= args[0].size * 4
    assert usage.residual_bytes >= residual.size * 2
    assert usage.output_bytes == usage.residual_bytes
    required = launch.resources.vmem_estimate_bytes + usage.vmem_bytes
    boundary = dataclasses.replace(info, vmem_capacity_bytes=math.ceil(required / 0.8))
    assert projection_fits(launch, args[0], weight, hc, boundary)
    assert not projection_fits(
        launch,
        args[0],
        weight,
        hc,
        dataclasses.replace(
            boundary, vmem_capacity_bytes=boundary.vmem_capacity_bytes - 1
        ),
    )
    bigger = replace_spec(weight, shape=(64, weight.shape[1], weight.shape[2]))
    assert not projection_fits(launch, args[0], bigger, hc, info)


@pytest.mark.parametrize(
    "field",
    [
        "K",
        "tiles",
        "weight_dtype",
        "residual_dtype",
        "gate_dtype",
        "branches",
        "residual_width",
    ],
)
def test_invalid_projection_abi_is_rejected(field):
    args, weight, residual, gate, _ = specs(GEOMETRIES[1], True)
    if field == "K":
        weight = replace_spec(weight, shape=(2, 769, 512))
    elif field == "tiles":
        weight = replace_spec(weight, shape=(3, 768, 512))
    elif field == "weight_dtype":
        weight = replace_spec(weight, dtype=jnp.float32)
    elif field == "residual_dtype":
        residual = replace_spec(residual, dtype=jnp.float32)
    elif field == "gate_dtype":
        gate = replace_spec(gate, dtype=jnp.bfloat16)
    elif field == "branches":
        gate = replace_spec(gate, shape=(3, 0))
    else:
        residual = replace_spec(residual, shape=(3, 1921))
    with pytest.raises(ValueError, match="Projection"):
        projection_width(args[0], weight, residual, gate)


@pytest.mark.parametrize(
    "batch,k,n", [(1, 128, 193), (3, 768, 640), (7, 3072, 1025), (16, 6144, 512)]
)
def test_real_projection_reduces_complete_k_at_existing_tolerance(batch, k, n):
    rng = np.random.default_rng(918)
    attended = jnp.asarray(rng.normal(0, 0.4, (batch, k)), jnp.bfloat16)
    weight = jnp.asarray(rng.normal(0, 0.025, (k, n)), jnp.bfloat16)
    # A final nonzero row catches floor-divided/dropped K tails.
    attended = attended.at[:, -1].set(jnp.bfloat16(3))
    weight = weight.at[-1].set(jnp.bfloat16(0.25))
    expected = np.asarray(attended, np.float64) @ np.asarray(weight, np.float64)
    actual = np.asarray(jax.jit(project_tile)(attended, weight), np.float64)
    assert (
        np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12) < 0.01
    )


@pytest.mark.parametrize("geometry", GEOMETRIES[:3])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("fuse", [False, True])
def test_projected_dispatch_padding_and_separate_path(geometry, fp8, fuse, monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    batch, heads, dim, kv, ratio, groups, page, width, branches = geometry
    args = _case(batch, 67, page, fp8, heads, kv, dim, ratio, groups)
    rng = np.random.default_rng(712)
    matrix = jnp.asarray(rng.normal(0, 0.025, (heads * dim, width)), jnp.bfloat16)
    packed = (
        jnp.pad(matrix, ((0, 0), (0, -width % 512)))
        .reshape(heads * dim, -1, 512)
        .transpose(1, 0, 2)
    )
    residual = jnp.asarray(rng.normal(0, 0.2, (batch, branches * width)), jnp.bfloat16)
    hc = jnp.asarray(rng.normal(1, 0.1, (batch, branches)), jnp.float32)
    attended = jnp.where((args[3] > 0)[:, None, None], args[0], 0)
    calls = []

    def attention(*a, **kw):
        calls.append("separate")
        return attended, a[5]

    def kernel(**kw):
        calls.append("fused")
        assert kw["cache"].dtype == args[5].dtype
        assert kw["output_weight"].shape == packed.shape
        projected = jnp.concatenate(
            [
                project_tile(attended.reshape(batch, -1), packed[tile])
                for tile in range(len(packed))
            ],
            axis=-1,
        )
        padded = kw["residual"].reshape(batch, branches, -1)
        return kw["query"], (
            padded.astype(jnp.float32) + hc[:, :, None] * projected[:, None, :]
        ).astype(jnp.bfloat16).reshape(batch, -1)

    monkeypatch.setattr(projection, "qsa_attention", attention)
    monkeypatch.setattr(projection, "decode_projected_kernel", kernel)
    monkeypatch.setattr(projection.schedule, "generate", lambda *a: None)
    if not fuse:
        # Attention still fits; only resident projection storage is unavailable.
        monkeypatch.setattr(projection, "plan_projected_decode", lambda *a, **kw: None)
    before = np.asarray(args[5]).view(np.uint8).copy()
    output, returned = projection.projected_attention(
        *args,
        jnp.arange(batch + 1, dtype=jnp.int32),
        packed,
        residual,
        hc,
        request_distribution=jnp.full((3,), batch, jnp.int32),
        kv_heads=kv,
        ratio=ratio,
        k_scale=0.375,
        v_scale=2.5,
    )
    projected = np.asarray(attended.reshape(batch, -1), np.float64) @ np.asarray(
        matrix, np.float64
    )
    projected = np.asarray(jnp.asarray(projected, jnp.bfloat16), np.float64)
    expected = (
        np.asarray(residual, np.float64).reshape(batch, branches, width)
        + np.asarray(hc, np.float64)[:, :, None] * projected[:, None, :]
    )
    actual = np.asarray(output, np.float64).reshape(expected.shape)
    assert (
        np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12) < 0.01
    )
    assert ("fused" in calls) == fuse
    assert "separate" in calls
    assert returned is args[5]
    np.testing.assert_array_equal(np.asarray(returned).view(np.uint8), before)
    idle = np.asarray(args[3]) == 0
    np.testing.assert_array_equal(np.asarray(output)[idle], np.asarray(residual)[idle])


@pytest.mark.parametrize("batch,capacity", [(8, 16), (3, 8), (16, 16), (9, 4)])
@pytest.mark.parametrize("layout", ["decode", "verify", "mixed", "idle"])
def test_prepared_projected_public_entry_sizes_cached_offsets(
    batch, capacity, layout, monkeypatch
):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    args = list(_case(batch, 67, 128, True, 4, 1, 128, 4, 16))
    args[0] = args[0].astype(jnp.float32)
    requests = max(batch, capacity)
    if layout == "decode":
        counts, distribution = [1] * (batch - 1), [batch - 1] * 3
    elif layout == "verify":
        counts, distribution = [1, 2], [1, 2, 2]
    elif layout == "mixed":
        counts, distribution = [1, 2], [1, 1, 2]
    else:
        counts, distribution = [], [0, 0, 0]
    counts += [0] * (requests - len(counts))
    starts = jnp.asarray(np.r_[0, np.cumsum(counts)], jnp.int32)
    args[3] = jnp.where(jnp.arange(batch) < sum(counts), args[3], 0)
    attended = jnp.where((args[3] > 0)[:, None, None], args[0], 0).astype(jnp.bfloat16)
    matrix = jnp.ones((512, 193), jnp.bfloat16) * jnp.bfloat16(0.01)
    weight = jnp.pad(matrix, ((0, 0), (0, 319)))[None]
    residual = jnp.ones((batch, 3 * 193), jnp.bfloat16)
    hc_gate = jnp.full((batch, 3), 0.75, jnp.float32)
    seen = []

    def attention(*arrays, **kwargs):
        return attended, arrays[5]

    def kernel(**kwargs):
        offsets = kwargs["cu_q_lens"]
        assert offsets.shape == (batch + 1,)
        jax.debug.callback(
            lambda value: seen.append(np.asarray(value).copy()), offsets, ordered=True
        )
        projected = project_tile(attended.reshape(batch, -1), weight[0])
        value = kwargs["residual"].reshape(batch, 3, 512).astype(jnp.float32)
        return kwargs["query"], (
            value + hc_gate[:, :, None] * projected[:, None, :]
        ).astype(jnp.bfloat16).reshape(batch, -1)

    monkeypatch.setattr(projection, "qsa_attention", attention)
    monkeypatch.setattr(projection, "decode_projected_kernel", kernel)
    monkeypatch.setattr(projection.schedule, "generate", lambda *args: None)
    cached = jnp.arange(capacity + 1, dtype=jnp.int32)
    before = np.asarray(cached).copy()
    output, returned = projection.projected_attention(
        *args,
        starts,
        weight,
        residual,
        hc_gate,
        request_distribution=jnp.asarray(distribution, jnp.int32),
        kv_heads=1,
        ratio=4,
        k_scale=0.375,
        v_scale=2.5,
        query_prepared=True,
        prepared_cu=cached,
    )
    output.block_until_ready()
    projected = np.asarray(attended.reshape(batch, -1), np.float64) @ np.asarray(
        matrix, np.float64
    )
    projected = np.asarray(jnp.asarray(projected, jnp.bfloat16), np.float64)
    expected = np.asarray(residual, np.float64).reshape(batch, 3, 193) + (
        np.asarray(hc_gate)[:, :, None] * projected[:, None, :]
    )
    actual = np.asarray(output, np.float64).reshape(expected.shape)
    assert (
        np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12) < 0.01
    )
    assert len(seen) == (layout != "mixed")
    for offsets in seen:
        np.testing.assert_array_equal(offsets, np.arange(batch + 1))
    np.testing.assert_array_equal(cached, before)
    assert returned is args[5]
    idle = np.asarray(args[3]) == 0
    np.testing.assert_array_equal(np.asarray(output)[idle], np.asarray(residual)[idle])


@pytest.mark.parametrize("batch,capacity,chunk", [(8, 16, 8), (9, 4, 9), (17, 32, 8)])
def test_pure_decode_uses_launch_local_cached_offsets(
    batch, capacity, chunk, monkeypatch
):
    from vllm_torchtpu.kernels.sparse_gqa_decode import decode

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    args = list(_case(batch, 67, 128, True, 4, 1, 128, 4, 16))
    args[0] = args[0].astype(jnp.float32)
    launch = plan_decode(
        args[0].astype(jnp.bfloat16),
        *args[1:],
        kv_heads=1,
        ratio=4,
        k_scale=0.375,
        v_scale=2.5,
    )
    assert launch is not None
    launch = dataclasses.replace(launch, config=launch.config_for_batch(chunk))
    monkeypatch.setattr(decode.schedule, "generate", lambda *args: None)

    def kernel(**kwargs):
        offsets = kwargs["cu_q_lens"]
        assert offsets.shape == (chunk + 1,)
        return kwargs["query"] + offsets[:-1][None, :, None, None, None]

    monkeypatch.setattr(decode, "decode_kernel", kernel)
    before = np.asarray(args[5]).view(np.uint8).copy()
    output, owner = decode.execute_decode(
        *args,
        launch=launch,
        k_scale=0.375,
        v_scale=2.5,
        prepared_cu=jnp.arange(capacity + 1, dtype=jnp.int32),
    )
    expected = jnp.where(
        (args[3] > 0)[:, None, None],
        args[0] + (jnp.arange(batch) % chunk)[:, None, None],
        0,
    ).astype(jnp.bfloat16)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(np.asarray(owner).view(np.uint8), before)
    np.testing.assert_array_equal(np.asarray(args[5]).view(np.uint8), before)


@pytest.mark.parametrize("shape,dtype", [((9,), jnp.float32), ((1, 9), jnp.int32)])
def test_cached_query_offsets_reject_invalid_integer_abi(shape, dtype):
    from vllm_torchtpu.kernels.sparse_gqa_decode.query_layout import query_offsets

    with pytest.raises(ValueError, match="int32 vector"):
        query_offsets(8, jnp.zeros(shape, dtype))


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "batch,capacity,requests,heads,kv,dim,width,branches,fp8",
    [
        (8, 16, 16, 24, 2, 256, 2560, 4, True),
        (16, 16, 16, 24, 2, 256, 2560, 4, True),
        (3, 8, 8, 6, 2, 128, 640, 3, False),
        (12, 4, 16, 4, 1, 128, 193, 2, True),
    ],
)
def test_native_torch_public_prepared_projected_offsets(
    batch,
    capacity,
    requests,
    heads,
    kv,
    dim,
    width,
    branches,
    fp8,
    monkeypatch,
    record_property,
):
    import torch

    from vllm_torchtpu.kernels.qwen4_exp import retained_preparation
    from vllm_torchtpu.kernels.sparse_gqa_decode import decode
    from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import (
        build_compact_pooled_qsa_op,
    )

    torch._dynamo.reset()
    generator = torch.Generator().manual_seed(18635)

    def sample(shape, scale=0.2):
        return (torch.randn(shape, generator=generator) * scale).bfloat16().to("tpu")

    producer = (
        sample((batch, 2 * heads * dim)),
        sample((batch, kv * dim)),
        sample((batch, kv * dim)),
        sample((batch, 5 * 128)),
        *[torch.zeros(n, dtype=torch.bfloat16).to("tpu") for n in (dim, dim, 128, 128)],
    )
    trig = torch.zeros((256, 128), dtype=torch.bfloat16)
    trig[:, :32] = 1
    trig = trig.to("tpu")
    multiplier = torch.ones(dim, dtype=torch.float32).to("tpu")
    cached = torch.arange(capacity + 1, dtype=torch.int32).to("tpu")
    matrix = sample((heads * dim, width), 0.025)
    packed = (
        torch.nn.functional.pad(matrix, (0, -width % 512))
        .reshape(heads * dim, -1, 512)
        .permute(1, 0, 2)
        .contiguous()
    )
    residual = sample((batch, branches * width))
    hc_gate = torch.full((batch, branches), 0.75, dtype=torch.float32).to("tpu")
    axes = tuple(i % 3 for i in range(32))
    prepared_calls, offset_calls = [], []
    original_prepare = retained_preparation.use_retained_preparation
    original_offsets = projection.query_offsets

    def preparation(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        prepared_calls.append(result)
        return result

    def offsets(rows, prefix=None):
        result = original_offsets(rows, prefix)
        offset_calls.append(
            (rows, None if prefix is None else len(prefix), len(result))
        )
        return result

    monkeypatch.setattr(retained_preparation, "use_retained_preparation", preparation)
    monkeypatch.setattr(projection, "query_offsets", offsets)
    monkeypatch.setattr(decode, "query_offsets", offsets)

    def build(projected):
        op = build_compact_pooled_qsa_op(project_output=projected, rope_axes=axes)

        def invoke(main, index, starts, context, slots, table, distribution, positions):
            options = dict(k_norm_multiplier=multiplier, decode_cu=cached)
            if projected:
                options.update(
                    output_weight=packed,
                    residual=residual,
                    hc_gate=hc_gate,
                    original_output_weight=matrix,
                )
            return op(
                *producer,
                positions,
                trig,
                main,
                index,
                starts,
                context,
                slots,
                table,
                table,
                distribution,
                compress_ratio=4,
                token_topk=64,
                norm_eps=1e-6,
                rotary_dim=64,
                k_scale=0.375,
                v_scale=2.5,
                **options,
            )

        return torch.compile(invoke, backend="tpu", fullgraph=True, dynamic=False)

    separate, fused = build(False), build(True)
    pages = 2 * requests + 1
    packing = 4 if fp8 else 2
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    initial = [
        torch.zeros(
            (pages, 1664, math.ceil(2 * kv / packing), packing, dim), dtype=dtype
        ),
        torch.zeros((pages, 208, 1, 2, 128), dtype=torch.bfloat16),
    ]
    for owner in initial:
        owner[0] = torch.nan
    readonly = (*producer, trig, multiplier, cached, matrix, packed, residual, hc_gate)
    before = [value.cpu().view(torch.uint8).clone() for value in readonly]
    cases = [
        ([1] * (batch - 1), [batch - 1] * 3),
        ([1, 2], [1, 2, 2]),
        ([1, 2], [1, 1, 2]),
        ([], [0, 0, 0]),
    ]
    with torch.inference_mode():
        for counts, distribution in cases:
            valid = sum(counts)
            counts += [0] * (requests - len(counts))
            starts = torch.tensor([0, *np.cumsum(counts)], dtype=torch.int32).to("tpu")
            context = torch.full((requests,), 3, dtype=torch.int32).to("tpu")
            slots = torch.arange(requests + 1, 2 * requests + 1, dtype=torch.int32).to(
                "tpu"
            )
            table = torch.arange(1, requests + 1, dtype=torch.int32)[:, None].to("tpu")
            positions = torch.zeros(batch, dtype=torch.int32)
            for request, count in enumerate(counts):
                first = sum(counts[:request])
                positions[first : first + count] = 3 + torch.arange(
                    count, dtype=torch.int32
                )
            positions = positions[None].expand(3, -1).contiguous().to("tpu")
            metadata = (
                starts,
                context,
                slots,
                table,
                torch.tensor(distribution, dtype=torch.int32).to("tpu"),
                positions,
            )
            base_owners = [owner.to("tpu") for owner in initial]
            fused_owners = [owner.to("tpu") for owner in initial]
            storage = [owner.data_ptr() for owner in fused_owners]
            attended, expected_ids = separate(*base_owners, *metadata)
            actual, selected = fused(*fused_owners, *metadata)
            expected_projection = (
                (attended.cpu().double() @ matrix.cpu().double()).bfloat16().float()
            )
            expected = (
                residual.cpu().float().reshape(batch, branches, width)
                + hc_gate.cpu()[:, :, None] * expected_projection[:, None, :]
            ).bfloat16()
            a, b = actual.cpu().double().reshape(expected.shape), expected.double()
            assert torch.isfinite(a).all()
            assert float((a - b).norm() / b.norm().clamp(min=1e-12)) < 0.01
            assert torch.equal(actual.cpu()[valid:], residual.cpu()[valid:])
            assert torch.equal(selected.cpu(), expected_ids.cpu())
            assert [owner.data_ptr() for owner in fused_owners] == storage
            for left, right, original in zip(
                base_owners, fused_owners, initial, strict=True
            ):
                assert torch.equal(
                    left.cpu().view(torch.uint8), right.cpu().view(torch.uint8)
                )
                assert torch.equal(
                    right[0].cpu().view(torch.uint8), original[0].view(torch.uint8)
                )
    for value, original in zip(readonly, before, strict=True):
        assert torch.equal(value.cpu().view(torch.uint8), original)
    assert prepared_calls and all(prepared_calls)
    assert (batch, capacity + 1, batch + 1) in offset_calls
    record_property("prepared_calls", prepared_calls)
    record_property("offset_calls", offset_calls)


@pytest.mark.tpu_test
@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("fp8", [False, True])
def test_native_generic_projected_decode(geometry, fp8):
    batch, heads, dim, kv, ratio, groups, page, width, branches = geometry
    context = 2051 if geometry == GEOMETRIES[-1] else 67
    args = _case(
        batch,
        context,
        page,
        fp8,
        heads,
        kv,
        dim,
        ratio,
        groups,
        poison_packed_padding=True,
    )
    launch = plan_decode(*args, kv_heads=kv, ratio=ratio, k_scale=0.375, v_scale=2.5)
    assert launch is not None
    rng = np.random.default_rng(5173)
    matrix = jnp.asarray(rng.normal(0, 0.025, (heads * dim, width)), jnp.bfloat16)
    packed = (
        jnp.pad(matrix, ((0, 0), (0, -width % 512)))
        .reshape(heads * dim, -1, 512)
        .transpose(1, 0, 2)
    )
    residual = jnp.asarray(rng.normal(0, 0.2, (batch, branches * width)), jnp.bfloat16)
    hc = jnp.asarray(rng.uniform(0.5, 1.5, (batch, branches)), jnp.float32)
    projected_launch = plan_projected_decode(
        *args, packed, hc, kv_heads=kv, ratio=ratio, k_scale=0.375, v_scale=2.5
    )
    assert projected_launch is not None
    assert projection_fits(projected_launch, args[0], packed, hc, pltpu.get_tpu_info())
    before = np.asarray(args[5]).view(np.uint8).copy()
    run = jax.jit(
        functools.partial(
            projection.projected_attention,
            kv_heads=kv,
            ratio=ratio,
            k_scale=0.375,
            v_scale=2.5,
        )
    )
    values = (*args, jnp.arange(batch + 1, dtype=jnp.int32), packed, residual, hc)
    lowered = run.lower(*values, request_distribution=jnp.full((3,), batch, jnp.int32))
    assert "qsa_attention_output_prefetch" in lowered.as_text()
    actual, returned = lowered.compile()(
        *values, request_distribution=jnp.full((3,), batch, jnp.int32)
    )
    separate, _ = run(
        *values, request_distribution=jnp.asarray([0, 0, batch], jnp.int32)
    )
    actual, separate = np.asarray(actual, np.float64), np.asarray(separate, np.float64)
    assert np.isfinite(actual).all()
    assert (
        np.linalg.norm(actual - separate) / max(np.linalg.norm(separate), 1e-12) < 0.01
    )
    idle = np.asarray(args[3]) == 0
    np.testing.assert_array_equal(actual[idle], np.asarray(residual, np.float64)[idle])
    np.testing.assert_array_equal(np.asarray(returned).view(np.uint8), before)
    np.testing.assert_array_equal(np.asarray(args[5]).view(np.uint8), before)


def test_attention_gate_budget_includes_sublane_padding(monkeypatch):
    from vllm_torchtpu.kernels.sparse_gqa_decode.planning import (
        make_config,
        resource_usage,
    )

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    cfg = make_config(3, 3, 128, 1, 4, 17, jnp.dtype(jnp.bfloat16))
    usage = resource_usage(cfg)
    assert usage.gate_bytes == 3 * info.get_sublane_tiling(jnp.bfloat16) * 128 * 2
    assert usage.gate_bytes > 3 * 3 * 128 * 2
