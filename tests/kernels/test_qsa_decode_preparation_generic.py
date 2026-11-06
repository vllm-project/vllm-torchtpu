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
"""Run actual preparation bodies outside the former model geometry."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import qsa, retained_preparation, tuned_block_sizes


def _check_producer(
    batch, heads, kv, dim, index_heads, rotary, axial, monkeypatch, *, interpreted
):
    info = (
        pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
        if interpreted
        else pltpu.get_tpu_info()
    )
    descriptor = tuned_block_sizes.DmaHardware.from_tpu_info(info)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setattr(tuned_block_sizes, "_hardware", lambda: descriptor)
    monkeypatch.setattr(retained_preparation, "_hardware", lambda: descriptor)
    rng = np.random.default_rng(581)

    def sample(shape):
        return jnp.asarray(rng.normal(0, 0.4, shape), jnp.bfloat16)

    q, k, v = (
        sample((batch, heads * 2 * dim)),
        sample((batch, kv * dim)),
        sample((batch, kv * dim)),
    )
    iqk = sample((batch, (index_heads + 1) * 128))
    nq, nk, niq = sample((dim,)), sample((dim,)), sample((128,))
    trig = sample((64, 128))
    positions = jnp.arange(batch, dtype=jnp.int32)
    axes = None
    if axial:
        positions = jnp.stack((positions, positions + 1, positions + 2))
        axes = tuple(i % 3 for i in range(rotary // 2))
    assert retained_preparation.use_retained_preparation(
        q, k, iqk, nq, nk, niq, 1 + nk.astype(jnp.float32), trig, rotary_dim=rotary
    )
    monkeypatch.setattr(tuned_block_sizes, "use_query_prepare", lambda *a: False)
    expected = qsa.produce(
        q, k, v, iqk, nq, nk, niq, positions, trig, 1e-6, rotary, axes
    )
    original_call = pl.pallas_call

    def interpret(*args, **kwargs):
        return original_call(*args, **dict(kwargs, interpret=True))

    if interpreted:
        monkeypatch.setattr(pl, "pallas_call", interpret)
    actual = retained_preparation.produce_prepared(
        q,
        k,
        v,
        iqk,
        nq,
        1 + nk.astype(jnp.float32),
        niq,
        positions,
        trig,
        1e-6,
        rotary,
        axes,
    )
    for value, reference in zip(actual, expected, strict=True):
        value, reference = (
            np.asarray(value, np.float64),
            np.asarray(reference, np.float64),
        )
        assert value.shape == reference.shape
        assert (
            np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1e-12)
            < 0.01
        )


@pytest.mark.parametrize(
    "batch,heads,kv,dim,index_heads,rotary",
    [(1, 3, 1, 128, 2, 32), (3, 6, 2, 128, 3, 128), (5, 12, 3, 256, 4, 64)],
)
@pytest.mark.parametrize("axial", [False, True])
def test_generic_producer_interpret_matches_original(
    batch, heads, kv, dim, index_heads, rotary, axial, monkeypatch
):
    _check_producer(
        batch, heads, kv, dim, index_heads, rotary, axial, monkeypatch, interpreted=True
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "batch,heads,kv,dim,index_heads,rotary",
    [(1, 3, 1, 128, 2, 32), (3, 6, 2, 128, 3, 128), (5, 12, 3, 256, 4, 64)],
)
@pytest.mark.parametrize("axial", [False, True])
def test_native_generic_producer_matches_original(
    batch, heads, kv, dim, index_heads, rotary, axial, monkeypatch
):
    _check_producer(
        batch,
        heads,
        kv,
        dim,
        index_heads,
        rotary,
        axial,
        monkeypatch,
        interpreted=False,
    )


def test_weight_residency_lower_bound_and_page_alignment(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    hw = tuned_block_sizes.DmaHardware.from_tpu_info(info)
    monkeypatch.setattr(tuned_block_sizes, "_hardware", lambda: hw)
    assert tuned_block_sizes.output_projection_tile((6144, 2560)) == 512
    assert tuned_block_sizes.output_projection_tile((768, 193)) == 512
    assert tuned_block_sizes.output_projection_tile((131072, 8192)) is None
    q = jax.ShapeDtypeStruct((3, 6, 128), jnp.bfloat16)
    cache = jax.ShapeDtypeStruct((20, 208, 2, 128), jnp.bfloat16)
    pages = tuned_block_sizes.get_decode_index_pages(
        q, cache, pages_per_query=41, ratio=4, topk=17
    )
    assert pages == 4
    assert pages * cache.shape[1] * 2 % info.num_lanes == 0


def test_index_tile_resource_rejection_preserves_lane_alignment(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    hw = tuned_block_sizes.DmaHardware.from_tpu_info(info)
    q = jax.ShapeDtypeStruct((3, 6, 128), jnp.bfloat16)
    cache = jax.ShapeDtypeStruct((20, 208, 2, 128), jnp.bfloat16)
    monkeypatch.setattr(
        tuned_block_sizes, "_hardware", lambda: dataclasses.replace(hw, vmem_bytes=1)
    )
    assert (
        tuned_block_sizes.get_decode_index_pages(
            q, cache, pages_per_query=41, ratio=4, topk=17
        )
        is None
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("width,ratio,axes", [(256, 8, 1), (128, 4, 3)])
def test_native_state_ragged_rows_and_null_owner(width, ratio, axes):
    from vllm_torchtpu.kernels.qwen4_exp.decode_state import update_decode_state
    from vllm_torchtpu.kernels.qwen4_exp.qsa_state import (
        read_ring_and_tags,
        write_ring_and_tags,
    )

    raw = jnp.asarray(
        np.random.default_rng(519).normal(0, 0.2, (3, width)), jnp.bfloat16
    )
    pool = jnp.zeros((4, 32, 1, 2, width), jnp.bfloat16)
    ring = jnp.zeros((4, 8, width), jnp.bfloat16)
    tags = jnp.zeros((4, 8, axes + 1), jnp.int32)
    ids = jnp.asarray([1, 0, 2], jnp.int32)
    starts = jnp.asarray([0, 2, 2, 3], jnp.int32)
    context = jnp.asarray([ratio - 1, 0, 2 * ratio - 1], jnp.int32)
    positions = jnp.arange(3, dtype=jnp.int32) + ratio - 1
    rope_axes = None
    if axes == 3:
        positions = jnp.stack((positions, positions + 1, positions + 2))
        rope_axes = tuple(i % 3 for i in range(32))
    norm = jnp.ones((width,), jnp.bfloat16)
    trig = jnp.ones((64, 128), jnp.bfloat16)
    assert tuned_block_sizes.use_decode_state_fusion(
        raw, pool, requests=3, ratio=ratio, position_axes=axes
    )
    owner = jax.jit(write_ring_and_tags)(
        pool, ring.reshape(4, -1), tags.reshape(4, -1), jnp.arange(4, dtype=jnp.int32)
    )
    before = np.asarray(owner).view(np.uint8).copy()
    expected, complete, _, _, expected_ring, expected_tags = qsa._indexer_updates(
        raw,
        positions,
        ring,
        tags,
        norm,
        trig,
        starts,
        context,
        ids,
        ratio,
        1e-6,
        64,
        rope_axes,
    )
    result, keys = jax.jit(
        lambda owner: update_decode_state(
            owner,
            raw,
            positions,
            ids,
            starts,
            context,
            norm,
            trig,
            ratio=ratio,
            norm_eps=1e-6,
            rotary_dim=64,
            rope_axes=rope_axes,
        )
    )(owner)
    actual_ring, actual_tags = jax.jit(
        lambda x: read_ring_and_tags(
            x, jnp.arange(4, dtype=jnp.int32), 8 * width, 8 * (axes + 1)
        )
    )(result)
    np.testing.assert_array_equal(
        np.asarray(actual_ring), np.asarray(expected_ring).reshape(4, -1)
    )
    np.testing.assert_array_equal(
        np.asarray(actual_tags), np.asarray(expected_tags).reshape(4, -1)
    )
    actual, wanted = (
        np.asarray(keys, np.float64)[np.asarray(complete)],
        np.asarray(expected, np.float64)[np.asarray(complete)],
    )
    assert np.linalg.norm(actual - wanted) / max(np.linalg.norm(wanted), 1e-12) < 0.01
    np.testing.assert_array_equal(np.asarray(result)[0].view(np.uint8), before[0])


def test_hardware_availability_uses_cpu_and_abstract_mesh(monkeypatch):
    assert jax.default_backend() == "cpu"
    calls = []
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)

    def descriptor():
        calls.append(1)
        return info

    monkeypatch.setattr(pltpu, "get_tpu_info", descriptor)
    assert tuned_block_sizes._hardware() is None
    assert calls == []
    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("device",),
        abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"),
    )
    with jax.sharding.use_abstract_mesh(mesh):
        assert (
            tuned_block_sizes._hardware()
            == tuned_block_sizes.DmaHardware.from_tpu_info(info)
        )
    assert calls == [1]


def _check_full_rotary_query_values(monkeypatch, *, interpreted):
    from vllm_torchtpu.kernels.qwen4_exp.query_prepare import prepare_query_values

    if interpreted:
        original_call = pl.pallas_call
        monkeypatch.setattr(
            pl,
            "pallas_call",
            lambda *a, **kw: original_call(*a, **dict(kw, interpret=True)),
        )
    rng = np.random.default_rng(427)
    source = jnp.asarray(rng.normal(0, 0.2, (3, 6, 256)), jnp.bfloat16)
    norm = jnp.asarray(rng.normal(0, 0.2, 128), jnp.bfloat16)
    trig = jnp.asarray(rng.normal(0, 0.2, (64, 128)), jnp.bfloat16)
    positions = jnp.asarray([0, 7, 63], jnp.int32)
    actual = prepare_query_values(source, norm, positions, trig, 1e-6, 128)
    expected = qsa.main_rope(
        qsa.gemma_norm(source[..., :128], norm, 1e-6), positions, trig, 128
    )
    a, b = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    assert np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12) < 0.01


def test_full_rotary_query_values_interpret(monkeypatch):
    _check_full_rotary_query_values(monkeypatch, interpreted=True)


@pytest.mark.tpu_test
def test_native_full_rotary_query_values(monkeypatch):
    _check_full_rotary_query_values(monkeypatch, interpreted=False)


def _check_full_rotary_pair_writer(dma, fp8, monkeypatch, *, interpreted):
    from vllm_torchtpu.kernels.qwen4_exp.compressed_prepare import write_prepared_pairs
    from vllm_torchtpu.kernels.qwen4_exp.compressed_prepare_dma import (
        write_prepared_pairs_dma,
    )

    if interpreted:
        original_call = pl.pallas_call
        monkeypatch.setattr(
            pl,
            "pallas_call",
            lambda *a, **kw: original_call(*a, **dict(kw, interpret=True)),
        )
    rng = np.random.default_rng(498)
    values = jnp.asarray(rng.normal(0, 0.2, (4, 128)), jnp.bfloat16)
    trig = jnp.asarray(rng.normal(0, 0.2, (64, 128)), jnp.bfloat16)
    positions = jnp.asarray([0, 15, 32, 63], jnp.int32)
    dtype, packing = (jnp.float8_e4m3fn, 4) if fp8 else (jnp.bfloat16, 2)
    raw = rng.integers(0, 256, (4, 1, 1, 2, 128, 2), dtype=np.uint8)
    # Seed a NaN in the inactive half of a partial pair.
    raw.view(np.uint16).reshape(4, 2, 128)[2, 1, 64] = 0xFFBF
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.bfloat16).reshape(
        4, 1, 1, 2, 128
    )
    if fp8:
        pool = jax.lax.bitcast_convert_type(pool, dtype).reshape(4, 1, 1, packing, 128)
    pair = jnp.asarray([1, 2, 3, 0], jnp.int32)
    active = jnp.asarray([1, 1, 0, 0], jnp.int32)
    mask = jnp.asarray([[1, 1], [1, 0], [1, 1], [1, 1]], jnp.int32)
    local = jnp.asarray([[0, 1], [2, 3], [0, 0], [0, 0]], jnp.int32)
    # Snapshot materialized device storage rather than a cached creation value.
    pool = jax.jit(lambda x: x)(pool)
    reference = np.asarray(pool).view(np.uint8).copy()
    if dma:
        actual = write_prepared_pairs_dma(
            pool,
            values,
            pair,
            active,
            mask,
            local=local,
            trig_cache=trig,
            positions=positions,
            rotary_dim=128,
        )
    else:
        actual = write_prepared_pairs(
            pool,
            values,
            pair,
            active,
            mask,
            local=local,
            trig_rows=trig[positions],
            rotary_dim=128,
        )
    words = np.asarray(qsa.index_rope(values, positions, trig, 128)).view(np.uint16)
    for destination, half, token in ((1, 0, 0), (1, 1, 1), (2, 0, 2)):
        if fp8:
            carrier = reference.reshape(4, packing, 128)
            carrier[destination, 2 * half] = words[token] & 255
            carrier[destination, 2 * half + 1] = words[token] >> 8
        else:
            reference.view(np.uint16).reshape(4, 2, 128)[destination, half] = words[
                token
            ]
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint8).reshape(-1), reference.reshape(-1)
    )


@pytest.mark.parametrize("dma", [False, True])
@pytest.mark.parametrize("fp8", [False, True])
def test_full_rotary_pair_writer_interpret(dma, fp8, monkeypatch):
    _check_full_rotary_pair_writer(dma, fp8, monkeypatch, interpreted=True)


@pytest.mark.tpu_test
@pytest.mark.parametrize("dma", [False, True])
@pytest.mark.parametrize("fp8", [False, True])
def test_native_full_rotary_pair_writer(dma, fp8, monkeypatch):
    _check_full_rotary_pair_writer(dma, fp8, monkeypatch, interpreted=False)
