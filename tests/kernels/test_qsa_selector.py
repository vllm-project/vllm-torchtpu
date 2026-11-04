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
"""Exact workload bounds and lazy prepared-TC dispatch contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from tests.kernels.test_qsa_prefill_tc_packing import inputs
from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as tc
from vllm_torchtpu.kernels.sparse_gqa_prefill import selector as sel

pytest_plugins = ["tests.kernels.test_qsa_prefill_tc_packing"]


@pytest.fixture(autouse=True)
def target(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


def profile_for(xs, **overrides):
    values = dict(
        tc_fixed_ms=0.0,
        tc_word_ms=0.0,
        tc_visit_ms=1.0,
        sc_fixed_ms=19.0,
        sc_tile_ms=0.0,
        sc_page_ms=0.0,
    )
    values.update(overrides)
    return sel.CostModel(**values)


def run(xs, profile):
    return sel.select_prefill(*xs, cost_model=profile, kv_heads=2, ratio=4)


@pytest.mark.parametrize("ratio,page,qb", [(2, 640, 32), (4, 1664, 256), (8, 128, 32)])
@pytest.mark.parametrize("seed", [7, 31, 99])
def test_bounds_use_original_membership_and_match_independent_sets(
    ratio, page, qb, seed
):
    rng = np.random.default_rng(seed)
    rows, slots, columns = 2 * qb, 16, 3
    groups = columns * page // ratio
    blocks = rng.integers(-3, groups + 4, (rows, slots), dtype=np.int32)
    blocks[:, 1] = blocks[:, 0]
    blocks[::3, -1] = np.iinfo(np.int32).max
    lengths = rng.integers(-2, columns * page + 3, rows, dtype=np.int32)
    lengths[:8] = [
        0,
        1,
        ratio,
        ratio + 1,
        page - 1,
        page,
        columns * page,
        columns * page + 1,
    ]
    expected = []
    for begin in range(0, rows, qb):
        union = set()
        for b, length in zip(blocks[begin : begin + qb], lengths[begin : begin + qb]):
            complete = min(int(length) // ratio, groups)
            for slot, value in enumerate(b):
                if length > 0 and slot < min(complete, slots) and 0 <= value < complete:
                    union.add(int(value) // 32)
            if length > 0 and length % ratio and length // ratio < groups:
                union.add(int(length) // ratio // 32)
        expected.append(len(union))
    b, ls = jnp.asarray(blocks), jnp.asarray(lengths)
    tile = ratio * 32
    prepared = tc._prepare_prefill_metadata(
        b,
        ls,
        jnp.zeros((rows, columns), jnp.int32),
        ratio=ratio,
        query_block=qb,
        kv_tile=tile,
        physical_page_size=page,
        cache_dtype=jnp.bfloat16,
    )
    np.testing.assert_array_equal(prepared.active_counts, expected)
    kwargs = dict(ratio=ratio, groups=groups, kv_tile=tile, query_block=qb)
    lower = np.asarray(sel.sampled_lower(b, ls, **kwargs))
    span = np.asarray(sel.span_upper(b, ls, **kwargs))
    coarse = np.asarray(sel.bucket_upper(b, ls, **kwargs))
    assert np.all(lower <= expected)
    assert np.all(span >= expected)
    assert np.all(coarse >= expected)


def test_prepared_consumption_exact_and_constructed_once(core_calls, monkeypatch):
    xs = list(inputs(256, [0, 256], columns=10))
    xs[2] = jnp.broadcast_to(jnp.arange(32, 544, dtype=jnp.int32), (256, 512))
    xs[2] = xs[2].at[1].set(jnp.arange(800, 1312, dtype=jnp.int32))
    xs[3] = jnp.full(256, 8192, jnp.int32)
    xs = tuple(xs)
    original = tc._prepare_prefill_metadata
    constructions = []

    def prepare(*args, **kwargs):
        constructions.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(tc, "_prepare_prefill_metadata", prepare)
    with jax.disable_jit():
        y, route = run(xs, profile_for(xs))
    assert int(route) == 1
    assert len(constructions) == 1
    assert len(core_calls) == 1
    constructions.clear()
    with jax.disable_jit():
        reference, _ = tc.block_sparse_gqa_attention(
            *xs, kv_heads=2, ratio=4, query_block=256, kv_tile=256
        )
    np.testing.assert_array_equal(y, reference)
    assert len(constructions) == 1


@pytest.mark.parametrize("which", ["cheap_tc", "sampled_sc"])
def test_decided_routes_do_not_prepare_twice(core_calls, monkeypatch, which):
    xs = inputs(256, [0, 256], columns=1)
    original = tc._prepare_prefill_metadata
    calls = []

    def prepare(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(tc, "_prepare_prefill_metadata", prepare)
    monkeypatch.setattr(
        sel.owner_window, "owner_window_attention", lambda *a, **k: jnp.zeros_like(a[0])
    )
    cost = 100000.0 if which == "cheap_tc" else 0.0
    with jax.disable_jit():
        _, route = run(xs, profile_for(xs, sc_fixed_ms=cost))
    assert int(route) == (0 if which == "cheap_tc" else 2)
    assert len(calls) == (1 if which == "cheap_tc" else 0)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "future",
        "negative",
        "ragged",
        "empty_requests",
    ],
)
def test_special_inputs_keep_original_tc_semantics(core_calls, mutation):
    xs = list(inputs(256, [0, 256], columns=1))
    xs[2] = jnp.tile(jnp.arange(4, dtype=jnp.int32), (256, 1))
    xs[3] = jnp.full(256, 16, jnp.int32)
    if mutation == "duplicate":
        xs[2] = xs[2].at[:, 1].set(0)
    if mutation == "future":
        xs[2] = xs[2].at[:, 1].set(500)
    if mutation == "negative":
        xs[2] = xs[2].at[:, 1].set(-1)
    if mutation == "ragged":
        xs[6] = jnp.array([0, 17, 256], jnp.int32)
    if mutation == "empty_requests":
        xs[6] = jnp.array([0, 0, 0], jnp.int32)
        xs[3] = jnp.zeros(256, jnp.int32)
    xs = tuple(xs)
    profile = profile_for(xs, sc_fixed_ms=100000.0)
    with jax.disable_jit():
        actual, route = run(xs, profile)
        expected, _ = tc.block_sparse_gqa_attention(
            *xs, kv_heads=2, ratio=4, query_block=256, kv_tile=256
        )
    assert int(route) == (0 if mutation in ["duplicate", "future", "negative"] else 4)
    np.testing.assert_array_equal(actual, expected)


def test_static_scales_are_preserved_on_safe_tc_fallback(monkeypatch):
    xs = inputs(256, [0, 256])
    captured = []

    def original(*args, **kwargs):
        captured.append(kwargs)
        return jnp.zeros_like(args[0]), args[5]

    from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa

    monkeypatch.setattr(qsa_rpa, "_prefill_attention_tc", original)
    _, route = sel.select_prefill(
        *xs,
        cost_model=profile_for(xs),
        kv_heads=2,
        ratio=4,
        k_scale=0.75,
        v_scale=1.25,
    )
    assert int(route) == 4
    assert captured[0]["k_scale"] == 0.75 and captured[0]["v_scale"] == 1.25


@pytest.mark.parametrize("duplicate", [False, True])
@pytest.mark.parametrize("choose_sc", [False, True])
def test_unordered_membership_refines_once_without_mutation(
    core_calls, monkeypatch, duplicate, choose_sc
):
    xs = list(inputs(256, [0, 256], columns=1))
    xs[3] = jnp.full(256, 7, jnp.int32)
    # One complete slot plus an implicit tail; padding has no membership.
    xs[2] = jnp.tile(jnp.array([0, -1, -1, -1], jnp.int32), (256, 1))
    xs[3] = xs[3].at[:128].set(17)
    prefix = jnp.array([2, 0, 3, 1], jnp.int32)
    if duplicate:
        prefix = prefix.at[2].set(2)
    xs[2] = xs[2].at[:128].set(prefix)
    xs = tuple(xs)
    before = np.asarray(xs[2]).tobytes()
    calls, sparse_calls = [], []
    original = tc._prepare_prefill_metadata

    def prepare(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    def sparse(*args, **kwargs):
        sparse_calls.append(1)
        return jnp.full_like(args[0], 5)

    monkeypatch.setattr(tc, "_prepare_prefill_metadata", prepare)
    monkeypatch.setattr(sel.owner_window, "owner_window_attention", sparse)
    with jax.disable_jit():
        output, route = run(
            xs, profile_for(xs, sc_fixed_ms=0.0 if choose_sc else 100000.0)
        )
    assert len(calls) == 1
    assert int(route) == (0 if not choose_sc else 4 if duplicate else 3)
    assert bool(sparse_calls) == (choose_sc and not duplicate)
    assert np.asarray(xs[2]).tobytes() == before
    if duplicate or not choose_sc:
        with jax.disable_jit():
            expected, _ = tc.block_sparse_gqa_attention(
                *xs, kv_heads=2, ratio=4, query_block=256, kv_tile=256
            )
        np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("kind", ["unordered", "duplicate", "invalid_padding"])
def test_cheap_tc_skips_sc_only_certificates(core_calls, monkeypatch, kind):
    xs = list(inputs(256, [0, 256], columns=1))
    xs[3] = jnp.full(256, 7, jnp.int32)
    xs[2] = jnp.tile(jnp.array([0, -1, -1, -1], jnp.int32), (256, 1))
    if kind != "invalid_padding":
        xs[3] = jnp.full(256, 17, jnp.int32)
        prefix = jnp.array([2, 0, 3, 1], jnp.int32)
        if kind == "duplicate":
            prefix = prefix.at[2].set(2)
        xs[2] = jnp.broadcast_to(prefix, (256, 4))
    xs = tuple(xs)

    def forbidden(*args, **kwargs):
        raise AssertionError("SC-only certificate executed on cheap TC")

    monkeypatch.setattr(sel, "sc_prefix_certificates", forbidden)
    monkeypatch.setattr(sel, "prepared_membership_is_unique", forbidden)
    with jax.disable_jit():
        output, route = run(xs, profile_for(xs, sc_fixed_ms=100000.0))
        expected, _ = tc.block_sparse_gqa_attention(
            *xs, kv_heads=2, ratio=4, query_block=256, kv_tile=256
        )
    assert int(route) == 0
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("choose_sc", [False, True])
def test_single_sc_execution_and_initialized_placeholder(
    core_calls, monkeypatch, choose_sc
):
    xs = inputs(256, [0, 256], columns=10)
    sparse_calls, initialized = [], []
    zeros_like = jnp.zeros_like

    def zeros(x, *args, **kwargs):
        output = zeros_like(x, *args, **kwargs)
        if x is xs[0]:
            initialized.append(np.asarray(output))
        return output

    def sparse(*args, **kwargs):
        sparse_calls.append(1)
        return jnp.full_like(args[0], 5)

    monkeypatch.setattr(sel.jnp, "zeros_like", zeros)
    monkeypatch.setattr(sel.owner_window, "owner_window_attention", sparse)
    with jax.disable_jit():
        output, route = run(
            xs, profile_for(xs, sc_fixed_ms=0.0 if choose_sc else 100000.0)
        )
    assert len(sparse_calls) == int(choose_sc)
    assert len(initialized) == int(choose_sc)
    if choose_sc:
        assert int(route) == 2
        np.testing.assert_array_equal(initialized[0], 0)
        np.testing.assert_array_equal(output, 5)
    else:
        assert int(route) == 0
        with jax.disable_jit():
            expected, _ = tc.block_sparse_gqa_attention(
                *xs, kv_heads=2, ratio=4, query_block=256, kv_tile=256
            )
        np.testing.assert_array_equal(output, expected)


def test_sc_body_traces_once_across_metadata_extent_branches(core_calls, monkeypatch):
    xs = inputs(256, [0, 256], columns=158)
    traces = []

    def sparse(*args, **kwargs):
        traces.append(1)
        return jnp.full_like(args[0], 5)

    monkeypatch.setattr(sel.owner_window, "owner_window_attention", sparse)
    jax.make_jaxpr(lambda *args: run(args, profile_for(xs)))(*xs)
    assert len(traces) == 1


def _sampled_tile_set_counts(blocks, lengths, *, ratio, groups, kv_tile, query_block):
    """Independent Python sets; no bitmap construction or selector helpers."""
    result = []
    for begin in range(0, len(blocks), query_block):
        touched = set()
        for row in range(begin, begin + query_block, 32):
            length = int(lengths[row])
            if length <= 0:
                continue
            complete = min(length // ratio, groups)
            for slot, group in enumerate(blocks[row]):
                if slot < min(complete, blocks.shape[1]) and 0 <= group < complete:
                    touched.add(int(group) * ratio // kv_tile)
            if length % ratio and length // ratio < groups:
                touched.add((length // ratio) * ratio // kv_tile)
        result.append(len(touched))
    return result


@pytest.mark.parametrize("ntiles", [1, 31, 32, 33, 63, 64, 65])
@pytest.mark.parametrize("ratio", [1, 4])
def test_sampled_lower_exact_sets_across_word_boundaries(ntiles, ratio):
    rows, query_block, budget, groups = 512, 256, 17, ntiles * 2 - 1
    rng = np.random.default_rng(120 + ntiles)
    blocks = rng.integers(-3, groups + 5, (rows, budget), dtype=np.int32)
    lengths = rng.integers(-2, groups * ratio + 2, rows, dtype=np.int32)
    blocks[:, 1] = blocks[:, 0]
    blocks[:, -1] = np.iinfo(np.int32).max
    # Nonascending samples include the sign bit and the next word's first bit.
    blocks[0, :7] = np.array([64, 62, 0, 62, -1, groups, groups + 1])
    lengths[0] = groups * ratio
    lengths[32] = max(0, (groups - 1) * ratio + ratio - 1)
    blocks[64, 0] = groups - 1
    lengths[64] = groups * ratio
    lengths[96] = 0
    lengths[128] = -3
    lengths[256:] = 0
    kwargs = dict(
        ratio=ratio, groups=groups, kv_tile=ratio * 2, query_block=query_block
    )
    expected = _sampled_tile_set_counts(blocks, lengths, **kwargs)
    actual = sel.sampled_lower(jnp.asarray(blocks), jnp.asarray(lengths), **kwargs)
    np.testing.assert_array_equal(actual, expected)
    # Every unsampled row may change without changing this strict subset union.
    unsampled = np.arange(rows) % 32 != 0
    changed = blocks.copy()
    changed[unsampled] = 0
    changed_lengths = lengths.copy()
    changed_lengths[unsampled] = groups * ratio
    np.testing.assert_array_equal(
        sel.sampled_lower(jnp.asarray(changed), jnp.asarray(changed_lengths), **kwargs),
        expected,
    )


def test_sampled_lower_partial_tail_future_and_masked_suffix():
    blocks = np.full((256, 7), -1, np.int32)
    lengths = np.zeros(256, np.int32)
    # Query0's group1 is partial; query32 fully selects that same group.
    blocks[0] = [0, 7, 6, 5, 4, 3, 2]
    lengths[0] = 7
    blocks[32] = [1, 1, 0, 9, -1, 7, 6]
    lengths[32] = 12
    # A non-sampled valid group must not inflate the lower bound.
    blocks[1, 0] = 7
    lengths[1] = 32
    kwargs = dict(ratio=4, groups=10, kv_tile=4, query_block=256)
    expected = _sampled_tile_set_counts(blocks, lengths, **kwargs)
    assert expected == [2]
    np.testing.assert_array_equal(
        sel.sampled_lower(jnp.asarray(blocks), jnp.asarray(lengths), **kwargs),
        expected,
    )
    lengths[:] = 0
    np.testing.assert_array_equal(
        sel.sampled_lower(jnp.asarray(blocks), jnp.asarray(lengths), **kwargs), [0]
    )
