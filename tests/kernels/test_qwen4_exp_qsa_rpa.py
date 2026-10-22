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
"""Native-layout QSA must preserve request state across prefill and decode."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qwen4_exp_qsa_reference as qsa
from jax.experimental import pallas as pl
from qwen4_pool_test_state import pack_state, unpack_state

from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa
from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import compact_pooled_qsa_step


@pytest.fixture
def page_metadata_backend(monkeypatch):
    # Execute the same Pallas body in CPU CI; TPU tests use native lowering.
    if jax.default_backend() == "cpu":
        monkeypatch.setattr(pl, "pallas_call", partial(pl.pallas_call, interpret=True))


def test_virtual_pages_preserve_sparse_order_and_tail(page_metadata_backend):
    blocks = jnp.array([[32, 0, -1], [-1, -1, -1]], jnp.int32)
    # A budget of two complete groups plus the 133rd token's partial group.
    blocks = blocks[:, :2]
    lengths = jnp.array([133, 0], jnp.int32)
    table = jnp.array([[7, 2], [0, 1]], jnp.int32)
    pages, lens, starts, _ = qsa_rpa.virtual_page_metadata(blocks, lengths, table)
    np.testing.assert_array_equal(pages, [[64, 224, 65], [0, 0, 0]])
    np.testing.assert_array_equal(lens, [9, 0])
    np.testing.assert_array_equal(starts, [0, 1, 1])


@pytest.mark.parametrize(
    "batch,columns,ratio", [(1, 1, 4), (8, 72, 4), (13, 129, 4), (8, 33, 8)]
)
def test_virtual_pages_match_scalar_addressing(
    batch, columns, ratio, page_metadata_backend
):
    """Preserve sparse order across vector chunks, partial groups and idle rows."""
    rng = np.random.default_rng(703)
    budget = 512
    table = rng.permutation(batch * columns).reshape(batch, columns).astype(np.int32)
    lengths = np.maximum(columns * 128 - np.arange(batch) * 17, 0).astype(np.int32)
    if batch > 1:
        lengths[batch // 2] = 0
    blocks = np.full((batch, budget), -1, np.int32)
    expected = np.zeros((batch, budget + 1), np.int32)
    expected_lengths = []
    for row, length in enumerate(lengths):
        complete, tail = divmod(int(length), ratio)
        count = min(complete, budget)
        blocks[row, :count] = rng.choice(complete, count, replace=False)
        groups = list(blocks[row, :count]) + ([complete] if tail else [])
        for col, group in enumerate(groups):
            logical_page, offset = divmod(int(group) * ratio, 128)
            expected[row, col] = (
                table[row, logical_page] * (128 // ratio) + offset // ratio
            )
        expected_lengths.append(count * ratio + tail)
    pages, lens, starts, _ = qsa_rpa.virtual_page_metadata(
        jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table), ratio=ratio
    )
    np.testing.assert_array_equal(pages, expected)
    np.testing.assert_array_equal(lens, expected_lengths)
    np.testing.assert_array_equal(starts, np.r_[0, np.cumsum(lengths > 0)])


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("heads", [1, 2])
def test_head_packing_preserves_values(fp8, heads):
    k = jnp.arange(3 * heads * 256).reshape(3, heads, 256).astype(jnp.bfloat16) / 32
    v = -k
    packed = qsa_rpa.pack_rows(k, v, fp8=fp8, k_scale=0.375, v_scale=2.5)
    assert packed.dtype == (jnp.float8_e4m3fn if fp8 else jnp.bfloat16)
    decoded = packed.reshape(3, -1, 256)
    expected_k = qsa.quantize_kv(k, 0.375) if fp8 else k
    expected_v = qsa.quantize_kv(v, 2.5) if fp8 else v
    np.testing.assert_array_equal(
        np.asarray(decoded[:, : 2 * heads : 2], np.float32),
        np.asarray(expected_k, np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(decoded[:, 1 : 2 * heads : 2], np.float32),
        np.asarray(expected_v, np.float32),
    )
    if fp8 and heads == 1:
        np.testing.assert_array_equal(np.asarray(decoded[:, 2:], np.float32), 0)


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize(
    "context_base,prefill_count",
    [(126, 8), (4094, 8), (8182, 8), (8190, 8), (9206, 8), (0, 128)],
)
def test_prefill_then_decode_preserves_cache_and_causality(
    fp8, context_base, prefill_count
):
    _check_lifecycle(fp8, context_base, prefill_count)


def _check_lifecycle(
    fp8,
    context_base,
    prefill_count,
    heads=2,
    hq=24,
    dim=256,
    index_dim=128,
    ratio=4,
    budget=2048,
    padded_requests=0,
):
    rng = np.random.default_rng(313)

    def rand(shape):
        return jnp.asarray(rng.normal(0, 0.3, shape), jnp.bfloat16)

    hi = 4
    stored_dim = (dim + 127) // 128 * 128
    stored_index_dim = (index_dim + 127) // 128 * 128
    pages_per_req = max(2, (context_base + prefill_count + 2 + 127) // 128)
    pages = 2 * pages_per_req
    table = jnp.asarray(rng.permutation(pages).reshape(2, pages_per_req), jnp.int32)
    k, v = rand((pages, 128, heads, dim)), rand((pages, 128, heads, dim))
    native = qsa_rpa.pack_rows(
        k.reshape(-1, heads, dim),
        v.reshape(-1, heads, dim),
        fp8=fp8,
        k_scale=0.375,
        v_scale=2.5,
    ).reshape(pages, 128, -1, 4 if fp8 else 2, stored_dim)
    compressed, ring = (
        rand((pages, 128 // ratio, 1, index_dim)),
        rand((3, max(8, ratio), index_dim)),
    )
    tags = jnp.zeros((3, max(8, ratio), 2), jnp.int32)
    # Give each old ring lane its actual logical position.
    for row, context in enumerate([context_base, context_base + 4]):
        tags = tags.at[row, :ratio, 0].set(
            context - 1 - (context - 1 - jnp.arange(ratio)) % ratio
        )
    state = (
        native,
        jnp.pad(
            compressed, ((0, 0), (0, 0), (0, 0), (0, stored_index_dim - index_dim))
        ),
        ring,
        tags,
    )
    if fp8:
        reference_state = (
            qsa.pack_fp8_kv(qsa.quantize_kv(k, 0.375), qsa.quantize_kv(v, 2.5)),
            compressed,
            ring,
            tags,
        )
        reference_fn = qsa.qsa_step_fp8
    else:
        reference_state = (k, v, compressed, ring, tags)
        reference_fn = qsa.qsa_step
    compressed_shape, ring_shape = state[1].shape, state[2].shape
    state, ring_ids = pack_state(*state)
    initial_nulls = [np.asarray(owner[0]).copy() for owner in state]
    rope_length = context_base + prefill_count + 2
    trig = jnp.stack(
        (
            jnp.ones((rope_length, min(dim // 8, index_dim // 2))),
            jnp.zeros((rope_length, min(dim // 8, index_dim // 2))),
        ),
        axis=1,
    ).astype(jnp.bfloat16)
    opts = dict(compress_ratio=ratio, token_topk=budget, norm_eps=1e-6)
    native_fn = jax.jit(
        partial(
            compact_pooled_qsa_step,
            **opts,
            rotary_dim=2 * trig.shape[-1],
            k_scale=0.375,
            v_scale=2.5,
        ),
        donate_argnums=(10, 11),
    )
    reference_fn = jax.jit(
        partial(
            reference_fn, **opts, **(dict(k_scale=0.375, v_scale=2.5) if fp8 else {})
        )
    )
    for phase in ["prefill", "decode", "idle", "reuse"]:
        decode = phase != "prefill"
        tokens = 8 if decode else prefill_count
        starts = jnp.array(
            [0, 1, 2, 2, 2, 2, 2, 2, 2] if decode else [0, 3, prefill_count], jnp.int32
        )
        context = jnp.array(
            [context_base + 3, context_base + prefill_count + 1] + [0] * 6
            if decode
            else [context_base, context_base + 4],
            jnp.int32,
        )
        slots = jnp.array([0, 1] + [-1] * 6 if decode else [0, 1], jnp.int32)
        tt = jnp.pad(table, ((0, 6), (0, 0))) if decode else table
        pos = jnp.array(
            [context_base + 3, context_base + prefill_count + 1] + [0] * 6
            if decode
            else list(range(context_base, context_base + 3))
            + list(range(context_base + 4, context_base + prefill_count + 1)),
            jnp.int32,
        )
        producer = (
            rand((tokens, 2 * hq * dim)),
            rand((tokens, heads * dim)),
            rand((tokens, heads * dim)),
            rand((tokens, (hi + 1) * index_dim)),
            jnp.zeros((dim,), jnp.bfloat16),
            jnp.zeros((dim,), jnp.bfloat16),
            jnp.zeros((index_dim,), jnp.bfloat16),
            jnp.zeros((index_dim,), jnp.bfloat16),
            pos,
            trig,
        )
        if phase == "idle":
            starts, context, slots = (
                jnp.zeros_like(starts),
                jnp.zeros_like(context),
                jnp.full_like(slots, -1),
            )
        if phase == "reuse":
            # New requests reuse the old slots in reversed order. Logical
            # context zero must invalidate stale raw-ring/RoPE history.
            context = jnp.zeros_like(context)
            slots = jnp.array([1, 0] + [-1] * 6, jnp.int32)
            producer = (*producer[:8], jnp.zeros_like(pos), producer[9])
        if not decode and padded_requests:
            extra = padded_requests - len(context)
            starts = jnp.pad(starts, (0, extra), constant_values=prefill_count)
            context = jnp.pad(context, (0, extra))
            slots = jnp.pad(slots, (0, extra), constant_values=-1)
            tt = jnp.pad(tt, ((0, extra), (0, 0)))
        metadata = (starts, context, slots, tt, tt)
        expected = reference_fn(*producer, *reference_state, *metadata)
        flat_trig = trig.reshape(trig.shape[0], -1)
        flat_trig = jnp.pad(flat_trig, ((0, 0), (0, (-flat_trig.shape[1]) % 128)))
        native_producer = (*producer[:9], flat_trig)
        native_metadata = (
            starts,
            context,
            jnp.where(slots >= 0, slots + pages + 1, 0),
            tt + 1,
            tt + 1,
        )
        actual_pooled = native_fn(
            *native_producer,
            # These owners are independent of the reference and donated once.
            # Numerical BF16 copies can flush the opaque integer tag planes.
            *state,
            *native_metadata,
        )
        decoded = unpack_state(*actual_pooled[:2], compressed_shape, ring_shape)
        actual = (*decoded, *actual_pooled[-2:])
        for owner, before in zip(actual_pooled[:2], initial_nulls):
            np.testing.assert_array_equal(np.asarray(owner[0]), before)
        jax.block_until_ready(actual)
        a, e = np.asarray(actual[-2], np.float32), np.asarray(expected[-2], np.float32)
        err = np.linalg.norm(a - e) / max(np.linalg.norm(e), 1e-30)
        assert np.isfinite(a).all() and err < 0.01, err
        if decode:
            np.testing.assert_array_equal(a[2:], 0)
        for n, (new, old) in enumerate(zip(actual[1:4], expected[-5:-2])):
            if n == 0:
                new = new[..., :index_dim]
            np.testing.assert_array_equal(np.asarray(new), np.asarray(old))
        for aa, ee in zip(np.asarray(actual[-1]), np.asarray(expected[-1])):
            np.testing.assert_array_equal(np.sort(aa), np.sort(ee))
        rows = actual[0]
        assert rows.dtype == (jnp.float8_e4m3fn if fp8 else jnp.bfloat16)
        if fp8:
            ek, ev = qsa.unpack_fp8_kv(expected[0])
        else:
            ek, ev = expected[:2]
        rows = rows.reshape(pages, 128, -1, stored_dim)[:, :, : 2 * heads, :dim]
        np.testing.assert_array_equal(
            np.asarray(rows[:, :, 0::2], np.float32), np.asarray(ek, np.float32)
        )
        np.testing.assert_array_equal(
            np.asarray(rows[:, :, 1::2], np.float32), np.asarray(ev, np.float32)
        )
        state, reference_state = actual_pooled[:2], expected[:-2]


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
def test_one_kv_head_rpa_lifecycle(fp8):
    _check_lifecycle(fp8, 126, 8, heads=1, hq=3)


@pytest.mark.parametrize("budget", [16, 512, 2048, 4096])
def test_query_batch_bounds_scalar_page_metadata(budget):
    batch = qsa_rpa.rpa_query_batch_size(budget + 1)
    assert 0 < batch <= 64
    assert batch * (budget + 1) * 4 <= 160 * 1024
    if budget == 512:
        assert batch == 64


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
def test_alternative_geometry_with_padded_head_dimensions(fp8):
    _check_lifecycle(
        fp8, 126, 8, heads=4, hq=8, dim=64, index_dim=32, ratio=8, budget=128
    )


@pytest.mark.parametrize("fp8", [False, True])
def test_empty_step_preserves_native_owners(fp8):
    state = (
        jnp.zeros(
            (2, 128, 1, 4 if fp8 else 2, 256),
            jnp.float8_e4m3fn if fp8 else jnp.bfloat16,
        ),
        jnp.zeros((2, 16, 1, 2, 128), jnp.bfloat16),
    )
    args = [jnp.zeros((0, n), jnp.bfloat16) for n in (1536, 256, 256, 640)]
    args += [jnp.zeros(n, jnp.bfloat16) for n in (256, 256, 128, 128)]
    args += [jnp.zeros(0, jnp.int32), jnp.zeros((128, 128), jnp.bfloat16)]
    result = compact_pooled_qsa_step(
        *args,
        *state,
        jnp.array([0], jnp.int32),
        jnp.zeros(0, jnp.int32),
        jnp.zeros(0, jnp.int32),
        jnp.zeros((0, 1), jnp.int32),
        jnp.zeros((0, 1), jnp.int32),
        compress_ratio=4,
        token_topk=2048,
        norm_eps=1e-6,
        rotary_dim=64,
    )
    for actual, expected in zip(result[:2], state):
        np.testing.assert_array_equal(actual, expected)
    assert result[2].shape == (0, 768)
    assert result[3].shape == (0, 2051)


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("tokens", [4, 8, 16])
def test_multitoken_prefill_with_idle_request_capacity(fp8, tokens):
    """T<R, T==R and T>R preserve the same causal lifecycle oracle."""
    _check_lifecycle(fp8, 126, tokens, heads=1, hq=3, padded_requests=8)


@pytest.mark.parametrize("ratio", [1, 2, 4, 8])
def test_selected_tokens_preserve_sparse_order_padding_and_partial_tail(ratio):
    blocks = np.array([[2, 0, -1], [0, 0, 1], [-1, -1, -1]], np.int32)
    lengths = np.array([2 * ratio + 1, 4 * ratio - 1, 0], np.int32)
    expected = []
    for row, length in zip(blocks, lengths):
        complete, remainder = divmod(int(length), ratio)
        tokens = [
            int(group) * ratio + offset if group >= 0 else -1
            for group in row
            for offset in range(ratio)
        ] + [-1] * (ratio - 1)
        start = min(complete, len(row)) * ratio
        tokens[start : start + ratio - 1] = [
            complete * ratio + offset if offset < remainder else -1
            for offset in range(ratio - 1)
        ]
        expected.append(tokens)
    actual = jax.jit(qsa_rpa.selected_tokens, static_argnums=(2, 3))(
        jnp.asarray(blocks), jnp.asarray(lengths), blocks.shape[1] * ratio, ratio
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("index_dim,ratio", [(32, 8), (128, 4)])
def test_dual_store_fixture_lane_words_match_mosaic(fp8, index_dim, ratio):
    """The independent host oracle must match Mosaic's sublane, not lane bitcast."""
    from jax.experimental.pallas import tpu as pltpu

    main = jnp.ones(
        (2, 128, 1, 4 if fp8 else 2, 128), jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    )
    keys = (
        jnp.arange(2 * (128 // ratio) * 128)
        .reshape(2, 128 // ratio, 1, 128)
        .astype(jnp.bfloat16)
    )
    ring = jnp.arange(2 * 8 * index_dim).reshape(2, 8, index_dim).astype(jnp.bfloat16)
    tags = jnp.tile(jnp.array([[0x1234567, -7], [-1, 31]], jnp.int32), (2, 4, 1))
    state, ring_ids = pack_state(main, keys, ring, tags)
    restored = unpack_state(*state, keys.shape, ring.shape)
    for actual, expected in zip(restored, (main, keys, ring, tags)):
        np.testing.assert_array_equal(actual, expected)
    tag_token = (8 * index_dim + 255) // 256
    encoded = state[1][ring_ids, tag_token : tag_token + 1]
    actual = jax.jit(lambda x: pltpu.bitcast(x, jnp.int32))(encoded)
    np.testing.assert_array_equal(
        np.asarray(actual).reshape(2, -1)[:, :16], np.asarray(tags).reshape(2, -1)
    )
    assert state[0].dtype == main.dtype and state[1].dtype == jnp.bfloat16
    np.testing.assert_array_equal(state[0][0], 0)
    np.testing.assert_array_equal(state[1][0], 0)


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
def test_dual_store_fixture_preserves_opaque_tag_bits(fp8):
    """Validate host packing and native transfer against original integer words."""
    from vllm_torchtpu.kernels.qwen4_exp.qsa_state import read_ring_and_tags

    main = jnp.ones(
        (2, 128, 1, 4 if fp8 else 2, 128), jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    )
    keys = jnp.zeros((2, 32, 1, 128), jnp.bfloat16)
    ring = jnp.ones((3, 8, 128), jnp.bfloat16)
    words = np.array(
        [0, 1, 125, 127, 128, 65535, 65536, 0x1234567, -1, -7, -(2**31), 2**31 - 1],
        np.int32,
    )
    tags = np.resize(words, (3, 8, 2))
    state, ids = pack_state(main, keys, ring, tags)
    restored = unpack_state(*state, keys.shape, ring.shape)
    for actual, expected in zip(restored, (main, keys, ring, tags)):
        np.testing.assert_array_equal(actual, expected)
    _, device_tags = read_ring_and_tags(state[1], jnp.asarray(ids), 8 * 128, 8 * 2)
    np.testing.assert_array_equal(np.asarray(device_tags).reshape(tags.shape), tags)
    np.testing.assert_array_equal(state[0][0], 0)
    np.testing.assert_array_equal(state[1][0], 0)
