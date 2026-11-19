# SPDX-License-Identifier: Apache-2.0
"""Native-layout QSA must preserve request state across prefill and decode."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qwen4_exp_qsa_reference as qsa
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa


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
    first_query=3,
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
            qsa_rpa.qsa_step,
            **opts,
            rotary_dim=2 * trig.shape[-1],
            k_scale=0.375,
            v_scale=2.5,
        ),
        donate_argnums=(10, 11, 12, 13),
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
            [0, 1, 2, 2, 2, 2, 2, 2, 2] if decode else [0, first_query, prefill_count],
            jnp.int32,
        )
        context = jnp.array(
            [context_base + first_query, context_base + prefill_count - first_query + 4]
            + [0] * 6
            if decode
            else [context_base, context_base + 4],
            jnp.int32,
        )
        slots = jnp.array([0, 1] + [-1] * 6 if decode else [0, 1], jnp.int32)
        tt = jnp.pad(table, ((0, 6), (0, 0))) if decode else table
        pos = jnp.array(
            [context_base + first_query, context_base + prefill_count - first_query + 4]
            + [0] * 6
            if decode
            else list(range(context_base, context_base + first_query))
            + list(
                range(context_base + 4, context_base + prefill_count - first_query + 4)
            ),
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
        metadata = (starts, context, slots, tt, tt)
        expected = reference_fn(*producer, *reference_state, *metadata)
        flat_trig = trig.reshape(trig.shape[0], -1)
        flat_trig = jnp.pad(flat_trig, ((0, 0), (0, (-flat_trig.shape[1]) % 128)))
        native_producer = (*producer[:9], flat_trig)
        actual = native_fn(
            *native_producer, *(jnp.array(x, copy=True) for x in state), *metadata
        )
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
        state, reference_state = actual[:-2], expected[:-2]


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
        jnp.zeros((2, 32, 1, 128), jnp.bfloat16),
        jnp.zeros((2, 8, 128), jnp.bfloat16),
        jnp.zeros((2, 8, 2), jnp.int32),
    )
    args = [jnp.zeros((0, n), jnp.bfloat16) for n in (1536, 256, 256, 640)]
    args += [jnp.zeros(n, jnp.bfloat16) for n in (256, 256, 128, 128)]
    args += [jnp.zeros(0, jnp.int32), jnp.zeros((128, 128), jnp.bfloat16)]
    result = qsa_rpa.qsa_step(
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
    for actual, expected in zip(result[:4], state):
        np.testing.assert_array_equal(actual, expected)
    assert result[4].shape == (0, 768)
    assert result[5].shape == (0, 2051)


@pytest.mark.tpu_test
@pytest.mark.parametrize("prefix_slice", [False, True])
@pytest.mark.parametrize("fp8", [False, True])
def test_mixed_prefill_decode_matches_independent_state_reference(fp8, prefix_slice, monkeypatch):
    """VLLM_TPU_QSA_PREFIX_SLICE moves the decode prefix with dynamic slices, not row gathers."""
    from vllm_torchtpu import envs

    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFIX_SLICE", prefix_slice)
    _check_lifecycle(fp8, 126, 8, first_query=1)


@pytest.mark.parametrize("cap", [0, 1, 16])
def test_prefix_slice_helpers_match_the_row_gathers(cap):
    rng = np.random.default_rng(cap)
    x = jnp.asarray(rng.normal(size=(64, 3, 8)), jnp.bfloat16)
    r = np.arange(64)
    for shift in range(cap + 1):
        s = jnp.int32(shift)
        np.testing.assert_array_equal(
            qsa_rpa._shift_rows_up(x, s, cap), x[np.minimum(r + shift, 63)]
        )
        np.testing.assert_array_equal(
            qsa_rpa._shift_rows_down(x, s, cap), x[np.maximum(r - shift, 0)]
        )
    if cap:
        d = x[:cap]
        np.testing.assert_array_equal(qsa_rpa._pad_rows(d, 64), d[np.minimum(r, cap - 1)])


@pytest.mark.tpu_test
@pytest.mark.parametrize("first_query", [1, 3])
@pytest.mark.parametrize("prefix_slice", [False, True])
@pytest.mark.parametrize("fp8", [False, True])
def test_zero_prefix_fast_lifecycle(fp8, prefix_slice, first_query, monkeypatch):
    """VLLM_TPU_QSA_ZERO_PREFIX_FAST on the real kernels: the prefill step has a
    one-token decode prefix (first_query=1) or none (3 > the 2-row capacity)."""
    from vllm_torchtpu import envs

    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFIX_SLICE", prefix_slice)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_ZERO_PREFIX_FAST", True)
    _check_lifecycle(fp8, 126, 8, first_query=first_query)


# VLLM_TPU_QSA_ZERO_PREFIX_FAST routing, on CPU: stand-in kernels that depend on
# every routed input, on each row's request and position under ``starts``, on a
# neighbouring row, on every row's length and on the cache, so a misrouted row, length, start or cache
# shows in the bits. The real kernels' step is test_zero_prefix_fast_lifecycle.
_ROWS, _SLOTS, _HEADS, _DIM = 48, 8, 2, 8


def _fake_decode(q, gate, blocks, lengths, table, cache, **_):
    q32 = q.astype(jnp.float32)
    side = (blocks.sum(1) * 3 + table.sum(1) * 5 + lengths * 7).astype(jnp.float32)
    out = q32 * 1.5 + gate.astype(jnp.float32) * 0.75 + side[:, None, None] / 64
    out = out + cache.sum().astype(jnp.float32) / 1024
    out = out + (lengths.sum() % 89).astype(jnp.float32) / 4
    out = jnp.where((lengths > 0)[:, None, None], out, 0)
    return out.astype(jnp.bfloat16), cache + 1


def _fake_prefill(q, gate, blocks, lengths, table, cache, starts, **_):
    rows = jnp.arange(len(q))
    req = jnp.clip(jnp.searchsorted(starts, rows, side="right") - 1, 0, len(starts) - 1)
    pos = rows - starts[req]
    first = starts[req]
    prev = q[jnp.maximum(rows - 1, first)].astype(jnp.float32)
    side = (blocks.sum(1) * 3 + table.sum(1) * 5 + lengths * 7 + req * 11 + pos * 13)
    out = q.astype(jnp.float32) * 2 + prev * 0.5 + gate.astype(jnp.float32)
    out = out + side.astype(jnp.float32)[:, None, None] / 128
    out = out + cache.sum().astype(jnp.float32) / 4096
    # Every row sees every length (a row-local kernel would hide the lengths of
    # rows the merge drops).
    out = out + (lengths.sum() % 97).astype(jnp.float32) / 4
    # No length mask: rows past the step stay nonzero, so the final mask shows.
    return out.astype(jnp.bfloat16), cache * 3


def _routing_step(starts_list, seed=0):
    rng = np.random.default_rng(seed)
    starts = np.asarray(starts_list + [starts_list[-1]] * (_SLOTS + 1 - len(starts_list)))
    lengths = np.zeros(_ROWS, np.int32)
    for r in range(_SLOTS):
        lo, hi = starts[r], starts[r + 1]
        lengths[lo:hi] = 100 * (r + 1) + np.arange(1, hi - lo + 1)
    shape = (_ROWS, _HEADS, _DIM)
    q = jnp.asarray(rng.normal(0, 1, shape), jnp.bfloat16)
    gate = jnp.asarray(rng.normal(0, 1, shape), jnp.bfloat16)
    blocks = jnp.asarray(rng.integers(0, 50, (_ROWS, 4)), jnp.int32)
    table = jnp.asarray(rng.integers(0, 9, (_ROWS, 3)), jnp.int32)
    cache = jnp.asarray(rng.integers(0, 5, (6, 4)), jnp.int32)
    return (q, gate, blocks, jnp.asarray(lengths), table, cache, jnp.asarray(starts, jnp.int32))


# (starts, decode prefix): none (a long first request); none on a step that
# fills every row; a 1- and a 3-token request; short requests until one ends
# past the 8-row capacity; one short request and padding; the capacity filled
# exactly by short requests; a prefix on a step that fills every row (the
# shifted rows clamp to the last, live row).
_ROUTING_STEPS = {
    "prefix0": ([0, 20, 30, 41], 0),
    "prefix0_full_rows": ([0, 30, 48], 0),
    "prefix4": ([0, 1, 4, 24, 40], 4),
    "prefix5_capacity_stop": ([0, 2, 5, 9, 28], 5),
    "prefix3_padding": ([0, 3], 3),
    "prefix8_full": ([0, 2, 4, 8, 30], 8),
    "prefix5_full_rows": ([0, 2, 5, 48], 5),
}


def _route(step, monkeypatch, *, fast, prefix_slice, decode=_fake_decode):
    from vllm_torchtpu import envs

    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFIX_SLICE", prefix_slice)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_ZERO_PREFIX_FAST", fast)
    monkeypatch.setattr(qsa_rpa, "decode_attention", decode)
    monkeypatch.setattr(qsa_rpa, "_prefill_attention", _fake_prefill)
    fn = jax.jit(partial(qsa_rpa.qsa_attention, kv_heads=1, ratio=4))
    out, cache = fn(*step)
    return np.asarray(out).view(np.uint16), np.asarray(cache)


@pytest.mark.parametrize("prefix_slice", [False, True])
@pytest.mark.parametrize("name", list(_ROUTING_STEPS))
def test_zero_prefix_fast_matches_the_routed_path(name, prefix_slice, monkeypatch):
    starts_list, prefix = _ROUTING_STEPS[name]
    step = _routing_step(starts_list)
    assert int(qsa_rpa.short_request_prefix_tokens(step[-1], _SLOTS)) == prefix
    ref_out, ref_cache = _route(step, monkeypatch, fast=False, prefix_slice=prefix_slice)
    out, cache = _route(step, monkeypatch, fast=True, prefix_slice=prefix_slice)
    np.testing.assert_array_equal(out, ref_out)
    np.testing.assert_array_equal(cache, ref_cache)
    # The stand-ins reach every row: live rows are nonzero, padding rows zero.
    live = np.asarray(step[3]) > 0
    assert (ref_out[live] != 0).any(axis=(1, 2)).all() and not ref_out[~live].any()
    # Both kernels ran once, in order: decode then prefill.
    np.testing.assert_array_equal(ref_cache, (np.asarray(step[5]) + 1) * 3)


@pytest.mark.parametrize("prefix_slice", [False, True])
def test_zero_prefix_fast_runs_the_rows_in_place(prefix_slice, monkeypatch):
    """With prefix 0 neither the row movement nor the merge runs: poisoned moves
    leave the output unchanged; with a prefix the same poison shows."""
    real = qsa_rpa._prefix_moves

    def poisoned(decoded, prefix, n, capacity):
        move_in, move_out, decoded_rows = real(decoded, prefix, n, capacity)
        return (
            lambda x: jnp.roll(move_in(x), 1, axis=0),
            lambda x: jnp.roll(move_out(x), 1, axis=0),
            lambda: decoded_rows() + 1,
        )

    for name, bites in (("prefix0", False), ("prefix4", True)):
        step = _routing_step(_ROUTING_STEPS[name][0], seed=1)
        ref_out, ref_cache = _route(step, monkeypatch, fast=False, prefix_slice=prefix_slice)
        monkeypatch.setattr(qsa_rpa, "_prefix_moves", poisoned)
        out, cache = _route(step, monkeypatch, fast=True, prefix_slice=prefix_slice)
        monkeypatch.setattr(qsa_rpa, "_prefix_moves", real)
        np.testing.assert_array_equal(cache, ref_cache)
        assert np.array_equal(out, ref_out) is not bites, name


def _traced(step, monkeypatch, **flags):
    calls = []

    def counting_prefill(*args, **kwargs):
        calls.append(len(args[0]))
        return _fake_prefill(*args, **kwargs)

    from vllm_torchtpu import envs

    for name, value in flags.items():
        monkeypatch.setattr(envs, name, value)
    monkeypatch.setattr(qsa_rpa, "decode_attention", _fake_decode)
    monkeypatch.setattr(qsa_rpa, "_prefill_attention", counting_prefill)
    jaxpr = jax.make_jaxpr(partial(qsa_rpa.qsa_attention, kv_heads=1, ratio=4))(*step)
    conds = [e for e in jaxpr.eqns if e.primitive.name == "cond"]
    return calls, conds, str(jaxpr)


def test_zero_prefix_fast_dispatch(monkeypatch):
    step = _routing_step(_ROUTING_STEPS["prefix0"][0])
    off = dict(VLLM_TPU_QSA_PREFIX_SLICE=False, VLLM_TPU_QSA_ZERO_PREFIX_FAST=False)
    calls, conds, text_off = _traced(step, monkeypatch, **off)
    assert calls == [_ROWS] and not conds
    # Off, the fast route is never entered: the program is the routed path's.
    monkeypatch.setattr(qsa_rpa, "_zero_prefix_fast_route", None)
    assert _traced(step, monkeypatch, **off)[2] == text_off
    monkeypatch.undo()
    # On: one prefill trace (the walk is not duplicated), one cond around the
    # row movement and one around the merge, both on prefix == 0.
    on = dict(off, VLLM_TPU_QSA_ZERO_PREFIX_FAST=True)
    calls, conds, text_on = _traced(step, monkeypatch, **on)
    assert calls == [_ROWS] and len(conds) == 2 and text_on != text_off
    for prefix_slice in (False, True):
        seen = []
        real = qsa_rpa._zero_prefix_fast_route

        def spy(*args, _seen=seen, _real=real, **kwargs):
            _seen.append(kwargs["capacity"])
            return _real(*args, **kwargs)

        monkeypatch.setattr(qsa_rpa, "_zero_prefix_fast_route", spy)
        _traced(step, monkeypatch, **dict(on, VLLM_TPU_QSA_PREFIX_SLICE=prefix_slice))
        assert seen == [_SLOTS]
        # Decode-only steps (no more rows than the RPA capacity) never route.
        seen.clear()
        _traced(tuple(x[:_SLOTS] if i < 5 else x for i, x in enumerate(step)), monkeypatch, **on)
        assert seen == []
        monkeypatch.setattr(qsa_rpa, "_zero_prefix_fast_route", real)


def test_zero_prefix_fast_is_a_compile_key(monkeypatch):
    """The flag changes the traced program, so it must key the compile caches:
    an env entry that tpu_compiler's ignore set does not list (read from the
    source: importing tpu_compiler needs torch_tpu)."""
    import ast
    import pathlib

    from vllm_torchtpu import envs

    path = pathlib.Path(qsa_rpa.__file__).parents[2] / "compilation" / "tpu_compiler.py"
    ignored = None
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "_TPU_COMPILE_ENV_IGNORED" for t in node.targets
        ):
            ignored = ast.literal_eval(node.value)
    assert ignored and "VLLM_TPU_QSA_PREFIX_SLICE" not in ignored
    assert "VLLM_TPU_QSA_ZERO_PREFIX_FAST" not in ignored
    assert "VLLM_TPU_QSA_ZERO_PREFIX_FAST" in envs.environment_variables
    # q38 BLSTACK-SEED (rw2g, Benliang Wang's tree): earlier tests' monkeypatch.setattr(envs, ...) undo leaves a real
    # module attribute (their oldval was the lazily read value), which shadows envs.__getattr__ for the rest of the
    # session on this tree; drop it so the registry entry itself is read.
    envs.__dict__.pop("VLLM_TPU_QSA_ZERO_PREFIX_FAST", None)
    monkeypatch.setenv("VLLM_TPU_QSA_ZERO_PREFIX_FAST", "1")
    assert envs.VLLM_TPU_QSA_ZERO_PREFIX_FAST is True
    monkeypatch.delenv("VLLM_TPU_QSA_ZERO_PREFIX_FAST")
    assert envs.VLLM_TPU_QSA_ZERO_PREFIX_FAST is False
