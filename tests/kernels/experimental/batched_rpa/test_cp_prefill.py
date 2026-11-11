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
"""CPU contract tests for the zigzag context-parallel prefill plan."""

import functools
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs,
    kernel,
    schedule,
    wrapper,
)
from vllm_torchtpu.kernels.experimental.batched_rpa import cp_prefill as C

P = pytest.mark.parametrize
FP8, BF16, F32, I32 = jnp.float8_e4m3fn, jnp.bfloat16, jnp.float32, jnp.int32
PAGE = 256
SDS = jax.ShapeDtypeStruct
TPU7X = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
H_SUB, S_LANE = configs.KVLayout.HEAD_ALONG_SUBLANE, configs.KVLayout.SEQ_ALONG_LANE
LAYOUTS, WORLDS = [H_SUB, S_LANE], [2, 8]


def _i32(*xs):
    return jnp.array(list(xs), I32)


def _plan(mult, world=8):
    return C.make_zigzag_plan(16 * mult * PAGE, world=world, page_size=PAGE)


def _kv(n, h, d=128, seed=0):
    r = np.random.default_rng(seed)
    return jnp.asarray(r.standard_normal((n, h, d), np.float32), FP8), jnp.asarray(
        r.standard_normal((n, h, d), np.float32), FP8
    )


def _rank_slices(plan, rank):
    p = plan.piece_tokens
    return [(int(i) * p, (int(i) + 1) * p) for i in plan.rank_pieces(rank)]


def _rank_rows(x, plan, rank):
    return jnp.concatenate([x[s:e] for s, e in _rank_slices(plan, rank)])


def _bytes(x):
    return np.asarray(x).view(np.uint8)


def _eq(a, b):
    np.testing.assert_array_equal(_bytes(a), _bytes(b))


def _int8(x):
    return jax.lax.bitcast_convert_type(x, jnp.int8)


def _id_view(cache):
    return cache, 1, lambda x: x


def _bt(n_cache, n_pages, seed):
    return jnp.asarray(np.random.default_rng(seed).permutation(n_cache)[:n_pages], I32)


def _gather_pages(k, v, p):
    return jnp.concatenate(
        [
            C.pack_kv_pages(_rank_rows(k, p, r), _rank_rows(v, p, r), PAGE)
            for r in range(p.world)
        ]
    )


def _gather_rows(k, v, p):
    return jnp.concatenate(
        [
            C.pack_kv_rows(_rank_rows(k, p, r), _rank_rows(v, p, r))
            for r in range(p.world)
        ]
    )


def _gather_sliding(rows, p, sl, cfg):
    return jnp.concatenate(
        [
            C.cp_sliding_exchange_rows(*rows[r], sl, _i32(r), PAGE, cfg=cfg)
            for r in range(p.world)
        ]
    )


def _new_kv(k, v, layout=H_SUB):
    with use_abstract_mesh(TPU7X):
        return wrapper.prepare_inputs(
            jnp.zeros(k.shape, BF16), k, v, BF16, FP8, kv_layout=layout, page_size=PAGE
        )[1]


def _serving_pages(k, v, layout):
    n, nk = k.shape[0], _new_kv(k, v, layout)
    if layout == S_LANE:
        return nk.reshape(*nk.shape[:-1], n // PAGE, PAGE).transpose(3, 0, 1, 2, 4)
    return nk.reshape(n // PAGE, PAGE, *nk.shape[1:])


def _win_len(p, where):
    t = p.piece_tokens
    return {
        "end": p.num_tokens,
        "two_ranks": (p.num_pieces - 1) * t + 100,
        "one_rank": p.world * t + 100,
        "short": 700,
    }[where]


def _cfg(nkv, window=None, layout=H_SUB, world=8, **kw):
    c = C.LayerConfig(
        world, 2 * nkv, nkv, 128, window, 1.0, None, 1.0, 1.0, False, kv_layout=layout
    )
    return replace(c, **kw) if kw else c


def _full(cache, *a, **kw):
    with pltpu.force_tpu_interpret_mode():
        return C.cp_full_attention(cache, *a, cache_view=_id_view, **kw)


def _sliding(cache, *a, **kw):
    with pltpu.force_tpu_interpret_mode():
        return C.cp_sliding_attention(cache, *a, cache_view=_id_view, **kw)


def _write_heads(base, g, first, writes, n=1, layout=S_LANE, rot=0):
    return C.write_kv_head_pages(
        base,
        g,
        I32(first),
        *writes,
        num_heads=n,
        kv_layout=layout,
        shared_kv_rotary_dims=rot,
    )


def _assert_cache(cache, written, expected):
    _eq(cache[written], expected)
    untouched = np.setdiff1d(np.arange(cache.shape[0]), written)
    assert (np.asarray(cache[untouched]).astype(np.float32) == 7).all()


@P("world", WORLDS)
def test_zigzag_plan_covers_the_prompt_and_balances_causal_pairs(world):
    plan, piece = C.make_zigzag_plan(81920, world, PAGE), 81920 // (2 * world)
    assert plan.piece_tokens == piece
    covered = sorted(r for rank in range(world) for r in _rank_slices(plan, rank))
    assert covered == [(i * piece, (i + 1) * piece) for i in range(2 * world)]
    pairs = [
        sum((s + e + 1) * (e - s) // 2 for s, e in _rank_slices(plan, r))
        for r in range(world)
    ]
    assert max(pairs) == min(pairs)


def test_zigzag_plan_rejects_unaligned_token_counts():
    with pytest.raises(ValueError, match="multiple of"):
        C.make_zigzag_plan(80128, 8, PAGE)


@P("world", WORLDS)
def test_gathered_page_table_finds_every_position(world):
    plan = _plan(2, world)
    k, v = _kv(plan.num_tokens, 4)
    table = plan.gathered_page_table()
    _eq(_gather_pages(k, v, plan)[table], C.pack_kv_pages(k, v, PAGE))
    assert set(table.tolist()) == set(range(plan.num_pages))


@P("world", WORLDS)
def test_traced_rank_gives_the_python_rank_metadata(world):
    p = _plan(8, world)
    fn = lambda r: (p.sub_request_metadata(r), p.sliding_page_tables(r, 4))  # noqa: E731
    traced = jax.jit(fn)
    for r in range(p.world):
        for g, w in zip(jax.tree.leaves(traced(r)), jax.tree.leaves(fn(r))):
            np.testing.assert_array_equal(g, w)


@P("halo", [0, 3])
def test_sliding_page_tables_refuse_halos_outside_one_piece(halo):
    with pytest.raises(ValueError, match="halo"):
        _plan(2).sliding_page_tables(0, halo)


def test_sliding_exchange_refuses_pieces_shorter_than_the_halo():
    p = _plan(2)
    z = jnp.zeros((2 * p.piece_tokens, 4, 128), FP8)
    with pytest.raises(ValueError, match="halo"):
        C.cp_sliding_exchange_rows(
            z, z, _i32(p.num_pages * PAGE), _i32(0), PAGE, cfg=_cfg(4, 1024)
        )


def test_layer_config_rpa_kwargs_and_plan():
    plan, base = C.make_zigzag_plan(81920, 8, PAGE), _cfg(16, 1024)
    assert base.plan(2 * plan.piece_tokens, PAGE) == plan
    for f in ("fence_output", "unmasked_interior"):
        assert replace(base, **{f: True}).rpa_kwargs()[f] is True
        assert f not in replace(base, **{f: False}).rpa_kwargs()
    assert replace(base, fp8_out_rows=True).rpa_kwargs()["output_quant_dtype"] == FP8
    assert "output_quant_dtype" not in base.rpa_kwargs()


def test_sub_request_metadata_places_queries_at_absolute_positions():
    cu, kv, dist = C.make_zigzag_plan(81920, 8, PAGE).sub_request_metadata(2)
    np.testing.assert_array_equal(cu, [0, 5120, 10240])
    np.testing.assert_array_equal(kv, [3 * 5120, 14 * 5120])
    np.testing.assert_array_equal(dist, [0, 0, 2])


_TP_CASES = [
    (4, 8, [(r // 2, 1) for r in range(8)]),
    (16, 8, [(2 * r, 2) for r in range(8)]),
    (2, 2, [(0, 1), (1, 1)]),
    (8, 2, [(0, 4), (4, 4)]),
]


@P("nkv,w,exp", _TP_CASES)
def test_tp_kv_heads_follow_tensor_parallel_sharding(nkv, w, exp):
    assert [C.tp_kv_heads(nkv, w, r) for r in range(w)] == exp


@P("nkv,layout", [(2, H_SUB), (4, H_SUB), (16, H_SUB), (2, S_LANE), (4, S_LANE)])
@P("world", WORLDS)
def test_head_page_writes_cache_what_slot_slices_cache(nkv, layout, world):
    if layout == S_LANE and C.tp_kv_heads(nkv, world, 0)[1] != 1:
        pytest.skip("SEQ_ALONG_LANE head pages hold one head per rank")
    p = _plan(2, world)
    k, v = _kv(p.num_tokens, nkv, seed=10)
    g = _gather_pages(k, v, p)
    sl = I32(p.num_tokens - PAGE - 44)
    wr = C.full_attention_page_writes(p, _bt(48, p.num_pages, 11), sl)
    for r in range(world):
        f, nh = C.tp_kv_heads(nkv, world, r)
        mine = C.cache_pages_from_slots(C.kv_head_slots(g, f, nh), layout)
        base = jnp.full((48, *mine.shape[1:]), 7, FP8)
        with pltpu.force_tpu_interpret_mode():
            _eq(
                _write_heads(base, g, f, wr, nh, layout),
                C.write_kv_pages(base, mine, *wr),
            )


@P("world", WORLDS)
@P("layout", LAYOUTS)
@P("hpw", [False, True])
def test_cp_full_attention_caches_rank_heads(monkeypatch, layout, world, hpw):
    def stub(q, p, *a, return_kv_pages=False, **kw):
        return (q, p) if return_kv_pages else q

    monkeypatch.setattr(C, "cp_prefill_attention", stub)
    p, nkv, seq_len = _plan(2, world), 2 if hpw else 4, 16 * 2 * PAGE - 300
    k, v = _kv(p.num_tokens, nkv, seed=6)
    g, bt = _int8(_gather_rows(k, v, p)), _bt(48, p.num_pages + 4, 7)
    bts, sl = jnp.concatenate([bt, jnp.zeros_like(bt)]), _i32(seq_len, 0)
    written = np.asarray(bt[: -(-seq_len // PAGE)])
    cfg = _cfg(nkv, None, layout, world, head_page_writes=hpw)
    q = jnp.zeros((2 * p.piece_tokens, cfg.num_heads, 128), BF16)
    for r in sorted({0, 5 % world, world - 1}):
        f, nh = C.tp_kv_heads(nkv, world, r)
        exp = _serving_pages(k[:, f : f + nh], v[:, f : f + nh], layout)
        c0 = jnp.full((48, *exp.shape[1:]), 7, FP8)
        cache, _ = _full(c0, q, g, sl, bts, _i32(r), cfg=cfg)
        _assert_cache(cache, written, exp[: len(written)])


def test_head_page_writes_refuse_two_heads_of_a_seq_along_lane_cache():
    with pytest.raises(NotImplementedError, match="SEQ_ALONG_LANE"):
        _cfg(16, None, S_LANE, head_page_writes=True)


@P("world", WORLDS)
@P("layout", LAYOUTS)
@P("where", ["end", "two_ranks", "one_rank", "short"])
def test_sliding_layers_cache_the_decode_window(monkeypatch, where, layout, world):
    att = []
    stub = lambda q, p, pt, *a, **kw: att.append((p, pt)) or q  # noqa: E731
    monkeypatch.setattr(C, "cp_prefill_attention", stub)
    win, nkv, n_cp, p = 1024, 16, 160, _plan(8, world)
    seq_len = _win_len(p, where)
    k, v = _kv(p.num_tokens, nkv, seed=8)
    in_order, bt, sl = (
        C.pack_kv_pages(k, v, PAGE),
        _bt(n_cp, p.num_pages, 9),
        _i32(seq_len),
    )
    q = jnp.zeros((2 * p.piece_tokens, 32, 128), BF16)
    rows = [(_rank_rows(k, p, r), _rank_rows(v, p, r)) for r in range(world)]
    cfg = _cfg(nkv, win, layout, world)
    g = _gather_sliding(rows, p, sl, cfg)
    pos_p, cnt = C.window_position_pages(I32(seq_len), win, PAGE)
    needed, al = np.asarray(pos_p)[: int(cnt)], C.window_page_align(win, PAGE)
    p0 = ((max(seq_len - win + 1, 0) // PAGE) // al) * al
    np.testing.assert_array_equal(needed, np.arange(p0, -(-seq_len // PAGE)))
    halo = C.num_halo_pages(win, PAGE)
    by_rank, ht, pp = g.reshape(world, -1, 2 * nkv * 128), halo * PAGE, p.piece_pages
    for r in range(world):
        rp = p.rank_pieces(r)
        for slot, src, pc in ((0, (r - 1) % world, rp[0]), (1, (r + 1) % world, rp[1])):
            if pc and not (r == world - 1 and slot == 1):
                a = by_rank[src, slot * ht : (slot + 1) * ht].reshape(-1)
                _eq(a, in_order[pc * pp - halo : pc * pp].reshape(-1))
        f, nh = C.tp_kv_heads(nkv, world, r)
        exp = _serving_pages(k[:, f : f + nh], v[:, f : f + nh], layout)
        c0 = jnp.full((n_cp, *exp.shape[1:]), 7, FP8)
        cache, _ = _sliding(c0, q, *rows[r], g, sl, bt, _i32(r), cfg=cfg)
        _assert_cache(cache, np.asarray(bt[needed]), exp[needed])
        loc, tbls = att[-1]
        tbls = np.asarray(jnp.broadcast_to(tbls, (2, p.num_pages)))
        for sub, pc in enumerate(rp):
            seen = np.arange(max(pc * pp - halo, 0), pc * pp)
            _eq(loc[tbls[sub][seen]], in_order[seen])


_HP_CASES = [
    (2, 512, H_SUB, 0),
    (16, 256, H_SUB, 0),
    (2, 512, S_LANE, 0),
    (8, 512, S_LANE, 64),
]


@P("nkv,hd,layout,rot", _HP_CASES)
def test_head_page_writes_lower_for_tpu7x(nkv, hd, layout, rot):
    f, nh = C.tp_kv_heads(nkv, 8, 3)
    pg = (2, hd // 4, 4, PAGE) if layout == S_LANE else (PAGE, max(nh // 2, 1), 4, hd)
    with use_abstract_mesh(TPU7X):
        fn = lambda c, p, s, d: _write_heads(c, p, f, (s, d, I32(40)), nh, layout, rot)  # noqa: E731
        c_s, p_s = SDS((128, *pg), FP8), SDS((64, PAGE, nkv // 2, 4, hd), FP8)
        t = jax.jit(fn).trace(c_s, p_s, SDS((64,), I32), SDS((64,), I32))
        assert "tpu_custom_call" in t.lower(lowering_platforms=("tpu",)).as_text()


def test_head_page_writes_refuse_an_odd_head_count_above_one():
    with pytest.raises(NotImplementedError, match="even number"):
        _cfg(24, head_page_writes=True)


def test_new_kv_rows_are_pack_kv_rows():
    k, v = _kv(64, 4, seed=20)
    _eq(_new_kv(k, v).reshape(64, -1), C.pack_kv_rows(k, v))


def _qkv_inputs(rows, nh, nkv, hd, seed):
    key = jax.random.key(seed)
    qkv = (jax.random.normal(key, (rows, (nh + 2 * nkv) * hd)) * 2).astype(BF16)
    inv = 10000.0 ** (jnp.arange(hd // 2) / (hd // 2))
    ang = jnp.arange(rows)[:, None] / inv[None]
    ws = [
        (1 + 0.1 * jax.random.normal(jax.random.fold_in(key, i), (hd,))).astype(BF16)
        for i in (1, 2)
    ]
    return qkv, jnp.cos(ang).astype(BF16), jnp.sin(ang).astype(BF16), ws[0], ws[1]


def _attention_stub(calls, pages_seen=None):
    def attention(q, pages, *args, return_kv_pages=False, **kwargs):
        calls.append(kwargs.get("prepared_inputs"))
        if pages_seen is not None:
            pages_seen.append(pages)
        return (q, pages) if return_kv_pages else q

    return attention


@P("world", WORLDS)
@P("where", ["end", "two_ranks"])
def test_sliding_exchange_rows_sources_match(where, world):
    nkv, win, p = 8, 1024, _plan(8, world)
    k, v = _kv(p.num_tokens, nkv, seed=21)
    b_cfg = _cfg(nkv, win, H_SUB, world)
    q_cfg = replace(b_cfg, prologue_eps=1e-6, exchange_from_qkv=True)
    p_cfg = replace(q_cfg, paged_gather=True)
    rows, sl = 2 * p.piece_tokens, _i32(_win_len(p, where))
    ex = C.cp_sliding_exchange_rows
    for r in sorted({0, 3 % world, world - 1}):
        kr, vr, args = _rank_rows(k, p, r), _rank_rows(v, p, r), (sl, _i32(r), PAGE)
        _eq(
            ex(_new_kv(kr, vr), *args, cfg=b_cfg, source="prepared"),
            ex(kr, vr, *args, cfg=b_cfg, source="kv"),
        )
        qi = _qkv_inputs(rows, 2 * nkv, nkv, 128, r)
        with pltpu.force_tpu_interpret_mode():
            _, nk = C.qkv_prologue(*qi, cfg=q_cfg)
            want = ex(nk, *args, cfg=q_cfg, source="prepared")
            got, got_p = (
                ex(*qi, *args, cfg=q_cfg, source="qkv"),
                ex(*qi, *args, cfg=p_cfg, source="qkv"),
            )
        _eq(got, want)
        assert got_p.shape == (want.shape[0], *nk.shape[1:])
        _eq(got_p.reshape(want.shape), want)


@P("hpw", [False, True])
@P("pg", [False, True])
def test_cp_full_attention_from_q_and_q8(monkeypatch, hpw, pg):
    calls = []
    monkeypatch.setattr(C, "cp_prefill_attention", _attention_stub(calls))
    p, nkv, seq_len = _plan(2), 2, 16 * 2 * PAGE - 300
    k, v = _kv(p.num_tokens, nkv, seed=22)
    nks = [_new_kv(_rank_rows(k, p, r), _rank_rows(v, p, r)) for r in range(8)]
    g = _int8(jnp.concatenate(nks).reshape(p.num_tokens, -1))
    bt = _bt(48, p.num_pages + 4, 23)
    bts, sl, rows = (
        jnp.concatenate([bt, jnp.zeros_like(bt)]),
        _i32(seq_len, 0),
        2 * p.piece_tokens,
    )
    q_hbm = jnp.ones((nkv, rows, 2, 2, 128), BF16)
    q_fp8, cfg = (
        wrapper.quantize_queries(q_hbm, FP8),
        _cfg(nkv, num_heads=8, head_page_writes=hpw),
    )
    for r in (0, 6):
        args, cache = (sl, bts, _i32(r)), jnp.full((48, PAGE, 1, 4, 128), 7, FP8)
        g_arg = g.reshape(p.num_tokens, 1, 4, 128) if pg else g
        want, _ = _full(cache, jnp.zeros((rows, 8, 128), BF16), g, *args, cfg=cfg)
        got, out = _full(
            cache, q_hbm, g_arg, *args, cfg=replace(cfg, paged_gather=pg), from_q=True
        )
        _eq(got, want)
        assert out.shape == (rows, 8, 128) and calls[-1][0] is q_hbm
        assert calls[-1][1].shape == (rows, 1, 4, 128)
        if hpw and not pg:
            kw8 = dict(fp8_compute=True, prologue_eps=1e-6, prologue_dma_stores=True)
            cfg8 = replace(cfg, prologue_fp8_queries=True, **kw8)
            got8, out8 = _full(cache, q_hbm, q_fp8, g, *args, cfg=cfg8, from_q8=True)
            _eq(got8, want)
            assert out8.shape == (rows, 8, 128) and calls[-1][2] is q_fp8


@P("world", WORLDS)
@P("pg", [False, True])
@P("ex_qkv", [False, True])
def test_sliding_attention_layer_from_q_caches_like_from_q_k_v(
    monkeypatch, ex_qkv, pg, world
):
    calls, pages_seen = [], []
    monkeypatch.setattr(C, "cp_prefill_attention", _attention_stub(calls, pages_seen))
    win, nkv, n_cp, p = 1024, 16, 160, _plan(8, world)
    k, v = _kv(p.num_tokens, nkv, seed=24)
    bt, sl = _bt(n_cp, p.num_pages, 25), _i32(_win_len(p, "two_ranks"))
    rows = [(_rank_rows(k, p, r), _rank_rows(v, p, r)) for r in range(world)]
    cfg = _cfg(nkv, win, H_SUB, world)
    g, nr = _gather_sliding(rows, p, sl, cfg), 2 * p.piece_tokens
    q_hbm = jnp.ones((nkv, nr, 1, 2, 128), BF16)
    wr = -(-2 * C.tp_kv_heads(nkv, world, 0)[1] // 4)
    for r in (0, world - 1):
        meta, cache = (sl, bt, _i32(r)), jnp.full((n_cp, PAGE, wr, 4, 128), 7, FP8)
        nk, g_arg = _new_kv(*rows[r]), g.reshape(g.shape[0], 8, 4, 128) if pg else g
        want, _ = _sliding(
            cache, jnp.zeros((nr, 32, 128), BF16), *rows[r], g, *meta, cfg=cfg
        )
        loc_w, cfg2 = calls[-1], replace(cfg, exchange_from_qkv=ex_qkv, paged_gather=pg)
        got, out = _sliding(cache, q_hbm, nk, g_arg, *meta, cfg=cfg2, from_q=True)
        _eq(got, want)
        _eq(pages_seen[-1], pages_seen[-2])
        assert out.shape == (nr, 32, 128) and loc_w is None and calls[-1] == (q_hbm, nk)


def test_cp_full_attention_fp8_rows_returns_the_o_proj_input(monkeypatch):
    from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor

    p = _plan(2)
    nkv, rows = 2, 2 * p.piece_tokens
    att = jax.random.normal(jax.random.PRNGKey(30), (rows, 4, 128)).astype(BF16)

    def stub(q, pg, *a, return_kv_pages=False, **kw):
        o = quantize_tensor(att.reshape(rows, -1), kw["output_quant_dtype"])
        return (o, pg) if return_kv_pages else o

    monkeypatch.setattr(C, "cp_prefill_attention", stub)
    k, v = _kv(p.num_tokens, nkv, seed=31)
    c0, q0 = jnp.full((48, PAGE, 1, 4, 128), 7, FP8), jnp.zeros((rows, 4, 128), BF16)
    bt, cfg = _bt(48, p.num_pages, 32), _cfg(nkv, fp8_out_rows=True)
    _, got_r, got_s = _full(
        c0, q0, _int8(_gather_rows(k, v, p)), _i32(p.num_tokens), bt, _i32(3), cfg=cfg
    )
    want_r, want_s = quantize_tensor(att.reshape(rows, -1), FP8)
    _eq(got_r, want_r)
    np.testing.assert_array_equal(np.asarray(got_s), np.asarray(want_s))


_LOW_CASES = [
    ((16, 2), 512, None, False, False),
    ((16, 8), 256, 1024, False, False),
    ((16, 2), 512, None, True, False),
    ((16, 8), 256, 1024, True, False),
    ((16, 2), 512, None, True, True),
]


@P("world", WORLDS)
@P("heads,hd,win,fp8_out,fq8", _LOW_CASES)
def test_cp_layers_lower_for_tpu7x(heads, hd, win, fp8_out, fq8, world):
    p, (nq, nkv) = _plan(8, world), heads
    rows, slots, grp = 2 * p.piece_tokens, nkv // 2, nq // nkv
    kw0 = dict(
        num_heads=nq, head_dim=hd, fp8_compute=win is None, head_page_writes=win is None
    )
    kw1 = dict(
        prologue_eps=1e-6,
        prologue_dma_stores=fq8,
        prologue_fp8_queries=fq8,
        fp8_out_rows=fp8_out,
    )
    cfg = _cfg(nkv, win, H_SUB, world, **kw0, **kw1)
    with use_abstract_mesh(TPU7X):
        q_hbm = SDS((nkv, rows, grp // 2, 2, hd), BF16)
        cache = SDS((64, PAGE, -(-2 * cfg.local_kv_heads // 4), 4, hd), FP8)
        meta = (SDS((1,), I32), SDS((p.num_pages,), I32), SDS((1,), I32))
        w, kw = 2 * nkv * hd, dict(cfg=cfg, cache_view=_id_view)
        if win is None:
            qa = (q_hbm, SDS((nkv, rows, grp // 4, 4, hd), FP8)) if fq8 else (q_hbm,)
            fn = functools.partial(
                C.cp_full_attention,
                **({"from_q8": True} if fq8 else {"from_q": True}),
                **kw,
            )
            tr = jax.jit(fn).trace(cache, *qa, SDS((p.num_tokens, w), jnp.int8), *meta)
        else:
            ex = (
                3 * C.num_halo_pages(win, PAGE) + C.window_page_align(win, PAGE)
            ) * PAGE
            fn = functools.partial(C.cp_sliding_attention, from_q=True, **kw)
            tr = jax.jit(fn).trace(
                cache,
                q_hbm,
                SDS((rows, slots, 4, hd), FP8),
                SDS((world * ex, w), jnp.int8),
                *meta,
            )
        text = tr.lower(lowering_platforms=("tpu",)).as_text()
    assert "RPAm-snh-p256" in text and (not fp8_out or "rpa_output_quantize" in text)


def test_sliding_exchange_rows_from_qkv_lower_for_tpu7x():
    p = C.make_zigzag_plan(122880, 8, PAGE)
    cfg = _cfg(
        8, 1024, num_heads=16, head_dim=256, prologue_eps=1e-6, exchange_from_qkv=True
    )
    r = 2 * p.piece_tokens
    with use_abstract_mesh(TPU7X):
        a = [
            SDS((r, 8192), BF16),
            SDS((r, 128), BF16),
            SDS((r, 128), BF16),
            SDS((256,), BF16),
            SDS((256,), BF16),
        ]
        fn = functools.partial(
            C.cp_sliding_exchange_rows, page_size=PAGE, cfg=cfg, source="qkv"
        )
        lo = (
            jax.jit(fn)
            .trace(*a, SDS((1,), I32), SDS((1,), I32))
            .lower(lowering_platforms=("tpu",))
        )
    assert (
        lo.out_info.shape == (14 * PAGE, 2 * 8 * 256)
        and "tpu_custom_call" in lo.as_text()
    )


def test_prologue_dma_stores_need_a_prologue_layer():
    with pytest.raises(ValueError, match="prologue_eps"):
        _cfg(2, prologue_dma_stores=True)


@P("stores,fp8_c,win", [(False, True, None), (True, False, None), (True, True, 1024)])
def test_prologue_fp8_queries_need_full_attention_fp8_dma_stores(stores, fp8_c, win):
    kw = dict(num_heads=8, fp8_compute=fp8_c, prologue_eps=1e-6)
    with pytest.raises(ValueError, match="prologue_fp8_queries"):
        _cfg(2, win, prologue_dma_stores=stores, prologue_fp8_queries=True, **kw)


@P("fp8_c", [False, True])
def test_qkv_prologue_dma_stores_write_the_same_bits(monkeypatch, fp8_c):
    fn = functools.partial(C.prologue.qkv_to_rpa_inputs, interpret=True)
    monkeypatch.setattr(C.prologue, "qkv_to_rpa_inputs", fn)
    cfg = _cfg(2, num_heads=8, fp8_compute=fp8_c, prologue_eps=1e-6)
    cfg_s = replace(cfg, prologue_dma_stores=True, prologue_fp8_queries=fp8_c)
    args = _qkv_inputs(512, 8, 2, 128, 40)
    want, got = C.qkv_prologue(*args, cfg=cfg), C.qkv_prologue(*args, cfg=cfg_s)
    assert len(want) == 2 and len(got) == (3 if fp8_c else 2)
    for g, w in zip(got, want):
        _eq(g, w)
    if fp8_c:
        _eq(got[2], wrapper.quantize_queries(want[0], FP8))


def _shared_kv_pages(std, rot):
    k, v = std[..., 0, :, :, :], std[..., 1, :, :, :]
    g, r = k.shape[-3], rot // 4
    z = jnp.zeros_like(k[..., : g - 2 * r, :, :])
    rot_k = jnp.concatenate(
        [k[..., :r, :, :], k[..., g // 2 : g // 2 + r, :, :], z], -3
    )
    return jnp.stack([v, rot_k], axis=-4)


@P("world", [2, 8])
def test_shared_kv_head_pages_cache_the_value_and_rotated_key(world):
    p = C.make_zigzag_plan(16 * world * PAGE, world, PAGE)
    k, v = _kv(p.num_tokens, world, seed=20)
    g = _gather_pages(k, v, p)
    sl = I32(p.num_tokens - PAGE - 44)
    wr = C.full_attention_page_writes(p, _bt(48, p.num_pages, 21), sl)
    for r in range(world):
        f, nh = C.tp_kv_heads(world, world, r)
        base = jnp.full((48, 2, 32, 4, PAGE), 7, FP8)
        with pltpu.force_tpu_interpret_mode():
            std = _write_heads(base, g, f, wr, nh)
            got = _write_heads(base, g, f, wr, nh, rot=32)
        written = np.unique(np.asarray(wr[1])[: int(wr[2])])
        want = np.asarray(_bytes(std)).copy()
        want[written] = _bytes(_shared_kv_pages(std[written], 32))
        np.testing.assert_array_equal(_bytes(got), want)


def test_shared_kv_pages_need_the_head_page_kernel():
    with pytest.raises(NotImplementedError, match="head_page_writes=True"):
        _cfg(8, None, S_LANE, shared_kv_rotary_dims=32)
    z, idx = jnp.zeros((4, PAGE, 1, 4, 128), FP8), jnp.zeros((4,), I32)
    with pytest.raises(NotImplementedError, match="SEQ_ALONG_LANE"):
        _write_heads(z, z, 0, (idx, idx, I32(0)), 1, H_SUB, 32)


def _page_write_pair(layout, seed, src, dst, count, rot=0):
    k, v = _kv(4 * PAGE, 2, seed=seed)
    g = C.pack_kv_pages(k, v, PAGE)
    mine = C.cache_pages_from_slots(C.kv_head_slots(g, 0, 1), layout)
    sen = jnp.full((8, *mine.shape[1:]), 7, FP8)
    with pltpu.force_tpu_interpret_mode():
        ap = C.write_kv_pages(sen, mine, src, dst, I32(count))
        ah = _write_heads(sen, g, 0, (src, dst, I32(count)), 1, layout, rot)
    return sen, mine, ap, ah


@P("layout", LAYOUTS)
def test_zero_count_writes_preserve_block_zero(layout):
    z = jnp.zeros((4,), I32)
    sen, _, ap, ah = _page_write_pair(
        layout, 99, z, z, 0, 32 if layout == S_LANE else 0
    )
    _eq(ap, sen)
    _eq(ah, sen)


@P("layout", LAYOUTS)
def test_page_writes_clamp_out_of_bounds_indices(layout):
    s, d = _i32(-3, 99, -9, 55), _i32(-4, 77, -8, 99)
    _, mine, ap, ah = _page_write_pair(layout, 101, s, d, 2)
    for arr in (ap, ah):
        _eq(arr[0], mine[0])
        _eq(arr[-1], mine[-1])
        assert (np.asarray(arr[1:-1]).astype(np.float32) == 7).all()


@P("layout", LAYOUTS)
def test_nan_kv_cache_sliding_cp_prefill_then_decode_step_is_finite(
    monkeypatch, layout
):
    world, win, seq_len, hd, nq, nkv, n_cp = 2, 1024, 16800, 128, 8, 2, 80
    p = C.make_zigzag_plan(18432, world, PAGE)
    k_all, v_all = _kv(p.num_tokens, nkv, hd, seed=7)
    bt = jnp.arange(n_cp - 1, n_cp - 1 - p.num_pages, -1, dtype=I32)
    sl, sm = _i32(seq_len), hd**-0.5
    cfg = _cfg(nkv, win, layout, world, num_heads=nq, sm_scale=sm)
    rows = [(_rank_rows(k_all, p, r), _rank_rows(v_all, p, r)) for r in range(world)]
    g = _gather_sliding(rows, p, sl, cfg)
    with use_abstract_mesh(TPU7X):
        kv_sh = wrapper.get_kv_cache_shape(n_cp, PAGE, 1, hd, FP8, kv_layout=layout)
    monkeypatch.setattr(C, "cp_prefill_attention", lambda q, *a, **kw: q)
    q0 = jnp.zeros((2 * p.piece_tokens, nq, hd), BF16)
    cache, _ = _sliding(
        jnp.full(kv_sh, jnp.nan, FP8), q0, *rows[0], g, sl, bt, _i32(0), cfg=cfg
    )
    assert jnp.isnan(cache[bt[:60]].astype(F32)).all()
    assert jnp.isfinite(cache[bt[60:66]].astype(F32)).all()

    rng = np.random.default_rng(19)
    q_dec = jnp.asarray(rng.standard_normal((1, 4, hd), np.float32), BF16)
    k_new = jnp.asarray(rng.standard_normal((1, 1, hd), np.float32), FP8)
    v_new = jnp.asarray(rng.standard_normal((1, 1, hd), np.float32), FP8)
    s0 = seq_len - win + 1
    k_win = jnp.concatenate([k_all[s0:seq_len, 0], k_new[:, 0]]).astype(F32)
    v_win = jnp.concatenate([v_all[s0:seq_len, 0], v_new[:, 0]]).astype(F32)
    ref = (jax.nn.softmax((q_dec[0].astype(F32) @ k_win.T) * sm, -1) @ v_win).astype(
        BF16
    )
    pad = ((0, 15), (0, 0), (0, 0))
    q_pad, k_pad, v_pad = jnp.pad(q_dec, pad), jnp.pad(k_new, pad), jnp.pad(v_new, pad)

    if jax.devices()[0].platform == "tpu":
        rpa_args = (
            q_pad,
            k_pad,
            v_pad,
            cache.copy(),
            _i32(seq_len + 1),
            bt,
            _i32(0, 1),
            _i32(1, 1, 1),
        )
        ot, _ = wrapper.ragged_paged_attention(
            *rpa_args, sm_scale=sm, sliding_window=win, kv_layout=layout
        )
        assert jnp.isfinite(ot[0]).all()
        np.testing.assert_allclose(
            np.asarray(ot[0], np.float32), np.asarray(ref, np.float32), atol=2e-2
        )

    bkv, rem = 2048, seq_len - 16384
    with use_abstract_mesh(TPU7X):
        mdl = configs.ModelConfigs(
            4, 1, hd, sliding_window=win, sm_scale=sm, mask_value=-1e30
        )
        srv = configs.ServingConfigs(
            1,
            PAGE,
            16,
            p.num_pages,
            dtype_q=BF16,
            dtype_kv=FP8,
            dtype_out=BF16,
            kv_layout=layout,
            mask_invalid_v=False,
        )
        cfgs = configs.RpaConfigs(
            configs.BlockSizes(1, 1, bkv, 1), mdl, srv, configs.RpaCase.DECODE, 64 << 20
        )
        q_hbm, nk = wrapper.prepare_inputs(
            q_pad, k_pad, v_pad, BF16, FP8, kv_layout=layout, page_size=PAGE
        )
        p0, p1, z0 = (
            cache[bt[56:64]],
            cache[bt[64:66]],
            jnp.zeros(cfgs.kv_vmem_shape, FP8),
        )
        nan0 = jnp.full(cfgs.kv_vmem_shape, jnp.nan, FP8)
        if layout == S_LANE:
            kv0 = z0.at[0, :, :, :, :bkv].set(
                p0.transpose(1, 2, 3, 0, 4).reshape(2, hd // 4, 4, bkv)
            )
            c1 = p1.transpose(1, 2, 3, 0, 4).reshape(2, hd // 4, 4, 2 * PAGE)
            kv1 = (
                nan0.at[0, :, :, :, :rem]
                .set(c1[:, :, :, :rem])
                .at[0, :, :, :, rem : rem + 1]
                .set(nk[:, :, :, :1])
            )
            s_cls = schedule.SeqAlongLaneDmaNew
        else:
            kv0 = z0.at[0, :bkv].set(p0.reshape(bkv, 1, 4, hd))
            kv1 = (
                nan0.at[0, :rem]
                .set(p1.reshape(2 * PAGE, 1, 4, hd)[:rem])
                .at[0, rem : rem + 1]
                .set(nk[:1])
            )
            s_cls = schedule.HeadAlongSublaneDmaNew
        st = schedule.RpaSchedule.create_shape_dtype(cfgs, dma_kv_new_struct_cls=s_cls)
        si = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), st)
        si = replace(
            si,
            k_idx=replace(si.k_idx, data=si.k_idx.data.at[0].set(7).at[1].set(8)),
            is_last_k=replace(si.is_last_k, data=si.is_last_k.data.at[1].set(1)),
        )
        vs = pl.BlockSpec(memory_space=pltpu.VMEM)
        ss = jax.tree_util.tree_map(lambda _: pl.BlockSpec(memory_space=pltpu.SMEM), st)

        def decode_steps(qr, kv0_r, kv1_r, sr, or_, mr, lr, ar):
            mr[...] = jnp.full(mr.shape, -jnp.inf, F32)
            lr[...], ar[...] = jnp.zeros(lr.shape, F32), jnp.zeros(ar.shape, F32)
            for s, kr, k0, nc, nn in (
                (0, kv0_r, 14336, bkv, 0),
                (1, kv1_r, 16384, rem, 1),
            ):
                sm_m = kernel.StitchMetadata([I32(nc)], [I32(nn)], [I32(0)])
                cm_m = kernel.ComputeMetadata([I32(k0 - seq_len)])
                pos_a = (I32(s), qr, kr, or_, None, sr, mr, lr, ar)
                kernel._attend_and_store(
                    *pos_a, stitch_meta=sm_m, step_meta=cm_m, cfgs=cfgs, masked=True
                )

        lm_s = SDS(cfgs.lm_scratch_shape, F32)
        osh = [
            SDS(cfgs.o_vmem_shape, BF16),
            lm_s,
            lm_s,
            SDS(cfgs.acc_scratch_shape, F32),
        ]
        fn = pl.pallas_call(
            decode_steps,
            in_specs=(vs, vs, vs, ss),
            out_specs=(vs,) * 4,
            out_shape=osh,
            interpret=True,
        )
        out = wrapper.prepare_outputs(fn(q_hbm[:, :1][None], kv0, kv1, si)[0][0])[0, 0]

    assert jnp.isfinite(out).all()
    np.testing.assert_allclose(
        np.asarray(out, np.float32), np.asarray(ref, np.float32), atol=2e-2
    )


def test_cp_sliding_cache_all_pages_writes_all_blocks_and_lowers(monkeypatch):
    p, nkv = _plan(4, 2), 8
    rows = 2 * p.piece_tokens
    seen_sw: list[int | None] = []
    monkeypatch.setattr(
        C,
        "cp_prefill_attention",
        lambda q, *a, **kw: (seen_sw.append(kw.get("sliding_window")), q)[1],
    )
    k, v = _kv(p.num_tokens, nkv, seed=41)
    bt = _bt(48, p.num_pages, 42)
    cfg = _cfg(nkv, 1024, S_LANE, 2, num_heads=16, head_dim=256, cache_all_pages=True)
    c0, q0 = jnp.full((48, 8, 64, 4, PAGE), 7, FP8), jnp.zeros((rows, 16, 256), BF16)
    with pltpu.force_tpu_interpret_mode():
        got, _ = C.cp_full_attention(
            c0,
            q0,
            _int8(_gather_rows(k, v, p)),
            _i32(p.num_tokens),
            bt,
            _i32(0),
            cfg=cfg,
            cache_view=_id_view,
        )
    assert seen_sw == [1024]
    written = np.asarray(bt)
    assert all(not np.array_equal(_bytes(got[idx]), _bytes(c0[idx])) for idx in written)
