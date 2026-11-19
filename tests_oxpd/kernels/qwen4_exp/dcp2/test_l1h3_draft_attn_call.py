# SPDX-License-Identifier: Apache-2.0
"""q38 L1-H3 (H3-DRAFT-ATTN-D14): the draft attention call (`megakernel_runtime.qwen38_draft_attn.
build_draft_attn_call`, kernel in Pallas interpret) against today's draft path on CPU, one fused-loop pass.

Today's path (DRAFT-1 on, VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE): per TP rank the projections (one XLA dot of the
block with the rank's [q|gate] / K / V columns, BF16), then `pooled_qsa_shared_noidx_step` (main-KV DCP 1) or
`pooled_qsa_shared_dcp_noidx_step` on the DCP group's gathered q heads + `qsa_rpa.dcp_combine` of the group's
partials for the rank's heads (DCP 2 / 4), attending the carried target selection (`selected_tokens` layout). Batched
RPA cannot run interpreted: decode attention is this directory's stand-in (`rpa_stand_in`; DCP partials via
`test_dcp2_op._reference_route`), the same oracle the DRAFT-1 tests use.

Rules (stated before data): (1) every live row's attention within 2 BF16 ulps of today's (or 1e-6 absolute, values near
0), elementwise, finite, idle rows 0; bitwise is not required: D14's union attention orders the online softmax by tiles (union_attn.py: "agree to
f32 rounding, not bit for bit"); the bitwise share is printed. (2) Each rank's pool after the pass: every byte
outside the rows this pass writes is unchanged and equal to today's; the written rows equal today's bytes (the K / V
projections are the kernel's BF16 dot; any FP8 byte difference is counted, printed and must be 0). Cases: a select-all
request, top-K requests with g_a listed (ctx % 4 == 3), a fresh group (ctx % 4 == 0), DCP owner changes, an idle seat.
"""

from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import dcp2_harness
from dcp2_harness import random_fp8
from jax.sharding import Mesh

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa, qsa_rpa
from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import selected_tokens
from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.models.qwen38_inputs import _rep, _rows
from vllm_torchtpu.megakernel.models.qwen38_pack import pack_weights
from vllm_torchtpu.megakernel.ops import BF16, F32, rnd
from vllm_torchtpu.megakernel.wholestep import qsa_draft as qd
from vllm_torchtpu.megakernel.wholestep.qsa_full import qkvg_width

TILES = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)
PB, HD, NH_RANK, RATIO, ROT, TOPK, MG = 128, 256, 3, 4, 64, 256, 1024
K_SCALE, V_SCALE, EPS = 0.375, 2.5, 1e-6
# Live requests (one token each) then one idle seat. 3071: g_a 767 complete + listed; 1024: fresh group, new page;
# 37: select-all; 1279 / 255: g_a complete at a page / owner edge.
CTX = [3071, 1024, 37, 1279, 255, 2046]  # 2046: a row whose partial group holds 3 tokens (ctx % 4 == 2)
LIVE = 6
SEATS = 7


def _route_any_dcp(monkeypatch):
    """`test_dcp2_op._reference_route` for a DCP group of any size: the harness's `reference_attend` (DCP 2 only: it asserts
    size == D == 2) with its owner rule `(t // B) % D` taken at the group's size (the stock writer's and reader's rule)."""
    original = qsa_rpa.decode_attention

    def route(*args, dcp=None, return_lse=False, **kw):
        if dcp is None:
            return original(*args, **kw)
        assert return_lse
        q, gate, blocks, lengths, table, cache = args[:6]
        size = int(dcp[1])
        monkeypatch.setattr(dcp2_harness, "D", size)
        try:
            out, lse = dcp2_harness.reference_attend(
                q, gate, np.asarray(blocks), np.asarray(lengths), np.asarray(table), cache, kv_heads=kw["kv_heads"],
                ratio=kw["ratio"], k_scale=kw.get("k_scale", 1.0), v_scale=kw.get("v_scale", 1.0),
                kv_head_offset=kw.get("kv_head_offset", 0), dcp=dcp,
                owner=lambda t, b: (np.asarray(t) // b) % size)
        finally:
            monkeypatch.setattr(dcp2_harness, "D", 2)
        return out, lse, cache

    monkeypatch.setattr(qsa_rpa, "decode_attention", route)


def _interpret_keep_params(monkeypatch):
    """This directory's `interpret` fixture (legacy interpret evaluator for every pallas_call, TPU7x info), except that a call
    that already asks for the TPU interpret mode (`pltpu.InterpretParams`: the megakernel, which needs its barrier semaphore and
    remote DMAs) keeps it; the fixture's blanket `interpret=True` breaks those (s6 j-c165daa3: no ir_type_handler for the barrier
    semaphore at DCP 1, remote-copy device ids at DCP 2 / 4)."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    original = pl.pallas_call

    def call(*args, **kwargs):
        if not isinstance(kwargs.get("interpret"), pltpu.InterpretParams):
            kwargs = {**kwargs, "interpret": True}
        return original(*args, **kwargs)

    monkeypatch.setattr(pl, "pallas_call", call)
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


def _ulp(x):
    a = np.abs(x).astype(np.float32)
    e = np.floor(np.log2(np.where(a > 0, a, 1e-30)))
    return np.where(a > 0, 2.0 ** (e - 7), 2.0**-133)


def _case(seed, R, D, main_slot):
    rng = np.random.default_rng(seed)
    g = dataclasses.replace(qwen38.TINY, NH=NH_RANK * R)
    if R > 2:
        g = dataclasses.replace(g, KH=R, VH=3 * R)
    n = SEATS
    ctx = np.zeros(n, np.int32)
    ctx[:LIVE] = CTX
    starts = np.concatenate([[0], np.cumsum([1] * LIVE + [0] * (n - LIVE))]).astype(np.int32)
    positions = np.zeros(n, np.int32)
    positions[:LIVE] = ctx[:LIVE]
    lens = np.where(np.arange(n) < LIVE, ctx + 1, 0)
    cols = int(-(-(lens.max() + 8) // (PB * D))) + 1
    pages = n * cols * D + 8
    table = (rng.permutation(pages - 1)[: n * cols] + 1).reshape(n, cols).astype(np.int32)
    pools = [random_fp8(rng, (pages, PB, 1, 4, HD)) for _ in range(R)]
    if D == 1:
        pools = [pools[0]] * R  # one KV head on every rank, whole pages
    # The carried target selection (taken at the previous position): every visible group where they fit, else
    # 64 groups incl. g_a when complete for the draft row.
    budget = TOPK // RATIO
    blocks = np.full((n, budget), -1, np.int32)
    for r in range(LIVE):
        groups = int(ctx[r]) // RATIO
        if groups <= budget:
            blocks[r, :groups] = np.arange(groups)
        else:
            pick = rng.permutation(groups)[:budget]
            ga = int(ctx[r]) // 4
            if (int(ctx[r]) + 1) % 4 == 0 and ga not in pick:
                pick[0] = min(ga, groups - 1)
            blocks[r] = pick
    shared = np.asarray(selected_tokens(jnp.asarray(blocks), jnp.asarray(ctx), TOPK, RATIO))
    nh = NH_RANK
    wq = [jnp.asarray(rng.normal(0, 0.06, (g.H, nh * 2 * HD)), BF16) for _ in range(R)]
    wkv = jnp.asarray(rng.normal(0, 0.06, (g.H, 2 * HD)), BF16)
    qkvg = jnp.stack([jnp.concatenate([w, wkv], axis=1) for w in wq])[:, None]  # [R, 1, H, W]
    nq = jnp.asarray(rng.normal(0, 0.05, (HD,)), F32)
    nk = jnp.asarray(rng.normal(0, 0.05, (HD,)), F32)
    inv = 1.0 / (g.theta ** (np.arange(0, ROT, 2) / ROT))
    ang = np.arange(8192)[:, None] * inv[None]
    trig = np.zeros((8192, 128), np.float32)
    trig[:, : ROT // 2], trig[:, ROT // 2 : ROT] = np.cos(ang), np.sin(ang)
    trig = jnp.asarray(trig, BF16)
    block = jnp.asarray(rng.normal(0, 0.5, (n, g.H)), BF16)
    return dict(g=g, n=n, ctx=ctx, starts=starts, positions=positions, table=table, pools=pools, shared=shared,
                qkvg=qkvg, nq=nq, nk=nk, trig=trig, block=block, main_slot=main_slot)


def _stock(c, R, D):
    """Today's draft pass per rank: `(attention [R, n, nh*HD], pools)`."""
    g, n, nh = c["g"], c["n"], NH_RANK
    y = [rnd(jnp.dot(c["block"], c["qkvg"][r, 0], preferred_element_type=F32)) for r in range(R)]
    qg = [x[:, : nh * 2 * HD] for x in y]
    key, value = y[0][:, -2 * HD : -HD], y[0][:, -HD:]
    common = dict(compress_ratio=RATIO, token_topk=TOPK, norm_eps=EPS, rotary_dim=ROT, k_scale=K_SCALE,
                  v_scale=V_SCALE, main_slot=c["main_slot"], max_query_len=1)
    args = lambda q, r: (q, key, value, c["block"], c["nq"], c["nk"], c["nq"][:128], c["nk"][:128],  # noqa: E731
                         jnp.asarray(c["positions"]), c["trig"], c["pools"][r], jnp.asarray(c["starts"]),
                         jnp.asarray(c["ctx"]), jnp.zeros((n,), jnp.int32), jnp.asarray(c["table"]),
                         jnp.zeros((n, 1), jnp.int32), jnp.asarray(c["shared"]))
    outs, pools = [], []
    if D == 1:
        for r in range(R):
            pool, out, _ = pooled_qsa.pooled_qsa_shared_noidx_step(*args(qg[r], r), **common)
            outs.append(np.asarray(out, np.float32).reshape(n, -1))
            pools.append(np.asarray(pool).view(np.uint8))
        return np.stack(outs), pools
    part = []
    for r in range(R):
        grp = [r // D * D + k for k in range(D)]
        qgrp = jnp.concatenate([qg[k] for k in grp], axis=-1)
        pool, out, lse, _ = pooled_qsa.pooled_qsa_shared_dcp_noidx_step(
            *args(qgrp, r), jnp.asarray([r % D], jnp.int32), dcp_size=D, **common)
        part.append((out, lse))
        pools.append(np.asarray(pool).view(np.uint8))
    for r in range(R):
        grp = [r // D * D + k for k in range(D)]
        gate = qg[r].reshape(n, nh, 2 * HD)[..., HD:]
        o = qsa_rpa.dcp_combine(jnp.stack([part[k][0] for k in grp]), jnp.stack([part[k][1] for k in grp]), gate,
                                jnp.asarray([r % D], jnp.int32))
        outs.append(np.asarray(o, np.float32).reshape(n, -1))
    return np.stack(outs), pools


def _kernel(c, R, D):
    from vllm_torchtpu.megakernel_runtime.qwen38_draft_attn import build_draft_attn_call, draft_indexed

    g, n = c["g"], c["n"]
    pps = int(c["table"].shape[1])
    fn, in_specs, out_specs, _ = build_draft_attn_call(
        Mesh(np.array(jax.devices()[:R]), ("tp",)), g, n, budget=TOPK // RATIO, max_groups=MG, pages_per_seq=pps,
        dcp=D, main_slot=c["main_slot"], k_scale=K_SCALE, v_scale=V_SCALE, norm_eps=EPS, rotary_dim=ROT,
        token_topk=TOPK, interpret=True, tiles=TILES)
    ex = ExecSpec(R=R, B=n, T=1, ctx=128, tiles=TILES, interpret=True)
    spec = qd.draft_attn_spec(g, R, budget=TOPK // RATIO, indexed=draft_indexed(MG, pps, D))
    w = pack_weights(spec, ex, {"qkvg": c["qkvg"]},
                     {"qsa_qn": _rows(_rep(c["nq"][None], R), (0,), 1), "qsa_kn": _rows(_rep(c["nk"][None], R), (0,), 1)})
    names = [x for x in kernel.make_step(Mesh(np.array(jax.devices()[:R]), ("tp",)), spec, ex)[1]["plan"].w_names]
    weights = [kernel.fold(w[k]) for k in names]
    fold = lambda xs: jnp.concatenate([jnp.asarray(x) for x in xs])  # noqa: E731
    out, pool = jax.jit(fn)(
        fold([c["block"]] * R), fold(c["pools"]), fold([c["shared"]] * R), jnp.asarray(c["table"]),
        jnp.asarray(c["positions"]), jnp.asarray(c["starts"]), jnp.asarray(c["ctx"]), c["trig"],
        jnp.asarray([r % D for r in range(R)], jnp.int32), c["nk"], *weights)
    out = np.asarray(out, np.float32).reshape(R, n, -1)
    pool = np.asarray(pool).view(np.uint8).reshape(R, *c["pools"][0].shape)
    return out, [pool[r] for r in range(R)]


@pytest.mark.parametrize("D,main_slot", [(1, 1), (2, 1), (4, 1), (4, -1)])
def test_draft_call_matches_todays_path(rpa_stand_in, monkeypatch, D, main_slot):
    _interpret_keep_params(monkeypatch)
    _route_any_dcp(monkeypatch)
    R = max(D, 2)
    if D > 1:
        from vllm_torchtpu.models.vllm.qwen4_exp import pool_layout

        monkeypatch.setenv(pool_layout.MAIN_KV_DCP_ENV, str(D))
    c = _case(7 + D, R, D, main_slot)
    want, want_pools = _stock(c, R, D)
    got, got_pools = _kernel(c, R, D)
    assert np.isfinite(got).all()
    live = np.arange(c["n"]) < LIVE
    err = np.abs(got[:, live] - want[:, live])
    ulps = err / _ulp(want[:, live])
    exact = float((err == 0).mean())
    byte_diff = [int((a != b).sum()) for a, b in zip(got_pools, want_pools)]
    changed = [int((b != np.asarray(p0).view(np.uint8)).sum()) for b, p0 in zip(want_pools, c["pools"])]
    print(f"L1-H3 call vs today D={D} slot={main_slot}: bitwise {exact:.6f}, max ulps {ulps.max():.1f}, "
          f"scale {np.abs(want[:, live]).mean():.4f}; pool bytes differing {byte_diff} (today's pass changed "
          f"{changed})", flush=True)
    assert np.abs(want[:, live]).mean() > 1e-3
    bad = err > 2 * _ulp(want[:, live]) + 1e-6
    assert not bad.any(), ("attention beyond 2 ulps", int(bad.sum()), float(ulps.max()))
    assert (got[:, ~live] == 0).all()
    # Control: today's pass writes on exactly the ranks that own a live token (main-KV DCP owner (t // B) % D).
    owners = {int(p) // PB % D for p in c["positions"][:LIVE]} if D > 1 else {0}
    assert [r % D in owners for r in range(R)] == [x > 0 for x in changed], ("write control", owners, changed)
    assert byte_diff == [0] * R, ("pool bytes differ", byte_diff)
