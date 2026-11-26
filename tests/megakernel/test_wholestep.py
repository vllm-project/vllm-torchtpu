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
"""Whole-step prototype: two `[GDN, GDN, GDN, QSA]` groups chained in one kernel (CPU).

Every QSA layer is a full in-kernel layer (HC mix, projection, norms, RoPE,
sparse attention, gate, o_proj). The reference chains the library's block
references with `project_reference` + `reference_rows` for the QSA layers.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.blocks.hc import RESIDUAL_OUT
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.models.qwen38_pack import shard_step_inputs
from vllm_torchtpu.megakernel.ops import BF16, F32
from vllm_torchtpu.megakernel.wholestep import inputs as wi
from vllm_torchtpu.megakernel.wholestep.qsa_full import KV_NEW
from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec

_TILES = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)
G = qwen38.TINY
BUDGET, PAGES, L = 64, 8, 8
KV_LEN = [0, 1, 9, 100, 255, 257, 130, 64]  # empty row, single token, tile edges


def _run(
    key: int = 11,
    T: int = 4,
    zero: tuple[str, ...] = (),
    identical_experts: bool = False,
    hc_fp8: bool = False,
):
    R = 2
    ex = ExecSpec(R=R, B=2 * T, T=T, ctx=128, interpret=True, tiles=_TILES)
    # Router gain: eight chained layers drift enough for a near-tied top-k router logit
    # to flip one row's expert set; the library tests use the same knob for the same
    # reason.
    d = wi.random_wholestep_inputs(
        key,
        G,
        ex,
        L,
        budget=BUDGET,
        n_pages=PAGES,
        kv_len=KV_LEN,
        zero=zero,
        identical_experts=identical_experts,
    )
    if hc_fp8:  # the reference sees the FP8-rounded HC weights
        for name in ("moe", "qsa_full", "gdn"):
            for k in ("down", "inj", "up"):
                w = d["hc"][name][k]
                q, sc = wi._fp8_columns(w)
                d["hc"][name][k] = (q.astype(jnp.float32) * sc[..., None, :]).astype(w.dtype)
    ref, kv_ref = wi.wholestep_reference(G, ex, d, L)
    spec = wholestep_spec(G, R, L=L, qsa=dict(budget=BUDGET), hc_fp8=hc_fp8)
    w_mk, states = wi.pack_wholestep(spec, G, ex, d, L, hc_fp8=hc_fp8)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(
        mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states
    )
    assert [x.name for x in kernel.extra_inputs(spec, ex)] == wi.EXTRA_NAMES
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *wi.extras(d, mesh, R))
    B = ex.B
    res = np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, -1)[:, :B], np.float32)
    kv = np.asarray(outs[KV_NEW], np.float32).reshape(R, L // G.att_every, ex.M, -1)[
        :, :, :B
    ]
    x = np.asarray(d["inp"].x[:B], np.float32)
    return res, np.asarray(ref, np.float32), x, kv, np.asarray(kv_ref, np.float32)


def _assert_within_budget(res: np.ndarray, ref: np.ndarray, layers: int = L) -> None:
    """The segment test's budget (mean 1e-2, q999 5e-2, max 1e-1 of mean |ref|, set for
    L = 4: eight BF16-rounded sublayers), scaled by sqrt(sublayers / 8): BF16 rounding
    of the
    residual stream accumulates as a random walk over the sublayers. Measured with
    identical
    experts (no router flips, no outlier rows): L = 4 mean 0.0046 / q999 0.0475 / max
    0.098;
    L = 8 mean 0.0084 / q999 0.0590 / max 0.0755 (library segment L = 4: 0.0047 / 0.0446
    /
    0.047)."""
    k = np.sqrt(2 * layers / 8)
    scale = float(np.abs(ref).mean())
    for r in range(res.shape[0]):
        d = np.abs(res[r] - ref)
        assert d.mean() < 1e-2 * k * scale, (r, d.mean(), scale)
        # q999 6e-2 (was 5e-2): the sigmoid(z) GDN gate passes more BF16 rounding (first
        # QSA
        # layer's K/V rows: 0.0313 vs 0.0281); the segment parity tests moved 5e-2 ->
        # 6e-2.
        assert np.quantile(d, 0.999) < 6e-2 * k * scale, (
            r,
            np.quantile(d, 0.999),
            scale,
        )
        assert d.max() < 1e-1 * k * scale, (r, d.max(), scale)


def test_two_groups_identical_experts_match_reference():
    """Strict: with routing flips made harmless, every row is inside the budget."""
    res, ref, x, kv, kv_ref = _run(identical_experts=True)
    assert float(np.std(ref - x)) > 0.05
    _assert_within_budget(res, ref)
    # The step's K/V rows (cache-write payload): the first QSA layer's input has seen
    # three
    # layers of drift, the second's seven; both inside the residual's depth-scaled
    # budget.
    # atol 4e-2: Qwen3.8's sigmoid(z) GDN gate passes more of the BF16-rounded GDN
    # output
    # than silu did (5 of 8192 elements at 0.035; the segment tests widened likewise).
    np.testing.assert_allclose(kv[:, 0], kv_ref[:, 0], rtol=2e-2, atol=4e-2)
    for o in range(kv.shape[1]):
        for r in range(kv.shape[0]):  # per-rank projections: per-rank references
            _assert_within_budget(kv[r : r + 1, o], kv_ref[r, o], layers=4 * (o + 1))


def test_two_groups_fp8_hc_weights_match_reference():
    """VLLM_TPU_MK_HC_FP8: with FP8 E4M3 HC mix weights (per-column scales) the kernel
    matches a reference built from the same FP8-rounded weights inside the BF16
    budget, and differs from the BF16-weight kernel (the FP8 path is live)."""
    res8, ref8, _, _, _ = _run(identical_experts=True, hc_fp8=True)
    res, _, _, _, _ = _run(identical_experts=True)
    assert not np.array_equal(res8, res)
    _assert_within_budget(res8, ref8)


def test_two_groups_random_experts_statistical():
    """Random experts: a 1-ulp difference upstream of a router may flip one row's top-k
    (seen as a single-row outlier, absent with MoE silenced); bound the mean and the
    number of outlier rows instead of every element."""
    res, ref, x, _, _ = _run()
    scale, k = float(np.abs(ref).mean()), np.sqrt(2 * L / 8)
    for r in range(res.shape[0]):
        d = np.abs(res[r] - ref)
        assert d.mean() < 1e-2 * k * scale, (r, d.mean(), scale)
        assert int((d.max(axis=1) > 1e-1 * k * scale).sum()) <= 2, (
            r,
            d.max(axis=1),
            scale,
        )


def test_two_groups_budget_catches_a_dropped_qsa_layer(monkeypatch):
    """Positive control: the strict budget rejects a kernel whose QSA layers add
    nothing."""
    from vllm_torchtpu.megakernel.wholestep.qsa_full import QSAFullBlock

    orig = QSAFullBlock.emit
    monkeypatch.setattr(
        QSAFullBlock,
        "emit",
        lambda self, ctx, ln, slot: 0.0 * orig(self, ctx, ln, slot),
    )
    res, ref, _, _, _ = _run(identical_experts=True)
    with pytest.raises(AssertionError):
        _assert_within_budget(res, ref)


# --------------------------------------------------------------------- stage B
MAX_GROUPS = 1024
CTX = [
    3000,
    100,
]  # request 0: n_vis 750 > K (top-K path); request 1: n_vis 25 <= K (select all)


def _run_b(key: int = 13, T: int = 4, identical_experts: bool = True):
    from vllm_torchtpu.megakernel.wholestep.qsa_full_b import DBG_IDS, DBG_IQ

    R = 2
    ex = ExecSpec(R=R, B=2 * T, T=T, ctx=128, interpret=True, tiles=_TILES)
    d = wi.random_wholestep_inputs(
        key, G, ex, L, budget=BUDGET, n_pages=PAGES, identical_experts=identical_experts
    )
    d = wi.add_indexer_inputs(key, G, ex, d, L, max_groups=MAX_GROUPS, ctx=CTX)
    ref, _ = wi.wholestep_reference(G, ex, d, L)
    pps = d["mtab"].shape[1]
    spec = wholestep_spec(
        G,
        R,
        L=L,
        qsa=dict(budget=BUDGET),
        indexed=dict(max_groups=MAX_GROUPS, pages_per_seq=pps, debug=True),
    )
    w_mk, states = wi.pack_wholestep(spec, G, ex, d, L)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(
        mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states
    )
    assert [x.name for x in kernel.extra_inputs(spec, ex)] == wi.EXTRA_NAMES_B
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *wi.extras_b(d, mesh, R))
    B, n_att = ex.B, L // G.att_every
    res = np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, -1)[:, :B], np.float32)
    iq = np.asarray(outs[DBG_IQ], np.float32).reshape(R, n_att, ex.M, -1)[0, :, :B]
    ids = np.asarray(outs[DBG_IDS]).reshape(R, n_att, ex.M, -1)[0, :, :B]
    return (
        res,
        np.asarray(ref, np.float32),
        np.asarray(d["inp"].x[:B], np.float32),
        iq,
        ids,
        d,
        ex,
    )


def test_in_kernel_selection_exact_on_its_own_queries():
    """Strict: the kernel's selected group ids equal the stock rule (top-K, ties ->
    lower id;
    all visible groups when n_vis <= K) applied to the kernel's own index queries, every
    row
    and QSA layer. Separates selection logic (must be exact) from upstream rounding."""
    from vllm_torchtpu.megakernel.wholestep.indexer import score_rows, topk_reference

    _, _, _, iq, ids, d, ex = _run_b()
    T = ex.T
    for o in range(iq.shape[0]):
        for r in range(ex.B):
            qr = jnp.asarray(iq[o, r].reshape(4, -1), BF16)
            st = np.asarray(score_rows(qr, d["ck"][r // T]))[0]
            n = int(d["nvis"][r])
            want = set(topk_reference(st, n, BUDGET))
            got = set(ids[o, r, : min(n, BUDGET)].tolist())
            assert got == want, (o, r, len(got ^ want))


def test_two_groups_in_kernel_indexer_match_reference():
    """End to end: select-all rows (request 1) strictly inside the depth-scaled budget;
    top-K rows (request 0) may see a boundary flip from 1-ulp drift upstream of the
    indexer
    (as MoE routing does), so they are bounded statistically."""
    res, ref, x, _, _, _, ex = _run_b()
    assert float(np.std(ref - x)) > 0.05
    T = ex.T
    _assert_within_budget(res[:, T : 2 * T], ref[T : 2 * T])
    scale, k = float(np.abs(ref).mean()), np.sqrt(2 * L / 8)
    for r in range(res.shape[0]):
        d = np.abs(res[r, :T] - ref[:T])
        assert int((d.max(axis=1) > 1e-1 * k * scale).sum()) <= 2, (
            r,
            d.max(axis=1),
            scale,
        )


def test_selection_check_catches_an_off_by_one_id(monkeypatch):
    """Positive control for the strict selection check: ids shifted by one group fail
    it."""
    from vllm_torchtpu.megakernel.wholestep import indexer as ix

    orig = ix.compact_row
    monkeypatch.setattr(ix, "compact_row", lambda *a, **k: orig(*a, **k) + 1)
    with pytest.raises(AssertionError):
        test_in_kernel_selection_exact_on_its_own_queries()


# --------------------------------------------------------------------- stage C
CTX_C = [
    3001,
    100,
]  # request 0: g_a holds one old token; request 1: g_a starts at the step


def _run_c(key: int = 17, T: int = 4, long: bool = False, shard: bool = False, rank_ids: bool = False,
           union: bool = False, stats: bool = False, loop: bool = False, ctx: list | None = None,
           max_groups: int | None = None, phys: bool = False, dcp: int = 1, dcp_pool: bool = True,
           shared_kv: bool = False, cdcp: bool = False, cdcp_keys: bool = True, ck_fn=None,
           async_dma: bool = False, B: int | None = None, xreq: int = 0,
           skip_idle: bool = False, xmerge: bool = False, key_mp: int = 0, key_slots: int = 2,
           req_async: int = 0, nvis_balance: bool = False, uattn_lite: bool = False,
           key_lanes: int = 0, export: bool = False, select_nvis: int = 0,
           nvis_rows: tuple | None = None, phys_mxu: bool = False):
    from vllm_torchtpu.megakernel.wholestep.qsa_full_b import DBG_IDS, DBG_IQ
    from vllm_torchtpu.megakernel.wholestep.qsa_full_c import (
        CK_NEW,
        IR_NEW,
        KR_NEW,
        RING_NEW,
    )

    CTX = CTX_C if ctx is None else ctx
    MG = MAX_GROUPS if max_groups is None else max_groups
    R = 2
    ex = ExecSpec(R=R, B=2 * T if B is None else B, T=T, ctx=128, interpret=True, tiles=_TILES)
    d = wi.random_wholestep_inputs(
        key, G, ex, L, budget=BUDGET, n_pages=PAGES, identical_experts=True
    )
    d = wi.add_indexer_inputs(key, G, ex, d, L, max_groups=MG, ctx=CTX)
    if ck_fn is not None:  # test-chosen compressed keys (e.g. tie-heavy scores)
        d["ck"] = ck_fn(d["ck"])
    d = wi.add_step_inputs(key, G, ex, d, L, ctx=CTX)
    if (
        long
    ):  # served long-context mode: HBM page tables, one request's scores at a time
        d = wi.add_carrier(key, d, mp=key_mp)
        d["hbm_tables"] = True
    if shared_kv:  # the ranks share one KV head (a DCP pair): same K/V columns, same pool
        kv = 2 * G.HD  # `[q0|g0|..|k|v]`: K and V are the last two 256-wide column blocks
        d["qkvg"] = d["qkvg"].at[1:, ..., -kv:].set(
            jnp.broadcast_to(d["qkvg"][:1, ..., -kv:], d["qkvg"][1:, ..., -kv:].shape)
        )
        d["pool"] = jnp.broadcast_to(d["pool"][:1], d["pool"].shape)
    ref, _ = wi.wholestep_reference(G, ex, d, L)
    steps = d.pop("_step_out")
    pps = d["mtab"].shape[1]
    if dcp > 1 and dcp_pool:
        _dcp_split(d)
    if cdcp and cdcp_keys:
        _ck_split(d)
    long_kw = dict(carrier="bf16", chunk_pages=8, hbm_tables=True, per_request_scores=True,
                   bounded_chunks=True, shard_index=shard, union_attn=union,
                   union_stats=stats, union_phys=phys or phys_mxu, loop_requests=loop, dcp=dcp,
                   compressed_dcp=cdcp, xreq_prefetch=xreq, skip_idle=skip_idle,
                   manager_pages=key_mp, key_slots=key_slots, ck_lanes=key_lanes,
                   **(dict(xmerge=True) if xmerge else {}),
                   **(dict(req_async=req_async) if req_async else {}),
                   **(dict(nvis_balance=True) if nvis_balance else {}),
                   **(dict(uattn_lite=True) if uattn_lite else {}),
                   **(dict(select_nvis=select_nvis) if select_nvis else {}),
                   **(dict(nvis_rows=nvis_rows) if nvis_rows else {}),
                   **(dict(union_phys_mxu=pps) if phys_mxu else {}))
    spec = wholestep_spec(
        G,
        R,
        L=L,
        qsa=dict(budget=BUDGET),
        step=True,
        indexed=dict(
            max_groups=MG, pages_per_seq=pps, debug=True, async_dma=async_dma,
            **(long_kw if long else {}), **(dict(export_ids=True) if export else {})
        ),
    )
    w_mk, states = wi.pack_wholestep(spec, G, ex, d, L)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(
        mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states
    )
    names = list(wi.EXTRA_NAMES_C)
    if long:
        names.insert(names.index("qsa_ck") + 1, "qsa_cktab")
    assert [x.name for x in kernel.extra_inputs(spec, ex)] == names
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *wi.extras_c(d, mesh, R))
    B, n_att, nreq = ex.B, L // G.att_every, ex.M // T
    res = np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, -1)[:, :B], np.float32)
    ck = np.asarray(outs[CK_NEW], np.float32).reshape(R, n_att, nreq, -1)[0]
    ring = np.asarray(outs[RING_NEW], np.float32).reshape(R, n_att, nreq, 8, -1)[0]
    iq = np.asarray(outs[DBG_IQ], np.float32).reshape(R, n_att, ex.M, -1)[0, :, :B]
    ids = np.asarray(outs[DBG_IDS]).reshape(R, n_att, ex.M, -1)[:, :, :B]
    ids = ids if rank_ids else ids[0]
    x0 = np.asarray(d["inp"].x[:B], np.float32)
    d["_kr_new"] = np.asarray(outs[KR_NEW], np.float32).reshape(R, n_att, ex.M, -1)[
        :, :, :B
    ]
    d["_ir_new"] = np.asarray(outs[IR_NEW], np.float32).reshape(R, n_att, ex.M, -1)[
        :, :, :B
    ]
    if export:
        from vllm_torchtpu.megakernel.wholestep.qsa_full_b import SEL_IDS

        d["_sel_ids"] = np.asarray(outs[SEL_IDS]).reshape(R, ex.M, -1)[:, :B]
    if stats:
        from vllm_torchtpu.megakernel.wholestep.qsa_full_c import (
            UNION_STATS,
            UNION_STATS_REQS,
        )

        d["_stats"] = np.asarray(outs[UNION_STATS]).reshape(
            R, n_att, UNION_STATS_REQS, -1
        )[:, :, :nreq]
    return res, np.asarray(ref, np.float32), x0, ck, ring, steps, iq, ids, d


def _dcp_split(d: dict) -> None:
    """Main-KV DCP-2 layout of the replicated test pool, in place: table column `c` now covers
    tokens `[2cB, 2(c+1)B)` at page `mtab[2c]`; rank 0 keeps that page (its first `B` tokens),
    rank 1's copy of it gets page `mtab[2c + 1]`'s rows (the next `B`), and every odd-column page
    is NaN-poisoned on both ranks (nothing may read it)."""
    pool = np.array(d["pool"])  # [R, pages, B, 1, 4, 256] FP8
    mtab = np.asarray(d["mtab"])
    even, odd = mtab[..., 0::2], mtab[..., 1::2]
    n = odd.shape[-1]
    nan = np.full(pool.shape[2:], np.nan, np.float32).astype(pool.dtype)
    for r in range(pool.shape[0]):
        src = pool[r].copy()
        for e, o in zip(even[..., :n].ravel(), odd.ravel()):
            if r == 1:
                pool[r, e] = src[o]
        for o in set(odd.ravel().tolist()) - set(even.ravel().tolist()):
            pool[r, o] = nan
    d["pool"] = jnp.asarray(pool)
    pad = np.zeros_like(mtab[..., : mtab.shape[-1] - even.shape[-1]])
    d["mtab"] = jnp.asarray(np.concatenate([even, pad], axis=-1))


def _ck_split(d: dict, R: int = 2) -> None:
    """Compressed-key DCP-2 layout of the test carrier, in place: rank r's copy of request q's
    first half of `cktab` pages holds its own groups in local order (local `l` -> group
    `(l // gpp) * 2 gpp + r gpp + l % gpp`, `gpp` = groups per pool block), and every other
    page is NaN-poisoned on that rank (nothing may read it)."""
    car = np.asarray(d["carrier"].astype(jnp.float32))  # [pages, 16, 2, width]
    n_pages, width = car.shape[0], car.shape[-1]
    flat = car.reshape(n_pages, 32, width)
    tab = np.asarray(d["cktab"])
    ck = np.asarray(d["ck"].astype(jnp.float32))  # [requests, groups, 128]
    gpp = d["pool"].shape[2] // 4
    half = tab.shape[1] // R
    l = np.arange(half * 32)
    out = np.full((R, n_pages, 32, width), np.nan, np.float32)
    for r in range(R):
        gid = (l // gpp) * (R * gpp) + r * gpp + l % gpp
        for q in range(tab.shape[0]):
            keys = np.zeros((half * 32, width), np.float32)
            keys[:, : ck.shape[-1]] = ck[q, gid]
            out[r, tab[q, :half]] = keys.reshape(half, 32, width)
    d["carrier"] = jnp.asarray(out.reshape(R, n_pages, 16, 2, width), BF16)


def _cdcp_runs(monkeypatch, **kw):
    """(replicated-key DCP-2 run, compressed-key DCP-2 run) of `_run_c(long, union, shard)`."""
    from vllm_torchtpu.models.vllm.qwen4_exp import pool_layout

    monkeypatch.setenv(pool_layout.MAIN_KV_DCP_ENV, "2")
    base_kw = dict(long=True, rank_ids=True, union=True, shard=True, shared_kv=True, dcp=2)
    base_kw.update(kw)
    return _run_c(**base_kw), _run_c(**base_kw, cdcp=True)


def _assert_same_selection(run: tuple, base: tuple) -> None:
    """Same pool-path ids per row (as sets: the pair layout lists rank 0's groups first), on
    every rank, and the same step outputs."""
    a, b = np.sort(run[7], axis=-1), np.sort(base[7], axis=-1)
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(run[0]).all()
    for i, name in ((0, "residual"), (3, "ck_new"), (4, "ring_new")):
        np.testing.assert_array_equal(run[i], base[i], err_msg=name)


@pytest.mark.parametrize("loop", [False, True])
def test_compressed_dcp2_selection_matches_replicated_keys(monkeypatch, loop):
    """Compressed-key DCP-2 (`compressed_dcp`, R = 2 = one pair): each rank holds only its own
    groups' keys (`_ck_split`, the rest NaN-poisoned), scores them, the pair swaps score halves
    and both select over the whole set. Same selection and bit-identical outputs as the
    replicated-key DCP-2 kernel; the compressed kernel on unsplit keys (it would score the
    wrong groups for every odd-gpp span) is caught."""
    base, run = _cdcp_runs(monkeypatch, loop=loop)
    _assert_same_selection(run, base)
    with pytest.raises(AssertionError):
        _assert_same_selection(_run_c(long=True, rank_ids=True, union=True, shard=True,
                                      shared_kv=True, dcp=2, loop=loop, cdcp=True,
                                      cdcp_keys=False), base)


def test_compressed_dcp2_ties_break_to_the_lower_group(monkeypatch):
    """Tie-heavy: every key repeats with period 3 groups, so most scores tie and the top-K
    boundary lands inside a run of equal scores that spans both ranks' groups. The pair
    selection keeps the lower group ids exactly as the replicated selection does."""
    def ties(ck):
        G_ = ck.shape[1]
        return ck[:, np.arange(G_) % 3]

    base, run = _cdcp_runs(monkeypatch, loop=True, ck_fn=ties)
    _assert_same_selection(run, base)


def test_dcp2_union_attention_matches_replicated(monkeypatch):
    """Main-KV DCP-2 (`dcp=2`, R = 2 = one pair sharing a KV head): each rank holds half of every request's main
    KV (`_dcp_split`, the other half NaN-poisoned), attends both ranks' heads over the groups it
    owns and the pair combines by LSE. Matches the replicated union kernel to f32 combine
    rounding; the DCP kernel on the unsplit pool (wrong rows for every odd-B span) is caught."""
    from vllm_torchtpu.models.vllm.qwen4_exp import pool_layout

    base = _run_c(long=True, rank_ids=True, union=True, shard=True, shared_kv=True)
    scale = float(np.abs(base[1]).mean())

    def close(run: tuple) -> None:
        dd = np.abs(run[0] - base[0])
        assert np.isfinite(run[0]).all()
        assert dd.max() < 1e-2 * scale, dd.max() / scale
        np.testing.assert_array_equal(run[7], base[7])  # selection untouched

    monkeypatch.setenv(pool_layout.MAIN_KV_DCP_ENV, "2")
    close(_run_c(long=True, rank_ids=True, union=True, shard=True, shared_kv=True, dcp=2))
    with pytest.raises(AssertionError):
        close(_run_c(long=True, rank_ids=True, union=True, shard=True, shared_kv=True, dcp=2,
                     dcp_pool=False))


@pytest.mark.parametrize("nvis", [3, 1])
def test_dcp2_union_lanepack_bit_identical(monkeypatch, nvis):
    """VLLM_TPU_MK_QSA_UNION_LANEPACK: under main-KV DCP-2 each pair rank's union scans only
    its packed rows — the ranged union (`select_nvis` 3) and the full-row one (`select_nvis`
    1) — so the step is bit-identical to the masked scan: residual, selection ids, compressed
    keys and ring. Negative control (ranged): the packed union's ids shifted by two rows."""
    from vllm_torchtpu.megakernel.wholestep import union_attn as ua
    from vllm_torchtpu.models.vllm.qwen4_exp import pool_layout

    monkeypatch.setenv(pool_layout.MAIN_KV_DCP_ENV, "2")
    kw = dict(long=True, rank_ids=True, union=True, shard=True, shared_kv=True, dcp=2,
              select_nvis=nvis, nvis_rows=(16, 16), max_groups=8192, ctx=[30001, 9000])
    calls = []
    if nvis >= 2:  # packed ranged union: `compact_union_chunk` gets the 7th (`id_of`) arg
        name, packed = "compact_union_chunk", lambda a, k: len(a) > 6 or "id_of" in k
    else:  # packed full union: its ids go through `unpack_union_ids`
        name, packed = "unpack_union_ids", lambda a, k: True
    orig = getattr(ua, name)

    def spy(*a, **k):
        calls.append(packed(a, k))
        return orig(*a, **k)

    monkeypatch.setattr(ua, name, spy)
    monkeypatch.setenv("VLLM_TPU_MK_QSA_UNION_LANEPACK", "0")
    off = _run_c(**kw)
    assert not any(calls)  # the masked union ran
    calls.clear()
    monkeypatch.setenv("VLLM_TPU_MK_QSA_UNION_LANEPACK", "1")
    on = _run_c(**kw)
    assert calls and all(calls)  # the packed union ran
    assert np.isfinite(on[0]).all()
    for i in (0, 3, 4, 7):  # residual, compressed keys, ring, selection ids
        np.testing.assert_array_equal(on[i], off[i])
    if nvis < 2:
        return
    chunk = orig

    def shifted(code, tot, row_of, KU, consts, store, id_of=None):
        return chunk(code, tot, lambda rj: row_of(rj) + 2, KU, consts, store,
                     None if id_of is None else lambda rj, lj: id_of(rj, lj) + 256)

    monkeypatch.setattr(ua, "compact_union_chunk", shifted)
    jax.clear_caches()  # same spec as `on`: retrace so the shifted ids are compiled in
    bad = _run_c(**kw)
    assert not np.array_equal(bad[0], off[0])


@pytest.mark.parametrize("long", [False, True])
def test_step_selection_exact_on_its_own_queries(long):
    """Strict: the pool-path ids are the stock selection (from the kernel's own queries
    and
    its own completed-group key) restricted to groups older than the step, every
    row/layer.
    `long`: HBM page tables with per-layer SMEM windows and per-request score buffers.
    """
    from vllm_torchtpu.megakernel.wholestep.indexer import score_rows, topk_reference

    _, _, _, ck, _, _, iq, ids, d = _run_c(long=long)
    T = 4
    for o in range(iq.shape[0]):
        for r in range(iq.shape[1]):
            q = r // T
            c0 = int(d["ctxq"][q])
            p3 = c0 + (3 - c0 % 4) % 4
            qr = jnp.asarray(iq[o, r].reshape(4, -1), BF16)
            st = np.array(score_rows(qr, d["ck"][q]))[0]
            st[p3 // 4] = np.asarray(
                score_rows(
                    qr, jnp.broadcast_to(jnp.asarray(ck[o, q], BF16), (8, ck.shape[-1]))
                )
            )[0, 0]
            n = int(d["nvis"][r])
            want = {i for i in topk_reference(st, n, BUDGET) if i < c0 // 4}
            got = set(ids[o, r, : len(want)].tolist())
            assert got == want, (o, r, len(got ^ want))


@pytest.mark.parametrize("kw", [
    dict(long=True),
    dict(long=True, union=True, shard=True, xmerge=True, req_async=2, loop=True, xreq=3,
         skip_idle=True, async_dma=True),
    # skip_idle with two idle request slots in a 16-row bucket: their rows are neither
    # selected nor exchanged, so their exported ids are a previous layer's (masked by starts).
    dict(long=True, union=True, shard=True, xmerge=True, req_async=2, loop=True, xreq=3,
         skip_idle=True, async_dma=True, B=16, ctx=[3001, 100, 0, 0]),
], ids=["long", "served", "served_idle"])
def test_export_ids_restore_the_full_stock_selection(kw):
    """VLLM_TPU_MK_WHOLE_EXPORT_SELECTION: the exported ids are the last QSA layer's
    pool-path ids on every rank, and `selection_groups` turns them into the stock
    selection of every row, g_a included (the kernel keeps g_a off the pool path), and
    `export_selection` into the stock `selected` rows of the draft's sink."""
    from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import selected_tokens
    from vllm_torchtpu.megakernel.wholestep.indexer import score_rows, topk_reference
    from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_inputs import (
        export_selection,
        selection_groups,
    )

    *_, ck, _, _, iq, ids, d = _run_c(rank_ids=True, export=True, **kw)
    sel = d["_sel_ids"]  # [R, B, K]
    o, T, (R, B, K) = iq.shape[0] - 1, 4, sel.shape
    for r in range(R):
        np.testing.assert_array_equal(sel[r], ids[r, o])
    ctx = jnp.asarray(d["ctxq"], jnp.int32)
    live = np.asarray(d["ctxq"]) > 0  # served: an idle slot has no query tokens
    starts = jnp.asarray(np.concatenate([[0], np.cumsum(np.where(live, T, 0))]), jnp.int32)
    groups, lens = selection_groups(jnp.asarray(sel[0]), ctx, starts, T=T, K=K)
    groups, lens = np.asarray(groups), np.asarray(lens)
    want_blocks = np.full((B, K), -1, np.int32)
    n_ga = 0
    for row in range(B):
        q = row // T
        c0 = int(d["ctxq"][q])
        if not live[q]:
            assert lens[row] == 0 and (groups[row] == -1).all(), row
            continue
        p3 = c0 + (3 - c0 % 4) % 4
        qr = jnp.asarray(iq[o, row].reshape(4, -1), BF16)
        st = np.array(score_rows(qr, d["ck"][q]))[0]
        st[p3 // 4] = np.asarray(
            score_rows(qr, jnp.broadcast_to(jnp.asarray(ck[o, q], BF16), (8, ck.shape[-1])))
        )[0, 0]
        n = int(d["nvis"][row])
        want = sorted(topk_reference(st, n, K))
        n_ga += int(c0 // 4 in want and n > K)
        assert sorted(groups[row][groups[row] >= 0].tolist()) == want, row
        assert lens[row] == c0 + row % T + 1
        want_blocks[row, : len(want)] = want
    got = np.asarray(export_selection(jnp.asarray(sel[0]), ctx, starts, T=T, K=K, rows=B, tails=True))
    sorted_groups = np.sort(np.where(groups < 0, np.iinfo(np.int32).max, groups), axis=1)
    sorted_groups = np.where(sorted_groups == np.iinfo(np.int32).max, -1, sorted_groups)
    np.testing.assert_array_equal(sorted_groups, want_blocks)
    ref = np.asarray(selected_tokens(jnp.asarray(want_blocks), jnp.asarray(lens), 4 * K, 4))
    np.testing.assert_array_equal(got, ref)
    print(f"export: rows with g_a selected past the budget: {n_ga}")


def test_sharded_indexer_matches_unsharded():
    """`shard_index`: each rank scores and selects its own request (R = 2, two
    requests), then
    the ranks exchange ids, lengths and `g_a` flags. Every rank ends with the unsharded
    kernel's
    pool-path ids, and the step's outputs are bit-identical (same ids, same attention
    order)."""
    base = _run_c(long=True, rank_ids=True)
    shard = _run_c(long=True, shard=True, rank_ids=True)
    ids_b, ids_s = base[7], shard[7]
    for r in range(
        ids_s.shape[0]
    ):  # every rank holds every row's ids after the exchange
        np.testing.assert_array_equal(ids_s[r], ids_b[0])
    for i, name in ((0, "residual"), (3, "ck_new"), (4, "ring_new")):
        np.testing.assert_array_equal(shard[i], base[i], err_msg=name)


@pytest.mark.parametrize("dcp", [1, 2])
def test_union_stacked_rows_bit_identical(monkeypatch, dcp):
    """`union_attn.STACK_ROWS`: a request's rows scored and accumulated as one stacked
    `[T * nh, 256]` matmul / softmax per tile give bit-identical outputs to the per-row loop,
    with and without main-KV DCP."""
    from vllm_torchtpu.megakernel.wholestep import union_attn as ua
    from vllm_torchtpu.models.vllm.qwen4_exp import pool_layout

    kw = dict(long=True, rank_ids=True, union=True, shard=True, shared_kv=True)
    if dcp > 1:
        monkeypatch.setenv(pool_layout.MAIN_KV_DCP_ENV, str(dcp))
        kw["dcp"] = dcp
    monkeypatch.setattr(ua, "STACK_ROWS", True)
    stacked = _run_c(**kw)
    monkeypatch.setattr(ua, "STACK_ROWS", False)
    per_row = _run_c(**kw)
    assert np.isfinite(stacked[0]).all()
    np.testing.assert_array_equal(stacked[0], per_row[0])


def test_union_attention_matches_per_row(monkeypatch):
    """`union_attn`: each request's rows attend one union of their selections (here 229
    of a
    possible 256 groups for request 0: four ring tiles instead of one per row). Every
    row attends
    the same tokens in the same order, so only the online softmax's tile partition
    differs: the
    outputs agree to BF16 rounding (on these inputs, exactly), alone and with
    `shard_index`. A
    corrupted row code (row 0 attending the whole union) is caught."""
    from vllm_torchtpu.megakernel.wholestep import union_attn as ua

    base = _run_c(long=True, rank_ids=True)
    scale = float(np.abs(base[1]).mean())

    def close(run: tuple) -> float:
        d = np.abs(run[0] - base[0])
        assert d.max() < 1e-2 * scale and (d == 0).mean() > 0.99, (
            d.max() / scale,
            (d == 0).mean(),
        )
        np.testing.assert_array_equal(
            run[7], base[7]
        )  # the selection itself is untouched
        return float(d.max())

    close(_run_c(long=True, rank_ids=True, union=True))
    close(_run_c(long=True, rank_ids=True, union=True, shard=True))
    orig = ua.compact_union

    def broken(code, KU, consts):  # noqa: ANN001
        ids, codes, n = orig(code, KU, consts)
        return ids, codes | 1, n

    monkeypatch.setattr(ua, "compact_union", broken)
    with pytest.raises(AssertionError):
        close(_run_c(long=True, rank_ids=True, union=True))


def test_physical_ids_match_the_copy_address_math():
    """`union_attn.physical_ids` (vector lookup) equals the per-copy scalar address
    `tab[v >> shift] * per_page + (v & (per_page - 1))` for every id of every table
    column,
    including id 0 (the zero-padded slots); a wrong shift is caught."""
    from vllm_torchtpu.megakernel.wholestep import union_attn as ua

    rng = np.random.default_rng(0)
    for per_page, pps in ((128, 256), (128, 512), (4, 9)):
        tab = rng.integers(0, 200_000, (1, pps)).astype(np.int32)
        ids = np.arange(per_page * pps, dtype=np.int32)
        ids = np.concatenate([ids, np.zeros((-len(ids)) % 128, np.int32)])
        want = tab[0, ids >> (per_page.bit_length() - 1)] * per_page + (
            ids & (per_page - 1)
        )
        got = np.concatenate(
            [
                np.asarray(
                    ua.physical_ids(
                        jnp.asarray(ids[None, i : i + 128]), jnp.asarray(tab), per_page
                    )
                )[0]
                for i in range(0, len(ids), 128)
            ]
        )
        np.testing.assert_array_equal(got, want)
        bad = np.asarray(
            ua.physical_ids(
                jnp.asarray(ids[None, :128] + 128 * per_page),
                jnp.asarray(tab),
                2 * per_page,
            )
        )[0]
        assert not np.array_equal(bad, want[:128])


def test_physical_ids_scan_only_the_visible_table_chunks():
    """`wholephysmxu`: `physical_ids` with `n_tab = table_chunks(n_vis, ...)` reads only the
    table chunks holding the columns of groups `< n_vis` (MXU byte-split gather) and equals
    the vector pass, for 1-4 byte entries incl. the largest; one chunk too few misses the ids
    of the last chunk (the bound is what makes it equal)."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.megakernel.wholestep import union_attn as ua

    rng = np.random.default_rng(1)
    for per_page, pps, dcp, nbytes in ((128, 512, 1, 3), (128, 512, 2, 3), (4, 256, 1, 3),
                                       (128, 256, 1, 1), (128, 256, 2, 2), (32, 384, 1, 4)):
        hi = min(1 << (8 * nbytes), 1 << 30)
        tab = rng.integers(0, hi, (1, pps)).astype(np.int32)
        tab[0, :2] = (0, hi - 1)
        cs = per_page.bit_length() - 1 + dcp.bit_length() - 1
        for n_vis in (1, 5 << cs, (128 << cs) + 3, (pps << cs) - 1):
            ids = rng.integers(0, n_vis, (1, 128)).astype(np.int32)
            ids[0, :4] = (0, n_vis - 1, 0, n_vis - 1)
            want = np.asarray(ua.physical_ids(jnp.asarray(ids), jnp.asarray(tab), per_page, dcp))

            def run(n_tab, ids=ids, tab=tab, per_page=per_page, dcp=dcp, nbytes=nbytes):
                def kern(ids_ref, tab_ref, out_ref):
                    out_ref[...] = ua.physical_ids(
                        ids_ref[...], tab_ref, per_page, dcp, n_tab, table_bytes=nbytes
                    )

                return np.asarray(pl.pallas_call(
                    kern, out_shape=jax.ShapeDtypeStruct((1, 128), jnp.int32),
                    in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 2,
                    out_specs=pl.BlockSpec(memory_space=pltpu.VMEM), interpret=True,
                )(jnp.asarray(ids), jnp.asarray(tab)))

            n_tab = int(ua.table_chunks(jnp.int32(n_vis), per_page, dcp, pps))
            np.testing.assert_array_equal(run(n_tab), want,
                                          err_msg=f"{per_page} {dcp} {nbytes} {n_vis}")
            if n_tab > 1:
                assert not np.array_equal(run(n_tab - 1), want)


# `wholephysmxu` on the full kernel needs a 128-lane-multiple table row: max_groups 4096
# (pps 128, one chunk) and 8192 with a context past 128 columns of 32 groups (pps 256, two).
PHYS_MXU_CASES = [
    dict(max_groups=4096),
    dict(max_groups=4096, shard=True, loop=True),
    dict(max_groups=8192, ctx=[17000, 100]),
]


@pytest.mark.parametrize("kw", PHYS_MXU_CASES)
def test_union_physical_ids_mxu_bit_identical(kw):
    """`wholephysmxu` (prefetched table rows, MXU gather of only the visible chunks): every
    output byte-identical to the union kernel without physical ids, and to `wholephys`."""
    import gc

    outs = []
    for arm in (dict(), dict(phys=True), dict(phys_mxu=True)):
        res = _run_c(long=True, rank_ids=True, union=True, **arm, **kw)
        outs.append([res[i] for i in (0, 3, 4, 7)])
        del res
        jax.clear_caches()  # one compiled step at a time (CPU memory)
        gc.collect()
    base, phys, run = outs
    for j, name in enumerate(("residual", "ck_new", "ring_new", "ids")):
        np.testing.assert_array_equal(run[j], base[j], err_msg=f"{name} {kw}")
        np.testing.assert_array_equal(run[j], phys[j], err_msg=f"{name} phys {kw}")


def test_union_physical_ids_mxu_short_scan_is_caught(monkeypatch):
    """Mutation: the gather scans no table chunk (every id -> page 0); the comparison must
    fail (the MXU path really makes the copies' addresses)."""
    from vllm_torchtpu.megakernel.wholestep import union_attn as ua

    kw = dict(long=True, rank_ids=True, union=True, max_groups=4096)
    base = _run_c(**kw)
    monkeypatch.setattr(ua, "table_chunks", lambda *a, **k: jnp.int32(0))
    run = _run_c(**kw, phys_mxu=True)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(run[0], base[0])


def test_union_physical_ids_bit_identical():
    """`union_phys`: the copies read the same addresses, so the step is bit-identical to
    the
    union kernel (alone and with `shard_index` + `loop_requests`)."""
    for kw in (dict(), dict(shard=True, loop=True)):
        base = _run_c(long=True, rank_ids=True, union=True, **kw)
        run = _run_c(long=True, rank_ids=True, union=True, phys=True, **kw)
        for i, name in ((0, "residual"), (3, "ck_new"), (4, "ring_new"), (7, "ids")):
            np.testing.assert_array_equal(run[i], base[i], err_msg=f"{name} {kw}")


def test_union_stats_count_the_rows_selections():
    """`union_stats`: per QSA layer and request, `[union size, sum of the rows'
    pool-path
    groups, max row, rows with groups]`, equal on every rank to the counts from the ids.
    """
    run = _run_c(long=True, rank_ids=True, union=True, shard=True, stats=True)
    ids, d = run[7][0], run[8]
    st = d["_stats"]
    for o in range(ids.shape[0]):
        for q in range(len(CTX_C)):
            ga = int(d["ctxq"][q]) // 4
            sets = []
            for t in range(4):
                r = q * 4 + t
                n = min(int(d["nvis"][r]), BUDGET)
                sets.append({int(x) for x in ids[o, r, :n] if x < ga})
            want = [
                len(set().union(*sets)),
                sum(map(len, sets)),
                max(map(len, sets)),
                sum(1 for x in sets if x),
            ]
            for r in range(st.shape[0]):
                assert st[r, o, q, :4].tolist() == want, (
                    r,
                    o,
                    q,
                    st[r, o, q, :4],
                    want,
                )


def test_request_loop_matches_unrolled():
    """`loop_requests`: the per-request work in a `fori_loop` (traced request index, the
    carrier prefetch chain on traced first/last flags) gives bit-identical outputs,
    alone and with
    the sharded indexer and union attention."""
    for kw in (dict(), dict(shard=True, union=True)):
        base = _run_c(long=True, rank_ids=True, **kw)
        loop = _run_c(long=True, rank_ids=True, loop=True, **kw)
        for i, name in ((0, "residual"), (3, "ck_new"), (4, "ring_new"), (7, "ids")):
            np.testing.assert_array_equal(loop[i], base[i], err_msg=f"{name} {kw}")


@pytest.mark.parametrize("kw", [dict(union=True, loop=True),
                                dict(shard=True, union=True, loop=True)])
def test_key_stream_ring_and_manager_pages_bit_identical(monkeypatch, kw):
    """VLLM_TPU_MEGAKERNEL wholekeyslots / wholekeymp: a 3- or 4-slot key ring and manager
    pages of 2 or 4 kernel pages per DMA give bit-identical residuals, completed keys, rings
    and selected ids. Request 0 streams 6 chunks (more than the slots), request 1 one chunk
    (fewer than the prefetch depth, after a cross-request prefetch); unsharded the two share
    one rank's stream (traced first/last flags), sharded each rank streams one. Positive
    control: scoring a stale ring (`SCORE_ARM = "compute"`, what a slot read before its copy
    lands sees) is caught."""
    from vllm_torchtpu.megakernel.wholestep import qsa_full_b

    run = functools.partial(_run_c, long=True, rank_ids=True, ctx=[6000, 100], max_groups=2048,
                            **kw)
    base = run()
    for mp, ks in ((0, 3), (4, 3), (2, 4)):
        got = run(key_mp=mp, key_slots=ks)
        for i, name in ((0, "residual"), (3, "ck_new"), (4, "ring_new"), (7, "ids")):
            np.testing.assert_array_equal(got[i], base[i], err_msg=f"{name} mp={mp} slots={ks}")
    # wholekeylanes=128: only the real lanes of each carrier row, alone and with manager pages.
    for mp, ks in ((0, 4), (4, 3)):
        got = run(key_mp=mp, key_slots=ks, key_lanes=128)
        for i, name in ((0, "residual"), (3, "ck_new"), (4, "ring_new"), (7, "ids")):
            np.testing.assert_array_equal(got[i], base[i], err_msg=f"{name} mp={mp} lanes=128")
    monkeypatch.setattr(qsa_full_b, "SCORE_ARM", "compute")
    bad = run(key_slots=3)
    assert not np.array_equal(bad[7], base[7]), "stale-ring scores must change the selection"


def test_step_tokens_match_stock_semantics():
    """Stage C: the completed group's key and the new ring rows equal the stock write
    for
    every request and QSA layer (bf16-close: their inputs carry the residual's drift);
    the
    residual, with every row's own tokens attended from VMEM, is inside the depth-scaled
    budget
    on select-all rows (request 1). Top-K rows (request 0, 750 groups -> 64) are not
    held to
    the per-row budget: a 1-ulp drift of the index query flips boundary groups there
    (the
    selection itself is exact on the kernel's own queries, the strict test above); only
    their
    mean error is bounded, at 3x the depth-scaled mean budget."""
    res, ref, x, ck, ring, steps, _, _, d = _run_c()
    assert float(np.std(ref - x)) > 0.05
    for o, s in enumerate(steps):
        # Cache-writer inputs: raw index key (replicated) and each rank's un-normed key.
        for r in range(d["_kr_new"].shape[0]):
            np.testing.assert_allclose(
                d["_ir_new"][r, o],
                np.asarray(s["raw"], np.float32),
                atol=0.1,
                rtol=0.05,
            )
            np.testing.assert_allclose(
                d["_kr_new"][r, o],
                np.asarray(s["kr"][r], np.float32),
                atol=0.1,
                rtol=0.05,
            )
        for q in range(len(CTX_C)):
            np.testing.assert_allclose(
                ck[o, q], np.asarray(s["ck"][q], np.float32)[0], atol=0.1, rtol=0.05
            )
            np.testing.assert_allclose(
                ring[o, q], np.asarray(s["ring"][q], np.float32), atol=0.1, rtol=0.05
            )
    T = 4
    _assert_within_budget(res[:, T : 2 * T], ref[T : 2 * T])
    scale, k = float(np.abs(ref).mean()), np.sqrt(2 * L / 8)
    for r in range(res.shape[0]):
        dd = np.abs(res[r, :T] - ref[:T])
        assert dd.mean() < 3e-2 * k * scale, (
            r,
            dd.mean() / scale,
            dd.max(axis=1) / scale,
        )


def test_step_with_served_key_layout_matches_contiguous():
    """Stage C reading the served compressed-key layout (32-group kernel pages, two BF16
    keys per
    carrier row, 256 lanes, shuffled pages, streamed 32 pages per chunk; the BF16 view
    here, the
    FP8 carrier on TPU) gives the same residual as the contiguous-key kernel, bit for
    bit."""
    from vllm_torchtpu.megakernel.wholestep.qsa_full_c import CK_NEW

    T, R = 4, 2
    ex = ExecSpec(R=R, B=2 * T, T=T, ctx=128, interpret=True, tiles=_TILES)
    d = wi.random_wholestep_inputs(
        17, G, ex, L, budget=BUDGET, n_pages=PAGES, identical_experts=True
    )
    maxg = (
        2 * MAX_GROUPS
    )  # two 32-page chunks per request: exercises the double-buffered ring
    d = wi.add_indexer_inputs(17, G, ex, d, L, max_groups=maxg, ctx=CTX_C)
    d = wi.add_step_inputs(17, G, ex, d, L, ctx=CTX_C)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pps = d["mtab"].shape[1]
    outs = []
    # contiguous keys; served layout per kernel page; per manager page (4 kernel pages
    # here,
    # 13 served) reading only the 128 real lanes.
    arms = (("", 0, 0), ("bf16", 0, 0), ("bf16", 4, 128))
    for carrier, mp, lanes in arms:
        dd = wi.add_carrier(17, d, mp=mp) if carrier else d
        idx = dict(
            max_groups=maxg,
            pages_per_seq=pps,
            carrier=carrier,
            chunk_pages=32,
            manager_pages=mp,
            ck_lanes=lanes,
        )
        spec = wholestep_spec(
            G, R, L=L, qsa=dict(budget=BUDGET), step=True, indexed=idx
        )
        w_mk, states = wi.pack_wholestep(spec, G, ex, dd, L)
        pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(
            mesh, ex, dd["inp"].pos, dd["inp"].x, w_mk, states
        )
        step, _ = kernel.make_step(mesh, spec, ex)
        _, o = step(pos_mk, hid_mk, w_mk, st_mk, *wi.extras_c(dd, mesh, R))
        outs.append(
            (np.asarray(o[RESIDUAL_OUT], np.float32), np.asarray(o[CK_NEW], np.float32))
        )
    for o in outs[1:]:
        np.testing.assert_array_equal(outs[0][0], o[0])
        np.testing.assert_array_equal(outs[0][1], o[1])


def test_qsa_first_layout_matches_reference():
    """Model layers 3..10 (cycle QSA, GDN, GDN, GDN): the layout that lets the head
    (layers 0-2,
    PLE before layer 1) stay a separate call while every QSA layer is in the whole-step
    kernel."""
    R, T = 2, 4
    ex = ExecSpec(R=R, B=2 * T, T=T, ctx=128, interpret=True, tiles=_TILES)
    d = wi.random_wholestep_inputs(
        11,
        G,
        ex,
        L,
        budget=BUDGET,
        n_pages=PAGES,
        kv_len=KV_LEN,
        identical_experts=True,
    )
    d["first"] = "qsa"
    ref, _ = wi.wholestep_reference(G, ex, d, L)
    d.pop("_step_out", None)
    spec = wholestep_spec(G, R, L=L, qsa=dict(budget=BUDGET), first="qsa")
    w_mk, states = wi.pack_wholestep(spec, G, ex, d, L)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(
        mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states
    )
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *wi.extras(d, mesh, R))
    res = np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, -1)[:, : ex.B], np.float32)
    assert (
        float(np.std(np.asarray(ref) - np.asarray(d["inp"].x[: ex.B], np.float32)))
        > 0.05
    )
    _assert_within_budget(res, np.asarray(ref, np.float32))


@pytest.mark.parametrize("V", [384, 8256])
def test_tail_greedy_tokens_match_reference(V):
    """Final HC mixer + lm_head + per-rank top token + argmax over ranks = the stock
    greedy
    verify (argmax of the full logits, ties to the lower index). `V = 8256`: two lm_head
    column blocks (`tail.vocab_chunks`), the second one padded past the vocab shard."""
    from vllm_torchtpu.megakernel.models.qwen38_inputs import (
        _hc_params,
        hc_weight_arrays,
    )
    from vllm_torchtpu.megakernel.wholestep.tail import (
        TOP,
        greedy_tokens,
        tail_reference,
        tail_spec,
    )

    R, T = 2, 4
    ex = ExecSpec(R=R, B=2 * T, T=T, ctx=128, interpret=True, tiles=_TILES)
    k = jax.random.split(jax.random.PRNGKey(5), 3)
    hc = _hc_params(k[0], G, 1)
    # Wider vocab, closer top-2 logits: spread them so the pick is decided.
    lm = (jax.random.normal(k[1], (R, G.H, V), F32) * (0.05 if V < 1024 else 0.5)).astype(BF16)
    x = jax.random.normal(k[2], (ex.B, G.NHC * G.H), F32).astype(BF16)
    want, logits = tail_reference(
        x, {n: v[0] for n, v in hc.items()}, lm, nhc=G.NHC, eps=G.eps
    )
    spec = tail_spec(G, R, vocab_shard=V)
    from vllm_torchtpu.megakernel.models.qwen38_pack import pack_weights

    from vllm_torchtpu.megakernel.wholestep.tail import vocab_chunk_width, vocab_chunks

    w = {**hc_weight_arrays("tail", hc, G, R, mix=True)}
    for name, c0, width in vocab_chunks(V, vocab_chunk_width(ex)):
        blk = lm[:, :, c0 : c0 + width]
        w[name] = jnp.pad(blk, ((0, 0), (0, 0), (0, width - blk.shape[-1])))[:, None]
    sm = {"tail_hcn": jnp.broadcast_to(hc["norm"][None], (R, 1, hc["norm"].shape[-1]))}
    w_mk = pack_weights(spec, ex, w, sm)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos = jnp.zeros((ex.M,), jnp.int32)
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(mesh, ex, pos, x, w_mk, {})
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk)
    got = np.asarray(greedy_tokens(outs[TOP], R, ex.M))[: ex.B]
    # Exact unless the reference's own top-2 gap is within rounding of the kernel's
    # logits.
    top2 = np.sort(np.asarray(logits), axis=-1)[:, -2:]
    clear = (top2[:, 1] - top2[:, 0]) > 1e-2
    assert clear.sum() >= ex.B - 1
    np.testing.assert_array_equal(got[clear], np.asarray(want)[clear])


def test_last_layer_with_tail_in_one_call():
    """Model layer 47 (QSA + MoE) and the decode tail (final HC mixer, lm_head, per-rank
    top
    token) in one kernel call: tokens equal the stock greedy pick on the reference
    residual."""
    from vllm_torchtpu.megakernel.wholestep.tail import (
        TOP,
        greedy_tokens,
        tail_reference,
    )

    R, T, V = 2, 4, 384
    ex = ExecSpec(R=R, B=2 * T, T=T, ctx=128, interpret=True, tiles=_TILES)
    d = wi.random_wholestep_inputs(
        19,
        G,
        ex,
        1,
        budget=BUDGET,
        n_pages=PAGES,
        kv_len=KV_LEN,
        identical_experts=True,
        first="qsa",
        vocab_shard=V,
    )
    d["lm_head"] = (d["lm_head"].astype(F32) * 10).astype(
        BF16
    )  # spread the top-2 logit gaps
    ref, _ = wi.wholestep_reference(G, ex, d, 1)
    d.pop("_step_out", None)
    want, logits = tail_reference(
        ref,
        {n: v[0] for n, v in d["hc"]["tail"].items()},
        d["lm_head"],
        nhc=G.NHC,
        eps=G.eps,
    )
    spec = wholestep_spec(
        G, R, L=1, qsa=dict(budget=BUDGET), first="qsa", tail=dict(vocab_shard=V)
    )
    w_mk, states = wi.pack_wholestep(spec, G, ex, d, 1)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(
        mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states
    )
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *wi.extras(d, mesh, R))
    got = np.asarray(greedy_tokens(outs[TOP], R, ex.M))[: ex.B]
    top2 = np.sort(np.asarray(logits), axis=-1)[:, -2:]
    clear = (top2[:, 1] - top2[:, 0]) > 5e-2  # the residual carries one layer of drift
    assert clear.sum() >= ex.B - 2, top2
    np.testing.assert_array_equal(got[clear], np.asarray(want)[clear])


@pytest.mark.parametrize("T", [1, 4])
def test_draft_step_kernel_attention_moe_and_greedy_tail(T):
    """MTP draft step as one kernel call (VLLM_TPU_MK_DRAFT): one QSA layer at the served
    stage-C packing (in-kernel indexer, selection, step K/V) + MoE + final HC mixer +
    lm_head + per-rank top token. `T = 1` is a draft loop step (one row per request),
    `T = 4` draft step 0 over the verify rows. Tokens equal the stock greedy pick."""
    from vllm_torchtpu.megakernel.wholestep.tail import (
        TOP,
        greedy_tokens,
        tail_reference,
    )

    R, V, Lt = 2, 384, 1
    ex = ExecSpec(R=R, B=2 * T, T=T, ctx=128, interpret=True, tiles=_TILES)
    d = wi.random_wholestep_inputs(
        31, G, ex, Lt, budget=BUDGET, n_pages=PAGES, identical_experts=True,
        first="qsa", vocab_shard=V,
    )
    d = wi.add_indexer_inputs(31, G, ex, d, Lt, max_groups=MAX_GROUPS, ctx=CTX_C)
    d = wi.add_step_inputs(31, G, ex, d, Lt, ctx=CTX_C)
    d = wi.add_per_layer(31, G, ex, d, Lt)
    d = wi.add_carrier(31, d)
    d["lm_head"] = (d["lm_head"].astype(F32) * 10).astype(BF16)
    ref, _ = wi.wholestep_reference(G, ex, d, Lt)
    d.pop("_step_out")
    want, logits = tail_reference(
        ref, {n: v[0] for n, v in d["hc"]["tail"].items()}, d["lm_head"],
        nhc=G.NHC, eps=G.eps,
    )
    idx = dict(
        max_groups=MAX_GROUPS, pages_per_seq=d["mtab"].shape[-1], carrier="bf16",
        chunk_pages=32, n_layers=1,
    )
    spec = wholestep_spec(
        G, R, L=Lt, qsa=dict(budget=BUDGET), step=True, indexed=idx, first="qsa",
        tail=dict(vocab_shard=V),
    )
    w_mk, states = wi.pack_wholestep(spec, G, ex, d, Lt)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(
        mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states
    )
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *wi.extras_c(d, mesh, R))
    got = np.asarray(greedy_tokens(outs[TOP], R, ex.M))[: ex.B]
    top2 = np.sort(np.asarray(logits), axis=-1)[:, -2:]
    clear = (top2[:, 1] - top2[:, 0]) > 5e-2
    assert clear.sum() >= ex.B - 2, top2
    np.testing.assert_array_equal(got[clear], np.asarray(want)[clear])


@pytest.mark.parametrize("T", [1, 4])
def test_draft_step_skip_idle_requests_bit_identical(T):
    """`skip_idle` (the MK draft's request loops stop after the last live request) at the
    served request loop / sharded selection / union attention: live rows' residual and
    greedy tokens are byte-identical to the full loop over every request slot (a T = 1
    tile holds 16 request slots, 2 live here)."""
    from vllm_torchtpu.megakernel.wholestep.tail import TOP, greedy_tokens

    R, V, Lt = 2, 384, 1
    ex = ExecSpec(R=R, B=2 * T, T=T, ctx=128, interpret=True, tiles=_TILES)
    d = wi.random_wholestep_inputs(
        37, G, ex, Lt, budget=BUDGET, n_pages=PAGES, identical_experts=True,
        first="qsa", vocab_shard=V,
    )
    # Live requests first (contexts > 0), the rest of the tile's request slots idle.
    nreq, live = ex.M // T, ex.B // T
    ctx = [CTX_C[q % len(CTX_C)] or 37 for q in range(live)] + [0] * (nreq - live)
    d = wi.add_indexer_inputs(37, G, ex, d, Lt, max_groups=MAX_GROUPS, ctx=ctx)
    d = wi.add_step_inputs(37, G, ex, d, Lt, ctx=ctx)
    d = wi.add_per_layer(37, G, ex, d, Lt)
    d = wi.add_carrier(37, d)
    d["hbm_tables"] = True
    wi.wholestep_reference(G, ex, d, Lt)
    d.pop("_step_out")
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    got = []
    for skip in (False, True):
        idx = dict(
            max_groups=MAX_GROUPS, pages_per_seq=d["mtab"].shape[-1], carrier="bf16",
            chunk_pages=8, n_layers=1, hbm_tables=True, per_request_scores=True,
            bounded_chunks=True, shard_index=True, union_attn=True, loop_requests=True,
            skip_idle=skip,
        )
        spec = wholestep_spec(
            G, R, L=Lt, qsa=dict(budget=BUDGET), step=True, indexed=idx, first="qsa",
            tail=dict(vocab_shard=V),
        )
        w_mk, states = wi.pack_wholestep(spec, G, ex, d, Lt)
        pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(
            mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states
        )
        step, _ = kernel.make_step(mesh, spec, ex)
        _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *wi.extras_c(d, mesh, R))
        res = np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, -1)[:, : ex.B], np.float32)
        tok = np.asarray(greedy_tokens(outs[TOP], R, ex.M))[: ex.B]
        got.append((res, tok))
    np.testing.assert_array_equal(got[0][0], got[1][0])
    np.testing.assert_array_equal(got[0][1], got[1][1])


def test_step_with_per_layer_metadata_matches_reference():
    """Served packing: every QSA layer has its own compressed keys, main page table, raw
    ring and
    head-pair offset (pages disjoint in one pool); the kernel indexes them by layer
    ordinal."""
    R, T = 2, 4
    ex = ExecSpec(R=R, B=2 * T, T=T, ctx=128, interpret=True, tiles=_TILES)
    d = wi.random_wholestep_inputs(
        23, G, ex, L, budget=BUDGET, n_pages=PAGES, identical_experts=True
    )
    d = wi.add_indexer_inputs(23, G, ex, d, L, max_groups=MAX_GROUPS, ctx=CTX_C)
    d = wi.add_step_inputs(23, G, ex, d, L, ctx=CTX_C)
    d = wi.add_per_layer(23, G, ex, d, L)
    d = wi.add_carrier(23, d)
    ref, _ = wi.wholestep_reference(G, ex, d, L)
    d.pop("_step_out")
    pps = d["mtab"].shape[-1]
    idx = dict(
        max_groups=MAX_GROUPS,
        pages_per_seq=pps,
        carrier="bf16",
        chunk_pages=32,
        n_layers=L // G.att_every,
    )
    spec = wholestep_spec(G, R, L=L, qsa=dict(budget=BUDGET), step=True, indexed=idx)
    w_mk, states = wi.pack_wholestep(spec, G, ex, d, L)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(
        mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states
    )
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *wi.extras_c(d, mesh, R))
    res = np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, -1)[:, : ex.B], np.float32)
    refn = np.asarray(ref, np.float32)
    assert float(np.std(refn - np.asarray(d["inp"].x[: ex.B], np.float32))) > 0.05
    _assert_within_budget(res[:, T : 2 * T], refn[T : 2 * T])  # select-all rows: strict
    scale, k = float(np.abs(refn).mean()), np.sqrt(2 * L / 8)
    for r in range(R):
        dd = np.abs(res[r, :T] - refn[:T])
        assert dd.mean() < 3e-2 * k * scale, (r, dd.mean() / scale)


@pytest.mark.parametrize("S", [8, 256, 512])
def test_compaction_exclusive_prefix_matches_reference(S):
    """`indexer.exclusive_prefix` (128-lane chunks + running total for S % 128 == 0,
    else one
    `[S, S]` matmul): `compact_row` returns the stock top-K ids ascending and
    `compact_union` the
    ascending union with its codes, at the 131K / 262K score widths (S = 256 / 512); a
    prefix
    shifted by one chunk is caught."""
    from vllm_torchtpu.megakernel.wholestep import indexer as ix
    from vllm_torchtpu.megakernel.wholestep import union_attn as ua

    G, K = S * 128, 64
    rng = np.random.default_rng(S)
    sc = rng.standard_normal(G).astype(np.float32)
    sc[rng.integers(0, G, 40)] = sc[5]  # ties across chunks
    nv = G - 37
    bits = np.where(np.arange(G) < nv, np.asarray(sc).view(np.int32), -1).reshape(
        1, S, 128
    )
    ids = np.arange(G, dtype=np.int32).reshape(1, S, 128)
    thr, tmax = ix.thresholds(
        jnp.asarray(bits), jnp.asarray(ids), jnp.full((1, 1, 1), nv), K, G
    )
    consts = ix.compact_consts(S, K)
    got = np.asarray(
        ix.compact_row(
            jnp.asarray(bits[0]), thr[0], tmax[0], jnp.full((1, 1), nv), K, consts
        )
    )[0]
    assert got.tolist() == sorted(ix.topk_reference(sc, nv, K))
    code = np.zeros((S, 128), np.int32)
    pick = rng.choice(G, 300, replace=False)
    code.reshape(-1)[pick] = rng.integers(1, 16, 300)
    KU = 512
    u_ids, u_codes, n = ua.compact_union(jnp.asarray(code), KU, ua.union_consts(S, KU))
    want = np.sort(pick)
    assert int(np.asarray(n)[0, 0]) == 300
    np.testing.assert_array_equal(np.asarray(u_ids)[0, :300], want)
    np.testing.assert_array_equal(np.asarray(u_codes)[0, :300], code.reshape(-1)[want])
    c = jnp.asarray(rng.integers(0, 129, (1, S)).astype(np.float32))
    want_p = np.concatenate([[0], np.cumsum(np.asarray(c)[0])[:-1]])
    np.testing.assert_array_equal(
        np.asarray(ix.exclusive_prefix(c, consts[1]))[0], want_p
    )
    if S % 128 == 0 and S > 128:  # positive control: drop the running total
        bad = jnp.concatenate(
            [
                jnp.dot(
                    c[:, i : i + 128].astype(jnp.bfloat16),
                    consts[1],
                    preferred_element_type=jnp.float32,
                )
                for i in range(0, S, 128)
            ],
            axis=1,
        )
        assert not np.array_equal(np.asarray(bad)[0], want_p)


@pytest.mark.parametrize("T", [4, 1])
def test_last_layer_step_with_per_layer_metadata_matches_reference(T):
    """Model layer 47 alone (`L = 1`, `first="qsa"`) at the served stage-C packing: the
    whole-step tail call that folds the last QSA layer (VLLM_TPU_MK_WHOLE_L47). `T = 1`:
    one query row per request, the MTP draft loop's shape (a draft-step kernel)."""
    R, Lt = 2, 1
    ex = ExecSpec(R=R, B=2 * T, T=T, ctx=128, interpret=True, tiles=_TILES)
    d = wi.random_wholestep_inputs(
        29, G, ex, Lt, budget=BUDGET, n_pages=PAGES, identical_experts=True, first="qsa"
    )
    d = wi.add_indexer_inputs(29, G, ex, d, Lt, max_groups=MAX_GROUPS, ctx=CTX_C)
    d = wi.add_step_inputs(29, G, ex, d, Lt, ctx=CTX_C)
    d = wi.add_per_layer(29, G, ex, d, Lt)
    d = wi.add_carrier(29, d)
    ref, _ = wi.wholestep_reference(G, ex, d, Lt)
    d.pop("_step_out")
    idx = dict(
        max_groups=MAX_GROUPS,
        pages_per_seq=d["mtab"].shape[-1],
        carrier="bf16",
        chunk_pages=32,
        n_layers=1,
    )
    spec = wholestep_spec(
        G, R, L=Lt, qsa=dict(budget=BUDGET), step=True, indexed=idx, first="qsa"
    )
    w_mk, states = wi.pack_wholestep(spec, G, ex, d, Lt)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(
        mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states
    )
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *wi.extras_c(d, mesh, R))
    res = np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, -1)[:, : ex.B], np.float32)
    refn = np.asarray(ref, np.float32)
    assert float(np.std(refn - np.asarray(d["inp"].x[: ex.B], np.float32))) > 0.01
    _assert_within_budget(res[:, T : 2 * T], refn[T : 2 * T], layers=Lt)
    scale, k = float(np.abs(refn).mean()), np.sqrt(2 * Lt / 8)
    for r in range(R):
        dd = np.abs(res[r, :T] - refn[:T])
        assert dd.mean() < 3e-2 * k * scale, (r, dd.mean() / scale)


def _partial_last_cycle_run(tiles, export: bool = False):
    """`L = 5`, `first="qsa"` whole-step call; returns (kernel residual, reference).
    `export`: also `(SEL_IDS, last QSA layer's DBG_IDS)` (the served L45 export)."""
    R, T, Lt = 2, 4, 5
    ex = ExecSpec(R=R, B=2 * T, T=T, ctx=128, interpret=True, tiles=tiles)
    d = wi.random_wholestep_inputs(
        29, G, ex, Lt, budget=BUDGET, n_pages=PAGES, identical_experts=True, first="qsa"
    )
    d = wi.add_indexer_inputs(29, G, ex, d, Lt, max_groups=MAX_GROUPS, ctx=CTX_C)
    d = wi.add_step_inputs(29, G, ex, d, Lt, ctx=CTX_C)
    d = wi.add_per_layer(29, G, ex, d, Lt)
    d = wi.add_carrier(29, d)
    ref, _ = wi.wholestep_reference(G, ex, d, Lt)
    d.pop("_step_out")
    idx = dict(
        max_groups=MAX_GROUPS,
        pages_per_seq=d["mtab"].shape[-1],
        carrier="bf16",
        chunk_pages=32,
        n_layers=2,
        **(dict(export_ids=True, debug=True) if export else {}),
    )
    spec = wholestep_spec(
        G, R, L=Lt, qsa=dict(budget=BUDGET), step=True, indexed=idx, first="qsa"
    )
    w_mk, states = wi.pack_wholestep(spec, G, ex, d, Lt)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(
        mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states
    )
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *wi.extras_c(d, mesh, R))
    res = np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, -1)[:, : ex.B], np.float32)
    refn = np.asarray(ref, np.float32)
    assert float(np.std(refn - np.asarray(d["inp"].x[: ex.B], np.float32))) > 0.01
    if export:
        from vllm_torchtpu.megakernel.wholestep.qsa_full_b import DBG_IDS, SEL_IDS

        sel = np.asarray(outs[SEL_IDS]).reshape(R, ex.M, -1)[:, : ex.B]
        dbg = np.asarray(outs[DBG_IDS]).reshape(R, 2, ex.M, -1)[:, -1, : ex.B]
        return res, refn, sel, dbg
    return res, refn


def test_export_ids_last_layer_of_a_per_layer_call():
    """`export_ids` with per-layer metadata (`n_layers = 2`, the served L45 kernel): the
    export is the last QSA layer's ids (its debug copy), on every rank, and the residual
    is bit-identical to the call without the export."""
    res, _ = _partial_last_cycle_run(_TILES)
    res_x, _, sel, dbg = _partial_last_cycle_run(_TILES, export=True)
    np.testing.assert_array_equal(res_x, res)
    np.testing.assert_array_equal(sel, dbg)
    assert (sel > 0).any()


def test_partial_last_cycle_ends_on_qsa_matches_reference():
    """`L = 5`, `first="qsa"`: one full cycle (QSA, GDN, GDN, GDN) and a partial cycle
    of its QSA layer, as layers 3..47 in one kernel (VLLM_TPU_MK_WHOLE_L45)."""
    R, T, Lt = 2, 4, 5
    res, refn = _partial_last_cycle_run(_TILES)
    _assert_within_budget(res[:, T : 2 * T], refn[T : 2 * T], layers=Lt)
    scale, k = float(np.abs(refn).mean()), np.sqrt(2 * Lt / 8)
    for r in range(R):
        dd = np.abs(res[r, :T] - refn[:T])
        assert dd.mean() < 3e-2 * k * scale, (r, dd.mean() / scale)


def test_partial_last_cycle_rel_fetch_bit_identical():
    """`rel_fetch` (trace-time-decoded dense refills, VLLM_TPU_MK_BUCKET_TUNE b4..b32) with a
    partial last cycle: its tiles must be refilled too. A `b < L // period` bound dropped
    them, and the L45 kernel hung on TPU at every rel_fetch bucket (exos j-ffbb37a8,
    j-a50b71fd). Scheduling only, so the outputs are byte-identical."""
    base, _ = _partial_last_cycle_run(_TILES)
    rel, _ = _partial_last_cycle_run(dataclasses.replace(_TILES, rel_fetch=True))
    np.testing.assert_array_equal(base, rel)


@pytest.mark.parametrize("kw", [
    dict(long=True, union=True, shard=True, loop=True),
    dict(long=True, union=True, shard=True),
    dict(long=True),
])
def test_async_selection_dmas_bit_identical(kw):
    """VLLM_TPU_MK_QSEL_ASYNC: the selection's SMEM id copies and the new group's loads /
    stores issued together (waited once) leave every output byte-identical."""
    base = _run_c(rank_ids=True, **kw)
    run = _run_c(rank_ids=True, async_dma=True, **kw)
    for a, b in ((base[0], run[0]), (base[3], run[3]), (base[4], run[4]), (base[7], run[7]),
                 (base[8]["_kr_new"], run[8]["_kr_new"]), (base[8]["_ir_new"], run[8]["_ir_new"])):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# Ragged requests for the cross-request prefetch: 4 / 1 / 0 / 4 / 1 / 4 / 0 / 3 union tiles
# (ctx 3: no pool-path group; a one-tile request ahead of a full one).
CTX_XREQ = [3001, 100, 3, 2000, 50, 3001, 1, 777]


def _xreq_pair(monkeypatch, B: int, dcp: int = 1, skip_idle: bool = False, phys: bool = False,
               xreq: int = 3):
    from vllm_torchtpu.models.vllm.qwen4_exp import pool_layout

    kw = dict(long=True, rank_ids=True, union=True, shard=True, loop=True, B=B, phys=phys,
              ctx=CTX_XREQ, skip_idle=skip_idle, async_dma=True)
    if dcp > 1:
        monkeypatch.setenv(pool_layout.MAIN_KV_DCP_ENV, str(dcp))
        kw.update(dcp=dcp, shared_kv=True)
    return _run_c(**kw), _run_c(**kw, xreq=xreq)


def _assert_xreq_identical(base: tuple, run: tuple) -> None:
    for i, name in ((0, "residual"), (3, "ck_new"), (4, "ring_new"), (7, "ids")):
        np.testing.assert_array_equal(run[i], base[i], err_msg=name)
    for k in ("_kr_new", "_ir_new"):
        np.testing.assert_array_equal(run[8][k], base[8][k], err_msg=k)


@pytest.mark.parametrize("B,dcp,xreq", [
    (8, 1, 3), (8, 2, 3), (32, 1, 3), (32, 2, 3), (64, 2, 3), (8, 1, 2), (8, 2, 2),
])
def test_xreq_prefetch_bit_identical(monkeypatch, B, dcp, xreq):
    """VLLM_TPU_MK_QSA_XREQ_PREFETCH: the union tile stream continues across the request loop
    (request q+1's ids staged in the second `qu_row` bank, its first tiles issued into the
    slots request q's last tiles free); ragged unions incl. empty requests and a one-tile
    request ahead of a full one; bit 2 issues the request loop's loads one request ahead
    (alone and with the tile stream). Every output byte-identical, with and without main-KV
    DCP."""
    base, run = _xreq_pair(monkeypatch, B, dcp, xreq=xreq)
    assert np.isfinite(base[0]).all()
    _assert_xreq_identical(base, run)


def test_xreq_prefetch_skip_idle_and_phys_bit_identical(monkeypatch):
    """The prefetch under `skip_idle` (the loop stops at the last live request: no prefetch
    past it) and under `union_phys` (the next request's physical ids staged early)."""
    global CTX_XREQ
    ctx = CTX_XREQ
    try:
        CTX_XREQ = [3001, 100, 3, 2000, 0, 0, 0, 0]  # live first, then idle slots
        base, run = _xreq_pair(monkeypatch, 32, skip_idle=True)
        _assert_xreq_identical(base, run)
    finally:
        CTX_XREQ = ctx
    base, run = _xreq_pair(monkeypatch, 32, phys=True)
    _assert_xreq_identical(base, run)


def test_xreq_prefetch_wrong_request_is_caught(monkeypatch):
    """Mutation: the prefetched tiles read the current request's ids instead of the next
    one's; the comparison must fail."""
    from vllm_torchtpu.megakernel.wholestep import union_attn as ua

    base, _ = _xreq_pair(monkeypatch, 32)
    orig = ua.attend_request

    def wrong(**kw):  # noqa: ANN003
        if kw.get("stream") is not None:
            sbase, first, (live, nrow, trow, n_next) = kw["stream"]
            kw["stream"] = (sbase, first, (live, 1 - nrow, trow - 1, n_next))
        return orig(**kw)

    monkeypatch.setattr(ua, "attend_request", wrong)
    run = _run_c(long=True, rank_ids=True, union=True, shard=True, loop=True, B=32,
                 ctx=CTX_XREQ, async_dma=True, xreq=1)
    with pytest.raises(AssertionError):
        _assert_xreq_identical(base, run)


# VLLM_TPU_MK_QSA_REQ_ASYNC: (B, dcp, XREQ_PREFETCH, phys, skip_idle, ctx, level). Ragged unions
# (CTX_XREQ: 4/1/0/4/1/4/0/3 tiles), one live request with an idle rank (B 8: 2 request slots,
# 1 per rank at R = 2), and the loop without the banked loads (xreq 0) / with `union_phys`.
REQ_ASYNC_CASES = [
    (8, 1, 2, False, False, None, 1),
    (8, 1, 2, False, False, None, 2),
    (32, 2, 2, False, False, None, 2),
    (8, 2, 2, False, True, [3001, 0], 2),
    (8, 1, 0, True, False, None, 2),
]


def _req_async_pair(monkeypatch, B, dcp, xreq, phys, skip_idle, ctx, level, mutate=None):
    from vllm_torchtpu.models.vllm.qwen4_exp import pool_layout

    kw = dict(long=True, rank_ids=True, union=True, shard=True, loop=True, B=B, phys=phys,
              ctx=CTX_XREQ if ctx is None else ctx, skip_idle=skip_idle, async_dma=True,
              xreq=xreq)
    if dcp > 1:
        monkeypatch.setenv(pool_layout.MAIN_KV_DCP_ENV, str(dcp))
        kw.update(dcp=dcp, shared_kv=True)
    base = _run_c(**kw)
    if mutate is not None:
        mutate()
    return base, _run_c(**kw, req_async=level)


@pytest.mark.parametrize("B,dcp,xreq,phys,skip_idle,ctx,level", REQ_ASYNC_CASES)
def test_req_async_bit_identical(monkeypatch, B, dcp, xreq, phys, skip_idle, ctx, level):
    """VLLM_TPU_MK_QSA_REQ_ASYNC: the new group's stores waited only before their bank's
    reuse / at the layer's end, the union ids staged one request ahead (after request q's
    attention, or into a second bank before it). Every output
    byte-identical (residual, ck_new, ring_new, ids, kr/ir_new), incl. 1 live request with an
    idle rank, main-KV DCP-2, ragged and empty unions, `union_phys`."""
    base, run = _req_async_pair(monkeypatch, B, dcp, xreq, phys, skip_idle, ctx, level)
    assert np.isfinite(base[0]).all()
    _assert_xreq_identical(base, run)


def test_req_async_wrong_store_row_is_caught(monkeypatch):
    """Mutation: the banked new-group stores go to request q ^ 1's output rows; the
    comparison must fail (the flag really takes the banked store path)."""
    from vllm_torchtpu.megakernel.wholestep import qsa_full_c as qc

    orig = qc.QSAFullStepBlock._new_group_stores

    def mutate():
        monkeypatch.setattr(qc.QSAFullStepBlock, "_new_group_stores",
                            lambda self, ctx, l, q, b: orig(self, ctx, l, q ^ 1, b))

    base, run = _req_async_pair(monkeypatch, 8, 1, 2, False, False, None, 2, mutate)
    with pytest.raises(AssertionError):
        _assert_xreq_identical(base, run)


# Main-call geometry (T = 4, 16-row tile: 4 request slots, 2 per rank at R = 2): live
# requests first, then idle slots (context 0) for B < 16.
CTX_MAIN = [3001, 100, 2000, 777]


def _main_run(monkeypatch, B: int, dcp: int = 1, **kw):
    """`_run_c` in the served main-call form (sharded selection, union attention, request
    loop, async selection DMAs, both cross-request prefetches)."""
    from vllm_torchtpu.models.vllm.qwen4_exp import pool_layout

    nlive = B // 4
    run_kw = dict(long=True, rank_ids=True, union=True, shard=True, loop=True, B=B,
                  ctx=CTX_MAIN[:nlive] + [0] * (4 - nlive), async_dma=True, xreq=3)
    if dcp > 1:
        monkeypatch.setenv(pool_layout.MAIN_KV_DCP_ENV, str(dcp))
        run_kw.update(dcp=dcp, shared_kv=True)
    run_kw.update(kw)
    return _run_c(**run_kw)


def _assert_live_identical(base: tuple, run: tuple, nlive: int) -> None:
    """Every live row / request output byte-identical (idle request slots' cache rows are
    unwritten under `skip_idle` and read by no one)."""
    for i, name in ((0, "residual"), (7, "ids")):
        np.testing.assert_array_equal(run[i], base[i], err_msg=name)
    for i, name in ((3, "ck_new"), (4, "ring_new")):
        np.testing.assert_array_equal(run[i][:, :nlive], base[i][:, :nlive], err_msg=name)
    for k in ("_kr_new", "_ir_new"):
        np.testing.assert_array_equal(run[8][k], base[8][k], err_msg=k)


@pytest.mark.parametrize("B,dcp", [(4, 1), (8, 1), (16, 1), (8, 2)])
def test_qsa_xmerge_bit_identical(monkeypatch, B, dcp):
    """VLLM_TPU_MK_QSA_XMERGE: the selection rows and the unions in one exchange per QSA
    layer; every output byte-identical (b4 / b8 / b16-like tiles, with idle request slots
    at b4 / b8, with and without main-KV DCP)."""
    base = _main_run(monkeypatch, B, dcp)
    run = _main_run(monkeypatch, B, dcp, xmerge=True)
    assert np.isfinite(base[0]).all()
    _assert_live_identical(base, run, 4)


def test_qsa_xmerge_unrolled_bit_identical():
    """The merged exchange from the unrolled sharded path (no request loop)."""
    kw = dict(long=True, rank_ids=True, union=True, shard=True, async_dma=True)
    base = _run_c(**kw)
    run = _run_c(**kw, xmerge=True)
    _assert_live_identical(base, run, 2)


@pytest.mark.parametrize("B", [4, 8, 16])
def test_main_skip_idle_and_xmerge_bit_identical(monkeypatch, B):
    """VLLM_TPU_MK_WHOLE_SKIP_IDLE on the main-call geometry (1 / 2 / 4 live requests of 4
    slots), alone and with the merged exchange: live outputs byte-identical to the full
    request loops."""
    nlive = B // 4
    base = _main_run(monkeypatch, B)
    skip = _main_run(monkeypatch, B, skip_idle=True)
    both = _main_run(monkeypatch, B, skip_idle=True, xmerge=True)
    _assert_live_identical(base, skip, nlive)
    _assert_live_identical(base, both, nlive)


@pytest.mark.parametrize("B,dcp,kw", [
    (8, 1, {}), (8, 2, {}), (8, 2, dict(skip_idle=True, xmerge=True)), (8, 1, dict(phys=True)),
])
def test_uattn_lite_bit_identical(monkeypatch, B, dcp, kw):
    """VLLM_TPU_MK_UATTN_LITE on the served main-call form (request loop, cross-request
    prefetch): every live output byte-identical to the default union attention, with and
    without main-KV DCP, skip_idle + xmerge, and physical ids."""
    nlive = 4 if not kw.get("skip_idle") else B // 4
    base = _main_run(monkeypatch, B, dcp, **kw)
    lite = _main_run(monkeypatch, B, dcp, uattn_lite=True, **kw)
    assert np.isfinite(base[0]).all()
    _assert_live_identical(base, lite, nlive)


def test_qsa_xmerge_lost_union_is_caught(monkeypatch):
    """Mutation: the peer's union payload of the merged row arrives zeroed (mis-packed
    exchange); the comparison must fail."""
    from jax.experimental import pallas as pl

    from vllm_torchtpu.megakernel.wholestep import qsa_full_c as qc

    base = _main_run(monkeypatch, 16)
    orig = qc.QSAFullStepBlock._gather_runs

    def lossy(self, ctx, send, recv, ssem, rsem, own_rank=False):  # noqa: ANN001
        orig(self, ctx, send, recv, ssem, rsem, own_rank)
        if getattr(self, "xmerge", False):
            u0, wm = self._xmerge_width(ctx.ex)
            peer = 1 - self._run(ctx)
            recv[peer, :, :, pl.ds(u0, wm - u0)] = jnp.zeros(
                (recv.shape[1], 1, wm - u0), jnp.int32
            )

    monkeypatch.setattr(qc.QSAFullStepBlock, "_gather_runs", lossy)
    run = _main_run(monkeypatch, 16, xmerge=True)
    with pytest.raises(AssertionError):
        _assert_live_identical(base, run, 4)


# VLLM_TPU_MK_QSA_NVIS_BALANCE_BUCKETS: main-call geometry (4 request slots, R = 2 -> two
# requests per run). Contexts chosen so the balanced runs differ from the consecutive ones:
# ragged 4 live ({0, 2} / {1, 3} instead of {0, 1} / {2, 3}), 3 live + an idle slot (odd
# count), 2 live (the consecutive split leaves rank 1 with idle slots only), and ties.
NVIS_BALANCE_CASES = [
    # B, dcp, ctx, key_slots, req_async
    (16, 1, [3001, 2000, 100, 777], 2, 0),
    (16, 2, [3001, 2000, 100, 777], 4, 1),
    (16, 1, [3001, 2000, 777, 0], 4, 1),
    (16, 2, [3001, 2000, 0, 0], 2, 1),
    (16, 1, [2000, 2000, 100, 2000], 4, 0),
]


def _nvis_balance_pair(monkeypatch, B, dcp, ctx, key_slots, req_async, mutate=None):
    kw = dict(ctx=ctx, xmerge=True, key_slots=key_slots, req_async=req_async)
    base = _main_run(monkeypatch, B, dcp, **kw)
    if mutate is not None:
        mutate()
    return base, _main_run(monkeypatch, B, dcp, nvis_balance=True, **kw)


@pytest.mark.parametrize("B,dcp,ctx,key_slots,req_async", NVIS_BALANCE_CASES)
def test_nvis_balance_bit_identical(monkeypatch, B, dcp, ctx, key_slots, req_async):
    """`nvis_balance`: the sharded selection's runs hold whole requests balanced by visible
    groups (largest with smallest) instead of consecutive requests. Each request is still
    selected whole by one rank, so every output is byte-identical (residual, ck_new, ring_new,
    ids, kr/ir_new): ragged contexts, an odd live count with an idle slot, a rank that is idle
    under the consecutive split, ties, the deep key ring, REQ_ASYNC, main-KV DCP-2."""
    base, run = _nvis_balance_pair(monkeypatch, B, dcp, ctx, key_slots, req_async)
    assert np.isfinite(base[0]).all()
    _assert_xreq_identical(base, run)


def test_nvis_balance_swapped_runs_are_caught(monkeypatch):
    """Mutation: the two runs' requests swapped in the ownership table only (each rank selects
    the other run's requests but packs its own run's slots); the comparison must fail (the
    flag takes the balanced path and the exchange reads the owners' rows)."""
    from vllm_torchtpu.megakernel.wholestep import qsa_full_c as qc

    orig = qc.QSAFullStepBlock._balance_tables

    def swapped(self, ctx):  # noqa: ANN001
        orig(self, ctx)
        for q in range(ctx.ex.MB // ctx.ex.T):
            ctx.scr["qb_run"][q] = 1 - ctx.scr["qb_run"][q]

    def mutate():
        monkeypatch.setattr(qc.QSAFullStepBlock, "_balance_tables", swapped)

    base, run = _nvis_balance_pair(monkeypatch, 16, 1, [3001, 2000, 100, 777], 2, 0, mutate)
    with pytest.raises(AssertionError):
        _assert_xreq_identical(base, run)


def test_nvis_balance_buckets_env(monkeypatch):
    """VLLM_TPU_MK_QSA_NVIS_BALANCE_BUCKETS: only the listed row buckets get the option;
    unset / unlisted / the bucketless spec add nothing (spec unchanged)."""
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wc

    monkeypatch.delenv("VLLM_TPU_MK_QSA_NVIS_BALANCE_BUCKETS", raising=False)
    assert wc.nvis_balance_options(48) == {}
    monkeypatch.setenv("VLLM_TPU_MK_QSA_NVIS_BALANCE_BUCKETS", "48,64")
    assert wc.nvis_balance_options(48) == {"nvis_balance": True}
    assert wc.nvis_balance_options(64) == {"nvis_balance": True}
    assert wc.nvis_balance_options(16) == {}
    assert wc.nvis_balance_options(None) == {}


def test_bucket_entries_helper(monkeypatch):
    """`bucket_entries`: "," and ";" separate entries, empties dropped; the three per-bucket
    envs parse through it unchanged (THR_FAST_BUCKETS values keep their ":")."""
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wc

    assert wc.bucket_entries("16,32") == ["16", "32"]
    assert wc.bucket_entries("16;32") == ["16", "32"]
    assert wc.bucket_entries("16:te;32:te,48:t") == ["16:te", "32:te", "48:t"]
    assert wc.bucket_entries(";16,,32;") == ["16", "32"]
    assert wc.bucket_entries("") == [] and wc.bucket_entries(None) == []
    monkeypatch.setenv("VLLM_TPU_MK_QSA_THR_FAST_BUCKETS", "16:te;32:ter2")
    assert wc.thr_fast_options(16) == {"thr_fast": "te"}
    assert wc.thr_fast_options(32) == {"thr_fast": "ter2"}
    assert wc.thr_fast_options(48) == {}
    monkeypatch.setenv("VLLM_TPU_MK_QSA_NVIS_BALANCE_BUCKETS", "48;64")
    assert wc.nvis_balance_options(64) == {"nvis_balance": True}
    assert wc.nvis_balance_options(32) == {}


def test_skip_idle_buckets_env(monkeypatch):
    """VLLM_TPU_MK_WHOLE_SKIP_IDLE_BUCKETS: unset -> VLLM_TPU_MK_WHOLE_SKIP_IDLE (global);
    set -> only the listed buckets ("," or ";"), overriding the global flag."""
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wc

    if not wc.LOOP_REQUESTS:
        monkeypatch.setattr(wc, "LOOP_REQUESTS", True)
    monkeypatch.delenv("VLLM_TPU_MK_WHOLE_SKIP_IDLE_BUCKETS", raising=False)
    monkeypatch.delenv("VLLM_TPU_MK_WHOLE_SKIP_IDLE", raising=False)
    assert not wc.skip_idle_on(16) and "skip_idle" not in wc.main_call_options(16)
    monkeypatch.setenv("VLLM_TPU_MK_WHOLE_SKIP_IDLE", "1")
    assert all(wc.skip_idle_on(b) for b in (16, 48, None))
    assert wc.main_call_options(48)["skip_idle"] is True
    for raw in ("16,32", "16;32"):
        monkeypatch.setenv("VLLM_TPU_MK_WHOLE_SKIP_IDLE_BUCKETS", raw)
        assert wc.skip_idle_on(16) and wc.skip_idle_on(32)
        assert not wc.skip_idle_on(48) and not wc.skip_idle_on(64) and not wc.skip_idle_on(None)
        assert "skip_idle" not in wc.main_call_options(48)
        assert wc.main_call_options(32)["skip_idle"] is True
    monkeypatch.delenv("VLLM_TPU_MK_WHOLE_SKIP_IDLE")
    assert wc.skip_idle_on(16) and not wc.skip_idle_on(48)


def test_skip_idle_buckets_keeps_nvis_balance_bit_identical(monkeypatch):
    """c32 served form: SKIP_IDLE_BUCKETS="16;32" + NVIS_BALANCE_BUCKETS="48;64" at b48 gives
    no skip_idle and nvis_balance on, so the kernel (main-call geometry, 4 request slots, R = 2,
    two requests per run) is byte-identical to skip_idle off + nvis_balance on, and the
    balanced path is really taken (the block keeps nvis_balance)."""
    from vllm_torchtpu.megakernel.wholestep import qsa_full_c as qc
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wc

    monkeypatch.setattr(wc, "LOOP_REQUESTS", True)
    monkeypatch.setenv("VLLM_TPU_MK_WHOLE_SKIP_IDLE", "1")
    monkeypatch.setenv("VLLM_TPU_MK_WHOLE_SKIP_IDLE_BUCKETS", "16;32")
    monkeypatch.setenv("VLLM_TPU_MK_QSA_NVIS_BALANCE_BUCKETS", "48;64")
    opts = {**wc.main_call_options(48), **wc.nvis_balance_options(48)}
    assert opts.get("nvis_balance") and not opts.get("skip_idle"), opts
    seen = []
    orig = qc.QSAFullStepBlock._balance_tables

    def spy(self, ctx):  # noqa: ANN001
        seen.append(self.skip_idle)
        return orig(self, ctx)

    monkeypatch.setattr(qc.QSAFullStepBlock, "_balance_tables", spy)
    kw = dict(ctx=[3001, 2000, 777, 0], xmerge=True, req_async=1)
    run = _main_run(monkeypatch, 16, skip_idle=bool(opts.get("skip_idle")),
                    nvis_balance=bool(opts.get("nvis_balance")), **kw)
    assert seen and not any(seen), seen
    base = _main_run(monkeypatch, 16, skip_idle=False, nvis_balance=True, **kw)
    assert np.isfinite(base[0]).all()
    _assert_xreq_identical(base, run)


@pytest.mark.parametrize("dcp,nvis,fast,cut", [(1, 3, "1", "1024"), (1, 3, "1", "16384"),
                                               (2, 3, "te", "1024"), (1, 1, "0", "1024")])
def test_lane_major_select_and_dense_read_bit_identical(monkeypatch, dcp, nvis, fast, cut):
    """VLLM_TPU_MK_QSA_LANE_MAJOR_SELECT (+ VLLM_TPU_MK_FP8_DENSE_READ): int16 count passes
    (THR_FAST early exit / tiers included), lane-major row, pool-path and union compactions
    (ranged union with the DCP-2 lane pack, full union) give a bit-identical step: residual,
    selection ids, compressed keys and ring. `cut` routes the rows to the full scan (1024) or
    the ranged one (16384)."""
    from vllm_torchtpu.megakernel.wholestep import indexer as ix
    from vllm_torchtpu.megakernel.wholestep import union_attn as ua
    from vllm_torchtpu.models.vllm.qwen4_exp import pool_layout

    if dcp > 1:
        monkeypatch.setenv(pool_layout.MAIN_KV_DCP_ENV, "2")
        monkeypatch.setenv("VLLM_TPU_MK_QSA_UNION_LANEPACK", "1")
    monkeypatch.setenv("VLLM_TPU_MK_QSA_THR_FAST", fast)
    monkeypatch.setenv("VLLM_TPU_MK_QSA_NVIS_CUT", cut)
    kw = dict(long=True, rank_ids=True, union=True, shard=True, shared_kv=True, dcp=dcp,
              select_nvis=nvis, nvis_rows=(16, 16), max_groups=8192, ctx=[30001, 9000])
    seen = []
    for mod, name in ((ix, "compact_sel"), (ix, "_value_passes16_early"),
                      (ua, "_compact_union_lm"), (ua, "_compact_union_chunk_lm")):
        orig = getattr(mod, name)

        def spy(*a, _orig=orig, _name=name, **k):
            seen.append(_name)
            return _orig(*a, **k)

        monkeypatch.setattr(mod, name, spy)
    monkeypatch.delenv("VLLM_TPU_MK_QSA_LANE_MAJOR_SELECT", raising=False)
    monkeypatch.delenv("VLLM_TPU_MK_FP8_DENSE_READ", raising=False)
    off = _run_c(**kw)
    assert not seen
    monkeypatch.setenv("VLLM_TPU_MK_QSA_LANE_MAJOR_SELECT", "1")
    monkeypatch.setenv("VLLM_TPU_MK_FP8_DENSE_READ", "1")
    jax.clear_caches()
    on = _run_c(**kw)
    if cut == "1024":  # full-scan rows: the lane-major row / pool-path compaction ran
        assert "compact_sel" in seen
    assert ("_compact_union_chunk_lm" if nvis >= 2 else "_compact_union_lm") in seen
    if cut == "1024" and "e" in (fast if fast != "1" else "ter2"):
        assert "_value_passes16_early" in seen
    assert np.isfinite(on[0]).all()
    for i in (0, 3, 4, 7):  # residual, compressed keys, ring, selection ids
        np.testing.assert_array_equal(on[i], off[i])
