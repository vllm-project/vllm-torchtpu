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
    ref, kv_ref = wi.wholestep_reference(G, ex, d, L)
    spec = wholestep_spec(G, R, L=L, qsa=dict(budget=BUDGET))
    w_mk, states = wi.pack_wholestep(spec, G, ex, d, L)
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
           shared_kv: bool = False, cdcp: bool = False, cdcp_keys: bool = True, ck_fn=None, R: int = 2,
           nreq: int = 2, budget: int | None = None, lean: dict | None = None, ring_fn=None, live=None, live_mode: int = 1):
    """q38 L1-D14: `R` ranks (one DCP group when `shared_kv`), 3 q heads per rank as served (NH 24 / TP 8), and
    `nreq` requests (`ctx` cycles); R = 2, nreq = 2 is the original two-request pair.
    q38 L1-D08S1: `budget` (default BUDGET), `lean` = extra indexed options (the D08 S1 stages / `union_probe`, long
    mode), `ring_fn(ring, ctxq)` = test-chosen raw index-key ring; with `union_probe` the probe dumps land in
    d["_probe_u"] [R, n_att, requests, 2 D KU + 128] and d["_probe_r"] [R, n_att, M, 256].
    q38 X-K-LIVE: `live` = the live-row mask (int32 [B]; `ExecSpec.live_rows` = `live_mode`, kernel input `live_rows`); the GDN
    checkpoints land in d["_gdn"] ({"rec_ckpt", "conv_ckpt"} as returned) on every run."""
    from vllm_torchtpu.megakernel.wholestep.qsa_full_b import DBG_IDS, DBG_IQ
    from vllm_torchtpu.megakernel.wholestep.qsa_full_c import (
        CK_NEW,
        IR_NEW,
        KR_NEW,
        RING_NEW,
    )

    CTX = CTX_C if ctx is None else ctx
    MG = MAX_GROUPS if max_groups is None else max_groups
    Gx = G if R == 2 else dataclasses.replace(G, NH=3 * R)
    if R > 4:  # q38 L1-D08S1: TINY's 4 / 12 GDN heads do not split over 8 ranks; keep R = 4's per-rank 1 / 3
        Gx = dataclasses.replace(Gx, KH=R, VH=3 * R)
    K_ = BUDGET if budget is None else budget
    ex = ExecSpec(R=R, B=nreq * T, T=T, ctx=128, interpret=True, tiles=_TILES, live_rows=live_mode if live is not None else False)
    d = wi.random_wholestep_inputs(
        key, Gx, ex, L, budget=K_, n_pages=PAGES, identical_experts=True
    )
    d = wi.add_indexer_inputs(key, Gx, ex, d, L, max_groups=MG, ctx=CTX)
    if ck_fn is not None:  # test-chosen compressed keys (e.g. tie-heavy scores)
        d["ck"] = ck_fn(d["ck"])
    d = wi.add_step_inputs(key, Gx, ex, d, L, ctx=CTX)
    if ring_fn is not None:  # q38 L1-D08S1: test-chosen raw index-key ring
        d["ring"] = ring_fn(d["ring"], d["ctxq"])
    if (
        long
    ):  # served long-context mode: HBM page tables, one request's scores at a time
        d = wi.add_carrier(key, d)
        d["hbm_tables"] = True
    if shared_kv:  # the ranks share one KV head (a DCP pair): same K/V columns, same pool
        kv = 2 * Gx.HD  # `[q0|g0|..|k|v]`: K and V are the last two 256-wide column blocks
        d["qkvg"] = d["qkvg"].at[1:, ..., -kv:].set(
            jnp.broadcast_to(d["qkvg"][:1, ..., -kv:], d["qkvg"][1:, ..., -kv:].shape)
        )
        d["pool"] = jnp.broadcast_to(d["pool"][:1], d["pool"].shape)
    ref, _ = wi.wholestep_reference(Gx, ex, d, L)
    steps = d.pop("_step_out")
    pps = d["mtab"].shape[1]
    if dcp > 1 and dcp_pool:
        _dcp_split(d, dcp)
    if cdcp and cdcp_keys:
        _ck_split(d)
    long_kw = dict(carrier="bf16", chunk_pages=8, hbm_tables=True, per_request_scores=True,
                   bounded_chunks=True, shard_index=shard, union_attn=union,
                   union_stats=stats, union_phys=phys, loop_requests=loop, dcp=dcp,
                   compressed_dcp=cdcp, **(lean or {}))
    spec = wholestep_spec(
        Gx,
        R,
        L=L,
        qsa=dict(budget=K_),
        step=True,
        indexed=dict(
            max_groups=MG, pages_per_seq=pps, debug=True, **(long_kw if long else {})
        ),
    )
    w_mk, states = wi.pack_wholestep(spec, Gx, ex, d, L)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(
        mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states
    )
    names = list(wi.EXTRA_NAMES_C)
    if long:
        names.insert(names.index("qsa_ck") + 1, "qsa_cktab")
    xlive = ()
    if live is not None:  # q38 X-K-LIVE: the planner's model-level input, last
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        names.append("live_rows")
        xlive = (jax.device_put(jnp.asarray(live, jnp.int32), NamedSharding(mesh, P())),)
    assert [x.name for x in kernel.extra_inputs(spec, ex)] == names
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *wi.extras_c(d, mesh, R), *xlive)
    d["_gdn"] = {k: np.asarray(outs[k]) for k in ("rec_ckpt", "conv_ckpt") if k in outs}
    B, n_att, nreq = ex.B, L // Gx.att_every, ex.M // T
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
    if stats:
        from vllm_torchtpu.megakernel.wholestep.qsa_full_c import (
            UNION_STATS,
            UNION_STATS_REQS,
        )

        d["_stats"] = np.asarray(outs[UNION_STATS]).reshape(
            R, n_att, UNION_STATS_REQS, -1
        )[:, :, :nreq]
    if (lean or {}).get("union_probe"):  # q38 L1-D08S1
        from vllm_torchtpu.megakernel.wholestep.qsa_full_lean import DBG_ROWS, DBG_UNION

        d["_probe_u"] = np.asarray(outs[DBG_UNION]).reshape(R, n_att, nreq, -1)
        d["_probe_r"] = np.asarray(outs[DBG_ROWS]).reshape(R, n_att, ex.M, -1)
    return res, np.asarray(ref, np.float32), x0, ck, ring, steps, iq, ids, d


def _dcp_split(d: dict, D: int = 2) -> None:
    """Main-KV DCP-D layout of the replicated test pool, in place: table column `c` now covers
    tokens `[D c B, D (c+1) B)` at page `mtab[D c]`; DCP rank `k = r % D` (stock `dcp_rank`) keeps, at that page,
    page `mtab[D c + k]`'s rows (its `B` tokens of the span), and every page of the other columns is
    NaN-poisoned on every rank (nothing may read it). D = 2 is the original pair layout (q38 L1-D14: any D)."""
    pool = np.array(d["pool"])  # [R, pages, B, 1, 4, 256] FP8
    mtab = np.asarray(d["mtab"])
    cols = [mtab[..., k::D] for k in range(D)]
    n = cols[-1].shape[-1]
    nan = np.full(pool.shape[2:], np.nan, np.float32).astype(pool.dtype)
    lead = set(cols[0].ravel().tolist())
    rest = set(np.concatenate([c.ravel() for c in cols[1:]]).tolist())
    for r in range(pool.shape[0]):
        src = pool[r].copy()
        k = r % D
        if k:
            for e, o in zip(cols[0][..., :n].ravel(), cols[k][..., :n].ravel()):
                pool[r, e] = src[o]
        for o in rest - lead:
            pool[r, o] = nan
    d["pool"] = jnp.asarray(pool)
    pad = np.zeros_like(mtab[..., : mtab.shape[-1] - cols[0].shape[-1]])
    d["mtab"] = jnp.asarray(np.concatenate([cols[0], pad], axis=-1))


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


# q38 L1-D14: main-KV DCP 4 = one stock DCP group of 4 TP ranks sharing a KV head (R = 4, `shared_kv`). The
# contexts put DCP-4 owner changes inside the step and the recent tile: 3070..3073 (page 23 -> 24, owner 3 -> 0;
# g_a 767 on page 23), 37 (select-all, owner 0), 1021..1024 (page 7 -> 8, owner 3 -> 0), 254..257 (page 1 -> 2,
# owner 1 -> 2) (PD's DCP-4 CTX32 subset, test_wholestep_qsa_dcp.py on 80b612e33).
CTX_D4 = [3070, 37, 1021, 254]
D4_KW = dict(long=True, rank_ids=True, union=True, shard=True, shared_kv=True, R=4, nreq=4, ctx=CTX_D4)


def _d4_close(run: tuple, two: tuple, base: tuple) -> None:
    """DCP 4 against the replicated union kernel, judged against MM's DCP-2 kernel on the same inputs (q38 L1-D14):
    finite; max error <= 1.25 x DCP-2's max error; mean error <= 1.25 x DCP-2's mean error + 1e-5 of mean |ref|;
    selection untouched. The DCP-2 anchor itself must stay loosely sane (max < 0.1, mean < 5e-3 of mean |ref|).
    proof-i1 (j-128756f4) measured on these inputs DCP 4 mean 6.429e-4 / max 3.268e-2 and DCP 2 mean 6.434e-4 /
    max 3.268e-2 of mean |ref|: an absolute 1e-2 max bound (the R = 2 DCP-2 test's) sat below DCP-2's own error."""
    scale = float(np.abs(base[1]).mean())
    e2, e4 = np.abs(two[0] - base[0]), np.abs(run[0] - base[0])
    print(f"L1-D14 DCP4 vs replicated: mean {e4.mean() / scale:.3e} max {e4.max() / scale:.3e}; DCP2 vs "
          f"replicated: mean {e2.mean() / scale:.3e} max {e2.max() / scale:.3e} (of mean |ref| {scale:.5f})",
          flush=True)
    assert np.isfinite(run[0]).all()
    assert e2.max() < 0.1 * scale and e2.mean() < 5e-3 * scale, (e2.max() / scale, e2.mean() / scale)
    assert e4.max() <= 1.25 * e2.max(), (e4.max() / scale, e2.max() / scale)
    assert e4.mean() <= 1.25 * e2.mean() + 1e-5 * scale, (e4.mean() / scale, e2.mean() / scale)
    np.testing.assert_array_equal(run[7], base[7])  # selection untouched


def test_dcp4_union_attention_matches_replicated(monkeypatch):
    """Main-KV DCP-4 (`dcp=4`; q38 L1-D14): each rank of the group holds a quarter of every request's main KV
    (`_dcp_split(D=4)`, the rest NaN-poisoned), gathers the group's q, attends the group's 4 x nh heads over
    the groups it owns, returns each peer its heads' partials and merges its own heads' four partials by LSE
    in DCP-rank order. Its error to the replicated union kernel is within 1.25x MM's DCP-2 kernel's on the same
    inputs (DCP 2 at R = 4: two pairs), max and mean (`_d4_close`); the DCP-4 kernel on the unsplit pool (wrong
    rows for 3 of every 4 B-spans) is caught."""
    from vllm_torchtpu.models.vllm.qwen4_exp import pool_layout

    base = _run_c(**D4_KW)
    monkeypatch.setenv(pool_layout.MAIN_KV_DCP_ENV, "2")
    two = _run_c(**D4_KW, dcp=2)
    monkeypatch.setenv(pool_layout.MAIN_KV_DCP_ENV, "4")
    _d4_close(_run_c(**D4_KW, dcp=4), two, base)
    with pytest.raises(AssertionError):
        _d4_close(_run_c(**D4_KW, dcp=4, dcp_pool=False), two, base)


def test_dcp4_refuses_compressed_keys_and_the_per_row_path():
    """q38 L1-D14: the DCP-4 group block (`QSAFullStepGroupBlock`, picked by `wholestep_spec` when dcp > 2) is union
    attention over the main KV only: it builds at DCP 4 with union and overrides MM's three DCP methods; compressed-key
    DCP (pair-only), DCP 4 without union and DCP 8 are refused, and MM's own block still refuses DCP 4."""
    import inspect

    from vllm_torchtpu.megakernel.wholestep import spec as ws
    from vllm_torchtpu.megakernel.wholestep.qsa_full_c import QSAFullStepBlock, QSAFullStepGroupBlock

    base = dict(max_groups=MAX_GROUPS, pages_per_seq=32, carrier="bf16", chunk_pages=8, hbm_tables=True,
                per_request_scores=True, bounded_chunks=True, shard_index=True, budget=BUDGET)
    blk = QSAFullStepGroupBlock(**base, union_attn=True, dcp=4)
    assert blk.dcp == 4
    for name in ("_dcp_scratch", "_dcp_attend", "_dcp_combine"):
        assert getattr(QSAFullStepGroupBlock, name) is not getattr(QSAFullStepBlock, name), name
    for cls, bad in ((QSAFullStepGroupBlock, dict(union_attn=True, dcp=4, compressed_dcp=True)),
                     (QSAFullStepGroupBlock, dict(union_attn=False, dcp=4)),
                     (QSAFullStepGroupBlock, dict(union_attn=True, dcp=8)),
                     (QSAFullStepBlock, dict(union_attn=True, dcp=4))):
        with pytest.raises(AssertionError):
            cls(**base, **bad)
    assert 'QSAFullStepGroupBlock if int(indexed.get("dcp", 1) or 1) > 2 else QSAFullStepBlock' in inspect.getsource(ws)


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


@pytest.mark.parametrize("dcp", [1, 2, 4])
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
    if dcp > 2:  # q38 L1-D14: one R = 4 DCP group, four requests
        kw.update(R=4, nreq=4, ctx=CTX_D4)
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

    def broken(code, KU, consts, **kw):  # noqa: ANN001
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


def test_tail_greedy_tokens_match_reference():
    """Final HC mixer + lm_head + per-rank top token + argmax over ranks = the stock
    greedy
    verify (argmax of the full logits, ties to the lower index)."""
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

    R, T, V = 2, 4, 384
    ex = ExecSpec(R=R, B=2 * T, T=T, ctx=128, interpret=True, tiles=_TILES)
    k = jax.random.split(jax.random.PRNGKey(5), 3)
    hc = _hc_params(k[0], G, 1)
    lm = (jax.random.normal(k[1], (R, G.H, V), F32) * 0.05).astype(BF16)
    x = jax.random.normal(k[2], (ex.B, G.NHC * G.H), F32).astype(BF16)
    want, logits = tail_reference(
        x, {n: v[0] for n, v in hc.items()}, lm, nhc=G.NHC, eps=G.eps
    )
    spec = tail_spec(G, R, vocab_shard=V)
    from vllm_torchtpu.megakernel.models.qwen38_pack import pack_weights

    w = {"lm_head": lm[:, None], **hc_weight_arrays("tail", hc, G, R, mix=True)}
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


# ======================================================================== q38 L1-D08S1
# D08 package S1 "UNION-LEAN" (wholestep/qsa_full_lean.py) on the union path: B1 union compaction bounded by the union
# count, B2 rowsel without per-row ids + the selection exchange folded into the union exchange (each rank receives its
# own DCP half), B3 per-layer batched completed groups / recent tiles + double-buffered union ids. Each stage must be
# bitwise identical to MM's / D14's block: the outputs AND the union state attention reads (`union_probe` dumps it on
# both sides). Scenarios cover DCP 1 / 2 / 4, 4-64 rows, padded slots, n_vis <= K rows, tie-heavy scores, g_a
# selected and unselected, ranks that own no request (R = 8 at 4 rows), and a ring that pins the f32 sum order.


class _s1_pool_layout:  # noqa: N801  (lazy, as the other tests: the vLLM model package imports on use)
    MAIN_KV_DCP_ENV = "VLLM_TPU_QSA_MAIN_KV_DCP"

S1_CONFIGS = {
    "OFF": dict(union_bounded=False, union_lean=False, slot_batch=False),  # the lean class, every stage off
    "B1": dict(union_bounded=True),
    "B2": dict(union_lean=True),
    "B3": dict(slot_batch=True),
    "ALL": dict(union_bounded=True, union_lean=True, slot_batch=True),
}
S1_PROBE = dict(union_probe=True)
S1_T = 4


def _s1_ties(ck):
    """Tie-heavy compressed keys: period 3 groups, so the top-K boundary sits inside runs of equal scores."""
    return ck[:, np.arange(ck.shape[1]) % 3]


def _s1_ring(ring, ctxq):
    """For every request whose completed group has three old members (ctx % 4 == 3) those ring rows are 2^24, 1,
    -2^24 in position order: MM's left-to-right f32 sum keeps the step's row exactly (2^24 + 1 rounds to 2^24), any
    other order does not, so the batched completed groups must keep MM's order to match."""
    r = np.array(jnp.asarray(ring, jnp.float32))
    for q, c0 in enumerate(np.asarray(ctxq).tolist()):
        if c0 % 4 == 3:
            for j, v in enumerate((2.0**24, 1.0, -(2.0**24))):
                r[q, (c0 - 3 + j) % 8, :] = v
    return jnp.asarray(r, BF16)


S1_SCEN = {
    # 8 rows (2 live requests, 2 padded slots), DCP 1, request loop, budget 512 (KU 2048: 8 compaction passes per
    # union, about 3 live), request 0 with its partial group in the pool, request 1 ctx % 4 == 3 (ring), union stats.
    "d1_r2_8": dict(R=2, nreq=2, dcp=1, loop=True, budget=512, ctx=[3001, 3003, 102, 255], ring=True, stats=True),
    # 32 rows, 8 requests over 2 ranks (4 each), unrolled, tie-heavy keys, budget 64.
    "d1_r2_32": dict(R=2, nreq=8, dcp=1, loop=False, budget=64, ctx=[3000, 100, 3003, 254], ties=True),
    # DCP 2, 16 rows, request loop, budget 512.
    "d2_r2_16": dict(R=2, nreq=4, dcp=2, loop=True, budget=512, ctx=[3001, 37, 1023, 254], ring=True),
    # DCP 2 over two pairs (R = 4), 64 rows unrolled (4 requests per rank), tie-heavy keys.
    "d2_r4_64": dict(R=4, nreq=16, dcp=2, loop=False, budget=64, ctx=CTX_D4, ties=True),
    # DCP 4 over two groups (R = 8), 4 rows: 1 live request + 3 padded slots, ranks 4-7 own no request; budget 128
    # (KU 512: 2 passes).
    "d4_r8_4": dict(R=8, nreq=1, dcp=4, loop=True, budget=128, ctx=[3003, 37, 1021, 254], ring=True),
    # DCP 4 (R = 4), 64 rows, request loop (4 requests per rank), tie-heavy keys.
    "d4_r4_64": dict(R=4, nreq=16, dcp=4, loop=True, budget=64, ctx=CTX_D4, ties=True),
}


def _s1_ku(budget: int) -> int:
    return -(-(S1_T * budget) // 128) * 128


def _s1_run(monkeypatch, scen: str, cfg: str | None, **over):
    """One kernel run of scenario `scen` with the probe: MM's / D14's block (`cfg` None) or the lean block."""
    kw = dict(S1_SCEN[scen])
    kw.update(over)
    dcp, ties, ring = kw.pop("dcp"), kw.pop("ties", False), kw.pop("ring", False)
    if dcp > 1:
        monkeypatch.setenv(_s1_pool_layout.MAIN_KV_DCP_ENV, str(dcp))
    else:
        monkeypatch.delenv(_s1_pool_layout.MAIN_KV_DCP_ENV, raising=False)
    lean = dict(S1_PROBE, **(S1_CONFIGS[cfg] if cfg else {}))
    kw.setdefault("shard", True)
    return _run_c(long=True, rank_ids=True, union=True, shared_kv=dcp > 1, dcp=dcp, lean=lean,
                  ck_fn=_s1_ties if ties else None, ring_fn=_s1_ring if ring else None, **kw)


def _s1_union(run: tuple, D: int, KU: int):
    """(qu_n [R, layers, requests, D], ids, codes [R, layers, requests, D KU], kvl, sga [R, layers, M], qi_z)."""
    pu, pr = run[8]["_probe_u"], run[8]["_probe_r"]
    t0 = 2 * D * KU
    return pu[..., t0 : t0 + D], pu[..., :D * KU], pu[..., D * KU : t0], pr[..., 0], pr[..., 1], pr[..., 128:]


def _s1_same(run: tuple, base: tuple, D: int, KU: int, what: str) -> None:
    """Bitwise: residual, ck_new, ring_new, index queries, kr_new / ir_new, union stats, and the union state
    attention reads on every rank (qu_n; this rank's half of the union ids and codes below its count; qi_kvl,
    qc_sga; qi_z zero)."""
    for i, name in ((0, "residual"), (3, "ck_new"), (4, "ring_new"), (6, "index queries")):
        np.testing.assert_array_equal(run[i], base[i], err_msg=f"{what}: {name}")
    for k in ("_kr_new", "_ir_new"):
        np.testing.assert_array_equal(run[8][k], base[8][k], err_msg=f"{what}: {k}")
    if "_stats" in base[8]:
        np.testing.assert_array_equal(run[8]["_stats"], base[8]["_stats"], err_msg=f"{what}: union stats")
    n_r, ids_r, cd_r, kvl_r, sga_r, z_r = _s1_union(run, D, KU)
    n_b, ids_b, cd_b, kvl_b, sga_b, z_b = _s1_union(base, D, KU)
    np.testing.assert_array_equal(n_r, n_b, err_msg=f"{what}: qu_n")
    R_, n_att, nq = n_b.shape[:3]
    for r in range(R_):
        h = r % D
        for o in range(n_att):
            for q in range(nq):
                a = slice(h * KU, h * KU + int(n_b[r, o, q, h]))
                np.testing.assert_array_equal(ids_r[r, o, q, a], ids_b[r, o, q, a],
                                              err_msg=f"{what}: union ids rank {r} layer {o} request {q}")
                np.testing.assert_array_equal(cd_r[r, o, q, a], cd_b[r, o, q, a],
                                              err_msg=f"{what}: union codes rank {r} layer {o} request {q}")
    np.testing.assert_array_equal(kvl_r, kvl_b, err_msg=f"{what}: qi_kvl")
    np.testing.assert_array_equal(sga_r, sga_b, err_msg=f"{what}: qc_sga")
    assert not z_b.any(), "MM's qi_z is not zero"
    assert not z_r.any(), (what, "qi_z not zero on ranks", sorted(set(np.nonzero(z_r)[0].tolist())))


def _s1_cover(base: tuple, scen: str) -> None:
    """The scenario exercises what it claims (printed, and asserted where it is the scenario's point)."""
    kw = S1_SCEN[scen]
    D, KU, R_ = kw["dcp"], _s1_ku(kw["budget"]), kw["R"]
    n, _, _, kvl, sga, z = _s1_union(base, D, KU)
    d = base[8]
    B = kw["nreq"] * S1_T
    nvis = np.asarray(d["nvis"])
    slots = max(16, -(-B // 16) * 16) // S1_T
    W = min(KU, 256)
    print(f"L1-D08S1 COVER {scen}: R {R_} DCP {D} rows {B} slots {slots} (padded {slots - kw['nreq']}) budget "
          f"{kw['budget']} KU {KU} passes/union {KU // W}; union n max {int(n.max())} (live passes max "
          f"{-(-int(n.max()) // W)}), n > {W} in {int((n > W).sum())} halves; g_a selected rows {int((sga[0] > 0).sum())}"
          f" / unselected {int((sga[0] == 0).sum())}; live rows with n_vis <= K {int((nvis[:B] <= kw['budget']).sum())}"
          f"; ranks owning no request {max(0, R_ - slots)}; ctx % 4 == 3 requests "
          f"{sum(1 for c in np.asarray(d['ctxq']).tolist() if c % 4 == 3)}", flush=True)
    assert n.max() > 0 and (sga[0] > 0).any() and (sga[0] == 0).any()
    if kw["budget"] >= 512:  # B1's point: a multi-pass union with live and skipped passes
        assert (n > W).any() and n.max() <= KU - W, (int(n.max()), W, KU)
    if scen == "d4_r8_4":
        assert R_ > slots


S1_IDENTITY = [(s, "each") for s in S1_SCEN] + [("d1_r2_8", "B1"), ("d1_r2_8", "B2"), ("d1_r2_8", "B3"),
                                               ("d4_r8_4", "B2")]


@pytest.mark.parametrize(("scen", "which"), S1_IDENTITY, ids=[f"{s}-{w}" for s, w in S1_IDENTITY])
def test_l1d08s1_identity(monkeypatch, scen, which):
    """Each D08 S1 stage alone, every stage on, and the lean class with every stage off (`each`), or one stage (the
    mutants' nodes): bitwise identical to MM's / D14's block on the scenario (`_s1_same`)."""
    kw = S1_SCEN[scen]
    D, KU = kw["dcp"], _s1_ku(kw["budget"])
    base = _s1_run(monkeypatch, scen, None)
    _s1_cover(base, scen)
    for cfg in (list(S1_CONFIGS) if which == "each" else [which]):
        _s1_same(_s1_run(monkeypatch, scen, cfg), base, D, KU, f"{scen} {cfg}")
        print(f"L1-D08S1 IDENTITY {scen} {cfg}: bitwise identical to MM's block", flush=True)


def _s1_rows_from_union(run: tuple, D: int, KU: int, T: int = S1_T):
    """Per (layer, row) the pool-path groups the union codes give each row (bit t of a slot's code = row t), from
    the rank holding each DCP half."""
    n, ids, cd, _, _, _ = _s1_union(run, D, KU)
    n_att, nq = n.shape[1], n.shape[2]
    out = {}
    for o in range(n_att):
        for q in range(nq):
            for t in range(T):
                s = set()
                for h in range(D):
                    m = int(n[h, o, q, h])
                    sl = slice(h * KU, h * KU + m)
                    s |= {int(g) for g, c in zip(ids[h, o, q, sl], cd[h, o, q, sl]) if (int(c) >> t) & 1}
                out[o, q * T + t] = s
    return out


@pytest.mark.parametrize("dcp", [1, 4])
def test_l1d08s1_step_selection_exact_on_its_own_queries(monkeypatch, dcp):
    """`test_step_selection_exact_on_its_own_queries` for D08 S1 (every stage on): B2 builds no per-row ids, so each
    row's selection is read back from the union codes (all DCP halves) and must equal the stock selection from the
    kernel's own queries and completed-group key, restricted to groups older than the step."""
    from vllm_torchtpu.megakernel.wholestep.indexer import score_rows, topk_reference

    kw = dict(R=2, nreq=2, ctx=CTX_C) if dcp == 1 else dict(R=4, nreq=4, ctx=CTX_D4)
    if dcp > 1:
        monkeypatch.setenv(_s1_pool_layout.MAIN_KV_DCP_ENV, str(dcp))
    run = _run_c(long=True, rank_ids=True, union=True, shard=True, loop=True, shared_kv=dcp > 1, dcp=dcp,
                 lean=dict(S1_PROBE, **S1_CONFIGS["ALL"]), **kw)
    _, _, _, ck, _, _, iq, _, d = run
    sets = _s1_rows_from_union(run, dcp, _s1_ku(BUDGET))
    T, checked = S1_T, 0
    for o in range(iq.shape[0]):
        for r in range(iq.shape[1]):
            q = r // T
            c0 = int(d["ctxq"][q])
            p3 = c0 + (3 - c0 % 4) % 4
            qr = jnp.asarray(iq[o, r].reshape(4, -1), BF16)
            st = np.array(score_rows(qr, d["ck"][q]))[0]
            st[p3 // 4] = np.asarray(
                score_rows(qr, jnp.broadcast_to(jnp.asarray(ck[o, q], BF16), (8, ck.shape[-1])))
            )[0, 0]
            n = int(d["nvis"][r])
            want = {i for i in topk_reference(st, n, BUDGET) if i < c0 // 4}
            assert sets[o, r] == want, (o, r, len(sets[o, r] ^ want))
            checked += 1
    print(f"L1-D08S1 SELECTION dcp {dcp}: {checked} rows exact from the union codes", flush=True)


def test_l1d08s1_sharded_indexer_matches_unsharded(monkeypatch):
    """`test_sharded_indexer_matches_unsharded` for D08 S1 on the union path: every stage on, with the sharded
    indexer (B2's folded exchange) and without it (B2 = no per-row ids only), bit-identical to MM's unsharded union
    kernel."""
    monkeypatch.delenv(_s1_pool_layout.MAIN_KV_DCP_ENV, raising=False)
    base = _run_c(long=True, rank_ids=True, union=True)
    for shard in (True, False):
        run = _run_c(long=True, rank_ids=True, union=True, shard=shard, lean=dict(S1_CONFIGS["ALL"]))
        for i, name in ((0, "residual"), (3, "ck_new"), (4, "ring_new")):
            np.testing.assert_array_equal(run[i], base[i], err_msg=f"shard {shard}: {name}")


@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_l1d08s1_union_stacked_rows_bit_identical(monkeypatch, dcp):
    """`test_union_stacked_rows_bit_identical` with every D08 S1 stage on."""
    from vllm_torchtpu.megakernel.wholestep import union_attn as ua

    kw = dict(long=True, rank_ids=True, union=True, shard=True, shared_kv=True, lean=dict(S1_CONFIGS["ALL"]))
    if dcp > 1:
        monkeypatch.setenv(_s1_pool_layout.MAIN_KV_DCP_ENV, str(dcp))
        kw["dcp"] = dcp
    if dcp > 2:
        kw.update(R=4, nreq=4, ctx=CTX_D4)
    monkeypatch.setattr(ua, "STACK_ROWS", True)
    stacked = _run_c(**kw)
    monkeypatch.setattr(ua, "STACK_ROWS", False)
    per_row = _run_c(**kw)
    assert np.isfinite(stacked[0]).all()
    np.testing.assert_array_equal(stacked[0], per_row[0])


def test_l1d08s1_dcp2_union_attention_matches_replicated(monkeypatch):
    """`test_dcp2_union_attention_matches_replicated` with every D08 S1 stage on: close to MM's replicated union
    kernel, same row lengths and g_a flags; the lean DCP-2 kernel on the unsplit pool is caught."""
    monkeypatch.delenv(_s1_pool_layout.MAIN_KV_DCP_ENV, raising=False)
    base = _run_c(long=True, rank_ids=True, union=True, shard=True, shared_kv=True, lean=dict(S1_PROBE))
    scale = float(np.abs(base[1]).mean())
    lean = dict(S1_PROBE, **S1_CONFIGS["ALL"])

    def close(run: tuple) -> None:
        dd = np.abs(run[0] - base[0])
        assert np.isfinite(run[0]).all()
        assert dd.max() < 1e-2 * scale, dd.max() / scale
        for k in (3, 4):  # qi_kvl, qc_sga
            np.testing.assert_array_equal(_s1_union(run, 2, _s1_ku(BUDGET))[k], _s1_union(base, 1, _s1_ku(BUDGET))[k])

    monkeypatch.setenv(_s1_pool_layout.MAIN_KV_DCP_ENV, "2")
    close(_run_c(long=True, rank_ids=True, union=True, shard=True, shared_kv=True, dcp=2, lean=lean))
    with pytest.raises(AssertionError):
        close(_run_c(long=True, rank_ids=True, union=True, shard=True, shared_kv=True, dcp=2, dcp_pool=False,
                     lean=lean))


def test_l1d08s1_dcp4_union_attention_matches_replicated(monkeypatch):
    """`test_dcp4_union_attention_matches_replicated` (D14) with every D08 S1 stage on: within 1.25x MM's DCP-2
    error to the replicated union kernel (max and mean), same row lengths and g_a flags; the lean DCP-4 kernel on
    the unsplit pool is caught."""
    KU = _s1_ku(BUDGET)
    lean = dict(S1_PROBE, **S1_CONFIGS["ALL"])
    monkeypatch.delenv(_s1_pool_layout.MAIN_KV_DCP_ENV, raising=False)
    base = _run_c(**D4_KW, lean=dict(S1_PROBE))
    monkeypatch.setenv(_s1_pool_layout.MAIN_KV_DCP_ENV, "2")
    two = _run_c(**D4_KW, dcp=2)
    monkeypatch.setenv(_s1_pool_layout.MAIN_KV_DCP_ENV, "4")

    def close(run: tuple) -> None:
        scale = float(np.abs(base[1]).mean())
        e2, e4 = np.abs(two[0] - base[0]), np.abs(run[0] - base[0])
        print(f"L1-D08S1 DCP4 lean vs replicated: mean {e4.mean() / scale:.3e} max {e4.max() / scale:.3e}; MM DCP2: "
              f"mean {e2.mean() / scale:.3e} max {e2.max() / scale:.3e}", flush=True)
        assert np.isfinite(run[0]).all()
        assert e2.max() < 0.1 * scale and e2.mean() < 5e-3 * scale
        assert e4.max() <= 1.25 * e2.max() and e4.mean() <= 1.25 * e2.mean() + 1e-5 * scale
        for k in (3, 4):
            np.testing.assert_array_equal(_s1_union(run, 4, KU)[k], _s1_union(base, 1, KU)[k])

    close(_run_c(**D4_KW, dcp=4, lean=lean))
    with pytest.raises(AssertionError):
        close(_run_c(**D4_KW, dcp=4, dcp_pool=False, lean=lean))


def test_l1d08s1_request_loop_matches_unrolled(monkeypatch):
    """`test_request_loop_matches_unrolled` with every D08 S1 stage on (sharded indexer, union): the request loop
    and the unrolled requests give bit-identical outputs and union state."""
    monkeypatch.delenv(_s1_pool_layout.MAIN_KV_DCP_ENV, raising=False)
    lean = dict(S1_PROBE, **S1_CONFIGS["ALL"])
    base = _run_c(long=True, rank_ids=True, shard=True, union=True, lean=lean)
    loop = _run_c(long=True, rank_ids=True, shard=True, union=True, loop=True, lean=lean)
    _s1_same(loop, base, 1, _s1_ku(BUDGET), "loop vs unrolled")


def test_l1d08s1_union_stats_count_the_rows_selections(monkeypatch):
    """`test_union_stats_count_the_rows_selections` with every D08 S1 stage on: per QSA layer and request, `[union
    size, sum of the rows' pool-path groups, max row, rows with groups]` on every rank equals the counts read from
    the union codes (B2 builds no per-row ids)."""
    monkeypatch.delenv(_s1_pool_layout.MAIN_KV_DCP_ENV, raising=False)
    run = _run_c(long=True, rank_ids=True, union=True, shard=True, stats=True, lean=dict(S1_PROBE, **S1_CONFIGS["ALL"]))
    st = run[8]["_stats"]
    n, ids, cd, _, _, _ = _s1_union(run, 1, _s1_ku(BUDGET))
    for o in range(st.shape[1]):
        for q in range(len(CTX_C)):
            m = int(n[0, o, q, 0])
            codes = cd[0, o, q, :m].astype(np.int64)
            per_row = [int(((codes >> t) & 1).sum()) for t in range(S1_T)]
            want = [m, sum(per_row), max(per_row), sum(1 for x in per_row if x)]
            assert len(set(ids[0, o, q, :m].tolist())) == m  # a union lists each group once
            for r in range(st.shape[0]):
                assert st[r, o, q, :4].tolist() == want, (r, o, q, st[r, o, q, :4], want)


def test_l1d08s1_block_selection_and_scratch(monkeypatch):
    """No kernel run. `lean_step_block`: None without an S1 / probe key (MM's / D14's block is built as before),
    else the lean / probe class for the DCP size; the stages drop and add exactly their buffers; the lean block
    refuses compressed-key DCP, the non-union path and B3 with wholephys; the envs map to the options."""
    from vllm_torchtpu.megakernel.models import qwen38
    from vllm_torchtpu.megakernel.wholestep import qsa_full_lean as ql
    from vllm_torchtpu.megakernel.wholestep.qsa_full_c import QSAFullStepBlock, QSAFullStepGroupBlock
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc
    from vllm_torchtpu.models.vllm.qwen4_exp import pool_layout

    assert pool_layout.MAIN_KV_DCP_ENV == _s1_pool_layout.MAIN_KV_DCP_ENV
    base = dict(max_groups=65536, pages_per_seq=2048, carrier="fp8", n_layers=11, hbm_tables=True,
                per_request_scores=True, bounded_chunks=True, shard_index=True, union_attn=True, loop_requests=True)
    qsa = dict(budget=512)
    assert ql.lean_step_block(dict(base, dcp=2), qsa) is None
    for dcp, lean_cls, probe_cls, mm_cls in ((2, ql.QSAFullStepLeanBlock, ql.QSAFullStepProbeBlock, QSAFullStepBlock),
                                             (4, ql.QSAFullStepGroupLeanBlock, ql.QSAFullStepGroupProbeBlock,
                                              QSAFullStepGroupBlock)):
        assert type(ql.lean_step_block(dict(base, dcp=dcp, union_lean=True), qsa)) is lean_cls
        assert type(ql.lean_step_block(dict(base, dcp=dcp, union_bounded=False), qsa)) is lean_cls
        assert type(ql.lean_step_block(dict(base, dcp=dcp, union_probe=True, debug=True), qsa)) is probe_cls
        assert issubclass(lean_cls, mm_cls) and issubclass(probe_cls, mm_cls)
    ex = ExecSpec(R=8, B=64, T=4, ctx=128, tiles=_TILES)
    g = qwen38.FULL

    def names(**kw):
        blk = ql.lean_step_block(dict(base, dcp=4, **kw), qsa) or QSAFullStepGroupBlock(**base, dcp=4, **qsa)
        return {s.name for s in blk.scratch(g, ex, None)}

    mm = names()
    b2_gone = {"qi_id", "qi_ids", "qg_send", "qg_recv", "qg_ssem", "qg_rsem", "qu_id", "qu_cd", "qgu_send", "qgu_recv",
               "qgu_ssem", "qgu_rsem"}
    b3_gone = {"qc_ring", "qc_rope", "qc_ck", "qc_rnew", "qc_old", "qu_row"}
    assert names(union_bounded=True) == mm
    assert names(union_lean=True) == (mm - b2_gone) | {"qgl_send", "qgl_recv", "qgl_ssem", "qgl_rsem"}
    assert names(slot_batch=True) == (mm - b3_gone) | {"qb_ring", "qb_rope", "qb_ck", "qb_rnew", "qb_old", "qb_sem",
                                                       "qu_rowb", "qb_usem"}
    for bad in (dict(base, dcp=2, compressed_dcp=True, union_lean=True), dict(base, dcp=1, union_attn=False,
                union_lean=True), dict(base, dcp=1, union_phys=True, slot_batch=True)):
        with pytest.raises(AssertionError):
            ql.lean_step_block(bad, qsa)
    for e in ("VLLM_TPU_MK_UNION_BOUNDED", "VLLM_TPU_MK_UNION_LEAN", "VLLM_TPU_MK_SLOT_BATCH"):
        monkeypatch.delenv(e, raising=False)
    assert wsc._d08s1_options() == {}
    for key, e in wsc.D08S1_ENVS:
        monkeypatch.setenv(e, "1")
        want = {k: k == key for k, _ in wsc.D08S1_ENVS}
        assert wsc._d08s1_options() == want, (e, wsc._d08s1_options())
        assert {k: wsc.indexed_options()[k] for k in want} == want
        monkeypatch.delenv(e)
    print(f"L1-D08S1 SCRATCH b64 DCP4: MM {len(mm)} buffers; B2 drops {sorted(b2_gone & mm)}; B3 drops "
          f"{sorted(b3_gone & mm)}", flush=True)


# ======================================================================== q38 L2-UAPIPE
# X-UA-PIPE (union_attn.attend_request_pipe / pipe_request_loop, qsa_full_lean._PipeMixin) on the whole step: with
# `union_pipe` the outputs and the union state attention reads (`union_probe`) are bitwise those of the same block
# without it (the D08 S1 stage set given per node; "OFF" = the lean class with every stage off, None = MM's / D14's
# block). The base must call `attend_request` and never `attend_request_pipe`; the pipe run the reverse (engagement).
UAPIPE_SCEN = dict(S1_SCEN)
# DCP 4, 4 requests, budget 512 (multi-tile per-rank unions: the cross-request prefetch with m = 1 and 2).
UAPIPE_SCEN["d4_r4_16_b512"] = dict(R=4, nreq=4, dcp=4, loop=True, budget=512, ctx=CTX_D4, ring=True)
UAPIPE_NODES = [
    ("d4_r4_64", "ALL", 3), ("d4_r4_64", "ALL", 1), ("d4_r4_64", "ALL", 2),
    ("d4_r4_16_b512", "ALL", 3), ("d4_r4_16_b512", "ALL", 7),
    ("d1_r2_8", "ALL", 3), ("d2_r2_16", "ALL", 3), ("d4_r8_4", "OFF", 1),
]


def _uapipe_run(monkeypatch, scen: str, cfg: str | None, pipe: int):
    """`_s1_run` on UAPIPE_SCEN, with `union_pipe=pipe` in the lean options when nonzero; returns (run, calls)."""
    from vllm_torchtpu.megakernel.wholestep import union_attn as ua

    calls = {"base": 0, "pipe": 0, "loop_xreq": 0}
    orig_b, orig_p, orig_l = ua.attend_request, ua.attend_request_pipe, ua.pipe_request_loop

    def wb(**k):
        calls["base"] += 1
        return orig_b(**k)

    def wp(**k):
        calls["pipe"] += 1
        return orig_p(**k)

    def wl(**k):
        calls["loop_xreq"] += bool(k["pipe"] & ua.PIPE_XREQ)
        return orig_l(**k)

    monkeypatch.setattr(ua, "attend_request", wb)
    monkeypatch.setattr(ua, "attend_request_pipe", wp)
    monkeypatch.setattr(ua, "pipe_request_loop", wl)
    kw = dict(UAPIPE_SCEN[scen])
    dcp, ties, ring = kw.pop("dcp"), kw.pop("ties", False), kw.pop("ring", False)
    if dcp > 1:
        monkeypatch.setenv(_s1_pool_layout.MAIN_KV_DCP_ENV, str(dcp))
    else:
        monkeypatch.delenv(_s1_pool_layout.MAIN_KV_DCP_ENV, raising=False)
    lean = dict(S1_PROBE, **(S1_CONFIGS[cfg] if cfg else {}))
    if pipe:
        lean["union_pipe"] = pipe
    kw.setdefault("shard", True)
    run = _run_c(long=True, rank_ids=True, union=True, shared_kv=dcp > 1, dcp=dcp, lean=lean,
                 ck_fn=_s1_ties if ties else None, ring_fn=_s1_ring if ring else None, **kw)
    monkeypatch.setattr(ua, "attend_request", orig_b)
    monkeypatch.setattr(ua, "attend_request_pipe", orig_p)
    monkeypatch.setattr(ua, "pipe_request_loop", orig_l)
    return run, calls


@pytest.mark.parametrize(("scen", "cfg", "pipe"), UAPIPE_NODES, ids=[f"{s}-{c}-p{p}" for s, c, p in UAPIPE_NODES])
def test_l2uapipe_identity(monkeypatch, scen, cfg, pipe):
    """X-UA-PIPE `pipe` on the scenario's block (D08 S1 `cfg`): bitwise identical to the block without it."""
    kw = UAPIPE_SCEN[scen]
    D, KU = kw["dcp"], _s1_ku(kw["budget"])
    base, cb = _uapipe_run(monkeypatch, scen, cfg if cfg != "OFF" else None, 0)
    run, cp = _uapipe_run(monkeypatch, scen, cfg, pipe)
    n = _s1_union(base, D, KU)[0]
    R_ = n.shape[0]
    tiles = [[[-(-int(n[r, o, q, r % D]) * 4 // 256) for q in range(n.shape[2])] for o in range(n.shape[1])]
             for r in range(R_)]
    flat = [x for a in tiles for b in a for x in b]
    print(f"L2-UAPIPE WCOVER {scen} {cfg} pipe={pipe}: R {R_} DCP {D} requests {n.shape[2]} layers {n.shape[1]}; "
          f"per-rank tiles per request-layer (rank 0) {tiles[0]}; all ranks: empty {flat.count(0)}, 1-tile "
          f"{flat.count(1)}, >= 2 {sum(1 for x in flat if x >= 2)}, max {max(flat)}; calls base {cb} pipe {cp}",
          flush=True)
    assert cb["base"] > 0 and cb["pipe"] == 0, cb
    assert cp["pipe"] > 0 and cp["base"] == 0, cp
    assert (cp["loop_xreq"] > 0) == bool(pipe & 2), cp
    assert flat.count(0) + flat.count(1) > 0
    _s1_same(run, base, D, KU, f"{scen} {cfg} pipe={pipe}")
    print(f"L2-UAPIPE WIDENTITY {scen} {cfg} pipe={pipe}: bitwise identical to the block without X-UA-PIPE", flush=True)


def test_l2uapipe_block_selection_and_scratch(monkeypatch):
    """No kernel run. `pipe_step_block`: None without `union_pipe` (then the lean / MM / D14 selection is unchanged),
    else the pipe class on the lean class for the DCP size; (ii) swaps B3's two id rows for three (one semaphore per
    row) and refuses a block without B3's ids, and wholephys is refused; the env maps to the option (absent at 0)."""
    from vllm_torchtpu.megakernel.models import qwen38
    from vllm_torchtpu.megakernel.wholestep import qsa_full_lean as ql
    from vllm_torchtpu.megakernel.wholestep import union_attn as ua
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    base = dict(max_groups=65536, pages_per_seq=2048, carrier="fp8", n_layers=11, hbm_tables=True,
                per_request_scores=True, bounded_chunks=True, shard_index=True, union_attn=True, loop_requests=True)
    qsa = dict(budget=512)
    assert ql.pipe_step_block(dict(base, dcp=4, slot_batch=True), qsa) is None
    for dcp, cls, lean_cls in ((2, ql.QSAFullStepPipeBlock, ql.QSAFullStepLeanBlock),
                               (4, ql.QSAFullStepGroupPipeBlock, ql.QSAFullStepGroupLeanBlock)):
        for p in ua.PIPE_MODES:
            blk = ql.pipe_step_block(dict(base, dcp=dcp, slot_batch=True, union_pipe=p), qsa)
            assert type(blk) is cls and isinstance(blk, lean_cls) and blk.union_pipe == p
    ex = ExecSpec(R=8, B=64, T=4, ctx=128, tiles=_TILES)
    g = qwen38.FULL

    def names(**kw):
        blk = ql.pipe_step_block(dict(base, dcp=4, **kw), qsa) or ql.lean_step_block(dict(base, dcp=4, **kw), qsa)
        return {s.name: s for s in blk.scratch(g, ex, None)}

    b3 = names(slot_batch=True)
    for p in (1, 5):
        assert set(names(slot_batch=True, union_pipe=p)) == set(b3)
    for p in (2, 3, 7):
        got = names(slot_batch=True, union_pipe=p)
        assert set(got) == (set(b3) - {"qu_rowb", "qb_usem"}) | {"qx_rows", "qx_usem"}, sorted(set(got) ^ set(b3))
    assert set(names(union_pipe=1)) == set(names(union_bounded=False))
    for bad in (dict(base, dcp=4, union_pipe=2), dict(base, dcp=4, slot_batch=True, slot_batch_parts="newgrp,recent",
                union_pipe=3), dict(base, dcp=1, union_phys=True, union_pipe=1), dict(base, dcp=4, slot_batch=True,
                union_pipe=4), dict(base, dcp=4, slot_batch=True, union_pipe=6), dict(base, dcp=4, union_pipe=0)):
        with pytest.raises((AssertionError, ValueError)):
            ql.pipe_step_block(bad, qsa)
    monkeypatch.delenv("VLLM_TPU_MK_UNION_PIPE", raising=False)
    assert wsc._union_pipe_options() == {} and "union_pipe" not in wsc.indexed_options()
    for p in ua.PIPE_MODES:
        monkeypatch.setenv("VLLM_TPU_MK_UNION_PIPE", str(p))
        assert wsc._union_pipe_options() == {"union_pipe": p} and wsc.indexed_options()["union_pipe"] == p
    monkeypatch.setenv("VLLM_TPU_MK_UNION_PIPE", "0")
    assert wsc._union_pipe_options() == {}
    monkeypatch.setenv("VLLM_TPU_MK_UNION_PIPE", "6")
    with pytest.raises(ValueError):
        wsc._union_pipe_options()
    rows = names(slot_batch=True, union_pipe=3)["qx_rows"].spec
    print(f"L2-UAPIPE SCRATCH b64 DCP4: B3 {len(b3)} buffers; (ii) swaps qu_rowb / qb_usem for qx_rows "
          f"{getattr(rows, 'shape', rows)} / qx_usem", flush=True)
