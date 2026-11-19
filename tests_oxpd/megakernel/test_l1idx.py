# SPDX-License-Identifier: Apache-2.0
"""q38 L1-IDX: the in-kernel indexer package on the whole step (D14 + D08 S1), every lever default off.

* S2P (VLLM_TPU_MK_SCORE_CHUNK / _LANES / _MP + VLLM_TPU_MK_SELECT_RADIX): PD W's scorer knobs and the manager-page key
  table (qwen38_wholestep_call.score_options / _idx_step_inputs, qwen38_wholestep_inputs.idx_manager_table) and the
  radix-2 thresholds (wholestep/qsa_full_idx.thresholds_radix2).
* QSPLIT (VLLM_TPU_MK_UNION_QSPLIT 1 / 2): each DCP rank's union compacted from its own rows of the code array.
* TSLIM (VLLM_TPU_MK_TABLE_SLIM 1 / 2): slim main page table (pages_per_seq) and the double-buffered, prefetched
  per-layer SMEM windows.
* RING-S1 (VLLM_TPU_MK_RING_S1): per-bucket ring depths, only with every D08 S1 env on at main-KV DCP > 2.

Kernel tests run the whole step in CPU interpret mode with the union probe (tests/megakernel/test_wholestep.py
`_s1_same`: outputs AND the union state attention reads, bitwise) and build the indexed options exactly as served:
`_d08s1_options()` + `_idx_options()` from the envs (so the env -> option plumbing is under test too). Test scale:
max_groups 1024 (32 key pages per request), so the scorer's chunk 32 vs the test default 8 is the served 128 vs 32
(one chunk vs three at these contexts); `test_l1idx_identity[d4_lit-...]` runs the served values themselves (chunk 128
vs 32, lanes 128 vs 256, MP 4 vs 0) at max_groups 4096. QSPLIT needs 512-token main-pool blocks (128 groups per block
= one sublane row): `page512` regroups the test pool into 512-token pages before the DCP split.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from tests.megakernel import test_wholestep as tw

DCP_ENV = "VLLM_TPU_QSA_MAIN_KV_DCP"
S1 = {"VLLM_TPU_MK_UNION_BOUNDED": "1", "VLLM_TPU_MK_UNION_LEAN": "1", "VLLM_TPU_MK_SLOT_BATCH": "1"}
IDX_ENVS = ("VLLM_TPU_MK_SCORE_CHUNK", "VLLM_TPU_MK_SCORE_LANES", "VLLM_TPU_MK_SCORE_MP", "VLLM_TPU_MK_SELECT_RADIX",
            "VLLM_TPU_MK_UNION_QSPLIT", "VLLM_TPU_MK_TABLE_SLIM", "VLLM_TPU_MK_RING_S1")
S2P_ENV = {"VLLM_TPU_MK_SCORE_CHUNK": "32", "VLLM_TPU_MK_SCORE_LANES": "128", "VLLM_TPU_MK_SCORE_MP": "4"}
CFG = {
    "BASE": dict(S1),  # D14 + D08 S1 (all three stages): the base of every lever arm
    "MM": {},  # MM's / D14's block (no S1; the probe class)
    "B1": {"VLLM_TPU_MK_UNION_BOUNDED": "1"},  # S1 B1 alone
    "S2P": dict(S1, **S2P_ENV),
    "CH": dict(S1, VLLM_TPU_MK_SCORE_CHUNK="32"),
    "LN": dict(S1, VLLM_TPU_MK_SCORE_LANES="128"),
    "MP": dict(S1, VLLM_TPU_MK_SCORE_MP="4"),
    "R2": dict(S1, VLLM_TPU_MK_SELECT_RADIX="2"),
    "Q1": dict(S1, VLLM_TPU_MK_UNION_QSPLIT="1"),
    "Q2": dict(S1, VLLM_TPU_MK_UNION_QSPLIT="2"),
    "Q1B1": {"VLLM_TPU_MK_UNION_BOUNDED": "1", "VLLM_TPU_MK_UNION_QSPLIT": "1"},  # vs B1
    "Q1MM": {"VLLM_TPU_MK_UNION_QSPLIT": "1"},  # vs MM's compaction (L1-IDX class, S1 stages off)
    "T1": dict(S1, VLLM_TPU_MK_TABLE_SLIM="1"),
    "T2": dict(S1, VLLM_TPU_MK_TABLE_SLIM="2"),
    "ALL": dict(S1, **S2P_ENV, VLLM_TPU_MK_SELECT_RADIX="2", VLLM_TPU_MK_UNION_QSPLIT="1", VLLM_TPU_MK_TABLE_SLIM="2"),
    "ALLNP": dict(S1, **S2P_ENV, VLLM_TPU_MK_SELECT_RADIX="2", VLLM_TPU_MK_UNION_QSPLIT="1",
                  VLLM_TPU_MK_TABLE_SLIM="1"),  # ALL without the prefetch (scenarios without per-layer tables)
    "LIT0": dict(S1, VLLM_TPU_MK_SCORE_CHUNK="32"),  # served defaults: chunk 32, lanes 256, MP 0
    "LIT1": dict(S1, VLLM_TPU_MK_SCORE_CHUNK="128", VLLM_TPU_MK_SCORE_LANES="128", VLLM_TPU_MK_SCORE_MP="4"),
}
# env -> (indexed-block option, expected value) for the engagement check (written here, not read from the code)
ENV_OPT = {
    "VLLM_TPU_MK_SCORE_CHUNK": ("chunk_pages", int),
    "VLLM_TPU_MK_SCORE_LANES": ("ck_lanes", int),
    "VLLM_TPU_MK_SCORE_MP": ("manager_pages", int),
    "VLLM_TPU_MK_SELECT_RADIX": ("radix_bits", int),
    "VLLM_TPU_MK_UNION_QSPLIT": ("union_qsplit", int),
    "VLLM_TPU_MK_UNION_BOUNDED": ("union_bounded", lambda v: v == "1"),
    "VLLM_TPU_MK_UNION_LEAN": ("union_lean", lambda v: v == "1"),
    "VLLM_TPU_MK_SLOT_BATCH": ("slot_batch", lambda v: v == "1"),
}
SCEN = {
    # 4 rows: 1 live request + 3 padded slots at DCP 4 over two groups (R = 8; ranks 4-7 own no request); budget 128.
    "d4_r8_4": dict(R=8, nreq=1, dcp=4, loop=True, budget=128, ctx=[3003, 37, 1021, 254], ring=True, page512=True),
    # 16 rows at DCP 4 on test_wholestep's D4_KW inputs (page512 closeness check).
    "d4_r4_16s": dict(R=4, nreq=4, dcp=4, loop=True, budget=64, ctx=tw.CTX_D4, page512=True),
    # PG diagnosis: the same D4_KW inputs at DCP 2 and DCP 1 on the regrouped 512-token pool (KV shared as in D4_KW).
    "d2_r4_16s": dict(R=4, nreq=4, dcp=2, loop=True, budget=64, ctx=tw.CTX_D4, page512=True),
    "d1_r4_16s": dict(R=4, nreq=4, dcp=1, loop=True, budget=64, ctx=tw.CTX_D4, page512=True, shared_kv=True),
    # 16 rows at DCP 4, per-layer tables (prefetch), request loop.
    "d4_r4_16": dict(R=4, nreq=4, dcp=4, loop=True, budget=64, ctx=tw.CTX_D4, page512=True, per_layer=True),
    # 32 rows at DCP 4, unrolled requests, tie-heavy keys.
    "d4_r4_32": dict(R=4, nreq=8, dcp=4, loop=False, budget=64, ctx=tw.CTX_D4, ties=True, page512=True),
    # 64 rows at DCP 4 (4 requests per rank), request loop, tie-heavy keys.
    "d4_r4_64": dict(R=4, nreq=16, dcp=4, loop=True, budget=64, ctx=tw.CTX_D4, ties=True, page512=True),
    # DCP 2, 16 rows, per-layer tables, budget 512 (multi-pass unions), ring that pins the f32 sum order.
    "d2_r2_16": dict(R=2, nreq=4, dcp=2, loop=True, budget=512, ctx=[3001, 37, 1023, 254], ring=True, page512=True,
                     per_layer=True),
    # DCP 2 over two pairs, 64 rows unrolled, tie-heavy keys.
    "d2_r4_64": dict(R=4, nreq=16, dcp=2, loop=False, budget=64, ctx=tw.CTX_D4, ties=True, page512=True),
    # DCP 1, 8 rows (2 live + 2 padded slots), per-layer tables, budget 512, ring.
    "d1_r2_8": dict(R=2, nreq=2, dcp=1, loop=True, budget=512, ctx=[3001, 3003, 102, 255], ring=True, per_layer=True),
    # DCP 1, 32 rows unrolled, tie-heavy keys.
    "d1_r2_32": dict(R=2, nreq=8, dcp=1, loop=False, budget=64, ctx=[3000, 100, 3003, 254], ties=True),
    # The served scorer values at max_groups 4096 (128 key pages per request; chunk 32 = 3 chunks, 128 = 1).
    "d4_lit": dict(R=4, nreq=4, dcp=4, loop=True, budget=64, ctx=[12001, 37, 5021, 254], page512=True, max_groups=4096),
}


def _page512(d: dict) -> None:
    """Regroup the test main pool (128-token pages) into 512-token pages, in place, before the DCP split: page2
    `i * p2 + c + 1` of table row i holds, at rows [128 j, 128 j + 128), the 128-token page `mtab[i, 4 c + j]` (so
    token t of the row stays at column t // 512, row t % 512); page 0 is unused."""
    pool = np.asarray(d["pool"])  # [R, pages, 128, 1, 4, HD]
    mtab = np.asarray(d["mtab"])
    lead, pps = mtab.shape[:-1], mtab.shape[-1]
    assert pps % 4 == 0 and pool.shape[2] == 128, (pps, pool.shape)
    flat = mtab.reshape(-1, pps)
    n, p2 = flat.shape[0], pps // 4
    new = np.zeros((pool.shape[0], n * p2 + 1, 512) + pool.shape[3:], pool.dtype)
    mt2 = np.zeros((n, p2), np.int32)
    for i in range(n):
        for c in range(p2):
            pid = i * p2 + c + 1
            mt2[i, c] = pid
            for j in range(4):
                new[:, pid, 128 * j : 128 * (j + 1)] = pool[:, flat[i, 4 * c + j]]
    d["pool"] = jnp.asarray(new)
    d["mtab"] = jnp.asarray(mt2.reshape(lead + (p2,)))


def _idx_run(monkeypatch, scen: str, cfg: str, key: int = 17, split: bool = True):
    """One whole-step kernel run (CPU interpret, union probe on) of scenario `scen` under config `cfg`'s envs; the
    indexed options are the served ones (`_d08s1_options()` + `_idx_options()`) over the test geometry. Returns
    test_wholestep `_run_c`'s tuple (residual, ref, x0, ck_new, ring_new, steps, iq, ids, d)."""
    from vllm_torchtpu.megakernel import kernel
    from vllm_torchtpu.megakernel.blocks.hc import RESIDUAL_OUT
    from vllm_torchtpu.megakernel.exec_spec import ExecSpec
    from vllm_torchtpu.megakernel.models.qwen38_pack import shard_step_inputs
    from vllm_torchtpu.megakernel.wholestep import inputs as wi
    from vllm_torchtpu.megakernel.wholestep.qsa_full_b import DBG_IDS, DBG_IQ
    from vllm_torchtpu.megakernel.wholestep.qsa_full_c import CK_NEW, IR_NEW, KR_NEW, RING_NEW
    from vllm_torchtpu.megakernel.wholestep.qsa_full_lean import DBG_ROWS, DBG_UNION
    from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    kw = dict(SCEN[scen])
    R, nreq, dcp, T = kw["R"], kw["nreq"], kw["dcp"], 4
    for e in tuple(S1) + IDX_ENVS:
        monkeypatch.delenv(e, raising=False)
    for e, v in CFG[cfg].items():
        monkeypatch.setenv(e, v)
    if dcp > 1:
        monkeypatch.setenv(DCP_ENV, str(dcp))
    else:
        monkeypatch.delenv(DCP_ENV, raising=False)
    from vllm_torchtpu import envs

    MG = kw.get("max_groups", tw.MAX_GROUPS)
    K_ = kw["budget"]
    Gx = tw.G if R == 2 else dataclasses.replace(tw.G, NH=3 * R)
    if R > 4:
        Gx = dataclasses.replace(Gx, KH=R, VH=3 * R)
    ex = ExecSpec(R=R, B=nreq * T, T=T, ctx=128, interpret=True, tiles=tw._TILES)
    d = wi.random_wholestep_inputs(key, Gx, ex, tw.L, budget=K_, n_pages=tw.PAGES, identical_experts=True)
    d = wi.add_indexer_inputs(key, Gx, ex, d, tw.L, max_groups=MG, ctx=kw["ctx"])
    if kw.get("ties"):
        d["ck"] = tw._s1_ties(d["ck"])
    d = wi.add_step_inputs(key, Gx, ex, d, tw.L, ctx=kw["ctx"])
    if kw.get("ring"):
        d["ring"] = tw._s1_ring(d["ring"], d["ctxq"])
    if kw.get("per_layer"):
        d = wi.add_per_layer(key, Gx, ex, d, tw.L)
    mp = int(envs.VLLM_TPU_MK_SCORE_MP or 0)
    d = wi.add_carrier(key, d, mp=mp)  # MP: manager pages of MP contiguous kernel pages (inputs.add_carrier)
    d["hbm_tables"] = True
    if dcp > 1 or kw.get("shared_kv"):  # the DCP group shares one KV head: same K / V columns, same pool
        kv = 2 * Gx.HD
        d["qkvg"] = d["qkvg"].at[1:, ..., -kv:].set(
            jnp.broadcast_to(d["qkvg"][:1, ..., -kv:], d["qkvg"][1:, ..., -kv:].shape))
        d["pool"] = jnp.broadcast_to(d["pool"][:1], d["pool"].shape)
    ref, _ = wi.wholestep_reference(Gx, ex, d, tw.L)
    steps = d.pop("_step_out")
    if kw.get("page512"):
        _page512(d)
    if dcp > 1 and split:  # split=False: the negative control (the DCP layout not applied)
        tw._dcp_split(d, dcp)
    pps = d["mtab"].shape[-1]
    if str(envs.VLLM_TPU_MK_TABLE_SLIM) != "0":  # TSLIM: the served column count for this pool's block size
        cols = wsc._idx_pps(MG, SimpleNamespace(pool_shape=(0, int(d["pool"].shape[2]))))
        assert cols <= pps, (cols, pps)
        d["mtab"] = jnp.asarray(np.asarray(d["mtab"])[..., :cols])
        pps = cols
    n_att = tw.L // Gx.att_every
    indexed = dict(max_groups=MG, pages_per_seq=pps, debug=True, carrier="bf16", chunk_pages=8, hbm_tables=True,
                   per_request_scores=True, bounded_chunks=True, shard_index=True, union_attn=True,
                   union_stats=False, union_phys=False, loop_requests=kw["loop"], dcp=dcp, compressed_dcp=False,
                   union_probe=True)
    if kw.get("per_layer"):
        indexed["n_layers"] = n_att
    indexed.update(wsc._d08s1_options())
    indexed.update(wsc._idx_options())
    spec = wholestep_spec(Gx, R, L=tw.L, qsa=dict(budget=K_), step=True, indexed=indexed)
    w_mk, states = wi.pack_wholestep(spec, Gx, ex, d, tw.L)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states)
    step, _ = kernel.make_step(mesh, spec, ex)
    from vllm_torchtpu.megakernel.wholestep.qsa_full_idx import lean_step_block

    blk = lean_step_block(dict(indexed), dict(budget=K_))  # the class wholestep_spec built (same call)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *wi.extras_c(d, mesh, R))
    B = ex.B
    res = np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, -1)[:, :B], np.float32)
    ck = np.asarray(outs[CK_NEW], np.float32).reshape(R, n_att, nreq, -1)[0]
    ring = np.asarray(outs[RING_NEW], np.float32).reshape(R, n_att, nreq, 8, -1)[0]
    iq = np.asarray(outs[DBG_IQ], np.float32).reshape(R, n_att, ex.M, -1)[0, :, :B]
    ids = None  # per-row ids are not built under B2 (union_lean); the union probe carries the selection
    d["_kr_new"] = np.asarray(outs[KR_NEW], np.float32).reshape(R, n_att, ex.M, -1)[:, :, :B]
    d["_ir_new"] = np.asarray(outs[IR_NEW], np.float32).reshape(R, n_att, ex.M, -1)[:, :, :B]
    d["_probe_u"] = np.asarray(outs[DBG_UNION]).reshape(R, n_att, nreq, -1)
    d["_probe_r"] = np.asarray(outs[DBG_ROWS]).reshape(R, n_att, ex.M, -1)
    d["_block"] = type(blk).__name__ if blk is not None else "QSAFullStep(Group)Block"
    d["_opts"] = {k: indexed[k] for k in sorted(indexed) if k not in ("max_groups", "debug")}
    x0 = np.asarray(d["inp"].x[:B], np.float32)
    return res, np.asarray(ref, np.float32), x0, ck, ring, steps, iq, ids, d


def _ku(budget: int) -> int:
    return -(-(4 * budget) // 128) * 128


def _qcover(run: tuple, D: int, KU: int, scen: str) -> None:
    """The scenario's union halves are non-trivial on every DCP rank (QSPLIT's point), printed."""
    n = tw._s1_union(run, D, KU)[0]
    print(f"L1-IDX COVER {scen}: block {run[8]['_block']}; union n per DCP half: max {int(n.max())}, min over halves "
          f"with live requests {int(n[n > 0].min()) if (n > 0).any() else 0}, halves > 0: {int((n > 0).sum())} of "
          f"{n.size}", flush=True)
    assert n.max() > 0


# (scenario, base config, configs compared to it bitwise)
IDENTITY = [
    ("d4_r4_64", "BASE", ["S2P", "R2", "Q1", "Q2", "T1", "ALLNP"]),
    ("d4_r4_32", "BASE", ["S2P", "R2", "Q1", "ALLNP"]),
    ("d4_r4_16", "BASE", ["S2P", "R2", "Q1", "Q2", "T1", "T2", "ALL"]),
    ("d4_r8_4", "BASE", ["S2P", "R2", "Q1", "T1", "ALLNP"]),
    ("d2_r2_16", "BASE", ["S2P", "R2", "Q1", "T2", "ALL"]),
    ("d2_r4_64", "BASE", ["S2P", "R2", "Q1", "ALLNP"]),
    ("d1_r2_8", "BASE", ["S2P", "R2", "Q1", "T2", "ALL"]),
    ("d1_r2_32", "BASE", ["S2P", "R2", "ALLNP"]),
    # one lever per node (the mutants' clean controls; small and fast)
    ("d4_r8_4", "BASE", ["CH"]),
    ("d4_r8_4", "BASE", ["LN"]),
    ("d4_r8_4", "BASE", ["MP"]),
    ("d4_r4_32", "BASE", ["R2"]),
    ("d4_r8_4", "BASE", ["Q1"]),
    ("d4_r8_4", "BASE", ["Q2"]),
    ("d4_r8_4", "B1", ["Q1B1"]),
    ("d4_r8_4", "MM", ["Q1MM"]),
    ("d4_r4_16", "BASE", ["T2"]),
    ("d4_lit", "LIT0", ["LIT1"]),
]


@pytest.mark.parametrize(("scen", "base", "cfgs"), IDENTITY, ids=[f"{s}-{b}-{'+'.join(c)}" for s, b, c in IDENTITY])
def test_l1idx_identity(monkeypatch, scen, base, cfgs):
    """Each L1-IDX config is bitwise identical to its base on the scenario: residual, ck_new, ring_new, index queries,
    kr_new / ir_new and the union state attention reads on every rank (`test_wholestep._s1_same`)."""
    kw = SCEN[scen]
    D, KU = kw["dcp"], _ku(kw["budget"])
    b = _idx_run(monkeypatch, scen, base)
    print(f"L1-IDX BASE {scen} {base}: block {b[8]['_block']} options {b[8]['_opts']}", flush=True)
    _qcover(b, D, KU, scen)
    for cfg in cfgs:
        r = _idx_run(monkeypatch, scen, cfg)
        want_idx = any(k in CFG[cfg] for k in ("VLLM_TPU_MK_SELECT_RADIX", "VLLM_TPU_MK_UNION_QSPLIT")) or \
            CFG[cfg].get("VLLM_TPU_MK_TABLE_SLIM") == "2"
        assert (r[8]["_block"].endswith("IdxBlock")) == want_idx, (cfg, r[8]["_block"])  # engagement: the class
        for e, (opt, conv) in ENV_OPT.items():  # engagement: every env of the config reached its block option
            if e in CFG[cfg]:
                assert r[8]["_opts"].get(opt) == conv(CFG[cfg][e]), (cfg, e, opt, r[8]["_opts"].get(opt))
        tw._s1_same(r, b, D, KU, f"{scen} {cfg} vs {base}")
        print(f"L1-IDX IDENTITY {scen} {cfg} vs {base}: bitwise identical (block {r[8]['_block']}, options "
              f"{r[8]['_opts']})", flush=True)


def test_l1idx_page512_dcp4_close_to_replicated(monkeypatch):
    """The 512-token regrouping is a faithful layout (so QSPLIT's scenarios are real DCP-4 steps): the D14 + S1 kernel
    on it at DCP 4 is close to MM's replicated (DCP 1) union kernel on test_wholestep's D4_KW inputs, within D14's
    DCP-4 bar (1.25x MM's DCP-2 error, max and mean, + 1e-5 of scale on the mean), with the same row lengths and g_a
    flags; the regrouped pool without the DCP split (wrong tokens read) fails the same check."""
    sc = "d4_r4_16s"
    kw = SCEN[sc]
    assert kw["ctx"] == tw.D4_KW["ctx"] and kw["nreq"] == tw.D4_KW["nreq"] and kw["budget"] == tw.BUDGET
    for e in tuple(S1) + IDX_ENVS:
        monkeypatch.delenv(e, raising=False)
    monkeypatch.delenv(DCP_ENV, raising=False)
    ref1 = tw._run_c(**tw.D4_KW, lean=dict(tw.S1_PROBE))
    monkeypatch.setenv(DCP_ENV, "2")
    two = tw._run_c(**tw.D4_KW, dcp=2)
    KU = _ku(kw["budget"])
    scale = float(np.abs(ref1[1]).mean())
    e2 = np.abs(two[0] - ref1[0])

    def close(run: tuple, what: str) -> None:
        e4 = np.abs(run[0] - ref1[0])
        print(f"L1-IDX PAGE512 {what} vs replicated: mean {np.nanmean(e4) / scale:.3e} max {np.nanmax(e4) / scale:.3e}; "
              f"MM DCP2: mean {e2.mean() / scale:.3e} max {e2.max() / scale:.3e}", flush=True)
        assert np.isfinite(run[0]).all()
        assert e2.max() < 0.1 * scale and e2.mean() < 5e-3 * scale
        assert e4.max() <= 1.25 * e2.max() and e4.mean() <= 1.25 * e2.mean() + 1e-5 * scale
        for k in (3, 4):
            np.testing.assert_array_equal(tw._s1_union(run, 4, KU)[k], tw._s1_union(ref1, 1, KU)[k])

    close(_idx_run(monkeypatch, sc, "BASE"), "DCP4 S1")
    with pytest.raises(AssertionError):
        close(_idx_run(monkeypatch, sc, "BASE", split=False), "DCP4 S1 unsplit (control)")


# ------------------------------------------------------------------ unit tests (no kernel)
def _bits_rows(rng, R, G, kind):
    """[R, S, 128] int32 bit patterns of non-negative f32 scores (-1 = invisible) and per-row n_vis."""
    S = G // 128
    if kind == "rand":
        sc = rng.random((R, G), dtype=np.float32) * 4
    elif kind == "ties":
        sc = rng.integers(0, 5, (R, G)).astype(np.float32) * 0.25
    elif kind == "zeros":
        sc = np.zeros((R, G), np.float32)
    else:  # "wide": huge and tiny magnitudes, exact zeros
        sc = np.where(rng.random((R, G)) < 0.3, 0.0, np.exp(rng.normal(0, 20, (R, G)))).astype(np.float32)
        sc = np.minimum(sc, np.float32(3e38))
    nv = rng.integers(0, G + 1, R).astype(np.int32)
    nv[0], nv[-1] = G, min(G, 7)
    ids = np.arange(G, dtype=np.int32)
    raw = sc.view(np.int32).reshape(R, S, 128)  # unmasked: callers mask with their own n_vis
    return raw, nv.reshape(R, 1, 1), np.ascontiguousarray(np.broadcast_to(ids.reshape(1, S, 128), (R, S, 128)))


@pytest.mark.parametrize("G", [1024, 8192, 65536])
@pytest.mark.parametrize("kind", ["rand", "ties", "zeros", "wide"])
def test_l1idx_radix2_thresholds_identical(G, kind):
    """thresholds_radix2 == indexer.thresholds ((thr, tmax) per row, bitwise) over random, tie-heavy, all-zero and
    wide-magnitude scores, n_vis from 0 to G (rows with exactly K, K + 1 and <= K visible included), K 64 / 512, id
    bits even (1024, 65536) and odd (8192). Sensitivity: test_l1idx_radix2_controls_are_caught."""
    from vllm_torchtpu.megakernel.wholestep import indexer as ix
    from vllm_torchtpu.megakernel.wholestep import qsa_full_idx as qi

    rng = np.random.default_rng(G + len(kind))
    R = 8
    raw, nv, ids = _bits_rows(rng, R, G, kind)
    for K in (64, 512):
        nvk = nv.copy()
        nvk[1] = K  # exactly K visible
        nvk[2] = K + 1
        b2 = np.where(ids < nvk, raw, -1)
        want = jax.jit(ix.thresholds, static_argnums=(3, 4))(jnp.asarray(b2), jnp.asarray(ids), jnp.asarray(nvk), K, G)
        got = jax.jit(qi.thresholds_radix2, static_argnums=(3, 4))(jnp.asarray(b2), jnp.asarray(ids), jnp.asarray(nvk),
                                                                     K, G)
        for w, g_, name in zip(want, got, ("thr", "tmax")):
            np.testing.assert_array_equal(np.asarray(g_), np.asarray(w), err_msg=f"{name} G {G} {kind} K {K}")
        over = int((nvk > K).sum())
        ties = int((np.asarray(want[1]) < G).sum())
        print(f"L1-IDX RADIX2 G {G} {kind} K {K}: {R} rows identical ({over} over K, {ties} with a tie cut)", flush=True)


def test_l1idx_radix2_controls_are_caught():
    """The radix-2 comparison is sensitive: local copies of thresholds_radix2 that skip the last value pair (bits 2..1)
    on random rows, or keep the wrong candidate (c2 for c3) in the tie pass on tie-heavy rows, give different
    (thr, tmax)."""
    import inspect

    from vllm_torchtpu.megakernel.wholestep import indexer as ix
    from vllm_torchtpu.megakernel.wholestep import qsa_full_idx as qi

    src = inspect.getsource(qi.thresholds_radix2)
    G, K = 8192, 64
    for kind, old, new in (("rand", "for b in range(29, 0, -2):", "for b in range(29, 2, -2):"),
                           ("ties", "t = jnp.where(n3 < need, c3,", "t = jnp.where(n3 < need, c2,")):
        raw, nv, ids = _bits_rows(np.random.default_rng(5), 8, G, kind)
        nv = np.maximum(nv, 4 * K)  # every row over K
        bits = jnp.asarray(np.where(ids < nv, raw, -1))
        want = ix.thresholds(bits, jnp.asarray(ids), jnp.asarray(nv), K, G)
        assert src.count(old) == 1, old
        ns = dict(qi.__dict__)
        exec(compile(src.replace(old, new), "mut", "exec"), ns)  # noqa: S102
        got = ns["thresholds_radix2"](bits, jnp.asarray(ids), jnp.asarray(nv), K, G)
        assert any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(got, want)), (kind, new)


@pytest.mark.parametrize("S", [8, 16, 512])
@pytest.mark.parametrize("D", [2, 4])
def test_l1idx_qsplit_compaction_identical(S, D):
    """QSPLIT's compaction of one DCP rank's rows (strided rows, and the row-gather matmul of mode 2) == MM's
    `compact_union` of the masked full code (group g on rank (g // 128) % D: 512-token blocks), ids / codes / count
    bitwise, all passes and (B1) every live pass; random codes < 2^4, sparse and dense, unions up to KU. Controls: the
    rows of rank r + 1 and the remap skipped both differ."""
    from vllm_torchtpu.megakernel.wholestep import qsa_full_idx as qi
    from vllm_torchtpu.megakernel.wholestep import qsa_full_lean as ql
    from vllm_torchtpu.megakernel.wholestep import union_attn as ua

    rng = np.random.default_rng(S * 10 + D)
    KU = 2048 if S == 512 else 256
    gidx = np.arange(S * 128).reshape(S, 128)
    for dens in (0.01, 0.1, 0.9):
        code = np.where(rng.random((S, 128)) < dens, rng.integers(1, 16, (S, 128)), 0).astype(np.int32)
        for r in range(D):
            masked = np.where((gidx // 128) % D == r, code, 0)
            if int((masked > 0).sum()) > KU:  # a request's union never exceeds KU slots
                keep = np.flatnonzero(masked.ravel())[KU:]
                masked.ravel()[keep] = 0
                code.ravel()[keep] = 0
            want = ua.compact_union(jnp.asarray(masked), KU, ua.union_consts(S, KU))
            rows = jnp.asarray(code[r::D])
            got1 = qi.compact_union_rows(rows, KU, ua.union_consts(S // D, KU), D, r)
            got2 = qi.compact_union_rows(qi.quarter_rows(jnp.asarray(code), D, r), KU, ua.union_consts(S // D, KU), D, r)
            for got, mode in ((got1, "strided"), (got2, "gather")):
                for w, g_, name in zip(want, got, ("ids", "codes", "n")):
                    np.testing.assert_array_equal(np.asarray(g_), np.asarray(w), err_msg=f"S {S} D {D} r {r} {mode} {name}")
            # B1: every pass below the count, pass by pass
            consts_f, consts_q = ua.union_consts(S, KU), ua.union_consts(S // D, KU)
            pro_f, pro_q = ql._union_prologue(jnp.asarray(masked), consts_f), ql._union_prologue(rows, consts_q)
            n = int(np.asarray(pro_f[-1]).sum())
            assert n == int(np.asarray(pro_q[-1]).sum()) == int((masked > 0).sum())
            W = consts_f[2].shape[0]
            for j0 in range(0, KU, W):
                if n > j0:
                    a = ql._union_pass(pro_f, consts_f, j0)
                    b = qi._union_pass_rows(pro_q, consts_q, j0, D, r)
                    for x, y in zip(a, b):
                        np.testing.assert_array_equal(np.asarray(y), np.asarray(x))
            if n:
                bad_rows = qi.compact_union_rows(jnp.asarray(code[(r + 1) % D :: D]), KU, consts_q, D, r)
                bad_map = qi.compact_union_rows(rows, KU, consts_q, 1, 0)
                assert not np.array_equal(np.asarray(bad_rows[0]), np.asarray(want[0])) or \
                    not np.array_equal(np.asarray(bad_rows[1]), np.asarray(want[1]))
                if r or S // D > 1:
                    assert not np.array_equal(np.asarray(bad_map[0]), np.asarray(want[0]))
        print(f"L1-IDX QSPLIT S {S} D {D} density {dens}: {D} halves identical (strided, gather, B1 passes)", flush=True)


def _fake_inputs(monkeypatch, block_tokens: int, nreq: int = 2, max_groups: int = 1024, n_att: int = 2):
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_inputs as wsi

    monkeypatch.setattr(wsi, "read_ring_and_tags", lambda pool, ids, n, m: (jnp.zeros((len(ids), n), jnp.bfloat16), None))
    pool = jnp.zeros((64, block_tokens, 1, 4, 256), jnp.float8_e4m3fn)
    n_ck = max_groups // 32
    cols = -(-n_ck // (block_tokens // 128))
    rng = np.random.default_rng(block_tokens)
    tables, slots = [], []
    for o in range(n_att):
        comp = rng.integers(1, 60, size=(nreq, cols)).astype(np.int32)
        main = rng.integers(1, 60, size=(nreq, 64)).astype(np.int32)
        ring = rng.integers(1, 60, size=(nreq, 8)).astype(np.int32)
        tables.append((jnp.asarray(ring.reshape(-1)), jnp.asarray(main.reshape(-1)), jnp.asarray(comp.reshape(-1))))
        slots.append(((o + 1) % 2, o % 2))
    T = 4
    ctx = jnp.asarray([3000, 100][:nreq], jnp.int32)
    starts = jnp.arange(nreq + 1, dtype=jnp.int32) * T
    positions = jnp.concatenate([ctx[q] + jnp.arange(T) for q in range(nreq)]).astype(jnp.int32)
    trig = jnp.zeros((4096, 128), jnp.bfloat16)
    kw = dict(M=nreq * T, T=T, pages_per_seq=max_groups * 4 // 128, max_groups=max_groups, layer_major_tables=True)
    return pool, tables, slots, (trig, positions, starts, ctx), kw


@pytest.mark.parametrize("cslots", ["01", "10"])
def test_l1idx_manager_table_matches_w_and_is_contiguous(monkeypatch, cslots):
    """S2P's manager-page key table through the served call path (`_idx_step_inputs(qsa_step_inputs)` with
    VLLM_TPU_MK_SCORE_MP=4, 512-token pool): equals PD W's `pages[:, :n_ck:4] // 4` built independently from the stock
    page formula, and every manager id m stands for the 4 contiguous kernel pages 4m .. 4m + 3 that the per-page table
    lists (so one 4-page DMA reads exactly the pages the per-page path reads)."""
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc
    from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_inputs import carrier_pages_per_block, qsa_step_inputs

    for e in IDX_ENVS:
        monkeypatch.delenv(e, raising=False)
    pool, tables, slots, rest, kw = _fake_inputs(monkeypatch, 512)
    slots = [(m, int(c)) for (m, _), c in zip(slots, cslots)]
    plain = qsa_step_inputs(pool, tables, slots, *rest, **kw)
    assert wsc._idx_step_inputs(qsa_step_inputs) is qsa_step_inputs  # flags off: the function itself
    monkeypatch.setenv("VLLM_TPU_MK_SCORE_MP", "4")
    got = wsc._idx_step_inputs(qsa_step_inputs)(pool, tables, slots, *rest, **kw)
    n_ck, nreq = kw["max_groups"] // 32, kw["M"] // kw["T"]
    phys, logical = carrier_pages_per_block(pool), pool.shape[1] // 128
    for o, ((_, _, comp), (_, cs)) in enumerate(zip(tables, slots)):
        comp = np.asarray(comp).reshape(nreq, -1)
        stock = (comp[:, :, None] * phys + np.arange(logical) + cs * logical).reshape(nreq, -1)  # independent formula
        np.testing.assert_array_equal(np.asarray(got["qsa_cktab"][o]), stock[:, :n_ck:4] // 4)
        per_page = np.asarray(plain["qsa_cktab"][o])
        mgr = np.asarray(got["qsa_cktab"][o])
        np.testing.assert_array_equal(mgr[:, :, None] * 4 + np.arange(4), per_page.reshape(nreq, -1, 4))
    for k in got:
        if k != "qsa_cktab":
            np.testing.assert_array_equal(np.asarray(got[k]), np.asarray(plain[k]), err_msg=k)
    print(f"L1-IDX MPTABLE slots {cslots}: manager table == W's, {n_ck // 4} runs of 4 contiguous pages per request-layer",
          flush=True)


def test_l1idx_manager_table_refuses_bad_layouts(monkeypatch):
    """The trace-time asserts: 128- and 256-token pool blocks (1 / 2 kernel pages per block) cannot hold 4-page runs."""
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc
    from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_inputs import qsa_step_inputs

    for e in IDX_ENVS:
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("VLLM_TPU_MK_SCORE_MP", "4")
    for bt in (128, 256):
        pool, tables, slots, rest, kw = _fake_inputs(monkeypatch, bt)
        with pytest.raises(AssertionError, match="manager pages"):
            wsc._idx_step_inputs(qsa_step_inputs)(pool, tables, slots, *rest, **kw)


def test_l1idx_tslim_columns_and_wrapper(monkeypatch):
    """TSLIM's column count is max_groups // (groups per block x DCP) (served: 128 at DCP 4 with 512-token blocks, 256
    at DCP 2, 512 at DCP 1; 512 at DCP 4 with 128-token blocks); off it is max_groups * 4 // 128. The wrapper refuses a
    width that cannot address the cap and leaves every other input unchanged."""
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc
    from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_inputs import qsa_step_inputs

    for e in IDX_ENVS:
        monkeypatch.delenv(e, raising=False)
    lay = lambda bt: SimpleNamespace(pool_shape=(0, bt))  # noqa: E731
    for dcp in (1, 2, 4):
        monkeypatch.setattr(wsc, "main_kv_dcp", lambda d=dcp: d)
        assert wsc._idx_pps(65536, lay(512)) == 2048
    for v in ("1", "2"):
        monkeypatch.setenv("VLLM_TPU_MK_TABLE_SLIM", v)
        for dcp, bt, want in ((4, 512, 128), (2, 512, 256), (1, 512, 512), (4, 128, 512)):
            monkeypatch.setattr(wsc, "main_kv_dcp", lambda d=dcp: d)
            assert wsc._idx_pps(65536, lay(bt)) == want, (v, dcp, bt)
    monkeypatch.setenv("VLLM_TPU_MK_TABLE_SLIM", "1")
    monkeypatch.setattr(wsc, "main_kv_dcp", lambda: 4)
    pool, tables, slots, rest, kw = _fake_inputs(monkeypatch, 512)
    plain = qsa_step_inputs(pool, tables, slots, *rest, **kw)
    slim = dict(kw, pages_per_seq=wsc._idx_pps(kw["max_groups"], lay(512)))
    got = wsc._idx_step_inputs(qsa_step_inputs)(pool, tables, slots, *rest, **slim)
    assert got["qsa_mtab"].shape[-1] == slim["pages_per_seq"] == 2
    np.testing.assert_array_equal(np.asarray(got["qsa_mtab"]), np.asarray(plain["qsa_mtab"])[..., :2])
    with pytest.raises(AssertionError, match="TSLIM"):
        wsc._idx_step_inputs(qsa_step_inputs)(pool, tables, slots, *rest, **dict(kw, pages_per_seq=1))


def test_l1idx_options_routing_and_scratch(monkeypatch):
    """No kernel run. Flags off: no option, the D08 S1 routing unchanged (None / lean class), pps and step inputs as
    before. Each env gives its option; an L1-IDX kernel key selects the L1-IDX class for the DCP size (a subclass of the
    lean class); the prefetch doubles the SMEM windows; bad values are refused."""
    from vllm_torchtpu.megakernel.exec_spec import ExecSpec
    from vllm_torchtpu.megakernel.models import qwen38
    from vllm_torchtpu.megakernel.wholestep import qsa_full_idx as qi
    from vllm_torchtpu.megakernel.wholestep import qsa_full_lean as ql
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    for e in tuple(S1) + IDX_ENVS:
        monkeypatch.delenv(e, raising=False)
    assert wsc._idx_options() == {} and wsc.score_options() == {}
    assert not (set(wsc.indexed_options()) & ({"chunk_pages", "ck_lanes", "manager_pages"} | set(qi.IDX_KEYS)))
    base = dict(max_groups=65536, pages_per_seq=2048, carrier="fp8", n_layers=11, hbm_tables=True,
                per_request_scores=True, bounded_chunks=True, shard_index=True, union_attn=True, loop_requests=True)
    qsa = dict(budget=512)
    assert qi.lean_step_block(dict(base, dcp=4), qsa) is None
    assert type(qi.lean_step_block(dict(base, dcp=4, union_lean=True), qsa)) is ql.QSAFullStepGroupLeanBlock
    want = {"VLLM_TPU_MK_SCORE_CHUNK": ("128", {"chunk_pages": 128}), "VLLM_TPU_MK_SCORE_LANES": ("128", {"ck_lanes": 128}),
            "VLLM_TPU_MK_SCORE_MP": ("4", {"manager_pages": 4}), "VLLM_TPU_MK_SELECT_RADIX": ("2", {"radix_bits": 2}),
            "VLLM_TPU_MK_UNION_QSPLIT": ("2", {"union_qsplit": 2}), "VLLM_TPU_MK_TABLE_SLIM": ("2", {"table_prefetch": True})}
    for e, (v, opt) in want.items():
        monkeypatch.setenv(e, v)
        assert wsc._idx_options() == opt, (e, wsc._idx_options())
        monkeypatch.delenv(e)
    monkeypatch.setenv("VLLM_TPU_MK_TABLE_SLIM", "1")
    assert wsc._idx_options() == {}  # slimming is pages_per_seq only
    monkeypatch.delenv("VLLM_TPU_MK_TABLE_SLIM")
    for dcp, cls in ((4, qi.QSAFullStepGroupIdxBlock), (2, qi.QSAFullStepIdxBlock), (1, qi.QSAFullStepIdxBlock)):
        blk = qi.lean_step_block(dict(base, dcp=dcp, radix_bits=2, union_lean=True), qsa)
        assert type(blk) is cls and isinstance(blk, ql.QSAFullStepLeanBlock if dcp <= 2 else ql.QSAFullStepGroupLeanBlock)
        assert blk.radix_bits == 2 and blk.union_lean and not blk.union_bounded
    for bad in (dict(radix_bits=3), dict(union_qsplit=3), dict(table_prefetch=True, n_layers=0)):
        with pytest.raises(AssertionError):
            qi.lean_step_block(dict(base, dcp=4, **bad), qsa)
    ex = ExecSpec(R=8, B=64, T=4, ctx=128, tiles=tw._TILES)
    g = qwen38.FULL

    def scr(**kw):
        return {s.name: s for s in qi.lean_step_block(dict(base, dcp=4, manager_pages=4, **kw), qsa).scratch(g, ex, None)}

    a, b = scr(union_qsplit=1), scr(union_qsplit=1, table_prefetch=True)
    assert set(a) == set(b)
    for k in ("qi_ckw", "qi_mtw"):
        sa, sb = tuple(a[k].spec.shape), tuple(b[k].spec.shape)
        assert sb == (2,) + sa, (k, sa, sb)  # one whole leading slot per buffer (no tiled-dimension slicing)
    assert tuple(a["qi_ckw"].spec.shape) == (16, 65536 // 32 // 4)
    ex48 = ExecSpec(R=8, B=48, T=4, ctx=128, tiles=tw._TILES)  # 12 requests: slots stay whole leading indices
    b48 = {x.name: x for x in qi.lean_step_block(dict(base, dcp=4, manager_pages=4, union_qsplit=1, table_prefetch=True),
                                                  qsa).scratch(g, ex48, None)}
    assert tuple(b48["qi_ckw"].spec.shape) == (2, 12, 512) and tuple(b48["qi_mtw"].spec.shape) == (2, 12, 2048), (
        b48["qi_ckw"].spec.shape, b48["qi_mtw"].spec.shape)
    print(f"L1-IDX SCRATCH b64 DCP4 MP4: qi_ckw {tuple(a['qi_ckw'].spec.shape)} -> {tuple(b['qi_ckw'].spec.shape)}, "
          f"qi_mtw {tuple(a['qi_mtw'].spec.shape)} -> {tuple(b['qi_mtw'].spec.shape)}", flush=True)


def test_l1idx_ring_s1_rule(monkeypatch):
    """RING-S1: the per-bucket rings apply only at main-KV DCP > 2 with all three D08 S1 envs on; listed buckets only;
    VLLM_TPU_MK_RINGS still overrides; malformed specs are refused."""
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    for e in tuple(S1) + IDX_ENVS + ("VLLM_TPU_MK_RINGS",):
        monkeypatch.delenv(e, raising=False)

    def rings(dcp):
        monkeypatch.setattr(wsc, "main_kv_dcp", lambda: dcp)
        return {B: (wsc.wholestep_tiles(B, 65536).nbe, wsc.wholestep_tiles(B, 65536).nbd) for B in (8, 16, 32, 48, 64)}

    off4 = rings(4)
    monkeypatch.setenv("VLLM_TPU_MK_RING_S1", "32=7:10,64=6:4")
    assert rings(4) == off4  # S1 off: inactive
    for e, v in S1.items():
        monkeypatch.setenv(e, v)
    on4 = rings(4)
    assert on4[32] == (7, 10) and on4[64] == (6, 4), on4
    for B in (8, 16, 48):
        assert on4[B] == off4[B], (B, on4, off4)
    for dcp in (1, 2):
        monkeypatch.delenv("VLLM_TPU_MK_RING_S1")
        a = rings(dcp)
        monkeypatch.setenv("VLLM_TPU_MK_RING_S1", "32=7:10,64=6:4")
        assert rings(dcp) == a, dcp
    monkeypatch.setenv("VLLM_TPU_MK_SLOT_BATCH", "0")
    assert rings(4) == off4  # one S1 env off: inactive
    monkeypatch.setenv("VLLM_TPU_MK_SLOT_BATCH", "1")
    monkeypatch.setenv("VLLM_TPU_MK_RINGS", "5:5")
    assert set(rings(4).values()) == {(5, 5)}
    monkeypatch.delenv("VLLM_TPU_MK_RINGS")
    assert wsc.parse_ring_s1("48=8:4") == {48: "8:4"}
    # q38 L1 STACK-D v4: the served form uses '+' (DECODE_EXTRA_SERVER_ENV splits on ','); both forms parse the same.
    assert wsc.parse_ring_s1("32=7:10+48=8:4+64=6:4") == wsc.parse_ring_s1("32=7:10,48=8:4,64=6:4") == {
        32: "7:10", 48: "8:4", 64: "6:4"}
    for bad in ("32=7", "32:7:10", "x=1:1", "32=0:10", "32=7:10,32=6:10", "32=7:10+32=6:10", "32=7:10++64=6:4"):
        with pytest.raises(ValueError):
            wsc.parse_ring_s1(bad)
    print(f"L1-IDX RING-S1 DCP4: off {off4} -> on {on4}", flush=True)


def test_l1idx_generated_select_matches_lean():
    """qsa_full_idx._IdxMixin._select is the lean `_select` with exactly the pinned substitutions (the generator's): with
    them undone the two sources are equal line for line."""
    import inspect

    from vllm_torchtpu.megakernel.wholestep import qsa_full_idx as qi
    from vllm_torchtpu.megakernel.wholestep import qsa_full_lean as ql

    a = inspect.getsource(qi._IdxMixin._select)
    b = inspect.getsource(ql._LeanMixin._select)
    a_body = a.split('"""', 2)[2]
    b_body = b.split('"""', 2)[2]
    a_body = a_body.replace("self._idx_thresholds(bits, ids, nv, K, G)  # q38 L1-IDX S2P radix-2",
                            "ix.thresholds(bits, ids, nv, K, G)")
    head = ("                if self.union_qsplit and D > 1:  # q38 L1-IDX QSPLIT: each DCP rank's own rows of the code array\n"
            "                    self._qsplit_compact(ctx, lo // T, KU, npr)\n"
            "                else:\n")
    assert a_body.count(head) == 1
    i = a_body.index(head)
    j = a_body.index("\n\n        self.load_tables(ctx, l)", i)
    block = a_body[i + len(head):j]
    dedent = "".join(x[4:] if x.startswith("    ") else x for x in block.splitlines(keepends=True))
    a_body = a_body[:i] + dedent + a_body[j:]
    assert a_body == b_body


def test_l1idx_page512_exact_match_diagnosis(monkeypatch):
    """Diagnosis of PG's exact match (proof-i4: D14 + S1 at DCP 4 on 512-token pages bit-equal to the replicated DCP-1
    union kernel). Rules out a self-comparison and localises the cause. Runs, all on test_wholestep's D4_KW inputs:
    ref1 = MM's DCP-1 union kernel (128-token pages, the PG reference); two128 / four128 = DCP 2 / DCP 4 on 128-token
    pages; one512 / two512 = DCP 1 / DCP 2 on the regrouped 512-token pool; four512_s1 / four512_d14 = DCP 4 on it with
    and without S1. Prints each run's error vs ref1 and the per-request union size on every DCP half. Asserts only what
    must hold whatever the cause: the comparator is live (two128 differs from ref1), DCP 1 does not depend on the page
    size (one512 == ref1), S1 is bitwise D14 (four512_s1 == four512_d14), and the DCP-4 512-token run really split
    work over ranks (some request has groups on at least 2 DCP halves)."""
    KU = _ku(tw.BUDGET)
    for e in tuple(S1) + IDX_ENVS:
        monkeypatch.delenv(e, raising=False)
    monkeypatch.delenv(DCP_ENV, raising=False)
    ref1 = tw._run_c(**tw.D4_KW, lean=dict(tw.S1_PROBE))
    monkeypatch.setenv(DCP_ENV, "2")
    two128 = tw._run_c(**tw.D4_KW, dcp=2)
    monkeypatch.setenv(DCP_ENV, "4")
    four128 = tw._run_c(**tw.D4_KW, dcp=4)
    one512 = _idx_run(monkeypatch, "d1_r4_16s", "MM")
    two512 = _idx_run(monkeypatch, "d2_r4_16s", "MM")
    four512_d14 = _idx_run(monkeypatch, "d4_r4_16s", "MM")
    four512_s1 = _idx_run(monkeypatch, "d4_r4_16s", "BASE")
    scale = float(np.abs(ref1[1]).mean())
    for name, run in (("two128", two128), ("four128", four128), ("one512", one512), ("two512", two512),
                      ("four512_d14", four512_d14), ("four512_s1", four512_s1)):
        d = np.abs(run[0] - ref1[0])
        per_req = [float(d[:, 4 * q : 4 * q + 4].max() / scale) for q in range(tw.D4_KW["nreq"])]
        print(f"L1-IDX PGDIAG {name} vs ref1: mean {d.mean() / scale:.3e} max {d.max() / scale:.3e} differing "
              f"{int((d > 0).sum())} of {d.size}; per-request max {['%.2e' % x for x in per_req]}", flush=True)
    for name, run, D in (("two512", two512, 2), ("four512_s1", four512_s1, 4), ("four128 (probe-less: n/a)", None, 4)):
        if run is None:
            continue
        n = tw._s1_union(run, D, KU)[0]  # [R, layers, requests, D]
        print(f"L1-IDX PGDIAG {name} union size per DCP half (rank 0's view, layer 0): "
              f"{n[0, 0].tolist()}; layer 1: {n[0, 1].tolist()}", flush=True)
    assert np.abs(two128[0] - ref1[0]).max() > 0, "comparator dead: DCP 2 equals the reference"
    np.testing.assert_array_equal(one512[0], ref1[0], err_msg="DCP 1 depends on the page size")
    np.testing.assert_array_equal(four512_s1[0], four512_d14[0], err_msg="S1 is not bitwise D14 here")
    n4 = tw._s1_union(four512_s1, 4, KU)[0]
    assert ((n4[0] > 0).sum(axis=-1) >= 2).any(), "no request split over DCP ranks"
