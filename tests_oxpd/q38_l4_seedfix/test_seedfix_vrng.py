# SPDX-License-Identifier: Apache-2.0
"""q38 L4 SEEDFIX x VRNG-VPS x VFY-ONE (composition on q38-l4-samp2-vfy a05fdf823). CDK CPU pod only.

The composed tree's real VPS / VFY methods on SAMP2-BUILD's composed fake runner (tests/q38_l4_samp2v/test_samp2_vfy.py:
TP1 VPS head with a fake vocab shard, compiled programs -> eager bodies), with an input batch whose seeded requests
carry per-request generators:
  * VRNG on: a chunk holding a seeded request takes the base path (VRNG counts it `seeded`; no in-graph draw, the
    counter does not move); on that base path the seeded rows read their own pieces and the unseeded rows the shared
    draws, bitwise as with SEEDFIX off; a chunk with no seeded request still runs VRNG;
  * VFY-ONE (1 and 2): the VFY program takes the overlaid eager draws as operands, so a seeded request's tokens are
    the same through the VFY program, through the D83 chain (VFY cold) and with VRNG on or off, in every batch
    composition; value 2's seed stash still comes from the step's tokens;
  * unseeded rows (tokens, shared generator, VRNG counter, VFY stats) are bitwise unchanged flag on vs off.
"""

import os
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "q38_l4_samp2v"))

import test_samp2_vfy as SV  # noqa: E402

import vllm_torchtpu.runner.tpu_runner as tr  # noqa: E402
from vllm_torchtpu.runner import l4_samp2, seedfix as sf, vfy_one  # noqa: E402

K, V, HID = SV.K, SV.V, SV.HID
FLAG = "VLLM_TPU_SEEDED_SAMPLING"
rt = SV.rt  # the composed fixture (eager bodies, identity gather, VRNG log capture)


@pytest.fixture
def sfx(rt, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    sf._reset_for_tests()
    rec = {"vfy": [], "chain": [], "vfy_rng": [], "chain_rng": []}
    for name, attr, impl, ui in (("vfy", "_vfy_one_compiled", tr._vfy_one_impl, 8),
                                 ("chain", "_vps_verify_fb_compiled", tr._vps_verify_fb_impl, 7),
                                 ("vfy_rng", "_vfy_one_rng_compiled", tr._vfy_one_rng_impl, None),
                                 ("chain_rng", "_vps_verify_rng_compiled", tr._vps_verify_rng_impl, None)):
        def wrapped(*a, _n=name, _f=impl, _u=ui, **k):
            rec[_n].append(None if _u is None else tuple(a[_u + i].clone() for i in range(3)))
            return _f(*a, **k)
        monkeypatch.setattr(tr, attr, wrapped)
    return NS(rt=rt, mp=monkeypatch, rec=rec)


class Req:
    def __init__(self, rid, seed, logit_seed):
        self.rid, self.seed, self.logit_seed = rid, seed, logit_seed
        self.gen = torch.Generator().manual_seed(seed) if seed is not None else None

    def rows(self, step):
        g = torch.Generator().manual_seed(self.logit_seed * 7919 + step * 31 + 3)
        hid = torch.randn(K + 1, HID, generator=g).bfloat16()
        logits = SV.PROC._get_logits(hid, SV.HEAD, None, skip_gather=True)
        am = logits[:K].float().argmax(-1)
        draft = torch.where(torch.rand(K, generator=g) < 0.6, am, torch.randint(0, V, (K,), generator=g))
        return hid, draft.to(torch.int32)


class SRunner(SV.Runner):
    """The composed fake runner plus an input batch (per-request generators, CPU params) and request states; every
    request in range (top_k 20, temperature 0.6): F = 0, as served."""

    def __init__(self, reqs, B, **kw):
        super().__init__(B, **kw)
        n = len(reqs)
        self.top_k = torch.full((256,), 20, dtype=torch.int32)
        self.temps = torch.full((256,), 0.6)
        self.vocab_size = V
        self.input_batch = NS(generators={i: r.gen for i, r in enumerate(reqs) if r.gen is not None}, num_reqs=n,
                              req_ids=[r.rid for r in reqs], temperature_cpu_tensor=self.temps[:n].clone(),
                              top_k_cpu_tensor=self.top_k[:n].clone(), top_p_cpu_tensor=torch.full((n,), 0.95))
        self.requests = {r.rid: NS(num_computed_tokens=96, num_tokens=100) for r in reqs}
        self.sched = NS(num_scheduled_tokens={r.rid: K + 1 for r in reqs})


def chunk_of(reqs, step, B, extra=8):
    """hidden [T, HID] + logits indices + md for the chunk (every request K drafts), padded to B requests."""
    n = len(reqs)
    N = n * (K + 1)
    hids, drafts = [], []
    for r in reqs:
        h, d = r.rows(step)
        hids.append(h)
        drafts.append(d)
    hidden = torch.cat(hids + [torch.zeros(extra + (B - n) * (K + 1), HID, dtype=torch.bfloat16)], 0)
    idx = torch.arange(B * (K + 1), dtype=torch.int32)
    rows = torch.arange(B * (K + 1)).view(B, K + 1)
    draft = torch.cat(drafts + [torch.zeros((B - n) * K, dtype=torch.int32)])
    md = NS(bonus_logits_indices=rows[:, K].contiguous().to(torch.int32),
            target_logits_indices=rows[:, :K].reshape(-1).contiguous().to(torch.int32),
            draft_token_ids=draft, draft_lengths=torch.tensor([K] * n + [0] * (B - n), dtype=torch.int32),
            draft_lengths_cpu=np.array([K] * n, dtype=np.int32),
            segment_ids=torch.cat([torch.arange(n).repeat_interleave(K), torch.full(((B - n) * K,), n)]).to(torch.int32),
            group_indices=torch.cat([torch.arange(K).repeat(n), torch.zeros((B - n) * K, dtype=torch.int64)]).to(torch.int32))
    assert N <= hidden.shape[0]
    return hidden, idx, md


def serve(r, reqs, B, step, path, gen):
    """One verify of the chunk on `path`: "vfy" (VFY program, warm), "cold" (VFY cold -> D83 chain on the step's
    logits) or "chain" (no VFY: `_sample_spec_verify_vps` on the local logits). SEEDFIX's plan is built as
    sample_tokens builds it."""
    hidden, idx, md = chunk_of(reqs, step, B)
    groups = SV.slot_groups(step, 2, 64, len(reqs))
    r._fused_read_offsets = (groups, len(reqs))
    r._fused_read_offsets_applied = False
    r._seed_plan = sf.build_plan(r, r.sched) if r.input_batch.generators else None
    SV.tick(r)  # q38 L4 AGREE port (v5 composition, V5COMP-BUILD): the engine step's clock tick (VRNG agrees on it)
    try:
        if path == "chain":
            local = SV.PROC._get_logits(hidden[idx.long()], SV.HEAD, None, skip_gather=True)
            return r._sample_spec_verify_vps(local, md, 0, len(reqs), gen, 0)[: len(reqs)]
        r._vfy.step_mode = "vfy"
        if path == "cold":
            r._vfy.warm = set()
        return r._sample_spec_verify_vps(vfy_one.VfyChunk(hidden, idx), md, 0, len(reqs), gen, 0)[: len(reqs)]
    finally:
        r._seed_plan = None


def make_runner(reqs, B, path, vrng, mode=1, spec=None):
    ctl = SV.mkctl() if vrng else None
    r = SRunner(reqs, B, vrng=ctl, mode=0 if path == "chain" else mode, spec=spec)
    if path != "chain":
        SV.warm_all(r)
    elif ctl is not None:
        ctl.warm = SV.AlwaysWarm()
    return r, ctl


def S_req():
    return Req("S", 4242, 777)


COMPS = [
    ("alone", lambda S: [S], 8),
    ("first", lambda S: [S] + [Req(f"a{i}", None, 10 + i) for i in range(4)], 8),
    ("last_seeded_nbrs", lambda S: [Req(f"b{i}", 900 + i if i % 2 else None, 20 + i) for i in range(5)] + [S], 8),
    ("middle_bigpad", lambda S: [Req("c0", None, 30), Req("c1", 901, 31), S, Req("c3", None, 33)], 16),
]


@pytest.mark.parametrize("vrng", [True, False])
@pytest.mark.parametrize("path", ["vfy", "cold", "chain"])
def test_seeded_reproducible_across_compositions_and_paths(sfx, path, vrng):
    """S's tokens over 4 steps are identical in every composition, on the VFY program, the VFY-cold D83 chain and the
    plain chain, with VRNG on or off; with VRNG on every chunk here holds S, so VRNG counts each call `seeded` and its
    counter never moves."""
    ref = getattr(test_seeded_reproducible_across_compositions_and_paths, "_ref", None)
    for name, mk, B in COMPS:
        S = S_req()
        reqs = mk(S)
        r, ctl = make_runner(reqs, B, path, vrng)
        gen = torch.Generator().manual_seed(5)
        b = [q.rid for q in reqs].index("S")
        toks = torch.stack([serve(r, reqs, B, s, path, gen)[b] for s in range(4)])
        if ref is None:
            ref = toks
            test_seeded_reproducible_across_compositions_and_paths._ref = ref
        assert torch.equal(toks, ref), (name, path, vrng, toks.tolist(), ref.tolist())
        if ctl is not None:
            assert ctl.stats["seeded"] == 4 and ctl.vrng_calls == 0, ctl.stats
    assert sf.counts()["rows_ovl"] > 0 and sf.counts()["tgt_ovl"] > 0
    print(f"\nSEEDFIX_VRNG_REPRO path={path} vrng={vrng} compositions={len(COMPS)} PASS")


def test_seeded_chunk_base_path_uniforms(sfx):
    """VRNG on, VFY 1: the seeded chunk runs the base VFY program (not the in-graph one) on eager draws whose seeded
    slices are the requests' pieces and whose unseeded slices equal the SEEDFIX-off draws."""
    out = {}
    for on in (True, False):
        sfx.mp.setenv(FLAG, "1" if on else "0")
        for k in sfx.rec:
            sfx.rec[k].clear()
        reqs = [Req("u0", None, 1), Req("s1", 501, 2), Req("u2", None, 3), Req("s3", 503, 4)]
        r, ctl = make_runner(reqs, 8, "vfy", True)
        gen = torch.Generator().manual_seed(9)
        toks = serve(r, reqs, 8, 0, "vfy", gen)
        assert len(sfx.rec["vfy"]) == 1 and not sfx.rec["vfy_rng"] and ctl.stats["seeded"] == 1 and ctl.vrng_calls == 0
        out[on] = (toks, sfx.rec["vfy"][0], gen.get_state())
    (t1, (b1, a1, r1), g1), (t0, (b0, a0, r0), g0) = out[True], out[False]
    assert torch.equal(g1, g0)
    for b in (0, 2):
        assert torch.equal(t1[b], t0[b]) and torch.equal(b1[b], b0[b])
        assert torch.equal(a1[b * K:(b + 1) * K], a0[b * K:(b + 1) * K]) and torch.equal(r1[b * K:(b + 1) * K], r0[b * K:(b + 1) * K])
    for b, seed in ((1, 501), (3, 503)):
        bonus, acc, rec, _, _ = sf.draw_pieces(torch.Generator().manual_seed(seed), V, K, 0, SV.CPU)
        assert torch.equal(b1[b], bonus) and torch.equal(a1[b * K:(b + 1) * K], acc) and torch.equal(r1[b * K:(b + 1) * K], rec)
    print("\nSEEDFIX_VRNG_BASE seeded_chunk=base unseeded=bitwise PASS")


def test_unseeded_chunk_keeps_vrng(sfx):
    """A chunk with no seeded request runs VRNG even while the batch holds a seeded request in another chunk (the
    VRNG test is per chunk), and SEEDFIX draws nothing for it."""
    reqs = [Req("u0", None, 1), Req("u1", None, 2), Req("u2", None, 3)]
    r, ctl = make_runner(reqs, 8, "vfy", True)
    r.input_batch.generators = {7: torch.Generator().manual_seed(1)}   # a seeded request outside the chunk [0, 3)
    r.input_batch.num_reqs = 8
    r.input_batch.req_ids = [q.rid for q in reqs] + [f"o{i}" for i in range(5)]
    for i in range(5):
        r.requests[f"o{i}"] = NS(num_computed_tokens=96, num_tokens=100)
        r.sched.num_scheduled_tokens[f"o{i}"] = K + 1
    gen = torch.Generator().manual_seed(2)
    g0 = gen.get_state().clone()
    serve(r, reqs, 8, 0, "vfy", gen)
    assert ctl.stats["vrng"] == 1 and len(sfx.rec["vfy_rng"]) == 1 and torch.equal(gen.get_state(), g0)
    assert sf.counts()["rows_ovl"] == 0
    print("\nSEEDFIX_VRNG_KEEP unseeded_chunk=vrng PASS")


@pytest.mark.parametrize("mode", [1, 2])
def test_unseeded_bitwise_flag_on_off(sfx, mode):
    """VRNG on, VFY 1 / 2 (value 2 with its seed spec), 4 steps of a mixed chunk: unseeded rows' tokens, the shared
    generator, the VRNG counter, VFY's stats and value 2's seed stash on unseeded rows are bitwise flag on vs off."""
    res = {}
    for on in (True, False):
        sfx.mp.setenv(FLAG, "1" if on else "0")
        reqs = [Req("u0", None, 1), Req("s1", 601, 2), Req("u2", None, 3), Req("u3", None, 4), Req("s4", 604, 5)]
        spec = vfy_one.VfySeedSpec(SV.seed_body, (torch.tensor(5),), (torch.tensor(5),)) if mode == 2 else None
        r, ctl = make_runner(reqs, 8, "vfy", True, mode=mode, spec=spec)
        gen = torch.Generator().manual_seed(3)
        toks = [serve(r, reqs, 8, s, "vfy", gen) for s in range(4)]
        stash = r.__dict__.get("_vfy_seed")
        res[on] = (toks, gen.get_state(), ctl.counter_word(), dict(ctl.stats), r._vfy.stats.copy() if hasattr(r._vfy.stats, "copy") else dict(r._vfy.stats), stash)
    (t1, g1, c1, s1, v1, st1), (t0, g0, c0, s0, v0, st0) = res[True], res[False]
    assert torch.equal(g1, g0) and c1 == c0 and s1 == s0 and v1 == v0
    for a, b in zip(t1, t0):
        assert torch.equal(a[[0, 2, 3]], b[[0, 2, 3]])
    assert any(not torch.equal(a[[1, 4]], b[[1, 4]]) for a, b in zip(t1, t0))
    if mode == 2:
        assert st1 is not None and st0 is not None
    print(f"\nSEEDFIX_VRNG_UNSEEDED vfy_mode={mode} PASS")


def test_partial_seeded_row_composed(sfx):
    """A partial seeded request in the chunk: VRNG still routes the chunk to the base path (as without SEEDFIX), the
    partial row draws nothing, and every row's tokens equal SEEDFIX off (no request samples from its generator)."""
    res = {}
    for on in (True, False):
        sfx.mp.setenv(FLAG, "1" if on else "0")
        reqs = [Req("u0", None, 1), Req("p1", 701, 2), Req("u2", None, 3)]
        r, ctl = make_runner(reqs, 8, "vfy", True)
        r.requests["p1"] = NS(num_computed_tokens=10, num_tokens=1000)
        p_state = reqs[1].gen.get_state().clone()
        gen = torch.Generator().manual_seed(4)
        toks = serve(r, reqs, 8, 0, "vfy", gen)
        assert ctl.stats["seeded"] == 1 and torch.equal(reqs[1].gen.get_state(), p_state)
        res[on] = toks
    assert torch.equal(res[True], res[False])
    print("\nSEEDFIX_VRNG_PARTIAL PASS")
