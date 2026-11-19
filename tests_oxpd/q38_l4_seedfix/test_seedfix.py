# SPDX-License-Identifier: Apache-2.0
"""q38 L4 SEEDFIX (VLLM_TPU_SEEDED_SAMPLING; runner/seedfix.py). CDK CPU pod only.

The runner's real sample_tokens and verify methods run on a fake runner (sf_fakes.py; compiled programs -> eager
bodies, TP1). Requests have fixed logits per step, so a seeded request's tokens depend only on its RNG.
  * admission: TpuPlatform.validate_request accepts a seeded request with the flag on (default) and raises base's
    ValueError with it off;
  * unseeded rows are bitwise unchanged (tokens, every uniform they read, the shared generator's state), in
    all-unseeded and mixed batches, on every route (VPS full with and without fallback rows, VPS candidate draws with
    fallback rows, stock separate, fused, non-spec, spec-step non-verify);
  * a seeded request's tokens are the same across batch compositions (neighbours, position, padding buckets, chunking,
    F, neighbours' draft counts) and across the full-width routes; another seed gives other tokens; its uniforms sit
    exactly in its slots; 8 simulated ranks agree;
  * partial (chunked-prefill) seeded rows draw nothing and are not rewound; flag off is base;
  * the overlay's eager programs are all warmed: after warm-up, serving mixed batches at every steady bucket issues
    no new eager key (census);
  * the seeded path's accept / reject outcomes match the analytic target (accepted-length chi2 < 11.34, |z| < 3).
"""

import ast
import inspect
import math
import os
import sys
import textwrap
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sf_fakes as F  # noqa: E402

import vllm_torchtpu.runner.tpu_runner as tr  # noqa: E402
from vllm_torchtpu.compilation import tpu_compiler  # noqa: E402
from vllm_torchtpu.layers.adapter.sample import vp_sampler as vp  # noqa: E402
from vllm_torchtpu.platforms import tpu_platform  # noqa: E402
from vllm_torchtpu.runner import seedfix as sf  # noqa: E402
from vllm_torchtpu.runner import vps_wiring  # noqa: E402

K = F.K
TV = F.TV
FLAG = "VLLM_TPU_SEEDED_SAMPLING"


# ---- fixtures -------------------------------------------------------------------------------------------------------
@pytest.fixture
def rt(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("VLLM_TPU_L6_", "VLLM_TPU_L4_", "VLLM_TPU_VPS_")) or k == FLAG:
            monkeypatch.delenv(k)
    F.patch_all(monkeypatch)
    F.clear_rec()
    sf._reset_for_tests()
    yield monkeypatch
    F.clear_rec()


def flag(mp, on):
    mp.setenv(FLAG, "1" if on else "0")


def R(rid, seed=None, **kw):
    return F.Req(rid, seed, **kw)


def mixed(tag="m", n=6, seeded=(1, 4), seed0=1000, range_rows=(), d=None, logit0=0):
    """n requests; `seeded` positions carry seeds seed0 + i; `range_rows` carry top_k 0 (fallback rows)."""
    out = []
    for i in range(n):
        out.append(R(f"{tag}{i}", seed0 + i if i in seeded else None, top_k=0 if i in range_rows else 20,
                     temp=[0.8, 0.6, 1.0][i % 3], top_p=[0.95, 1.0, 0.9][i % 3], logit_seed=logit0 + i,
                     d=K if d is None else d[i]))
    return out


def runner_for(route):
    if route in ("vps_fb", "vps_nofb"):
        return F.Runner(vps=True, fb_max=8)
    if route == "vps_cand":
        return F.Runner(vps=True, cand=True, fb_max=8)
    if route == "vps_plainprog":
        return F.Runner(vps=True, fb_max=0)
    return F.Runner(vps=False)


ROUTE_RUN = {"vps_fb": "vps", "vps_nofb": "vps", "vps_cand": "vps", "vps_plainprog": "vps", "stock": "stock",
             "plain": "plain"}


def steps(runner, make_chunks, route, n_steps, spec=True, pads=None, plain_chunks=()):
    outs, routes = [], []
    for s in range(n_steps):
        o, r = F.run_step(runner, make_chunks(), route=ROUTE_RUN.get(route, route), spec=spec, step=s, pads=pads,
                          plain_chunks=plain_chunks)
        outs.append(torch.cat([x.reshape(x.shape[0], -1) for x in o], 0))
        routes.append(tuple(r))
    return outs, routes


# ---- import origin, flag, admission ----------------------------------------------------------------------------------
def test_imports_the_tree_under_test():
    root = os.path.realpath(os.environ.get("Q38_SEEDFIX_EXPECT_ROOT") or os.path.join(HERE, "..", "..", "src"))
    for mod in (tr, sf, vp, tpu_platform, tpu_compiler, vps_wiring):
        where = os.path.realpath(mod.__file__)
        assert where.startswith(root + os.sep), f"{mod.__name__} imported from {where}, not under {root}"


def test_flag_registered_default_on_host_only(monkeypatch):
    from vllm_torchtpu import envs

    assert FLAG in envs.environment_variables and FLAG in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    monkeypatch.delenv(FLAG, raising=False)
    assert envs.VLLM_TPU_SEEDED_SAMPLING is True and sf.enabled()
    for v, want in (("0", False), ("false", False), ("1", True), ("on", True)):
        monkeypatch.setenv(FLAG, v)
        assert sf.enabled() is want
    print("\nSEEDFIX_FLAG registered=1 ignored=1 default=on")


def test_admission_on_off(monkeypatch):
    from vllm.sampling_params import SamplingParams

    seeded, plain, greedy = (SamplingParams(temperature=1.0, seed=7), SamplingParams(temperature=1.0),
                             SamplingParams(temperature=0.0, seed=7))
    for v in (None, "1", "0"):
        if v is None:
            monkeypatch.delenv(FLAG, raising=False)
        else:
            monkeypatch.setenv(FLAG, v)
        tpu_platform.TpuPlatform.validate_request(None, plain)
        tpu_platform.TpuPlatform.validate_request(None, greedy)
        if v == "0":
            with pytest.raises(ValueError, match=r"^JAX does not support per-request seed\.$"):
                tpu_platform.TpuPlatform.validate_request(None, seeded)
        else:
            tpu_platform.TpuPlatform.validate_request(None, seeded)
        print(f"\nSEEDFIX_ADMISSION flag={v or 'unset'} seeded={'REJECT' if v == '0' else 'ACCEPT'}")


# ---- unseeded bitwise ------------------------------------------------------------------------------------------------
def _uniform_rows(rec_list, key, rows):
    """Concatenate, per recorded call, the rows of `key` NOT in `rows` (the unseeded slices)."""
    out = []
    for r in rec_list:
        t = r[key]
        if t is None:
            continue
        keep = [i for i in range(t.shape[0]) if i not in rows]
        out.append(t[keep])
    return out


def _seeded_slots(chunk, start=0):
    """(bonus rows, target slots) of the seeded requests of one chunk."""
    b_rows, t_slots, t = set(), set(), 0
    for b, r in enumerate(chunk):
        if r.seed is not None:
            b_rows.add(b)
            t_slots.update(range(t, t + r.d))
        t += r.d
    return b_rows, t_slots


def _fb_seeded_slots(runner, chunk, md, F_):
    if runner._vps_head is None or not runner._vps_cand_draws:
        return set()
    sb, st = sf.fallback_slots(runner, 0, len(chunk), md, torch.bfloat16, vp.VP_KC)
    _, t_slots = _seeded_slots(chunk)
    out = {sb[b] for b, r in enumerate(chunk) if r.seed is not None and b in sb}
    out |= {st[t] for t in t_slots if t in st}
    return {s for s in out if s < F_}


ROUTES = ["vps_fb", "vps_nofb", "vps_cand", "vps_plainprog", "stock", "plain"]


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("kind", ["all_unseeded", "mixed"])
def test_unseeded_bitwise(rt, route, kind):
    """Flag on vs flag off on the same steps: unseeded rows' tokens, every uniform slice they read and the shared
    generator's final state are bit-identical; with no seeded request everything is (and the plan is never built)."""
    spec = route != "plain"
    rng = (1,) if route in ("vps_fb", "vps_cand") else ()
    seeded = () if kind == "all_unseeded" else (0, 3, 5)
    built = []
    real = sf.build_plan
    rt.setattr(sf, "build_plan", lambda *a: built.append(1) or real(*a))
    res = {}
    for on in (False, True):
        flag(rt, on)
        F.clear_rec()
        runner = runner_for(route)
        chunk = mixed("u", 7, seeded=seeded, range_rows=rng, d=[3, 2, 3, 1, 3, 3, 0] if spec else None)
        outs, routes = steps(runner, lambda: [chunk], route, 4, spec=spec)
        res[on] = (outs, routes, {k: list(v) for k, v in F.REC.items()}, runner._sampling_generator.get_state())
    (o0, r0, rec0, g0), (o1, r1, rec1, g1) = res[False], res[True]
    assert r0 == r1
    assert torch.equal(g0, g1), "shared generator state differs"
    sb, st = _seeded_slots(chunk)
    rows = sorted(sb)
    for a, b in zip(o0, o1):
        keep = [i for i in range(a.shape[0]) if i not in sb]
        assert torch.equal(a[keep], b[keep])
    md0 = F.compose_spec(chunk, 0)[1] if spec else None
    for key_list, keys in (("vps", ("bonus_u", "accept_u", "recover_u", "fb_u")), ("fused", ("bonus_u", "accept_u", "recover_u")),
                           ("rs", ("accept_u", "recover_u")), ("sfl", ("u",))):
        assert len(rec0[key_list]) == len(rec1[key_list])
        for c0, c1 in zip(rec0[key_list], rec1[key_list]):
            for key in keys:
                a, b = c0.get(key), c1.get(key)
                if a is None:
                    assert b is None
                    continue
                if key in ("bonus_u", "u"):
                    excl = sb
                elif key == "fb_u":
                    excl = set()
                    if seeded:
                        rr = runner_for(route)
                        rr.set_batch(chunk)
                        excl = _fb_seeded_slots(rr, chunk, md0, a.shape[0])
                else:
                    excl = st
                keep = [i for i in range(a.shape[0]) if i not in excl]
                assert torch.equal(a[keep], b[keep]), (key_list, key)
    if kind == "all_unseeded":
        assert built == [] and sf.counts()["steps"] == 0
        assert all(torch.equal(a, b) for a, b in zip(o0, o1))
    else:
        assert sf.counts()["steps"] == 4 and sf.counts()["rows"] == 4 * len(seeded)
        assert any(not torch.equal(a[rows], b[rows]) for a, b in zip(o0, o1)), "seeded rows never changed"
    print(f"\nSEEDFIX_UNSEEDED route={route} kind={kind} routes={sorted(set(r1))} steps=4 PASS")


@pytest.mark.parametrize("kind", ["all_unseeded", "mixed"])
def test_unseeded_bitwise_fused(rt, kind):
    """The fused verify (VLLM_TPU_VERIFY_FUSED; TPU-only dispatch, called directly as _sample_spec_verify_chunk does):
    unseeded rows' tokens and uniform slices are bit-identical flag on vs off."""
    seeded = () if kind == "all_unseeded" else (1, 4)
    res = {}
    for on in (False, True):
        flag(rt, on)
        F.clear_rec()
        runner = F.Runner(vps=False)
        chunk = mixed("f", 6, seeded=seeded, d=[3, 3, 1, 3, 2, 3])
        outs = []
        for s in range(3):
            sched = runner.set_batch(chunk)
            logits, md = F.compose_spec(chunk, s)
            runner._seed_plan = sf.build_plan(runner, sched) if runner.input_batch.generators else None
            outs.append(runner._sample_spec_verify_fused(logits, md, 0, len(chunk), runner._get_sampling_generator())[:6])
            runner._seed_plan = None
        res[on] = (outs, list(F.REC["fused"]), runner._sampling_generator.get_state())
    sb, st = _seeded_slots(chunk)
    assert torch.equal(res[False][2], res[True][2])
    for a, b in zip(res[False][0], res[True][0]):
        keep = [i for i in range(6) if i not in sb]
        assert torch.equal(a[keep], b[keep])
    for c0, c1 in zip(res[False][1], res[True][1]):
        for key, excl in (("bonus_u", sb), ("accept_u", st), ("recover_u", st)):
            keep = [i for i in range(c0[key].shape[0]) if i not in excl]
            assert torch.equal(c0[key][keep], c1[key][keep])
    if seeded:
        assert any(not torch.equal(a[sorted(sb)], b[sorted(sb)]) for a, b in zip(res[False][0], res[True][0]))
    print(f"\nSEEDFIX_UNSEEDED route=fused kind={kind} PASS")


# ---- seeded reproducibility ------------------------------------------------------------------------------------------
def _S(seed=4242, d=K, top_k=20, top_p=0.95, temp=0.8):
    return R("S", seed, top_k=top_k, top_p=top_p, temp=temp, logit_seed=777, d=d)


def _nb(tag, n, seeded=(), range_rows=(), d=None, logit0=50, seed0=2000):
    return mixed(tag, n, seeded=seeded, range_rows=range_rows, d=d, logit0=logit0, seed0=seed0)


def _compositions(rng=False):
    """(name, builder S -> chunks, pads): the batch compositions S is placed in."""
    out = [
        ("alone", lambda S: [[S]], None),
        ("front", lambda S: [[S] + _nb("a", 5)], None),
        ("back", lambda S: [_nb("b", 5, logit0=60) + [S]], None),
        ("middle_seeded_nbrs", lambda S: [_nb("c", 3, seeded=(0, 2), logit0=70) + [S]
                                          + _nb("c2", 4, seeded=(1,), logit0=80, seed0=3000)], None),
        ("big_pad", lambda S: [_nb("e", 2, logit0=90) + [S] + _nb("e2", 3, logit0=95)], [(64, 128)]),
        ("nbr_drafts", lambda S: [_nb("g", 4, d=[0, 1, 2, 0], logit0=110) + [S] + _nb("g2", 2, d=[1, 3], logit0=120)],
         None),
        ("chunk2", lambda S: [_nb("h", 5, logit0=130), _nb("h2", 2, logit0=140) + [S]], None),
    ]
    if rng:
        out += [("range_nbr_before", lambda S: [_nb("r", 3, range_rows=(1,), logit0=150) + [S]], None),
                ("range_nbr_after", lambda S: [[S] + _nb("r2", 3, range_rows=(0,), logit0=160)], None)]
    return out


def _where(chunks, rid):
    for ci, c in enumerate(chunks):
        for b, r in enumerate(c):
            if r.rid == rid:
                return ci, b
    raise AssertionError(rid)


@pytest.mark.parametrize("route", ["vps_fb", "vps_nofb", "vps_plainprog", "stock", "plain", "vps_cand"])
def test_seeded_reproducible_across_compositions(rt, route):
    """S (seed 4242, fixed logits) gives the same tokens on every step in every batch composition: alone, first, last,
    between seeded and unseeded neighbours, a larger padding bucket, neighbours with other draft counts, the second
    chunk; with DRAWS=candidate also next to fallback-row neighbours (other F, other fallback slots) while S is itself
    a fallback-row request (top_k 0)."""
    cand = route == "vps_cand"
    S_kw = dict(top_k=0) if cand else {}
    ref = None
    seen_routes = set()
    comps = _compositions(rng=cand)
    for comp_name, mk, pads in comps:
        S = _S(**S_kw)
        chunks = mk(S)
        runner = runner_for(route)
        toks, rts = [], []
        for s in range(5):
            o, r = F.run_step(runner, chunks, route=ROUTE_RUN[route], spec=route != "plain", step=s, pads=pads)
            ci, b = _where(chunks, "S")
            toks.append(o[ci][b].clone())
            rts.append(r[ci])
        toks = torch.stack(toks)
        seen_routes.update(rts)
        if ref is None:
            ref = (comp_name, toks)
        else:
            assert torch.equal(toks, ref[1]), (comp_name, ref[0], toks.tolist(), ref[1].tolist())
    if cand:
        assert seen_routes == {"vps"}, seen_routes
        assert sf.counts()["fb_ovl"] > 0, sf.counts()
    print(f"\nSEEDFIX_REPRO route={route} compositions={len(comps)} routes={sorted(seen_routes)} steps=5 PASS")


def test_seeded_same_tokens_across_full_width_routes(rt):
    """VPS full draws (with and without fallback rows, and the plain VPS program), the stock separate verify and the
    fused verify give S the same tokens: they read the same full pieces (top_p 1 avoids VPS's documented float-sum
    caveat at a top-p boundary)."""
    got = {}
    for route in ("vps_fb", "vps_plainprog", "stock", "fused"):
        S = _S(top_p=1.0)
        nb = mixed("x", 4, seeded=(2,), logit0=300, seed0=5000)
        chunk = nb[:2] + [S] + nb[2:]
        toks = []
        runner = runner_for("stock" if route == "fused" else route)
        for s in range(6):
            if route == "fused":
                sched = runner.set_batch(chunk)
                logits, md = F.compose_spec(chunk, s)
                runner._seed_plan = sf.build_plan(runner, sched)
                out = runner._sample_spec_verify_fused(logits, md, 0, len(chunk), runner._get_sampling_generator())
                runner._seed_plan = None
                toks.append(out[2].clone())
            else:
                o, r = F.run_step(runner, [chunk], route=ROUTE_RUN[route], step=s)
                toks.append(o[0][2].clone())
        got[route] = torch.stack(toks)
    base = got["stock"]
    for k, v in got.items():
        assert torch.equal(v, base), (k, v.tolist(), base.tolist())
    print(f"\nSEEDFIX_ROUTES routes={sorted(got)} steps=6 PASS")


def test_different_seeds_differ(rt):
    outs = {}
    for seed in (11, 11, 12):
        S = _S(seed=seed)
        runner = runner_for("vps_fb")
        toks = []
        for s in range(8):
            o, _ = F.run_step(runner, [[S] + mixed("y", 3, logit0=400)], step=s)
            toks.append(o[0][0].clone())
        outs.setdefault(seed, []).append(torch.stack(toks))
    assert torch.equal(outs[11][0], outs[11][1])
    assert not torch.equal(outs[11][0], outs[12][0])
    print("\nSEEDFIX_SEEDS same=equal other=differs PASS")


# ---- the uniforms sit in the right slots -----------------------------------------------------------------------------
@pytest.mark.parametrize("route", ["vps_fb", "vps_cand", "stock", "fused", "plain"])
def test_seeded_uniforms_in_their_slots(rt, route):
    """The recorded uniforms of each seeded request's slots equal its pieces (re-drawn here from a fresh generator with
    its seed: bonus row, accept / recover at its draft slots t0 .. t0+d-1, fallback slots under candidate draws)."""
    seeded = (0, 2, 5)
    rng = (2, 4) if route == "vps_cand" else ()
    d = [3, 1, 2, 3, 3, 2]
    chunk = mixed("z", 6, seeded=seeded, range_rows=rng, d=d, seed0=6000)
    runner = runner_for("stock" if route == "fused" else route)
    if route == "fused":
        sched = runner.set_batch(chunk)
        logits, md = F.compose_spec(chunk, 0)
        runner._seed_plan = sf.build_plan(runner, sched)
        runner._sample_spec_verify_fused(logits, md, 0, len(chunk), runner._get_sampling_generator())
        runner._seed_plan = None
        rec = F.REC["fused"][-1]
    else:
        F.run_step(runner, [chunk], route=ROUTE_RUN[route], spec=route != "plain")
        rec = (F.REC["vps"][-1] if route.startswith("vps") else
               dict(bonus_u=F.REC["sfl"][-1]["u"], **F.REC["rs"][-1]) if route == "stock" else
               dict(bonus_u=F.REC["sfl"][-1]["u"]))
    md = F.compose_spec(chunk, 0)[1]
    cw = 32 if route == "vps_cand" else 0
    t0 = 0
    n_checked = 0
    for b, r in enumerate(chunk):
        if r.seed is not None:
            g = torch.Generator().manual_seed(r.seed)
            bonus, acc, rec_full, cb, cr = sf.draw_pieces(g, TV, K, cw, F.CPU)
            want_b = cb if route == "vps_cand" else bonus
            assert torch.equal(rec["bonus_u"][b], want_b)
            n_checked += 1
            if route != "plain":
                dd = r.d
                assert torch.equal(rec["accept_u"][t0:t0 + dd], acc[:dd])
                want_r = cr if route == "vps_cand" else rec_full
                assert torch.equal(rec["recover_u"][t0:t0 + dd], want_r[:dd])
        t0 += r.d
    if route == "vps_cand":
        fb = rec["fb_u"]
        assert fb is not None
        slot_b, slot_t = sf.fallback_slots(runner, 0, len(chunk), md, torch.bfloat16, vp.VP_KC)
        b = 2  # seeded and a range row
        g = torch.Generator().manual_seed(chunk[b].seed)
        bonus, acc, rec_full, _, _ = sf.draw_pieces(g, TV, K, cw, F.CPU)
        assert torch.equal(fb[slot_b[b]], bonus)
        tb = sum(d[:b])
        for j in range(d[b]):
            assert torch.equal(fb[slot_t[tb + j]], rec_full[j])
        n_checked += 1
    assert n_checked >= 3
    print(f"\nSEEDFIX_SLOTS route={route} seeded={len(seeded)} PASS")


def test_fallback_slots_match_the_graph(rt):
    """seedfix.fallback_slots equals vp_verify_fb's own slot order (its _need_full range flags, stable sort)."""
    for trial in range(6):
        rng = {0: (1,), 1: (0, 3), 2: (2,), 3: (), 4: (4,), 5: (1, 2)}[trial]
        d = [[3, 2, 0, 3, 1], [1, 3, 3, 2, 3], [3, 3, 3, 3, 3], [2, 2, 2, 2, 2], [0, 1, 2, 3, 3], [3, 0, 3, 1, 2]][trial]
        chunk = mixed("q", 5, seeded=(), range_rows=rng, d=d, logit0=trial * 10)
        chunk[2].temp = [0.8, 0.0, -1.0, 0.7, 1e-6, 0.9][trial]
        runner = F.Runner(vps=True, cand=True, fb_max=8)
        runner.set_batch(chunk)
        logits, md = F.compose_spec(chunk, 0)
        b_pad = md.bonus_logits_indices.shape[0]
        temps, top_k, top_p = runner._padded_sampling_params(0, 5, b_pad, logits.dtype, F.CPU)
        vals, ids, sent = vp.local_candidates_ext(logits, 0)
        cs = sent
        num_bonus = b_pad
        bonus_c = torch.index_select(vals, 0, md.bonus_logits_indices)
        target_c = torch.index_select(vals, 0, md.target_logits_indices)
        active_b = torch.arange(num_bonus) < 5
        seg64 = md.segment_ids.to(torch.int64)
        seg_c = seg64.clamp(min=0, max=num_bonus - 1)
        active_t = (seg64 >= 0) & (seg64 < num_bonus) & (seg_c < 5) & (md.group_indices.to(torch.int64) < md.draft_lengths.to(torch.int64)[seg_c])
        range_b, _ = vp._need_full(vp._scaled_bonus(bonus_c, temps), vp._scaled_bonus(torch.index_select(cs, 0, md.bonus_logits_indices), temps),
                                   (temps <= vp.SAMPLING_EPS).squeeze(-1), top_k, active_b, vp.VP_KC)
        tt = temps[seg_c]
        range_t, _ = vp._need_full(vp._scaled_target(target_c, tt), vp._scaled_target(torch.index_select(cs, 0, md.target_logits_indices), tt),
                                   tt.squeeze(-1) == 0, top_k[seg_c], active_t, vp.VP_KC)
        first = torch.cat([range_b, range_t])
        order = torch.sort(torch.where(first, 0, 2).to(torch.int32), stable=True).indices[: int(first.sum())].tolist()
        sb, st = sf.fallback_slots(runner, 0, 5, md, logits.dtype, vp.VP_KC)
        host = {**{b: s for b, s in sb.items()}, **{num_bonus + t: s for t, s in st.items()}}
        graph = {row: s for s, row in enumerate(order)}
        assert host == graph, (trial, host, graph)
    print("\nSEEDFIX_FBSLOTS trials=6 PASS")


# ---- first token -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("site", ["plain", "spec_plain"])
def test_first_token_reproducible(rt, site):
    """The first token (non-spec step, and a non-verify chunk of a spec step) of S is the same across compositions,
    and its u row is S's bonus piece."""
    ref = None
    for comp in range(4):
        S = _S(d=0)
        nb = mixed(f"p{comp}", 2 + comp, seeded=(0,) if comp % 2 else (), logit0=500 + comp * 7, seed0=7000)
        plain_chunk = nb[: comp] + [S] + nb[comp:]
        if site == "plain":
            runner = F.Runner(vps=False)
            F.clear_rec()
            o, r = F.run_step(runner, [plain_chunk], spec=False, pads=[(8 * (1 + comp % 2), None)])
            tok = o[0][comp]
        else:
            runner = F.Runner(vps=True, fb_max=8)
            verify = mixed(f"v{comp}", 3, seeded=(1,), logit0=600 + comp, seed0=8000)
            F.clear_rec()
            o, r = F.run_step(runner, [verify, plain_chunk], plain_chunks=(1,), pads=[(None, None), (8 * (1 + comp % 2), None)])
            assert r == ["vps", "spec_plain"]
            tok = o[1][comp][:1]
        u = F.REC["sfl"][-1]["u"]
        g = torch.Generator().manual_seed(S.seed)
        assert torch.equal(u[comp], sf.draw_pieces(g, TV, K, 0, F.CPU)[0])
        if ref is None:
            ref = tok.clone()
        assert torch.equal(tok.reshape(-1)[:1], ref.reshape(-1)[:1]), (comp, tok, ref)
    print(f"\nSEEDFIX_FIRST site={site} compositions=4 PASS")


# ---- partial rows, flag off ------------------------------------------------------------------------------------------
class RewindSpy:
    def __init__(self):
        self.calls = []

    def get_offset(self):
        self.calls.append("get")
        return 100

    def set_offset(self, v):
        self.calls.append(("set", v))


@pytest.mark.parametrize("on", [True, False])
def test_partial_rows(rt, on):
    """A partial (chunked-prefill) seeded row draws nothing; flag on skips the -4 rewind (nothing was drawn), flag off
    keeps base's rewind (get_offset, then set_offset(offset - 4))."""
    flag(rt, on)
    runner = F.Runner(vps=False)
    P = R("P", 99, partial=True)
    Q = R("Q", 98, logit_seed=5)
    spy = RewindSpy()
    chunk = [Q, P] + mixed("w", 2, seeded=(), logit0=700)
    q0 = Q.gen.get_state().clone()
    sched = runner.set_batch(chunk, spec=False)
    runner.input_batch.generators[1] = spy
    logits = F.compose_plain(chunk, 0)
    runner.execute_model_state = NS(scheduler_output=sched, draft_chunks=None, top_tokens=False, vps_chunks=None,
                                    vps_fallback=None, spec_decode_metadata_list=[None], logits_list=[logits],
                                    num_reqs_list=[len(chunk)], mamba_state_indices_list=None)
    try:
        runner.sample_tokens(None)
    except F.Stop:
        pass
    if on:
        assert spy.calls == [] and sf.counts()["partial"] == 1 and sf.counts()["rows"] == 1
        assert not torch.equal(Q.gen.get_state(), q0)
    else:
        assert spy.calls == ["get", ("set", 96)] and sf.counts()["steps"] == 0
        assert torch.equal(Q.gen.get_state(), q0)
    print(f"\nSEEDFIX_PARTIAL flag={'on' if on else 'off'} rewind_calls={spy.calls} PASS")


def test_flag_off_runner_ast_is_base():
    """Flag off, AST for AST: tpu_runner.py with SEEDFIX's statements resolved to their flag-off arm
    (seedfix_flagoff_ast: the plan is None, so every overlay block, plan assignment, warm-up call and the rewind's
    extra guard drop out) is the base tree's (ref_*_tpu_runner.py.txt, the composition base on the composed tree);
    the check bites on an overlay block left unguarded."""
    import seedfix_flagoff_ast as so

    src = open(tr.__file__).read()
    ref_name = [f for f in os.listdir(HERE) if f.startswith("ref_") and f.endswith("_tpu_runner.py.txt")]
    assert len(ref_name) == 1, ref_name
    ref = open(os.path.join(HERE, ref_name[0])).read()
    assert src != ref
    d = so.check(src, ref, pre=getattr(sys.modules[__name__], "_FLAGOFF_PRE", None))
    assert not d, d
    guard = "        if seed_plan is not None:\n            _seedfix.overlay_targets(\n"
    assert src.count(guard) == 1
    bad = so.check(src.replace(guard, "        if True:\n            _seedfix.overlay_targets(\n"), ref,
                   pre=getattr(sys.modules[__name__], "_FLAGOFF_PRE", None))
    assert bad == ["differs: TPUModelRunner._sample_spec_verify_chunk"], bad
    print(f"\nSEEDFIX_FLAGOFF_AST ref={ref_name[0]} identical=1 bites=1 PASS")


def test_flag_off_is_base(rt):
    """Flag off: seeded rows read the shared draws exactly as if they had no generator (tokens and uniforms), their
    generators are never advanced and no plan is built."""
    res = []
    for gens in (True, False):
        flag(rt, False)
        F.clear_rec()
        runner = F.Runner(vps=True, fb_max=8)
        chunk = mixed("o", 5, seeded=(0, 3) if gens else (), range_rows=(1,))
        states = [r.gen.get_state().clone() for r in chunk if r.gen is not None]
        outs, _ = steps(runner, lambda: [chunk], "vps_fb", 3)
        if gens:
            assert all(torch.equal(r.gen.get_state(), s) for r, s in zip([r for r in chunk if r.gen is not None], states))
        res.append((outs, [dict(v) for v in F.REC["vps"]]))
    assert all(torch.equal(a, b) for a, b in zip(res[0][0], res[1][0]))
    for a, b in zip(res[0][1], res[1][1]):
        for k in ("bonus_u", "accept_u", "recover_u"):
            assert torch.equal(a[k], b[k])
    assert sf.counts()["steps"] == 0
    print("\nSEEDFIX_FLAGOFF base=identical PASS")


# ---- multirank -------------------------------------------------------------------------------------------------------
def test_eight_ranks_agree(rt):
    """8 simulated TP ranks (threads; each its own runner and per-request generators, seeded identically) over 5
    mixed steps on the VPS fallback route and the stock route: identical tokens, identical per-request and shared
    generator states."""
    import vllm.distributed

    local = threading.local()
    rt.setattr(vllm.distributed, "get_tensor_model_parallel_rank", lambda: getattr(local, "rank", 0), raising=False)
    results = [None] * 8
    errors = []
    barrier = threading.Barrier(8)

    def rank(i):
        try:
            local.rank = i
            out = []
            for route in ("vps_fb", "stock"):
                runner = runner_for(route)
                chunk = mixed(f"k{route}", 6, seeded=(0, 2, 5), range_rows=(3,), seed0=9000)
                for s in range(5):
                    barrier.wait()
                    o, _ = F.run_step(runner, [chunk], route=ROUTE_RUN[route], step=s)
                    out.append(o[0].clone())
                out.append(torch.stack([r.gen.get_state() for r in chunk if r.gen is not None]))
                out.append(runner._sampling_generator.get_state())
            results[i] = out
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))
            barrier.abort()

    th = [threading.Thread(target=rank, args=(i,)) for i in range(8)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    assert not errors, errors
    for i in range(1, 8):
        assert len(results[i]) == len(results[0])
        for a, b in zip(results[0], results[i]):
            assert torch.equal(a, b), f"rank {i} differs"
    print("\nSEEDFIX_RANKS ranks=8 routes=2 steps=5 PASS")


# ---- warm-up census --------------------------------------------------------------------------------------------------
_SEEDFIX_FILE = os.sep + os.path.join("runner", "seedfix.py")
_HOST_FUNCS = {"fallback_slots", "dev_index"}
_HOST_OPS = ("aten.arange", "aten.lift_fresh")


def _meta(a):
    if isinstance(a, torch.Tensor):
        return ("T", tuple(a.shape), str(a.dtype), a.device.type, tuple(a.stride()), a.storage_offset())
    if isinstance(a, torch.Generator):
        return "G"
    if isinstance(a, (list, tuple)):
        return tuple(_meta(x) for x in a)
    return repr(a)


class Census(TorchDispatchMode):
    """Eager-op keys (op, argument layouts) issued from runner/seedfix.py, except its host-side helpers (index
    construction, the fallback-slot arithmetic on CPU params); index uploads are counted by their seedfix_h2d notes."""

    def __init__(self):
        super().__init__()
        self.keys = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        f = sys._getframe(1)
        while f is not None and not f.f_code.co_filename.endswith(_SEEDFIX_FILE):
            f = f.f_back
        name = str(func)
        if f is not None and f.f_code.co_name not in _HOST_FUNCS and not name.startswith(_HOST_OPS):
            self.keys.add((name, _meta(list(args)), _meta(sorted(kwargs.items(), key=lambda kv: kv[0]))))
        return func(*args, **kwargs)


def _census_runner(cand, notes):
    r = F.Runner(vps=True, cand=cand, fb_max=8)
    r.num_reqs_paddings = [8, 16, 32, 64]
    r.num_tokens_paddings = [16, 32, 64, 128, 256]
    r.max_num_reqs = 64
    r._eager_shape_note = lambda site, key: notes.add((site, key))
    return r


def _steady_buckets(r):
    """The (num_tokens, num_reqs) pairs `_precompile_rejection_sampler` sweeps (no cap), as it iterates them."""
    max_logits_len = tr._get_padded_token_len(r.num_tokens_paddings, min(r.max_num_reqs * (K + 1), r.num_tokens_paddings[-1]))
    req_bucket = next((b for b in r.num_reqs_paddings if b >= r.max_num_reqs), r.num_reqs_paddings[-1])
    out = []
    for nt in r.num_tokens_paddings:
        if nt > max_logits_len:
            break
        for nr in r.num_reqs_paddings:
            if nr > req_bucket:
                break
            if not r._spec_verify_bucket_reachable(nt, nr):
                continue
            out.append((nt, nr))
            if nr >= min(nt, r.max_num_reqs):
                break
    return out


def _bucket_batch(nt, nr, cand, tag):
    """A mixed batch at padded shapes (nr, nt): seeded rows with d = 1, 2, 3 (cycling), unseeded neighbours, and
    under candidate draws one seeded and one unseeded fallback-row request."""
    n = min(nr - 1, 12)
    reqs, total = [], 0
    for i in range(n):
        seeded = i % 2 == 0
        dmax = (i // 2) % K + 1 if seeded else (i % (K + 1))
        d = max(0, min(dmax, nt - total - (n - i)))
        if total + d + 1 > nt:
            break
        top_k = 0 if i in (0, 1) else 20
        reqs.append(R(f"{tag}{i}", 10000 + i if seeded else None, top_k=top_k, logit_seed=i, d=d))
        total += d + 1
    return reqs


def _serve_all(runners, buckets_for):
    for r in runners:
        cand = r._vps_cand_draws
        for nt, nr in buckets_for(r):
            reqs = _bucket_batch(nt, nr, cand, f"s{nt}_{nr}_")
            F.run_step(r, [reqs], route="vps", pads=[(nr, nt)])
            if not cand:
                F.run_step(r, [reqs], route="stock", pads=[(nr, nt)])
                sched = r.set_batch(reqs)
                logits, md = F.compose_spec(reqs, 0, b_pad=nr, t_pad=nt)
                r._seed_plan = sf.build_plan(r, sched)
                r._sample_spec_verify_fused(logits, md, 0, len(reqs), r._get_sampling_generator())
                r._seed_plan = None
        for nr in r.num_reqs_paddings:
            reqs = [R(f"pl{nr}_{i}", 20000 + i if i % 2 == 0 else None, logit_seed=i, d=0) for i in range(min(nr - 1, 6))]
            F.run_step(r, [reqs], spec=False, pads=[(nr, None)])


def _warm_all(runners):
    """What the runner's warm-up calls: warm_bucket over the verify sweep's buckets and warm_rows over the non-spec
    sampling buckets (tpu_runner _precompile_rejection_sampler / _precompile_sample_from_logits)."""
    for r in runners:
        fb_rows = vps_wiring.fallback_variants(r._vps_fb_max, r._vps_tiegap) if r._vps_cand_draws else ()
        for nt, nr in _steady_buckets(r):
            sf.warm_bucket(r, nt, nr, fb_rows)
        for nr in r.num_reqs_paddings:
            sf.warm_rows(r, nr, r.vocab_size)


def test_census_no_new_eager_key_after_warmup(rt):
    """Warm up (as the runner does), then serve mixed seeded / unseeded batches at every steady bucket on every route
    (VPS full with F 0 / > 0, VPS candidate with fallback rows, stock, fused, non-spec): 0 eager keys and 0 index-upload
    / shape notes that warm-up did not issue."""
    notes_w, notes_s = set(), set()
    rw = [_census_runner(False, notes_w), _census_runner(True, notes_w)]
    with Census() as cw:
        _warm_all(rw)
    for r in rw:
        r._eager_shape_note = lambda site, key: notes_s.add((site, key))
    c0 = sf.counts()
    with Census() as cs:
        _serve_all(rw, _steady_buckets)
    c1 = sf.counts()
    new_keys = cs.keys - cw.keys
    notes_w = {x for x in notes_w if x[0].startswith("seedfix")}     # the overlay's own notes (base sites note too)
    notes_s = {x for x in notes_s if x[0].startswith("seedfix")}
    new_notes = notes_s - notes_w
    n_b = sum(len(_steady_buckets(r)) for r in rw)
    print(f"\nSEEDFIX_CENSUS buckets={n_b} warm_keys={len(cw.keys)} serve_keys={len(cs.keys)} new_keys={len(new_keys)} "
          f"warm_notes={len(notes_w)} serve_notes={len(notes_s)} new_notes={len(new_notes)} "
          f"rows_ovl={c1['rows_ovl'] - c0['rows_ovl']} tgt_ovl={c1['tgt_ovl'] - c0['tgt_ovl']} fb_ovl={c1['fb_ovl'] - c0['fb_ovl']}")
    for k in sorted(new_keys, key=repr)[:10]:
        print(f"\nSEEDFIX_CENSUS_NEW {k}")
    for k in sorted(new_notes, key=repr)[:10]:
        print(f"\nSEEDFIX_CENSUS_NEWNOTE {k}")
    assert len(cs.keys) > 0 and c1["rows_ovl"] > c0["rows_ovl"] and c1["tgt_ovl"] > c0["tgt_ovl"] and c1["fb_ovl"] > c0["fb_ovl"]
    assert not new_keys and not new_notes


def test_runner_warmup_wiring():
    """tpu_runner's warm-up calls the overlay's warm-up inside the verify bucket sweep (after the reachability check),
    in the full-draft sweep, and in the non-spec sampling sweep."""
    src = inspect.getsource(tr.TPUModelRunner._precompile_rejection_sampler)
    tree = ast.parse(textwrap.dedent(src))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "warm_bucket" and getattr(n.func.value, "id", "") == "_seedfix"]
    assert len(calls) == 2, len(calls)
    i_reach = src.index("if not self._spec_verify_bucket_reachable(num_tokens, num_reqs):")
    i_warm = src.index("_seedfix.warm_bucket(self, num_tokens, num_reqs, seed_fb_rows)")
    assert i_reach < i_warm < src.index("for num_tokens, num_reqs in full_draft:")
    assert src.count("_seedfix.warm_bucket(self, num_tokens, num_reqs, seed_fb_rows)") == 2
    src2 = inspect.getsource(tr.TPUModelRunner._precompile_sample_from_logits)
    assert "_seedfix.warm_rows(self, num_reqs, dummy_logits.shape[-1])" in src2
    assert src2.index("for num_reqs in self.num_reqs_paddings:") < src2.index("_seedfix.warm_rows(")
    print("\nSEEDFIX_WIRING warm_bucket=2 warm_rows=1 PASS")


def test_warmup_noop_with_flag_off(rt):
    flag(rt, False)
    r = _census_runner(False, set())
    with Census() as c:
        sf.warm_bucket(r, 32, 8)
        sf.warm_rows(r, 8, TV)
    assert c.keys == set() and "_seedfix_warmed" not in r.__dict__


# ---- engagement line -------------------------------------------------------------------------------------------------
def test_engagement_line(rt):
    lines = []
    rt.setattr(sf.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    runner = runner_for("vps_fb")
    F.run_step(runner, [mixed("l", 4, seeded=(1,))])
    assert len(lines) == 1 and lines[0].startswith("q38 SEEDFIX steps=1 rows=1 ")
    print(f"\nSEEDFIX_LINE {lines[0]}")


# ---- distribution of the seeded path's accept / reject outcomes ------------------------------------------------------
PEAKS = [5.0, 4.7, 4.2, 3.8, 3.2, 2.6, 2.0, 1.5]


def test_seeded_accept_reject_distribution(rt):
    """>= 1e6 decisions of the served VPS route (fallback program, full draws, top_k 20, top_p 0.95, temperature 1),
    every row a seeded request with its own seed: the accepted-length histogram matches the analytic target
    (chi2 < 11.34 on df 3) and every position's acceptance |z| < 3."""
    P, calls = 256, 4000
    vocab = TV
    g = torch.Generator().manual_seed(77)
    req = torch.randn(K + 1, vocab, generator=g) * 0.5
    for j in range(K + 1):
        top = torch.randperm(vocab, generator=g)[:8]
        req[j, top] = torch.tensor(PEAKS) + 0.1 * j
    req = req.bfloat16()
    second = [int(torch.sort(req[j].float(), descending=True, stable=True).indices[1]) for j in range(K)]
    temps = torch.ones(1, 1, dtype=torch.bfloat16)
    top_k = torch.full((1, 1), 20, dtype=torch.int32)
    top_p = torch.full((1, 1), 0.95, dtype=torch.float32)
    cv, ci = vp.local_candidates(req, 0)
    pacc = []
    for j in range(K):
        m = vp._candidate_mask(vp._scaled_target(cv[j:j + 1], temps), top_k, top_p)
        p = dict(zip(ci[j].tolist(), torch.softmax(m, dim=-1)[0].tolist()))
        pacc.append(p.get(second[j], 0.0))
    al_p = [math.prod(pacc[:a]) * (1 - pacc[a]) for a in range(K)] + [math.prod(pacc)]

    class DReq(F.Req):
        def rows(self, step):
            return req, torch.tensor(second + [0] * (K - len(second)), dtype=torch.int32)

    reqs = [DReq(f"d{i}", 123457 + 7919 * i, temp=1.0, top_k=20, top_p=0.95, d=K) for i in range(P)]
    runner = runner_for("vps_fb")
    hist = [0] * (K + 1)
    reached, accepted = [0] * K, [0] * K
    d = torch.tensor(second, dtype=torch.int32)
    for s in range(calls):
        F.clear_rec()  # the recorder keeps a clone of every uniform array: 52 MB per step here (v2d j-305d65cd OOM)
        o, r = F.run_step(runner, [reqs], step=s, pads=[(P + 8, P * (K + 1))])
        assert r == ["vps"]
        out = o[0]
        acc_mask = out[:, :K] == d
        first_rej = torch.where(~acc_mask, torch.arange(K).expand(P, K), torch.full((P, K), K)).min(dim=1).values
        for a, c in enumerate(torch.bincount(first_rej, minlength=K + 1).tolist()):
            hist[a] += c
        for j in range(K):
            reached[j] += int((first_rej >= j).sum())
            accepted[j] += int((first_rej > j).sum())
    n = P * calls
    chi = sum((h - n * p) ** 2 / (n * p) for h, p in zip(hist, al_p))
    z = [abs(accepted[j] - reached[j] * pacc[j]) / math.sqrt(reached[j] * pacc[j] * (1 - pacc[j])) for j in range(K)]
    c = sf.counts()
    print(f"\nSEEDFIX_OUTCOME rows={n} decisions={sum(reached)} pacc={[round(p, 6) for p in pacc]} al_hist={hist} "
          f"expect={[round(n * p) for p in al_p]} al_chi2={chi:.2f} z={[round(x, 2) for x in z]} seeded_rows={c['rows']}")
    assert n >= 1_000_000 and sum(reached) >= 1_000_000 and c["rows"] == n
    assert chi < 11.34 and max(z) < 3.0
