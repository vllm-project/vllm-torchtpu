# SPDX-License-Identifier: Apache-2.0
"""q38 L4 T1-K1FUSE (VLLM_TPU_K1FUSE; spec_decode/k1fuse.py) against today's unfused chain (VFY-ONE value 2 +
DRAFT-ASM + DH-D), through the real runner verify and the real propose (tests/q38_k1fuse/k1fuse_harness.py).
CDK CPU pod only.

A. Bitwise over 8 serving steps with carry (each step's draft ids are that chain's previous drafts): tokens, random
   stream, stats, read offsets, chunk counts, drafts, source, rejected count, index dicts and the draft KV writes;
   8 / 32 / 48 / 64 requests, fallback rows F = 0 and F = 8 (with two flagged rows).
B. Seam: K1's outputs in exactly the R4 seam's shapes (nt [B, 1+k], source [max_reqs*(1+k)], rejected [max_reqs],
   int32; drafts [p, k]) equal to the unfused chain's.
C. Census: verify + draft compiled calls 2 -> 1 per step; the eager ops outside programs are the same multiset.
D. Programs: one instance per static signature with its own code object, serving only its steady token buckets;
   instance / program count over the steady sweep against DH-D's; warm keys = serving keys (0 cold at steady shapes).
E. Window switching over K1 / DH-D steps never makes a new program key after the warm-up and stays bitwise.
F. No host read in the K1 body; the uniform draws stay eager (no RNG op inside the program; same stream).
G. CHECK: compares then arms; a perturbed K1 output is caught and turns K1 off; serving results stay the chain's.
H. Take: by construction no write between the verify and the propose touches the tokens or a K1 draft input;
   a write to a draft input makes the take raise; a result never taken raises at the next verify.
"""

import collections
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import k1fuse_harness as kh
from vllm_torchtpu.runner import tpu_runner, vfy_one
from vllm_torchtpu.spec_decode import draft_asm, draft_one, k1fuse

h = kh.h
COUNTS: collections.Counter = collections.Counter()


def _pair(monkeypatch, F, check=0, k1_ab=None, seed=0):
    """(K1 proposer, unfused proposer) on equal state, each warmed on the same steps."""
    kh.install(monkeypatch, k1=False)
    bp = kh.make_pair_proposer("one", F=F, seed=seed)
    kh.install(monkeypatch, k1=True, check=check, k1_ab=k1_ab)
    kp = kh.make_pair_proposer("k1", F=F, seed=seed)
    return kp, bp


def _warm_both(monkeypatch, kp, bp, rs, check=0, k1_ab=None):
    kh.install(monkeypatch, k1=False)
    kh.warm(bp, rs)
    kh.install(monkeypatch, k1=True, check=check, k1_ab=k1_ab)
    kh.warm(kp, rs)
    for p in (kp, bp):
        p.draft_model.kv.zero_()  # K1's warm-up calls wrote the synthetic requests' KV too


def _serve_both(monkeypatch, kp, bp, vk, vb, F, seed, check=0, k1_ab=None, between=None):
    kh.install(monkeypatch, k1=False)
    ob = kh.serve(bp, vb, F, seed)
    kh.install(monkeypatch, k1=True, check=check, k1_ab=k1_ab)
    ok = kh.serve(kp, vk, F, seed, between=between)
    return ok, ob


def _agree_both(monkeypatch, kp, bp, check=0, k1_ab=None):
    kh.install(monkeypatch, k1=False)
    kh.agree(bp)
    kh.install(monkeypatch, k1=True, check=check, k1_ab=k1_ab)
    kh.agree(kp)


# ---------------------------------------------------------------- A. bitwise over 8 steps with carry


@pytest.mark.parametrize("F", [0, 8])
@pytest.mark.parametrize("R", [8, 32, 48, 64])
def test_k1_is_bitwise_the_unfused_chain_over_8_steps(monkeypatch, R, F):
    kp, bp = _pair(monkeypatch, F, seed=R)
    _warm_both(monkeypatch, kp, bp, [R])
    _agree_both(monkeypatch, kp, bp)
    assert kp._draft_asm.k1_agreed and kp._draft_asm.one_agreed
    prev_k = prev_b = None
    for i in range(8):
        seed = 20000 + 100 * R + 10 * F + i
        vk, vb = kh.make_vstep(seed, R, prev_k), kh.make_vstep(seed, R, prev_b)
        ok, ob = _serve_both(monkeypatch, kp, bp, vk, vb, F, seed)
        assert ok.calls.get("k1", 0) == 1 and ok.calls.get("vfy_one", 0) == 0 and ok.calls.get("one", 0) == 0, ok.calls
        assert ob.calls.get("vfy_one", 0) == 1 and ob.calls.get("one", 0) == 1 and ob.calls.get("k1", 0) == 0, ob.calls
        assert isinstance(ok.result, k1fuse.K1Result) and "_vfy_seed" not in kp.runner.__dict__
        kh.assert_same_step(ok, ob, kp.runner, bp.runner, f"R={R} F={F} step {i}")
        prev_k, prev_b = ok.drafts, ob.drafts
    ctl = kp._draft_asm
    assert ctl.k1_stats["k1"] == 8 and ctl.counts()["folded_k1"] == 8 and not ctl.k1_reasons, dict(ctl.k1_reasons)
    if F:
        assert int(kp.runner._vps_fb_stats[1]) > 0, kp.runner._vps_fb_stats  # fallback rows covered: not vacuous
    assert int(kp.draft_model.kv.abs().sum()) > 0
    COUNTS["bitwise_steps"] += 8
    print(f"\nK1FUSE_BITWISE R={R} F={F} steps=8 tokens / stream / stats / read offsets / drafts / source / rejected / "
          f"index dicts / draft KV equal to the VFY + DH-D chain; k1 calls 8, stats {kp.runner._vps_fb_stats.tolist()}")


# ---------------------------------------------------------------- B. the seam


@pytest.mark.parametrize("R", [12, 40])
def test_seam_shapes_are_the_r4_contracts(monkeypatch, R):
    kp, bp = _pair(monkeypatch, 8)
    _warm_both(monkeypatch, kp, bp, [R])
    _agree_both(monkeypatch, kp, bp)
    vk, vb = kh.make_vstep(23000 + R, R), kh.make_vstep(23000 + R, R)
    ok, ob = _serve_both(monkeypatch, kp, bp, vk, vb, 8, 23000 + R)
    res = ok.result
    B = vk.step.chunk.draft_lengths.shape[0]
    p_rows = kp._loop_bucket(R)
    assert ok.nt.shape == (B, kh.K + 1) and ok.nt.dtype == torch.int32
    assert res.src.shape == (kh.MAX_REQS * (kh.K + 1),) and res.src.dtype == torch.int32
    assert res.num_rejected.shape == (kh.MAX_REQS,) and res.num_rejected.dtype == torch.int32
    assert res.k1_drafts.shape == (p_rows, kh.K) and res.k1_drafts.dtype == torch.int32
    assert res.next_tokens is ok.nt and res.draft_lengths is vk.step.chunk.draft_lengths and res.num_reqs == R
    kh.same(ok.nt, ob.nt, "seam nt")
    kh.same(res.src, ob.src, "seam source")
    kh.same(res.num_rejected, ob.rejected, "seam rejected")
    kh.same(res.drafts(), ob.drafts, "seam drafts")
    COUNTS["seam"] += 1
    print(f"\nK1FUSE_SEAM R={R}: nt {tuple(ok.nt.shape)} int32, source [{res.src.numel()}] int32, rejected "
          f"[{res.num_rejected.numel()}] int32, drafts [{p_rows}, {kh.K}]; equal to the unfused chain's")


# ---------------------------------------------------------------- C. census


@pytest.mark.parametrize("R", [8, 32, 48, 64])
def test_census_verify_plus_draft_two_calls_become_one(monkeypatch, R):
    kp, bp = _pair(monkeypatch, 0)
    _warm_both(monkeypatch, kp, bp, [R])
    _agree_both(monkeypatch, kp, bp)
    vk, vb = kh.make_vstep(24000 + R, R), kh.make_vstep(24000 + R, R)
    ok, ob = _serve_both(monkeypatch, kp, bp, vk, vb, 0, 24000 + R)
    nk, nb = kh.program_calls(ok), kh.program_calls(ob)
    assert (nb, nk) == (2, 1), (ob.calls, ok.calls)
    assert collections.Counter(ok.eager) == collections.Counter(ob.eager), (
        collections.Counter(ok.eager) - collections.Counter(ob.eager),
        collections.Counter(ob.eager) - collections.Counter(ok.eager))
    COUNTS["census"] += 1
    print(f"\nK1FUSE_CENSUS R={R} verify+draft compiled calls {nb} -> {nk} ({dict(ob.calls)} -> {dict(ok.calls)}); "
          f"eager ops outside programs equal ({len(ok.eager)})")


# ---------------------------------------------------------------- D. programs


@pytest.mark.parametrize("F", [0, 8])
def test_instances_programs_and_token_buckets_over_the_steady_sweep(monkeypatch, F):
    rec = {"info": [], "error": [], "warning": []}
    kh.install(monkeypatch, k1=True)
    with kh.quiet(monkeypatch, rec):
        kp = kh.make_pair_proposer("k1", F=F)
        kh.warm(kp, range(1, kh.MAX_REQS + 1), seed0=26000)
        kh.agree(kp)
    ctl = kp._draft_asm
    shapes = ctl.decode_shapes()
    steady = {(t, w, lp, c) for t, w, lp, c in shapes}
    dhd = len({(t, lp, c) for t, _w, lp, c in steady})
    regions = list(ctl.k1_modules.values())
    codes = {type(r.module).forward.__code__ for r in regions}
    assert len(codes) == len(regions) and sorted(r.index for r in regions) == list(range(len(regions)))
    assert k1fuse._K1FuseProgram.forward.__code__ not in codes
    assert ctl.k1_fvars == [F] and not ctl.k1_restricted
    from vllm_torchtpu.compilation import bucket_filter

    want_inst = {(lp, w, c) for _t, w, lp, c in steady}
    assert len(ctl.k1_modules) == len(want_inst), (len(ctl.k1_modules), sorted(want_inst))
    seen_inst = set()
    for sig, m in ctl.k1_modules.items():
        pred = bucket_filter.assigned(m)
        st = m.statics
        assert sig[0][0] == "T" and sig[0][1][0] == "tok"  # the verify hidden, its token dim abstracted
        width = sig[k1fuse.ARG_NAMES.index("bonus_logits_indices")][1][0]  # the request bucket B
        served = {t for t in kh.PADDINGS if pred(t)}
        want = {t for t, w, lp, c in steady if (lp, w, c) == (st.p, width, st.num_rows)}
        assert served and served == want, (st, width, served, want)
        seen_inst.add((st.p, width, st.num_rows))
    assert seen_inst == want_inst
    keys_t = {k[ k1fuse.ARG_NAMES.index("input_ids") ][1][0] for k in ctl.warm["k1"]}
    assert keys_t == {t for t, *_ in steady}
    assert len(ctl.warm["k1"]) == len(steady)
    total = len(ctl.warm["k1"])
    assert total <= 2 * dhd, (total, dhd)
    lines = [m for m in rec["info"] if "K1FUSE warm-up" in m]
    assert len(lines) == 1 and f"{len(ctl.k1_modules)} instances, {total} programs" in lines[0], rec["info"]
    COUNTS["programs"] += 1
    print(f"\nK1FUSE_PROGRAMS F={F}: {len(ctl.k1_modules)} instances (p, B, cap) {sorted(want_inst)}, {total} programs "
          f"(token buckets) vs DH-D {dhd} programs over {len(ctl.one_modules)} loop buckets; each instance its own "
          f"code object and only its steady token buckets; {lines[0]}")


def test_restriction_rule_keeps_the_served_fallback_rows_only(monkeypatch):
    """F variants {0, 4, 8} (fallback max 8, no tie gap): 3 x 12 programs > 2 x DH-D's 12, so K1 warms only the
    served F (0); a step at F = 8 runs today's chain (counted cold)."""
    kh.install(monkeypatch, k1=True)
    kp = kh.make_pair_proposer("k1", F=0)
    kp.runner._vps_fb_max = 8
    kh.warm(kp, [12])
    kh.agree(kp)
    ctl = kp._draft_asm
    assert ctl.k1_restricted and ctl.k1_fvars == [0], ctl.k1_fvars
    vs = kh.make_vstep(26500, 12)
    o = kh.serve(kp, vs, 8, 26500)
    assert o.calls.get("k1", 0) == 0 and o.calls.get("vfy_one", 0) == 1 and ctl.k1_reasons["cold"] == 1, (
        o.calls, dict(ctl.k1_reasons))
    COUNTS["restriction"] += 1
    print("\nK1FUSE_RESTRICT F variants [0, 4, 8] -> 36 programs > 2 x 12: warmed F [0] only; an F=8 step runs the "
          "VFY + DH-D chain (cold 1)")


def test_warm_keys_are_the_serving_keys(monkeypatch):
    kh.install(monkeypatch, k1=True)
    kp = kh.make_pair_proposer("k1", F=8)
    rs = [1, 5, 9, 16, 17, 31, 33, 47, 49, 64]
    kh.warm(kp, rs, seed0=27000)
    kh.agree(kp)
    for i, R in enumerate(rs):
        o = kh.serve(kp, kh.make_vstep(27100 + i, R), 8, 27100 + i)
        assert o.calls.get("k1", 0) == 1, (R, o.calls, dict(kp._draft_asm.k1_reasons))
    assert kp._draft_asm.k1_reasons["cold"] == 0
    COUNTS["warm_keys"] += len(rs)
    print(f"\nK1FUSE_WARMKEYS {len(rs)} serving steps at R {rs}: every K1 key was warmed (cold 0)")


# ---------------------------------------------------------------- E. window switching


class _FakeL6:
    def __init__(self, sched):
        self.LEVERS = {"k1fuse": "VLLM_TPU_K1FUSE_AB", "draftasmone": "x", "draftasmcl": "x"}
        self.STEPS = 1
        self.sched = sched

    def install(self, monkeypatch):
        import vllm_torchtpu.core as core

        mod = ModuleType("vllm_torchtpu.core.l6_ab")
        mod.LEVERS, mod.STEPS = self.LEVERS, self.STEPS
        mod.on = lambda lever: self.sched.get(lever, True)
        mod.note = lambda *a, **k: None
        monkeypatch.setitem(sys.modules, "vllm_torchtpu.core.l6_ab", mod)
        monkeypatch.setattr(core, "l6_ab", mod, raising=False)
        monkeypatch.setattr(draft_asm, "_L6_AB", [])


def test_window_switching_never_compiles_after_warmup_and_stays_bitwise(monkeypatch):
    sched = {"k1fuse": True}
    fake = _FakeL6(sched)
    kp, bp = _pair(monkeypatch, 8, k1_ab="ab:1")
    rs = [8, 13, 20, 32, 41, 48, 57, 64]
    _warm_both(monkeypatch, kp, bp, rs, k1_ab="ab:1")
    fake.install(monkeypatch)
    monkeypatch.setattr(draft_asm, "AGREE_EVERY", 1)
    kp._draft_asm.period = None
    warm_keys = {k[: len(k1fuse.ARG_NAMES)] for k in kp._draft_asm.warm["k1"]}
    used = collections.Counter()
    new = set()
    for i, R in enumerate(rs * 2):
        # a boundary on every propose: the window set here applies from the next verify on
        sched["k1fuse"] = i % 2 == 0
        seed = 28000 + i
        vk, vb = kh.make_vstep(seed, R), kh.make_vstep(seed, R)
        kh.install(monkeypatch, k1=False)
        fake.install(monkeypatch)
        ob = kh.serve(bp, vb, 8, seed)
        kh.install(monkeypatch, k1=True, k1_ab="ab:1")
        fake.install(monkeypatch)
        ok = kh.serve(kp, vk, 8, seed)
        kh.assert_same_step(ok, ob, kp.runner, bp.runner, f"window step {i} R={R}")
        used["k1" if ok.calls.get("k1") else "chain"] += 1
        new.update(k for name, k in ok.keys if name == "k1" and k not in warm_keys)
    assert not new, len(new)
    assert used["k1"] >= 6 and used["chain"] >= 6, dict(used)
    COUNTS["window_steps"] += len(rs) * 2
    print(f"\nK1FUSE_COMPILE window steps={len(rs) * 2} k1={used['k1']} chain={used['chain']} new K1 keys after "
          f"warm-up=0 (warm {len(warm_keys)}); every step bitwise the VFY + DH-D chain")


# ---------------------------------------------------------------- F. no host read; eager uniforms


def _body_call(monkeypatch, R=20, F=8):
    kh.install(monkeypatch, k1=True)
    kp = kh.make_pair_proposer("k1", F=F)
    kh.warm(kp, [R])
    kh.agree(kp)
    captured = {}
    real = kp._draft_asm._k1_call

    def grab(module, md, args, T, serves):
        captured.update(module=module, md=md, args=args, T=T, serves=serves)
        return real(module, md, args, T, serves)

    monkeypatch.setattr(kp._draft_asm, "_k1_call", grab)
    kh.serve(kp, kh.make_vstep(29000, R), F, 29000)
    return kp, captured


def test_k1_body_has_no_host_read_and_traces_as_one_graph(monkeypatch):
    """Eager: no explicit host read (.item / .tolist / .cpu / .numpy refused), and every aten scalar read
    (_local_scalar_dense) sits in VFY-ONE's pre-existing verify code (vp_sampler / top_k_top_p: eager-only argument
    checks of aten ops), none in k1fuse.py, draft_one.py, draft_asm.py or the draft. Compiled: ONE fullgraph trace
    with Dynamo's scalar / dynamic-output capture off (any host read would be a graph break, an error), and the FX
    graph holds no scalar-read node; outputs equal."""
    kp, c = _body_call(monkeypatch)
    region = c["module"]
    prog = region.module
    import traceback

    sites = collections.Counter()

    class ScalarReads(TorchDispatchMode):
        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            if func.overloadpacket.__name__ == "_local_scalar_dense":
                frames = [f for f in traceback.extract_stack()[:-1]
                          if "/src/vllm_torchtpu/" in f.filename or "/tests/q38_" in f.filename]
                f = frames[-1] if frames else None
                sites[f"{f.filename.rsplit('/', 1)[-1]}:{f.name}" if f else "unknown"] += 1
            return func(*args, **(kwargs or {}))

    def refuse(name):
        def f(*a, **k):
            raise AssertionError(f"Tensor.{name} in the K1 body")
        return f

    def run(fn):
        snap = [b.clone() for b in c["args"][-3:]]
        kv = kp.draft_model.kv.clone()
        with h.fake_forward_context(c["md"]):
            out = fn(*c["args"])
        for b, s in zip(c["args"][-3:], snap):
            b.copy_(s)
        kp.draft_model.kv.copy_(kv)
        return out

    want = run(prog)
    with pytest.MonkeyPatch.context() as m:
        for name in ("item", "tolist", "cpu", "numpy"):
            m.setattr(torch.Tensor, name, refuse(name))
        with ScalarReads():
            got = run(prog)
    for a, b in zip(got, want):
        kh.same(a, b, "K1 body under the scalar-read census")
    allowed = ("vp_sampler.py:", "top_k_top_p.py:")
    assert all(k.startswith(allowed) for k in sites), dict(sites)
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    with torch._dynamo.config.patch(capture_scalar_outputs=False, capture_dynamic_output_shape_ops=False):
        compiled = torch.compile(prog.forward, backend=backend, fullgraph=True, dynamic=False)
        got2 = run(compiled)
    assert len(graphs) == 1
    bad = [str(n.target) for n in graphs[0].graph.nodes
           if "_local_scalar_dense" in str(n.target) or (n.op == "call_method" and n.target in ("item", "tolist"))]
    assert not bad, bad
    for a, b in zip(got2, want):
        kh.same(a, b, "K1 body as one compiled graph")
    COUNTS["no_host_read"] += 1
    print(f"\nK1FUSE_NOHOST K1 body: no .item/.tolist/.cpu/.numpy; eager aten scalar reads only in VFY's verify code "
          f"{dict(sites)} (none in K1 / draft_body); compiled: one fullgraph trace, 0 scalar-read nodes; outputs equal")


RNG_OPS = {"uniform_", "uniform", "rand", "rand_like", "randn", "randn_like", "random_", "randint", "bernoulli",
           "bernoulli_", "normal_", "multinomial", "exponential_", "randperm"}


def test_uniforms_stay_eager_in_order_and_none_is_drawn_inside_the_program(monkeypatch):
    kp, bp = _pair(monkeypatch, 8)
    _warm_both(monkeypatch, kp, bp, [24])
    _agree_both(monkeypatch, kp, bp)
    vk, vb = kh.make_vstep(29500, 24), kh.make_vstep(29500, 24)
    regions = list(kp._draft_asm.k1_modules.values())
    for r in regions:
        r.ops.clear()
    ok, ob = _serve_both(monkeypatch, kp, bp, vk, vb, 8, 29500)
    assert torch.equal(ok.gen_state, ob.gen_state)
    draws_k = [o for o in ok.eager if o in RNG_OPS]
    draws_b = [o for o in ob.eager if o in RNG_OPS]
    assert draws_k == draws_b and len(draws_k) == 4, (draws_k, draws_b)  # bonus, accept, recover, fallback rows
    ops = [o for r in regions for o in r.ops]
    inside = [o for o in ops if o in RNG_OPS]
    assert ok.calls.get("k1") == 1 and len(ops) > 50 and not inside, (ok.calls, len(ops), inside)
    COUNTS["uniforms"] += 1
    print(f"\nK1FUSE_UNIFORMS eager draws {draws_k} (same order as the chain), random stream equal, 0 RNG ops inside "
          f"the K1 program")


# ---------------------------------------------------------------- G. CHECK


def test_check_runs_k1_first_compares_then_arms(monkeypatch):
    rec = {"info": [], "error": [], "warning": []}
    kp, bp = _pair(monkeypatch, 8, check=3)
    _warm_both(monkeypatch, kp, bp, [16], check=3)
    monkeypatch.setattr(draft_asm, "AGREE_EVERY", 4)
    kh.install(monkeypatch, k1=True, check=3)
    with kh.quiet(monkeypatch, rec):
        _agree_both(monkeypatch, kp, bp, check=3)
    ctl = kp._draft_asm
    paths = []
    for i in range(9):
        seed = 30000 + i
        vk, vb = kh.make_vstep(seed, 16), kh.make_vstep(seed, 16)
        kh.install(monkeypatch, k1=False)
        ob = kh.serve(bp, vb, 8, seed)
        kh.install(monkeypatch, k1=True, check=3)
        with kh.quiet(monkeypatch, rec):
            ok = kh.serve(kp, vk, 8, seed)
        kh.assert_same_step(ok, ob, kp.runner, bp.runner, f"CHECK run step {i}")
        if ok.calls.get("k1") and ok.calls.get("vfy_one") and ok.calls.get("one"):
            paths.append("C")  # K1 first, then the VFY + DH-D chain serves
        elif ok.calls.get("k1") and not ok.calls.get("vfy_one") and not ok.calls.get("one"):
            paths.append("K")
        else:
            paths.append("-" if not ok.calls.get("k1") else "?")
    assert ctl.k1_stats["checked"] == 3 and ctl.k1_mismatches == 0 and not ctl.k1_failed
    path = "".join(paths)
    assert path.startswith("CCC") and "K" in path and "C" not in path[3:], path
    assert any("K1FUSE CHECK: 3 steps, 0 mismatches" in m for m in rec["info"]), rec["info"]
    COUNTS["check_steps"] += 3
    print(f"\nK1FUSE_CHECK window 3: path {path} (C = K1 first then the chain serves, K = K1 serves); 0 mismatches; "
          f"armed at the next boundary; every step bitwise the chain")


@pytest.mark.parametrize("what", ["drafts", "source", "tokens"])
def test_check_catches_a_perturbed_k1_output_and_turns_k1_off(monkeypatch, what):
    rec = {"info": [], "error": [], "warning": []}
    kp, bp = _pair(monkeypatch, 8, check=2)
    _warm_both(monkeypatch, kp, bp, [16], check=2)
    monkeypatch.setattr(draft_asm, "AGREE_EVERY", 3)
    kh.install(monkeypatch, k1=True, check=2)
    with kh.quiet(monkeypatch, rec):
        _agree_both(monkeypatch, kp, bp, check=2)
    ctl = kp._draft_asm
    real = ctl._k1_call
    slot = {"drafts": 4, "source": 5, "tokens": 0}[what]

    def perturbed(module, md, args, T, serves):
        out = list(real(module, md, args, T, serves))
        out[slot] = out[slot].clone()
        out[slot].view(-1)[0] += 1
        return tuple(out)

    monkeypatch.setattr(ctl, "_k1_call", perturbed)
    k1_served = 0
    for i in range(7):
        seed = 31000 + i
        vk, vb = kh.make_vstep(seed, 16), kh.make_vstep(seed, 16)
        kh.install(monkeypatch, k1=False)
        ob = kh.serve(bp, vb, 8, seed)
        kh.install(monkeypatch, k1=True, check=2)
        with kh.quiet(monkeypatch, rec):
            ok = kh.serve(kp, vk, 8, seed)
        kh.assert_same_step(ok, ob, kp.runner, bp.runner, f"perturbed CHECK step {i}")
        k1_served += bool(ok.calls.get("k1")) and not ok.calls.get("one")
    assert ctl.k1_mismatches > 0 and ctl.k1_failed and not ctl.k1_agreed, (ctl.k1_mismatches, ctl.k1_failed)
    assert k1_served == 0
    assert any("K1FUSE CHECK mismatch" in m for m in rec["error"]), rec["error"]
    assert any("every rank runs the VFY + DH-D chain" in m for m in rec["error"]), rec["error"]
    COUNTS["check_caught"] += 1
    print(f"\nK1FUSE_CHECK perturbed {what}: caught ({ctl.k1_mismatches} entries), K1 off on every rank at the next "
          f"boundary, 0 K1-served steps; serving results bitwise the chain throughout")


# ---------------------------------------------------------------- H. the take


class _MutLog(TorchDispatchMode):
    """Every tensor an aten op writes (in-place or out=) while active."""

    def __init__(self):
        super().__init__()
        self.written: list = []

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        kwargs = kwargs or {}
        schema = func._schema
        for i, a in enumerate(schema.arguments):
            if a.alias_info is not None and a.alias_info.is_write:
                v = args[i] if i < len(args) else kwargs.get(a.name)
                if isinstance(v, torch.Tensor):
                    self.written.append((func.overloadpacket.__name__, v))
        return func(*args, **kwargs)


class _FakeEvent:
    def record(self):
        pass

    def synchronize(self):
        pass

    def query(self):
        return True


def test_no_write_between_verify_and_propose_touches_a_k1_input(monkeypatch):
    """By construction: the writes sample_tokens makes between the verify and the propose on a K1 step are D83's
    read-offset index_put_ (inside _vfy_verify, after the program), the stats rebinding, the sampled-token copy
    (AsyncTPUCopyState.from_device_chunks) and the host-side rollback of the previous step (_modify_prev_results:
    host lists only). The spec substitution and the placeholder update run after the propose. None writes the tokens
    or any K1 draft input: the mutation log records no such tensor and the in-place version counters are equal."""
    from vllm_torchtpu.runner import tpu_runner_async_output as tao

    if hasattr(torch, "tpu"):
        monkeypatch.setattr(torch.tpu, "Event", _FakeEvent, raising=False)
    else:
        monkeypatch.setattr(torch, "tpu", SimpleNamespace(Event=_FakeEvent), raising=False)
    monkeypatch.setattr(tao.envs, "VLLM_TPU_PINNED_TOKEN_COPY", False, raising=False)
    kh.install(monkeypatch, k1=True)
    kp = kh.make_pair_proposer("k1", F=8)
    kh.warm(kp, [20])
    kh.agree(kp)
    ctl = kp._draft_asm
    log = _MutLog()
    seen = {}
    real_verify = ctl.k1_verify

    def spy(runner, vchunk, call, spec, num_active):
        res = real_verify(runner, vchunk, call, spec, num_active)
        seen["stash"] = ctl.k1_stash
        log.__enter__()  # from right after the program call to the take
        return res

    monkeypatch.setattr(ctl, "k1_verify", spy)

    def between(r, nt):
        seen["copy"] = tao.AsyncTPUCopyState.from_device_chunks([nt], [20])
        # the previous step's rollback, as sample_tokens runs it before the propose (host lists / host tokens)
        req = SimpleNamespace(output_token_ids=[1, 2, 3, 4, 5])
        r.input_batch.req_id_to_index = {"r0": 0}
        r.input_batch.num_tokens_no_spec = np.array([10], dtype=np.int64)
        r.input_batch.token_ids_cpu = np.zeros((1, 64), dtype=np.int32)  # host numpy, as InputBatch's
        prev = torch.full((1, kh.K + 1), -1, dtype=torch.int32)
        prev[0, :2] = torch.tensor([7, 8], dtype=torch.int32)
        r._pre_async_results = SimpleNamespace(
            req_ids=["r0"], request_seq_lens=[(0, req, 10, "r0")], discard_sampled_tokens_req_indices=[],
            num_draft_per_req={0: 3}, wait_for_copy=lambda: prev)
        r._modify_prev_results()
        assert req.output_token_ids == [1, 7, 8] and r.input_batch.token_ids_cpu[0, 6:8].tolist() == [7, 8]

    real_take = ctl.k1_result

    def take(chunks, nts, rd):
        log.__exit__(None, None, None)
        stash = ctl.k1_stash
        watched = {id(t) for t in stash.watch}
        hits = [(op, tuple(t.shape)) for op, t in log.written if id(t) in watched]
        seen["hits"], seen["writes"] = hits, sorted({op for op, _ in log.written})
        seen["vers_equal"] = vfy_one.versions(stash.watch) == stash.vers
        return real_take(chunks, nts, rd)

    monkeypatch.setattr(ctl, "k1_result", take)
    o = kh.serve(kp, kh.make_vstep(32000, 20), 8, 32000, between=between)
    assert o.calls.get("k1") == 1 and isinstance(o.result, k1fuse.K1Result)
    assert seen["hits"] == [] and seen["vers_equal"], seen
    assert "index_put_" in seen["writes"], seen["writes"]  # D83's scatter ran in the window (on the read offsets)
    assert len(seen["stash"].watch) >= 11
    COUNTS["take_construction"] += 1
    print(f"\nK1FUSE_TAKE writes between the verify and the propose: {seen['writes']}; on the tokens or a K1 draft "
          f"input ({len(seen['stash'].watch)} tensors watched): 0; version counters equal; taken")


@pytest.mark.parametrize("target", ["seq_lens", "input_ids", "tokens"])
def test_a_write_to_a_k1_input_makes_the_take_raise(monkeypatch, target):
    kh.install(monkeypatch, k1=True)
    kp = kh.make_pair_proposer("k1", F=0)
    kh.warm(kp, [20])
    kh.agree(kp)
    vs = kh.make_vstep(33000, 20)

    def between(r, nt):
        t = {"seq_lens": vs.step.chunk.attn_ctx.seq_lens, "input_ids": vs.step.chunk.input_ids, "tokens": nt}[target]
        t.add_(0)  # same values, an in-place write all the same

    with pytest.raises(RuntimeError, match="q38 K1FUSE: the propose after a fused verify"):
        kh.serve(kp, vs, 0, 33000, between=between)
    assert kp._draft_asm.k1_stats["take_miss"] == 1
    COUNTS["take_raise"] += 1


def test_a_result_never_taken_raises_at_the_next_verify(monkeypatch):
    kh.install(monkeypatch, k1=True)
    kp = kh.make_pair_proposer("k1", F=0)
    kh.warm(kp, [12])
    kh.agree(kp)
    vs = kh.make_vstep(33500, 12)
    kp.runner.warmup_done = True
    kh.verify(kp.runner, vs, 0, 33500)
    assert kp._draft_asm.k1_stash is not None
    with pytest.raises(RuntimeError, match="never taken"):
        kh.verify(kp.runner, kh.make_vstep(33501, 12), 0, 33501)
    COUNTS["untaken"] += 1


def test_zz_counts():
    print("Q38_K1FUSE_COUNTS " + " ".join(f"{k}={v}" for k, v in sorted(COUNTS.items())))
