# SPDX-License-Identifier: Apache-2.0
"""q38 L4 v6 composition (q38-l4-v6c2): T1-K1FUSE with DRAFT-CAP, VRNG-VPS, SEEDFIX, KEYLITE and the AGREE clock.
CDK CPU pod only. V6COMP-BUILD (helper of L4-E).

Driven through K1FUSE's own harness (tests/q38_k1fuse/k1fuse_harness.py: the real TPUModelRunner._sample_spec_verify_vps
-> _vfy_verify on a stub runner of the real class, the real Eagle3Proposer.propose and async spec substitution; every
compiled program its own body in a counted region), with DRAFT-CAP's per-group metadata (tests/q38_draftcap). Each case
is bitwise against its own flag-off path: the same steps through DraftOne (today's VFY + DH-D chain), same composition.

K1C  K1 x DRAFT-CAP, 8 serving steps at 8 / 20 / 33 / 64 rows, fallback rows 0 / 8: K1 serves every step through its
     R_d program class (k1fuse._K1FuseCapProgram), bitwise the VFY + DH-D (DRAFT-CAP) chain (tokens, generator, drafts,
     source, rejected count, index dicts, stats, read offsets, draft KV writes); every draft pass inside K1's program
     reads seq_lens / query_start_loc / every group's table at R_d = p (the loop bucket) slots, the same per-pass census
     as DRAFT-CAP's DH-D program on the same step.
K1V  K1 x VRNG-VPS (configured, path off "0" and on "1"): K1 refuses for the run (static reason vrng_ingraph_uniforms,
     ONE warning), warms and builds no K1 program, never serves; bitwise the chain with the same VRNG; with "1" every
     step ran VRNG's in-graph-draw VFY program. Control: the same harness without VRNG serves K1.
K1O  structural: in tpu_runner._vfy_verify the K1 hook sits only in the `else` of `if rng is not None:` (VRNG first).
K1S  K1 x SEEDFIX, all-seeded / mixed / unseeded batches, fallback rows 0 / 8: K1 serves (seeded chunks are NOT sent to
     the base path under VFY-ONE / K1: SEEDFIX overlays the eager draws before the call), bitwise the chain, overlay
     engaged.
K1K  K1 x KEYLITE: K1 serves every warm step with VFY's KEYLITE verdict memoised (hits > 0), bitwise the chain; a step
     whose K1 key was never warmed runs the VFY program (K1 reason cold) even though KEYLITE says warm.
K1E  eagle3.__init__'s own DRAFT-ASM block builds k1fuse.DraftK1 only under VLLM_TPU_K1FUSE (the flag at its build site).
K1A  K1 x l4_agree: K1's agreement bits (8 / 16 / 32 / 64) are disjoint from DraftOne's (1 / 2 / 4) and K1 adds no slot
     (the same lever set, widths and header as a DraftOne rank); through one Agreement gather with two other levers'
     slots, DH-D on / K1 off, DH-D off / K1 on and both on decide exactly as their own bits say, and the other slots
     get their own entries untouched.
Run: python -m pytest -q -p no:cacheprovider -p no:logging -s tests/q38_l4_v5c/test_v6c_k1.py
"""

import ast
import collections
import os
import pathlib
import sys
from types import SimpleNamespace as NS

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "q38_k1fuse"))
sys.path.insert(0, str(HERE.parent / "q38_draftcap"))
import draftcap_harness as hc  # noqa: E402
import k1fuse_harness as kh  # noqa: E402

from vllm_torchtpu.runner import l4_agree, l4_samp2, tpu_runner, vfy_one  # noqa: E402
from vllm_torchtpu.runner import seedfix as sf  # noqa: E402
from vllm_torchtpu.spec_decode import draft_asm, draft_cap, draft_one, k1fuse  # noqa: E402

COUNTS: collections.Counter = collections.Counter()


# ---------------------------------------------------------------- K1C: K1 x DRAFT-CAP


class FakeMtpCapKV(hc.RecMtpCap):
    """DRAFT-CAP's recording stub draft (every group's table row per slot, seq_lens / query_start_loc / request
    distribution) plus K1FUSE's KV writes (kv[slot] = round(1000 * h[:, 0]), slot from positions, seq_lens, step and the
    slot's table features), so a draft pass at the wrong slot count or table changes the cache."""

    def __init__(self):
        super().__init__()
        self.kv = torch.zeros(kh.KV_SLOTS, dtype=torch.int64)

    def forward(self, input_ids, positions, hidden_states, intermediate_tensors=None, inputs_embeds=None,
                spec_step_idx=0, draft_rows=None):
        hh, nxt = super().forward(input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds,
                                  spec_step_idx, draft_rows)
        md = kh.h.CTX["md"]
        m = md[hc.MAIN]
        pos = positions[draft_rows.clamp(min=0)] if draft_rows is not None else positions
        n = hh.shape[0]
        feat = (hc._slot_features(md, m.seq_lens.shape[0]) * 1000).round().long()
        slot = (kh._fit(pos.long(), n) * 131 + kh._fit(m.seq_lens.long(), n) * 7 + spec_step_idx * 17
                + kh._fit(feat, n) * 3) % kh.KV_SLOTS
        self.kv.index_put_((slot,), (hh[:, 0].float() * 1000).round().to(torch.int64))
        return hh, nxt


class CapK1Region(kh.K1Region):
    """K1's program as a counted region, for either program class (K1's 33 arguments, + DRAFT-CAP's 8)."""

    def __call__(self, *args):
        assert len(args) in (len(k1fuse.ARG_NAMES), len(k1fuse.ARG_NAMES) + len(k1fuse.CAP_ARG_NAMES)), len(args)
        kh.CEN.calls[self.name] += 1
        kh.CEN.keys.append((self.name, vfy_one.key(*args)))
        kh.CEN.depth += 1
        try:
            return self.module(*args)
        finally:
            kh.CEN.depth -= 1


def _cap_install(monkeypatch, k1, check=0):
    kh.install(monkeypatch, k1=k1, check=check)
    monkeypatch.setenv(hc.CAP_ENV, "1")

    def build_cap(*, vllm_config, prefix, mtp, num_steps, dynamic_arg_dims, dvp=False, max_reqs, p):
        m = draft_cap._cap_program_class(p)(vllm_config=None, prefix=prefix, mtp=mtp, num_steps=num_steps, dvp=dvp,
                                             max_reqs=max_reqs)
        region = hc.CapRegion(m, "one")
        region.p = p
        return region

    def build_k1(*, vllm_config, prefix, mtp, num_steps, dvp=False, vh, statics, index, cap=False):
        cls = (k1fuse._cap_program_class if cap else k1fuse._program_class)(index)
        region = CapK1Region(cls(vllm_config=None, prefix=prefix, mtp=mtp, num_steps=num_steps, dvp=dvp, vh=vh,
                                 statics=statics))
        region.index, region.statics, region.cap = index, statics, cap
        kh.BUILT.append(region)
        return region

    monkeypatch.setattr(draft_cap, "build_draft_cap_program", build_cap)
    monkeypatch.setattr(k1fuse, "build_k1fuse_program", build_k1)


def _cap_proposer(kind, F):
    p = kh.make_pair_proposer(kind, F=F, mtp_cls=FakeMtpCapKV)
    p._draft_attn_layer_names = set(hc.NAMES)
    return p


def _vstep_cap(seed, R):
    vs = kh.make_vstep(seed, R)
    hc.group_md(vs.step, seed)
    return vs


def _census(seen):
    """Per draft forward: (step index, seq_lens, query_start_loc, request_distribution shapes, every group's table
    shape, draft_rows shape)."""
    return [(s, tuple(sl.shape), tuple(q.shape), tuple(rd.shape), tuple(sorted((n, tuple(t.shape)) for n, t in tb.items())),
             None if rows is None else tuple(rows.shape)) for s, sl, q, rd, tb, rows in seen]


@pytest.mark.parametrize("F", [0, 8])
@pytest.mark.parametrize("R", [8, 20, 33, 64])
def test_k1c_k1_cap_runs_at_r_d_and_is_bitwise_the_cap_chain(monkeypatch, R, F):
    with kh.quiet(monkeypatch):
        _cap_install(monkeypatch, k1=False)
        bp = _cap_proposer("one", F)
        _cap_install(monkeypatch, k1=True)
        kp = _cap_proposer("k1", F)
        assert kp._draft_asm.one_cap and bp._draft_asm.one_cap
        for p in (bp, kp):
            kh.h.run_step(p, hc.make_step(51000 + R, R, cap=kh.CAP), serving=False)
            p.draft_model.kv.zero_()  # K1's warm-up calls wrote the synthetic requests' KV too (K1FUSE's _warm_both)
            kh.agree(p)
        p_rows = kp._loop_bucket(R)
        for i in range(8):
            seed = 52000 + 100 * R + 10 * F + i
            vb, vk = _vstep_cap(seed, R), _vstep_cap(seed, R)
            nb, nk = len(bp.draft_model.seen), len(kp.draft_model.seen)
            ob = kh.serve(bp, vb, F, seed)
            ok = kh.serve(kp, vk, F, seed)
            tag = f"K1C R={R} F={F} step {i}"
            kh.assert_same_step(ok, ob, kp.runner, bp.runner, tag)
            assert ok.calls.get("k1", 0) == 1 and ok.calls.get("one", 0) == 0, (tag, ok.calls,
                                                                              dict(kp._draft_asm.k1_reasons))
            assert ob.calls.get("one", 0) == 1, (tag, ob.calls, dict(bp._draft_asm.reasons))
            sb, sk = _census(bp.draft_model.seen[nb:]), _census(kp.draft_model.seen[nk:])
            assert sk == sb, (tag, sk, sb)
            assert len(sk) == kh.K and all(c[1] == (p_rows,) and c[2] == (p_rows + 1,) for c in sk), (tag, p_rows, sk)
            assert all(dict(c[4])[g] == (p_rows * hc.COLS[g],) for c in sk for g in hc.GROUPS), (tag, sk)
    regions = [r for r in kh.BUILT if isinstance(r, CapK1Region)]
    assert regions and all(r.cap and isinstance(r.module, k1fuse._K1FuseCapProgram) for r in regions), regions
    assert kp._draft_asm.counts()["folded_k1"] == 8
    COUNTS["k1c"] += 1
    print(f"\nV6C_K1 K1C R={R} F={F} steps=8 K1 R_d program 8/8 (p = R_d = {p_rows}, capacity {kh.CAP}); per-pass census "
          f"= DRAFT-CAP's DH-D program's; bitwise the VFY + DH-D (DRAFT-CAP) chain incl. draft KV writes")


# ---------------------------------------------------------------- K1V / K1O: K1 x VRNG-VPS


def _vrng(mode):
    ctl = l4_samp2.VrngCtl(3, torch.device("cpu"), ab=mode)
    ctl.event_fn = lambda: NS(query=lambda: True)
    if mode == "1":
        ctl.warm = kh.AlwaysWarm()
        ctl.agreed = True
    return ctl


@pytest.mark.parametrize("R", [8, 33, 64])
@pytest.mark.parametrize("mode", ["0", "1"])
def test_k1v_k1_refuses_for_the_run_when_vrng_is_configured(monkeypatch, mode, R):
    """v6 configuration V (DRAFT-CAP + VRNG-VPS + K1 configured): K1 refuses; DH-D (DRAFT-CAP) serves."""
    rec = {"info": [], "error": [], "warning": []}
    F = 0
    with kh.quiet(monkeypatch, rec):
        _cap_install(monkeypatch, k1=False)
        bp = _cap_proposer("one", F)
        _cap_install(monkeypatch, k1=True)
        kp = _cap_proposer("k1", F)
        monkeypatch.setattr(tpu_runner, "_vfy_one_rng_compiled",
                            kh.h.region("vfy_one_rng", tpu_runner._vfy_one_rng_impl))
        ctls = {}
        for name, p in (("base", bp), ("k1", kp)):
            ctls[name] = p.runner._vrng = _vrng(mode)
            kh.h.run_step(p, hc.make_step(53000 + R, R, cap=kh.CAP), serving=False)
            p.draft_model.kv.zero_()
            kh.agree(p)
        for i in range(6):
            seed = 53100 + 10 * R + i
            ob = kh.serve(bp, _vstep_cap(seed, R), F, seed)
            ok = kh.serve(kp, _vstep_cap(seed, R), F, seed)
            kh.assert_same_step(ok, ob, kp.runner, bp.runner, f"K1V mode={mode} R={R} step {i}")
            assert ok.calls.get("k1", 0) == 0 and ok.calls.get("one", 0) == 1, (i, ok.calls)
            assert ctls["k1"].counter_word() == ctls["base"].counter_word()
    ctl = kp._draft_asm
    assert ctl.k1_static_reason == "vrng_ingraph_uniforms", ctl.k1_static_reason
    assert not ctl.k1_modules and not ctl.warm["k1"], (len(ctl.k1_modules), len(ctl.warm["k1"]))
    # mode "0": every step reaches K1's hook and is refused (static); mode "1": every step is a VRNG step, which never
    # reaches the hook (the rng branch comes first), so the refusal shows only as the static reason and no K1 warm-up
    want_static = 6 if mode == "0" else 0
    assert ctl.k1_reasons["static"] == want_static and ctl.stats["folded_k1"] == 0, dict(ctl.k1_reasons)
    warns = [m for m in rec["warning"] if "K1FUSE stays off on this runner: vrng_ingraph_uniforms" in m]
    assert len(warns) == 1, rec["warning"]
    if mode == "1":
        assert ctls["k1"].vrng_calls == 6 and kp.runner._vfy.stats["vfy_vrng"] == 6, (
            ctls["k1"].vrng_calls, dict(kp.runner._vfy.stats))
    else:
        assert ctls["k1"].vrng_calls == 0 and ctls["k1"].stats["window_off"] == 6, dict(ctls["k1"].stats)
    # control: the same harness, no VRNG (configuration K): K1 serves
    with kh.quiet(monkeypatch):
        _cap_install(monkeypatch, k1=True)
        cp = _cap_proposer("k1", F)
        kh.h.run_step(cp, hc.make_step(53000 + R, R, cap=kh.CAP), serving=False)
        kh.agree(cp)
        oc = kh.serve(cp, _vstep_cap(53100 + 10 * R, R), F, 53100 + 10 * R)
    assert oc.calls.get("k1", 0) == 1, (oc.calls, dict(cp._draft_asm.k1_reasons))
    COUNTS["k1v"] += 1
    print(f"\nV6C_K1 K1V config V (DRAFT-CAP + VRNG mode={mode} + K1 configured) R={R}: K1 refused for the run (reason vrng_ingraph_uniforms, 1 warning), 0 K1 programs "
          f"warmed / built, 6/6 steps bitwise the chain with the same VRNG (VRNG calls {ctls['k1'].vrng_calls}); "
          f"control without VRNG serves K1")


def test_k1o_the_k1_hook_sits_only_in_the_non_vrng_arm():
    src = pathlib.Path(tpu_runner.__file__).read_text()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "_vfy_verify")

    def reads_k1(node):
        return any(isinstance(x, ast.Constant) and x.value == "k1_verify" for x in ast.walk(node))

    hooks = [n for n in ast.walk(fn) if isinstance(n, ast.If) and reads_k1(n.test)]
    assert len(hooks) == 1, len(hooks)
    # the serving step's program choice: a statement of the function body itself (the CHECK branch's own
    # `if rng is not None:` sits inside `if mode == "check":`)
    rng_ifs = [n for n in fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == "rng is not None"
               and any("_vfy_one_rng_compiled" in ast.unparse(s) for s in n.body) and n.orelse]
    assert len(rng_ifs) == 1, len(rng_ifs)
    inside = [x for s in rng_ifs[0].orelse for x in ast.walk(s)]
    assert any(x is hooks[0] for x in inside), "the K1 hook is not in the non-VRNG arm"
    assert not any(reads_k1(x) for s in rng_ifs[0].body for x in ast.walk(s))
    COUNTS["k1o"] += 1
    print(f"\nV6C_K1 K1O tpu_runner._vfy_verify: the one K1 hook (line {hooks[0].lineno}) is in the else of "
          f"`if rng is not None:` (line {rng_ifs[0].lineno}); VRNG steps never reach K1")


# ---------------------------------------------------------------- K1E: eagle3 builds DraftK1 only under the flag


def test_k1e_eagle3_builds_draft_k1_only_under_its_flag(monkeypatch):
    """Eagle3Proposer.__init__'s own DRAFT-ASM construction block (`if envs.VLLM_TPU_DRAFT_ASM_FOLD:`, taken from the
    tree's eagle3.py source and run with eagle3's module globals on the harness proposer): VLLM_TPU_K1FUSE=1 builds
    k1fuse.DraftK1, unset builds draft_one.DraftOne (K1FUSE's own build test constructs DraftK1 directly)."""
    from vllm_torchtpu.spec_decode import eagle3

    src = pathlib.Path(eagle3.__file__).read_text()
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "Eagle3Proposer")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    block = [s for s in init.body if isinstance(s, ast.If) and ast.unparse(s.test) == "envs.VLLM_TPU_DRAFT_ASM_FOLD"]
    assert len(block) == 1, len(block)
    mod = ast.parse("def _q38_build(self):\n    pass\n")
    mod.body[0].body = block
    ast.fix_missing_locations(mod)
    ns = dict(vars(eagle3))
    exec(compile(mod, eagle3.__file__, "exec"), ns)  # noqa: S102 - the tree's own source block
    got = {}
    with kh.quiet(monkeypatch):
        for k1 in (True, False):
            kh.install(monkeypatch, k1=k1)
            p = kh.h1.make_proposer(fold=False, cap=kh.CAP)
            p.runner.__dict__.pop("_l4_agree", None)
            ns["_q38_build"](p)
            got[k1] = type(p._draft_asm)
    assert got[True] is k1fuse.DraftK1 and got[False] is draft_one.DraftOne, got
    COUNTS["k1e"] += 1
    print(f"\nV6C_K1 K1E eagle3.__init__'s DRAFT-ASM block: VLLM_TPU_K1FUSE=1 -> {got[True].__name__}, unset -> "
          f"{got[False].__name__}")


# ---------------------------------------------------------------- K1S: K1 x SEEDFIX


def _seed_runner(p, R, seeded):
    r = p.runner
    r.vocab_size = kh.VT
    r.device = torch.device("cpu")
    r._q38_gens = {i: torch.Generator().manual_seed(7700 + i) for i in range(R) if seeded(i)}
    r.requests = {f"r{i}": NS(num_computed_tokens=96, num_tokens=100) for i in range(R)}
    return r


def _seed_serve(p, vs, F, seed):
    """kh.serve with the seeded requests' generators in the input batch and SEEDFIX's plan built as sample_tokens
    builds it (before the verify; cleared after the step)."""
    r = p.runner
    real = kh.SimpleNamespace

    def ns(**kw):
        if set(kw) == {"num_reqs", "req_ids"}:
            n = kw["num_reqs"]
            kw["generators"] = r._q38_gens
            # the CPU sampling params SEEDFIX's fallback-slot map reads, as the harness's _padded_sampling_params
            kw.update(temperature_cpu_tensor=r.temps[:n].clone(), top_k_cpu_tensor=r.top_k[:n].clone(),
                      top_p_cpu_tensor=torch.full((n,), 0.95))
            sched = NS(num_scheduled_tokens={rid: kh.K + 1 for rid in kw["req_ids"]})
            out = real(**kw)
            r.input_batch = out
            r._seed_plan = sf.build_plan(r, sched) if r._q38_gens else None
            return out
        return real(**kw)

    kh.SimpleNamespace = ns
    try:
        return kh.serve(p, vs, F, seed)
    finally:
        kh.SimpleNamespace = real
        r._seed_plan = None


@pytest.mark.parametrize("F", [0, 8])
@pytest.mark.parametrize("batch", ["all", "mixed", "none"])
def test_k1s_k1_with_seeded_chunks_is_bitwise_the_chain(monkeypatch, batch, F):
    monkeypatch.delenv("VLLM_TPU_SEEDED_SAMPLING", raising=False)
    sf._reset_for_tests()
    R = 20
    seeded = {"all": lambda i: True, "mixed": lambda i: i % 2 == 1, "none": lambda i: False}[batch]
    with kh.quiet(monkeypatch):
        kh.install(monkeypatch, k1=False)
        bp = kh.make_pair_proposer("one", F=F)
        kh.install(monkeypatch, k1=True)
        kp = kh.make_pair_proposer("k1", F=F)
        for p in (bp, kp):
            _seed_runner(p, R, seeded)
            kh.warm(p, [R], seed0=54000)
            p.draft_model.kv.zero_()
            kh.agree(p)
        for i in range(6):
            seed = 54100 + 10 * F + i
            ob = _seed_serve(bp, kh.make_vstep(seed, R), F, seed)
            ok = _seed_serve(kp, kh.make_vstep(seed, R), F, seed)
            kh.assert_same_step(ok, ob, kp.runner, bp.runner, f"K1S batch={batch} F={F} step {i}")
            assert ok.calls.get("k1", 0) == 1, (i, ok.calls, dict(kp._draft_asm.k1_reasons))
            for j in kp.runner._q38_gens:
                assert torch.equal(kp.runner._q38_gens[j].get_state(), bp.runner._q38_gens[j].get_state()), j
    c = sf.counts()
    if batch == "none":
        assert c["rows"] == 0
    else:
        assert c["rows"] > 0 and c["rows_ovl"] > 0 and c["tgt_ovl"] > 0, c
    COUNTS["k1s"] += 1
    print(f"\nV6C_K1 K1S batch={batch} F={F} steps=6: K1 served 6/6, bitwise the chain (tokens, shared generator, every "
          f"seeded request's generator, drafts, KV); SEEDFIX counts {c}")


# ---------------------------------------------------------------- K1K: K1 x KEYLITE


def test_k1k_k1_with_keylite(monkeypatch):
    rs = [12, 20, 33]
    with kh.quiet(monkeypatch):
        kh.install(monkeypatch, k1=False)
        bp = kh.make_pair_proposer("one", F=8)
        kh.install(monkeypatch, k1=True)
        kp = kh.make_pair_proposer("k1", F=8)
        for p in (bp, kp):
            p.runner._vfy.keylite = True
            kh.warm(p, rs, seed0=55000)
            p.draft_model.kv.zero_()
            kh.agree(p)
        for i in range(9):
            R, seed = rs[i % 3], 55100 + i
            ob = kh.serve(bp, kh.make_vstep(seed, R), 8, seed)
            ok = kh.serve(kp, kh.make_vstep(seed, R), 8, seed)
            kh.assert_same_step(ok, ob, kp.runner, bp.runner, f"K1K step {i} R={R}")
            assert ok.calls.get("k1", 0) == 1, (i, ok.calls, dict(kp._draft_asm.k1_reasons))
        st = kp.runner._vfy.stats
        assert st["keylite_hit"] > 0 and st["keylite_miss"] == 3, dict(st)
        # R = 60: loop / request bucket 64, which no warmed count (12 / 20 / 33 -> 16 / 32 / 48) reaches (smoke w4
        # j-9d3112b9: R = 40 shares 33's bucket 48, so its K1 key was warm)
        ob = kh.serve(bp, kh.make_vstep(55200, 60), 8, 55200)
        ok = kh.serve(kp, kh.make_vstep(55200, 60), 8, 55200)
        kh.assert_same_step(ok, ob, kp.runner, bp.runner, "K1K cold R=60")
    assert ok.calls.get("k1", 0) == 0 and ok.calls.get("vfy_one", 0) == 1, ok.calls
    rs_ = kp._draft_asm.k1_reasons  # an unwarmed bucket: no warm-up steady shape ("bucket") or an unwarmed key ("cold")
    assert rs_["cold"] + rs_["bucket"] == 1, dict(rs_)
    COUNTS["k1k"] += 1
    print(f"\nV6C_K1 K1K KEYLITE on: K1 served 9/9 warm steps (VFY keylite hit {st['keylite_hit']} miss "
          f"{st['keylite_miss']}), bitwise the chain; an unwarmed K1 key (R=60) ran the VFY program (K1 reasons {dict(kp._draft_asm.k1_reasons)})")


# ---------------------------------------------------------------- K1A: K1 x l4_agree


class _Stub:
    def __init__(self, name, width, base):
        self.AGREE_NAME, self.AGREE_WIDTH = name, width
        self.vec = [base + i for i in range(width)]
        self.rows = None

    def local_vector(self):
        return list(self.vec)

    def apply(self, rows):
        self.rows = [list(r) for r in rows]

    def agree_period(self):
        return 0


def test_k1a_k1_bits_ride_draftasm_entry_0_without_collision(monkeypatch):
    k1_bits = (k1fuse.BIT_K1, k1fuse.BIT_K1CONF, k1fuse.BIT_K1CLOSE, k1fuse.BIT_K1BAD)
    d1_bits = (draft_one.BIT_FOLD, draft_one.BIT_CL, draft_one.BIT_ONE)
    allb = k1_bits + d1_bits
    assert all(b and not b & (b - 1) for b in allb) and len(set(allb)) == len(allb), allb
    assert not set(k1_bits) & set(d1_bits)
    with kh.quiet(monkeypatch):
        kh.install(monkeypatch, k1=False)
        dp = kh.make_pair_proposer("one", F=0)
        kh.install(monkeypatch, k1=True)
        kp = kh.make_pair_proposer("k1", F=0)
        ad, ak = dp.runner._l4_agree, kp.runner._l4_agree
        assert [(s.name, s.width) for s in ak.slots] == [(s.name, s.width) for s in ad.slots] == [("draftasm", 5)]
        assert ak.schema == ad.schema
        stubs = [_Stub("vfyone", 8, 1000), _Stub("vrng", 4, 2000)]
        for s in stubs:
            l4_agree.register(kp.runner, s)
        kh.warm(kp, [20], seed0=56000)
        world = 8
        ak.transport = lambda vec: [list(vec)] * world
        wish = {}
        real = draft_one.DraftOne._stage_window
        monkeypatch.setattr(draft_one.DraftOne, "_stage_window",
                            lambda self, lever, ab: wish.get(lever, real(self, lever, ab)))
        ctl = kp._draft_asm
        kp.runner.warmup_done = True
        seen = []
        for one, k1 in ((True, False), (False, True), (True, True), (False, False)):
            wish.update({draft_one.ONE_LEVER: one, k1fuse.LEVER: k1})
            ak._agree()
            w0 = ak.local_vector()[l4_agree.HEADER + 0]  # draftasm sorts first
            assert ctl.one_agreed is one and ctl.k1_agreed is (one and k1), (one, k1, ctl.one_agreed, ctl.k1_agreed)
            assert bool(w0 & k1fuse.BIT_K1) is k1 and bool(w0 & draft_one.BIT_ONE) is one and w0 & k1fuse.BIT_K1CONF, w0
            for s in stubs:
                assert s.rows == [s.vec] * world, (s.AGREE_NAME, s.rows)
            seen.append((one, k1, ctl.one_agreed, ctl.k1_agreed, w0))
    COUNTS["k1a"] += 1
    print(f"\nV6C_K1 K1A bits K1 {k1_bits} vs DraftOne {d1_bits} disjoint; K1 adds no slot (lever set draftasm(5), schema "
          f"equal to a DraftOne rank's); through one gather with vfyone(8) + vrng(4) stub slots over {world} ranks: "
          f"(DH-D wish, K1 wish, DH-D agreed, K1 agreed, entry 0) = {seen}; the other slots got their own entries")


def test_zz_counts():
    print("\nV6C_K1_COUNTS " + " ".join(f"{k}={v}" for k, v in sorted(COUNTS.items())))
