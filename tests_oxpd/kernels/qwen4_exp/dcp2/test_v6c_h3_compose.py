# SPDX-License-Identifier: Apache-2.0
"""q38 L4 v6 composition (q38-l4-v6c2): the H3 draft attention kind (L1-H3 + ATTNWIRE, VLLM_TPU_MK_DRAFT_ATTN) inside
DRAFT-CAP's DH-D programs, inside T1-K1FUSE's verify + draft program, and beside LMH-STREAM's draft head. CPU, Pallas
interpret. V6COMP-BUILD (helper of L4-E).

Built on ATTNWIRE's T2 world (test_attnwire_compose.World: R TP ranks as threads, the real attach / DraftAttn / kind op /
build_draft_attn_call, the real Qwen4ExpMultiTokenPredictor.forward with the mk_tail path, stub layers around attention,
L1's oracle as today's attention per pass, run on a copy of the pool). The kind's numerics are judged by L1-H3's rule
(2 BF16 ulps + 1e-6, >= 99.9 % bitwise, pools byte-identical to today's after every pass: test_attnwire_compose
._check_recs); everything else bitwise against its own flag-off path.

HC  H3 x DH-D x DRAFT-CAP (VLLM_TPU_DRAFT_ASM_CAP), 4 steps, p = 16 / 48: DRAFT-CAP's R_d program class
    (draft_cap._cap_program_class: per-R_d metadata buffers and the request-major flat block table's [R_d * cols] buffer
    in the forward context) with the kind: every fused-loop pass calls the kind at B = p rows (the kernel built at p only)
    with R_d = p request slots (context / query_start_loc / main table at R_d); rule A on every pass; tokens, source and
    every pass's attention and pool digest bitwise the capacity DH-D program with the kind on the same inputs.
HK  H3 x T1-K1FUSE (capacity, and K1's DRAFT-CAP R_d program), 4 steps, p = 16 / 48: K1's program (VFY-ONE's verify,
    the superset seed and DH-D's body, k1fuse._program_class / _cap_program_class) with the kind vs the unfused chain
    (tpu_runner._vfy_one_impl with the seed = VFY-ONE value 2, then DH-D's / DRAFT-CAP's program on the seed outputs),
    same inputs, both with the kind: tokens, stats, read-offset operands, drafts, source, rejected count and every pass's
    attention and pool digest bitwise; B = p on every pass of both; rule A on every pass.
HL  H3 x LMH-STREAM stage 0 (VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT): with the draft's own hidden-major LM head armed before the
    attach (load order), the kind's packed operands and fingerprint are byte-identical to an attach without LMH, the attach
    leaves the head's bytes untouched, and a DH-D program with both (the draft's tokens through
    HiddenMajorLMHeadMethod via get_top_tokens) drafts the same tokens with bitwise the same per-pass attention as the kind
    with the stock-layout head over 4 steps.
Run (CDK CPU pod): JAX_PLATFORMS=cpu VLLM_TARGET_DEVICE=cpu python -m pytest -q -s -p q38_cpu_device
  tests/kernels/qwen4_exp/dcp2/test_v6c_h3_compose.py
"""

from __future__ import annotations

import hashlib
import logging
import os
import types
from types import SimpleNamespace as NS
from unittest.mock import patch

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import attnwire_harness as aw  # noqa: E402
import test_attnwire_compose as cw  # noqa: E402
import test_l1h3_draft_attn_call as l1c  # noqa: E402

K, CAP = cw.K, cw.CAP
STEPS = 4
VT, HIDV = 5440, 32  # the verify's TP1 vocab shard (K1FUSE's harness geometry: 34 whole VP_BLOCKs)
COUNTS: dict = {}


class _ShapeRec:
    """Outermost mk_attn wrapper: per call (rows B, metadata request slots, main-table entries)."""

    def __init__(self, inner, world, rec):
        self.inner, self.world, self.rec = inner, world, rec

    def __call__(self, block, pos, md, attn):
        main = self.world.spmd.tl.ctx.attn_metadata[attn.main_cache_name].block_tables
        self.rec.append((int(block.shape[0]), int(md.context_lens.shape[0]), int(main.numel())))
        return self.inner(block, pos, md, attn)


class V6World(cw.World):
    """ATTNWIRE's world plus: flat request-major tables (DRAFT-CAP), the DRAFT-CAP R_d program, a verify-capable vocab
    (the draft embeds the verify's tokens), K1FUSE's program and the unfused VFY + DH-D chain, per-call shape records."""

    def __init__(self, monkeypatch, R, D, seed, mk=True, vocab=None):
        super().__init__(monkeypatch, R, D, seed, mk=mk)
        if vocab is not None:
            gen = torch.Generator().manual_seed(seed + 1)
            self.emb = torch.randn(vocab, self.g.H, generator=gen) * 0.5
            self.wv = torch.randn(self.g.H, vocab, generator=gen)
        self.shapes = [[] for _ in range(R)]

    def setup(self, r):
        out = super().setup(r)
        rk = self.ranks[r]
        if self.mk:
            rk.pred.mk_attn = _ShapeRec(rk.pred.mk_attn, self, self.shapes[r])
        return out

    def _stock(self, block, pos, md, attn, pool, write=False):
        ctx = self.spmd.tl.ctx
        main = ctx.attn_metadata[attn.main_cache_name].block_tables
        if main.dim() == 1:  # DRAFT-CAP's flat request-major table: L1's oracle takes [requests, cols]
            md_main = ctx.attn_metadata[attn.main_cache_name]
            saved = md_main.block_tables
            md_main.block_tables = main.reshape(-1, self.cols)
            try:
                return super()._stock(block, pos, md, attn, pool, write)
            finally:
                md_main.block_tables = saved
        return super()._stock(block, pos, md, attn, pool, write)

    def positions_for(self, si):
        """Token positions consistent with the verify window: request i's 4 tokens at seq0[i] - 4 .. seq0[i] - 1."""
        pos = np.zeros(si["input_ids"].shape[0], np.int32)
        qsl = si["md0"][1].numpy()
        seq0 = si["md0"][0].numpy()
        for i in range(si["n"]):
            pos[qsl[i]:qsl[i] + 4] = seq0[i] - 4 + np.arange(4)
        return torch.from_numpy(pos)

    def install_cap(self, r, si, p):
        rk = self.ranks[r]
        r_d = min(p, CAP)
        md0 = si["md0"]
        bufs = (torch.zeros_like(md0[0][:r_d]), torch.zeros_like(md0[1][: r_d + 1]), torch.zeros_like(md0[2]))
        flat = torch.from_numpy(self.table.reshape(-1).copy())
        tbuf = torch.zeros(r_d * self.cols, dtype=flat.dtype)
        md = NS(seq_lens=bufs[0], query_start_loc=bufs[1], request_distribution=bufs[2], block_tables=tbuf)
        self.spmd.tl.ctx = NS(attn_metadata={rk.attn.prefix: md})
        rk.attn._selection_rows = torch.from_numpy(si["sel"])
        return bufs, flat, tbuf

    def cap_prog(self, r, p):
        from vllm_torchtpu.spec_decode import draft_cap

        rk = self.ranks[r]
        progs = rk.__dict__.setdefault("cap_progs", {})
        if p not in progs:
            progs[p] = draft_cap._cap_program_class(p)(vllm_config=None, mtp=rk.mtp, num_steps=K - 1, dvp=False,
                                                       max_reqs=CAP)
        return progs[p]

    def dhd_cap(self, r, si, p):
        from vllm_torchtpu.spec_decode import draft_one

        rk = self.ranks[r]
        bufs, flat, tbuf = self.install_cap(r, si, p)
        n0 = len(self.recs[r])
        args = (si["input_ids"], si["positions"], si["hidden"], si["draft_rows"][:p], si["pos1"], *si["md0"],
                *si["md1"], *bufs, torch.zeros(CAP, dtype=torch.int32), torch.tensor(si["n"]), flat, None, None, None,
                tbuf, None, None, None)
        with draft_one._meta_i32(rk.pred):
            out = tuple(self.cap_prog(r, p)(*args))
        calls = self.recs[r][n0:]
        assert len(calls) == K - 1 and all(c["rows"] == p for c in calls), [c["rows"] for c in calls]
        return out

    # ---- K1FUSE (HK)
    def verify_inputs(self, si, i, p):
        """VFY-ONE's verify operands for the step (every rank and both worlds identical): a TP1 vocab shard of VT, the
        chunk's 3 drafts per live request (60 % the verify row's argmax), the served padding (segment min(n, B - 1),
        group K), full-vocabulary uniforms (DRAWS=full), no fallback rows."""
        from vllm_torchtpu.runner import vps_wiring

        class Proc:
            def _get_logits(self, hidden, head, bias, skip_gather=False):
                return torch.nn.functional.linear(hidden, head.weight)

        if not hasattr(self, "vh"):
            gw = torch.Generator().manual_seed(4242)
            self.head_v = NS(weight=(torch.randn(VT, HIDV, generator=gw) * 0.5).bfloat16())
            self.vh = vps_wiring.VpsHead(Proc(), self.head_v, 0, VT, VT, 1)
        n, B = si["n"], CAP
        T = int(si["input_ids"].shape[0])
        g = torch.Generator().manual_seed(self.seed * 77 + 13 * i + p)
        hidden = torch.randn(T, HIDV, generator=g).bfloat16()
        logits = torch.nn.functional.linear(hidden, self.head_v.weight)
        qsl = si["md0"][1].numpy()
        t_idx, dti = torch.zeros(T, dtype=torch.int32), torch.zeros(T, dtype=torch.int32)
        seg, grp = torch.full((T,), min(n, B - 1), dtype=torch.int64), torch.full((T,), K, dtype=torch.int32)
        b_idx = torch.zeros(B, dtype=torch.int32)
        pos = 0
        for q in range(n):
            for j in range(K):
                t = int(qsl[q]) + j
                t_idx[pos], seg[pos], grp[pos] = t, q, j
                dti[pos] = int(logits[t].float().argmax()) if float(torch.rand((), generator=g)) < 0.6 else int(
                    torch.randint(0, VT, (), generator=g))
                pos += 1
            b_idx[q] = int(qsl[q]) + K
        num_draft = torch.tensor([K if q < n else 0 for q in range(B)], dtype=torch.int32)
        slots = (torch.randperm(4095, generator=g)[: 3 * CAP] + 1).to(torch.int32).view(3, CAP)
        slots[:, n:] = 0
        return dict(
            hidden=hidden, idx=torch.arange(T, dtype=torch.int32), b_idx=b_idx, t_idx=t_idx,
            temps=torch.full((B, 1), 0.6, dtype=torch.bfloat16), top_k=torch.full((B, 1), 20, dtype=torch.int32),
            top_p=torch.full((B, 1), 0.95, dtype=torch.float32),
            bonus_u=torch.rand((B, VT), generator=g), accept_u=torch.rand((T,), generator=g),
            recover_u=torch.rand((T, VT), generator=g), dti=dti, num_draft=num_draft, seg=seg, grp=grp,
            rates=torch.tensor([1.0, 0.32, 0.0, 0.0]), n_active=torch.tensor([n], dtype=torch.int32),
            groups=[slots[k].clone() for k in range(3)], ro_n=torch.tensor(n, dtype=torch.int64),
            acc=torch.zeros(3, dtype=torch.int64), n_dev=torch.tensor(n, dtype=torch.int64))

    def _vargs(self, v):
        return (v["hidden"], v["idx"], v["b_idx"], v["t_idx"], v["temps"], v["top_k"], v["top_p"], v["bonus_u"],
                v["accept_u"], v["recover_u"], v["dti"], v["num_draft"], v["seg"], v["grp"], v["rates"])

    def k1(self, r, si, v, p, cap):
        from vllm_torchtpu.spec_decode import draft_one, k1fuse

        rk = self.ranks[r]
        if cap:
            bufs, flat, tbuf = self.install_cap(r, si, p)
        else:
            bufs = self.install(r, si)
        md0, md1 = si["md0"], si["md1"]
        st = k1fuse.K1Statics(K, True, 0, False, torch.int32, p, CAP, CAP)
        progs = rk.__dict__.setdefault("k1_progs", {})
        if (p, cap) not in progs:
            cls = (k1fuse._cap_program_class if cap else k1fuse._program_class)(len(progs))
            progs[(p, cap)] = cls(vllm_config=None, prefix=f"k1_{p}_{int(cap)}", mtp=rk.mtp, num_steps=K - 1,
                                  dvp=False, vh=self.vh, statics=st)
        args = (*self._vargs(v), v["n_active"], None, list(v["groups"]), v["ro_n"], v["acc"], v["num_draft"], md0[1],
                v["n_dev"], si["input_ids"], si["positions"], md0[0], md0[2], si["hidden"], md1[1], md1[2], *bufs)
        assert len(args) == len(k1fuse.ARG_NAMES)
        if cap:
            args = (*args, flat, None, None, None, tbuf, None, None, None)
        n0 = len(self.recs[r])
        with draft_one._meta_i32(rk.pred):
            out = tuple(progs[(p, cap)](*args))
        calls = self.recs[r][n0:]
        assert len(calls) == K - 1 and all(c["rows"] == p for c in calls), [c["rows"] for c in calls]
        return out

    def chain(self, r, si, v, p, cap):
        """VFY-ONE value 2 (the verify program with the superset seed, tpu_runner._vfy_one_impl) then DH-D's program
        (DRAFT-CAP's R_d program when `cap`) on the seed's outputs, as today's served chain runs them."""
        from vllm_torchtpu.runner import tpu_runner as tr
        from vllm_torchtpu.spec_decode import draft_one

        rk = self.ranks[r]
        md0, md1 = si["md0"], si["md1"]
        res = tr._vfy_one_impl(self.vh, *self._vargs(v), K, True, v["n_active"], 0, False, None, list(v["groups"]),
                               v["ro_n"], torch.int32, v["acc"], draft_one._seed_cl_impl, v["num_draft"], md0[1],
                               v["n_dev"], si["input_ids"], md0[0], si["positions"], p, CAP, CAP)
        seed = res[4:]
        n0 = len(self.recs[r])
        if cap:
            bufs, flat, tbuf = self.install_cap(r, si, p)
            rows = seed[1] if seed[1].shape[0] == min(p, CAP) else seed[3]
            args = (seed[0], si["positions"], si["hidden"], rows, seed[draft_one.SEED_POS1], *md0, seed[draft_one.SEED_SEQ1],
                    md1[1], md1[2], *bufs, seed[draft_one.SEED_BONUS], v["n_dev"], flat, None, None, None, tbuf, None,
                    None, None)
            prog = self.cap_prog(r, p)
        else:
            bufs = self.install(r, si)
            args = (seed[0], si["positions"], si["hidden"], seed[3], seed[draft_one.SEED_POS1], *md0,
                    seed[draft_one.SEED_SEQ1], md1[1], md1[2], *bufs, seed[draft_one.SEED_BONUS], v["n_dev"])
            prog = self.prog(r, p)
        with draft_one._meta_i32(rk.pred):
            out = tuple(prog(*args))
        calls = self.recs[r][n0:]
        assert len(calls) == K - 1 and all(c["rows"] == p for c in calls), [c["rows"] for c in calls]
        return (*res[:4], torch.stack(out[:-1], dim=1), out[-1], seed[draft_one.SEED_REJ])


def _passes_equal(wa, wb, tag):
    for r, (ra, rb) in enumerate(zip(wa.recs, wb.recs)):
        assert len(ra) == len(rb), (tag, r, len(ra), len(rb))
        for k, (ca, cb) in enumerate(zip(ra, rb)):
            assert torch.equal(ca["att"], cb["att"]), (f"{tag} rank {r} pass {k}: attention differs")
            assert ca["pool_sha"] == cb["pool_sha"], (f"{tag} rank {r} pass {k}: pool differs")


def _gather_identity(monkeypatch):
    import vllm.distributed

    monkeypatch.setattr(vllm.distributed, "tensor_model_parallel_all_gather", lambda x, dim=-1: x, raising=False)


# ---------------------------------------------------------------- HC


@pytest.mark.parametrize("R,D,p", [(2, 2, 16), (2, 2, 48), (4, 4, 16)])
def test_hc_kind_inside_draft_cap_programs_at_r_d(rpa_stand_in, monkeypatch, caplog, R, D, p):
    cw._setup_dcp(monkeypatch, D)
    caplog.set_level(logging.INFO)
    n = cw._live_for(p)
    outs, worlds = {}, {}
    for mode in ("cap", "full"):
        world = V6World(monkeypatch, R, D, seed=9100 + 100 * D + p)

        def body(r, world=world, mode=mode):
            world.setup(r)
            run = world.dhd_cap if mode == "cap" else (lambda rr, si, pp: world.dhd(rr, si, pp, meta=True))
            return [run(r, world.step_inputs(i, p, n), p) for i in range(STEPS)]

        outs[mode], worlds[mode] = world.spmd.threads(body), world
        assert sorted(set(world.builds)) == [p], (mode, world.builds)
    wc, wf = worlds["cap"], worlds["full"]
    for r in range(R):
        for a, b in zip(outs["cap"][r], outs["full"][r]):
            for x, y in zip(a, b):
                assert torch.equal(x, y), f"HC p={p} rank {r}: DRAFT-CAP + kind vs capacity + kind drafts differ"
    _passes_equal(wc, wf, f"HC p={p}")
    stc = cw._check_recs(wc, f"HC cap p={p} R={R} D={D}", p)
    assert stc["passes"] == R * STEPS * (K - 1), stc
    assert all(s == (p, min(p, CAP), min(p, CAP) * wc.cols) for rs in wc.shapes for s in rs), wc.shapes[0][:3]
    assert all(s == (p, CAP, CAP * wf.cols) for rs in wf.shapes for s in rs), wf.shapes[0][:3]
    COUNTS[f"hc_{R}{D}_{p}"] = stc["passes"]
    print(f"\nV6C_H3 HC p={p} R={R} D={D} live={n} steps={STEPS}: DRAFT-CAP R_d program + kind, {stc['passes']} passes all "
          f"at B = {p} rows over R_d = {min(p, CAP)} slots (capacity program: {CAP} slots), kernel built at {p} only; "
          f"rule A max ulps {stc['worst_ulps']:.1f} bitwise min {stc['min_exact']:.6f}; drafts, source, every pass's "
          f"attention and pool digest bitwise the capacity DH-D program with the kind", flush=True)


# ---------------------------------------------------------------- HK


@pytest.mark.parametrize("cap", [False, True])
@pytest.mark.parametrize("p", [16, 48])
def test_hk_kind_inside_k1fuse_program_is_bitwise_the_chain(rpa_stand_in, monkeypatch, caplog, p, cap):
    R, D = 2, 2
    cw._setup_dcp(monkeypatch, D)
    _gather_identity(monkeypatch)
    caplog.set_level(logging.INFO)
    n = cw._live_for(p)
    outs, worlds = {}, {}
    for mode in ("k1", "chain"):
        world = V6World(monkeypatch, R, D, seed=9300 + p + 7 * cap, vocab=VT)

        def body(r, world=world, mode=mode):
            world.setup(r)
            res = []
            for i in range(STEPS):
                si = world.step_inputs(i, p, n)
                si["positions"] = world.positions_for(si)
                v = world.verify_inputs(si, i, p)
                res.append(world.k1(r, si, v, p, cap) if mode == "k1" else world.chain(r, si, v, p, cap))
            return res

        outs[mode], worlds[mode] = world.spmd.threads(body), world
        assert sorted(set(world.builds)) == [p], (mode, world.builds)
    names = ("tokens", "stats", "ro_idx", "ro_vals", "drafts", "source", "rejected")
    for r in range(R):
        for i, (a, b) in enumerate(zip(outs["k1"][r], outs["chain"][r])):
            assert len(a) == len(b) == 7
            for nm, x, y in zip(names, a, b):
                assert x.shape == y.shape and torch.equal(x, y), f"HK p={p} cap={cap} rank {r} step {i}: {nm} differ"
    wk, wc = worlds["k1"], worlds["chain"]
    _passes_equal(wk, wc, f"HK p={p} cap={cap}")
    st = cw._check_recs(wk, f"HK p={p} cap={cap}", p)
    assert st["passes"] == R * STEPS * (K - 1), st
    slots = min(p, CAP) if cap else CAP
    assert all(s == (p, slots, slots * wk.cols) for rs in wk.shapes + wc.shapes for s in rs), wk.shapes[0][:3]
    COUNTS[f"hk_{p}_{int(cap)}"] = st["passes"]
    print(f"\nV6C_H3 HK p={p} cap={int(cap)} R={R} D={D} live={n} steps={STEPS}: K1FUSE's "
          f"{'R_d (DRAFT-CAP) ' if cap else ''}program with the kind == VFY-ONE value 2 + "
          f"{'DRAFT-CAP' if cap else 'DH-D'} program with the kind: tokens / stats / read offsets / drafts / source / "
          f"rejected and {st['passes']} passes' attention + pool digests bitwise; B = {p}, {slots} slots per pass; "
          f"rule A max ulps {st['worst_ulps']:.1f}", flush=True)


# ---------------------------------------------------------------- HL


_ORIG_MTP = cw._Mtp


class _LmhWorld(V6World):
    """The draft's LM head as a real vLLM ParallelLMHead (TP-disabled, integer weights) behind a LogitsProcessor, shared
    with a target model as Qwen4ExpMTP shares it; tokens through get_top_tokens (DH-D's dvp path). With `lmh`,
    LMH-STREAM's runner_utils.arm_lm_head_hidden_major(draft=True) gives the draft its own hidden-major copy BEFORE the
    attach (load order). Heads and processors are built in the main thread (no per-thread patching)."""

    def __init__(self, monkeypatch, R, D, seed, lmh):
        import vllm.model_executor.layers.vocab_parallel_embedding as vpe
        from vllm.model_executor.layers import logits_processor as lp_module
        from vllm.model_executor.layers.logits_processor import LogitsProcessor

        super().__init__(monkeypatch, R, D, seed)
        self.lmh, self.census, self.head_sha = lmh, [None] * R, [None] * R
        self.heads = []
        for r in range(R):
            head = vpe.ParallelLMHead(cw.V, self.g.H, params_dtype=torch.bfloat16, prefix="lm_head", disable_tp=True)
            gen = torch.Generator().manual_seed(seed + 5)
            head.weight.data.copy_(torch.randint(-2, 3, tuple(head.weight.shape), generator=gen).to(torch.bfloat16))
            # Under VLLM_TARGET_DEVICE=cpu (ATTNWIRE's environment) vLLM's unquantized head method calls layer.cpu_linear,
            # which its process_weights_after_loading would set; the plain case's value (vllm linear.py / layers/utils.py),
            # i.e. the stock F.linear the TPU path computes (w1 h3c A j-0d8ad8cb: AttributeError without it)
            head.cpu_linear = torch.nn.functional.linear
            with patch.object(lp_module, "get_current_vllm_config", return_value=NS(model_config=None)):
                procs = (LogitsProcessor(cw.V), LogitsProcessor(cw.V))
            target = torch.nn.Module()
            target.model = NS(embed_tokens=NS(weight=None))
            target.lm_head, target.logits_processor = head, procs[0]
            self.heads.append((head, procs[1], target))
        world = self

        class LmhMtp(_ORIG_MTP):
            def __init__(self, pred, wv):
                super().__init__(pred, wv)
                r = world.spmd.tl.rank
                head, proc, target = world.heads[r]
                self.lm_head, self.logits_processor = head, proc
                pred.embed_tokens = NS(weight=None)
                if world.lmh:
                    from vllm_torchtpu.runner import utils as ru

                    world.census[r] = ru.arm_lm_head_hidden_major(target, self, draft=True, verify=False)

            def compute_logits(self, h):
                return self.logits_processor(self.lm_head, h)

        monkeypatch.setattr(cw, "_Mtp", LmhMtp)

    def setup(self, r):
        out = super().setup(r)
        rk = self.ranks[r]
        mk = rk.pred.mk_attn.inner.real  # _ShapeRec -> _Shadow -> DraftAttn
        ops = b"".join(getattr(mk, f"w_{i}").contiguous().view(torch.uint8).numpy().tobytes() for i in range(mk.n_w))
        rk.ops_sha = hashlib.sha256(ops).hexdigest()
        rk.fp = repr((mk.kind.buckets, int(mk.pack_rows), sorted(mk.kind.opts.items())))
        return out

    def prog(self, r, p):
        from vllm_torchtpu.spec_decode import draft_one

        rk = self.ranks[r]
        if p not in rk.progs:
            rk.progs[p] = draft_one._program_class(p)(vllm_config=None, mtp=rk.mtp, num_steps=K - 1, dvp=True,
                                                      max_reqs=CAP)
        return rk.progs[p]


def _sha_head(m):
    return hashlib.sha256(m.lm_head.weight.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def test_hl_kind_beside_lmh_hidden_major_draft_head(rpa_stand_in, monkeypatch):
    from vllm_torchtpu.runner import utils as ru

    R, D, p = 2, 1, 16
    cw._setup_dcp(monkeypatch, D)
    n = cw._live_for(p)
    monkeypatch.setattr(ru, "_to_row_major", lambda t: t.clone())
    monkeypatch.setattr(ru, "_install_entry_layout_pin", lambda: None)
    seen = []
    orig = ru.HiddenMajorLMHeadMethod.apply

    def spy(self, layer, x, bias=None):
        seen.append(tuple(layer.weight.shape))
        return orig(self, layer, x, bias)

    monkeypatch.setattr(ru.HiddenMajorLMHeadMethod, "apply", spy)
    outs, worlds, calls = {}, {}, {}
    for lmh in (True, False):
        world = _LmhWorld(monkeypatch, R, D, seed=9500, lmh=lmh)
        n_seen = len(seen)

        def body(r, world=world):
            world.setup(r)
            sha_after_attach = _sha_head(world.ranks[r].mtp)
            res = [world.dhd(r, world.step_inputs(i, p, n), p, meta=True) for i in range(STEPS)]
            return res, sha_after_attach

        outs[lmh], worlds[lmh] = world.spmd.threads(body), world
        calls[lmh] = len(seen) - n_seen
    wl, ws = worlds[True], worlds[False]
    assert all(c is not None and "own hidden-major copy" in c["draft"] for c in wl.census), wl.census
    assert calls[True] >= R * STEPS and calls[False] == 0, calls
    for r in range(R):
        assert wl.ranks[r].ops_sha == ws.ranks[r].ops_sha and wl.ranks[r].fp == ws.ranks[r].fp, r
        assert outs[True][r][1] == _sha_head(wl.ranks[r].mtp), r  # the attach left the armed head's bytes as they were
        for a, b in zip(outs[True][r][0], outs[False][r][0]):
            for x, y in zip(a, b):
                assert torch.equal(x, y), f"HL rank {r}: drafts / source differ with LMH armed"
    _passes_equal(wl, ws, "HL")
    st = cw._check_recs(wl, "HL", p)
    COUNTS["hl"] = st["passes"]
    print(f"\nV6C_H3 HL p={p} R={R} D={D} steps={STEPS}: LMH-STREAM's own hidden-major draft head armed before the attach "
          f"({wl.census[0]['draft']}); kind operands / fingerprint byte-identical to an attach without LMH; head bytes "
          f"unchanged across the attach; drafts (HiddenMajorLMHeadMethod calls {calls[True]}) and {st['passes']} passes' "
          f"attention + pools bitwise the kind with the stock-layout head", flush=True)


def test_zz_counts():
    print("\nV6C_H3_COUNTS " + " ".join(f"{k}={v}" for k, v in sorted(COUNTS.items())))
