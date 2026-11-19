# SPDX-License-Identifier: Apache-2.0
"""q38 L4 ATTNWIRE T2: the draft attention kind (q38 L1-H3, VLLM_TPU_MK_DRAFT_ATTN) inside DRAFT1-B's draft programs, CPU.

What runs for real: `attach_draft_attn` (loop buckets, refusal rules, packing ONCE at max(driver.buckets), the all-rank
agreement), `DraftAttn.forward`, the kind's torch custom op, `DraftAttnKind.callable_for` (one kernel build per rows count,
its "built at %d rows" line), `build_draft_attn_call` (Pallas interpret), `Qwen4ExpMultiTokenPredictor.forward` (the
mk_tail path with L1's mk_attn branch and DRAFT1-B's meta_i32 switch), DH-D's per-loop-bucket program class
(`draft_one._program_class(p)`, i.e. `dh_one_body`: draft step 0, the carry and the K-1 = 2 fused-loop passes) under
`draft_one._meta_i32` exactly as `DraftOne.forward0` wraps it; stage A's `carry_loop_body`, DRAFT-ASM's
`_MtpFusedLoopAsm.forward` and the plain `_MtpFusedLoop.forward`.

Stand-ins (attnwire_harness): R TP ranks are threads; torch_tpu's JaxCallable is `aw.bridge` (assumed semantics, I); the
megakernel builds in interpret mode with L1's CPU tiles (production tiles' hier-2 all-reduce asserts R == 8; T1 shows the
packed bytes are the same under both); the predictor's layers around attention are small deterministic torch stubs: the
embedding, the HCs, draft step 0's attention (stock in both paths, not under test; it writes no KV), the MTP tail (o_proj +
an all-reduce over the rank threads) and the LM head. Today's draft attention is L1's oracle (`l1c._stock`: DRAFT-1's step +
DCP combine, the batched-RPA stand-in), run per pass on a copy of each rank's pool.

Rules (stated before data):
  A. Every fused-loop pass of every DH-D step (8 steps, loop bucket p = 8 / 16 / 32 / 48 / 64, DCP 1 / 2 / 4): the kind's
     attention vs today's on the same block and pool within 2 BF16 ulps (+ 1e-6 absolute; L1-H3's rule) on live rows, >= 99.9 % bitwise,
     finite; every rank's pool after the pass byte-identical to today's; the kernel built at B = p ("built at p rows" logged),
     operands packed at max(driver.buckets) = 64; mk_attn called exactly K-1 = 2 times per DH-D call, each with p rows, never
     on draft step 0; every integer operand of the op int32.
  B. meta_i32 on (DH-D's mode) vs off: tokens, source, every pass's attention and every pool byte-identical.
  C. One kernel build per rows count across window-switched paths: after DH-D at p, stage A, DRAFT-ASM's loop and the
     plain loop at the same p add no build, and their passes also take the kind (rule A's numerics).
  D. Informational (not a gate): the kind's DH-D drafts vs a DH-D run on today's attention, same inputs, over 8 steps.
"""

from __future__ import annotations

import hashlib
import logging
import os
import types

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

import attnwire_harness as aw  # noqa: E402
import test_l1h3_draft_attn_call as l1c  # noqa: E402

K = 3
CAP = 64
V = 97
STEPS = 8
PADDINGS = [8, 16, 32, 48, 64, 80, 96, 128]
DRIVER_BUCKETS = (8, 16, 32, 48, 64)
BUILD_LINE = "Qwen3.8 MTP draft attention kernel built at %d rows (q38 L1-H3)"
_REAL = {}


def _real_build():
    from vllm_torchtpu.megakernel_runtime import qwen38_draft_attn as da

    return _REAL.setdefault("build", da.build_draft_attn_call)


def _pad(n):
    return next(x for x in PADDINGS if x >= n)


class _Layer:
    def __init__(self, attn):
        self.self_attn = attn
        self.attn_hyper_connection = types.SimpleNamespace(mix=lambda h: (h.to(torch.bfloat16), None))


class _Tail:
    """The MTP tail stand-in: o_proj of this rank's heads, summed over the rank threads (the TP all-reduce), plus a residual."""

    def __init__(self, spmd, wo):
        self.spmd, self.wo = spmd, wo

    def __call__(self, hidden, attention):
        part = attention.float() @ self.wo
        return (hidden.float() * 0.5 + self.spmd.allreduce(part)).to(torch.bfloat16)


class _Pred:
    """Predictor-shaped stub; `forward` IS `Qwen4ExpMultiTokenPredictor.forward`."""

    def __init__(self, emb, attn, tail):
        from vllm_torchtpu.models.vllm.qwen4_exp.mtp import Qwen4ExpMultiTokenPredictor

        self.forward = types.MethodType(Qwen4ExpMultiTokenPredictor.forward, self)
        self.meta_i32 = False
        self.mk_attn = None
        self.mk_tail = tail
        self.layers = [_Layer(attn)]
        self.hyper_connection_mixer = types.SimpleNamespace(mix=lambda h: (h, None))
        self.emb = emb

    def fuse_inputs(self, input_ids, hidden_states, inputs_embeds=None):
        return (hidden_states.float() + self.emb[input_ids.long()]).to(torch.bfloat16)


class _Mtp:
    """Qwen4ExpMTP.forward's call shape (`self.model(..., spec_step_idx, draft_rows)`) and an LM head."""

    def __init__(self, pred, wv):
        self.model, self.wv = pred, wv

    def forward(self, input_ids, positions, hidden_states, intermediate_tensors=None, inputs_embeds=None,
                spec_step_idx=0, draft_rows=None):
        return self.model.forward(input_ids, positions, hidden_states, inputs_embeds, spec_step_idx, draft_rows)

    def compute_logits(self, h):
        return h.float() @ self.wv


class _Shadow:
    """mk_attn as the predictor sees it: today's attention on a COPY of the pool first, then the real kind (which writes the
    live pool); every call recorded on this rank."""

    def __init__(self, real, stock, rec):
        self.real, self.stock, self.rec = real, stock, rec

    def __call__(self, block, pos, md, attn):
        s_att, s_pool = self.stock(block, pos, md, attn, attn.cache_owners[0].clone())
        out = self.real(block, pos, md, attn)
        live_pool = attn.cache_owners[0].view(torch.uint8)
        # The pools are compared here and only the count and a digest are kept (a whole-pool copy per pass per rank ran a
        # 4-rank world out of memory: attnwire-proof-pfb1 j-f7d61b2b OOMKilled).
        self.rec.append(dict(rows=int(block.shape[0]), pos_dtype=pos.dtype, ctx_dtype=md.context_lens.dtype,
                             att=out.float().clone(), s_att=s_att.float(),
                             pool_diff=int((live_pool != s_pool.view(torch.uint8)).sum()),
                             pool_sha=hashlib.sha256(live_pool.contiguous().numpy().tobytes()).hexdigest(),
                             live=int((md.query_start_loc[1:] > md.query_start_loc[:-1]).sum())))
        del s_pool
        return out


class World:
    """R rank threads, each with its own attention stub, pool, predictor, attached kind and DH-D program per loop bucket."""

    def __init__(self, monkeypatch, R, D, seed, mk=True):
        from vllm_torchtpu.megakernel_runtime import qwen38_draft_attn as da

        self.R, self.D, self.seed, self.mk = R, D, seed, mk
        self.spmd = aw.Spmd(R)
        self.log = aw.BridgeLog()
        self.builds = []
        aw.unique_custom_op(monkeypatch)
        monkeypatch.setattr(da, "_LOGGED", set())
        monkeypatch.setattr(da, "sharded_jax_callable", aw.bridge(self.spmd, self.log))
        monkeypatch.setattr(da, "segment_tiles", lambda B, *a, **k: l1c.TILES)
        real_build = _real_build()

        def build(mesh, g, B, **kw):
            self.builds.append(B)
            return real_build(mesh, g, B, **{**kw, "interpret": True})

        monkeypatch.setattr(da, "build_draft_attn_call", build)
        monkeypatch.setattr(da, "_gather", lambda vec: self.spmd.run(list(vec), lambda slots: [slots] * len(slots)))
        from vllm import forward_context as vfc

        from vllm_torchtpu.models.vllm.qwen4_exp import mtp

        get = lambda: self.spmd.tl.ctx  # noqa: E731
        monkeypatch.setattr(mtp, "get_forward_context", get)
        monkeypatch.setattr(vfc, "get_forward_context", get)
        rng = np.random.default_rng(seed)
        self.g = aw.geometry(R)
        self.w = aw.weights(rng, R, self.g)
        H, nh = self.g.H, aw.NH_RANK
        gen = torch.Generator().manual_seed(seed)
        self.emb = torch.randn(V, H, generator=gen) * 0.5
        self.wv = torch.randn(H, V, generator=gen)
        self.wo = [torch.randn(nh * aw.HD, H, generator=gen) * 0.02 for _ in range(R)]
        self.wq0 = [torch.randn(H, nh * aw.HD, generator=gen) * 0.05 for _ in range(R)]
        ctx_max = 1200
        self.cols = int(-(-(ctx_max + 8 + STEPS * (K + 1) + 8) // (aw.PB * D))) + 1
        pages = CAP * self.cols * D + 8
        self.table = (rng.permutation(pages - 1)[: CAP * self.cols] + 1).reshape(CAP, self.cols).astype(np.int32)
        pools = [aw.random_fp8(rng, (pages, aw.PB, 1, 4, aw.HD)) for _ in range(R)]
        if D == 1:
            pools = [pools[0]] * R
        self.pools0 = pools
        self.ctx0 = rng.integers(40, ctx_max, CAP).astype(np.int32)
        self.recs = [[] for _ in range(R)]
        self.ranks = [None] * R

    # ---- per-rank setup (inside the rank thread)
    def setup(self, r):
        from vllm_torchtpu.megakernel_runtime import qwen38_draft_attn as da

        c = dict(self.w, pools=self.pools0, shared=np.zeros((CAP, aw.TOPK + aw.RATIO - 1), np.int32), main_slot=1)
        attn = aw.torch_attn(c, r, self.D)
        attn.attend_with_indices = lambda block, pos, md, rows=None: (
            (block[rows.clamp(min=0)].float() @ self.wq0[r]).to(torch.bfloat16), None)
        attn.attend = lambda block, pos, md: self._stock(block, pos, md, attn, attn.cache_owners[0], write=True)[0]
        pred = _Pred(self.emb, attn, _Tail(self.spmd, self.wo[r]))
        mtp_obj = _Mtp(pred, self.wv)
        drafter = types.SimpleNamespace(draft_model=types.SimpleNamespace(model=pred), _loop_bucket=_pad)
        runner = types.SimpleNamespace(drafter=drafter, max_num_reqs=CAP, num_tokens_paddings=list(PADDINGS),
                                       max_model_len=4096)
        driver = types.SimpleNamespace(g=self.g, mesh=Mesh(np.array(jax.devices()[: self.R]), ("tp",)),
                                       buckets=DRIVER_BUCKETS, whole_cap=4096)
        attached = da.attach_draft_attn(runner, driver) if self.mk else False
        assert attached == self.mk, attached
        if self.mk:
            assert isinstance(pred.mk_attn, da.DraftAttn)
            assert pred.mk_attn.pack_rows == max(DRIVER_BUCKETS), ("packed at", pred.mk_attn.pack_rows)
            assert pred.mk_attn.kind.buckets == DRIVER_BUCKETS, pred.mk_attn.kind.buckets
            pred.mk_attn = _Shadow(pred.mk_attn, self._stock, self.recs[r])
        self.ranks[r] = types.SimpleNamespace(attn=attn, pred=pred, mtp=mtp_obj, progs={})
        return attached

    def _stock(self, block, pos, md, attn, pool, write=False):
        """Today's draft pass (L1's oracle) for this rank: every rank contributes its block / metadata / pool; rank 0 runs
        `l1c._stock` over all of them; (attention [p, nh*HD] BF16, the pool after the pass)."""
        R, D, p = self.R, self.D, int(block.shape[0])
        main = self.spmd.tl.ctx.attn_metadata[attn.main_cache_name].block_tables
        payload = (block, pos.to(torch.int32), md.query_start_loc.to(torch.int32), md.context_lens.to(torch.int32),
                   main.to(torch.int32), attn._selection_rows[:p].to(torch.int32), pool)

        def run(slots):
            b0 = slots[0]
            for s in slots[1:]:
                for i in range(6):
                    assert torch.equal(s[i].view(torch.uint8) if s[i].dtype == torch.bfloat16 else s[i],
                                       b0[i].view(torch.uint8) if b0[i].dtype == torch.bfloat16 else b0[i]), i
            c = dict(g=self.g, n=p, ctx=b0[3][:p].numpy(), starts=b0[2][: p + 1].numpy(), positions=b0[1].numpy(),
                     table=b0[4][:p].numpy(), pools=[aw.tj(s[6]) for s in slots], shared=b0[5].numpy(), block=aw.tj(b0[0]),
                     main_slot=1, qkvg=self.w["qkvg"], nq=self.w["nq"], nk=self.w["nk"], trig=self.w["trig"])
            att, pools = l1c._stock(c, R, D)
            out = []
            for r in range(len(slots)):
                a = aw.jt(np.asarray(att[r], np.float32).astype(jax.numpy.bfloat16))
                pl = torch.from_numpy(np.asarray(pools[r]).copy()).view(torch.float8_e4m3fn).reshape(slots[r][6].shape)
                out.append((a, pl))
            return out

        att, new_pool = self.spmd.run(payload, run)
        if write:
            pool.copy_(new_pool)
        return att, new_pool

    def prog(self, r, p):
        from vllm_torchtpu.spec_decode import draft_one

        rk = self.ranks[r]
        if p not in rk.progs:
            rk.progs[p] = draft_one._program_class(p)(vllm_config=None, mtp=rk.mtp, num_steps=K - 1, dvp=False, max_reqs=CAP)
        return rk.progs[p]

    # ---- one DH-D step (inside the rank thread)
    def step_inputs(self, i, p, n):
        """Step i's inputs (identical on every rank): n live requests at loop bucket p."""
        rng = np.random.default_rng(self.seed * 1000 + 17 * i + p)
        ctx1 = np.zeros(CAP, np.int32)
        ctx1[:n] = self.ctx0[:n] + (K + 1) * i + rng.integers(0, 3, n)
        seq1 = np.where(np.arange(CAP) < n, ctx1 + 1, 0).astype(np.int32)
        seq0 = np.where(np.arange(CAP) < n, ctx1 - 1 + 4, 0).astype(np.int32)  # the verify step's 4 tokens per request
        T = -(-(4 * n) // 16) * 16  # step 0's token bucket (16-multiples cover 4 x 64 requests)
        qsl0 = np.minimum(np.arange(CAP + 1) * 4, 4 * n).astype(np.int32)
        rows = np.full(CAP, -1, np.int64)
        rows[:n] = np.arange(n) * 4 + 3
        H = self.g.H
        return dict(
            input_ids=torch.from_numpy(rng.integers(1, V, T).astype(np.int32)),
            positions=torch.from_numpy(rng.integers(0, 4000, T).astype(np.int32)),
            hidden=torch.from_numpy(rng.normal(0, 0.5, (T, H)).astype(np.float32)).to(torch.bfloat16),
            draft_rows=torch.from_numpy(rows),
            pos1=torch.from_numpy(np.where(np.arange(p) < n, ctx1[:p], 0).astype(np.int32)),
            md0=(torch.from_numpy(seq0), torch.from_numpy(qsl0), torch.tensor([0, n, n], dtype=torch.int32)),
            md1=(torch.from_numpy(seq1), torch.from_numpy(np.minimum(np.arange(CAP + 1), n).astype(np.int32)),
                 torch.tensor([n, n, n], dtype=torch.int32)),
            sel=aw.selection(rng, ctx1, n, CAP), n=n)

    def install(self, r, si):
        rk = self.ranks[r]
        bufs = tuple(torch.zeros_like(t) for t in si["md0"])
        md = types.SimpleNamespace(seq_lens=bufs[0], query_start_loc=bufs[1], request_distribution=bufs[2],
                                   block_tables=torch.from_numpy(self.table))
        self.spmd.tl.ctx = types.SimpleNamespace(attn_metadata={rk.attn.prefix: md})
        rk.attn._selection_rows = torch.from_numpy(si["sel"])
        return bufs

    def dhd(self, r, si, p, meta):
        from vllm_torchtpu.spec_decode import draft_one

        rk = self.ranks[r]
        bufs = self.install(r, si)
        n0 = len(self.recs[r])
        args = (si["input_ids"], si["positions"], si["hidden"], si["draft_rows"], si["pos1"], *si["md0"], *si["md1"],
                *bufs, torch.zeros(CAP, dtype=torch.int32), torch.tensor(si["n"]))
        with draft_one._meta_i32(rk.pred) if meta else _null():
            assert rk.pred.meta_i32 is bool(meta)
            out = tuple(self.prog(r, p)(*args))
        assert rk.pred.meta_i32 is False
        calls = self.recs[r][n0:]
        if self.mk:
            assert len(calls) == K - 1 and all(c["rows"] == p for c in calls), \
                ("mk_attn calls per DH-D call", [c["rows"] for c in calls], p)
        return out


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _check_recs(world, tag, p):
    """Rule A on every recorded pass of every rank; returns aggregate stats."""
    worst_ulps, exact, n_cmp, n_pass = 0.0, [], 0, 0
    for r, recs in enumerate(world.recs):
        for k, c in enumerate(recs):
            live = np.arange(c["rows"]) < c["live"]
            got, want = c["att"].numpy()[None], c["s_att"].numpy()[None]
            st = aw.compare(got, want, live)
            assert st["finite"] and st["idle_zero"], (tag, r, k, st)
            assert st["bad"] == 0, (f"{tag} rank {r} pass {k}: attention beyond 2 ulps", st)
            assert st["exact"] >= aw.EXACT_BOUND, (f"{tag} rank {r} pass {k}: bitwise share below 99.9 %", st)
            diff = c["pool_diff"]
            assert diff == 0, (f"{tag} rank {r} pass {k}: pool bytes differ", diff)
            worst_ulps = max(worst_ulps, st["max_ulps"])
            exact.append(st["exact"])
            n_cmp += int(live.sum())
            n_pass += 1
    return dict(passes=n_pass, worst_ulps=worst_ulps, min_exact=min(exact) if exact else -1.0,
                mean_exact=float(np.mean(exact)) if exact else -1.0, live_rows=n_cmp)


def _setup_dcp(monkeypatch, D):
    l1c._interpret_keep_params(monkeypatch)
    l1c._route_any_dcp(monkeypatch)
    if D > 1:
        from vllm_torchtpu.models.vllm.qwen4_exp import pool_layout

        monkeypatch.setenv(pool_layout.MAIN_KV_DCP_ENV, str(D))


def _live_for(p):
    return max(1, p - 3) if p > 8 else 6


GROUPS = [(2, 1), (2, 2), (4, 4)]


@pytest.mark.parametrize("p", [8, 16, 32, 48, 64])
@pytest.mark.parametrize("R,D", GROUPS)
def test_dhd_with_draft_attention_kind_over_8_steps(rpa_stand_in, monkeypatch, caplog, R, D, p):
    _setup_dcp(monkeypatch, D)
    caplog.set_level(logging.INFO)
    world = World(monkeypatch, R, D, seed=7000 + 100 * D + p)
    n = _live_for(p)

    def body(r):
        world.setup(r)
        return [world.dhd(r, world.step_inputs(i, p, n), p, meta=True) for i in range(STEPS)]

    outs = world.spmd.threads(body)
    for r in range(1, R):
        for a, b in zip(outs[0], outs[r]):
            for x, y in zip(a, b):
                assert torch.equal(x, y), "ranks drafted differently"
    st = _check_recs(world, f"p={p} R={R} D={D}", p)
    assert st["passes"] == R * STEPS * (K - 1), st
    assert all(c["pos_dtype"] == torch.int64 and c["ctx_dtype"] == torch.int32 for recs in world.recs for c in recs), \
        "mk_attn inputs under meta_i32: positions int64 (mk_tail path), context int32"
    builds = [m for m in (x.getMessage() for x in caplog.records) if "MTP draft attention kernel built at" in m]
    assert builds and all(m == BUILD_LINE % p for m in builds), builds
    assert sorted(set(world.builds)) == [p], world.builds
    assert {(rows, out_rows) for _n, rows, out_rows in world.log.calls} == {(p, p)}, set(world.log.calls)
    agree = [m for m in (x.getMessage() for x in caplog.records) if "draft attention agreement" in m]
    assert agree and f"attached {R}, refused 0 of {R}" in agree[0], agree
    print(f"ATTNWIRE DHD p={p} R={R} D={D} live={n} steps={STEPS} passes={st['passes']} (K-1 = {K - 1} per DH-D call per rank): "
          f"max ulps {st['worst_ulps']:.1f}, bitwise min {st['min_exact']:.6f} mean {st['mean_exact']:.6f}, pools byte-identical "
          f"to today's after every pass; engagement '{builds[0]}'; packed at {max(DRIVER_BUCKETS)}; agreement '{agree[0]}'",
          flush=True)


@pytest.mark.parametrize("p", [16, 48])
@pytest.mark.parametrize("R,D", [(2, 1), (4, 4)])
def test_meta_i32_on_off_bitwise(rpa_stand_in, monkeypatch, R, D, p):
    _setup_dcp(monkeypatch, D)
    res = {}
    for meta in (True, False):
        world = World(monkeypatch, R, D, seed=8100 + 10 * D + p)
        n = _live_for(p)

        def body(r, meta=meta, world=world):
            world.setup(r)
            return [world.dhd(r, world.step_inputs(i, p, n), p, meta=meta) for i in range(STEPS)]

        res[meta] = (world.spmd.threads(body), world)
    (on, w_on), (off, w_off) = res[True], res[False]
    for a, b in zip(on[0], off[0]):
        for x, y in zip(a, b):
            assert torch.equal(x, y), "meta_i32 on / off: drafts differ"
    for ra, rb in zip(w_on.recs, w_off.recs):
        for ca, cb in zip(ra, rb):
            assert torch.equal(ca["att"], cb["att"]) and ca["pool_sha"] == cb["pool_sha"], "meta_i32 on / off: pass differs"
            assert ca["ctx_dtype"] == torch.int32 and cb["ctx_dtype"] == torch.int64
    st = _check_recs(w_off, f"meta off p={p}", p)
    print(f"ATTNWIRE META p={p} R={R} D={D}: meta_i32 on (context int32) vs off (context int64): drafts, source, "
          f"{len(w_on.recs[0])} passes' attention and pools byte-identical on every rank; off-mode vs today: max ulps "
          f"{st['worst_ulps']:.1f}", flush=True)


def test_one_build_per_rows_across_window_switched_paths(rpa_stand_in, monkeypatch, caplog):
    from vllm_torchtpu.spec_decode import draft_asm, draft_one
    from vllm_torchtpu.spec_decode.mtp_fused_loop import _MtpFusedLoop

    R, D, p = 2, 1, 16
    _setup_dcp(monkeypatch, D)
    caplog.set_level(logging.INFO)
    world = World(monkeypatch, R, D, seed=8800)
    n = 11

    def body(r):
        world.setup(r)
        world.dhd(r, world.step_inputs(0, p, n), p, meta=True)  # warm-up: builds B = p once per rank
        after_warm = len(world.builds)
        counts = {}
        for i, path in enumerate(("stageA", "draftasm", "plain", "dhd")):
            si = world.step_inputs(1 + i, p, n)
            n0 = len(world.recs[r])
            if path == "dhd":
                world.dhd(r, si, p, meta=True)
            else:
                bufs = world.install(r, si)
                for b, v in zip(bufs, si["md1"]):
                    b.copy_(v)
                prog, rk = world.prog(r, p), world.ranks[r]
                h0 = torch.from_numpy(np.random.default_rng(i).normal(0, 0.5, (CAP, world.g.H)).astype(np.float32)).to(
                    torch.bfloat16)
                tok0 = torch.from_numpy(np.random.default_rng(i + 9).integers(1, V, p).astype(np.int32))
                if path == "stageA":
                    draft_one.carry_loop_body(prog, si["pos1"], h0, h0, bufs[0], torch.zeros(CAP, dtype=torch.int32),
                                              torch.tensor(n), CAP)
                elif path == "draftasm":
                    draft_asm._MtpFusedLoopAsm.forward(prog, tok0, si["pos1"], h0[:p], bufs[0],
                                                       torch.zeros(CAP, dtype=torch.int32), torch.tensor(n))
                else:
                    _MtpFusedLoop.forward(prog, tok0, si["pos1"], h0[:p], bufs[0])
                assert rk.pred.meta_i32 is False
            counts[path] = [c["rows"] for c in world.recs[r][n0:]]
        return after_warm, counts

    outs = world.spmd.threads(body)
    st = _check_recs(world, "paths", p)
    for after_warm, counts in outs:
        for path, rows in counts.items():
            assert rows == [p] * (K - 1), (path, rows)
    assert len(world.builds) == R and set(world.builds) == {p}, ("kernel builds after warm-up", world.builds)
    builds = [m for m in (x.getMessage() for x in caplog.records) if "MTP draft attention kernel built at" in m]
    print(f"ATTNWIRE PATHS p={p}: after the DH-D warm-up ({R} builds at {p} rows, one per rank), stage A / DRAFT-ASM loop / "
          f"plain loop / DH-D each ran {K - 1} kind passes of {p} rows with 0 new builds ({len(world.builds)} total); build "
          f"lines {sorted(set(builds))}; numerics over {st['passes']} passes: max ulps {st['worst_ulps']:.1f}, bitwise min "
          f"{st['min_exact']:.6f}, pools byte-identical", flush=True)


def test_drafts_vs_todays_attention_informational(rpa_stand_in, monkeypatch):
    R, D, p = 2, 1, 16
    _setup_dcp(monkeypatch, D)
    n = 13
    runs = {}
    for mk in (True, False):
        world = World(monkeypatch, R, D, seed=8900, mk=mk)

        def body(r, world=world):
            world.setup(r)
            return [world.dhd(r, world.step_inputs(i, p, n), p, meta=True) for i in range(STEPS)]

        runs[mk] = world.spmd.threads(body)[0]
    agree = total = 0
    for a, b in zip(runs[True], runs[False]):
        for x, y in zip(a[:K], b[:K]):
            agree += int((x[:n] == y[:n]).sum())
            total += n
    print(f"ATTNWIRE DRAFTS (informational) p={p} R={R} D={D} steps={STEPS}: the kind's DH-D drafts equal today's on "
          f"{agree} / {total} draft tokens ({agree / total:.4f}); not a gate (the attention is not bitwise by construction)",
          flush=True)
    assert total == STEPS * K * n
