# SPDX-License-Identifier: Apache-2.0
"""q38 L4 ATTNWIRE: CPU harness for the draft attention kind (q38 L1-H3, VLLM_TPU_MK_DRAFT_ATTN) composed with DRAFT1-B's draft
programs. Not collected (no test_ prefix). Builds on L1-H3's call test (test_l1h3_draft_attn_call: `_stock`, `_ulp`,
`_route_any_dcp`, `_interpret_keep_params`, its constants), the same oracle: today's draft pass per rank (DRAFT-1's
`pooled_qsa_shared_noidx_step` / `pooled_qsa_shared_dcp_noidx_step` + `qsa_rpa.dcp_combine`, RPA via this directory's stand-in).

* `case_b`: one fused-loop pass at `seats` request slots / rows (L1's `_case` with the seat count, live count and contexts as
  parameters) plus each rank's torch attention stub (`torch_attn`), whose q / k / v projections and norms hold the case's weights.
* `pack_torch`: each rank's operands through the real `qwen38_draft_attn.pack_draft_attn_operands` at a given rows count, folded
  over the ranks (the kind's `P("tp")` weight spec), with the sha256 of every operand.
* `kernel_at`: the real `build_draft_attn_call` at B rows (Pallas interpret), called with given operands.
* `Spmd` / `bridge`: a stand-in for torch_tpu's `JaxCallable` over R rank THREADS (assumed semantics, I: a `P("tp")` operand is
  the concatenation of the ranks' local tensors along dim 0, a `P()` operand is identical on every rank, outputs split the same
  way). Every rank thread calls with its local torch tensors; rank 0 runs the jitted JAX function once on the global arrays;
  each rank gets its slice back. It asserts the kind's dtype contract (every integer operand int32) and that each rank's output
  rows equal its block rows (the op's fake).
* `unique_custom_op`: one process holds several ranks' kinds here (one per rank thread, one per case), so each
  `torch.library.custom_op("qwen38_mk::draft_attn")` registration gets a unique suffix (test-only; the op body and fake are
  the kind's own).
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import threading
import types

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

import test_l1h3_draft_attn_call as l1c
from dcp2_harness import random_fp8
from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import selected_tokens
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.ops import BF16, F32

PB, HD, NH_RANK, RATIO, ROT, TOPK = l1c.PB, l1c.HD, l1c.NH_RANK, l1c.RATIO, l1c.ROT, l1c.TOPK
K_SCALE, V_SCALE, EPS = l1c.K_SCALE, l1c.V_SCALE, l1c.EPS
MG = 1024  # max_groups of a table sized by max_model_len <= 4096 (attach_draft_attn's rule)
PPS = MG * RATIO // PB  # 32 main-table columns (attach_draft_attn's pages_per_seq at max_model_len 4096)
ULP_BOUND = 2  # L1-H3's pre-stated rule (test_l1h3_draft_attn_call.py:12 / :240): err > 2 ulp(want) + 1e-6 is bad, elementwise
EXACT_BOUND = 0.999


def geometry(R):
    g = dataclasses.replace(qwen38.TINY, NH=NH_RANK * R)
    if R > 2:
        g = dataclasses.replace(g, KH=R, VH=3 * R)
    return g


def ctx_values(rng, live, ctx_max):
    """L1's call-test contexts first (select-all 37, fresh group 1024, page / owner edges 1279 / 255, a 3-token partial group
    2046, a listed g_a 3071), capped at ctx_max, then random ones covering every ctx % 4."""
    base = [c for c in (3071, 1024, 37, 1279, 255, 2046) if c <= ctx_max]
    extra = list(rng.integers(5, ctx_max, max(0, live - len(base))))
    return np.asarray((base + extra)[:live], np.int32)


def weights(rng, R, g):
    wq = [jnp.asarray(rng.normal(0, 0.06, (g.H, NH_RANK * 2 * HD)), BF16) for _ in range(R)]
    wkv = jnp.asarray(rng.normal(0, 0.06, (g.H, 2 * HD)), BF16)
    nq = jnp.asarray(rng.normal(0, 0.05, (HD,)), F32)
    nk = jnp.asarray(rng.normal(0, 0.05, (HD,)), F32)
    inv = 1.0 / (g.theta ** (np.arange(0, ROT, 2) / ROT))
    ang = np.arange(8192)[:, None] * inv[None]
    trig = np.zeros((8192, 128), np.float32)
    trig[:, : ROT // 2], trig[:, ROT // 2 : ROT] = np.cos(ang), np.sin(ang)
    return dict(wq=wq, wkv=wkv, nq=nq, nk=nk, trig=jnp.asarray(trig, BF16),
                qkvg=jnp.stack([jnp.concatenate([w, wkv], axis=1) for w in wq])[:, None])


def selection(rng, ctx, live, seats):
    """The carried target selection (L1's rule): every visible group where they fit, else 64 groups incl. g_a when complete."""
    budget = TOPK // RATIO
    blocks = np.full((seats, budget), -1, np.int32)
    for r in range(live):
        groups = int(ctx[r]) // RATIO
        if groups <= budget:
            blocks[r, :groups] = np.arange(groups)
        else:
            pick = rng.permutation(groups)[:budget]
            ga = int(ctx[r]) // 4
            if (int(ctx[r]) + 1) % 4 == 0 and ga not in pick:
                pick[0] = min(ga, groups - 1)
            blocks[r] = pick
    return np.asarray(selected_tokens(jnp.asarray(blocks), jnp.asarray(ctx), TOPK, RATIO))


def case_b(seed, R, D, main_slot, seats, live, ctx_max=2100, w=None):
    """L1's `_case` at `seats` slots with `live` live requests (one token each), the rest idle."""
    rng = np.random.default_rng(seed)
    g = geometry(R)
    n = seats
    ctx = np.zeros(n, np.int32)
    ctx[:live] = ctx_values(rng, live, ctx_max)
    starts = np.concatenate([[0], np.cumsum([1] * live + [0] * (n - live))]).astype(np.int32)
    positions = np.zeros(n, np.int32)
    positions[:live] = ctx[:live]
    lens = np.where(np.arange(n) < live, ctx + 1, 0)
    cols = int(-(-(lens.max() + 8) // (PB * D))) + 1
    assert cols <= PPS, cols
    pages = n * cols * D + 8
    table = (rng.permutation(pages - 1)[: n * cols] + 1).reshape(n, cols).astype(np.int32)
    pools = [random_fp8(rng, (pages, PB, 1, 4, HD)) for _ in range(R)]
    if D == 1:
        pools = [pools[0]] * R
    w = w or weights(rng, R, g)
    block = jnp.asarray(rng.normal(0, 0.5, (n, g.H)), BF16)
    return dict(g=g, n=n, live=live, ctx=ctx, starts=starts, positions=positions, table=table, pools=pools,
                shared=selection(rng, ctx, live, n), block=block, main_slot=main_slot, **w)


# ---------------------------------------------------------------- torch side


def jt(x):
    """A jax / numpy array as a torch tensor of the same dtype (BF16 / FP8 by bit pattern)."""
    a = np.asarray(x)
    if a.dtype == jnp.bfloat16:
        return torch.from_numpy(a.view(np.int16).copy()).view(torch.bfloat16)
    if a.dtype == jnp.float8_e4m3fn:
        return torch.from_numpy(a.view(np.uint8).copy()).view(torch.float8_e4m3fn)
    return torch.from_numpy(a.copy())


def tj(t):
    """A torch tensor as a jax array of the same dtype (BF16 / FP8 by bit pattern)."""
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        return jax.lax.bitcast_convert_type(jnp.asarray(t.view(torch.int16).numpy()), jnp.bfloat16)
    if t.dtype == torch.float8_e4m3fn:
        return jax.lax.bitcast_convert_type(jnp.asarray(t.view(torch.uint8).numpy()), jnp.float8_e4m3fn)
    return jnp.asarray(t.numpy())


def torch_attn(c, r, D, pool=None):
    """Rank r's draft attention layer stub: what `attach_draft_attn`, `pack_draft_attn_operands` and `DraftAttn.forward` read.
    Projections stored `[k, n]` (`qwen38_weights.kn` takes that layout as is), norms F32 (the case's values exactly)."""
    lin = lambda w: types.SimpleNamespace(weight=types.SimpleNamespace(data=jt(w)))  # noqa: E731
    wkv = c["wkv"]
    nq, nk = jt(c["nq"]), jt(c["nk"])
    return types.SimpleNamespace(
        prefix="mtp.layers.0.self_attn", main_cache_name="mtp.layers.0.self_attn",
        q_proj=lin(c["wq"][r]), k_proj=lin(wkv[:, :HD]), v_proj=lin(wkv[:, HD:]),
        q_norm=types.SimpleNamespace(weight=nq), k_norm=types.SimpleNamespace(weight=nk),
        _shared_ops=(object(), object()), _shared_no_index_state=True, _selection_source=torch.zeros(1), pooled=True,
        main_kv_dcp=D, token_topk=TOPK, compress_ratio=RATIO, fp8_kv=True, _k_scale_float=K_SCALE, _v_scale_float=V_SCALE,
        cache_owners=(jt(c["pools"][r]) if pool is None else pool,),
        _select_keywords=dict(compress_ratio=RATIO, token_topk=TOPK, norm_eps=EPS, rotary_dim=ROT, main_slot=c["main_slot"]),
        dcp_rank=torch.tensor([r % D], dtype=torch.int32), rope_cache=jt(c["trig"]),
        _selection_rows=jt(c["shared"]))


def spec_ex(c, R, D, B, tiles=None):
    from vllm_torchtpu.megakernel.wholestep import qsa_draft as qd
    from vllm_torchtpu.megakernel_runtime.qwen38_draft_attn import draft_indexed
    from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import segment_tiles

    ex = ExecSpec(R=R, B=B, T=1, ctx=128, tiles=segment_tiles(B) if tiles is None else tiles)
    spec = qd.draft_attn_spec(c["g"], R, budget=TOPK // RATIO, indexed=draft_indexed(MG, PPS, D))
    return spec, ex


def pack_torch(c, R, D, B, tiles=None):
    """(folded jax weights in plan order, [[sha256 per operand] per rank]) through the real torch packer at B rows."""
    from vllm_torchtpu.megakernel_runtime.qwen38_draft_attn import pack_draft_attn_operands

    spec, ex = spec_ex(c, R, D, B, tiles)
    per_rank = [pack_draft_attn_operands(torch_attn(c, r, D), c["g"], spec, ex) for r in range(R)]
    shas = [[hashlib.sha256(np.asarray(tj(t)).tobytes()).hexdigest() for t in ops] for ops in per_rank]
    folded = [jnp.concatenate([tj(per_rank[r][i]) for r in range(R)]) for i in range(len(per_rank[0]))]
    return folded, shas


def kernel_at(c, R, D, B, weights_folded, tiles):
    """The real `build_draft_attn_call` at B rows (Pallas interpret) on the case's operands; (attention [R, n, .], pools)."""
    from vllm_torchtpu.megakernel_runtime.qwen38_draft_attn import build_draft_attn_call

    assert B == c["n"], (B, c["n"])
    fn, _in, _out, _ = build_draft_attn_call(
        Mesh(np.array(jax.devices()[:R]), ("tp",)), c["g"], B, budget=TOPK // RATIO, max_groups=MG, pages_per_seq=PPS,
        dcp=D, main_slot=c["main_slot"], k_scale=K_SCALE, v_scale=V_SCALE, norm_eps=EPS, rotary_dim=ROT, token_topk=TOPK,
        interpret=True, tiles=tiles)
    fold = lambda xs: jnp.concatenate([jnp.asarray(x) for x in xs])  # noqa: E731
    out, pool = jax.jit(fn)(
        fold([c["block"]] * R), fold(c["pools"]), fold([c["shared"]] * R), jnp.asarray(c["table"]),
        jnp.asarray(c["positions"]), jnp.asarray(c["starts"]), jnp.asarray(c["ctx"]), c["trig"],
        jnp.asarray([r % D for r in range(R)], jnp.int32), c["nk"], *weights_folded)
    n = c["n"]
    out = np.asarray(out, np.float32).reshape(R, n, -1)
    pool = np.asarray(pool).view(np.uint8).reshape(R, *np.asarray(c["pools"][0]).shape)
    return out, [pool[r] for r in range(R)]


def compare(got, want, live):
    """L1-H3's elementwise rule: |got - want| <= ULP_BOUND (2) ulps + 1e-6 on live rows, finite, idle rows 0; `bad3` (beyond
    3 ulps + 1e-6) is printed for reference only."""
    g, w = got[:, live], want[:, live]
    err = np.abs(g - w)
    ulps = err / l1c._ulp(w)
    bad = err > ULP_BOUND * l1c._ulp(w) + 1e-6
    bad3 = err > 3 * l1c._ulp(w) + 1e-6
    return dict(exact=float((err == 0).mean()), max_ulps=float(ulps.max()), bad=int(bad.sum()), bad3=int(bad3.sum()),
                finite=bool(np.isfinite(got).all()), idle_zero=bool((got[:, ~live] == 0).all()),
                scale=float(np.abs(w).mean()))


# ---------------------------------------------------------------- rank threads


class Spmd:
    """R rank threads meeting at a barrier; rank 0 runs `fn(slots)` once and every rank reads its own result."""

    def __init__(self, R, timeout=3600):
        self.R = R
        self.barrier = threading.Barrier(R, timeout=timeout)
        self.slots = [None] * R
        self.res = None
        self.tl = threading.local()

    @property
    def rank(self):
        return self.tl.rank

    def run(self, payload, fn):
        r = self.rank
        self.slots[r] = payload
        self.barrier.wait()
        if r == 0:
            try:
                self.res = ("ok", fn(list(self.slots)))
            except BaseException as e:  # noqa: BLE001 - re-raised on every rank
                self.res = ("err", e)
        self.barrier.wait()
        st, val = self.res
        mine = val if st == "err" else val[r]
        self.barrier.wait()
        if st == "err":
            raise val
        return mine

    def allreduce(self, t):
        """Sum over the ranks in rank order (every rank gets the same bits)."""
        def f(slots):
            s = slots[0].clone()
            for x in slots[1:]:
                s = s + x
            return [s.clone() for _ in slots]
        return self.run(t.detach().clone(), f)

    def threads(self, body, *args):
        errs = [None] * self.R
        outs = [None] * self.R

        def go(r):
            self.tl.rank = r
            try:
                outs[r] = body(r, *args)
            except BaseException as e:  # noqa: BLE001
                errs[r] = e
                self.barrier.abort()

        ts = [threading.Thread(target=go, args=(r,)) for r in range(self.R)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        first = next((e for e in errs if e is not None and not isinstance(e, threading.BrokenBarrierError)), None)
        if first is None:
            first = next((e for e in errs if e is not None), None)
        if first is not None:
            raise first
        return outs


class BridgeLog:
    def __init__(self):
        self.calls = []  # (name, rows per rank, int dtypes)


def bridge(spmd, log):
    """`sharded_jax_callable` stand-in (see the module docstring)."""

    def sharded_jax_callable(name, fn, *, mesh, input_partition_specs, output_partition_specs, donate_argnums=None):
        jit_fn = jax.jit(fn)
        specs = tuple(input_partition_specs)

        def call(*targs):
            assert len(targs) == len(specs), (name, len(targs), len(specs))
            for i, t in enumerate(targs):
                if not t.dtype.is_floating_point:
                    assert t.dtype == torch.int32, ("op operand dtype", name, i, t.dtype)
            rows = int(targs[0].shape[0])

            def run(slots):
                glob = []
                for i, sp in enumerate(specs):
                    parts = [tj(s[i]) for s in slots]
                    if sp == P("tp"):
                        glob.append(jnp.concatenate(parts))
                    else:
                        assert sp == P(), (name, i, sp)
                        for x in parts[1:]:
                            assert np.array_equal(np.asarray(x).view(np.uint8), np.asarray(parts[0]).view(np.uint8)), \
                                ("replicated operand differs across ranks", name, i)
                        glob.append(parts[0])
                outs = jit_fn(*glob)
                per = [[] for _ in slots]
                for o, sp in zip(outs, output_partition_specs):
                    assert sp == P("tp"), sp
                    for r, x in enumerate(jnp.split(o, len(slots))):
                        per[r].append(jt(x))
                log.calls.append((name, rows, int(outs[0].shape[0]) // len(slots)))
                return per

            res = spmd.run(list(targs), run)
            assert int(res[0].shape[0]) == rows, ("kernel rows", name, int(res[0].shape[0]), rows)
            return tuple(res)

        return call

    return sharded_jax_callable


_OP_N = itertools.count(1)
_OP_LOCK = threading.Lock()


def unique_custom_op(monkeypatch):
    real = torch.library.custom_op

    def custom_op(name, *a, **k):
        if name != "qwen38_mk::draft_attn":
            return real(name, *a, **k)
        name = f"qwen38_mk::draft_attn_attnwire{next(_OP_N)}"
        deco = real(name, *a, **k)

        def locked(fn):
            with _OP_LOCK:
                return deco(fn)

        return locked

    monkeypatch.setattr(torch.library, "custom_op", custom_op)
