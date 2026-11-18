"""TPU gates of the fused MoE decode kernel's AR2D exchange (FUSED_MOE_DECODE_KERNEL_AR2D_MIN_TOKENS).

Runs on one tpu7x 2x2x1 host (8 cores) at the Qwen3.8-Flash EP8 decode shapes: 64 local
experts a rank (512 global), top-10, hidden 2560, routed intermediate 640, FP8 weights
with per-channel scales, the BF16 shared expert's 128-wide TP slice, the shared gate's
sigmoid in XLA (FUSED_MOE_DECODE_KERNEL_FOLD=xla). The mesh is the served EP mesh's
order (device id), so mesh index m and m ^ 1 are the two cores of one chip.

Phases (each in its own process; records as `MA {json}` lines and DIR/<phase>.json):
  g4   every served row bucket compiles with the adapter's folded builder (`_layer`),
       AR2D flag off and on (and with FUSED_MOE_DECODE_EXACT_WIDTH); the plan (ring
       depth, routed tile rows, VMEM count, token cap) and which exchange each bucket
       runs. Also: another hidden width and another rank count keep the all-to-all.
  ew   FUSED_MOE_DECODE_EXACT_WIDTH alone at the buckets AR2D leaves alone: output equal
       in bits to the padded-width call.
  g2   skew before the exchange (one rank, one chip, alternate ranks late at 5 us, 50 us,
       5 ms; every rank and chip in turn), unrouted at 128 rows and routed at 256:
       output equal to ar2d_reference of the call's own partials on all 8 ranks.
  g3   recorded AgentX routing at 128, 192, 256, 384, 512 rows: 200 consecutive calls in
       one compiled execution, inputs changing every call, each call's output checked
       on device against the summation-order model of its partials and across ranks;
       then the kernel time per call with the flag off and on.
  delay a pl.delay calibration: a kernel that sleeps n times d ns, timed at two n.
  pm   (not in the default list) FUSED_MOE_DECODE_KERNEL_PADDING_MASK: the folded call
       with the real-row count in rank[0, 2] and the rows past it holding NaN, inf and
       routes to any expert returns, on every row, the bits of the call on the masked
       inputs, at served buckets (dense, AR2D 192, routed from 208), EXACT_WIDTH on.
  g5num the paired read of G5: recorded routing at 128 to 512 rows, flag on and off on
       equal inputs, both against the exact (f64) sum of the bf16 partials that enter
       the exchange: absolute error, its distribution in bf16 steps of the exact
       result, RMS absolute error and the tensor-wide max |error| / max |result|.

    python scripts/bench_moe_ar2d.py --out-dir /tmp/moe_ar2d [--phases g4,ew,g2,g3]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import traceback

import ml_dtypes
import numpy as np

BF16_NP = ml_dtypes.bfloat16
N, E, H, I, I_S, TOPK = 8, 64, 2560, 640, 128, 10
AR2D_MIN = 128
ROUTED_MIN = 208  # FUSED_MOE_DECODE_KERNEL_ROUTED_MIN_TOKENS, served
MAX_TOKENS = 512  # FUSED_MOE_DECODE_KERNEL_MAX_TOKENS, served
SERVED_BUCKETS = [16, 32, 64, 96, 112, 128, 144, 160, 192, 208, 224, 240, 256, 320, 384,
                  448, 512]
PHASES = ("g4", "ew", "g2", "g3", "g5num", "delay")
OPT_PHASES = ("pm",)  # run only when named in --phases
HERE = os.path.dirname(os.path.abspath(__file__))
ROUTING_NPZ = os.path.join(HERE, "data", "agentx_routing.npz")

RECORDS: list[dict] = []
OUT_FILE = None


def emit(rec):
    RECORDS.append(rec)
    print("MA " + json.dumps(rec, sort_keys=True, default=str), flush=True)
    if OUT_FILE:
        with open(OUT_FILE, "w") as f:
            json.dump(RECORDS, f, indent=1, default=str)


def log(msg):
    print(f"[ma {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- driver


# A record is a check when it carries one of these fields; it fails when the
# field is False or when it carries an "error".
CHECK_FIELDS = ("ok", "equal", "compiled")


def check_outcome(rec):
    """None for a record that is not a check, else True (passed) or False."""
    if "error" in rec:
        return False
    vals = [rec[f] for f in CHECK_FIELDS if f in rec]
    if not vals:
        return None
    return all(v is True for v in vals)


def record_name(rec):
    keys = ("kind", "tokens", "pattern", "delay_ns", "routed", "exact_width", "ar2d_min",
            "name")
    return "/".join(str(rec[k]) for k in keys if k in rec)


def check_summary(recs, label):
    """One line naming the failed checks; returns the number failed."""
    outcome = [(record_name(r), check_outcome(r)) for r in recs]
    passed = [n for n, o in outcome if o is True]
    failed = [n for n, o in outcome if o is False]
    print(f"{label} checks passed {len(passed)} failed {len(failed)}"
          + (f": {', '.join(failed[:40])}" if failed else ""), flush=True)
    return len(failed)


def run_all(args):
    os.makedirs(args.out_dir, exist_ok=True)
    rc_all = 0
    recs = []
    for ph in args.phases.split(","):
        cmd = [sys.executable, os.path.abspath(__file__), "--phase", ph, "--out-dir",
               args.out_dir]
        log(f"phase {ph}: {' '.join(cmd)}")
        t0 = time.time()
        try:
            rc = subprocess.run(cmd, timeout=args.phase_timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        log(f"phase {ph} rc={rc} in {time.time() - t0:.0f} s")
        print(f"MA_PHASE {ph} {rc}", flush=True)
        if rc != 0:
            rc_all = 1
        path = os.path.join(args.out_dir, f"{ph}.json")
        if os.path.exists(path):
            with open(path) as f:
                recs += json.load(f)
        else:
            rc_all = 1
            print(f"MA_PHASE {ph} wrote no records", flush=True)
    if check_summary(recs, "MA_SUMMARY"):
        rc_all = 1
    return rc_all


def phase_selftest(ctx):
    """Driver check without a TPU: one passing and one failing record."""
    emit(dict(kind="selftest", name="passes", ok=True))
    emit(dict(kind="selftest", name="fails", ok=False))


# --------------------------------------------------------------------------- setup


class Ctx:
    def __init__(self):
        import jax
        import jax.numpy as jnp
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P

        self.jax, self.jnp, self.P = jax, jnp, P
        devs = sorted(jax.devices(), key=lambda d: d.id)[:N]
        for m in range(0, N, 2):
            assert tuple(devs[m].coords) == tuple(devs[m + 1].coords), devs
        self.mesh = Mesh(np.asarray(devs), ("d",))
        self.sh = NamedSharding(self.mesh, P("d"))
        self.rep = NamedSharding(self.mesh, P())
        emit(dict(kind="env", jax=jax.__version__, device_kind=devs[0].device_kind,
                  devices=[(d.id, list(d.coords), d.core_on_chip) for d in devs],
                  libtpu_init_args=os.environ.get("LIBTPU_INIT_ARGS", "")))
        self.weights = self._weights(0)
        self.rank = jax.device_put(
            np.stack([np.arange(N), np.arange(N) * E], 1).astype(np.int32), self.sh)

    def _weights(self, seed):
        """Per-rank FP8 experts and BF16 shared expert, generated on device."""
        jax, jnp, P = self.jax, self.jnp, self.P
        f8 = jnp.float8_e4m3fn

        def gen(key):
            k = jax.random.split(key[0], 4)
            w1 = (jax.random.normal(k[0], (E, H, 2 * I)) * 64).astype(f8)
            w2 = (jax.random.normal(k[1], (E, I, H)) * 64).astype(f8)
            w1s = jnp.full((E, 1, 1, 2 * I), 0.02 / 64, jnp.float32)
            w2s = jnp.full((E, 1, 1, H), 0.02 / 64, jnp.float32)
            sw1 = (jax.random.normal(k[2], (H, 2 * I_S)) * 0.02).astype(jnp.bfloat16)
            sw2 = (jax.random.normal(k[3], (I_S, H)) * 0.02).astype(jnp.bfloat16)
            return w1, w2, w1s, w2s, sw1[None], sw2[None]

        keys = jax.device_put(jax.random.split(jax.random.PRNGKey(seed), N), self.sh)
        f = jax.jit(jax.shard_map(gen, mesh=self.mesh, in_specs=(P("d"),),
                                  out_specs=(P("d"),) * 6, check_vma=False))
        w1, w2, w1s, w2s, sw1, sw2 = f(keys)
        gate = jax.device_put(
            (jax.random.normal(jax.random.PRNGKey(seed + 1), (1, H)) * 0.02).astype(
                jnp.bfloat16), self.rep)
        return dict(w1=w1, w2=w2, w1s=w1s, w2s=w2s, sw1=sw1, sw2=sw2, gate=gate)

    def kernel_call(self, tokens, *, ar2d_min, exact_width=False, partials=False,
                    delay=None, routed="served", valid_rows=False):
        """The served folded call (fused_moe_decode as the adapter calls it with
        FOLD=xla) as a per-rank function f(rank, w..., x, gate, tw, ti)."""
        from vllm_torchtpu.kernels.experimental.fused_moe_decode import (
            fused_moe_decode_kernel as kmod,
        )

        jnp = self.jnp
        m = tokens + (-tokens) % 16
        if routed == "served":
            routed = (kmod.ROUTED_BLOCK, kmod.routed_tile_rows(m)) if m >= ROUTED_MIN else None

        def f(rank, w1, w2, w1s, w2s, sw1, sw2, x, gate, tw, ti):
            out = kmod.fused_moe_decode(
                rank, x, w1, w2, w1s, w2s, sw1[0], jnp.ones((2 * I_S,), jnp.float32),
                sw2[0], jnp.ones((H,), jnp.float32), gate, tw, ti,
                n_ranks=N, expert_start="rank", fold=True, routed=routed,
                ar2d_min_tokens=ar2d_min, exact_width=exact_width,
                ar2d_partials=partials, ar2d_delay_ns=delay, valid_rows=valid_rows)
            return out

        return f

    def operands(self):
        w = self.weights
        return (self.rank, w["w1"], w["w2"], w["w1s"], w["w2s"], w["sw1"], w["sw2"])

    def in_specs(self):
        P = self.P
        return (P("d"),) * 7 + (P(),) * 4


def routing_inputs(m, n_calls, seed, recorded=True):
    """x [n, m, H] bf16 values (f32), tw [n, m, 10] f32, ti [n, m, 10] int32: AgentX
    routing (4 rows a verify window, distinct replies, layers in turn) or uniform."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_calls, m, H)).astype(np.float32)
    tw = np.exp(rng.normal(size=(n_calls, m, TOPK)))
    tw = (tw / tw.sum(-1, keepdims=True)).astype(np.float32)
    if recorded and os.path.exists(ROUTING_NPZ):
        z = np.load(ROUTING_NPZ)
        experts, groups = z["experts"], z["groups"]  # [windows, 4, layers, 10]
        replies = np.unique(groups)
        L = experts.shape[2]
        ti = np.empty((n_calls, m, TOPK), np.int32)
        for c in range(n_calls):
            pick = rng.choice(len(replies), m // 4, replace=False)
            wins = [int(rng.choice(np.nonzero(groups == g)[0])) for g in replies[pick]]
            ti[c] = experts[wins, :, c % L, :].reshape(m, TOPK)
    else:
        ti = np.stack([np.stack([rng.choice(N * E, TOPK, replace=False) for _ in range(m)])
                       for _ in range(n_calls)]).astype(np.int32)
    return x, tw, ti


# --------------------------------------------------------------------------- models


def kernel_names(text):
    """The fused MoE decode kernel names a lowered program carries."""
    return sorted(set(re.findall(r"fused_moe_decode[\w\-]*", text)))


def model_jnp(jnp, acc):
    """ar2d_reference in jnp: acc [8, m, h] f32 accumulators (all m rows) -> [m, h]
    bf16. Own operands stay f32, sent ones are rounded to bf16; reduce_precision
    pins each rounding, since XLA may otherwise keep f32 across a convert pair
    (excess precision) and skip it."""
    from jax import lax

    f32, bf16 = jnp.float32, jnp.bfloat16

    def rb(a):
        return lax.reduce_precision(a, 8, 7)

    a = acc.astype(f32)
    _, m, h = a.shape
    half, part = h // 2, h // 4
    i = jnp.arange(m)
    t = (i // (m // 2))[:, None]
    u = ((i % (m // 2)) // (m // 4))[:, None]
    outs = []
    for c in (0, 1):
        cs = slice(c * half, (c + 1) * half)
        A = {(x, y): a[4 * x + 2 * y + c][:, cs] + rb(a[4 * x + 2 * y + 1 - c][:, cs])
             for x in (0, 1) for y in (0, 1)}
        b0 = {y: jnp.where(t == 0, A[0, y] + rb(A[1, y]), A[1, y] + rb(A[0, y]))
              for y in (0, 1)}
        c0 = jnp.where(u == 0, b0[0] + rb(b0[1]), b0[1] + rb(b0[0]))
        b1 = {x: jnp.where(t == 0, A[x, 0] + rb(A[x, 1]), A[x, 1] + rb(A[x, 0]))
              for x in (0, 1)}
        c1 = jnp.where(u == 0, b1[0] + rb(b1[1]), b1[1] + rb(b1[0]))
        outs += [rb(c0[:, :part]), rb(c1[:, part:])]
    return jnp.concatenate(outs, axis=1).astype(bf16)


# --------------------------------------------------------------------------- phases


def phase_g4(ctx):
    """Compile every served bucket through the adapter's folded builder, flag off and
    on, and record the plan and the exchange each call runs."""
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.experimental.fused_moe_decode import (
        fused_moe_decode_kernel as kmod,
    )
    from vllm_torchtpu.layers.adapter import fused_moe_decode as fmd

    cap = kmod._vmem_capacity()
    f8 = jnp.float8_e4m3fn
    for ew in (False, True):
        for ar2d_min in (0, AR2D_MIN):
            need = kmod.vmem_bytes(MAX_TOKENS, H, I, I_S, E, N, f8,
                                   routed=(kmod.ROUTED_BLOCK, kmod.routed_tile_rows(512)),
                                   ar2d_min_tokens=ar2d_min)
            emit(dict(kind="token_cap", exact_width=ew, ar2d_min=ar2d_min,
                      max_tokens=MAX_TOKENS, vmem_need=need, vmem_capacity=cap,
                      fits=need <= cap))
    for tokens in SERVED_BUCKETS:
        m = tokens
        plan = kmod.plan_weight_ring(m, H, I, I_S, E, N, f8)
        routed = (kmod.ROUTED_BLOCK, kmod.routed_tile_rows(m)) if m >= ROUTED_MIN else None
        for ew in (False, True):
            for ar2d_min in (0, AR2D_MIN):
                rec = dict(kind="g4", tokens=tokens, exact_width=ew, ar2d_min=ar2d_min,
                           plan=list(plan), routed=routed,
                           ar2d=kmod.ar2d_applies(m, H, N, ar2d_min,
                                                  whole_experts=plan[0] is None),
                           vmem_bytes=kmod.vmem_bytes(tokens, H, I, I_S, E, N, f8,
                                                      routed=routed,
                                                      ar2d_min_tokens=ar2d_min))
                local = lambda rank, w1, w2, w1s, w2s, sw1, sw2, x, gate, tw, ti, a=ar2d_min, e=ew: (  # noqa: E731
                    fmd._layer(rank, x, w1, w2, w1s, w2s, sw1[0],
                               jnp.ones((2 * I_S,), jnp.float32), sw2[0],
                               jnp.ones((H,), jnp.float32), gate, tw, ti,
                               n_ranks=N, topk=TOPK, expert_parallel=True,
                               max_tokens=MAX_TOKENS, shared_gated=True, fold="xla",
                               routed_min=ROUTED_MIN, ar2d_min=a, exact_width=e))
                fn = jax.jit(jax.shard_map(local, mesh=ctx.mesh, in_specs=ctx.in_specs(),
                                           out_specs=ctx.P("d"), check_vma=False))
                args = (*ctx.operands(),
                        jax.ShapeDtypeStruct((tokens, H), jnp.bfloat16, sharding=ctx.rep),
                        ctx.weights["gate"],
                        jax.ShapeDtypeStruct((tokens, TOPK), jnp.bfloat16, sharding=ctx.rep),
                        jax.ShapeDtypeStruct((tokens, TOPK), jnp.int32, sharding=ctx.rep))
                t0 = time.time()
                try:
                    lowered = fn.lower(*args)
                    names = kernel_names(lowered.as_text())
                    lowered.compile()
                    rec.update(compiled=True, compile_s=round(time.time() - t0, 1),
                               kernel=names[:3],
                               kernel_has_ar2d=any("-ar2d" in n for n in names))
                except Exception as e:  # noqa: BLE001
                    rec.update(compiled=False, error=f"{type(e).__name__}: {e}"[:1500])
                emit(rec)
    # Other shapes keep the all-to-all.
    other = [
        dict(what="hidden 1536", m=128, h=1536, n=N),
        dict(what="4 ranks", m=128, h=H, n=4),
        dict(what="160 rows", m=160, h=H, n=N),
    ]
    for o in other:
        emit(dict(kind="g4_other", **o,
                  ar2d=kmod.ar2d_applies(o["m"], o["h"], o["n"], AR2D_MIN),
                  plan_off=list(kmod.plan_weight_ring(o["m"], o["h"], I, I_S, E, o["n"], f8)),
                  vmem_on=kmod.vmem_bytes(o["m"], o["h"], I, I_S, E, o["n"], f8,
                                          ar2d_min_tokens=AR2D_MIN),
                  vmem_off=kmod.vmem_bytes(o["m"], o["h"], I, I_S, E, o["n"], f8)))
    # Lowered with the flag on: 4 ranks (half the host) and hidden 1536 build the
    # all-to-all kernel (no -ar2d in its name), 8 ranks at 2560 the AR2D one.
    from jax.sharding import Mesh, NamedSharding

    for n, h in ((4, H), (N, 1536), (N, H)):
        mesh = Mesh(np.asarray(sorted(jax.devices(), key=lambda d: d.id)[:n]), ("d",))
        sh, rep = NamedSharding(mesh, ctx.P("d")), NamedSharding(mesh, ctx.P())

        def local(rank, w1, w2, w1s, w2s, sw1, sw2, x, gate, tw, ti, n=n):
            return kmod.fused_moe_decode(
                rank, x, w1, w2, w1s, w2s, sw1[0], jnp.ones((2 * I_S,), jnp.float32),
                sw2[0], jnp.ones((h,), jnp.float32), gate, tw, ti, n_ranks=n,
                expert_start="rank", fold=True, ar2d_min_tokens=AR2D_MIN)[None]

        S = jax.ShapeDtypeStruct
        args = (S((n, 2), jnp.int32, sharding=sh), S((n * E, h, 2 * I), f8, sharding=sh),
                S((n * E, I, h), f8, sharding=sh),
                S((n * E, 1, 1, 2 * I), jnp.float32, sharding=sh),
                S((n * E, 1, 1, h), jnp.float32, sharding=sh),
                S((n, h, 2 * I_S), jnp.bfloat16, sharding=sh),
                S((n, I_S, h), jnp.bfloat16, sharding=sh),
                S((128, h), jnp.bfloat16, sharding=rep), S((1, h), jnp.bfloat16, sharding=rep),
                S((128, TOPK), jnp.bfloat16, sharding=rep),
                S((128, TOPK), jnp.int32, sharding=rep))
        rec = dict(kind="g4_lowered", ranks=n, hidden=h, tokens=128)
        try:
            fn = jax.jit(jax.shard_map(local, mesh=mesh, in_specs=ctx.in_specs(),
                                       out_specs=ctx.P("d"), check_vma=False))
            names = kernel_names(fn.lower(*args).as_text())
            rec.update(kernel=names[:3], kernel_has_ar2d=any("-ar2d" in x for x in names))
        except Exception as e:  # noqa: BLE001
            rec.update(error=f"{type(e).__name__}: {e}"[:1500])
        emit(rec)


def phase_ew(ctx):
    """FUSED_MOE_DECODE_EXACT_WIDTH on its own at the buckets AR2D leaves alone:
    bitwise the padded call (the all-to-all with the balanced split)."""
    jax = ctx.jax
    for tokens in [16, 32, 64, 96, 112, 144, 160, 208, 224, 240]:
        x, tw, ti = routing_inputs(tokens, 1, seed=tokens)
        args = (*ctx.operands(),
                jax.device_put(ctx.jnp.asarray(x[0], ctx.jnp.bfloat16), ctx.rep),
                ctx.weights["gate"],
                jax.device_put(ctx.jnp.asarray(tw[0], ctx.jnp.bfloat16), ctx.rep),
                jax.device_put(ti[0], ctx.rep))
        outs = {}
        for ew in (False, True):
            f = ctx.kernel_call(tokens, ar2d_min=AR2D_MIN, exact_width=ew)
            g = jax.jit(jax.shard_map(lambda *a, f=f: f(*a)[None], mesh=ctx.mesh,
                                      in_specs=ctx.in_specs(), out_specs=ctx.P("d"),
                                      check_vma=False))
            outs[ew] = np.asarray(g(*args)).view(np.uint16)
        same = bool(np.array_equal(outs[False], outs[True]))
        emit(dict(kind="ew", tokens=tokens, equal=same, shape=list(outs[True].shape)))


def phase_pm(ctx, ar2d_min=192, buckets=(16, 64, 128, 192, 256, 512)):
    """FUSED_MOE_DECODE_KERNEL_PADDING_MASK on device (see the module docstring).
    Masked inputs: rows from n on zero, weights zero, routes to experts 0 .. 9 (what
    the decoder layer's mask and zero_routing_weights_for_padding leave today)."""
    jax, jnp = ctx.jax, ctx.jnp
    for tokens in buckets:
        x, tw, ti = routing_inputs(tokens, 1, seed=3000 + tokens)
        for n in sorted({tokens - 3, tokens // 2}):
            rng = np.random.default_rng(tokens * 1000 + n)
            masked = [a[0].copy() for a in (x, tw, ti)]
            masked[0][n:], masked[1][n:], masked[2][n:] = 0.0, 0.0, np.arange(TOPK)
            junk = [a[0].copy() for a in (x, tw, ti)]
            junk[0][n:] = rng.normal(size=junk[0][n:].shape) * 1e4
            junk[0][n:, ::97], junk[0][n:, 5::89] = np.nan, np.inf
            junk[1][n:] = np.nan
            junk[2][n:] = rng.integers(-N * E, 2 * N * E, size=junk[2][n:].shape)
            rec = dict(kind="pm", tokens=tokens, name=f"n{n}")
            try:
                outs = {}
                for flag, (xx, ww, ii) in ((False, masked), (True, junk)):
                    cols = [np.arange(N), np.arange(N) * E] + ([np.full(N, n)] if flag else [])
                    rank = jax.device_put(np.stack(cols, 1).astype(np.int32), ctx.sh)
                    f = ctx.kernel_call(tokens, ar2d_min=ar2d_min, exact_width=True,
                                        valid_rows=flag)
                    g = jax.jit(jax.shard_map(lambda *a, f=f: f(*a)[None], mesh=ctx.mesh,
                                              in_specs=ctx.in_specs(), out_specs=ctx.P("d"),
                                              check_vma=False))
                    args = (rank, *ctx.operands()[1:],
                            jax.device_put(jnp.asarray(xx, jnp.bfloat16), ctx.rep),
                            ctx.weights["gate"],
                            jax.device_put(jnp.asarray(ww, jnp.bfloat16), ctx.rep),
                            jax.device_put(ii, ctx.rep))
                    outs[flag] = np.asarray(g(*args)).view(np.uint16)
                diff = outs[True] != outs[False]
                rec.update(equal=not diff.any(), mismatched=int(diff.sum()),
                           ok=bool(np.isfinite(outs[False].view(BF16_NP).astype(np.float32)).all()))
            except Exception as e:  # noqa: BLE001
                rec.update(ok=False, error=f"{type(e).__name__}: {e}"[:1500])
                traceback.print_exc()
            emit(rec)


def _check_call(ctx, tokens, routed, delay, x, tw, ti, ew=False):
    jax, jnp = ctx.jax, ctx.jnp
    f = ctx.kernel_call(tokens, ar2d_min=AR2D_MIN, exact_width=ew, partials=True,
                        delay=delay, routed=routed)
    g = jax.jit(jax.shard_map(lambda *a: tuple(o[None] for o in f(*a)), mesh=ctx.mesh,
                              in_specs=ctx.in_specs(), out_specs=(ctx.P("d"),) * 2,
                              check_vma=False))
    args = (*ctx.operands(), jax.device_put(jnp.asarray(x, jnp.bfloat16), ctx.rep),
            ctx.weights["gate"], jax.device_put(jnp.asarray(tw, jnp.bfloat16), ctx.rep),
            jax.device_put(ti, ctx.rep))
    t0 = time.time()
    out, parts = g(*args)
    out, parts = np.asarray(out), np.asarray(parts)
    wall = time.time() - t0
    # The compiled call again: its time shows the injected delay.
    t0 = time.perf_counter()
    ctx.jax.block_until_ready(g(*args))
    wall = (wall, time.perf_counter() - t0)
    from vllm_torchtpu.kernels.experimental.fused_moe_decode.ar2d_reference import (
        ar2d_mismatches,
    )

    return ar2d_mismatches(out, parts), wall


def phase_g2(ctx):
    """Skew injected after the entry signals and before s1."""
    from vllm_torchtpu.kernels.experimental.fused_moe_decode import (
        fused_moe_decode_kernel as kmod,
    )

    patterns = [(f"rank {r} late", [r]) for r in range(N)]
    patterns += [(f"chip {c} late", [2 * c, 2 * c + 1]) for c in range(N // 2)]
    patterns += [("even ranks late", [0, 2, 4, 6]), ("odd ranks late", [1, 3, 5, 7])]
    for tokens, routed in ((128, None), (256, (kmod.ROUTED_BLOCK, 64))):
        x, tw, ti = routing_inputs(tokens, 1, seed=1000 + tokens)
        res, wall = _check_call(ctx, tokens, routed, None, x[0], tw[0], ti[0])
        emit(dict(kind="g2", tokens=tokens, routed=routed, delay_ns=0, pattern="none",
                  late=[], **res, rerun_ms=round(wall[1] * 1e3, 3),
                  ok=res["mismatched"] == 0 and res["ranks_identical"]))
        for ns in (50_000, 5_000, 5_000_000):
            for name, late in patterns:
                delay = [ns if r in late else 0 for r in range(N)]
                rec = dict(kind="g2", tokens=tokens, routed=routed, delay_ns=ns,
                           pattern=name, late=late)
                try:
                    res, wall = _check_call(ctx, tokens, routed, delay, x[0], tw[0], ti[0])
                    rec.update(res, wall_s=round(wall[0], 3), rerun_ms=round(wall[1] * 1e3, 3),
                               ok=res["mismatched"] == 0 and res["ranks_identical"])
                except Exception as e:  # noqa: BLE001
                    rec.update(ok=False, error=f"{type(e).__name__}: {e}"[:1500])
                    traceback.print_exc()
                emit(rec)


def phase_g3(ctx):
    """Recorded routing, 200 consecutive calls per execution, checked on device."""
    jax, jnp, P = ctx.jax, ctx.jnp, ctx.P
    from jax import lax

    n_calls = 200
    for tokens in (128, 192, 256, 384, 512):
        x, tw, ti = routing_inputs(tokens, n_calls, seed=2000 + tokens)
        xs = jax.device_put(jnp.asarray(x, jnp.bfloat16), ctx.rep)
        tws = jax.device_put(jnp.asarray(tw, jnp.bfloat16), ctx.rep)
        tis = jax.device_put(ti, ctx.rep)
        # Correctness: every call against the model of its own partials, on device.
        f = ctx.kernel_call(tokens, ar2d_min=AR2D_MIN, partials=True)

        def checked(rank, w1, w2, w1s, w2s, sw1, sw2, xs, gate, tws, tis, f=f):
            def body(c, carry):
                bad, diff_rank = carry
                out, part = f(rank, w1, w2, w1s, w2s, sw1, sw2, xs[c], gate, tws[c], tis[c])
                parts = lax.all_gather(part, "d")  # [8, m, h] f32
                outs = lax.all_gather(out, "d")
                ref = model_jnp(jnp, parts)[: out.shape[0]]
                bad = bad + jnp.sum(out.view(jnp.uint16) != ref.view(jnp.uint16))
                diff_rank = diff_rank + jnp.sum(outs.view(jnp.uint16)
                                                != out.view(jnp.uint16)[None])
                return bad, diff_rank

            bad, diff = lax.fori_loop(0, xs.shape[0], body, (jnp.int32(0), jnp.int32(0)))
            return jnp.stack([bad, diff])[None]

        g = jax.jit(jax.shard_map(checked, mesh=ctx.mesh, in_specs=ctx.in_specs(),
                                  out_specs=P("d"), check_vma=False))
        rec = dict(kind="g3_check", tokens=tokens, calls=n_calls)
        try:
            t0 = time.time()
            res = np.asarray(g(*ctx.operands(), xs, ctx.weights["gate"], tws, tis))
            rec.update(mismatched_per_rank=res[:, 0].tolist(),
                       rank_diff_per_rank=res[:, 1].tolist(),
                       ok=bool((res == 0).all()), wall_s=round(time.time() - t0, 2))
        except Exception as e:  # noqa: BLE001
            rec.update(ok=False, error=f"{type(e).__name__}: {e}"[:1500])
            traceback.print_exc()
        emit(rec)
        # The jnp model against the NumPy model on the first call.
        try:
            res, _ = _check_call(ctx, tokens, "served", None, x[0], tw[0], ti[0])
            emit(dict(kind="g3_np_check", tokens=tokens, **res))
        except Exception as e:  # noqa: BLE001
            emit(dict(kind="g3_np_check", tokens=tokens, error=str(e)[:1500]))
        # Timing: the same 200 calls, flag off and on, wall clock per call from
        # (t(200 calls) - t(20 calls)) / 180, median of 7.
        xu, twu, tiu = routing_inputs(tokens, n_calls, seed=2500 + tokens, recorded=False)
        uniform = (jax.device_put(jnp.asarray(xu, jnp.bfloat16), ctx.rep),
                   jax.device_put(jnp.asarray(twu, jnp.bfloat16), ctx.rep),
                   jax.device_put(tiu, ctx.rep))
        for routing, ar2d_min, ew in [("recorded", a, e) for a in (0, AR2D_MIN)
                                      for e in (False, True)] + [
                ("uniform", a, False) for a in (0, AR2D_MIN)]:
            f = ctx.kernel_call(tokens, ar2d_min=ar2d_min, exact_width=ew)

            def timed(rank, w1, w2, w1s, w2s, sw1, sw2, xs, gate, tws, tis, n, f=f):
                def body(c, acc):
                    out = f(rank, w1, w2, w1s, w2s, sw1, sw2, xs[c], gate, tws[c],
                            tis[c])
                    return acc + out[0, 0].astype(jnp.float32)

                return lax.fori_loop(0, n[0], body, jnp.float32(0))[None]

            gt = jax.jit(jax.shard_map(timed, mesh=ctx.mesh,
                                       in_specs=(*ctx.in_specs(), P()),
                                       out_specs=P("d"), check_vma=False))
            if routing == "recorded":
                a = (*ctx.operands(), xs, ctx.weights["gate"], tws, tis)
            else:
                a = (*ctx.operands(), uniform[0], ctx.weights["gate"], uniform[1],
                     uniform[2])
            n20 = jax.device_put(np.array([20], np.int32), ctx.rep)
            n200 = jax.device_put(np.array([200], np.int32), ctx.rep)
            try:
                jax.block_until_ready(gt(*a, n20))
                jax.block_until_ready(gt(*a, n200))
                t20, t200 = [], []
                for _ in range(7):
                    t = time.perf_counter()
                    jax.block_until_ready(gt(*a, n20))
                    t20.append(time.perf_counter() - t)
                    t = time.perf_counter()
                    jax.block_until_ready(gt(*a, n200))
                    t200.append(time.perf_counter() - t)
                us = (statistics.median(t200) - statistics.median(t20)) / 180 * 1e6
                emit(dict(kind="g3_time", tokens=tokens, ar2d_min=ar2d_min,
                          exact_width=ew, routing=routing, us_per_call=round(us, 2),
                          t20_ms=round(statistics.median(t20) * 1e3, 3),
                          t200_ms=round(statistics.median(t200) * 1e3, 3)))
            except Exception as e:  # noqa: BLE001
                emit(dict(kind="g3_time", tokens=tokens, ar2d_min=ar2d_min,
                          exact_width=ew, error=f"{type(e).__name__}: {e}"[:1500]))


def phase_delay(ctx):
    """How long pl.delay(ns) sleeps: a kernel of n sleeps, (t(n=41) - t(n=1)) / 40,
    and the AR2D kernel's stepped delay (all ranks late) against no delay."""
    jax, jnp = ctx.jax, ctx.jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    def call(ns, n):
        def body(x_ref, o_ref):
            def step(i, c):
                pl.delay(jnp.int32(ns))
                return c

            lax.fori_loop(0, n, step, 0)
            o_ref[...] = x_ref[...]

        return pl.pallas_call(body, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
                              compiler_params=pltpu.CompilerParams(has_side_effects=True))

    x = jnp.zeros((8, 128), jnp.float32)
    for ns in (1_000, 5_000, 50_000, 200_000, 500_000, 1_000_000, 5_000_000):
        ts = {}
        for n in (1, 41):
            f = jax.jit(call(ns, n))
            jax.block_until_ready(f(x))
            samples = []
            for _ in range(9):
                t0 = time.perf_counter()
                jax.block_until_ready(f(x))
                samples.append(time.perf_counter() - t0)
            ts[n] = statistics.median(samples)
        emit(dict(kind="delay", ns=ns, us_per_delay=round((ts[41] - ts[1]) / 40 * 1e6, 3),
                  t1_ms=round(ts[1] * 1e3, 4), t41_ms=round(ts[41] * 1e3, 4)))
    # The AR2D kernel's own delay (steps of _DELAY_STEP_NS), every rank late.
    x, tw, ti = routing_inputs(128, 1, seed=77)
    for ns in (0, 5_000, 50_000, 5_000_000):
        delay = None if ns == 0 else [ns] * N
        f = ctx.kernel_call(128, ar2d_min=AR2D_MIN, delay=delay)
        g = jax.jit(jax.shard_map(lambda *a, f=f: f(*a)[None], mesh=ctx.mesh,
                                  in_specs=ctx.in_specs(), out_specs=ctx.P("d"),
                                  check_vma=False))
        args = (*ctx.operands(), jax.device_put(jnp.asarray(x[0], jnp.bfloat16), ctx.rep),
                ctx.weights["gate"], jax.device_put(jnp.asarray(tw[0], jnp.bfloat16), ctx.rep),
                jax.device_put(ti[0], ctx.rep))
        jax.block_until_ready(g(*args))
        samples = []
        for _ in range(9):
            t0 = time.perf_counter()
            jax.block_until_ready(g(*args))
            samples.append(time.perf_counter() - t0)
        emit(dict(kind="delay_ar2d", ns=ns, call_ms=round(statistics.median(samples) * 1e3, 4)))


def phase_g5num(ctx, n_calls=16):
    """Paired numerics on equal inputs: today (flag off), revision 2 (its model,
    bit-exact to its kernel on TPU in j-dac0e9c1), revision 3 (the kernel), each
    against the exact sum of the unrounded f32 accumulators, and against the exact
    sum of the bf16 partials beside it."""
    jax, jnp = ctx.jax, ctx.jnp
    from vllm_torchtpu.kernels.experimental.fused_moe_decode.ar2d_reference import (
        ar2d_reference_bf16_operands,
    )

    edges = [0.0, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0, np.inf]
    for tokens in (128, 192, 256, 320, 384, 448, 512):
        x, tw, ti = routing_inputs(tokens, n_calls, seed=3000 + tokens)
        f_on = ctx.kernel_call(tokens, ar2d_min=AR2D_MIN, partials=True)
        f_off = ctx.kernel_call(tokens, ar2d_min=0)
        g_on = jax.jit(jax.shard_map(lambda *a: tuple(o[None] for o in f_on(*a)),
                                     mesh=ctx.mesh, in_specs=ctx.in_specs(),
                                     out_specs=(ctx.P("d"),) * 2, check_vma=False))
        g_off = jax.jit(jax.shard_map(lambda *a: f_off(*a)[None], mesh=ctx.mesh,
                                      in_specs=ctx.in_specs(), out_specs=ctx.P("d"),
                                      check_vma=False))
        forms = ("today", "rev2", "rev3")
        errs = {(f, r): [] for f in forms for r in ("acc", "bf16")}
        steps = {(f, r): [] for f in forms for r in ("acc", "bf16")}
        rowsteps = {(f, r): [] for f in forms for r in ("acc", "bf16")}
        ref_max = 0.0
        for c in range(n_calls):
            args = (*ctx.operands(), jax.device_put(jnp.asarray(x[c], jnp.bfloat16), ctx.rep),
                    ctx.weights["gate"], jax.device_put(jnp.asarray(tw[c], jnp.bfloat16), ctx.rep),
                    jax.device_put(ti[c], ctx.rep))
            on, acc = (np.asarray(a) for a in g_on(*args))
            off = np.asarray(g_off(*args))
            acc = acc[:, :tokens]
            rev2 = ar2d_reference_bf16_operands(acc)
            outs = {"today": off[0], "rev2": rev2, "rev3": on[0]}
            exact = {"acc": acc.astype(np.float64).sum(axis=0),
                     "bf16": acc.astype(BF16_NP).astype(np.float64).sum(axis=0)}
            for r, ex in exact.items():
                ulp = np.exp2(np.floor(np.log2(np.maximum(np.abs(ex), 1e-30))) - 7)
                row_rms = np.sqrt((ex ** 2).mean(axis=1, keepdims=True))
                row_ulp = np.exp2(np.floor(np.log2(np.maximum(row_rms, 1e-30))) - 7)
                for f in forms:
                    e = np.abs(outs[f].astype(np.float64) - ex)
                    errs[f, r].append(e.ravel())
                    steps[f, r].append((e / ulp).ravel())
                    rowsteps[f, r].append((e / row_ulp).ravel())
            ref_max = max(ref_max, float(np.abs(exact["acc"]).max()))
        rec = dict(kind="g5num3", tokens=tokens, calls=n_calls, ref_max=ref_max)
        for (f, r), lst in errs.items():
            e = np.concatenate(lst)
            st = np.concatenate(steps[f, r])
            rs = np.concatenate(rowsteps[f, r])
            hist = np.histogram(st, bins=edges)[0] / st.size
            rec[f"{f}_vs_{r}"] = dict(
                rms_abs=float(np.sqrt((e ** 2).mean())), max_abs=float(e.max()),
                tensor_wide=float(e.max() / ref_max),
                frac_steps={f"<={b}" if np.isfinite(b) else ">8": round(float(v), 5)
                            for b, v in zip(edges[1:], hist)},
                rms_row_steps=float(np.sqrt((rs ** 2).mean())), max_row_steps=float(rs.max()))
        for r in ("acc", "bf16"):
            base = rec[f"today_vs_{r}"]["rms_abs"]
            rec[f"rms_ratio_vs_{r}"] = {f: rec[f"{f}_vs_{r}"]["rms_abs"] / base
                                        for f in ("rev2", "rev3")}
        emit(rec)


def main():
    global OUT_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=("all", "selftest") + PHASES + OPT_PHASES)
    ap.add_argument("--phases", default=",".join(PHASES))
    ap.add_argument("--out-dir", default="/tmp/moe_ar2d")
    ap.add_argument("--phase-timeout", type=int, default=2400)
    args = ap.parse_args()
    if args.phase == "all":
        sys.exit(run_all(args))
    os.makedirs(args.out_dir, exist_ok=True)
    OUT_FILE = os.path.join(args.out_dir, f"{args.phase}.json")
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src")))
    ctx = None if args.phase == "selftest" else Ctx()
    log(f"phase {args.phase}")
    {"g4": phase_g4, "ew": phase_ew, "g2": phase_g2, "g3": phase_g3,
     "g5num": phase_g5num, "delay": phase_delay, "selftest": phase_selftest,
     "pm": phase_pm}[args.phase](ctx)
    # A failed check fails the phase, whatever else ran.
    sys.exit(1 if check_summary(RECORDS, f"MA_CHECKS {args.phase}") else 0)


if __name__ == "__main__":
    main()
