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
"""Single-layer EP routed-expert prototype for wide decode batches (V2 of the AgentX plan).

One rank's share of one MoE layer: NE local experts, each with the megakernel's tiling
(w13 as T13 tiles of [TK13, 2 * IR] fp8, w2 as T2 tiles of [TK2, H] fp8, 1 MiB per tile),
streamed once through a 12-slot VMEM ring while the previous expert computes. Three modes
share the ring and the loop and differ only in how rows meet experts:

- "stream": DMA only. The achievable HBM read rate of the ring.
- "allrows": every listed expert is applied to all MB rows and masked by its routing weight
  column (the megakernel's `MoEBlock.expert` formulation). MXU work grows with MB x NE.
- "gather": each expert's rows are gathered with a one-hot matmul into a CAP-row tile,
  computed, weighted, and scattered back with the transposed one-hot. MXU work grows with
  the number of routed (row, expert) pairs, not with MB x NE.

The kernel is a measurement vehicle, not a drop-in block: routing is an input (row lists
per expert), there is no shared expert, no router, no collective, and no software
pipelining across experts.

Phase 3 harness copy of agentx-p1-v2-moewide src/vllm_torchtpu/kernels/qwen35_megakernel/
proto_moe_wide.py, placed next to tests/kernels/test_p3_hbm_bw.py (the G1 HBM read
bandwidth bench) so that the kernel package of the target tree is not modified; only the
config import path changed. The megakernel config of HEAD 25ab0a7ee is identical to the
one this kernel was written against.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.qwen35_megakernel.config import (  # noqa: E402
    BF16, F32, FP8, I32, NN, rnd, rup)

MODES = ("stream", "allrows", "gather")


@dataclasses.dataclass(frozen=True)
class ProtoCfg:
    H: int = 4096
    IR: int = 1024  # routed-expert intermediate width per rank (EP: the whole expert)
    NE: int = 64  # listed (hit) experts of this rank
    MB: int = 80  # rows in the step
    CAP: int = 32  # gathered rows per expert (gather mode)
    TK13: int = 512  # [TK13, 2 * IR] fp8 = 1 MiB
    TK2: int = 256  # [TK2, H] fp8 = 1 MiB
    mode: str = "allrows"
    depth: int = 1  # experts in flight in the ring (depth x 12 MiB of VMEM)
    whole: bool = False  # stream mode: one DMA per w13 / w2 expert block instead of 1 MiB tiles
    reps: int = 1  # stream mode: passes over the NE experts inside one call
    queues: int = 1  # stream mode: 2 alternates the tiles between DMA priority 0 and 1
    scope: bool = False  # wrap the expert loop in a named scope (an in-kernel trace region)
    tag: str = ""  # appended to the kernel name so a trace tells configurations apart
    vmem_mib: int = 62

    @property
    def T13(self):
        return self.H // self.TK13

    @property
    def T2(self):
        return self.IR // self.TK2

    @property
    def expert_bytes(self):
        return self.T13 * self.TK13 * 2 * self.IR + self.T2 * self.TK2 * self.H

    @property
    def weight_bytes(self):
        return self.NE * self.expert_bytes

    @property
    def read_bytes(self):
        """HBM bytes one call reads: weight_bytes per pass."""
        return self.reps * self.weight_bytes

    def check(self):
        assert self.mode in MODES, self.mode
        assert self.H % self.TK13 == 0 and self.IR % self.TK2 == 0
        assert self.TK13 * 2 * self.IR <= 4 << 20 and self.TK2 * self.H <= 4 << 20  # tile bytes
        assert self.depth >= 1 and self.NE >= self.depth
        assert not self.whole or self.mode == "stream", "whole-expert DMAs are a stream-mode probe"
        assert self.reps >= 1 and (self.reps == 1 or (self.mode == "stream" and not self.whole))
        assert self.queues in (1, 2) and (self.queues == 1 or (self.mode == "stream" and not self.whole))
        assert self.MB % 8 == 0 and self.CAP % 16 == 0 and self.H % 128 == 0 and self.IR % 128 == 0


def _kernel(c: ProtoCfg, interpret: bool):
    H, IR, NE, MB, CAP, T13, T2, TK13, TK2, D = (
        c.H, c.IR, c.NE, c.MB, c.CAP, c.T13, c.T2, c.TK13, c.TK2, c.depth)
    BN = 2 * IR
    mode = c.mode
    N = c.reps * NE  # loop iterations; iteration i reads expert i % NE

    def nxt(i):  # the expert iteration i + D refills
        return lax.rem(i + D, NE) if c.reps > 1 else i + D

    def p13(t):  # DMA priority of w13 tile t (w2 tile t continues the alternation)
        return t % 2 if c.queues == 2 else 0

    def p2(t):
        return (T13 + t) % 2 if c.queues == 2 else 0

    def mm(x, w):
        if interpret and x.dtype != w.dtype:
            w = w.astype(x.dtype)
        return lax.dot_general(x, w, NN, preferred_element_type=F32)

    def kernel(x_ref, wn_ref, idx_c_ref, idx_r_ref, wt_ref, es_ref, w13_hbm, w2_hbm,
               out_ref, r13, r2, sem13, sem2, acc13, acc, xg, mid):
        def bank(e):
            return e % D if D > 1 else 0

        def cp13(e, t, b):
            return pltpu.make_async_copy(w13_hbm.at[e, t], r13.at[b, t], sem13.at[b, t])

        def cp2(e, t, b):
            return pltpu.make_async_copy(w2_hbm.at[e, t], r2.at[b, t], sem2.at[b, t])

        def cp13w(e, b):  # whole w13 block of one expert: T13 tiles in one DMA
            return pltpu.make_async_copy(w13_hbm.at[e], r13.at[b], sem13.at[b, 0])

        def cp2w(e, b):
            return pltpu.make_async_copy(w2_hbm.at[e], r2.at[b], sem2.at[b, 0])

        if c.whole:
            for d in range(D):
                cp13w(d, d).start()
                cp2w(d, d).start()
            out_ref[...] = jnp.zeros((MB, H), F32)

            def body_whole(e, carry):
                b = bank(e)
                cp13w(0, b).wait()
                cp2w(0, b).wait()

                @pl.when(e + D < NE)
                def _():
                    cp13w(e + D, b).start()
                    cp2w(e + D, b).start()

                return carry

            lax.fori_loop(0, NE, body_whole, 0)
            return

        for d in range(D):
            for t in range(T13):
                cp13(d % NE, t, d).start(priority=p13(t))
            for t in range(T2):
                cp2(d % NE, t, d).start(priority=p2(t))
        out_ref[...] = jnp.zeros((MB, H), F32)

        def refill13(i, t, b):
            @pl.when(i + D < N)
            def _():
                cp13(nxt(i), t, b).start(priority=p13(t))

        def refill2(i, t, b):
            @pl.when(i + D < N)
            def _():
                cp2(nxt(i), t, b).start(priority=p2(t))

        def body(e, carry):
            b = bank(e)
            if mode == "stream":
                for t in range(T13):
                    cp13(0, t, b).wait()
                    refill13(e, t, b)
                for t in range(T2):
                    cp2(0, t, b).wait()
                    refill2(e, t, b)
                return carry

            if mode == "gather":
                # one-hot [CAP, MB]: slot i of expert e selects row idx[e, i] (MB = padding)
                idx_c = idx_c_ref[e][:, 0:1]  # [CAP, 1]
                oh = (idx_c == lax.broadcasted_iota(I32, (CAP, MB), 1)).astype(BF16)
                xg[...] = mm(oh, x_ref[...]).astype(BF16)
                xin = xg
            else:
                xin = x_ref

            for t in range(T13):
                cp13(0, t, b).wait()
                m13 = mm(xin[:, t * TK13:(t + 1) * TK13], r13[b, t])
                if t == 0:
                    acc13[...] = m13
                else:
                    acc13[...] += m13
                refill13(e, t, b)
            es = es_ref[e]  # [1, 2 * IR + H]
            h = acc13[...] * es[:, :BN]
            mid[...] = rnd(rnd(jax.nn.silu(rnd(h[:, :IR]))) * rnd(h[:, IR:])).astype(BF16)
            for t in range(T2):
                cp2(0, t, b).wait()
                m2 = mm(mid[:, t * TK2:(t + 1) * TK2], r2[b, t])
                if t == 0:
                    acc[...] = m2
                else:
                    acc[...] += m2
                refill2(e, t, b)
            y = rnd(acc[...] * es[:, BN:])  # [rows, H] bf16-rounded f32
            if mode == "gather":
                yw = (y * wt_ref[e][:, 0:1]).astype(BF16)  # per-slot routing weight
                idx_r = idx_r_ref[e][0:1, :]  # [1, CAP]
                oht = (lax.broadcasted_iota(I32, (MB, CAP), 0) == idx_r).astype(BF16)
                out_ref[...] += mm(oht, yw)
            else:
                lane = lax.broadcasted_iota(I32, (MB, 128), 1)
                wcol = jnp.sum(jnp.where(lane == e, wn_ref[...], 0.0), axis=1, keepdims=True)
                out_ref[...] += wcol * y
            return carry

        with jax.named_scope("proto_ring") if c.scope else contextlib.nullcontext():
            lax.fori_loop(0, N, body, 0)

    return kernel


def build(c: ProtoCfg, interpret: bool = False):
    """pallas_call for one rank's routed experts of one layer; see the module docstring."""
    c.check()
    rows = c.CAP if c.mode == "gather" else c.MB
    scratch = [
        pltpu.VMEM((c.depth, c.T13, c.TK13, 2 * c.IR), FP8),
        pltpu.VMEM((c.depth, c.T2, c.TK2, c.H), FP8),
        pltpu.SemaphoreType.DMA((c.depth, c.T13)),
        pltpu.SemaphoreType.DMA((c.depth, c.T2)),
        pltpu.VMEM((rows, 2 * c.IR), F32),
        pltpu.VMEM((rows, c.H), F32),
        pltpu.VMEM((rows, c.H), BF16),
        pltpu.VMEM((rows, c.IR), BF16),
    ]
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    return pl.pallas_call(
        _kernel(c, interpret),
        out_shape=jax.ShapeDtypeStruct((c.MB, c.H), F32),
        in_specs=[vmem, vmem, vmem, vmem, vmem, vmem, hbm, hbm],
        out_specs=vmem,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=c.vmem_mib << 20),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=(f"proto_moe_wide_{c.mode}_mb{c.MB}_ne{c.NE}_cap{c.CAP}_d{c.depth}"
              f"{'_whole' if c.whole else ''}{f'_r{c.reps}' if c.reps > 1 else ''}"
              f"{'_q2' if c.queues == 2 else ''}{'_scope' if c.scope else ''}"
              f"{'_' + c.tag if c.tag else ''}"),
    )


def build_hbm_copy(src_shape, dtype, depth: int = 2, reps: int = 1, tag: str = ""):
    """pallas_call copying src to a new HBM array block by block with DMAs only.

    Block j is src[j] (the leading axis); depth blocks are in flight; reps passes. HBM traffic
    per call is reps x (read + write) of the whole array.
    """
    nb = src_shape[0]
    n = reps * nb
    assert 1 <= depth <= nb

    def kernel(src, dst, sem):
        def cp(j, b):
            return pltpu.make_async_copy(src.at[j], dst.at[j], sem.at[b])

        for d in range(depth):
            cp(d, d).start()

        def body(i, carry):
            b = lax.rem(i, depth)
            cp(0, b).wait()

            @pl.when(i + depth < n)
            def _():
                cp(lax.rem(i + depth, nb), b).start()

            return carry

        lax.fori_loop(0, n, body, 0)

    hbm = pl.BlockSpec(memory_space=pl.ANY)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(tuple(src_shape), dtype),
        in_specs=[hbm],
        out_specs=hbm,
        scratch_shapes=[pltpu.SemaphoreType.DMA((depth,))],
        name=f"proto_hbm_copy_nb{nb}_d{depth}{f'_r{reps}' if reps > 1 else ''}{'_' + tag if tag else ''}",
    )


def route_uniform(c: ProtoCfg, seed: int, topk: int = 10, n_experts: int = 512):
    """Uniform random top-k routing of MB rows over n_experts, kept for this rank's NE experts.

    Returns (wn [MB, 128] f32 routing weight per local expert lane, idx_c [NE, CAP, 128] i32,
    idx_r [NE, 8, CAP] i32, wt [NE, CAP, 128] f32, max_rows_per_expert). Slots past an
    expert's row count hold MB (never matched by a row iota) and weight 0.
    """
    rng = np.random.default_rng(seed)
    wn = np.zeros((c.MB, 128), np.float32)
    rows = [[] for _ in range(c.NE)]
    for r in range(c.MB):
        picks = rng.choice(n_experts, size=topk, replace=False)
        w = rng.random(topk).astype(np.float32)
        w /= w.sum()
        for p, wv in zip(picks, w):
            if p < c.NE:  # this rank owns experts 0..NE-1 of the global table
                wn[r, p] = wv
                rows[p].append(r)
    mx = max(len(rr) for rr in rows)
    assert c.mode != "gather" or mx <= c.CAP, (mx, c.CAP)
    idx = np.full((c.NE, c.CAP), c.MB, np.int32)
    wt = np.zeros((c.NE, c.CAP), np.float32)
    for e, rr in enumerate(rows):
        rr = rr[:c.CAP]  # only gather mode reads the slots, and it asserts mx <= CAP above
        idx[e, :len(rr)] = rr
        wt[e, :len(rr)] = wn[rr, e]
    idx_c = np.broadcast_to(idx[:, :, None], (c.NE, c.CAP, 128)).copy()
    idx_r = np.broadcast_to(idx[:, None, :], (c.NE, 8, c.CAP)).copy()
    wt_c = np.broadcast_to(wt[:, :, None], (c.NE, c.CAP, 128)).copy()
    return wn, idx_c, idx_r, wt_c, mx


def make_weights(c: ProtoCfg, seed: int):
    rng = np.random.default_rng(seed)
    w13 = rng.standard_normal((c.NE, c.T13, c.TK13, 2 * c.IR), np.float32) * 0.02
    w2 = rng.standard_normal((c.NE, c.T2, c.TK2, c.H), np.float32) * 0.02
    es = np.concatenate(
        [rng.uniform(0.5, 1.5, (c.NE, 1, 2 * c.IR)), rng.uniform(0.5, 1.5, (c.NE, 1, c.H))],
        axis=-1).astype(np.float32)
    x = rng.standard_normal((c.MB, c.H), np.float32)
    return x, w13, w2, es


def reference(c: ProtoCfg, x, wn, w13, w2, es):
    """Pure-JAX f32 reference of the routed-expert sum (same bf16 rounding points)."""
    xb = jnp.asarray(x, BF16)
    w13f = jnp.asarray(w13, FP8).astype(F32).reshape(c.NE, c.H, 2 * c.IR)
    w2f = jnp.asarray(w2, FP8).astype(F32).reshape(c.NE, c.IR, c.H)
    out = jnp.zeros((c.MB, c.H), F32)
    for e in range(c.NE):
        h = xb.astype(F32) @ w13f[e] * es[e, :, :2 * c.IR]
        mid = rnd(rnd(jax.nn.silu(rnd(h[:, :c.IR]))) * rnd(h[:, c.IR:])).astype(BF16)
        y = rnd(mid.astype(F32) @ w2f[e] * es[e, :, 2 * c.IR:])
        out = out + wn[:, e:e + 1] * y
    return out


def run(c: ProtoCfg, interpret: bool = False, seed: int = 0, device=None):
    """Build, run once and return (out, inputs) for c."""
    x, w13, w2, es = make_weights(c, seed)
    wn, idx_c, idx_r, wt, mx = route_uniform(c, seed)
    put = (lambda a: jax.device_put(a, device)) if device is not None else jnp.asarray
    args = (
        put(jnp.asarray(x, BF16)), put(jnp.asarray(wn)), put(jnp.asarray(idx_c)),
        put(jnp.asarray(idx_r)), put(jnp.asarray(wt)), put(jnp.asarray(es)),
        put(jnp.asarray(w13, FP8)), put(jnp.asarray(w2, FP8)),
    )
    fn = jax.jit(build(c, interpret))
    out = fn(*args)
    return out, fn, args, dict(x=x, wn=wn, w13=w13, w2=w2, es=es, max_rows=mx)


def model(c: ProtoCfg, hbm_tbs: float = 3.69):
    """Byte floor and MXU work of one call: bytes, floor_us at hbm_tbs TB/s, flop."""
    expert_flop = 2 * (c.H * 2 * c.IR + c.IR * c.H)
    if c.mode == "stream":
        flop = 0
    elif c.mode == "allrows":
        flop = c.NE * c.MB * expert_flop
    else:
        flop = c.NE * (c.CAP * expert_flop + 2 * 2 * c.CAP * c.MB * c.H)
    return dict(bytes=c.read_bytes, floor_us=c.read_bytes / (hbm_tbs * 1e12) * 1e6, flop=flop)


__all__ = ["ProtoCfg", "MODES", "build", "build_hbm_copy", "run", "reference", "route_uniform",
           "model", "make_weights", "rup", "functools"]
