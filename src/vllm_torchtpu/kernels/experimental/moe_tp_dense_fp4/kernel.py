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
"""TP-sharded small-T decode MoE, dense over tokens, sparse over hit experts (own design).

Every core holds a 1/TP slice of every expert: w1 [E, K, 2Fs] (gate slice | up slice), w2 [E, Fs, N], fp4 with
128-row block scales. All T tokens are on every core (x [Mp, K]). For each HIT expert (scalar-prefetched list) the
kernel DMAs that expert's whole slice (w1 + w2 + scales, ~3.5 MB) through a ring of VMEM slots, computes
h = act(x @ w1) for all Mp rows, scales row m by the routing weight gw[e, m] (0 if token m is not routed to e),
and accumulates h @ w2 into one f32 [Mp, N] VMEM accumulator. One output write per call; no permute, no entries.
"""
import functools
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

QB = 128


def _kernel(hit_ref, nhit_ref, x_ref, gw_hbm, w1_hbm, s1_hbm, w2_hbm, s2_hbm, out_ref,
            w1v, s1v, w2v, s2v, gwv, acc, sem, *, slots, fs):
    k = x_ref.shape[1]
    n = w2_hbm.shape[2]
    nh = nhit_ref[0]
    acc[...] = jnp.zeros_like(acc)

    def copies(t):
        slot = t % slots
        e = hit_ref[t]
        return (pltpu.make_async_copy(w1_hbm.at[e], w1v.at[slot], sem.at[slot, 0]),
                pltpu.make_async_copy(s1_hbm.at[e], s1v.at[slot], sem.at[slot, 1]),
                pltpu.make_async_copy(w2_hbm.at[e], w2v.at[slot], sem.at[slot, 2]),
                pltpu.make_async_copy(s2_hbm.at[e], s2v.at[slot], sem.at[slot, 3]),
                pltpu.make_async_copy(gw_hbm.at[e], gwv.at[slot], sem.at[slot, 4]))

    for i in range(slots - 1):
        @pl.when(i < nh)
        def _():
            for c in copies(i):
                c.start()

    x = x_ref[...]

    def body(t, carry):
        @pl.when(t + slots - 1 < nh)
        def _():
            for c in copies(t + slots - 1):
                c.start()
        for c in copies(t):
            c.wait()
        slot = t % slots
        w1 = w1v.at[slot]
        s1 = s1v.at[slot]
        g = jnp.zeros((x.shape[0], 2 * fs), jnp.float32)
        for kb in range(k // QB):
            blk = jnp.matmul(x[:, kb * QB:(kb + 1) * QB], w1[kb * QB:(kb + 1) * QB, :].astype(jnp.float8_e4m3fn),
                             preferred_element_type=jnp.float32)
            g = g + blk * s1[kb:kb + 1, :]
        h = jax.nn.silu(g[:, :fs]) * g[:, fs:]
        h = (h * gwv[slot][:, 0:1]).astype(x.dtype)
        w2 = w2v.at[slot]
        s2 = s2v.at[slot]
        for kb in range(fs // QB):
            blk = jnp.matmul(h[:, kb * QB:(kb + 1) * QB], w2[kb * QB:(kb + 1) * QB, :].astype(jnp.float8_e4m3fn),
                             preferred_element_type=jnp.float32)
            acc[...] += blk * s2[kb:kb + 1, :]
        return carry

    lax.fori_loop(0, nh, body, 0)
    out_ref[...] = acc[...].astype(out_ref.dtype)


def tp_dense_moe(x, gw, w1, s1, w2, s2, hit, nhit, *, slots=3, interpret=False):
    """x [Mp,K] bf16; gw [E,Mp,128] f32 (lane-replicated); w1 [E,K,2Fs] fp4; s1 [E,K/128,2Fs] f32;
    w2 [E,Fs,N] fp4; s2 [E,Fs/128,N] f32; hit int32[E] (first nhit valid); nhit int32[1]."""
    mp, k = x.shape
    e, _, f2 = w1.shape
    fs = f2 // 2
    n = w2.shape[2]
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vm = pl.BlockSpec(memory_space=pltpu.VMEM)
    kern = functools.partial(_kernel, slots=slots, fs=fs)
    extra = {"interpret": pltpu.InterpretParams()} if interpret else {}
    return pl.pallas_call(
        kern, **extra,
        out_shape=jax.ShapeDtypeStruct((mp, n), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[vm, hbm, hbm, hbm, hbm, hbm],
            out_specs=vm,
            scratch_shapes=[
                pltpu.VMEM((slots, k, f2), w1.dtype),
                pltpu.VMEM((slots, k // QB, f2), jnp.float32),
                pltpu.VMEM((slots, fs, n), w2.dtype),
                pltpu.VMEM((slots, fs // QB, n), jnp.float32),
                pltpu.VMEM((slots, mp, 128), jnp.float32),
                pltpu.VMEM((mp, n), jnp.float32),
                pltpu.SemaphoreType.DMA((slots, 5)),
            ]),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=48 << 20),
        name=f"tp_dense_moe-e{e}-m{mp}-fs{fs}",
    )(hit, nhit, x, gw, w1, s1, w2, s2)


# Packed layout: one fp4 record + one f32 scale record per expert (2 weight DMAs instead of 4).
# W[e] [K + (N/2Fs)*Fs, 2Fs]: rows [0,K) = w1 (gate|up slice), then chunk j of w2 (columns j*2Fs..) as Fs rows.
# S[e] [K/128 + (N/2Fs)*(Fs/128), 2Fs] f32: s1 rows, then the s2 rows of chunk j.


def pack(w1, s1, w2, s2):
    """w1 [E,K,2Fs] fp4, s1 [E,K/128,2Fs], w2 [E,Fs,N] fp4, s2 [E,Fs/128,N] -> (W, S)."""
    e, k, f2 = w1.shape
    fs, n = w2.shape[1], w2.shape[2]
    nch = n // f2
    w2r = w2.reshape(e, fs, nch, f2).transpose(0, 2, 1, 3).reshape(e, nch * fs, f2)
    s2r = s2.reshape(e, fs // QB, nch, f2).transpose(0, 2, 1, 3).reshape(e, nch * (fs // QB), f2)
    return jnp.concatenate([w1, w2r], axis=1), jnp.concatenate([s1, s2r], axis=1)


def _kernel_packed(hit_ref, nhit_ref, x_ref, gw_hbm, w_hbm, s_hbm, out_ref, wv, sv, gwv, acc, sem,
            *, slots, fs, k, nch, compute):
    f2 = 2 * fs
    nh = nhit_ref[0]
    acc[...] = jnp.zeros_like(acc)

    def copies(t):
        slot = t % slots
        e = hit_ref[t]
        return (pltpu.make_async_copy(w_hbm.at[e], wv.at[slot], sem.at[slot, 0]),
                pltpu.make_async_copy(s_hbm.at[e], sv.at[slot], sem.at[slot, 1]),
                pltpu.make_async_copy(gw_hbm.at[e], gwv.at[slot], sem.at[slot, 2]))

    for i in range(slots - 1):
        @pl.when(i < nh)
        def _():
            for c in copies(i):
                c.start()

    x = x_ref[...]

    def body(t, carry):
        @pl.when(t + slots - 1 < nh)
        def _():
            for c in copies(t + slots - 1):
                c.start()
        for c in copies(t):
            c.wait()
        if not compute:
            return carry
        slot = t % slots
        w = wv.at[slot]
        s = sv.at[slot]
        g = jnp.zeros((x.shape[0], f2), jnp.float32)
        for kb in range(k // QB):
            blk = jnp.matmul(x[:, kb * QB:(kb + 1) * QB], w[kb * QB:(kb + 1) * QB, :].astype(jnp.float8_e4m3fn),
                             preferred_element_type=jnp.float32)
            g = g + blk * s[kb:kb + 1, :]
        h = jax.nn.silu(g[:, :fs]) * g[:, fs:]
        h = (h * gwv[slot][:, 0:1]).astype(x.dtype)
        nkb = fs // QB
        for j in range(nch):
            part = jnp.zeros((x.shape[0], f2), jnp.float32)
            for kb in range(nkb):
                r0 = k + j * fs + kb * QB
                blk = jnp.matmul(h[:, kb * QB:(kb + 1) * QB], w[r0:r0 + QB, :].astype(jnp.float8_e4m3fn),
                                 preferred_element_type=jnp.float32)
                srow = k // QB + j * nkb + kb
                part = part + blk * s[srow:srow + 1, :]
            acc[:, j * f2:(j + 1) * f2] += part
        return carry

    lax.fori_loop(0, nh, body, 0)
    out_ref[...] = acc[...].astype(out_ref.dtype)


def tp_dense_moe_packed(x, gw, W, S, hit, nhit, *, fs, slots=3, compute=True, interpret=False):
    mp, k = x.shape
    e, rows, f2 = W.shape
    nch = (rows - k) // fs
    n = nch * f2
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vm = pl.BlockSpec(memory_space=pltpu.VMEM)
    kern = functools.partial(_kernel_packed, slots=slots, fs=fs, k=k, nch=nch, compute=compute)
    extra = {"interpret": pltpu.InterpretParams()} if interpret else {}
    return pl.pallas_call(
        kern, **extra,
        out_shape=jax.ShapeDtypeStruct((mp, n), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[vm, hbm, hbm, hbm],
            out_specs=vm,
            scratch_shapes=[
                pltpu.VMEM((slots, rows, f2), W.dtype),
                pltpu.VMEM((slots, S.shape[1], f2), jnp.float32),
                pltpu.VMEM((slots, mp, 128), jnp.float32),
                pltpu.VMEM((mp, n), jnp.float32),
                pltpu.SemaphoreType.DMA((slots, 3)),
            ]),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=48 << 20),
        name=f"tp_dense_moe_packed-e{e}-m{mp}-fs{fs}",
    )(hit, nhit, x, gw, W, S)


def reference(x, gw, w1, s1, w2, s2, hit, nhit):
    fs = w1.shape[2] // 2
    out = jnp.zeros((x.shape[0], w2.shape[2]), jnp.float32)
    for t in range(int(nhit[0])):
        e = int(hit[t])
        w1d = (w1[e].astype(jnp.float32).reshape(-1, QB, 2 * fs) * s1[e][:, None, :]).reshape(-1, 2 * fs)
        w2d = (w2[e].astype(jnp.float32).reshape(-1, QB, w2.shape[2]) * s2[e][:, None, :]).reshape(fs, -1)
        g = x.astype(jnp.float32) @ w1d
        h = (jax.nn.silu(g[:, :fs]) * g[:, fs:]) * gw[e][:, 0:1]
        out = out + h.astype(jnp.bfloat16).astype(jnp.float32) @ w2d
    return out
