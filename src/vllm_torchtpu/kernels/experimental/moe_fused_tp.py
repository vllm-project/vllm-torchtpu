# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Fused MoE for tensor-parallel experts at decode-sized token counts.

Under MoE tensor parallelism every rank holds all experts with the
intermediate dimension sharded, so each expert's local slice is small (for
Qwen3.5-397B at TP8: i = 128, 1.5 MiB of FP8 per expert) and rows per expert
at decode are ~1-3. The GMM path pays a fixed cost per active expert in each
of its two kernels plus a permute/sort/combine around them.

This kernel visits each active expert once. Its w13/w2 slices stream from HBM
through a ring of `num_buffers` VMEM slots while the previous expert computes
on ALL tokens in their original order; the route weight zeroes rows the
expert does not own. The first matmul of expert j+1 is issued in step j, ahead
of expert j's elementwise chain and second matmul, so the MXU stays busy while
the VPU quantizes. Nothing is permuted and no intermediate goes to HBM.

The same call also runs the layer's shared expert slice and the TP
all-reduce (see fused_moe_tp_ar), so one kernel replaces the whole MoE block
after the router.

Numerics: hidden states are quantized to FP8 once per call with one scale per
row; the activated intermediate is quantized per row per expert; both
matmuls are FP8 x FP8 with FP32 accumulation into an FP32 VMEM accumulator.
Router top-k stays with the caller.

Derived from moe_dense_expert.py, which targets i % 512 == 0 (EP layouts).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_F8 = jnp.float8_e4m3fn
_F8_MAX = 448.0
# Output columns per second-matmul chunk; bounds the live f32 result.
_N_CHUNK = 512
# Weight ring depth. One slot holds the expert whose first matmul is in
# flight, so num_buffers - 2 transfers run ahead of compute.
_NUM_BUFFERS = 6


def can_use_fused_tp(hidden, w1, w2, s1, s2, b1, b2, *, activation, rhs_quant_dtype):
    """Arithmetic/layout support only; the token cap is applied by the caller."""
    if activation != "silu" or rhs_quant_dtype not in (None, _F8):
        return False
    if b1 is not None or b2 is not None or s1 is None or s2 is None:
        return False
    if hidden.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        return False
    _, h = hidden.shape
    e, i, out = w2.shape
    if min(h, i, e) <= 0 or i % 128 or h % _N_CHUNK or out != h:
        return False
    if w1.shape != (e, h, 2 * i):
        return False
    if hidden.dtype != jnp.bfloat16 or w1.dtype != _F8 or w2.dtype != _F8:
        return False
    return (
        s1.shape == (e, 1, 1, 2 * i)
        and s2.shape == (e, 1, 1, h)
        and s1.dtype == jnp.float32
        and s2.dtype == jnp.float32
    )


# --- Shared expert + in-kernel TP all-reduce -------------------------------
#
# The expert loop, plus the layer's shared expert (its TP slice, gated by
# sigmoid(x . w_g)) and the TP all-reduce, in one call per rank under
# shard_map. The all-reduce is latency bound at decode sizes, so it runs once
# at the end; the barrier it needs is signalled at entry and waited only
# before the reduce-scatter, so the expert loop hides it. bf16 on the wire,
# FP32 sum of the slots.

_MESH = pl.DeviceIdType.MESH


def _ar_kernel(
    active_ref,
    rank_ref,
    x_ref,
    coeff_ref,
    sgate_ref,
    w1_hbm,
    w2_hbm,
    s1_hbm,
    s2_hbm,
    w1s_hbm,
    s1s_hbm,
    w2s_hbm,
    s2s_hbm,
    out_ref,
    w1_ref,
    w2_ref,
    s1_ref,
    s2_ref,
    w1s_v,
    s1s_v,
    w2s_v,
    s2s_v,
    xq_ref,
    xs_ref,
    owner_ref,
    own_bf,
    recv,
    red,
    ids_ref,
    sem_ref,
    ssem,
    rs_send,
    rs_recv,
    ag_send,
    ag_recv,
    *,
    m,
    h,
    i,
    experts,
    nb,
    n_ranks,
):
    me = rank_ref[0]
    chunk = h // n_ranks
    barrier = pltpu.get_barrier_semaphore()
    for d in range(n_ranks):
        pl.semaphore_signal(
            barrier, inc=1, device_id=(jnp.int32(d),), device_id_type=_MESH
        )

    shared_copies = [
        pltpu.make_async_copy(w1s_hbm, w1s_v, ssem.at[0]),
        pltpu.make_async_copy(w2s_hbm, w2s_v, ssem.at[1]),
        pltpu.make_async_copy(s1s_hbm, s1s_v, ssem.at[2]),
        pltpu.make_async_copy(s2s_hbm, s2s_v, ssem.at[3]),
    ]
    for copy in shared_copies:
        copy.start()

    owner_ref[...] = jnp.zeros((m, h), jnp.float32)
    x = x_ref[...].astype(jnp.float32)
    xsc = jnp.max(jnp.abs(x), axis=1, keepdims=True) / _F8_MAX
    xq_ref[...] = (x * jnp.where(xsc == 0, 0.0, 1.0 / xsc)).astype(_F8)
    xs_ref[...] = jnp.broadcast_to(xsc, (m, 2 * i))
    cols = lax.broadcasted_iota(jnp.int32, coeff_ref.shape, 1)

    def compact(e, count):
        @pl.when(active_ref[e] != 0)
        def record():
            ids_ref[count] = e

        return count + (active_ref[e] != 0).astype(jnp.int32)

    count = lax.fori_loop(0, experts, compact, jnp.int32(0))

    def transfers(step):
        e, slot = ids_ref[step], step % nb
        return [
            pltpu.make_async_copy(w1_hbm.at[e], w1_ref.at[slot], sem_ref.at[slot, 0]),
            pltpu.make_async_copy(w2_hbm.at[e], w2_ref.at[slot], sem_ref.at[slot, 1]),
            pltpu.make_async_copy(
                s1_hbm.at[e, 0], s1_ref.at[slot], sem_ref.at[slot, 2]
            ),
            pltpu.make_async_copy(
                s2_hbm.at[e, 0], s2_ref.at[slot], sem_ref.at[slot, 3]
            ),
        ]

    def accumulate(acc1, coeff, s1, w2, s2):
        acc1 = acc1 * xs_ref[...] * s1
        mid = jnp.where(coeff != 0, jax.nn.silu(acc1[:, :i]) * acc1[:, i:] * coeff, 0.0)
        ms = jnp.max(jnp.abs(mid), axis=1, keepdims=True) / _F8_MAX
        mq = (mid * jnp.where(ms == 0, 0.0, 1.0 / ms)).astype(_F8)
        for n in range(0, h, _N_CHUNK):
            acc2 = jnp.matmul(
                mq, w2[:, n : n + _N_CHUNK], preferred_element_type=jnp.float32
            )
            owner_ref[:, n : n + _N_CHUNK] += acc2 * s2[:, n : n + _N_CHUNK] * ms

    @pl.when(count > 0)
    def prime_routed():
        for initial in range(nb):

            @pl.when(count > initial)
            def prime(*, initial=initial):
                for copy in transfers(jnp.int32(initial)):
                    copy.start()

    # Shared expert first; its slabs were issued at entry with the routed primes.
    for copy in shared_copies:
        copy.wait()
    shared_coeff = jnp.max(sgate_ref[...], axis=1, keepdims=True)
    accumulate(
        jnp.matmul(xq_ref[...], w1s_v[...], preferred_element_type=jnp.float32),
        shared_coeff,
        s1s_v[...],
        w2s_v,
        s2s_v[...],
    )

    @pl.when(count > 0)
    def run_routed():
        for copy in transfers(jnp.int32(0)):
            copy.wait()

        def expert_body(step, acc1):
            nxt = step + 1

            @pl.when(nxt < count)
            def wait_next():
                for copy in transfers(nxt):
                    copy.wait()

            acc1_next = jnp.matmul(
                xq_ref[...], w1_ref[nxt % nb], preferred_element_type=jnp.float32
            )
            slot = step % nb
            e = ids_ref[step]
            coeff = jnp.sum(
                jnp.where(cols == e, coeff_ref[...], 0.0), axis=1, keepdims=True
            )
            accumulate(acc1, coeff, s1_ref[slot], w2_ref.at[slot], s2_ref[slot])

            @pl.when(step + nb < count)
            def refill():
                for copy in transfers(step + nb):
                    copy.start()

            return acc1_next

        lax.fori_loop(
            0,
            count,
            expert_body,
            jnp.matmul(xq_ref[...], w1_ref[0], preferred_element_type=jnp.float32),
        )

    # All-reduce: reduce-scatter column chunks, sum, all-gather.
    own_bf[...] = owner_ref[...].astype(own_bf.dtype)
    # Every peer has entered this call, so its receive buffer is live.
    pl.semaphore_wait(barrier, n_ranks)
    scatter = [
        pltpu.make_async_remote_copy(
            own_bf.at[:, pl.ds(d * chunk, chunk)],
            recv.at[me],
            rs_send.at[d],
            rs_recv,
            device_id=(jnp.int32(d),),
            device_id_type=_MESH,
        )
        for d in range(n_ranks)
    ]
    for copy in scatter:
        copy.start()
    pltpu.make_async_copy(recv, recv, rs_recv).wait()
    acc = recv[0].astype(jnp.float32)
    for s in range(1, n_ranks):
        acc = acc + recv[s].astype(jnp.float32)
    red[...] = acc.astype(red.dtype)
    for copy in scatter:
        copy.wait_send()
    col = pl.multiple_of(me * chunk, chunk)
    gather = [
        pltpu.make_async_remote_copy(
            red,
            out_ref.at[:, pl.ds(col, chunk)],
            ag_send.at[d],
            ag_recv,
            device_id=(jnp.int32(d),),
            device_id_type=_MESH,
        )
        for d in range(n_ranks)
    ]
    for copy in gather:
        copy.start()
    pltpu.make_async_copy(out_ref, out_ref, ag_recv).wait()
    for copy in gather:
        copy.wait_send()


def can_use_fused_tp_ar(hidden, w1, w2, s1, s2, w1s, s1s, w2s, s2s, wg, *, n_ranks):
    """Layout support for the shared-expert + all-reduce variant."""
    if not can_use_fused_tp(
        hidden, w1, w2, s1, s2, None, None, activation="silu", rhs_quant_dtype=None
    ):
        return False
    _, h = hidden.shape
    _, i, _ = w2.shape
    return (
        n_ranks > 1
        and h % (n_ranks * 128) == 0
        and w1s.shape == (h, 2 * i)
        and w2s.shape == (i, h)
        and s1s.shape == (1, 2 * i)
        and s2s.shape == (1, h)
        and w1s.dtype == _F8
        and w2s.dtype == _F8
        and s1s.dtype == jnp.float32
        and s2s.dtype == jnp.float32
        and wg.shape == (h, 1)
    )


def fused_moe_tp_ar(
    rank,
    hidden,
    w1,
    w2,
    s1,
    s2,
    w1s,
    s1s,
    w2s,
    s2s,
    wg,
    topk_weights,
    topk_ids,
    *,
    n_ranks,
    num_buffers=_NUM_BUFFERS,
):
    """Per-rank body; call inside shard_map over the TP mesh.

    Returns the TP-all-reduced (shared + routed) MoE output, BF16 [tokens,
    hidden], identical on every rank. `rank` is this rank's mesh index as an
    int32 array of one element (partition-id cannot be exported through
    torch_tpu). w1s [H, 2i] / w2s [i, H] FP8 with s1s [1, 2i] / s2s [1, H]
    FP32 are this rank's shared-expert slice; wg [H, 1] is the replicated
    shared_expert_gate.
    """
    tokens, h = hidden.shape
    experts, i, _ = w2.shape
    pad = (-tokens) % 16
    if pad:
        hidden = jnp.pad(hidden, ((0, pad), (0, 0)))
        topk_weights = jnp.pad(topk_weights, ((0, pad), (0, 0)))
        topk_ids = jnp.pad(topk_ids, ((0, pad), (0, 0)))
    m = tokens + pad
    expert_ids = jnp.arange(experts, dtype=jnp.int32)[None, :]
    weights = topk_weights.astype(hidden.dtype).astype(jnp.float32)
    coeff = jnp.zeros((m, experts), jnp.float32)
    for route in range(topk_ids.shape[1]):
        selected = topk_ids[:, route : route + 1] == expert_ids
        coeff = coeff + selected.astype(jnp.float32) * weights[:, route : route + 1]
    active = jnp.any(coeff != 0, axis=0).astype(jnp.int32)
    coeff = jnp.pad(coeff, ((0, 0), (0, (-experts) % 128)))
    shared_gate = jax.nn.sigmoid(
        jnp.dot(hidden, wg.astype(hidden.dtype), preferred_element_type=jnp.float32)
    )
    shared_gate = jnp.broadcast_to(shared_gate, (m, 128))
    nb = num_buffers
    chunk = h // n_ranks
    out = pl.pallas_call(
        functools.partial(
            _ar_kernel, m=m, h=h, i=i, experts=experts, nb=nb, n_ranks=n_ranks
        ),
        out_shape=jax.ShapeDtypeStruct((m, h), hidden.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 3
            + [pl.BlockSpec(memory_space=pltpu.HBM)] * 8,
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                pltpu.VMEM((nb, h, 2 * i), _F8),
                pltpu.VMEM((nb, i, h), _F8),
                pltpu.VMEM((nb, 1, 2 * i), jnp.float32),
                pltpu.VMEM((nb, 1, h), jnp.float32),
                pltpu.VMEM((h, 2 * i), _F8),
                pltpu.VMEM((1, 2 * i), jnp.float32),
                pltpu.VMEM((i, h), _F8),
                pltpu.VMEM((1, h), jnp.float32),
                pltpu.VMEM((m, h), _F8),
                pltpu.VMEM((m, 2 * i), jnp.float32),
                pltpu.VMEM((m, h), jnp.float32),
                pltpu.VMEM((m, h), hidden.dtype),
                pltpu.VMEM((n_ranks, m, chunk), hidden.dtype),
                pltpu.VMEM((m, chunk), hidden.dtype),
                pltpu.SMEM((experts,), jnp.int32),
                pltpu.SemaphoreType.DMA((nb, 4)),
                pltpu.SemaphoreType.DMA((4,)),
                pltpu.SemaphoreType.DMA((n_ranks,)),
                pltpu.SemaphoreType.DMA(()),
                pltpu.SemaphoreType.DMA((n_ranks,)),
                pltpu.SemaphoreType.DMA(()),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=0,
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
        ),
        name=f"fused_moe_tp_ar-e_{experts}-m_{m}-h_{h}-i_{i}-n_{n_ranks}-buffers_{nb}",
    )(
        active,
        rank.reshape(1).astype(jnp.int32),
        hidden,
        coeff,
        shared_gate,
        w1,
        w2,
        s1,
        s2,
        w1s,
        s1s,
        w2s,
        s2s,
    )
    return out[:tokens] if pad else out
