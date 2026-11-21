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
"""A decode-sized SwiGLU-OAI MLP on e4m3 K-major weights in one Pallas call.

The DSpark drafter's dense MLP (``models/minimax_m3/moe.py`` MiniMaxM3MLP on
per-channel e4m3 linears) at a few rows is weight-streaming bound: the call
copies both weights HBM -> VMEM in K-row chunks, a bounded number in flight,
and contracts each chunk as it lands (bf16 rows x e4m3 tiles on the MXU, f32
accumulation in VMEM), so gate_up, the activation and down run under one
weight stream instead of two XLA matmuls plus the elementwise ops between.

Rounding follows the torch path (`SharedExpert` of moe_dense_expert_fp4): each
projection ``bf16(bf16(x @ w) * s)``, the activation per op
(`shared_activation`). Only the f32 summation order over K can differ from
XLA's dot.

With an `AllReduceSpec` the call also all-reduces the down projection over
the TP ranks (kernels/experimental/moe_tp_decode/all_reduce.py's epilogue),
its barrier hidden under the weight stream.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.moe_dense_expert_fp4 import (
    shared_activation,
)
from vllm_torchtpu.kernels.experimental.moe_tp_decode import all_reduce as ar

BF16, F32 = jnp.bfloat16, jnp.float32
# Above this many rows the weights stop being the whole cost; XLA keeps them.
MAX_ROWS = 32
# bf16 MXU operands pack 16 sublanes.
_ROW_ALIGN = 16
# Bytes per weight DMA, and how many are in flight (6: 4.5 MiB, enough to
# cover the DMA latency). The weights stream through rings of one more slot
# than that: the kernel's VMEM stays small, which leaves XLA the room it uses
# to prefetch the neighbouring ops' weights (all of VMEM to this call cost
# the drafter's other matmuls ~5 us each, c4 twin j-319ca531).
_CHUNK_BYTES = 3 << 18
_IN_FLIGHT = 6
_SLOTS = _IN_FLIGHT + 1
_VMEM_HEADROOM = 4 << 20


def _chunks(rows: int, row_bytes: int) -> list[tuple[int, int]]:
    """(start, size) K-row chunks of ~_CHUNK_BYTES, sizes multiples of 128."""
    size = max(128, (_CHUNK_BYTES // row_bytes) // 128 * 128)
    return [(k, min(size, rows - k)) for k in range(0, rows, size)]


def _in_hbm(a: jax.Array) -> jax.Array:
    """Pin a weight operand to HBM: left free, XLA prefetches the whole
    weight into VMEM just ahead of the call, and that copy stayed exposed
    (6.5 us per layer at 8 and 24 rows, c4 twin j-084c2b02) while the kernel
    streams it at the same bandwidth anyway."""
    return pltpu.with_memory_space_constraint(a, pltpu.HBM)


def _project(acc, scale):
    """bf16(bf16(acc) * scale), held in f32."""
    y = acc.astype(BF16).astype(F32) * scale.astype(F32)
    return y.astype(BF16).astype(F32)


def _kernel(meta_ref, *refs, gu_chunks, d_chunks, spec, eps):
    """meta_ref: SMEM int32 [1, 1], this device's mesh index (all-reduce).

    refs, with ``eps`` (the residual add + RMSNorm prologue) the bracketed:
      inputs  (HBM): x [m, h] bf16, s_gu [1, 2i] f32, s_d [1, h] f32,
                     w_gu [h, 2i] e4m3 ([gate | up]), w_d [i, h] e4m3,
                     [residual [m, h] bf16, norm weight [1, h] f32]
      outputs (VMEM): out [m, h] bf16, [new residual [m, h] bf16]
      scratch (VMEM): x, s_gu, s_d copies; the gate_up and down rings
                     [slots, chunk rows, n] e4m3; gu_acc [mp, 2i] f32,
                     act [mp, i] bf16, out_acc [mp, h] f32; DMA semaphores;
                     [residual, norm weight copies]; with an all-reduce
                     [mp, h] bf16 then ar.scratch_shapes.
    """
    norm = eps is not None
    k = 7 if norm else 5
    x_hbm, sgu_hbm, sd_hbm, wgu_hbm, wd_hbm = refs[:5]
    o_ref, *outs = refs[k : k + (2 if norm else 1)]
    scratch = refs[k + 1 + len(outs) :]
    x_ref, sgu_ref, sd_ref, wgu_v, wd_v, gu_acc, act_ref, out_acc, sem = scratch[:9]
    extra = scratch[9:]
    if norm:
        res_hbm, nw_hbm = refs[5:7]
        (res_out,) = outs
        res_ref, nw_ref, *ar_scratch = extra
        prologue = [(res_hbm, res_ref), (nw_hbm, nw_ref)]
    else:
        ar_scratch, prologue = extra, []
    m, h = x_ref.shape
    mp, i = act_ref.shape
    if spec is not None:
        ar.prologue(spec)
    # The rows and scales first: their latency hides under the first chunk's.
    small = [
        pltpu.make_async_copy(src, dst, sem.at[j])
        for j, (src, dst) in enumerate(
            [(x_hbm, x_ref), *prologue, (sgu_hbm, sgu_ref), (sd_hbm, sd_ref)]
        )
    ]
    x_copy, *prologue_copies, sgu_copy, sd_copy = small
    for copy in small:
        copy.start()
    # Chunk n of a weight goes to slot n % _SLOTS of its ring; it starts once
    # chunk n - _SLOTS (that slot's last tenant) has been contracted.
    copies = [
        pltpu.make_async_copy(
            src.at[pl.ds(k, c)],
            ring.at[n % _SLOTS, pl.ds(0, c)],
            sem.at[len(small) + n % _SLOTS + (_SLOTS if ring is wd_v else 0)],
        )
        for src, ring, chunks in ((wgu_hbm, wgu_v, gu_chunks), (wd_hbm, wd_v, d_chunks))
        for n, (k, c) in enumerate(chunks)
    ]
    for copy in copies[:_IN_FLIGHT]:
        copy.start()

    def landed(j):
        copies[j].wait()
        if j + _IN_FLIGHT < len(copies):
            copies[j + _IN_FLIGHT].start()

    x_copy.wait()
    x = x_ref[...].astype(F32)
    if norm:
        # fused_add_rms_norm (layers/adapter/ir_ops.py): the bf16 sum is the
        # new residual and what is normalized, in f32, times (1 + w).
        for copy in prologue_copies:
            copy.wait()
        r = (x + res_ref[...].astype(F32)).astype(BF16)
        res_out[...] = r
        r = r.astype(F32)
        var = jnp.mean(r * r, axis=1, keepdims=True)
        x = (r * lax.rsqrt(var + eps) * nw_ref[...]).astype(BF16).astype(F32)
    if mp != m:
        x = jnp.concatenate([x, jnp.zeros((mp - m, h), F32)], 0)
    x = x.astype(BF16)
    gu_acc[...] = jnp.zeros_like(gu_acc)
    for j, (k, c) in enumerate(gu_chunks):
        landed(j)
        gu_acc[...] += jnp.dot(
            x[:, k : k + c], wgu_v[j % _SLOTS, :c, :], preferred_element_type=F32
        )
    sgu_copy.wait()
    gate = _project(gu_acc[:, :i], sgu_ref[:, :i])
    up = _project(gu_acc[:, i:], sgu_ref[:, i:])
    act_ref[...] = shared_activation(gate, up).astype(BF16)
    out_acc[...] = jnp.zeros_like(out_acc)
    for n, (k, c) in enumerate(d_chunks):
        landed(len(gu_chunks) + n)
        out_acc[...] += jnp.dot(
            act_ref[:, k : k + c], wd_v[n % _SLOTS, :c, :], preferred_element_type=F32
        )
    sd_copy.wait()
    if spec is None:
        o_ref[...] = _project(out_acc[:m, :], sd_ref[...]).astype(o_ref.dtype)
    else:
        reduced, *scr = ar_scratch
        partial = _project(out_acc[...], sd_ref[...])
        ar.epilogue(spec, meta_ref[0, 0], m, partial, reduced, scr)
        o_ref[...] = reduced[:m, :]


def _vmem_bytes(mp: int, h: int, i: int, spec, info) -> int:
    t = functools.partial(ar.tiled_bytes, info=info)
    total = (
        t((_SLOTS, _chunks(h, 2 * i)[0][1], 2 * i), jnp.float8_e4m3fn)
        + t((_SLOTS, _chunks(i, h)[0][1], h), jnp.float8_e4m3fn)
        + 2 * t((mp, h), BF16)  # x, out
        + t((1, 2 * i), F32)
        + t((1, h), F32)
        + t((mp, 2 * i), F32)
        + t((mp, i), BF16)
        + t((mp, h), F32)
        # f32 temporaries: the padded rows, a chunk product, the activation.
        + t((mp, h), F32)
        + t((mp, 2 * i), F32)
        + 4 * t((mp, i), F32)
    )
    if spec is not None:
        total += t((mp, h), BF16) + ar.vmem_bytes(spec, info)
    return total


def supported(m: int, h: int, i: int, n: int = 1) -> bool:
    """Whether [m, h] rows against [h, 2i] / [i, h] weights fit the kernel
    (with n > 1: and its all-reduce over n ranks)."""
    info = pltpu.get_tpu_info()
    mp = m + (-m % _ROW_ALIGN)
    spec = None
    if n > 1:
        if not ar.supported(n, mp, h, info):
            return False
        spec = ar.make_spec(n, mp, h)
    return (
        0 < m <= MAX_ROWS
        and h % 128 == 0
        and i % 128 == 0
        and _vmem_bytes(mp, h, i, spec, info) + _VMEM_HEADROOM
        <= info.vmem_capacity_bytes * ar.VMEM_FRACTION
    )


def fused_mlp(
    x: jax.Array,
    w_gu: jax.Array,
    s_gu: jax.Array,
    w_d: jax.Array,
    s_d: jax.Array,
    rank: jax.Array | None = None,
    n: int = 1,
    interpret: bool = False,
    norm: tuple[jax.Array, jax.Array, float] | None = None,
):
    """``down(swiglu_oai(x @ w_gu * s_gu)) * s_d`` of bf16 ``x [m, h]`` (at
    most MAX_ROWS rows) on e4m3 ``w_gu [h, 2i]`` ([gate | up])
    and ``w_d [i, h]`` with f32 ``s_gu [1, 2i]`` / ``s_d [1, h]``: bf16
    ``[m, h]``.

    With ``n > 1`` the result is summed over the n ranks of the 1-D mesh this
    runs under (shard_map); ``rank`` (int32 [1, 1]) is this device's mesh index.

    With ``norm = (residual [m, h] bf16, weight [1, h] f32, eps)`` the MLP's
    input is GemmaRMSNorm's fused add: ``r = bf16(x + residual)``, normalized
    with ``weight`` (1 + w); returns ``(out, r)``.
    """
    m, h = x.shape
    i = w_d.shape[0]
    if not supported(m, h, i, n):
        raise ValueError(f"fused_mlp: unsupported [{m}, {h}] x [{h}, {2 * i}]")
    if m % 8:  # whole f32 sublane tiles of rows
        rows = m + (-m % 8)

        def pad(a):
            return jnp.pad(a, ((0, rows - m), (0, 0)))

        if norm is not None:
            norm = (pad(norm[0]), *norm[1:])
        out = fused_mlp(pad(x), w_gu, s_gu, w_d, s_d, rank, n, interpret, norm)
        return tuple(o[:m] for o in out) if norm is not None else out[:m]
    info = pltpu.get_tpu_info()
    mp = m + (-m % _ROW_ALIGN)
    spec = ar.make_spec(n, mp, h) if n > 1 else None
    if rank is None:
        rank = jnp.zeros((1, 1), jnp.int32)
    gu_chunks = _chunks(h, 2 * i * w_gu.dtype.itemsize)
    d_chunks = _chunks(i, h * w_d.dtype.itemsize)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    limit = _vmem_bytes(mp, h, i, spec, info) + _VMEM_HEADROOM
    out_shape = jax.ShapeDtypeStruct((m, h), BF16)
    operands = [x, *(_in_hbm(a) for a in (s_gu, s_d, w_gu, w_d))]
    norm_scratch, eps = [], None
    if norm is not None:
        residual, weight, eps = norm
        operands += [residual, _in_hbm(weight)]
        norm_scratch = [pltpu.VMEM(x.shape, x.dtype), pltpu.VMEM(weight.shape, F32)]
        limit += 3 * ar.tiled_bytes((mp, h), F32, info)
    params = dict(vmem_limit_bytes=limit)
    if spec is not None:
        params["collective_id"] = ar.COLLECTIVE_ID
    return pl.pallas_call(
        functools.partial(
            _kernel, gu_chunks=gu_chunks, d_chunks=d_chunks, spec=spec, eps=eps
        ),
        out_shape=(out_shape, out_shape) if norm is not None else out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[hbm] * len(operands),
            out_specs=[vmem, vmem] if norm is not None else vmem,
            scratch_shapes=[
                pltpu.VMEM(x.shape, x.dtype),
                pltpu.VMEM(s_gu.shape, s_gu.dtype),
                pltpu.VMEM(s_d.shape, s_d.dtype),
                pltpu.VMEM((_SLOTS, gu_chunks[0][1], 2 * i), w_gu.dtype),
                pltpu.VMEM((_SLOTS, d_chunks[0][1], h), w_d.dtype),
                pltpu.VMEM((mp, 2 * i), F32),
                pltpu.VMEM((mp, i), BF16),
                pltpu.VMEM((mp, h), F32),
                pltpu.SemaphoreType.DMA((len(operands) - 2 + 2 * _SLOTS,)),
                *norm_scratch,
                *(
                    [pltpu.VMEM((mp, h), BF16), *ar.scratch_shapes(spec)]
                    if spec is not None
                    else []
                ),
            ],
        ),
        compiler_params=pltpu.CompilerParams(**params),
        name=f"dspark_mlp-m_{mp}-h_{h}-i_{i}"
        + ("-norm" if norm is not None else "")
        + (f"-ar_{n}" if n > 1 else ""),
        interpret=interpret,
    )(rank.reshape(1, 1).astype(jnp.int32), *operands)


def reference(x, w_gu, s_gu, w_d, s_d, norm=None):
    """The torch path's math in XLA (any row count): `fused_mlp` without the
    all-reduce."""
    i = w_d.shape[0]
    if norm is not None:
        residual, weight, eps = norm
        r = (x.astype(F32) + residual.astype(F32)).astype(BF16)
        rf = r.astype(F32)
        var = jnp.mean(rf * rf, axis=1, keepdims=True)
        x = (rf * lax.rsqrt(var + eps) * weight).astype(BF16)
        return reference(x, w_gu, s_gu, w_d, s_d), r

    def project(lhs, w, s):
        y = jnp.dot(lhs, w.astype(BF16), preferred_element_type=F32)
        return _project(y, s)

    gu = project(x, w_gu, s_gu)
    act = shared_activation(gu[:, :i], gu[:, i:]).astype(BF16)
    return project(act, w_d, s_d).astype(BF16)
