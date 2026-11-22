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
"""A decode-sized TP all-reduce of the residual stream fused with the ops next
to it, in one Pallas call.

A decoder layer at decode rows runs, around each of its two TP all-reduces,
the all-reduce, the residual add and the RMSNorm as separate programs (plus
reshapes into and out of the all-reduce's view and a per-layer (1 + w)
convert). Here:

  `matmul_ar_norm` (attention exit): o_proj ``p = proj(a)`` contracted as its
      e4m3 weight streams in (the barrier hides under it), all-reduced,
      ``r = bf16(p + residual)`` and its GemmaRMSNorm;
  `ar_add_norm` (FFN exit): the FFN's partial all-reduced, ``r = bf16(x +
      residual)`` and its GemmaRMSNorm, optionally as e4m3 rows with per-row
      scales (kernels/sp_norm.py's quantization) for an e4m3 x e4m3 qkv_proj.

The all-reduce is moe_tp_decode/all_reduce.py's epilogue on a layout without
padding rows (`_exchange_shape`). Rounding follows the torch path: ``proj(v) =
bf16(bf16(v @ w) * s)`` (power-of-two channel scale, MiniMaxM3Mxfp8LinearMethod),
the all-reduce sums the n bf16 partials in f32 in rank order and rounds once
(bit-equal to kernels/collectives/tp_all_reduce.py), the norm is
fused_add_rms_norm (layers/adapter/ir_ops.py) with weight (1 + w) in f32. Only
o_proj's f32 summation order over K can differ from XLA's dot. The new
residual is a result in HBM, where the next call reads it: left to XLA it
lands in VMEM and is copied out, exposed (c4 twin j-f58b8bd8).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.moe_tp_decode import all_reduce as ar

BF16, F32, FP8 = jnp.bfloat16, jnp.float32, jnp.float8_e4m3fn
FP8_MAX = 448.0
# The all-reduce moves whole bf16 row tiles; bf16 MXU operands pack 16 rows.
_ROW_ALIGN = 16
# o_proj's weight streams in K-row chunks of this size through _SLOTS slots:
# the call's VMEM stays small.
_CHUNK_BYTES = 3 << 19
_SLOTS = 4
_VMEM_HEADROOM = 2 << 20


def _chunks(rows: int, row_bytes: int) -> list[tuple[int, int]]:
    """(start, size) K-row chunks of ~_CHUNK_BYTES, sizes multiples of 128."""
    size = max(128, (_CHUNK_BYTES // row_bytes) // 128 * 128)
    return [(k, min(size, rows - k)) for k in range(0, rows, size)]


def _padded(m: int) -> int:
    return m + (-m % _ROW_ALIGN)


def _pad_rows(v: jax.Array, rows: int) -> jax.Array:
    m = v.shape[0]
    if m == rows:
        return v
    return jnp.concatenate([v, jnp.zeros((rows - m, v.shape[1]), v.dtype)], 0)


def _exchange_shape(m: int, h: int) -> tuple[int, int]:
    """The [rows, cols] layout the all-reduce moves an [m, h] tensor in: rows
    a multiple of the 16-row bf16 tile, so no padding rows travel (+0.5 to
    1.7 us at 8 to 40 rows, probe j-7dceb3b1). m = 16k + 8 stacks the two
    column halves ([2m, h / 2]); the sum is elementwise, so any layout every
    rank shares is the same all-reduce."""
    return (2 * m, h // 2) if m % _ROW_ALIGN else (m, h)


def _to_exchange(v: jax.Array, h: int) -> jax.Array:
    if v.shape[0] % _ROW_ALIGN == 0:
        return v
    return jnp.concatenate([v[:, : h // 2], v[:, h // 2 :]], 0)


def _from_exchange(v: jax.Array, m: int) -> jax.Array:
    if v.shape[0] == m:
        return v
    return jnp.concatenate([v[:m], v[m:]], 1)


def _project(acc: jax.Array, scale: jax.Array) -> jax.Array:
    """bf16(bf16(acc) * scale), held in f32."""
    y = acc.astype(BF16).astype(F32) * scale.astype(F32)
    return y.astype(BF16).astype(F32)


def _gemma(w):
    """GemmaRMSNorm's weight (1 + w), in f32, from its stored w."""
    return w.astype(F32) + 1.0


def _add_norm(x32, res, nw, eps):
    """fused_add_rms_norm: (bf16 normalized, bf16 new residual)."""
    r = (x32 + res.astype(F32)).astype(BF16)
    rf = r.astype(F32)
    var = jnp.mean(rf * rf, axis=1, keepdims=True)
    return (rf * lax.rsqrt(var + eps) * nw.astype(F32)).astype(BF16), r


def _quantize(normed):
    """e4m3 rows of bf16 `normed` and their f32 [m, 1] scales amax / 448
    (kernels/sp_norm.py)."""
    nf = normed.astype(F32)
    scale = jnp.maximum(jnp.max(jnp.abs(nf), axis=-1, keepdims=True), 1e-12) / FP8_MAX
    return (nf * (1.0 / scale)).astype(FP8), scale


class _Ring:
    """A weight's K-row chunks streamed HBM -> VMEM through its slots, chunk
    j into slot j % slots, the next tenant started once a slot's chunk is
    contracted."""

    def __init__(self, w_hbm, w_v, sem, chunks):
        self.w_hbm, self.w_v, self.sem, self.chunks = w_hbm, w_v, sem, chunks
        self.slots = w_v.shape[0]
        for j in range(min(self.slots, len(chunks))):
            self._copy(j).start()

    def _copy(self, j):
        k, c = self.chunks[j]
        slot = j % self.slots
        return pltpu.make_async_copy(
            self.w_hbm.at[pl.ds(k, c)],
            self.w_v.at[slot, pl.ds(0, c)],
            self.sem.at[slot],
        )

    def contract(self, lhs, acc_ref):
        """acc_ref = lhs @ w, each chunk contracted as it lands."""
        acc_ref[...] = jnp.zeros(acc_ref.shape, F32)
        for j, (k, c) in enumerate(self.chunks):
            self._copy(j).wait()
            acc_ref[...] += jnp.dot(
                lhs[:, k : k + c],
                self.w_v[j % self.slots, pl.ds(0, c), :],
                preferred_element_type=F32,
            )
            if j + self.slots < len(self.chunks):
                self._copy(j + self.slots).start()


def _load(pairs, sem):
    """Start HBM -> VMEM copies of (src, dst) pairs; returns them."""
    copies = [pltpu.make_async_copy(s, d, sem.at[i]) for i, (s, d) in enumerate(pairs)]
    for copy in copies:
        copy.start()
    return copies


def _row(v_ref) -> jax.Array:
    """A 1-D [n] VMEM vector as an f32 [1, n] row."""
    return v_ref[...].astype(F32).reshape(1, -1)


def _add_norm_store(x32, res_v, nw_v, eps, outs, res_out, sem):
    """fused_add_rms_norm of x32: the new residual into res_out = (VMEM,
    result), its store (to HBM, the slower one) started before the norm is
    computed; the normed rows into outs: one (VMEM, result) pair for bf16,
    or two for e4m3 rows and their scales (`_quantize`)."""
    r = (x32 + res_v[...].astype(F32)).astype(BF16)
    res_out[0][...] = r
    copies = [pltpu.make_async_copy(*res_out, sem.at[0])]
    copies[0].start()
    rf = r.astype(F32)
    var = jnp.mean(rf * rf, axis=1, keepdims=True)
    normed = (rf * lax.rsqrt(var + eps) * _gemma(_row(nw_v))).astype(BF16)
    if len(outs) == 1:
        outs[0][0][...] = normed
    else:
        outs[0][0][...], outs[1][0][...] = _quantize(normed)
    for i, pair in enumerate(outs):
        copies.append(pltpu.make_async_copy(*pair, sem.at[1 + i]))
        copies[-1].start()
    for copy in copies:
        copy.wait()


def _ar_add_norm_kernel(meta_ref, *refs, spec, eps, quant):
    n_out = 3 if quant else 2
    x_hbm, res_hbm, nw_hbm = refs[:3]
    outs_hbm = refs[3 : 3 + n_out]
    x_v, res_v, nw_v = refs[3 + n_out : 6 + n_out]
    outs_v = refs[6 + n_out : 6 + 2 * n_out]
    red, io_sem, *scr = refs[6 + 2 * n_out :]
    m, h = x_v.shape
    ar.prologue(spec)
    x_in, *rest = _load([(x_hbm, x_v), (res_hbm, res_v), (nw_hbm, nw_v)], io_sem)
    x_in.wait()
    x = _to_exchange(x_v[...].astype(F32), h)
    ar.epilogue(spec, meta_ref[0, 0], spec.m, x, red, scr)
    for copy in rest:
        copy.wait()
    pairs = list(zip(outs_v, outs_hbm))
    _add_norm_store(
        _from_exchange(red[...].astype(F32), m),
        res_v,
        nw_v,
        eps,
        pairs[:-1],
        pairs[-1],
        io_sem,
    )


def _matmul_ar_norm_kernel(
    meta_ref,
    a_hbm,
    res_hbm,
    nw_hbm,
    s_hbm,
    w_hbm,
    o_hbm,
    r_hbm,
    a_v,
    res_v,
    nw_v,
    s_v,
    o_v,
    r_v,
    w_v,
    acc,
    red,
    io_sem,
    w_sem,
    *scr,
    spec,
    chunks,
    eps,
):
    m, h = res_v.shape
    ar.prologue(spec)
    a_in, s_in, *rest = _load(
        [(a_hbm, a_v), (s_hbm, s_v), (res_hbm, res_v), (nw_hbm, nw_v)], io_sem
    )
    ring = _Ring(w_hbm, w_v, w_sem, chunks)
    a_in.wait()
    ring.contract(_pad_rows(a_v[...], acc.shape[0]), acc)
    s_in.wait()
    partial = _to_exchange(_project(acc[pl.ds(0, m), :], _row(s_v)), h)
    ar.epilogue(spec, meta_ref[0, 0], spec.m, partial, red, scr)
    for copy in rest:
        copy.wait()
    _add_norm_store(
        _from_exchange(red[...].astype(F32), m),
        res_v,
        nw_v,
        eps,
        [(o_v, o_hbm)],
        (r_v, r_hbm),
        io_sem,
    )


def _ring_shape(k: int, nout: int, itemsize: int) -> tuple[list, tuple[int, int, int]]:
    chunks = _chunks(k, nout * itemsize)
    return chunks, (min(_SLOTS, len(chunks)), chunks[0][1], nout)


def _vmem_bytes(m: int, h: int, spec, info, weight=None) -> int:
    """An upper bound of a call's VMEM: the rows, the f32 temporaries, the
    all-reduce's scratch and, for o_proj (weight = (k, nout)), the weight
    ring and the accumulator."""
    t = functools.partial(ar.tiled_bytes, info=info)
    total = (
        4 * t((m, h), BF16)
        + 2 * t((1, h), BF16)
        + 6 * t((_padded(m), h), F32)
        + t((spec.m, spec.h), BF16)
        + ar.vmem_bytes(spec, info)
    )
    if weight is not None:
        k, nout = weight
        _, ring = _ring_shape(k, nout, 1)
        total += t(ring, FP8) + t((m, k), BF16) + 2 * t((_padded(m), nout), F32)
    return total


def supported(m: int, h: int, n: int, max_rows: int, weight=None) -> bool:
    """Whether an all-reduce of [m, h] over n ranks (and, for o_proj, m rows
    of the e4m3 weight [k, nout] = `weight`) fits one call (m <= max_rows)."""
    if not 0 < m <= max_rows or m % 8 or n < 2:
        return False
    if weight is not None and (weight[0] % 128 or weight[1] % 128):
        return False
    info = pltpu.get_tpu_info()
    mv, hv = _exchange_shape(m, h)
    if not ar.supported(n, mv, hv, info):
        return False
    need = _vmem_bytes(m, h, ar.make_spec(n, mv, hv), info, weight) + _VMEM_HEADROOM
    return need <= info.vmem_capacity_bytes * ar.VMEM_FRACTION


def _residual_result(shape, interpret):
    """The new residual, in HBM where the next call reads it."""
    if interpret:
        return jax.ShapeDtypeStruct(shape, BF16)
    return pltpu.HBM(shape, BF16)


def _in_hbm(a: jax.Array) -> jax.Array:
    """Pin an operand to HBM. Left free, XLA's memory-space assignment moves
    the call's operands into VMEM with async copies it issues right at the
    call, exposed ~0.6-0.9 us each (c4 twin j-951f7341); the weight:
    kernels/dspark_mlp.py."""
    return pltpu.with_memory_space_constraint(a, pltpu.HBM)


def ar_add_norm(
    x, residual, norm_w, rank, n: int, eps: float, quant=False, interpret=False
):
    """All-reduce of bf16 x [m, h] over the n ranks of the 1-D mesh this runs
    under (shard_map; rank int32 [1, 1] this device's mesh index), then
    ``r = bf16(x + residual)`` and its GemmaRMSNorm, norm_w the norm's stored
    weight w (it scales by 1 + w), [h]. Returns (norm(r), r), or with
    `quant` (e4m3 rows, f32 [m, 1] scales, r)."""
    m, h = x.shape
    info = pltpu.get_tpu_info()
    spec = ar.make_spec(n, *_exchange_shape(m, h))
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    normed = (
        [jax.ShapeDtypeStruct((m, h), FP8), jax.ShapeDtypeStruct((m, 1), F32)]
        if quant
        else [jax.ShapeDtypeStruct((m, h), BF16)]
    )
    out_shape = (*normed, _residual_result((m, h), interpret))
    return pl.pallas_call(
        functools.partial(_ar_add_norm_kernel, spec=spec, eps=eps, quant=quant),
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[hbm] * 3,
            out_specs=[hbm] * len(out_shape),
            scratch_shapes=[
                pltpu.VMEM(x.shape, x.dtype),
                pltpu.VMEM(residual.shape, residual.dtype),
                pltpu.VMEM(norm_w.shape, norm_w.dtype),
                *(pltpu.VMEM(o.shape, o.dtype) for o in out_shape),
                pltpu.VMEM((spec.m, spec.h), BF16),
                pltpu.SemaphoreType.DMA((3,)),
                *ar.scratch_shapes(spec),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=_vmem_bytes(m, h, spec, info) + _VMEM_HEADROOM,
            collective_id=ar.COLLECTIVE_ID,
        ),
        name=f"ar_add_norm{'_quant' if quant else ''}-m_{m}-h_{h}-ar_{n}",
        interpret=interpret,
    )(
        rank.reshape(1, 1).astype(jnp.int32),
        *(_in_hbm(a) for a in (x, residual, norm_w)),
    )


def matmul_ar_norm(
    a, w, s, residual, norm_w, rank, n: int, eps: float, interpret=False
):
    """``p = bf16(bf16(a @ w) * s)`` of bf16 a [m, K] on e4m3 w [K, h] with
    channel scale s [h], all-reduced over the n ranks (as `ar_add_norm`),
    then ``r = bf16(p + residual)`` and its GemmaRMSNorm. Returns (norm(r),
    r)."""
    m, h = residual.shape
    k, nout = w.shape
    info = pltpu.get_tpu_info()
    spec = ar.make_spec(n, *_exchange_shape(m, h))
    chunks, ring = _ring_shape(k, nout, w.dtype.itemsize)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    limit = _vmem_bytes(m, h, spec, info, (k, nout)) + _VMEM_HEADROOM
    return pl.pallas_call(
        functools.partial(_matmul_ar_norm_kernel, spec=spec, chunks=chunks, eps=eps),
        # The normed rows go to XLA ops, which read them fastest where XLA
        # puts them.
        out_shape=(
            jax.ShapeDtypeStruct((m, h), BF16),
            _residual_result((m, h), interpret),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[hbm] * 5,
            out_specs=[hbm, hbm],
            scratch_shapes=[
                pltpu.VMEM(a.shape, a.dtype),
                pltpu.VMEM(residual.shape, residual.dtype),
                pltpu.VMEM(norm_w.shape, norm_w.dtype),
                pltpu.VMEM(s.shape, s.dtype),
                pltpu.VMEM((m, h), BF16),
                pltpu.VMEM((m, h), BF16),
                pltpu.VMEM(ring, w.dtype),
                pltpu.VMEM((_padded(m), nout), F32),
                pltpu.VMEM((spec.m, spec.h), BF16),
                pltpu.SemaphoreType.DMA((4,)),
                pltpu.SemaphoreType.DMA((ring[0],)),
                *ar.scratch_shapes(spec),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=limit, collective_id=ar.COLLECTIVE_ID
        ),
        name=f"matmul_ar_norm-m_{m}-k_{k}-n_{nout}-ar_{n}",
        interpret=interpret,
    )(
        rank.reshape(1, 1).astype(jnp.int32),
        # The attention heads stay where XLA put them.
        a,
        *(_in_hbm(v) for v in (residual, norm_w, s, w)),
    )


def reference_matmul(v, w, s):
    """torch's ``matmul(v, bf16(w)) * s`` in bf16 (the scale multiply fuses
    into the dot)."""
    acc = jnp.dot(v, w.astype(BF16), preferred_element_type=F32).astype(BF16)
    return acc * s.reshape(1, -1).astype(BF16)


def reference_add_norm(x_reduced, residual, norm_w, eps: float):
    """`ar_add_norm` after its all-reduce, in XLA."""
    return _add_norm(
        x_reduced.astype(F32), residual, _gemma(norm_w.reshape(1, -1)), eps
    )


def quantize_rows(v):
    """`ar_add_norm`'s e4m3 rows and scales, in XLA."""
    return _quantize(v)


def e4m3_rows_matmul(q, row_scale, w, s):
    """``q * row_scale`` (e4m3 rows, f32 scales) times e4m3 w with channel
    scale s, e4m3 x e4m3 with f32 accumulation (models/minimax_m3/quant.py
    `_e4m3_rows_matmul`): bf16. At decode rows XLA runs it on the fp8 MXU
    path, 1.7-3.5 us faster than converting the weight to bf16 inside the
    dot's fusion (probe j-86f817ac)."""
    acc = jax.lax.dot(q, w, preferred_element_type=F32)
    # s is a power of two: scaling the rounded bf16 is exact, and a bf16
    # multiply fuses into the dot (an f32 copy of s is a separate op).
    return (acc * row_scale).astype(BF16) * s.reshape(1, -1).astype(BF16)
