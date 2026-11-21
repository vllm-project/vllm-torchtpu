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
# Bytes per weight DMA, and how many are in flight: ~1.5 MiB chunks keep the
# compute behind the last one short; 6 in flight cover the DMA latency.
_CHUNK_BYTES = 3 << 19
_IN_FLIGHT = 6
_VMEM_HEADROOM = 4 << 20


def _chunks(rows: int, row_bytes: int) -> list[tuple[int, int]]:
    """(start, size) K-row chunks of ~_CHUNK_BYTES, sizes multiples of 128."""
    size = max(128, (_CHUNK_BYTES // row_bytes) // 128 * 128)
    return [(k, min(size, rows - k)) for k in range(0, rows, size)]


def _project(acc, scale):
    """bf16(bf16(acc) * scale), held in f32."""
    y = acc.astype(BF16).astype(F32) * scale.astype(F32)
    return y.astype(BF16).astype(F32)


def _kernel(
    meta_ref,  # SMEM int32 [1]: this device's mesh index (used with an all-reduce)
    x_hbm,  # HBM [m, h] bf16
    sgu_hbm,  # HBM [1, 2i] f32
    sd_hbm,  # HBM [1, h] f32
    wgu_hbm,  # HBM [h, 2i] e4m3: [gate | up]
    wd_hbm,  # HBM [i, h] e4m3
    o_ref,  # VMEM [m, h] bf16 (with an all-reduce: [mp, h])
    x_ref,  # VMEM [m, h] bf16
    sgu_ref,  # VMEM [1, 2i] f32
    sd_ref,  # VMEM [1, h] f32
    wgu_v,  # VMEM [h, 2i] e4m3
    wd_v,  # VMEM [i, h] e4m3
    gu_acc,  # VMEM [mp, 2i] f32
    act_ref,  # VMEM [mp, i] bf16
    out_acc,  # VMEM [mp, h] f32
    sem,  # DMA [3 + chunks]: x, scales, weight chunks
    *ar_scratch,
    gu_chunks,
    d_chunks,
    spec,
):
    m, h = x_ref.shape
    mp, i = act_ref.shape
    if spec is not None:
        ar.prologue(spec)
    # The rows and scales first: their latency hides under the first chunk's.
    small = [
        pltpu.make_async_copy(src, dst, sem.at[j])
        for j, (src, dst) in enumerate(
            [(x_hbm, x_ref), (sgu_hbm, sgu_ref), (sd_hbm, sd_ref)]
        )
    ]
    for copy in small:
        copy.start()
    copies = [
        pltpu.make_async_copy(src.at[pl.ds(k, c)], dst.at[pl.ds(k, c)], sem.at[j])
        for j, (src, dst, k, c) in enumerate(
            [(wgu_hbm, wgu_v, k, c) for k, c in gu_chunks]
            + [(wd_hbm, wd_v, k, c) for k, c in d_chunks],
            start=len(small),
        )
    ]
    for copy in copies[:_IN_FLIGHT]:
        copy.start()

    def landed(j):
        copies[j].wait()
        if j + _IN_FLIGHT < len(copies):
            copies[j + _IN_FLIGHT].start()

    small[0].wait()
    x = x_ref[...].astype(F32)
    if mp != m:
        x = jnp.concatenate([x, jnp.zeros((mp - m, h), F32)], 0)
    x = x.astype(BF16)
    gu_acc[...] = jnp.zeros_like(gu_acc)
    for j, (k, c) in enumerate(gu_chunks):
        landed(j)
        gu_acc[...] += jnp.dot(
            x[:, k : k + c], wgu_v[k : k + c, :], preferred_element_type=F32
        )
    small[1].wait()
    gate = _project(gu_acc[:, :i], sgu_ref[:, :i])
    up = _project(gu_acc[:, i:], sgu_ref[:, i:])
    act_ref[...] = shared_activation(gate, up).astype(BF16)
    out_acc[...] = jnp.zeros_like(out_acc)
    for j, (k, c) in enumerate(d_chunks, start=len(gu_chunks)):
        landed(j)
        out_acc[...] += jnp.dot(
            act_ref[:, k : k + c], wd_v[k : k + c, :], preferred_element_type=F32
        )
    small[2].wait()
    if spec is None:
        o_ref[...] = _project(out_acc[:m, :], sd_ref[...]).astype(o_ref.dtype)
    else:
        partial = _project(out_acc[...], sd_ref[...])
        ar.epilogue(spec, meta_ref[0], m, partial, o_ref, ar_scratch)


def _vmem_bytes(mp: int, h: int, i: int, spec, info) -> int:
    t = functools.partial(ar.tiled_bytes, info=info)
    total = (
        t((h, 2 * i), jnp.float8_e4m3fn)
        + t((i, h), jnp.float8_e4m3fn)
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
        total += ar.vmem_bytes(spec, info)
    return total


def supported(m: int, h: int, i: int) -> bool:
    """Whether [m, h] rows against [h, 2i] / [i, h] weights fit the kernel."""
    info = pltpu.get_tpu_info()
    mp = m + (-m % _ROW_ALIGN)
    return (
        0 < m <= MAX_ROWS
        and h % 128 == 0
        and i % 128 == 0
        and _vmem_bytes(mp, h, i, None, info) + _VMEM_HEADROOM
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
) -> jax.Array:
    """``down(swiglu_oai(x @ w_gu * s_gu)) * s_d`` of bf16 ``x [m, h]`` (at
    most MAX_ROWS rows) on e4m3 ``w_gu [h, 2i]`` ([gate | up])
    and ``w_d [i, h]`` with f32 ``s_gu [1, 2i]`` / ``s_d [1, h]``: bf16
    ``[m, h]``.

    With ``n > 1`` the result is summed over the n ranks of the 1-D mesh this
    runs under (shard_map); ``rank`` (int32 [1]) is this device's mesh index.
    """
    m, h = x.shape
    i = w_d.shape[0]
    if not supported(m, h, i):
        raise ValueError(f"fused_mlp: unsupported [{m}, {h}] x [{h}, {2 * i}]")
    if m % 8:  # whole f32 sublane tiles of rows
        rows = m + (-m % 8)
        x = jnp.pad(x, ((0, rows - m), (0, 0)))
        return fused_mlp(x, w_gu, s_gu, w_d, s_d, rank, n, interpret)[:m]
    info = pltpu.get_tpu_info()
    mp = m + (-m % _ROW_ALIGN)
    spec = ar.make_spec(n, mp, h) if n > 1 else None
    if rank is None:
        rank = jnp.zeros((1,), jnp.int32)
    gu_chunks = _chunks(h, 2 * i * w_gu.dtype.itemsize)
    d_chunks = _chunks(i, h * w_d.dtype.itemsize)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    out_rows = m if spec is None else mp
    # All of VMEM: room left over lets XLA prefetch the weights into VMEM
    # ahead of the call, serialized before it.
    params = dict(vmem_limit_bytes=info.vmem_capacity_bytes)
    if spec is not None:
        params["collective_id"] = ar.COLLECTIVE_ID
    out = pl.pallas_call(
        functools.partial(_kernel, gu_chunks=gu_chunks, d_chunks=d_chunks, spec=spec),
        out_shape=jax.ShapeDtypeStruct((out_rows, h), BF16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[hbm] * 5,
            out_specs=vmem,
            scratch_shapes=[
                pltpu.VMEM(x.shape, x.dtype),
                pltpu.VMEM(s_gu.shape, s_gu.dtype),
                pltpu.VMEM(s_d.shape, s_d.dtype),
                pltpu.VMEM((h, 2 * i), w_gu.dtype),
                pltpu.VMEM((i, h), w_d.dtype),
                pltpu.VMEM((mp, 2 * i), F32),
                pltpu.VMEM((mp, i), BF16),
                pltpu.VMEM((mp, h), F32),
                pltpu.SemaphoreType.DMA((3 + len(gu_chunks) + len(d_chunks),)),
                *(ar.scratch_shapes(spec) if spec is not None else []),
            ],
        ),
        compiler_params=pltpu.CompilerParams(**params),
        name=f"dspark_mlp-m_{mp}-h_{h}-i_{i}" + (f"-ar_{n}" if n > 1 else ""),
        interpret=interpret,
    )(rank.reshape(1).astype(jnp.int32), x, s_gu, s_d, w_gu, w_d)
    return out if out_rows == m else out[:m]


def reference(x, w_gu, s_gu, w_d, s_d):
    """The torch path's math in XLA (any row count): `fused_mlp` without the
    all-reduce."""
    i = w_d.shape[0]

    def project(lhs, w, s):
        y = jnp.dot(lhs, w.astype(BF16), preferred_element_type=F32)
        return _project(y, s)

    gu = project(x, w_gu, s_gu)
    act = shared_activation(gu[:, :i], gu[:, i:]).astype(BF16)
    return project(act, w_d, s_d).astype(BF16)
