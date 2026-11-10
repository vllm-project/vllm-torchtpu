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
"""TPU_SP_TP2_PALLAS: the sequence-parallel collectives of a DP x TP2 decoder layer as Pallas kernels over the
TP pair (the two cores of one chip) instead of XLA's.

* ``gather_q_s_or`` -- the pre-attention all-gather of the fp8 rows ``x_q`` [h, hidden] and their f32 scales
  ``x_s`` [h, 1] as ONE pair-exchange kernel (each core DMAs its block to its partner's VMEM slot; block r = TP
  rank r's rows, the order of the XLA all-gather it replaces). Pure copies.
* ``reduce_scatter_or`` -- the post-attention reduce-scatter of the bf16 partials [R, hidden] as the small
  collectives' ``small_reduce_scatter`` over the pair (each output element is the f32 sum of two bf16 values
  rounded once -- the bits of XLA's reduce-scatter; measured bitwise at R = 96/112/128, decdnsmb1).

Measured (decdnsmb2, marginal us per op in one program, r28 shape R = 112): pair gather 2.04 vs XLA x_q + x_s
gathers 3.25 + 2.40 (serving), reduce-scatter 2.23 vs XLA 5.74 (served as all-reduce + slice at 56 rows/rank).
Built in ``prebuild`` (called from ``small_all_reduce_op.prebuild``) only when the TP group is 2 cores that are a
block of the expert-parallel mesh (``small_all_reduce_op._tp_pairs_match_mesh``); otherwise every call falls back.
Like the small all-reduce, the kernels are taken only inside the warmup-compiled programs (static shapes).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, PartitionSpec

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.distributed.ep_mesh import build_ep_mesh, ep_mesh_index
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.kernels.experimental.small_collectives import common as sc
from vllm_torchtpu.kernels.experimental.small_collectives import reduce_scatter as srs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

TP = 2
AXES = ("dp_", "tp_")
# Largest per-rank row block either kernel takes (VMEM: 2 x rows x hidden); larger steps keep XLA's collectives.
MAX_GATHER_ROWS = 256
MAX_RS_ROWS = 512
# Barrier semaphore group of the pair gather (the small collectives use 5, the fused EP MoE 0, the MK 61).
GATHER_COLLECTIVE_ID = 7

_GATHER_OP: Callable | None = None
_RS_OP: Callable | None = None
_RANK: torch.Tensor | None = None  # this core's TP index (0/1) as an int32 [1] device tensor (kernels take it as data)
_TP_RANKS: tuple[int, ...] = ()


def pair_gather(xq: jax.Array, xs: jax.Array, rank: jax.Array, *, mutant: bool = False, interpret: bool = False):
    """Per-core body (inside a shard_map over AXES): x_q [h, W] fp8, x_s [h] f32 -> [2h, W], [2h]."""
    h, w = xq.shape
    hp = -(-h // 32) * 32  # fp8 sublane tile
    xq_p = jnp.pad(xq, ((0, hp - h), (0, 0))) if hp != h else xq
    xs_p = jnp.pad(xs.reshape(h, 1), ((0, hp - h), (0, 127)))

    def kernel(rank_ref, xq_ref, xs_ref, oq_ref, os_ref, sq, ss, ssem, rsem):
        me = rank_ref[0]
        sc.barrier_peers(me, (1,), "tp_", True)
        peer = sc.peer(me ^ 1, "tp_", True)
        cq = pltpu.make_async_remote_copy(src_ref=xq_ref, dst_ref=sq.at[me], send_sem=ssem.at[0],
                                          recv_sem=rsem.at[0], device_id=peer, device_id_type=sc.MESH)
        cs = pltpu.make_async_remote_copy(src_ref=xs_ref, dst_ref=ss.at[me], send_sem=ssem.at[1],
                                          recv_sem=rsem.at[1], device_id=peer, device_id_type=sc.MESH)
        cq.start()
        cs.start()
        sq[me] = xq_ref[...]
        ss[me] = xs_ref[...]
        cq.wait_recv()
        cs.wait_recv()
        cq.wait_send()
        cs.wait_send()
        oq_ref[...] = sq[...]
        os_ref[...] = ss[...]

    with sc.mesh_axes(AXES):
        oq, os_ = pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct((2, hp, w), xq.dtype), jax.ShapeDtypeStruct((2, hp, 128), jnp.float32)),
            in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM), pl.BlockSpec(), pl.BlockSpec()],
            scratch_shapes=[pltpu.VMEM((2, hp, w), xq.dtype), pltpu.VMEM((2, hp, 128), jnp.float32),
                            pltpu.SemaphoreType.DMA((2,)), pltpu.SemaphoreType.DMA((2,))],
            compiler_params=pltpu.CompilerParams(collective_id=GATHER_COLLECTIVE_ID),
            interpret=pltpu.InterpretParams() if interpret else False,
            name=f"tp2_sp_pair_gather_h{h}",
        )(rank.astype(jnp.int32), xq_p, xs_p)
    if mutant:  # test only: the two ranks' rows swapped (wrong order) -> the tokens must change
        oq, os_ = oq[::-1], os_[::-1]
    return oq[:, :h].reshape(2 * h, w), os_[:, :h, 0].reshape(2 * h)


def pair_reduce_scatter(x: jax.Array, rank: jax.Array, *, mutant: bool = False) -> jax.Array:
    """Per-core body: bf16 partials [R, W] -> this core's [R/2, W] sum (block r to TP rank r)."""
    with sc.mesh_axes(AXES):
        out = srs.small_reduce_scatter(x, num_cores=TP, axis="tp_", rank=rank)
    if mutant:  # test only (TPU_SP_TP2_PALLAS_RS_MUTANT): this core's summed rows reversed -> tokens must change
        out = out[::-1]
    return out


def _mesh2():
    mesh = build_ep_mesh()
    if mesh is None:
        return None
    n = int(np.asarray(mesh.devices).size)
    return Mesh(np.asarray(mesh.devices).reshape(n // TP, TP), axis_names=AXES)


def _build_ops(mesh2, mutant: bool, rs_mutant: bool = False):
    spec = PartitionSpec(AXES)

    def g_fn(xq: jax.Array, xs: jax.Array, rank: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.shard_map(lambda a, b, r: pair_gather(a, b, r, mutant=mutant), mesh=mesh2,
                             in_specs=(spec, spec, spec), out_specs=(spec, spec), check_vma=False)(xq, xs, rank)

    def g_fake(xq, xs, rank):
        return xq.new_empty((2 * xq.shape[0], xq.shape[1])), xs.new_empty((2 * xs.shape[0],))

    def r_fn(x: jax.Array, rank: jax.Array) -> jax.Array:
        return jax.shard_map(lambda a, r: pair_reduce_scatter(a, r, mutant=rs_mutant), mesh=mesh2,
                             in_specs=(spec, spec), out_specs=spec, check_vma=False)(x, rank)

    def r_fake(x, rank):
        return x.new_empty((x.shape[0] // TP, x.shape[1]))

    gop = sharded_jax_op("pallas::tp2_sp_pair_gather", g_fn, mesh=mesh2, input_partition_specs=(spec, spec, spec),
                         output_partition_specs=(spec, spec), fake=g_fake)
    rop = sharded_jax_op("pallas::tp2_sp_reduce_scatter", r_fn, mesh=mesh2, input_partition_specs=(spec, spec),
                         output_partition_specs=spec, fake=r_fake)
    return gop, rop


def prebuild() -> None:
    """Build both ops and the rank tensor outside any traced region (the mesh comes from a collective)."""
    global _GATHER_OP, _RS_OP, _RANK, _TP_RANKS
    if not envs.TPU_SP_TP2_PALLAS or _GATHER_OP is not None:
        return
    from vllm.distributed.parallel_state import get_tp_group

    from vllm_torchtpu.layers.adapter.small_all_reduce_op import _tp_pairs_match_mesh

    tp = int(get_tp_group().world_size)
    mesh2 = _mesh2()
    idx = ep_mesh_index()
    if tp != TP or mesh2 is None or idx is None or not _tp_pairs_match_mesh(idx, TP):
        logger.warning("TPU_SP_TP2_PALLAS=1 ignored: needs a TP group of 2 cores that is a block of the "
                       "expert-parallel mesh (tp=%d, mesh=%s)", tp, None if mesh2 is None else dict(mesh2.shape))
        return
    _TP_RANKS = tuple(int(r) for r in get_tp_group().ranks)
    _RANK = torch.tensor([idx % TP], dtype=torch.int32, device="tpu")
    parts = envs.TPU_SP_TP2_PALLAS_PARTS or "both"
    _GATHER_OP, _RS_OP = _build_ops(mesh2, bool(envs.TPU_SP_TP2_PALLAS_MUTANT), bool(envs.TPU_SP_TP2_PALLAS_RS_MUTANT))
    if parts == "rs":  # bisect: the gather keeps XLA's all-gathers
        _GATHER_OP = None
    if parts == "gather":  # bisect: the reduce-scatter keeps XLA's
        _RS_OP = None
    logger.info("TPU_SP_TP2_PALLAS armed | tp pairs of the %s mesh | SP x_q/x_s gather + reduce-scatter as Pallas "
                "pair kernels (gather rows <= %d, reduce-scatter rows <= %d) | parts=%s%s", dict(mesh2.shape), MAX_GATHER_ROWS,
                MAX_RS_ROWS, parts, (" | MUTANT (test only)" if envs.TPU_SP_TP2_PALLAS_MUTANT else "")
                + (" | RS_MUTANT (test only)" if envs.TPU_SP_TP2_PALLAS_RS_MUTANT else ""))


def _static_ok() -> bool:
    return torch.compiler.is_compiling() and shape_variants.in_warmup()


def _retain(t: torch.Tensor) -> torch.Tensor:
    """A zero read of the rank buffer riding on ``t`` (float), so a warmup bucket that skips the kernel lifts the
    same buffers as one that calls it (see small_all_reduce_op.retain)."""
    if _RANK is None or not shape_variants.in_warmup():
        return t
    v = _RANK[0].to(t.dtype)
    return t + (v - v)


def _gather_rows_cap() -> int:
    """Largest per-rank row block the pair gather takes (TPU_SP_TP2_PALLAS_MAX_STEP_ROWS / TP when set)."""
    cap = int(envs.TPU_SP_TP2_PALLAS_MAX_STEP_ROWS or 0)
    return MAX_GATHER_ROWS if cap <= 0 else min(MAX_GATHER_ROWS, cap // TP)


def _rs_rows_cap() -> int:
    """Largest step (rows of the [R, W] partials) the pair reduce-scatter takes."""
    cap = int(envs.TPU_SP_TP2_PALLAS_MAX_STEP_ROWS or 0)
    return MAX_RS_ROWS if cap <= 0 else min(MAX_RS_ROWS, cap)


def gather_q_s_or(x_q: torch.Tensor, x_s: torch.Tensor, fallback: Callable):
    """The SP all-gather of (x_q [h, W] fp8, x_s [h, 1] f32) over the TP pair; ``fallback(x_q, x_s)`` otherwise."""
    if (
        _GATHER_OP is not None
        and _static_ok()
        and x_q.dim() == 2
        and x_q.dtype == torch.float8_e4m3fn
        and x_q.shape[1] % 128 == 0
        and x_s.dtype == torch.float32
        and x_s.numel() == x_q.shape[0]
    ):
        if x_q.shape[0] <= _gather_rows_cap():
            q, s = _GATHER_OP(x_q, x_s.reshape(-1), _RANK)
            return q, s.reshape((2 * x_s.shape[0],) + tuple(x_s.shape[1:]))
        q, s = fallback(x_q, x_s)
        return q, _retain(s)
    return fallback(x_q, x_s)


def reduce_scatter_or(x: torch.Tensor, fallback: Callable) -> torch.Tensor:
    """The SP reduce-scatter of bf16 [R, W] partials over the TP pair; ``fallback(x)`` otherwise."""
    if (
        _RS_OP is not None
        and _static_ok()
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.shape[1] % 128 == 0
        and x.shape[0] % TP == 0
    ):
        if x.shape[0] <= _rs_rows_cap():
            return _RS_OP(x, _RANK)
        return _retain(fallback(x))
    return fallback(x)
