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
"""Row-invariant tensor-parallel reductions for Muse Spark Open.

The XLA all-reduce on TPU v7x sums an element in an order that depends on its
offset in the buffer, so identical rows placed at different batch positions get
results one ulp apart. Over 62 layers that changes greedy tokens with
batch composition. This helper reduces partials in fixed rank order in fp32.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.mesh_utils import get_or_create_tp_mesh
from vllm_torchtpu.logger import init_logger

_AXIS = "muse_tp"
logger = init_logger(__name__)
_TP_MESH_CACHE: dict[int, Any] = {}
_TP_ORDER_CACHE: dict[int, tuple[int, ...]] = {}
_TP_ORDER_OPS: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
_TP_GATHER_OPS: dict[
    tuple[int, int, int, torch.dtype], Callable[[torch.Tensor], torch.Tensor]
] = {}
_AR_OPS: dict[
    tuple[int, torch.dtype, torch.dtype], Callable[[torch.Tensor], torch.Tensor]
] = {}
_ATTN_POST_OPS: dict[
    tuple[int, int, float, float, float, bool, bool, bool, bool], Callable
] = {}
_MOE_POST_OPS: dict[tuple[int, float, float, bool, bool, bool], Callable] = {}
_DENSE_POST_OPS: dict[tuple[int, float, float, bool, bool], Callable] = {}


def _fmt_eps(val: float) -> str:
    return f"{val:.6e}".replace(".", "p").replace("-", "m").replace("+", "")


def _dtype_tag(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _sharded_mean_sq(x: jax.Array, world: int) -> jax.Array:
    sq = jnp.mean(x * x, axis=-1, keepdims=True)
    sq_all = jax.lax.all_gather(sq, _AXIS, axis=0, tiled=False)
    acc = sq_all[0]
    for i in range(1, world):
        acc = acc + sq_all[i]
    return acc * jnp.float32(1.0 / world)


def _det_reduce_scatter(x: jax.Array, world: int) -> jax.Array:
    """[m, h] partials -> [m, h / world] f32 sum of this position's column block.

    Fixed source-order sum after an all_to_all. Hardware psum_scatter picks its
    reduction order from the row count, which breaks batch invariance.
    """
    m, h = x.shape
    # Exchange in the input dtype (bf16 partials halve the bytes); sum in fp32.
    ex = jax.lax.all_to_all(
        x.reshape(m, world, h // world), _AXIS, split_axis=1, concat_axis=1, tiled=True
    )
    acc = ex[:, 0].astype(jnp.float32)
    for i in range(1, world):
        acc = acc + ex[:, i].astype(jnp.float32)
    return acc


def _tp_mesh():
    # Plain dict cache rather than @lru_cache: Dynamo's lru_cache polyfill
    # bypasses the cache and traces into jax.default_backend()'s StrongLRUCache.
    tp = get_tp_group()
    world = int(tp.world_size)
    if world <= 1:
        return None
    cached = _TP_MESH_CACHE.get(world)
    if cached is not None:
        return cached
    if jax.default_backend() != "tpu":
        return None
    mesh = get_or_create_tp_mesh(axis_name=_AXIS, group=tp)
    _TP_MESH_CACHE[world] = mesh
    return mesh


def tp_collective_order() -> tuple[int, ...]:
    """TP ranks in the order Muse collectives place shards: position j holds rank
    order[j].

    Under sharded_jax_op the collectives order shards by physical device, not by
    rank_in_group, so host-side shard slicing must use order.index(rank).
    """
    world = get_tensor_model_parallel_world_size()
    if world <= 1:
        return tuple(range(world))
    cached = _TP_ORDER_CACHE.get(world)
    if cached is not None:
        return cached
    rank = get_tensor_model_parallel_rank()
    mesh = _tp_mesh()
    if mesh is None:
        return tuple(range(world))
    op = _TP_ORDER_OPS.get(world)
    if op is None:
        from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op

        spec = P(_AXIS, None)

        def fn(r: jax.Array) -> jax.Array:
            def local(r_p: jax.Array) -> jax.Array:
                g = jax.lax.all_gather(r_p[:, :1], _AXIS, axis=0, tiled=True)[:, 0]
                return (
                    jnp.zeros((1, r_p.shape[1]), jnp.int32).at[0, : g.shape[0]].set(g)
                )

            return jax.shard_map(
                local, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False
            )(r)

        op = sharded_jax_op(
            f"muse::tp_collective_order_w{world}",
            fn,
            mesh=mesh,
            input_partition_specs=(spec,),
            output_partition_specs=spec,
        )
        op.register_fake(lambda r: r.new_empty(r.shape))
        _TP_ORDER_OPS[world] = op

    probe = torch.full((1, 128), rank, dtype=torch.int32, device="tpu")
    row = op(probe).cpu().tolist()[0]
    order = tuple(int(v) for v in row[:world])
    if sorted(order) != list(range(world)):
        raise RuntimeError(
            f"Muse TP collective order probe invalid: rank={rank} row={row[:world]}"
        )
    logger.info(
        "Muse TP collective order: %s (rank %d at position %d)",
        order,
        rank,
        order.index(rank),
    )
    _TP_ORDER_CACHE[world] = order
    return order


def tp_shard_index() -> int:
    """Hidden column block this rank owns in the sharded residual stream."""
    return tp_collective_order().index(get_tensor_model_parallel_rank())


def tp_position_block(w: torch.Tensor, dim: int) -> torch.Tensor:
    """Block tp_shard_index() of the full tensor whose rank r shard along `dim` is `w`.

    Lets a column-parallel weight produce the residual block this rank owns when
    the collective order differs from rank order.
    """
    order = tp_collective_order()
    shard = tp_shard_index()
    # Global decision: every rank must enter the gather collective together.
    if order == tuple(range(len(order))):
        return w
    mesh = _tp_mesh()
    if mesh is None:
        raise RuntimeError(
            "tp_position_block requires an active TPU tensor-parallel mesh."
        )
    world = len(order)
    x = w.transpose(0, dim).contiguous() if dim != 0 else w.contiguous()
    rows, cols = int(x.shape[0]), int(x.shape[1])
    key = (world, rows, cols, x.dtype)
    op = _TP_GATHER_OPS.get(key)
    if op is None:
        from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op

        spec = P(_AXIS, None)

        def fn(a: jax.Array) -> jax.Array:
            return jax.shard_map(
                lambda a_p: jax.lax.all_gather(a_p, _AXIS, axis=0, tiled=True),
                mesh=mesh,
                in_specs=spec,
                out_specs=spec,
                check_vma=False,
            )(a)

        op = sharded_jax_op(
            f"muse::tp_gather_rows_w{world}_{rows}_{cols}_{_dtype_tag(x.dtype)}",
            fn,
            mesh=mesh,
            input_partition_specs=(spec,),
            output_partition_specs=spec,
        )
        op.register_fake(lambda a: a.new_empty((a.shape[0] * world, a.shape[1])))
        _TP_GATHER_OPS[key] = op

    full = op(x)
    # Position p of the gather holds rank order[p]'s block, which is block order[p].
    pos = order.index(shard)
    blk = full[pos * rows : (pos + 1) * rows].contiguous()
    return blk.transpose(0, dim).contiguous() if dim != 0 else blk


def _build_ar_op(in_dtype: torch.dtype, out_dtype: torch.dtype):
    mesh = _tp_mesh()
    if mesh is None:
        return None
    world = int(mesh.shape[_AXIS])
    key = (world, in_dtype, out_dtype)
    cached = _AR_OPS.get(key)
    if cached is not None:
        return cached
    from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op

    spec = P(_AXIS, None)
    out_jax_dtype = jnp.bfloat16 if out_dtype == torch.bfloat16 else jnp.float32

    def collective(x: jax.Array) -> jax.Array:
        def local(part: jax.Array) -> jax.Array:
            m, h = part.shape
            exchanged = jax.lax.all_to_all(
                part, _AXIS, split_axis=1, concat_axis=0, tiled=True
            ).reshape(world, m, h // world)
            acc = exchanged[0].astype(jnp.float32)
            for i in range(1, world):
                acc = acc + exchanged[i].astype(jnp.float32)
            return jax.lax.all_gather(
                acc.astype(out_jax_dtype), _AXIS, axis=1, tiled=True
            )

        return jax.shard_map(
            local, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False
        )(x)

    in_tag = "bf16" if in_dtype == torch.bfloat16 else "f32"
    out_tag = "bf16" if out_dtype == torch.bfloat16 else "f32"
    op = sharded_jax_op(
        f"muse::det_all_reduce_w{world}_{in_tag}_{out_tag}",
        collective,
        mesh=mesh,
        input_partition_specs=(spec,),
        output_partition_specs=spec,
    )

    def fake(x: torch.Tensor) -> torch.Tensor:
        return x.new_empty(x.shape, dtype=out_dtype)

    op.register_fake(fake)
    _AR_OPS[key] = op
    return op


def _get_attn_post_op(
    o_eps: float,
    post_attn_eps: float,
    pre_ffn_eps: float,
    gate_rhs_nk: bool,
    o_rhs_nk: bool,
    has_pre_exp: bool,
    pre_rhs_nk: bool,
    head_dim: int = 64,
):
    mesh = _tp_mesh()
    if mesh is None:
        return None
    world = int(mesh.shape[_AXIS])
    key = (
        world,
        int(head_dim),
        float(o_eps),
        float(post_attn_eps),
        float(pre_ffn_eps),
        bool(gate_rhs_nk),
        bool(o_rhs_nk),
        bool(has_pre_exp),
        bool(pre_rhs_nk),
    )
    op = _ATTN_POST_OPS.get(key)
    if op is not None:
        return op
    from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
    from vllm_torchtpu.kernels.batch_invariant.matmul import det_matmul
    from vllm_torchtpu.kernels.batch_invariant.rms_norm import det_o_norm_sigmoid_gate

    spec = P(_AXIS, None)
    name_base = (
        f"muse::det_attn_post_w{world}_hd{head_dim}_"
        f"{_fmt_eps(o_eps)}_{_fmt_eps(post_attn_eps)}_{_fmt_eps(pre_ffn_eps)}"
        f"_gnk{int(gate_rhs_nk)}_onk{int(o_rhs_nk)}_pe{int(has_pre_exp)}_pnk{int(pre_rhs_nk)}"
    )

    # pre_expert_proj column block j lives on rank j, which all_gather puts at position
    # order.index(j).
    order = tp_collective_order()
    z_perm = tuple(order.index(j) for j in range(world))

    if has_pre_exp:

        def fn_pe(
            ao: jax.Array,
            h_in: jax.Array,
            gw: jax.Array,
            ow: jax.Array,
            r_s: jax.Array,
            a_s: jax.Array,
            b_s: jax.Array,
            w_s: jax.Array,
            pew: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            def local(ao_p, h_p, gw_p, ow_p, rs_p, as_p, bs_p, ws_p, pew_p):
                m, h_dim = h_p.shape
                hs = h_dim // world
                attn_w = ao_p.shape[1]
                if attn_w % head_dim != 0:
                    raise NotImplementedError(
                        "det_attn_post_sandwich requires per-rank attention width "
                        f"divisible by head_dim={head_dim}, got {attn_w}."
                    )
                gate = det_matmul(h_p, gw_p, out_dtype=jnp.bfloat16, rhs_nk=gate_rhs_nk)
                ao_g = det_o_norm_sigmoid_gate(
                    ao_p.reshape(-1, head_dim),
                    gate.reshape(-1, head_dim),
                    eps=o_eps,
                ).reshape(m, attn_w)
                part = det_matmul(ao_g, ow_p, out_dtype=jnp.bfloat16, rhs_nk=o_rhs_nk)
                acc = _det_reduce_scatter(part, world)
                yf = acc.astype(jnp.bfloat16).astype(jnp.float32)
                inv1 = jax.lax.rsqrt(
                    _sharded_mean_sq(yf, world) + jnp.float32(post_attn_eps)
                )
                normed_s = (yf * inv1).astype(jnp.bfloat16).astype(jnp.float32)
                hid_s = (
                    as_p.astype(jnp.float32) * rs_p.astype(jnp.float32)
                    + bs_p.astype(jnp.float32) * normed_s
                )
                inv2 = jax.lax.rsqrt(
                    _sharded_mean_sq(hid_s, world) + jnp.float32(pre_ffn_eps)
                )
                h_ffn_s = (hid_s * inv2 * (1.0 + ws_p.astype(jnp.float32))).astype(
                    jnp.bfloat16
                )
                h_ffn = jax.lax.all_gather(
                    h_ffn_s.reshape(1, m, hs), _AXIS, axis=2, tiled=True
                ).reshape(m, h_dim)
                z_s = det_matmul(
                    h_ffn, pew_p, out_dtype=jnp.bfloat16, rhs_nk=pre_rhs_nk
                )
                n_s = z_s.shape[1]
                z_pre = jax.lax.all_gather(
                    z_s.reshape(1, m, n_s), _AXIS, axis=2, tiled=True
                ).reshape(m, world, n_s)
                if z_perm != tuple(range(world)):
                    z_pre = z_pre[:, jnp.asarray(z_perm, jnp.int32), :]
                return hid_s, h_ffn, z_pre.reshape(m, n_s * world)

            return jax.shard_map(
                local,
                mesh=mesh,
                in_specs=(spec,) * 9,
                out_specs=(spec, spec, spec),
                check_vma=False,
            )(ao, h_in, gw, ow, r_s, a_s, b_s, w_s, pew)

        op = sharded_jax_op(
            name_base,
            fn_pe,
            mesh=mesh,
            input_partition_specs=(spec,) * 9,
            output_partition_specs=(spec, spec, spec),
        )

        def fake_pe(ao, h_in, gw, ow, r_s, a_s, b_s, w_s, pew):
            m, h_dim = h_in.shape
            n_pe = pew.shape[0] if pre_rhs_nk else pew.shape[1]
            return (
                r_s.new_empty((m, r_s.shape[1]), dtype=torch.float32),
                h_in.new_empty((m, h_dim), dtype=torch.bfloat16),
                h_in.new_empty((m, n_pe * world), dtype=torch.bfloat16),
            )

        op.register_fake(fake_pe)
    else:

        def fn_dense(
            ao: jax.Array,
            h_in: jax.Array,
            gw: jax.Array,
            ow: jax.Array,
            r_s: jax.Array,
            a_s: jax.Array,
            b_s: jax.Array,
            w_s: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            def local(ao_p, h_p, gw_p, ow_p, rs_p, as_p, bs_p, ws_p):
                m, h_dim = h_p.shape
                hs = h_dim // world
                attn_w = ao_p.shape[1]
                if attn_w % head_dim != 0:
                    raise NotImplementedError(
                        "det_attn_post_sandwich requires per-rank attention width "
                        f"divisible by head_dim={head_dim}, got {attn_w}."
                    )
                gate = det_matmul(h_p, gw_p, out_dtype=jnp.bfloat16, rhs_nk=gate_rhs_nk)
                ao_g = det_o_norm_sigmoid_gate(
                    ao_p.reshape(-1, head_dim),
                    gate.reshape(-1, head_dim),
                    eps=o_eps,
                ).reshape(m, attn_w)
                part = det_matmul(ao_g, ow_p, out_dtype=jnp.bfloat16, rhs_nk=o_rhs_nk)
                acc = _det_reduce_scatter(part, world)
                yf = acc.astype(jnp.bfloat16).astype(jnp.float32)
                inv1 = jax.lax.rsqrt(
                    _sharded_mean_sq(yf, world) + jnp.float32(post_attn_eps)
                )
                normed_s = (yf * inv1).astype(jnp.bfloat16).astype(jnp.float32)
                hid_s = (
                    as_p.astype(jnp.float32) * rs_p.astype(jnp.float32)
                    + bs_p.astype(jnp.float32) * normed_s
                )
                inv2 = jax.lax.rsqrt(
                    _sharded_mean_sq(hid_s, world) + jnp.float32(pre_ffn_eps)
                )
                h_ffn_s = (hid_s * inv2 * (1.0 + ws_p.astype(jnp.float32))).astype(
                    jnp.bfloat16
                )
                h_ffn = jax.lax.all_gather(
                    h_ffn_s.reshape(1, m, hs), _AXIS, axis=2, tiled=True
                ).reshape(m, h_dim)
                return hid_s, h_ffn

            return jax.shard_map(
                local,
                mesh=mesh,
                in_specs=(spec,) * 8,
                out_specs=(spec, spec),
                check_vma=False,
            )(ao, h_in, gw, ow, r_s, a_s, b_s, w_s)

        op = sharded_jax_op(
            name_base,
            fn_dense,
            mesh=mesh,
            input_partition_specs=(spec,) * 8,
            output_partition_specs=(spec, spec),
        )

        def fake_dense(ao, h_in, gw, ow, r_s, a_s, b_s, w_s):
            m, h_dim = h_in.shape
            return (
                r_s.new_empty((m, r_s.shape[1]), dtype=torch.float32),
                h_in.new_empty((m, h_dim), dtype=torch.bfloat16),
            )

        op.register_fake(fake_dense)

    _ATTN_POST_OPS[key] = op
    return op


def _get_moe_post_op(
    post_ffn_eps: float,
    next_in_eps: float,
    post_rhs_nk: bool,
    has_next_norm: bool,
    gather_hid: bool,
):
    mesh = _tp_mesh()
    if mesh is None:
        return None
    world = int(mesh.shape[_AXIS])
    key = (
        world,
        float(post_ffn_eps),
        float(next_in_eps),
        bool(post_rhs_nk),
        bool(has_next_norm),
        bool(gather_hid),
    )
    op = _MOE_POST_OPS.get(key)
    if op is not None:
        return op
    from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
    from vllm_torchtpu.kernels.batch_invariant.matmul import det_matmul

    spec = P(_AXIS, None)
    name_base = (
        f"muse::det_moe_post_w{world}_{_fmt_eps(post_ffn_eps)}_{_fmt_eps(next_in_eps)}"
        f"_pnk{int(post_rhs_nk)}_nn{int(has_next_norm)}_gh{int(gather_hid)}"
    )

    def _compute_hid_s(m_p, pw_p, rs_p, pnw_p, as_p, bs_p):
        m, h_moe = m_p.shape
        m_sh = _det_reduce_scatter(m_p, world).astype(jnp.bfloat16)
        m_act = jax.lax.all_gather(
            m_sh.reshape(1, m, h_moe // world), _AXIS, axis=2, tiled=True
        ).reshape(m, h_moe)
        ff_s = det_matmul(m_act, pw_p, out_dtype=jnp.bfloat16, rhs_nk=post_rhs_nk)
        scaled_s = (
            (ff_s.astype(jnp.float32) * (1.0 + pnw_p.astype(jnp.float32)))
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        inv1 = jax.lax.rsqrt(
            _sharded_mean_sq(scaled_s, world) + jnp.float32(post_ffn_eps)
        )
        normed_s = (scaled_s * inv1).astype(jnp.bfloat16).astype(jnp.float32)
        return (
            as_p.astype(jnp.float32) * rs_p.astype(jnp.float32)
            + bs_p.astype(jnp.float32) * normed_s
        )

    if has_next_norm and gather_hid:

        def fn_n_g(
            m_fp32: jax.Array,
            pw: jax.Array,
            r_s: jax.Array,
            pnw_s: jax.Array,
            a_s: jax.Array,
            b_s: jax.Array,
            nw_s: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            def local(m_p, pw_p, rs_p, pnw_p, as_p, bs_p, nw_p):
                m = m_p.shape[0]
                hs = rs_p.shape[1]
                hid_s = _compute_hid_s(m_p, pw_p, rs_p, pnw_p, as_p, bs_p)
                inv2 = jax.lax.rsqrt(
                    _sharded_mean_sq(hid_s, world) + jnp.float32(next_in_eps)
                )
                h_next_s = (hid_s * inv2 * (1.0 + nw_p.astype(jnp.float32))).astype(
                    jnp.bfloat16
                )
                h_next = jax.lax.all_gather(
                    h_next_s.reshape(1, m, hs), _AXIS, axis=2, tiled=True
                ).reshape(m, hs * world)
                hid_full = jax.lax.all_gather(
                    hid_s.reshape(1, m, hs), _AXIS, axis=2, tiled=True
                ).reshape(m, hs * world)
                return hid_s, h_next, hid_full

            return jax.shard_map(
                local,
                mesh=mesh,
                in_specs=(spec,) * 7,
                out_specs=(spec, spec, spec),
                check_vma=False,
            )(m_fp32, pw, r_s, pnw_s, a_s, b_s, nw_s)

        op = sharded_jax_op(
            name_base,
            fn_n_g,
            mesh=mesh,
            input_partition_specs=(spec,) * 7,
            output_partition_specs=(spec, spec, spec),
        )

        def fake_n_g(m_fp32, pw, r_s, pnw_s, a_s, b_s, nw_s):
            m, hs = r_s.shape
            return (
                r_s.new_empty((m, hs), dtype=torch.float32),
                r_s.new_empty((m, hs * world), dtype=torch.bfloat16),
                r_s.new_empty((m, hs * world), dtype=torch.float32),
            )

        op.register_fake(fake_n_g)
    elif has_next_norm and not gather_hid:

        def fn_n_only(
            m_fp32: jax.Array,
            pw: jax.Array,
            r_s: jax.Array,
            pnw_s: jax.Array,
            a_s: jax.Array,
            b_s: jax.Array,
            nw_s: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            def local(m_p, pw_p, rs_p, pnw_p, as_p, bs_p, nw_p):
                m = m_p.shape[0]
                hs = rs_p.shape[1]
                hid_s = _compute_hid_s(m_p, pw_p, rs_p, pnw_p, as_p, bs_p)
                inv2 = jax.lax.rsqrt(
                    _sharded_mean_sq(hid_s, world) + jnp.float32(next_in_eps)
                )
                h_next_s = (hid_s * inv2 * (1.0 + nw_p.astype(jnp.float32))).astype(
                    jnp.bfloat16
                )
                h_next = jax.lax.all_gather(
                    h_next_s.reshape(1, m, hs), _AXIS, axis=2, tiled=True
                ).reshape(m, hs * world)
                return hid_s, h_next

            return jax.shard_map(
                local,
                mesh=mesh,
                in_specs=(spec,) * 7,
                out_specs=(spec, spec),
                check_vma=False,
            )(m_fp32, pw, r_s, pnw_s, a_s, b_s, nw_s)

        op = sharded_jax_op(
            name_base,
            fn_n_only,
            mesh=mesh,
            input_partition_specs=(spec,) * 7,
            output_partition_specs=(spec, spec),
        )

        def fake_n_only(m_fp32, pw, r_s, pnw_s, a_s, b_s, nw_s):
            m, hs = r_s.shape
            return (
                r_s.new_empty((m, hs), dtype=torch.float32),
                r_s.new_empty((m, hs * world), dtype=torch.bfloat16),
            )

        op.register_fake(fake_n_only)
    else:

        def fn_final(
            m_fp32: jax.Array,
            pw: jax.Array,
            r_s: jax.Array,
            pnw_s: jax.Array,
            a_s: jax.Array,
            b_s: jax.Array,
        ) -> jax.Array:
            def local(m_p, pw_p, rs_p, pnw_p, as_p, bs_p):
                m = m_p.shape[0]
                hs = rs_p.shape[1]
                hid_s = _compute_hid_s(m_p, pw_p, rs_p, pnw_p, as_p, bs_p)
                return jax.lax.all_gather(
                    hid_s.reshape(1, m, hs), _AXIS, axis=2, tiled=True
                ).reshape(m, hs * world)

            return jax.shard_map(
                local,
                mesh=mesh,
                in_specs=(spec,) * 6,
                out_specs=spec,
                check_vma=False,
            )(m_fp32, pw, r_s, pnw_s, a_s, b_s)

        op = sharded_jax_op(
            name_base,
            fn_final,
            mesh=mesh,
            input_partition_specs=(spec,) * 6,
            output_partition_specs=spec,
        )

        def fake_final(m_fp32, pw, r_s, pnw_s, a_s, b_s):
            m, hs = r_s.shape
            return r_s.new_empty((m, hs * world), dtype=torch.float32)

        op.register_fake(fake_final)

    _MOE_POST_OPS[key] = op
    return op


def _get_dense_post_op(
    post_ffn_eps: float,
    next_in_eps: float,
    has_next_norm: bool,
    gather_hid: bool,
):
    mesh = _tp_mesh()
    if mesh is None:
        return None
    world = int(mesh.shape[_AXIS])
    key = (
        world,
        float(post_ffn_eps),
        float(next_in_eps),
        bool(has_next_norm),
        bool(gather_hid),
    )
    op = _DENSE_POST_OPS.get(key)
    if op is not None:
        return op
    from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op

    spec = P(_AXIS, None)
    name_base = (
        f"muse::det_dense_post_w{world}_{_fmt_eps(post_ffn_eps)}_{_fmt_eps(next_in_eps)}"
        f"_nn{int(has_next_norm)}_gh{int(gather_hid)}"
    )

    def _compute_dense_hid_s(ff_s, rs_p, pnw_p, as_p, bs_p):
        scaled_s = (
            (ff_s.astype(jnp.float32) * (1.0 + pnw_p.astype(jnp.float32)))
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        inv1 = jax.lax.rsqrt(
            _sharded_mean_sq(scaled_s, world) + jnp.float32(post_ffn_eps)
        )
        normed_s = (scaled_s * inv1).astype(jnp.bfloat16).astype(jnp.float32)
        return (
            as_p.astype(jnp.float32) * rs_p.astype(jnp.float32)
            + bs_p.astype(jnp.float32) * normed_s
        )

    if has_next_norm and gather_hid:

        def fn_dn_g(
            ff_s: jax.Array,
            r_s: jax.Array,
            pnw_s: jax.Array,
            a_s: jax.Array,
            b_s: jax.Array,
            nw_s: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            def local(ffs_p, rs_p, pnw_p, as_p, bs_p, nw_p):
                m, hs = rs_p.shape
                hid_s = _compute_dense_hid_s(ffs_p, rs_p, pnw_p, as_p, bs_p)
                inv2 = jax.lax.rsqrt(
                    _sharded_mean_sq(hid_s, world) + jnp.float32(next_in_eps)
                )
                h_next_s = (hid_s * inv2 * (1.0 + nw_p.astype(jnp.float32))).astype(
                    jnp.bfloat16
                )
                h_next = jax.lax.all_gather(
                    h_next_s.reshape(1, m, hs), _AXIS, axis=2, tiled=True
                ).reshape(m, hs * world)
                hid_full = jax.lax.all_gather(
                    hid_s.reshape(1, m, hs), _AXIS, axis=2, tiled=True
                ).reshape(m, hs * world)
                return hid_s, h_next, hid_full

            return jax.shard_map(
                local,
                mesh=mesh,
                in_specs=(spec,) * 6,
                out_specs=(spec, spec, spec),
                check_vma=False,
            )(ff_s, r_s, pnw_s, a_s, b_s, nw_s)

        op = sharded_jax_op(
            name_base,
            fn_dn_g,
            mesh=mesh,
            input_partition_specs=(spec,) * 6,
            output_partition_specs=(spec, spec, spec),
        )

        def fake_dn_g(ff_s, r_s, pnw_s, a_s, b_s, nw_s):
            m, hs = r_s.shape
            return (
                r_s.new_empty((m, hs), dtype=torch.float32),
                r_s.new_empty((m, hs * world), dtype=torch.bfloat16),
                r_s.new_empty((m, hs * world), dtype=torch.float32),
            )

        op.register_fake(fake_dn_g)
    else:

        def fn_dn(
            ff_s: jax.Array,
            r_s: jax.Array,
            pnw_s: jax.Array,
            a_s: jax.Array,
            b_s: jax.Array,
            nw_s: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            def local(ffs_p, rs_p, pnw_p, as_p, bs_p, nw_p):
                m, hs = rs_p.shape
                hid_s = _compute_dense_hid_s(ffs_p, rs_p, pnw_p, as_p, bs_p)
                inv2 = jax.lax.rsqrt(
                    _sharded_mean_sq(hid_s, world) + jnp.float32(next_in_eps)
                )
                h_next_s = (hid_s * inv2 * (1.0 + nw_p.astype(jnp.float32))).astype(
                    jnp.bfloat16
                )
                h_next = jax.lax.all_gather(
                    h_next_s.reshape(1, m, hs), _AXIS, axis=2, tiled=True
                ).reshape(m, hs * world)
                return hid_s, h_next

            return jax.shard_map(
                local,
                mesh=mesh,
                in_specs=(spec,) * 6,
                out_specs=(spec, spec),
                check_vma=False,
            )(ff_s, r_s, pnw_s, a_s, b_s, nw_s)

        op = sharded_jax_op(
            name_base,
            fn_dn,
            mesh=mesh,
            input_partition_specs=(spec,) * 6,
            output_partition_specs=(spec, spec),
        )

        def fake_dn(ff_s, r_s, pnw_s, a_s, b_s, nw_s):
            m, hs = r_s.shape
            return (
                r_s.new_empty((m, hs), dtype=torch.float32),
                r_s.new_empty((m, hs * world), dtype=torch.bfloat16),
            )

        op.register_fake(fake_dn)

    _DENSE_POST_OPS[key] = op
    return op


def prepare_det_collectives(
    *,
    rms_norm_eps: float = 1e-5,
    post_norm_eps: float = 1e-8,
    head_dim: int = 64,
) -> None:
    """Eagerly construct TP mesh and all_to_all reduction ops before torch.compile."""
    if not envs.VLLM_TPU_MUSE_BATCH_INVARIANT:
        return
    mesh = _tp_mesh()
    if mesh is None:
        return
    tp_collective_order()
    for in_dt, out_dt in (
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.bfloat16),
        (torch.float32, torch.float32),
    ):
        _build_ar_op(in_dt, out_dt)
    for g_nk in (False, True):
        for o_nk in (False, True):
            _get_attn_post_op(
                rms_norm_eps,
                post_norm_eps,
                rms_norm_eps,
                g_nk,
                o_nk,
                False,
                False,
                head_dim=head_dim,
            )
            for p_nk in (False, True):
                _get_attn_post_op(
                    rms_norm_eps,
                    post_norm_eps,
                    rms_norm_eps,
                    g_nk,
                    o_nk,
                    True,
                    p_nk,
                    head_dim=head_dim,
                )
    for p_nk in (False, True):
        _get_moe_post_op(post_norm_eps, rms_norm_eps, p_nk, True, False)
        _get_moe_post_op(post_norm_eps, rms_norm_eps, p_nk, True, True)
        _get_moe_post_op(post_norm_eps, rms_norm_eps, p_nk, False, True)
    _get_dense_post_op(post_norm_eps, rms_norm_eps, True, False)
    _get_dense_post_op(post_norm_eps, rms_norm_eps, True, True)


def _fixed_order_sum(chunks: torch.Tensor) -> torch.Tensor:
    """Sum dim 0 sequentially in fp32 so the order never depends on layout."""
    acc = chunks[0].float()
    for i in range(1, chunks.shape[0]):
        acc = acc + chunks[i].float()
    return acc


def det_all_reduce(
    x: torch.Tensor, *, out_dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Sum `x` [..., H] over the TP group with a row-invariant fixed reduction."""
    target_dtype = out_dtype if out_dtype is not None else x.dtype
    n = get_tensor_model_parallel_world_size()
    if n == 1:
        return x.to(target_dtype)
    if not envs.VLLM_TPU_MUSE_BATCH_INVARIANT:
        return tensor_model_parallel_all_reduce(x).to(target_dtype)
    if x.device.type == "tpu" and x.shape[-1] % (n * 128) == 0:
        op = _build_ar_op(x.dtype, target_dtype)
        if op is not None:
            lead = x.shape[:-1]
            x2 = x.reshape(-1, x.shape[-1]).contiguous()
            return op(x2).reshape(*lead, x.shape[-1])
    gathered = tensor_model_parallel_all_gather(x.unsqueeze(0), dim=0)  # [n, ..., H]
    return _fixed_order_sum(gathered).to(target_dtype)


def det_attn_post_sandwich(
    attn_out: torch.Tensor,
    h_in: torch.Tensor,
    gate_w: torch.Tensor,
    o_w: torch.Tensor,
    res_s: torch.Tensor,
    alpha_s: torch.Tensor,
    beta_s: torch.Tensor,
    pre_ffn_w_s: torch.Tensor,
    *,
    o_eps: float,
    post_attn_eps: float,
    pre_ffn_eps: float,
    gate_rhs_nk: bool,
    o_rhs_nk: bool,
    pre_exp_w: torch.Tensor | None = None,
    pre_rhs_nk: bool = False,
    head_dim: int = 64,
) -> tuple[torch.Tensor, ...]:
    """Fused post-attention gate, o_proj, reduce_scatter, sandwich norms, and
    pre_expert_proj.
    """
    has_pe = pre_exp_w is not None
    op = _get_attn_post_op(
        o_eps,
        post_attn_eps,
        pre_ffn_eps,
        gate_rhs_nk,
        o_rhs_nk,
        has_pe,
        pre_rhs_nk,
        head_dim=head_dim,
    )
    if op is None:
        raise RuntimeError(
            "det_attn_post_sandwich requires TPU with tensor_parallel_size > 1."
        )
    if has_pe:
        return op(
            attn_out,
            h_in,
            gate_w,
            o_w,
            res_s,
            alpha_s,
            beta_s,
            pre_ffn_w_s,
            pre_exp_w,
        )
    return op(attn_out, h_in, gate_w, o_w, res_s, alpha_s, beta_s, pre_ffn_w_s)


def det_moe_post_sandwich(
    m_fp32: torch.Tensor,
    post_exp_w: torch.Tensor,
    res_s: torch.Tensor,
    post_ffn_w_s: torch.Tensor,
    alpha_s: torch.Tensor,
    beta_s: torch.Tensor,
    next_in_w_s: torch.Tensor | None,
    *,
    post_ffn_eps: float,
    next_in_eps: float,
    post_rhs_nk: bool,
    gather_hid: bool = False,
):
    """Fused MoE combine, post_expert_proj, post-FFN norm/gate, and next
    input_layernorm.
    """
    has_next = next_in_w_s is not None
    op = _get_moe_post_op(
        post_ffn_eps,
        next_in_eps,
        post_rhs_nk,
        has_next,
        gather_hid,
    )
    if op is None:
        raise RuntimeError(
            "det_moe_post_sandwich requires TPU with tensor_parallel_size > 1."
        )
    if has_next:
        return op(
            m_fp32,
            post_exp_w,
            res_s,
            post_ffn_w_s,
            alpha_s,
            beta_s,
            next_in_w_s,
        )
    return op(m_fp32, post_exp_w, res_s, post_ffn_w_s, alpha_s, beta_s)


def det_dense_post_sandwich(
    ff_s: torch.Tensor,
    res_s: torch.Tensor,
    post_ffn_w_s: torch.Tensor,
    alpha_s: torch.Tensor,
    beta_s: torch.Tensor,
    next_in_w_s: torch.Tensor,
    *,
    post_ffn_eps: float,
    next_in_eps: float,
    gather_hid: bool = False,
):
    """Fused dense MLP post-FFN norm/gate and next layer input_layernorm."""
    op = _get_dense_post_op(
        post_ffn_eps,
        next_in_eps,
        True,
        gather_hid,
    )
    if op is None:
        raise RuntimeError(
            "det_dense_post_sandwich requires TPU with tensor_parallel_size > 1."
        )
    return op(ff_s, res_s, post_ffn_w_s, alpha_s, beta_s, next_in_w_s)
