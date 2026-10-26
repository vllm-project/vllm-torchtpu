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
"""TPU_DSV4_MHC_AR_SEAM: the TP all-reduce of a sublayer partial fused with
the mHC seam that consumes it, as one torch op over the expert-parallel mesh.

Built like ``small_all_reduce_op``: a shard_map over the EP mesh, this core's
TP-group position and the group's mesh indices as data. Inside JAX, a block
of at most the small all-reduce's row cap runs the fused kernel
(``ar_seam_kernel``); a larger one runs XLA's all-reduce over the same groups
followed by the unfused seam kernel -- exactly the pair it replaces. Either
way the outputs are those of ``all_reduce`` -> ``mhc_ops.fused``, bit for bit.
"""

import functools
from typing import Any

import jax
import torch
from jax import lax
from jax.sharding import PartitionSpec

from vllm_torchtpu.kernels.deepseek_v4.mhc import ar_seam_kernel, fused_post_pre_kernel, utils
from vllm_torchtpu.kernels.experimental.small_collectives import all_reduce
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_cache: dict[str, Any] = {}


def _body(groups, cap, axis, consts, kernel_kw, reduced_above_cap=False, fn_mode="vmem"):
    """reduced_above_cap (the MoE side): above ``cap`` rows the operand is
    already the reduced sum (the fused EP op hands over a partial only up to
    the same cap), so only the seam runs. fn_mode: ``ar_seam``'s fn
    placement ("pin": entry DMA from HBM under the all-reduce)."""
    cores = len(groups[0])
    index_groups = [list(g) for g in groups]
    form, chunks, skip = all_reduce.group_form(groups)
    (rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat) = consts

    def body(x_p, residual, post, comb, fn_cat, hc_scale, hc_base, norm_w, rank, peers):
        if x_p.shape[0] > cap:
            x = x_p if reduced_above_cap else lax.psum(
                x_p, axis, axis_index_groups=index_groups
            )
            return fused_post_pre_kernel.mhc_fused_post_pre(
                x, residual, post, comb, utils.CatFn(fn_cat), hc_scale, hc_base,
                rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                norm_weight=norm_w, norm_eps=rms_eps, **kernel_kw,
            )
        return ar_seam_kernel.mhc_ar_seam(
            x_p, residual, post, comb, utils.CatFn(fn_cat), hc_scale, hc_base,
            rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
            norm_weight=norm_w, norm_eps=rms_eps, num_cores=cores, axis=axis, rank=rank,
            peers=peers, form=form, chunks=chunks, fn_chunks=kernel_kw["fn_chunks"],
            small_block=kernel_kw["small_block"], skip_device_barrier=skip,
            fn_mode=fn_mode,
        )

    return body


def _fake(x_p, residual, *args):
    outer = residual.shape[:-2]
    m, h = residual.shape[-2:]
    return (
        torch.empty_like(residual),
        torch.empty((*outer, m, 1), dtype=torch.float32, device=residual.device),
        torch.empty((*outer, m, m), dtype=torch.float32, device=residual.device),
        torch.empty((*outer, h), dtype=residual.dtype, device=residual.device),
    )


def get_ar_seam_op(consts, kernel_kw, *, moe_fn_mode=None):
    """The op, or None when the small all-reduce is not armed. ``consts`` =
    (rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
    sinkhorn_repeat); ``kernel_kw`` = the unfused seam's options
    (gates_in_kernel=True, gates_max_tokens, token_block_size, small_block,
    fn_chunks). Call outside the trace.

    op(x_p, residual, post_mix, res_mix, fn_cat, hc_scale, hc_base, norm_w2d,
    rank, peers) -> (residual_cur, post_mix_cur, res_mix_cur, layer_input).
    """
    from torch_tpu._internal.pallas import jax_op

    from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME, build_ep_mesh, ep_token_replica_groups
    from vllm_torchtpu.layers.adapter import small_all_reduce_op as sar

    sar.prebuild()
    if sar._OP is None:
        return None
    name = "pallas::deepseek_v4_mhc_ar_seam_v1_" + "_".join(
        f"{v:.6g}".replace(".", "p").replace("-", "m") for v in consts
    ) + "_" + "_".join(f"{k}{v}" for k, v in sorted(kernel_kw.items()))
    name = name.replace("+", "")
    if moe_fn_mode is not None:
        # MoE side (TPU_DSV4_MHC_MOE_AR_SEAM): the fused EP op's partial.
        name += f"_moe_f{moe_fn_mode}"
    op = _cache.get(name)
    if op is not None:
        return op
    mesh = build_ep_mesh()
    groups = ep_token_replica_groups()
    cap = sar.max_rows()
    # The fused EP op hands over a partial up to max_rows(replica width)
    # (fused_moe_ep._small_ar_max_rows); the MoE-side seam must sum exactly
    # those buckets.
    assert cap == sar.max_rows(len(groups[0])), (cap, groups)
    spec = PartitionSpec(EP_AXIS_NAME)
    body = _body(
        groups, cap, EP_AXIS_NAME, consts, kernel_kw,
        reduced_above_cap=moe_fn_mode is not None, fn_mode=moe_fn_mode or "vmem",
    )

    def fn(
        x_p: jax.Array,
        residual: jax.Array,
        post: jax.Array,
        comb: jax.Array,
        fn_cat: jax.Array,
        hc_scale: jax.Array,
        hc_base: jax.Array,
        norm_w: jax.Array,
        rank: jax.Array,
        peers: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return jax.shard_map(
            body, mesh=mesh, in_specs=(spec,) * 10, out_specs=(spec,) * 4, check_vma=False
        )(x_p, residual, post, comb, fn_cat, hc_scale, hc_base, norm_w, rank, peers)

    op = jax_op(
        name.replace(".", "p"),
        fn,
        mesh=mesh,
        input_partition_specs=(spec,) * 10,
        output_partition_specs=(spec,) * 4,
    )
    op.register_fake(_fake)
    _cache[name] = op
    logger.info(
        "mHC AR+seam armed | groups=%s | cap=%d | form=%s | %s | moe_fn_mode=%s",
        groups, cap, all_reduce.group_form(groups), kernel_kw, moe_fn_mode,
    )
    return op


def moe_cap() -> int:
    """Token rows up to which the MoE side hands over a partial (the small
    all-reduce's cap for this core's group); the model gates on it."""
    from vllm_torchtpu.layers.adapter import small_all_reduce_op as sar

    return sar.max_rows()


def rank_peers():
    from vllm_torchtpu.layers.adapter import small_all_reduce_op as sar

    return sar._RANK, sar._PEERS


def tp_ranks():
    from vllm_torchtpu.layers.adapter import small_all_reduce_op as sar

    return sar._TP_RANKS
