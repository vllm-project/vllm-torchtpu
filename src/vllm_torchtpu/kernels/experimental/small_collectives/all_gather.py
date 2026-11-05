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
"""Low-latency Pallas ICI fused all-gather kernel for small token batches.

Pushes both ``hidden_states`` and ``router_logits`` directly to peer VMEM in
rank order behind a single hardware barrier semaphore.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.small_collectives.common import (
    LANES,
    barrier,
    normalize_rank_of_core,
    push,
    split_rank,
    wait_bytes_of,
)

# Distinct barrier group from all-reduce (5), reduce-scatter (7), and
# moe_lowtok fused collectives (9).
COLLECTIVE_ID = 8
# Each block's rows are padded to this multiple so DMA slices sit on whole
# sublane tiles.
SUBLANES = 8


def _check_width(width: int, name: str) -> None:
    if width % LANES:
        raise ValueError(f"{name} {width} is not a multiple of {LANES} lanes")


def small_all_gather_pair(
    x,
    logits,
    *,
    num_cores,
    axis="d",
    rank=None,
    my_rank=None,
    rank_of_core=None,
    dict_ids=None,
    collective_id=COLLECTIVE_ID,
):
    """Fused all-gather of ``(x, logits)`` across ``num_cores`` cores in rank order.

    Each core provides ``x`` ``[rows, hidden]`` and ``logits``
    ``[rows, num_experts]`` for its own rank (from ``my_rank``, ``rank_of_core``,
    or mesh index) and receives ``([num_cores * rows, hidden],
    [num_cores * rows, num_experts])`` with block ``r`` holding rank ``r``'s rows.
    """
    rows, hidden = x.shape
    l_rows, num_experts = logits.shape
    if rows != l_rows:
        raise ValueError(f"row mismatch: x has {rows}, logits has {l_rows}")
    _check_width(hidden, "hidden")
    _check_width(num_experts, "num_experts")
    n = num_cores
    if my_rank is not None and rank_of_core is not None:
        raise ValueError("pass at most one of my_rank and rank_of_core")
    order = normalize_rank_of_core(rank_of_core, n)
    if n == 1:
        return x, logits
    if dict_ids is None:
        dict_ids = rank is not None

    if rows % SUBLANES:
        padded_rows = pl.cdiv(rows, SUBLANES) * SUBLANES
        x_pad = jnp.pad(x, ((0, padded_rows - rows), (0, 0)))
        logits_pad = jnp.pad(logits, ((0, padded_rows - rows), (0, 0)))
        out_x, out_logits = small_all_gather_pair(
            x_pad,
            logits_pad,
            num_cores=n,
            axis=axis,
            rank=rank,
            my_rank=my_rank,
            rank_of_core=order if rank_of_core is not None else None,
            dict_ids=dict_ids,
            collective_id=collective_id,
        )
        out_x = out_x.reshape(n, padded_rows, hidden)[:, :rows, :].reshape(
            n * rows, hidden
        )
        out_logits = out_logits.reshape(
            n, padded_rows, num_experts
        )[:, :rows, :].reshape(n * rows, num_experts)
        return out_x, out_logits

    dynamic_my_rank = my_rank is not None

    def kernel(*refs):
        me, refs = split_rank(refs, rank is not None, axis)
        if dynamic_my_rank:
            my_rank_ref, *refs = refs
            mine = my_rank_ref[0]
        (
            x_ref,
            logits_ref,
            o_x_ref,
            o_logits_ref,
            x_slices,
            logits_slices,
            send_x_sem,
            recv_x_sem,
            send_logits_sem,
            recv_logits_sem,
        ) = refs
        barrier(n, axis, dict_ids)
        if dynamic_my_rank:
            for r in range(n):

                @pl.when(jnp.int32(r) == mine)
                def _(r=r):
                    for j in range(n):

                        @pl.when(jnp.int32(j) != me)
                        def _(j=j, r=r):
                            push(
                                x_ref,
                                x_slices.at[r],
                                send_x_sem,
                                recv_x_sem,
                                j,
                                axis,
                                dict_ids,
                            )
                            push(
                                logits_ref,
                                logits_slices.at[r],
                                send_logits_sem,
                                recv_logits_sem,
                                j,
                                axis,
                                dict_ids,
                            )

                    x_slices[pl.ds(r, 1)] = x_ref[...][None]
                    logits_slices[pl.ds(r, 1)] = logits_ref[...][None]
        else:
            for c in range(n):
                r = order[c]

                @pl.when(jnp.int32(c) == me)
                def _(c=c, r=r):
                    for j in range(n):

                        @pl.when(jnp.int32(j) != me)
                        def _(j=j, r=r):
                            push(
                                x_ref,
                                x_slices.at[r],
                                send_x_sem,
                                recv_x_sem,
                                j,
                                axis,
                                dict_ids,
                            )
                            push(
                                logits_ref,
                                logits_slices.at[r],
                                send_logits_sem,
                                recv_logits_sem,
                                j,
                                axis,
                                dict_ids,
                            )

                    x_slices[pl.ds(r, 1)] = x_ref[...][None]
                    logits_slices[pl.ds(r, 1)] = logits_ref[...][None]

        wait_bytes_of(x_slices.at[pl.ds(0, n - 1)], recv_x_sem)
        wait_bytes_of(logits_slices.at[pl.ds(0, n - 1)], recv_logits_sem)
        for r in range(n):
            o_x_ref[r] = x_slices[r]
            o_logits_ref[r] = logits_slices[r]
        wait_bytes_of(x_slices.at[pl.ds(0, n - 1)], send_x_sem)
        wait_bytes_of(logits_slices.at[pl.ds(0, n - 1)], send_logits_sem)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    in_specs = []
    args = []
    if rank is not None:
        in_specs.append(smem)
        args.append(rank.astype(jnp.int32))
    if dynamic_my_rank:
        in_specs.append(smem)
        args.append(my_rank.astype(jnp.int32))
    in_specs.extend([pl.BlockSpec(), pl.BlockSpec()])
    args.extend([x, logits])

    scratch = [
        pltpu.VMEM((n, rows, hidden), x.dtype),
        pltpu.VMEM((n, rows, num_experts), logits.dtype),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
    ]
    out_x, out_logits = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((n, rows, hidden), x.dtype),
            jax.ShapeDtypeStruct((n, rows, num_experts), logits.dtype),
        ),
        in_specs=in_specs,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=f"small_all_gather_pair_r{rows}",
    )(*args)
    return out_x.reshape(n * rows, hidden), out_logits.reshape(
        n * rows, num_experts
    )
