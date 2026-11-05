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
"""Score all-to-all for the DCP indexer top-k merge as one Pallas kernel.

In the DCP indexer's ``a2a_merge`` stage, each of ``n`` cores holds flat local
top-k score words ``x`` for ``T = n * tokens_per_core`` tokens of width ``k``
(32-bit words, ``k`` a multiple of ``8 * LANES = 1024``). Core ``d`` needs
every sender's ``k``-word list for its token slice ``[d * tokens_per_core,
(d + 1) * tokens_per_core)``, arranged token-major and rank-major within each
token: logical shape ``[tokens_per_core, n * k]``, returned flat as
``[tokens_per_core * n * k]`` so both SparseCore endpoints (the local top-k
producer and the cooperative merge consumer) keep their native 1-D ``(1024,)``
HBM tiling without 2-D sublane relayouts.

Every core signals one barrier over the DCP axis, pushes one contiguous
``tokens_per_core * k`` block to each of the ``n - 1`` peers in a single round
of remote DMAs, copies its own block locally, and writes the token-major output
to HBM. Because no arithmetic touches the words, the flat output is bit-for-bit
identical to ``lax.all_to_all(x.reshape(T, k), axis, split_axis=0,
concat_axis=1, tiled=True).reshape(-1)``.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.small_collectives.common import (
    LANES,
    MESH,
    wait_bytes_of,
)

# Distinct from fused MoE (0), all-reduce (5), reduce-scatter (7),
# small all-gather (9), MLA send small all-to-all (11), sparse MLA send (13),
# rotated collectives (15), prefill (17, 19), and online-LSE combine (21).
COLLECTIVE_ID = 23

# Default scoped VMEM cap when scratch is used so XLA keeps most of VMEM free
# for surrounding weight prefetches across the custom call.
DEFAULT_MIN_VMEM_LIMIT_BYTES = 16 * 1024 * 1024


def reference(blocks, *, tokens_per_core: int | None = None, k: int | None = None):
    """Oracle for ``small_indexer_score_all_to_all``.

    Accepts either a 3-D ``[cores, cores * tokens_per_core, k]`` array (returning
    ``[cores, tokens_per_core, cores * k]``) or a flat 2-D
    ``[cores, cores * tokens_per_core * k]`` array with ``tokens_per_core`` and
    ``k`` specified (returning ``[cores, tokens_per_core * cores * k]``),
    matching ``lax.all_to_all(scores_2d, axis, split_axis=0, concat_axis=1,
    tiled=True)``.
    """
    n = blocks.shape[0]
    is_3d = blocks.ndim == 3
    if is_3d:
        t_total, k_inferred = blocks.shape[1], blocks.shape[2]
        assert t_total % n == 0, blocks.shape
        tokens_per_core = t_total // n if tokens_per_core is None else tokens_per_core
        k = k_inferred if k is None else k
    else:
        if tokens_per_core is None or k is None:
            raise ValueError("tokens_per_core and k are required when blocks is 2-D")
    shaped = blocks.reshape(n, n, tokens_per_core, k)
    # Destination d gets shaped[s, d, t, :] at out[d, t, s, :].
    transposed = shaped.transpose(1, 2, 0, 3).copy()
    if is_3d:
        return transposed.reshape(n, tokens_per_core, n * k)
    return transposed.reshape(n, tokens_per_core * n * k)


def supports_indexer_score_all_to_all(
    num_tokens: int, dcp_size: int, k: int, max_tokens: int
) -> bool:
    """Whether ``(num_tokens, dcp_size, k)`` is eligible for the kernel."""
    return (
        dcp_size > 1
        and num_tokens > 0
        and num_tokens <= max_tokens
        and num_tokens % dcp_size == 0
        and k > 0
        and k % (8 * LANES) == 0
    )


def small_indexer_score_all_to_all(
    x: jax.Array,
    *,
    num_cores: int,
    tokens_per_core: int,
    k: int,
    axis: str,
    rank: jax.Array | None = None,
    lane_axis: str | None = None,
    lane: jax.Array | None = None,
    collective_id: int = COLLECTIVE_ID,
    hbm_operand: bool = False,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:
    """Flat 1-D score exchange for ``[T, k] -> [T // n, n * k]``.

    Args:
      x: 1-D ``[num_cores * tokens_per_core * k]`` 32-bit array on this core.
      num_cores: size of ``axis``.
      tokens_per_core: ``T // num_cores`` (>= 1).
      k: candidates per token, a multiple of ``8 * LANES`` (1024).
      axis: mesh axis name to exchange across.
      rank: optional ``int32[1]`` index on ``axis`` passed as data.
      lane_axis: optional orthogonal mesh axis name (at most one).
      lane: optional ``int32[1]`` index on ``lane_axis`` passed as data.
      collective_id: barrier semaphore ID.
      hbm_operand: constrain ``x`` to HBM so XLA does not stage SparseCore's
        output into VMEM ahead of the call.
      vmem_limit_bytes: optional scoped VMEM ceiling for the Pallas call.
    """
    if x.ndim != 1:
        raise ValueError(
            f"small_indexer_score_all_to_all takes a flat 1-D block, got {x.shape}"
        )
    if x.dtype.itemsize != 4:
        raise ValueError(
            f"small_indexer_score_all_to_all expects a 32-bit dtype, got {x.dtype}"
        )
    if num_cores <= 1:
        raise ValueError(f"num_cores must be > 1, got {num_cores}")
    if tokens_per_core <= 0:
        raise ValueError(f"tokens_per_core must be > 0, got {tokens_per_core}")
    if k <= 0 or k % (8 * LANES) != 0:
        raise ValueError(f"k={k} must be a positive multiple of {8 * LANES}")
    expected = num_cores * tokens_per_core * k
    if x.shape[0] != expected:
        raise ValueError(
            f"x length {x.shape[0]} != num_cores({num_cores}) * "
            f"tokens_per_core({tokens_per_core}) * k({k}) = {expected}"
        )
    if lane is not None and lane_axis is None:
        raise ValueError("lane requires lane_axis to be specified")

    n = num_cores
    k_rows = k // LANES  # multiple of 8
    block_rows = tokens_per_core * k_rows  # multiple of 8
    total_rows = n * block_rows
    data_rank = rank is not None
    data_lane = lane_axis is not None and lane is not None
    direct_hbm = tokens_per_core == 1

    def kernel(*refs):
        refs_list = list(refs)
        me = refs_list.pop(0)[0] if data_rank else lax.axis_index(axis)
        if lane_axis is None:
            my_lane = None
        elif data_lane:
            my_lane = refs_list.pop(0)[0]
        else:
            my_lane = lax.axis_index(lane_axis)

        if direct_hbm:
            x_ref, o_ref, send_sem, recv_sem, local_sem = refs_list
        else:
            (
                x_ref,
                o_ref,
                recv_ref,
                out_ref,
                send_sem,
                recv_sem,
                local_sem,
                out_sem,
            ) = refs_list

        def peer(j):
            ids = {axis: jnp.int32(j)}
            if lane_axis is not None:
                ids[lane_axis] = my_lane
            return ids

        sem = pltpu.get_barrier_semaphore()
        for j in range(n):
            pl.semaphore_signal(sem, 1, device_id=peer(j), device_id_type=MESH)
        pl.semaphore_wait(sem, n)

        if direct_hbm:
            mine = o_ref.at[pl.ds(pl.multiple_of(me * block_rows, 8), block_rows)]
            for j in range(n):

                @pl.when(jnp.int32(j) != me)
                def _(j=j):
                    pltpu.make_async_remote_copy(
                        src_ref=x_ref.at[pl.ds(j * block_rows, block_rows)],
                        dst_ref=mine,
                        send_sem=send_sem,
                        recv_sem=recv_sem,
                        device_id=peer(j),
                        device_id_type=MESH,
                    ).start()

            local = pltpu.make_async_copy(
                x_ref.at[pl.ds(pl.multiple_of(me * block_rows, 8), block_rows)],
                mine,
                local_sem,
            )
            local.start()
            others = o_ref.at[pl.ds(0, (n - 1) * block_rows)]
            wait_bytes_of(others, recv_sem)
            local.wait()
            wait_bytes_of(others, send_sem)
            return

        mine = recv_ref.at[me]
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                pltpu.make_async_remote_copy(
                    src_ref=x_ref.at[pl.ds(j * block_rows, block_rows)],
                    dst_ref=mine,
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=peer(j),
                    device_id_type=MESH,
                ).start()

        local = pltpu.make_async_copy(
            x_ref.at[pl.ds(pl.multiple_of(me * block_rows, 8), block_rows)],
            mine,
            local_sem,
        )
        local.start()

        others = recv_ref.at[pl.ds(0, n - 1)]
        wait_bytes_of(others, recv_sem)
        local.wait()

        token_span = n * k_rows
        for t in range(tokens_per_core):
            src_slice = pl.ds(t * k_rows, k_rows)
            for s in range(n):
                out_ref[t, pl.ds(s * k_rows, k_rows), :] = recv_ref[s, src_slice, :]
            pltpu.make_async_copy(
                out_ref.at[t],
                o_ref.at[pl.ds(t * token_span, token_span)],
                out_sem,
            ).start()

        wait_bytes_of(o_ref, out_sem)
        wait_bytes_of(others, send_sem)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    any_ = pl.BlockSpec(memory_space=pl.ANY)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    scalars = []
    if data_rank:
        scalars.append(jnp.reshape(rank, (-1,))[:1].astype(jnp.int32))
    if data_lane:
        scalars.append(jnp.reshape(lane, (-1,))[:1].astype(jnp.int32))

    # Reshaping 1-D (1024,)-tiled 32-bit words to [total_rows, 128] (8, 128) is
    # a pure bitcast on TPU.
    block = x.reshape(total_rows, LANES)
    if hbm_operand:
        block = pltpu.with_memory_space_constraint(block, pltpu.HBM)

    if direct_hbm:
        scratch = [pltpu.SemaphoreType.DMA] * 3
        compiler_params = pltpu.CompilerParams(collective_id=collective_id)
    else:
        scratch = [
            pltpu.VMEM((n, block_rows, LANES), x.dtype),
            pltpu.VMEM((tokens_per_core, n * k_rows, LANES), x.dtype),
            *[pltpu.SemaphoreType.DMA] * 4,
        ]
        scratch_bytes = 2 * total_rows * LANES * x.dtype.itemsize
        eff_vmem_limit = vmem_limit_bytes
        if eff_vmem_limit is None:
            eff_vmem_limit = max(DEFAULT_MIN_VMEM_LIMIT_BYTES, 4 * scratch_bytes)
        compiler_params = pltpu.CompilerParams(
            collective_id=collective_id,
            vmem_limit_bytes=eff_vmem_limit,
        )

    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((total_rows, LANES), x.dtype),
        in_specs=[smem] * len(scalars) + [hbm if hbm_operand else any_],
        out_specs=hbm,
        scratch_shapes=scratch,
        compiler_params=compiler_params,
        name=f"small_idx_score_a2a_n{n}_t{tokens_per_core}_k{k}",
    )(*scalars, block)
    return out.reshape(expected)
