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
"""SparseCore Pallas DMA scatter into a paged KV cache."""

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.mla import kv_cache_utils


class Plan(NamedTuple):
    """How one step's tokens are spread over the vector subcores.

    Tokens are cut into descriptor blocks of `lanes_per_block` lanes.
    `cores_per_block` subcores share a block, each owning `lanes_per_core`
    consecutive lanes of it, and the chip walks `steps_per_core` rounds.
    """

    lanes_per_block: int  # == num_lanes; forced by the SparseCore tiling
    cores_per_block: int  # subcores sharing one descriptor block
    num_cores: int  # vector subcores on the chip
    steps_per_core: int  # pipeline grid

    @property
    def lanes_per_core(self) -> int:
        return self.lanes_per_block // self.cores_per_block

    @property
    def blocks_per_round(self) -> int:
        return self.num_cores // self.cores_per_block

    @property
    def total_tokens(self) -> int:
        """Tokens the grid covers; the inputs must be padded to this."""
        return self.steps_per_core * self.blocks_per_round * self.lanes_per_block


def split_plan(num_tokens: int, num_lanes: int, num_cores: int) -> Plan:
    """Choose the token -> subcore mapping for a step of ``num_tokens``."""
    assert num_tokens > 0, f"num_tokens must be positive (got {num_tokens})"
    num_blocks = pl.cdiv(num_tokens, num_lanes)

    # Subcores left over once every block has an owner split a block's lanes
    # instead. Powers of two only, so the lanes divide evenly and a subcore's
    # (block, lane range) is a shift and a mask of its index.
    spare = min(num_cores // num_blocks, num_lanes)
    cores_per_block = 1 << (int(spare).bit_length() - 1) if spare > 0 else 1

    return Plan(
        lanes_per_block=num_lanes,
        cores_per_block=cores_per_block,
        num_cores=num_cores,
        steps_per_core=pl.cdiv(num_blocks, num_cores // cores_per_block),
    )


def _scatter_kernel(
    dst_hbm_ref: Any,
    src_ref: Any,
    cache_ref: Any,
    sem: Any,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    plan: Plan,
    tc_tiled: bool,
) -> None:
    core_index = lax.axis_index((core_axis_name, subcore_axis_name))

    if tc_tiled:
        src_words = src_ref.bitcast(jnp.int32)
        cache_words = cache_ref.bitcast(jnp.int32)
    else:
        src_words = src_ref
        cache_words = cache_ref
    num_rows = cache_words.shape[0]

    block_slot = core_index // plan.cores_per_block
    lane_group = lax.rem(core_index, plan.cores_per_block)

    def block_index(step):
        return step * plan.blocks_per_round + block_slot

    def body(dst_ref, src_buf):
        dst_rows = dst_ref[...]
        copies = []
        for i in range(plan.lanes_per_block):
            dst = dst_rows[i]
            write = jnp.logical_and(dst >= 0, dst < num_rows)
            if plan.cores_per_block > 1:
                write = jnp.logical_and(write, lane_group == i // plan.lanes_per_core)
            row = jnp.clip(dst, 0, num_rows - 1)
            copy = pltpu.make_async_copy(
                src_buf.at[pl.ds(i, 1)],
                cache_words.at[pl.ds(row, 1)],
                sem.at[i],
            )
            copies.append((write, copy))

            @pl.when(write)
            def _(copy=copy):
                copy.start()

        for write, copy in copies:

            @pl.when(write)
            def _(copy=copy):
                copy.wait()

    pltpu.emit_pipeline(
        body,
        grid=(plan.steps_per_core,),
        tiling=pltpu.Tiling.SPARSE_CORE,
        in_specs=(
            pl.BlockSpec(
                (plan.lanes_per_block,),
                lambda r: (block_index(r),),
            ),
            pl.BlockSpec(
                (plan.lanes_per_block, src_words.shape[-1]),
                lambda r: (block_index(r), 0),
            ),
        ),
    )(dst_hbm_ref, src_words)


def scatter_rows(
    cache: jax.Array,
    src: jax.Array,
    dst_rows: jax.Array,
    name: str = "scatter_kv_cache",
) -> jax.Array:
    """Writes one row of `src` per token into `cache`.

    `cache.dtype` selects the layout: uint32 is the native SparseCore cache, one
    word row per token; uint8 is the TensorCore-tiled cache, where the kernel
    folds WORD_BYTES uint8 rows into one word row.

    Args:
      cache: KV cache in HBM, any leading dims; flattened to one row per token.
      src: [num_tokens, row_width] tokens, same dtype as `cache`.
      dst_rows: [num_tokens] destination row; negative or out of range drops the
        token.
      name: Pallas kernel name, as it appears in a profile.

    Returns:
      The updated cache, same shape and dtype as `cache`.
    """
    assert src.dtype == cache.dtype, (
        f"src {src.dtype} does not match cache {cache.dtype}"
    )
    tc_tiled = cache.dtype == jnp.uint8
    assert tc_tiled or cache.dtype == jnp.uint32, (
        f"cache must be uint8 (TensorCore) or uint32 (SparseCore), got {cache.dtype}"
    )
    tc_row_bytes = kv_cache_utils.WORD_BYTES * kv_cache_utils.TILE_LANE_BYTES
    assert not tc_tiled or src.shape[-1] == tc_row_bytes, (
        f"TensorCore tiling folds exactly {kv_cache_utils.WORD_BYTES} rows, so a"
        f" token must be {tc_row_bytes}B (got {src.shape[-1]}B)"
    )

    sc_info = pltpu.get_tpu_info().sparse_core
    assert sc_info is not None, "SparseCore info is missing."
    num_cores = sc_info.num_cores * sc_info.num_subcores
    plan = split_plan(src.shape[0], sc_info.num_lanes, num_cores)

    pad_tokens = plan.total_tokens - src.shape[0]
    if pad_tokens > 0:
        dst_rows = jnp.pad(
            dst_rows, (0, pad_tokens), constant_values=kv_cache_utils.SKIP_ROW
        )
        src = jnp.pad(src, ((0, pad_tokens), (0, 0)))

    # A TC-tiled token is WORD_BYTES uint8 rows the kernel folds back into one.
    row_width = kv_cache_utils.TILE_LANE_BYTES if tc_tiled else src.shape[-1]
    cache_ref = jax.new_ref(cache.reshape(-1, row_width))

    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    pl.kernel(
        functools.partial(
            _scatter_kernel,
            core_axis_name=vector_mesh.core_axis_name,
            subcore_axis_name=vector_mesh.subcore_axis_name,
            plan=plan,
            tc_tiled=tc_tiled,
        ),
        out_type=(),
        scratch_types=(pltpu.SemaphoreType.DMA((plan.lanes_per_block,)),),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=tc_tiled,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        mesh=vector_mesh,
        name=name,
    )(dst_rows, src.reshape(-1, row_width), cache_ref)

    return jax.freeze(cache_ref).reshape(cache.shape)


def _scatter_pair_kernel(
    dst_hbm_ref: Any,
    nope_src_ref: Any,
    rope_src_ref: Any,
    nope_cache_ref: Any,
    rope_cache_ref: Any,
    sem: Any,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    plan: Plan,
) -> None:
    """Copies one nope row and one rope row per lane to the same row index.

    The structure is the same as `_scatter_kernel` with `tc_tiled=False`. Each
    lane starts both of its copies before any lane waits, so the two DMAs of a
    token are in flight together. `sem` holds `2 * plan.lanes_per_block`
    semaphores: entry `i` belongs to the nope copy of lane `i`, and entry
    `plan.lanes_per_block + i` to its rope copy.
    """
    core_index = lax.axis_index((core_axis_name, subcore_axis_name))
    # Both caches hold one row per token slot, so a row index addresses the
    # same (page, slot) in each. `scatter_rows_pair` asserts equal row counts.
    num_rows = nope_cache_ref.shape[0]

    block_slot = core_index // plan.cores_per_block
    lane_group = lax.rem(core_index, plan.cores_per_block)

    def block_index(step):
        return step * plan.blocks_per_round + block_slot

    def body(dst_ref, nope_buf, rope_buf):
        dst_rows = dst_ref[...]
        copies = []
        for i in range(plan.lanes_per_block):
            dst = dst_rows[i]
            write = jnp.logical_and(dst >= 0, dst < num_rows)
            if plan.cores_per_block > 1:
                write = jnp.logical_and(write, lane_group == i // plan.lanes_per_core)
            row = jnp.clip(dst, 0, num_rows - 1)
            nope_copy = pltpu.make_async_copy(
                nope_buf.at[pl.ds(i, 1)],
                nope_cache_ref.at[pl.ds(row, 1)],
                sem.at[i],
            )
            rope_copy = pltpu.make_async_copy(
                rope_buf.at[pl.ds(i, 1)],
                rope_cache_ref.at[pl.ds(row, 1)],
                sem.at[plan.lanes_per_block + i],
            )
            copies.append((write, nope_copy, rope_copy))

            @pl.when(write)
            def _(nope_copy=nope_copy, rope_copy=rope_copy):
                nope_copy.start()
                rope_copy.start()

        for write, nope_copy, rope_copy in copies:

            @pl.when(write)
            def _(nope_copy=nope_copy, rope_copy=rope_copy):
                nope_copy.wait()
                rope_copy.wait()

    pltpu.emit_pipeline(
        body,
        grid=(plan.steps_per_core,),
        tiling=pltpu.Tiling.SPARSE_CORE,
        in_specs=(
            pl.BlockSpec(
                (plan.lanes_per_block,),
                lambda r: (block_index(r),),
            ),
            pl.BlockSpec(
                (plan.lanes_per_block, nope_src_ref.shape[-1]),
                lambda r: (block_index(r), 0),
            ),
            pl.BlockSpec(
                (plan.lanes_per_block, rope_src_ref.shape[-1]),
                lambda r: (block_index(r), 0),
            ),
        ),
    )(dst_hbm_ref, nope_src_ref, rope_src_ref)


def scatter_rows_pair(
    nope_cache: jax.Array,
    nope_src: jax.Array,
    rope_cache: jax.Array,
    rope_src: jax.Array,
    dst_rows: jax.Array,
    name: str = "scatter_nope_rope_sparsecore_cache",
) -> tuple[jax.Array, jax.Array]:
    """Writes one token row into each of two caches in one SparseCore program.

    The result is the same as two `scatter_rows` calls with the same
    `dst_rows`. One program instead of two removes, per call, the second
    program's launch and the idle time between the two programs, and it lets
    the nope and rope DMAs of a token run at the same time. Only the native
    SparseCore layout (uint32, one word row per token) is supported.

    Args:
      nope_cache: uint32 nope KV cache in HBM, any leading dims; flattened to
        one row of `nope_src.shape[-1]` words per token.
      nope_src: [num_tokens, nope_words] uint32 packed nope tokens.
      rope_cache: uint32 rope KV cache in HBM, any leading dims; flattened to
        one row of `rope_src.shape[-1]` words per token.
      rope_src: [num_tokens, rope_words] uint32 packed rope tokens.
      dst_rows: [num_tokens] destination row, the same in both caches. A
        negative or out-of-range row drops the token from both caches.
      name: Pallas kernel name, as it appears in a profile.

    Returns:
      The updated (nope_cache, rope_cache), same shapes and dtypes as the
      inputs.
    """
    for label, cache, src in (
        ("nope", nope_cache, nope_src),
        ("rope", rope_cache, rope_src),
    ):
        assert cache.dtype == jnp.uint32 and src.dtype == jnp.uint32, (
            f"{label}: the pair writer supports only the uint32 SparseCore"
            f" layout (cache {cache.dtype}, src {src.dtype})"
        )
    assert nope_src.shape[0] == rope_src.shape[0] == dst_rows.shape[0], (
        f"token counts differ: nope {nope_src.shape[0]}, rope"
        f" {rope_src.shape[0]}, dst_rows {dst_rows.shape[0]}"
    )
    nope_rows = nope_cache.size // nope_src.shape[-1]
    rope_rows = rope_cache.size // rope_src.shape[-1]
    assert nope_rows == rope_rows, (
        f"the caches hold different token counts ({nope_rows} nope rows,"
        f" {rope_rows} rope rows), so one dst_rows cannot address both"
    )

    sc_info = pltpu.get_tpu_info().sparse_core
    assert sc_info is not None, "SparseCore info is missing."
    num_cores = sc_info.num_cores * sc_info.num_subcores
    plan = split_plan(nope_src.shape[0], sc_info.num_lanes, num_cores)

    pad_tokens = plan.total_tokens - nope_src.shape[0]
    if pad_tokens > 0:
        dst_rows = jnp.pad(
            dst_rows, (0, pad_tokens), constant_values=kv_cache_utils.SKIP_ROW
        )
        nope_src = jnp.pad(nope_src, ((0, pad_tokens), (0, 0)))
        rope_src = jnp.pad(rope_src, ((0, pad_tokens), (0, 0)))

    nope_ref = jax.new_ref(nope_cache.reshape(-1, nope_src.shape[-1]))
    rope_ref = jax.new_ref(rope_cache.reshape(-1, rope_src.shape[-1]))

    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    pl.kernel(
        functools.partial(
            _scatter_pair_kernel,
            core_axis_name=vector_mesh.core_axis_name,
            subcore_axis_name=vector_mesh.subcore_axis_name,
            plan=plan,
        ),
        out_type=(),
        scratch_types=(pltpu.SemaphoreType.DMA((2 * plan.lanes_per_block,)),),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        mesh=vector_mesh,
        name=name,
    )(dst_rows, nope_src, rope_src, nope_ref, rope_ref)

    return (
        jax.freeze(nope_ref).reshape(nope_cache.shape),
        jax.freeze(rope_ref).reshape(rope_cache.shape),
    )
