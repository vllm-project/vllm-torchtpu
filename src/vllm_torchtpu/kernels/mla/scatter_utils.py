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


def _sparse_core_plan(num_tokens: int) -> Plan:
    """`split_plan` for this chip's SparseCore vector subcores."""
    sc_info = pltpu.get_tpu_info().sparse_core
    assert sc_info is not None, "SparseCore info is missing."
    num_cores = sc_info.num_cores * sc_info.num_subcores
    return split_plan(num_tokens, sc_info.num_lanes, num_cores)


def padded_num_tokens(num_tokens: int) -> int:
    """Rows that `scatter_rows` reads for a step of `num_tokens` tokens.

    A caller that builds `src` itself can pad it to this size in the same op
    that produces it. `scatter_rows` then pads only `dst_rows`, which is
    layer-invariant within a step.
    """
    return _sparse_core_plan(num_tokens).total_tokens


def pad_to_plan(
    src: jax.Array, dst_rows: jax.Array, total_tokens: int
) -> tuple[jax.Array, jax.Array]:
    """Pads `src` with zero rows and `dst_rows` with SKIP_ROW to `total_tokens`.

    The two inputs are padded independently, so `src` may already have
    `total_tokens` rows (see `kv_cache_utils.pack_tokens(num_rows=...)`) while
    `dst_rows` still has one entry per real token. A padded row is never
    written, because its destination is SKIP_ROW.

    Args:
      src: `[n, row_width]` rows, where `n` is the real token count or
        `total_tokens`.
      dst_rows: `[num_tokens]` destination rows of the real tokens.
      total_tokens: rows the scatter grid covers (`Plan.total_tokens`).

    Returns:
      `(src, dst_rows)`, both with `total_tokens` rows.
    """
    num_tokens = dst_rows.shape[0]
    assert src.shape[0] in (num_tokens, total_tokens), (
        f"src has {src.shape[0]} rows; expected {num_tokens} tokens or the"
        f" padded {total_tokens}"
    )
    if total_tokens > num_tokens:
        dst_rows = jnp.pad(
            dst_rows,
            (0, total_tokens - num_tokens),
            constant_values=kv_cache_utils.SKIP_ROW,
        )
    if total_tokens > src.shape[0]:
        src = jnp.pad(src, ((0, total_tokens - src.shape[0]), (0, 0)))
    return src, dst_rows


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
    # The plan is sized from the real token count (`dst_rows`), because `src`
    # may arrive already padded to the plan size.
    plan = _sparse_core_plan(dst_rows.shape[0])
    src, dst_rows = pad_to_plan(src, dst_rows, plan.total_tokens)

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
    src_a_ref: Any,
    src_b_ref: Any,
    cache_a_ref: Any,
    cache_b_ref: Any,
    sem: Any,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    plan: Plan,
) -> None:
    """`_scatter_kernel` for two uint32 caches that share `dst_rows`.

    Lane `i` of a block issues two DMAs: row `i` of `src_a` to row `dst` of
    `cache_a` on semaphore `i`, and row `i` of `src_b` to row `dst` of
    `cache_b` on semaphore `lanes_per_block + i`. Both DMAs use the same write
    predicate as `_scatter_kernel`, so each cache receives exactly the rows
    that a separate `scatter_rows` call would write.
    """
    core_index = lax.axis_index((core_axis_name, subcore_axis_name))
    num_rows_a = cache_a_ref.shape[0]
    num_rows_b = cache_b_ref.shape[0]
    lanes = plan.lanes_per_block

    block_slot = core_index // plan.cores_per_block
    lane_group = lax.rem(core_index, plan.cores_per_block)

    def block_index(step):
        return step * plan.blocks_per_round + block_slot

    def body(dst_ref, src_a_buf, src_b_buf):
        dst_rows = dst_ref[...]
        copies = []
        for i in range(lanes):
            dst = dst_rows[i]
            owned = dst >= 0
            if plan.cores_per_block > 1:
                owned = jnp.logical_and(owned, lane_group == i // plan.lanes_per_core)
            for src_buf, cache, num_rows, sem_index in (
                (src_a_buf, cache_a_ref, num_rows_a, i),
                (src_b_buf, cache_b_ref, num_rows_b, lanes + i),
            ):
                write = jnp.logical_and(owned, dst < num_rows)
                copy = pltpu.make_async_copy(
                    src_buf.at[pl.ds(i, 1)],
                    cache.at[pl.ds(jnp.clip(dst, 0, num_rows - 1), 1)],
                    sem.at[sem_index],
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
            pl.BlockSpec((lanes,), lambda r: (block_index(r),)),
            pl.BlockSpec((lanes, src_a_ref.shape[-1]), lambda r: (block_index(r), 0)),
            pl.BlockSpec((lanes, src_b_ref.shape[-1]), lambda r: (block_index(r), 0)),
        ),
    )(dst_hbm_ref, src_a_ref, src_b_ref)


def scatter_rows_pair(
    cache_a: jax.Array,
    src_a: jax.Array,
    cache_b: jax.Array,
    src_b: jax.Array,
    dst_rows: jax.Array,
    name: str = "scatter_kv_cache_pair",
) -> tuple[jax.Array, jax.Array]:
    """Writes row `t` of `src_a` and of `src_b` to row `dst_rows[t]` of each cache.

    The result equals `scatter_rows(cache_a, src_a, dst_rows)` followed by
    `scatter_rows(cache_b, src_b, dst_rows)`. The difference is that one
    SparseCore program issues both writes. The MLA KV cache write stores every
    token in two caches (nope and rope) at the same destination row, so one
    program per layer saves the fixed cost of a second SparseCore launch: the
    launch preamble, the idle gap between two programs, and one wait for the
    input DMA of `dst_rows`.

    Only the native SparseCore layout (uint32 caches, one word row per token)
    is supported. The TensorCore-tiled uint8 layout folds several rows into one
    word row and stays on `scatter_rows`.

    Args:
      cache_a: first uint32 KV cache in HBM, any leading dims; flattened to one
        row per token.
      src_a: `[n, row_width_a]` uint32 rows for `cache_a`. `n` is the real token
        count or `padded_num_tokens(num_tokens)`.
      cache_b: second uint32 KV cache, same contract as `cache_a`. The row width
        may differ from `cache_a`, for example 16 words of rope and 128 words of
        nope per token.
      src_b: `[n, row_width_b]` uint32 rows for `cache_b`.
      dst_rows: `[num_tokens]` destination row, shared by both caches. A
        negative value, or a value outside a cache, drops that token for that
        cache.
      name: Pallas kernel name, as it appears in a profile.

    Returns:
      `(cache_a, cache_b)` updated, with the input shapes and dtypes.
    """
    for label, cache, src in (("a", cache_a, src_a), ("b", cache_b, src_b)):
        assert cache.dtype == jnp.uint32 and src.dtype == jnp.uint32, (
            f"scatter_rows_pair supports uint32 (SparseCore) caches only; cache_{label}"
            f" is {cache.dtype}, src_{label} is {src.dtype}"
        )
        assert cache.size % src.shape[-1] == 0, (
            f"cache_{label} of {cache.size} words is not a whole number of"
            f" {src.shape[-1]}-word rows"
        )

    sc_info = pltpu.get_tpu_info().sparse_core
    assert sc_info is not None, "SparseCore info is missing."
    # Sized from the real token count, as in `scatter_rows`. Both sources use
    # the same plan, so the padded `dst_rows` is shared.
    plan = _sparse_core_plan(dst_rows.shape[0])
    src_a, padded_dst_rows = pad_to_plan(src_a, dst_rows, plan.total_tokens)
    src_b, _ = pad_to_plan(src_b, dst_rows, plan.total_tokens)

    cache_a_ref = jax.new_ref(cache_a.reshape(-1, src_a.shape[-1]))
    cache_b_ref = jax.new_ref(cache_b.reshape(-1, src_b.shape[-1]))

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
    )(padded_dst_rows, src_a, src_b, cache_a_ref, cache_b_ref)

    return (
        jax.freeze(cache_a_ref).reshape(cache_a.shape),
        jax.freeze(cache_b_ref).reshape(cache_b.shape),
    )
