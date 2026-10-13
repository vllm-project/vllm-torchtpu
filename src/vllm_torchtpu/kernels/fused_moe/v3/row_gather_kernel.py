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
"""Row gathers on the sparse cores, for the combine's arrival rows.

Every one of the chip's vector subcores owns a contiguous run of the output
rows and streams them with index-driven DMAs through a ring of VMEM
blocks, keeping `ring` gathers in flight while finished blocks drain. The
fp8 rows are moved as 32-bit words (nothing is unpacked), and each row's
f32 scale is gathered from the kernel's arrival scale rows by a second
index-driven DMA per block.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.fused_moe.v3 import device
from vllm_torchtpu.kernels.fused_moe.v3.config import (
    GATHER_BLOCK_ROWS_DEFAULT,
    GATHER_RING_DEFAULT,
)
from vllm_torchtpu.kernels.fused_moe.v3.layout import LANES, WORD_BYTES

CORE_AXIS = "core"
SUBCORE_AXIS = "subcore"
# 1-D int32 slices on the sparse core start at multiples of this many words.
SLICE_ALIGN_WORDS = 8
# The widest row the default ring stages at its full depth and block: 4096
# lanes. The staging budget is that many lane blocks by the default ring
# by the default block rows.
MAX_ROW_BLOCKS_AT_FULL_RING = 32
# The arrival rows array: [rows, lane blocks, LANES].
ROWS_RANK = 3
# The sort program (gather_rows_prefix) is a static schedule over the
# blocks of one subcore's run, so its size follows the table, and a
# sparse-core program holds at most SORT_PROGRAM_BUNDLE_LIMIT bundles (the
# compiler's xla_sc_tile_overlays_size). Compiled without the device across
# widths and tables (measured offline): a row of a power of two
# words (1024, 2048, 4096, 8192 fp8 values) moves as one copy per block, 23
# bundles a block on 8 of prologue whatever the table (3872 for 168 blocks
# a subcore, 2032 for 88); any other width costs about 92 bundles a block
# and 1.25 a row (2560 wide, 16-row blocks: 8578 for 88 blocks, 16,238 for
# 168; 32-row blocks: 5146 for 44; 1536 and 3584 wide: 18,797 for 168
# blocks of 16). The estimate stands on the largest costs seen.
SORT_PROGRAM_BUNDLE_LIMIT = 8144
SORT_BUNDLES_FIXED = 8
SORT_BUNDLES_PER_BLOCK = 23
SORT_BUNDLES_PER_ROW_BLOCK = 92
SORT_BUNDLES_PER_ROW = 1.25
COMPILER_PARAMS = dict(
    use_tc_tiling_on_sc=True, needs_layout_passes=True, disable_bounds_checks=True
)


def _jax_private():
    """jax's private core-map lowering and ref constructor. Both are bound
    here so a jax that moved them fails with a message naming the versions
    this was gated on, not with an AttributeError inside a kernel."""
    try:
        from jax._src import core as jax_core
        from jax._src import tree_util
        from jax._src.pallas import core as pallas_core

        return (
            jax_core.new_ref,
            jax_core.MemorySpace,
            tree_util,
            pallas_core.core_map,
            pallas_core._convert_out_shape_to_aval,
        )
    except (ImportError, AttributeError) as e:
        raise NotImplementedError(
            "the sparse-core programs go through jax's private core-map "
            "lowering (jax._src.pallas.core.core_map, jax._src.core.new_ref) "
            f"and jax {device.jax_version()} no longer carries it as "
            "written"
        ) from e


def _empty_output_ref(out_type):
    new_ref, memory_space_type, _, _, out_shape_to_aval = _jax_private()
    # A Pallas memory-space aval; the private helper is typed AbstractValue.
    aval: Any = out_shape_to_aval(out_type)
    memory_space = (
        None if isinstance(aval.memory_space, memory_space_type) else aval.memory_space
    )
    value = lax.empty(aval.shape, aval.dtype, out_sharding=aval.sharding)
    return new_ref(value, memory_space=memory_space)


def sparse_core_program(
    body, *, out_types, mesh, scratch_types, compiler_params, name, in_place_outputs=0
):
    """Wraps `body(*operand_refs, *out_refs, *scratch_refs)` as a jitted
    function over the vector-subcore mesh, returning the outputs as a tuple
    in out_types order. With in_place_outputs, that many trailing operands
    ARE the outputs: taken with their contents and updated in place."""
    new_ref, _, tree_util, core_map, _ = _jax_private()
    assert in_place_outputs in (0, len(out_types)), (in_place_outputs, out_types)

    @jax.jit
    def run(*operands):
        n_in = len(operands) - in_place_outputs
        operand_refs = tree_util.tree_map(new_ref, operands[:n_in])
        if in_place_outputs:
            out_refs = [new_ref(v) for v in operands[n_in:]]
        else:
            out_refs = [_empty_output_ref(t) for t in out_types]

        @core_map(
            mesh,
            scratch_shapes=scratch_types,
            compiler_params=compiler_params,
            name=name,
        )
        def _(*scratch_refs):
            return body(*operand_refs, *out_refs, *scratch_refs)

        return tuple(ref[...] for ref in out_refs)

    return run


def subcore_mesh():
    """The chip's vector-subcore mesh and its subcore count."""
    sparse_core = pltpu.get_tpu_info().sparse_core
    if sparse_core is None:
        raise ValueError(
            "this chip has no sparse cores. The pallas mode's "
            "gather and scatter programs run on them"
        )
    mesh = plsc.VectorSubcoreMesh(
        num_cores=sparse_core.num_cores,
        num_subcores=sparse_core.num_subcores,
        core_axis_name=CORE_AXIS,
        subcore_axis_name=SUBCORE_AXIS,
    )
    return mesh, sparse_core.num_cores * sparse_core.num_subcores


def ring_for(
    lane_blocks,
    itemsize=1,
    ring=GATHER_RING_DEFAULT,
    block_rows=GATHER_BLOCK_ROWS_DEFAULT,
):
    """The gather's ring depth for a row width: `ring`, halved while the
    ring's staging (`block_rows` rows of `lane_blocks` fp8 lane blocks,
    the same bytes in any element type, `ring` deep) exceeds the default
    geometry's at MAX_ROW_BLOCKS_AT_FULL_RING blocks, so it stays inside
    the sparse core's tile memory. The block height is the caller's: a
    block is a whole number of the sparse core's 16-lane vectors."""
    budget = (
        MAX_ROW_BLOCKS_AT_FULL_RING * GATHER_RING_DEFAULT * GATHER_BLOCK_ROWS_DEFAULT
    )
    while ring > 1 and lane_blocks * itemsize * ring * block_rows > budget:
        ring //= 2
    return ring


def rows_gatherable(lane_blocks, dtype=jnp.float8_e4m3fn):
    """Whether rows of `lane_blocks` lane blocks of `dtype` can travel
    through the gather: it moves a row as 32-bit words, so the lane blocks
    must be a whole number of word blocks (four fp8, two bf16, eight fp4).
    The layer keeps the per-row fetch for widths that are not."""
    return lane_blocks % (WORD_BYTES * 8 // jnp.finfo(dtype).bits) == 0


def row_words_whole_copy(lane_blocks, dtype=jnp.float8_e4m3fn):
    """Whether a block of rows of `lane_blocks` lane blocks of `dtype`
    moves as one copy: the row's words are a power of two."""
    words = lane_blocks * LANES // (WORD_BYTES * 8 // jnp.finfo(dtype).bits)
    return words & (words - 1) == 0


def sort_program_bundles(
    rows_per_subcore,
    lane_blocks,
    dtype=jnp.float8_e4m3fn,
    block_rows=GATHER_BLOCK_ROWS_DEFAULT,
):
    """The sort program's estimated bundles for a run of `rows_per_subcore`
    rows of `lane_blocks` blocks of `dtype` in `block_rows`-row blocks."""
    blocks = -(-rows_per_subcore // block_rows)
    if row_words_whole_copy(lane_blocks, dtype):
        return SORT_BUNDLES_FIXED + blocks * SORT_BUNDLES_PER_BLOCK
    return SORT_BUNDLES_FIXED + int(
        blocks * (SORT_BUNDLES_PER_ROW_BLOCK + SORT_BUNDLES_PER_ROW * block_rows) + 0.5
    )


def sort_program_fits(
    rows_per_subcore,
    lane_blocks,
    dtype=jnp.float8_e4m3fn,
    block_rows=GATHER_BLOCK_ROWS_DEFAULT,
):
    """Whether the sort program of such a run compiles: its estimated
    bundles within the sparse core's program size."""
    return (
        sort_program_bundles(rows_per_subcore, lane_blocks, dtype, block_rows)
        <= SORT_PROGRAM_BUNDLE_LIMIT
    )


def gather_rows_multiple(block_rows=GATHER_BLOCK_ROWS_DEFAULT):
    """Rows a gather splits evenly: one block per vector subcore."""
    return subcore_mesh()[1] * block_rows


def rows_per_subcore(n_rows, n_subcores, block_rows):
    """A subcore's run of the rows: whole blocks, aligned to the slice."""
    if n_rows % n_subcores:
        raise ValueError(
            f"{n_rows} rows do not split evenly over the chip's "
            f"{n_subcores} vector subcores"
        )
    run = n_rows // n_subcores
    if run % block_rows:
        raise ValueError(
            f"a subcore's {run} rows are not a whole number of {block_rows}-row blocks"
        )
    if run % SLICE_ALIGN_WORDS or block_rows % SLICE_ALIGN_WORDS:
        raise ValueError(
            f"a subcore's run ({run} rows) and the block "
            f"({block_rows}) have to be multiples of "
            f"{SLICE_ALIGN_WORDS}, the sparse core's slice "
            "alignment"
        )
    return run


def _gather_body(*refs, run, block_rows, ring, with_scales):
    if with_scales:
        (
            index_hbm,
            rows_hbm,
            scale_words_hbm,
            out_rows_hbm,
            out_scales_hbm,
            index_vmem,
            rows_vmem,
            scales_vmem,
            row_gather_sems,
            scale_gather_sems,
            row_store_sems,
            scale_store_sems,
        ) = refs
    else:
        (
            index_hbm,
            rows_hbm,
            out_rows_hbm,
            index_vmem,
            rows_vmem,
            row_gather_sems,
            row_store_sems,
            scale_store_sems,
        ) = refs
    rows_hbm = rows_hbm.bitcast(jnp.int32)  # the rows as words
    out_rows_hbm = out_rows_hbm.bitcast(jnp.int32)
    subcore = lax.axis_index((CORE_AXIS, SUBCORE_AXIS))
    base = subcore * run
    n_blocks = run // block_rows
    lanes = pltpu.get_tpu_info().sparse_core.num_lanes
    lax.broadcasted_iota(jnp.int32, (lanes,), 0)

    fetch = pltpu.make_async_copy(
        index_hbm.at[pl.ds(base, run)], index_vmem, scale_store_sems.at[ring]
    )
    fetch.start()
    fetch.wait()

    def gather_rows(b, slot):
        return pltpu.make_async_copy(
            rows_hbm.at[index_vmem.at[pl.ds(b * block_rows, block_rows)]],
            rows_vmem.at[slot],
            row_gather_sems.at[slot],
        )

    def gather_scales(b, slot):
        """The block's scales, a word per row by the same index."""
        return pltpu.make_async_copy(
            scale_words_hbm.at[index_vmem.at[pl.ds(b * block_rows, block_rows)]],
            scales_vmem.at[slot],
            scale_gather_sems.at[slot],
        )

    def store_rows(b, slot):
        return pltpu.make_async_copy(
            rows_vmem.at[slot],
            out_rows_hbm.at[pl.ds(base + b * block_rows, block_rows)],
            row_store_sems.at[slot],
        )

    def store_scales(b, slot):
        return pltpu.make_async_copy(
            scales_vmem.at[slot],
            out_scales_hbm.at[pl.ds(base + b * block_rows, block_rows)],
            scale_store_sems.at[slot],
        )

    for b in range(min(ring, n_blocks)):
        gather_rows(b, b).start()
        if with_scales:
            gather_scales(b, b).start()
    for b in range(n_blocks):
        slot = b % ring
        gather_rows(b, slot).wait()
        store_rows(b, slot).start()
        if with_scales:
            gather_scales(b, slot).wait()
            store_scales(b, slot).start()
        if b + ring < n_blocks:
            store_rows(b, slot).wait()
            if with_scales:
                store_scales(b, slot).wait()
            gather_rows(b + ring, slot).start()
            if with_scales:
                gather_scales(b + ring, slot).start()
    for b in range(max(0, n_blocks - ring), n_blocks):
        store_rows(b, b % ring).wait()
        if with_scales:
            store_scales(b, b % ring).wait()


def _gather_prefix_body(
    index_hbm,
    rows_hbm,
    meta_hbm,
    out_rows_hbm,
    index_vmem,
    rows_vmem,
    meta_vmem,
    row_gather_sems,
    row_store_sems,
    misc_sems,
    *,
    run,
    block_rows,
    ring,
):
    """gather_rows_prefix's body: every subcore gathers `run_live` rows
    (meta_hbm[0], a multiple of block_rows at most `run`), its run starting
    at subcore * run_live, so the first n_subcores * run_live output rows
    are rows[index] and the rest are left as they are."""
    rows_hbm = rows_hbm.bitcast(jnp.int32)  # the rows as words
    out_rows_hbm = out_rows_hbm.bitcast(jnp.int32)
    subcore = lax.axis_index((CORE_AXIS, SUBCORE_AXIS))
    meta = pltpu.make_async_copy(meta_hbm, meta_vmem, misc_sems.at[0])
    meta.start()
    meta.wait()
    run_live = meta_vmem[...][0]  # the vector loaded, its first element
    base = pl.multiple_of(subcore * run_live, block_rows)
    n_blocks = run_live // block_rows
    n_blocks_max = run // block_rows
    # The subcore's index slice at its full static length: base + run never
    # passes the index's end (base <= subcore * run).
    fetch = pltpu.make_async_copy(
        index_hbm.at[pl.ds(base, run)], index_vmem, misc_sems.at[1]
    )
    fetch.start()
    fetch.wait()

    def gather_rows(b, slot):
        return pltpu.make_async_copy(
            rows_hbm.at[index_vmem.at[pl.ds(b * block_rows, block_rows)]],
            rows_vmem.at[slot],
            row_gather_sems.at[slot],
        )

    def store_rows(b, slot):
        return pltpu.make_async_copy(
            rows_vmem.at[slot],
            out_rows_hbm.at[
                pl.ds(pl.multiple_of(base + b * block_rows, block_rows), block_rows)
            ],
            row_store_sems.at[slot],
        )

    # The static schedule of the full run, each block predicated on the
    # live count: every in-VMEM offset stays static.
    for b in range(min(ring, n_blocks_max)):

        @pl.when(b < n_blocks)
        def _(b=b):
            gather_rows(b, b).start()

    for b in range(n_blocks_max):
        slot = b % ring

        @pl.when(b < n_blocks)
        def _(b=b, slot=slot):
            gather_rows(b, slot).wait()
            store_rows(b, slot).start()

        if b + ring < n_blocks_max:

            @pl.when(b + ring < n_blocks)
            def _(b=b, slot=slot):
                store_rows(b, slot).wait()
                gather_rows(b + ring, slot).start()

    for b in range(n_blocks_max):

        @pl.when(jnp.logical_and(b >= n_blocks - ring, b < n_blocks))
        def _(b=b):
            store_rows(b, b % ring).wait()


def prefix_run(n_live, n_rows, block_rows=GATHER_BLOCK_ROWS_DEFAULT):
    """Rows per subcore for gather_rows_prefix covering `n_live` rows (a
    traced int32): a whole number of block_rows, at most the static run of
    an `n_rows` gather."""
    _, n_subcores = subcore_mesh()
    run = rows_per_subcore(n_rows, n_subcores, block_rows)
    per = -(-n_live // n_subcores)
    per = -(-per // block_rows) * block_rows
    return jnp.minimum(per, run).astype(jnp.int32)


def gather_rows_prefix(
    rows,
    index,
    run_live,
    *,
    block_rows=GATHER_BLOCK_ROWS_DEFAULT,
    ring=GATHER_RING_DEFAULT,
):
    """rows[index] as [N, lane blocks, LANES] for the first
    n_subcores * run_live rows of `index` (prefix_run's value, traced), the
    rest of the output left unwritten: the sort of the routed rows, whose
    used prefix is routing-dependent while the table is sized to its
    bound. Rows only, fp8 or bf16, gathered on the sparse cores."""
    if rows.ndim != ROWS_RANK or jnp.dtype(rows.dtype) not in (
        jnp.dtype(jnp.float8_e4m3fn),
        jnp.dtype(jnp.bfloat16),
    ):
        raise ValueError(
            f"the rows are {rows.shape} {rows.dtype}. The gather "
            "takes [rows, lane blocks, LANES] fp8 e4m3 or bf16"
        )
    n_rows, lane_blocks, lanes = rows.shape
    per_word = WORD_BYTES // jnp.dtype(rows.dtype).itemsize
    if lanes != LANES or lane_blocks % per_word:
        raise ValueError(
            f"rows of {lane_blocks} x {lanes} lanes: the gather "
            f"moves a row as 32-bit words, {per_word} lane "
            "blocks to a word block"
        )
    if index.ndim != 1 or index.dtype != jnp.int32:
        raise ValueError(
            f"the gather index is {index.shape} {index.dtype}. A "
            "flat int32 array is expected"
        )
    mesh, n_subcores = subcore_mesh()
    run = rows_per_subcore(index.shape[0], n_subcores, block_rows)
    sc_lanes = pltpu.get_tpu_info().sparse_core.num_lanes
    if block_rows % sc_lanes:
        raise ValueError(
            f"the block of {block_rows} rows is not a whole "
            f"number of the sparse core's {sc_lanes}-lane vectors"
        )
    n = index.shape[0]
    meta = jnp.full((SLICE_ALIGN_WORDS,), run_live, jnp.int32)
    out_types = [jax.ShapeDtypeStruct((n, lane_blocks, lanes), rows.dtype)]
    scratch_types = [
        pltpu.VMEM((run,), jnp.int32),  # index
        pltpu.VMEM((ring, block_rows, lane_blocks // per_word, lanes), jnp.int32),
        pltpu.VMEM((SLICE_ALIGN_WORDS,), jnp.int32),  # meta: the run length
        pltpu.SemaphoreType.DMA((ring,)),  # row gathers
        pltpu.SemaphoreType.DMA((ring,)),  # row stores
        pltpu.SemaphoreType.DMA((2,)),  # meta and index fetches
    ]
    program = sparse_core_program(
        functools.partial(
            _gather_prefix_body, run=run, block_rows=block_rows, ring=ring
        ),
        out_types=out_types,
        mesh=mesh,
        scratch_types=scratch_types,
        compiler_params=pltpu.CompilerParams(**COMPILER_PARAMS),
        name=f"fused_ep_sort_rows_b{block_rows}x{ring}",
    )
    return program(index, rows, meta)[0]


def gather_rows_and_scales(
    rows,
    scale_rows,
    index,
    *,
    block_rows=GATHER_BLOCK_ROWS_DEFAULT,
    ring=GATHER_RING_DEFAULT,
):
    """rows [R, lane blocks, LANES] fp8 or bf16, scale_rows [R] f32 (row
    r's scale, a word per row) or None for rows without scales, index [N]
    int32 -> (rows[index] as [N, lane blocks, LANES], the scale of each
    gathered row as [N] f32 or None), gathered on the sparse cores."""
    if rows.ndim != ROWS_RANK or jnp.dtype(rows.dtype) not in (
        jnp.dtype(jnp.float8_e4m3fn),
        jnp.dtype(jnp.bfloat16),
    ):
        raise ValueError(
            f"the arrival rows is {rows.shape} {rows.dtype}. The "
            "gather takes [rows, lane blocks, LANES] fp8 e4m3 or "
            "bf16"
        )
    n_rows, lane_blocks, lanes = rows.shape
    per_word = WORD_BYTES // jnp.dtype(rows.dtype).itemsize
    if lanes != LANES or lane_blocks % per_word:
        raise ValueError(
            f"arrival rows of {lane_blocks} x {lanes} lanes: the "
            f"gather moves a row as 32-bit words, {per_word} "
            f"lane blocks of {rows.dtype} to a word block"
        )
    with_scales = scale_rows is not None
    if with_scales and (
        scale_rows.ndim != 1
        or scale_rows.shape[0] != n_rows
        or scale_rows.dtype != jnp.float32
    ):
        raise ValueError(
            f"the arrival scales are {scale_rows.shape} "
            f"{scale_rows.dtype}. "
            f"one f32 word per arrival row ({n_rows}) is expected"
        )
    if index.ndim != 1 or index.dtype != jnp.int32:
        raise ValueError(
            f"the gather index is {index.shape} {index.dtype}. A "
            "flat int32 array is expected"
        )
    mesh, n_subcores = subcore_mesh()
    run = rows_per_subcore(index.shape[0], n_subcores, block_rows)
    sc_lanes = pltpu.get_tpu_info().sparse_core.num_lanes
    if block_rows % sc_lanes:
        raise ValueError(
            f"the block of {block_rows} rows is not a whole "
            f"number of the sparse core's {sc_lanes}-lane vectors"
        )
    n = index.shape[0]
    rows_scratch = pltpu.VMEM(
        (ring, block_rows, lane_blocks // per_word, lanes), jnp.int32
    )
    if with_scales:
        out_types = [
            jax.ShapeDtypeStruct((n, lane_blocks, lanes), rows.dtype),
            jax.ShapeDtypeStruct((n,), jnp.float32),
        ]
        scratch_types = [
            pltpu.VMEM((run,), jnp.int32),  # index
            rows_scratch,
            pltpu.VMEM((ring, block_rows), jnp.float32),  # scales
            pltpu.SemaphoreType.DMA((ring,)),  # row gathers
            pltpu.SemaphoreType.DMA((ring,)),  # scale gathers
            pltpu.SemaphoreType.DMA((ring,)),  # row stores
            pltpu.SemaphoreType.DMA((ring + 1,)),  # scale stores + index fetch
        ]
        operands = (index, rows, scale_rows)
    else:
        out_types = [jax.ShapeDtypeStruct((n, lane_blocks, lanes), rows.dtype)]
        scratch_types = [
            pltpu.VMEM((run,), jnp.int32),  # index
            rows_scratch,
            pltpu.SemaphoreType.DMA((ring,)),  # row gathers
            pltpu.SemaphoreType.DMA((ring,)),  # row stores
            pltpu.SemaphoreType.DMA((ring + 1,)),  # index fetch
        ]
        operands = (index, rows)
    program = sparse_core_program(
        functools.partial(
            _gather_body,
            run=run,
            block_rows=block_rows,
            ring=ring,
            with_scales=with_scales,
        ),
        out_types=out_types,
        mesh=mesh,
        scratch_types=scratch_types,
        compiler_params=pltpu.CompilerParams(**COMPILER_PARAMS),
        name=f"fused_ep_gather_rows_b{block_rows}x{ring}"
        f"{'' if with_scales else '_bf16'}",
    )
    outs = program(*operands)
    return (outs[0], outs[1]) if with_scales else (outs[0], None)
