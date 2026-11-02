# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SparseCore packed candidate-key gather without expanded HBM row addresses."""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.deepseek_v41.index_key_layout import ROW_WORDS, SLOTS
from vllm_torchtpu.kernels.deepseek_v41.index_selection import tuned_block_sizes


def is_supported_by_gather_packed_candidate_keys(
    packed_key_banks: jax.Array | tuple[jax.Array, jax.Array],
    block_start_key_indices: jax.Array,
    keys_per_block: int,
) -> bool:
    """Whether packed D128 key blocks can use the native SparseCore gather."""
    banks = (
        packed_key_banks if isinstance(packed_key_banks, tuple) else (packed_key_banks,)
    )
    if (
        len(banks) not in (1, 2)
        or keys_per_block <= 0
        or keys_per_block % SLOTS
        or block_start_key_indices.dtype != jnp.int32
        or any(
            bank.ndim != 2
            or bank.shape[1] != ROW_WORDS
            or not bank.shape[0]
            or bank.dtype != jnp.int32
            for bank in banks
        )
    ):
        return False
    if not block_start_key_indices.size:
        return True
    return (
        jax.default_backend() == "tpu" and pltpu.get_tpu_info().sparse_core is not None
    )


@functools.partial(jax.jit, static_argnames=("keys_per_block",))
def gather_packed_candidate_keys(
    packed_key_banks: jax.Array | tuple[jax.Array, jax.Array],
    block_start_key_indices: jax.Array,
    keys_per_block: int,
) -> jax.Array:
    """Gather packed D128 index keys as INT32 [blocks * keys_per_block/2,128].

    Each bank row packs two 128-dimensional BF16 index keys. Block starts
    are physical key indices of any shape, aligned to two keys rather than
    whole blocks. Tuple banks share global key indexing; blocks may cross
    their boundary. Negative, odd, or out-of-range whole-block starts produce
    zero rows in their original slots. Nonempty gathers require TPU
    SparseCore and nonempty INT32 [rows,128] banks. Other cache geometries
    must retain their existing gather path.
    """
    rows_per_block = keys_per_block // SLOTS
    count = block_start_key_indices.size
    if not count:
        return jnp.empty((0, ROW_WORDS), jnp.int32)
    banks = (
        packed_key_banks if isinstance(packed_key_banks, tuple) else (packed_key_banks,)
    )
    info = plsc.get_sparse_core_info()
    lanes = info.num_lanes
    cores = info.num_cores * info.num_subcores
    chunks = max(
        1,
        tuned_block_sizes.CANDIDATE_GATHER_ROWS_PER_SUBCORE // (lanes * rows_per_block),
    )
    starts_per_core = chunks * lanes
    streams = math.gcd(
        chunks * rows_per_block,
        tuned_block_sizes.CANDIDATE_GATHER_MAX_READS // len(banks),
    )
    steps = chunks * rows_per_block // streams
    tile = cores * starts_per_core
    padding = -count % tile
    starts = jnp.pad(
        block_start_key_indices.reshape(-1), (0, padding), constant_values=-1
    )
    total_keys = sum(bank.shape[0] for bank in banks) * SLOTS
    mesh = plsc.VectorSubcoreMesh(
        num_cores=info.num_cores,
        num_subcores=info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct(
            ((count + padding) * rows_per_block, ROW_WORDS), jnp.int32
        ),
        scratch_types=(
            pltpu.VMEM((2, streams, lanes, ROW_WORDS), jnp.int32),
            pltpu.SemaphoreType.DMA((2, len(banks), streams)),
            pltpu.SemaphoreType.DMA((streams,)),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        mesh=mesh,
        name="candidate_banked_word_gather",
    )
    def kernel(bank_refs, starts_ref, out_ref, word_buf, read_sem, write_sem):
        core = jax.lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))

        def outer(ids):
            base = (pl.program_id(0) * cores + core) * starts_per_core

            def gather(step, wait=False):
                slot = step % 2
                for stream in range(streams):
                    index = step * streams + stream
                    chunk, offset = index // rows_per_block, index % rows_per_block
                    selected = ids[pl.ds(chunk * lanes, lanes)]
                    valid = (
                        (selected >= 0)
                        & (selected % SLOTS == 0)
                        & (selected <= total_keys - keys_per_block)
                    )
                    rows = selected // SLOTS + offset
                    if not wait:
                        word_buf[slot, stream] = jnp.zeros(
                            (lanes, ROW_WORDS), jnp.int32
                        )
                    bank_base = 0
                    for bank, bank_ref in enumerate(bank_refs):
                        own = (
                            valid
                            & (rows >= bank_base)
                            & (rows < bank_base + bank_ref.shape[0])
                        )
                        indexed = plsc.Indices(
                            jnp.where(own, rows - bank_base, -1), ignored_value=-1
                        )
                        copy = pltpu.make_async_copy(
                            bank_ref.at[indexed],
                            word_buf.at[slot, stream],
                            read_sem.at[slot, bank, stream],
                        )
                        if wait:
                            copy.wait()
                        else:
                            copy.start()
                        bank_base += bank_ref.shape[0]

            def step(index, _):
                slot = index % 2
                gather(index, wait=True)

                @pl.when(index + 1 < steps)
                def prefetch():
                    gather(index + 1)

                copies = []
                for stream in range(streams):
                    expanded = index * streams + stream
                    chunk = expanded // rows_per_block
                    offset = expanded % rows_per_block
                    destinations = (
                        base + chunk * lanes + jnp.arange(lanes)
                    ) * rows_per_block + offset
                    copy = pltpu.make_async_copy(
                        word_buf.at[slot, stream],
                        out_ref.at[plsc.Indices(destinations)],
                        write_sem.at[stream],
                    )
                    copy.start()
                    copies.append(copy)
                for copy in copies:
                    copy.wait()

            gather(jnp.int32(0))
            jax.lax.fori_loop(0, steps, step, None)

        pltpu.emit_pipeline(
            outer,
            grid=((count + padding) // tile,),
            in_specs=(pl.BlockSpec((starts_per_core,), lambda b: (b * cores + core,)),),
        )(starts_ref)

    out = kernel(banks, starts)
    return out[: count * rows_per_block]
