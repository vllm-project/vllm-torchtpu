# SPDX-License-Identifier: Apache-2.0
"""Run-based main KV writer for prefill-sized steps.

``write_main_cache``'s tiled kernels take their one-DMA path only for a
128-token tile that is page aligned and fully valid; any other tile issues a
read and a write per row (paired pages), waiting row by row when the tile has
repeated destinations. A prefill request whose context is not a multiple of
128, decode rows and request edges all land there, so mixed serving chunks
pay one DMA issue per row.

This writer splits the step into runs of consecutive rows (same page,
consecutive offsets, inside one tile) and moves each run with power-of-two
DMAs that are all in flight at once (VLLM_TPU_QSA_MAIN_RUN_WRITE). Paired
pages read the runs' old rows, merge this layer's head pair word-wise and
write the runs back. A tile with repeated destinations writes its runs one
at a time in token order, so the last writer wins as in the tiled kernels.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def main_runs(page, offsets, valid, tile, rows_per_page):
    """Run starts and lengths, each tile's first run and its unique flag.

    ``page``/``offsets``/``valid`` are padded to a multiple of ``tile``. A run
    is a maximal stretch of valid rows of one tile with one page and
    consecutive offsets. Returns ``(start, length, tile_first, unique)``:
    ``start``/``length`` (int32 ``[N]``, the first ``tile_first[-1]``
    entries used, in token order), ``tile_first`` (int32 ``[N // tile + 1]``)
    and ``unique`` (int32 ``[N // tile]``, 1 when no two valid rows of the
    tile share a destination).
    """
    tokens = len(page)
    index = jnp.arange(tokens, dtype=jnp.int32)
    follows = (
        valid
        & jnp.concatenate([jnp.zeros((1,), bool), valid[:-1]])
        & (page == jnp.roll(page, 1))
        & (offsets == jnp.roll(offsets, 1) + 1)
        & (index % tile != 0)
    )
    begins = valid & ~follows
    ends = valid & ~jnp.concatenate([follows[1:], jnp.zeros((1,), bool)])
    (start,) = jnp.nonzero(begins, size=tokens, fill_value=0)
    (end,) = jnp.nonzero(ends, size=tokens, fill_value=-1)
    length = end - start + 1
    per_tile = jnp.sum(begins.reshape(-1, tile), axis=1, dtype=jnp.int32)
    tile_first = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(per_tile)])
    # Invalid rows get distinct negative keys; a sort exposes repeats.
    key = jnp.where(valid, page * rows_per_page + offsets, -1 - index)
    key = jnp.sort(key.reshape(-1, tile), axis=1)
    unique = ~jnp.any(key[:, 1:] == key[:, :-1], axis=1)
    return (
        start.astype(jnp.int32),
        length.astype(jnp.int32),
        tile_first.astype(jnp.int32),
        unique.astype(jnp.int32),
    )


def write_main_runs(pool, rows, page, offsets, valid, *, keep=None, tile=512):
    """``pool`` with ``rows[i]`` stored at ``pool[page[i], offsets[i]]``.

    ``keep`` (paired pages): the uint32 mask of the partner's bytes in every
    word; those bytes keep their old value and ``rows`` must be zero there
    (``main_slot_rows``). ``None`` writes whole rows.
    """
    tokens = len(rows)
    padded = -(-tokens // tile) * tile
    if padded != tokens:
        extra = padded - tokens
        rows = jnp.pad(rows, ((0, extra),) + ((0, 0),) * (rows.ndim - 1))
        page = jnp.pad(page, (0, extra))
        offsets = jnp.pad(offsets, (0, extra))
        valid = jnp.pad(valid, (0, extra))
    valid = valid.astype(jnp.bool_)
    start, length, tile_first, unique = main_runs(page, offsets, valid, tile, pool.shape[1])
    # A run stays inside one tile and one page.
    longest = min(tile, pool.shape[1])
    sizes = tuple(1 << b for b in range(longest.bit_length() - 1, -1, -1))
    width = pool.shape[-1]
    paired = keep is not None

    def kernel(
        pages_ref,
        offsets_ref,
        start_ref,
        length_ref,
        first_ref,
        unique_ref,
        source_ref,
        _old_ref,
        target_ref,
        *scratch,
    ):
        if paired:
            buffer_ref, semaphores, sent = scratch
        else:
            semaphores, sent = scratch
            buffer_ref = source_ref
        program = pl.program_id(0)
        base = program * tile
        runs = (first_ref[program], first_ref[program + 1])

        def copies(r, write, action):
            token, run = start_ref[r], length_ref[r]
            local = token - base
            for s, size in enumerate(sizes):

                @pl.when((run & size) != 0)
                def _(s=s, size=size):
                    done = run & ~(2 * size - 1)
                    stored = target_ref.at[
                        pages_ref[token], pl.ds(offsets_ref[token] + done, size)
                    ]
                    staged = buffer_ref.at[pl.ds(local + done, size)]
                    source, dest = (staged, stored) if write else (stored, staged)
                    action(s, pltpu.make_async_copy(source, dest, semaphores.at[s]))

        def start(s, copy):
            copy.start()
            sent[s] = sent[s] + 1

        def all_in_flight(write):
            for s in range(len(sizes)):
                sent[s] = 0

            def issue(r, carry):
                copies(r, write, start)
                return carry

            jax.lax.fori_loop(*runs, issue, 0)
            for s, size in enumerate(sizes):

                def wait(k, carry, s=s, size=size):
                    pltpu.make_async_copy(
                        buffer_ref.at[pl.ds(0, size)],
                        buffer_ref.at[pl.ds(0, size)],
                        semaphores.at[s],
                    ).wait()
                    return carry

                jax.lax.fori_loop(0, sent[s], wait, 0)

        if paired:
            all_in_flight(write=False)
            old = buffer_ref.bitcast(jnp.uint32).reshape(-1, width)
            new = source_ref.bitcast(jnp.uint32).reshape(-1, width)
            old[...] = (old[...] & jnp.uint32(keep)) | new[...]

        @pl.when(unique_ref[program] != 0)
        def store_unique():
            all_in_flight(write=True)

        @pl.when(unique_ref[program] == 0)
        def store_ordered():
            def store(s, copy):
                copy.start()
                copy.wait()

            def issue(r, carry):
                copies(r, True, store)
                return carry

            jax.lax.fori_loop(*runs, issue, 0)

    scratch = [pltpu.SemaphoreType.DMA((len(sizes),)), pltpu.SMEM((len(sizes),), jnp.int32)]
    if paired:
        scratch = [pltpu.VMEM((tile, *rows.shape[1:]), pool.dtype)] + scratch
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            grid=(padded // tile,),
            in_specs=[
                pl.BlockSpec(
                    (tile, *rows.shape[1:]),
                    lambda program, *_: (program,) + (0,) * (rows.ndim - 1),
                ),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch,
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={7: 0},
        name="qwen4_exp_main_kv_run_write",
    )(
        page.astype(jnp.int32),
        offsets.astype(jnp.int32),
        start,
        length,
        tile_first,
        unique,
        rows,
        pool,
    )
