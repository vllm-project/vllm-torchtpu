# SPDX-License-Identifier: Apache-2.0
"""Run-based compressed-key pool writer for prefill-sized steps.

``write_compressed`` writes one BF16 pair (an FP8 carrier row) per completed
pair of compressed groups. Its tiled kernel runs one 128-token tile per grid
step and waits each DMA; a tile takes the one-DMA path only when it is page
aligned and fully covered, otherwise it issues up to 128 serial per-row
writes. Mixed serving chunks rarely align, so an 8192-token chunk spends
~1.5 ms per layer there for ~1k pair rows.

This writer compacts the writers (one per touched pair), stages their carrier
rows in VMEM with one vector store, merges the few partial pairs (request
edges) with their old rows, and writes each run of consecutive pair rows with
power-of-two DMAs that are all in flight at once (VLLM_TPU_QSA_COMPRESSED_RUN_WRITE).
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def writer_capacity(tokens: int, requests: int, ratio: int) -> int:
    """Upper bound on the pairs one step writes.

    A pair covers ``2 * ratio`` tokens of one request, so a request of ``n``
    tokens touches at most ``n // (2 * ratio) + 2`` pairs.
    """
    cap = tokens // (2 * ratio) + 2 * requests + 8
    return -(-cap // 8) * 8


def write_runs(view, pair, write, mask, values, requests, ratio):
    """``view`` (``pool.reshape(-1, packing, width)``) with the writers applied.

    ``pair``/``write``/``mask``/``values`` are ``compressed_pairs``' outputs:
    ``write`` marks the one writer of each touched pair, ``mask[:, h]`` the
    BF16 halves it fills from ``values[:, h]``.
    """
    tokens = len(pair)
    packing, width = view.shape[1:]
    cap = writer_capacity(tokens, requests, ratio)
    (idx,) = jnp.nonzero(write, size=cap, fill_value=tokens)
    valid = idx < tokens
    safe = jnp.minimum(idx, tokens - 1)
    pairs = jnp.where(valid, pair[safe], 0).astype(jnp.int32)
    halves = jnp.where(valid[:, None], mask[safe], False)
    rows = values[safe]
    partial = valid & ~(halves[:, 0] & halves[:, 1])
    (partial_list,) = jnp.nonzero(partial, size=cap, fill_value=0)
    previous = jnp.concatenate([jnp.full((1,), -2, jnp.int32), pairs[:-1]])
    prev_valid = jnp.concatenate([jnp.zeros((1,), bool), valid[:-1]])
    start = valid & ~(prev_valid & (pairs == previous + 1))
    (run_off,) = jnp.nonzero(start, size=cap, fill_value=0)
    num_valid = jnp.sum(valid).astype(jnp.int32)
    num_runs = jnp.sum(start).astype(jnp.int32)
    run_end = jnp.where(
        jnp.arange(cap) + 1 < num_runs,
        jnp.concatenate([run_off[1:], jnp.zeros((1,), run_off.dtype)]),
        num_valid,
    )
    run_len = jnp.where(jnp.arange(cap) < num_runs, run_end - run_off, 0)
    run_pair = pairs[run_off]
    counts = jnp.stack([jnp.sum(partial).astype(jnp.int32), num_runs])
    sizes = tuple(1 << b for b in range(cap.bit_length() - 1, -1, -1))

    def kernel(
        pair_ref,
        mask_ref,
        partial_ref,
        off_ref,
        run_pair_ref,
        len_ref,
        count_ref,
        rows_ref,
        _old,
        out,
        staged,
        tmp,
        row_sem,
        size_sems,
        sent,
    ):
        staged_bf = staged.bitcast(jnp.bfloat16) if view.dtype != jnp.bfloat16 else staged
        staged_bf[...] = rows_ref[...]

        # Partial pairs (request edges): merge the new half into the old row.
        def merge(k, carry):
            i = partial_ref[k]
            copy = pltpu.make_async_copy(out.at[pair_ref[i]], tmp, row_sem)
            copy.start()
            copy.wait()
            tmp_bf = tmp.bitcast(jnp.bfloat16) if view.dtype != jnp.bfloat16 else tmp
            half = jnp.broadcast_to(jnp.arange(2)[:, None], (2, width))
            keep = jnp.where(half == 0, mask_ref[i, 0], mask_ref[i, 1]) != 0
            staged_bf[i] = jnp.where(keep, rows_ref[i], tmp_bf[...])
            return carry

        jax.lax.fori_loop(0, count_ref[0], merge, 0)

        # Runs of consecutive pair rows: power-of-two DMAs, all in flight.
        for s in range(len(sizes)):
            sent[s] = 0

        def issue(r, carry):
            off, dest, length = off_ref[r], run_pair_ref[r], len_ref[r]
            for s, size in enumerate(sizes):

                @pl.when((length & size) != 0)
                def _(s=s, size=size):
                    done = length & ~(2 * size - 1)
                    pltpu.make_async_copy(
                        staged.at[pl.ds(off + done, size)],
                        out.at[pl.ds(dest + done, size)],
                        size_sems.at[s],
                    ).start()
                    sent[s] = sent[s] + 1

            return carry

        jax.lax.fori_loop(0, count_ref[1], issue, 0)
        for s, size in enumerate(sizes):

            def wait(k, carry, s=s, size=size):
                pltpu.make_async_copy(
                    staged.at[pl.ds(0, size)], out.at[pl.ds(0, size)], size_sems.at[s]
                ).wait()
                return carry

            jax.lax.fori_loop(0, sent[s], wait, 0)

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=7,
            grid=(1,),
            in_specs=[
                pl.BlockSpec((cap, 2, width), lambda i, *_: (0, 0, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((cap, packing, width), view.dtype),
                pltpu.VMEM((packing, width), view.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((len(sizes),)),
                pltpu.SMEM((len(sizes),), jnp.int32),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(view.shape, view.dtype),
        input_output_aliases={8: 0},
        name="qwen4_exp_compressed_kv_run_write",
    )(
        pairs,
        halves.astype(jnp.int32),
        partial_list.astype(jnp.int32),
        run_off.astype(jnp.int32),
        run_pair.astype(jnp.int32),
        run_len.astype(jnp.int32),
        counts,
        rows,
        view,
    )
