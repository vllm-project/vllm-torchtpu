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

# Provenance: copied from src/vllm_torchtpu/kernels/qwen4_exp/compressed_write.py
#   at 473dda4acce94f0d133b73352ede6c9f9819a7f2, a commit of the public
#   branch dongruizi/qwen38-flash-agentx (the served Qwen3.8-Flash stack; the
#   file is not on main). Owned by models/qwen4_exp from this copy on.
"""Run-based compressed-key pool writer for prefill-sized steps.

The pooled QSA writer (``pooled_qsa.write_compressed``, added by #3234)
writes one BF16 pair (an FP8 carrier row) per completed pair of compressed
groups. Its tiled kernel runs one 128-token tile per grid
step and waits each DMA; a tile takes the one-DMA path only when it is page
aligned and fully covered, otherwise it issues up to 128 serial per-row
writes. Mixed serving chunks rarely align, so an 8192-token chunk spends
~1.5 ms per layer there for ~1k pair rows.

This writer compacts the writers (one per touched pair), stages their carrier
rows in VMEM with one vector store, merges the few partial pairs (request
edges) with their old rows, and writes each run of consecutive pair rows with
power-of-two DMAs that are all in flight at once. ``pooled_qsa`` (#3234)
calls it for steps of at least 512 tokens when its
``VLLM_TPU_QSA_COMPRESSED_RUN_WRITE`` plan option is on.
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
    """``view`` with the writers applied (the input buffer is aliased).

    Inputs, for ``tokens`` step tokens:

    - ``view``: the compressed-key pool as ``(pairs, packing, width)``; one
      row is one BF16 pair (two BF16 rows of ``width``), so ``packing *
      itemsize == 4`` (FP8 carrier: packing 4; BF16: packing 2).
    - ``pair``: ``(tokens,)`` int, the destination row of each token; every
      writer's ``pair`` is in ``[0, view.shape[0])``.
    - ``write``: ``(tokens,)`` bool, marks the one writer of each touched
      pair (at most one writer per pair).
    - ``mask``: ``(tokens, 2)`` bool, the BF16 halves the writer fills; a
      half it does not fill keeps its old value.
    - ``values``: ``(tokens, 2, width)`` BF16, the new halves.
    - ``requests``: requests in the step; ``ratio``: tokens per compressed
      group. Together with ``tokens`` they bound the writer count
      (``writer_capacity``).

    Non-writer tokens are ignored. Writers are applied in token order;
    consecutive destination rows form one run of DMAs.
    """
    tokens = len(pair)
    if view.ndim != 3:
        raise ValueError(f"view must be (pairs, packing, width), got {view.shape}")
    packing, width = view.shape[1:]
    if packing * jnp.dtype(view.dtype).itemsize != 4:
        raise ValueError(
            f"a view row must hold one BF16 pair (packing * itemsize == 4), "
            f"got packing {packing} of {view.dtype}"
        )
    if tokens <= 0 or requests <= 0 or ratio <= 0:
        raise ValueError(
            f"tokens, requests and ratio must be positive, got {tokens}, "
            f"{requests}, {ratio}"
        )
    if values.shape != (tokens, 2, width) or values.dtype != jnp.bfloat16:
        raise ValueError(
            f"values must be BF16 {(tokens, 2, width)}, got {values.dtype} "
            f"{values.shape}"
        )
    if write.shape != (tokens,) or mask.shape != (tokens, 2):
        raise ValueError(
            f"write must be {(tokens,)} and mask {(tokens, 2)}, got "
            f"{write.shape} and {mask.shape}"
        )
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
        staged_bf = (
            staged.bitcast(jnp.bfloat16) if view.dtype != jnp.bfloat16 else staged
        )
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
