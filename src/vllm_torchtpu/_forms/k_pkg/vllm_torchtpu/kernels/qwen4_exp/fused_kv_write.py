# SPDX-License-Identifier: Apache-2.0
"""One QSA layer's decode/verify cache writes as a single Pallas kernel.

A decode or MTP verify step writes three disjoint pool regions per QSA
layer: main KV rows (one per token, read/merged/written when two layers
pair their heads in a row), completed compressed-key pairs (merged into
the half-written BF16 pair) and every request's raw ring plus tags. The
separate writers each launch a kernel and mostly serialize their row DMAs.
Here every row read is issued before the first wait, merges run once in
VMEM, and every row write is issued before the first wait, so the step
pays about two DMA latencies per layer instead of one per row.

The regions never alias: main KV, compressed keys and rings live in pages
of different cache groups. Within the main region the writer preserves the
last-writer order of duplicate destinations by issuing those writes one at
a time.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import typed_ldst
from vllm_torchtpu.kernels.qwen4_exp.cache_regions import region_rows

# Decode/verify steps only: per-token scalars live in SMEM and the ring
# staging is unrolled per request.
MAX_FUSED_TOKENS = 512
MAX_FUSED_REQUESTS = 128

_MAIN_READ, _COMP_READ, _MAIN_WRITE, _COMP_WRITE, _RING_WRITE = range(5)
_RUN_READ, _RUN_WRITE = 5, 6


def fused_write_supported(pool, tokens: int, requests: int, main_slot) -> bool:
    """Whether ``fused_layer_write`` covers this step's static geometry."""
    packing = pool.shape[-2]
    if pool.ndim != 5 or pool.shape[2] != 1:
        return False
    if pool.dtype == jnp.float8_e4m3fn:
        if packing != 4:
            return False
    elif pool.dtype != jnp.bfloat16 or packing != 2:
        return False
    if main_slot is not None and (
        pool.dtype != jnp.float8_e4m3fn or not 0 <= main_slot < packing // 2
    ):
        return False
    return 0 < tokens <= MAX_FUSED_TOKENS and 0 < requests <= MAX_FUSED_REQUESTS


def main_keep_mask(slot: int) -> int:
    # Slot s of a paired FP8 row owns bytes 2s (K) and 2s + 1 (V) of every
    # 32-bit word; see ``pooled_qsa._main_slot_keep_mask``.
    return ~(0xFFFF << (16 * slot)) & 0xFFFFFFFF


def fused_layer_write(
    pool: jax.Array,
    *,
    main_rows: jax.Array,
    main_page: jax.Array,
    main_offset: jax.Array,
    main_valid: jax.Array,
    main_slot: int | None,
    comp_pair: jax.Array,
    comp_write: jax.Array,
    comp_mask: jax.Array,
    comp_values: jax.Array,
    ring: jax.Array,
    tags: jax.Array,
    ring_ids: jax.Array,
    run_starts: jax.Array | None = None,
    run_lengths: jax.Array | None = None,
    run_length: int = 1,
) -> jax.Array:
    """Write main KV, compressed pairs and rings of one layer in place.

    ``main_rows`` are packed rows already moved to ``main_slot``'s head
    pair (``None``: whole rows). ``comp_pair`` indexes
    ``pool.reshape(-1, packing, width)`` like ``pooled_qsa.write_compressed``;
    ``comp_mask[:, h]`` selects which BF16 half of the pair takes
    ``comp_values[:, h]``. ``ring``/``tags`` are flat per-request elements
    written to ``ring_ids`` > 0 as ``qsa_state.write_ring_and_tags`` does.

    ``run_starts``/``run_lengths`` (first token and token count of each
    request) with a static ``run_length`` > 1 move each request that holds
    exactly ``run_length`` page-contiguous rows (an MTP verify step) to one
    multi-row DMA each way instead of one per row.
    """
    tokens = len(main_rows)
    page_rows = pool.shape[1]
    width = pool.shape[-1]
    packing = pool.shape[-2]
    payload = tuple(pool.shape[2:])
    paired = main_slot is not None
    keep = main_keep_mask(main_slot) if paired else 0

    main_valid = main_valid.astype(jnp.int32)
    page_safe = jnp.where(main_valid != 0, main_page, 0).astype(jnp.int32)
    offset = (main_offset % page_rows).astype(jnp.int32)
    # Duplicate destinations among valid rows (none in a well-formed step)
    # keep their order through serial writes.
    flat = jnp.where(
        main_valid != 0, page_safe * page_rows + offset, -1 - jnp.arange(tokens)
    )
    ordered = jnp.sort(flat)
    duplicate = jnp.any((ordered[1:] == ordered[:-1]) & (ordered[1:] >= 0))
    unique = jnp.logical_not(duplicate).astype(jnp.int32).reshape(1)

    use_runs = run_starts is not None and run_length > 1 and tokens >= run_length
    if use_runs:
        assert run_lengths is not None
        span = jnp.arange(run_length, dtype=jnp.int32)
        first = jnp.clip(run_starts.astype(jnp.int32), 0, tokens - run_length)
        members = first[:, None] + span[None]
        run_live = (
            (run_lengths == run_length)
            & (run_starts + run_length <= tokens)
            & jnp.all(main_valid[members] != 0, axis=1)
            & jnp.all(page_safe[members] == page_safe[first][:, None], axis=1)
            & jnp.all(offset[members] == offset[first][:, None] + span[None], axis=1)
            & (unique[0] != 0)
        )
        token = jnp.arange(tokens, dtype=jnp.int32)[:, None]
        covered = jnp.any(
            run_live[None]
            & (token >= first[None])
            & (token < first[None] + run_length),
            axis=1,
        )
        row_valid = jnp.where(covered, 0, main_valid)
        run_page = page_safe[first]
        run_offset = offset[first]
    else:
        first = run_page = run_offset = jnp.zeros((1,), jnp.int32)
        run_live = jnp.zeros((1,), jnp.bool_)
        row_valid = main_valid
    run_live = run_live.astype(jnp.int32)
    runs = len(first)

    comp_tokens = len(comp_pair)
    comp_write = comp_write.astype(jnp.int32)
    comp_mask = comp_mask.astype(jnp.int32)
    comp_safe = jnp.where(comp_write != 0, comp_pair, 0).astype(jnp.int32)
    comp_page = comp_safe // page_rows
    comp_offset = comp_safe % page_rows
    comp_full = (comp_mask[:, 0] != 0) & (comp_mask[:, 1] != 0)
    # 0: skip; 1: write both halves; 2: read, merge one half, write.
    comp_mode = jnp.where(comp_write != 0, jnp.where(comp_full, 1, 2), 0)
    comp_mode = comp_mode.astype(jnp.int32)
    comp_mask_flat = comp_mask.reshape(-1)
    comp_values = jnp.pad(
        comp_values.astype(jnp.bfloat16),
        ((0, 0), (0, 0), (0, width - comp_values.shape[-1])),
    )

    count, ring_elements = ring.shape
    tag_elements = tags.shape[-1]
    ring_tokens, ring_rows = region_rows(pool, ring_elements, ring.dtype)
    tag_tokens, tag_rows = region_rows(pool, tag_elements, tags.dtype)
    state_tokens = ring_tokens + tag_tokens
    assert state_tokens <= page_rows
    lanes = pool.shape[-1]
    ring = jnp.pad(ring, ((0, 0), (0, ring_rows * lanes - ring_elements))).reshape(
        count, ring_rows, lanes
    )
    tags = jnp.pad(tags, ((0, 0), (0, tag_rows * lanes - tag_elements))).reshape(
        count, tag_rows, lanes
    )
    ring_ids = ring_ids.astype(jnp.int32)

    def kernel(
        page_ref,
        offset_ref,
        valid_ref,
        unique_ref,
        cpage_ref,
        coffset_ref,
        cmode_ref,
        cmask_ref,
        ids_ref,
        run_tok_ref,
        run_page_ref,
        run_off_ref,
        run_live_ref,
        rows_ref,
        cvalues_ref,
        ring_ref,
        tags_ref,
        _old_ref,
        target_ref,
        main_buf,
        comp_buf,
        staged,
        sems,
    ):
        def main_copy(i, read):
            stored = target_ref.at[page_ref[i], pl.ds(offset_ref[i], 1)]
            if read:
                return pltpu.make_async_copy(
                    stored, main_buf.at[pl.ds(i, 1)], sems.at[_MAIN_READ]
                )
            source = main_buf if paired else rows_ref
            return pltpu.make_async_copy(
                source.at[pl.ds(i, 1)], stored, sems.at[_MAIN_WRITE]
            )

        def run_copy(r, read):
            stored = target_ref.at[run_page_ref[r], pl.ds(run_off_ref[r], run_length)]
            if read:
                return pltpu.make_async_copy(
                    stored,
                    main_buf.at[pl.ds(run_tok_ref[r], run_length)],
                    sems.at[_RUN_READ],
                )
            source = main_buf if paired else rows_ref
            return pltpu.make_async_copy(
                source.at[pl.ds(run_tok_ref[r], run_length)],
                stored,
                sems.at[_RUN_WRITE],
            )

        def comp_copy(i, read):
            stored = target_ref.at[cpage_ref[i], coffset_ref[i], 0]
            staged_pair = comp_buf.at[i]
            source, dest = (stored, staged_pair) if read else (staged_pair, stored)
            return pltpu.make_async_copy(
                source, dest, sems.at[_COMP_READ if read else _COMP_WRITE]
            )

        def ring_copy(lane):
            return pltpu.make_async_copy(
                staged.at[lane],
                target_ref.at[ids_ref[lane], pl.ds(0, state_tokens)],
                sems.at[_RING_WRITE],
            )

        def each(n, predicate, action):
            def body(i, carry):
                @pl.when(predicate(i))
                def _():
                    action(i)

                return carry

            jax.lax.fori_loop(0, n, body, 0)

        main_live = lambda i: valid_ref[i] != 0  # noqa: E731
        comp_merge = lambda i: cmode_ref[i] == 2  # noqa: E731
        comp_live = lambda i: cmode_ref[i] != 0  # noqa: E731
        ring_live = lambda i: ids_ref[i] > 0  # noqa: E731
        run_live_at = lambda r: run_live_ref[r] != 0  # noqa: E731

        # Phase 1: every read in flight.
        if paired:
            each(tokens, main_live, lambda i: main_copy(i, True).start())
            if use_runs:
                each(runs, run_live_at, lambda r: run_copy(r, True).start())
        each(comp_tokens, comp_merge, lambda i: comp_copy(i, True).start())

        # Ring staging overlaps the reads.
        for lane in range(count):
            typed_ldst.store_typed(
                staged.at[lane, pl.ds(0, ring_tokens)], ring_ref[lane]
            )
            typed_ldst.store_typed(
                staged.at[lane, pl.ds(ring_tokens, tag_tokens)], tags_ref[lane]
            )

        if paired:
            each(tokens, main_live, lambda i: main_copy(i, True).wait())
            if use_runs:
                each(runs, run_live_at, lambda r: run_copy(r, True).wait())
            old = main_buf.bitcast(jnp.uint32).reshape(-1, width)
            new = rows_ref.bitcast(jnp.uint32).reshape(-1, width)
            old[...] = (old[...] & jnp.uint32(keep)) | new[...]
        each(comp_tokens, comp_merge, lambda i: comp_copy(i, True).wait())

        # Phase 2: merge compressed halves in VMEM.
        bf = comp_buf.bitcast(jnp.bfloat16) if pool.dtype != jnp.bfloat16 else comp_buf
        half = jax.lax.broadcasted_iota(jnp.int32, (2, width), 0)

        def merge_comp(i):
            take = (
                jnp.where(half == 0, cmask_ref[2 * i], cmask_ref[2 * i + 1]) != 0
            ) | (cmode_ref[i] == 1)
            bf[i] = jnp.where(take, cvalues_ref[i], bf[i])

        each(comp_tokens, comp_live, merge_comp)

        # Phase 3: every write in flight.
        @pl.when(unique_ref[0] != 0)
        def write_main_unique():
            each(tokens, main_live, lambda i: main_copy(i, False).start())
            if use_runs:
                each(runs, run_live_at, lambda r: run_copy(r, False).start())

        @pl.when(unique_ref[0] == 0)
        def write_main_ordered():
            def store(i):
                copy = main_copy(i, False)
                copy.start()
                copy.wait()

            each(tokens, main_live, store)

        each(comp_tokens, comp_live, lambda i: comp_copy(i, False).start())
        each(count, ring_live, lambda i: ring_copy(i).start())

        @pl.when(unique_ref[0] != 0)
        def wait_main_unique():
            each(tokens, main_live, lambda i: main_copy(i, False).wait())
            if use_runs:
                each(runs, run_live_at, lambda r: run_copy(r, False).wait())

        each(comp_tokens, comp_live, lambda i: comp_copy(i, False).wait())
        each(count, ring_live, lambda i: ring_copy(i).wait())

    def full(shape):
        return pl.BlockSpec(shape, lambda i, *_: (0,) * len(shape))

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=13,
            grid=(1,),
            in_specs=[
                full(main_rows.shape),
                full(comp_values.shape),
                full(ring.shape),
                full(tags.shape),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM(main_rows.shape if paired else (8, *payload), pool.dtype),
                pltpu.VMEM((max(comp_tokens, 1), packing, width), pool.dtype),
                pltpu.VMEM((count, state_tokens, *payload), pool.dtype),
                pltpu.SemaphoreType.DMA((7,)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={17: 0},
        name="qwen4_exp_qsa_fused_kv_write",
    )(
        page_safe,
        offset,
        row_valid,
        unique,
        comp_page.astype(jnp.int32),
        comp_offset.astype(jnp.int32),
        comp_mode,
        comp_mask_flat,
        ring_ids,
        first,
        run_page.astype(jnp.int32),
        run_offset.astype(jnp.int32),
        run_live,
        main_rows.astype(pool.dtype),
        comp_values,
        ring,
        tags,
        pool,
    )


def read_ring_and_tags_concurrent(
    pool: jax.Array, ids: jax.Array, ring_elements: int, tag_elements: int
) -> tuple[jax.Array, jax.Array]:
    """``qsa_state.read_ring_and_tags`` with every request's DMA in flight."""
    count = len(ids)
    ring_tokens, ring_rows = region_rows(pool, ring_elements, jnp.bfloat16)
    tag_tokens, tag_rows = region_rows(pool, tag_elements, jnp.int32)
    tokens = ring_tokens + tag_tokens
    payload = tuple(pool.shape[2:])
    lanes = pool.shape[-1]

    def kernel(indices, source, ring_out, tags_out, staged, semaphore):
        def copy(lane):
            return pltpu.make_async_copy(
                source.at[jnp.maximum(indices[lane], 0), pl.ds(0, tokens)],
                staged.at[lane],
                semaphore,
            )

        for lane in range(count):
            copy(lane).start()
        for lane in range(count):
            copy(lane).wait()
        for lane in range(count):
            ring_out[lane] = typed_ldst.load_typed(
                staged.at[lane, pl.ds(0, ring_tokens)], view_dtype=jnp.bfloat16
            )
            tags_out[lane] = typed_ldst.load_typed(
                staged.at[lane, pl.ds(ring_tokens, tag_tokens)], view_dtype=jnp.int32
            )

    ring, tags = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=[
                pl.BlockSpec((count, ring_rows, lanes), lambda i, *_: (0, 0, 0)),
                pl.BlockSpec((count, tag_rows, lanes), lambda i, *_: (0, 0, 0)),
            ],
            scratch_shapes=[
                pltpu.VMEM((count, tokens, *payload), pool.dtype),
                pltpu.SemaphoreType.DMA,
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((count, ring_rows, lanes), jnp.bfloat16),
            jax.ShapeDtypeStruct((count, tag_rows, lanes), jnp.int32),
        ],
        name="qwen4_exp_ring_tags_concurrent_read",
    )(ids.astype(jnp.int32), pool)
    return (
        ring.reshape(count, -1)[:, :ring_elements],
        tags.reshape(count, -1)[:, :tag_elements],
    )


__all__ = [
    "MAX_FUSED_REQUESTS",
    "MAX_FUSED_TOKENS",
    "fused_layer_write",
    "fused_write_supported",
    "read_ring_and_tags_concurrent",
]
