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

import os

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

# q38mm DSTOCK-B lean writer: VLLM_TPU_QSA_FUSED_KV_WRITE_LEAN=1 (with VLLM_TPU_QSA_FUSED_KV_WRITE=1)
# routes fused_layer_write to _lean_layer_write (same interface, same bytes, compacted work
# lists). Unset or 0: the fused kernel exactly as before. Other values raise. Not registered in
# envs.py; compilation/tpu_compiler.py keys it only when set.
LEAN_ENV = "VLLM_TPU_QSA_FUSED_KV_WRITE_LEAN"


def lean_writer_enabled() -> bool:
    value = os.environ.get(LEAN_ENV, "").strip()
    if value in ("", "0"):
        return False
    if value == "1":
        return True
    raise ValueError(f"{LEAN_ENV} must be 0 or 1, got {value!r}")

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
    if _d73_traced(len(main_rows), len(ring)):  # = lean_writer_enabled(), + D73's trace-time line (q38 L2 D73MK)
        return _lean_layer_write(
            pool, main_rows=main_rows, main_page=main_page, main_offset=main_offset,
            main_valid=main_valid, main_slot=main_slot, comp_pair=comp_pair,
            comp_write=comp_write, comp_mask=comp_mask, comp_values=comp_values,
            ring=ring, tags=tags, ring_ids=ring_ids, run_starts=run_starts,
            run_lengths=run_lengths, run_length=run_length,
        )
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
        name=D73_FUSED_KERNEL,
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


def _lean_layer_write(
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
    """``fused_layer_write`` with compacted work lists (q38mm DSTOCK-B lean writer, LEAN_ENV).

    Same operands, same DMAs and the same merges, so the same pool bytes. The fused kernel
    walks every token slot in each of its ~13 phases behind ``pl.when`` (about 1,344 loop
    iterations per layer at 128 tokens while ~32 rows are live). Here one scalar pass per
    category writes the live indices into SMEM lists (branch-free: store, then advance the
    count by the predicate), and every DMA phase loops over its list only. The compressed
    merge is one vector ``where`` over the staged block (``comp_take``); the ring staging
    and ring DMAs run only when some ring id is live (``ring_any``).
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

    comp_take = jnp.where(
        (comp_mask != 0) | (comp_mode[:, None] == 1), jnp.int32(1), jnp.int32(0)
    )
    comp_take = jnp.broadcast_to(comp_take[:, :, None], (comp_tokens, 2, width))
    ring_any = jnp.any(ring_ids > 0).astype(jnp.int32).reshape(1)

    def kernel(
        page_ref,
        offset_ref,
        valid_ref,
        unique_ref,
        cpage_ref,
        coffset_ref,
        cmode_ref,
        ids_ref,
        run_tok_ref,
        run_page_ref,
        run_off_ref,
        run_live_ref,
        ring_any_ref,
        rows_ref,
        cvalues_ref,
        ctake_ref,
        ring_ref,
        tags_ref,
        _old_ref,
        target_ref,
        main_buf,
        comp_buf,
        staged,
        sems,
        main_list,
        run_list,
        merge_list,
        live_list,
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

        def compact(n, pred, out):
            # Branch-free: always store i at the count, advance the count only for live i.
            def body(i, c):
                out[c] = i
                return c + jnp.where(pred(i), jnp.int32(1), jnp.int32(0))

            return jax.lax.fori_loop(0, n, body, jnp.int32(0))

        def compact_comp(n):
            def body(i, c):
                m, lv = c
                merge_list[m] = i
                live_list[lv] = i
                mode = cmode_ref[i]
                return (
                    m + jnp.where(mode == 2, jnp.int32(1), jnp.int32(0)),
                    lv + jnp.where(mode != 0, jnp.int32(1), jnp.int32(0)),
                )

            return jax.lax.fori_loop(0, n, body, (jnp.int32(0), jnp.int32(0)))

        def over(n, lst, action):
            def body(k, carry):
                action(lst[k])
                return carry

            jax.lax.fori_loop(0, n, body, 0)

        n_main = compact(tokens, lambda i: valid_ref[i] != 0, main_list)
        n_runs = (
            compact(runs, lambda r: run_live_ref[r] != 0, run_list) if use_runs else 0
        )
        n_merge, n_live = compact_comp(comp_tokens)

        # Phase 1: every read in flight.
        if paired:
            over(n_main, main_list, lambda i: main_copy(i, True).start())
            if use_runs:
                over(n_runs, run_list, lambda r: run_copy(r, True).start())
        over(n_merge, merge_list, lambda i: comp_copy(i, True).start())

        # Ring staging overlaps the reads, only when some ring id is live.
        @pl.when(ring_any_ref[0] != 0)
        def stage_rings():
            for lane in range(count):
                typed_ldst.store_typed(
                    staged.at[lane, pl.ds(0, ring_tokens)], ring_ref[lane]
                )
                typed_ldst.store_typed(
                    staged.at[lane, pl.ds(ring_tokens, tag_tokens)], tags_ref[lane]
                )

        if paired:
            over(n_main, main_list, lambda i: main_copy(i, True).wait())
            if use_runs:
                over(n_runs, run_list, lambda r: run_copy(r, True).wait())
            old = main_buf.bitcast(jnp.uint32).reshape(-1, width)
            new = rows_ref.bitcast(jnp.uint32).reshape(-1, width)
            old[...] = (old[...] & jnp.uint32(keep)) | new[...]
        over(n_merge, merge_list, lambda i: comp_copy(i, True).wait())

        # Phase 2: merge every compressed pair in one vector select. Rows that are not live
        # are never written back, so what they hold does not matter.
        bf = comp_buf.bitcast(jnp.bfloat16) if pool.dtype != jnp.bfloat16 else comp_buf
        bf[...] = jnp.where(ctake_ref[...] != 0, cvalues_ref[...], bf[...])

        # Phase 3: every write in flight.
        @pl.when(unique_ref[0] != 0)
        def write_main_unique():
            over(n_main, main_list, lambda i: main_copy(i, False).start())
            if use_runs:
                over(n_runs, run_list, lambda r: run_copy(r, False).start())

        @pl.when(unique_ref[0] == 0)
        def write_main_ordered():
            def store(i):
                copy = main_copy(i, False)
                copy.start()
                copy.wait()

            over(n_main, main_list, store)

        over(n_live, live_list, lambda i: comp_copy(i, False).start())

        @pl.when(ring_any_ref[0] != 0)
        def write_rings():
            def body(lane, carry):
                @pl.when(ids_ref[lane] > 0)
                def _():
                    ring_copy(lane).start()

                return carry

            jax.lax.fori_loop(0, count, body, 0)

        @pl.when(unique_ref[0] != 0)
        def wait_main_unique():
            over(n_main, main_list, lambda i: main_copy(i, False).wait())
            if use_runs:
                over(n_runs, run_list, lambda r: run_copy(r, False).wait())

        over(n_live, live_list, lambda i: comp_copy(i, False).wait())

        @pl.when(ring_any_ref[0] != 0)
        def wait_rings():
            def body(lane, carry):
                @pl.when(ids_ref[lane] > 0)
                def _():
                    ring_copy(lane).wait()

                return carry

            jax.lax.fori_loop(0, count, body, 0)

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
                full(comp_take.shape),
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
                pltpu.SMEM((tokens,), jnp.int32),
                pltpu.SMEM((runs,), jnp.int32),
                pltpu.SMEM((max(comp_tokens, 1),), jnp.int32),
                pltpu.SMEM((max(comp_tokens, 1),), jnp.int32),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={18: 0},
        name=D73_LEAN_KERNEL,
    )(
        page_safe,
        offset,
        row_valid,
        unique,
        comp_page.astype(jnp.int32),
        comp_offset.astype(jnp.int32),
        comp_mode,
        ring_ids,
        first,
        run_page.astype(jnp.int32),
        run_offset.astype(jnp.int32),
        run_live,
        ring_any,
        main_rows.astype(pool.dtype),
        comp_values,
        comp_take,
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


# q38 L2 D73MK begin
# D73's observable engagement markers. D73 = VLLM_TPU_QSA_FUSED_KV_WRITE=1 (+ VLLM_TPU_QSA_FUSED_KV_WRITE_LEAN=1): the stock
# QSA cache writes of decode / verify-sized steps (`pooled_qsa.qsa_cache_writes`' fused branch) go through this module's one
# Pallas writer (`fused_layer_write`; `_lean_layer_write` under LEAN). Log only: nothing traced or written changes. Two lines:
# - the startup line (`log_d73_state`, from Qwen4ExpQSA.bind_pool), once per process and so once per rank, flags on or off:
#   the two env values as this code reads them, the writer the fused branch traces, whether the whole step's D73 gate is in
#   the imported code (`_d73_wholestep_gate`) and the peer-replay switch;
# - the trace-time line (`_d73_traced`, at `fused_layer_write`'s writer choice), once per (tokens, requests, lean, kernel,
#   site) per process.
# The trace-time line prints only where Python traces the writer. A program this process does not lower prints nothing for
# it: vLLM's AOT artifact load (`TpuCompilerAdaptor.load` -> `_replay`, which skips the FX->MLIR lowering), a peer rank's
# replayed artifact (TPU_PRECOMPILE_PEER_REPLAY=1, `_replay_from_peer`), and JAX's own trace cache for a second identical
# lowering. The startup line does not depend on tracing. Appended at the end of the module; the edits above keep their lines.
D73_ENV = "VLLM_TPU_QSA_FUSED_KV_WRITE"
D73_FUSED_KERNEL = "qwen4_exp_qsa_fused_kv_write"
D73_LEAN_KERNEL = "qwen4_exp_qsa_lean_kv_write"
D73_TRACES: dict = {}  # (tokens, requests, lean, kernel, site) -> traces seen by this process
_D73_STATE_LOGGED = False
_D73_ENTRY_DEPTH = 12
_D73_STACK_DEPTH = 400


def _d73_site() -> str:
    """``<entry>@<program>`` for the trace-time line; best effort, never raises.

    ``entry``: the nearest ``pooled_qsa_*`` step (the custom op's JAX function, e.g. O1s
    ``pooled_qsa_request_select_dcp_step``, ``pooled_qsa_request_plain_dcp_step``, ``pooled_qsa_rows_dcp_step``, the
    ``pooled_qsa_shared_*`` draft ops) or ``wholestep_qsa_writes`` above the writer, else the writer's direct caller.
    ``program``: the prefix of vLLM's compile-cache dir of the program being lowered (``<hash>/rank_<r>_<dp>/<prefix>``:
    backbone, model_mk_whole, model_mk_segments, ...), read off the ``TpuCompilerAdaptor`` frame on this thread's stack;
    ``none`` without such a frame (a direct or eager trace, or a lowering on another thread), ``?`` without a cache dir.
    """
    try:
        import sys

        caller = sys._getframe(3)  # 0 _d73_site, 1 _d73_traced, 2 fused_layer_write, 3 its caller
        entry = caller.f_code.co_name
        f, depth = caller, 0
        while f is not None and depth < _D73_ENTRY_DEPTH:
            name = f.f_code.co_name
            if name.startswith("pooled_qsa_") or name == "wholestep_qsa_writes":
                entry = name
                break
            f, depth = f.f_back, depth + 1
        program = "none"
        f, depth = caller, 0
        while f is not None and depth < _D73_STACK_DEPTH:
            if f.f_code.co_name in ("_run_backend", "compile", "load"):
                owner = f.f_locals.get("self")
                if type(owner).__name__ == "TpuCompilerAdaptor":
                    cache_dir = getattr(owner, "cache_dir", None)
                    program = os.path.basename(os.path.normpath(cache_dir)) if cache_dir else "?"
                    break
            f, depth = f.f_back, depth + 1
        return f"{entry}@{program}"
    except Exception:  # noqa: BLE001 - a log label must never break a trace
        return "?@?"


def _d73_traced(tokens: int, requests: int) -> bool:
    """``fused_layer_write``'s writer choice, ``lean_writer_enabled()`` exactly (True: the lean writer; an invalid LEAN
    value raises before anything is logged), plus D73's trace-time line, once per (tokens, requests, lean, kernel, site)
    per process: ``q38 D73 fused KV write traced: tokens <T>, requests <R>, lean <0|1>, kernel <pallas_call name>,
    site <entry>@<program>``. ``tokens`` = the step's token rows, ``requests`` = its request slots (ring rows)."""
    lean = lean_writer_enabled()
    key = (int(tokens), int(requests), int(lean), D73_LEAN_KERNEL if lean else D73_FUSED_KERNEL, _d73_site())
    D73_TRACES[key] = D73_TRACES.get(key, 0) + 1
    if D73_TRACES[key] == 1:
        from vllm_torchtpu.logger import init_logger

        init_logger(__name__).info(
            "q38 D73 fused KV write traced: tokens %d, requests %d, lean %d, kernel %s, site %s", *key
        )
    return lean


def _d73_wholestep_gate(source: str | None = None) -> str:
    """``present`` when ``megakernel_runtime.qwen38_wholestep_cache`` has no code that reads VLLM_TPU_QSA_FUSED_KV_WRITE
    (q38 L2 D73 gate a4333fb7d: the whole step's post-kernel writes take the batched writer whatever D73's env, ``if
    batched:``); ``absent`` when it has (the pre-gate ``if batched and not envs.VLLM_TPU_QSA_FUSED_KV_WRITE:``);
    ``unknown (<error>)`` when its source cannot be read. Docstrings and comments do not count. ``source``: that module's
    text (tests); default the module this process imports."""
    try:
        import ast

        if source is None:
            import inspect

            from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_cache

            source = inspect.getsource(qwen38_wholestep_cache)
        tree = ast.parse(source)
        docs = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body:
                first = body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                    docs.add(id(first.value))
        reads = sum(
            1
            for node in ast.walk(tree)
            if (isinstance(node, ast.Attribute) and node.attr == D73_ENV)
            or (isinstance(node, ast.Name) and node.id == D73_ENV)
            or (isinstance(node, ast.Constant) and node.value == D73_ENV and id(node) not in docs)
        )
        return "absent" if reads else "present"
    except Exception as exc:  # noqa: BLE001 - a log label must never break startup
        return f"unknown ({type(exc).__name__})"


def log_d73_state(rank: int) -> bool:
    """D73's startup line, once per process (True when this call logged it), flags on or off:
    ``q38 D73 state: fused_kv_write <0|1|invalid> (raw <'value'|unset>), lean <0|1|invalid> (raw <'value'|unset>),
    stock writer <pallas_call name|off|invalid>, wholestep gate <present|absent|unknown (..)>, peer replay <0|1|?>,
    rank <r>``. The values are the ones this code acts on (``envs.VLLM_TPU_QSA_FUSED_KV_WRITE``, ``lean_writer_enabled()``,
    ``envs.TPU_PRECOMPILE_PEER_REPLAY``); "stock writer" is the kernel ``qsa_cache_writes``' fused branch traces."""
    global _D73_STATE_LOGGED
    if _D73_STATE_LOGGED:
        return False
    _D73_STATE_LOGGED = True
    from vllm_torchtpu import envs
    from vllm_torchtpu.logger import init_logger

    def raw(name):
        value = os.environ.get(name)
        return "unset" if value is None else repr(value)

    try:
        fused = int(bool(envs.VLLM_TPU_QSA_FUSED_KV_WRITE))
    except Exception:  # noqa: BLE001 - reported, not raised, in a log line
        fused = "invalid"
    try:
        lean = int(lean_writer_enabled())
    except Exception:  # noqa: BLE001
        lean = "invalid"
    if fused == 0:
        writer = "off"
    elif fused == 1 and lean in (0, 1):
        writer = D73_LEAN_KERNEL if lean else D73_FUSED_KERNEL
    else:
        writer = "invalid"
    try:
        replay = int(bool(envs.TPU_PRECOMPILE_PEER_REPLAY))
    except Exception:  # noqa: BLE001
        replay = "?"
    init_logger(__name__).info(
        "q38 D73 state: fused_kv_write %s (raw %s), lean %s (raw %s), stock writer %s, wholestep gate %s, "
        "peer replay %s, rank %d",
        fused, raw(D73_ENV), lean, raw(LEAN_ENV), writer, _d73_wholestep_gate(), replay, int(rank),
    )
    return True
# q38 L2 D73MK end
