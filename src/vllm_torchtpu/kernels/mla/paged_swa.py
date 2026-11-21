# SPDX-License-Identifier: Apache-2.0
"""Publish sliding-window attention into the shared INT32 word pool."""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


def swa_word_pages(
    raw: jax.Array | tuple[jax.Array, jax.Array], page_size: int
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """View shared INT32 [rows,128] owners as 128-word SWA pages.

    The allocator's physical page stride/offset must map to whole 2*S-row
    pages. Banks split on page boundaries. This does not change ownership or
    allocate a second cache. Each token pair occupies four 128-word rows.
    """
    if type(page_size) is not int or (
        page_size not in (8, 16, 32, 64) and (page_size < 128 or page_size % 128)
    ):
        raise ValueError(
            "SWA page size must be 8,16,32,64 or a positive multiple of 128"
        )
    if isinstance(raw, tuple):
        return tuple(swa_word_pages(bank, page_size) for bank in raw)
    if (
        raw.ndim != 2
        or raw.shape[1] != 128
        or raw.dtype != jnp.int32
        or raw.shape[0] % (2 * page_size)
    ):
        raise ValueError("SWA owner must be INT32 [rows,128] with whole physical pages")
    return raw.reshape(-1, 2 * page_size, 128)


@functools.partial(jax.jit, donate_argnums=(0,), static_argnames=("page_size",))
def write_swa_word_pool(
    raw: jax.Array | tuple[jax.Array, jax.Array],
    records: jax.Array,
    values: jax.Array,
    *,
    page_size: int,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Publish SWA while preserving the shared flat word-row owner interface."""
    swa_word_pages(raw, page_size)
    return _publish_swa_sc(raw, records, values, page_size)


def _publish_swa_sc(raw, records, values, page_size, planned=None):
    """Publish each token pair once, preserving any unmodified BF16 half.

    ``planned`` (plan, group, shift) packs ``records`` from a
    `swa_publish_plan` of the unshifted records instead of sorting them again.
    """
    if values.ndim != 2 or values.shape[1] != 512 or values.dtype != jnp.bfloat16:
        raise ValueError("SWA values must be BF16 [N,512]")
    if records.shape != values.shape[:1] or records.dtype != jnp.int32:
        raise ValueError("SWA records must be INT32 [N]")
    if not records.size:
        return raw
    banks = raw if isinstance(raw, tuple) else (raw,)
    capacity = sum(bank.shape[0] for bank in banks) // 2
    info = pltpu.get_tpu_info().sparse_core
    assert info is not None
    lanes, cores = info.num_lanes, info.num_cores * info.num_subcores
    from vllm_torchtpu.kernels.experimental import decode_preparation as prep

    if prep.admit(
        "swa_pack",
        None if page_size == 64 else f"SWA page size {page_size}, kernel 64",
        tokens=records.shape[0],
        buckets=prep.SWA_PACK_TOKENS,
    ):
        padded = -(-records.size // (lanes * cores)) * (lanes * cores)
        ids, words, masks = prep.swa_word_pool_pack(
            records, values, capacity=capacity, padded=padded
        )
    elif planned is not None:
        ids, words, masks = _pack_planned(values, *planned, page_size, capacity)
    else:
        ids, words, masks = _pack_native(records, values, capacity, page_size)
        padding = -ids.size % (lanes * cores)
        ids = jnp.pad(ids, (0, padding), constant_values=-1)
        words = jnp.pad(words, ((0, padding), (0, 0)))
        masks = jnp.pad(masks, ((0, padding), (0, 0)))
    return _publish_packed(banks, ids, words, masks, info, raw)


def _pack_native(records, values, capacity, page_size):
    """XLA sort/pair of SWA records into 128-word physical rows (unpadded)."""
    valid = (records >= page_size) & (records < capacity)
    local = jnp.where(valid, records, capacity)
    order = jnp.argsort(local, stable=True)
    ids = local[order]
    next_ids = jnp.concatenate((ids[1:], jnp.array([capacity], jnp.int32)))
    pairs = ids // 2
    boundary = (ids < capacity) & jnp.concatenate(
        (jnp.ones(1, jnp.bool_), pairs[1:] != pairs[:-1])
    )
    even = ids % 2 == 0
    has_odd = (~even) | (next_ids // 2 == pairs)
    bits = jax.lax.bitcast_convert_type(values[order], jnp.uint16).astype(jnp.uint32)
    next_bits = jnp.concatenate((bits[1:], jnp.zeros((1, 512), jnp.uint32)))
    low = jnp.where(even[:, None], bits, jnp.uint32(0))
    high = jnp.where(even[:, None], next_bits, bits)
    high = jnp.where(has_odd[:, None], high, jnp.uint32(0))
    words = jax.lax.bitcast_convert_type(low | (high << 16), jnp.int32).reshape(-1, 128)
    masks = jnp.where(even, jnp.int32(65535), jnp.int32(0)) | jnp.where(
        has_odd, jnp.int32(-65536), jnp.int32(0)
    )
    masks = jnp.broadcast_to(masks[:, None, None], (records.size, 4, 128)).reshape(
        -1, 128
    )
    physical = pairs[:, None] * 4 + jnp.arange(4, dtype=jnp.int32)[None, :]
    physical = jnp.where(boundary[:, None], physical, -1).reshape(-1)
    return physical, words, masks


# A plan's pack keeps the layer's values, their F32 and 32-bit copies, its
# words and masks in VMEM, double-buffered: about 14 KiB per token. Larger
# calls keep the native pack.
SWA_PLAN_MAX_TOKENS = 1024


class SwaPublishPlan(NamedTuple):
    """Record-only part of SWA publishes, for G record vectors at once (one
    per SWA block table); layers using one vector differ by a whole-page row
    shift (see `swa_publish_plan`)."""

    # [G, 4, P] per sorted slot: the value row of its low halves, the value
    # row of its high halves, and the masks (0 or -1) keeping either.
    slots: jax.Array
    lead: jax.Array  # [G, 1, 4P] per 128-word row: its pair's record, or -1


def swa_plan_slots(records: int) -> int:
    """Slots P of a plan: 4P rows fill whole SparseCore writer blocks."""
    info = pltpu.get_tpu_info().sparse_core
    assert info is not None
    block = info.num_lanes * info.num_cores * info.num_subcores
    return -(-4 * records // block) * block // 4


def swa_publish_plan(records: jax.Array) -> SwaPublishPlan:
    """Sort and pair each row of ``records`` [G, N] once for every layer that
    publishes it.

    A row shift by whole pages keeps the sort order, the pairs and their
    parity, so only the bounds check and the row ids move per layer
    (`write_swa_word_pool_planned`). Every non-padding record is planned here;
    the per-layer check drops what `_pack_native` would drop for the shifted
    records. Both slots of a pair share a page, so they share that verdict.
    """
    if records.ndim != 2 or records.dtype != jnp.int32:
        raise ValueError("SWA records must be INT32 [G, N]")
    return jax.vmap(_plan_records)(records)


def _plan_records(records):
    n = records.size
    pad = swa_plan_slots(n) - n
    padding = jnp.iinfo(jnp.int32).max
    key = jnp.where(records >= 0, records, padding)
    order = jnp.argsort(key, stable=True).astype(jnp.int32)
    ids = key[order]
    pairs = ids // 2
    first = (ids < padding) & jnp.concatenate(
        (jnp.ones(1, jnp.bool_), pairs[1:] != pairs[:-1])
    )[:n]
    next_ids = jnp.concatenate((ids[1:], jnp.array([padding], jnp.int32)))[:n]
    next_order = jnp.concatenate((order[1:], jnp.zeros(1, jnp.int32)))[:n]
    even = ids % 2 == 0
    has_odd = (~even) | (next_ids // 2 == pairs)
    slots = jnp.stack(
        (
            order,
            jnp.where(even, next_order, order),
            jnp.where(even, jnp.int32(-1), jnp.int32(0)),
            jnp.where(has_odd, jnp.int32(-1), jnp.int32(0)),
        )
    )
    lead = jnp.pad(jnp.where(first, ids, -1), (0, pad), constant_values=-1)
    return SwaPublishPlan(
        slots=jnp.pad(slots, ((0, 0), (0, pad))), lead=jnp.repeat(lead, 4)[None, :]
    )


@functools.partial(
    jax.jit, donate_argnums=(0,), static_argnames=("group", "shift", "page_size")
)
def write_swa_word_pool_planned(
    raw: jax.Array | tuple[jax.Array, jax.Array],
    records: jax.Array,
    values: jax.Array,
    plan: SwaPublishPlan,
    *,
    group: int,
    shift: int,
    page_size: int,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """`write_swa_word_pool` of ``records``: row ``group`` of ``plan``'s
    records moved by ``shift`` rows (``where(r >= 0, r + shift, r)``).

    Where the fused TC pack is not admitted, one kernel packs the values and
    moves the row ids; the plan supplies everything that depends on records.
    """
    swa_word_pages(raw, page_size)
    if records.size > SWA_PLAN_MAX_TOKENS:
        raise ValueError(f"SWA plans cover at most {SWA_PLAN_MAX_TOKENS} records")
    return _publish_swa_sc(raw, records, values, page_size, (plan, group, shift))


def _pack_planned(values, plan, group, shift, page_size, capacity, interpret=False):
    """Row ids, words and masks of `_pack_native` for planned, shifted records."""
    if shift % page_size:
        raise ValueError("SWA plans move by whole pages only")
    tokens, rows = values.shape[0], plan.lead.shape[-1]

    def kernel(slots_ref, values_ref, lead_ref, ids_ref, words_ref, masks_ref, bits):
        # The F32 high half holds the BF16 bits: the convert neither flushes
        # denormals nor quiets NaNs (test_csa2_word_pool publishes all 2^16
        # patterns; checked on v7x).
        widened = values_ref[...].astype(jnp.float32)
        bits[...] = jax.lax.shift_right_logical(
            jax.lax.bitcast_convert_type(widened, jnp.int32), 16
        )

        def pack(slot, carry):
            keep_low, keep_high = slots_ref[2, slot], slots_ref[3, slot]
            low = bits[pl.ds(slots_ref[0, slot], 1), :] & keep_low
            high = bits[pl.ds(slots_ref[1, slot], 1), :] & keep_high
            word = low | (high << 16)
            mask = jnp.full((1, 128), (keep_low & 65535) | (keep_high & -65536))
            for segment in range(4):  # rows 4 * slot + segment, as _pack_native
                row = pl.ds(4 * slot + segment, 1)
                words_ref[row, :] = word[:, 128 * segment : 128 * (segment + 1)]
                masks_ref[row, :] = mask
            return carry

        # Padding slots stay unwritten: their row ids are -1.
        jax.lax.fori_loop(0, tokens, pack, 0, unroll=min(tokens, 8))
        lead = lead_ref[...]
        shifted = lead + shift
        written = (lead >= 0) & (shifted >= page_size) & (shifted < capacity)
        segment = jax.lax.broadcasted_iota(jnp.int32, lead.shape, 1) & 3
        ids_ref[...] = jnp.where(written, (shifted >> 1) * 4 + segment, -1)

    def whole(*shape):
        return pl.BlockSpec(shape, lambda _: (0,) * len(shape))

    ids, words, masks = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((1, rows), jnp.int32),
            jax.ShapeDtypeStruct((rows, 128), jnp.int32),
            jax.ShapeDtypeStruct((rows, 128), jnp.int32),
        ),
        # One step: the grid only lets the specs select this layer's group.
        grid=(1,),
        in_specs=(
            pl.BlockSpec(
                (None, 4, rows // 4),
                lambda _: (group, 0, 0),
                memory_space=pltpu.SMEM,
            ),
            whole(tokens, 512),
            pl.BlockSpec((None, 1, rows), lambda _: (group, 0, 0)),
        ),
        out_specs=(whole(1, rows), whole(rows, 128), whole(rows, 128)),
        scratch_shapes=(pltpu.VMEM((tokens, 512), jnp.int32),),
        name="swa_planned_pack",
        interpret=interpret,
        compiler_params=None
        if interpret
        else pltpu.CompilerParams(
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes // 2
        ),
    )(plan.slots, values, plan.lead)
    return ids.reshape(-1), words, masks


def _publish_packed(banks, ids, words, masks, info, raw):
    """Read-modify-write packed SWA rows into the word pool on SparseCore.

    Row r's mask is ``masks[r % len(masks)]``: the fused pack shares one mask
    row across a pair's four segment-major rows; native masks are per row.
    """
    lanes, cores = info.num_lanes, info.num_cores * info.num_subcores
    if ids.size % masks.shape[0] or masks.shape[0] % lanes:
        raise ValueError("SWA masks must tile the row ids in whole lane blocks")
    mask_blocks = masks.shape[0] // lanes
    mesh = plsc.VectorSubcoreMesh(
        num_cores=info.num_cores,
        num_subcores=info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    owners = tuple(jax.new_ref(bank) for bank in banks)

    def kernel(ids_ref, values_ref, masks_ref, out_refs, old_ref, sem):
        core = jax.lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))

        def store(selected, word, mask):
            base = 0
            for bank in range(len(banks)):
                valid = (selected[...] >= base) & (
                    selected[...] < base + banks[bank].shape[0]
                )
                indices = plsc.Indices(
                    jnp.where(valid, selected[...] - base, -1), ignored_value=-1
                )
                read = pltpu.make_async_copy(
                    out_refs[bank].at[indices], old_ref, sem.at[bank]
                )
                read.start()
                read.wait()
                old_ref[...] = (old_ref[...] & ~mask[...]) | word[...]
                write = pltpu.make_async_copy(
                    old_ref, out_refs[bank].at[indices], sem.at[bank]
                )
                write.start()
                write.wait()
                base += banks[bank].shape[0]

        pltpu.emit_pipeline(
            store,
            grid=(ids.size // (lanes * cores),),
            in_specs=(
                pl.BlockSpec((lanes,), lambda b: (b * cores + core,)),
                pl.BlockSpec((lanes, 128), lambda b: (b * cores + core, 0)),
                pl.BlockSpec(
                    (lanes, 128), lambda b: ((b * cores + core) % mask_blocks, 0)
                ),
            ),
        )(ids_ref, values_ref, masks_ref)

    pl.kernel(
        kernel,
        mesh=mesh,
        out_type=(),
        scratch_types=(
            pltpu.VMEM((lanes, 128), jnp.int32),
            pltpu.SemaphoreType.DMA((len(banks),)),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        name="publish_swa_word_pool",
    )(ids, words, masks, owners)
    result = tuple(jax.freeze(owner) for owner in owners)
    return result if isinstance(raw, tuple) else result[0]
