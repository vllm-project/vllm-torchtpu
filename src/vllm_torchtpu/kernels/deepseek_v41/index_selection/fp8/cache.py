# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature-major FP8 pages with an in-page FP32 scale per key.

For token t and feature d, byte d%4 of word [d//4,t] contains K[t,d].
Field 32 holds scales[t] as FP32 bits. Each [33,page_size] page is rounded
to whole owner rows. Unwritten keys are masked by sequence lengths.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .prepare import prepare_sorted_fp8, supports_sorted_fp8

PAGE_RECORDS = 128
PAGE_ROWS = 33

# Stage up to 128 touched pages, stopping at the sorted invalid-record tail.
PACKED_FP8_WRITE_BATCH = 128


def quantize_index_rows(values: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return E4M3 values, FP32 scales and E8M0 scale bytes for D128 vectors."""
    # One power-of-two scale covers each post-RoPE vector, independent of layout.
    if values.shape[-1] != 128:
        raise ValueError("V4.1 FP8 index vectors must have 128 channels")
    values = values.astype(jnp.float32)
    amax = jnp.maximum(jnp.abs(values).max(axis=-1), 1e-4)
    exponent = jnp.ceil(jnp.log2(amax / 448.0)).astype(jnp.int32)
    scale_bytes = (exponent + 127).astype(jnp.uint8)
    scales = jax.lax.bitcast_convert_type(
        scale_bytes.astype(jnp.uint32) << 23, jnp.float32
    )
    quantized = (values / scales[..., None]).astype(jnp.float8_e4m3fn)
    return quantized, scales, scale_bytes


@dataclass(frozen=True, init=False)
class PackedFP8Layout:
    """Location of this layer's pages inside one or two shared word-row owners."""

    pages: int
    page_stride: int
    page_offset: int
    page_size: int

    def __init__(
        self,
        pages: int,
        page_stride: int | None = None,
        page_offset: int = 0,
        page_size: int = PAGE_RECORDS,
    ) -> None:
        rows = (PAGE_ROWS * page_size + PAGE_RECORDS - 1) // PAGE_RECORDS
        for name, value in (
            ("pages", pages),
            ("page_stride", rows if page_stride is None else page_stride),
            ("page_offset", page_offset),
            ("page_size", page_size),
        ):
            object.__setattr__(self, name, value)

    @property
    def page_rows(self) -> int:
        return (PAGE_ROWS * self.page_size + PAGE_RECORDS - 1) // PAGE_RECORDS

    def validate(self, bank_rows: tuple[int, ...]) -> None:
        if self.page_size < 1 or self.page_size & (self.page_size - 1):
            raise ValueError("Packed FP8 page capacity must be a power of two")
        if len(bank_rows) not in (1, 2) or any(
            rows < self.page_rows for rows in bank_rows
        ):
            raise ValueError("Packed FP8 cache requires one or two nonempty banks")
        if self.pages < 1 or self.page_stride < self.page_rows or self.page_offset < 0:
            raise ValueError("Packed FP8 pages require nonoverlapping page slots")
        end = self.page_offset + (self.pages - 1) * self.page_stride + self.page_rows
        if end > sum(bank_rows) or max(bank_rows) >= 1 << 29:
            raise ValueError("Packed FP8 page addresses exceed their owners")
        if len(bank_rows) == 2:
            page = min(
                self.pages - 1,
                max(0, (bank_rows[0] - self.page_offset) // self.page_stride),
            )
            start = self.page_offset + page * self.page_stride
            if start < bank_rows[0] < start + self.page_rows:
                raise ValueError("A packed FP8 page must not cross the bank boundary")


def _record_addresses(records: jax.Array, layout: PackedFP8Layout):
    valid = (records >= 0) & (records < layout.pages * layout.page_size)
    safe = jnp.clip(records, 0, layout.pages * layout.page_size - 1)
    base = layout.page_offset + (safe // layout.page_size) * layout.page_stride
    return base, safe % layout.page_size, valid


def _sort_by_slot(bases: jax.Array, columns: jax.Array):
    """Order records by page, then column, so consecutive slots merge as one run."""
    order = jnp.arange(bases.size, dtype=jnp.int32)
    return jax.lax.sort((bases, columns.astype(jnp.int32), order), num_keys=2)


def write_fp8_records(
    banks: tuple[jax.Array, ...],
    records: jax.Array,
    keys: jax.Array,
    layout: PackedFP8Layout,
) -> tuple[jax.Array, ...]:
    """Quantize and write new keys, using fused preparation for verified decode."""
    if not supports_sorted_fp8(banks, records, keys, layout.page_size):
        return _write_fp8_records_native(banks, records, keys, layout)
    layout.validate(tuple(bank.shape[0] for bank in banks))
    bases, columns, valid = _record_addresses(records, layout)
    bases = jnp.where(valid, bases, -1)
    count = records.size
    first_rows = banks[0].shape[0]
    total_rows = sum(bank.shape[0] for bank in banks)
    valid = (bases >= 0) & (bases + layout.page_rows <= first_rows)
    valid |= (bases >= first_rows) & (bases + layout.page_rows <= total_rows)
    bases, _, order = _sort_by_slot(jnp.where(valid, bases, total_rows), columns)
    _, scales, _ = quantize_index_rows(keys)
    columns, values = prepare_sorted_fp8(keys, columns, order, scales)
    return _write_sorted_packed_pages(
        banks,
        bases,
        columns,
        values,
        count=count,
        fields=PAGE_ROWS,
        page_size=layout.page_size,
        interpret=False,
    )


def _write_fp8_records_native(
    banks: tuple[jax.Array, ...],
    records: jax.Array,
    keys: jax.Array,
    layout: PackedFP8Layout,
) -> tuple[jax.Array, ...]:
    """Quantize only new keys and update their columns; valid slots are unique."""
    layout.validate(tuple(bank.shape[0] for bank in banks))
    if keys.shape != (records.size, PAGE_RECORDS) or records.ndim != 1:
        raise ValueError("Expected one logical record slot per D128 key")
    quantized, scales, _ = quantize_index_rows(keys)
    bits = jax.lax.bitcast_convert_type(quantized, jnp.uint8).astype(jnp.uint32)
    bits = bits.reshape(-1, 32, 4)
    words = bits[:, :, 0] | (bits[:, :, 1] << 8)
    words |= (bits[:, :, 2] << 16) | (bits[:, :, 3] << 24)
    values = jnp.concatenate(
        (
            words.astype(jnp.int32),
            jax.lax.bitcast_convert_type(scales, jnp.int32)[:, None],
        ),
        axis=1,
    )
    base, columns, valid = _record_addresses(records, layout)
    if jax.default_backend() == "tpu":
        return write_packed_pages(
            banks,
            jnp.where(valid, base, -1),
            columns,
            values,
            page_size=layout.page_size,
        )
    updated = []
    offset = 0
    for bank in banks:
        words = (
            base[:, None] * PAGE_RECORDS
            + jnp.arange(PAGE_ROWS)[None, :] * layout.page_size
            + columns[:, None]
        )
        local = words // PAGE_RECORDS - offset
        active = valid[:, None] & (local >= 0) & (local < bank.shape[0])
        local = jnp.where(active, local, bank.shape[0])
        updated.append(bank.at[local, words % PAGE_RECORDS].set(values, mode="drop"))
        offset += bank.shape[0]
    return tuple(updated)


def read_fp8_records(
    banks: tuple[jax.Array, ...], records: jax.Array, layout: PackedFP8Layout
) -> jax.Array:
    """Decode selected records to BF16 for eager consumers; invalid slots are zero."""
    layout.validate(tuple(bank.shape[0] for bank in banks))
    shape = records.shape
    base, columns, valid = _record_addresses(records.reshape(-1), layout)
    addresses = (
        base[:, None] * PAGE_RECORDS
        + jnp.arange(PAGE_ROWS)[None, :] * layout.page_size
        + columns[:, None]
    )
    words = jnp.zeros(addresses.shape, jnp.int32)
    offset = 0
    for bank in banks:
        local = addresses // PAGE_RECORDS - offset
        active = valid[:, None] & (local >= 0) & (local < bank.shape[0])
        gathered = bank[jnp.clip(local, 0, bank.shape[0] - 1), addresses % PAGE_RECORDS]
        words = jnp.where(active, gathered, words)
        offset += bank.shape[0]
    bits = (words[:, :32, None].astype(jnp.uint32) >> (jnp.arange(4) * 8)) & 255
    values = jax.lax.bitcast_convert_type(
        bits.astype(jnp.uint8).reshape(-1, PAGE_RECORDS), jnp.float8_e4m3fn
    )
    scales = jax.lax.bitcast_convert_type(words[:, 32], jnp.float32)
    result = (values.astype(jnp.float32) * scales[:, None]).astype(jnp.bfloat16)
    return jnp.where(valid[:, None], result, 0).reshape(*shape, PAGE_RECORDS)


def write_packed_pages(
    banks: tuple[jax.Array, ...],
    bases: jax.Array,
    columns: jax.Array,
    values: jax.Array,
    *,
    page_size: int | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, ...]:
    """Update unique token slots in one or two banks using global row bases."""
    count, fields = values.shape
    if not count:
        return banks
    page_tokens = banks[0].shape[1]
    page_size = page_tokens if page_size is None else page_size
    page_rows = pl.cdiv(fields * page_size, page_tokens)
    padded_fields = page_rows * page_tokens // page_size
    first_rows = banks[0].shape[0]
    total_rows = sum(bank.shape[0] for bank in banks)
    valid = (bases >= 0) & (bases + page_rows <= first_rows)
    if len(banks) == 2:
        valid |= (bases >= first_rows) & (bases + page_rows <= total_rows)
    bases, columns, order = _sort_by_slot(jnp.where(valid, bases, total_rows), columns)
    values = values[order]
    values = jnp.pad(values.T, ((0, padded_fields - fields), (0, -count % page_tokens)))

    return _write_sorted_packed_pages(
        banks,
        bases,
        columns,
        values,
        count=count,
        fields=fields,
        page_size=page_size,
        interpret=interpret,
    )


def _write_sorted_packed_pages(
    banks,
    bases,
    columns,
    values,
    *,
    count: int,
    fields: int,
    page_size: int,
    interpret: bool,
):
    page_tokens = banks[0].shape[1]
    page_rows = pl.cdiv(fields * page_size, page_tokens)
    padded_fields = page_rows * page_tokens // page_size
    first_rows = banks[0].shape[0]
    total_rows = sum(bank.shape[0] for bank in banks)
    batch = min(count, PACKED_FP8_WRITE_BATCH)
    steps = pl.cdiv(count, batch)
    # Page and run boundaries come from vector ops, so the kernel loops over
    # pages and runs of consecutive slots rather than over records. Sorting
    # put valid records first; the last entry of each list is its end.
    bases, columns = bases[:count], columns[:count].astype(jnp.int32)
    valid = bases < total_rows
    page_first = valid & (bases != jnp.pad(bases[:-1], (1, 0), constant_values=-1))
    run_first = page_first | (
        valid & (columns != jnp.pad(columns[:-1], (1, 0), constant_values=-2) + 1)
    )
    run_ends = jnp.cumsum(run_first, dtype=jnp.int32)
    page_ends = jnp.cumsum(page_first, dtype=jnp.int32)
    used = valid.sum(dtype=jnp.int32)
    index = jnp.arange(steps * batch + 1, dtype=jnp.int32)

    def marked(first, ends, values, end):
        # The value at the i-th marked record, or `end` past the last mark:
        # one fused compare and reduce, with no gather or scatter.
        hit = first[None, :] & (ends[None, :] == index[:, None] + 1)
        found = jnp.where(hit, values[None, :], 0).sum(1, dtype=jnp.int32)
        return jnp.where(index < ends[-1], found, end)

    run_starts = marked(run_first, run_ends, jnp.arange(count, dtype=jnp.int32), used)
    page_bases = marked(page_first, page_ends, bases, total_rows)
    page_runs = marked(page_first, page_ends, run_ends - 1, run_ends[-1])

    def kernel(page_base_ref, page_run_ref, run_ref, column_ref, value_ref, *refs):
        outputs = refs[len(banks) : 2 * len(banks)]
        pages, sem = refs[2 * len(banks) :]

        def each_page(action):
            def body(slot):
                action(slot, pl.program_id(0) * batch + slot)
                return slot + 1

            jax.lax.while_loop(
                lambda slot: (
                    (slot < batch)
                    & (page_base_ref[pl.program_id(0) * batch + slot] < total_rows)
                ),
                body,
                jnp.int32(0),
            )

        def copies(*, write, wait):
            def body(slot, page):
                base = page_base_ref[page]

                def move(bank, offset):
                    hbm = outputs[bank].at[pl.ds(base - offset, page_rows), :]
                    vmem = pages.at[slot]
                    src, dst = (vmem, hbm) if write else (hbm, vmem)
                    copy = pltpu.make_async_copy(src, dst, sem)
                    if wait:
                        copy.wait()
                    else:
                        copy.start()

                if len(banks) == 1:
                    move(0, 0)
                else:
                    jax.lax.cond(
                        base < first_rows,
                        lambda: move(0, 0),
                        lambda: move(1, first_rows),
                    )

            each_page(body)

        copies(write=False, wait=False)
        copies(write=False, wait=True)

        value_lanes = value_ref.shape[1]
        blocks = max(1, page_size // page_tokens)
        lanes = jax.lax.broadcasted_iota(jnp.int32, (padded_fields, page_tokens), 1)
        positions = jnp.arange(page_rows * page_tokens).reshape(page_rows, page_tokens)

        def window(slot, start, count):
            """Merge records start..start+count, whose columns are consecutive."""
            column = column_ref[start]
            # Column c takes value lane `start - column + c`: aligned lane
            # blocks from `origin`, rotated left by `skew`. A block clamped to
            # the value bounds holds only lanes outside the run, masked below.
            source = start - column + blocks * page_tokens  # biased >= 0
            skew = source % page_tokens
            origin = source - skew - blocks * page_tokens
            shift = (page_tokens - skew) % page_tokens

            def block(index):
                index = jnp.clip(
                    origin + index * page_tokens, 0, value_lanes - page_tokens
                )
                return pltpu.roll(
                    value_ref[
                        :, pl.ds(pl.multiple_of(index, page_tokens), page_tokens)
                    ],
                    shift,
                    1,
                )

            def lane_block(index):
                return jnp.where(
                    lanes < page_tokens - skew, block(index), block(index + 1)
                )

            if page_size <= page_tokens:
                word = lane_block(0)[:, :page_size].reshape(page_rows, page_tokens)
                offset = positions % page_size - column
                pages[slot] = jnp.where(
                    (offset >= 0) & (offset < count) & (positions < fields * page_size),
                    word,
                    pages[slot],
                )
                return
            # Field f's columns j*128..j*128+127 sit in page row f * blocks + j;
            # update only the lane blocks the run reaches.
            for index in range(blocks):
                low = index * page_tokens

                @pl.when((column < low + page_tokens) & (column + count > low))
                def _():
                    rows = pl.ds(index, padded_fields, stride=blocks)
                    offset = lanes + low - column
                    pages[slot, rows, :] = jnp.where(
                        (offset >= 0) & (offset < count),
                        lane_block(index),
                        pages[slot, rows, :],
                    )

        def merge(slot, page):
            def run(index, carry):
                start = run_ref[index]
                window(slot, start, run_ref[index + 1] - start)
                return carry

            jax.lax.fori_loop(page_run_ref[page], page_run_ref[page + 1], run, 0)

        each_page(merge)
        copies(write=True, wait=False)
        copies(write=True, wait=True)

    return pl.pallas_call(
        kernel,
        out_shape=tuple(jax.ShapeDtypeStruct(bank.shape, bank.dtype) for bank in banks),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(steps,),
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.VMEM),
                *(pl.BlockSpec(memory_space=pltpu.HBM) for _ in banks),
            ),
            out_specs=tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in banks),
            scratch_shapes=(
                pltpu.VMEM((batch, page_rows, page_tokens), jnp.int32),
                pltpu.SemaphoreType.DMA,
            ),
        ),
        input_output_aliases={5 + i: i for i in range(len(banks))},
        interpret=pltpu.InterpretParams() if interpret else False,
        name="write_packed_fp8_pages",
    )(page_bases, page_runs, run_starts, columns, values, *banks)
