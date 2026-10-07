"""Device-resident staged block tables with delta uploads.

`stage_block_table_uploads` packs every KV cache group's block-table rows into
one flat int32 buffer and uploads it on each metadata walk. At long context
that buffer is large (DeepSeek-V4.1 at 100k context, 51 groups, 16 requests:
`s32[2166208]`, 8.67 MB per rank per step). Pin memory is unsupported on TPU,
so the copy is synchronous on the host and it sits on the decode critical
path. Yet between two decode steps only a handful of entries change: one new
block id each time a request crosses a block boundary in some group.

`BlockTableDeltaStager` keeps the last uploaded packed buffer on device, plus a
host mirror of its contents, for each staging layout. On the next walk with the
same layout it diffs only the rows that may have changed against the mirror.
It uploads the changed `(flat_index, value)` pairs as one fixed-size
`int32[2, DELTA_CAPACITY]` tensor and scatters them into the device copy.
A walk with no changes uploads nothing. A walk with more changes than
`DELTA_CAPACITY`, or with a new layout, falls back to the full upload.

Rows that may have changed are found through per-row version counters. They
are installed on each upstream `BlockTable` by swapping its class for a
subclass whose row mutators (`append_row` / `add_row` / `clear_row` /
`move_row` / `swap_row` / `clear`) bump the counters. Those methods are the
only writers of the host table: no `vllm_torchtpu` code writes it directly.
`TPU_VERIFY_INCREMENTAL_BLOCK_TABLE=1` rechecks the whole table every step.

The device tensor is only ever read by `_staged_walk_products_compiled`, whose
outputs are fresh buffers, so reusing it across steps is safe. The scatter
program is functional (clone + index_put), so the previous buffer is never
written in place.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

# Changed entries one delta upload carries. Larger walks take the full path.
# 128 requests crossing a block boundary together in every one of 51 groups is
# 6528 entries.
DELTA_CAPACITY = 8192
# Staging layouts kept on device (target walk, drafter walk, chunk offsets).
MAX_LAYOUTS = 4

_token_counter = itertools.count(1)
_tracked_classes: dict[type, type] = {}


def _versioned_subclass(base: type) -> type:
    """A subclass of `base` whose row mutators record what they changed.

    `_row_version[row]` bumps on every mutation. `_row_rewrite[row]` is set to
    the new version by every mutation other than `append_row`, which only
    writes columns `[num_blocks_per_row[row], ...)`. So a row whose version
    moved past a snapshot but whose rewrite did not changed only in the
    columns between the snapshot's and the current `num_blocks_per_row`.
    """
    cls = _tracked_classes.get(base)
    if cls is not None:
        return cls

    class _Versioned(base):  # type: ignore[valid-type, misc]
        def _rewrote(self, rows):
            self._bt_mutations += 1
            self._row_version[rows] += 1
            self._row_rewrite[rows] = self._row_version[rows]

        def append_row(self, block_ids, row_idx):
            super().append_row(block_ids, row_idx)
            self._bt_mutations += 1
            self._row_version[row_idx] += 1

        def add_row(self, block_ids, row_idx):
            super().add_row(block_ids, row_idx)
            self._rewrote(row_idx)

        def clear_row(self, row_idx):
            super().clear_row(row_idx)
            self._rewrote(row_idx)

        def move_row(self, src, tgt):
            super().move_row(src, tgt)
            self._rewrote([src, tgt])

        def swap_row(self, src, tgt):
            super().swap_row(src, tgt)
            self._rewrote([src, tgt])

        def clear(self):
            super().clear()
            self._rewrote(slice(None))

    _Versioned.__name__ = f"Versioned{base.__name__}"
    _Versioned.__qualname__ = _Versioned.__name__
    _tracked_classes[base] = _Versioned
    return _Versioned


def ensure_tracked(block_table: Any) -> int:
    """Install row-version tracking on `block_table`; return its identity token.

    The token, not `id()`, keys staging layouts, so a re-created block table
    (KV cache re-initialization) can never alias a stale device copy.
    """
    token = getattr(block_table, "_bt_delta_token", None)
    if token is None:
        num_rows = block_table.get_cpu_tensor().shape[0]
        block_table._row_version = np.zeros(num_rows, dtype=np.int64)
        block_table._row_rewrite = np.zeros(num_rows, dtype=np.int64)
        block_table._bt_mutations = 0
        block_table.__class__ = _versioned_subclass(type(block_table))
        token = next(_token_counter)
        block_table._bt_delta_token = token
    return token


def _apply_block_table_delta(
    staged: torch.Tensor, update: torch.Tensor
) -> torch.Tensor:
    """`staged` with `staged[update[0]] = update[1]`, as a new tensor.

    Padding slots repeat a real `(index, value)` pair, so duplicate indices
    always carry identical values and the scatter result is order-independent.
    """
    out = staged.clone()
    out[update[0].to(torch.int64)] = update[1]
    return out


_apply_block_table_delta_compiled = torch.compile(
    _apply_block_table_delta, backend="tpu", fullgraph=True, dynamic=False
)


def apply_block_table_delta(staged: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    if staged.device.type == "tpu":
        return _apply_block_table_delta_compiled(staged, update)
    return _apply_block_table_delta(staged, update)


@dataclass
class _EntryState:
    """What the device copy holds for one group's region."""

    offset: int
    num_blocks: int  # row width (target_num_blocks)
    active_cols: int  # columns copied from the host table; zeros beyond
    num_reqs: int  # live rows; rows past this are zeros
    versions: np.ndarray  # source-row versions at upload, length num_reqs
    nblocks: np.ndarray  # source-row num_blocks_per_row at upload
    mutations: int  # source table's total mutation count at upload


@dataclass
class _Layout:
    dev: torch.Tensor
    mirror: np.ndarray  # int32, equal to `dev` element for element
    entries: list[_EntryState]


class BlockTableDeltaStager:
    """Per-runner cache of device-resident staged block tables."""

    def __init__(self) -> None:
        self._layouts: OrderedDict[tuple, _Layout] = OrderedDict()
        self._zero_update: dict[torch.device, torch.Tensor] = {}
        self.num_full = 0
        self.num_delta = 0
        self.num_reuse = 0

    @staticmethod
    def layout_key(
        entries: list[tuple[int, Any, int]], num_reqs_padded: int, start_index: int
    ) -> tuple:
        return (
            tuple((gid, ensure_tracked(bt), tnb) for gid, bt, tnb in entries),
            num_reqs_padded,
            start_index,
        )

    def try_stage(
        self,
        key: tuple,
        ctx: Any,
        entries: list[tuple[int, Any, int]],
        builders: list[Any],
        num_reqs_padded: int,
    ) -> torch.Tensor | None:
        """The up-to-date device copy for `key`, or None for a full upload."""
        layout = self._layouts.get(key)
        if layout is None:
            return None
        start, num_reqs = ctx.start_index, ctx.num_reqs
        end = start + num_reqs
        idx: list[int] = []
        vals: list[int] = []
        for builder, (_, bt, tnb), st in zip(
            builders, entries, layout.entries, strict=True
        ):
            active_cols = builder._active_block_table_columns(ctx, tnb)
            mutations = bt._bt_mutations
            if (
                mutations == st.mutations
                and active_cols == st.active_cols
                and num_reqs == st.num_reqs
            ):
                continue
            versions = bt._row_version[start:end]
            nblocks = bt.num_blocks_per_row[start:end]
            region = layout.mirror[st.offset : st.offset + num_reqs_padded * tnb]
            mirror = region.reshape(num_reqs_padded, tnb)
            source = bt.get_cpu_tensor().numpy()[start:end]
            span = max(num_reqs, st.num_reqs)
            if active_cols != st.active_cols:
                full_rows = np.arange(span)
                app_rows: list[int] = []
            else:
                common = min(num_reqs, st.num_reqs)
                changed = np.flatnonzero(versions[:common] != st.versions[:common])
                rewritten = bt._row_rewrite[start + changed] > st.versions[changed]
                full_rows = np.concatenate(
                    [changed[rewritten], np.arange(common, span)]
                )
                app_rows = changed[~rewritten].tolist()
            if full_rows.size:
                # Rows rewritten in place, or entering / leaving the live range:
                # compare their whole active width against the mirror.
                width = max(active_cols, st.active_cols)
                expected = np.zeros((full_rows.size, width), dtype=np.int32)
                live = full_rows < num_reqs
                expected[live, :active_cols] = source[full_rows[live], :active_cols]
                r, c = np.nonzero(expected != mirror[full_rows, :width])
                rows = full_rows[r]
                new = expected[r, c]
                mirror[rows, c] = new
                idx.extend((st.offset + rows * tnb + c).tolist())
                vals.extend(new.tolist())
            for row in app_rows:
                # Append-only since the snapshot: columns below the snapshot's
                # block count, and at or past the current one, are unchanged.
                lo = min(int(st.nblocks[row]), active_cols)
                hi = min(int(nblocks[row]), active_cols)
                if hi <= lo:
                    continue
                base = st.offset + row * tnb
                new_row = source[row, lo:hi]
                mirror[row, lo:hi] = new_row
                idx.extend(range(base + lo, base + hi))
                vals.extend(new_row.tolist())
            if len(idx) > DELTA_CAPACITY:
                # The mirror is partially updated; rebuild from scratch.
                del self._layouts[key]
                return None
            st.active_cols = active_cols
            st.num_reqs = num_reqs
            st.versions = versions.copy()
            st.nblocks = nblocks.copy()
            st.mutations = mutations
        self._layouts.move_to_end(key)
        count = len(idx)
        if count:
            update = np.empty((2, DELTA_CAPACITY), dtype=np.int32)
            update[0, :count] = idx
            update[1, :count] = vals
            update[0, count:] = idx[0]
            update[1, count:] = vals[0]
            update_dev = torch.from_numpy(update).to(layout.dev.device)
            layout.dev = apply_block_table_delta(layout.dev, update_dev)
            self.num_delta += 1
        else:
            self.num_reuse += 1
        return layout.dev

    def record_full(
        self,
        key: tuple,
        ctx: Any,
        entries: list[tuple[int, Any, int]],
        builders: list[Any],
        num_reqs_padded: int,
        flat: torch.Tensor,
        staged_dev: torch.Tensor,
    ) -> None:
        """Remember a full upload so the next walk can send a delta."""
        start, num_reqs = ctx.start_index, ctx.num_reqs
        states: list[_EntryState] = []
        offset = 0
        for builder, (_, bt, tnb) in zip(builders, entries, strict=True):
            states.append(
                _EntryState(
                    offset,
                    tnb,
                    builder._active_block_table_columns(ctx, tnb),
                    num_reqs,
                    bt._row_version[start : start + num_reqs].copy(),
                    bt.num_blocks_per_row[start : start + num_reqs].copy(),
                    bt._bt_mutations,
                )
            )
            offset += num_reqs_padded * tnb
        if staged_dev.device.type == "cpu":
            # `.to()` on a CPU device returns the shared host scratch itself,
            # which the next full upload overwrites.
            staged_dev = staged_dev.clone()
        self._layouts[key] = _Layout(staged_dev, flat.numpy().copy(), states)
        self._layouts.move_to_end(key)
        while len(self._layouts) > MAX_LAYOUTS:
            self._layouts.popitem(last=False)
        self.num_full += 1

    def warm(self, staged_dev: torch.Tensor) -> None:
        """Compile the scatter for this layout during dummy runs (no-op update)."""
        device = staged_dev.device
        zero = self._zero_update.get(device)
        if zero is None:
            zero = torch.zeros((2, DELTA_CAPACITY), dtype=torch.int32).to(device)
            self._zero_update[device] = zero
        apply_block_table_delta(staged_dev, zero)

    def verify(self, key: tuple, expected_flat: torch.Tensor) -> None:
        """Raise unless the mirror and the device copy equal `expected_flat`."""
        layout = self._layouts[key]
        expected = expected_flat.numpy()
        if not np.array_equal(layout.mirror, expected):
            bad = np.flatnonzero(layout.mirror != expected)
            raise RuntimeError(
                f"incremental block table mirror diverged at {bad.size} entries "
                f"(first {bad[:8].tolist()})"
            )
        device_copy = layout.dev.cpu().numpy()
        if not np.array_equal(device_copy, expected):
            bad = np.flatnonzero(device_copy != expected)
            raise RuntimeError(
                f"incremental block table device copy diverged at {bad.size} "
                f"entries (first {bad[:8].tolist()})"
            )


def get_stager(runner: Any) -> BlockTableDeltaStager:
    stager = getattr(runner, "_block_table_delta_stager", None)
    if not isinstance(stager, BlockTableDeltaStager):
        stager = BlockTableDeltaStager()
        runner._block_table_delta_stager = stager
    return stager
