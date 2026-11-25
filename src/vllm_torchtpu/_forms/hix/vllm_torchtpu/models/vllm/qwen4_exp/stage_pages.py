# SPDX-License-Identifier: Apache-2.0
"""Per-stage physical pages behind a pipeline engine's shared block IDs.

A Qwen4Exp pipeline engine plans the whole model's cache groups on every
stage, so all stages share one block-ID space and, without this map, each
stage's pool holds one page per block ID. A stage writes only the pages of
groups that have a layer on it: of the 24 token groups a cached 3200-token
block takes (12 main KV, 12 compressed keys) a stage writes 2 or 4, and of a
GDN checkpoint's 37 state groups 4 to 6. The rest of its pool is never
touched.

``StagePageMap`` lets the scheduler hand out more block IDs than a stage has
pages. A stage maps a block ID to one of its pages while one of its own
groups holds that ID, and returns the page when the ID shows up in another
group. Every allocation reaches every stage in the scheduler output (new
requests' tables, appended IDs, replaced rows), so each stage tracks the
owner of every ID it has mapped exactly, including IDs that sit freed but
cached in the scheduler's free queue: such an ID keeps its page, and its
prefix-cache hit finds the bytes it wrote, until the scheduler gives the ID
to a group of another stage.

Block ID 0 is the scheduler's null block and maps to page 0. Pages of groups
without a layer on this stage are never read or written here, so their
table entries map to page 0 as well, which keeps every device-side index in
range of the pool. Freed pages are reused last-in first-out.
"""

import dataclasses
from collections.abc import Callable, Iterable, Sequence

import numpy as np


class StagePagesExhausted(RuntimeError):
    pass


class StagePageMap:
    def __init__(
        self,
        physical_pages: int,
        logical_pages: int,
        own_groups: Iterable[int],
        num_groups: int,
    ) -> None:
        if physical_pages < 2:
            raise ValueError("A stage pool needs the null page and one more")
        if logical_pages < physical_pages:
            raise ValueError("The scheduler cannot have fewer block IDs than pages")
        self.physical_pages = int(physical_pages)
        self.logical_pages = int(logical_pages)
        self.num_groups = int(num_groups)
        self.own = np.zeros(self.num_groups, dtype=bool)
        for group in own_groups:
            if not 0 <= group < self.num_groups:
                raise ValueError(f"Own group {group} is outside the plan")
            self.own[group] = True
        # Logical ID -> page, -1 when unmapped; page -> logical ID, -1 free.
        self._page = np.full(self.logical_pages, -1, dtype=np.int64)
        self._page[0] = 0
        self._owner = np.full(self.logical_pages, -1, dtype=np.int64)
        self._free = list(range(self.physical_pages - 1, 0, -1))
        self.peak = 0

    @property
    def mapped(self) -> int:
        return self.physical_pages - 1 - len(self._free)

    def _check(self, block_id: int) -> None:
        if not 0 <= block_id < self.logical_pages:
            raise ValueError(
                f"Block ID {block_id} is outside the scheduler's "
                f"{self.logical_pages} IDs"
            )

    def _release(self, block_id: int) -> None:
        page = int(self._page[block_id])
        if page > 0:
            self._free.append(page)
        self._page[block_id] = -1
        self._owner[block_id] = -1

    def assign(self, group: int, block_ids: Sequence[int]) -> list[int]:
        """Record that ``group`` now holds ``block_ids``; return its pages."""
        out = []
        own = bool(self.own[group])
        for block_id in block_ids:
            block_id = int(block_id)
            self._check(block_id)
            if block_id == 0:
                out.append(0)
                continue
            if not own:
                if self._page[block_id] > 0:
                    self._release(block_id)
                out.append(0)
                continue
            page = int(self._page[block_id])
            if page <= 0:
                if not self._free:
                    raise StagePagesExhausted(
                        f"Stage pool of {self.physical_pages} pages is full "
                        f"({self.mapped} mapped) mapping block {block_id} of "
                        f"group {group}; lower "
                        "QWEN4_EXP_PP_LOGICAL_BLOCKS_FACTOR"
                    )
                page = self._free.pop()
                self._page[block_id] = page
                self.peak = max(self.peak, self.mapped)
            self._owner[block_id] = group
            out.append(page)
        return out

    def lookup(self, group: int, block_ids: Sequence[int]) -> tuple[int, ...]:
        """Pages of IDs ``group`` already holds (no ownership change)."""
        if not self.own[group]:
            return tuple(0 for _ in block_ids)
        out = []
        for block_id in block_ids:
            block_id = int(block_id)
            self._check(block_id)
            if block_id == 0:
                out.append(0)
                continue
            page = int(self._page[block_id])
            if page <= 0 or self._owner[block_id] != group:
                raise ValueError(
                    f"Block {block_id} of group {group} has no page on this stage"
                )
            out.append(page)
        return tuple(out)

    def page_of(self, block_id: int) -> int | None:
        """The page of a mapped ID whatever its group, else None."""
        block_id = int(block_id)
        self._check(block_id)
        page = int(self._page[block_id])
        return page if page > 0 or block_id == 0 else None

    def _assign_table(
        self, table: tuple[list[int], ...] | None
    ) -> tuple[list[int], ...] | None:
        if table is None:
            return None
        if len(table) != self.num_groups:
            raise ValueError(
                f"A block table has {len(table)} groups, the plan {self.num_groups}"
            )
        return tuple(self.assign(g, ids) for g, ids in enumerate(table))

    def translate(self, scheduler_output):
        """A copy of ``scheduler_output`` addressing this stage's pages.

        Allocations are recorded first (new requests, appended or replaced
        rows), then the zero list and the copy list resolve against the
        updated map.
        """
        new_reqs = [
            dataclasses.replace(req, block_ids=self._assign_table(req.block_ids))
            for req in scheduler_output.scheduled_new_reqs
        ]
        cached = scheduler_output.scheduled_cached_reqs
        cached = dataclasses.replace(
            cached,
            new_block_ids=[self._assign_table(t) for t in cached.new_block_ids],
        )
        changes = {
            "scheduled_new_reqs": new_reqs,
            "scheduled_cached_reqs": cached,
        }
        updates = getattr(scheduler_output, "block_table_updates", None)
        if updates:
            changes["block_table_updates"] = {
                req_id: self._assign_table(table) for req_id, table in updates.items()
            }
        zeros = getattr(scheduler_output, "new_block_ids_to_zero", None)
        if zeros:
            pages = (self.page_of(b) for b in zeros)
            changes["new_block_ids_to_zero"] = [p for p in pages if p]
        copies = getattr(scheduler_output, "kv_cache_block_copies", None)
        if copies:
            changes["kv_cache_block_copies"] = self._translate_copies(copies)
        return dataclasses.replace(scheduler_output, **changes)

    def _translate_copies(self, copies):
        out = []
        for copy in copies:
            dst = self.page_of(copy.dst_block_id)
            if dst is None or dst == 0:
                continue  # another stage's page
            src = self.page_of(copy.src_block_id)
            if src is None:
                raise ValueError(
                    f"Copy into page {dst} reads block {copy.src_block_id}, "
                    "which has no page on this stage"
                )
            out.append(type(copy)(src, dst))
        return out


def table_translator(
    page_map: StagePageMap | None,
) -> Callable[[int, Sequence[int]], tuple[int, ...]] | None:
    return None if page_map is None else page_map.lookup
