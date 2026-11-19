# SPDX-License-Identifier: Apache-2.0
"""Compressed-key pairs once per decode host (QWEN4_EXP_PD_SHARED_COMPRESSED).

A port, generalised, of the Qwen3.5 head-unique Stage-3 transfer (commit
3eeddf339: ``raiden/head_unique.py`` and ``raiden/gather_kernels.py``). There
a *group* is the TP ranks holding one KV head; page ``p`` of a load lands on
replica ``p % G`` only, and after landing the replicas all-gather the pages
they did not receive. Here a group is the set of decode ranks that need
identical bytes for one span class:

- compressed keys: every rank of the TP group (``G = tp_size``), this module;
- one main-KV page: the ranks of its DCP slice (S3, later);
- GDN state: no group (each rank keeps different heads).

The owner rule is unchanged: page ``b`` goes to rank ``b % G``.

**Geometry.** A compressed-key tag's request-global pair space holds pair
``q`` at row ``q`` (S12's compact rows: producer page ``i``'s 400 live rows
at ``400 i``), cut into pages of ``PAGE_ROWS`` = 256 one-KiB rows (S12's
compact staging page). Global page ``b`` lands on rank ``b % G`` at that
rank's compact page ``b // G``. Raiden needs every rank's coverage of a tag
to be dense and the same length (one destination page list per tag for all
units; only the final page partial), so a rank with fewer pages re-sends its
own last page as padding (``seq_on_lane``'s rule). A load with fewer than
``G`` pages stays broadcast (head-unique's ``load_group_size``); both legs
derive that from the token count alone.

**Prefix-aware loads.** Raiden clips one ``dst_skip_bytes`` per tag, the same
on every unit, so a decode-local prefix may clip only whole pages that hold
the same pairs' share on every rank: multiples of ``G * PAGE_ROWS`` pairs
(``clip_tokens``: 16,384 tokens at ratio 4, ``G`` 8). Compressed tags clip
``skip // clip_tokens`` pages per rank; the repack ignores the received pairs
below ``skip`` (they sit in shared prefix pages and are never written).

**Landing** (decode, inside the L1-named step, before the repack). Per pass of
at most ``GATHER_PAGE_BUCKETS[-1]`` pages per rank: ``pack`` copies this
rank's compact pages from the staging carrier into its send buffer, the send
buffer is all-gathered over the TP group, and ``scatter`` writes every pair
of the gathered buffer (this rank's own included) to its pool row, addressed
exactly as ``pd_staging.compressed_rows_np``. Every index list here is the
same on every rank: staging pages come from the TP-MIN free map and decode
tables from one scheduler. So every rank compiles and runs the same
programs, the rule a compiled TP collective needs (``tpu_runner``'s
``_tpu_pin_drafts_to_rank0``).

This module is pure numpy; both legs derive their addresses from it.
"""

import dataclasses
import math

import numpy as np

PAGE_ROWS = 256
# Pages per rank per gather pass. A pass's scatter writes G * Q * 256 rows:
# 32,768 at Q = 16, G = 8, inside the DMA writer's ~40k-row SMEM budget
# (pd_staging.REPACK_CHUNK_ROWS). Send buffer 4 MiB, gathered 32 MiB.
GATHER_PAGE_BUCKETS = (4, 8, 16)
# Destination rows of one contiguous pool run (pd_staging.REPACK_TILE).
TILE = 64


def check_admissible(*, tp_size: int, sync_repack: bool, sync_repack_ab_s: int) -> None:
    """Refuse a decode on which the landing gather could hang its TP group.

    The gather is a TP collective, so every rank must run it for the same
    loads in the same step: only QWEN4_EXP_PD_SYNC_REPACK (L1) names loads
    once they landed on every rank, to every rank, in one step. Its wall-clock
    A/B repacks on landing (rank by rank) in off periods, so it is refused.
    """
    if tp_size < 2:
        raise ValueError("QWEN4_EXP_PD_SHARED_COMPRESSED needs a TP>1 staged decode")
    if not sync_repack or sync_repack_ab_s > 0:
        raise ValueError(
            "QWEN4_EXP_PD_SHARED_COMPRESSED needs QWEN4_EXP_PD_SYNC_REPACK=1 and "
            "QWEN4_EXP_PD_SYNC_REPACK_AB_S=0: its landing all-gather is a TP "
            "collective every rank must reach in the same step"
        )


@dataclasses.dataclass(frozen=True)
class SharedGeometry:
    group: int  # G: decode ranks that need the same compressed keys (TP size)
    ratio: int  # indexer_compress_ratio
    src_block: int  # producer tokens per page (3200 on a TP1 stage)
    page_rows: int = PAGE_ROWS

    def __post_init__(self):
        if self.group < 2:
            raise ValueError("a shared transfer needs a group of at least two ranks")
        if self.src_block % self.pair_tokens:
            raise ValueError("a producer page must hold whole compressed pairs")
        if self.src_rows < self.page_rows:
            # The filler of a last page's tail reads one producer page.
            raise ValueError("a producer page must hold at least one staging page")

    @property
    def pair_tokens(self) -> int:
        return 2 * self.ratio

    @property
    def src_rows(self) -> int:
        """Live pair rows of one producer page (400)."""
        return self.src_block // self.pair_tokens

    @property
    def tile(self) -> int:
        """Scatter segments are padded to this many rows (a DMA tile)."""
        return min(TILE, self.page_rows)

    @property
    def clip_tokens(self) -> int:
        """Prefix clip quantum of a compressed tag (16,384 tokens)."""
        return self.pair_tokens * self.page_rows * self.group

    def pairs(self, num_tokens: int) -> int:
        return math.ceil(math.ceil(num_tokens / self.ratio) / 2)

    def pages(self, num_tokens: int) -> int:
        """Global pages of one tag's pair space."""
        return -(-self.pairs(num_tokens) // self.page_rows)

    def shared(self, num_tokens: int) -> bool:
        """Whether a load goes once per host (else broadcast, as S12)."""
        return self.pages(num_tokens) >= self.group

    def unit_pages(self, num_tokens: int) -> int:
        """Compact pages every rank covers per tag (before a prefix clip)."""
        return -(-self.pages(num_tokens) // self.group)

    def clip_pages(self, skip_tokens: int) -> int:
        """Compact pages a prefix clips from every rank's coverage of a tag."""
        return skip_tokens // self.clip_tokens

    def staged_pages(self, num_tokens: int, skip_tokens: int = 0) -> int:
        """Staging pages per compressed tag per rank."""
        count = self.unit_pages(num_tokens) - self.clip_pages(skip_tokens)
        if count <= 0:
            raise ValueError("A prefix-aware load must leave every rank a page")
        return count

    def owner(self, page: int) -> int:
        """The head-unique owner rule: page ``b`` lands on rank ``b % G``."""
        return page % self.group

    def is_real(self, num_tokens: int, unit: int, k: int) -> bool:
        """Whether rank ``unit``'s compact page ``k`` holds its own global page."""
        return self.group * k + unit < self.pages(num_tokens)

    def unit_page_source(self, num_tokens: int, unit: int, k: int) -> int:
        """Global page carried by rank ``unit``'s compact page ``k``; past
        its own pages, its last own page again (padding)."""
        own = -(-(self.pages(num_tokens) - unit) // self.group)
        if own <= 0:
            raise ValueError("a shared load gives every rank at least one page")
        return self.group * min(k, own - 1) + unit


def enabled() -> bool:
    from vllm_torchtpu import envs

    return bool(envs.QWEN4_EXP_PD_SHARED_COMPRESSED)


def geometry_for(route) -> SharedGeometry:
    """The shared geometry of a routed decode (``pd_route.RouteGeometry``):
    the group is its TP, pages are its compact staging pages."""
    return SharedGeometry(
        group=route.tp, ratio=route.ratio, src_block=route.src_block, page_rows=route.stage_rows
    )


def producer_runs(
    num_tokens: int, geo: SharedGeometry, num_src_pages: int | None = None
) -> np.ndarray:
    """One compressed tag's producer runs, ``[n, 5]`` int64 rows of
    ``(unit, dst_row, src_page_ordinal, src_row, rows)``.

    Rank ``unit``'s compact rows ``[0, unit_pages * page_rows)`` are covered
    densely, page ``k`` from global page ``unit_page_source(unit, k)``. A run
    never crosses a producer page (``src_rows`` rows each). Rows past the
    producer's last page (only the tail of the last global page) are filled
    from the start of that page: bytes nothing reads, sent so the coverage is
    exact. A span is ``PoolByteSpan(src_block_ordinal=src_page_ordinal,
    src_offset_bytes=src_row * row_bytes, dst_block_index=0,
    dst_offset_bytes=dst_row * row_bytes, size_bytes=rows * row_bytes,
    dst_unit_ordinal=unit)`` in a ``dst_space_version=1`` tag.
    """
    if not geo.shared(num_tokens):
        raise ValueError("a load with fewer pages than ranks stays broadcast")
    n_src = -(-num_tokens // geo.src_block)
    if num_src_pages is not None and num_src_pages != n_src:
        raise ValueError(
            f"a {num_tokens}-token load has {n_src} producer pages, not {num_src_pages}"
        )
    avail = n_src * geo.src_rows
    pr, sr = geo.page_rows, geo.src_rows
    out = []
    for unit in range(geo.group):
        for k in range(geo.unit_pages(num_tokens)):
            b = geo.unit_page_source(num_tokens, unit, k)
            lo, hi, dst = b * pr, (b + 1) * pr, k * pr
            q = lo
            while q < hi:
                if q >= avail:
                    out.append((unit, dst + q - lo, n_src - 1, 0, hi - q))
                    break
                page, row = divmod(q, sr)
                rows = min(hi, (page + 1) * sr, avail) - q
                out.append((unit, dst + q - lo, page, row, rows))
                q += rows
    return np.asarray(out, dtype=np.int64).reshape(-1, 5)


def shared_spans(num_tokens: int, num_src_pages: int, geo: SharedGeometry, row_bytes: int):
    """``producer_runs`` as ``pd_route`` spans: (src page ordinal, src byte
    offset, dst byte offset, bytes, dst unit). A run never leaves its
    producer page, so ``pd_route.subpage_spans`` keeps each in its slot."""
    return [
        (int(page), int(row) * row_bytes, int(dst) * row_bytes, int(rows) * row_bytes, int(unit))
        for unit, dst, page, row, rows in producer_runs(num_tokens, geo, num_src_pages)
    ]


def agree_named_loads(present, cpu_group=None) -> list[bool]:
    """Which named loads every rank of the TP group holds (MIN over ranks).

    Run before the first landing collective of a step: a load some rank
    lacks (its staged pages were already released) must be dropped on every
    rank, or the ranks that hold it would wait in an all-gather the other
    never enters. One CPU all-reduce of a few int32 (gloo ``cpu_group``)."""
    import torch

    have = torch.tensor([1 if p else 0 for p in present], dtype=torch.int32)
    if cpu_group is not None and len(have):
        import torch.distributed as dist

        dist.all_reduce(have, op=dist.ReduceOp.MIN, group=cpu_group)
    return [bool(x) for x in have.tolist()]


def gather_bucket(pages: int) -> int:
    """The smallest bucket holding one pass of ``pages`` pages per rank."""
    for bucket in GATHER_PAGE_BUCKETS:
        if pages <= bucket:
            return bucket
    raise ValueError(f"{pages} pages exceed the largest gather pass")


@dataclasses.dataclass(frozen=True)
class SharedTag:
    """One compressed-key owner's landing: its staging pages (compact pages
    ``clip, clip + 1, ...`` of every rank, the same ids on every rank) and its
    decode destination (``compressed_destinations``)."""

    staging_pages: tuple[int, ...]
    dst_pages: tuple[int, ...]
    dst_offset: int = 0


@dataclasses.dataclass(frozen=True)
class GatherPass:
    """One pass: ``pack`` lists the staging page of each send-buffer slot
    (``bucket`` slots); ``scatter_src`` rows index the gathered buffer
    (``G * bucket * page_rows`` rows, rank ``u``'s slot ``s`` at
    ``(u * bucket + s) * page_rows``); ``scatter_dst`` are flat pool rows.
    Both scatter lists have ``G * bucket * page_rows`` entries, tile aligned,
    padded by repeating the last pair (a rewrite of the same bytes)."""

    bucket: int
    pack: np.ndarray
    scatter_src: np.ndarray
    scatter_dst: np.ndarray


def pair_dst(pairs: np.ndarray, dst_pages, dst_block: int, ratio: int, dst_offset: int = 0):
    """Pool row of each pair (``compressed_rows_np``'s destination rule)."""
    t = pairs * (2 * ratio)
    dp = np.asarray(dst_pages, dtype=np.int64)
    return dp[t // dst_block] * dst_block + dst_offset + (t % dst_block) // (2 * ratio)


def _align(rows: np.ndarray, tile: int = TILE) -> np.ndarray:
    pad = -len(rows) % tile
    return rows if not pad else np.concatenate((rows, np.full(pad, rows[-1], rows.dtype)))


def _pad_to(rows: np.ndarray, length: int) -> np.ndarray:
    if len(rows) > length:
        raise ValueError(f"{len(rows)} row copies exceed the bucket's {length}")
    return np.concatenate((rows, np.full(length - len(rows), rows[-1], rows.dtype)))


def gather_passes(
    num_tokens: int,
    tags: list[SharedTag],
    dst_block: int,
    geo: SharedGeometry,
    skip_tokens: int = 0,
) -> list[GatherPass]:
    """The passes of one load's landing gather, identical on every rank.

    Every rank's staging list is the same: tag after tag, compact pages
    ``clip ..`` each. A pass takes the next ``<= GATHER_PAGE_BUCKETS[-1]``
    entries of it. In the gathered buffer, rank ``u``'s slot ``s`` holds
    global page ``G * k + u`` of the slot's tag (``k`` its compact page); the
    scatter writes its pairs in ``[skip / pair_tokens, pairs)`` and skips the
    padding pages.
    """
    if skip_tokens % geo.pair_tokens:
        raise ValueError("A repack must start on a compressed pair")
    clip = geo.clip_pages(skip_tokens)
    count = geo.staged_pages(num_tokens, skip_tokens)
    entries = []
    for t, tag in enumerate(tags):
        if len(tag.staging_pages) != count:
            raise ValueError(
                f"compressed tag {t} stages {len(tag.staging_pages)} pages, expected {count}"
            )
        entries.extend((t, clip + j) for j in range(count))
    first, total = skip_tokens // geo.pair_tokens, geo.pairs(num_tokens)
    pr, g = geo.page_rows, geo.group
    largest = GATHER_PAGE_BUCKETS[-1]
    passes = []
    for start in range(0, len(entries), largest):
        part = entries[start : start + largest]
        bucket = gather_bucket(len(part))
        pack = np.asarray(
            [tags[t].staging_pages[k - clip] for t, k in part], dtype=np.int64
        )
        src, dst = [], []
        for unit in range(g):
            for slot, (t, k) in enumerate(part):
                if not geo.is_real(num_tokens, unit, k):
                    continue
                b = g * k + unit
                q = np.arange(max(b * pr, first), min((b + 1) * pr, total), dtype=np.int64)
                if not len(q):
                    continue
                tag = tags[t]
                src.append(_align((unit * bucket + slot) * pr + q - b * pr, geo.tile))
                dst.append(_align(
                    pair_dst(q, tag.dst_pages, dst_block, geo.ratio, tag.dst_offset), geo.tile
                ))
        length = g * bucket * pr
        if src:
            src_rows, dst_rows = np.concatenate(src), np.concatenate(dst)
        else:  # nothing of this pass is past the prefix: rewrite the null row
            src_rows = dst_rows = np.zeros(1, dtype=np.int64)
        passes.append(
            GatherPass(
                bucket=bucket,
                pack=_pad_to(pack, bucket),
                scatter_src=_pad_to(src_rows, length),
                scatter_dst=_pad_to(dst_rows, length),
            )
        )
    return passes
