# SPDX-License-Identifier: Apache-2.0
"""Routed Stage-3 spans for a staged TP>1 Qwen decode (QWEN4_EXP_PD_ROUTED_DST).

Today a TP1 pipeline producer sends every page to every decode rank (pd_staging,
"the producer broadcasts each span to every decode unit"): 8 copies per decode
host, of which a rank reads about a third. Routed, each decode rank receives
only what its staged repack reads, into its own dense compact staging space:

- main KV: the 512-token runs of its main-KV DCP slice (both KV heads stay in
  each 1,024-byte row: they are byte-interleaved within the row's 32-bit
  words, so a head is not a byte span). Slice ``d`` goes to every rank with
  ``rank % dcp == d``, at compact row ``entry * block + t % block``;
- compressed keys: every page's live rows, to every rank (compact, dense);
- GDN state: its own value heads' SSM rows (contiguous 64 KiB per head) plus
  the whole conv state;
- raw ring and PLE state: the first ``min(producer block, decode block)`` rows,
  which is all the repack copies.

This module is the contract both legs derive their addresses from: the
producer's byte spans (destination offsets in the rank-local request-global
space, ``dst_space_version=1``) and the decode's staging page counts, prefix
clips and repack source rows. Raiden's planner requires each destination
rank's coverage to be dense, with only its final destination block partial,
and one destination page list per tag for all ranks: every slice is padded to
the longest slice with copies of producer rows the repack never reads.
"""

import dataclasses
import math

import numpy as np

ROW_BYTES = 1024  # one [4 sublanes, 256 lanes] FP8 row of the pooled carriers


@dataclasses.dataclass(frozen=True)
class RouteGeometry:
    """What both legs must agree on (it is part of the layout fingerprint)."""

    tp: int  # decode TP workers (destination units, tp_rank order)
    dcp: int  # decode main-KV DCP degree
    dst_block: int  # decode manager block: the DCP run length in tokens
    src_block: int  # producer page tokens
    ratio: int  # indexer compress ratio (a compressed pair covers 2 * ratio tokens)
    ssm_rows_src: int  # producer GDN page: SSM rows (all value heads)
    conv_rows_src: int  # producer GDN page: conv rows (all channels)
    ssm_rows_dst: int  # one decode rank's SSM rows (its value heads)
    stage_rows: int = 256  # rows per compact staging page

    def __post_init__(self):
        if self.tp <= 0 or self.dcp <= 0 or self.tp % self.dcp:
            raise ValueError(f"routed decode needs dcp | tp, got tp={self.tp} dcp={self.dcp}")
        if self.ssm_rows_dst * self.tp != self.ssm_rows_src:
            raise ValueError(
                "routed GDN state needs each rank's value heads to fill whole rows: "
                f"{self.tp} x {self.ssm_rows_dst} != {self.ssm_rows_src}"
            )
        if self.src_block % (2 * self.ratio):
            raise ValueError("a producer page must hold whole compressed pairs")
        if self.prefix_rows % self.stage_rows or self.gdn_rows % self.stage_rows:
            raise ValueError("ring/PLE and GDN extents must be whole staging pages")

    @property
    def entry(self) -> int:
        """Tokens of one decode main-KV table entry (D x B)."""
        return self.dcp * self.dst_block

    @property
    def pairs_per_src_page(self) -> int:
        return self.src_block // (2 * self.ratio)

    @property
    def prefix_rows(self) -> int:
        """Rows of a ring / PLE page the repack copies."""
        return min(self.src_block, self.dst_block)

    @property
    def gdn_rows(self) -> int:
        return self.ssm_rows_dst + self.conv_rows_src

    def ranks_of_slice(self, d: int) -> list[int]:
        return [r for r in range(self.tp) if r % self.dcp == d]

    def key(self) -> list[int]:
        return [self.tp, self.dcp, self.dst_block, self.src_block, self.ratio,
                self.ssm_rows_src, self.conv_rows_src, self.ssm_rows_dst, self.stage_rows]


def parse_route_env(value: str | None) -> tuple[int, int, int] | None:
    """``"<decode tp>,<decode dcp>,<decode block>"`` or empty (routing off).

    ``":"`` separates too: the recipes' comma-separated ``*_EXTRA_SERVER_ENV``
    lists cannot carry a comma inside a value.
    """
    value = (value or "").strip()
    if not value:
        return None
    parts = [int(x) for x in value.replace(":", ",").split(",")]
    if len(parts) != 3 or min(parts) <= 0:
        raise ValueError(f"QWEN4_EXP_PD_ROUTED_DST must be 'tp,dcp,block', got {value!r}")
    return parts[0], parts[1], parts[2]


# --- extents --------------------------------------------------------------------


def slice_rows(num_tokens: int, g: RouteGeometry, d: int) -> int:
    """Tokens of main-KV DCP slice ``d`` in ``[0, num_tokens)``."""
    runs = math.ceil(num_tokens / g.dst_block)
    return sum(min(g.dst_block, num_tokens - k * g.dst_block) for k in range(d, runs, g.dcp))


def main_rows(num_tokens: int, g: RouteGeometry) -> int:
    """Every rank's main-KV compact extent: the longest slice."""
    return max(slice_rows(num_tokens, g, d) for d in range(g.dcp))


def compressed_rows(num_src_pages: int, g: RouteGeometry) -> int:
    return num_src_pages * g.pairs_per_src_page


def skip_rows(kind: str, skip_tokens: int, g: RouteGeometry) -> int:
    """Compact rows a prefix-aware load leaves out (its decode-local prefix)."""
    if not skip_tokens:
        return 0
    if skip_tokens % g.entry or skip_tokens % (2 * g.ratio):
        raise ValueError(f"routed prefix {skip_tokens} is not on a {g.entry}-token entry")
    rows = {"main": skip_tokens // g.dcp, "compressed": skip_tokens // (2 * g.ratio)}.get(kind, 0)
    if rows % g.stage_rows:
        raise ValueError(
            f"routed prefix {skip_tokens} clips {kind} at row {rows}, not a "
            f"{g.stage_rows}-row staging page"
        )
    return rows


def extent_rows(kind: str, num_tokens: int, num_src_pages: int, g: RouteGeometry) -> int:
    if kind == "main":
        return main_rows(num_tokens, g)
    if kind == "compressed":
        return compressed_rows(num_src_pages, g)
    if kind == "gdn":
        return g.gdn_rows
    if kind in ("ring", "ple"):
        return g.prefix_rows
    raise ValueError(f"no routed extent for cache kind {kind}")


def staging_pages(kind: str, num_tokens: int, num_src_pages: int, skip_tokens: int,
                  g: RouteGeometry) -> int:
    """Compact staging pages one owner of ``kind`` takes for a load (equal on all ranks)."""
    rows = extent_rows(kind, num_tokens, num_src_pages, g) - skip_rows(kind, skip_tokens, g)
    if rows <= 0:
        raise ValueError("A prefix-aware load must leave every group a suffix")
    return math.ceil(rows / g.stage_rows)


# --- producer spans --------------------------------------------------------------
# (src page ordinal, src byte offset, dst byte offset, bytes, dst unit or None)
Span = tuple[int, int, int, int, int | None]


def _main_spans(num_tokens: int, g: RouteGeometry) -> list[Span]:
    rb, P, B = ROW_BYTES, g.src_block, g.dst_block
    runs = math.ceil(num_tokens / B)
    common = main_rows(num_tokens, g)
    spans: list[Span] = []
    for d in range(g.dcp):
        units = g.ranks_of_slice(d)
        pos = 0
        pieces = []
        for k in range(d, runs, g.dcp):
            t, end = k * B, min(num_tokens, (k + 1) * B)
            while t < end:
                page = t // P
                stop = min(end, (page + 1) * P)
                pieces.append((page, (t % P) * rb, pos * rb, (stop - t) * rb))
                pos += stop - t
                t = stop
        # Pad to the common extent with producer rows nothing reads: one
        # destination page list per tag serves every rank, and Raiden only
        # accepts a partial final destination block.
        pad = common - pos
        if pad:
            if pad > P:
                raise ValueError("routed main-KV padding exceeds one producer page")
            pieces.append((0, 0, pos * rb, pad * rb))
        spans.extend((p, so, do, n, u) for p, so, do, n in pieces for u in units)
    return spans


def producer_spans(kind: str, num_tokens: int, num_src_pages: int, g: RouteGeometry) -> list[Span]:
    """Every span one producer owner declares for a load (all destination ranks)."""
    rb = ROW_BYTES
    if kind == "main":
        return _main_spans(num_tokens, g)
    if kind == "compressed":
        n = g.pairs_per_src_page * rb
        return [(i, 0, i * n, n, None) for i in range(num_src_pages)]
    if kind == "gdn":
        spans: list[Span] = [
            (0, r * g.ssm_rows_dst * rb, 0, g.ssm_rows_dst * rb, r) for r in range(g.tp)
        ]
        spans.append((0, g.ssm_rows_src * rb, g.ssm_rows_dst * rb, g.conv_rows_src * rb, None))
        return spans
    if kind in ("ring", "ple"):
        return [(0, 0, 0, g.prefix_rows * rb, None)]
    raise ValueError(f"no routed spans for cache kind {kind}")


# --- decode repack source rows ----------------------------------------------------


def _flat(pages, rows: np.ndarray, g: RouteGeometry) -> np.ndarray:
    pg = np.asarray(pages, dtype=np.int64)
    return pg[rows // g.stage_rows] * g.stage_rows + rows % g.stage_rows


def main_src_rows(num_tokens: int, skip_tokens: int, rank: int, pages, g: RouteGeometry):
    """(staging row, compact row) of every token ``rank`` keeps from ``skip_tokens`` on,
    in the order ``pd_staging.main_rows_np`` lists them."""
    t = np.arange(skip_tokens, num_tokens, dtype=np.int64)
    t = t[(t // g.dst_block) % g.dcp == rank % g.dcp]
    rows = (t // g.entry) * g.dst_block + t % g.dst_block - skip_rows("main", skip_tokens, g)
    return _flat(pages, rows, g)


def compressed_src_rows(num_tokens: int, skip_tokens: int, pages, g: RouteGeometry):
    span = 2 * g.ratio
    pairs = math.ceil(math.ceil(num_tokens / g.ratio) / 2)
    rows = np.arange(skip_tokens // span, pairs, dtype=np.int64) - skip_rows(
        "compressed", skip_tokens, g
    )
    return _flat(pages, rows, g)


def prefix_src_rows(pages, g: RouteGeometry):
    return _flat(pages, np.arange(g.prefix_rows, dtype=np.int64), g)


def unit_bytes(kind: str, num_tokens: int, num_src_pages: int, g: RouteGeometry) -> int:
    """Bytes one decode rank receives for one owner of ``kind`` (logging, tests)."""
    return extent_rows(kind, num_tokens, num_src_pages, g) * ROW_BYTES
