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
"""Stage B: `QSAFullBlock` with the indexer in-kernel (no selection input).

Per QSA layer, after the stage-A projection:

  iq      = index_query(mixed @ W_idx)                      4 heads x 128, every row
  scores  = per request: its compressed keys DMA'd into VMEM, `score_rows` on the MXU
  ids     = exact top-K per row (`indexer.thresholds` + `compact_row`), lane-major ->
  SMEM
  pages   = ids + the row's partial group (if `len % 4`), kv_len = 4 min(n_vis, K) + len
  % 4
  attn    = `attend_row(table=...)` (group ids -> pool micro-pages via the main table at
  DMA issue)

Rows are grouped by request, `ex.T` verify rows each (MTP verify). Prototype key layout:
`qsa_ck [requests, max_groups, 128]` BF16, contiguous per request (the served cache is
paged;
see DESIGN.md). Groups completed during this step are stage C.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..blocks.qsa_attn.block import MICRO, attend_row, fp8_dense_read
from ..exec_spec import ExecSpec
from ..ops import BF16, F32, I32, rnd
from ..spec import ExtraInput, ScratchReq, SmallField, StateDecl, WeightDecl
from . import indexer as ix
from .qsa_full import OFFSET, POOL, ROPE, SCALES, QSAFullBlock

_LOGGED: set[str] = set()


def _log_once(msg: str) -> None:
    """One INFO line per message and process, on TP rank 0 (kernel build time)."""
    if msg in _LOGGED:
        return
    _LOGGED.add(msg)
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        if get_tensor_model_parallel_rank() != 0:
            return
    except Exception:  # no TP group (CPU tests, tools): log
        pass
    from vllm_torchtpu.logger import init_logger

    init_logger(__name__).info(msg)

CK = "qsa_ck"
CKTAB = "qsa_cktab"  # [requests, max_groups // 32] kernel pages of the FP8 carrier
GPK = 32  # groups per compressed-key kernel page
NVIS = "qsa_nvis"  # [M] complete groups visible per row (len // 4)
TAIL = "qsa_tail"  # [M] tokens of the row's partial group (len % 4)
MTAB = "qsa_mtab"  # [requests, pages_per_seq] main-pool page table
IDX_W = "idx_w"
DBG_IQ = (
    "qsa_dbg_iq"  # debug outputs: the kernel's index queries and selected ids per layer
)
DBG_IDS = "qsa_dbg_ids"
# `export_ids`: the last QSA layer's pool-path ids `[M, 1, K]` (`qi_id`: ascending, slots past
# the row's count 0, g_a's flag in slot K-1 as `selection_groups` reads it) for the MTP
# draft's shared selection (VLLM_TPU_MK_WHOLE_EXPORT_SELECTION). Not per layer: one buffer.
SEL_IDS = "qsa_sel_ids"
UNION_SLOTS = 3  # attention ring slots with `union_attn`
NVIS_DIGIT = int(os.environ.get("VLLM_TPU_MK_QSA_NVIS_DIGIT", "3"))  # bits per bisection pass
# Timing-only attribution arm of the key stream (tests/megakernel/tpu_bench_wholestep_call.py
# `--score-arm`; wrong scores): "dma" keeps the copies and waits and stores zero scores,
# "compute" skips the copies and waits and scores the stale ring. Read at trace time.
SCORE_ARM = ""
IDX_N = 4 * ix.IDX_DIM + ix.IDX_DIM  # 512 query + 128 raw-key columns


class QSAFullIndexedBlock(QSAFullBlock):
    """`QSAFullBlock` + in-kernel indexer and selection."""

    def __init__(
        self,
        *,
        max_groups: int,
        pages_per_seq: int,
        debug: bool = False,
        carrier: str = "",
        ck_width: int = 256,
        chunk_pages: int = 32,
        manager_pages: int = 0,
        ck_lanes: int = 0,
        key_slots: int = 2,
        n_layers: int = 0,
        hbm_tables: bool = False,
        per_request_scores: bool = False,
        bounded_chunks: bool = False,
        shard_index: bool = False,
        union_attn: bool = False,
        union_stats: bool = False,
        union_phys: bool = False,
        loop_requests: bool = False,
        dcp: int = 1,
        compressed_dcp: bool = False,
        async_dma: bool = False,
        skip_idle: bool = False,
        xreq_prefetch: int = 0,
        select_nvis: int = 0,
        nvis_rows: tuple = (32, 64),
        xmerge: bool = False,
        thr_fast: str | None = None,
        req_async: int = 0,
        nvis_balance: bool = False,
        uattn_lite: bool = False,
        export_ids: bool = False,
        **kw: Any,
    ) -> None:
        """`carrier`: "" = prototype contiguous keys `[requests, max_groups, 128]`;
        "fp8" = the
        served FP8 carrier `[kernel_pages, 16, 4, ck_width]` (32 groups per kernel page,
        two BF16
        keys per carrier row), paged by `qsa_cktab [requests, max_groups // 32]` and
        streamed
        `chunk_pages` pages per chunk; "bf16" = its BF16 view `[kernel_pages, 16, 2,
        ck_width]`
        (CPU interpret: same layout, no in-kernel bitcast). `manager_pages` > 0:
        `qsa_cktab` holds
        manager pages of that many contiguous kernel pages (served 1664-token pages: 13)
        and each
        is one DMA (P4: per-kernel-page DMAs are issue-bound); `ck_lanes` = 128 reads
        only the
        real lanes of each carrier row. `key_slots` (> 2 needs `bounded_chunks`): slots of the
        key-stream ring, so `key_slots - 1` chunks are in flight while one is scored (2: one
        chunk of lookahead). `n_layers` > 0: per-QSA-layer metadata — the
        main-pool and
        key tables are `[n_layers * requests, …]` (row = layer ordinal * requests +
        request) and the
        head-pair offset `[n_layers]`, as the served pool packs layers into shared
        pages.
        `hbm_tables`: the key and main page tables stay in HBM and each QSA layer's rows
        are
        copied into SMEM windows `[requests, …]` at the layer's start (served contexts
        up to
        262K tokens: all layers' tables do not fit SMEM). `per_request_scores`: the
        score
        scratch holds one request's `T` rows (`[T, max_groups]` F32), selected before
        the next
        request is scored (an all-rows buffer is 16 MiB at 64 rows and 64K groups).
        `bounded_chunks`: each request streams only the key chunks its rows can see
        (`ceil(n_vis / chunk groups)`, at least 1) instead of all `max_groups`; scores
        past a
        row's `n_vis` are stale and masked by the selection. `shard_index` (needs
        `per_request_scores`): each rank scores and selects only its own contiguous run
        of
        `ceil(requests / R)` requests, then the ranks exchange the selected ids, lengths
        and
        `g_a` flags so every rank attends every row (in-kernel remote DMA, int32:
        exact).
        `union_attn` (needs `per_request_scores`): each request's verify rows attend one
        union of
        their selections, streamed once (`union_attn.attend_request`). `union_stats`
        (needs
        `union_attn`): per QSA layer and request, output `qsa_union_stats` lanes `[union
        groups,
        sum of the rows' groups, max row groups, rows with groups]` (int32), read by the
        runner
        outside the graph to measure how much a request's rows share. `union_phys`
        (needs
        `union_attn`, `hbm_tables`): each request's union ids are turned into physical
        micro-page
        ids by one vector pass before its tile loop (`union_attn.physical_ids`), so
        every copy
        issues from one SMEM load instead of the ids -> table chain (same addresses).
        `loop_requests` (needs
        `per_request_scores`): the per-request work of a QSA layer (completed group, key
        streaming and scores, selection, recent tile) runs in a `fori_loop` over requests instead
        of a Python loop, so the kernel body, and its compile time, does not grow with the
        bucket's request count. `dcp` = 2 (VLLM_TPU_QSA_MAIN_KV_DCP, needs `union_attn`): the
        main KV is split by token over TP rank pairs (`me`, `me ^ 1`; owner `(t // B) % 2`, one
        table column = `2 B` tokens); each rank attends both ranks' heads over the groups it owns
        and the pair combines the partials by LSE (`QSAFullStepBlock`). `compressed_dcp`
        (VLLM_TPU_QSA_COMPRESSED_DCP, needs `dcp` = 2 and the carrier with `bounded_chunks`):
        the compressed keys are split the same way (group `g` on rank `(g // gpp) % 2`, `gpp` =
        groups per pool block; `qsa_cktab` lists a rank's own pages in local order). Both ranks
        of a pair score the same request, each over its own groups, swap their halves of the
        scores (in-kernel remote DMA, F32 bits: exact) and select over the whole set; with
        `shard_index` the requests are sharded over pairs."""
        super().__init__(**kw)
        assert max_groups % 1024 == 0, max_groups  # whole (8, 128) score tiles per row
        self.max_groups, self.pps, self.debug = max_groups, pages_per_seq, debug
        self.carrier, self.ck_width = carrier, ck_width
        self.mp, self.ck_lanes = manager_pages, ck_lanes or ck_width
        self.key_lanes = self.ck_lanes  # carrier lanes the key stream copies (wholekeylanes)
        self.key_slots = int(key_slots)
        assert self.key_slots >= 2, key_slots
        assert self.key_slots == 2 or (carrier and bounded_chunks and not compressed_dcp), (
            "a deeper key ring: carrier keys, bounded per-request chunks, no compressed DCP"
        )
        self.n_layers = n_layers
        # `export_ids`: copy the last QSA layer's `qi_id` to the `SEL_IDS` output (stage C;
        # with `n_layers` the layer ordinal `n_layers - 1`, else every layer, the last wins).
        self.export_ids = bool(export_ids)
        self.hbm_tables, self.per_request_scores = hbm_tables, per_request_scores
        self.bounded_chunks = bounded_chunks
        self.shard_index = shard_index
        # Selection-path DMAs to distinct destinations issued together and waited once
        # (row ids into SMEM, the new group's loads/stores) instead of start-wait pairs.
        self.async_dma = bool(async_dma)
        # `skip_idle` (needs `loop_requests`; the MK draft at one row per request): the
        # per-request loops stop after the last live request (context > 0), so the 16
        # request slots of a T = 1 tile do no work when idle. Their outputs are left
        # unwritten: the cache writer and the caller read only live requests.
        self.skip_idle = bool(skip_idle)
        # `xreq_prefetch` bit 0 (union attention): one tile stream across the request loop;
        # request q+1's ids are staged (second `qu_row` bank) and its first tiles issued into
        # the ring slots request q's last tiles free (`union_attn.attend_request`'s `stream`).
        # Bit 1 (`loop_requests`): the request loop's own loads (raw ring row, RoPE row, recent
        # tile's pool rows) issued one request ahead into a second bank. Exact.
        self.xreq_prefetch = bool(int(xreq_prefetch) & 1) and bool(union_attn)
        self.xreq_loads = bool(int(xreq_prefetch) & 2) and bool(loop_requests)
        # `req_async` (VLLM_TPU_MK_QSA_REQ_ASYNC, `loop_requests`): the request loops wait on
        # nothing they do not read yet. The new group's two output stores go to bank q mod 2
        # and are waited before that bank's next write (q + 2) or at the layer's end; the union
        # attention's ids staging into `qu_row` (without bit 0's stream) for q + 1 is issued
        # right after request q's attention (1) or, into a second `qu_row` bank, before it (2),
        # and waited just before q + 1's attention. Exact.
        self.req_async = bool(int(req_async)) and bool(loop_requests)
        self.req_async_ids = self.req_async and bool(union_attn) and not self.xreq_prefetch
        self.req_async_bank = self.req_async_ids and int(req_async) >= 2
        assert not skip_idle or loop_requests, "skip_idle bounds the request loops"
        # `xmerge` (VLLM_TPU_MK_QSA_XMERGE; `shard_index` + `union_attn`, not
        # `compressed_dcp`): the selection rows and the union of each request go out in one
        # all-gather per QSA layer (`_exchange_merged`) instead of two. Both payloads are
        # ready at the same point and copied as int32, so every rank receives the same bytes.
        self.xmerge = bool(xmerge) and bool(shard_index) and bool(union_attn) and not compressed_dcp
        # `uattn_lite` (VLLM_TPU_MK_UATTN_LITE): union attention's per-tile op cuts
        # (`attend_request`'s `lite`); outputs unchanged.
        self.uattn_lite = bool(uattn_lite) and bool(union_attn)
        assert not shard_index or per_request_scores, (
            "the sharded indexer selects per request"
        )
        self.union_attn = union_attn
        assert not union_attn or per_request_scores, "the union is built per request"
        self.dcp = max(int(dcp), 1)
        assert self.dcp in (1, 2), dcp
        assert self.dcp == 1 or union_attn, "main-KV DCP is implemented on the union path"
        self.compressed_dcp = compressed_dcp
        assert not compressed_dcp or (
            self.dcp == 2 and carrier and per_request_scores and bounded_chunks
        ), "compressed-key DCP: pair split, carrier keys, per-request bounded scores"
        if union_attn:
            # The union path's scratch (codes, union ids, compaction) pushed the 16-row
            # kernel
            # past scoped VMEM (j-3d8ad822: 62.6 of 62.1 MiB; with loop_requests' row
            # copies and
            # 4 slots still 62.21, j-3d196b4b); a 3-slot ring (2 tiles in flight) keeps
            # a
            # request's union stream ahead of its reads.
            self.slots = min(self.slots, UNION_SLOTS)
        self.union_stats = union_stats
        assert not union_stats or union_attn, "the stats count the union"
        self.union_phys = union_phys
        assert not union_phys or (union_attn and hbm_tables), (
            "phys ids: union over HBM tables"
        )
        self.loop_requests = loop_requests
        assert not loop_requests or per_request_scores, (
            "the request loop selects per request"
        )
        self.chunk_pages = chunk_pages if carrier else max_groups // GPK
        assert (max_groups // GPK) % self.chunk_pages == 0, (
            max_groups,
            self.chunk_pages,
        )
        assert (self.chunk_pages * GPK) % 128 == 0 and (max_groups // 128) % 8 == 0
        assert not self.mp or self.chunk_pages % self.mp == 0
        assert not compressed_dcp or self.n_chunks % self.dcp == 0, self.n_chunks
        # `select_nvis` (VLLM_TPU_MK_QSA_SELECT_NVIS, the per-request selection): the
        # threshold bisection and the row compaction scan only the score rows that hold
        # visible groups (`nvis_rows` = (bisection chunk rows, compaction half-chunk rows))
        # instead of all `max_groups / 128`; the rows past `n_vis` are invisible (-1) and
        # add nothing, so thresholds, ids and order are unchanged.
        self.select_nvis = bool(int(select_nvis)) and per_request_scores
        # 2: also the union compaction (`qsa_ucompact`) over those rows.
        self.select_nvis_union = self.select_nvis and int(select_nvis) >= 2 and union_attn
        # Parts (VLLM_TPU_MK_QSA_NVIS_PARTS, measurement only): t = thresholds, c = row
        # compaction, u = union compaction.
        parts = os.environ.get("VLLM_TPU_MK_QSA_NVIS_PARTS", "tcu")
        self.nvis_thr = self.select_nvis and "t" in parts
        self.nvis_cmp = self.select_nvis and "c" in parts
        self.select_nvis_union = self.select_nvis_union and "u" in parts
        # Routing by visible groups (per request; per row for the compaction): the ranged
        # thresholds / compaction only up to VLLM_TPU_MK_QSA_NVIS_CUT groups (past it the
        # chunk loops cost more than the static full scan: exos ctx 100K), the ranged union
        # up to VLLM_TPU_MK_QSA_NVIS_UCUT; larger requests take the original path.
        self.nvis_cut = int(os.environ.get("VLLM_TPU_MK_QSA_NVIS_CUT", "16384"))
        self.nvis_ucut = int(os.environ.get("VLLM_TPU_MK_QSA_NVIS_UCUT", str(max_groups)))
        # 3: also the bisection `NVIS_DIGIT` bits per pass (fewer dependent count passes).
        self.nvis_digit = NVIS_DIGIT if self.select_nvis and int(select_nvis) >= 3 else 1
        self.nvis_rows = tuple(int(x) for x in nvis_rows)
        # VLLM_TPU_MK_QSA_UNION_LANEPACK (default off; exact): under main-KV DCP2 without
        # `compressed_dcp`, the ranged union of pair rank r packs the lanes it owns from two
        # adjacent code rows into one row (`_nvis_union`), so each of the D unions scans half
        # the rows instead of the whole visible range under a mask. Same ids, codes and order.
        self.union_lanepack = os.environ.get(
            "VLLM_TPU_MK_QSA_UNION_LANEPACK", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        # VLLM_TPU_MK_FP8_DENSE_READ (default off; exact): the FP8 key carrier slot is read as
        # dense int32 word rows and bitcast in registers (`_consume`), and the union attention
        # unpacks K / V ring slots from dense word rows (`qsa_attn.block._unpack_kv`).
        self.fp8_dense_read = fp8_dense_read()
        if self.fp8_dense_read:
            _log_once("MK FP8 dense read active: K/V and key ring slots as dense word rows")
        # VLLM_TPU_MK_QSA_THR_FAST (default off; exact): the full-scan selection's
        # thresholds skip the id passes unless a tie is truncated (t), stop the value
        # passes once the count at `lo` is exactly the budget (e), and scan only the
        # smallest static row tier {S/2, 3S/4, S} that holds the visible groups, for the
        # thresholds and the row compaction (r). "1" = "ter2" (exos c16 mix best; three
        # tiers' static copies slowed every rank's select at b16); letters pick parts.
        # VLLM_TPU_MK_QSA_THR_EXIT = "bit,every": first while-loop bit, passes per trip.
        # `thr_fast` (not None): this kernel's value instead of the env (per row bucket:
        # VLLM_TPU_MK_QSA_THR_FAST_BUCKETS, `qwen38_wholestep_call.thr_fast_options`).
        fast = thr_fast if thr_fast is not None else os.environ.get("VLLM_TPU_MK_QSA_THR_FAST", "0")
        fast = "ter2" if fast == "1" else "" if fast == "0" else fast
        self.thr_fast_mode = fast  # resolved value, read by the startup option log
        ex_bit, ex_every = (int(x) for x in
                            os.environ.get("VLLM_TPU_MK_QSA_THR_EXIT", "19,2").split(","))
        self.thr_kw = dict(tie_skip="t" in fast,
                           early_exit=(ex_bit, ex_every) if "e" in fast else ())
        # Tiers for the thresholds (h) / the compaction (c) / both (r); "2": tiers S/2, S only
        # (each tier is one more static copy of the scan: code size).
        # `nvis_balance` (VLLM_TPU_MK_QSA_NVIS_BALANCE_BUCKETS, per row bucket; exact): the
        # sharded selection gives each run whole requests balanced by visible groups (the
        # largest request with the smallest) instead of consecutive runs of `npr` requests.
        # Needs the served request loop with `xmerge`; only buckets with two requests per run
        # (`QSAFullCompleting._balanced`). Each request is still selected whole on one rank.
        self.nvis_balance = (
            bool(nvis_balance) and bool(shard_index) and bool(loop_requests) and self.xmerge
            and not skip_idle and not compressed_dcp
        )
        self.thr_tiers = "r" in fast or "h" in fast
        self.cmp_tiers = "r" in fast or "c" in fast
        self.tier_two = "2" in fast
        if self.select_nvis:
            S, rs = max_groups // 128, max_groups // 128 // self.n_chunks
            Sh = S // (self.dcp if compressed_dcp else 1)
            for x in self.nvis_rows:
                assert x % rs == 0 and Sh % x == 0, (self.nvis_rows, rs, Sh)

    @property
    def n_chunks(self) -> int:
        return self.max_groups // GPK // self.chunk_pages

    # ------------------------------------------------- compressed-key DCP (`compressed_dcp`)
    # Score layout: rank r's own groups, in local order, fill chunks [r * NC / D, (r + 1) * NC / D)
    # of `qi_s`; so score position p holds group `_cdcp_gid(p)` and each half is ascending.
    def _cdcp_gpp(self, ctx: Any) -> int:
        """Groups per pool block: the unit of the owner rule `(g // gpp) % D`."""
        return ctx.extra[POOL].shape[1] // MICRO

    def _cdcp_local_nv(self, ctx: Any, nv: Any, rk: Any) -> Any:
        """How many of rank `rk`'s own groups are among the first `nv` (its visible prefix)."""
        gpp, D = self._cdcp_gpp(ctx), self.dcp
        return (nv // (D * gpp)) * gpp + jnp.clip(nv % (D * gpp) - rk * gpp, 0, gpp)

    def _cdcp_gid(self, ctx: Any, p: Any) -> Any:
        """Group id at score position `p` (`cp_local_to_global` of the half's local index)."""
        gpp, D, Gh = self._cdcp_gpp(ctx), self.dcp, self.max_groups // self.dcp
        loc = jax.lax.rem(p, Gh)
        return (loc // gpp) * (D * gpp) + (p // Gh) * gpp + jax.lax.rem(loc, gpp)

    def outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        out = super().outputs(g, ex)
        if self.export_ids:
            out.append(StateDecl(SEL_IDS, (ex.M, 1, self.budget), I32, "w", per_layer=False,
                                 memory_space="hbm"))  # never VMEM: 128 KiB per 64 rows
        if self.debug:
            out += [
                StateDecl(DBG_IQ, (ex.M, 4 * ix.IDX_DIM), F32, "w"),
                StateDecl(DBG_IDS, (ex.M, 1, self.budget), I32, "w"),
            ]
        return out

    def host_update_states(self, g, ex, pos, states, raw_outputs):  # noqa: ANN001
        out = super().host_update_states(g, ex, pos, states, raw_outputs)
        if self.export_ids:
            out[SEL_IDS] = raw_outputs[SEL_IDS]
        if self.debug:
            out.update({k: raw_outputs[k] for k in (DBG_IQ, DBG_IDS)})
        return out

    def weights(self, g: Any, ex: ExecSpec) -> list[WeightDecl]:
        # Declaration order is the stream's consumption order: qkvg, idx_w, then o_proj.
        qkvg, o_proj = super().weights(g, ex)
        return [qkvg, WeightDecl(IDX_W, K=g.H, N=IDX_N, dtype=BF16), o_proj]

    def smalls(self, g: Any, ex: ExecSpec) -> list[SmallField]:
        return super().smalls(g, ex) + [SmallField("qsa_iqn", ix.IDX_DIM)]

    def meta_rows(self, ex: ExecSpec) -> int:
        """Rows of the per-request tables: requests, times the QSA layers if
        per-layer.
        """
        return ex.M // ex.T * max(1, self.n_layers)

    def row_base(self, ctx: Any, l: Any) -> Any:  # noqa: E741
        """This layer's first table row (0 without per-layer metadata)."""
        if not self.n_layers:
            return 0
        return ctx.stream.sp.block_ord(l, self.name) * (ctx.ex.M // ctx.ex.T)

    def table_shape(self, ex: ExecSpec, cols: int) -> tuple[int, ...]:
        """`[rows, cols]` SMEM tables, or `[layers, requests, cols]` with
        `hbm_tables`.
        """
        if self.hbm_tables:
            return (max(1, self.n_layers), ex.M // ex.T, cols)
        return (self.meta_rows(ex), cols)

    def tab_base(self, ctx: Any, l: Any) -> Any:  # noqa: E741
        """First row of this layer's tables in the refs `tables` returns."""
        return 0 if self.hbm_tables else self.row_base(ctx, l)

    def tables(self, ctx: Any) -> tuple[Any, Any]:
        """`(key page table, main page table)` refs (SMEM windows with `hbm_tables`)."""
        if self.hbm_tables:
            return ctx.scr["qi_ckw"], ctx.scr["qi_mtw"]
        return (ctx.extra[CKTAB] if self.carrier else None), ctx.extra[MTAB]

    def load_tables(self, ctx: Any, l: Any) -> None:  # noqa: E741
        """Copy this layer's table rows into the SMEM windows (`hbm_tables` only)."""
        if not self.hbm_tables:
            return
        # HBM tables are [layers, requests, cols]: a layer is one whole leading-dim
        # block (a
        # row offset `o * requests` into a 2-D table is not tile-aligned for 12
        # requests).
        o = ctx.stream.sp.block_ord(l, self.name) if self.n_layers else 0
        scr = ctx.scr
        cps = [
            pltpu.make_async_copy(ctx.extra[src].at[o], scr[dst], scr["qi_tsem"].at[i])
            for i, (src, dst) in enumerate(((CKTAB, "qi_ckw"), (MTAB, "qi_mtw")))
        ]
        for c in cps:
            c.start()
        for c in cps:
            c.wait()

    def offset(self, ctx: Any, l: Any) -> Any:  # noqa: E741
        if not self.n_layers:
            return ctx.extra[OFFSET][0]
        return ctx.extra[OFFSET][ctx.stream.sp.block_ord(l, self.name)]

    def extra_inputs(self, g: Any, ex: ExecSpec) -> list[ExtraInput]:
        nreq = ex.M // ex.T
        return [
            ExtraInput(POOL, "hbm"),
            *(
                [
                    ExtraInput(CK, "hbm"),
                    ExtraInput(
                        CKTAB,
                        "hbm" if self.hbm_tables else "smem",
                        self.table_shape(ex, self.max_groups // GPK // (self.mp or 1)),
                        I32,
                    ),
                ]
                if self.carrier
                else [ExtraInput(CK, "hbm", (nreq, self.max_groups, ix.IDX_DIM), BF16)]
            ),
            ExtraInput(NVIS, "smem", (ex.M,), I32),
            ExtraInput(TAIL, "smem", (ex.M,), I32),
            ExtraInput(
                MTAB,
                "hbm" if self.hbm_tables else "smem",
                self.table_shape(ex, self.pps),
                I32,
            ),
            ExtraInput(OFFSET, "smem", (max(1, self.n_layers),), I32),
            ExtraInput(SCALES, "smem", (2,), F32),
            ExtraInput(ROPE, "hbm", (ex.M, 128), F32),
        ]

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> list[ScratchReq]:
        G, S, K, MB = self.max_groups, self.max_groups // 128, self.budget, ex.MB
        return super().scratch(g, ex, plan) + [
            *(
                [
                    ScratchReq(
                        "qi_ring",
                        pltpu.VMEM(
                            (
                                self.key_slots,
                                self.chunk_pages,
                                16,
                                4 if self.carrier == "fp8" else 2,
                                self.ck_lanes,
                            ),
                            jnp.float8_e4m3fn if self.carrier == "fp8" else BF16,
                        ),
                    ),
                    ScratchReq("qi_ring_sem", pltpu.SemaphoreType.DMA((self.key_slots,))),
                ]
                if self.carrier
                else [ScratchReq("qi_ck", pltpu.VMEM((G, ix.IDX_DIM), BF16))]
            ),
            ScratchReq(
                "qi_q", pltpu.VMEM((MB // ex.T, ex.T * ix.IDX_HEADS, ix.IDX_DIM), F32)
            ),
            *(
                [ScratchReq("qi_cc", pltpu.SMEM((1,), I32))]
                if self.bounded_chunks
                else []
            ),
            ScratchReq(
                "qi_s",
                pltpu.VMEM(
                    (
                        ex.T if self.per_request_scores else MB,
                        self.n_chunks,
                        S // self.n_chunks,
                        128,
                    ),
                    F32,
                ),
            ),
            *(
                [
                    ScratchReq(
                        "qi_ckw",
                        pltpu.SMEM((MB // ex.T, G // GPK // (self.mp or 1)), I32),
                    ),
                    ScratchReq("qi_mtw", pltpu.SMEM((MB // ex.T, self.pps), I32)),
                    ScratchReq("qi_tsem", pltpu.SemaphoreType.DMA((2,))),
                ]
                if self.hbm_tables
                else []
            ),
            ScratchReq("qi_th", pltpu.VMEM((MB, 1, 128), I32)),
            ScratchReq("qi_tm", pltpu.VMEM((MB, 1, 128), I32)),
            ScratchReq("qi_nv", pltpu.VMEM((MB, 1, 128), I32)),
            ScratchReq("qi_id", pltpu.VMEM((MB, 1, K), I32)),
            ScratchReq("qi_ids", pltpu.SMEM((MB, K + 128), I32)),
            ScratchReq("qi_kvl", pltpu.SMEM((MB,), I32)),
            ScratchReq("qi_z", pltpu.VMEM((1, 128), I32)),
            *(
                [ScratchReq("qi_dbg", pltpu.VMEM((MB, 4 * ix.IDX_DIM), F32))]
                if self.debug
                else []
            ),
            ScratchReq("qi_sem", pltpu.SemaphoreType.DMA((3,))),
            *([ScratchReq("qe_sem", pltpu.SemaphoreType.DMA((1,)))] if self.export_ids else []),
            *(
                [
                    ScratchReq("qx_ssem", pltpu.SemaphoreType.DMA((2,))),
                    ScratchReq("qx_rsem", pltpu.SemaphoreType.DMA((2,))),
                    ScratchReq("qx_credit", pltpu.SemaphoreType.REGULAR((1,))),
                ]
                if self.compressed_dcp
                else []
            ),
        ]

    # ------------------------------------------------------------ in kernel
    def _score_request(
        self, ctx: Any, q: int, qr: jax.Array, base: Any = 0, last: bool = True, row0: Any = None,
        first: bool | None = None, c_off: Any = 0, nxt: Any = None,
    ) -> None:
        """Scores of request `q`'s `T` rows into `qi_s[q*T : (q+1)*T]` (chunk-major).

        Carrier mode streams chunks through a 2-slot ring whose slot follows a chunk counter
        across requests (`q * n_chunks + c`): request q's last chunk prefetches request q+1's
        first chunk (unless `last`), so no request starts its key stream cold; request 0's
        first chunk is issued here. `first` (default `q == 0`) marks the first request of the
        rank's run when the indexer is sharded by request (`shard_index`). `compressed_dcp`:
        only this rank's own groups (`qsa_cktab` lists its pages), scored into chunks from
        `c_off` on. `nxt` (default q + 1): the request whose first chunk this request's last
        chunk prefetches (`nvis_balance`: the run's next request)."""
        scr, T = ctx.scr, ctx.ex.T
        nxt = q + 1 if nxt is None else nxt
        r0 = q * T if row0 is None else row0
        rows = slice(r0, r0 + T)
        if not self.carrier:
            cp = pltpu.make_async_copy(
                ctx.extra[CK].at[q], scr["qi_ck"], scr["qi_sem"].at[0]
            )
            cp.start()
            cp.wait()
            st = ix.score_rows(qr, scr["qi_ck"][...])  # [T, G]
            scr["qi_s"][rows, 0] = st.reshape(T, self.max_groups // 128, 128)
            return
        CH, NC, W, MP = self.chunk_pages, self.n_chunks, self.ck_lanes, self.mp
        CG = CH * GPK
        ring, sems = scr["qi_ring"], scr["qi_ring_sem"]
        lanes = pl.ds(0, W)

        def copies(c: Any, s: Any, qq: int = q) -> list[Any]:
            if SCORE_ARM == "compute":  # timing arm: no key copies
                return []
            tab, qb = self.tables(ctx)[0], base + qq
            if MP:  # one DMA per manager page of MP contiguous kernel pages
                return [
                    pltpu.make_async_copy(
                        ctx.extra[CK].at[
                            pl.ds(tab[qb, c * (CH // MP) + m] * MP, MP), :, :, lanes
                        ],
                        ring.at[s, pl.ds(m * MP, MP)],
                        sems.at[s],
                    )
                    for m in range(CH // MP)
                ]
            return [
                pltpu.make_async_copy(
                    ctx.extra[CK].at[tab[qb, c * CH + p], :, :, lanes],
                    ring.at[s, p],
                    sems.at[s],
                )
                for p in range(CH)
            ]

        first = (q == 0) if first is None else first
        if self.key_slots > 2:
            self._score_deep(ctx, q, qr, copies, rows, first, last, c_off, nxt)
            return

        def issue_first() -> None:
            for cp in copies(0, 0):
                cp.start()
            if self.bounded_chunks:
                scr["qi_cc"][0] = 0

        if isinstance(first, bool):
            if first:
                issue_first()
        else:  # traced (`loop_requests`)
            pl.when(first)(issue_first)
        if self.bounded_chunks:
            # Chunks this request's rows can see (the last row sees the most); slots
            # follow a
            # chunk counter carried across requests, as the fixed count's `q * NC + c`
            # does.
            nv = ctx.extra[NVIS][q * T + T - 1]
            if self.compressed_dcp:
                nv = self._cdcp_local_nv(ctx, nv, jax.lax.rem(ctx.me, self.dcp))
            n_q = jnp.clip((nv + CG - 1) // CG, 1, NC // (self.dcp if self.compressed_dcp else 1))
            c0 = scr["qi_cc"][0]
        else:
            n_q, c0 = NC, q * NC

        def chunk(c: Any, carry: Any) -> Any:
            s = jax.lax.rem(c0 + c, 2)

            @pl.when(c + 1 < n_q)
            def _():
                for cp in copies(c + 1, 1 - s):
                    cp.start()

            if (
                last is not True
            ):  # the next request's first chunk into the slot this chunk frees

                @pl.when((c + 1 == n_q) & jnp.logical_not(last))
                def _():
                    for cp in copies(0, 1 - s, nxt):
                        cp.start()

            self._consume(ctx, qr, s, rows, c_off + c)
            return carry

        jax.lax.fori_loop(0, n_q, chunk, 0)
        if self.bounded_chunks:
            scr["qi_cc"][0] = c0 + n_q

    def _consume(self, ctx: Any, qr: jax.Array, s: Any, rows: slice, col: Any) -> None:
        """Wait for ring slot `s` and score its chunk into score column `col` of `rows`."""
        scr, T = ctx.scr, ctx.ex.T
        ring, sems = scr["qi_ring"], scr["qi_ring_sem"]
        CG, W = self.chunk_pages * GPK, self.ck_lanes
        if SCORE_ARM != "compute":
            # One wait for the whole slot (the page copies add up to its bytes).
            pltpu.make_async_copy(ring.at[s], ring.at[s], sems.at[s]).wait()
        if SCORE_ARM == "dma":
            scr["qi_s"][rows, col] = jnp.zeros((T, CG // 128, 128), F32)
            return
        if self.carrier == "fp8" and self.fp8_dense_read:
            # The slot is row-major words `[CG / 2 * W / 128, 128]`, two BF16 keys per word
            # (key 2i in the low half, as the ref bitcast FP8 -> BF16 reads it): whole-vreg
            # reads of the first 128 lanes, then an in-register bitcast with the same row
            # order. A `[CH, 16, 2, W]` BF16 read loads two rows per vreg and relayouts.
            wrows = ring.at[s].bitcast(I32).reshape(CG // 2 * (W // 128), 128)
            words = wrows[pl.ds(0, CG // 2, stride=W // 128)] if W > 128 else wrows[...]
            keys = pltpu.bitcast(words, BF16)
        else:
            # Stock BF16KeyReader read: ref bitcast FP8 -> BF16 when carrier == "fp8".
            keys = ring.at[s].bitcast(BF16)[...] if self.carrier == "fp8" else ring[s]
            keys = keys.reshape(CG, W)[:, : ix.IDX_DIM]
        st = ix.score_rows(qr, keys)  # [T, CG]
        scr["qi_s"][rows, col] = st.reshape(T, CG // 128, 128)

    def _score_deep(
        self, ctx: Any, q: Any, qr: jax.Array, copies: Any, rows: slice, first: Any, last: Any,
        c_off: Any, nxt: Any = None,
    ) -> None:
        """`_score_request`'s key stream through a `key_slots`-slot ring (bounded chunks):
        `key_slots - 1` chunks are in flight while one is scored. Chunk `c` of the rank's
        stream (counter `c0 + c` carried across requests) lives in slot `(c0 + c) % KS`; at
        chunk `c` the copy of chunk `c + KS - 1` goes into the slot chunk `c - 1` freed. A
        request's last chunk prefetches the next request's first chunk (unless `last`), as
        the 2-slot ring does; the next request issues its chunks 1..KS-2 on entry. Same keys
        and the same scores as the 2-slot ring, only the copy schedule differs."""
        scr, T, KS = ctx.scr, ctx.ex.T, self.key_slots
        nxt = q + 1 if nxt is None else nxt
        CG, NC = self.chunk_pages * GPK, self.n_chunks
        nv = ctx.extra[NVIS][q * T + T - 1]
        n_q = jnp.clip((nv + CG - 1) // CG, 1, NC)
        c0 = jnp.where(first, 0, scr["qi_cc"][0])

        def start(c: Any, pos: Any = None, qq: Any = q) -> None:
            """Copy chunk `c` of request `qq` into the slot of stream position c0 + pos."""
            for cp in copies(c, jax.lax.rem(c0 + (c if pos is None else pos), KS), qq):
                cp.start()

        for k in range(KS - 1):  # chunks 0..KS-2 (chunk 0 came with the previous request)
            cond = k < n_q
            if k == 0:
                if isinstance(first, bool):
                    if not first:
                        continue
                else:
                    cond = cond & first
            pl.when(cond)(functools.partial(start, k))

        def chunk(c: Any, carry: Any) -> Any:
            pl.when(c + KS - 1 < n_q)(lambda: start(c + KS - 1))
            if last is not True:
                # The next request's first chunk, into the slot after this request's last.
                pl.when((c + 1 == n_q) & jnp.logical_not(last))(lambda: start(0, n_q, nxt))
            self._consume(ctx, qr, jax.lax.rem(c0 + c, KS), rows, c_off + c)
            return carry

        jax.lax.fori_loop(0, n_q, chunk, 0)
        scr["qi_cc"][0] = c0 + n_q

    def _select(self, ctx: Any, l: Any, slot: Any) -> None:  # noqa: E741
        ex, scr, g = ctx.ex, ctx.scr, ctx.g
        MB, T, K, G = ex.MB, ex.T, self.budget, self.max_groups
        S, rot = G // 128, self.rot
        with ctx.stream.segment(lazy=False):
            yi = rnd(ctx.stream.dmatmul(IDX_W, scr["xb"], IDX_N))
        cos = scr["qf_rope"][:, : rot // 2]
        sin = scr["qf_rope"][:, rot // 2 : rot]
        iq = ix.index_query(
            yi, ctx.sm(slot, "qsa_iqn"), cos, sin, rot, g.eps, pltpu.bitcast
        )
        nreq = MB // T
        for q in range(
            nreq
        ):  # row-major (row, head) query tile per request, static stores
            for t in range(T):
                for h in range(ix.IDX_HEADS):
                    i = t * ix.IDX_HEADS + h
                    scr["qi_q"][q, i : i + 1, :] = iq[h][q * T + t : q * T + t + 1]
        self.load_tables(ctx, l)
        base = self.tab_base(ctx, l)
        for q in range(nreq):
            self._score_request(
                ctx, q, scr["qi_q"][q].astype(BF16), base, last=q == nreq - 1
            )
        rid = jax.lax.broadcasted_iota(I32, (MB, 1, 1), 0)
        nv = jnp.zeros((MB, 1, 1), I32)
        for r in range(MB):
            nv = jnp.where(rid == r, ctx.extra[NVIS][r], nv)
        ids = jax.lax.broadcasted_iota(
            I32, (MB, S, 128), 1
        ) * 128 + jax.lax.broadcasted_iota(I32, (MB, S, 128), 2)
        bits = jnp.where(
            ids < nv, pltpu.bitcast(scr["qi_s"][...].reshape(MB, S, 128), I32), -1
        )
        thr, tmax = ix.thresholds(bits, ids, nv, K, G, **self.thr_kw)
        scr["qi_th"][...] = jnp.broadcast_to(thr, (MB, 1, 128))
        scr["qi_tm"][...] = jnp.broadcast_to(tmax, (MB, 1, 128))
        scr["qi_nv"][...] = jnp.broadcast_to(nv, (MB, 1, 128))
        consts = ix.compact_consts(S, K)
        lid = consts[-1]
        scr["qi_z"][...] = jnp.zeros((1, 128), I32)

        def row(r: Any, carry: Any) -> Any:
            nvr = scr["qi_nv"][r][:, :1]
            b = jnp.where(
                lid < nvr, pltpu.bitcast(scr["qi_s"][r].reshape(S, 128), I32), -1
            )
            scr["qi_id"][r] = ix.compact_row(
                b, scr["qi_th"][r][:, :1], scr["qi_tm"][r][:, :1], nvr, K, consts
            )
            # Columns past K are read by the last tile's copies: keep them at group 0.
            cs = [
                pltpu.make_async_copy(
                    scr["qi_id"].at[r],
                    scr["qi_ids"].at[pl.ds(r, 1), pl.ds(0, K)],
                    scr["qi_sem"].at[1],
                ),
                pltpu.make_async_copy(
                    scr["qi_z"],
                    scr["qi_ids"].at[pl.ds(r, 1), pl.ds(K, 128)],
                    scr["qi_sem"].at[2],
                ),
            ]
            for c in cs:
                c.start()
            for c in cs:
                c.wait()
            n, tail = ctx.extra[NVIS][r], ctx.extra[TAIL][r]
            cnt = jnp.minimum(n, K)

            @pl.when(tail > 0)
            def _():
                scr["qi_ids"][r, cnt] = n  # the partial group follows the selection

            scr["qi_kvl"][r] = 4 * cnt + tail
            return carry

        jax.lax.fori_loop(0, MB, row, 0)
        if self.debug:
            for h in range(ix.IDX_HEADS):
                scr["qi_dbg"][:, h * ix.IDX_DIM : (h + 1) * ix.IDX_DIM] = iq[h]
            o = ctx.stream.sp.block_ord(l, self.name)
            for src, dst in ((scr["qi_dbg"], DBG_IQ), (scr["qi_id"], DBG_IDS)):
                c = pltpu.make_async_copy(
                    src, ctx.O[dst].at[o, pl.ds(0, MB)], scr["qi_sem"].at[0]
                )
                c.start()
                c.wait()

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        ex, scr, g = ctx.ex, ctx.scr, ctx.g
        nh = self.nh(g, ex)
        kv = self._project(ctx, l, slot)
        self._select(ctx, l, slot)
        rows = jax.lax.broadcasted_iota(I32, (ex.MB, 1), 0)

        def row(r: Any, carry: Any) -> Any:
            *outs, g0, pre = carry
            out, g0, pre = attend_row(
                r,
                g0,
                pre,
                nrows=ex.MB,
                pool=ctx.extra[POOL],
                ring=scr["qf_ring"],
                sems=scr["qf_ring_sem"],
                pages=scr["qi_ids"],
                kv_len=scr["qi_kvl"],
                table=self.tables(ctx)[1],
                rows_per_req=ex.T,
                table_row_base=self.tab_base(ctx, l),
                q=scr["qf_q"][r],
                gate=scr["qf_g"][r],
                off=self.offset(ctx, l),
                scale_k=ctx.extra[SCALES][0],
                scale_v=ctx.extra[SCALES][1],
                sm_scale=g.HD**-0.5,
                interpret=ex.interpret,
            )
            return (
                *(jnp.where(rows == r, out[h : h + 1], o) for h, o in enumerate(outs)),
                g0,
                pre,
            )

        zeros = tuple(jnp.zeros((ex.MB, g.HD), F32) for _ in range(nh))
        *outs, _, _ = jax.lax.fori_loop(
            0, ex.MB, row, (*zeros, jnp.int32(0), jnp.int32(0))
        )
        for h, o in enumerate(outs):
            scr["xb2"][:, pl.ds(h * g.HD, g.HD)] = o.astype(BF16)
        kv.wait()
        with ctx.stream.segment(lazy=False):
            return ctx.stream.dmatmul("o_proj", scr["xb2"], g.H)
