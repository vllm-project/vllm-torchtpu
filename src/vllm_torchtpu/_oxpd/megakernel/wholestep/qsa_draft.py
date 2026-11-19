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
"""q38 L1-H3 (H3-DRAFT-ATTN-D14): the MTP draft layer's QSA attention as an L = 1, attention-only
whole-step kind on D14's in-kernel attention and DCP-exchange blocks.

The stock draft attends the target's carried selection (VLLM_TPU_QSA_DRAFT_SHARED_SELECTION,
`pooled_qsa.shared_blocks`) without indexer state (VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE, DRAFT-1):
per pass the projections, norms and RoPE, the main K/V write, RPAd over the selected groups plus
the row's partial group, and under main-KV DCP the q all-gather, the partial all-gather and
`qsa_rpa.dcp_combine`. This kind runs, in one kernel call per pass:

  qkvg = block @ W_qkvg, q / k gemma norm + partial RoPE      (stage A `_project`, unchanged)
  per request: the attended row's pool-path groups (this rank's owned ones, from the host)
               -> D14's union scratch (one row's union: its own list; code = its row bit)
  the recent tile of the step's own tokens                    (stage C `_recent_tile`, unchanged)
  union attention + the DCP q / partial exchange and merge    (D14 `_attend_unions` / `_dcp_*`)
  outputs: the gated attention before o_proj `qsa_attn_out [M, nh * HD]` BF16 (the d14mtp tail's
           `attn` operand), and the step's K / V rows `qsa_kv_new` / `qsa_kr_new` for the
           post-kernel stock main-KV writer (as D14: the pool is not written in-kernel).

`block` is the draft layer's attention input after its HC mix (the kernel's hidden input), so no
HC sublayer and no all-reduce (`reduction = "none"`). The index projection, compressed keys and ring
are not computed (DRAFT-1). `ex.T` rows per request: 1 for the fused loop's passes, `T_VERIFY` for
draft step 0 (every verify row's K / V is written; only the draft row per request is attended:
the host gives the other rows no groups and their outputs are discarded).

Per-rank host input `qsa_shared [requests, 1, 2 KU + 128]` int32 (`shared_rows`): lanes [0, KU) the
attended row's pool-path groups (ids < g_a = ctx // 4) owned by this rank under main-KV DCP,
ascending, 0-padded; lanes [KU, 2 KU) their row codes (1 << t of the attended row; 0 past the
count); lane 2 KU the count. Replicated SMEM `qsa_draft_rows [2 M]`: per row the pool-path token
count over all ranks (4 x groups; the per-row path's `qi_kvl`), then whether g_a is listed (`qc_sga`).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..blocks.qsa_attn.block import MICRO, TILE
from ..exec_spec import ExecSpec
from ..ops import BF16, F32, I32
from ..spec import ExtraInput, LayerType, ModelSpec, ScratchReq, SmallField, StateDecl, WeightDecl
from .qsa_full import KV_NEW, OFFSET, POOL, QKVG, ROPE, SCALES, qkvg_width
from .qsa_full_b import MTAB, NVIS, TAIL
from .qsa_full_c import CTXQ, KR_NEW, QSAFullStepBlock, QSAFullStepGroupBlock

DRAFT_ATTN = "qwen38_draft_attn"
ATTN_OUT = "qsa_attn_out"
SHARED = "qsa_shared"
ROWS = "qsa_draft_rows"
EXTRA_NAMES = [POOL, NVIS, TAIL, MTAB, OFFSET, SCALES, ROPE, CTXQ, SHARED, ROWS]


class _DraftAttn:
    """The draft kind's overrides on D14's stage-C QSA block (`QSAFullStepBlock` /
    `QSAFullStepGroupBlock`); union attention over HBM tables with the request loop only."""

    reduction = "none"

    def _draft_check(self) -> None:
        assert self.union_attn and self.hbm_tables and self.loop_requests, (
            "the draft kind runs D14's union path over HBM tables with the request loop"
        )
        assert not (self.shard_index or self.compressed_dcp or self.union_stats or self.n_layers), (
            "the draft kind has no indexer: no sharded selection, compressed-key DCP, stats or per-layer tables"
        )

    def ku(self, ex: ExecSpec) -> int:
        return self._union_width(ex)

    # ------------------------------------------------------------ declarations
    def weights(self, g: Any, ex: ExecSpec) -> list[WeightDecl]:
        # K = the block input's true width (`h_in`): the core's residual may be wider (`core_width`), and
        # `TwoRing.dmatmul` reads only `xb[:, :K]`.
        return [WeightDecl(QKVG, K=self.h_in or g.H, N=qkvg_width(self.nh(g, ex), g.HD), dtype=BF16)]

    def smalls(self, g: Any, ex: ExecSpec) -> list[SmallField]:
        return [SmallField("qsa_qn", g.HD), SmallField("qsa_kn", g.HD)]

    def outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        return [
            StateDecl(KV_NEW, (ex.M, 2 * g.HD), BF16, "w"),
            StateDecl(KR_NEW, (ex.M, g.HD), BF16, "w"),
            StateDecl(ATTN_OUT, (ex.M, self.nh(g, ex) * g.HD), BF16, "w"),
        ]

    def host_update_states(self, g, ex, pos, states, raw_outputs):  # noqa: ANN001
        return {k: raw_outputs[k] for k in (KV_NEW, KR_NEW, ATTN_OUT)}

    def extra_inputs(self, g: Any, ex: ExecSpec) -> list[ExtraInput]:
        nreq = ex.M // ex.T
        return [
            ExtraInput(POOL, "hbm"),
            ExtraInput(NVIS, "smem", (ex.M,), I32),
            ExtraInput(TAIL, "smem", (ex.M,), I32),
            ExtraInput(MTAB, "hbm", self.table_shape(ex, self.pps), I32),
            ExtraInput(OFFSET, "smem", (1,), I32),
            ExtraInput(SCALES, "smem", (2,), F32),
            ExtraInput(ROPE, "hbm", (ex.M, 128), F32),
            ExtraInput(CTXQ, "smem", (nreq,), I32),
            ExtraInput(SHARED, "hbm", (nreq, 1, 2 * self.ku(ex) + 128), I32),
            ExtraInput(ROWS, "smem", (2 * ex.M,), I32),
        ]

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> list[ScratchReq]:
        nh, MB, T, f8 = self.nh(g, ex), ex.MB, ex.T, jnp.float8_e4m3fn
        nreq, KU, D = MB // T, self.ku(ex), max(self.dcp, 1)
        out = [
            # stage A (`_project`)
            ScratchReq("qf_ring", pltpu.VMEM((self.slots, TILE, 1, 4, 256), f8)),
            ScratchReq("qf_ring_sem", pltpu.SemaphoreType.DMA((self.slots,))),
            ScratchReq("qf_q", pltpu.VMEM((MB, nh, g.HD), F32)),
            ScratchReq("qf_g", pltpu.VMEM((MB, nh, g.HD), F32)),
            ScratchReq("qf_kv", pltpu.VMEM((MB, 2 * g.HD), BF16)),
            ScratchReq("qf_rope", pltpu.VMEM((MB, 128), F32)),
            ScratchReq("qf_sem", pltpu.SemaphoreType.DMA((2,))),
            ScratchReq("qf_kr", pltpu.VMEM((MB, g.HD), BF16)),
            # tables, row metadata
            ScratchReq("qi_mtw", pltpu.SMEM((nreq, self.pps), I32)),
            ScratchReq("qi_tsem", pltpu.SemaphoreType.DMA((1,))),
            ScratchReq("qi_kvl", pltpu.SMEM((MB,), I32)),
            ScratchReq("qi_z", pltpu.VMEM((1, 128), I32)),
            ScratchReq("qi_sem", pltpu.SemaphoreType.DMA((3,))),
            # stage C recent tile (`_recent_tile`, request-major K / V copies of the loop)
            ScratchReq("qc_old", pltpu.VMEM((MICRO, 1, 4, 256), f8)),
            ScratchReq("qc_rk", pltpu.VMEM((nreq, 8, 256), F32)),
            ScratchReq("qc_rv", pltpu.VMEM((nreq, 8, 256), F32)),
            ScratchReq("qc_sga", pltpu.SMEM((MB,), I32)),
            ScratchReq("qc_sem", pltpu.SemaphoreType.DMA((3,))),
            ScratchReq("qc_osem", pltpu.SemaphoreType.DMA((2,))),
            ScratchReq("qc_kvq", pltpu.VMEM((nreq, T, 2 * g.HD), F32)),
            # union attention (`_attend_unions`): one union per DCP rank slot, this rank's filled
            ScratchReq("qu_id", pltpu.VMEM((nreq, 1, D * KU), I32)),
            ScratchReq("qu_cd", pltpu.VMEM((nreq, 1, D * KU), I32)),
            ScratchReq("qu_n", pltpu.SMEM((nreq, D), I32)),
            ScratchReq("qu_row", pltpu.SMEM((1, KU + 128), I32)),
            ScratchReq("qu_sem", pltpu.SemaphoreType.DMA((3,))),
            ScratchReq("qs_cnt", pltpu.VMEM((1, 128), I32)),
        ]
        if self.union_phys:
            out += [
                ScratchReq("qu_tab", pltpu.VMEM((1, self.pps), I32)),
                ScratchReq("qu_pv", pltpu.VMEM((1, KU), I32)),
            ]
        return out + (self._dcp_scratch(g, ex) if self.dcp > 1 else [])

    # ------------------------------------------------------------ tables (main table only)
    def tables(self, ctx: Any) -> tuple[Any, Any]:
        return None, ctx.scr["qi_mtw"]

    def load_tables(self, ctx: Any, l: Any) -> None:  # noqa: E741
        cp = pltpu.make_async_copy(ctx.extra[MTAB].at[0], ctx.scr["qi_mtw"], ctx.scr["qi_tsem"].at[0])
        cp.start()
        cp.wait()

    # ------------------------------------------------------------ in kernel
    def _select(self, ctx: Any, l: Any, slot: Any) -> None:  # noqa: E741
        """The host's selection into D14's union scratch, the per-row metadata, the recent tiles;
        `qsa_kr_new` out."""
        ex, scr = ctx.ex, ctx.scr
        MB, T = ex.MB, ex.T
        nreq, KU = MB // T, self.ku(ex)
        o = ctx.stream.sp.block_ord(l, self.name)
        kr = pltpu.make_async_copy(scr["qf_kr"], ctx.O[KR_NEW].at[o, pl.ds(0, MB)], scr["qc_osem"].at[0])
        kr.start()
        self.load_tables(ctx, l)
        for q in range(nreq):  # static stores into the request-major copies (`_recent_tile`)
            scr["qc_kvq"][q] = scr["qf_kv"][q * T : (q + 1) * T].astype(F32)
        rk = jax.lax.rem(ctx.me, self.dcp) if self.dcp > 1 else 0
        uoff = pl.multiple_of(rk * KU, 128) if self.dcp > 1 else 0
        lane = jax.lax.broadcasted_iota(I32, (1, 128), 1)

        def req(q: Any, carry: Any) -> Any:
            src = ctx.extra[SHARED].at[q]
            cs = [
                pltpu.make_async_copy(src.at[:, pl.ds(0, KU)], scr["qu_id"].at[q, :, pl.ds(uoff, KU)],
                                      scr["qi_sem"].at[0]),
                pltpu.make_async_copy(src.at[:, pl.ds(KU, KU)], scr["qu_cd"].at[q, :, pl.ds(uoff, KU)],
                                      scr["qi_sem"].at[1]),
                pltpu.make_async_copy(src.at[:, pl.ds(2 * KU, 128)], scr["qs_cnt"], scr["qi_sem"].at[2]),
            ]
            for c in cs:
                c.start()
            for c in cs:
                c.wait()
            scr["qu_n"][q, rk] = jnp.sum(jnp.where(lane == 0, scr["qs_cnt"][...], 0))
            self._recent_tile(ctx, l, q)
            return carry

        jax.lax.fori_loop(0, nreq, req, 0)

        def row(r: Any, carry: Any) -> Any:
            scr["qi_kvl"][r] = ctx.extra[ROWS][r]
            scr["qc_sga"][r] = ctx.extra[ROWS][ex.M + r]
            return carry

        jax.lax.fori_loop(0, MB, row, 0)
        scr["qi_z"][...] = jnp.zeros((1, 128), I32)
        kr.wait()

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        ex, scr, g = ctx.ex, ctx.scr, ctx.g
        nh, MB = self.nh(g, ex), ex.MB
        sc = ctx.stream.sc
        scr["xb"][...] = scr["res"][...].astype(BF16)  # the BF16 block input, exact
        with sc("qsa_proj"):
            kv = self._project(ctx, l, slot)
        with sc("qsa_select"):
            self._select(ctx, l, slot)
        zeros = tuple(jnp.zeros((MB, g.HD), F32) for _ in range(nh))
        with sc("qsa_attend"):
            if self.dcp > 1:
                self._dcp_attend(ctx, l, zeros)
                with sc("qsa_dcp_combine", 2):
                    outs = self._dcp_combine(ctx)
            else:
                outs = self._attend_unions(ctx, l, zeros)
        with sc("qsa_store"):
            for h, o_ in enumerate(outs):
                scr["xb2"][:, pl.ds(h * g.HD, g.HD)] = o_.astype(BF16)
            o = ctx.stream.sp.block_ord(l, self.name)
            cp = pltpu.make_async_copy(scr["xb2"], ctx.O[ATTN_OUT].at[o, pl.ds(0, MB)], scr["qc_osem"].at[1])
            cp.start()
            cp.wait()
            kv.wait()
        return None


class QSADraftAttnBlock(_DraftAttn, QSAFullStepBlock):
    """The draft attention kind at main-KV DCP 1 / 2 (q38 L1-H3)."""

    def __init__(self, *, h_in: int = 0, **kw: Any) -> None:
        super().__init__(**kw)
        self.h_in = h_in
        self._draft_check()


class QSADraftAttnGroupBlock(_DraftAttn, QSAFullStepGroupBlock):
    """The draft attention kind at main-KV DCP 4 (q38 L1-H3; D14's group exchange)."""

    def __init__(self, *, h_in: int = 0, **kw: Any) -> None:
        super().__init__(**kw)
        self.h_in = h_in
        self._draft_check()


def core_width(H: int) -> int:
    """The kernel core's residual width for a block input of width `H`: the core chunks its rows by
    `min(1024, H)` columns (`vmem.build_scratch` asserts `H % min(1024, H) == 0`), which Qwen3.8's single
    stream (2560) is not; the input is then zero-padded to the next multiple of 1024 (3072)."""
    return H if H % min(1024, H) == 0 else -(-H // 1024) * 1024


def draft_attn_spec(g: Any, R: int, *, budget: int, indexed: dict[str, Any]) -> ModelSpec:
    """The L = 1 draft attention kind; `indexed` = D14's stage-C options (`max_groups`,
    `pages_per_seq`, `union_attn`, `hbm_tables`, `per_request_scores`, `loop_requests`,
    `union_phys`, `dcp`); the selection options are refused by the block. The config is the inner
    geometry at the core width (`core_width`); the block's projection reads the true width `g.H`."""
    gi = g.inner(R, 1)
    gc = dataclasses.replace(gi, H=core_width(gi.H))
    dcp = int(indexed.get("dcp", 1) or 1)
    blk = (QSADraftAttnGroupBlock if dcp > 2 else QSADraftAttnBlock)(**indexed, budget=budget, h_in=gi.H)
    return ModelSpec(name=DRAFT_ATTN, config=gc, layer_type=LayerType(name=DRAFT_ATTN, slots=(blk,)))


# ---------------------------------------------------------------- host-side inputs
def shared_rows(
    blocks: Any,
    lengths: Any,
    ctxq: Any,
    attended: Any,
    *,
    T: int,
    KU: int,
    ratio: int,
    page_size: int,
    dcp: int = 1,
    dcp_rank: Any = 0,
) -> tuple[Any, Any]:
    """`(qsa_shared [nreq, 1, 2 KU + 128], qsa_draft_rows [2 M])` for one rank (JAX, traceable).

    `blocks [M, K]` int32: each kernel row's attended groups in stock `pooled_qsa.shared_blocks`
    layout (only columns below `min(lengths // ratio, K)` count; negative ids never count).
    `lengths [M]` the row's visible tokens (position + 1), `ctxq [nreq]` the request's tokens
    before the step, `attended [M]` bool: the row attended (one per request at most; others get
    no groups). Pool path = listed ids below g_a = ctx // 4 (the recent tile takes g_a itself);
    this rank keeps the groups it owns under main-KV DCP (`(4 g // page_size) % dcp == rank`)."""
    blocks = jnp.asarray(blocks, I32)
    lengths = jnp.asarray(lengths, I32)
    M, K = blocks.shape
    nreq = M // T
    ga = jnp.repeat(jnp.asarray(ctxq, I32) // 4, T)[:M]
    col = jnp.arange(K, dtype=I32)[None, :]
    count = jnp.minimum(lengths // ratio, K)
    listed = (col < count[:, None]) & (blocks >= 0) & jnp.asarray(attended)[:, None]
    pool = listed & (blocks < ga[:, None])
    kvl = 4 * jnp.sum(pool.astype(I32), axis=1)
    sga = jnp.any(listed & (blocks == ga[:, None]), axis=1).astype(I32)
    own = pool
    if dcp > 1:
        own = own & (jax.lax.rem((blocks * ratio) // page_size, dcp) == jnp.asarray(dcp_rank, I32).reshape(()))
    big = jnp.int32(2**31 - 1)
    ids = jnp.sort(jnp.where(own, blocks, big), axis=1)  # ascending, unowned last
    n = jnp.sum(own.astype(I32), axis=1)
    # One attended row per request: its list is the request's union.
    pick = jnp.asarray(attended).reshape(nreq, T)
    t_r = jnp.argmax(pick, axis=1).astype(I32)
    ridx = jnp.arange(nreq, dtype=I32) * T + t_r
    ids_q, n_q = ids[ridx], n[ridx]
    lanes = jnp.arange(KU, dtype=I32)[None, :]
    ids_q = jnp.pad(ids_q, ((0, 0), (0, KU - K))) if KU > K else ids_q[:, :KU]
    live = lanes < n_q[:, None]
    uid = jnp.where(live, ids_q, 0)
    code = jnp.where(live, jnp.left_shift(1, t_r)[:, None], 0)
    cnt = jnp.zeros((nreq, 128), I32).at[:, 0].set(n_q)
    shared = jnp.concatenate([uid, code, cnt], axis=1)[:, None, :]
    return shared, jnp.concatenate([kvl, sga])


def selection_sets(blocks: np.ndarray, lengths: np.ndarray, attended: np.ndarray, ratio: int) -> dict:
    """Per attended row, the set of listed groups (`shared_rows`' rule), for references."""
    out = {}
    K = blocks.shape[1]
    for r in range(blocks.shape[0]):
        if not attended[r]:
            continue
        c = min(int(lengths[r]) // ratio, K)
        out[r] = {int(x) for x in blocks[r, :c] if x >= 0}
    return out
