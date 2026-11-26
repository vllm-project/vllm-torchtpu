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
"""Stage C: the step's own tokens, with no in-place cache write inside the kernel.

The stock layer writes this step's K/V (FP8), raw ring rows and any completed group's
compressed key BEFORE scoring and attention. In-kernel writes would need the pool
aliased
inside a communicating call (refused by the core, and the pattern behind the
fused-KV-writer
hang), so the kernel instead keeps the step's tokens in VMEM and emits the cache rows as
outputs for a post-kernel scatter:

* compressed key of the group completed this step (`g_c`, one per request when T = 4):
  members = this step's raw index-key rows or the request's raw ring, bf16 mean, gemma
  norm,
  index RoPE at the group's first position; scored in place of the (stale) cached
  column;
* attention: the pool path reads only selected groups below `g_a = ctx // 4` (wholly
  older
  than the step); the <= 2 groups holding the step's tokens form a per-request "recent
  tile"
  of 8 slots (old members from the pool, the step's tokens FP8-quantised like the stock
  write), masked per row: causal, the row's partial group, and `g_a` only if complete
  for
  the row and selected;
* outputs: `qsa_kv_new` (stage A), `qsa_ck_new [requests, 128]`, `qsa_ring_new
  [requests, 8, 128]`.
"""

from __future__ import annotations

import functools

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..blocks.qsa_attn.block import MICRO, _unpack_kv, attend_row
from ..exec_spec import ExecSpec
from ..ops import BF16, F32, I32, rmsnorm, rnd
from ..spec import ExtraInput, ScratchReq, SmallField, StateDecl
from . import indexer as ix
from . import union_attn as ua
from .qsa_full import OFFSET, POOL, ROPE, SCALES
from .qsa_full_b import CK, GPK, IDX_N, IDX_W, MTAB, NVIS, TAIL, QSAFullIndexedBlock

CTXQ = "qsa_ctx"  # [requests] tokens before this step
RING = "qsa_ring"  # [requests, 8, 128] raw index-key ring (lane = position % 8)
ROPE_CK = (
    "qsa_rope_ck"  # [requests, 128] RoPE row at the completed group's first position
)
CK_NEW = "qsa_ck_new"
RING_NEW = "qsa_ring_new"
# Projections the post-kernel scatter feeds to the stock cache writers
# (`pooled_qsa.qsa_cache_writes`): the un-normed key and the raw index key per row.
KR_NEW = "qsa_kr_new"
UNION_STATS = (
    "qsa_union_stats"  # [UNION_STATS_REQS, 1, 128] int32 per QSA layer (`union_stats`)
)
# Fixed request capacity of the stats output (64-row buckets / T): its shape must not
# follow the
# bucket, or the torch op's fake output makes vLLM's trace branch on the token count
# (w06g/w07g:
# ShapeSpecializationError 's72 <= 4' at bucket 32).
UNION_STATS_REQS = 16
IR_NEW = "qsa_ir_new"
FP8_MAX = 448.0



_LANEPACK_LOGGED = False


def _log_lanepack_active(gpp: int) -> None:
    """VLLM_TPU_MK_QSA_UNION_LANEPACK self-check: one line, once per process on TP rank 0,
    when a kernel is traced with the packed union (trace time; no runtime cost)."""
    global _LANEPACK_LOGGED
    if _LANEPACK_LOGGED:
        return
    _LANEPACK_LOGGED = True
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        if get_tensor_model_parallel_rank() != 0:
            return
    except Exception:  # no TP group (CPU tests, tools): log
        pass
    from vllm_torchtpu.logger import init_logger

    init_logger(__name__).info("LANEPACK active gpp=%d", gpp)

def _scoped(name: str, level: int = 2):
    """Run the method under `ctx.stream.sc(name, level)` (VLLM_TPU_MK_SCOPES)."""

    def wrap(fn):
        @functools.wraps(fn)
        def inner(self, ctx, *a, **k):
            with ctx.stream.sc(name, level):
                return fn(self, ctx, *a, **k)

        return inner

    return wrap


def quant_fp8(x: jax.Array, scale: Any) -> jax.Array:
    """Stock cache write: `clip(x / scale, +-448) -> e4m3`, read back as f32 `*
    scale`.
    """
    return pool_units(x, scale) * scale


def pool_units(x: jax.Array, scale: Any) -> jax.Array:
    """The FP8 value the stock write would store (`clip(x / scale) -> e4m3`), as f32:
    the units
    `attend_row` reads pool tiles in (it applies the scales itself)."""
    return jnp.clip(x / scale, -FP8_MAX, FP8_MAX).astype(jnp.float8_e4m3fn).astype(F32)


def pick_row(rows: jax.Array, idx: Any) -> jax.Array:
    """`rows[idx]` for a traced `idx` by a masked sum (exact: one nonzero term)."""
    sel = jax.lax.broadcasted_iota(I32, (rows.shape[0], 1), 0) == idx
    return jnp.sum(jnp.where(sel, rows, 0.0), axis=0, keepdims=True)


class QSAFullStepBlock(QSAFullIndexedBlock):
    """Stage B + the step's own tokens (compressed key, recent tile, cache-row
    outputs).
    """

    @_scoped("qsa_score")
    def _score_request(self, ctx: Any, *a: Any, **k: Any) -> Any:
        return super()._score_request(ctx, *a, **k)

    def smalls(self, g: Any, ex: ExecSpec) -> list[SmallField]:
        return super().smalls(g, ex) + [SmallField("qsa_ikn", ix.IDX_DIM)]

    def outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        nreq = ex.M // ex.T
        return (
            super().outputs(g, ex)
            + [
                StateDecl(CK_NEW, (nreq, 1, ix.IDX_DIM), BF16, "w"),
                StateDecl(RING_NEW, (nreq, 8, ix.IDX_DIM), BF16, "w"),
                StateDecl(KR_NEW, (ex.M, g.HD), BF16, "w"),
                StateDecl(IR_NEW, (ex.M, ix.IDX_DIM), BF16, "w"),
            ]
            + (
                [StateDecl(UNION_STATS, (UNION_STATS_REQS, 1, 128), I32, "w")]
                if self.union_stats
                else []
            )
        )

    def host_update_states(self, g, ex, pos, states, raw_outputs):  # noqa: ANN001
        out = super().host_update_states(g, ex, pos, states, raw_outputs)
        out.update({k: raw_outputs[k] for k in (CK_NEW, RING_NEW, KR_NEW, IR_NEW)})
        if self.union_stats:
            out[UNION_STATS] = raw_outputs[UNION_STATS]
        return out

    def extra_inputs(self, g: Any, ex: ExecSpec) -> list[ExtraInput]:
        nreq = ex.M // ex.T
        return super().extra_inputs(g, ex) + [
            ExtraInput(CTXQ, "smem", (nreq,), I32),
            ExtraInput(
                RING, "hbm", (self.meta_rows(ex), 8, ix.IDX_DIM), BF16
            ),  # per layer
            ExtraInput(ROPE_CK, "hbm", (nreq, 1, 128), F32),
        ]

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> list[ScratchReq]:
        nreq, f8 = ex.MB // ex.T, jnp.float8_e4m3fn
        return super().scratch(g, ex, plan) + [
            ScratchReq("qc_ring", pltpu.VMEM((8, ix.IDX_DIM), BF16)),
            ScratchReq("qc_rope", pltpu.VMEM((1, 128), F32)),
            ScratchReq("qc_ck", pltpu.VMEM((1, ix.IDX_DIM), BF16)),
            ScratchReq("qc_rnew", pltpu.VMEM((8, ix.IDX_DIM), BF16)),
            ScratchReq("qc_old", pltpu.VMEM((MICRO, 1, 4, 256), f8)),
            ScratchReq("qc_rk", pltpu.VMEM((nreq, 8, 256), F32)),
            ScratchReq("qc_rv", pltpu.VMEM((nreq, 8, 256), F32)),
            ScratchReq("qc_sga", pltpu.SMEM((ex.MB,), I32)),
            ScratchReq("qc_sem", pltpu.SemaphoreType.DMA((4,))),
            ScratchReq("qf_kr", pltpu.VMEM((ex.MB, g.HD), BF16)),
            ScratchReq("qc_ir", pltpu.VMEM((ex.MB, ix.IDX_DIM), BF16)),
            ScratchReq("qc_osem", pltpu.SemaphoreType.DMA((2,))),
        ] + ([
            # Request-major copies for the traced request index of `loop_requests`: a dynamic
            # `[q * T, (q + 1) * T)` sublane window of a row-major tile is not provably aligned
            # (Mosaic E2003, j-bfd3a7f8); a leading-dim index is.
            ScratchReq("qc_irq", pltpu.VMEM((nreq, ex.T, ix.IDX_DIM), F32)),
            ScratchReq("qc_kvq", pltpu.VMEM((nreq, ex.T, 2 * g.HD), F32)),
        ] if self.loop_requests else []) + ([
            # `xreq_loads`: the request loop's loads, banked by request parity (q mod 2).
            ScratchReq("qx_ring", pltpu.VMEM((2, 8, ix.IDX_DIM), BF16)),
            ScratchReq("qx_rope", pltpu.VMEM((2, 1, 128), F32)),
            ScratchReq("qx_old", pltpu.VMEM((2, MICRO, 1, 4, 256), f8)),
            ScratchReq("qx_sem", pltpu.SemaphoreType.DMA((6,))),
        ] if self.xreq_loads else []) + ([
            # `req_async`: the new group's output stores, banked by request parity (q mod 2).
            ScratchReq("qa_ck", pltpu.VMEM((2, 1, ix.IDX_DIM), BF16)),
            ScratchReq("qa_rnew", pltpu.VMEM((2, 8, ix.IDX_DIM), BF16)),
            ScratchReq("qa_sem", pltpu.SemaphoreType.DMA((4,))),
        ] if self.req_async else []) + (self._shard_scratch(ex) if self.shard_index else []) + (
            self._union_scratch(ex) if self.union_attn else []
        ) + (self._dcp_scratch(g, ex) if self.dcp > 1 else [])

    def _dcp_scratch(self, g: Any, ex: ExecSpec) -> list[ScratchReq]:
        """Main-KV DCP: the pair rank's q (`qd_q`), this rank's partial over its own heads
        (`qd_o` / `qd_l`) and the partial over the pair rank's heads sent to it (`qd_os` /
        `qd_ls`, received into `qd_or` / `qd_lr`); lse rows lane-broadcast `[nh, 128]`."""
        nh = self.nh(g, ex)
        o, lse = pltpu.VMEM((ex.MB, nh, g.HD), F32), pltpu.VMEM((ex.MB, nh, 128), F32)
        return [
            ScratchReq("qd_q", pltpu.VMEM((ex.MB, nh, g.HD), F32)),
            ScratchReq("qd_o", o), ScratchReq("qd_l", lse),
            ScratchReq("qd_os", o), ScratchReq("qd_ls", lse),
            ScratchReq("qd_or", o), ScratchReq("qd_lr", lse),
        ] + [
            ScratchReq(f"qd_{k}sem{i}", pltpu.SemaphoreType.DMA((1,)))
            for i in range(3) for k in "sr"
        ]

    def _dcp_owned(self, ctx: Any, pos: Any) -> Any:
        """Whether this rank owns token positions `pos` (main-KV DCP: `(pos // B) % 2`)."""
        B = ctx.extra[POOL].shape[1]
        return jax.lax.rem(pos // B, self.dcp) == jax.lax.rem(ctx.me, self.dcp)

    def _union_width(self, ex: ExecSpec) -> int:
        """Union slots per request: every row's whole selection, 128-lane aligned."""
        return -(-(ex.T * self.budget) // 128) * 128

    def _union_scratch(self, ex: ExecSpec) -> list[ScratchReq]:
        nreq, KU, S = ex.MB // ex.T, self._union_width(ex), self.max_groups // 128
        D = max(self.dcp, 1)
        out = [
            ScratchReq("qu_code", pltpu.VMEM((S, 128), I32)),
            # Under main-KV DCP one union per pair rank (its owned groups only) at lanes
            # [r * KU, (r + 1) * KU), counts in qu_n[q, r].
            ScratchReq("qu_id", pltpu.VMEM((nreq, 1, D * KU), I32)),
            ScratchReq("qu_cd", pltpu.VMEM((nreq, 1, D * KU), I32)),
            ScratchReq("qu_n", pltpu.SMEM((nreq, D), I32)),
            # `xreq_prefetch`: two banks (request q's ids, request q+1's staged).
            # `req_async` 2: two banks (request q's ids, request q+1's staged).
            ScratchReq("qu_row", pltpu.SMEM(
                (2 if self.xreq_prefetch or self.req_async_bank else 1, KU + 128), I32)),
            ScratchReq("qu_sem", pltpu.SemaphoreType.DMA((3,))),
            ScratchReq("qu_st", pltpu.VMEM((1, 128), I32)),
        ]
        if self.union_phys:
            out += [
                ScratchReq("qu_tab", pltpu.VMEM((1, self.pps), I32)),
                ScratchReq("qu_pv", pltpu.VMEM((1, KU), I32)),
            ]
        if self.shard_index and not self.xmerge:
            npr, n = self._shard_rows(ex)[0], max(ex.R - 1, 1)
            out += [
                ScratchReq("qgu_send", pltpu.VMEM((npr, 1, 2 * D * KU + 128), I32)),
                ScratchReq("qgu_recv", pltpu.VMEM((self._runs(ex), npr, 1, 2 * D * KU + 128), I32)),
                ScratchReq("qgu_ssem", pltpu.SemaphoreType.DMA((n,))),
                ScratchReq("qgu_rsem", pltpu.SemaphoreType.DMA((n,))),
            ]
        return out

    def _shard_rows(self, ex: ExecSpec) -> tuple[int, int, int]:
        """(requests per rank, rows per rank, exchange row width) of the sharded
        indexer."""
        nreq = ex.MB // ex.T
        npr = -(-nreq // self._runs(ex))
        return npr, npr * ex.T, -(-(self.budget + 2) // 128) * 128

    def _runs(self, ex: ExecSpec) -> int:
        """Request runs of the sharded indexer: one per rank, or per pair (`compressed_dcp`:
        both ranks of a pair score the same requests, each over its own groups)."""
        return ex.R // self.dcp if self.compressed_dcp else ex.R

    def _run(self, ctx: Any) -> Any:
        """This rank's run of requests (`_runs`)."""
        return ctx.me // self.dcp if self.compressed_dcp else ctx.me

    def _balanced(self, ex: ExecSpec) -> bool:
        """`nvis_balance` in this bucket: two requests per run (more requests than runs)."""
        return self.nvis_balance and self._shard_rows(ex)[0] == 2

    def _balance_tables(self, ctx: Any) -> None:
        """`nvis_balance`: this layer's request -> run assignment (scalar, SMEM). Requests are
        ranked by visible groups (their last row's `n_vis`, descending; ties by index), the
        ranking is padded with empty places to `2 * runs`, and place `k < runs` shares run `k`
        with place `2 * runs - 1 - k` (the largest with the smallest). Per request `q`:
        `qb_run` its run, `qb_first` 1 if it is the run's first request in loop order,
        `qb_nxt` the run's next request (-1: the run's last); `qb_slot[run * 2 + j]` the run's
        j-th request (-1: empty). Every rank computes the same tables from the same inputs."""
        ex, scr, T = ctx.ex, ctx.scr, ctx.ex.T
        nreq, R = ex.MB // T, self._runs(ex)
        P = 2 * R
        nv = [ctx.extra[NVIS][q * T + T - 1] for q in range(nreq)]
        for k in range(P):
            scr["qb_inv"][k] = jnp.int32(-1)
            scr["qb_slot"][k] = jnp.int32(-1)
        for q in range(nreq):
            k = jnp.int32(0)
            for p in range(nreq):
                if p != q:
                    k = jnp.where((nv[p] >= nv[q]) if p < q else (nv[p] > nv[q]), k + 1, k)
            scr["qb_rk"][q] = k
            scr["qb_inv"][k] = jnp.int32(q)
        for q in range(nreq):
            k = scr["qb_rk"][q]
            run = jnp.where(k < R, k, P - 1 - k)
            mate = scr["qb_inv"][P - 1 - k]
            first = (mate < 0) | (q < mate)
            scr["qb_run"][q] = run
            scr["qb_first"][q] = first.astype(I32)
            scr["qb_nxt"][q] = jnp.where(first & (mate >= 0), mate, -1)
            scr["qb_slot"][run * 2 + jnp.where(first, 0, 1)] = jnp.int32(q)

    def _gather_runs(
        self, ctx: Any, send: Any, recv: Any, ssem: Any, rsem: Any, own_rank: bool = False
    ) -> None:
        """All-gather every run's `send` into `recv [runs, ...]` (slot = run). Under
        `compressed_dcp` both ranks of a pair hold their run's results; only pair rank 0 sends
        them, to every rank outside the pair, so `recv` holds one slot per run. `own_rank`:
        each pair rank holds the part for its pair rank (its own union) and sends it to the
        ranks of the same pair rank."""
        ex, me = ctx.ex, ctx.me
        recv[self._run(ctx)] = send[...]
        if ex.R == 1:
            return
        if not self.compressed_dcp:
            ctx.comm.exchange(ctx.comm.group_copies(lambda off: send, recv.at[me], ssem, rsem, ex.R))
            return
        D = self.dcp
        descs = {
            off: pltpu.make_async_remote_copy(
                src_ref=send, dst_ref=recv.at[me // D], send_sem=ssem.at[off - 1],
                recv_sem=rsem.at[off - 1], device_id=(me ^ off,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            for off in range(D, ex.R)  # me ^ off with off < D is the pair itself
        }
        if own_rank:
            mine = [d for off, d in descs.items() if off % D == 0]
            for d in mine:
                d.start()
            for d in mine:
                d.wait_recv()
            for d in mine:
                d.wait_send()
            return
        lead = jax.lax.rem(me, D) == 0

        @pl.when(lead)
        def _():
            for d in descs.values():
                d.start()

        for off, d in descs.items():
            pl.when(jax.lax.rem(me ^ off, D) == 0)(d.wait_recv)

        @pl.when(lead)
        def _():
            for d in descs.values():
                d.wait_send()

    def _cdcp_credit(self, ctx: Any) -> None:
        """Tell the pair rank its next score half may land here (`_cdcp_swap`)."""
        pl.semaphore_signal(
            ctx.scr["qx_credit"].at[0], 1, device_id=(ctx.me ^ 1,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    @_scoped("qsa_cdcp_swap")
    def _cdcp_swap(self, ctx: Any, q: Any) -> None:
        """Swap request `q`'s score halves with the pair rank: this rank's chunks (scored by
        `_score_request` at `c_off = rank * NC / 2`) go to the same chunks of the peer's `qi_s`,
        the peer's to ours. Both send the chunks rank 0's half of the last row's visible groups
        spans (rank 0 owns at least as many as rank 1); slots past a rank's own count hold stale
        scores of groups >= n_vis, masked by the selection. A send waits for the peer's credit
        (`_cdcp_credit`: its previous request's selection has read the half it overwrites)."""
        ex, scr, T = ctx.ex, ctx.scr, ctx.ex.T
        NCh, CG = self.n_chunks // self.dcp, self.chunk_pages * GPK
        nv0 = self._cdcp_local_nv(ctx, ctx.extra[NVIS][q * T + T - 1], 0)
        n_x = jnp.clip((nv0 + CG - 1) // CG, 1, NCh)
        off = jax.lax.rem(ctx.me, self.dcp) * NCh

        def desc(c: Any) -> Any:
            ref = scr["qi_s"].at[pl.ds(0, T), off + c]
            return pltpu.make_async_remote_copy(
                src_ref=ref, dst_ref=ref, send_sem=scr["qx_ssem"].at[0],
                recv_sem=scr["qx_rsem"].at[0], device_id=(ctx.me ^ 1,),
                device_id_type=pl.DeviceIdType.MESH,
            )

        def each(fn: Any) -> None:
            def body(c: Any, carry: Any) -> Any:
                fn(desc(c))
                return carry

            jax.lax.fori_loop(0, n_x, body, 0)

        pl.semaphore_wait(scr["qx_credit"].at[0], 1)
        each(lambda d: d.start())
        each(lambda d: d.wait_recv())
        each(lambda d: d.wait_send())

    # ------------------------------------------- visible-chunk selection (`select_nvis`)
    def _nvis_route(self, nv: Any, cut: int, ranged: Any, full: Any) -> Any:
        """`ranged()` when `nv` (scalar visible groups) <= `cut`, else `full()` (the original
        full-row scan): both exact, so the choice only moves time. `cut` >= max_groups:
        always ranged (no branch)."""
        if cut >= self.max_groups:
            return ranged()
        return jax.lax.cond(nv <= cut, ranged, full)

    def _thr_tiers(self, S: int, RS: int) -> tuple:
        """Static score-row tiers of VLLM_TPU_MK_QSA_THR_FAST "r": S/2, 3S/4, S (served
        256 / 384 / 512), each a whole number of score chunks (`RS` rows) and of 128-row
        prefix chunks when S is (`ix.exclusive_prefix`)."""
        ok = lambda x: x % RS == 0 and (x % 128 == 0 or S % 128)  # noqa: E731
        lows = (S // 2,) if self.tier_two else (S // 2, 3 * S // 4)
        return tuple(x for x in lows if x and ok(x)) + (S,)

    def _by_tier(self, nv: Any, tiers: tuple, fn: Any) -> Any:
        """`fn(Sx)` for the smallest tier `Sx` with `Sx * 128 >= nv` (scalar visible
        groups): one static scan per tier, chosen by nested conds."""
        if len(tiers) == 1:
            return fn(tiers[0])
        return jax.lax.cond(nv <= tiers[0] * 128, lambda: fn(tiers[0]),
                            lambda: self._by_tier(nv, tiers[1:], fn))

    def _nvis_plan(self, ctx: Any, nv: Any, CH: int) -> tuple:
        """(count, base row of chunk i) of the `CH`-row score chunks holding groups < `nv`
        (scalar), in score-position order: under `compressed_dcp` rank 0's half's visible
        prefix, then rank 1's."""
        S, span = self.max_groups // 128, CH * 128
        nv = jnp.clip(nv, 0, self.max_groups)  # hardening: out-of-range n_vis cannot widen m
        if not self.compressed_dcp:
            return jnp.minimum((nv + span - 1) // span, S // CH), lambda i: i * CH
        Sh = S // self.dcp
        m0, m1 = (
            jnp.minimum((self._cdcp_local_nv(ctx, nv, rk) + span - 1) // span, Sh // CH)
            for rk in range(self.dcp)
        )
        return m0 + m1, lambda i: jnp.where(i < m0, i * CH, Sh + (i - m0) * CH)

    def _nvis_slab(self, ctx: Any, r0: Any, n: int, base: Any, CH: int) -> Any:
        """`qi_s` rows `[r0, r0 + n)`, score rows `[base, base + CH)` as `pl.ds` indices."""
        rs = self.max_groups // 128 // self.n_chunks
        return (pl.ds(r0, n), pl.ds(base // rs, CH // rs))

    def _nvis_ids(self, ctx: Any, shape: tuple, base: Any) -> Any:
        """Group ids of score rows `[base, base + shape[-2])` (`_cdcp_gid` order)."""
        d = len(shape)
        p = (jax.lax.broadcasted_iota(I32, shape, d - 2) + base) * 128 + (
            jax.lax.broadcasted_iota(I32, shape, d - 1)
        )
        return self._cdcp_gid(ctx, p) if self.compressed_dcp else p

    def _nvis_thresholds(self, ctx: Any, r0: Any, n: int, nv: Any, nv_max: Any) -> tuple:
        """`ix.thresholds` of the `n` rows at `qi_s[r0 …]` (per-row `nv [n, 1, 1]`) over the
        chunks holding groups < `nv_max` (scalar); none when no row has more than `budget`
        visible groups (thresholds 0 / max_groups whatever the counts). Masks those chunks'
        scores in place first (-1 past each row's `n_vis`; the compaction masks again)."""
        CH, K, G = self.nvis_rows[0], self.budget, self.max_groups
        rs = self.max_groups // 128 // self.n_chunks
        m, base_of = self._nvis_plan(ctx, nv_max, CH)
        m = jnp.where(nv_max > K, m, 0)
        qs = ctx.scr["qi_s"]
        shape = (n, CH, 128)

        def mask(c: Any, carry: Any) -> Any:
            base = base_of(c)
            ix_ = self._nvis_slab(ctx, r0, n, base, CH)
            sc = pltpu.bitcast(qs[ix_].reshape(shape), I32)
            b = jnp.where(self._nvis_ids(ctx, shape, base) < nv, sc, -1)
            qs[ix_] = pltpu.bitcast(b, F32).reshape(n, CH // rs, rs, 128)
            return carry

        jax.lax.fori_loop(0, m, mask, 0)

        def bits(c: Any) -> Any:
            ix_ = self._nvis_slab(ctx, r0, n, base_of(c), CH)
            return pltpu.bitcast(qs[ix_].reshape(shape), I32)

        return ix.thresholds_chunks(
            bits, lambda c: self._nvis_ids(ctx, shape, base_of(c)), m, nv, K, G, shape,
            digit=self.nvis_digit,
        )

    def _nvis_row(self, ctx: Any, r: Any, lo: Any, s0: Any, ga: Any, consts: tuple) -> tuple:
        """`row`'s selection and compaction of row `r` over the score rows holding its
        visible groups, two `CM`-row halves per compaction chunk (under `compressed_dcp` the
        two halves may sit in the two ranks' score halves). Returns (ids `[1, K]` int32
        positions, g_a selected, kept)."""
        scr, K, CM = ctx.scr, self.budget, self.nvis_rows[1]
        rs = self.max_groups // 128 // self.n_chunks
        nvs = ctx.extra[NVIS][r]
        m, base_of = self._nvis_plan(ctx, nvs, CM)
        th, tm = scr["qi_th"][r][:, :1], scr["qi_tm"][r][:, :1]
        ov = scr["qi_nv"][r][:, :1] > K
        qs = scr["qi_s"]

        def it(i: Any, carry: Any) -> Any:
            acc, tot, sga, kept = carry
            sels, bases = [], []
            for k in range(2):
                c = 2 * i + k
                base = jnp.where(c < m, base_of(c), 0)
                lid = self._nvis_ids(ctx, (CM, 128), base)
                ix_ = (r - lo + s0, pl.ds(base // rs, CM // rs))
                sc = pltpu.bitcast(qs[ix_].reshape(CM, 128), I32)
                b = jnp.where(lid < jnp.where(c < m, nvs, 0), sc, -1)  # past m: none
                sel = (ov & ((b > th) | ((b == th) & (lid <= tm)))) | (~ov & (b >= 0))
                # Elementwise counts, reduced once after the loop (no per-chunk vector ->
                # scalar moves).
                sga = sga + (sel & (lid == ga)).astype(I32)
                s2 = sel & (lid < ga)
                kept = kept + s2.astype(I32)
                if self.union_attn:  # row bit r - lo of every pool-path group it selects
                    rows = pl.ds(pl.multiple_of(base, 8), CM)
                    scr["qu_code"][rows] = scr["qu_code"][rows] | jnp.where(
                        s2, jnp.left_shift(1, r - lo), 0
                    )
                sels.append(s2 & (b >= 0))
                bases.append(base)

            def row_of(rj: Any) -> Any:
                return jnp.where(rj < CM, bases[0] + rj, bases[1] + rj - CM)

            part, tot = ix.compact_chunk(jnp.concatenate(sels, axis=0), tot, row_of, consts)
            return acc + part, tot, sga, kept

        z = jnp.zeros((CM, 128), I32)
        acc, _, sga, kept = jax.lax.fori_loop(
            0, (m + 1) // 2, it,
            (jnp.zeros((1, K), F32), jnp.zeros((1, 1), F32), z, z),
        )
        return acc.astype(I32), jnp.sum(sga), jnp.sum(kept)

    def _nvis_union(self, ctx: Any, lo: Any, r: Any, KU: int, D: int, gpp: int,
                    nv_max: Any) -> None:
        """The `qsa_ucompact` union of request `lo // T` for pair rank `r` (its owned groups
        under main-KV DCP) over the score rows holding groups < `nv_max` (no row's union code
        is set past them), two `CM`-row halves per chunk; slot windows past the union are
        skipped."""
        scr, T, CM = ctx.scr, ctx.ex.T, self.nvis_rows[1]
        q = lo // T
        m, base_of = self._nvis_plan(ctx, nv_max, CM)
        at = pl.multiple_of(r * KU, 128) if self.compressed_dcp else r * KU
        lm = self.lane_major_select
        consts = ua.union_chunk_consts(2 * CM, KU, lm)
        scr["qu_id"][q, :, pl.ds(at, KU)] = jnp.zeros((1, KU), I32)
        scr["qu_cd"][q, :, pl.ds(at, KU)] = jnp.zeros((1, KU), I32)

        def store(j0: int, ids: Any, codes: Any) -> None:
            W = ids.shape[1]
            for k, v in (("qu_id", ids), ("qu_cd", codes)):
                ref = scr[k].at[q, :, pl.ds(at + j0, W)]
                ref[...] = ref[...] + v

        def it(i: Any, tot: Any) -> Any:
            codes, bases = [], []
            for k in range(2):
                c = 2 * i + k
                base = jnp.where(c < m, base_of(c), 0)
                cd = scr["qu_code"][pl.ds(pl.multiple_of(base, 8), CM)]
                keep = jnp.where(c < m, 1, 0)
                if D > 1:  # pair rank r's groups (under `compressed_dcp`: its half)
                    gidx = self._nvis_ids(ctx, (CM, 128), base)
                    cd = jnp.where(jax.lax.rem(gidx // gpp, D) == r, cd, 0)
                codes.append(cd * keep)
                bases.append(base)

            def row_of(rj: Any) -> Any:
                return jnp.where(rj < CM, bases[0] + rj, bases[1] + rj - CM)

            return ua.compact_union_chunk(
                jnp.concatenate(codes, axis=0), tot, row_of, KU, consts, store, lane_major=lm
            )

        H = CM // 2
        rows_own = gpp == 128  # served 512-token blocks: rank r owns whole rows 2p + r
        if (self.union_lanepack and D == 2 and not self.compressed_dcp and isinstance(r, int)
                and (rows_own or 128 % (2 * gpp) == 0) and H % 8 == 0):
            _log_lanepack_active(gpp)
            # Pack: rank r's owned groups of code rows 2p and 2p + 1 form packed row p in
            # ascending id order. gpp = 128: rank r owns all of row 2p + r (a strided row read);
            # gpp | 64: it owns lanes l with (l // gpp) % 2 == r of every row (row 2p's at
            # packed lanes [0, 64), row 2p + 1's at [64, 128)). Four CM-row chunks pack into
            # one 2 * CM-row chunk: half the iterations, same ids, codes and order.
            def itp(i: Any, tot: Any) -> Any:
                codes, bases = [], []
                for k in range(4):
                    c = 4 * i + k
                    base = jnp.where(c < m, base_of(c), 0)
                    b8 = pl.multiple_of(base, 8)
                    if rows_own:
                        pk = scr["qu_code"][pl.ds(b8 + r, H, stride=2)]
                    else:
                        ev = scr["qu_code"][pl.ds(b8, H, stride=2)]
                        od = scr["qu_code"][pl.ds(b8 + 1, H, stride=2)]
                        pk = ua.lane_pack(ev, od, r, gpp, pltpu.roll)
                    codes.append(pk * jnp.where(c < m, 1, 0))
                    bases.append(base)

                def row_of(rj: Any) -> Any:
                    b = bases[3]
                    for k in (2, 1, 0):
                        b = jnp.where(rj < (k + 1) * H, bases[k], b)
                    return b + 2 * jax.lax.rem(rj, H) + (r if rows_own else 0)

                def id_of(rj: Any, lj: Any) -> Any:
                    return ua.lane_pack_id(row_of(rj), lj, r, gpp)

                return ua.compact_union_chunk(
                    jnp.concatenate(codes, axis=0), tot, row_of, KU, consts, store,
                    None if rows_own else id_of, lane_major=lm,
                )

            n = jax.lax.fori_loop(0, (m + 3) // 4, itp, jnp.zeros((), I32))
        else:
            n = jax.lax.fori_loop(0, (m + 1) // 2, it, jnp.zeros((), I32))
        if self.compressed_dcp:
            ref = scr["qu_id"].at[q, :, pl.ds(at, KU)]
            ref[...] = self._cdcp_gid(ctx, ref[...])
        scr["qu_n"][q, r] = n

    def _xmerge_width(self, ex: ExecSpec) -> tuple[int, int]:
        """`xmerge`: (union payload offset, row width) of a request's merged exchange row:
        its `T` selection rows at lanes `[t * W, (t + 1) * W)` (`_shard_rows` width), then
        its union ids, codes (`D * KU` lanes each) and counts (128 lanes)."""
        WU = max(self.dcp, 1) * self._union_width(ex)
        u0 = ex.T * self._shard_rows(ex)[2]
        return u0, u0 + 2 * WU + 128

    def _shard_scratch(self, ex: ExecSpec) -> list[ScratchReq]:
        _, rows, W = self._shard_rows(ex)
        n = max(ex.R - 1, 1)
        if self.xmerge:
            npr, Wm = self._shard_rows(ex)[0], self._xmerge_width(ex)[1]
            nreq, P = ex.MB // ex.T, 2 * self._runs(ex)
            return [
                ScratchReq("qm_send", pltpu.VMEM((npr, 1, Wm), I32)),
                ScratchReq("qm_recv", pltpu.VMEM((self._runs(ex), npr, 1, Wm), I32)),
                ScratchReq("qm_ssem", pltpu.SemaphoreType.DMA((n,))),
                ScratchReq("qm_rsem", pltpu.SemaphoreType.DMA((n,))),
            ] + ([  # `nvis_balance`: the request -> run tables (`_balance_tables`)
                ScratchReq(name, pltpu.SMEM((size,), I32))
                for name, size in (("qb_rk", nreq), ("qb_inv", P), ("qb_run", nreq),
                                   ("qb_first", nreq), ("qb_nxt", nreq), ("qb_slot", P))
            ] if self._balanced(ex) else [])
        return [
            ScratchReq("qg_send", pltpu.VMEM((rows, 1, W), I32)),
            ScratchReq("qg_recv", pltpu.VMEM((self._runs(ex), rows, 1, W), I32)),
            ScratchReq("qg_ssem", pltpu.SemaphoreType.DMA((n,))),
            ScratchReq("qg_rsem", pltpu.SemaphoreType.DMA((n,))),
        ]

    # ------------------------------------------------------------ in kernel
    @_scoped("qsa_newgrp")
    def _new_group(
        self, ctx: Any, l: Any, q: int, raw_q: jax.Array, bank: Any = None, sbank: Any = None
    ) -> jax.Array:
        """Compressed key of request `q`'s group completed this step; writes its outputs.
        `bank`: its loads were issued ahead (`_xreq_loads`) into that bank. `sbank`
        (`req_async`): its stores go out of that bank and are left in flight
        (`_wait_new_group_stores`)."""
        ex, scr, g, T = ctx.ex, ctx.scr, ctx.g, ctx.ex.T
        c0 = ctx.extra[CTXQ][q]
        if bank is not None:
            for cp in self._xreq_loads(ctx, l, q, bank)[:2]:
                cp.wait()
            ring = scr["qx_ring"][bank].astype(F32)
            rope = scr["qx_rope"][bank]
            return self._new_group_body(ctx, l, q, raw_q, c0, ring, rope, sbank)
        loads = [
            pltpu.make_async_copy(src, dst, scr["qc_sem"].at[i])
            for src, dst, i in (
                (ctx.extra[RING].at[self.row_base(ctx, l) + q], scr["qc_ring"], 0),
                (ctx.extra[ROPE_CK].at[q], scr["qc_rope"], 1),
            )
        ]
        for cp in loads:
            cp.start()
            if not self.async_dma:
                cp.wait()
        if self.async_dma:
            for cp in loads:
                cp.wait()
        return self._new_group_body(
            ctx, l, q, raw_q, c0, scr["qc_ring"][...].astype(F32), None, sbank
        )

    def _new_group_stores(self, ctx: Any, l: Any, q: Any, sbank: Any) -> list:
        """`req_async`: request `q`'s two output stores out of bank `sbank`."""
        scr, o = ctx.scr, ctx.stream.sp.block_ord(l, self.name)
        return [
            pltpu.make_async_copy(src.at[sbank], ctx.O[dst].at[o, q], scr["qa_sem"].at[2 * sbank + i])
            for i, (src, dst) in enumerate(((scr["qa_ck"], CK_NEW), (scr["qa_rnew"], RING_NEW)))
        ]

    def _wait_new_group_stores(self, ctx: Any, l: Any, sbank: Any) -> None:
        """`req_async`: wait bank `sbank`'s two stores (the wait needs only the semaphore and
        the byte count; request 0's descriptor has both)."""
        for cp in self._new_group_stores(ctx, l, 0, sbank):
            cp.wait()

    def _drain_new_group_stores(self, ctx: Any, l: Any) -> None:
        """`req_async`: each used bank's last stores, still in flight after the request loop
        (waited at the layer's end, before the next layer's loop reuses the banks)."""
        n_loop = self._n_live(ctx) if self.skip_idle else ctx.ex.MB // ctx.ex.T
        for b in range(2):
            pl.when(n_loop > b)(lambda b=b: self._wait_new_group_stores(ctx, l, b))

    def _new_group_body(
        self, ctx: Any, l: Any, q: Any, raw_q: jax.Array, c0: Any, ring: jax.Array, rope: Any,
        sbank: Any = None,
    ) -> jax.Array:
        ex, scr, g, T = ctx.ex, ctx.scr, ctx.g, ctx.ex.T
        p3 = c0 + jax.lax.rem(3 - jax.lax.rem(c0, 4) + 4, 4)  # first position = 3 mod 4
        j0 = p3 - 3
        acc: Any = None
        for j in range(
            4
        ):  # sequential f32 sum: one fixed order, mirrored by the reference
            pos = j0 + j
            cur = pick_row(raw_q, pos - c0)
            old = pick_row(ring, jax.lax.rem(pos, 8))
            m = jnp.where(pos >= c0, cur, old)
            acc = m if acc is None else acc + m
        pooled = rnd(acc * 0.25)
        if rope is None:
            rope = scr["qc_rope"][...]
        cos, sin = rope[:, : self.rot // 2], rope[:, self.rot // 2 : self.rot]
        ck = ix.index_rope(
            rmsnorm(pooled, ctx.sm(ctx._slot, "qsa_ikn"), g.eps, zero_centered=True),
            cos,
            sin,
            self.rot,
            pltpu.bitcast,
        )
        scr["qc_ck"][...] = ck.astype(BF16)
        # Ring: lane l keeps the newest position = l (mod 8) below end = ctx + T.
        end = c0 + T
        lane = jax.lax.broadcasted_iota(I32, (8, 1), 0)
        p_l = end - 8 + jax.lax.rem(lane - jax.lax.rem(end - 8, 8) + 16, 8)
        cur_rows = jnp.zeros((8, ix.IDX_DIM), F32)
        for t in range(T):
            cur_rows = jnp.where(p_l == c0 + t, raw_q[t : t + 1], cur_rows)
        rnew = jnp.where(p_l >= c0, cur_rows, ring).astype(BF16)
        if sbank is not None:  # `req_async`: the bank's stores of request q - 2 first
            pl.when(q >= 2)(lambda: self._wait_new_group_stores(ctx, l, sbank))
            scr["qa_ck"][sbank] = ck.astype(BF16)
            scr["qa_rnew"][sbank] = rnew
            for cp in self._new_group_stores(ctx, l, q, sbank):
                cp.start()
            return ck, p3 // 4
        scr["qc_ck"][...] = ck.astype(BF16)
        scr["qc_rnew"][...] = rnew
        o = ctx.stream.sp.block_ord(l, self.name)
        stores = [
            pltpu.make_async_copy(src, ctx.O[dst].at[o, q], scr["qc_sem"].at[2 + i])
            if self.async_dma
            else pltpu.make_async_copy(src, ctx.O[dst].at[o, q], scr["qc_sem"].at[2])
            for i, (src, dst) in enumerate(((scr["qc_ck"], CK_NEW), (scr["qc_rnew"], RING_NEW)))
        ]
        for cp in stores:
            cp.start()
            if not self.async_dma:
                cp.wait()
        if self.async_dma:
            for cp in stores:
                cp.wait()
        return ck, p3 // 4

    @_scoped("qsa_recent")
    def _recent_micro(self, ctx: Any, l: Any, q: Any) -> Any:
        """Micro-page of request `q`'s group `g_a` (its recent tile's old members)."""
        ga = ctx.extra[CTXQ][q] // 4
        per_page = ctx.extra[POOL].shape[1] // MICRO
        shift = per_page.bit_length() - 1
        cshift = shift + self.dcp.bit_length() - 1
        qb = self.tab_base(ctx, l) + q
        return self.tables(ctx)[1][qb, ga >> cshift] * per_page + (ga & (per_page - 1))

    def _xreq_loads(self, ctx: Any, l: Any, q: Any, bank: Any) -> list:
        """`xreq_loads`: request `q`'s loads of the request loop (raw ring row, RoPE row for
        `_new_group`; `g_a`'s pool rows for `_recent_tile`) into `bank`."""
        scr = ctx.scr
        per_page = ctx.extra[POOL].shape[1] // MICRO
        shift = per_page.bit_length() - 1
        micro = self._recent_micro(ctx, l, q)
        return [
            pltpu.make_async_copy(src, dst.at[bank], scr["qx_sem"].at[3 * bank + i])
            for i, (src, dst) in enumerate((
                (ctx.extra[RING].at[self.row_base(ctx, l) + q], scr["qx_ring"]),
                (ctx.extra[ROPE_CK].at[q], scr["qx_rope"]),
                (ctx.extra[POOL].at[
                    micro >> shift, pl.ds((micro & (per_page - 1)) * MICRO, MICRO)
                ], scr["qx_old"]),
            ))
        ]

    def _recent_tile(self, ctx: Any, l: Any, q: int, bank: Any = None) -> None:
        """K / V of the 8 positions from `4 (ctx // 4)` in pool units (FP8 values, unscaled, as
        `attend_row`'s `recent` expects): old members from the pool, the step's tokens
        FP8-quantised from `qf_kv` (stock write-then-read)."""
        ex, scr, T = ctx.ex, ctx.scr, ctx.ex.T
        c0 = ctx.extra[CTXQ][q]
        ga = c0 // 4
        per_page = ctx.extra[POOL].shape[1] // MICRO
        shift = per_page.bit_length() - 1
        cshift = shift + self.dcp.bit_length() - 1  # DCP: a column covers `dcp * B` tokens
        qb = self.tab_base(ctx, l) + q
        micro = self.tables(ctx)[1][qb, ga >> cshift] * per_page + (ga & (per_page - 1))
        cp = pltpu.make_async_copy(
            ctx.extra[POOL].at[
                micro >> shift, pl.ds((micro & (per_page - 1)) * MICRO, MICRO)
            ],
            scr["qc_old"],
            scr["qc_sem"].at[0],
        )
        old = scr["qc_old"]
        if bank is None:
            cp.start()
            cp.wait()
        else:  # issued ahead (`_xreq_loads`)
            self._xreq_loads(ctx, l, q, bank)[2].wait()
            old = scr["qx_old"].at[bank]
        sk, sv = ctx.extra[SCALES][0], ctx.extra[SCALES][1]
        k8, v8 = _unpack_kv(old, self.offset(ctx, l), ex.interpret)
        pad = jnp.zeros((4, 256), F32)
        k_old = jnp.concatenate([k8.astype(F32), pad], axis=0)
        v_old = jnp.concatenate([v8.astype(F32), pad], axis=0)
        if self.loop_requests:
            kv = scr["qc_kvq"][q]
        else:
            kv = scr["qf_kv"][q * T : (q + 1) * T].astype(F32)
        k_cur, v_cur = pool_units(kv[:, :256], sk), pool_units(kv[:, 256:], sv)
        pos = 4 * ga + jax.lax.broadcasted_iota(I32, (8, 1), 0)
        k_t = jnp.where(pos < c0, k_old, 0.0)
        v_t = jnp.where(pos < c0, v_old, 0.0)
        for t in range(T):
            k_t = jnp.where(pos == c0 + t, k_cur[t : t + 1], k_t)
            v_t = jnp.where(pos == c0 + t, v_cur[t : t + 1], v_t)
        if self.dcp > 1:  # the pair rank's slots: its pool rows / tokens, scored there
            own = self._dcp_owned(ctx, pos)
            k_t, v_t = jnp.where(own, k_t, 0.0), jnp.where(own, v_t, 0.0)
        scr["qc_rk"][q] = k_t
        scr["qc_rv"][q] = v_t

    def _select(self, ctx: Any, l: Any, slot: Any) -> None:  # noqa: E741
        ex, scr, g = ctx.ex, ctx.scr, ctx.g
        MB, T, K, G = ex.MB, ex.T, self.budget, self.max_groups
        S, rot = G // 128, self.rot
        ctx._slot = slot
        with ctx.stream.sc("qsa_idx_proj", 2), ctx.stream.segment(lazy=False):
            yi = rnd(ctx.stream.dmatmul(IDX_W, scr["xb"], IDX_N))
        cos = scr["qf_rope"][:, : rot // 2]
        sin = scr["qf_rope"][:, rot // 2 : rot]
        iq = ix.index_query(
            yi, ctx.sm(slot, "qsa_iqn"), cos, sin, rot, g.eps, pltpu.bitcast
        )
        raw = yi[:, 4 * ix.IDX_DIM : 5 * ix.IDX_DIM]
        scr["qc_ir"][...] = raw.astype(BF16)
        o = ctx.stream.sp.block_ord(l, self.name)
        outs = [
            pltpu.make_async_copy(
                scr[src], ctx.O[dst].at[o, pl.ds(0, MB)], scr["qc_osem"].at[i]
            )
            for i, (src, dst) in enumerate((("qf_kr", KR_NEW), ("qc_ir", IR_NEW)))
        ]
        for c in outs:
            c.start()
        nreq = MB // T
        for q in range(nreq):
            for t in range(T):
                for h in range(ix.IDX_HEADS):
                    i = t * ix.IDX_HEADS + h
                    scr["qi_q"][q, i : i + 1, :] = iq[h][q * T + t : q * T + t + 1]
        NC = self.n_chunks
        gid = (
            jax.lax.broadcasted_iota(I32, (T, NC, S // NC, 128), 1) * (G // NC)
            + jax.lax.broadcasted_iota(I32, (T, NC, S // NC, 128), 2) * 128
            + jax.lax.broadcasted_iota(I32, (T, NC, S // NC, 128), 3)
        )
        cdcp = self.compressed_dcp
        if cdcp:  # score position -> group id (`_cdcp_gid`)
            gid = self._cdcp_gid(ctx, gid)

        def select_rows(lo: Any, n: int, s0: Any) -> None:
            """Thresholds and compaction of rows `[lo, lo + n)`; scores at
            `qi_s[s0 …]`.
            """
            rid = jax.lax.broadcasted_iota(I32, (n, 1, 1), 0) + lo
            nv = jnp.zeros((n, 1, 1), I32)
            for i in range(n):  # `lo` may be traced (`loop_requests`)
                nv = jnp.where(rid == lo + i, ctx.extra[NVIS][lo + i], nv)
            ids = (
                jax.lax.broadcasted_iota(I32, (n, S, 128), 1) * 128
                + jax.lax.broadcasted_iota(I32, (n, S, 128), 2)
            )
            if cdcp:
                ids = self._cdcp_gid(ctx, ids)

            def nv_max(r0: Any, k: int) -> Any:  # scalar max `n_vis` of rows [r0, r0 + k)
                v = ctx.extra[NVIS][r0]
                for i in range(1, k):
                    v = jnp.maximum(v, ctx.extra[NVIS][r0 + i])
                return v

            RS = S // NC
            # VLLM_TPU_MK_QSA_THR_FAST "r": static row tiers (not under compressed_dcp, whose
            # score positions are not in group order).
            tiers_all = self._thr_tiers(S, RS) if not cdcp else ()
            tiers = tiers_all if self.thr_tiers else ()
            ctiers = tiers_all if self.cmp_tiers else ()

            def thr_full(nvm: Any) -> Any:
                """`ix.thresholds` of rows [lo, lo + n) over all S score rows, or (tiers) over
                the smallest tier holding `nvm` (scalar max n_vis) groups: the rows past it
                are invisible (-1) and count nothing."""
                def scan(Sx: int) -> Any:
                    sc = pltpu.bitcast(
                        scr["qi_s"][pl.ds(s0, n), pl.ds(0, Sx // RS)].reshape(n, Sx, 128), I32
                    )
                    idx = ids[:, :Sx]
                    return ix.thresholds(jnp.where(idx < nv, sc, -1), idx, nv, K, G,
                                         **self.thr_kw)

                return self._by_tier(nvm, tiers, scan) if tiers else scan(S)

            if cdcp and n % self.dcp == 0:
                # Both pair ranks hold every row's scores: each finds the thresholds of its
                # share of the rows (the bisection is most of the selection) and the pair swaps
                # them (rows are disjoint across requests: no overwrite hazard).
                h = n // self.dcp
                o = jax.lax.rem(ctx.me, self.dcp) * h
                rid_h = jax.lax.broadcasted_iota(I32, (h, 1, 1), 0)
                nv_h = jnp.zeros((h, 1, 1), I32)
                for i in range(h):
                    nv_h = jnp.where(rid_h == i, ctx.extra[NVIS][lo + o + i], nv_h)
                ids_h = ids[:h]
                if self.nvis_thr:
                    nvm = nv_max(lo + o, h)

                    def full_h() -> Any:
                        sc = pltpu.bitcast(
                            scr["qi_s"][pl.ds(s0 + o, h)].reshape(h, S, 128), I32
                        )
                        return ix.thresholds(jnp.where(ids_h < nv_h, sc, -1), ids_h, nv_h, K, G,
                                             **self.thr_kw)

                    with ctx.stream.sc("qsa_thr", 2):
                        thr, tmax = self._nvis_route(
                            nvm, self.nvis_cut,
                            lambda: self._nvis_thresholds(ctx, s0 + o, h, nv_h, nvm), full_h,
                        )
                else:
                    sc = pltpu.bitcast(scr["qi_s"][pl.ds(s0 + o, h)].reshape(h, S, 128), I32)
                    with ctx.stream.sc("qsa_thr", 2):
                        thr, tmax = ix.thresholds(
                            jnp.where(ids_h < nv_h, sc, -1), ids_h, nv_h, K, G, **self.thr_kw
                        )
                scr["qi_th"][pl.ds(lo + o, h)] = jnp.broadcast_to(thr, (h, 1, 128))
                scr["qi_tm"][pl.ds(lo + o, h)] = jnp.broadcast_to(tmax, (h, 1, 128))
                with ctx.stream.sc("qsa_cdcp_thx", 2):
                    xs = [
                        pltpu.make_async_remote_copy(
                            src_ref=scr[k].at[pl.ds(lo + o, h)], dst_ref=scr[k].at[pl.ds(lo + o, h)],
                            send_sem=scr["qx_ssem"].at[1], recv_sem=scr["qx_rsem"].at[1],
                            device_id=(ctx.me ^ 1,), device_id_type=pl.DeviceIdType.MESH,
                        )
                        for k in ("qi_th", "qi_tm")
                    ]
                    ctx.comm.exchange(xs)
            elif self.nvis_thr:
                nvm = nv_max(lo, n)

                def full_n() -> Any:
                    return thr_full(nvm)

                with ctx.stream.sc("qsa_thr", 2):
                    thr, tmax = self._nvis_route(
                        nvm, self.nvis_cut,
                        lambda: self._nvis_thresholds(ctx, s0, n, nv, nvm), full_n,
                    )
                scr["qi_th"][pl.ds(lo, n)] = jnp.broadcast_to(thr, (n, 1, 128))
                scr["qi_tm"][pl.ds(lo, n)] = jnp.broadcast_to(tmax, (n, 1, 128))
            else:
                with ctx.stream.sc("qsa_thr", 2):
                    thr, tmax = thr_full(nv_max(lo, n) if tiers else None)
                scr["qi_th"][pl.ds(lo, n)] = jnp.broadcast_to(thr, (n, 1, 128))
                scr["qi_tm"][pl.ds(lo, n)] = jnp.broadcast_to(tmax, (n, 1, 128))
            scr["qi_nv"][pl.ds(lo, n)] = jnp.broadcast_to(nv, (n, 1, 128))
            lm = self.lane_major_select
            consts = ix.compact_consts(S, K, lm)
            lid = consts[-1]
            if cdcp:  # selection masks by group id; compaction yields positions (mapped below)
                lid = self._cdcp_gid(ctx, lid)
            scr["qi_z"][...] = jnp.zeros((1, 128), I32)
            if self.union_attn:
                scr["qu_code"][...] = jnp.zeros((S, 128), I32)

            if self.nvis_cmp:
                nconsts = ix.chunk_consts(2 * self.nvis_rows[1], K)

            def row(r: Any, carry: Any) -> Any:
                if self.nvis_cmp:
                    ga = ctx.extra[CTXQ][r // T] // 4

                    def ranged() -> Any:
                        pos, sga, kept = self._nvis_row(ctx, r, lo, s0, ga, nconsts)
                        scr["qc_sga"][r] = sga
                        return pos, kept

                    pos, kept = self._nvis_route(
                        ctx.extra[NVIS][r], self.nvis_cut, ranged, lambda: full_row(r)
                    )
                    return tail(r, pos, kept, carry)
                return tail(r, *full_row(r), carry)

            def full_row(r: Any) -> Any:
                if not ctiers:
                    return scan_row(r, S, consts, lid)

                def scan(Sx: int) -> Any:
                    cx = ix.compact_consts(Sx, K, lm)
                    return scan_row(r, Sx, cx, cx[-1])

                return self._by_tier(ctx.extra[NVIS][r], ctiers, scan)

            def scan_row(r: Any, Sx: int, consts: Any, lid: Any) -> Any:
                """Row `r`'s selection and compaction over score rows [0, Sx) (Sx < S: a
                row tier holding all its visible groups)."""
                nvr = scr["qi_nv"][r][:, :1]
                ga = ctx.extra[CTXQ][r // T] // 4
                sc = scr["qi_s"][r - lo + s0] if Sx == S else scr["qi_s"][
                    r - lo + s0, pl.ds(0, Sx // RS)]
                b = jnp.where(lid < nvr, pltpu.bitcast(sc.reshape(Sx, 128), I32), -1)
                th, tm = scr["qi_th"][r][:, :1], scr["qi_tm"][r][:, :1]
                ov = nvr > K
                sel = (ov & ((b > th) | ((b == th) & (lid <= tm)))) | (~ov & (b >= 0))
                # Recent groups (>= g_a) leave the pool path; remember whether g_a was
                # selected.
                scr["qc_sga"][r] = jnp.sum((sel & (lid == ga)).astype(I32))
                bp = jnp.where(lid < ga, b, -1)
                # Selection of the pool path: the same set restricted to ids < g_a. With
                # `n_vis`
                # clamped to g_a the compaction sees exactly those ids (select-all of
                # the kept).
                kept = jnp.sum((sel & (lid < ga)).astype(I32))
                b2 = jnp.where(sel & (lid < ga), bp, -1)
                if (
                    self.union_attn
                ):  # row bit r - lo of every pool-path group it selects
                    rows = pl.ds(0, Sx)
                    scr["qu_code"][rows] = scr["qu_code"][rows] | jnp.where(
                        sel & (lid < ga), jnp.left_shift(1, r - lo), 0
                    )
                if lm:  # the pool-path set compacted whole (at most K groups)
                    pos = ix.compact_sel(sel & (lid < ga), jnp.zeros((1, 1), I32) + kept, K,
                                         consts)
                else:
                    pos = ix.compact_row(
                        b2, jnp.zeros((1, 1), I32), jnp.zeros((1, 1), I32) + G,
                        jnp.zeros((1, 1), I32) + kept, K, consts,
                    )
                return pos, kept

            def tail(r: Any, pos: Any, kept: Any, carry: Any) -> Any:
                scr["qi_id"][r] = self._cdcp_gid(ctx, pos) if cdcp else pos
                cs = self._ids_copies(ctx, r)
                for c in cs:
                    c.start()
                if not self.async_dma:
                    for c in cs:
                        c.wait()
                scr["qi_kvl"][r] = 4 * kept
                return carry

            with ctx.stream.sc("qsa_rowsel", 2):
                jax.lax.fori_loop(lo, lo + n, row, 0)
                if self.async_dma:
                    self._wait_ids_copies(ctx, n)
            if self.union_attn:  # per-request call: rows [lo, lo + T) are request lo // T
                KU, D = self._union_width(ex), max(self.dcp, 1)
                with ctx.stream.sc("qsa_ucompact", 2):
                    # Main-KV DCP: one union per pair rank over the groups it owns (group g
                    # lives on rank (g // groups-per-page) % D), each compacted ascending.
                    code_all = scr["qu_code"][...]
                    gpp = ctx.extra[POOL].shape[1] // ua.MICRO
                    gidx = (
                        jax.lax.broadcasted_iota(I32, (S, 128), 0) * 128
                        + jax.lax.broadcasted_iota(I32, (S, 128), 1)
                    )
                    if cdcp:  # each rank's groups are one ascending half: ids map in order
                        gidx = self._cdcp_gid(ctx, gidx)
                    # Under `compressed_dcp` both pair ranks hold the request: each builds only
                    # its own union (the one it attends), and `_exchange_unions` sends it to the
                    # ranks of the same pair rank.
                    pack = (self.union_lanepack and D == 2 and not cdcp and S // 2 % 8 == 0
                            and (gpp == 128 or 128 % (2 * gpp) == 0))

                    def full_union(r: Any) -> None:
                        if pack:  # `_nvis_union`'s pack over all S rows: S / 2 packed rows
                            if gpp == 128:
                                code = scr["qu_code"][pl.ds(r, S // 2, stride=2)]
                            else:
                                code = ua.lane_pack(
                                    scr["qu_code"][pl.ds(0, S // 2, stride=2)],
                                    scr["qu_code"][pl.ds(1, S // 2, stride=2)],
                                    r, gpp, pltpu.roll,
                                )
                            ids, codes, nu = ua.compact_union(
                                code, KU, ua.union_consts(S // 2, KU, lm), lm
                            )
                            ids = ua.unpack_union_ids(ids, nu, r, gpp)
                        else:
                            code = code_all if D == 1 else jnp.where(
                                jax.lax.rem(gidx // gpp, D) == r, code_all, 0
                            )
                            ids, codes, nu = ua.compact_union(
                                code, KU, ua.union_consts(S, KU, lm), lm
                            )
                        if cdcp:
                            ids = self._cdcp_gid(ctx, ids)
                        at = pl.multiple_of(r * KU, 128) if cdcp else r * KU
                        scr["qu_id"][lo // T, :, pl.ds(at, KU)] = ids
                        scr["qu_cd"][lo // T, :, pl.ds(at, KU)] = codes
                        scr["qu_n"][lo // T, r] = jnp.sum(nu)

                    for r in [jax.lax.rem(ctx.me, D)] if cdcp else range(D):
                        if self.select_nvis_union:
                            nvm = nv_max(lo, T)
                            self._nvis_route(
                                nvm, self.nvis_ucut,
                                lambda r=r, nvm=nvm: self._nvis_union(ctx, lo, r, KU, D, gpp, nvm),
                                lambda r=r: full_union(r),
                            )
                            continue
                        full_union(r)
        self.load_tables(ctx, l)
        npr = self._shard_rows(ex)[0] if self.shard_index else nreq
        c_off = jax.lax.rem(ctx.me, self.dcp) * (NC // self.dcp) if cdcp else 0
        if cdcp:  # the first swap's credit (when the pair has a request this layer)
            if self.shard_index:
                pl.when(self._run(ctx) * npr < nreq)(lambda: self._cdcp_credit(ctx))
            else:
                self._cdcp_credit(ctx)

        def score_merge(
            q: Any, qr: Any, ck: Any, gc: Any, first: Any, last: Any, nxt: Any = None
        ) -> None:
            """Request `q`'s scores (this rank's groups, then the pair's under `compressed_dcp`)
            with the completed group's fresh key in its stale cached column; then selected."""
            self._score_request(
                ctx, q, qr, self.tab_base(ctx, l), row0=0, first=first, last=last, c_off=c_off,
                nxt=nxt,
            )
            if cdcp:
                self._cdcp_swap(ctx, q)
            s_new = ix.score_rows(qr, jnp.broadcast_to(ck.astype(BF16), (8, ix.IDX_DIM)))
            scr["qi_s"][0:T] = jnp.where(
                gid == gc, s_new[:, :1].reshape(T, 1, 1, 1), scr["qi_s"][0:T]
            )
            select_rows(q * T, T, 0)
            if cdcp:  # this request's scores are read: the pair may send the next one
                if isinstance(last, bool):
                    if not last:
                        self._cdcp_credit(ctx)
                else:
                    pl.when(jnp.logical_not(last))(lambda: self._cdcp_credit(ctx))

        n_loop = self._n_live(ctx) if self.skip_idle else nreq
        bal = self.shard_index and self._balanced(ex)
        if bal:
            self._balance_tables(ctx)
        if self.loop_requests:
            for q in range(nreq):  # static stores into the request-major copies
                scr["qc_irq"][q] = raw[q * T : (q + 1) * T]
                scr["qc_kvq"][q] = scr["qf_kv"][q * T : (q + 1) * T].astype(F32)

            xl = self.xreq_loads

            def issue_loads(q: Any, b: Any) -> None:
                for cp in self._xreq_loads(ctx, l, q, b):
                    cp.start()

            def req(q: Any, carry: Any) -> Any:
                bank = jax.lax.rem(q, 2) if xl else None
                if xl:  # the next request's loads go out before this request's work

                    @pl.when(q + 1 < n_loop)
                    def _():
                        issue_loads(jnp.minimum(q + 1, nreq - 1), 1 - bank)

                ck, gc = self._new_group(
                    ctx, l, q, scr["qc_irq"][q], bank,
                    jax.lax.rem(q, 2) if self.req_async else None,
                )

                def own_bal() -> None:  # `nvis_balance`: the run from `_balance_tables`
                    nxt = scr["qb_nxt"][q]
                    score_merge(
                        q, scr["qi_q"][q].astype(BF16), ck, gc, scr["qb_first"][q] == 1,
                        nxt < 0, jnp.maximum(nxt, 0),
                    )

                def own() -> None:
                    score_merge(
                        q, scr["qi_q"][q].astype(BF16), ck, gc, jax.lax.rem(q, npr) == 0,
                        # The last request of a run stops the key prefetch (`n_loop`: the
                        # last live one under `skip_idle`).
                        (jax.lax.rem(q, npr) == npr - 1) | (q == n_loop - 1),
                    )

                if bal:
                    pl.when(scr["qb_run"][q] == self._run(ctx))(own_bal)
                elif self.shard_index:
                    pl.when(q // npr == self._run(ctx))(own)
                else:
                    own()
                self._recent_tile(ctx, l, q, bank)
                return carry

            if xl:

                @pl.when(n_loop > 0)
                def _():
                    issue_loads(0, 0)

            jax.lax.fori_loop(0, n_loop, req, 0)
            if self.shard_index:
                self._exchange_selection(ctx)
        elif self.shard_index:
            # Each rank scores and selects its own run of `npr` requests; every rank still
            # builds every request's completed group (outputs) and recent tile (its KV head).
            for q in range(nreq):
                ck, gc = self._new_group(ctx, l, q, raw[q * T : (q + 1) * T])

                @pl.when(q // npr == self._run(ctx))
                def _(q=q, ck=ck, gc=gc):
                    score_merge(
                        q, scr["qi_q"][q].astype(BF16), ck, gc, q % npr == 0,
                        q % npr == npr - 1 or q == nreq - 1,
                    )

                self._recent_tile(ctx, l, q)
            self._exchange_selection(ctx)
        for q in range(0 if self.shard_index or self.loop_requests else nreq):
            qr = scr["qi_q"][q].astype(BF16)
            if cdcp:
                ck, gc = self._new_group(ctx, l, q, raw[q * T : (q + 1) * T])
                score_merge(q, qr, ck, gc, q == 0, q == nreq - 1)
                self._recent_tile(ctx, l, q)
                continue
            s0 = 0 if self.per_request_scores else q * T
            self._score_request(
                ctx, q, qr, self.tab_base(ctx, l), last=q == nreq - 1, row0=s0
            )
            ck, gc = self._new_group(ctx, l, q, raw[q * T : (q + 1) * T])
            s_new = ix.score_rows(
                qr, jnp.broadcast_to(ck.astype(BF16), (8, ix.IDX_DIM))
            )[:, :1]
            rows = slice(s0, s0 + T)
            # The completed group's fresh key replaces its stale cached column.
            scr["qi_s"][rows] = jnp.where(
                gid == gc, s_new.reshape(T, 1, 1, 1), scr["qi_s"][rows]
            )
            self._recent_tile(ctx, l, q)
            if (
                self.per_request_scores
            ):  # select this request's rows before the next scores
                select_rows(q * T, T, 0)
        if not self.per_request_scores:
            select_rows(0, MB, 0)
        for c in outs:
            c.wait()
        if (
            self.debug
        ):  # the kernel's queries and pool-path ids; g_a's flag in id slot K-1
            from .qsa_full_b import DBG_IDS, DBG_IQ

            for h in range(ix.IDX_HEADS):
                scr["qi_dbg"][:, h * ix.IDX_DIM : (h + 1) * ix.IDX_DIM] = iq[h]
            o = ctx.stream.sp.block_ord(l, self.name)
            for src, dst in ((scr["qi_dbg"], DBG_IQ), (scr["qi_id"], DBG_IDS)):
                c = pltpu.make_async_copy(
                    src, ctx.O[dst].at[o, pl.ds(0, MB)], scr["qi_sem"].at[0]
                )
                c.start()
                c.wait()

    def _ids_copies(self, ctx: Any, r: Any) -> list:
        """Row `r`'s pool-path ids (`qi_id`) and the zero tail into the SMEM table `qi_ids`."""
        scr, K = ctx.scr, self.budget
        return [
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

    def _wait_ids_copies(self, ctx: Any, n: Any) -> None:
        """`async_dma`: retire `n` rows' `_ids_copies` (equal-sized copies per semaphore; the
        rows' destinations are distinct, so their order is unobservable)."""

        def body(i: Any, carry: Any) -> Any:
            for c in self._ids_copies(ctx, 0):
                c.wait()
            return carry

        jax.lax.fori_loop(0, n, body, 0)

    @_scoped("qsa_selx")
    def _exchange_selection(self, ctx: Any) -> None:
        """All-gather the sharded indexer's per-row results: pool-path ids (`qi_id` and
        its
        SMEM copy `qi_ids`), `qi_kvl` and the `g_a` flag `qc_sga`. Each rank sends its
        run of
        rows (lanes `[0, K)` ids, `K` length, `K + 1` flag) to every peer's slot `me`;
        int32
        throughout, so the gathered selection is the owner's exactly."""
        if self.xmerge:
            self._exchange_merged(ctx)
            return
        ex, scr, K, T, MB = ctx.ex, ctx.scr, self.budget, ctx.ex.T, ctx.ex.MB
        npr, rows, W = self._shard_rows(ex)
        me = ctx.me
        lane = jax.lax.broadcasted_iota(I32, (1, W), 1)
        lo = self._run(ctx) * rows
        scr["qi_z"][...] = jnp.zeros((1, 128), I32)  # set by select_rows only on owners

        def pack(i: Any, carry: Any) -> Any:
            r = jnp.minimum(lo + i, MB - 1)  # past the last request: never read
            scr["qg_send"][i, :, pl.ds(0, K)] = scr["qi_id"][r]
            head = jnp.where(lane == K, scr["qi_kvl"][r], 0)
            scr["qg_send"][i] = jnp.where(
                lane < K,
                scr["qg_send"][i],
                jnp.where(lane == K + 1, scr["qc_sga"][r], head),
            )
            return carry

        jax.lax.fori_loop(0, rows, pack, 0)
        self._gather_runs(ctx, scr["qg_send"], scr["qg_recv"], scr["qg_ssem"], scr["qg_rsem"])

        def unpack(r: Any, carry: Any) -> Any:
            o = r // T // npr
            row = scr["qg_recv"][o, r - o * rows]
            scr["qi_id"][r] = row[:, :K]
            scr["qi_kvl"][r] = jnp.sum(jnp.where(lane == K, row, 0))
            scr["qc_sga"][r] = jnp.sum(jnp.where(lane == K + 1, row, 0))
            cs = self._ids_copies(ctx, r)
            for c in cs:
                c.start()
            if not self.async_dma:
                for c in cs:
                    c.wait()
            return carry

        jax.lax.fori_loop(0, MB, unpack, 0)
        if self.async_dma:
            self._wait_ids_copies(ctx, MB)
        if self.union_attn:
            self._exchange_unions(ctx)

    def _exchange_merged(self, ctx: Any) -> None:
        """`xmerge`: `_exchange_selection` and `_exchange_unions` as one all-gather. Request
        `i` of this rank's run sends one row (`_xmerge_width`): its `T` selection rows (the
        same lanes as `qg_send`) and its union (the same lanes as `qgu_send`, offset `u0`);
        unpacked into the same scratch as the two exchanges. int32 copies only."""
        ex, scr, K, T, MB = ctx.ex, ctx.scr, self.budget, ctx.ex.T, ctx.ex.MB
        nreq = MB // T
        npr, rows, WS = self._shard_rows(ex)
        u0, _ = self._xmerge_width(ex)
        D = max(self.dcp, 1)
        W = D * self._union_width(ex)
        lane = jax.lax.broadcasted_iota(I32, (1, WS), 1)
        ulane = jax.lax.broadcasted_iota(I32, (1, 2 * W + 128), 1)
        run = self._run(ctx)
        send, recv = scr["qm_send"], scr["qm_recv"]
        scr["qi_z"][...] = jnp.zeros((1, 128), I32)  # set by select_rows only on owners

        bal = self._balanced(ex)

        def pack(i: Any, carry: Any) -> Any:
            if bal:  # `nvis_balance`: the run's i-th request (an empty slot: never read)
                qb = jnp.maximum(scr["qb_slot"][run * npr + i], 0)
            for t in range(T):
                r = (
                    qb * T + t if bal
                    else jnp.minimum(run * rows + i * T + t, MB - 1)  # past the last request: never read
                )
                at = t * WS
                send[i, :, pl.ds(at, K)] = scr["qi_id"][r]
                head = jnp.where(lane == K, scr["qi_kvl"][r], 0)
                send[i, :, pl.ds(at, WS)] = jnp.where(
                    lane < K,
                    send[i, :, pl.ds(at, WS)],
                    jnp.where(lane == K + 1, scr["qc_sga"][r], head),
                )
            q = qb if bal else jnp.minimum(run * npr + i, nreq - 1)  # past the last: never read
            send[i, :, pl.ds(u0, W)] = scr["qu_id"][q]
            send[i, :, pl.ds(u0 + W, W)] = scr["qu_cd"][q]
            nl = jnp.zeros((1, 128), I32)
            for r in range(D):
                nl = jnp.where(ulane[:, :128] == r, scr["qu_n"][q, r], nl)
            send[i, :, pl.ds(u0 + 2 * W, 128)] = nl
            return carry

        jax.lax.fori_loop(0, npr, pack, 0)
        self._gather_runs(ctx, send, recv, scr["qm_ssem"], scr["qm_rsem"])

        def unpack_slot(q: Any, o: Any, j: Any) -> None:
            for t in range(T):
                r = q * T + t
                row = recv[o, j, :, pl.ds(t * WS, WS)]
                scr["qi_id"][r] = row[:, :K]
                scr["qi_kvl"][r] = jnp.sum(jnp.where(lane == K, row, 0))
                scr["qc_sga"][r] = jnp.sum(jnp.where(lane == K + 1, row, 0))
                cs = self._ids_copies(ctx, r)
                for c in cs:
                    c.start()
                if not self.async_dma:
                    for c in cs:
                        c.wait()
            urow = recv[o, j, :, pl.ds(u0, 2 * W + 128)]
            scr["qu_id"][q] = urow[:, :W]
            scr["qu_cd"][q] = urow[:, W : 2 * W]
            for r in range(D):
                scr["qu_n"][q, r] = jnp.sum(jnp.where(ulane == 2 * W + r, urow, 0))

        def unpack(q: Any, carry: Any) -> Any:
            o = q // npr
            unpack_slot(q, o, q - o * npr)
            return carry

        def unpack_bal(sl: Any, carry: Any) -> Any:  # `nvis_balance`: every slot of every run
            q = scr["qb_slot"][sl]
            o = sl // npr
            pl.when(q >= 0)(lambda: unpack_slot(q, o, sl - o * npr))
            return carry

        if bal:
            jax.lax.fori_loop(0, self._runs(ex) * npr, unpack_bal, 0)
        else:
            jax.lax.fori_loop(0, nreq, unpack, 0)
        if self.async_dma:
            self._wait_ids_copies(ctx, MB)

    @_scoped("qsa_ux")
    def _exchange_unions(self, ctx: Any) -> None:
        """All-gather the owners' per-request unions (ids, row codes, count; int32)."""
        ex, scr = ctx.ex, ctx.scr
        nreq, KU = ex.MB // ex.T, self._union_width(ex)
        npr = self._shard_rows(ex)[0]
        me = ctx.me
        D = max(self.dcp, 1)
        W = D * KU
        lane = jax.lax.broadcasted_iota(I32, (1, 2 * W + 128), 1)

        def pack(i: Any, carry: Any) -> Any:
            q = jnp.minimum(self._run(ctx) * npr + i, nreq - 1)  # past the last request: never read
            scr["qgu_send"][i, :, pl.ds(0, W)] = scr["qu_id"][q]
            scr["qgu_send"][i, :, pl.ds(W, W)] = scr["qu_cd"][q]
            nl = jnp.zeros((1, 128), I32)
            for r in range(D):
                nl = jnp.where(lane[:, :128] == r, scr["qu_n"][q, r], nl)
            scr["qgu_send"][i, :, pl.ds(2 * W, 128)] = nl
            return carry

        jax.lax.fori_loop(0, npr, pack, 0)
        self._gather_runs(
            ctx, scr["qgu_send"], scr["qgu_recv"], scr["qgu_ssem"], scr["qgu_rsem"], own_rank=True
        )

        def unpack(q: Any, carry: Any) -> Any:
            o = q // npr
            row = scr["qgu_recv"][o, q - o * npr]
            scr["qu_id"][q] = row[:, :W]
            scr["qu_cd"][q] = row[:, W : 2 * W]
            for r in range(D):
                scr["qu_n"][q, r] = jnp.sum(jnp.where(lane == 2 * W + r, row, 0))
            return carry

        jax.lax.fori_loop(0, nreq, unpack, 0)

    def _n_live(self, ctx: Any) -> Any:
        """`skip_idle`: requests up to the last one with a context (live requests come
        first; an idle request slot has context 0)."""
        n = jnp.int32(0)
        for q in range(ctx.ex.MB // ctx.ex.T):
            n = jnp.where(ctx.extra[CTXQ][q] > 0, q + 1, n)
        return n

    def _recent_ok(self, ctx: Any, r: Any, s: Any) -> Any:
        """Row `r`'s valid recent-tile slots `s`: causal, its partial group, and `g_a`
        when
        complete for the row and selected (the per-row attention's `ok`)."""
        ex, scr, T = ctx.ex, ctx.scr, ctx.ex.T
        q = r // T
        c0 = ctx.extra[CTXQ][q]
        ga = c0 // 4
        p_r = c0 + r - q * T
        n_r, tail = ctx.extra[NVIS][r], ctx.extra[TAIL][r]
        pos = 4 * ga + s
        grp = pos // 4
        part = (grp == n_r) & (tail > 0)
        full = (grp == ga) & (ga < n_r) & (scr["qc_sga"][r] > 0)
        ok = (r < ex.B) & (pos <= p_r) & (part | full)
        return ok & self._dcp_owned(ctx, pos) if self.dcp > 1 else ok

    def _attend_unions(
        self,
        ctx: Any,
        l: Any,  # noqa: E741
        zeros: tuple,
    ) -> list[jax.Array]:
        """Every request's rows over the union of their selections (`union_attn`)."""
        ex, scr, g, T = ctx.ex, ctx.scr, ctx.g, ctx.ex.T
        nh, KU, nreq = self.nh(g, ex), self._union_width(ex), ex.MB // ex.T
        rows = jax.lax.broadcasted_iota(I32, (ex.MB, 1), 0)
        srow = jax.lax.broadcasted_iota(I32, (1, 8), 1)
        scol = jax.lax.broadcasted_iota(I32, (8, 1), 0)

        sc = ctx.stream.sc

        rk = jax.lax.rem(ctx.me, self.dcp) if self.dcp > 1 else 0
        uoff = pl.multiple_of(rk * KU, 128) if self.dcp > 1 else 0

        xreq = self.xreq_prefetch
        rasync = self.req_async_ids
        rbank = self.req_async_bank

        def stage_ids(q: Any, bank: Any, wait: bool = True) -> None:
            """Request `q`'s union ids (physical with `union_phys`) into `qu_row[bank]`.
            `wait=False` (`req_async`): the copies into `qu_row` are left in flight
            (`wait_ids`)."""
            # This rank's union: under DCP only the groups it owns (the pair rank's rows are
            # neither streamed nor scored here).
            src = scr["qu_id"].at[q, :, pl.ds(uoff, KU)]
            n_u = scr["qu_n"][q, rk]
            with sc("qsa_u_ids", 2):
                if self.union_phys:
                    # The request's main-table row, then its union ids -> physical micro-page ids for
                    # every slot a tile reads (< ceil(n / 64) * 64 <= ceil(n / 128) * 128).
                    o = ctx.stream.sp.block_ord(l, self.name) if self.n_layers else 0
                    cp = pltpu.make_async_copy(
                        ctx.extra[MTAB].at[o, pl.ds(q, 1)], scr["qu_tab"], scr["qu_sem"].at[2]
                    )
                    cp.start()
                    cp.wait()
                    per_page = ctx.extra[POOL].shape[1] // ua.MICRO
                    tab = scr["qu_tab"][...]

                    def chunk(c: Any, carry: Any) -> Any:
                        lo = pl.multiple_of(c * 128, 128)
                        scr["qu_pv"][:, pl.ds(lo, 128)] = ua.physical_ids(
                            src[:, pl.ds(lo, 128)], tab, per_page, self.dcp
                        )
                        return carry

                    jax.lax.fori_loop(0, (n_u + 127) // 128, chunk, 0)
                    src = scr["qu_pv"]
                cs = [
                    pltpu.make_async_copy(
                        src, scr["qu_row"].at[pl.ds(bank, 1), pl.ds(0, KU)],
                        scr["qu_sem"].at[0],
                    ),
                    pltpu.make_async_copy(
                        scr["qi_z"], scr["qu_row"].at[pl.ds(bank, 1), pl.ds(KU, 128)],
                        scr["qu_sem"].at[1],
                    ),
                ]
                for c in cs:
                    c.start()
                if wait:
                    for c in cs:
                        c.wait()
            return n_u

        def wait_ids() -> None:
            """`req_async`: wait the `qu_row` copies `stage_ids(.., wait=False)` issued (the
            waits need only the semaphores and byte counts)."""
            with sc("qsa_u_ids", 2):
                for c in (
                    pltpu.make_async_copy(
                        scr["qu_id"].at[0, :, pl.ds(0, KU)], scr["qu_row"].at[pl.ds(0, 1), pl.ds(0, KU)],
                        scr["qu_sem"].at[0],
                    ),
                    pltpu.make_async_copy(
                        scr["qi_z"], scr["qu_row"].at[pl.ds(0, 1), pl.ds(KU, 128)],
                        scr["qu_sem"].at[1],
                    ),
                ):
                    c.wait()

        def req(q: Any, carry: Any) -> Any:
            if xreq:
                outs, sbase = carry
                bank = jax.lax.rem(q, 2)
                qn = jnp.minimum(q + 1, nreq - 1)
                nlive = q + 1 < n_loop
                # The next request's ids, staged while this request's first tiles (issued by
                # the previous request) are in flight.
                @pl.when(nlive)
                def _():
                    stage_ids(qn, 1 - bank)

                stream = (
                    sbase, q == 0,
                    (nlive, 1 - bank, self.tab_base(ctx, l) + qn, scr["qu_n"][qn, rk]),
                )
                n_u = scr["qu_n"][q, rk]
            elif rasync:  # staged by the previous request (or before the loop)
                outs, stream = carry, None
                bank = jax.lax.rem(q, 2) if rbank else 0
                n_u = scr["qu_n"][q, rk]
                if rbank:  # q's ids are in: q + 1's go to the other bank during q's attention
                    wait_ids()

                    @pl.when(q + 1 < n_loop)
                    def _():
                        stage_ids(jnp.minimum(q + 1, nreq - 1), 1 - bank, wait=False)
            else:
                outs, bank, stream = carry, 0, None
                n_u = stage_ids(q, 0)
            rs = [q * T + t for t in range(T)]
            ok = [self._recent_ok(ctx, r, srow) for r in rs]
            ok_any = self._recent_ok(ctx, rs[0], scol)
            for r in rs[1:]:
                ok_any = ok_any | self._recent_ok(ctx, r, scol)
            if self.dcp > 1:  # both pair ranks' heads, this rank's first
                qs = [jnp.concatenate([scr["qf_q"][r], scr["qd_q"][r]], axis=0) for r in rs]
            else:
                qs = [scr["qf_q"][r] for r in rs]
            if rasync and not rbank:
                wait_ids()
            with sc("qsa_u_attend", 2):
                res = ua.attend_request(
                    T=T, nh=nh * self.dcp, pool=ctx.extra[POOL], ring=scr["qf_ring"],
                    sems=scr["qf_ring_sem"],
                    ids=scr["qu_row"], codes=scr["qu_cd"].at[q, :, pl.ds(uoff, KU)],
                        n_union=n_u,
                    table=self.tables(ctx)[1], table_row=self.tab_base(ctx, l) + q,
                    qs=qs,
                    gates=[scr["qf_g"][r] for r in rs],
                    has=[scr["qi_kvl"][r] > 0 for r in rs],
                    off=self.offset(ctx, l), scale_k=ctx.extra[SCALES][0],
                    scale_v=ctx.extra[SCALES][1], sm_scale=g.HD**-0.5, interpret=ex.interpret,
                    recent=(scr["qc_rk"][q], scr["qc_rv"][q], ok, ok_any), phys=self.union_phys,
                    dcp=self.dcp, sc=ctx.stream.sc,
                    **(dict(ids_row=bank, stream=stream) if xreq else {}),
                    **(dict(ids_row=bank) if rbank else {}),
                    **(dict(lite=True) if self.uattn_lite else {}),
                )
            if rasync and not rbank:  # request q's ids are read: the next request's go into `qu_row`

                @pl.when(q + 1 < n_loop)
                def _():
                    stage_ids(jnp.minimum(q + 1, nreq - 1), 0, wait=False)

            if self.dcp > 1:
                for r, (att, lse) in zip(rs, res):
                    lse = jnp.broadcast_to(lse, (2 * nh, 128))
                    scr["qd_o"][r], scr["qd_os"][r] = att[:nh], att[nh:]
                    scr["qd_l"][r], scr["qd_ls"][r] = lse[:nh], lse[nh:]
                return (outs, nxt_base(sbase, n_u)) if xreq else outs
            if self.union_stats:
                assert nreq <= UNION_STATS_REQS, (nreq, UNION_STATS_REQS)
                kept = [scr["qi_kvl"][r] // 4 for r in rs]
                lane = jax.lax.broadcasted_iota(I32, (1, 128), 1)
                vals = [n_u, sum(kept[1:], kept[0]), kept[0], (kept[0] > 0).astype(I32)]
                for k in kept[1:]:
                    vals[2] = jnp.maximum(vals[2], k)
                    vals[3] = vals[3] + (k > 0).astype(I32)
                st = jnp.zeros((1, 128), I32)
                for i, v in enumerate(vals):
                    st = jnp.where(lane == i, v, st)
                scr["qu_st"][...] = st
                cp = pltpu.make_async_copy(
                    scr["qu_st"],
                    ctx.O[UNION_STATS].at[ctx.stream.sp.block_ord(l, self.name), q],
                    scr["qu_sem"].at[2],
                )
                cp.start()
                cp.wait()
            outs = list(outs)
            for t, r in enumerate(rs):
                for h in range(nh):
                    outs[h] = jnp.where(rows == r, res[t][h : h + 1], outs[h])
            return (tuple(outs), nxt_base(sbase, n_u)) if xreq else tuple(outs)

        def nxt_base(sbase: Any, n_u: Any) -> Any:
            return sbase + (n_u * ua.MICRO + ua.TILE - 1) // ua.TILE

        n_loop = self._n_live(ctx) if self.skip_idle else nreq
        if rasync:

            @pl.when(n_loop > 0)
            def _():
                stage_ids(0, 0, wait=False)

        if not xreq:
            return list(jax.lax.fori_loop(0, n_loop, req, zeros))
        @pl.when(n_loop > 0)
        def _():
            stage_ids(0, 0)

        outs, _ = jax.lax.fori_loop(0, n_loop, req, (zeros, jnp.int32(0)))
        return list(outs)

    def _dcp_pair(self, ctx: Any, src: Any, dst: Any, i: int) -> None:
        """`src` -> the pair rank's `dst` (and its `src` -> ours), waited on both sides."""
        scr = ctx.scr
        ctx.comm.exchange(ctx.comm.group_copies(
            lambda off: src, dst, scr[f"qd_ssem{i}"], scr[f"qd_rsem{i}"], 2,
        ))

    def _dcp_attend(self, ctx: Any, l: Any, zeros: tuple) -> None:
        """Pair q exchange, then the union attention over this rank's groups for both ranks'
        heads (partials into `qd_o` / `qd_l` and, for the pair rank, `qd_os` / `qd_ls`)."""
        scr, sc = ctx.scr, ctx.stream.sc
        with sc("qsa_dcp_qx", 2):
            self._dcp_pair(ctx, scr["qf_q"], scr["qd_q"], 0)
        self._attend_unions(ctx, l, zeros)
        with sc("qsa_dcp_px", 2):
            self._dcp_pair(ctx, scr["qd_os"], scr["qd_or"], 1)
            self._dcp_pair(ctx, scr["qd_ls"], scr["qd_lr"], 2)

    def _dcp_combine(self, ctx: Any) -> list[jax.Array]:
        """This rank's heads: the pair's partials merged by LSE, then gated (`dcp_combine`'s
        math and rounding; rows no rank owns a token of stay 0). `nh` `[MB, HD]` F32."""
        ex, scr, g = ctx.ex, ctx.scr, ctx.g
        nh = self.nh(g, ex)
        rows = jax.lax.broadcasted_iota(I32, (ex.MB, 1), 0)

        def row(r: Any, outs: Any) -> Any:
            l0, l1 = scr["qd_l"][r][:, 0:1], scr["qd_lr"][r][:, 0:1]
            top = jnp.maximum(l0, l1)
            fin = top > -jnp.inf
            safe = jnp.where(fin, top, 0.0)
            w0 = jnp.where(fin, jnp.exp(l0 - safe), 0.0)
            w1 = jnp.where(fin, jnp.exp(l1 - safe), 0.0)
            tot = w0 + w1
            o = (w0 * scr["qd_o"][r] + w1 * scr["qd_or"][r]) / jnp.where(tot > 0, tot, 1.0)
            gated = (
                o.astype(BF16).astype(F32)
                * jax.nn.sigmoid(scr["qf_g"][r].astype(F32)).astype(BF16).astype(F32)
            ).astype(BF16).astype(F32)
            res = jnp.where(fin, gated, 0.0)
            return tuple(jnp.where(rows == r, res[h : h + 1], o_) for h, o_ in enumerate(outs))

        zeros = tuple(jnp.zeros((ex.MB, g.HD), F32) for _ in range(nh))
        return list(jax.lax.fori_loop(0, ex.MB, row, zeros))

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        ex, scr, g = ctx.ex, ctx.scr, ctx.g
        nh, T = self.nh(g, ex), ex.T
        sc = ctx.stream.sc
        with sc("qsa_proj"):
            kv = self._project(ctx, l, slot)
        with sc("qsa_select"):
            self._select(ctx, l, slot)
        self._export_ids(ctx, l, start=True)
        rows = jax.lax.broadcasted_iota(I32, (ex.MB, 1), 0)
        srow = jax.lax.broadcasted_iota(I32, (1, 8), 1)
        scol = jax.lax.broadcasted_iota(I32, (8, 1), 0)

        def row(r: Any, carry: Any) -> Any:
            *outs, g0, pre = carry
            q = r // T
            c0 = ctx.extra[CTXQ][q]
            ga = c0 // 4
            p_r = c0 + r - q * T
            n_r, tail = ctx.extra[NVIS][r], ctx.extra[TAIL][r]
            live = r < ex.B

            def ok(s: Any) -> Any:
                pos = 4 * ga + s
                grp = pos // 4
                part = (grp == n_r) & (tail > 0)
                full = (grp == ga) & (ga < n_r) & (scr["qc_sga"][r] > 0)
                return live & (pos <= p_r) & (part | full)

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
                rows_per_req=T,
                table_row_base=self.tab_base(ctx, l),
                q=scr["qf_q"][r],
                gate=scr["qf_g"][r],
                off=self.offset(ctx, l),
                scale_k=ctx.extra[SCALES][0],
                scale_v=ctx.extra[SCALES][1],
                sm_scale=g.HD**-0.5,
                interpret=ex.interpret,
                recent=(scr["qc_rk"][q], scr["qc_rv"][q], ok(srow), ok(scol)),
            )
            return (
                *(jnp.where(rows == r, out[h : h + 1], o) for h, o in enumerate(outs)),
                g0,
                pre,
            )

        zeros = tuple(jnp.zeros((ex.MB, g.HD), F32) for _ in range(nh))
        with sc("qsa_attend"):
            if self.dcp > 1:
                self._dcp_attend(ctx, l, zeros)
                with sc("qsa_dcp_combine", 2):
                    outs = self._dcp_combine(ctx)
            elif self.union_attn:
                outs = self._attend_unions(ctx, l, zeros)
            else:
                *outs, _, _ = jax.lax.fori_loop(
                    0, ex.MB, row, (*zeros, jnp.int32(0), jnp.int32(0))
                )
        with sc("qsa_store"):
            for h, o in enumerate(outs):
                scr["xb2"][:, pl.ds(h * g.HD, g.HD)] = o.astype(BF16)
            kv.wait()
            if self.req_async:
                self._drain_new_group_stores(ctx, l)
            self._export_ids(ctx, l, start=False)
        with ctx.stream.segment(lazy=False):
            return ctx.stream.dmatmul("o_proj", scr["xb2"], g.H)

    def _export_ids(self, ctx: Any, l: Any, *, start: bool) -> None:  # noqa: E741
        """`export_ids`: start (after the selection) or wait (after the attention, which
        does not write `qi_id`) the copy of the last QSA layer's pool-path ids to `SEL_IDS`.
        Per row: ascending ids below g_a, slots past the count 0; g_a, when selected, is in
        no slot (`selection_groups` restores it). No VMEM: `qi_id` is the selection's own."""
        if not self.export_ids:
            return
        from .qsa_full_b import SEL_IDS

        MB = ctx.ex.MB
        cp = pltpu.make_async_copy(
            ctx.scr["qi_id"], ctx.O[SEL_IDS].at[pl.ds(0, MB)], ctx.scr["qe_sem"].at[0]
        )
        go = cp.start if start else cp.wait
        if not self.n_layers:
            go()
            return
        last = ctx.stream.sp.block_ord(l, self.name) == self.n_layers - 1
        if isinstance(last, bool):  # a static layer index
            if last:
                go()
            return
        pl.when(last)(go)
