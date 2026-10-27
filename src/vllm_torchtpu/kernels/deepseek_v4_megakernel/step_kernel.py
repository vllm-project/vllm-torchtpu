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
"""DSv4 whole decode/verify step: one grid-less pallas_call per rank (L4e).

Every rank runs the dense chain of every layer on all T global rows
(replicated activations, weights from the cross-layer ring of dense_ring);
the sharded projections give each rank its slice: wq_b heads
[me*hpr, (me+1)*hpr), wo_a o-group `me`, wo_b rows of group `me`, the shared
expert's 1/R intermediate slice. Per layer:

  mHC pre + norm, wqa, q/kv norm, compressor/indexer projections, wq_b (own
  heads, all rows) -> qnorm + rope -> A2A#1 (each token's heads to its
  owner) -> owner attention phase (own requests) -> A2A#2 (o-group g of
  every token to rank g) -> wo_a(g), wo_b rows(g) -> AR#1 -> mHC + ffn norm,
  gate, shared slice -> MoE phase (routed experts + AR#2).

Token metadata (owner rank and owner-local row of every global row) is built
in XLA before the call (`step_metadata`) and read from SMEM. The final head
collapse / norm and the aux means run in XLA on the kernel's own-row outputs
(the residual streams after the last post and after each aux layer).
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4_megakernel import allreduce as AR
from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_ring as DR
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_phases as SP
from vllm_torchtpu.kernels.deepseek_v4_megakernel.dense_config import (
    CSA,
    HCA,
    SWA,
    DenseChainConfig,
)

F32, BF16, FP8 = jnp.float32, jnp.bfloat16, jnp.float8_e4m3fn
LANE = 128
NEG_ROW = -1
ROLES = ("swa", "kv", "state", "kv_rope", "indexer_k", "indexer_state")
META_ROLES = ROLES + ("swa_pages",)


@dataclasses.dataclass(frozen=True)
class AttnLayout:
    """Static layout of the served caches handed to the attention phase:
    distinct arrays (shape, dtype) and, per role, the array of each entry and
    its decoder layer (step_phases.AttnCaches / LayerRefs)."""

    arrays: tuple[tuple[tuple[int, ...], str], ...]
    cache_index: tuple[tuple[int, ...], ...]  # per ROLES entry
    cache_layers: tuple[tuple[int, ...], ...]


@dataclasses.dataclass(frozen=True)
class StepKernelConfig:
    """One compiled step: dense geometry + mesh + rows."""

    dense: DenseChainConfig  # tp = R, rows = T
    q: int  # rows per request
    B: int  # this rank's request slots
    aux_layers: tuple[int, ...] = ()  # 0-based layers whose post state is output
    ar_mode: str = "hrsag"
    ar_group: int = 8
    attn_impl: str = "stub"  # "stub": per-row RMSNorm of the gathered q; "real"
    max_ctx: int = 131072  # attention: max tokens per request of this bucket
    debug_att: bool = False  # also output layer 0's attention output (own reqs)
    # Attention caches are inputs the kernel DMA-writes in place, with no
    # outputs. A communicating custom call's outputs are compiler-owned
    # buffers, so with aliased outputs XLA copies every cache in and out of
    # each step. In place relies on XLA neither copying nor reusing a
    # read-only, non-donated parameter, so nothing else in the program may read
    # the caches after the call. Interpret mode does not mutate inputs: keep
    # the aliased path there.
    cache_inplace: bool = False
    # time attribution only (wrong numerics): "a2a" (no all-to-alls), "ar" (no
    # all-reduces: partials kept local), "fence" (no ring fence: phases must not
    # borrow the pool), "dense_dma" (no weight DMA: compute on stale banks)
    skip: tuple[str, ...] = ()
    moe_impl: str = "stub"  # "stub": shared-expert partial only
    moe_leased: bool = True  # MoE expert ring in the dense pool (device only)
    # positive control of the numerics check: "a2a_heads" lands each rank's
    # heads in the next rank's head slots
    fault: str = ""
    # DSpark draft step (SWA layers): the context rows per request (the target
    # step's verify rows, global layout); each SWA layer first projects their
    # kv from main_x (wqa_ctx stream) for its attention to insert. 0: none.
    ctx_q: int = 0
    num_hash_layers: int = 3  # the first layers route by token id (0: draft)
    # Ranks per attention group (`head_block`): 1 = DP attention (each request
    # on one owner rank, all heads); 8 = TP8 x DP2 serving, where a DP group's
    # 8 ranks share its KV replica and each attends NA = NH / 8 heads of every
    # group request.
    attn_tp: int = 1
    # static expert placement (moe_phase.placement_tables): "" none, "perm"
    # (moe_w["held"]), "replicas" (also moe_w["copy"])
    moe_placement: str = ""
    strict_causal: bool = False  # StepGeometry.strict_causal

    @property
    def n_debug_outputs(self) -> int:
        """debug_att: layer 0's attention output and residual taps, plus the
        context kv with ctx_q."""
        return (3 if self.ctx_q else 2) if self.debug_att else 0

    def __post_init__(self):
        assert self.R * self.dense.heads_per_group == self.dense.n_heads, (
            "one o-group (and its heads) per rank"
        )
        assert self.T % 16 == 0 and self.T <= LANE and self.T % self.QP == 0
        # the bf16 all-to-all blocks pair two tokens (or two ranks' heads)
        # into one (16, 128) tile
        assert self.hpr == 8 and self.R % 2 == 0 and self.NH % 16 == 0
        assert all(0 <= a < self.dense.n_layers for a in self.aux_layers)
        assert self.R % self.attn_tp == 0 and self.G % 2 == 0
        assert self.moe_placement in ("", "perm", "replicas")

    @property
    def R(self) -> int:
        return self.dense.tp

    @property
    def T(self) -> int:
        return self.dense.rows

    @property
    def G(self) -> int:  # attention groups = head sources of an attending rank
        return self.R // self.attn_tp

    @property
    def NA(self) -> int:  # heads a rank attends per request row
        return self.NH // self.attn_tp

    @property
    def M(self) -> int:
        return self.B * self.q

    @property
    def QP(self) -> int:  # global rows per request block (f32-tile aligned)
        return -(-self.q // 8) * 8

    @property
    def hpr(self) -> int:  # heads per rank (= heads of one o-group)
        return self.dense.heads

    @property
    def NH(self) -> int:
        return self.dense.n_heads

    @property
    def RP(self) -> int:
        return -(-self.q * self.NA // LANE) * LANE

    def geometry(self) -> SP.StepGeometry:
        cfg = self.dense
        return SP.StepGeometry(
            R=self.R, T=self.T, q=self.q, B=self.B,
            layer_kinds=tuple(
                {CSA: "csa", HCA: "hca", SWA: "swa"}[t] for t in cfg.layer_types
            ),
            hidden=cfg.hidden, hc_mult=cfg.hc_mult, n_heads=cfg.n_heads,
            head_dim=cfg.head_dim, rope_dim=cfg.rope_dim, q_lora=cfg.q_lora,
            o_groups=cfg.n_heads // cfg.heads_per_group, o_lora=cfg.o_lora,
            idx_heads=cfg.idx_heads, idx_head_dim=cfg.idx_head_dim,
            n_experts=cfg.n_experts, moe_inter=cfg.moe_inter,
            lease_bytes=cfg.n_banks * cfg.max_tile_bytes, max_ctx=self.max_ctx,
            num_hash_layers=self.num_hash_layers, attn_heads=self.NA,
            attn_tp=self.attn_tp, strict_causal=self.strict_causal,
        )  # fmt: skip

    def ar_spec(self) -> AR.AllReduceSpec:
        mode = self.ar_mode if self.ar_group <= self.R else "flat"
        return AR.AllReduceSpec(
            R=self.R, rows=self.T, hidden=self.dense.hidden, mode=mode,
            group=min(self.ar_group, self.R),
        )  # fmt: skip


# ---------------------------------------------------------------------------
# metadata (XLA, per rank, inside shard_map)
# ---------------------------------------------------------------------------
def head_block(r, R: int, attn_tp: int):
    """The head block (8 heads, one wo_a o-group, one 1/R shared-expert slice)
    mesh rank r = g * attn_tp + j holds: half g of attention-TP shard j, so the
    NA heads rank r attends are the blocks of ranks g' * attn_tp + j."""
    G = R // attn_tp
    return (r % attn_tp) * G + r // attn_tp


def step_metadata(scfg: StepKernelConfig, me, input_ids, positions, seq_lens, axis):
    """All-gather the per-group row metadata (each attention group's ranks
    hold the same batch; its first rank's is used) and lay the active requests
    of all groups out as global row blocks of QP rows (group-major; rows
    t >= q of a block are padding). Returns int32 arrays: tokm [4 * T] (id,
    position, owner group or -1, group-local row), ownt [M] (global row of own
    row k, or -1), counts [4] (live global rows, own live rows, global
    requests, own requests), own_greq [B] (global request of own slot r, or
    -1). "Own" = this rank's group (= this rank at attn_tp 1)."""
    q, M, T, QP, B = scfg.q, scfg.M, scfg.T, scfg.QP, scfg.B
    # one all-gather (each is a ~12 us collective at R=16)
    mine = jnp.concatenate([input_ids, positions, seq_lens]).astype(jnp.int32)
    allr = lax.all_gather(mine, axis)[:: scfg.attn_tp]  # [G, 2M + B]
    me = me // scfg.attn_tp  # own group
    ids, pos, sl = allr[:, :M], allr[:, M : 2 * M], allr[:, 2 * M :]
    nreq = jnp.sum(sl > 0, axis=1).astype(jnp.int32)  # active requests per rank
    roff = jnp.cumsum(nreq) - nreq
    total = jnp.sum(nreq)
    t = jnp.arange(T, dtype=jnp.int32)
    gr, j = t // QP, t % QP
    owner = jnp.sum(gr[:, None] >= (roff + nreq)[None, :], axis=1).astype(jnp.int32)
    owner_c = jnp.minimum(owner, scfg.G - 1)
    lrow = (gr - roff[owner_c]) * q + j
    live = (gr < total) & (j < q)
    lrow_c = jnp.clip(lrow, 0, M - 1)
    tid = jnp.where(live, ids[owner_c, lrow_c], 0)
    tpos = jnp.where(live, pos[owner_c, lrow_c], 0)
    tokm = jnp.concatenate([
        tid, tpos, jnp.where(live, owner_c, NEG_ROW), jnp.where(live, lrow, NEG_ROW)
    ])  # fmt: skip
    k = jnp.arange(M, dtype=jnp.int32)
    ownt = jnp.where(k // q < nreq[me], (roff[me] + k // q) * QP + k % q, NEG_ROW)
    r = jnp.arange(B, dtype=jnp.int32)
    own_greq = jnp.where(r < nreq[me], roff[me] + r, NEG_ROW).astype(jnp.int32)
    counts = jnp.stack([total * q, nreq[me] * q, total, nreq[me]]).astype(jnp.int32)
    return tokm, ownt.astype(jnp.int32), counts, tid, tpos, own_greq


def rope_rows(cos_sin_cache, positions, inverse: bool, packed: bool = False):
    """[T, 256] f32: cos | sin over the last 128 lanes of a head, identity on
    the NoPE lanes, as rope.py `_cos_sin_lanes` expands a cache row.
    packed: the table is SP.pack_cos_sin's [ceil(max_pos / 2), 128] (a
    64-lane table gets a column-major layout and a relayout copy per step)."""
    if packed:
        rows = cos_sin_cache[positions // 2].astype(F32)
        half = rows.shape[1] // 2
        odd = (positions % 2 == 1)[:, None]
        cs = jnp.where(odd, rows[:, half:], rows[:, :half])  # [T, rotary]
    else:
        cs = cos_sin_cache[positions].astype(F32)  # [T, rotary]
    rd = cs.shape[1]
    half = rd // 2
    cos = jnp.repeat(cs[:, :half], 2, axis=-1)
    sin = jnp.repeat(cs[:, half:], 2, axis=-1)
    if inverse:
        sin = -sin
    pad = LANE - rd
    n = positions.shape[0]
    cos = jnp.concatenate([jnp.ones((n, pad), F32), cos], axis=-1)
    sin = jnp.concatenate([jnp.zeros((n, pad), F32), sin], axis=-1)
    return jnp.concatenate([cos, sin], axis=-1)


# ---------------------------------------------------------------------------
# kernel
# ---------------------------------------------------------------------------
def _scratch(scfg: StepKernelConfig, typed: bool):
    """name -> scratch shape (dense chain buffers first, as dense_ring)."""
    cfg = scfg.dense
    T, h, st = scfg.T, cfg.hidden, cfg.streams
    names = cfg.stream_names
    kmax = max(cfg.hidden, cfg.groups * cfg.o_lora, cfg.inter,
               cfg.heads_per_group * cfg.head_dim)  # fmt: skip
    nmax = max(s.n for s in st.values() if not s.nt)
    d = cfg.head_dim
    scr = {}
    if typed:
        for n in names:
            s = st[n]
            scr["pool_" + n] = pltpu.VMEM(
                (cfg.n_banks, s.tile_rows, s.tile_cols), jnp.dtype(s.dtype)
            )
    else:
        scr["pool"] = pltpu.VMEM(
            (cfg.n_banks * cfg.max_tile_bytes // 128, 128), jnp.uint8
        )
    scr.update(
        dsem=pltpu.SemaphoreType.DMA((cfg.n_banks,)),
        sm_v=pltpu.VMEM((2, cfg.smalls_rows, LANE), F32),
        ssem=pltpu.SemaphoreType.DMA((2,)),
        res=pltpu.VMEM((T, cfg.hc_hidden), BF16),
        mixst=pltpu.VMEM((T, 2 * LANE), F32),
        tsc=pltpu.VMEM((LANE, LANE), F32),
        xin=pltpu.VMEM((T, h), BF16),
        hin=pltpu.VMEM((T, h), BF16),
        xa=pltpu.VMEM((T, h), BF16),
        xq=pltpu.VMEM((T, kmax), FP8),
        acc=pltpu.VMEM((T, nmax), F32),
        qkv=pltpu.VMEM((T, max(cfg.qkv_cols, 2 * cfg.inter)), BF16),
        qr=pltpu.VMEM((T, cfg.q_lora), BF16),
        q=pltpu.VMEM((T, cfg.heads * d), BF16),
        xg=pltpu.VMEM((T, max(cfg.heads_per_group * d, d)), BF16),
        z=pltpu.VMEM((T, cfg.groups * cfg.o_lora), BF16),
        act=pltpu.VMEM((T, cfg.inter), BF16),
        # step-only buffers
        kvn=pltpu.VMEM((T, d), F32),  # kv_norm, then kv rope (bf16 values)
        **(
            {"ctx_raw": pltpu.VMEM((T, d), F32), "ctx_kvn": pltpu.VMEM((T, d), F32)}
            if scfg.ctx_q
            else {}
        ),  # the draft context rows' kv: projection, then kv_norm + rope
        xp=pltpu.VMEM((T, max(cfg.xproj_cols.values())), F32),
        idxq=pltpu.VMEM((T, cfg.idx_heads * cfg.idx_head_dim), BF16),
        logits=pltpu.VMEM((T, cfg.n_experts), F32),
        # f32 so a token's hpr(=8)-head block is one (8, 128)-tile row group:
        # bf16 8-row blocks at 8-row offsets are half tiles, which a DMA
        # cannot address. Values stay bf16-exact.
        qs=pltpu.VMEM((T, scfg.hpr, d), F32),
        # bf16 (the values are bf16-exact); rows t * NA + head, so the heads
        # of two adjacent ranks fill one bf16 (16, 128) tile
        q_in=pltpu.VMEM((scfg.B, scfg.RP, d), BF16),
        og=pltpu.VMEM((T, scfg.hpr, d), F32),
        # Both all-to-alls travel as bf16 (the values are bf16-exact): one
        # [QP * hpr, d] block (rows t * hpr + j, row pairs fill whole bf16
        # tiles) per (request, rank) instead of one f32 [hpr, d] DMA per
        # (row, rank) -- half the bytes, 1/q the DMAs.
        qs_b=pltpu.VMEM((T * scfg.hpr, d), BF16),  # A2A#1 send
        a2a_rx=pltpu.VMEM((scfg.B, scfg.G, scfg.QP * scfg.hpr, d), BF16),
        # A2A#2 receive, one [QP * hpr, d] block per global request. Never
        # a2a_rx: a peer may still be unpacking its A2A#1 heads from there
        # when an owner's A2A#2 lands.
        og_b=pltpu.VMEM((T // scfg.QP, scfg.QP * scfg.hpr, d), BF16),
        part=pltpu.VMEM((T, h), F32),  # wo_b / shared-expert partials
        post_st=pltpu.VMEM((16, cfg.hc_hidden), BF16),
        a2a_sem=pltpu.SemaphoreType.DMA((2, 2)),  # [A2A#1, #2] x [send, recv]
        osem=pltpu.SemaphoreType.DMA((1,)),
        fence=pltpu.SMEM((1,), jnp.int32),
        li_ref=pltpu.SMEM((1,), jnp.int32),
        part_dbg=pltpu.VMEM((T, h) if scfg.debug_att else (8, LANE), F32),
    )
    for k, v in scfg.ar_spec().scratch().items():
        scr[k] = v
    if scfg.attn_impl == "real":
        B = scfg.B
        scr.update(
            at_kv=pltpu.VMEM((B, 8, d), BF16),
            at_comp=pltpu.VMEM((B, 8, 4 * d), F32),
            at_idxq=pltpu.VMEM((B, 8, cfg.idx_heads * cfg.idx_head_dim), BF16),
            at_idxw=pltpu.VMEM((B, 8, LANE), F32),
            at_idxc=pltpu.VMEM((B, 8, 4 * cfg.idx_head_dim), F32),
        )
        if scfg.ctx_q:  # the draft step's context rows for the swa phase
            scr.update(
                at_ctx=pltpu.VMEM((B, 8, d), BF16),
                at_ctxpos=pltpu.SMEM((B * 8,), jnp.int32),
            )
    return scr


def _kernel(scfg: StepKernelConfig, typed: bool, ax: str, names_scr, n_moe_scr,
            alay: AttnLayout | None, n_attn_scr, *refs):  # fmt: skip
    cfg = scfg.dense
    names = cfg.stream_names
    T, q, B, R = scfg.T, scfg.q, scfg.B, scfg.R
    h, hcm, d, hpr = cfg.hidden, cfg.hc_mult, cfg.head_dim, scfg.hpr
    L = cfg.n_layers
    (sched, base, ltype, tokm, ownt, counts, x0, cs_f, cs_i, smalls_hbm) = refs[:10]
    tiles = dict(zip(names, refs[10 : 10 + len(names)]))
    n_in = 10 + len(names)
    if scfg.ctx_q:  # main_x [T, H] bf16, context rope rows [T, 256], positions
        main_x, cs_ctx, ctx_pos = refs[n_in : n_in + 3]
        n_in += 3
    moe = scfg.moe_impl == "real"
    held = copy = None
    if moe:
        hash_ids, row_valid, eblk, moe_bias = refs[n_in : n_in + 4]
        moe_w = refs[n_in + 4 : n_in + 4 + 4 * L]
        n_in += 4 + 4 * L
        if scfg.moe_placement:
            held = refs[n_in]
            n_in += 1
        if scfg.moe_placement == "replicas":
            copy = refs[n_in]
            n_in += 1
    attn = alay is not None
    n_arr = len(alay.arrays) if attn else 0
    if attn:
        a_smem = refs[n_in : n_in + 20]
        a_w = refs[n_in + 20 : n_in + 28]
        n_in += 28 + n_arr
    res_out, aux_out = refs[n_in : n_in + 2]
    if scfg.cache_inplace:
        cache_out = refs[n_in - n_arr : n_in]
        n_arr = 0  # no cache outputs
    else:  # the cache inputs are aliased to these outputs
        cache_out = refs[n_in + 2 : n_in + 2 + n_arr]
    n_dbg = scfg.n_debug_outputs
    dbg_out = refs[n_in + 2 + n_arr] if n_dbg else None
    dbg_x = refs[n_in + 3 + n_arr] if n_dbg else None  # f32 [3, T, H] layer 0
    dbg_ctx = refs[n_in + 4 + n_arr] if n_dbg == 3 else None  # f32 [T, d] layer 0
    scr_refs = refs[n_in + 2 + n_arr + n_dbg :]
    s = dict(zip(names_scr, scr_refs[: len(names_scr)]))
    moe_scr = scr_refs[len(names_scr) : len(names_scr) + n_moe_scr]
    attn_scr = scr_refs[len(names_scr) + n_moe_scr :]
    assert len(attn_scr) == n_attn_scr
    me = lax.axis_index(ax)
    geom = scfg.geometry()

    def dbg_x_store(k, src):
        """Layer 0 debug: src (all T rows) -> dbg_x[k]."""
        if dbg_x is None:
            return

        @pl.when(s["li_ref"][0] == 0)
        def _():
            s["part_dbg"][...] = src[...].astype(F32)
            cp = pltpu.make_async_copy(s["part_dbg"], dbg_x.at[k], s["osem"].at[0])
            cp.start()
            cp.wait()

    lease_rows = cfg.n_banks * cfg.max_tile_bytes // 128
    faults = DR.Faults(
        no_dma="dense_dma" in scfg.skip,
        skip=frozenset(x[3:] for x in scfg.skip if x.startswith("dr_")),
    )
    skip = scfg.skip
    pool = {n: s["pool_" + n] for n in names} if typed else s["pool"]
    # the phases borrow the whole pool (the ring is fenced around them)
    attn_lease = SP.PoolLease(None if typed else pool, 0, 0 if typed else lease_rows)
    ring = DR._Ring(cfg, sched, tiles, pool, s["dsem"], typed, faults, s["fence"])

    og = s["og"]

    def head_src(gi, hh, lo, n):  # o-group gi is always this rank's (gi == 0)
        del gi
        return og[:, hh, pl.ds(lo, n)].astype(BF16)

    ops = DR.make_dense_ops(
        cfg, ring, faults, sm_v=s["sm_v"], res=s["res"], mixst=s["mixst"],
        tsc=s["tsc"], xin=s["xin"], hin=s["hin"], xa=s["xa"], xq=s["xq"],
        acc=s["acc"], q=s["q"], xg=s["xg"], z=s["z"], cos_sin=cs_i,
        head_src=head_src,
    )  # fmt: skip
    eps = cfg.rms_eps
    lane = lax.broadcasted_iota(jnp.int32, (T, LANE), 1)
    even = lane % 2 == 0

    def rot(x):  # rope.py _rotate_gptj on 128 lanes
        return jnp.where(even, -pltpu.roll(x, LANE - 1, 1), pltpu.roll(x, 1, 1))

    # ---- own heads: qnorm + rope (stock deepseek_v4_qnorm_rope) -----------
    def qnorm_rope_heads():
        cos, sin = cs_f[:, :LANE], cs_f[:, LANE:]
        for hh in range(hpr):
            v = s["q"][:, hh * d : (hh + 1) * d].astype(F32)
            y = v * lax.rsqrt(jnp.mean(v * v, axis=1, keepdims=True) + eps)
            tail = y[:, d - LANE :]
            # stock rounding: bf16 output (held as f32 for the all-to-all)
            if d > LANE:
                s["qs"][:, hh, : d - LANE] = y[:, : d - LANE].astype(BF16).astype(F32)
            roped = (tail * cos + rot(tail) * sin).astype(BF16)
            s["qs"][:, hh, d - LANE :] = roped.astype(F32)

    # ---- the two all-to-alls -------------------------------------------------
    sem = s["a2a_sem"]

    def a2a1_desc(src, dst, dev):
        return pltpu.make_async_remote_copy(
            src_ref=src, dst_ref=dst, send_sem=sem.at[0, 0], recv_sem=sem.at[0, 1],
            device_id=(dev,), device_id_type=pl.DeviceIdType.MESH,
        )  # fmt: skip

    def a2a2_desc(src, dst, dev):
        return pltpu.make_async_remote_copy(
            src_ref=src, dst_ref=dst, send_sem=sem.at[1, 0], recv_sem=sem.at[1, 1],
            device_id=(dev,), device_id_type=pl.DeviceIdType.MESH,
        )  # fmt: skip

    def times(n, fn):
        lax.fori_loop(0, n, lambda i, c: (fn(), c)[1], 0)

    QP = scfg.QP
    blk = QP * hpr
    G, GS, NA = scfg.G, scfg.attn_tp, scfg.NA
    tj = lax.rem(me, GS)  # rank within the attention group

    def a2a_1():
        """qs[g] (this rank's heads of global row g) -> a2a_rx[r, src] of the
        rank of the owner group attending them (bf16 rows t*hpr + j), one DMA
        per global request; src = this rank's group."""
        qsb = s["qs_b"]
        for u in range(T // 2):
            pair = jnp.concatenate([s["qs"][2 * u], s["qs"][2 * u + 1]])
            qsb[2 * u * hpr : (2 * u + 2) * hpr, :] = pair.astype(BF16)
        src = me // GS
        slot_rank = lax.rem(src + 1, G) if scfg.fault == "a2a_heads" else src

        def send(gr, c):
            g0 = gr * QP
            owner = tokm[2 * T + g0]

            @pl.when(owner >= 0)
            def _():
                r = tokm[3 * T + g0] // q
                blk_src = qsb.at[pl.ds(pl.multiple_of(g0 * hpr, blk), blk)]
                dst = s["a2a_rx"].at[r, slot_rank]
                a2a1_desc(blk_src, dst, owner * GS + tj).start()

            return c

        lax.fori_loop(0, T // QP, send, 0)

    def a2a_1_wait():
        """Land every source's heads of the own requests, widened into
        q_in[r, t*NA + src*hpr : +hpr]."""
        rx = s["a2a_rx"]
        dummy = a2a1_desc(s["qs_b"].at[pl.ds(0, blk)], rx.at[0, 0], me)
        times(counts[3] * G, dummy.wait_recv)
        times(counts[2], dummy.wait_send)
        for r in range(B):

            @pl.when(ownt[r * q] >= 0)
            def _(r=r):
                def unpack(sp, c):  # sources 2 sp, 2 sp + 1: one q_in tile per row
                    for u in range(QP // 2):
                        rr = slice(2 * u * hpr, (2 * u + 2) * hpr)
                        a = rx[r, 2 * sp, rr, :].astype(F32)
                        b = rx[r, 2 * sp + 1, rr, :].astype(F32)
                        for k in range(2):
                            t = 2 * u + k
                            if t < q:
                                kk = slice(k * hpr, (k + 1) * hpr)
                                row = pl.multiple_of(t * NA + 2 * sp * hpr, 2 * hpr)
                                s["q_in"][r, pl.ds(row, 2 * hpr), :] = jnp.concatenate(
                                    [a[kk], b[kk]]
                                ).astype(BF16)
                    return c

                lax.fori_loop(0, G // 2, unpack, 0)

    def a2a_2():
        """att[r, t*NA + g*hpr : +hpr] of own row t of request slot r -> source
        g's (rank g * GS + tj) og[global row, :], packed per (request, source)
        as bf16 rows t*hpr + j of a2a2_stg[g] and landed in og_b, then widened
        into og."""
        # this rank has unpacked its own A2A#1 heads: its slot 0 is free
        stg = s["a2a_rx"].at[0]
        ogb = s["og_b"]
        dummy = a2a2_desc(stg.at[0], ogb.at[0], me)
        for r in range(B):
            gk0 = ownt[r * q]  # global row of the request's first row

            @pl.when(gk0 >= 0)
            def _(r=r, gk0=gk0):
                def pack_send(gp, c):  # sources 2 gp, 2 gp + 1
                    def rows(t):  # [2 hpr, d]: source 2 gp's heads, then 2 gp + 1's
                        if t >= q:
                            return jnp.zeros((2 * hpr, d), F32)
                        row = pl.multiple_of(t * NA + 2 * gp * hpr, 2 * hpr)
                        return s["q_in"][r, pl.ds(row, 2 * hpr), :].astype(F32)

                    for u in range(QP // 2):  # two rows = one bf16 tile
                        x0, x1 = rows(2 * u), rows(2 * u + 1)
                        rr = slice(2 * u * hpr, (2 * u + 2) * hpr)
                        for k in range(2):
                            kk = slice(k * hpr, (k + 1) * hpr)
                            stg[2 * gp + k, rr, :] = jnp.concatenate(
                                [x0[kk], x1[kk]]
                            ).astype(BF16)
                    dst = ogb.at[gk0 // QP]
                    for k in range(2):
                        a2a2_desc(
                            stg.at[2 * gp + k], dst, (2 * gp + k) * GS + tj
                        ).start()
                    return c

                lax.fori_loop(0, G // 2, pack_send, 0)
                times(G, dummy.wait_send)  # a2a2_stg is reused by the next slot

        times(counts[2], dummy.wait_recv)
        og[...] = ogb[0 : T // QP].astype(F32).reshape(T, hpr, d)

    # ---- owner attention phase ---------------------------------------------------
    if attn:
        meta = SP.AttnMeta(
            num_req=a_smem[0], seq_lens=a_smem[1], positions=a_smem[2],
            bt_flat=a_smem[3], bt_off=dict(zip(META_ROLES, a_smem[4:11])),
            bt_w=dict(zip(META_ROLES, a_smem[11:18])),
            ctx_positions=s["at_ctxpos"] if scfg.ctx_q else None,
        )  # fmt: skip
        own_greq, csa_ord = a_smem[18], a_smem[19]
        SP.reset_table_stage(geom, attn_scr, not typed)
        caches = SP.AttnCaches(**{
            role: SP.LayerRefs([cache_out[i] for i in alay.cache_index[k]],
                               alay.cache_layers[k])
            for k, role in enumerate(ROLES)
        })  # fmt: skip
        aweights = SP.AttnWeights(*a_w)

    def kv_rope(kv=None, cs=None):
        """stock deepseek_v4_kv_rope on the kv_norm rows (bf16 rounding)."""
        kv, cs = s["kvn"] if kv is None else kv, cs_f if cs is None else cs
        cos, sin = cs[:, :LANE], cs[:, LANE:]
        tail = kv[:, d - LANE :]
        kv[:, d - LANE :] = (tail * cos + rot(tail) * sin).astype(BF16).astype(F32)

    def attention_real(lt, li):
        """Gather the own requests' rows (QP-row global blocks) into the phase's
        inputs, then run it; its output overwrites q_in in place."""
        if lt == SWA:
            return attention_swa(li)
        csa = lt == CSA
        nc = 4 * d if csa else 2 * d
        c0 = cfg.wp_cols + 4 * cfg.idx_head_dim if csa else 0
        for r in range(B):
            gr = own_greq[r]

            @pl.when(gr >= 0)
            def _(r=r, gr=gr):
                rows = pl.ds(pl.multiple_of(gr * scfg.QP, 8), 8)
                s["at_kv"][r] = s["kvn"][rows, :].astype(BF16)
                s["at_comp"][r, :, :nc] = s["xp"][rows, c0 : c0 + nc]
                if csa:
                    s["at_idxw"][r] = s["xp"][rows, 0:LANE]
                    s["at_idxc"][r] = s["xp"][rows, cfg.wp_cols : c0]
                    # idxq is bf16: load the aligned 16-row pair of request
                    # blocks and keep this request's half
                    pair = s["idxq"][pl.ds(pl.multiple_of(lax.div(gr, 2) * 16, 16), 16)]
                    pair = pair.astype(F32)
                    odd = lax.rem(gr, 2) == 1
                    s["at_idxq"][r] = jnp.where(odd, pair[8:], pair[:8]).astype(BF16)

        inputs = SP.AttnInputs(
            q=s["q_in"], kv=s["at_kv"], comp=s["at_comp"].at[:, :, pl.ds(0, nc)],
            idx_q=s["at_idxq"] if csa else None, idx_w=s["at_idxw"] if csa else None,
            idx_comp=s["at_idxc"] if csa else None,
        )  # fmt: skip
        SP.owner_attention_phase(
            geom, "csa" if csa else "hca", li, csa_ord[li], meta, inputs, caches,
            aweights, s["q_in"], attn_lease, attn_scr, me=me,
        )  # fmt: skip

    def attention_swa(li):
        """The DSpark draft layer: the draft rows' kv and the context rows' kv
        (ctx_pass) of each own request, then the swa phase."""
        for r in range(B):
            gr = own_greq[r]
            for c in range(8):
                s["at_ctxpos"][r * 8 + c] = jnp.int32(-1)

            @pl.when(gr >= 0)
            def _(r=r, gr=gr):
                rows = pl.ds(pl.multiple_of(gr * scfg.QP, 8), 8)
                s["at_kv"][r] = s["kvn"][rows, :].astype(BF16)
                s["at_ctx"][r] = s["ctx_kvn"][rows, :].astype(BF16)
                for c in range(min(8, scfg.ctx_q)):
                    s["at_ctxpos"][r * 8 + c] = ctx_pos[gr * scfg.QP + c]

        inputs = SP.AttnInputs(
            q=s["q_in"], kv=s["at_kv"], comp=None, idx_q=None, idx_w=None,
            idx_comp=None, ctx_kv=s["at_ctx"],
        )  # fmt: skip
        SP.owner_attention_phase(
            geom, "swa", li, 0, meta, inputs, caches, aweights, s["q_in"],
            attn_lease, attn_scr, me=me,
        )  # fmt: skip

    # ---- phases (stubs) -------------------------------------------------------
    def attention_stub():
        """Per-row RMSNorm of the gathered q (the dense_reference stub)."""
        for r in range(B):
            for c0 in range(0, scfg.RP, 64):
                v = s["q_in"][r, c0 : c0 + 64, :].astype(F32)
                y = v * lax.rsqrt(jnp.mean(v * v, axis=1, keepdims=True) + eps)
                s["q_in"][r, c0 : c0 + 64, :] = y.astype(BF16)

    spec = scfg.ar_spec()
    # One all-reduce scratch: AR#1 always uses slot 0, AR#2 slot 1, so
    # consecutive all-reduces alternate slots (the cross-rank WAR rule).
    ar = AR.AllReduce(spec, {k: v for k, v in s.items() if k.startswith("ar_")}, me)

    def allreduce(li, slot):
        if "ar" in skip:
            s["xin"][...] = s["part"][...].astype(BF16)
            return

        @pl.when(li >= 1)
        def _():
            ar.wait_sends(slot)  # this slot's previous use (one layer back)

        s["xin"][...] = ar(s["part"][...], slot).astype(BF16)

    # ---- post only (aux layers, final) ---------------------------------------
    def post_rows_to(dst_hbm):
        """dst_hbm[T, 4H] <- the residual streams after the pending post
        (mhc_post), 16 global rows at a time through `post_st`."""
        post = [s["mixst"][:, DR.POST0 + j : DR.POST0 + j + 1] for j in range(hcm)]
        comb = [
            s["mixst"][:, DR.COMB0 + k : DR.COMB0 + k + 1] for k in range(hcm * hcm)
        ]
        ew = min(DR.EW, h)
        for r0 in range(0, T, 16):
            rows = slice(r0, r0 + 16)

            def body(ci, c, rows=rows):
                c0 = pl.multiple_of(ci * ew, ew)
                xv = s["xin"][rows, pl.ds(c0, ew)].astype(F32)
                old = [s["res"][rows, pl.ds(i * h + c0, ew)].astype(F32)
                       for i in range(hcm)]  # fmt: skip
                for j in range(hcm):
                    a = post[j][rows] * xv
                    for i in range(hcm):
                        a = a + comb[i * hcm + j][rows] * old[i]
                    s["post_st"][:, pl.ds(j * h + c0, ew)] = a.astype(BF16)
                return c

            lax.fori_loop(0, h // ew, body, 0)
            cp = pltpu.make_async_copy(
                s["post_st"], dst_hbm.at[pl.ds(r0, 16)], s["osem"].at[0]
            )
            cp.start()
            cp.wait()

    # ---- one layer -------------------------------------------------------------
    xin, hin, xa, qkv, acc = s["xin"], s["hin"], s["xa"], s["qkv"], s["acc"]

    kinds = [k for k in (CSA, HCA, SWA) if k in cfg.layer_types]
    assert len(kinds) <= 2, "the layer body branches between at most two kinds"

    def layer(li, slot):
        """One decoder layer, traced once for every kind: the code shared by
        CSA and HCA layers exists once (kernel code size is a cost: see
        B6_step.md), only the kind-specific parts branch."""
        is_k0 = ltype[li] == kinds[0]
        g_l = base[li]

        def g(name, lt=None):
            """First global tile of stream `name` in this layer (per-kind
            offsets selected at run time unless `lt` is given)."""
            if lt is not None:
                return g_l + cfg.stream_offsets[lt][name]
            offs = {cfg.stream_offsets[k][name] for k in kinds}
            if len(offs) == 1:
                return g_l + offs.pop()
            k0, k1 = kinds
            return g_l + jnp.where(
                is_k0, cfg.stream_offsets[k0][name], cfg.stream_offsets[k1][name]
            )

        def by_kind(fn):
            if len(kinds) == 1:
                fn(kinds[0])
                return
            pl.when(is_k0)(lambda: fn(kinds[0]))
            pl.when(jnp.logical_not(is_k0))(lambda: fn(kinds[1]))

        def emit(dst, col0=0):
            def f(c0, cw, out):
                dst[:, pl.ds(c0 + col0, cw)] = out.astype(dst.dtype)

            return f

        # The phases borrow the whole pool, so the ring runs between them:
        # restart at the layer's first tile, fenced before attention (layers
        # after the first were restarted before the previous layer's last
        # all-reduce, see prefetch_next) ...
        if "fence" not in skip:

            @pl.when(li == 0)
            def _():
                with jax.named_scope("mk_restart"):
                    ring.restart(g_l, g("wo_a"))
        else:

            @pl.when(li == 0)
            def _():
                ring.restart(0, cfg.n_tiles_total)

        sc = jax.named_scope

        def ctx_pass(lt):
            """DSpark context KV: kv columns of fused_wqa_wkv on main_x (no
            mHC, no attn_norm), kv_norm, rope at the context positions."""
            if lt != SWA or not scfg.ctx_q:
                return
            ops.fp8_linear(
                "wqa_ctx", g("wqa_ctx", lt), main_x, slot, emit(s["ctx_raw"])
            )
            ops.rms_norm_to(
                s["ctx_raw"], 0, d, s["ctx_kvn"], slot, "kv_norm", weight_dtype=BF16
            )
            kv_rope(s["ctx_kvn"], cs_ctx)
            if dbg_ctx is not None:

                @pl.when(li == 0)
                def _():
                    cp = pltpu.make_async_copy(s["ctx_kvn"], dbg_ctx, s["osem"].at[0])
                    cp.start()
                    cp.wait()

        with sc("mk_ctx_kv"):
            by_kind(ctx_pass)
        with sc("mk_attn_in"):
            with sc("mk_mhc_attn"):
                sk, ss = ops.mhc(
                    "hc_attn", g("hc_attn"), slot, "hc_attn_base", "hc_attn_scale"
                )
            with sc("mk_wqa"):
                amax = ops.rms_norm_to(
                    hin, 0, h, xa, slot, "attn_norm", ss, weight_dtype=F32
                )
                ops.finish_sinkhorn(
                    ops.fp8_linear(
                        "wqa", g("wqa"), xa, slot, emit(qkv), bg=sk, amax=amax
                    )
                )
            with sc("mk_wq_b"):
                amax_q = ops.rms_norm_to(
                    qkv, 0, cfg.q_lora, s["qr"], slot, "q_norm", weight_dtype=BF16
                )
                ops.rms_norm_to(
                    qkv, cfg.q_lora, d, s["kvn"], slot, "kv_norm", weight_dtype=BF16
                )
                if attn:
                    kv_rope()
                ops.fp8_linear(
                    "wq_b", g("wq_b"), s["qr"], slot, emit(s["q"]), amax=amax_q
                )
                qnorm_rope_heads()
            with sc("mk_a2a1_send"):
                if "a2a" not in skip:
                    a2a_1()  # sends only: they fly while the projections below stream

        def xproj(lt):
            if lt == SWA:  # no compressor / indexer on sliding-window layers
                return
            xname = "xproj_csa" if lt == CSA else "xproj_hca"
            ops.matmul(xname, g(xname, lt), lambda k0, w: xa[:, pl.ds(k0, w)])
            nx = cfg.xproj_cols[lt]
            s["xp"][:, :nx] = acc[:, :nx]
            if lt == CSA:
                ops.fp8_linear(
                    "idx_wq_b", g("idx_wq_b", lt), s["qr"], slot, emit(s["idxq"]),
                    amax=amax_q,
                )  # fmt: skip

        with jax.named_scope("mk_xproj"):
            by_kind(xproj)
        with jax.named_scope("mk_a2a1_wait"):
            if "a2a" not in skip:
                a2a_1_wait()
        with jax.named_scope("mk_attn"):
            if attn:
                by_kind(lambda lt: attention_real(lt, li))
            else:
                attention_stub()
        if dbg_out is not None:

            @pl.when(li == 0)
            def _():
                cp = pltpu.make_async_copy(s["q_in"], dbg_out, s["osem"].at[0])
                cp.start()
                cp.wait()

        # ... and restart after it, fenced before the next layer (MoE phase)
        nxt = jnp.where(li + 1 < L, base[jnp.minimum(li + 1, L - 1)], cfg.n_tiles_total)

        def prefetch_next():
            """The next layer's head streams under this layer's last
            all-reduce (the pool is idle there): restart at its first tile,
            fenced before its attention."""
            if "fence" in skip:
                return
            nl = jnp.minimum(li + 1, L - 1)
            wo_a = [cfg.stream_offsets[k]["wo_a"] for k in kinds]
            off = wo_a[0] if len(kinds) == 1 else jnp.where(
                ltype[nl] == CSA, cfg.stream_offsets[CSA]["wo_a"],
                cfg.stream_offsets[HCA]["wo_a"],
            )  # fmt: skip
            with jax.named_scope("mk_restart"):
                ring.restart(nxt, jnp.where(li + 1 < L, nxt + off, cfg.n_tiles_total))

        with jax.named_scope("mk_restart"):
            if "fence" not in skip:
                ring.restart(g("wo_a"), nxt)
        with sc("mk_wo"):
            with sc("mk_a2a2"):
                if "a2a" not in skip:
                    a2a_2()
            with sc("mk_wo_a"):
                ops.wo_a(g("wo_a"), slot)
            with sc("mk_wo_b"):
                ops.fp8_linear("wo_b", g("wo_b"), s["z"], slot, emit(s["part"]))
        s["li_ref"][0] = li
        dbg_x_store(0, s["part"])
        with jax.named_scope("mk_ar1"):
            allreduce(li, 0)
        dbg_x_store(1, s["xin"])

        with sc("mk_ffn_dense"):
            with sc("mk_mhc_ffn"):
                sk, ss = ops.mhc(
                    "hc_ffn", g("hc_ffn"), slot, "hc_ffn_base", "hc_ffn_scale"
                )
            with sc("mk_gate"):
                amax_f = ops.rms_norm_to(
                    hin, 0, h, xa, slot, "ffn_norm", ss, weight_dtype=F32
                )
                ops.finish_sinkhorn(
                    ops.matmul("gate", g("gate"), lambda k0, w: xa[:, pl.ds(k0, w)], sk)
                )
                s["logits"][...] = acc[:, : cfg.n_experts]
            inter = cfg.inter

            def shared_expert_fn():
                with sc("mk_w13"):
                    ops.fp8_linear("w13", g("w13"), xa, slot, emit(qkv), amax=amax_f)
                for c, w in DR._chunks(inter, DR.EW):
                    gt = jnp.minimum(qkv[:, c : c + w].astype(F32), cfg.swiglu_limit)
                    up = jnp.clip(
                        qkv[:, inter + c : inter + c + w].astype(F32),
                        -cfg.swiglu_limit,
                        cfg.swiglu_limit,
                    )
                    s["act"][:, c : c + w] = (gt * jax.nn.sigmoid(gt) * up).astype(BF16)
                ops.fp8_linear("w2", g("w2"), s["act"], slot, emit(s["part"]))

            if not moe:
                shared_expert_fn()
        if moe:
            leased = scfg.moe_leased and not typed
            lease = SP.PoolLease(
                pool if leased else None, 0, lease_rows if leased else 0
            )
            is_hash = (
                False if geom.num_hash_layers == 0 else (li < geom.num_hash_layers)
            )
            mw_refs = (
                tuple(moe_w)
                if len(moe_w) == 4
                else tuple(SP.LayerRefs(moe_w[j::4]) for j in range(4))
            )
            SP.moe_phase(
                geom, li, is_hash, me,
                SP.MoEInputs(xa, s["logits"], s["part"], hash_ids, row_valid, amax=amax_f),
                SP.MoEWeights(*mw_refs, moe_bias),
                xin, lease, moe_scr, ar=ar, ar_slot=1, expert_block=eblk[0],
                before_ar=prefetch_next, expert_held=held, expert_copy=copy,
                shared_expert=shared_expert_fn,
            )  # fmt: skip
        else:
            prefetch_next()
            with jax.named_scope("mk_ar2"):
                allreduce(li, 1)  # MoE stub: the shared-expert partials only
        dbg_x_store(2, s["xin"])

    # ---- prologue ----------------------------------------------------------------
    # Zero the remote-landing buffers before the barrier: past it, a fast peer
    # may already be sending its layer-0 A2A#1 heads here.
    s["q_in"][...] = jnp.zeros(s["q_in"].shape, BF16)
    s["og"][...] = jnp.zeros(s["og"].shape, F32)
    s["og_b"][...] = jnp.zeros(s["og_b"].shape, BF16)
    # every block og may read is finite (inactive requests' blocks are stale)
    s["a2a_rx"][...] = jnp.zeros(s["a2a_rx"].shape, BF16)
    AR.barrier(R, me)

    def smalls_copy(li, slot):
        return pltpu.make_async_copy(
            smalls_hbm.at[li], s["sm_v"].at[slot], s["ssem"].at[slot]
        )

    smalls_copy(0, 0).start()
    x0v = x0[...]
    xin[...] = jnp.zeros((T, h), BF16)
    for j in range(hcm):
        s["res"][:, j * h : (j + 1) * h] = x0v
    mixst = s["mixst"]
    mixst[...] = jnp.zeros(mixst.shape, F32)
    for j in range(hcm):  # post = 0, comb = I: layer 0's fused post is a no-op
        mixst[:, DR.COMB0 + j * hcm + j : DR.COMB0 + j * hcm + j + 1] = jnp.ones(
            (T, 1), F32
        )

    def layer_body(li, c):
        slot = lax.rem(li, 2)
        smalls_copy(0, slot).wait()

        @pl.when(li + 1 < L)
        def _():
            smalls_copy(li + 1, 1 - slot).start()

        layer(li, slot)
        for k, al in enumerate(scfg.aux_layers):

            @pl.when(li == al)
            def _(k=k):
                post_rows_to(aux_out.at[k])

        return c

    lax.fori_loop(0, L, layer_body, 0)
    if "ar" not in skip:
        ar.wait_sends(0)  # the last AR#1
        if not moe:
            ar.wait_sends(1)
    post_rows_to(res_out)


def build(scfg: StepKernelConfig, *, axis: str, interpret=False):
    """Per-rank step fn (call under shard_map over `axis`):

    f(input_ids [M], positions [M], seq_lens [B], embed [V, H], cos_sin_cache
      [ceil(max_pos / 2), 128] (SP.pack_cos_sin), smalls, tiles: dict) ->
      (res_final [M, 4H] bf16,
      aux_res [A, M, 4H] bf16, tokm, ownt, counts)
    """
    assert not (scfg.cache_inplace and interpret), "interpret copies its inputs"
    cfg = scfg.dense
    names = cfg.stream_names
    typed = bool(interpret)
    scr = _scratch(scfg, typed)
    names_scr = list(scr)
    moe = scfg.moe_impl == "real"
    leased = scfg.moe_leased and not typed
    # the MoE phase all-reduces on the skeleton's AR scratch (slot 1)
    moe_scr = (
        SP.moe_scratch_shapes(scfg.geometry(), leased, own_ar=False) if moe else []
    )
    L = cfg.n_layers
    sched, base = cfg.schedule
    ltype = jnp.asarray(cfg.layer_types, jnp.int32)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    anyspace = pl.BlockSpec(memory_space=pl.ANY)
    A = max(1, len(scfg.aux_layers))
    vmem_limit = None if typed else 64 * 2**20

    def make_call(alay: AttnLayout | None):
        # the attention phase borrows the dense pool (idle while it runs)
        attn_scr = (
            SP.attention_scratch_shapes(scfg.geometry(), not typed) if alay else []
        )
        n_arr = len(alay.arrays) if alay else 0
        in_specs = (
            [smem] * 6 + [vmem] * 3 + [anyspace] * (1 + len(names))
            + ([vmem, vmem, smem] if scfg.ctx_q else [])
            + ([smem] * 3 + [anyspace] * (1 + 4 * L) if moe else [])
            + [smem] * (moe and {"": 0, "perm": 1, "replicas": 2}[scfg.moe_placement])
            # bt_flat (operand 3 of 20) stays in HBM: the phase stages its tables
            + ([smem] * 3 + [anyspace] + [smem] * 16 + [anyspace] * (8 + n_arr)
               if alay else [])
        )  # fmt: skip
        n_in = len(in_specs)
        arr_shapes = [jax.ShapeDtypeStruct(sh, jnp.dtype(dt)) for sh, dt in
                      (alay.arrays if alay else ())]  # fmt: skip
        n_out_arr = 0 if scfg.cache_inplace else n_arr
        return pl.pallas_call(
            functools.partial(
                _kernel, scfg, typed, axis, names_scr, len(moe_scr), alay, len(attn_scr)
            ),  # fmt: skip
            in_specs=in_specs,
            out_specs=[anyspace] * (2 + n_out_arr + scfg.n_debug_outputs),
            out_shape=[
                jax.ShapeDtypeStruct((scfg.T, cfg.hc_hidden), BF16),
                jax.ShapeDtypeStruct((A, scfg.T, cfg.hc_hidden), BF16),
                *arr_shapes[:n_out_arr],
            ]
            + (
                [
                    jax.ShapeDtypeStruct((scfg.B, scfg.RP, cfg.head_dim), BF16),
                    jax.ShapeDtypeStruct((3, scfg.T, cfg.hidden), F32),
                ][: scfg.n_debug_outputs]
                + (
                    [jax.ShapeDtypeStruct((scfg.T, cfg.head_dim), F32)]
                    if scfg.n_debug_outputs == 3
                    else []
                )
            ),  # fmt: skip
            input_output_aliases={n_in - n_arr + i: 2 + i for i in range(n_out_arr)},
            scratch_shapes=[scr[k] for k in names_scr] + list(moe_scr) + list(attn_scr),
            compiler_params=pltpu.CompilerParams(
                collective_id=AR.COLLECTIVE_ID, vmem_limit_bytes=vmem_limit
            ),
            interpret=(
                interpret
                if isinstance(interpret, pltpu.InterpretParams)
                else (pltpu.InterpretParams() if interpret else False)
            ),
            name="dsv4_step",
        )

    sched_j, base_j = jnp.asarray(sched), jnp.asarray(base)

    csa_ord = np.cumsum([0] + [t == CSA for t in cfg.layer_types])[:-1].astype(np.int32)

    def step(input_ids, positions, seq_lens, embed, cos_sin_cache, smalls, tiles,
             moe_w=None, attn=None, me=None, eblk=None, ctx=None,
             with_all=False, vocab_start=None):  # fmt: skip
        """moe_w (moe_impl "real"): dict(tid2eid=[num_hash x int32 [V, 6]],
        bias=f32 [L, E], w13/w13_scale/w2/w2_scale=[L arrays, served layout],
        with moe_placement also held (/ copy) = this rank's int32 [L, slots]
        rows of moe_phase.placement_tables, the arrays in slot order).
        attn (attn_impl "real"): dict(arrays=[distinct served caches],
        cache_index={role: [array per entry]}, cache_layers={role: [layer]},
        bt_flat, bt_off={META_ROLES: int32 [L]}, bt_w={...}, weights=dict in
        step_phases.AttnWeights field order). With attn, the updated arrays are
        returned last (aliased, same order). me: this rank's mesh index as a
        traced int32 (the adapter's meta[META_RANK]); default lax.axis_index,
        which lowers to partition-id, unsupported by the torch_tpu SPMD
        export. eblk (moe "real"): which EP block of the global expert ids
        moe_w's expert arrays hold, a traced int32; required, since the mesh
        index is not the EP rank on every host (ep_mesh.ep_rank_order).
        ctx (ctx_q > 0, the DSpark draft step): dict(main_x=bf16 [T, H] and
        positions=int32 [T] of the context rows in the global row layout of
        the target step, -1 where no row). with_all: also return the final
        residual of every global row, [T, 4H] (identical on every rank), last.
        vocab_start: embed is this rank's vocab-parallel shard starting at this
        (traced) id, the shards of each attention group covering the vocab."""
        me = lax.axis_index(axis) if me is None else me
        tokm, ownt, counts, tid, tpos, own_greq = step_metadata(
            scfg, me, input_ids, positions, seq_lens, axis
        )
        if vocab_start is None:
            x0 = embed[tid].astype(BF16)  # [T, H]
        else:
            # every group sums to x0, so the R-wide sum is G * x0, exactly
            # (one nonzero term per group and element; G a power of two)
            assert scfg.G & (scfg.G - 1) == 0
            lid = tid - vocab_start
            hit = (lid >= 0) & (lid < embed.shape[0])
            row = embed[jnp.clip(lid, 0, embed.shape[0] - 1)]
            part = jnp.where(hit[:, None], row, 0)
            x0 = (lax.psum(part.astype(BF16), axis) * (1.0 / scfg.G)).astype(BF16)
        assert cos_sin_cache.shape[1] == 2 * cfg.rope_dim, "pack_cos_sin table"
        cs_f = rope_rows(cos_sin_cache, tpos, inverse=False, packed=True)
        cs_i = rope_rows(cos_sin_cache, tpos, inverse=True, packed=True)
        extra = []
        if scfg.ctx_q:
            cpos = ctx["positions"].astype(jnp.int32)
            cs_ctx = rope_rows(cos_sin_cache, jnp.maximum(cpos, 0), False, True)
            extra += [ctx["main_x"].astype(BF16), cs_ctx, cpos]
        if moe:
            if scfg.num_hash_layers:
                hid = jnp.stack(
                    [jnp.pad(t[tid], ((0, 0), (0, 2))) for t in moe_w["tid2eid"]]
                )
                hid = hid.reshape(len(moe_w["tid2eid"]), -1).astype(jnp.int32)
            else:  # no hash layer reads it; the phase still takes the operand
                hid = jnp.zeros((1, scfg.T * 8), jnp.int32)
            valid = (tokm[2 * scfg.T : 3 * scfg.T] >= 0).astype(jnp.int32)
            per_layer = []
            for li in range(L):
                per_layer += [moe_w["w13"][li], moe_w["w13_scale"][li],
                              moe_w["w2"][li], moe_w["w2_scale"][li]]  # fmt: skip
            assert eblk is not None, "moe real needs eblk (the expert block)"
            eblk_a = jnp.asarray(eblk, jnp.int32).reshape(1)
            extra += [hid, valid, eblk_a, moe_w["bias"], *per_layer]
            if scfg.moe_placement:
                extra.append(jnp.asarray(moe_w["held"], jnp.int32))
            if scfg.moe_placement == "replicas":
                extra.append(jnp.asarray(moe_w["copy"], jnp.int32))
        alay = None
        if scfg.attn_impl == "real":
            assert scfg.QP == 8
            alay = AttnLayout(
                arrays=tuple((tuple(a.shape), str(a.dtype)) for a in attn["arrays"]),
                cache_index=tuple(tuple(attn["cache_index"][r]) for r in ROLES),
                cache_layers=tuple(tuple(attn["cache_layers"][r]) for r in ROLES),
            )
            own_pos = positions[: scfg.M].astype(jnp.int32)
            nreq = jnp.sum(seq_lens > 0).astype(jnp.int32).reshape(1)
            extra += [
                nreq, seq_lens.astype(jnp.int32), own_pos,
                SP.pad_tables(attn["bt_flat"].astype(jnp.int32)),
                *[attn["bt_off"][r] for r in META_ROLES],
                *[attn["bt_w"][r] for r in META_ROLES],
                own_greq, jnp.asarray(csa_ord),
                *[attn["weights"][f] for f in SP.AttnWeights._fields],
                *attn["arrays"],
            ]  # fmt: skip
        outs = make_call(alay)(
            sched_j, base_j, ltype, tokm, ownt, counts, x0, cs_f, cs_i, smalls,
            *[tiles[n] for n in names], *extra,
        )  # fmt: skip
        res_all, aux_all = outs[:2]
        own = ownt >= 0
        sel = jnp.maximum(ownt, 0)
        res = jnp.where(own[:, None], res_all[sel], 0).astype(BF16)
        # row gathers per aux layer: a gather along axis 1 of the 3-D output
        # relays the whole [A, T, 4H] array out first (10 MiB at T=64)
        aux = jnp.stack(
            [jnp.where(own[:, None], aux_all[k][sel], 0) for k in range(A)]
        ).astype(BF16)
        tail = (res_all,) if with_all else ()
        if scfg.debug_att:
            nd = scfg.n_debug_outputs
            dbg = outs[len(outs) - nd :]
            return (res, aux, tokm, ownt, counts, list(outs[2 : len(outs) - nd]),
                    *dbg, *tail)  # fmt: skip
        if alay is not None:  # cache_inplace: no arrays (the inputs were updated)
            return res, aux, tokm, ownt, counts, list(outs[2:]), *tail
        return res, aux, tokm, ownt, counts, *tail

    return step
