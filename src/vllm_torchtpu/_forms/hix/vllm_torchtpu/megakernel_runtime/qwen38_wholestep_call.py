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
"""One Qwen3.8 whole-step call (model layers 3..46) as a single SPMD JAX function.

`build_wholestep_call` returns `(fn, in_specs, out_specs, donate_argnums)` at one row
count `B`. `fn(hidden, *weights, pool, *gdn_meta, *qsa_tables, positions, starts,
context,
trig)`: pads rows to the kernel's `M`, derives the GDN read / write blocks (as the
segment
call) and the QSA layers' extras with the stock addressing (`qwen38_wholestep_inputs`),
runs
the kernel, then — in the same function, on the donated pool — scatters the GDN
checkpoints
and writes every QSA layer's step tokens through the stock writer sequence
(`qwen38_wholestep_cache`). Returns `(residual, pool)`.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import os

from vllm_torchtpu import envs
from vllm_torchtpu.megakernel.exec_spec import DebugOptions, ExecSpec, Tiles
from vllm_torchtpu.megakernel.ops import F32, REC
from vllm_torchtpu.megakernel.models.qwen38_step import make_flat_segment_step
from vllm_torchtpu.megakernel.wholestep.qsa_full import KV_NEW
from vllm_torchtpu.megakernel.wholestep.qsa_full_c import IR_NEW, KR_NEW, UNION_STATS
from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec

from .qwen38_scatter import scatter_gdn_ckpts_fast
from .qwen38_segment_call import GDN_META, T_VERIFY, gdn_blocks, ring_depths, segment_tiles
from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import main_kv_dcp

from .qwen38_wholestep_cache import QSALayerCache, wholestep_qsa_writes
from .qwen38_wholestep_inputs import qsa_step_inputs

QSA_TABLES = 3  # per QSA layer: ring ids, main table, compressed table
# Served indexer options: page tables in HBM with per-layer SMEM windows, one request's
# scores
# at a time, and only the key chunks a request's rows can see (exact vs the stock
# selection,
# tests/megakernel/test_wholestep.py long mode). Without the last, every request streams
# max_groups keys per QSA layer: 22.4 ms at 64 rows for max_groups 8192 (j-dcafcac7).
LONG_CONTEXT = dict(hbm_tables=True, per_request_scores=True, bounded_chunks=True)
# `VLLM_TPU_MEGAKERNEL=...,wholeshard=1`: each rank scores and selects its own run of
# requests and
# the ranks exchange the selections in-kernel (`QSAFullIndexedBlock.shard_index`;
# exact).
SHARD_INDEX: bool = False
# `...,wholeunion=1`: each request's verify rows attend one union of their selections,
# streamed
# once (`QSAFullIndexedBlock.union_attn`; same tokens per row, f32 order of the softmax
# tiles).
UNION_ATTN: bool = False
# `...,wholestats=N` (with `wholeunion=1`): the call also returns per QSA layer and
# request the
# union size vs the rows' selections (`qsa_union_stats`); the runner logs them every N
# calls.
UNION_STATS_EVERY: int = 0
# `...,wholeloop=1`: a QSA layer's per-request work in a fori_loop (compile size flat in
# B).
LOOP_REQUESTS: bool = False
# `...,wholephys=1` (with `wholeunion=1`): union ids -> physical micro-page ids in one
# vector pass
# per request, so each attention copy issues from one SMEM load (same addresses; RPAd's
# recipe).
UNION_PHYS: bool = False


def union_stats_on() -> bool:
    return UNION_ATTN and UNION_STATS_EVERY > 0


def compressed_dcp() -> bool:
    """VLLM_TPU_QSA_COMPRESSED_DCP under main-KV DCP: compressed keys split like the main KV."""
    return main_kv_dcp() > 1 and bool(envs.VLLM_TPU_QSA_COMPRESSED_DCP)


def indexed_options() -> dict:
    """The served indexer options (`LONG_CONTEXT`, the request-sharded selection, union attention,
    the main-KV DCP size, compressed-key DCP)."""
    return dict(
        LONG_CONTEXT, shard_index=SHARD_INDEX, union_attn=UNION_ATTN, union_stats=union_stats_on(),
        loop_requests=LOOP_REQUESTS, union_phys=UNION_ATTN and UNION_PHYS, dcp=main_kv_dcp(),
        compressed_dcp=compressed_dcp(), **_d08s1_options(), **_idx_options(), **_selloop_options(), **_union_pipe_options(), **({"twin_split": 1} if _idx_twin_on() else {}),  # q38 L1-D08S1 / L1-IDX / L1-SELLOOP, q38 L2-UAPIPE, q38 HXIDX (twin selection split, wholestep/qsa_full_twin.py); q38 S3A-ENG T2: _idx_twin_on (end of module)
    )


def wholestep_tiles(B: int, max_groups: int = 0) -> Tiles:
    """`_wholestep_tiles`; with main-KV DCP-2 up to 32 rows the expert ring drops one more 2 MiB
    slot: the pair exchange buffers and the doubled-head union attention put b32 (f32 state)
    over scoped VMEM (tpu_bench_wholestep_call j-677db355: CompileTimeScopedVmemOom), and the
    owned-only per-rank unions (`qu_id` / `qu_cd` / `qgu_*` x2) b16 too (served j-af65b422:
    62.49M at qwen38_mk_whole_b16)."""
    t = _group_dcp_rings(_wholestep_tiles(B, max_groups), B)  # q38 L1-D14
    if main_kv_dcp() > 1 and B <= 32:
        # Above 32 rows the GDN chunk gather (no NS-sized pg_rec) leaves room for the full
        # expert ring: at b48 the trimmed 5 slots cost 0.3-0.4 ms per call vs 6 (j-74ef730a).
        t = dataclasses.replace(t, nbe=t.nbe - 1)
    # Head-major GDN state update on the chunked (WIDE, > 16 rows) path: per-head states stay
    # in vregs (same-pod A/B j-ed2d3a52: b32 8.42 -> 8.24 ms, b64 14.28 -> 13.99 ms; gdn_rec was
    # 1.7x its VPU floor). Reassociates the f32 sums (`GDNTunables` "approx").
    # VLLM_TPU_MK_GDN_HM=0 restores the whole-state update.
    if os.environ.get("VLLM_TPU_MK_GDN_HM", "1") not in ("", "0"):
        t = t.with_blocks({"gdn": {"gdn_head_major": True}})
    # VLLM_TPU_MK_HIER=N (bench sweeps): override the all-reduce hierarchy.
    hier = os.environ.get("VLLM_TPU_MK_HIER", "")
    if hier:
        t = dataclasses.replace(t, hier=int(hier))
    # VLLM_TPU_MK_RINGS=nbe:nbd (bench sweeps): override the ring depths.
    rings = os.environ.get("VLLM_TPU_MK_RINGS", "") or _ring_s1(B)  # q38 L1-IDX RING-S1 (per bucket, S1 on)
    if rings:
        nbe, nbd = (int(x) for x in rings.split(":"))
        t = dataclasses.replace(t, nbe=nbe, nbd=nbd)
    return t if not _gdnink_rel_fetch(B) else dataclasses.replace(t, rel_fetch=True)  # q38 L1-GDNINK (GDN-REL-FETCH), same line


def _wholestep_tiles(B: int, max_groups: int = 0) -> Tiles:
    """Segment tiles, with the 64-row ring depths from 48 rows up: at 48 rows the segment's
    (6, 10) rings leave the whole-step kernel 0.65 MiB over the 62 MiB VMEM limit (w01
    j-508c1d10: CompileTimeScopedVmemOom 62.63M at qwen38_mk_whole_b48). Past 32768 groups
    (262K contexts) the 16-row kernel's selection temporaries double and miss by 0.13 MiB
    (j-38d236d7: 62.23M at 65536 groups), so up to 16 rows the expert ring drops one 2 MiB slot.
    An f32 GDN state (`VLLM_TPU_MK_GDN_REC_F32=1`) on the WIDE rows (17..47) misses by 0.43 MiB
    (j-f4f467fc: 64.37M at qwen38_mk_whole_b32; 2.16 MiB in tpu_bench_wholestep_call j-5ae41ffb,
    where one expert slot fell 8 KiB short): the chunked GDN state ring (f32 chunk 3 MiB) drops
    from 3 to 2 buffers there; scheduling only. Up to 16 rows (whole-state path, f32 `recb`) it
    misses by 0.30 MiB (j-4aeaf924: 62.30M at qwen38_mk_whole_b16): one more expert slot.
    Above 32 rows the GDN chunks read their states from the pool per `recb` slot (chunk
    gather, `qwen38_gdn_pool`; no NS-sized `pg_rec`); above 64 rows the dense ring drops to
    3 slots. Scratch (f32 state): b48 42.1 / b64 47.9 / b96 53.7 MiB (CPU plan)."""
    if B > 64:
        t = segment_tiles(B, rings=(6, 3))
        return t.with_blocks({"gdn": {"gdn_bufs": 2}}) if REC == F32 else t
    if B >= 48:
        t = segment_tiles(B, rings=(6, 4))
        return t.with_blocks({"gdn": {"gdn_bufs": 2}}) if REC == F32 else t
    if REC == F32 and B > 16:
        return segment_tiles(B).with_blocks({"gdn": {"gdn_bufs": 2}})
    if B <= 16 and (max_groups > 32768 or REC == F32):
        nbe, nbd = ring_depths(B)
        drop = int(max_groups > 32768) + int(REC == F32)
        return segment_tiles(B, rings=(nbe - drop, nbd))
    return segment_tiles(B)


def build_wholestep_call(
    mesh: Any,
    g: Any,
    B: int,
    layout: Any,
    qsa_layers: list[QSALayerCache],
    *,
    L: int = 44,
    budget: int,
    max_groups: int,
    norm_eps: float,
    rotary_dim: int,
    write: bool = True,
    interpret: bool = False,
    tiles: Tiles | None = None,
) -> tuple[Any, tuple, tuple, tuple]:
    R = mesh.size
    T = T_VERIFY
    from .hix700_pregate import pregate_tiles  # q38 HIX700 PREGATE (function-local import)
    # VLLM_TPU_MK_SCOPES=N: per-phase named scopes in the kernel (profiling only).
    scopes = int(os.environ.get("VLLM_TPU_MK_SCOPES", "0") or 0)
    ex = ExecSpec(
        R=R, B=B, T=T, ctx=128, tiles=pregate_tiles(wholestep_tiles(B, max_groups), B) if tiles is None else tiles,  # q38 HIX700 PREGATE (off: the same tiles)
        interpret=interpret, debug=DebugOptions(scopes=scopes), live_rows=_live_rows_on(B),
    )
    n_att = -(-L // g.att_every)  # q38 L1-L47F: L 45's partial last cycle is its QSA layer (= L // att_every at L 44)
    n_gdn = L - n_att
    assert len(qsa_layers) == n_att, (len(qsa_layers), n_att)
    pps = _idx_pps(max_groups, layout)  # q38 L1-IDX TSLIM (max_groups * 4 // 128 when off)
    spec = _side_building(B, wholestep_spec)(  # q38 HIX700 GDN-SIDECAR: the bucket rows in context (wholestep_spec itself when off)
        g,
        R,
        L=L,
        first="qsa",
        pool_gdn=True,
        qsa=dict(budget=budget),
        step=True,
        indexed=dict(
            max_groups=max_groups,
            pages_per_seq=pps,
            carrier="fp8",
            n_layers=n_att, **_l47f_options(L, g.att_every),  # q38 L1-L47F: the sink export ({} when off)
            **indexed_options(),
        ),
    )
    stats = union_stats_on() or _l47f_sink_out(L, g.att_every)  # q38 L1-L47F: outs[6] = the sink export (not with stats)
    fn, k_specs, _, n_w = make_flat_segment_step(
        mesh,
        spec,
        ex,
        extra_outs=(KV_NEW, KR_NEW, IR_NEW) + ((UNION_STATS,) if union_stats_on() else ()) + _l47f_outs(L, g.att_every) + _kvw_extra_outs() + _ink_extra_outs(spec, interpret, layout) + _side_extra_outs(spec, interpret, g, B, R),  # q38 X-GLUE-1 / L1-L47F / L1-KVW
    )
    M, NS = ex.M, B // T
    tp, rep = P("tp"), P()
    # Planned global weight shapes: the packer hands tiled operands over 3-D ([L*nt, tk,
    # N],
    # the same bytes) and they are reshaped to [L, nt, tk, N] here, under jit, for free.
    # Every 4-D operand is re-laid out by an execute_as_strided program per call at the
    # torch_tpu boundary (q32/q40: all copied operands are 4-D, no 3-D one is).
    from vllm_torchtpu.megakernel import kernel

    _, info = kernel.make_step(mesh, spec, ex)
    targets: list[tuple[int, ...] | None] = []
    for name in info["plan"].w_names:
        sh = info["wshape"][name]
        if kernel.is_tupled_weight(sh):
            targets += [None] * len(sh)
        else:
            targets.append((R * sh[0][0],) + tuple(sh[0][1:]))
    x_specs = k_specs[2 : 2 + len(fn.extra_names)]
    hbm = {x for x, s in zip(fn.extra_names, x_specs) if s == tp}

    pad = jax.shard_map(
        lambda h: jnp.pad(h, ((0, M - B), (0, 0))),
        mesh=mesh,
        in_specs=tp,
        out_specs=tp,
        check_vma=False,
    )
    unpad = jax.shard_map(
        lambda r: r[:B], mesh=mesh, in_specs=tp, out_specs=tp, check_vma=False
    )
    pos0 = jnp.zeros((B,), jnp.int32)
    n_meta = GDN_META * n_gdn
    n_tab = QSA_TABLES * n_att
    slots = _uaunpack_slots([(lay.main_slot, lay.compressed_slot) for lay in qsa_layers])  # q38 L2-UNPACK
    scales = (qsa_layers[0].k_scale, qsa_layers[0].v_scale)
    assert all((lay.k_scale, lay.v_scale) == scales for lay in qsa_layers), (
        "one KV scale pair"
    )

    names = (
        "qsa_pool",
        "qsa_ck",
        "qsa_cktab",
        "qsa_nvis",
        "qsa_tail",
        "qsa_mtab",
        "qsa_off",
        "qsa_scales",
        "qsa_rope",
        "qsa_ctx",
        "qsa_ring",
        "qsa_rope_ck",
    )
    out_specs_in = tuple(tp if n in hbm else rep for n in names)

    def per_rank_inputs(pool, tables_flat, positions, starts, context, trig):
        tables = [
            tuple(tables_flat[i : i + QSA_TABLES]) for i in range(0, n_tab, QSA_TABLES)
        ]
        d = _idx_step_inputs(qsa_step_inputs)(  # q38 L1-IDX S2P / TSLIM (qsa_step_inputs itself when off)
            pool,
            tables,
            slots,
            trig,
            positions,
            starts,
            context,
            M=M,
            T=T,
            pages_per_seq=pps,
            max_groups=max_groups,
            k_scale=scales[0],
            v_scale=scales[1],
            layer_major_tables=LONG_CONTEXT["hbm_tables"],
        )
        return tuple(d[n] for n in names)

    inputs = jax.shard_map(
        per_rank_inputs,
        mesh=mesh,
        in_specs=(tp, rep, rep, rep, rep, rep),
        out_specs=out_specs_in,
        check_vma=False,
    )

    dcp_size = main_kv_dcp()

    def per_rank_writes(pool, conv_ck, rec_ck, wb, kv, kr, ir, tables_flat, positions, starts,
                        context, trig, norms_flat, *rank):
        pool = _g1_scatter(scatter_gdn_ckpts_fast, write)(pool, conv_ck, rec_ck, wb, layout=layout, interpret=interpret) if (conv_ck.shape, conv_ck.dtype) != (pool.shape, pool.dtype) else (conv_ck if interpret else pool)  # q38 X-GLUE-1: in-kernel checkpoints pass the pool through; q38 L1-G1: no scatter with VLLM_TPU_MK_GDN_COMMIT1
        if not write:
            return pool
        Bq = len(positions)
        rows = [
            (kr[o, :Bq], kv[o, :Bq, kv.shape[-1] // 2 :], ir[o, :Bq])
            for o in range(n_att)
        ]
        norms = [tuple(norms_flat[2 * o : 2 * o + 2]) for o in range(n_att)]
        tables = [tuple(tables_flat[i : i + QSA_TABLES]) for i in range(0, n_tab, QSA_TABLES)]
        # Main-KV DCP: this rank's pair rank, the QSA layers' `dcp_rank` buffer (int32 [1] per
        # rank; `axis_index` lowers to a partition id, which the served SPMD compile refuses).
        dcp = (rank[0].reshape(1), dcp_size) if dcp_size > 1 else None
        return wholestep_qsa_writes(
            pool, rows, norms, tables, qsa_layers, positions, trig, starts, context,
            compress_ratio=4, norm_eps=norm_eps, rotary_dim=rotary_dim, max_query_len=T,
            num_requests=M // T, dcp=dcp, compressed_dcp=dcp if compressed_dcp() else None, **_kvw_kw(kv, rank[int(dcp_size > 1):], Bq),  # q38 L1-KVW
        )

    writes = jax.shard_map(
        per_rank_writes,
        mesh=mesh,
        in_specs=(tp, tp, tp, rep, tp, tp, tp, rep, rep, rep, rep, rep, rep)
        + ((tp,) if dcp_size > 1 else ()) + _kvw_specs(tp),  # q38 L1-KVW: + (ck_new, ring_new) under KVAL
        out_specs=tp,
        check_vma=False,
    )

    def call(*args):
        hidden = args[0]
        w = [
            x if t is None or tuple(x.shape) == t else x.reshape(t)
            for x, t in zip(args[1 : 1 + n_w], targets, strict=True)
        ]
        i = 1 + n_w
        pool = args[i]; side = _side_take(args, i, spec); i += 3 if side is not None else 0  # noqa: E702  q38 HIX700 GDN-SIDECAR: the 3 sidecar operands right after the pool
        meta = args[i + 1 : i + 1 + n_meta]
        i += 1 + n_meta
        tables = args[i : i + n_tab]
        i += n_tab
        positions, starts, context, trig = args[i : i + 4]
        norms = args[i + 4 : i + 4 + 2 * n_att]  # per QSA layer: k_norm, index k_norm (cache writer)
        rank = args[i + 4 + 2 * n_att :]  # main-KV DCP: the pair rank, int32 [1] per rank
        x = pad(hidden)
        rb, wb, hi = gdn_blocks(meta, NS, write)
        xs = dict(zip(names, inputs(pool, tables, positions, starts, context, trig)))
        xs.update(gdn_pool=pool, gdn_rb=rb, gdn_hi=hi.astype(jnp.int32)); xs.update(_live_xs(ex, meta[3], NS, T)); xs.update(_ink_xs(fn, wb)); xs.update(_side_xs(fn, side, meta[1], meta[0::GDN_META], NS, T))  # noqa: E702  q38 X-GLUE-1; q38 HIX700 GDN-SIDECAR
        outs = fn(pos0, x, *(xs[k] for k in fn.extra_names), *w); outs = _ink_outs(fn, outs, pool, interpret); outs, side_ret = _side_outs(fn, outs, pool, side, interpret)  # noqa: E702  q38 X-GLUE-1; q38 HIX700 GDN-SIDECAR
        res, conv_ck, rec_ck, kv, kr, ir = outs[:6]
        pool = writes(pool, conv_ck, rec_ck, wb, kv, kr, ir, tables, positions, starts, context,
                      trig, norms, *rank, *_kvw_outs(outs))  # q38 L1-KVW
        return _side_ret(_g1_ret((unpad(res), pool, outs[6]) if stats else (unpad(res), pool), write, conv_ck, rec_ck, wb, meta[0::GDN_META], NS, R), side_ret, wb, R)  # q38 L1-G1; q38 HIX700 GDN-SIDECAR

    # q38 L1-D07B1 begin
    # q38 L1-D07B1, VLLM_TPU_MK_GDN_META_DEDUP=1: `call` with the GDN metadata as
    # `off, qsl, seq` once (the step's tensors every GDN layer's metadata shares) followed by
    # the n_gdn state ids and the n_gdn checkpoint ids, and the block ids from one vectorised
    # pass (`gdn_blocks_shared`, bit-identical to `gdn_blocks`) instead of n_gdn per-layer chains
    # (about 259 small XLA ops per step at 16 rows, 139 us in the E4 M2u profile). Defined after
    # `call`, whose lines (and so the flag-off lowering, source locations included) are
    # unchanged; keep the two in step.
    from .qwen38_segment_call import GDN_SHARED, gdn_blocks_shared

    n_meta_shared = GDN_SHARED + 2 * n_gdn

    def call_shared(*args):
        hidden = args[0]
        w = [
            x if t is None or tuple(x.shape) == t else x.reshape(t)
            for x, t in zip(args[1 : 1 + n_w], targets, strict=True)
        ]
        i = 1 + n_w
        pool = args[i]; side = _side_take(args, i, spec); i += 3 if side is not None else 0  # noqa: E702  q38 HIX700 GDN-SIDECAR (call_shared)
        off, qsl, seq = args[i + 1 : i + 1 + GDN_SHARED]
        sts = args[i + 1 + GDN_SHARED : i + 1 + GDN_SHARED + n_gdn]
        cks = args[i + 1 + GDN_SHARED + n_gdn : i + 1 + n_meta_shared]
        i += 1 + n_meta_shared
        tables = args[i : i + n_tab]
        i += n_tab
        positions, starts, context, trig = args[i : i + 4]
        norms = args[i + 4 : i + 4 + 2 * n_att]
        rank = args[i + 4 + 2 * n_att :]
        x = pad(hidden)
        rb, wb, hi = gdn_blocks_shared(off, qsl, seq, sts, cks, NS, write)
        xs = dict(zip(names, inputs(pool, tables, positions, starts, context, trig)))
        xs.update(gdn_pool=pool, gdn_rb=rb, gdn_hi=hi.astype(jnp.int32)); xs.update(_live_xs(ex, qsl, NS, T)); xs.update(_ink_xs(fn, wb)); xs.update(_side_xs_gm(fn, side, off, sts, NS, T))  # noqa: E702  q38 X-GLUE-1; q38 HIX700 GDN-SIDECAR; q38 HIX-GLUE GM (vectorised with the key, else _side_xs)
        outs = fn(pos0, x, *(xs[k] for k in fn.extra_names), *w); outs = _ink_outs(fn, outs, pool, interpret); outs, side_ret = _side_outs(fn, outs, pool, side, interpret)  # noqa: E702  q38 X-GLUE-1 (call_shared); q38 HIX700 GDN-SIDECAR
        res, conv_ck, rec_ck, kv, kr, ir = outs[:6]
        pool = writes(pool, conv_ck, rec_ck, wb, kv, kr, ir, tables, positions, starts, context,
                      trig, norms, *rank, *_kvw_outs(outs))  # q38 L1-KVW (call_shared)
        return _side_ret(_g1_ret((unpad(res), pool, outs[6]) if stats else (unpad(res), pool), write, conv_ck, rec_ck, wb, sts, NS, R), side_ret, wb, R)  # q38 L1-G1 (call_shared); q38 HIX700 GDN-SIDECAR

    # q38 L1-D07B1 end
    in_specs = (
        (tp,) + (tp,) * n_w + (tp,) + (rep,) * n_meta + (rep,) * n_tab + (rep,) * 4
        + (rep,) * (2 * n_att) + ((tp,) if dcp_size > 1 else ())
    )
    # q38 L1-D07B1 begin
    if gdn_meta_shared(B):  # q38 L1-GLUEFIX: per bucket (VLLM_TPU_MK_GDN_META_DEDUP_MIN_ROWS)
        # Same operands up to the pool (donated, unmoved), then n_meta_shared GDN ids, then as before.
        k = 2 + n_w
        in_specs = in_specs[:k] + (rep,) * n_meta_shared + in_specs[k + n_meta :]
        return _side_specs(spec, call_shared, in_specs, _g1_specs((tp, tp, tp) if stats else (tp, tp), write, tp, rep), (1 + n_w,), tp, rep)  # q38 L1-G1 (call_shared); q38 HIX700 GDN-SIDECAR
    # q38 L1-D07B1 end
    return _side_specs(spec, call, in_specs, _g1_specs((tp, tp, tp) if stats else (tp, tp), write, tp, rep), (1 + n_w,), tp, rep)  # q38 L1-G1; q38 HIX700 GDN-SIDECAR


# q38 L1-D14: main-KV DCP 4 (the whole step's group path, qsa_full_c `QSAFullStepGroupBlock`) at the 64-row bucket.
def _group_dcp_rings(t: Tiles, B: int) -> Tiles:
    """At main-KV DCP 4 and 48 < B <= 64 the expert ring drops from 6 to 4 slots: with the default (6, 4) rings the
    DCP-4 whole step needs 66.09M of 63.94M scoped VMEM and 5 slots also fail; 4 compiles (l1-d14-aot-i1
    j-7d24d656, b64, 63.693 GiB HBM). Every other bucket, and DCP 1 / 2, keep their tiles (16 / 32 / 48 rows fit
    at DCP 4 on the default rings, same job). `VLLM_TPU_MK_RINGS` still overrides afterwards."""
    if main_kv_dcp() > 2 and 48 < B <= 64:
        return dataclasses.replace(t, nbe=min(t.nbe, 4))
    return t


# q38 L1-D08S1: D08 package S1 "UNION-LEAN" (wholestep/qsa_full_lean.py), one default-off env per stage; each is registered
# in envs.py, so it keys the TPU compile cache.
D08S1_ENVS = (
    ("union_bounded", "VLLM_TPU_MK_UNION_BOUNDED"),  # B1: union compaction stops at the union count
    ("union_lean", "VLLM_TPU_MK_UNION_LEAN"),  # B2: no per-row compacted ids; selection exchange folded into the union's
    ("slot_batch", "VLLM_TPU_MK_SLOT_BATCH"),  # B3: per-layer batched completed groups / recent tiles, double-buffered ids
)


def _d08s1_options() -> dict:
    """{} with every D08 S1 env off: `wholestep_spec` then builds MM's / D14's block exactly as before. Otherwise every
    stage's option (on or off), which selects `QSAFullStep(Group)LeanBlock`; the lean block refuses a non-union step."""
    global _D08S1_LOGGED
    on = _d08s1_on()  # q38 S3A-ENG T2: the pure resolver (end of module), shared with the bind twin s2_construct_line
    issue = {**_uaissue_options(), **_uaunpack_options()}  # q38 L2-UAISSUE, q38 L2-UNPACK: {} when off
    if not any(on.values()):
        return issue
    if not _D08S1_LOGGED:  # engagement line for STEP cars (once per process; vLLM's handlers via the "vllm." prefix)
        _D08S1_LOGGED = True
        import logging

        logging.getLogger("vllm." + __name__).info(
            "q38 D08S1 whole step: %s", " ".join(f"{e}={int(on[k])}" for k, e in D08S1_ENVS)
        )
    return {**on, **issue}


_D08S1_LOGGED = False


# q38 L1-D07B1 begin


def gdn_meta_shared(B: int | None = None) -> bool:
    """q38 L1-D07B1: VLLM_TPU_MK_GDN_META_DEDUP (the whole step's shared-GDN-metadata operand list).

    Read when the call is built and when the module binds its pool; registered in envs, so it keys
    the TPU compile cache.

    q38 L1-GLUEFIX: with a bucket `B`, also VLLM_TPU_MK_GDN_META_DEDUP_MIN_ROWS: the shared program
    only for B >= that many rows (0 = every bucket, the D07-b1 behaviour). A bucket below it builds
    the flag-off program `call`; the module still passes the shared operand list and
    `_WholeKind.callable_for` expands it to the per-layer list on the host (`expand_gdn_meta`)."""
    on = bool(envs.VLLM_TPU_MK_GDN_META_DEDUP)
    if not on or B is None:
        return on
    if _hxg_gm_on():  # q38 HIX-GLUE GM: the shared program at every bucket (b16 included)
        return True
    return int(B) >= int(envs.VLLM_TPU_MK_GDN_META_DEDUP_MIN_ROWS)


def expand_gdn_meta(args: Any, k: int, n_gdn: int) -> list:
    """q38 L1-GLUEFIX: the call's operand list with its shared GDN metadata run (`off, qsl, seq`, the n_gdn
    state ids, the n_gdn checkpoint ids; at index k) replaced by the flag-off run (`st, off, ck, qsl, seq` per
    layer), the same objects in the order `Qwen38WholeStep.forward` passes with the flag off."""
    off, qsl, seq = args[k : k + GDN_META_SHARED]
    sts = args[k + GDN_META_SHARED : k + GDN_META_SHARED + n_gdn]
    cks = args[k + GDN_META_SHARED + n_gdn : k + GDN_META_SHARED + 2 * n_gdn]
    assert len(sts) == len(cks) == n_gdn, (len(sts), len(cks), n_gdn)
    per = [x for st, ck in zip(sts, cks, strict=True) for x in (st, off, ck, qsl, seq)]
    return [*args[:k], *per, *args[k + GDN_META_SHARED + 2 * n_gdn :]]


GDN_META_SHARED = 3  # = qwen38_segment_call.GDN_SHARED (off, qsl, seq); kept here so this block imports nothing
# q38 L1-D07B1 end


# q38 L1-IDX: the in-kernel indexer package on the whole step (S2P / QSPLIT / TSLIM / RING-S1; wholestep/qsa_full_idx.py).
# Every knob is a default-off env registered in envs.py (hashed, so it keys the TPU compile cache). With all of them unset
# `_idx_options()` is {}, `_idx_pps` is max_groups * 4 // 128, `_idx_step_inputs(f)` is f and `_ring_s1` is "", so every
# program is built exactly as before (the in-place call sites keep their lines and columns).
IDX_SCORE_ENVS = (
    ("chunk_pages", "VLLM_TPU_MK_SCORE_CHUNK"),  # PD W `wholechunk`: key pages per scorer chunk (kernel default 32)
    ("ck_lanes", "VLLM_TPU_MK_SCORE_LANES"),  # W `wholelanes`: carrier lanes read per row (default all 256)
    ("manager_pages", "VLLM_TPU_MK_SCORE_MP"),  # W `wholemp`: kernel pages per key DMA (needs the manager-page table)
)
_IDX_LOGGED: set = set()


def _idx_log(key: str, msg: str, *args: Any) -> None:
    """One engagement line per key and process (STEP / script-bench cars grep them; vLLM's handlers via "vllm.")."""
    if key in _IDX_LOGGED:
        return
    _IDX_LOGGED.add(key)
    import logging

    logging.getLogger("vllm." + __name__).info(msg, *args)


def score_options() -> dict:
    """S2P: the key scorer's chunk / lane / DMA-run overrides (PD W's `wholechunk`, `wholelanes`, `wholemp`; s2
    j-59c78af4: 128 / 128 / 4 cut W's B4 scoring 751 -> 300 us, bit-exact). {} when the three envs are unset / "0"."""
    return {k: int(getattr(envs, e)) for k, e in IDX_SCORE_ENVS if int(getattr(envs, e) or 0)}


def _idx_options() -> dict:
    """The L1-IDX indexed-block options: the scorer overrides plus the kernel keys of `qsa_full_idx` (`radix_bits`,
    `union_qsplit`, `table_prefetch`; any of them selects the L1-IDX block class). {} with every env off."""
    out = _idx_base()  # q38 S3A-ENG T2: the pure resolver (end of module), shared with the bind twin s2_construct_line. It holds
    # the six lines that stood here, verbatim: score_options(), then
    #   radix_bits = 2      iff VLLM_TPU_MK_SELECT_RADIX == "2"
    #   union_qsplit = N    iff VLLM_TPU_MK_UNION_QSPLIT is N != 0
    #   table_prefetch      iff VLLM_TPU_MK_TABLE_SLIM == "2"
    # These comment lines stand in place of the moved code, so every later line of this module keeps its number
    # (the traced whole step keeps its source locations).
    if out or str(envs.VLLM_TPU_MK_TABLE_SLIM) != "0":
        _idx_log("opts", "q38 L1-IDX whole step: %s TABLE_SLIM=%s", " ".join(f"{k}={v}" for k, v in sorted(out.items())),
                 envs.VLLM_TPU_MK_TABLE_SLIM)
    from ..megakernel.wholestep.qsa_full_idx import scan_options; return dict(out, **_qx_options(), **scan_options(out, envs))  # noqa: E702  q38 L1-QX (qx_early; end of module); q38 HXSCAN (scan_bound, same line)


def _idx_pps(max_groups: int, layout: Any) -> int:
    """Main page-table columns of the whole step. Default max_groups * 4 // 128 (2048 at 65536 groups). TSLIM
    (VLLM_TPU_MK_TABLE_SLIM 1 / 2): max_groups // (groups per pool block * main-KV DCP), the columns any group below
    max_groups addresses (a column covers DCP x block tokens; 128 at DCP 4 with 512-token blocks)."""
    pps = max_groups * 4 // 128
    if str(envs.VLLM_TPU_MK_TABLE_SLIM) == "0":
        return pps
    bt, d = int(layout.pool_shape[1]), max(main_kv_dcp(), 1)
    assert bt % 128 == 0 and (max_groups * 4) % (bt * d) == 0, ("TSLIM: block tokens / DCP must divide the cap", bt, d)
    slim = max_groups * 4 // (bt * d)
    assert 1 <= slim <= pps, (slim, pps)
    _idx_log(f"pps{slim}", "q38 L1-IDX TSLIM: main table %d -> %d columns (block %d tokens, DCP %d)", pps, slim, bt, d)
    return slim


def _idx_step_inputs(fn: Any) -> Any:
    """`fn` (qsa_step_inputs) itself unless S2P's manager pages or TSLIM are on; then `fn` followed by the trace-time
    layout checks and (MP) the manager-page key table `pages[:, :n_ck:MP] // MP` (qwen38_wholestep_inputs)."""
    mp = int(envs.VLLM_TPU_MK_SCORE_MP or 0)
    slim = str(envs.VLLM_TPU_MK_TABLE_SLIM) != "0"
    if not (mp or slim):
        return fn
    from .qwen38_wholestep_inputs import idx_manager_table

    def wrapped(pool, tables, slots, *args, **kw):
        d = fn(pool, tables, slots, *args, **kw)
        if slim:  # the slim width must still cover every group below the cap for this pool's block size
            cols, cap = int(kw["pages_per_seq"]), int(kw["max_groups"]) * 4
            assert cols * int(pool.shape[1]) * max(main_kv_dcp(), 1) >= cap, ("TSLIM columns", cols, pool.shape)
            assert d["qsa_mtab"].shape[-1] == cols, d["qsa_mtab"].shape
        if mp:
            d["qsa_cktab"] = idx_manager_table(d["qsa_cktab"], pool, slots, int(kw["max_groups"]), mp)
        return d

    return wrapped


def parse_ring_s1(spec: str) -> dict[int, str]:
    """"B=nbe:nbd[,B=nbe:nbd...]" -> {B: "nbe:nbd"} (raises on anything else). '+' is accepted as the separator too
    (q38 L1 STACK-D v4): the served DECODE_EXTRA_SERVER_ENV splits its items on ',', so a served car writes
    "32=7:10+48=8:4+64=6:4"; the bench ARM_ENV splits on '+', so a bench car keeps the ',' form."""
    out: dict[int, str] = {}
    for item in spec.replace("+", ",").split(","):
        b, _, r = item.strip().partition("=")
        nbe, _, nbd = r.partition(":")
        if not (b.isdigit() and nbe.isdigit() and nbd.isdigit() and int(b) > 0 and int(nbe) > 0 and int(nbd) > 0):
            raise ValueError(f"VLLM_TPU_MK_RING_S1: bad entry {item!r} in {spec!r} (want B=nbe:nbd)")
        if int(b) in out:
            raise ValueError(f"VLLM_TPU_MK_RING_S1: bucket {b} listed twice in {spec!r}")
        out[int(b)] = f"{int(nbe)}:{int(nbd)}"
    return out


def _ring_s1(B: int) -> str:
    """RING-S1: "nbe:nbd" for bucket B from VLLM_TPU_MK_RING_S1, only at main-KV DCP > 2 with all three D08 S1 envs on
    (the composed AOT's configuration: S1's freed VMEM pays for the deeper expert ring); "" otherwise. Read by
    `wholestep_tiles` in place of an unset VLLM_TPU_MK_RINGS (which still overrides every bucket)."""
    spec = str(envs.VLLM_TPU_MK_RING_S1 or "").strip()
    if not spec:
        return ""
    table = parse_ring_s1(spec)
    if main_kv_dcp() <= 2 or not all(bool(getattr(envs, e)) for _, e in D08S1_ENVS):
        _idx_log("ring-off", "q38 L1-IDX RING-S1 %r inactive (needs main-KV DCP > 2 and %s)", spec,
                 "+".join(e for _, e in D08S1_ENVS))
        return ""
    r = table.get(int(B), "")
    if r:
        _idx_log(f"ring{B}", "q38 L1-IDX RING-S1: b%d rings %s", int(B), r)
    return r
# q38 X-K-LIVE begin
# VLLM_TPU_MK_LIVE_ROWS=1: the kernel takes the live-row mask (megakernel/live_rows.py) and skips pad rows' GDN work (L1)
# and MoE work (L3). Kept at the end of the module, with `call` / `call_shared` / `build_wholestep_call` edited only
# within existing lines (the ExecSpec line and each `xs.update` line), so the flag-off lowering keeps its source
# locations. The mask comes from the query start locations already passed (layer 0's in `call`, the shared tensor in
# `call_shared`), so the call's operand list does not change.


def _live_rows_on(B: int = 0) -> int:
    """`ExecSpec.live_rows` for the call being built at `B` rows: 0 (VLLM_TPU_MK_LIVE_ROWS unset), 1 (the mask; GDN skips
    pad requests' pool-gather DMAs, bitwise vs the flag off), 2 (+ VLLM_TPU_MK_LIVE_GDN_COMPUTE: GDN also skips pad
    requests' recurrence and state copies). Mode 2 is refused at B <= LIVE_GDN_COMPUTE_MIN_ROWS - 1 (the whole-state GDN
    path): there it builds mode 1 (l1-live-c-aot-i2 j-3cac436b: mode 2 at 16 rows needs 63.72M of 63.50M scoped VMEM at
    the served rings 10:10; the 16-row recurrence guard alone causes it; L1 ruling: no ring cut at b16). With the flag,
    one engagement line per process (STEP / served readers key on it; vLLM's handlers via the "vllm." prefix), as D08
    S1's, and one line per refused bucket."""
    global _LIVE_LOGGED
    from vllm_torchtpu.megakernel.live_rows import live_rows_mode

    mode = live_rows_mode()
    if mode >= 2 and 0 < B < LIVE_GDN_COMPUTE_MIN_ROWS:
        mode = 1
        if B not in _LIVE_REFUSED:
            _LIVE_REFUSED.add(B)
            import logging

            logging.getLogger("vllm." + __name__).info(
                "q38 X-K-LIVE whole step: VLLM_TPU_MK_LIVE_GDN_COMPUTE refused at %d rows (whole-state GDN path, VMEM); "
                "mode 1 there", B
            )
    if mode and not _LIVE_LOGGED:
        _LIVE_LOGGED = True
        import logging

        logging.getLogger("vllm." + __name__).info(
            "q38 X-K-LIVE whole step: VLLM_TPU_MK_LIVE_ROWS=1 VLLM_TPU_MK_LIVE_GDN_COMPUTE=%d (kernel input live_rows; "
            "GDN pad requests: %s)", int(mode >= 2), "gather DMAs + recurrence skipped" if mode >= 2 else "gather DMAs skipped"
        )
    return mode


_LIVE_LOGGED = False
_LIVE_REFUSED: set = set()
# Mode 2 (VLLM_TPU_MK_LIVE_GDN_COMPUTE) only from this many rows (the chunked GDN path, ExecSpec.WIDE: B > 16).
LIVE_GDN_COMPUTE_MIN_ROWS = 17


def _live_xs(ex: Any, qsl: Any, NS: int, T: int) -> dict:
    """{} unless `ex.live_rows`; else the kernel's `live_rows` input from the step's query start locations."""
    if not ex.live_rows:
        return {}
    from vllm_torchtpu.megakernel.live_rows import LIVE_ROWS, live_rows_from_qsl

    return {LIVE_ROWS: live_rows_from_qsl(qsl, NS, T)}
# q38 X-K-LIVE end


# q38 L1-GDNINK begin (GDN-REL-FETCH, VLLM_TPU_MK_GDN_INK bit 2): the whole step's dense weight-stream refills decoded at trace
# time (MM's `Tiles.rel_fetch`, NUMERICS "exact": scheduling only; the Qwen3.5 candidate-30 profile's flag). Appended after
# every existing line; `wholestep_tiles`' return line is extended in place (no line moves), so the flag-off lowering is
# unchanged. Why (l1-gdnink-cen-i2 j-ce2d6f3f, static LLO census, d4_16): each `lin_in` K-tile iteration of the GDN input
# projection is 905 bundles with SALU slots 1050 (lower bound 525 bundles, scalar-bound) for 72 MXU slots; the scalar work is
# `TwoRing.fetch`'s run-time decode of the refilled tile (stream.py:277 / :211 / :304): 31 predicated DMA descriptor set-ups
# (one per weight segment of every layer type) per tile. `fetch_rel` decodes the tile's (layer type, weight, row) at trace
# time and issues one DMA. Same tiles into the same ring banks in the same order, so the values are unchanged.
_GDNINK_RF_LOGGED: set = set()


def _gdnink_rel_fetch(B: int = 0) -> bool:
    """VLLM_TPU_MK_GDN_INK bit 2 set (e.g. "2" / "3"): build the whole step with `Tiles.rel_fetch=True`; one engagement line
    per process and bucket."""
    v = os.environ.get("VLLM_TPU_MK_GDN_INK", "0").strip()
    on = v.isdigit() and bool(int(v) & 2)
    if on and B not in _GDNINK_RF_LOGGED:
        _GDNINK_RF_LOGGED.add(B)
        import logging

        logging.getLogger("vllm." + __name__).info(
            "q38 L1-GDNINK GDN-REL-FETCH on: whole step at %d rows built with Tiles.rel_fetch=True (dense refills decoded at "
            "trace time)", B
        )
    return on
# q38 L1-GDNINK end


# q38 X-GLUE-1 begin
# VLLM_TPU_MK_GDN_CKPT_INK=1 (megakernel/models/qwen38_gdn_ckpt_ink.py): the kernel writes the GDN checkpoints into the pool
# itself and takes the write ids `gdn_wb`; `call` / `call_shared` hand it the ids and pass the pool through the scatter
# line (its operand list and outputs are unchanged). Kept at the end of the module, with the existing lines edited only
# at their ends (the extra_outs argument, the scatter line, each xs.update / outs line), so the flag-off lowering keeps its
# source locations.


def _ink_extra_outs(spec: Any, interpret: Any, layout: Any) -> tuple:
    from vllm_torchtpu.megakernel.models.qwen38_gdn_ckpt_ink import ink_extra_outs

    return ink_extra_outs(spec, interpret, layout)


def _ink_xs(fn: Any, wb: Any) -> dict:
    from vllm_torchtpu.megakernel.models.qwen38_gdn_ckpt_ink import ink_xs

    return ink_xs(fn, wb)


def _ink_outs(fn: Any, outs: Any, pool: Any, interpret: Any) -> Any:
    from vllm_torchtpu.megakernel.models.qwen38_gdn_ckpt_ink import ink_outs

    return ink_outs(fn, outs, pool, interpret)
# q38 X-GLUE-1 end


# q38 L1-QX begin
# X-DCP-QX-EARLY (cycle 6): VLLM_TPU_MK_DCP_QX_EARLY=1 adds the kernel option `qx_early` to `_idx_options()` under main-KV
# DCP 2 / 4 (wholestep/qsa_full_idx.py `_QxEarlyMixin`, routed by its `lean_step_block`): the DCP q exchange (`qf_q` to the
# group peers) starts right after the QSA projection `_project` and is waited just before the union attention, so the transfer
# hides under the indexer chain `_select`; the same copies carry the same values (bitwise). At main-KV DCP 1 there is no
# exchange and under compressed-key DCP the option stays off (outside the proven envelope): no option, one log line. Default
# off, hashed (envs.py). Appended at the end of the module, with one in-place line in `_idx_options`, so every traced line of
# this module keeps its line number.
_QX_LOGGED = False


def _qx_options() -> dict:
    """{"qx_early": True} when VLLM_TPU_MK_DCP_QX_EARLY is "1" under main-KV DCP > 1 (no compressed-key DCP), else {}; one
    engagement line per process on vLLM's handlers ("vllm." prefix): 'q38 L1-QX whole step: DCP q exchange starts after
    _project (main-KV DCP <d>)', or the inactive reason."""
    global _QX_LOGGED
    qx = _qx_state()  # q38 S3A-ENG T2: the pure resolver (end of module), shared with the bind twin s2_construct_line
    if qx is None:  # VLLM_TPU_MK_DCP_QX_EARLY is not "1"
        return {}
    on, dcp, cdcp = qx  # on = main-KV DCP > 1 and no compressed-key DCP
    if not _QX_LOGGED:
        _QX_LOGGED = True
        import logging

        if on:
            logging.getLogger("vllm." + __name__).info(
                "q38 L1-QX whole step: DCP q exchange starts after _project (main-KV DCP %d)", dcp)
        else:
            logging.getLogger("vllm." + __name__).info(
                "q38 L1-QX whole step: VLLM_TPU_MK_DCP_QX_EARLY=1 inactive (main-KV DCP %d, compressed-key DCP %s)",
                dcp, cdcp)
    return {"qx_early": True} if on else {}
# q38 L1-QX end


# q38 L1-L47F begin
# VLLM_TPU_MK_L47_FOLD=1 (L47-FOLD-S1, megakernel_runtime/qwen38_l47fold.py): model layers 3..47 in the kernel (L 45) and
# the last QSA layer's per-row selection exported for the MTP draft's sink. Appended after every existing line, with
# `build_wholestep_call` edited only within four existing lines (n_att, the indexed options, `stats`, `extra_outs`), so the
# flag-off lowering keeps its source locations. With the flag off each helper is {} / False / ().


def _l47f_options(L: int, att_every: int) -> dict:
    from .qwen38_l47fold import l47f_options

    return l47f_options(L, att_every)


def _l47f_sink_out(L: int, att_every: int) -> bool:
    """Whether the call returns the sink export as its third output (`outs[6]`, where the union stats would be)."""
    on = bool(_l47f_options(L, att_every))
    assert not (on and union_stats_on()), "q38 L1-L47F: the sink export and wholestats share the call's third output"
    return on


def _l47f_outs(L: int, att_every: int) -> tuple:
    from vllm_torchtpu.megakernel.wholestep.qsa_full_l47f import SINK_IDS

    return (SINK_IDS,) if _l47f_sink_out(L, att_every) else ()
# q38 L1-L47F end
# q38 L1-SELLOOP begin
# VLLM_TPU_MK_SELECT_OWNED (1 / 2): the in-kernel selection loops only over this rank's owned request slots
# (wholestep/qsa_full_lean.py `_LeanMixin._owned_slots`). Appended at the end of the module, with `indexed_options` edited
# only within its existing line, so the flag-off lowering keeps its source locations. The option reaches the block through
# qsa_full_idx.lean_step_block, which refuses it without the D08 S1 lean block (the block itself refuses it without
# wholeloop=1, wholeshard=1 and VLLM_TPU_MK_SLOT_BATCH=1).


def _selloop_options() -> dict:
    """{} with VLLM_TPU_MK_SELECT_OWNED unset / "0"; else {"select_owned": 1 | 2} and one engagement line per process
    (STEP / served readers key on it; vLLM's handlers via the "vllm." prefix)."""
    global _SELLOOP_LOGGED
    mode = _selloop_mode()  # q38 S3A-ENG T2: the pure resolver (end of module), shared with the bind twin; it raises as before
    if not mode:
        return {}
    # The q38 L1-V6 refusal that stood here (VLLM_TPU_MK_SELECT_OWNED with VLLM_TPU_MK_L47_FOLD=1 while
    # HXP1A_SELLOOP_L47F is False) moved verbatim into _selloop_mode, same exception type and text, raised before
    # this point. These comment lines stand in place of the moved code, so every later line of this module keeps
    # its number.
    #
    if not _SELLOOP_LOGGED:
        _SELLOOP_LOGGED = True
        import logging

        logging.getLogger("vllm." + __name__).info(
            "q38 L1-SELLOOP whole step: VLLM_TPU_MK_SELECT_OWNED=%d (selection loop over owned request slots only; %s)",
            mode, "every owned slot unrolled" if mode == 2 else "straight-line at one request per rank, else a loop",
        )
    return {"select_owned": mode}


_SELLOOP_LOGGED = False
# q38 L1-SELLOOP end
# q38 L1-KVW begin
# KVW-ALL + QSA-WRITE-FOLD stage A (q38 L2 2e68a7f15 / d1b5247ca / 094e7d381 / 6150b16c9), wired by q38 L1-V6 (STACK-D v6).
# VLLM_TPU_MK_QSA_WRITE_ALL=1: `wholestep_qsa_writes` routes the post-kernel QSA cache writes through the all-layer writer
# (qwen38_wholestep_cache `_kvw_writer`). With VLLM_TPU_MK_QSA_WRITE_KVAL=1 as well (`qsa_write_kval_enabled`), the kernel's
# completed-group keys and rings (qsa_ck_new / qsa_ring_new) join the kernel's extra outputs after every other output but
# before the in-kernel GDN checkpoint writer's interpret-mode pool output (so the union-stats / layer-47 sink output keeps
# outs[6] and `_ink_outs` still finds its pool last), ride `writes` after the DCP rank, and reach the writer as `KernelValues` with the key rows of
# qsa_kv_new. The layer list is the call's own `qsa_layers` (12 QSA layers with the layer-47 fold on, 11 off). Appended at the end of
# the module, with five existing lines extended in place, so the flag-off lowering keeps its source locations. Flags off:
# every helper is () / {} and logs nothing.
_KVW_LOGGED = False


def _kvw_kval() -> bool:
    from .qwen38_qsa_write_all import qsa_write_kval_enabled

    return qsa_write_kval_enabled()


def _kvw_extra_outs() -> tuple:
    """(CK_NEW, RING_NEW) under KVAL (+ WRITE_ALL), else (); one engagement line per process when either is on, read at
    build time: the WFOLD line (both flags) or the KVW-ALL line (VLLM_TPU_MK_QSA_WRITE_ALL alone)."""
    global _KVW_LOGGED
    kval, wall = _kvw_kval(), bool(envs.VLLM_TPU_MK_QSA_WRITE_ALL)
    if wall and not _KVW_LOGGED:
        _KVW_LOGGED = True
        import logging

        if kval:
            logging.getLogger("vllm." + __name__).info(
                "q38 WFOLD whole step: QSA cache writes take the kernel's values (VLLM_TPU_MK_QSA_WRITE_ALL=1 "
                "VLLM_TPU_MK_QSA_WRITE_KVAL=1)"
            )
        else:
            logging.getLogger("vllm." + __name__).info(
                "q38 KVW-ALL whole step: QSA cache writes as one all-layer Pallas call (VLLM_TPU_MK_QSA_WRITE_ALL=1)"
            )
    if not kval:
        return ()
    from vllm_torchtpu.megakernel.wholestep.qsa_full_c import CK_NEW, RING_NEW

    return (CK_NEW, RING_NEW)


def _kvw_outs(outs: Any) -> tuple:
    """The call's last two kernel outputs (qsa_ck_new, qsa_ring_new) under KVAL, else ()."""
    return tuple(outs[len(outs) - 2 :]) if _kvw_kval() else ()


def _kvw_specs(tp: Any) -> tuple:
    return (tp, tp) if _kvw_kval() else ()


def _kvw_kw(kv: Any, kouts: tuple, tokens: int) -> dict:
    """{"kernel_values": KernelValues} from the rank's qsa_kv_new and (ck_new, ring_new) under KVAL, else {}."""
    if not kouts:
        return {}
    from .qwen38_qsa_write_all import kernel_values_from_outputs

    return {"kernel_values": kernel_values_from_outputs(kv, *kouts, tokens)}
# q38 L1-KVW end


# q38 L2-UAISSUE: X-UA-ISSUE (wholestep/union_issue.py), VLLM_TPU_MK_UNION_ISSUE 0 / 1 / 2; registered in envs.py, so it
# keys the TPU compile cache. Merged into the D08 S1 options above: the lean class (qsa_full_lean) serves it.
def _uaissue_options() -> dict:
    """{} with VLLM_TPU_MK_UNION_ISSUE unset / 0 (the whole step is built exactly as before); else
    {"union_issue": N}, refused without union attention or together with `wholephys` (the level builds its own
    physical ids). Logs one engagement line per process."""
    global _UAISSUE_LOGGED
    lvl = _uaissue_level()  # q38 S3A-ENG T2: the pure resolver (end of module), shared with the bind twin; it raises as before
    if not lvl:
        return {}
    # The two refusals that stood here moved verbatim into _uaissue_level, same exception types and texts, same
    # order, raised before this point:
    #   a level outside 0 / 1 / 2;
    #   a level without union attention (wholeunion=1) or with wholephys.
    # These comment lines stand in place of the moved code, so every later line of this module keeps its
    # number.
    #
    if not _UAISSUE_LOGGED:
        _UAISSUE_LOGGED = True
        import logging

        logging.getLogger("vllm." + __name__).info(
            "q38 UAISSUE whole step: VLLM_TPU_MK_UNION_ISSUE=%d (%s)", lvl,
            "stage a: slab-bounded physical ids" if lvl == 1 else "stages a+b: slab-bounded physical ids + F0a issue",
        )
    return {"union_issue": lvl}


_UAISSUE_LOGGED = False


# q38 L2-UNPACK: X-UA-UNPACK (wholestep/union_unpack.py), VLLM_TPU_MK_UNION_UNPACK 0 / 1; registered in envs.py, so it keys the
# TPU compile cache. Merged into the D08 S1 options above (with X-UA-ISSUE's): the lean class (qsa_full_lean) serves it.
def _uaunpack_options() -> dict:
    """{} with VLLM_TPU_MK_UNION_UNPACK unset / 0 (the whole step is built exactly as before); else {"union_unpack": True},
    refused without union attention. Logs one engagement line per process."""
    global _UAUNPACK_LOGGED
    if not _uaunpack_on():  # q38 S3A-ENG T2: the pure resolver (end of module), shared with the bind twin; it raises as before
        return {}
    # The refusal that stood here (UNION_UNPACK without union attention) moved verbatim into _uaunpack_on, same
    # exception type and text. These comment lines keep every later line of this module at its number.
    if not _UAUNPACK_LOGGED:
        _UAUNPACK_LOGGED = True
        import logging

        logging.getLogger("vllm." + __name__).info(
            "q38 UAUNPACK whole step: VLLM_TPU_MK_UNION_UNPACK=1 (union tile body: full-vreg strided ring loads, pair 0)"
        )
    return {"union_unpack": True}


def _uaunpack_slots(slots: list) -> list:
    """`build_wholestep_call`'s per-QSA-layer (main slot, compressed slot) list, unchanged. With X-UA-UNPACK on, every
    layer's main-KV head-pair offset (`qsa_off` = max(main slot, 0)) must be 0: the strided reader takes K / V from bytes
    0 / 1 only (an unpaired pool, QWEN4_EXP_PACK_MAIN_KV=0); a paired layout is refused here, before anything is traced."""
    if _uaunpack_options():
        bad = [(i, m) for i, (m, _) in enumerate(slots) if max(int(m), 0) != 0]
        if bad:
            raise ValueError(
                f"VLLM_TPU_MK_UNION_UNPACK=1 reads head pair 0 only, but QSA layers {bad[:8]} (index, main slot) have a "
                "non-zero main-KV offset (paired main KV, QWEN4_EXP_PACK_MAIN_KV=1?); unset VLLM_TPU_MK_UNION_UNPACK"
            )
    return slots


_UAUNPACK_LOGGED = False


# q38 L2-UAPIPE: X-UA-PIPE (wholestep/union_attn.py `attend_request_pipe`, qsa_full_lean.py `_PipeMixin`).
# VLLM_TPU_MK_UNION_PIPE = bit mask: 1 co-issue (i), 2 cross-request (ii, needs VLLM_TPU_MK_SLOT_BATCH=1), 3 both, 5 / 7 =
# 1 / 3 with the starts after the unpack. Registered in envs.py, so it keys the TPU compile cache. 0 / unset: no option,
# so `wholestep_spec` builds MM's / D14's / the D08 S1 block exactly as before. Appended at the end of the file: no line
# above (some of which the whole-step trace passes through) moves.
def _union_pipe_options() -> dict:
    """{} when VLLM_TPU_MK_UNION_PIPE is 0, else {"union_pipe": mode}; one engagement log line per process."""
    global _UNION_PIPE_LOGGED
    # q38 S3A-ENG T2: pipe_mode(VLLM_TPU_MK_UNION_PIPE) moved into the pure resolver _union_pipe_mode (end of module)

    mode = _union_pipe_mode()  # shared with the bind twin s2_construct_line; validates and raises as before
    if not mode:
        return {}
    _xuav5_pipe_refusal(mode)  # q38 L2-XUAV5: the pipe class is not composed with the in-kernel indexer / early-q classes
    if not _UNION_PIPE_LOGGED:
        _UNION_PIPE_LOGGED = True
        import logging

        logging.getLogger("vllm." + __name__).info("q38 UAPIPE whole step: VLLM_TPU_MK_UNION_PIPE=%d", mode)
    return {"union_pipe": mode}


_UNION_PIPE_LOGGED = False


# q38 L2-XUAV5 (X-UA on STACK-D v5 F-v5c): `pipe_step_block` builds X-UA-PIPE's class on the lean block without the in-kernel
# indexer mixin (wholestep/qsa_full_idx.py `_IdxMixin`) or the early DCP q exchange (`_QxEarlyMixin`); with any of their
# options present it would take precedence over their routing and drop or reject those options. Until a composed class
# exists the pipe is refused together with them (L2 ruling 2026-11-14, option (a)).
XUAV5_PIPE_REFUSED_ENVS = tuple(e for _, e in IDX_SCORE_ENVS) + (
    "VLLM_TPU_MK_SELECT_RADIX", "VLLM_TPU_MK_UNION_QSPLIT", "VLLM_TPU_MK_TABLE_SLIM", "VLLM_TPU_MK_RING_S1",
    "VLLM_TPU_MK_DCP_QX_EARLY")


def _xuav5_pipe_refusal(mode: int) -> None:
    """Raise ValueError when VLLM_TPU_MK_UNION_PIPE (`mode` != 0) meets any in-kernel indexer env (scorer chunk / lanes /
    manager pages, radix, QSPLIT, TABLE_SLIM, RING_S1) or VLLM_TPU_MK_DCP_QX_EARLY set to a non-default value, or any of
    their block options."""
    dflt = {"VLLM_TPU_MK_SELECT_RADIX": "1"}  # every other refused env defaults to "0" / ""
    on = [e for e in XUAV5_PIPE_REFUSED_ENVS if str(getattr(envs, e) or "").strip() not in ("", "0", dflt.get(e, "0"))]
    opts = sorted(_idx_options())
    if on or opts:
        raise ValueError(
            f"VLLM_TPU_MK_UNION_PIPE={mode} is not composed with the in-kernel indexer / early-q options on this tree "
            f"(set: {on}; block options: {opts}); unset VLLM_TPU_MK_UNION_PIPE (q38 L2-XUAV5)"
        )


# q38 L1-G1 begin (GDN-COMMIT1, VLLM_TPU_MK_GDN_COMMIT1): with the flag a writing call skips the GDN checkpoint scatter and
# returns conv_ckpt / rec_ckpt / write ids / state ids, and the runner commits the accepted checkpoint per request and layer
# (megakernel_runtime/qwen38_gdn_commit.py); off, each helper hands back its first argument, so the flag-off program is
# unchanged. Appended last (after the X-UA tail); the call's lines are edited only in place.
def _g1_scatter(scatter: Any, write: bool) -> Any:
    from .qwen38_gdn_commit import scatter_fn

    return scatter_fn(scatter, write)


def _g1_ret(ret: tuple, write: bool, conv_ck: Any, rec_ck: Any, wb: Any, sts: Any, NS: int, R: int) -> tuple:
    from .qwen38_gdn_commit import call_outputs

    return call_outputs(ret, write, conv_ck, rec_ck, wb, sts, NS, R)


def _g1_specs(specs: tuple, write: bool, tp: Any, rep: Any) -> tuple:
    from .qwen38_gdn_commit import out_specs

    return out_specs(specs, write, tp, rep)
# q38 L1-G1 end
# q38 HXP1A begin
# L1-SELLOOP composed with the layer-47 fold (wholestep/qsa_full_l47f.py `_hxp1a_build` engages `select_owned` on the fold's block
# class), so `_selloop_options` no longer refuses the pair under VLLM_TPU_MK_L47_FOLD=1. False restores the L1-V6 refusal.
HXP1A_SELLOOP_L47F = True
# q38 HXP1A end


# q38 XKL begin
# XKL (DSTEP lane, 2026-11-23): the X-K-LIVE line of record, printed at whole-step module construction. `_live_rows_on` logs
# only when `build_wholestep_call` runs, i.e. when the whole step is TRACED; a warm compile cache loads the compiled program and
# never runs the custom op's body (M: HX-STEP-1 T62z j-0bb8a641, warm arm uu: 0 trace-time whole-step lines, the construction
# lines of qwen38_wholestep.py printed). `Qwen38WholeStep.__init__` (every engine start, warm or cold) calls this; it logs the
# per-bucket mode the trace builds (`_live_rows_on`'s rule: mode 2 refused below LIVE_GDN_COMPUTE_MIN_ROWS rows), without
# `_live_rows_on`'s side effects (its own trace-time line still prints on a cold trace). Flag unset / "0": no line, returns "".
# Appended after every existing line; nothing here is traced.
XKL_LINE_PREFIX = "q38 X-K-LIVE whole-step ("


def live_rows_construct_line(tag: str, buckets: Any) -> str:
    """One INFO line per call with VLLM_TPU_MK_LIVE_ROWS on (else none): the module's tag, its buckets and each bucket's
    `ExecSpec.live_rows` mode. Returns the message ("" when off)."""
    from vllm_torchtpu.megakernel.live_rows import live_rows_mode

    mode = live_rows_mode()
    if not mode:
        return ""
    modes = ",".join(
        "b%d:%d" % (int(b), 1 if (mode >= 2 and 0 < int(b) < LIVE_GDN_COMPUTE_MIN_ROWS) else mode) for b in buckets
    )
    msg = (
        "%s%s) built: VLLM_TPU_MK_LIVE_ROWS=1 modes %s (kernel input live_rows; the compiled whole step is keyed by the flag)"
        % (XKL_LINE_PREFIX, tag, modes)
    )
    import logging

    logging.getLogger("vllm." + __name__).info(msg)
    return msg
# q38 XKL end


# q38 HIX700 GDN-SIDECAR begin (VLLM_TPU_MK_GDN_SIDECAR; megakernel/models/qwen38_gdn_sidecar.py): with the flag the 16-row whole-step call takes
# three more operands right after the pool (gdn_side_rec / gdn_side_conv per rank, donated; gdn_side_on int32 [1] replicated; not last, so the
# operand tail L1-L47F's wrap_call indexes from the end keeps its place), feeds the kernel the accepted
# checkpoint index per layer and request (read_offsets[state id]), skips the checkpoint scatter (the kernel's checkpoints never leave as
# outputs on TPU: it writes them into the two donated sidecar buffers) and returns the two buffers and the write ids after its other outputs. Off, every helper
# hands back its argument, so the flag-off program is unchanged. Appended last; the call's lines are edited only in place (same lines).
def _side_blocks(spec: Any) -> list:
    from vllm_torchtpu.megakernel.models.qwen38_gdn_sidecar import Qwen38SidecarGDN

    if spec is None:
        return []
    return [getattr(b, "inner", b) for b in spec.all_blocks() if isinstance(getattr(b, "inner", b), Qwen38SidecarGDN)]


def _side_extra_outs(spec: Any, interpret: Any, g: Any, B: int, R: int) -> tuple:
    """CPU interpret mode only: the two interpret outputs (the sidecar buffers the kernel reads and writes there), else ()."""
    from vllm_torchtpu.megakernel.models.qwen38_gdn_sidecar import SIDE_CONV_OUT, SIDE_REC_OUT

    blks = _side_blocks(spec)
    if not blks or not interpret:
        return ()
    for b in blks:
        b.interpret_shapes = side_shapes_of(b, B, R)
    return (SIDE_REC_OUT, SIDE_CONV_OUT)


def side_shapes_of(blk: Any, B: int, R: int) -> tuple:
    """Per-rank shapes of the sidecar buffers of sidecar block `blk` at bucket B (= its rec_ckpt / conv_ckpt stacked over its GDN layers).
    `blk.geometry` is the whole step's inner geometry (`g.inner(R, L)`: global heads, NGDN); R = ranks."""
    from vllm_torchtpu.megakernel.models.qwen38_gdn_sidecar import side_shapes

    gi = blk.geometry
    lay = blk.lin_in_layout(gi, R)  # per-rank heads
    return side_shapes(gi.NGDN, B, B // T_VERIFY, T_VERIFY, lay.vh, lay.KD, lay.VD, lay.QKV)


def _side_take(args: Any, i: int, spec: Any) -> Any:
    """The three sidecar operands (rec, conv, on) right after the pool at args[i], or None without sidecar blocks in `spec`. They sit
    after the pool, not last, so the operand tail (positions .. the DCP rank) keeps its place for wrappers that index it from the end
    (L1-L47F's `wrap_call`)."""
    if not _side_blocks(spec):
        return None
    return tuple(args[i + 1 : i + 4])


def _side_xs(fn: Any, side: Any, off: Any, sts: Any, NS: int, T: int) -> dict:
    """The kernel's sidecar inputs: the two buffers, the on flag and the accepted index per layer and request, {} without the flag."""
    from vllm_torchtpu.megakernel.models.qwen38_gdn_sidecar import SIDE_CONV, SIDE_OFF, SIDE_ON, SIDE_REC

    if side is None or SIDE_REC not in getattr(fn, "extra_names", ()):
        return {}
    from .qwen38_segment_call import _fit

    s_rec, s_conv, s_on = side
    offs = jnp.stack([jnp.clip(off[_fit(jnp.asarray(st), NS)], 0, T - 1) for st in sts]).astype(jnp.int32)
    return {SIDE_REC: s_rec, SIDE_CONV: s_conv, SIDE_ON: s_on.reshape(1).astype(jnp.int32), SIDE_OFF: offs}


def _side_outs(fn: Any, outs: Any, pool: Any, side: Any, interpret: Any) -> tuple:
    """(outs with the pool in the two checkpoint slots, (new sidecar rec, new sidecar conv)); `outs`, () without the flag. On TPU the
    kernel declares no checkpoint outputs (it writes the donated sidecar inputs); in CPU interpret mode its two sidecar outputs come last."""
    from vllm_torchtpu.megakernel.models.qwen38_gdn_sidecar import SIDE_REC

    if side is None or SIDE_REC not in getattr(fn, "extra_names", ()):
        return outs, ()
    outs = tuple(outs)
    if interpret:
        ret, outs = outs[-2:], outs[:-2]
    else:
        ret = (side[0], side[1])
    return (outs[0], pool, pool) + outs[1:], ret


def _side_ret(ret: tuple, side_ret: tuple, wb: Any = None, R: int = 1) -> tuple:
    """`ret` + (sidecar rec, sidecar conv, the step's write ids tiled R times: each rank's shard = [NGDN, NS, T], as G1 returns them)."""
    if not side_ret:
        return tuple(ret)
    return tuple(ret) + tuple(side_ret) + (jnp.tile(wb.astype(jnp.int32), (R, 1, 1)),)


def _side_specs(spec: Any, call: Any, in_specs: tuple, out_specs: tuple, donate: tuple, tp: Any, rep: Any) -> tuple:
    """(call, in_specs, out_specs, donate_argnums) with the sidecar's three operands inserted right after the pool (donate[0]; the two
    buffers donated) and its two outputs appended after the call's others; unchanged without the flag."""
    if not _side_blocks(spec):
        return call, in_specs, out_specs, donate
    k = donate[0]  # the pool's operand index (1 + n_w)
    ins = tuple(in_specs[: k + 1]) + (tp, tp, rep) + tuple(in_specs[k + 1 :])
    return call, ins, tuple(out_specs) + (tp, tp, tp), (k, k + 1, k + 2)


def _side_building(B: int, fn: Any) -> Any:
    from vllm_torchtpu.megakernel.models.qwen38_gdn_sidecar import building_rows, sidecar_on

    return building_rows(B, fn) if sidecar_on() else fn
# q38 HIX700 GDN-SIDECAR end


# q38 S3A-ENG T2 begin (the S2 options bind twin; HIX-LEAD-6 rulings 2026-11-24 items 1-4 and 9). The S2 option helpers above log their
# engagement lines once per process, when they first run; on the served path that is `indexed_options()` (called by
# `Qwen38WholeStep.__init__` and by `build_wholestep_call`), and the lines carry no rank. This block holds each helper's decision as a
# PURE resolver (no logging, no module state written), which the helper itself now calls (in place above, same line count), and the
# twin `s2_construct_line`, called from `Qwen38WholeStep.__init__` on every rank at every engine start (warm or cold), which logs ONE
# line through logger.info (never info_once):
#   'q38 S2 construct rank=<r>: idx_twin=<0|1> scan_bound=<0|1> union_issue=<0|1|2> union_unpack=<0|1> selloop=<0|1|2> qx=<0|1>
#    d08s1=<bounded>,<lean>,<slot_batch> radix_bits=<1|2> table_slim=<0|1|2> union_qsplit=<0|1|2> union_pipe=<mode>'
# Each field is the value the helper puts into the indexed options (absent = 0; radix_bits absent = the kernel default 1), except
# table_slim, the strict env value VLLM_TPU_MK_TABLE_SLIM (its 2 is the option table_prefetch; 1 and 2 slim pages_per_seq). idx_twin is
# `indexed_options()`'s twin_split (this module, the HXIDX line), not qsa_full_idx. Appended after every existing line.
S2_CONSTRUCT_PREFIX = "q38 S2 construct rank="


def _idx_twin_on() -> bool:
    """`indexed_options()`'s twin_split: VLLM_TPU_MK_IDX_TWIN == "1" (strict getter)."""
    return str(envs.VLLM_TPU_MK_IDX_TWIN or "0") == "1"


def _d08s1_on() -> dict:
    """`_d08s1_options`' stage switches {union_bounded, union_lean, slot_batch} -> bool."""
    return {k: bool(getattr(envs, e)) for k, e in D08S1_ENVS}


def _uaissue_level() -> int:
    """`_uaissue_options`' level: 0 off, else 1 / 2, raising exactly as that helper did."""
    lvl = int(envs.VLLM_TPU_MK_UNION_ISSUE)
    if not lvl:
        return 0
    if lvl not in (1, 2):
        raise ValueError(f"VLLM_TPU_MK_UNION_ISSUE={lvl}: 0 off, 1 stage (a), 2 stages (a) + (b)")
    if not UNION_ATTN or UNION_PHYS:
        raise ValueError(
            f"VLLM_TPU_MK_UNION_ISSUE={lvl} needs wholeunion=1 and wholephys off "
            f"(wholeunion={int(UNION_ATTN)}, wholephys={int(UNION_PHYS)})"
        )
    return lvl


def _uaunpack_on() -> bool:
    """`_uaunpack_options`' switch, raising exactly as that helper did."""
    if not envs.VLLM_TPU_MK_UNION_UNPACK:
        return False
    if not UNION_ATTN:
        raise ValueError("VLLM_TPU_MK_UNION_UNPACK=1 needs wholeunion=1 (it rewrites the union tile body's ring read)")
    return True


def _idx_base() -> dict:
    """`_idx_options`' dict before the QX and HXSCAN options: the scorer overrides, radix_bits, union_qsplit, table_prefetch."""
    out = score_options()
    if str(envs.VLLM_TPU_MK_SELECT_RADIX) == "2":
        out["radix_bits"] = 2
    if int(envs.VLLM_TPU_MK_UNION_QSPLIT or 0):
        out["union_qsplit"] = int(envs.VLLM_TPU_MK_UNION_QSPLIT)
    if str(envs.VLLM_TPU_MK_TABLE_SLIM) == "2":
        out["table_prefetch"] = True
    return out


def _qx_state() -> Any:
    """`_qx_options`' decision: None when VLLM_TPU_MK_DCP_QX_EARLY is not "1", else (on, main-KV DCP, compressed-key DCP) with
    on = DCP > 1 and no compressed-key DCP."""
    if str(envs.VLLM_TPU_MK_DCP_QX_EARLY) != "1":
        return None
    dcp, cdcp = main_kv_dcp(), compressed_dcp()
    return (dcp > 1 and not cdcp), dcp, cdcp


def _selloop_mode() -> int:
    """`_selloop_options`' mode (0 / 1 / 2), raising the L1-V6 refusal exactly as that helper did."""
    mode = int(envs.VLLM_TPU_MK_SELECT_OWNED or 0)
    if not mode:
        return 0
    if bool(envs.VLLM_TPU_MK_L47_FOLD) and not HXP1A_SELLOOP_L47F:
        raise ValueError(
            f"VLLM_TPU_MK_SELECT_OWNED={mode} is refused together with VLLM_TPU_MK_L47_FOLD=1: the fold rebuilds the block "
            "class with the sink export and would drop the owned-slot loop; unset one of them (q38 L1-V6)"
        )
    return mode


def _union_pipe_mode() -> int:
    """`_union_pipe_options`' mode: pipe_mode(VLLM_TPU_MK_UNION_PIPE), validated there (0 = off)."""
    from vllm_torchtpu.megakernel.wholestep.union_attn import pipe_mode

    return pipe_mode(envs.VLLM_TPU_MK_UNION_PIPE)


def _xuav5_pipe_check(mode: int, idx_opts: dict) -> None:
    """`_xuav5_pipe_refusal(mode)` with the indexer options given (the twin passes its pure `_idx_options` dict, so nothing is logged);
    same condition, same exception type and text."""
    dflt = {"VLLM_TPU_MK_SELECT_RADIX": "1"}
    on = [e for e in XUAV5_PIPE_REFUSED_ENVS if str(getattr(envs, e) or "").strip() not in ("", "0", dflt.get(e, "0"))]
    opts = sorted(idx_opts)
    if on or opts:
        raise ValueError(
            f"VLLM_TPU_MK_UNION_PIPE={mode} is not composed with the in-kernel indexer / early-q options on this tree "
            f"(set: {on}; block options: {opts}); unset VLLM_TPU_MK_UNION_PIPE (q38 L2-XUAV5)"
        )


def s2_construct_fields() -> dict:
    """The resolved S2 build options, evaluated in `indexed_options()`' order (D08 S1 with UAISSUE / UAUNPACK, the indexer options with
    QX then HXSCAN, SELLOOP, UAPIPE, twin_split), so an invalid setting raises what `indexed_options()` raises. Pure."""
    from ..megakernel.wholestep.qsa_full_idx import scan_bound_resolve

    d08 = _d08s1_on()
    issue = _uaissue_level()
    unpack = _uaunpack_on()
    base = _idx_base()
    qx = _qx_state()
    qx_on = bool(qx is not None and qx[0])
    scan = scan_bound_resolve(base, envs)
    idx = dict(base, **({"qx_early": True} if qx_on else {}), **({"scan_bound": 1} if scan else {}))
    sel = _selloop_mode()
    pipe = _union_pipe_mode()
    if pipe:
        _xuav5_pipe_check(pipe, idx)
    return dict(
        idx_twin=int(_idx_twin_on()), scan_bound=int(scan), union_issue=int(issue), union_unpack=int(unpack), selloop=int(sel),
        qx=int(qx_on), d08s1=",".join(str(int(d08[k])) for k, _ in D08S1_ENVS) if any(d08.values()) else "0,0,0",
        radix_bits=int(base.get("radix_bits", 1)), table_slim=int(str(envs.VLLM_TPU_MK_TABLE_SLIM)),
        union_qsplit=int(base.get("union_qsplit", 0)), union_pipe=int(pipe),
    )


S2_FIELDS = ("idx_twin", "scan_bound", "union_issue", "union_unpack", "selloop", "qx", "d08s1", "radix_bits", "table_slim",
             "union_qsplit", "union_pipe")


def s2_construct_line(rank: int) -> str:
    """Log (logger.info, every call, every rank) and return the S2 construct line of this rank. A resolve error is reported in the
    line, not raised (`indexed_options()` in the same `__init__` has already raised for any invalid setting)."""
    import logging

    try:
        f = s2_construct_fields()
        body = " ".join("%s=%s" % (k, f[k]) for k in S2_FIELDS)
    except Exception as e:  # noqa: BLE001  never stop an engine start at bind
        body = "resolve error %s: %s" % (type(e).__name__, e)
    msg = "%s%d: %s" % (S2_CONSTRUCT_PREFIX, int(rank), body)
    logging.getLogger("vllm." + __name__).info(msg)
    return msg
# q38 S3A-ENG T2 end


# q38 HIX-GLUE GM begin (VLLM_TPU_HXG_GDN_META; vllm_torchtpu/hixglue.py). Appended; `gdn_meta_shared` and `call_shared` carry
# one-line hooks only, so the flag-off lowering is unchanged.
def _hxg_gm_on() -> bool:
    from vllm_torchtpu import hixglue

    return hixglue.on("VLLM_TPU_HXG_GDN_META")


def side_offsets_vectorised(off: Any, sts: Any, NS: int, T: int) -> Any:
    """The GDN-SIDECAR accepted index per layer and request, `[NL, NS]` int32: `clip(off[st_l[:NS]], 0, T - 1)` for every layer in
    one gather over the stacked state ids (`_side_xs`'s list does one fit + gather + clip chain per layer). Bit-identical."""
    from .qwen38_segment_call import _fit, _fit_requests

    if len({tuple(jnp.shape(s)) for s in sts}) == 1:
        st = _fit_requests(jnp.stack([jnp.asarray(s) for s in sts]), NS)
    else:
        st = jnp.stack([_fit(jnp.asarray(s), NS) for s in sts])
    return jnp.clip(off[st], 0, T - 1).astype(jnp.int32)


def _side_xs_gm(fn: Any, side: Any, off: Any, sts: Any, NS: int, T: int) -> dict:
    """`_side_xs` with the accepted-index operand built by `side_offsets_vectorised` when VLLM_TPU_HXG_GDN_META is on."""
    if not _hxg_gm_on():
        return _side_xs(fn, side, off, sts, NS, T)
    from vllm_torchtpu.megakernel.models.qwen38_gdn_sidecar import SIDE_CONV, SIDE_OFF, SIDE_ON, SIDE_REC

    if side is None or SIDE_REC not in getattr(fn, "extra_names", ()):
        return {}
    s_rec, s_conv, s_on = side
    return {SIDE_REC: s_rec, SIDE_CONV: s_conv, SIDE_ON: s_on.reshape(1).astype(jnp.int32),
            SIDE_OFF: side_offsets_vectorised(off, sts, NS, T)}


def gm_bind_line(rank: int, tag: str, gdn_shared: bool, buckets: Any) -> str:
    """GM's per-rank bind line (logger.info, every call); refuses GM without the shared program (VLLM_TPU_MK_GDN_META_DEDUP=0)."""
    from vllm_torchtpu import hixglue

    if not gdn_shared:
        raise ValueError("VLLM_TPU_HXG_GDN_META=1 needs VLLM_TPU_MK_GDN_META_DEDUP=1 (the shared-GDN-metadata program it extends)")
    return hixglue.bind_line("VLLM_TPU_HXG_GDN_META", rank, "whole step (%s): shared GDN metadata program at every bucket (%s), "
                             "sidecar accepted index vectorised" % (tag, ",".join(str(int(b)) for b in buckets)))
# q38 HIX-GLUE GM end
