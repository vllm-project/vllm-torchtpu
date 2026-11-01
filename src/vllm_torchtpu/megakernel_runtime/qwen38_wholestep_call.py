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
count `B`. `fn(hidden, *weights, pool, *gdn_meta, *qsa_tables, positions, starts, context,
trig)`: pads rows to the kernel's `M`, derives the GDN read / write blocks (as the segment
call) and the QSA layers' extras with the stock addressing (`qwen38_wholestep_inputs`), runs
the kernel, then — in the same function, on the donated pool — scatters the GDN checkpoints
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
from vllm_torchtpu.megakernel.wholestep.qsa_full_c import IR_NEW, KR_NEW, UNION_STATS
from vllm_torchtpu.megakernel.wholestep.qsa_full import KV_NEW
from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec

from .qwen38_scatter import scatter_gdn_ckpts_fast
from .qwen38_segment_call import GDN_META, T_VERIFY, gdn_blocks, ring_depths, segment_tiles
from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import main_kv_dcp

from .qwen38_wholestep_cache import QSALayerCache, wholestep_qsa_writes
from .qwen38_wholestep_inputs import qsa_step_inputs

QSA_TABLES = 3  # per QSA layer: ring ids, main table, compressed table
# Served indexer options: page tables in HBM with per-layer SMEM windows, one request's scores
# at a time, and only the key chunks a request's rows can see (exact vs the stock selection,
# tests/megakernel/test_wholestep.py long mode). Without the last, every request streams
# max_groups keys per QSA layer: 22.4 ms at 64 rows for max_groups 8192 (j-dcafcac7).
LONG_CONTEXT = dict(hbm_tables=True, per_request_scores=True, bounded_chunks=True)
# `VLLM_TPU_MEGAKERNEL=...,wholeshard=1`: each rank scores and selects its own run of requests and
# the ranks exchange the selections in-kernel (`QSAFullIndexedBlock.shard_index`; exact).
SHARD_INDEX = False
# `...,wholeunion=1`: each request's verify rows attend one union of their selections, streamed
# once (`QSAFullIndexedBlock.union_attn`; same tokens per row, f32 order of the softmax tiles).
UNION_ATTN = False
# `...,wholestats=N` (with `wholeunion=1`): the call also returns per QSA layer and request the
# union size vs the rows' selections (`qsa_union_stats`); the runner logs them every N calls.
UNION_STATS_EVERY = 0
# `...,wholeloop=1`: a QSA layer's per-request work in a fori_loop (compile size flat in B).
LOOP_REQUESTS = False
# `...,wholephys=1` (with `wholeunion=1`): union ids -> physical micro-page ids in one vector pass
# per request, so each attention copy issues from one SMEM load (same addresses; RPAd's recipe).
UNION_PHYS = False


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
        compressed_dcp=compressed_dcp(),
    )


def wholestep_tiles(B: int, max_groups: int = 0) -> Tiles:
    """`_wholestep_tiles`; with main-KV DCP-2 up to 32 rows the expert ring drops one more 2 MiB
    slot: the pair exchange buffers and the doubled-head union attention put b32 (f32 state)
    over scoped VMEM (tpu_bench_wholestep_call j-677db355: CompileTimeScopedVmemOom), and the
    owned-only per-rank unions (`qu_id` / `qu_cd` / `qgu_*` x2) b16 too (served j-af65b422:
    62.49M at qwen38_mk_whole_b16)."""
    t = _wholestep_tiles(B, max_groups)
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
    rings = os.environ.get("VLLM_TPU_MK_RINGS", "")
    if rings:
        nbe, nbd = (int(x) for x in rings.split(":"))
        t = dataclasses.replace(t, nbe=nbe, nbd=nbd)
    return t


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
    # VLLM_TPU_MK_SCOPES=N: per-phase named scopes in the kernel (profiling only).
    scopes = int(os.environ.get("VLLM_TPU_MK_SCOPES", "0") or 0)
    ex = ExecSpec(
        R=R, B=B, T=T, ctx=128, tiles=wholestep_tiles(B, max_groups) if tiles is None else tiles,
        interpret=interpret, debug=DebugOptions(scopes=scopes),
    )
    n_att = L // g.att_every
    n_gdn = L - n_att
    assert len(qsa_layers) == n_att, (len(qsa_layers), n_att)
    pps = max_groups * 4 // 128
    spec = wholestep_spec(
        g, R, L=L, first="qsa", pool_gdn=True, qsa=dict(budget=budget), step=True,
        indexed=dict(max_groups=max_groups, pages_per_seq=pps, carrier="fp8", n_layers=n_att, **indexed_options()),
    )
    stats = union_stats_on()
    fn, k_specs, _, n_w = make_flat_segment_step(
        mesh, spec, ex, extra_outs=(KV_NEW, KR_NEW, IR_NEW) + ((UNION_STATS,) if stats else ())
    )
    M, NS = ex.M, B // T
    tp, rep = P("tp"), P()
    # Planned global weight shapes: the packer hands tiled operands over 3-D ([L*nt, tk, N],
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
        lambda h: jnp.pad(h, ((0, M - B), (0, 0))), mesh=mesh, in_specs=tp, out_specs=tp,
        check_vma=False,
    )
    unpad = jax.shard_map(lambda r: r[:B], mesh=mesh, in_specs=tp, out_specs=tp, check_vma=False)
    pos0 = jnp.zeros((B,), jnp.int32)
    n_meta = GDN_META * n_gdn
    n_tab = QSA_TABLES * n_att
    slots = [(lay.main_slot, lay.compressed_slot) for lay in qsa_layers]
    scales = (qsa_layers[0].k_scale, qsa_layers[0].v_scale)
    assert all((lay.k_scale, lay.v_scale) == scales for lay in qsa_layers), "one KV scale pair"

    names = (
        "qsa_pool", "qsa_ck", "qsa_cktab", "qsa_nvis", "qsa_tail", "qsa_mtab", "qsa_off",
        "qsa_scales", "qsa_rope", "qsa_ctx", "qsa_ring", "qsa_rope_ck",
    )
    out_specs_in = tuple(tp if n in hbm else rep for n in names)

    def per_rank_inputs(pool, tables_flat, positions, starts, context, trig):
        tables = [tuple(tables_flat[i : i + QSA_TABLES]) for i in range(0, n_tab, QSA_TABLES)]
        d = qsa_step_inputs(
            pool, tables, slots, trig, positions, starts, context, M=M, T=T,
            pages_per_seq=pps, max_groups=max_groups, k_scale=scales[0], v_scale=scales[1],
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
        pool = scatter_gdn_ckpts_fast(pool, conv_ck, rec_ck, wb, layout=layout, interpret=interpret)
        if not write:
            return pool
        Bq = len(positions)
        rows = [(kr[o, :Bq], kv[o, :Bq, kv.shape[-1] // 2 :], ir[o, :Bq]) for o in range(n_att)]
        norms = [tuple(norms_flat[2 * o : 2 * o + 2]) for o in range(n_att)]
        tables = [tuple(tables_flat[i : i + QSA_TABLES]) for i in range(0, n_tab, QSA_TABLES)]
        # Main-KV DCP: this rank's pair rank, the QSA layers' `dcp_rank` buffer (int32 [1] per
        # rank; `axis_index` lowers to a partition id, which the served SPMD compile refuses).
        dcp = (rank[0].reshape(1), dcp_size) if dcp_size > 1 else None
        return wholestep_qsa_writes(
            pool, rows, norms, tables, qsa_layers, positions, trig, starts, context,
            compress_ratio=4, norm_eps=norm_eps, rotary_dim=rotary_dim, max_query_len=T,
            num_requests=M // T, dcp=dcp, compressed_dcp=dcp if compressed_dcp() else None,
        )

    writes = jax.shard_map(
        per_rank_writes,
        mesh=mesh,
        in_specs=(tp, tp, tp, rep, tp, tp, tp, rep, rep, rep, rep, rep, rep)
        + ((tp,) if dcp_size > 1 else ()),
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
        pool = args[i]
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
        xs.update(gdn_pool=pool, gdn_rb=rb, gdn_hi=hi.astype(jnp.int32))
        outs = fn(pos0, x, *(xs[k] for k in fn.extra_names), *w)
        res, conv_ck, rec_ck, kv, kr, ir = outs[:6]
        pool = writes(pool, conv_ck, rec_ck, wb, kv, kr, ir, tables, positions, starts, context,
                      trig, norms, *rank)
        return (unpad(res), pool, outs[6]) if stats else (unpad(res), pool)

    in_specs = (
        (tp,) + (tp,) * n_w + (tp,) + (rep,) * n_meta + (rep,) * n_tab + (rep,) * 4
        + (rep,) * (2 * n_att) + ((tp,) if dcp_size > 1 else ())
    )
    return call, in_specs, (tp, tp, tp) if stats else (tp, tp), (1 + n_w,)

