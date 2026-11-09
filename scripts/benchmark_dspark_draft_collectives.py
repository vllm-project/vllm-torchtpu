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
"""Microbenchmark of the collectives of one MiniMax-M3.1 DSpark draft forward.

Served setup: TP8 + DCP8 on the 8 cores of a 2x2x1 pod (the DCP group is the
TP group), draft batch ``T = padded_reqs * 7`` rows (c64: T=512, R=73; c16:
T=128, R=18). One draft forward (`DSparkProposer._dflash_forward_and_sample`)
runs 33 collectives:

  ar      11 x all-reduce bf16[T, 6144]: embedding (vocab-parallel), and per
          layer o_proj and MLP down (RowParallelLinear)
  agqkv    5 x all-gather bf16[T, 10, 128] -> [T, 80, 128] (`gather_group`:
          q | k | v of the rank; only q is read afterwards)
  aglse    5 x all-gather f32[T, 64] -> [8, T, 64] (`cp_lse_ag_out_rs`)
  rs       5 x reduce-scatter f32[64, T, 128] -> [8, T, 128] (same; the
          weighted partial outputs of all 64 group heads, heads-major)
  markov   7 x all-gather f32[R, 2] -> [R, 16] (vocab-parallel Markov argmax,
          serial over the 7 draft slots)

Two modes, both inside one `shard_map` over the 8 cores (axis ``x``), every
iteration fed from the previous one's output (no CSE, no hoisting), and the
per-iteration time taken as the slope ``(t(2L) - t(L)) / L`` of a `fori_loop`:

  class   each collective class alone, served shape, per variant, minus the
          same loop with the collective replaced by a local op ("net").
  step    a mock draft forward at served per-rank shapes: real Pallas draft
          flash attention (`attend_cached_flash`) on an fp8 paged cache,
          real block attention + LSE merge, bf16 matmuls of the per-rank
          weight shards, the vocab-parallel LM head and the 7-step Markov
          loop. Variants change only the collectives (or their flags), so
          ``t(variant) - t(base)`` is what the change saves in a step and
          ``t(base) - t(nocoll)`` is the step's exposed collective time.

Variants (``--variants``; the per-program XLA options are passed with
``Lowered.compile(compiler_options=...)``, which is how a flag can be scoped
to the draft program without touching the target's):

  base        served collectives, served LIBTPU_INIT_ARGS
  ar_tc       all-reduce on the TensorCore (SC offload threshold 1 TiB)
  ar_flat     SC offload without the hierarchical algorithm
  ar_rsag     all-reduce as reduce-scatter + all-gather
  rs_tc       merge reduce-scatter on the TensorCore (RS SC offload off)
  ag_sc       all-gathers SC-offloaded (min size 0)
  agq         all-gather only q (k/v of the group are not read)  [exact]
  rs_bf16     merge reduce-scatter in bf16 (`rs_dtype`)  [LOWER PRECISION]
  a2a2        merge as all-to-all of bf16 out + all-to-all of f32 LSE,
              then a local f32 merge  [same precision as base]
  a2a1        the same in one all-to-all (bf16 out and f32 LSE packed as
              32-bit words per destination rank)  [same precision as base]
  split2      the post-attention tail (o_proj AR, MLP, down AR) in two row
              halves so one half's AR can hide under the other's MLP
  split2nc    split2 with the SC AR combiner limited to one operand
  best        agq + a2a1 + ar_tc
  nocoll      every collective replaced by a local op of the same shape
              (step: the collective-free floor; numerically meaningless)

Before timing, ``a2a1``/``a2a2``/``rs_bf16`` are checked on device against the
served merge (max |diff| of the merged bf16 output and LSE). The compiled
HLO of every program is scanned for collectives and the SparseCore marker,
so the offload choice of each flag is read from the program, not assumed.

Usage (TPU, 8 cores; served flags in LIBTPU_INIT_ARGS, see the run script):
  PYTHONPATH=<worktree>/src python3 benchmark_dspark_draft_collectives.py \\
      --mode class,step --cells c64,c16
CPU (correctness only):
  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
      PYTHONPATH=<worktree>/src python3 benchmark_dspark_draft_collectives.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import vllm_torchtpu  # noqa: F401  (env_override: served libtpu defaults before jax init)
from vllm_torchtpu.layers.adapter import cp_draft_attention as cda

AX = "x"
N = 8  # TP = DCP
HID, HD, QH, GQ, G, FFN = (
    6144,
    128,
    8,
    64,
    4,
    1536,
)  # per rank: 8 q heads; 64 group q heads
VOCAB, VSHARD, MRANK = 200064, 25008, 256
LAYERS, K, PAGE, WINDOW = 5, 7, 128, 4096
K_SCALE, V_SCALE = 0.7, 0.3
SM_SCALE = HD**-0.5
CELLS = {"c4": (4, 32), "c16": (16, 128), "c64": (64, 512), "c128": (128, 1024)}
TIB = str(1 << 40)

# Per-program XLA options of each variant (on top of LIBTPU_INIT_ARGS).
OPTS = {
    "ar_tc": {"xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes": TIB},
    "ar_flat": {"xla_tpu_enable_sparse_core_hierarchical_all_reduce": "false"},
    "rs_tc": {"xla_tpu_enable_sparse_core_collective_offload_reduce_scatter": "false"},
    "ag_sc": {
        "xla_tpu_enable_sparse_core_collective_offload_all_gather": "true",
        "xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes": "0",
    },
    "split2nc": {
        "xla_tpu_sparse_core_all_reduce_offload_max_num_operands": "1",
        "xla_tpu_arf_combiner_threshold_in_bytes": "0",
    },
    "best": {"xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes": TIB},
}

SMOKE = False


# ----------------------------------------------------------------------------
# Collectives as the served torch code lowers them (vLLM GroupCoordinator:
# all_gather(dim) gathers a new major axis then moves it to `dim`;
# reduce_scatter(dim) moves `dim` to the front first).
# ----------------------------------------------------------------------------
def ag_dim(x, dim, local=False):
    if local:
        g = jnp.broadcast_to(x[None], (N,) + x.shape) * jnp.asarray(1, x.dtype)
    else:
        g = lax.all_gather(x, AX, axis=0, tiled=False)
    g = jnp.moveaxis(g, 0, dim)
    return g.reshape(x.shape[:dim] + (N * x.shape[dim],) + x.shape[dim + 1 :])


def rs_dim1(x, local=False):
    y = jnp.swapaxes(x, 0, 1)
    if local:
        y = y[: y.shape[0] // N] * jnp.asarray(N, y.dtype)
    else:
        y = lax.psum_scatter(y, AX, scatter_dimension=0, tiled=True)
    return jnp.swapaxes(y, 0, 1)


def all_reduce(x, how="psum"):
    if how == "none":
        return x * jnp.asarray(1, x.dtype)
    if how == "rsag":
        r = lax.psum_scatter(x, AX, scatter_dimension=0, tiled=True)
        return lax.all_gather(r, AX, axis=0, tiled=True)
    return lax.psum(x, AX)


# ----------------------------------------------------------------------------
# DCP LSE merge of the cached-window partials (`dcp_comm.cp_lse_ag_out_rs`).
# out [T, 64, D] bf16 (all group heads, this rank's KV), lse [T, 64] f32.
# Returns this rank's 8 heads: merged [T, 8, D] bf16 and merged lse [T, 8].
# ----------------------------------------------------------------------------
def merge_agrs(out, lse, rs_dtype=None, local=False):
    """Served: AG of the f32 LSE, f32 weights, RS of the weighted f32 out."""
    lse = lse.astype(jnp.float32)
    lses = ag_dim(lse[None], 0, local)  # [8, T, 64]
    merged_lse = jax.nn.logsumexp(lses, axis=0)
    weight = jnp.exp(lse - merged_lse)[..., None]
    weighted = out.astype(jnp.float32) * weight
    weighted = jnp.where(weight > 0, weighted, 0.0)
    if rs_dtype is not None:
        weighted = weighted.astype(rs_dtype)
    merged = rs_dim1(weighted, local).astype(out.dtype)
    rank = lax.axis_index(AX)
    mlse = lax.dynamic_slice_in_dim(merged_lse, rank * QH, QH, axis=1)
    return merged, mlse


def _local_merge(outs, lses, dtype):
    """outs [8(src), T, 8, D] bf16, lses [8(src), T, 8] f32 -> the served merge
    of this rank's heads, in f32 (same terms as `merge_agrs`)."""
    merged_lse = jax.nn.logsumexp(lses, axis=0)
    weight = jnp.exp(lses - merged_lse)[..., None]
    weighted = outs.astype(jnp.float32) * weight
    weighted = jnp.where(weight > 0, weighted, 0.0)
    return jnp.sum(weighted, axis=0).astype(dtype), merged_lse


def merge_a2a2(out, lse, local=False):
    """All-to-all of the bf16 partial outputs and of the f32 LSEs (each rank
    gets every rank's partials of its own 8 heads), then a local f32 merge."""
    T = out.shape[0]
    o = out.reshape(T, N, QH, HD).transpose(1, 0, 2, 3)  # [dst, T, 8, D]
    w = lse.astype(jnp.float32).reshape(T, N, QH).transpose(1, 0, 2)  # [dst, T, 8]
    if not local:
        o = lax.all_to_all(o, AX, 0, 0, tiled=True)  # [src, T, 8, D]
        w = lax.all_to_all(w, AX, 0, 0, tiled=True)
    return _local_merge(o, w, out.dtype)


def merge_a2a1(out, lse, local=False):
    """`merge_a2a2` in one all-to-all: per destination, the bf16 out (as
    32-bit words) then the f32 LSE bits."""
    T = out.shape[0]
    o = out.reshape(T, N, QH, HD // 2, 2).transpose(1, 0, 2, 3, 4)
    ow = lax.bitcast_convert_type(o, jnp.uint32).reshape(N, -1)  # [dst, T*8*64]
    lw = lax.bitcast_convert_type(
        lse.astype(jnp.float32).reshape(T, N, QH).transpose(1, 0, 2), jnp.uint32
    ).reshape(N, -1)  # [dst, T*8]
    buf = jnp.concatenate([ow, lw], axis=1)
    if not local:
        buf = lax.all_to_all(buf, AX, 0, 0, tiled=True)  # [src, ...]
    no = T * QH * (HD // 2)
    outs = lax.bitcast_convert_type(
        buf[:, :no].reshape(N, T, QH, HD // 2), jnp.bfloat16
    )
    outs = outs.reshape(N, T, QH, HD)
    lses = lax.bitcast_convert_type(buf[:, no:].reshape(N, T, QH), jnp.float32)
    return _local_merge(outs, lses, out.dtype)


MERGES = {
    "agrs": merge_agrs,
    "rs_bf16": lambda o, s, local=False: merge_agrs(o, s, jnp.bfloat16, local),
    "a2a2": merge_a2a2,
    "a2a1": merge_a2a1,
}


def variant_cfg(v):
    cfg = dict(ar="psum", agq="qkv", merge="agrs", split=1, local=False, markov=True)
    if v == "ar_rsag":
        cfg["ar"] = "rsag"
    elif v == "agq":
        cfg["agq"] = "q"
    elif v in ("rs_bf16", "a2a2", "a2a1"):
        cfg["merge"] = v
    elif v in ("split2", "split2nc"):
        cfg["split"] = 2
    elif v == "best":
        cfg.update(agq="q", merge="a2a1")
    elif v == "nocoll":
        cfg.update(ar="none", local=True)
    return cfg


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def make_cell(rng, live, T, num_pages):
    R = T // K
    ctx = np.zeros(R, np.int32)
    lo, hi = (300, 3000) if SMOKE else (8192, 120000)
    ctx[:live] = rng.integers(lo, hi, live)
    q_lens = np.zeros(R, np.int32)
    q_lens[:live] = K
    logical = int(-(-(ctx.max() + K) // (PAGE * N))) + 2
    perm = rng.permutation(num_pages)
    tables = np.stack([np.roll(perm, -i * logical)[:logical] for i in range(R)]).astype(
        np.int32
    )
    cu = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    return dict(seq_lens=ctx + q_lens, tables=tables.reshape(-1), cu=cu, R=R)


def per_device(mesh, shape, dtype, seed, scale=1.0, fp8=False):
    """[N, *shape] sharded one slice per device, generated on device."""
    sh = NamedSharding(mesh, P(AX))

    def gen(key):
        if fp8:  # random e4m3 bit patterns without NaN codes
            bits = jax.random.bits(key, (N,) + shape, jnp.uint8)
            return lax.bitcast_convert_type(bits & jnp.uint8(0x77), jnp.float8_e4m3fn)
        return (jax.random.normal(key, (N,) + shape, jnp.float32) * scale).astype(dtype)

    return jax.jit(gen, out_shardings=sh)(jax.random.PRNGKey(seed))


def replicate_meta(mesh, arr):
    sh = NamedSharding(mesh, P(AX))
    return jax.device_put(np.broadcast_to(arr, (N,) + arr.shape).copy(), sh)


# ----------------------------------------------------------------------------
# Mock draft forward (per device)
# ----------------------------------------------------------------------------
def rms(x, w):
    xf = x.astype(jnp.float32)
    return (
        xf * lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + 1e-6) * (1 + w)
    ).astype(jnp.bfloat16)


def draft_forward(ids, p, cell, cfg):
    """ids [T] int32 -> (draft ids [R, K] int32, hidden [T, HID])."""
    T = ids.shape[0]
    R = T // K
    local = cfg["local"]
    rank = lax.axis_index(AX)
    sl, bt, cu = cell
    # Embedding: vocab-parallel gather, out-of-shard rows zero, all-reduce.
    lid = ids - rank * VSHARD
    inside = (lid >= 0) & (lid < VSHARD)
    h = jnp.where(inside[:, None], p["table"][jnp.clip(lid, 0, VSHARD - 1)], 0).astype(
        jnp.bfloat16
    )
    h = all_reduce(h, cfg["ar"])
    res = jnp.zeros_like(h)
    for i in range(LAYERS):
        res = res + h
        hn = rms(res, p["ln1"][i])
        qkv = jnp.dot(hn, p["wqkv"][i], preferred_element_type=jnp.float32).astype(
            jnp.bfloat16
        )
        qkv = qkv.reshape(T, QH + 2, HD)
        q = rms(qkv[:, :QH], p["qn"][i])
        k = rms(qkv[:, QH : QH + 1], p["kn"][i])
        v = qkv[:, QH + 1 :]
        if cfg["agq"] == "q":
            q_all = ag_dim(q, 1, local)
        else:
            g = ag_dim(jnp.concatenate([q, k, v], axis=1), 1, local)  # [T, 80, D]
            q_all = g.reshape(T, N, QH + 2, HD)[:, :, :QH].reshape(T, GQ, HD)
        ctx_out, ctx_lse = cda.attend_cached_flash(
            q_all,
            p["cache"],
            sl,
            bt,
            cu,
            kv_heads=G,
            max_query_len=K,
            sliding_window=WINDOW,
            sm_scale=SM_SCALE,
            dcp=N,
            rank=rank,
            k_scale=K_SCALE,
            v_scale=V_SCALE,
            interpret=SMOKE,
        )
        merged, mlse = MERGES[cfg["merge"]](ctx_out, ctx_lse, local=local)
        kq = (
            (k.astype(jnp.float32) / K_SCALE)
            .astype(jnp.float8_e4m3fn)
            .astype(jnp.float32)
            * K_SCALE
        ).astype(jnp.bfloat16)
        vq = (
            (v.astype(jnp.float32) / V_SCALE)
            .astype(jnp.float8_e4m3fn)
            .astype(jnp.float32)
            * V_SCALE
        ).astype(jnp.bfloat16)
        bo, bl = cda.attend_block(
            q, kq, vq, sl, cu, max_query_len=K, sm_scale=SM_SCALE, causal=False
        )
        attn = cda.merge(merged.astype(bo.dtype), mlse, bo, bl)[0].reshape(T, QH * HD)

        def tail(a, r, i=i):
            o = jnp.dot(a, p["wo"][i], preferred_element_type=jnp.float32).astype(
                jnp.bfloat16
            )
            o = all_reduce(o, cfg["ar"])
            r = r + o
            x = rms(r, p["ln2"][i])
            gu = jnp.dot(x, p["wgu"][i], preferred_element_type=jnp.float32)
            gate, up = gu[:, :FFN], gu[:, FFN:]
            act = (gate * jax.nn.sigmoid(1.702 * gate) * (up + 1)).astype(jnp.bfloat16)
            d = jnp.dot(act, p["wd"][i], preferred_element_type=jnp.float32).astype(
                jnp.bfloat16
            )
            return all_reduce(d, cfg["ar"]), r

        if cfg["split"] == 1:
            h, res = tail(attn, res)
        else:
            hf = T // 2
            ha, ra = tail(attn[:hf], res[:hf])
            hb, rb = tail(attn[hf:], res[hf:])
            h, res = jnp.concatenate([ha, hb]), jnp.concatenate([ra, rb])
    hidden = rms(res + h, p["lnf"])
    # Vocab-parallel LM head (no gather) + 7-step greedy Markov loop.
    logits = lax.dot_general(
        hidden[: R * K],
        p["table"],
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    logits = logits.reshape(R, K, VSHARD)
    prev = ids[: R * K].reshape(R, K)[:, 0]
    out = []
    for s in range(K):
        me = p["mw1"][jnp.clip(prev, 0, VOCAB - 1)]  # [R, MRANK]
        scores = logits[:, s].astype(jnp.float32) + jnp.dot(
            me, p["mw2"].T, preferred_element_type=jnp.float32
        )
        pair = jnp.stack(
            [
                scores.max(-1),
                (jnp.argmax(scores, -1) + rank * VSHARD).astype(jnp.float32),
            ],
            axis=-1,
        )  # [R, 2]
        # [R, 16]: every rank's (max, id), rank-major.
        gp = ag_dim(pair, 1) if cfg["markov"] and not local else jnp.tile(pair, (1, N))
        gp = gp.reshape(R, N, 2)
        best = jnp.argmax(gp[..., 0], -1)
        prev = jnp.take_along_axis(gp[..., 1], best[:, None], -1)[:, 0].astype(
            jnp.int32
        )
        out.append(prev)
    return jnp.stack(out, 1), hidden


def make_params(mesh):
    s = 0.02
    return dict(
        table=per_device(mesh, (VSHARD, HID), jnp.bfloat16, 10, s),
        wqkv=per_device(mesh, (LAYERS, HID, (QH + 2) * HD), jnp.bfloat16, 11, s),
        wo=per_device(mesh, (LAYERS, QH * HD, HID), jnp.bfloat16, 12, s),
        wgu=per_device(mesh, (LAYERS, HID, 2 * FFN), jnp.bfloat16, 13, s),
        wd=per_device(mesh, (LAYERS, FFN, HID), jnp.bfloat16, 14, s),
        ln1=per_device(mesh, (LAYERS, HID), jnp.float32, 15, 0.1),
        ln2=per_device(mesh, (LAYERS, HID), jnp.float32, 16, 0.1),
        qn=per_device(mesh, (LAYERS, HD), jnp.float32, 17, 0.1),
        kn=per_device(mesh, (LAYERS, HD), jnp.float32, 18, 0.1),
        lnf=per_device(mesh, (HID,), jnp.float32, 19, 0.1),
        mw1=per_device(mesh, (VOCAB, MRANK), jnp.bfloat16, 20, s),
        mw2=per_device(mesh, (VSHARD, MRANK), jnp.bfloat16, 21, s),
    )


# ----------------------------------------------------------------------------
# HLO scan and timing
# ----------------------------------------------------------------------------
_COLL = {
    "all-reduce": re.compile(r"\ball-reduce(-start)?\("),
    "all-gather": re.compile(r"\ball-gather(-start)?\("),
    "reduce-scatter": re.compile(r"\breduce-scatter\("),
    "all-to-all": re.compile(r"\ball-to-all\("),
}


def hlo_scan(text):
    """Collective count per type, how many of them carry a SparseCore marker on
    their line, and the SparseCore mentions in the whole module."""
    res = {}
    for name, rx in _COLL.items():
        lines = [ln for ln in text.splitlines() if rx.search(ln)]
        if lines:
            res[name] = len(lines)
            sc = sum("sparsecore" in ln.lower() for ln in lines)
            if sc:
                res[name + "_sc"] = sc
    res["sparsecore_mentions"] = text.lower().count("sparsecore")
    return res


def compile_loop(mesh, body, args, opts, dump=None, tag=""):
    f = jax.jit(
        jax.shard_map(body, mesh=mesh, in_specs=P(AX), out_specs=P(AX), check_vma=False)
    )
    low = f.lower(*args)
    try:
        comp = (
            low.compile(compiler_options=opts)
            if (opts and not SMOKE)
            else low.compile()
        )
    except Exception as e:  # an option this libtpu does not know
        return None, {"compile_error": repr(e)[:300]}
    text = comp.as_text() or ""
    if dump:
        with open(os.path.join(dump, f"{tag}.hlo.txt"), "w") as fh:
            fh.write(text)
    return comp, hlo_scan(text)


def time_compiled(comp, args, reps):
    out = comp(*args)
    jax.block_until_ready(out)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = comp(*args)
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e6


def slope(mesh, make_body, args, L, opts, reps, dump, tag):
    """Per-iteration us from loops of L and 2L iterations."""
    c1, info = compile_loop(mesh, make_body(L), args, opts, dump, tag)
    if c1 is None:
        return None, info
    c2, _ = compile_loop(mesh, make_body(2 * L), args, opts)
    t1, t2 = time_compiled(c1, args, reps), time_compiled(c2, args, reps)
    return (t2 - t1) / L, info


def emit(res, rec):
    print("BENCH " + json.dumps(rec), flush=True)
    res.append(rec)


# ----------------------------------------------------------------------------
# class mode: each collective alone at its served shape
# ----------------------------------------------------------------------------
def class_bodies(T, R):
    """name -> (arg shapes, {variant: step fn}). Each step maps the previous
    output to the next input of the same shape with a cheap op that reads
    the whole collective output."""

    def ar(how):
        return lambda x: (all_reduce(x, how).astype(jnp.float32) * (1.0 / N)).astype(
            x.dtype
        )

    def agqkv(loc):
        return lambda x: (
            ag_dim(x, 1, loc).reshape(T, N, QH + 2, HD).astype(jnp.float32).sum(1)
            * (1.0 / N)
        ).astype(x.dtype)

    def agq(loc):
        return lambda x: jnp.concatenate(
            [
                (
                    ag_dim(x[:, :QH], 1, loc)
                    .reshape(T, N, QH, HD)
                    .astype(jnp.float32)
                    .sum(1)
                    * (1.0 / N)
                ).astype(x.dtype),
                x[:, QH:],
            ],
            axis=1,
        )

    def aglse(loc):
        return lambda x: ag_dim(x[None], 0, loc).max(0) - 1e-3

    def markov(loc):
        return lambda x: ag_dim(x, 1, loc).reshape(R, N, 2).max(1) - 1e-3

    def merge(name, loc):
        def f(c):
            out, lse = c
            m, ml = MERGES[name](out, lse, local=loc)
            nxt = jnp.tile(m, (1, N, 1))
            return nxt, jnp.tile(ml, (1, N)) - 1e-3

        return f

    bf, f32 = jnp.bfloat16, jnp.float32
    return {
        "ar": (
            [((T, HID), bf)],
            {
                "base": ar("psum"),
                "ar_tc": ar("psum"),
                "ar_flat": ar("psum"),
                "ar_rsag": ar("rsag"),
                "local": ar("none"),
            },
        ),
        "agqkv": (
            [((T, QH + 2, HD), bf)],
            {
                "base": agqkv(False),
                "ag_sc": agqkv(False),
                "agq": agq(False),
                "local": agqkv(True),
            },
        ),
        "aglse": (
            [((T, GQ), f32)],
            {"base": aglse(False), "ag_sc": aglse(False), "local": aglse(True)},
        ),
        "merge": (
            [((T, GQ, HD), bf), ((T, GQ), f32)],
            {
                "base": merge("agrs", False),
                "rs_tc": merge("agrs", False),
                "rs_bf16": merge("rs_bf16", False),
                "a2a2": merge("a2a2", False),
                "a2a1": merge("a2a1", False),
                "local": merge("agrs", True),
            },
        ),
        "markov": (
            [((R, 2), f32)],
            {"base": markov(False), "ag_sc": markov(False), "local": markov(True)},
        ),
    }


def run_class(res, mesh, cell_name, L, reps, dump, only):
    _, T = CELLS[cell_name]
    R = T // K
    for cname, (shapes, variants) in class_bodies(T, R).items():
        args = [per_device(mesh, s, d, 30 + j) for j, (s, d) in enumerate(shapes)]
        times = {}
        for v, fn in variants.items():
            if only and v not in only and v not in ("base", "local"):
                continue

            def make_body(n_iter, fn=fn):
                def it(i, c):
                    r = fn(c[0]) if len(c) == 1 else fn(c)
                    return r if isinstance(r, tuple) else (r,)

                def body(*xs):
                    c = lax.fori_loop(0, n_iter, it, tuple(x[0] for x in xs))
                    return tuple(x[None] for x in c)

                return body

            us, info = slope(
                mesh,
                make_body,
                args,
                L,
                OPTS.get(v),
                reps,
                dump,
                f"class_{cell_name}_{cname}_{v}",
            )
            times[v] = us
            emit(
                res,
                dict(
                    mode="class",
                    cell=cell_name,
                    T=T,
                    coll=cname,
                    variant=v,
                    us=None if us is None else round(us, 2),
                    **info,
                ),
            )
        if times.get("local") is not None:  # local ran last; fill the nets
            for v, us in times.items():
                if v != "local" and us is not None:
                    emit(
                        res,
                        dict(
                            mode="class_net",
                            cell=cell_name,
                            coll=cname,
                            variant=v,
                            net_us=round(us - times["local"], 2),
                        ),
                    )


# ----------------------------------------------------------------------------
# merge precision check
# ----------------------------------------------------------------------------
def check_merges(res, mesh, cell_name):
    _, T = CELLS[cell_name]
    out = per_device(mesh, (T, GQ, HD), jnp.bfloat16, 40)
    lse = per_device(mesh, (T, GQ), jnp.float32, 41, 3.0)
    # Some heads see none of a token's window on a rank (-inf, as in serving).
    lse = jnp.where(per_device(mesh, (T, GQ), jnp.float32, 42) > 1.5, -jnp.inf, lse)

    def body(o, s):
        o, s = o[0], s[0]
        outs = [MERGES[m](o, s) for m in ("agrs", "rs_bf16", "a2a2", "a2a1")]
        return tuple(x[None] for pair in outs for x in pair)

    f = jax.jit(
        jax.shard_map(body, mesh=mesh, in_specs=P(AX), out_specs=P(AX), check_vma=False)
    )
    r = [np.asarray(x.astype(jnp.float32)) for x in f(out, lse)]
    ref_o, ref_l = r[0], r[1]
    fin = np.isfinite(ref_l)
    for j, m in enumerate(("rs_bf16", "a2a2", "a2a1")):
        o, ml = r[2 + 2 * j], r[3 + 2 * j]
        d = np.abs(o - ref_o)
        ulp = np.maximum(np.abs(ref_o), 2.0**-100) * 2.0**-8
        emit(
            res,
            dict(
                mode="merge_check",
                cell=cell_name,
                variant=m,
                out_max_abs_diff=float(d.max()),
                out_frac_differ=float((d > 0).mean()),
                out_max_diff_in_bf16_ulps=float((d / ulp).max()),
                lse_max_abs_diff=float(np.abs(ml - ref_l)[fin].max())
                if fin.any()
                else 0.0,
                lse_inf_pattern_equal=bool((np.isfinite(ml) == fin).all()),
            ),
        )


# ----------------------------------------------------------------------------
# step mode: whole mock draft forward
# ----------------------------------------------------------------------------
def run_step(res, mesh, cell_name, variants, L, reps, dump, params):
    live, T = CELLS[cell_name]
    if SMOKE:
        live, T = min(live, 4), min(T, 32)
    rng = np.random.default_rng(0)
    num_pages = 256 if SMOKE else 8192
    c = make_cell(rng, live, T, num_pages)
    cell = [
        replicate_meta(mesh, c["seq_lens"]),
        replicate_meta(mesh, c["tables"]),
        replicate_meta(mesh, c["cu"]),
    ]
    words = -(-2 * G // 4)
    cache = per_device(mesh, (num_pages, PAGE, words, 4, HD), None, 50, fp8=True)
    ids = replicate_meta(mesh, rng.integers(0, VOCAB, T).astype(np.int32))
    keys = sorted(params)
    args = [ids, cache] + cell + [params[k] for k in keys]
    base_us = None
    for v in variants:
        cfg = variant_cfg(v)

        def make_body(n_iter, cfg=cfg):
            def body(ids, cache, sl, bt, cu, *ps):
                p = {k: x[0] for k, x in zip(keys, ps)}
                p["cache"] = cache[0]
                cl = (sl[0], bt[0], cu[0])

                def it(i, ids):
                    d, hid = draft_forward(ids, p, cl, cfg)
                    nxt = jnp.zeros_like(ids).at[: d.size].set(d.reshape(-1))
                    return nxt + (hid[:, 0] > 1e9).astype(jnp.int32)

                return lax.fori_loop(0, n_iter, it, ids[0])[None]

            return body

        us, info = slope(
            mesh, make_body, args, L, OPTS.get(v), reps, dump, f"step_{cell_name}_{v}"
        )
        if v == "base":
            base_us = us
        rec = dict(
            mode="step",
            cell=cell_name,
            T=T,
            variant=v,
            us_per_fwd=None if us is None else round(us, 1),
            **info,
        )
        if base_us is not None and us is not None and v != "base":
            rec["saving_vs_base_us"] = round(base_us - us, 1)
        emit(res, rec)


ALL_STEP = [
    "base",
    "nocoll",
    "ar_tc",
    "ar_flat",
    "ar_rsag",
    "rs_tc",
    "ag_sc",
    "agq",
    "rs_bf16",
    "a2a2",
    "a2a1",
    "split2",
    "split2nc",
    "best",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="class,step")
    ap.add_argument("--cells", default="c64,c16")
    ap.add_argument("--variants", default=",".join(ALL_STEP))
    ap.add_argument(
        "--L", type=int, default=8, help="class loop length (slope from L, 2L)"
    )
    ap.add_argument("--step-L", type=int, default=2)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--dump-dir", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.environ.get("CDK_OUTPUT_DIR", "/tmp"))
    a = ap.parse_args()
    global SMOKE, HID, FFN, VOCAB, VSHARD
    SMOKE = a.smoke
    if SMOKE:  # CPU: shrink the dense dims (attention dims stay as served)
        HID, FFN, VOCAB, VSHARD = 512, 128, 4096, 512
    devs = jax.devices()
    assert len(devs) >= N, f"needs {N} devices, got {len(devs)}"
    mesh = Mesh(np.array(devs[:N]), (AX,))
    res = []
    emit(
        res,
        dict(
            mode="env",
            devices=len(devs),
            platform=devs[0].platform,
            jax=jax.__version__,
            libtpu_init_args=os.environ.get("LIBTPU_INIT_ARGS", ""),
        ),
    )
    if a.dump_dir:
        os.makedirs(a.dump_dir, exist_ok=True)
    variants = [v for v in a.variants.split(",") if v]
    modes = a.mode.split(",")
    cells = a.cells.split(",")
    if SMOKE:
        cells, L, step_L, reps = ["c4"], 1, 1, 1
    else:
        L, step_L, reps = a.L, a.step_L, a.reps
    for cell in cells:
        check_merges(res, mesh, cell)
    if "class" in modes:
        for cell in cells:
            run_class(res, mesh, cell, L, reps, a.dump_dir or None, set(variants))
    if "step" in modes:
        params = make_params(mesh)
        for cell in cells:
            run_step(
                res, mesh, cell, variants, step_L, reps, a.dump_dir or None, params
            )
    path = os.path.join(a.out, "benchmark_dspark_draft_collectives.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print("BENCH wrote", path, flush=True)


if __name__ == "__main__":
    main()
