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
"""The MLA attention block of a GLM-5.2 long-prompt prefill step, on the
sequence-parallel residual rows: today's token-parallel split vs the same
split on each rank's own rows (`TokenParallelLayout.rows_aligned`).

Over a mesh of every device (both hosts of a slice with COORD/NPROC set), with
GLM-5.2 shapes (hidden 6144, q_lora 2048, kv_lora 512 + rope 64, 64 heads of
192 + 64 / v 256, indexer 32 x 128 top-2048, per-channel fp8 weights, fp8
KV, page 256, interleave 4, 64 DCP rows) and random weights, one request with
a `--prefix`-token context and a `--tokens` chunk; each rank starts from its
sequence-parallel rows of the block input, as the MoE split leaves them:

  today: fused_qkv_a + indexer wk on own rows, all-gather of the projections,
    norms / index q on every row, q for the DCP rows and the rank's
    token-parallel rows `split + j * R`, indexer + attention, an all-to-all of
    the outputs back to heads, W_UV + o_proj on 4 heads of every row and a
    reduce-scatter to own rows;
  rows: every per-row op on own rows, one all-gather of the cache-ready rows
    (fp8 latent, fp8 rope, index K), the DCP rows' q_c from rank 0, indexer +
    attention on own rows, and W_UV + o_proj of every head of them with the
    layer's o_proj / W_UV shards gathered a layer ahead (`rows_oproj`).

Checks rows against today on the same inputs (caches bitwise, top-k rows,
attention outputs, block output within tolerance), times one layer and a
chain of `--chain` layers (each with its own weights and caches, optionally
with the served MoE split between them), and with `--trace-dir` captures the
chains. `--aot` only compiles the chains for the slice's topology.

Usage: python3 scripts/benchmark_mla_prefill_rows.py --tokens 8192 16384
"""

import argparse
import dataclasses
import functools
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    KVLayout,
    coalesced_kv_append_dcp,
    get_kv_cache_shape,
    streamindex_topk_dcp,
)
from vllm_torchtpu.kernels.mla import dcp_token_parallel as tp
from vllm_torchtpu.kernels.mla import kv_cache_utils as kvu
from vllm_torchtpu.kernels.mla import page_gather
from vllm_torchtpu.kernels.mla import rows_oproj as ro
from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor,
    xla_quantized_matmul,
)
from vllm_torchtpu.layers.adapter import dcp_token_parallel_op as tp_op
from vllm_torchtpu.layers.core.attention_interface import sparse_mla_attention_dcp

AX = "dcp"
HID, QL, KVL, ROPE, NOPE, VH, HEADS = 6144, 2048, 512, 64, 192, 256, 64
IDXH, IDXD, TOPK = 32, 128, 2048
PAGE, IL, MAX_REQS, SPLIT = 256, 4, 64, 64
QKV, KW = QL + KVL + ROPE, IDXD + IDXH
K_SCALE, SM_SCALE = 0.5, (NOPE + ROPE) ** -0.5
IDX_SCALE = IDXD**-0.5 * IDXH**-0.5
MAX_PAGES = 44  # dcp_token_parallel_op.layout_for at max-model-len 163840
PREFIX_PIECE = 32768
FP8 = jnp.float8_e4m3fn
SAL = KVLayout.SEQ_ALONG_LANE
# The cache writer's row views: 128-word nope tokens, 32-word rope tokens.
READ_PAGES = (
    functools.partial(page_gather.gather_pages, row_width=128),
    functools.partial(page_gather.gather_pages, row_width=32),
)


def _host(x):
    if x.is_fully_addressable:
        return np.asarray(x)
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


def _timeit(fn, *args, iters):
    jax.block_until_ready(fn(*args))
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e3


def _smap(fn, mesh, in_specs, out_specs):
    return jax.shard_map(
        fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
    )


def _rank(dcp):
    return tp.rank_as_data(AX, dcp)


# ---------------------------------------------------------------------------
# Weights and per-row math (the same functions on both paths)
# ---------------------------------------------------------------------------


def make_layer_weights(mesh, seed):
    """Random weights of one MLA block in the served layouts: replicated
    projections, o_proj / W_UV sharded by heads (TP). Keys are made inside
    the jitted builders from the int `seed`, the same on every process."""
    rep, shd = NamedSharding(mesh, P()), NamedSharding(mesh, P(AX))
    dcp = mesh.devices.size

    def fp8(k, n_in, n_out):
        kw, ks = jax.random.split(k)
        w = (jax.random.normal(kw, (n_in, n_out), jnp.float32) * 48).astype(FP8)
        s = jax.random.uniform(ks, (n_out,), jnp.float32, 0.5, 1.5) / (48 * n_in**0.5)
        return w, s

    def replicated(key):
        ks = jax.random.split(key, 8)
        return {
            "qkv": fp8(ks[0], HID, QKV),
            "kw": fp8(ks[1], HID, KW),
            "qb": fp8(ks[2], QL, HEADS * (NOPE + ROPE)),
            "qbi": fp8(ks[3], QL, IDXH * IDXD),
            "qn": jax.random.uniform(ks[4], (QL,), jnp.float32, 0.5, 1.5),
            "kvn": jax.random.uniform(ks[5], (KVL,), jnp.float32, 0.5, 1.5),
            "kn": jax.random.uniform(ks[6], (2, IDXD), jnp.float32, 0.5, 1.5),
            "uk": (jax.random.normal(ks[7], (HEADS, NOPE, KVL)) * NOPE**-0.5).astype(
                jnp.bfloat16
            ),
        }

    def sharded(key):
        ks = jax.random.split(key, 3)
        uv = (jax.random.normal(ks[0], (HEADS, KVL, VH)) * KVL**-0.5).astype(
            jnp.bfloat16
        )
        wo, _ = fp8(ks[1], HEADS * VH, HID)
        # Per-rank per-channel scale, as each TP shard is requantized.
        so = jax.random.uniform(ks[2], (dcp, HID), jnp.float32, 0.5, 1.5) / (
            48 * (HEADS * VH) ** 0.5
        )
        return {"uv": uv, "o": wo, "so": so}

    w = jax.jit(
        lambda: replicated(jax.random.fold_in(jax.random.key(seed), 0)),
        out_shardings=rep,
    )()
    w.update(
        jax.jit(
            lambda: sharded(jax.random.fold_in(jax.random.key(seed), 1)),
            out_shardings=shd,
        )()
    )
    return w


def rmsnorm(x, w):
    xf = x.astype(jnp.float32)
    xf = xf * lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + 1e-6)
    return (xf * w).astype(jnp.bfloat16)


def layernorm(x, wb):
    xf = x.astype(jnp.float32)
    xf = xf - jnp.mean(xf, -1, keepdims=True)
    xf = xf * lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + 1e-6)
    return (xf * wb[0] + wb[1]).astype(jnp.bfloat16)


def rope(x, pos):
    """NeoX rope on the leading 64 channels of x `[n, ..., d]`."""
    inv = 1.0 / (10000.0 ** (jnp.arange(0, ROPE, 2, dtype=jnp.float32) / ROPE))
    ang = pos.astype(jnp.float32)[:, None] * inv[None, :]
    shape = (pos.shape[0],) + (1,) * (x.ndim - 2) + (ROPE // 2,)
    cos, sin = jnp.cos(ang).reshape(shape), jnp.sin(ang).reshape(shape)
    xf = x.astype(jnp.float32)
    x1, x2, rest = xf[..., : ROPE // 2], xf[..., ROPE // 2 : ROPE], xf[..., ROPE:]
    out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], -1)
    return out.astype(x.dtype)


def quant_kv(x):
    return (x.astype(jnp.float32) / K_SCALE).clip(-448, 448).astype(FP8)


def quant_index_k(k):
    """fp8 values + one e8m0 scale byte per row, `[n, IDXD + 1]` u8."""
    kf = k.astype(jnp.float32)
    amax = jnp.maximum(jnp.max(jnp.abs(kf), -1, keepdims=True), 1e-12)
    e = jnp.ceil(jnp.log2(amax / 448.0))
    kq = (kf * jnp.exp2(-e)).astype(FP8)
    scale = (e + 127).astype(jnp.uint8)
    return jnp.concatenate([lax.bitcast_convert_type(kq, jnp.uint8), scale], -1)


def block_rows(w, x, pos):
    """The per-row part of the block on rows x `[n, HID]` (positions pos):
    q_c, the cache-ready latent / rope / index K, and the raw index weights."""
    qkv = xla_quantized_matmul(x, *w["qkv"])
    kw = xla_quantized_matmul(x, *w["kw"])
    return per_row(w, qkv, kw, pos)


def per_row(w, qkv, kw, pos):
    q_c = rmsnorm(qkv[:, :QL], w["qn"])
    kv = quant_kv(rmsnorm(qkv[:, QL : QL + KVL], w["kvn"]))
    kpe = quant_kv(rope(qkv[:, QL + KVL :], pos))
    kp = quant_index_k(rope(layernorm(kw[:, :IDXD], w["kn"]), pos))
    return q_c, kv, kpe, kp, kw[:, IDXD:]


def mla_q(w, q_c, pos):
    """(ql_nope `[n, H, KVL]`, q_pe `[n, H, ROPE]`): q_b, rope, W_UK."""
    q = xla_quantized_matmul(q_c, *w["qb"]).reshape(-1, HEADS, NOPE + ROPE)
    q_pe = rope(q[..., NOPE:], pos)
    ql = jnp.einsum("nhd,hdl->nhl", q[..., :NOPE], w["uk"])
    return ql, q_pe


def index_q(w, q_c, pos, w_raw):
    """(fp8 index q `[n, IDXH, IDXD]`, weights `[n, IDXH]`)."""
    q = xla_quantized_matmul(q_c, *w["qbi"]).reshape(-1, IDXH, IDXD)
    qq, qs = quantize_tensor(rope(q, pos), FP8)
    wts = (w_raw.astype(jnp.float32) * qs[..., 0] * IDX_SCALE).astype(jnp.bfloat16)
    return qq, wts


# ---------------------------------------------------------------------------
# The step: metadata, caches, and the ops both paths share
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Step:
    """Static shape of the step; its metadata travels as `md` (`step_data`)."""

    mesh: Mesh
    tokens: int
    prefix: int
    lay: tp.TokenParallelLayout
    nope_spec: kvu.SparseMLAKVCacheSpec
    rope_spec: kvu.SparseMLAKVCacheSpec
    idx_shape: tuple
    vpages: int

    @property
    def dcp(self):
        return self.lay.dcp_size

    @property
    def rows(self):
        return self.tokens // self.dcp


def make_step(mesh, tokens, prefix):
    dcp = mesh.devices.size
    caps = tuple(c for c in tp_op.GATHER_CAPS if c < MAX_PAGES) + (MAX_PAGES,)
    lay = tp.TokenParallelLayout(dcp, IL, PAGE, SPLIT, MAX_PAGES, gather_caps=caps)
    n_vp = -(-(prefix + tokens) // lay.virtual_page_size)
    pages_per_rank = n_vp + 4
    sc = kvu.KVCacheLayout.SPARSECORE
    return Step(
        mesh=mesh,
        tokens=tokens,
        prefix=prefix,
        lay=lay,
        nope_spec=kvu.SparseMLAKVCacheSpec.create(
            kvu.KVCacheType.NOPE, sc, pages_per_rank, PAGE, KVL
        ),
        rope_spec=kvu.SparseMLAKVCacheSpec.create(
            kvu.KVCacheType.ROPE, sc, pages_per_rank, PAGE, ROPE
        ),
        idx_shape=get_kv_cache_shape(pages_per_rank, PAGE, IDXD, SAL),
        vpages=n_vp,
    )


def step_metadata(st):
    """Host metadata of the step (one request, the chunk at its end):
    `meta` (seq_lens, block_tables, qsl, dist), `sub_meta` (the same cut
    after the DCP rows) and the chunk's positions."""
    seq_lens = np.zeros(MAX_REQS, np.int32)
    seq_lens[0] = st.prefix + st.tokens
    qsl = np.full(MAX_REQS + 1, st.tokens, np.int32)
    qsl[0] = 0
    dist = np.array([0, 0, 1], np.int32)
    perm = np.random.default_rng(0).permutation(st.nope_spec.num_pages)[: st.vpages]
    bt = np.zeros((MAX_REQS, st.vpages + 1), np.int32)
    bt[0, : st.vpages] = perm
    qsl_s = np.minimum(qsl, SPLIT)
    seq_s = seq_lens - (np.diff(qsl) - np.diff(qsl_s))
    return {
        "meta": [seq_lens, bt.reshape(-1), qsl, dist],
        "sub_meta": [seq_s.astype(np.int32), bt.reshape(-1), qsl_s, dist],
        "pos": np.arange(st.prefix, st.prefix + st.tokens, dtype=np.int32),
    }


def step_data(st):
    rep = NamedSharding(st.mesh, P())

    def put(x):
        return jax.make_array_from_callback(x.shape, rep, lambda i: x[i])

    return jax.tree.map(put, step_metadata(st))


def step_avals(st):
    rep = NamedSharding(st.mesh, P())
    return jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=rep),
        step_metadata(st),
    )


def make_caches(st, md, seed):
    """DCP-sharded caches holding a random `prefix`-token context."""
    mesh, dcp = st.mesh, st.dcp
    shd = NamedSharding(mesh, P(AX))

    def zeros(aval):
        return jax.jit(lambda: jnp.zeros(aval.shape, aval.dtype), out_shardings=shd)()

    nope, rope_c, icache = (zeros(a) for a in cache_avals(st))

    def write(nope, rope_c, icache, kv, kpe, kp, sl, bt, q_):
        rank = _rank(dcp)
        nope, rope_c = kvu.update_sparse_mla_kv_cache_dcp_jax(
            nope,
            rope_c,
            kv,
            kpe,
            sl,
            bt,
            q_,
            rank,
            nope_spec=st.nope_spec,
            rope_spec=st.rope_spec,
            dcp_size=dcp,
            interleave_size=IL,
        )
        icache = coalesced_kv_append_dcp(
            icache,
            kp,
            sl,
            bt,
            q_,
            rank,
            dcp_size=dcp,
            interleave_size=IL,
            kv_layout=SAL,
        )
        return nope, rope_c, icache

    write = jax.jit(
        _smap(write, mesh, (P(AX),) * 3 + (P(),) * 6, (P(AX),) * 3),
        donate_argnums=(0, 1, 2),
    )
    rng = np.random.default_rng(seed)
    for start in range(0, st.prefix, PREFIX_PIECE):
        end = min(start + PREFIX_PIECE, st.prefix)
        n = end - start
        # Host values, the same on every process (inputs replicate).
        kv, kpe, k = (
            jnp.asarray(rng.standard_normal((n, d)), jnp.float32)
            for d in (KVL, ROPE, IDXD)
        )
        kv, kpe = (np.asarray((v * 2).astype(FP8)) for v in (kv, kpe))
        kp = np.asarray(quant_index_k(k))
        sl = np.zeros(MAX_REQS, np.int32)
        sl[0] = end
        cu = np.full(MAX_REQS + 1, n, np.int32)
        cu[0] = 0
        nope, rope_c, icache = write(
            nope, rope_c, icache, kv, kpe, kp, sl, md["meta"][1], cu
        )
    return nope, rope_c, icache


def cache_avals(st):
    shd = NamedSharding(st.mesh, P(AX))
    return tuple(
        jax.ShapeDtypeStruct((st.dcp * shape[0], *shape[1:]), dtype, sharding=shd)
        for shape, dtype in (
            (st.nope_spec.shape, jnp.uint32),
            (st.rope_spec.shape, jnp.uint32),
            (st.idx_shape, jnp.uint8),
        )
    )


def ops(st):
    """The kernels both paths call, built once per step shape; each takes
    the step's metadata `md` last."""
    mesh, dcp = st.mesh, st.dcp
    attend = tp_op.make_attend(
        sm_scale=SM_SCALE,
        k_scale=K_SCALE,
        cache_layout="sparsecore",
        packed_rope=False,
        lkv_dim=KVL,
    )
    select = tp_op.make_select_topk(
        TOPK,
        SAL,
        chunk_tokens=envs.TPU_STREAMIDX_CHUNK_TOKENS,
        small_bucket=(None, None, 0),
        skip_empty_decode=False,
        skip_empty_decode_chunked=False,
        mixed_bkvp=None,
        decode_pack=False,
        shared_meta=False,
    )
    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        _streamidx_block_sizes,
    )

    bkvp, bq = _streamidx_block_sizes(SPLIT, None, None, 0)

    def write_kv(nope, rope_c, kv, kpe, sl, bt, q_, rank):
        return kvu.update_sparse_mla_kv_cache_dcp(
            nope,
            rope_c,
            kv,
            kpe,
            sl,
            bt,
            q_,
            rank,
            nope_spec=st.nope_spec,
            rope_spec=st.rope_spec,
            dcp_size=dcp,
            interleave_size=IL,
        )

    def write_index_k(icache, kp, md):
        def local(c, kp, sl, bt, q_):
            return coalesced_kv_append_dcp(
                c,
                kp,
                sl,
                bt,
                q_,
                _rank(dcp),
                dcp_size=dcp,
                interleave_size=IL,
                kv_layout=SAL,
            )

        return _smap(local, mesh, (P(AX),) + (P(),) * 4, P(AX))(
            icache, kp, *md["meta"][:3]
        )

    def dcp_index(icache, qq, wts, md):
        return streamindex_topk_dcp(
            qq,
            wts,
            icache,
            *md["sub_meta"],
            mesh=mesh,
            k=TOPK,
            compression_ratio=1,
            dcp_size=dcp,
            interleave_size=IL,
            num_kv_pages_per_block=bkvp,
            num_queries_per_block=bq,
            kv_layout=SAL,
            chunk_tokens=envs.TPU_STREAMIDX_CHUNK_TOKENS,
            hist_merge=True,
        )

    def merge(partial, lse):
        lses = lax.all_gather(lse, AX, axis=0, tiled=False)
        m = jax.nn.logsumexp(lses, axis=0)
        wgt = jnp.exp(lse - m)
        weighted = (partial.astype(jnp.float32) * wgt[..., None]).astype(jnp.bfloat16)
        return lax.psum_scatter(weighted, AX, scatter_dimension=1, tiled=True)

    def dcp_attend(nope, rope_c, ql, qpe, kv, kpe, local_topk, md):
        """The DCP rows: today's sharded attention + log-sum-exp merge;
        `[dcp * S, H / dcp, KVL]`, each rank's heads."""
        nope, rope_c, partial, lse = sparse_mla_attention_dcp(
            ql,
            qpe,
            kv,
            kpe,
            nope,
            rope_c,
            local_topk,
            *md["sub_meta"],
            mesh,
            st.nope_spec,
            st.rope_spec,
            sm_scale=SM_SCALE,
            k_scale=K_SCALE,
            dcp_size=dcp,
            interleave_size=IL,
        )
        head = _smap(merge, mesh, (P(AX), P(AX)), P(AX))(partial, lse)
        return nope, rope_c, head

    def tp_index(lay, icache, qq, wts, md):
        return tp.indexer_topk(
            lay, mesh, AX, icache, qq, wts, *md["meta"], select_topk=select
        )

    def tp_attend(lay, nope, rope_c, ql, qpe, kv, kpe, topk, md):
        return tp.sparse_mla_attention(
            lay,
            mesh,
            AX,
            nope,
            rope_c,
            ql,
            qpe,
            kv,
            kpe,
            topk,
            *md["meta"],
            write_kv=write_kv,
            attend=attend,
            q_own_rows=True,
            read_pages=READ_PAGES,
        )

    return dict(
        write_index_k=write_index_k,
        dcp_index=dcp_index,
        dcp_attend=dcp_attend,
        tp_index=tp_index,
        tp_attend=tp_attend,
    )


# ---------------------------------------------------------------------------
# The two block layouts: own rows in, own rows out
# ---------------------------------------------------------------------------


REP_KEYS = ("qkv", "kw", "qb", "qbi", "qn", "kvn", "kn", "uk")


def _rep(w):
    return {k: w[k] for k in REP_KEYS}


def today_block(st, o, w, caches, x, md):
    """Today's split; returns (block output own rows `[T, HID]` sharded,
    caches, attention outputs for the checks)."""
    mesh, dcp, s, rows, t = st.mesh, st.dcp, SPLIT, st.rows, st.tokens
    lay = st.lay
    nope, rope_c, icache = caches

    def prep(x, pos, w):
        rank = _rank(dcp)
        proj = jnp.concatenate(
            [xla_quantized_matmul(x, *w["qkv"]), xla_quantized_matmul(x, *w["kw"])], -1
        )
        proj = lax.all_gather(proj, AX, tiled=True)  # every row
        q_c, kv, kpe, kp, w_raw = per_row(w, proj[:, :QKV], proj[:, QKV:], pos)
        qq, wts = index_q(w, q_c, pos, w_raw)
        # q of the DCP rows and of this rank's token-parallel rows.
        i = jnp.arange(s + rows, dtype=jnp.int32)
        own = jnp.minimum(i + jnp.where(i >= s, rank * rows, 0), t - 1)
        ql, qpe = mla_q(w, q_c[own], pos[own])
        return kv, kpe, kp, qq, wts, ql[:s], qpe[:s], ql[s:], qpe[s:]

    kv, kpe, kp, qq, wts, ql_d, qpe_d, ql_t, qpe_t = _smap(
        prep, mesh, (P(AX), P(), P()), (P(),) * 7 + (P(AX),) * 2
    )(x, md["pos"], _rep(w))
    icache = o["write_index_k"](icache, kp, md)
    topk_d = o["dcp_index"](icache, qq[:s], wts[:s], md)
    topk_t = o["tp_index"](lay, icache, qq, wts, md)
    nope, rope_c, head = o["dcp_attend"](
        nope, rope_c, ql_d, qpe_d, kv[:s], kpe[:s], topk_d, md
    )
    nope, rope_c, tail = o["tp_attend"](
        lay, nope, rope_c, ql_t, qpe_t, kv, kpe, topk_t, md
    )

    def out(head, tail, uv, wo, so):
        lat = jnp.concatenate([head, tail[: t - s]])  # [T, H / dcp, KVL]
        v = jnp.einsum("thl,hlv->thv", lat, uv).reshape(t, -1)
        partial = xla_quantized_matmul(v, wo, so[0])
        return lax.psum_scatter(partial, AX, scatter_dimension=0, tiled=True), lat

    y, lat = _smap(out, mesh, (P(AX),) * 5, (P(AX), P(AX)))(
        head, tail, w["uv"], w["o"], w["so"]
    )
    return y, (nope, rope_c, icache), (lat, topk_t)


GATHERS = tuple(ro.GATHERS)


def prefetch(st, w, after, gather="allgather"):
    """Gather a layer's o_proj / W_UV / o_proj scale shards, starting once
    `after` exists: one all-gather (offloaded to the SparseCores) or
    collective-permutes around a ring of the slice (TensorCore DMAs)."""
    ring = None
    if ro.needs_ring(gather):
        ring = ro.ring_order(st.mesh.devices.reshape(-1))

    def local(wo, uv, so, after):
        return ro.gather_shards(wo, uv, so[0], after, AX, gather, ring)

    return _smap(local, st.mesh, (P(AX),) * 4, (P(),) * 3)(
        w["o"], w["uv"], w["so"], after
    )


def rows_block(st, o, w, caches, x, md, gathered):
    """The split on own rows; `gathered` is `prefetch`'s shards."""
    mesh, dcp, s, rows = st.mesh, st.dcp, SPLIT, st.rows
    lay = dataclasses.replace(st.lay, rows_aligned=True)
    nope, rope_c, icache = caches

    def prep(x, pos, w):
        rank = _rank(dcp)
        pos_own = lax.dynamic_slice(pos, (rank * rows,), (rows,))
        q_c, kv, kpe, kp, w_raw = block_rows(w, x, pos_own)
        # Every rank writes its own cache positions of every row.
        payload = jnp.concatenate(
            [
                lax.bitcast_convert_type(kv, jnp.uint8),
                lax.bitcast_convert_type(kpe, jnp.uint8),
                kp,
            ],
            -1,
        )
        payload = lax.all_gather(payload, AX, tiled=True)
        kv_all = lax.bitcast_convert_type(payload[:, :KVL], FP8)
        kpe_all = lax.bitcast_convert_type(payload[:, KVL : KVL + ROPE], FP8)
        kp_all = payload[:, KVL + ROPE :]
        # The DCP rows are rank 0's: every rank needs their q_c.
        src = jnp.concatenate([q_c[:s], w_raw[:s]], -1)
        src = lax.psum(jnp.where(rank == 0, src, jnp.zeros_like(src)), AX)
        q_c_d, w_raw_d = src[:, :QL], src[:, QL:]
        ql_d, qpe_d = mla_q(w, q_c_d, pos[:s])
        qq_d, wts_d = index_q(w, q_c_d, pos[:s], w_raw_d)
        ql, qpe = mla_q(w, q_c, pos_own)
        qq, wts = index_q(w, q_c, pos_own, w_raw)
        return kv_all, kpe_all, kp_all, ql_d, qpe_d, qq_d, wts_d, ql, qpe, qq, wts

    kv, kpe, kp, ql_d, qpe_d, qq_d, wts_d, ql, qpe, qq, wts = _smap(
        prep, mesh, (P(AX), P(), P()), (P(),) * 7 + (P(AX),) * 4
    )(x, md["pos"], _rep(w))
    icache = o["write_index_k"](icache, kp, md)
    topk_d = o["dcp_index"](icache, qq_d, wts_d, md)
    topk = o["tp_index"](lay, icache, qq, wts, md)
    nope, rope_c, head = o["dcp_attend"](
        nope, rope_c, ql_d, qpe_d, kv[:s], kpe[:s], topk_d, md
    )
    nope, rope_c, lat = o["tp_attend"](lay, nope, rope_c, ql, qpe, kv, kpe, topk, md)

    def out(head, lat, w_o, w_uv, w_scale):
        rank = _rank(dcp)
        head = lax.all_gather(head, AX, axis=1, tiled=True)  # [S, H, KVL]
        first = jnp.where(rank == 0, head, lat[:s])
        lat = lax.dynamic_update_slice(lat, first, (0, 0, 0))
        return ro.rows_oproj(lat, w_uv, w_o, w_scale), lat

    y, lat = _smap(out, mesh, (P(AX), P(AX)) + (P(),) * 3, (P(AX), P(AX)))(
        head, lat, *gathered
    )
    return y, (nope, rope_c, icache), (lat, topk)


# ---------------------------------------------------------------------------
# Chains of layers
# ---------------------------------------------------------------------------


def make_moe(mesh, kind, avals=False):
    """What runs between two attention blocks: nothing, a TensorCore-only
    stand-in, or the served MoE split (scripts/benchmark_moe_prefill_sp.py).
    With `avals`, the weights as abstract values (`--aot`)."""
    if kind == "none":
        return None, None
    build = jax.eval_shape if avals else (lambda f: f())
    if kind == "standin":
        w = build(
            jax.jit(
                lambda: (
                    jax.random.normal(jax.random.key(7), (HID, HID)) * HID**-0.5
                ).astype(jnp.bfloat16),
                out_shardings=NamedSharding(mesh, P()),
            )
        )

        def standin(x, w):
            def body(_, y):
                return jnp.dot(y, w, preferred_element_type=jnp.float32).astype(y.dtype)

            return lax.fori_loop(0, 20, body, x)

        return _smap(standin, mesh, (P(AX), P()), P(AX)), w
    import benchmark_moe_prefill_sp as bm

    bm.AX = AX  # its shard_maps name the mesh axis at call time
    w = build(lambda: bm.make_weights(mesh, mesh.devices.size, jax.random.key(8)))
    moe = bm.split_fn(mesh, mesh.devices.size, wire="bf16", sharded_plan=True)

    def run(x, w):
        logits = _smap(
            lambda x, g: jnp.dot(x, g.T, preferred_element_type=jnp.float32),
            mesh,
            (P(AX), P()),
            P(AX),
        )(x, w["gate"])
        return moe(x, logits, w)

    return run, w


def _residual(x, y):
    yf = x.astype(jnp.float32) + y.astype(jnp.float32)
    yf = yf * lax.rsqrt(jnp.mean(yf * yf, -1, keepdims=True) + 1e-6)
    return yf.astype(x.dtype)


def chain_fn(st, o, path, moe_fn, gather="allgather"):
    """A chain of blocks, each on its own weights and caches, with `moe_fn`
    (or nothing) after each; the rows path gathers layer l + 1's o_proj while
    layer l runs, and takes layer 0's gathered (`g0`, as a step would have it
    from the layer before). `(ws, caches, x, md, moe_w, g0) -> (x, caches)`."""

    def run(ws, caches, x, md, moe_w, g0):
        out_caches = []
        nxt = g0
        for i, w in enumerate(ws):
            if path == "rows":
                g = nxt
                if i + 1 < len(ws):
                    nxt = prefetch(st, ws[i + 1], x, gather)
                y, c, _ = rows_block(st, o, w, caches[i], x, md, g)
            else:
                y, c, _ = today_block(st, o, w, caches[i], x, md)
            out_caches.append(c)
            x = _residual(x, y)
            if moe_fn is not None:
                x = _residual(x, moe_fn(x, moe_w))
        return x, out_caches

    return jax.jit(run, donate_argnums=(1,))


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def compare(st, o, w, caches, x, md):
    """One block both ways on the same inputs."""
    dcp, s, rows, t = st.dcp, SPLIT, st.rows, st.tokens
    copy = jax.jit(lambda c: jax.tree.map(jnp.copy, c))
    y_t, c_t, (lat_t, topk_t) = jax.jit(
        lambda w, c, x, md: today_block(st, o, w, c, x, md)
    )(w, copy(caches), x, md)
    res = {}
    # Every rank's shards, as each gather leaves them on this process's first
    # and last device, against the weights themselves.
    want = (
        _host(w["o"]).reshape(dcp, -1, HID).view(np.uint8),
        _host(w["uv"]),
        _host(w["so"]),
    )
    ys = {}
    for gather in GATHERS:
        g = jax.jit(lambda w, x, gather=gather: prefetch(st, w, x, gather))(w, x)
        for name, got, ref in zip(("w_o", "w_uv", "w_scale"), g, want):
            shards = got.addressable_shards
            res[f"gather_{gather}_{name}_differ"] = int(
                sum(
                    np.sum(
                        np.asarray(sh.data).view(ref.dtype).reshape(ref.shape) != ref
                    )
                    for sh in (shards[0], shards[-1])
                )
            )
        y_r, c_r, (lat_r, topk_r) = jax.jit(
            lambda w, c, x, md, g: rows_block(st, o, w, c, x, md, g)
        )(w, copy(caches), x, md, g)
        ys[gather] = _host(y_r).astype(np.float32)
    for name, a, b in zip(("nope", "rope", "index_k"), c_t, c_r):
        res[f"cache_{name}_bitwise"] = bool(np.array_equal(_host(a), _host(b)))
    # Attention latent outputs in step row order; today's are [rank, T, 4, L].
    lat_t = _host(lat_t).reshape(dcp, t, HEADS // dcp, KVL).transpose(1, 0, 2, 3)
    lat_t = lat_t.reshape(t, HEADS, KVL).astype(np.float32)
    lat_r = _host(lat_r).reshape(t, HEADS, KVL).astype(np.float32)
    diff = np.abs(lat_t - lat_r).reshape(t, -1).max(-1)
    res["attn_rows_bitwise"] = int(np.sum(diff == 0))
    res["attn_rows"] = t
    res["attn_dcp_rows_max_abs"] = float(diff[:s].max())
    res["attn_tp_rows_max_abs"] = float(diff[s:].max())
    res["attn_finite"] = bool(np.isfinite(lat_r).all())
    # Top-k sets of the token-parallel rows (today: rank-major rows S + m).
    kt = _host(topk_t).reshape(dcp * rows, TOPK)[: t - s]
    kr = _host(topk_r).reshape(t, TOPK)[s:]
    res["topk_rows_differ"] = int(
        sum(set(a[a >= 0].tolist()) != set(b[b >= 0].tolist()) for a, b in zip(kt, kr))
    )
    yt = _host(y_t).astype(np.float32)
    for gather, yr in ys.items():
        res[f"out_{gather}_max_rel"] = float(np.abs(yt - yr).max() / np.abs(yt).max())
        res[f"out_{gather}_rel_l2"] = float(
            np.linalg.norm(yt - yr) / np.linalg.norm(yt)
        )
        res[f"out_{gather}_finite"] = bool(np.isfinite(yr).all())
    return res


# ---------------------------------------------------------------------------


def weight_avals(mesh):
    """`make_layer_weights`' pytree as sharded avals (deviceless compile)."""
    rep, shd = NamedSharding(mesh, P()), NamedSharding(mesh, P(AX))

    def a(shape, dtype, s=rep):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=s)

    def lin(n_in, n_out):
        return (a((n_in, n_out), FP8), a((n_out,), jnp.float32))

    return {
        "qkv": lin(HID, QKV),
        "kw": lin(HID, KW),
        "qb": lin(QL, HEADS * (NOPE + ROPE)),
        "qbi": lin(QL, IDXH * IDXD),
        "qn": a((QL,), jnp.float32),
        "kvn": a((KVL,), jnp.float32),
        "kn": a((2, IDXD), jnp.float32),
        "uk": a((HEADS, NOPE, KVL), jnp.bfloat16),
        "uv": a((HEADS, KVL, VH), jnp.bfloat16, shd),
        "o": a((HEADS * VH, HID), FP8, shd),
        "so": a((mesh.devices.size, HID), jnp.float32, shd),
    }


def aot(a):
    """Compile the chains for a 2x2x2 slice without a device; write each
    optimized HLO to `--out` (a directory)."""
    from jax.experimental import topologies

    devices = topologies.get_topology_desc("tpu7x:2x2x2", platform="tpu").devices
    mesh = Mesh(np.array(devices), (AX,))
    os.makedirs(a.out, exist_ok=True)
    for tokens in a.tokens:
        st = make_step(mesh, tokens, a.prefix)
        o = ops(st)
        shd = NamedSharding(mesh, P(AX))
        x = jax.ShapeDtypeStruct((tokens, HID), jnp.bfloat16, sharding=shd)
        ws, cs = [weight_avals(mesh)] * a.chain, [cache_avals(st)] * a.chain
        for kind, path, gather in (
            (kind, path, gather)
            for kind in a.moe
            for path, gather in (("today", None), *(("rows", g) for g in a.gather))
        ):
            moe_fn, moe_w = make_moe(mesh, kind, avals=True)
            t0 = time.time()
            with jax.default_device(devices[0]):
                g0 = None
                if path == "rows":
                    rep = NamedSharding(mesh, P())
                    g0 = (
                        jax.ShapeDtypeStruct(
                            (st.dcp, HEADS * VH // st.dcp, HID), FP8, sharding=rep
                        ),
                        jax.ShapeDtypeStruct(
                            (HEADS, KVL, VH), jnp.bfloat16, sharding=rep
                        ),
                        jax.ShapeDtypeStruct((st.dcp, HID), jnp.float32, sharding=rep),
                    )
                c = (
                    chain_fn(st, o, path, moe_fn, gather)
                    .lower(ws, cs, x, step_avals(st), moe_w, g0)
                    .compile()
                )
            name = (
                f"T{tokens}_{kind}_{path}{'_' + gather if gather else ''}"
                f"_chain{a.chain}"
            )
            with open(os.path.join(a.out, name + ".hlo"), "w") as f:
                f.write(c.as_text())
            ma = c.memory_analysis()
            print(
                f"{name}: compiled in {time.time() - t0:.1f}s, temp "
                f"{ma.temp_size_in_bytes / 2**20:.0f} MiB",
                flush=True,
            )


def run_tokens(a, mesh, tokens):
    """Check and time one bucket; returns its results."""
    dcp = mesh.devices.size
    shd = NamedSharding(mesh, P(AX))
    copy = jax.jit(lambda c: jax.tree.map(jnp.copy, c))
    st = make_step(mesh, tokens, a.prefix)
    md = step_data(st)
    o = ops(st)
    print(f"T={tokens} prefix={a.prefix} dcp={dcp} rows/rank={st.rows}", flush=True)
    ws = [make_layer_weights(mesh, 10 + i) for i in range(a.chain)]
    caches = [make_caches(st, md, 20 + i) for i in range(a.chain)]
    x = jax.jit(
        lambda: jax.random.normal(jax.random.key(3), (tokens, HID)).astype(
            jnp.bfloat16
        ),
        out_shardings=shd,
    )()
    res = {"tokens": tokens, "prefix": a.prefix, "dcp": dcp, "chain": a.chain}
    if not a.no_check:
        res.update(compare(st, o, ws[0], caches[0], x, md))
        print("check:", res, flush=True)
    if not a.no_single:
        # One block alone; the rows path pays its weight gather in the call.
        one_t = jax.jit(lambda w, c, x, md: today_block(st, o, w, c, x, md)[0])
        one_r = jax.jit(
            lambda w, c, x, md: rows_block(st, o, w, c, x, md, prefetch(st, w, x))[0]
        )
        one_rg = jax.jit(lambda w, c, x, md, g: rows_block(st, o, w, c, x, md, g)[0])
        g0 = jax.jit(lambda w, x: prefetch(st, w, x))(ws[0], x)
        for name, fn, args in (
            ("today", one_t, ()),
            ("rows", one_r, ()),
            ("rows_pregathered", one_rg, (g0,)),
        ):
            res[f"one_{name}_ms"] = _timeit(
                fn, ws[0], caches[0], x, md, *args, iters=a.iters
            )
            print(f"  one block {name}: {res[f'one_{name}_ms']:.3f} ms", flush=True)
        del g0
    for kind in a.moe:
        moe_fn, moe_w = make_moe(mesh, kind)
        for path, gather in (("today", None), *(("rows", g) for g in a.gather)):
            fn = chain_fn(st, o, path, moe_fn, gather)
            g0 = None
            if path == "rows":
                g0 = jax.jit(lambda w, x, g=gather: prefetch(st, w, x, g))(ws[0], x)
            state = {"c": copy(caches)}

            def call(fn=fn, moe_w=moe_w, state=state, g0=g0):
                y, state["c"] = fn(ws, state["c"], x, md, moe_w, g0)
                return y

            ms = _timeit(call, iters=a.iters)
            name = f"chain_{kind}_{path}" + (f"_{gather}" if gather else "")
            res[f"{name}_ms_per_layer"] = ms / a.chain
            print(f"  {name}: {ms:.3f} ms ({ms / a.chain:.3f} per layer)", flush=True)
            if a.trace_dir:
                d = os.path.join(a.trace_dir, f"T{tokens}_{name}")
                with jax.profiler.trace(d):
                    for _ in range(3):
                        jax.block_until_ready(call())
            del state
    print("RESULT", json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[8192])
    ap.add_argument("--prefix", type=int, default=40960)
    ap.add_argument("--chain", type=int, default=4)
    ap.add_argument(
        "--moe", choices=("none", "standin", "split"), nargs="+", default=["none"]
    )
    ap.add_argument("--gather", choices=GATHERS, nargs="+", default=list(GATHERS))
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--no-single", action="store_true")
    ap.add_argument("--trace-dir", default="")
    ap.add_argument("--aot", action="store_true", help="compile only, no device")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.aot:
        aot(a)
        return
    if os.environ.get("NPROC"):
        jax.distributed.initialize(
            coordinator_address=os.environ["COORD"],
            num_processes=int(os.environ["NPROC"]),
            process_id=int(os.environ["TPU_WORKER_ID"]),
        )
    mesh = Mesh(np.array(jax.devices()), (AX,))
    print("LIBTPU_INIT_ARGS:", os.environ.get("LIBTPU_INIT_ARGS", ""), flush=True)
    results = [run_tokens(a, mesh, tokens) for tokens in a.tokens]
    if a.out and jax.process_index() == 0:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
