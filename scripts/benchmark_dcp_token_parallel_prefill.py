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
"""One sparse MLA layer of a chunked-prefill step, token-parallel vs DCP.

Builds one request with a `--prefix`-token context and a `--tokens`-token
chunk over a DCP mesh of every local device (GLM-5.2 geometry: 64 heads,
512 + 64 latent, 32 x 128 index heads, top-2048, fp8 caches, page 256,
interleave 4), then

  * checks the token-parallel indexer against the single-device StreamIndex
    top-k on the unsharded cache (per-row sets);
  * checks the token-parallel attention against the single-device sparse MLA
    kernel on the unsharded cache with the same top-k (expected bitwise), and
    both it and the DCP path against an f32 reference on sampled rows;
  * times both paths (median of `--iters` after warmup).

Needs a TPU with SparseCores. Usage:
  python3 scripts/benchmark_dcp_token_parallel_prefill.py --prefix 40960
"""

import argparse
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

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    KVLayout,
    coalesced_kv_append,
    coalesced_kv_append_dcp,
    get_kv_cache_shape,
    streamindex_topk,
    streamindex_topk_dcp,
)
from vllm_torchtpu.kernels.mla import dcp_token_parallel as tp
from vllm_torchtpu.kernels.mla import kv_cache_utils as kvu
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel
from vllm_torchtpu.layers.adapter import dcp_token_parallel_op as tp_op
from vllm_torchtpu.layers.core.attention_interface import sparse_mla_attention_dcp

AXIS = "dcp"
HEADS, LKV, ROPE, TOPK = 64, 512, 64, 2048
IDX_HEADS, IDX_DIM = 32, 128
PAGE, IL, MAX_REQS = 256, 4, 64
PREFIX_PIECE = 32768
SAL = KVLayout.SEQ_ALONG_LANE


def _timeit(fn, *args, iters):
    out = fn(*args)
    jax.block_until_ready(out)
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e3, out


def _global(sharding, x):
    """A global array of host value `x` (the same on every process)."""
    x = np.asarray(x)
    return jax.make_array_from_callback(x.shape, sharding, lambda i: x[i])


def _host(x):
    """A (possibly multi-host) global array, whole, on this host."""
    if x.is_fully_addressable:
        return np.asarray(x)
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


def _fp8(key, shape, scale):
    x = jax.random.normal(key, shape, jnp.float32) * scale
    return x.astype(jnp.float8_e4m3fn)


def _collective_breakdown(mesh, lay, T, h, iters):
    """Each collective of the split alone, on buffers of its served shape:
    median time, bytes each device sends, effective GB/s, and how XLA lowered
    it (the collective ops in the compiled HLO)."""
    dcp = lay.dcp_size
    rows = T // dcp
    out = {}

    def run(name, fn, x, sent_bytes):
        f = jax.jit(
            jax.shard_map(
                fn, mesh=mesh, in_specs=P(AXIS), out_specs=P(AXIS), check_vma=False
            )
        )
        t, _ = _timeit(f, x, iters=iters)
        hlo = f.lower(x).compile().as_text()
        ops = [
            ln.strip()[:200]
            for ln in hlo.splitlines()
            if any(k in ln for k in ("all-to-all", "all-gather", "collective-permute"))
            and "=" in ln
        ][:4]
        out[f"coll_{name}_ms"] = t
        out[f"coll_{name}_GBps"] = sent_bytes / t / 1e6
        out[f"coll_{name}_hlo_ops"] = ops
        print(
            f"  {name}: {t:.3f} ms, {sent_bytes / 1e6:.1f} MB sent, "
            f"{sent_bytes / t / 1e6:.1f} GB/s, ops {ops}"
        )

    def zeros(shape, dtype):
        return jax.jit(
            lambda: jnp.zeros((dcp * shape[0], *shape[1:]), dtype),
            out_shardings=NamedSharding(mesh, P(AXIS)),
        )()

    frac = (dcp - 1) / dcp
    q = zeros((T, h, LKV + ROPE), jnp.bfloat16)
    qb = T * h * (LKV + ROPE) * 2 * frac
    run("noop", lambda x: x + 1, zeros((8, 128), jnp.float32), 1)
    run("q_a2a_heads", lambda x: lax.all_to_all(x, AXIS, 0, 1, tiled=True), q, qb)
    run(
        "q_a2a_flat",
        lambda x: lax.all_to_all(x.reshape(dcp, -1), AXIS, 0, 0, tiled=True),
        q,
        qb,
    )
    run(
        "q_a2a_flat_u32",
        lambda x: lax.all_to_all(
            lax.bitcast_convert_type(x.reshape(dcp, -1, 2), jnp.uint32),
            AXIS,
            0,
            0,
            tiled=True,
        ),
        q,
        qb,
    )
    o = zeros((rows, HEADS, LKV), jnp.bfloat16)
    ob = rows * HEADS * LKV * 2 * frac
    run("out_a2a_heads", lambda y: lax.all_to_all(y, AXIS, 1, 0, tiled=True), o, ob)
    run(
        "out_a2a_flat",
        lambda y: lax.all_to_all(
            y.reshape(rows, dcp, -1).swapaxes(0, 1).reshape(dcp, -1),
            AXIS,
            0,
            0,
            tiled=True,
        ),
        o,
        ob,
    )
    for cap in sorted({8, 24, lay.max_pages}):
        page = zeros((cap, PAGE, 128), jnp.uint32)
        run(
            f"latent_ag_{cap}p",
            lambda x: lax.all_gather(x, AXIS, axis=0, tiled=True),
            page,
            cap * PAGE * 128 * 4 * (dcp - 1),
        )
    run(
        "q_all_gather_dcp",
        lambda x: lax.all_gather(x, AXIS, axis=1, tiled=True),
        q,
        qb * dcp,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=int, default=40960)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--split-rows", type=int, default=64)
    ap.add_argument("--max-pages", type=int, default=44)
    ap.add_argument("--static-gather", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--check-rows", type=int, default=256)
    ap.add_argument("--out", default="")
    ap.add_argument("--trace-dir", default="", help="capture a profile of each op")
    a = ap.parse_args()

    if os.environ.get("NPROC"):
        # Multi-host (one process per host): every process builds the same
        # host data; mesh inputs are global arrays, results fetched back.
        jax.distributed.initialize(
            coordinator_address=os.environ["COORD"],
            num_processes=int(os.environ["NPROC"]),
            process_id=int(os.environ["TPU_WORKER_ID"]),
        )
    devices = jax.devices()
    dcp = len(devices)
    mesh = Mesh(np.array(devices), (AXIS,))
    shd = NamedSharding(mesh, P(AXIS))
    T, S, prefix = a.tokens, a.split_rows, a.prefix
    total = prefix + T
    caps = tuple(c for c in tp_op.GATHER_CAPS if c < a.max_pages) + (a.max_pages,)
    if a.static_gather:
        caps = (a.max_pages,)
    lay = tp.TokenParallelLayout(dcp, IL, PAGE, S, a.max_pages, gather_caps=caps)
    rows = lay.rows_per_rank(T)
    vps = lay.virtual_page_size
    n_vp = -(-total // vps)
    pages_per_rank = n_vp + 4
    k_scale, sm_scale = 0.5, 2.0 / np.sqrt(LKV + ROPE)
    print(f"dcp={dcp} T={T} prefix={prefix} rows/rank={rows} vpages={n_vp}")

    keys = jax.random.split(jax.random.key(0), 8)
    lat = _fp8(keys[0], (total, LKV), 1.0)
    pe = _fp8(keys[1], (total, ROPE), 1.0)
    idx_vals = _fp8(keys[2], (total, IDX_DIM), 1.0)
    idx_rec = jnp.concatenate(
        [
            lax.bitcast_convert_type(idx_vals, jnp.uint8),
            jnp.full((total, 1), 127, jnp.uint8),  # e8m0 scale 1.0
        ],
        -1,
    )
    q_idx = _fp8(keys[3], (T, IDX_HEADS, IDX_DIM), 0.5)
    w_idx = jax.random.normal(keys[4], (T, IDX_HEADS), jnp.float32).astype(jnp.bfloat16)
    q = jax.random.normal(keys[5], (T, HEADS, LKV + ROPE), jnp.float32).astype(
        jnp.bfloat16
    )

    lat, pe, idx_rec, q_idx, w_idx, q = (
        np.asarray(x) for x in (lat, pe, idx_rec, q_idx, w_idx, q)
    )

    # Step metadata: request 0 is the prefill chunk, the rest are padding.
    seq_lens = np.zeros(MAX_REQS, np.int32)
    seq_lens[0] = total
    qsl = np.full(MAX_REQS + 1, T, np.int32)
    qsl[0] = 0
    dist = np.array([0, 0, 1], np.int32)
    perm = np.random.default_rng(0).permutation(pages_per_rank)[:n_vp]
    bt = np.zeros((MAX_REQS, n_vp + 1), np.int32)
    bt[0, :n_vp] = perm
    meta = [seq_lens, bt.reshape(-1), qsl, dist]
    # DCP shards hold the prefix; the ops under test write the chunk.
    nope_spec = kvu.SparseMLAKVCacheSpec.create(
        kvu.KVCacheType.NOPE, kvu.KVCacheLayout.SPARSECORE, pages_per_rank, PAGE, LKV
    )
    rope_spec = kvu.SparseMLAKVCacheSpec.create(
        kvu.KVCacheType.ROPE, kvu.KVCacheLayout.SPARSECORE, pages_per_rank, PAGE, ROPE
    )
    idx_shape = get_kv_cache_shape(pages_per_rank, PAGE, IDX_DIM, SAL)

    def _write_prefix(nope, rope, icache, lat_, pe_, rec_, sl, bt_, q_):
        rank = tp.rank_as_data(AXIS, dcp)
        nope, rope = kvu.update_sparse_mla_kv_cache_dcp_jax(
            nope,
            rope,
            lat_,
            pe_,
            sl,
            bt_,
            q_,
            rank,
            nope_spec=nope_spec,
            rope_spec=rope_spec,
            dcp_size=dcp,
            interleave_size=IL,
        )
        icache = coalesced_kv_append_dcp(
            icache,
            rec_,
            sl,
            bt_,
            q_,
            rank,
            dcp_size=dcp,
            interleave_size=IL,
            kv_layout=SAL,
        )
        return nope, rope, icache

    def zeros(shape, dtype):
        return jax.jit(
            lambda: jnp.zeros((dcp * shape[0], *shape[1:]), dtype), out_shardings=shd
        )()

    write_prefix = jax.jit(
        jax.shard_map(
            _write_prefix,
            mesh=mesh,
            in_specs=(P(AXIS),) * 3 + (P(),) * 6,
            out_specs=(P(AXIS),) * 3,
            check_vma=False,
        )
    )
    nope0 = zeros(nope_spec.shape, jnp.uint32)
    rope0 = zeros(rope_spec.shape, jnp.uint32)
    icache_pre = zeros(idx_shape, jnp.uint8)
    # The K writer takes at most PREFIX_PIECE rows per call.
    for start in range(0, prefix, PREFIX_PIECE):
        end = min(start + PREFIX_PIECE, prefix)
        sl = np.zeros(MAX_REQS, np.int32)
        sl[0] = end
        cu = np.full(MAX_REQS + 1, end - start, np.int32)
        cu[0] = 0
        nope0, rope0, icache_pre = write_prefix(
            nope0,
            rope0,
            icache_pre,
            lat[start:end],
            pe[start:end],
            idx_rec[start:end],
            sl,
            meta[1],
            cu,
        )
    # The indexer ops expect this step's K already written (the adapter does).
    icache = jax.jit(
        jax.shard_map(
            lambda c, r, sl, bt_, q_: coalesced_kv_append_dcp(
                c,
                r,
                sl,
                bt_,
                q_,
                tp.rank_as_data(AXIS, dcp),
                dcp_size=dcp,
                interleave_size=IL,
                kv_layout=SAL,
            ),
            mesh=mesh,
            in_specs=(P(AXIS),) + (P(),) * 4,
            out_specs=P(AXIS),
            check_vma=False,
        )
    )(icache_pre, idx_rec[prefix:], meta[0], meta[1], meta[2])

    # Unsharded natural caches for the references (one device).
    nat_pages = -(-total // PAGE) + 1
    nat_nope_spec = kvu.SparseMLAKVCacheSpec.create(
        kvu.KVCacheType.NOPE, kvu.KVCacheLayout.SPARSECORE, nat_pages, PAGE, LKV
    )
    nat_rope_spec = kvu.SparseMLAKVCacheSpec.create(
        kvu.KVCacheType.ROPE, kvu.KVCacheLayout.SPARSECORE, nat_pages, PAGE, ROPE
    )
    nat_bt = jnp.arange(nat_pages, dtype=jnp.int32)
    one = (jnp.array([total], jnp.int32), jnp.array([0, total], jnp.int32))
    nat_nope, nat_rope = jax.jit(
        lambda n, r: kvu.update_sparse_mla_kv_cache_jax(
            n,
            r,
            lat,
            pe,
            one[0],
            nat_bt,
            one[1],
            nope_spec=nat_nope_spec,
            rope_spec=nat_rope_spec,
        )
    )(
        jnp.zeros(nat_nope_spec.shape, jnp.uint32),
        jnp.zeros(nat_rope_spec.shape, jnp.uint32),
    )
    nat_idx = jnp.zeros(get_kv_cache_shape(nat_pages, PAGE, IDX_DIM, SAL), jnp.uint8)
    append = jax.jit(
        lambda c, r, sl, cu: coalesced_kv_append(c, r, sl, nat_bt, cu, SAL)
    )
    for start in range(0, total, PREFIX_PIECE):
        end = min(start + PREFIX_PIECE, total)
        nat_idx = append(
            nat_idx,
            jnp.asarray(idx_rec[start:end]),
            jnp.array([end], jnp.int32),
            jnp.array([0, end - start], jnp.int32),
        )

    # Timed inputs live on device: replicated globals for the mesh, local
    # arrays for the single-device references.
    rep = NamedSharding(mesh, P())
    meta = [_global(rep, x) for x in meta]
    q_idx_g, w_idx_g = _global(rep, q_idx), _global(rep, w_idx)
    lat_c, pe_c = _global(rep, lat[prefix:]), _global(rep, pe[prefix:])

    results = {"dcp": dcp, "tokens": T, "prefix": prefix, "rows_per_rank": rows}
    from vllm_torchtpu import envs

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

    # ---- indexer: token-parallel vs single-device reference -------------
    tp_index = jax.jit(
        lambda c, qi, wi, *m: tp.indexer_topk(
            lay, mesh, AXIS, c, qi, wi, *m, select_topk=select
        )
    )
    t_tp_idx, flat = _timeit(tp_index, icache, q_idx_g, w_idx_g, *meta, iters=a.iters)
    flat = _host(flat).reshape(dcp, rows, TOPK).reshape(dcp * rows, TOPK)
    n_tp = T - S
    ref_pos = np.asarray(
        jax.jit(
            lambda qi, wi: streamindex_topk(
                qi,
                wi,
                nat_idx,
                one[0],
                nat_bt,
                jnp.array([0, n_tp], jnp.int32),
                jnp.array([0, 0, 1], jnp.int32),
                k=TOPK,
                compression_ratio=1,
                num_kv_pages_per_block=2,
                num_queries_per_block=128,
                kv_layout=SAL,
            )
        )(q_idx[S:], w_idx[S:])
    )
    ref_flat = np.asarray(
        tp.flat_rows(lay, jnp.asarray(ref_pos), jnp.zeros((n_tp,), jnp.int32))
    )
    mismatch = sum(
        set(flat[i][flat[i] >= 0].tolist())
        != set(ref_flat[i][ref_flat[i] >= 0].tolist())
        for i in range(n_tp)
    )
    results["indexer_rows_mismatch"] = int(mismatch)
    results["indexer_rows"] = n_tp
    print(f"indexer: {mismatch}/{n_tp} rows differ from the single-device top-k")

    # ---- attention: token-parallel vs the same kernel on natural cache ---
    attend = tp_op.make_attend(
        sm_scale=float(sm_scale),
        k_scale=k_scale,
        cache_layout="sparsecore",
        packed_rope=False,
        lkv_dim=LKV,
    )

    def write_kv(nope, rope, kv_c, k_pe, sl, bt_, q_, rank):
        return kvu.update_sparse_mla_kv_cache_dcp(
            nope,
            rope,
            kv_c,
            k_pe,
            sl,
            bt_,
            q_,
            rank,
            nope_spec=nope_spec,
            rope_spec=rope_spec,
            dcp_size=dcp,
            interleave_size=IL,
        )

    h = HEADS // dcp
    q_sharded = _global(
        shd, q.reshape(T, dcp, h, -1).transpose(1, 0, 2, 3).reshape(dcp * T, h, -1)
    )
    flat_dev = _global(shd, flat)
    tp_attn = jax.jit(
        lambda n, r, qq, kc, kp, ft, *m: tp.sparse_mla_attention(
            lay,
            mesh,
            AXIS,
            n,
            r,
            qq,
            kc,
            kp,
            ft,
            *m,
            write_kv=write_kv,
            attend=attend,
        )
    )
    t_tp_attn, (_, _, out_tp) = _timeit(
        tp_attn,
        nope0,
        rope0,
        q_sharded,
        lat_c,
        pe_c,
        flat_dev,
        *meta,
        iters=a.iters,
    )
    out_tp = _host(out_tp).astype(np.float32).reshape(dcp, T, h, LKV)
    out_tp = out_tp.transpose(1, 0, 2, 3).reshape(T, HEADS, LKV)[:n_tp]

    single = jax.jit(
        lambda qq, tk: sparse_mla_kernel.sparse_ragged_paged_attention(
            qq,
            nat_nope,
            nat_rope,
            tk,
            nat_bt,
            jnp.array([0, n_tp], jnp.int32),
            jnp.array([0, 0, 1], jnp.int32),
            sm_scale=float(sm_scale),
            k_scale=k_scale,
            cache_layout="sparsecore",
        )[..., :LKV]
    )
    t_single, out_single = _timeit(
        single, jnp.asarray(q[S:]), jnp.asarray(ref_pos), iters=a.iters
    )
    out_single = np.asarray(out_single, np.float32)
    same_topk = np.array(
        [
            set(flat[i][flat[i] >= 0].tolist())
            == set(ref_flat[i][ref_flat[i] >= 0].tolist())
            for i in range(n_tp)
        ]
    )
    assert same_topk.any(), "no row selected the reference top-k"
    diff = np.abs(out_tp[same_topk] - out_single[same_topk])
    results["tp_vs_single_max_abs"] = float(diff.max())
    results["tp_vs_single_bitwise_rows"] = int(
        np.sum(np.all(diff.reshape(diff.shape[0], -1) == 0, axis=1))
    )
    print(
        f"attention: TP vs single-device kernel max|d|={diff.max():.3g}, "
        f"bitwise rows {results['tp_vs_single_bitwise_rows']}/{int(same_topk.sum())}"
    )

    # ---- today's DCP path on the same step ------------------------------
    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        _streamidx_block_sizes,
    )

    bkvp, bq = _streamidx_block_sizes(T, None, None, 0)
    dcp_index = jax.jit(
        lambda c, qi, wi, *m: streamindex_topk_dcp(
            qi,
            wi,
            c,
            *m,
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
    )
    t_dcp_idx, local_topk = _timeit(
        dcp_index, icache, q_idx_g, w_idx_g, *meta, iters=a.iters
    )

    def gather_q(qs):
        return lax.all_gather(qs, AXIS, axis=1, tiled=True)

    def merge(partial, lse):
        lses = lax.all_gather(lse, AXIS, axis=0, tiled=False)
        m = jax.nn.logsumexp(lses, axis=0)
        w = jnp.exp(lse - m)
        weighted = (partial.astype(jnp.float32) * w[..., None]).astype(jnp.bfloat16)
        return lax.psum_scatter(weighted, AXIS, scatter_dimension=1, tiled=True)

    def dcp_attn_fn(n, r, qq, kc, kp, lt, *m):
        q_full = jax.shard_map(
            gather_q, mesh=mesh, in_specs=P(AXIS), out_specs=P(), check_vma=False
        )(qq)
        n, r, partial, lse = sparse_mla_attention_dcp(
            q_full[..., :LKV],
            q_full[..., LKV:],
            kc,
            kp,
            n,
            r,
            lt,
            *m,
            mesh,
            nope_spec,
            rope_spec,
            sm_scale=float(sm_scale),
            k_scale=k_scale,
            dcp_size=dcp,
            interleave_size=IL,
        )
        out = jax.shard_map(
            merge,
            mesh=mesh,
            in_specs=(P(AXIS), P(AXIS)),
            out_specs=P(AXIS),
            check_vma=False,
        )(partial, lse)
        return n, r, out

    t_dcp_attn, (_, _, out_dcp) = _timeit(
        jax.jit(dcp_attn_fn),
        nope0,
        rope0,
        q_sharded,
        lat_c,
        pe_c,
        local_topk,
        *meta,
        iters=a.iters,
    )
    out_dcp = _host(out_dcp).astype(np.float32).reshape(dcp, T, h, LKV)
    out_dcp = out_dcp.transpose(1, 0, 2, 3).reshape(T, HEADS, LKV)[S:]

    # ---- f32 reference on sampled rows ----------------------------------
    rng = np.random.default_rng(1)
    sample = np.sort(rng.choice(n_tp, min(a.check_rows, n_tp), replace=False))
    lat32 = np.asarray(lat.astype(jnp.float32)) * k_scale
    pe32 = np.asarray(pe.astype(jnp.float32)) * k_scale
    q32 = np.asarray(q.astype(jnp.float32))[S:]
    errs = {"tp": [], "dcp": []}
    for i in sample:
        sel = ref_pos[i][ref_pos[i] >= 0]
        kk = np.concatenate([lat32[sel], pe32[sel]], -1)
        s = q32[i] @ kk.T * sm_scale
        p = np.exp(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        want = p @ lat32[sel]
        scale = np.abs(want).max() + 1e-6
        errs["tp"].append(np.abs(out_tp[i] - want).max() / scale)
        errs["dcp"].append(np.abs(out_dcp[i] - want).max() / scale)
    for k_, v in errs.items():
        results[f"{k_}_vs_f32_max_rel"] = float(np.max(v))
        results[f"{k_}_vs_f32_mean_rel"] = float(np.mean(v))
    print(
        "vs f32 reference (max/mean of per-row max|d|/max|ref|): "
        f"TP {np.max(errs['tp']):.3g}/{np.mean(errs['tp']):.3g}, "
        f"DCP {np.max(errs['dcp']):.3g}/{np.mean(errs['dcp']):.3g}"
    )

    # ---- pieces of the token-parallel path ------------------------------
    def gather_only(n, r, *m):
        def _l(n, r, sl, bt_, q_, d_):
            pages, offs = tp.step_pages(lay, sl, q_, d_[2])
            ids = tp.gather_page_ids(lay, bt_, pages, offs, n.shape[0])
            return tp.gather_latent(lay, n, r, ids, jnp.sum(pages), AXIS)

        return jax.shard_map(
            _l,
            mesh=mesh,
            in_specs=(P(AXIS), P(AXIS)) + (P(),) * 4,
            out_specs=(P(AXIS), P(AXIS)),
            check_vma=False,
        )(n, r, *m)

    def a2a_only(qq):
        def _l(x):
            y = tp.heads_to_tokens(lay, x, AXIS)
            return tp.tokens_to_heads(y[..., :LKV], AXIS)

        return jax.shard_map(
            _l, mesh=mesh, in_specs=P(AXIS), out_specs=P(AXIS), check_vma=False
        )(qq)

    def index_pieces(c, qi, wi, *m, select_too):
        def _l(c, qi, wi, sl, bt_, q_, d_):
            pages, offs = tp.step_pages(lay, sl, q_, d_[2])
            ids = tp.gather_page_ids(lay, bt_, pages, offs, c.shape[0])
            nat = tp.gather_index_k(lay, c, ids, jnp.sum(pages), AXIS)
            if not select_too:
                return nat
            rank = tp.rank_as_data(AXIS, dcp)
            rr = tp.rank_rows(lay, rank, T, sl, q_, d_[2])
            table = tp.natural_page_table(lay, rr, offs, bt_.shape[0] // sl.shape[0])
            return select(qi[:rows], wi[:rows], nat, rr, table)

        return jax.shard_map(
            _l,
            mesh=mesh,
            in_specs=(P(AXIS),) + (P(),) * 6,
            out_specs=P(AXIS),
            check_vma=False,
        )(c, qi, wi, *m)

    t_igather, _ = _timeit(
        jax.jit(lambda *x: index_pieces(*x, select_too=False)),
        icache,
        q_idx_g,
        w_idx_g,
        *meta,
        iters=a.iters,
    )
    t_iboth, _ = _timeit(
        jax.jit(lambda *x: index_pieces(*x, select_too=True)),
        icache,
        q_idx_g,
        w_idx_g,
        *meta,
        iters=a.iters,
    )
    results.update(tp_index_gather_ms=t_igather, tp_index_gather_select_ms=t_iboth)
    t_gather, _ = _timeit(jax.jit(gather_only), nope0, rope0, *meta, iters=a.iters)
    t_a2a, _ = _timeit(jax.jit(a2a_only), q_sharded, iters=a.iters)
    t_qag, _ = _timeit(
        jax.jit(
            jax.shard_map(
                gather_q,
                mesh=mesh,
                in_specs=P(AXIS),
                out_specs=P(AXIS),
                check_vma=False,
            )
        ),
        q_sharded,
        iters=a.iters,
    )
    if a.trace_dir:
        # Three calls of each path, after warmup, in one capture; ops are
        # told apart by their module names.
        with jax.profiler.trace(a.trace_dir):
            for fn, args in (
                (tp_attn, (nope0, rope0, q_sharded, lat_c, pe_c, flat_dev, *meta)),
                (tp_index, (icache, q_idx_g, w_idx_g, *meta)),
                (
                    jax.jit(dcp_attn_fn),
                    (nope0, rope0, q_sharded, lat_c, pe_c, local_topk, *meta),
                ),
                (dcp_index, (icache, q_idx_g, w_idx_g, *meta)),
            ):
                for _ in range(3):
                    jax.block_until_ready(fn(*args))
    results.update(_collective_breakdown(mesh, lay, T, h, a.iters))
    # The kernel alone, on this rank's gathered buffers and rows.
    g_nope, g_rope = jax.jit(gather_only)(nope0, rope0, *meta)
    q_tok = jax.jit(
        jax.shard_map(
            lambda x: tp.heads_to_tokens(lay, x, AXIS),
            mesh=mesh,
            in_specs=P(AXIS),
            out_specs=P(AXIS),
            check_vma=False,
        )
    )(q_sharded)
    kernel_only = jax.jit(
        jax.shard_map(
            lambda qq, gn, gr, ft: attend(qq, gn, gr, ft, jnp.int32(rows)),
            mesh=mesh,
            in_specs=(P(AXIS),) * 4,
            out_specs=P(AXIS),
            check_vma=False,
        )
    )
    t_kernel, _ = _timeit(kernel_only, q_tok, g_nope, g_rope, flat_dev, iters=a.iters)
    results.update(tp_kernel_only_ms=t_kernel)
    results.update(
        tp_attention_ms=t_tp_attn,
        tp_indexer_ms=t_tp_idx,
        dcp_attention_ms=t_dcp_attn,
        dcp_indexer_ms=t_dcp_idx,
        single_device_kernel_all_rows_ms=t_single,
        tp_latent_gather_ms=t_gather,
        tp_q_and_out_all_to_all_ms=t_a2a,
        dcp_q_all_gather_ms=t_qag,
    )
    for k_, v in results.items():
        print(f"  {k_}: {v}")
    if a.out and jax.process_index() == 0:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
