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
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
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


def _fp8(key, shape, scale):
    x = jax.random.normal(key, shape, jnp.float32) * scale
    return x.astype(jnp.float8_e4m3fn)


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
    a = ap.parse_args()

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

    # Step metadata: request 0 is the prefill chunk, the rest are padding.
    seq_lens = np.zeros(MAX_REQS, np.int32)
    seq_lens[0] = total
    qsl = np.full(MAX_REQS + 1, T, np.int32)
    qsl[0] = 0
    dist = np.array([0, 0, 1], np.int32)
    perm = np.random.default_rng(0).permutation(pages_per_rank)[:n_vp]
    bt = np.zeros((MAX_REQS, n_vp + 1), np.int32)
    bt[0, :n_vp] = perm
    meta = [jnp.asarray(x) for x in (seq_lens, bt.reshape(-1), qsl, dist)]
    pre = [
        jnp.asarray(x)
        for x in (
            np.array([prefix] + [0] * (MAX_REQS - 1), np.int32),
            np.array([0] + [prefix] * MAX_REQS, np.int32),
        )
    ]

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
        return jax.device_put(jnp.zeros((dcp * shape[0], *shape[1:]), dtype), shd)

    nope0, rope0, icache_pre = jax.jit(
        jax.shard_map(
            _write_prefix,
            mesh=mesh,
            in_specs=(P(AXIS),) * 3 + (P(),) * 6,
            out_specs=(P(AXIS),) * 3,
            check_vma=False,
        )
    )(
        zeros(nope_spec.shape, jnp.uint32),
        zeros(rope_spec.shape, jnp.uint32),
        zeros(idx_shape, jnp.uint8),
        lat[:prefix],
        pe[:prefix],
        idx_rec[:prefix],
        pre[0],
        meta[1],
        pre[1],
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
    nat_idx = jax.jit(
        lambda c: coalesced_kv_append(c, idx_rec, one[0], nat_bt, one[1], SAL)
    )(jnp.zeros(get_kv_cache_shape(nat_pages, PAGE, IDX_DIM, SAL), jnp.uint8))

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
    t_tp_idx, flat = _timeit(tp_index, icache, q_idx, w_idx, *meta, iters=a.iters)
    flat = np.asarray(flat).reshape(dcp, rows, TOPK).reshape(dcp * rows, TOPK)
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
    q_sharded = jax.device_put(
        q.reshape(T, dcp, h, -1).transpose(1, 0, 2, 3).reshape(dcp * T, h, -1), shd
    )
    flat_dev = jax.device_put(jnp.asarray(flat), shd)
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
        lat[prefix:],
        pe[prefix:],
        flat_dev,
        *meta,
        iters=a.iters,
    )
    out_tp = np.asarray(out_tp, np.float32).reshape(dcp, T, h, LKV)
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
    t_single, out_single = _timeit(single, q[S:], jnp.asarray(ref_pos), iters=a.iters)
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
        dcp_index, icache, q_idx, w_idx, *meta, iters=a.iters
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
        lat[prefix:],
        pe[prefix:],
        local_topk,
        *meta,
        iters=a.iters,
    )
    out_dcp = np.asarray(out_dcp, np.float32).reshape(dcp, T, h, LKV)
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
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
