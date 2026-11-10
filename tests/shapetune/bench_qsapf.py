# SPDX-License-Identifier: Apache-2.0
"""QSA prefill attention tile sweep by (chunk tokens x context) bucket.

Served AgentX runs main-KV DCP-2, so prefill attention takes the DCP
block-sparse path (the union walk requires dcp=None): ``block_sparse_gqa_attention``
with kv_tile = one 512-token page and (query_block, query_tile) from
``prefill_tiles.SHARD_PREFILL_TILES`` (Q3: 512 / 256, halved for short chunks)
unless VLLM_TPU_QSA_DCP_PREFILL_QUERY_BLOCK / _QUERY_TILE override it (served: 0).

One sample = one prefill chunk of one request through ``qsa_rpa.qsa_attention``
for both paired slots (2 layer calls); reported per call. Selections: each
64-token query group shares a clustered top-512 base (recent window + runs),
each token re-draws 10% of it; causal (groups < (ctx + i + 1) // 4).
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

from common import bench, digest, env, try_config

PAGE = 512
RATIO = 4
BUDGET = 512
HEADS = 3
HD = 256
DCP = 2
POOL_PAGES = int(os.environ.get("ST_PF_POOL_PAGES", "8192"))
CHUNKS = [int(x) for x in os.environ.get("ST_PF_CHUNKS", "1024,2048,4096,8192").split(",")]
CTXS = [int(x) for x in os.environ.get("ST_PF_CTXS", "64000,128000,200000").split(",")]

# (query_block, query_tile); (0, 0) = served default (table, env 0).
GRID = [(0, 0), (256, 128), (256, 256), (512, 128), (512, 256), (512, 512),
        (1024, 128), (1024, 256), (1024, 512), (2048, 256)]


def clustered(rng, visible, recent=128, mean_run=4):
    picked = set(range(max(visible - recent, 0), visible))
    while len(picked) < min(BUDGET, visible):
        s = int(rng.integers(0, visible))
        picked.update(range(s, min(s + int(rng.geometric(1 / mean_run)), visible)))
    return np.fromiter(picked, np.int32)[:BUDGET]


def build(rng, chunk, ctx, dcp=DCP):
    rows = chunk
    lengths = (ctx + 1 + np.arange(rows)).astype(np.int32)
    cols = -(-(ctx + chunk + 8) // (dcp * PAGE))
    own = rng.integers(0, POOL_PAGES, cols).astype(np.int32)
    table = np.broadcast_to(own, (rows, cols)).copy()
    blocks = np.full((2, rows, BUDGET), -1, np.int32)
    for layer in range(2):
        for g0 in range(0, rows, 64):
            base = clustered(rng, int(lengths[g0]) // RATIO)
            for i in range(g0, min(g0 + 64, rows)):
                vis = int(lengths[i]) // RATIO
                sel = base.copy()
                k = len(sel) // 10
                sel[:k] = rng.choice(vis, k, replace=False)
                # the newest groups the token can see (local window)
                sel[k:k + 16] = np.arange(max(vis - 16, 0), vis)[: 16]
                u = np.unique(sel)
                blocks[layer, i, : len(u)] = rng.permutation(u)
    q = jax.random.normal(jax.random.key(1), (2, rows, HEADS, HD), jnp.bfloat16)
    g = jax.random.normal(jax.random.key(2), (2, rows, HEADS, HD), jnp.bfloat16)
    starts = jnp.asarray([0, rows], jnp.int32)
    return q, g, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table), starts


def make_cache():
    k = jax.random.key(0)
    mag = jax.random.randint(k, (POOL_PAGES, PAGE, 1, 4, HD), 0, 0x48, jnp.uint8)
    sign = jax.random.bernoulli(jax.random.key(9), 0.5, mag.shape).astype(jnp.uint8) << 7
    return jax.lax.bitcast_convert_type(mag | sign, jnp.float8_e4m3fn)


def make_step(q, g, blocks, lengths, table, starts, dcp=True):
    from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa

    rank = jnp.zeros((1,), jnp.int32)

    def step(cache, acc):
        lens = lengths + (acc > 1e30).astype(jnp.int32)
        outs = []
        for layer in range(2):
            r = qsa_rpa.qsa_attention(
                q[layer], g[layer], blocks[layer], lens, table, cache, starts,
                kv_heads=1, ratio=RATIO, k_scale=0.375, v_scale=2.5,
                # no-DCP: both calls on slot 0 (the union walk takes slot 0 only)
                **({"kv_head_offset": 1} if layer and dcp else {}),
                max_query_len=4, **({"dcp": (rank, DCP)} if dcp else {}),
            )
            if dcp:
                out, lse, cache = r
            else:
                out, cache = r
                lse = jnp.zeros(out.shape[:2], jnp.float32)
            outs.append((out, lse))
        return cache, outs

    def body(carry):
        cache, acc = carry
        cache, outs = step(cache, acc)
        s = sum(jnp.sum(o[-1, 0]) + jnp.sum(jnp.where(jnp.isfinite(l[-1]), l[-1], 0))
                for o, l in outs)
        return cache, acc + s

    return step, body


def main(emit):
    rng = np.random.default_rng(0)
    cache = make_cache()
    for ctx in CTXS:
        for chunk in CHUNKS:
            inputs = build(rng, chunk, ctx)
            bucket = {"phase": "prefill", "chunk": chunk, "ctx": ctx, "dcp": DCP}
            for qb, qt in GRID:
                if qb > 2 * chunk:
                    continue
                cfg = {"VLLM_TPU_QSA_DCP_PREFILL_QUERY_BLOCK": qb,
                       "VLLM_TPU_QSA_DCP_PREFILL_QUERY_TILE": qt}

                def run():
                    nonlocal cache
                    with env(**cfg):
                        step, body = make_step(*inputs)
                        one = jax.jit(step, donate_argnums=(0,))
                        cache, outs = one(cache, jnp.float32(0))
                        dg = digest(outs)
                        us, (cache, _) = bench(body, (cache, jnp.float32(0)), iters=4, reps=3)
                    return us / 2, dg

                us, _ = try_config(emit, bucket, cfg, run, default=(qb == 0))
                if us is None and cache.is_deleted():
                    cache = make_cache()
            del inputs


# ---- no-DCP (c1-c16): union walk for chunks >= 1024, block-sparse shard tiles below.
ND_CHUNKS = [int(x) for x in os.environ.get("ST_PFND_CHUNKS", "128,256,512,1024,2048,4096,8192").split(",")]
ND_CTXS = [int(x) for x in os.environ.get("ST_PFND_CTXS", "64000,160000").split(",")]
# chunks < 1024: (query_block, kv_tile, query_tile) for block_sparse_gqa_attention; None = served table
SMALL_GRID = [None, (128, 512, 128), (128, 1024, 128), (128, 2048, 128), (256, 512, 256),
              (256, 1024, 128), (256, 1024, 256), (256, 2048, 256), (512, 1024, 256), (512, 2048, 256)]
# chunks >= 1024: union walk env
UNION_GRID = [None] + [dict(VLLM_TPU_QSA_PREFILL_UNION_WALK=w, VLLM_TPU_QSA_PREFILL_UNION_KV_TILE=t)
                       for w in (64, 128) for t in (512, 1024, 2048) if (w, t) != (128, 1024)] + [
    dict(VLLM_TPU_QSA_PREFILL_UNION_CAP=8192), dict(VLLM_TPU_QSA_PREFILL_UNION_CAP=16384),
    dict(VLLM_TPU_QSA_PREFILL_UNION_WALK=0)]


def main_nodcp(emit):
    from vllm_torchtpu.kernels.qwen4_exp import prefill_tiles

    served_tiles = prefill_tiles._get_sparse_prefill_tiles
    rng = np.random.default_rng(0)
    cache = make_cache()
    for ctx in ND_CTXS:
        for chunk in ND_CHUNKS:
            inputs = build(rng, chunk, ctx, dcp=1)
            bucket = {"phase": "prefill", "chunk": chunk, "ctx": ctx, "dcp": 1}
            grid = SMALL_GRID if chunk < 1024 else UNION_GRID
            for item in grid:
                if chunk < 1024:
                    cfg = {"tiles": item} if item else {}
                    envs_ = {}
                    tiles = item
                else:
                    cfg = dict(item or {})
                    envs_ = cfg
                    tiles = None
                if tiles and tiles[0] > 2 * chunk:
                    continue

                def run():
                    nonlocal cache
                    prefill_tiles._get_sparse_prefill_tiles = (
                        (lambda *a, **k: tiles) if tiles else served_tiles)
                    try:
                        with env(**envs_):
                            step, body = make_step(*inputs, dcp=False)
                            one = jax.jit(step, donate_argnums=(0,))
                            cache, outs = one(cache, jnp.float32(0))
                            dg = digest(outs[0][0])
                            us, (cache, _) = bench(body, (cache, jnp.float32(0)), iters=4, reps=3)
                    finally:
                        prefill_tiles._get_sparse_prefill_tiles = served_tiles
                    return us / 2, dg

                us, _ = try_config(emit, bucket, cfg, run, default=item is None)
                if us is None and cache.is_deleted():
                    cache = make_cache()
            del inputs
