# SPDX-License-Identifier: Apache-2.0
"""QSA stock decode attention (RPAd) tile sweep by decode-rows bucket.

Served geometry: TP8 rank with 3 Q heads / 1 KV head (d 256), FP8 main KV
rows (1, 4, 256) holding two paired layers (slot read), 512-token pages,
top-512 groups of 4 tokens, MTP K=3 (4 verify rows per request), main-KV
DCP-2 (this rank attends its owned half and returns the LSE pair). One sample
is one decode step of the 12 QSA layers (alternating paired slots, per-layer
Q and selection) through ``qsa_rpa.qsa_attention``; reported per call =
step / 12. The cache threads through the donated loop (no pool copies).

Knobs (read at trace time): VLLM_TPU_QSA_DECODE_RPA_BKV (KV tokens per
block), VLLM_TPU_QSA_DECODE_RPA_LANES (decode rows per kernel batch),
VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES (rows per RPA call; more rows are split
into a scan of calls).
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
NL = 12
DCP = 2
POOL_PAGES = int(os.environ.get("ST_RPAD_POOL_PAGES", "8192"))
ROWS = [int(x) for x in os.environ.get("ST_RPAD_ROWS", "8,16,32,64,96,128").split(",")]
CTX_LO, CTX_HI = 60_000, 200_000

DEFAULT = (512, 4, 128)
GRID = [(b, l) for b in (256, 512, 1024, 2048) for l in (2, 4, 8, 16)] + [(128, 4), (128, 8)]


def clustered(rng, visible, recent=128, mean_run=4):
    picked = set(range(max(visible - recent, 0), visible))
    while len(picked) < min(BUDGET, visible):
        s = int(rng.integers(0, visible))
        picked.update(range(s, min(s + int(rng.geometric(1 / mean_run)), visible)))
    out = np.fromiter(picked, np.int32)[:BUDGET]
    return rng.permutation(out)


def build(rng, rows):
    reqs = rows // 4
    ctx = rng.integers(CTX_LO, CTX_HI, reqs)
    cols = -(-(CTX_HI + 8) // (DCP * PAGE))
    lengths = np.zeros(rows, np.int32)
    table = np.zeros((rows, cols), np.int32)
    blocks = np.full((NL, rows, BUDGET), -1, np.int32)
    for r in range(reqs):
        own = rng.integers(0, POOL_PAGES, cols)
        for t in range(4):
            i = 4 * r + t
            lengths[i] = ctx[r] + t + 1
            table[i] = own
        for layer in range(NL):
            base = clustered(rng, int(ctx[r] + 1) // RATIO)
            for t in range(4):
                i = 4 * r + t
                sel = base.copy()
                k = BUDGET // 10  # verify rows share ~90% of a request's pick
                vis = lengths[i] // RATIO
                sel[:k] = rng.choice(vis, k, replace=False)
                u = np.unique(sel)
                blocks[layer, i, : len(u)] = rng.permutation(u)
    q = jax.random.normal(jax.random.key(1), (NL, rows, HEADS, HD), jnp.bfloat16)
    g = jax.random.normal(jax.random.key(2), (NL, rows, HEADS, HD), jnp.bfloat16)
    starts = jnp.arange(0, rows + 1, 4, dtype=jnp.int32)
    return q, g, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table), starts


def make_cache():
    # Random finite FP8 bytes (|x| < 4): magnitude bits < 0x48, random sign.
    k = jax.random.key(0)
    mag = jax.random.randint(k, (POOL_PAGES, PAGE, 1, 4, HD), 0, 0x48, jnp.uint8)
    sign = (jax.random.bernoulli(jax.random.key(9), 0.5, mag.shape).astype(jnp.uint8) << 7)
    return jax.lax.bitcast_convert_type(mag | sign, jnp.float8_e4m3fn)


def make_step(q, g, blocks, lengths, table, starts):
    from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa

    rank = jnp.zeros((1,), jnp.int32)

    def step(cache, acc):
        # Data-dependent no-op on lengths: XLA cannot hoist the metadata.
        lens = lengths + (acc > 1e30).astype(jnp.int32)
        outs = []
        for layer in range(NL):
            out, lse, cache = qsa_rpa.qsa_attention(
                q[layer], g[layer], blocks[layer], lens, table, cache, starts,
                kv_heads=1, ratio=RATIO, k_scale=0.375, v_scale=2.5,
                **({"kv_head_offset": 1} if layer % 2 else {}),
                max_query_len=4, dcp=(rank, DCP),
            )
            outs.append((out, lse))
        return cache, outs

    def body(carry):
        cache, acc = carry
        cache, outs = step(cache, acc)
        s = sum(jnp.sum(o[0, 0]) + jnp.sum(jnp.where(jnp.isfinite(l[0]), l[0], 0)) for o, l in outs)
        return cache, acc + s

    return step, body


def main(emit):
    rng = np.random.default_rng(0)
    cache = make_cache()
    for rows in ROWS:
        inputs = build(rng, rows)
        bucket = {"phase": "decode", "rows": rows, "ctx": "60-200K", "dcp": DCP}
        configs = [DEFAULT] + [(b, l, 128) for b, l in GRID if (b, l) != DEFAULT[:2]]
        if rows > 32:
            configs += [(512, 4, 32), (512, 4, 64), (1024, 8, 64)]
        for bkv, lanes, mq in configs:
            cfg = {"VLLM_TPU_QSA_DECODE_RPA_BKV": bkv, "VLLM_TPU_QSA_DECODE_RPA_LANES": lanes,
                   "VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES": mq}

            def run():
                nonlocal cache
                with env(**cfg):
                    step, body = make_step(*inputs)
                    one = jax.jit(step, donate_argnums=(0,))
                    cache, outs = one(cache, jnp.float32(0))
                    dg = digest(outs)
                    us, (cache, _) = bench(body, (cache, jnp.float32(0)), iters=10)
                return us / NL, dg

            us, _ = try_config(emit, bucket, cfg, run, default=(bkv, lanes, mq) == DEFAULT)
            if us is None and cache.is_deleted():
                cache = make_cache()
        del inputs
