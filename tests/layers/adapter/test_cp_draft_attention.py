# SPDX-License-Identifier: Apache-2.0
"""DCP-sharded sliding-window draft attention vs the replicated reference.

The reference is `ref_ragged_paged_attention` (causal, sliding window) over
one full KV cache, as the replicated draft runs today. The sharded path
writes the context into `dcp` per-rank caches (`write_kv`), attends each
rank's share (`attend_cached`), merges the ranks by LSE, and merges the
query block's own attention (`attend_block`).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    ref_ragged_paged_attention,
)
from vllm_torchtpu.layers.adapter import cp_draft_attention as cda

PAGE = 128
HEAD_DIM = 128
WINDOW = 512  # several pages, so the window is binding and page-unaligned
MAX_SEQS = 4


def _cache(pages, kv_heads, dtype):
    packing = 4 // jnp.dtype(dtype).itemsize
    words = -(-2 * kv_heads // packing)
    return jnp.zeros((pages, PAGE, words, packing, HEAD_DIM), dtype)


def _dense_reference(q, keys, values, total, cu_q, window, sm_scale, causal):
    """The RPA kernel's mask (`ragged_paged_attention` v3): keys below the
    sequence length within the window, plus q >= k if causal."""
    out = np.zeros(q.shape, np.float32)
    q_heads, kv_heads = q.shape[1], keys[0].shape[1]
    for r, (k, v) in enumerate(zip(keys, values)):
        lo, hi = int(cu_q[r]), int(cu_q[r + 1])
        q_pos = int(total[r]) - (hi - lo) + np.arange(hi - lo)
        k_pos = np.arange(k.shape[0])
        mask = q_pos[:, None] - k_pos[None, :] < window
        if causal:
            mask &= q_pos[:, None] >= k_pos[None, :]
        kk = np.repeat(k, q_heads // kv_heads, axis=1)
        vv = np.repeat(v, q_heads // kv_heads, axis=1)
        s = np.einsum("qhd,khd->hqk", q[lo:hi].astype(np.float32), kk) * sm_scale
        s = np.where(mask[None], s, -np.inf)
        p = np.exp(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        out[lo:hi] = np.einsum("hqk,khd->qhd", p, vv)
    return out


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
@pytest.mark.parametrize("dcp,kv_heads,q_heads", [(1, 1, 8), (2, 1, 16), (8, 4, 64)])
def test_sharded_matches_replicated(dcp, kv_heads, q_heads, dtype, causal):
    rng = np.random.default_rng(dcp * 10 + kv_heads)
    ctx = np.array([1000, 37, 600, 0], np.int32)[:MAX_SEQS]  # 0: empty context
    q_len = np.array([7, 7, 3, 7], np.int32)
    num_seqs = MAX_SEQS
    total = ctx + q_len
    max_q = 7
    pages_per_seq = -(-int(total.max()) // PAGE) + 1
    scale = None if dtype == jnp.bfloat16 else 0.5

    def rand(*shape):
        return jnp.asarray(rng.standard_normal(shape), jnp.float32)

    ctx_k = [rand(int(c), kv_heads, HEAD_DIM) for c in ctx]
    ctx_v = [rand(int(c), kv_heads, HEAD_DIM) for c in ctx]
    new_k = rand(int(q_len.sum()), kv_heads, HEAD_DIM)
    new_v = rand(int(q_len.sum()), kv_heads, HEAD_DIM)
    q = rand(int(q_len.sum()), q_heads, HEAD_DIM).astype(jnp.bfloat16)
    cu_q = jnp.asarray(np.concatenate([[0], np.cumsum(q_len)]), jnp.int32)
    dist = jnp.asarray([0, 0, num_seqs], jnp.int32)
    sm_scale = HEAD_DIM**-0.5

    # Replicated reference: context written at dcp=1, then RPA over the block.
    tables_1 = jnp.arange(num_seqs * pages_per_seq, dtype=jnp.int32).reshape(num_seqs, -1)
    cache_1 = _cache(num_seqs * pages_per_seq + 1, kv_heads, dtype)
    cu_ctx = jnp.asarray(np.concatenate([[0], np.cumsum(ctx)]), jnp.int32)
    all_ctx_k = jnp.concatenate(ctx_k)
    all_ctx_v = jnp.concatenate(ctx_v)
    cache_1 = cda.write_kv(
        cache_1, all_ctx_k, all_ctx_v, jnp.asarray(ctx), tables_1.reshape(-1), cu_ctx,
        dcp=1, rank=jnp.int32(0), k_scale=scale, v_scale=scale,
    )
    # The reference casts probabilities to the cache dtype; run it on the
    # dequantized values in bf16 (exact for fp8 * 0.5) so it stays precise.
    def stored(x):
        x = x if scale is None else (x / scale).astype(dtype).astype(jnp.float32) * scale
        return x.astype(dtype if scale is None else jnp.bfloat16)

    ref_cache = cache_1 if scale is None else (
        cache_1.astype(jnp.float32) * scale
    ).astype(jnp.bfloat16).reshape(cache_1.shape[0], PAGE, -1, 2, HEAD_DIM)[:, :, :, :, :]
    if scale is not None:  # repack fp8 rows (4 per word) as bf16 rows (2 per word)
        ref_cache = ref_cache.reshape(cache_1.shape[0], PAGE, -1, HEAD_DIM)
        words = -(-2 * kv_heads // 2)
        ref_cache = ref_cache[:, :, : 2 * words].reshape(cache_1.shape[0], PAGE, words, 2, HEAD_DIM)
    ref_out, _ = ref_ragged_paged_attention(
        q, stored(new_k), stored(new_v), ref_cache, jnp.asarray(total),
        tables_1.reshape(-1), cu_q, dist, sm_scale=sm_scale, sliding_window=WINDOW,
    )

    # Sharded: logical block j of dcp * PAGE tokens -> page tables[seq, j] on every rank.
    logical = -(-pages_per_seq // dcp) + 1
    tables = jnp.arange(num_seqs * logical, dtype=jnp.int32).reshape(num_seqs, -1)
    outs, lses = [], []
    for r in range(dcp):
        cache_r = cda.write_kv(
            _cache(num_seqs * logical + 1, kv_heads, dtype), all_ctx_k, all_ctx_v,
            jnp.asarray(ctx), tables.reshape(-1), cu_ctx,
            dcp=dcp, rank=jnp.int32(r), k_scale=scale, v_scale=scale,
        )
        o, l = cda.attend_cached(
            q, cache_r, jnp.asarray(total), tables.reshape(-1), cu_q,
            kv_heads=kv_heads, max_query_len=max_q, sliding_window=WINDOW,
            sm_scale=sm_scale, dcp=dcp, rank=jnp.int32(r), k_scale=scale, v_scale=scale,
            interpret=True,
        )
        outs.append(o)
        lses.append(l)
    out, lse = outs[0], lses[0]
    for o, l in zip(outs[1:], lses[1:]):
        out, lse = cda.merge(out, lse, o, l)
    # The block's own K/V, as stored (quantized) by the replicated write.
    blk_k = new_k if scale is None else (new_k / scale).astype(dtype).astype(jnp.float32) * scale
    blk_v = new_v if scale is None else (new_v / scale).astype(dtype).astype(jnp.float32) * scale
    if scale is None:
        blk_k = blk_k.astype(dtype).astype(jnp.float32)
        blk_v = blk_v.astype(dtype).astype(jnp.float32)
    b_out, b_lse = cda.attend_block(
        q, blk_k, blk_v, jnp.asarray(total), cu_q, max_query_len=max_q,
        sm_scale=sm_scale, causal=causal,
    )
    out, _ = cda.merge(out, lse, b_out, b_lse)
    if not causal:
        # `ref_ragged_paged_attention` drops the window when not causal; the
        # kernel keeps it. Compare against the kernel's mask.
        def stored_np(x):
            if scale is None:
                return np.asarray(x.astype(dtype).astype(jnp.float32))
            return np.asarray((x / scale).astype(dtype).astype(jnp.float32) * scale)

        keys = [np.concatenate([stored_np(ck), stored_np(new_k[a:b])]) for ck, a, b in
                zip(ctx_k, np.asarray(cu_q[:-1]), np.asarray(cu_q[1:]))]
        values = [np.concatenate([stored_np(cv), stored_np(new_v[a:b])]) for cv, a, b in
                  zip(ctx_v, np.asarray(cu_q[:-1]), np.asarray(cu_q[1:]))]
        ref_out = _dense_reference(
            np.asarray(q, np.float32), keys, values, total, np.asarray(cu_q), WINDOW,
            sm_scale, causal=False,
        )

    np.testing.assert_allclose(
        np.asarray(out, np.float32), np.asarray(ref_out, np.float32), rtol=3e-2, atol=3e-2
    )


def test_write_kv_keeps_only_owned_blocks():
    """Rank r of 2 stores exactly the tokens of global blocks b with b % 2 == r."""
    kv_heads, dcp = 1, 2
    ctx = jnp.asarray([300, 0, 0, 0], jnp.int32)
    cu = jnp.asarray([0, 300, 300, 300, 300], jnp.int32)
    k = jnp.ones((300, kv_heads, HEAD_DIM), jnp.float32)
    tables = jnp.asarray([[1, 2], [0, 0], [0, 0], [0, 0]], jnp.int32).reshape(-1)
    for r in range(dcp):
        cache = cda.write_kv(
            _cache(3, kv_heads, jnp.bfloat16), k, k, ctx, tables, cu, dcp=dcp, rank=jnp.int32(r)
        )
        written = np.asarray(jnp.any(cache != 0, axis=(2, 3, 4)))  # [pages, slots]
        if r == 0:  # global blocks 0 (page 1, all 128) and 2 (page 2, tokens 256..299)
            assert written[1].all() and written[2, :44].all() and not written[2, 44:].any()
        else:  # global block 1 -> page 1 (logical block 0 of rank 1)
            assert written[1].all() and not written[2].any()
        assert not written[0].any()


def test_gather_pages_lowers_for_tpu():
    from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    args = (
        jax.ShapeDtypeStruct((4096, PAGE, 2, 4, HEAD_DIM), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((64, 6), jnp.int32),
    )
    with use_abstract_mesh(tpu7x):
        lowered = jax.jit(cda.gather_pages).trace(*args).lower(lowering_platforms=("tpu",))
    assert "tpu_custom_call" in lowered.as_text()


def test_gather_pages_matches_take():
    rng = np.random.default_rng(9)
    cache = jnp.asarray(rng.standard_normal((10, PAGE, 1, 2, HEAD_DIM)), jnp.bfloat16)
    pages = jnp.asarray(rng.integers(0, 10, (3, 4)), jnp.int32)
    np.testing.assert_array_equal(cda.gather_pages(cache, pages, interpret=True), cache[pages])
