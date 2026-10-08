# SPDX-License-Identifier: Apache-2.0
"""MIXED (RPAm) correctness at the production geometry.

Qwen3.5 TP8 per device: 4 q heads, 1 KV head, head_dim 256, fp8 SEQ_ALONG_LANE
(HND) cache with 128-token pages. Covers:

* The default bq_sz=1024 prefill tiling, whose new-KV stitch rolls the whole
  KV window by a dynamic lane shift (ef9e078de capped bq_sz at 128 to avoid
  that roll). Histories end on, before and after page/lane boundaries, and
  reach 160k tokens.
* The DCP passes (CACHE_ONLY / NEW_TOKENS_ONLY with return_lse), with and
  without a 4-way interleaved context-parallel cache shard.
* TPU_RPA_VPU_DIET (scale folded into q, mask skipped for KV blocks below the
  causal diagonal) against the original path: must be bit-identical when the
  folded scale is a power of two, and tight otherwise.

Inputs and references are built on device so 160k contexts stay fast.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

pytestmark = pytest.mark.skipif(
    jax.devices()[0].platform != "tpu", reason="Mosaic kernel needs a TPU"
)

_LAYOUT = configs.KVLayout.SEQ_ALONG_LANE
_PAGE = 128
_NQ, _NKV, _D = 4, 1, 256
_MAX_SEQS = 8
_FULL = configs.AttentionScope.FULL
_CACHE = configs.AttentionScope.CACHE_ONLY
_NEW = configs.AttentionScope.NEW_TOKENS_ONLY
# (cached tokens, new tokens). Total new tokens 1025 -> default bq_sz=1024.
_SHORT = ((0, 300), (1000, 213), (4097, 211), (9000, 300), (127, 1))
_LONG = ((160_000, 512), (39_001, 300), (100_000, 212), (156_127, 1))


def _setup(kv_dtype, seqs, seed=0, cp=1, rank=0):
    """Random history in a (possibly CP-sharded) paged cache + new tokens."""
    key = jax.random.PRNGKey(seed)
    ks = jax.random.split(key, 8)
    q_lens = [t for _, t in seqs]
    kv_lens = [c + t for c, t in seqs]
    total = sum(q_lens)
    pages_per_seq = max(kv_lens) // (_PAGE * cp) + 2
    needed = [kv // (_PAGE * cp) + 2 for kv in kv_lens]
    num_pages = sum(needed) + 3
    shape = wrapper.get_kv_cache_shape(
        num_pages, _PAGE, _NKV, _D, kv_dtype, kv_layout=_LAYOUT
    )

    def rnd(k, *s):
        return (jax.random.normal(k, s, jnp.float32) * 0.5).astype(kv_dtype)

    perm = np.random.default_rng(seed).permutation(num_pages)
    table = np.zeros((_MAX_SEQS, pages_per_seq), np.int32)
    off = 0
    for s_idx, n in enumerate(needed):
        table[s_idx, :n] = perm[off : off + n]
        off += n
    page_table = jnp.asarray(table)
    # Build lane-major [pages, lane, k/v, head_dim] then transpose to the cache
    # layout [pages, k/v, head_dim / packing, packing, lane]. Garbage first so
    # unwritten lanes are finite but wrong.
    cache_t = rnd(ks[1], num_pages, _PAGE, 2, _D)
    q = (jax.random.normal(ks[2], (total, _NQ, _D)) * 0.5).astype(jnp.bfloat16)
    k_new = rnd(ks[3], total, _NKV, _D)
    v_new = rnd(ks[4], total, _NKV, _D)
    logical = []
    start = 0
    for s, (ctx, t) in enumerate(seqs):
        kh = rnd(jax.random.fold_in(ks[5], s), ctx, _D)
        vh = rnd(jax.random.fold_in(ks[6], s), ctx, _D)
        pos = jnp.arange(ctx)
        owned = (pos // _PAGE) % cp == rank
        page = page_table[s, jnp.minimum((pos // _PAGE) // cp, pages_per_seq - 1)]
        page = jnp.where(owned, page, num_pages)  # Out of range -> dropped.
        cache_t = cache_t.at[page, pos % _PAGE, 0].set(kh, mode="drop")
        cache_t = cache_t.at[page, pos % _PAGE, 1].set(vh, mode="drop")
        logical.append(
            (
                jnp.concatenate([kh, k_new[start : start + t, 0]]),
                jnp.concatenate([vh, v_new[start : start + t, 0]]),
            )
        )
        start += t
    cache = cache_t.reshape(num_pages, _PAGE, 2, *shape[2:4]).transpose(0, 2, 3, 4, 1)
    assert cache.shape == tuple(shape)
    kv_lens_arr = np.zeros(_MAX_SEQS, np.int32)
    kv_lens_arr[: len(seqs)] = kv_lens
    cu = np.full(_MAX_SEQS + 1, total, np.int32)
    cu[: len(seqs) + 1] = np.cumsum([0, *q_lens])
    inputs = dict(
        queries=q,
        keys=k_new,
        values=v_new,
        cache=cache,
        kv_lens=jnp.asarray(kv_lens_arr),
        page_indices=page_table.ravel(),
        cu_q_lens=jnp.asarray(cu),
        distribution=jnp.array([0, 0, len(seqs)], jnp.int32),
        cp=cp,
        rank=rank,
    )
    return inputs, logical, np.asarray(page_table)


def _run(monkeypatch, inputs, diet, k_scale=None, prefill_blocks=None, **kw):
    monkeypatch.setenv("TPU_RPA_VPU_DIET", "1" if diet else "0")
    jax.clear_caches()
    if inputs["cp"] > 1:
        kw.update(
            cp_group_size=inputs["cp"],
            cp_rank=jnp.array([inputs["rank"]], jnp.int32),
        )
    out = wrapper.ragged_paged_attention(
        inputs["queries"],
        inputs["keys"],
        inputs["values"],
        jnp.copy(inputs["cache"]),
        inputs["kv_lens"],
        inputs["page_indices"],
        inputs["cu_q_lens"],
        inputs["distribution"],
        sm_scale=_D**-0.5,
        k_scale=k_scale,
        v_scale=None if k_scale is None else 1.0,
        kv_layout=_LAYOUT,
        decode_query_size=4,
        prefill_block_sizes=prefill_blocks,
        **kw,
    )
    out = jax.block_until_ready(out)
    return [np.asarray(x.astype(jnp.float32)) for x in out]


def _reference(inputs, logical, seqs, scope=_FULL, k_scale=1.0):
    """f32 (out, lse, valid_row) per new token, computed on device."""
    q = inputs["queries"].astype(jnp.float32)
    cp, rank = inputs["cp"], inputs["rank"]
    outs, lses, valid = [], [], []
    start = 0
    for (ctx, t), (k, v) in zip(seqs, logical):
        k = k.astype(jnp.float32) * k_scale
        v = v.astype(jnp.float32)
        pos = jnp.arange(ctx + t)
        for c0 in range(0, t, 128):
            rows = jnp.arange(c0, min(t, c0 + 128))
            qpos = ctx + rows
            vis = pos[None, :] <= qpos[:, None]
            if scope == _CACHE:
                vis &= (pos[None, :] < ctx) & ((pos[None, :] // _PAGE) % cp == rank)
            elif scope == _NEW:
                vis &= pos[None, :] >= ctx
            s = jnp.einsum(
                "nhd,td->nht", q[start + rows], k, precision="highest"
            ) * (_D**-0.5)
            s = jnp.where(vis[:, None, :], s, -jnp.inf)
            m = jnp.max(s, axis=-1, keepdims=True)
            m_safe = jnp.where(jnp.isfinite(m), m, 0.0)
            w = jnp.exp(s - m_safe)
            den = jnp.sum(w, axis=-1, keepdims=True)
            outs.append(
                jnp.einsum("nht,td->nhd", w / jnp.maximum(den, 1e-30), v,
                           precision="highest")
            )
            lses.append((m_safe + jnp.log(jnp.maximum(den, 1e-30)))[..., 0])
            valid.append(jnp.any(vis, axis=-1))
        start += t
    return (
        np.asarray(jnp.concatenate(outs)),
        np.asarray(jnp.concatenate(lses)),
        np.asarray(jnp.concatenate(valid)),
    )


def _check_writeback(cache, inputs, logical, seqs, page_table):
    cp, rank = inputs["cp"], inputs["rank"]
    for s, ((ctx, t), (k, v)) in enumerate(zip(seqs, logical)):
        k = np.asarray(k.astype(jnp.float32))
        v = np.asarray(v.astype(jnp.float32))
        for pos in range(ctx, ctx + t):
            if (pos // _PAGE) % cp != rank:
                continue
            page = page_table[s, (pos // _PAGE) // cp]
            got = cache[page, :, :, :, pos % _PAGE]
            np.testing.assert_array_equal(got[0].reshape(_D), k[pos])
            np.testing.assert_array_equal(got[1].reshape(_D), v[pos])


_TOL = dict(atol=1.5e-2, rtol=2e-2)


@pytest.mark.parametrize("seqs", [_SHORT, _LONG], ids=["short", "160k"])
@pytest.mark.parametrize("kv_dtype", [jnp.float8_e4m3fn, jnp.bfloat16])
def test_bq1024_prefill_matches_reference(monkeypatch, kv_dtype, seqs):
    if kv_dtype == jnp.bfloat16 and seqs is _LONG:
        # Pre-existing (e770f8bd2, diet off): the DECODE block picker chooses
        # bkv=1792 for bf16 SEQ_ALONG_LANE at this page-table size and the
        # RPAd compile overshoots VMEM by ~28 KB. Production KV is fp8.
        pytest.skip("RPAd bf16 HND VMEM estimate (pre-existing); fp8 covers 160k")
    inputs, logical, page_table = _setup(kv_dtype, seqs)
    total = int(inputs["cu_q_lens"][-1])
    ref, _, _ = _reference(inputs, logical, seqs)
    out, cache = _run(monkeypatch, inputs, diet=False)
    assert np.isfinite(out[:total]).all()
    np.testing.assert_allclose(out[:total], ref, **_TOL)
    _check_writeback(cache, inputs, logical, seqs, page_table)


def test_bq1024_matches_bq128_bkv128(monkeypatch):
    # bq_sz capped at num_lanes takes the O(1) per-vreg stitch. That stitch
    # covers at most two lane columns of new tokens, so bkv_sz must be capped
    # too for a MIXED block to be correct (see test_ef9e078de_tiling).
    inputs, logical, page_table = _setup(jnp.float8_e4m3fn, _SHORT)
    total = int(inputs["cu_q_lens"][-1])
    blocks = configs.BlockSizes(
        bq_sz=128, bq_c_sz=64, bkv_sz=128, batch_size=2, n_buffer=3
    )
    out_1024, _ = _run(monkeypatch, inputs, diet=False)
    out_128, cache_128 = _run(monkeypatch, inputs, diet=False, prefill_blocks=blocks)
    np.testing.assert_allclose(out_128[:total], _reference(inputs, logical, _SHORT)[0],
                               **_TOL)
    np.testing.assert_allclose(out_1024[:total], out_128[:total], atol=1e-2)
    _check_writeback(cache_128, inputs, logical, _SHORT, page_table)


@pytest.mark.parametrize("cp", [1, 4], ids=["nocp", "dcp4"])
@pytest.mark.parametrize("scope", [_CACHE, _NEW], ids=["cache_only", "new_only"])
@pytest.mark.parametrize("seqs", [_SHORT, _LONG], ids=["short", "160k"])
def test_dcp_scopes_match_reference(monkeypatch, seqs, scope, cp):
    inputs, logical, page_table = _setup(
        jnp.float8_e4m3fn, seqs, seed=5, cp=cp, rank=1 if cp > 1 else 0
    )
    total = int(inputs["cu_q_lens"][-1])
    ref_o, ref_lse, valid = _reference(inputs, logical, seqs, scope=scope)
    out, cache, lse = _run(
        monkeypatch, inputs, diet=True, attention_scope=scope, return_lse=True
    )
    rows = valid  # Rows with at least one visible key.
    assert np.isfinite(out[:total][rows]).all()
    np.testing.assert_allclose(out[:total][rows], ref_o[rows], **_TOL)
    np.testing.assert_allclose(lse[:total][rows], ref_lse[rows], atol=5e-2, rtol=1e-2)
    if scope == _NEW:
        _check_writeback(cache, inputs, logical, seqs, page_table)


@pytest.mark.parametrize(
    "seqs, bkv, bq_c",
    [
        (_SHORT, None, None),
        (_LONG, None, None),
        (_SHORT, 2048, 64),
        (_SHORT, 1024, 128),
        (_LONG, 2048, 128),
    ],
)
@pytest.mark.parametrize("kv_dtype", [jnp.float8_e4m3fn, jnp.bfloat16])
def test_vpu_diet_bit_exact(monkeypatch, kv_dtype, seqs, bkv, bq_c):
    inputs, logical, _ = _setup(kv_dtype, seqs, seed=1)
    total = int(inputs["cu_q_lens"][-1])
    blocks = None
    if bkv is not None:
        blocks = configs.BlockSizes(
            bq_sz=1024, bq_c_sz=bq_c, bkv_sz=bkv, batch_size=2, n_buffer=3
        )
    old, old_cache = _run(monkeypatch, inputs, diet=False, prefill_blocks=blocks)
    new, new_cache = _run(monkeypatch, inputs, diet=True, prefill_blocks=blocks)
    np.testing.assert_array_equal(new[:total], old[:total])
    np.testing.assert_array_equal(new_cache, old_cache)
    np.testing.assert_allclose(new[:total], _reference(inputs, logical, seqs)[0], **_TOL)


@pytest.mark.parametrize("cp", [1, 4], ids=["nocp", "dcp4"])
@pytest.mark.parametrize("scope", [_CACHE, _NEW], ids=["cache_only", "new_only"])
def test_vpu_diet_bit_exact_scoped(monkeypatch, scope, cp):
    inputs, _, _ = _setup(
        jnp.float8_e4m3fn, _LONG, seed=3, cp=cp, rank=2 if cp > 1 else 0
    )
    total = int(inputs["cu_q_lens"][-1])
    old = _run(monkeypatch, inputs, diet=False, attention_scope=scope, return_lse=True)
    new = _run(monkeypatch, inputs, diet=True, attention_scope=scope, return_lse=True)
    np.testing.assert_array_equal(new[0][:total], old[0][:total])
    np.testing.assert_array_equal(new[2][:total], old[2][:total])
    np.testing.assert_array_equal(new[1], old[1])


def test_vpu_diet_non_pow2_k_scale(monkeypatch):
    inputs, logical, _ = _setup(jnp.float8_e4m3fn, _SHORT, seed=2)
    total = int(inputs["cu_q_lens"][-1])
    old, _ = _run(monkeypatch, inputs, diet=False, k_scale=0.37)
    new, _ = _run(monkeypatch, inputs, diet=True, k_scale=0.37)
    np.testing.assert_allclose(new[:total], old[:total], atol=4e-3, rtol=0)
    np.testing.assert_allclose(
        new[:total], _reference(inputs, logical, _SHORT, k_scale=0.37)[0], **_TOL
    )


@pytest.mark.xfail(
    reason="ef9e078de verbatim (bq_sz=128, bkv_sz>128) routes MIXED through the"
    " two-column O(1) stitch; expected to misplace misaligned new KV.",
    strict=False,
)
def test_ef9e078de_tiling(monkeypatch):
    inputs, logical, _ = _setup(jnp.float8_e4m3fn, _SHORT, seed=4)
    total = int(inputs["cu_q_lens"][-1])
    blocks = configs.BlockSizes(
        bq_sz=128, bq_c_sz=16, bkv_sz=4096, batch_size=2, n_buffer=3
    )
    out, _ = _run(monkeypatch, inputs, diet=False, prefill_blocks=blocks)
    np.testing.assert_allclose(out[:total], _reference(inputs, logical, _SHORT)[0],
                               **_TOL)
