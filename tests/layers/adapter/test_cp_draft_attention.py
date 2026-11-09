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
@pytest.mark.parametrize("attend", ["xla", "flash"])
def test_sharded_matches_replicated(dcp, kv_heads, q_heads, dtype, causal, attend):
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
        attend_fn = cda.attend_cached_flash if attend == "flash" else cda.attend_cached
        o, l = attend_fn(
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


# ---------------------------------------------------------------------------
# attend_cached_flash: the Pallas paged sliding-window flash kernel.
# Not bitwise equal to attend_cached (online softmax, P rounded to bf16 for
# PV); checked against a float64 oracle and against attend_cached's argmax.
# ---------------------------------------------------------------------------

# float64-oracle tolerances: the output is bf16 (the op's contract), so each
# element within one bf16 ulp of the oracle plus 1e-5 (f32 accumulation of
# near-zero outputs); LSE (f32) within 1e-4.
FLASH_OUT_ULPS = 1.0
FLASH_OUT_ATOL = 1e-5
FLASH_LSE_ATOL = 1e-4


def _bf16_ulp(*xs):
    mag = np.maximum(np.max([np.abs(x) for x in xs], axis=0), 2.0**-126)
    return 2.0 ** (np.floor(np.log2(mag)) - 7)


def _flash_case(dcp, kv_heads, q_heads, dtype, seed, ctx=None, q_len=None, window=WINDOW):
    """Per-rank caches of a random context plus a query block."""
    rng = np.random.default_rng(seed)
    ctx = np.array([1000, 37, 600, 0, 2000, 130] if ctx is None else ctx, np.int32)
    q_len = np.array([7, 7, 3, 7, 0, 7] if q_len is None else q_len, np.int32)
    num_seqs = len(ctx)
    total = ctx + q_len
    pages_per_seq = -(-int(total.max()) // PAGE) + 1
    logical = -(-pages_per_seq // dcp) + 1
    # Shuffled pages, so a wrong page lookup reads someone else's KV.
    tables = jnp.asarray(
        rng.permutation(num_seqs * logical + 1)[: num_seqs * logical].reshape(num_seqs, -1),
        jnp.int32,
    )
    scale = None if dtype == jnp.bfloat16 else 0.5
    ck = rng.standard_normal((int(ctx.sum()), kv_heads, HEAD_DIM)).astype(np.float32)
    cv = rng.standard_normal((int(ctx.sum()), kv_heads, HEAD_DIM)).astype(np.float32)
    num_tokens = int(q_len.sum()) + 5  # padding tokens
    q = jnp.asarray(rng.standard_normal((num_tokens, q_heads, HEAD_DIM)), jnp.bfloat16)
    cu_q = jnp.asarray(np.concatenate([[0], np.cumsum(q_len)]), jnp.int32)
    cu_ctx = jnp.asarray(np.concatenate([[0], np.cumsum(ctx)]), jnp.int32)
    caches = [
        cda.write_kv(
            _cache(num_seqs * logical + 1, kv_heads, dtype), jnp.asarray(ck), jnp.asarray(cv),
            jnp.asarray(ctx), tables.reshape(-1), cu_ctx,
            dcp=dcp, rank=jnp.int32(r), k_scale=scale, v_scale=scale,
        )
        for r in range(dcp)
    ]

    def stored(x):
        if scale is None:
            return np.asarray(jnp.asarray(x).astype(dtype).astype(jnp.float32), np.float64)
        return np.asarray(
            (jnp.asarray(x) / scale).astype(dtype).astype(jnp.float32), np.float64
        ) * scale

    return dict(
        ctx=ctx, q_len=q_len, total=jnp.asarray(total), tables=tables.reshape(-1), cu_q=cu_q,
        q=q, caches=caches, scale=scale, keys=stored(ck), values=stored(cv),
        cu_ctx=np.asarray(cu_ctx), window=window, dcp=dcp, kv_heads=kv_heads,
    )


def _oracle(case, rank):
    """float64 partial attention of rank `rank`: keys of its blocks, below
    the cache length, in the window. Returns (out [T, H, D], lse [T, H])."""
    q = np.asarray(case["q"], np.float64)
    num_tokens, q_heads, _ = q.shape
    group = q_heads // case["kv_heads"]
    out = np.zeros(q.shape)
    lse = np.full((num_tokens, q_heads), -np.inf)
    cu_q = np.asarray(case["cu_q"])
    for s, c in enumerate(case["ctx"]):
        lo, hi = int(cu_q[s]), int(cu_q[s + 1])
        if hi == lo or c == 0:
            continue
        k = case["keys"][case["cu_ctx"][s] : case["cu_ctx"][s + 1]]
        v = case["values"][case["cu_ctx"][s] : case["cu_ctx"][s + 1]]
        k_pos = np.arange(c)
        q_pos = c + np.arange(hi - lo)
        mask = ((k_pos // PAGE) % case["dcp"] == rank)[None, :] & (
            q_pos[:, None] - k_pos[None, :] < case["window"]
        )
        kk = np.repeat(k, group, axis=1)
        vv = np.repeat(v, group, axis=1)
        sc = np.einsum("qhd,khd->hqk", q[lo:hi], kk) * HEAD_DIM**-0.5
        sc = np.where(mask[None], sc, -np.inf)
        m = sc.max(-1, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        p = np.exp(sc - m)
        l = p.sum(-1, keepdims=True)
        o = np.einsum("hqk,khd->qhd", p, vv) / np.where(l > 0, l, 1.0).transpose(1, 0, 2)
        out[lo:hi] = np.where((l > 0).transpose(1, 0, 2), o, 0.0)
        l0 = l[..., 0]
        lse[lo:hi] = np.where(l0 > 0, m[..., 0] + np.log(np.where(l0 > 0, l0, 1)), -np.inf).T
    return out, lse


def _attend(case, cache_rank, fn=None, **over):
    fn = fn or cda.attend_cached_flash
    kw = dict(
        kv_heads=case["kv_heads"], max_query_len=7, sliding_window=case["window"],
        sm_scale=HEAD_DIM**-0.5, dcp=case["dcp"], rank=jnp.int32(cache_rank),
        k_scale=case["scale"], v_scale=case["scale"], interpret=True,
    )
    kw.update(over)
    tables = kw.pop("tables", case["tables"])
    o, l = fn(case["q"], case["caches"][cache_rank], case["total"], tables, case["cu_q"], **kw)
    return np.asarray(o, np.float64), np.asarray(l, np.float64)


def _errors(case, rank, got):
    ref_o, ref_l = _oracle(case, rank)
    o, l = got
    live = np.isfinite(ref_l)
    if not (live == np.isfinite(l)).all():
        return np.inf, np.inf
    if not live.any():
        return float(np.abs(o).max()), 0.0
    # Output error in units of (one bf16 ulp + FLASH_OUT_ATOL): <= 1 passes.
    err = np.abs(o - ref_o) / (FLASH_OUT_ULPS * _bf16_ulp(o, ref_o) + FLASH_OUT_ATOL)
    return float(err[live].max()), float(np.abs(l - ref_l)[live].max())


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
@pytest.mark.parametrize("dcp,kv_heads,q_heads", [(1, 1, 8), (2, 1, 16), (8, 4, 64)])
def test_flash_matches_float64_oracle(dcp, kv_heads, q_heads, dtype):
    case = _flash_case(dcp, kv_heads, q_heads, dtype, seed=dcp)
    for r in range(dcp):
        flash = _attend(case, r)
        xla = _attend(case, r, cda.attend_cached)
        out_err, lse_err = _errors(case, r, flash)
        assert out_err <= 1.0 and lse_err < FLASH_LSE_ATOL, (r, out_err, lse_err)
        # Against the XLA path: within one ulp of the bf16 output, plus 1e-5
        # absolute for near-zero outputs (cancellation; f32 accumulation order).
        ulp = _bf16_ulp(xla[0], flash[0])
        diff = np.abs(flash[0] - xla[0])
        assert (diff <= ulp + 1e-5).all(), (r, (diff - ulp).max())
        live = np.isfinite(xla[1])
        assert (live == np.isfinite(flash[1])).all()
        assert np.abs(flash[1] - xla[1])[live].max(initial=0.0) < 1e-5


@pytest.mark.parametrize(
    "fault",
    ["wrong_rank", "window_minus_one_page", "page_table_swap", "no_v_scale"],
)
def test_flash_fault_controls(fault):
    """The oracle comparison catches each injected fault (positive controls)."""
    case = _flash_case(2, 1, 16, jnp.float8_e4m3fn, seed=11)
    rank, over = 0, {}
    if fault == "wrong_rank":
        over = dict(rank=jnp.int32(1))
    elif fault == "window_minus_one_page":
        over = dict(sliding_window=WINDOW - PAGE)
    elif fault == "page_table_swap":
        t = np.asarray(case["tables"]).reshape(len(case["ctx"]), -1).copy()
        t[[0, 4]] = t[[4, 0]]
        over = dict(tables=jnp.asarray(t.reshape(-1)))
    elif fault == "no_v_scale":
        over = dict(v_scale=None)
    out_err, lse_err = _errors(case, rank, _attend(case, rank, **over))
    assert out_err > 1.0 or lse_err >= FLASH_LSE_ATOL, (fault, out_err, lse_err)


def _merge_np(parts):
    out, lse = parts[0]
    for o, l in parts[1:]:
        m = np.maximum(lse, l)
        m = np.where(np.isfinite(m), m, 0.0)
        wa, wb = np.exp(lse - m), np.exp(l - m)
        norm = np.where(wa + wb > 0, wa + wb, 1.0)
        out = (wa[..., None] * out + wb[..., None] * o) / norm[..., None]
        lse = np.where(wa + wb > 0, m + np.log(norm), -np.inf)
    return out


def test_flash_argmax_matches_xla_path():
    """Draft-logit argmax (both ranks merged, through a random LM head).

    Both paths return bf16 (the op's contract), and that rounding alone moves
    the argmax of near-tied rows: the XLA path agrees with its own unrounded
    (f32) output on ~99.5% of rows here. So: every row whose float64 top-2
    logit gap is clear of that noise (> 1e-2; one output ulp moves a logit by
    < 4e-3 here) has the same argmax in the kernel, the XLA path and float64,
    and the kernel's overall agreement with the XLA path is no worse than the
    XLA path's own rounding noise."""
    rng = np.random.default_rng(5)
    head = rng.standard_normal((16 * HEAD_DIM, 2048))
    rows = 0
    agree = dict(flash_xla=0, flash_f64=0, xla_f64=0, xla_unrounded=0)
    for seed in range(4):
        n = 24
        ctx = rng.integers(0, 2600, n).astype(np.int32)
        q_len = np.full(n, 7, np.int32)
        case = _flash_case(2, 1, 16, jnp.float8_e4m3fn, seed=100 + seed, ctx=ctx, q_len=q_len)
        f32_case = dict(case, q=case["q"].astype(jnp.float32))  # same values, f32 output
        num = int(q_len.sum())
        outs = dict(
            flash=_merge_np([_attend(case, r) for r in range(2)]),
            xla=_merge_np([_attend(case, r, cda.attend_cached) for r in range(2)]),
            unrounded=_merge_np([_attend(f32_case, r, cda.attend_cached) for r in range(2)]),
            f64=_merge_np([_oracle(case, r) for r in range(2)]),
        )
        logits = {k: v[:num].reshape(num, -1) @ head for k, v in outs.items()}
        top = {k: v.argmax(-1) for k, v in logits.items()}
        live = np.asarray(ctx > 0).repeat(7)
        srt = np.sort(logits["f64"], axis=-1)
        clear = live & (srt[:, -1] - srt[:, -2] > 1e-2)
        for k in ("flash", "xla"):
            assert (top[k] == top["f64"])[clear].all(), k
        agree["flash_xla"] += int((top["flash"] == top["xla"])[live].sum())
        agree["flash_f64"] += int((top["flash"] == top["f64"])[live].sum())
        agree["xla_f64"] += int((top["xla"] == top["f64"])[live].sum())
        agree["xla_unrounded"] += int((top["xla"] == top["unrounded"])[live].sum())
        rows += int(live.sum())
    print("argmax agreement over", rows, "rows:", {k: v / rows for k, v in agree.items()})
    assert rows >= 500
    assert agree["flash_xla"] >= agree["xla_unrounded"] - 2, agree
    assert agree["flash_f64"] >= agree["xla_f64"] - 2, agree


@pytest.mark.parametrize("dcp,kv_heads,q_heads", [(2, 1, 16), (8, 4, 64)])
@pytest.mark.parametrize("num_tokens,max_seqs", [(32, 4), (128, 18), (512, 73), (1024, 146)])
def test_flash_compiles_for_tpu7x(dcp, kv_heads, q_heads, num_tokens, max_seqs):
    """Deviceless tpu7x AOT compile at the served draft shapes (T // 7 requests)."""
    from jax.experimental import topologies
    from jax.sharding import SingleDeviceSharding

    try:
        topology = topologies.get_topology_desc("tpu7x:2x2x1", "tpu")
    except Exception as e:  # no libtpu
        pytest.skip(f"no deviceless tpu7x topology: {e}")
    sharding = SingleDeviceSharding(topology.devices[0])

    def spec(shape, dt):
        return jax.ShapeDtypeStruct(shape, dt, sharding=sharding)

    words = -(-2 * kv_heads // 4)
    args = (
        spec((num_tokens, q_heads, HEAD_DIM), jnp.bfloat16),
        spec((65536, PAGE, words, 4, HEAD_DIM), jnp.float8_e4m3fn),
        spec((max_seqs,), jnp.int32),
        spec((max_seqs * 64,), jnp.int32),
        spec((max_seqs + 1,), jnp.int32),
        spec((1,), jnp.int32),
    )

    def fn(q, cache, seq_lens, tables, cu_q, rank):
        return cda.attend_cached_flash(
            q, cache, seq_lens, tables, cu_q, kv_heads=kv_heads, max_query_len=7,
            sliding_window=4096, sm_scale=HEAD_DIM**-0.5, dcp=dcp, rank=rank[0],
            k_scale=0.7, v_scale=0.3,
        )

    from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = jax.jit(fn).lower(*args)
    compiled = lowered.compile()
    assert "tpu_custom_call" in compiled.as_text()
