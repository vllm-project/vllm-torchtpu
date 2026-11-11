# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_PREFILL_UNION_WALK at a TP8 rank's head geometry (3 query heads,
one KV head, 512-row pages): the union walk agrees with float64 within twice
the block-sparse kernel's error, its cap / min-context fallbacks are the
block-sparse call bit for bit, and the ``_prefill_attention`` dispatch takes the
walk only with the flag on. Off TPU the Pallas kernels run in interpret mode.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as bsp
from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa
from vllm_torchtpu.kernels.qwen4_exp import qsa_union_dense as ud
from vllm_torchtpu.kernels.qwen4_exp.prefill_tiles import get_sparse_prefill_tiles

RATIO, TOPK, PAGE, HEADS = 4, 512, 512, 3
CAP = 8192
ROWS, CACHED = (37, 300, 129, 201), (5000, 1500, 4000, 700)


@pytest.fixture
def kernels(monkeypatch):
    if jax.default_backend() != "tpu":
        from jax._src import tpu_info

        monkeypatch.setitem(
            tpu_info.registry,
            jax.devices()[0].device_kind,
            lambda: tpu_info._get_tpu_info_impl(tpu_info.ChipVersion.TPU_7X, 1),
        )
        original = pl.pallas_call

        def interpreted(*args, **kwargs):
            kwargs["interpret"] = True
            return original(*args, **kwargs)

        monkeypatch.setattr(bsp.pl, "pallas_call", interpreted)
    for name in (
        "VLLM_TPU_QSA_PREFILL_UNION_WALK",
        "VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT",
        "VLLM_TPU_QSA_PREFILL_UNION_CAP",
        "VLLM_TPU_QSA_PREFILL_UNION_KV_TILE",
        "VLLM_TPU_QSA_PREFILL_LEAN",
        "VLLM_TPU_QSA_PREFILL_PREFIX_PAGES",
        "VLLM_TPU_QSA_PREFILL_UNION_QK_AHEAD",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    jax.clear_caches()


def _step(seed, tokens=1024):
    rng = np.random.default_rng(seed)
    pages_per_seq = max(-(-(c + n) // PAGE) for n, c in zip(ROWS, CACHED))
    pool_pages = pages_per_seq * len(ROWS)
    starts = np.concatenate([[0], np.cumsum(ROWS)]).astype(np.int32)
    owner = rng.permutation(pool_pages)
    lengths = np.zeros(tokens, np.int32)
    blocks = np.full((tokens, TOPK), -1, np.int32)
    table = np.zeros((tokens, pages_per_seq), np.int32)
    for r, (n, ctx) in enumerate(zip(ROWS, CACHED)):
        pages = owner[r * pages_per_seq : (r + 1) * pages_per_seq]
        for i in range(n):
            t = starts[r] + i
            length = ctx + i + 1
            lengths[t] = length
            table[t] = pages
            complete = length // RATIO
            if complete:
                local = list(range(max(0, complete - 32), complete))
                rest = rng.permutation(complete)[: TOPK - len(local)]
                pick = np.unique(np.concatenate([local, rest]))[:TOPK]
                blocks[t, : len(pick)] = rng.permutation(pick)
    q = jnp.asarray(rng.normal(0, 0.5, (tokens, HEADS, 256)), jnp.bfloat16)
    gate = jnp.asarray(rng.uniform(-3, 3, (tokens, HEADS, 256)), jnp.bfloat16)
    raw = rng.integers(0, 0x40, size=(pool_pages, PAGE, 1, 4, 256), dtype=np.uint8)
    raw |= rng.integers(0, 2, size=raw.shape, dtype=np.uint8) << 7
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    return (q, gate, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table),
            pool, jnp.asarray(starts))


def _bits(x):
    return np.asarray(x).view(np.uint16)


def _walk(step, **options):
    q, gate, blocks, lengths, table, pool, starts = step
    params = dict(group=128, kv_tile=1024, union_cap=CAP, min_context=0, lean=True)
    params.update(options)
    out, cache = ud.union_walk_prefill(
        q, gate, blocks, lengths, table, pool, starts, kv_heads=1, ratio=RATIO, **params
    )
    assert cache is pool
    return np.asarray(out)


def _served(step):
    q, gate, blocks, lengths, table, pool, starts = step
    query_block, kv_tile, query_tile = get_sparse_prefill_tiles(
        q, pool, kv_heads=1, ratio=RATIO, with_query_tile=True
    )
    out, _ = bsp.block_sparse_gqa_attention(
        q, gate, blocks, lengths, table, pool, starts, kv_heads=1, ratio=RATIO,
        query_block=query_block, kv_tile=kv_tile,
        **({"query_tile": query_tile} if query_tile else {}),
    )
    return np.asarray(out)


def _f64_reference(step, rows_sel):
    q, gate, blocks, lengths, table, pool, _ = step
    q64, g64 = np.asarray(q, np.float64), np.asarray(gate, np.float64)
    flat = np.asarray(pool.reshape(-1, 4, 256)).astype(np.float64)
    table, blocks, lengths = np.asarray(table), np.asarray(blocks), np.asarray(lengths)
    out = np.zeros((len(rows_sel), HEADS, 256))
    for n, row in enumerate(rows_sel):
        length = int(lengths[row])
        groups = {int(g) for g in blocks[row] if g >= 0}
        if length % RATIO:
            groups.add(length // RATIO)
        toks = np.asarray(sorted(
            t for g in groups for t in range(g * RATIO, g * RATIO + RATIO) if t < length
        ))
        kv = flat[table[row][toks // PAGE] * PAGE + toks % PAGE]
        for h in range(HEADS):
            k, v = kv[:, 0], kv[:, 1]
            s = k @ q64[row, h] / 16.0
            p = np.exp(s - s.max())
            out[n, h] = (p / p.sum()) @ v
    return out / (1.0 + np.exp(-g64[rows_sel]))


def test_tp_rank_walk_within_twice_the_served_error(kernels):
    step = _step(13)
    served = _served(step)
    jax.clear_caches()
    got = _walk(step)
    starts = np.asarray(step[-1])
    assert not got[starts[-1]:].astype(np.float32).any()
    rows_sel = np.concatenate([
        np.linspace(starts[r], starts[r + 1] - 1, 4).astype(int) for r in range(len(ROWS))
    ])
    ref = _f64_reference(step, rows_sel)
    rms = np.sqrt(np.mean(ref**2, axis=-1, keepdims=True)) + 1e-30

    def err(x):
        e = np.abs(x[rows_sel].astype(np.float32).astype(np.float64) - ref) / rms
        return e.max(), np.quantile(e, 0.99)

    served_max, served_p99 = err(served)
    got_max, got_p99 = err(got)
    assert served_max < 0.1, served_max
    assert got_max <= 2 * served_max and got_p99 <= 2 * served_p99, (
        got_max, got_p99, served_max, served_p99
    )


def test_tp_rank_fallbacks_are_the_served_call(kernels):
    step = _step(11)
    served = _served(step)
    jax.clear_caches()
    np.testing.assert_array_equal(_bits(_walk(step, union_cap=32)), _bits(served))
    jax.clear_caches()
    longest = int(np.asarray(step[3]).max())
    np.testing.assert_array_equal(_bits(_walk(step, min_context=longest + 1)), _bits(served))


def test_tp_rank_dispatch(kernels, monkeypatch):
    step = _step(15)
    q, gate, blocks, lengths, table, pool, starts = step
    calls = []
    original = ud.union_walk_prefill

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(ud, "union_walk_prefill", spy)

    def dispatch():
        out, cache = qsa_rpa._prefill_attention(
            q, gate, blocks, lengths, table, pool, starts, kv_heads=1, ratio=RATIO
        )
        assert cache is pool
        return np.asarray(out)

    off = dispatch()
    assert calls == []
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_UNION_WALK", "128")
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT", "1000")
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_LEAN", "1")
    # 512-key walk tiles: at these short contexts a 128-row group's union is
    # nearly every visible group, so 1024-key walk tiles would line up with the
    # block-sparse kernel's 1024-key tiles and accumulate in the same order.
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_UNION_KV_TILE", "512")
    jax.clear_caches()
    on = dispatch()
    assert len(calls) == 1 and calls[0]["kv_heads"] == 1 and calls[0]["group"] == 128
    jax.clear_caches()
    want = _walk(step, kv_tile=512, min_context=1000)
    np.testing.assert_array_equal(_bits(on), _bits(want))
    assert not np.array_equal(_bits(on), _bits(off))


@pytest.mark.parametrize("qk16", [False, True])
def test_tp_rank_walk_qk_ahead_bitwise(kernels, monkeypatch, qk16):
    """VLLM_TPU_QSA_PREFILL_UNION_QK_AHEAD only reorders the tile pair's work."""
    step = _step(17)
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_UNION_QK_AHEAD", "0")
    jax.clear_caches()
    base = _walk(step, qk16=qk16)
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_UNION_QK_AHEAD", "1")
    jax.clear_caches()
    ahead = _walk(step, qk16=qk16)
    np.testing.assert_array_equal(_bits(ahead), _bits(base))
