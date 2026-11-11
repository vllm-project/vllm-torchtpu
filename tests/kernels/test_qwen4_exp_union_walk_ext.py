# SPDX-License-Identifier: Apache-2.0
"""The union walk on a paired page's second layer (VLLM_TPU_QSA_PREFILL_UNION_OFFSET,
``kv_head_offset=1``) and under main-KV DCP-2 (VLLM_TPU_QSA_PREFILL_UNION_DCP: this
rank's owned keys, with the LSE the DCP combine merges), at a TP8 rank's head
geometry (3 query heads per KV head; 6 gathered heads under DCP) on 512-row FP8
pages.

The walk is not bitwise against the block-sparse kernel (union-order online
softmax), so each output and LSE is held to float64 within twice the
block-sparse kernel's own error; the cap fallback is the block-sparse call bit
for bit; the two DCP ranks combined match the full float64 attention; and
``_prefill_attention`` takes the walk for these layers only with the sub-flag
on. Off TPU the Pallas kernels run in interpret mode.
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

RATIO, TOPK, PAGE = 4, 512, 512
CAP = 8192
ROWS, CACHED = (37, 300, 129, 201), (5000, 1500, 4000, 700)
FLAGS = (
    "VLLM_TPU_QSA_PREFILL_UNION_WALK",
    "VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT",
    "VLLM_TPU_QSA_PREFILL_UNION_CAP",
    "VLLM_TPU_QSA_PREFILL_UNION_KV_TILE",
    "VLLM_TPU_QSA_PREFILL_UNION_OFFSET",
    "VLLM_TPU_QSA_PREFILL_UNION_DCP",
    "VLLM_TPU_QSA_PREFILL_LEAN",
    "VLLM_TPU_QSA_PREFILL_PREFIX_PAGES",
    "VLLM_TPU_QSA_PREFILL_UNION_QK_AHEAD",
    "VLLM_TPU_QSA_DCP_PREFILL_QUERY_BLOCK",
    "VLLM_TPU_QSA_DCP_PREFILL_QUERY_TILE",
)


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
    for name in FLAGS:
        monkeypatch.delenv(name, raising=False)
    yield
    jax.clear_caches()


def _step(seed, *, heads, world=1, tokens=1024):
    """A packed 4-request step in global token numbering, plus each DCP rank's table.

    Returns ``(q, gate, blocks, lengths, tables, pool, starts, phys)``:
    ``tables[r]`` int32[T, entries] is rank r's local page table (entry e holds
    global page ``e * world + r``), ``phys`` int32[T, global pages] the pool page
    of each global page (for the float64 reference).
    """
    rng = np.random.default_rng(seed)
    global_pages = max(-(-(c + n) // PAGE) for n, c in zip(ROWS, CACHED))
    entries = -(-global_pages // world)
    pool_pages = entries * world * len(ROWS)
    starts = np.concatenate([[0], np.cumsum(ROWS)]).astype(np.int32)
    owner = rng.permutation(pool_pages)
    lengths = np.zeros(tokens, np.int32)
    blocks = np.full((tokens, TOPK), -1, np.int32)
    phys = np.zeros((tokens, entries * world), np.int32)
    for r, (n, ctx) in enumerate(zip(ROWS, CACHED)):
        pages = owner[r * entries * world : (r + 1) * entries * world]
        for i in range(n):
            t = starts[r] + i
            length = ctx + i + 1
            lengths[t] = length
            phys[t] = pages
            complete = length // RATIO
            if complete:
                local = list(range(max(0, complete - 32), complete))
                rest = rng.permutation(complete)[: TOPK - len(local)]
                pick = np.unique(np.concatenate([local, rest]))[:TOPK]
                blocks[t, : len(pick)] = rng.permutation(pick)
    tables = [jnp.asarray(phys[:, rank::world]) for rank in range(world)]
    q = jnp.asarray(rng.normal(0, 0.5, (tokens, heads, 256)), jnp.bfloat16)
    gate = jnp.asarray(rng.uniform(-3, 3, (tokens, heads, 256)), jnp.bfloat16)
    raw = rng.integers(0, 0x40, size=(pool_pages, PAGE, 1, 4, 256), dtype=np.uint8)
    raw |= rng.integers(0, 2, size=raw.shape, dtype=np.uint8) << 7
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    return (q, gate, jnp.asarray(blocks), jnp.asarray(lengths), tables, pool,
            jnp.asarray(starts), phys)


def _reference(step, rows_sel, *, offset=0, world=1, rank=0):
    """float64 (normalized ungated out [n, H, D], lse [n, H]) over the rank's owned keys."""
    q, _, blocks, lengths, _, pool, _, phys = step
    heads = q.shape[1]
    q64 = np.asarray(q, np.float64)
    flat = np.asarray(pool.reshape(-1, 4, 256)).astype(np.float64)
    blocks, lengths = np.asarray(blocks), np.asarray(lengths)
    out = np.zeros((len(rows_sel), heads, 256))
    lse = np.full((len(rows_sel), heads), -np.inf)
    for n, row in enumerate(rows_sel):
        length = int(lengths[row])
        groups = {int(g) for g in blocks[row] if g >= 0}
        if length % RATIO:
            groups.add(length // RATIO)
        toks = np.asarray(sorted(
            t for g in groups for t in range(g * RATIO, g * RATIO + RATIO)
            if t < length and (t // PAGE) % world == rank
        ), np.int64)
        if not len(toks):
            continue
        kv = flat[phys[row][toks // PAGE] * PAGE + toks % PAGE]
        k, v = kv[:, 2 * offset], kv[:, 2 * offset + 1]
        for h in range(heads):
            s = k @ q64[row, h] / 16.0
            p = np.exp(s - s.max())
            out[n, h] = (p / p.sum()) @ v
            lse[n, h] = s.max() + np.log(p.sum())
    return out, lse


def _gated(step, rows_sel, out):
    g64 = np.asarray(step[1], np.float64)[rows_sel]
    return out / (1.0 + np.exp(-g64))


def _rows(step):
    starts = np.asarray(step[6])
    return np.concatenate([
        np.linspace(starts[r], starts[r + 1] - 1, 4).astype(int) for r in range(len(ROWS))
    ])


def _err(x, ref):
    rms = np.sqrt(np.mean(ref**2, axis=-1, keepdims=True)) + 1e-30
    e = np.abs(np.asarray(x, np.float32).astype(np.float64) - ref) / rms
    return e.max(), np.quantile(e, 0.99)


def _bits(x):
    return np.asarray(x, np.float32).view(np.uint32)


def _dcp_served(step, rank, world=2):
    q, gate, blocks, lengths, tables, pool, starts, _ = step
    return bsp.block_sparse_gqa_attention(
        q, gate, blocks, lengths, tables[rank], pool, starts, kv_heads=1, ratio=RATIO,
        query_block=1024, kv_tile=PAGE, query_tile=256,
        dcp=(jnp.asarray([rank], jnp.int32), world), return_lse=True,
    )


def _dcp_walk(step, rank, world=2, **options):
    q, gate, blocks, lengths, tables, pool, starts, _ = step
    params = dict(group=128, kv_tile=1024, union_cap=CAP, min_context=0, lean=True)
    params.update(options)
    out, lse, cache = ud.union_walk_prefill(
        q, gate, blocks, lengths, tables[rank], pool, starts, kv_heads=1, ratio=RATIO,
        dcp=(jnp.asarray([rank], jnp.int32), world), return_lse=True,
        served=lambda: _dcp_served(step, rank, world), **params,
    )
    assert cache is pool
    return np.asarray(out), np.asarray(lse)


def test_offset_walk_reads_the_second_layer(kernels):
    step = _step(23, heads=3)
    q, gate, blocks, lengths, tables, pool, starts, _ = step
    query_block, kv_tile, query_tile = get_sparse_prefill_tiles(
        q, pool, kv_heads=1, ratio=RATIO, with_query_tile=True
    )
    served, _ = bsp.block_sparse_gqa_attention(
        q, gate, blocks, lengths, tables[0], pool, starts, kv_heads=1, ratio=RATIO,
        query_block=query_block, kv_tile=kv_tile, kv_head_offset=1,
        **({"query_tile": query_tile} if query_tile else {}),
    )
    jax.clear_caches()

    def walk(offset):
        out, cache = ud.union_walk_prefill(
            q, gate, blocks, lengths, tables[0], pool, starts, kv_heads=1, ratio=RATIO,
            group=128, kv_tile=1024, union_cap=CAP, min_context=0, lean=True,
            kv_head_offset=offset,
        )
        assert cache is pool
        return np.asarray(out)

    got = walk(1)
    assert not got[int(starts[-1]):].astype(np.float32).any()
    rows_sel = _rows(step)
    ref = _gated(step, rows_sel, _reference(step, rows_sel, offset=1)[0])
    served_max, served_p99 = _err(np.asarray(served)[rows_sel], ref)
    got_max, got_p99 = _err(got[rows_sel], ref)
    print(f"OFFSET1 served max/p99 {served_max:.3e}/{served_p99:.3e} walk {got_max:.3e}/{got_p99:.3e}")
    assert served_max < 0.1, served_max
    assert got_max <= 2 * served_max and got_p99 <= 2 * served_p99, (
        got_max, got_p99, served_max, served_p99
    )
    # Positive control: the first layer's slots are far from the second's reference.
    jax.clear_caches()
    first_max, _ = _err(walk(0)[rows_sel], ref)
    assert first_max > 10 * served_max, first_max


@pytest.mark.parametrize("rank", [0, 1])
def test_dcp_walk_owned_keys_and_lse(kernels, rank):
    step = _step(29, heads=6, world=2)
    served_out, served_lse, _ = _dcp_served(step, rank)
    served_out, served_lse = np.asarray(served_out), np.asarray(served_lse)
    jax.clear_caches()
    got_out, got_lse = _dcp_walk(step, rank)
    starts = np.asarray(step[6])
    assert not got_out[starts[-1]:].any() and np.isneginf(got_lse[starts[-1]:]).all()
    rows_sel = _rows(step)
    ref_out, ref_lse = _reference(step, rows_sel, world=2, rank=rank)
    live = np.isfinite(ref_lse[:, 0])
    # Rows with no owned key: out 0 and LSE -inf in both kernels.
    assert np.array_equal(np.isfinite(got_lse[rows_sel]), np.isfinite(ref_lse))
    assert not got_out[rows_sel][~live].any()
    served_max, served_p99 = _err(served_out[rows_sel][live], ref_out[live])
    got_max, got_p99 = _err(got_out[rows_sel][live], ref_out[live])
    served_lse_err = np.abs(served_lse[rows_sel][live] - ref_lse[live]).max()
    got_lse_err = np.abs(got_lse[rows_sel][live] - ref_lse[live]).max()
    print(f"DCP rank{rank} out served {served_max:.3e}/{served_p99:.3e} walk {got_max:.3e}/{got_p99:.3e} "
          f"lse served {served_lse_err:.3e} walk {got_lse_err:.3e}")
    assert served_max < 0.05, served_max
    assert got_max <= 2 * served_max + 1e-6 and got_p99 <= 2 * served_p99 + 1e-6
    assert got_lse_err <= 2 * served_lse_err + 1e-5, (got_lse_err, served_lse_err)


def test_dcp_ranks_combine_to_full_attention(kernels):
    step = _step(31, heads=6, world=2)
    rows_sel = _rows(step)
    gate = step[1]
    outs, lses, served_outs, served_lses = [], [], [], []
    for rank in (0, 1):
        jax.clear_caches()
        out, lse = _dcp_walk(step, rank)
        outs.append(out)
        lses.append(lse)
        jax.clear_caches()
        s_out, s_lse, _ = _dcp_served(step, rank)
        served_outs.append(np.asarray(s_out))
        served_lses.append(np.asarray(s_lse))
    got = np.asarray(qsa_rpa.dcp_combine_own(jnp.asarray(np.stack(outs)), jnp.asarray(np.stack(lses)), gate))
    served = np.asarray(qsa_rpa.dcp_combine_own(
        jnp.asarray(np.stack(served_outs)), jnp.asarray(np.stack(served_lses)), gate))
    ref = _gated(step, rows_sel, _reference(step, rows_sel)[0])
    served_max, served_p99 = _err(served[rows_sel], ref)
    got_max, got_p99 = _err(got[rows_sel], ref)
    print(f"DCP combined served {served_max:.3e}/{served_p99:.3e} walk {got_max:.3e}/{got_p99:.3e}")
    assert served_max < 0.1, served_max
    assert got_max <= 2 * served_max and got_p99 <= 2 * served_p99


def test_dcp_cap_fallback_is_the_served_call(kernels):
    step = _step(37, heads=6, world=2)
    served_out, served_lse, _ = _dcp_served(step, 1)
    jax.clear_caches()
    got_out, got_lse = _dcp_walk(step, 1, union_cap=32)
    np.testing.assert_array_equal(_bits(got_out), _bits(served_out))
    np.testing.assert_array_equal(_bits(got_lse), _bits(served_lse))


def test_dispatch_needs_the_sub_flags(kernels, monkeypatch):
    calls = []
    original = ud.union_walk_prefill

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(ud, "union_walk_prefill", spy)
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_UNION_WALK", "128")
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT", "0")
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_LEAN", "1")
    off_step = _step(41, heads=3)
    q, gate, blocks, lengths, tables, pool, starts, _ = off_step

    def offset_layer():
        return qsa_rpa._prefill_attention(
            q, gate, blocks, lengths, tables[0], pool, starts, kv_heads=1, ratio=RATIO,
            kv_head_offset=1,
        )

    jax.eval_shape(offset_layer)
    assert calls == []
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_UNION_OFFSET", "1")
    jax.clear_caches()  # eval_shape caches the trace of the same callable
    jax.eval_shape(offset_layer)
    assert len(calls) == 1 and calls[0]["kv_head_offset"] == 1 and calls[0]["dcp"] is None

    calls.clear()
    dcp_step = _step(43, heads=6, world=2)
    q, gate, blocks, lengths, tables, pool, starts, _ = dcp_step

    def dcp_layer():
        return qsa_rpa._prefill_attention(
            q, gate, blocks, lengths, tables[1], pool, starts, kv_heads=1, ratio=RATIO,
            dcp=(jnp.asarray([1], jnp.int32), 2), return_lse=True,
        )

    jax.eval_shape(dcp_layer)
    assert calls == []
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_UNION_DCP", "1")
    jax.clear_caches()
    jax.eval_shape(dcp_layer)
    assert len(calls) == 1 and calls[0]["return_lse"] and calls[0]["dcp"][1] == 2
