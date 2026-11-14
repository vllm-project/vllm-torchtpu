"""MiniMax M3 MSA prefill attention: per-call tile configuration
(``TPU_MINIMAX_M3_MSA_PREFILL_ADAPTIVE``).

Each call must give the bits of the configuration it picks, and pick by the
mean distinct pages per query tile of its selections."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_attention_lse as lse_mod
from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_fp4

S, D, HG, TOPK = 128, 128, 16, 16
PAGES = 64  # pages per sequence; the call's tokens sit in the last one


def _wide_reference(sel, block_q, threshold, max_tiles=64):
    t, g, k = sel.shape
    pad = -t % block_q
    sel = np.concatenate([sel, np.full((pad, g, k), -1, sel.dtype)])
    stride = max(1, (t + pad) // block_q // max_tiles)
    distinct = tiles = 0
    for h in range(g):
        for t0 in range(0, t + pad, block_q * stride):
            keys = sel[t0 : t0 + block_q, h].reshape(-1)
            keys = keys[keys >= 0]
            distinct += len(np.unique(keys))
            tiles += int(keys.size > 0)
    return distinct > threshold * tiles


@pytest.mark.parametrize("seed", range(4))
def test_wide_selection_matches_reference(seed):
    rng = np.random.default_rng(seed)
    t = int(rng.integers(20, 90))
    sel = rng.integers(-1, 30, size=(t, 2, 8)).astype(np.int32)
    sel[: t // 3] = -1  # empty tiles do not count
    for block_q in (8, 16):
        for threshold in (4, 10, 16, 24):
            got = bool(lse_mod.wide_selection(jnp.asarray(sel), block_q, threshold))
            assert got == _wide_reference(sel, block_q, threshold)
            for max_tiles in (2, 3):  # sampled tiles
                got = lse_mod.wide_selection(jnp.asarray(sel), block_q, threshold, max_tiles)
                assert bool(got) == _wide_reference(sel, block_q, threshold, max_tiles)


def _inputs(num_heads, union, seed):
    """One sequence of 128 prefill tokens at the end of a 64-page context;
    each 16-token tile picks its tokens' 16 blocks from `union` blocks."""
    rng = np.random.default_rng(seed)
    t = 128
    blocks = np.zeros((t, num_heads, TOPK), np.int32)
    for t0 in range(0, t, 16):
        for h in range(num_heads):
            pool = rng.choice(PAGES - 1, size=union, replace=False)
            for i in range(t0, t0 + 16):
                blocks[i, h] = rng.choice(pool, size=TOPK, replace=False)
    scores = -np.sort(-rng.standard_normal((t, num_heads, TOPK)), axis=-1)
    cand = np.concatenate([scores, blocks.astype(np.float64)], axis=-1)
    cand = jnp.asarray(cand.reshape(1, t, -1), jnp.float32)
    q = jnp.asarray(rng.standard_normal((t, num_heads * HG, D)) * 0.5, jnp.bfloat16)
    nib = jnp.asarray(
        rng.integers(0, 16, size=(PAGES, S, num_heads * 2 * D)), jnp.uint8
    )
    kv = msa_fp4.pack_rows(nib)
    sc = jnp.asarray(
        rng.uniform(0.01, 0.03, size=(PAGES, msa_fp4.scale_rows(num_heads), S)),
        jnp.float32,
    )
    seq_lens = jnp.asarray([PAGES * S], jnp.int32)
    block_tables = jnp.asarray(rng.permutation(PAGES)[None], jnp.int32)
    cu_q_lens = jnp.asarray([0, t], jnp.int32)
    return q, kv, cand, seq_lens, block_tables, cu_q_lens, sc


def _attend(args):
    q, kv, cand, seq_lens, block_tables, cu_q_lens, sc = args
    out, lse = msa_dcp.msa_attend_dcp(
        q, kv, cand, seq_lens, block_tables, cu_q_lens,
        sm_scale=1 / np.sqrt(D), topk=TOPK, rank=0, dcp=1, scale_cache=sc,
        interpret=True,
    )
    return np.asarray(out.astype(jnp.float32)), np.asarray(lse)


@pytest.fixture
def prefill_config(monkeypatch):
    monkeypatch.setattr(msa_dcp, "_MIN_CANDIDATE_TOKENS", 64)
    monkeypatch.setattr(msa_dcp, "DCP_SMALL_BATCH_TOKENS", 64)
    monkeypatch.setattr(lse_mod, "PREFILL_PAGE_CACHE_SLOTS", 512)
    monkeypatch.setattr(lse_mod, "PREFILL_PAGES_PER_STEP", 32)
    monkeypatch.setattr(lse_mod, "PREFILL_STEP_WIDTHS", (2, 4, 8, 12, 16, 20, 24))
    monkeypatch.setattr(lse_mod, "PREFILL_BLOCK_Q", 0)
    monkeypatch.setattr(lse_mod, "PAGE_CACHE_TILE_PAGES", 64)
    monkeypatch.delenv(lse_mod.ADAPTIVE_ENV, raising=False)
    return monkeypatch


def _record_runs(monkeypatch):
    """Each kernel configuration that runs (not merely traced): the taken
    branch of a `lax.cond` alone executes its callback. On CPU the 8- and
    16-token tiles give the same bits, so the outputs cannot tell."""
    ran = []
    for name in ("msa_sparse_attention_lse", "msa_sparse_attention_lse_heads"):
        real = getattr(lse_mod, name)

        def spy(*args, _real=real, **kw):
            tag = (kw.get("block_q"), kw.get("page_cache_tile_pages", 0))
            jax.debug.callback(lambda: ran.append(tag))
            return _real(*args, **kw)

        monkeypatch.setattr(lse_mod, name, spy)
    return ran


@pytest.mark.parametrize("num_heads", [1, 2])
def test_adaptive_gives_the_picked_configuration(prefill_config, num_heads):
    narrow = _inputs(num_heads, union=16, seed=1)
    wide = _inputs(num_heads, union=40, seed=2)
    # Fixed configurations: 16-token tiles / 64 cached pages, and 8-token
    # tiles / 32 cached pages.
    jax.clear_caches()
    base = [_attend(narrow), _attend(wide)]
    prefill_config.setattr(lse_mod, "PREFILL_BLOCK_Q", 8)
    prefill_config.setattr(lse_mod, "PAGE_CACHE_TILE_PAGES", 32)
    jax.clear_caches()
    small = [_attend(narrow), _attend(wide)]
    prefill_config.setattr(lse_mod, "PREFILL_BLOCK_Q", 0)
    prefill_config.setattr(lse_mod, "PAGE_CACHE_TILE_PAGES", 64)

    prefill_config.setenv(lse_mod.ADAPTIVE_ENV, "28:8:32")
    ran = _record_runs(prefill_config)
    jax.clear_caches()
    got = [_attend(narrow), _attend(wide)]
    jax.effects_barrier()
    assert ran == [(16, 0), (8, 32)]
    for (o, l), (eo, el) in zip(got, [base[0], small[1]]):
        np.testing.assert_array_equal(o, eo)
        np.testing.assert_array_equal(l, el)


def test_adaptive_off_runs_one_configuration(prefill_config):
    ran = _record_runs(prefill_config)
    jax.clear_caches()
    _attend(_inputs(1, union=40, seed=2))
    jax.effects_barrier()
    assert ran == [(16, 0)]


def test_adaptive_spec_parsing(monkeypatch):
    monkeypatch.setenv(lse_mod.ADAPTIVE_ENV, "30:8:64")
    assert lse_mod.adaptive_prefill() == (30, 8, 64)
    monkeypatch.setenv(lse_mod.ADAPTIVE_ENV, "")
    assert lse_mod.adaptive_prefill() is None
    monkeypatch.setenv(lse_mod.ADAPTIVE_ENV, "30:8")
    with pytest.raises(ValueError):
        lse_mod.adaptive_prefill()
