"""TPU_MINIMAX_M3_INDEXER_PREFILL_OPT items causal / chunk<N>: the DCP prefill
candidates (`msa_dcp._class_candidates`: large-class block_scores, then each
rank's top-k) are the same bits as with the option off, on a folded 4-bit
index cache with folded index heads (Pallas interpret mode)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa, msa_dcp, msa_fp4, msa_indexer

S, D, BS, G, DCP = 128, 128, 128, 4, 2
MAX_SEQS, PAGES_PER_SEQ = 4, 32


def _case(seed, reqs, rank):
    rng = np.random.default_rng(seed)
    n = len(reqs)
    seq_lens = np.zeros(MAX_SEQS, np.int32)
    q_lens = np.zeros(MAX_SEQS, np.int32)
    for i, (c, l) in enumerate(reqs):
        seq_lens[i], q_lens[i] = c + l, l
    cu = np.zeros(MAX_SEQS + 1, np.int32)
    cu[1:] = np.cumsum(q_lens)
    num_tokens = int(cu[n])
    pool = MAX_SEQS * PAGES_PER_SEQ + 8
    bt = rng.permutation(pool)[: MAX_SEQS * PAGES_PER_SEQ].astype(np.int32)
    key = jax.random.PRNGKey(seed)
    nib = jax.random.randint(key, (pool, S, D), 0, 16, jnp.int32).astype(jnp.uint8)
    scl = jax.random.uniform(jax.random.fold_in(key, 1), (pool, D), jnp.float32, 0.01, 0.03)
    cache = msa_fp4.pack_rows(nib, scl)
    q = jax.random.normal(jax.random.fold_in(key, 2), (num_tokens, G, D), jnp.float32)
    sl, cuj = jnp.asarray(seq_lens), jnp.asarray(cu)
    local = msa_dcp.local_seq_lens(sl, block_size=BS, rank=rank, dcp=DCP)
    seq_id, _, _ = msa.token_positions(num_tokens, sl, cuj)
    return (
        q.astype(jnp.bfloat16),
        cache,
        sl,
        local,
        jnp.asarray(bt),
        cuj,
        jnp.asarray(n, jnp.int32),
        seq_id,
    )


def _run(monkeypatch, spec, args, rank):
    monkeypatch.setenv("TPU_MINIMAX_M3_INDEXER_PREFILL_OPT", spec)
    jax.clear_caches()
    cand = functools.partial(
        msa_dcp.local_candidates,
        block_size=BS,
        topk=4,
        init_blocks=1,
        local_blocks=1,
        rank=rank,
        dcp=DCP,
        interpret=True,
    )
    f = jax.jit(
        lambda *a: msa_dcp._class_candidates(
            *a,
            cand,
            interpret=True,
            scale_cache=None,
            scale_row=None,
            block_size=BS,
            rank=jnp.asarray(rank, jnp.int32),
            dcp=DCP,
        )
    )
    return np.asarray(f(*args))


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("spec", ["causal", "causal+chunk32", "chunk32", "causal+counts"])
def test_prefill_opt_same_candidates(monkeypatch, spec, rank):
    # A chunk on top of a context (its early tiles see far fewer pages than
    # its late ones) and a fresh request whose tiles straddle head boundaries.
    args = _case(0, [(2944, 1100), (0, 300)], rank)
    base = _run(monkeypatch, "", args, rank)
    arm = _run(monkeypatch, spec, args, rank)
    np.testing.assert_array_equal(arm, base)


def test_prefill_opt_parse(monkeypatch):
    monkeypatch.setenv("TPU_MINIMAX_M3_INDEXER_PREFILL_OPT", "causal+chunk32")
    assert msa_indexer.prefill_opt() == msa_indexer.PrefillOpt(causal=True, chunk=32)
    monkeypatch.setenv("TPU_MINIMAX_M3_INDEXER_PREFILL_OPT", "bogus")
    with pytest.raises(ValueError):
        msa_indexer.prefill_opt()
