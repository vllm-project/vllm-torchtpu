"""TPU_MINIMAX_M3_INDEX_FOLD_OPT gather / compose / lookup: `msa.partial_block_scores`
folds the index heads by a gather and unfolds the scores inside
`block_scores`' own layout gather, with the same bits as the scatter and the
second row gather (Pallas interpret mode, 4-bit folded index cache)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa, msa_fp4

S, D, H = 128, 128, 2


def _case(seed, ctx, q_lens, num_tokens, max_num_seqs, pages_per_seq):
    rng = np.random.default_rng(seed)
    num = len(ctx)
    pool = max_num_seqs * pages_per_seq + 8
    bt = rng.permutation(pool)[: max_num_seqs * pages_per_seq].astype(np.int32)
    seq_lens = np.zeros(max_num_seqs, np.int32)
    seq_lens[:num] = ctx
    cu = np.zeros(max_num_seqs + 1, np.int32)
    cu[1 : num + 1] = np.cumsum(q_lens)
    cu[num + 1 :] = cu[num]
    key = jax.random.PRNGKey(seed)
    nib = jax.random.randint(key, (pool, S, D), 0, 16, jnp.int32).astype(jnp.uint8)
    scl = jax.random.uniform(
        jax.random.fold_in(key, 1), (pool, D), jnp.float32, 0.01, 0.03
    )
    cache = msa_fp4.pack_rows(nib, scl)
    q = jax.random.normal(jax.random.fold_in(key, 2), (num_tokens, H, D), jnp.float32)
    args = (
        q.astype(jnp.bfloat16),
        cache,
        jnp.asarray(seq_lens),
        jnp.asarray(bt),
        jnp.asarray(cu),
        jnp.asarray(num, jnp.int32),
    )
    return args, cu


def _run(monkeypatch, opt, args):
    monkeypatch.setenv("TPU_MINIMAX_M3_INDEX_FOLD_OPT", opt)
    jax.clear_caches()
    f = jax.jit(functools.partial(msa.partial_block_scores, shards=1, interpret=True))
    return np.asarray(f(*args), np.float32)


CASES = [
    # decode: 4 requests of 8 rows, 8 padding tokens
    ([128 * 40 + 5, 128 * 15, 128 * 33 + 1, 128 * 13], [8, 8, 8, 8], 40),
    # mixed lengths incl. an empty request, no padding
    ([128 * 20 + 3, 128 * 7, 128 * 9, 128 * 11 + 9], [8, 3, 0, 13], 24),
]


@pytest.mark.parametrize(
    "opt", ["gather", "compose", "gather+compose", "gather+compose+lookup", "lookup"]
)
@pytest.mark.parametrize("case", range(len(CASES)))
def test_index_fold_opt_same_bits(monkeypatch, opt, case):
    ctx, q_lens, num_tokens = CASES[case]
    args, cu = _case(5 + case, ctx, q_lens, num_tokens, 6, 41)
    base = _run(monkeypatch, "", args)
    got = _run(monkeypatch, opt, args)
    assert got.shape == base.shape
    for i, c in enumerate(ctx):
        pages = -(-c // S)
        b = base[:, cu[i] : cu[i + 1], :pages]
        assert np.isfinite(b).all()
        np.testing.assert_array_equal(got[:, cu[i] : cu[i + 1], :pages], b)


@pytest.mark.parametrize("case", range(len(CASES)))
def test_fold_gather_same_rows(monkeypatch, case):
    ctx, q_lens, num_tokens = CASES[case]
    args, _ = _case(9, ctx, q_lens, num_tokens, 6, 41)
    q, _, seq_lens, _, cu, _ = args
    out = {}
    opts = ("", "gather", "lookup", "gather+lookup")
    for opt in opts:
        monkeypatch.setenv("TPU_MINIMAX_M3_INDEX_FOLD_OPT", opt)
        out[opt] = [np.asarray(x) for x in msa.fold_index_heads(q, seq_lens, cu)]
    for opt in opts[1:]:
        for a, b in zip(out[""], out[opt]):
            np.testing.assert_array_equal(a, b)
