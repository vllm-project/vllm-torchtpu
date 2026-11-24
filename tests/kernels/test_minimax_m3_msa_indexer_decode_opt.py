"""TPU_MINIMAX_M3_INDEXER_DECODE_OPT items noclamp / full / counts / tail<N> /
defer / lanes: the decode (token-major, transposed) block_scores tiles give
the same bits as the default kernel on a folded 4-bit index cache (Pallas
interpret mode)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa, msa_fp4, msa_indexer

S, D = 128, 128


def _case(seed, ctx, max_num_seqs, pages_per_seq):
    rng = np.random.default_rng(seed)
    num = len(ctx)
    pool = max_num_seqs * pages_per_seq + 8
    bt = rng.permutation(pool)[: max_num_seqs * pages_per_seq].astype(np.int32)
    seq_lens = np.zeros(max_num_seqs, np.int32)
    seq_lens[:num] = ctx
    cu = np.zeros(max_num_seqs + 1, np.int32)
    cu[1 : num + 1] = np.cumsum([8] * num)
    cu[num + 1 :] = cu[num]
    key = jax.random.PRNGKey(seed)
    nib = jax.random.randint(key, (pool, S, D), 0, 16, jnp.int32).astype(jnp.uint8)
    scl = jax.random.uniform(
        jax.random.fold_in(key, 1), (pool, D), jnp.float32, 0.01, 0.03
    )
    cache = msa_fp4.pack_rows(nib, scl)
    q = jax.random.normal(jax.random.fold_in(key, 2), (8 * num, 1, D), jnp.float32)
    args = (
        q.astype(jnp.bfloat16),
        cache,
        jnp.asarray(seq_lens),
        jnp.asarray(bt),
        jnp.asarray(cu),
        jnp.asarray(num, jnp.int32),
    )
    return args, cu


def _run(monkeypatch, spec, args, lower=False):
    monkeypatch.setenv("TPU_MINIMAX_M3_INDEXER_DECODE_OPT", spec)
    jax.clear_caches()
    opts = msa.block_score_options(1)
    f = jax.jit(
        functools.partial(
            msa_indexer.block_scores.__wrapped__, interpret=True, fp8_q=False, **opts
        )
    )
    if lower:
        return f.lower(*args).as_text()
    return np.asarray(f(*args), np.float32)


def _valid(out, cu, ctx):
    return [out[cu[i] : cu[i + 1], 0, : -(-int(c) // S)] for i, c in enumerate(ctx)]


# Requests: 1 to 4 page groups of 128 pages, last groups with 1..4 live
# chunks of 32 pages; pages_per_seq not a multiple of 128 (the clamp case).
CTX = [128 * 300 + 5, 128 * 150, 128 * 33 + 1, 128 * 130, 128 * 470 + 77]


@pytest.mark.parametrize(
    "spec",
    [
        "noclamp",
        "noclamp,full",
        "noclamp,full,counts",
        "full,counts,tail16",
        "noclamp,tail8",
        "noclamp,defer",
        "defer,counts",
        "noclamp,lanes",
        "noclamp,lanes,defer",
        "noclamp,lanes,full",
        "lanes,defer,counts",
    ],
)
def test_decode_opt_same_bits(monkeypatch, spec):
    args, cu = _case(3, CTX, 6, 471)
    base = _valid(_run(monkeypatch, "", args), cu, CTX)
    got = _valid(_run(monkeypatch, spec, args), cu, CTX)
    for b, g in zip(base, got):
        assert np.isfinite(b).all()
        np.testing.assert_array_equal(g, b)
    # The item reaches the kernel: its lowering differs from the default's.
    assert _run(monkeypatch, spec, args, lower=True) != _run(
        monkeypatch, "", args, lower=True
    )


def test_decode_opt_parse(monkeypatch):
    monkeypatch.setenv(
        "TPU_MINIMAX_M3_INDEXER_DECODE_OPT", "noclamp,full,counts,tail16"
    )
    o = msa_indexer.decode_opt()
    assert (o.noclamp, o.full, o.every_count, o.tail) == (True, True, True, 16)
    monkeypatch.setenv("TPU_MINIMAX_M3_INDEXER_DECODE_OPT", "tail5")
    with pytest.raises(ValueError):
        msa_indexer.decode_opt()
