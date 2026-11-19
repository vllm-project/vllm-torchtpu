# SPDX-License-Identifier: Apache-2.0
"""q38 L2-META D26 (selection port of MM 9f3e367d2): ``select_empty_pass`` builds
the StreamIndex pass schedule with a select instead of ``lax.cond`` and must give
the same schedule, leaf for leaf and bit for bit, for empty and non-empty passes
(decode passes with 4 / 8 sequences per tile, mixed passes, chunked schedules).
The selection-level test (test_indexer_step_flags_keep_the_selection) cannot see
schedule entries past ``num_steps``; this one does.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import importlib

md = importlib.import_module("vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.metadata")


def _inputs(seed, seqs=64, tokens=64):
    rng = np.random.default_rng(seed)
    qlens = np.zeros(seqs, np.int32)
    live = rng.integers(4, 17)
    qlens[:live] = rng.integers(1, 5, live)
    qlens[: max(1, live // 3)] = 1
    cu = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)
    assert cu[-1] <= tokens
    seq_lens = np.where(qlens > 0, qlens + rng.integers(0, 60000, seqs), 0).astype(np.int32)
    return jnp.asarray(seq_lens), jnp.asarray(cu), live


PASSES = {
    # (static_q_len, seq_batch_size, bq_sz, bkv_sz)
    "decode_b4": (1, 4, 1, 1024),
    "decode_b8": (1, 8, 1, 1024),
    "mixed_q4": (None, 1, 4, 3328),
    "mixed_q4_kv6656": (None, 1, 4, 6656),
}


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("pass_kind", sorted(PASSES))
@pytest.mark.parametrize("window", ["empty", "head", "tail", "all", "chunked"])
def test_select_matches_cond(seed, pass_kind, window):
    seq_lens, cu, live = _inputs(seed)
    static_q_len, batch, bq, bkv = PASSES[pass_kind]
    start, end = {"empty": (live, live), "head": (0, live // 2), "tail": (live // 2, live),
                  "all": (0, live), "chunked": (0, live)}[window]
    kw = dict(
        seq_lens=seq_lens,
        cu_q_lens=cu,
        start_seq_idx=jnp.int32(start),
        end_seq_idx=jnp.int32(end),
        bq_sz=bq,
        bkv_sz=bkv,
        pages_per_seq=512,
        page_size=128,
        max_num_tokens=64,
        compression_ratio=4,
        static_q_len=static_q_len,
        seq_batch_size=batch,
    )
    if window == "chunked":
        kw.update(chunk_token_start=(0, 32), chunk_tokens=32)
    want = md.compute_metadata(**kw, select_empty_pass=False)
    got = md.compute_metadata(**kw, select_empty_pass=True)
    w_leaves, w_tree = jax.tree.flatten(want)
    g_leaves, g_tree = jax.tree.flatten(got)
    assert w_tree == g_tree
    for i, (w, g) in enumerate(zip(w_leaves, g_leaves, strict=True)):
        w, g = np.asarray(w), np.asarray(g)
        assert w.dtype == g.dtype, i
        np.testing.assert_array_equal(g, w, err_msg=f"leaf {i}")
    if window == "empty":
        assert int(np.asarray(want.num_steps).sum()) == 0
    else:
        assert int(np.asarray(want.num_steps).sum()) > 0


def test_select_has_no_cond_in_the_lowering():
    """Engagement: the select form lowers without the schedule's `case`."""
    seq_lens, cu, live = _inputs(0)

    def f(flag, s, e):
        return jax.tree.leaves(md.compute_metadata(
            seq_lens=seq_lens, cu_q_lens=cu, start_seq_idx=s, end_seq_idx=e, bq_sz=4,
            bkv_sz=3328, pages_per_seq=512, page_size=128, max_num_tokens=64,
            compression_ratio=4, static_q_len=None, seq_batch_size=1, select_empty_pass=flag))

    args = (jnp.int32(0), jnp.int32(live))
    off = jax.jit(lambda s, e: f(False, s, e)).lower(*args).as_text()
    on = jax.jit(lambda s, e: f(True, s, e)).lower(*args).as_text()
    print("Q38_L2META_SEL_LOWERED case off=%d on=%d" % (off.count("stablehlo.case"), on.count("stablehlo.case")))
    assert off.count("stablehlo.case") == 1 and on.count("stablehlo.case") == 0
