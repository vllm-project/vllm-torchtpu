# SPDX-License-Identifier: Apache-2.0
"""q38 PSTEP-IDX: VLLM_TPU_QSA_PREFILL_INDEXER_NO_COND against the ragged prefill indexer's early-exit cond.

Today the ragged prefill indexer (pooled_qsa ``_select_blocks`` prefill branch) calls StreamIndex with
``enable_early_exit=True``: one ``lax.cond`` per QSA layer (``max(seq_lens) // ratio <= k``) that lists every
visible group on a fast path instead of scoring. The cond is opaque to XLA's scheduler, so the layer's
selection-independent ops cannot run beside the SparseCore top-k waits (M, j-1545b7bc: 662-775 us of
TensorCore wait per QSA layer at 6,400 rows). With the flag, StreamIndex runs ``early_exit_select``: the
scoring path always runs and the fast-path rows are taken by a select on the same predicate.

T70g fix (PSTEP-IDX3): the select engages only with the layer's own scheduling-group range (``indexer_group_base``,
passed by the flag-on op ``pooled_qsa_nocond_step``), because without the cond every QSA layer's chunk groups share
one computation and a layer that hits a trace cache would reuse an earlier layer's ids ('annotation groups with
gaps' on the device, T70g j-e61e85e4). The two-layer compile itself is proved by the deviceless tpu7x AOT harness.

The real StreamIndex kernel runs in TPU interpret mode on CPU with the JAX reference top-k standing in for
the SparseCore selector (test_qwen4_exp_shared_stream_indexer's harness). Rule 23: excess precision is pinned
off in this file (both arms run the same Pallas bodies inside different program structure), and a one-ulp
score change at a tie boundary must read NOT identical under the same flags.
"""

import functools
import os

os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_allow_excess_precision=false"
).strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from test_qwen4_exp_shared_stream_indexer import (  # noqa: E402,F401 (interpret is a fixture)
    BLOCK_TOPK,
    RATIO,
    ReferenceTopK,
    inputs,
    interpret,
    streamindex,
)

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import config as sconfig  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa  # noqa: E402

FLAG = "VLLM_TPU_QSA_PREFILL_INDEXER_NO_COND"
BASE = sconfig.LAYER_SCHEDULING_GROUP_FIRST + 3 * sconfig.LAYER_SCHEDULING_GROUP_STRIDE
_MISSING = object()


def test_excess_precision_is_pinned_off():
    assert "--xla_allow_excess_precision=false" in os.environ["XLA_FLAGS"]


@pytest.fixture
def envset(monkeypatch):
    """os.environ (monkeypatch) AND the envs module attribute, restored on teardown (L2's leak-free pattern)."""
    saved = {}

    def set_(name, value):
        if name not in saved:
            saved[name] = envs.__dict__.get(name, _MISSING)
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, str(value))
        envs.__dict__[name] = envs.environment_variables[name]()

    yield set_
    for name, old in saved.items():
        if old is _MISSING:
            envs.__dict__.pop(name, None)
        else:
            envs.__dict__[name] = old


# (q_lens per request slot, context per slot, tokens, block_topk, chunk). Prefill: tokens > request slots.
# Harness budget 16 groups = 64 tokens (fast path when every sequence <= 67 tokens); production 512 = 2048.
CASES = {
    # Every sequence inside the harness budget: the cond takes the FAST path today.
    "short_k16": ([24, 8, 0], [10, 30, 0], 32, BLOCK_TOPK, 0),
    "short_k16_chunk8": ([24, 8, 0], [10, 30, 0], 32, BLOCK_TOPK, 8),
    # Exactly at the budget (64 // 4 = 16 = k): still the fast path (the predicate is <=).
    "exact_k16": ([24, 8, 0], [40, 30, 0], 32, BLOCK_TOPK, 0),
    # One sequence just past it: the scoring path today.
    "edge_k16": ([24, 8, 0], [44, 30, 0], 32, BLOCK_TOPK, 0),
    # Served-like: long histories, several table columns, an idle slot; unchunked and chunked (the 640 form).
    "long_k16": ([24, 0, 8], [600, 0, 3000], 32, BLOCK_TOPK, 0),
    "long_k16_chunk8": ([24, 0, 8], [600, 0, 3000], 32, BLOCK_TOPK, 8),
    "long_k16_chunk16": ([20, 12, 0], [1203, 2999, 0], 32, BLOCK_TOPK, 16),
    # Production budget, every sequence inside it (fast path) and one past it (scoring path).
    "short_k512": ([24, 8, 0], [1500, 2000, 0], 32, 512, 0),
    "edge_k512": ([24, 8, 0], [2030, 2000, 0], 32, 512, 8),
}


def _spy(monkeypatch, backend=None):
    seen = []
    real = functools.partial(streamindex.streamindex_topk, topk_backend=backend or ReferenceTopK())

    def spy(*args, **kwargs):
        seen.append((kwargs.get("enable_early_exit"), kwargs.get("early_exit_select", False)))
        assert kwargs.get("scheduling_group_base") == (BASE if seen[-1][1] else None), kwargs
        return real(*args, **kwargs)

    monkeypatch.setattr(pooled_qsa, "streamindex_topk", spy)
    return seen


def _select(args, k, base=None):
    extra = {} if base is None else {"indexer_group_base": base}
    return np.asarray(pooled_qsa.select_blocks(**args, ratio=RATIO, block_topk=k, **extra))


def _visible(args):
    """Fast-path ground truth: every causally visible group, ascending (lengths are per-token causal lengths)."""
    lengths = np.asarray(args["lengths"])
    return [list(range(max(0, int(n) // RATIO))) for n in lengths]


@pytest.mark.parametrize("case", sorted(CASES))
def test_no_cond_is_bit_identical_to_the_cond(monkeypatch, interpret, envset, case):
    q_lens, context, tokens, k, chunk = CASES[case]
    args = inputs(len(case), q_lens, context, tokens, "integer")
    assert tokens > len(context), "the ragged prefill path"
    envset("VLLM_TPU_QSA_PREFILL_INDEXER_CHUNK", chunk)
    envset("VLLM_TPU_QSA_PREFILL_SC_BOUNDED_COPY", None)
    seen = _spy(monkeypatch)
    envset(FLAG, None)
    off = _select(args, k)
    assert seen and all(s == (True, False) for s in seen), seen
    seen.clear()
    envset(FLAG, 1)
    on = _select(args, k, BASE)
    assert seen and all(s == (False, True) for s in seen), seen
    np.testing.assert_array_equal(on, off)
    fast = int(np.max(np.asarray(args["context"]) + np.asarray(q_lens))) // RATIO <= k
    if fast:
        for row, want in zip(on, _visible(args)):
            np.testing.assert_array_equal(row[: len(want)], want)
            assert np.all(row[len(want):] == -1)
    print(f"Q38_PSTEP_IDX_NOCOND case={case} fast_path={fast} rows={len(on)} k={k} chunk={chunk} identical=1")


def test_default_is_off():
    assert envs.environment_variables[FLAG]() is False


def test_flag_off_passes_no_new_kwarg(monkeypatch, envset):
    """Flag off: the call is exactly today's (enable_early_exit=True, no early_exit_select key)."""
    seen = []

    class Stop(Exception):
        pass

    def capture(*args, **kwargs):
        seen.append(dict(kwargs))
        raise Stop

    monkeypatch.setattr(pooled_qsa, "streamindex_topk", capture)
    envset(FLAG, None)
    for extra in ({}, {"indexer_group_base": BASE}):
        with pytest.raises(Stop):
            pooled_qsa.select_blocks(**inputs(0, [24, 0], [600, 0], 24, "integer"), ratio=RATIO,
                                     block_topk=BLOCK_TOPK, **extra)
    assert len(seen) == 2 and seen[0] == seen[1], seen
    assert seen[0]["enable_early_exit"] is True and "early_exit_select" not in seen[0]
    assert "scheduling_group_base" not in seen[0]


def test_engagement_line(monkeypatch, envset, caplog):
    class Stop(Exception):
        pass

    def capture(*args, **kwargs):
        raise Stop

    monkeypatch.setattr(pooled_qsa, "streamindex_topk", capture)
    pooled_qsa._Q38_PREFILL_NO_COND_LOGGED.clear()
    lines = []
    monkeypatch.setattr(
        pooled_qsa, "_log_prefill_no_cond",
        lambda t, c, e, b: lines.append((t, c, e, b)),
    )
    envset(FLAG, 1)
    for extra in ({"indexer_group_base": BASE}, {}):
        with pytest.raises(Stop):
            pooled_qsa.select_blocks(**inputs(0, [24, 0], [600, 0], 24, "integer"), ratio=RATIO,
                                     block_topk=BLOCK_TOPK, **extra)
    assert lines == [(24, None, True, BASE), (24, None, False, None)]


def test_engagement_line_text_keeps_the_reader_form(caplog):
    """The locked engagement reader (q38_pidx_nceng.py ENG_RE, re.search) must still parse the line: the group base
    is appended after the closing parenthesis."""
    import logging
    import re

    eng_re = re.compile(r'q38 PSTEP-IDX prefill indexer no-cond (ENGAGED|NOT ENGAGED): tokens (\d+), chunk (\S+?) '
                        r'\(VLLM_TPU_QSA_PREFILL_INDEXER_NO_COND\)')
    pooled_qsa._Q38_PREFILL_NO_COND_LOGGED.clear()
    with caplog.at_level(logging.INFO):
        pooled_qsa._log_prefill_no_cond(6400, 640, True, BASE)
        pooled_qsa._log_prefill_no_cond(6400, 640, True, BASE + sconfig.LAYER_SCHEDULING_GROUP_STRIDE)
        pooled_qsa._log_prefill_no_cond(6400, 640, True, BASE)
    text = [r.getMessage() for r in caplog.records if "PSTEP-IDX prefill indexer no-cond" in r.getMessage()]
    assert len(text) == 2, text
    for t, b in zip(text, (BASE, BASE + sconfig.LAYER_SCHEDULING_GROUP_STRIDE)):
        m = eng_re.search(t)
        assert m and m.groups() == ("ENGAGED", "6400", "640"), t
        assert t.endswith(f" group base {b}"), t


def test_flag_on_without_a_group_base_keeps_the_cond(monkeypatch, interpret, envset):
    """A call without the layer's range (any op other than the no-cond op) keeps today's cond, bit for bit."""
    q_lens, context, tokens, k, chunk = CASES["long_k16_chunk8"]
    args = inputs(7, q_lens, context, tokens, "integer")
    envset("VLLM_TPU_QSA_PREFILL_INDEXER_CHUNK", chunk)
    envset("VLLM_TPU_QSA_PREFILL_SC_BOUNDED_COPY", None)
    seen = _spy(monkeypatch)
    envset(FLAG, None)
    off = _select(args, k)
    envset(FLAG, 1)
    on = _select(args, k)
    assert seen == [(True, False), (True, False)], seen
    np.testing.assert_array_equal(on, off)


class _GroupSpy(ReferenceTopK):
    """ReferenceTopK that overlaps scoring (so the chunk pipeline is annotated) and records the group of each top-k."""

    overlaps_scoring = True
    ids: list = []

    def select(self, score_bits, k, row_lengths, *, scheduling_group_id, return_scores):
        _GroupSpy.ids.append(scheduling_group_id)
        return super().select(score_bits, k, row_lengths, scheduling_group_id=scheduling_group_id,
                              return_scores=return_scores)

    def __hash__(self):
        return 2

    def __eq__(self, other):
        return isinstance(other, _GroupSpy)


def test_group_ids_come_from_the_layer_range(monkeypatch, interpret, envset):
    """Flag on with a base: chunk g's top-k carries group base + g (the last chunk none), the process counter is never
    touched, and two layers with their own ranges get disjoint ids even when the second call would hit the jit cache
    of the first (same shapes and statics apart from the base)."""
    q_lens, context, tokens, k, chunk = CASES["long_k16_chunk8"]
    args = inputs(9, q_lens, context, tokens, "integer")
    envset("VLLM_TPU_QSA_PREFILL_INDEXER_CHUNK", chunk)
    envset("VLLM_TPU_QSA_PREFILL_SC_BOUNDED_COPY", None)
    envset(FLAG, 1)
    real = functools.partial(streamindex.streamindex_topk, topk_backend=_GroupSpy())
    monkeypatch.setattr(pooled_qsa, "streamindex_topk", real)

    def no_counter(count):
        raise AssertionError("reserve_scheduling_group_ids called on the based path")

    monkeypatch.setattr(sconfig, "reserve_scheduling_group_ids", no_counter)
    chunks = tokens // chunk
    other = BASE + sconfig.LAYER_SCHEDULING_GROUP_STRIDE
    got = {}
    for base in (BASE, other):
        _GroupSpy.ids = []
        _select(args, k, base)
        got[base] = list(_GroupSpy.ids)
    assert got[BASE] == [BASE + g for g in range(chunks - 1)] + [None], got
    assert got[other] == [other + g for g in range(chunks - 1)] + [None], got
    assert not set(got[BASE]) - {None} & set(got[other]) - {None}
    print(f"Q38_PSTEP_IDX_GROUPS chunks={chunks} layer_a={got[BASE]} layer_b={got[other]}")


def test_scheduling_group_base_validation():
    args = dict(q=jnp.zeros((8, 4, 128), jnp.bfloat16), indexer_weights=jnp.zeros((8, 4), jnp.float32),
                cache_kv=jnp.zeros((4, 16, 2, 128), jnp.bfloat16), seq_lens=jnp.zeros((2,), jnp.int32),
                page_indices=jnp.zeros((8,), jnp.int32), cu_q_lens=jnp.zeros((3,), jnp.int32),
                distribution=jnp.zeros((3,), jnp.int32))
    for bad in (dict(scheduling_group_base=BASE), dict(scheduling_group_base=BASE, enable_early_exit=True),
                dict(scheduling_group_base=0, early_exit_select=True)):
        with pytest.raises(ValueError, match="scheduling_group_base|early_exit_select"):
            streamindex.streamindex_topk(**args, k=16, compression_ratio=4, num_kv_pages_per_block=1,
                                         num_queries_per_block=8, **bad)


def test_layer_range_is_checked_against_the_chunk_count(monkeypatch, interpret, envset):
    q_lens, context, tokens, k, chunk = CASES["long_k16_chunk8"]
    args = inputs(11, q_lens, context, tokens, "integer")
    envset("VLLM_TPU_QSA_PREFILL_INDEXER_CHUNK", chunk)
    envset("VLLM_TPU_QSA_PREFILL_SC_BOUNDED_COPY", None)
    envset(FLAG, 1)
    monkeypatch.setattr(pooled_qsa, "streamindex_topk",
                        functools.partial(streamindex.streamindex_topk, topk_backend=_GroupSpy()))
    monkeypatch.setattr(sconfig, "LAYER_SCHEDULING_GROUP_STRIDE", tokens // chunk - 2)
    with pytest.raises(ValueError, match="more than a layer's range"):
        _select(args, k, BASE + 1)


def test_layer_scheduling_group_base(monkeypatch):
    monkeypatch.setattr(sconfig, "_layer_scheduling_group_bases", {})
    a = sconfig.layer_scheduling_group_base("model.layers.6.self_attn")
    b = sconfig.layer_scheduling_group_base("model.layers.11.self_attn")
    assert a == sconfig.LAYER_SCHEDULING_GROUP_FIRST
    assert b == a + sconfig.LAYER_SCHEDULING_GROUP_STRIDE
    assert sconfig.layer_scheduling_group_base("model.layers.6.self_attn") == a
    sconfig.reset_scheduling_group_ids()  # a new program restarts the counter, never the layer ranges
    assert sconfig.layer_scheduling_group_base("model.layers.11.self_attn") == b
    assert sconfig.reserve_scheduling_group_ids(15) == 1 and a > 15


def test_flag_off_op_and_step_are_unchanged():
    """The flag-off torch op wraps pooled_qsa_step with today's signature (its schema and torch_tpu kernel key come
    from it); the no-cond op is a separate step with one extra static argument."""
    import inspect

    off = inspect.signature(pooled_qsa.pooled_qsa_step)
    on = inspect.signature(pooled_qsa.pooled_qsa_nocond_step)
    assert "indexer_group_base" not in off.parameters
    assert list(on.parameters) == list(off.parameters) + ["indexer_group_base"]
    assert on.parameters["indexer_group_base"].default == 0


def test_early_exit_select_validation():
    args = dict(q=jnp.zeros((8, 4, 128), jnp.bfloat16), indexer_weights=jnp.zeros((8, 4), jnp.float32),
                cache_kv=jnp.zeros((4, 16, 2, 128), jnp.bfloat16), seq_lens=jnp.zeros((2,), jnp.int32),
                page_indices=jnp.zeros((8,), jnp.int32), cu_q_lens=jnp.zeros((3,), jnp.int32),
                distribution=jnp.zeros((3,), jnp.int32))
    for bad in (dict(enable_early_exit=True), dict(return_scores=True)):
        with pytest.raises(ValueError, match="early_exit_select"):
            streamindex.streamindex_topk(**args, k=16, compression_ratio=4, num_kv_pages_per_block=1,
                                         num_queries_per_block=8, early_exit_select=True, **bad)


@pytest.mark.parametrize("case", ["long_k16", "long_k16_chunk8"])
def test_positive_control_one_ulp_score_change_is_not_identical(monkeypatch, interpret, envset, case):
    """Rule 23 (b): under the same pinned flags a one-ulp change of one score at a tie boundary must change the
    selection, so the identity above is not a comparator that passes regardless."""
    q_lens, context, tokens, k, chunk = CASES[case]
    args = inputs(len(case), q_lens, context, tokens, "integer")
    envset("VLLM_TPU_QSA_PREFILL_INDEXER_CHUNK", chunk)
    envset(FLAG, 1)
    _spy(monkeypatch)
    base = _select(args, k, BASE)

    class OneUlp(ReferenceTopK):
        """Adds one ulp to the first row's best-ranked loser that ties the k-th winner (integer data ties often)."""

        def select(self, score_bits, k, row_lengths, *, scheduling_group_id, return_scores):
            s = jax.lax.bitcast_convert_type(score_bits, jnp.float32)
            live = jnp.arange(s.shape[1])[None, :] < row_lengths[:, None]
            s = jnp.where(live, s, -jnp.inf)
            vals, ids = jax.lax.top_k(s, k + 1)
            r = jnp.argmax((vals[:, k - 1] == vals[:, k]) & (vals[:, k] > -jnp.inf))
            bumped = score_bits.at[r, ids[r, k]].add(jnp.where(vals[r, k - 1] == vals[r, k], 1, 0))
            return super().select(bumped, k, row_lengths, scheduling_group_id=scheduling_group_id,
                                  return_scores=return_scores)

        def __hash__(self):
            return 1

        def __eq__(self, other):
            return isinstance(other, OneUlp)

    _spy(monkeypatch, OneUlp())
    bumped = _select(args, k, BASE)
    assert not np.array_equal(bumped, base), "one-ulp positive control read identical"
    print(f"Q38_PSTEP_IDX_POSCTL case={case} identical=0")


@pytest.mark.parametrize("query_block", [0, 4])
def test_decode_paths_ignore_the_flag(monkeypatch, interpret, envset, query_block):
    """Decode / verify steps (tokens <= slots x max_query_len) never take the prefill branch: flag on, the call and
    the selection are today's."""
    args = inputs(3, [4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0], [3001, 2048, 1203] + [0] * 9, 12, "integer")
    envset("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", query_block)
    for name in ("VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE", "VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA",
                 "VLLM_TPU_QSA_INDEXER_MIXED_ONLY", "VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY"):
        envset(name, None)
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 1 << 20)
    seen = _spy(monkeypatch)

    def run():
        return np.asarray(pooled_qsa.select_blocks(**args, ratio=RATIO, block_topk=BLOCK_TOPK, max_query_len=4))

    envset(FLAG, None)
    off = run()
    calls_off = list(seen)
    seen.clear()
    envset(FLAG, 1)
    on = run()
    assert seen == calls_off and all(not s[1] for s in seen), (seen, calls_off)
    np.testing.assert_array_equal(on, off)
