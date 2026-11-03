# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Score-free QSA selection preserves causal winner sets and fallback work."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp import decode_indexer


def _visibility(counts, context, tokens, ratio):
    visible = np.zeros(tokens, np.int32)
    first = 0
    for count, history in zip(counts, context):
        for local in range(int(count)):
            if first + local < tokens:
                visible[first + local] = max(0, (int(history) + local + 1) // ratio)
        first += int(count)
    return visible


@pytest.mark.parametrize("ratio", [1, 2, 4, 8])
@pytest.mark.parametrize("tokens", [1, 7, 64, 129])
def test_prefill_visibility_matches_request_loop(ratio, tokens):
    rng = np.random.default_rng(2281 + ratio + tokens)
    for _ in range(12):
        counts = rng.multinomial(tokens - tokens // 7, [0.2, 0, 0.4, 0.4, 0])
        context = rng.integers(0, 131072, len(counts), dtype=np.int32)
        starts = np.r_[0, counts.cumsum()].astype(np.int32)
        actual = decode_indexer._prefill_visible_groups(
            jnp.asarray(starts), jnp.asarray(context), tokens, ratio
        )
        np.testing.assert_array_equal(
            actual, _visibility(counts, context, tokens, ratio)
        )


def _inputs(kind, ratio=4, heads=4):
    if kind == "decode-fit":
        counts = [1] * 9
        context = np.resize([0, ratio * 16 - 1, ratio * 17 - 2], 9)
        distribution = [9, 9, 9]
        tokens = 9
    elif kind == "decode-fallback":
        counts = [1] * 9
        context = np.resize([0, ratio * 17 - 1, ratio * 80], 9)
        distribution = [9, 9, 9]
        tokens = 9
    elif kind == "prefill-fit":
        counts, context, distribution, tokens = (
            [17, 0, 13, 0],
            [0, 300, 2, 300],
            [0, 0, 3],
            64,
        )
    elif kind == "prefill-crosses-budget":
        counts, context, distribution, tokens = (
            [37, 0, 91, 0],
            [0, 300, 3, 300],
            [0, 0, 3],
            128,
        )
    elif kind == "prefill-fallback":
        counts, context, distribution, tokens = (
            [37, 0, 91, 0],
            [ratio * 17, 300, ratio * 80, 300],
            [0, 0, 3],
            128,
        )
    elif kind == "mixed":
        counts, context, distribution, tokens = (
            [1, 4, 65, 0],
            [ratio * 17, 3, 0, 300],
            [1, 2, 3],
            128,
        )
    else:
        raise AssertionError(kind)
    counts = np.asarray(counts, np.int32)
    context = np.asarray(context, np.int32)
    starts = np.r_[0, counts.cumsum()].astype(np.int32)
    visible = _visibility(counts, context, tokens, ratio)
    req = np.minimum(
        np.searchsorted(starts[1:], np.arange(tokens), side="right"), len(counts) - 1
    )
    lengths = np.where(
        np.arange(tokens) < starts[-1],
        context[req] + np.arange(tokens) - starts[req] + 1,
        0,
    ).astype(np.int32)
    rng = np.random.default_rng(2839 + heads + ratio)
    pages = 8
    table = (
        rng.permutation(len(counts) * pages)
        .reshape(len(counts), pages)
        .astype(np.int32)
    )
    # Exactly representable scalar keys make the finite scoring oracle independent
    # of matrix multiplication precision, while physical pages remain permuted.
    cache = np.zeros((len(counts) * pages, 32, 1, 128), np.float32)
    for r in range(len(counts)):
        cache[table[r], :, 0, 0] = rng.permutation(pages * 32).reshape(pages, 32) + 1
    query = np.zeros((tokens, heads, 128), np.float32)
    query[:, :, 0] = 1
    args = tuple(
        map(
            jnp.asarray,
            (
                query,
                cache,
                lengths,
                table,
                req,
                starts,
                context,
                np.asarray(distribution, np.int32),
            ),
        )
    )
    args = (args[0].astype(jnp.bfloat16), args[1].astype(jnp.bfloat16), *args[2:])
    return args, visible


def _assert_winners(actual, args, visible):
    ids = np.asarray(actual)
    cache, table, req = map(np.asarray, (args[1], args[3], args[4]))
    for row, count in enumerate(visible):
        live = ids[row][ids[row] >= 0]
        assert len(live) == min(int(count), 16)
        assert len(set(live)) == len(live)
        assert np.all(live < count)
        scores = cache[table[req[row]], :, 0, 0].reshape(-1).astype(np.float32)
        expected = np.sort(scores[:count])[-16:]
        np.testing.assert_array_equal(np.sort(scores[live]), expected)
        np.testing.assert_array_equal(ids[row, len(live) :], -1)


def _select(args, **kwargs):
    return decode_indexer.select_decode_blocks(
        *args[:5],
        starts=args[5],
        context=args[6],
        request_distribution=args[7],
        compress_ratio=kwargs.pop("ratio", 4),
        block_topk=16,
        **kwargs,
    )


@pytest.mark.parametrize(
    "kind",
    [
        "decode-fit",
        "decode-fallback",
        "prefill-fit",
        "prefill-crosses-budget",
        "prefill-fallback",
        "mixed",
    ],
)
def test_score_free_routes_only_complete_winner_sets(monkeypatch, kind):
    args, visible = _inputs(kind)
    calls = []
    monkeypatch.setattr(decode_indexer, "decode_query_chunk_size", lambda _: 4)

    def score(q, weights, cache, seq_lens, pages, starts, distribution, **kwargs):
        calls.append(len(q))
        pages = np.asarray(pages).reshape(len(seq_lens), -1)
        counts = np.diff(starts)
        history = np.asarray(seq_lens) - counts
        row_counts = _visibility(counts, history, len(q), 4)
        owners = np.minimum(
            np.searchsorted(np.asarray(starts)[1:], np.arange(len(q)), side="right"),
            len(counts) - 1,
        )
        result = np.full((len(q), 16), -1, np.int32)
        data = np.asarray(cache).reshape(len(cache), 32, 128)
        for row, count in enumerate(row_counts):
            scores = data[pages[owners[row]], :, 0].reshape(-1)[:count]
            n = min(int(count), 16)
            result[row, :n] = np.argsort(-scores, kind="stable")[:n]
        return jnp.asarray(result)

    monkeypatch.setattr(decode_indexer, "streamindex_topk", score)
    with jax.disable_jit():
        result = _select(args)
    _assert_winners(result, args, visible)
    if kind.endswith("fit"):
        assert not calls
    elif kind.startswith("prefill-"):
        assert calls == [len(args[0])]
    else:
        assert calls


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "kind",
    [
        "decode-fit",
        "decode-fallback",
        "prefill-fit",
        "prefill-crosses-budget",
        "prefill-fallback",
        "mixed",
    ],
)
@pytest.mark.parametrize("ratio,heads", [(2, 1), (4, 4), (8, 2)])
def test_native_visible_groups_preserve_sets_and_input_bytes(kind, ratio, heads):
    args, visible = _inputs(kind, ratio, heads)
    before = [np.asarray(x).tobytes() for x in args]
    run = jax.jit(lambda *xs: _select(xs, ratio=ratio))
    compiled = run.lower(*args).compile()
    actual = compiled(*args)
    jax.block_until_ready(actual)
    _assert_winners(actual, args, visible)
    np.testing.assert_array_equal(compiled(*args), actual)
    assert [np.asarray(x).tobytes() for x in args] == before


@pytest.mark.parametrize("ratio", [2, 4, 8])
@pytest.mark.parametrize("groups", [0, 511, 512, 513, 32768])
def test_decode_budget_is_in_groups_and_falls_back(monkeypatch, ratio, groups):
    calls = []
    monkeypatch.setattr(decode_indexer, "decode_query_chunk_size", lambda _: 4)

    def score(q, weights, cache, lengths, pages, starts, distribution, **kwargs):
        calls.append(int(lengths[0]))
        return jnp.full((1, 512), 7, jnp.int32)

    monkeypatch.setattr(decode_indexer, "streamindex_topk", score)
    query = jnp.zeros((1, 1, 128), jnp.bfloat16)
    cache = jnp.zeros((1, 16, 2, 128), jnp.bfloat16)
    lengths = jnp.array([groups * ratio + ratio - 1], jnp.int32)
    table = jnp.zeros((1, (32768 + 31) // 32), jnp.int32)
    with jax.disable_jit():
        actual = decode_indexer.select_index_blocks(
            query,
            cache,
            lengths,
            table,
            jnp.zeros(1, jnp.int32),
            starts=None,
            context=None,
            ratio=ratio,
            block_topk=512,
        )
    if groups <= 512:
        assert not calls
        wanted = np.where(np.arange(512) < groups, np.arange(512), -1)[None]
        np.testing.assert_array_equal(actual, wanted)
    else:
        assert calls == [int(lengths[0])]
        np.testing.assert_array_equal(actual, 7)


def test_explicit_backend_keeps_its_scoring_contract(monkeypatch):
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
        SparseCoreTopK,
    )

    args, _ = _inputs("prefill-fit")
    backend = SparseCoreTopK()
    calls = []
    monkeypatch.setattr(decode_indexer, "decode_query_chunk_size", lambda _: 4)

    def score(q, weights, cache, lengths, pages, starts, distribution, **kwargs):
        assert kwargs["topk_backend"] is backend
        calls.append(len(q))
        return jnp.full((len(q), 16), -1, jnp.int32)

    monkeypatch.setattr(decode_indexer, "streamindex_topk", score)
    with jax.disable_jit():
        _select(args, topk_backend=backend)
    assert calls == [len(args[0])]


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "kind,calls",
    [
        ("prefill-fit", []),
        ("prefill-crosses-budget", [128]),
        ("prefill-fallback", [128]),
    ],
)
def test_native_scoring_uses_whole_chunk_unless_all_visible_groups_fit(
    monkeypatch, kind, calls
):
    args, visible = _inputs(kind)
    observed = []
    original = decode_indexer.streamindex_topk

    def score(q, *a, **kw):
        result = original(q, *a, **kw)
        jax.debug.callback(
            lambda n: observed.append(int(n)), jnp.int32(len(q)), ordered=True
        )
        return result

    monkeypatch.setattr(decode_indexer, "streamindex_topk", score)
    q, cache, lengths, table, req, starts, history, distribution = args

    def run(q, cache, lengths, table, req, starts, history, distribution):
        return decode_indexer.select_index_blocks(
            q,
            cache.reshape(len(cache), 16, 2, 128),
            lengths,
            table,
            req,
            starts=starts,
            context=history,
            request_distribution=distribution,
            ratio=4,
            block_topk=16,
        )

    actual = jax.jit(run)(*args)
    jax.block_until_ready(actual)
    jax.effects_barrier()
    _assert_winners(actual, args, visible)
    assert observed == calls


@pytest.mark.parametrize("tokens", [70, 133])
@pytest.mark.parametrize(
    "kind", ["prefill-crosses-budget", "prefill-fallback", "mixed"]
)
def test_prefill_preserves_request_crossings_and_partial_tail(
    monkeypatch, kind, tokens
):
    args, visible = _inputs(kind)
    args = list(args)
    padding = max(0, tokens - len(visible))
    args[0] = jnp.pad(args[0], ((0, padding), (0, 0), (0, 0)))[:tokens]
    args[2] = jnp.pad(args[2], (0, padding))[:tokens]
    args[4] = jnp.pad(args[4], (0, padding))[:tokens]
    args[5] = jnp.minimum(args[5], tokens)
    visible = np.pad(visible, (0, padding))[:tokens]
    monkeypatch.setattr(decode_indexer, "decode_query_chunk_size", lambda _: 4)

    def score(q, weights, cache, seq_lens, pages, starts, distribution, **kwargs):
        pages = np.asarray(pages).reshape(len(seq_lens), -1)
        counts = np.diff(starts)
        history = np.asarray(seq_lens) - counts
        row_counts = _visibility(counts, history, len(q), 4)
        owners = np.minimum(
            np.searchsorted(np.asarray(starts)[1:], np.arange(len(q)), side="right"),
            len(counts) - 1,
        )
        result = np.full((len(q), 16), -1, np.int32)
        data = np.asarray(cache).reshape(len(cache), 32, 128)
        for row, count in enumerate(row_counts):
            scores = data[pages[owners[row]], :, 0].reshape(-1)[:count]
            n = min(int(count), 16)
            result[row, :n] = np.argsort(-scores, kind="stable")[:n]
        return jnp.asarray(result)

    monkeypatch.setattr(decode_indexer, "streamindex_topk", score)
    with jax.disable_jit():
        actual = _select(tuple(args))
    _assert_winners(actual, args, visible)


@pytest.mark.parametrize(
    "tokens,history,active,injected",
    [
        (8192, 0, 8192, False),
        (8192, 8192, 8192, False),
        (8192, 8192, 8019, False),
        (4097, 8192, 3989, False),
        (2048, 0, 1997, False),
        (512, 0, 497, True),
    ],
)
def test_prefill_scoring_receives_complete_query_and_metadata(
    monkeypatch, tokens, history, active, injected
):
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
        SparseCoreTopK,
    )

    monkeypatch.setattr(decode_indexer, "decode_query_chunk_size", lambda _: 512)
    query = jnp.zeros((tokens, 1, 128), jnp.bfloat16)
    query = query.at[:, 0, 0].set(jnp.arange(tokens) % 127)
    starts = jnp.array([0, active] + [active] * 7, jnp.int32)
    context = jnp.array([history] + [73] * 7, jnp.int32)
    lengths = jnp.where(
        jnp.arange(tokens) < active, history + jnp.arange(tokens) + 1, 0
    )
    visible = np.where(
        np.arange(tokens) < active, (history + np.arange(tokens) + 1) // 4, 0
    )
    backend = SparseCoreTopK() if injected else None
    calls = []

    def score(q, weights, cache, seq_lens, pages, cu, distribution, **kwargs):
        np.testing.assert_array_equal(q, query)
        np.testing.assert_array_equal(cu, starts)
        np.testing.assert_array_equal(seq_lens, [history + active] + [0] * 7)
        np.testing.assert_array_equal(distribution, [0, 0, 1])
        assert kwargs["chunk_tokens"] == (512 if tokens % 512 == 0 else None)
        if injected:
            assert kwargs["topk_backend"] is backend
        calls.append(len(q))
        groups = jnp.arange(512)[None]
        return jnp.where(groups < jnp.asarray(visible)[:, None], groups, -1)

    monkeypatch.setattr(decode_indexer, "streamindex_topk", score)
    with jax.disable_jit():
        actual = decode_indexer.select_index_blocks(
            query,
            jnp.zeros((1, 16, 2, 128), jnp.bfloat16),
            lengths,
            jnp.zeros((8, 2054), jnp.int32),
            jnp.zeros(tokens, jnp.int32),
            starts=starts,
            context=context,
            request_distribution=jnp.array([0, 0, 1], jnp.int32),
            ratio=4,
            block_topk=512,
            topk_backend=backend,
        )
    groups = np.arange(512)[None]
    np.testing.assert_array_equal(
        actual, np.where(groups < visible[:, None], groups, -1)
    )
    assert calls == ([tokens] if injected or np.any(visible > 512) else [])
