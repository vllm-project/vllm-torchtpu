"""CPU checks for DeepSeek V4.1 cache addressing and gather paths."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention import (
    _query_split,
    _row_slots,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
    _attend_window,
    _attention_state,
    paged_attention_jax,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
    gather_cache_pieces,
    gather_cache_values,
    merge_halves,
    merge_sub_rows,
    scatter_cache_values,
    split_halves,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_select import _visible_top_k


@pytest.mark.parametrize("width", [7, 128, 300, 2560])
def test_row_slots_matches_elementwise_gather(width):
    rows_per_page = 32
    generator = torch.Generator().manual_seed(width)
    table = torch.randint(0, 1000, (4, 90), generator=generator, dtype=torch.int32)
    requests = torch.randint(0, 4, (17,), generator=generator)
    logical = torch.randint(
        -5, 90 * rows_per_page + 5, (17, width), generator=generator
    )
    cache = torch.empty((1, rows_per_page))
    safe = logical.clamp(0, table.shape[1] * rows_per_page - 1)
    expected = (
        table[requests[:, None], safe // rows_per_page].long() * rows_per_page
        + safe % rows_per_page
    )
    torch.testing.assert_close(_row_slots(cache, table, requests, logical), expected)


@pytest.mark.parametrize("elements,fp32", [(512, False), (128, False), (64, True)])
def test_gather_cache_values_negative_records(elements, fp32):
    pool = jnp.asarray(
        np.random.default_rng(0).integers(0, 2**31, (64, 128)), dtype=jnp.int32
    ) & jnp.int32(0x3FFF3FFF)  # finite BF16/FP32 values
    record_words = elements if fp32 else elements // 2
    count = 64 * 128 // record_words
    records = jnp.asarray([3, -1, count - 1, -1, -7, 0], dtype=jnp.int32)
    got = gather_cache_values(pool, records, elements, fp32)
    valid = records >= 0
    expected = gather_cache_values(pool, jnp.where(valid, records, 0), elements, fp32)
    np.testing.assert_array_equal(np.asarray(got)[valid], np.asarray(expected)[valid])
    assert got.shape == (records.size, elements)


@pytest.mark.parametrize("elements", [512, 128])
def test_gather_cache_values_split_order(elements):
    # Finite, normal BF16 values (permutations are matmuls; subnormals
    # flush); signs are kept.
    pool = jnp.asarray(
        np.random.default_rng(1).integers(-(2**31), 2**31, (64, 128)), jnp.int32
    ) & jnp.int32(-0x40004001) | jnp.int32(0x00800080)  # 0xBFFFBFFF
    records = jnp.asarray([5, 0, 17, 2], dtype=jnp.int32)
    natural = gather_cache_values(pool, records, elements)
    split = gather_cache_values(pool, records, elements, split=True)
    np.testing.assert_array_equal(
        np.asarray(split, np.float32), np.asarray(split_halves(natural), np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(merge_halves(split), np.float32), np.asarray(natural, np.float32)
    )
    np.testing.assert_array_equal(
        np.concatenate(
            [
                np.asarray(h, np.float32)
                for h in gather_cache_pieces(pool, records, elements)
            ],
            axis=1,
        ),
        np.asarray(split, np.float32),
    )
    numerator = jnp.asarray(
        np.random.default_rng(2).standard_normal((3, elements)), jnp.float32
    )
    np.testing.assert_array_equal(
        np.asarray(merge_halves(split_halves(numerator))), np.asarray(numerator)
    )


@pytest.mark.parametrize("elements", [512, 128])
def test_pack_bf16_words_and_split_pieces_roundtrip(elements):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
        _pack_bf16_words,
        merge_pieces,
        split_pieces,
    )

    rng = np.random.default_rng(elements)
    raw = rng.standard_normal((16, elements)).astype(np.float32)
    values = jnp.asarray(raw, dtype=jnp.bfloat16)
    got_words = _pack_bf16_words(values)
    ref_words = jax.lax.bitcast_convert_type(
        values.reshape(values.shape[0], -1, 2), jnp.int32
    )
    np.testing.assert_array_equal(np.asarray(got_words), np.asarray(ref_words))

    pieces = elements // 128
    q = jnp.asarray(rng.standard_normal((8, 64, elements)), dtype=jnp.bfloat16)
    q_pieces = split_pieces(q, pieces)
    assert q_pieces.shape == (pieces, 8, 64, 128)
    num = jnp.asarray(rng.standard_normal((pieces, 8, 64, 128)), dtype=jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(merge_pieces(split_pieces(merge_pieces(num), pieces)), np.float32),
        np.asarray(merge_pieces(num), np.float32),
    )


@pytest.mark.parametrize("width", [64, 200])
def test_paged_attention_matches_dense(width):
    rng = np.random.default_rng(width)
    tokens, heads, dim = 4, 2, 512
    pool = jnp.asarray(rng.integers(0, 2**31, (512, 128)), jnp.int32) & jnp.int32(
        0x3EFF3EFF
    )  # finite, moderate BF16 values
    records = rng.integers(0, 256, (tokens, width)).astype(np.int32)
    records[:, :3] = -1
    q = jnp.asarray(rng.standard_normal((tokens, heads, dim)), jnp.bfloat16)
    sink = jnp.asarray(rng.standard_normal(heads), jnp.float32)
    empty = jnp.zeros((tokens, 0), jnp.int32)
    scale = dim**-0.5
    got = paged_attention_jax(q, pool, pool, jnp.asarray(records), empty, sink, scale)
    kv = gather_cache_values(pool, jnp.asarray(np.maximum(records, 0)).reshape(-1), dim)
    kv = np.asarray(kv, np.float32).reshape(tokens, width, dim)
    s = np.einsum("thd,tkd->thk", np.asarray(q, np.float32), kv) * scale
    s = np.where((records >= 0)[:, None], s, -np.inf)
    m = np.maximum(s.max(-1), np.asarray(sink)[None])
    p = np.exp(s - m[..., None])
    den = p.sum(-1) + np.exp(np.asarray(sink)[None] - m)
    ref = np.einsum("thk,tkd->thd", p, kv) / den[..., None]
    np.testing.assert_allclose(np.asarray(got, np.float32), ref, atol=2e-3)


def _paged_records(rng, rows, width, page, pool_records):
    """Per-request records: random pages, contiguous inside."""
    pages = -(-width // page)
    table = rng.integers(0, pool_records // page, (rows, pages))
    logical = np.arange(width)
    return (table[:, logical // page] * page + logical % page).astype(np.int32)


def _request_inputs(rng, requests_count, tokens, width, heads=4, dim=128):
    pool = jnp.asarray(rng.integers(-(2**31), 2**31, (256, 128)), dtype=jnp.int32)
    pool = pool & jnp.int32(0x3FFF3FFF)  # finite BF16 keys
    records = rng.integers(0, 256 * 2, (requests_count, width)).astype(np.int32)
    records[:, width - 5 :] = -1
    requests = np.sort(rng.integers(0, requests_count, tokens)).astype(np.int32)
    lengths = rng.integers(1, width + 1, tokens).astype(np.int32)
    q = jnp.asarray(rng.standard_normal((tokens, heads, dim)), jnp.bfloat16)
    weights = jnp.asarray(rng.standard_normal((tokens, heads)), jnp.float32)
    return pool, records, requests, lengths, q, weights


@pytest.mark.parametrize("width,count", [(64, 40), (32, 40), (64, 1), (32, 7)])
def test_merge_sub_rows_matches_lane_scatter(width, count):
    rng = np.random.default_rng(width + count)
    per_row = 128 // width
    cache = jnp.asarray(rng.integers(-(2**31), 2**31, (32, 128)), dtype=jnp.int32)
    # Unique records, dense enough that rows hold several batch records.
    records = rng.permutation(32 * per_row)[:count].astype(np.int32)
    records[::5] = -1
    words = jnp.asarray(rng.integers(-(2**31), 2**31, (count, width)), jnp.int32)
    expected = scatter_cache_values(cache, jnp.asarray(records), words)
    rows, full = merge_sub_rows(cache, jnp.asarray(records), words)
    got = cache.at[jnp.where(rows >= 0, rows, cache.shape[0])].set(full, mode="drop")
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


@pytest.mark.parametrize("rows", [3, 16])
@pytest.mark.parametrize("bound", [0, 7, 512, 1024, 1025, 3000, 5000])
def test_visible_top_k_matches_full(bound, rows):
    rng = np.random.default_rng(bound)
    width, k = 5000, 512
    # Coarse values create ties (incl. -0.0 vs +0.0); columns >= bound and a
    # filtered gap are invisible.
    scores = np.round(rng.standard_normal((rows, width)), 1).astype(np.float32)
    zeros = np.where(rng.random(width) < 0.5, -0.0, 0.0).astype(np.float32)
    scores[::2] = np.where(np.abs(scores[::2]) < 0.3, zeros, scores[::2])
    scores[1, 100:900] = -np.inf
    scores[:, bound:] = -np.inf
    scores = jnp.asarray(scores)
    got = _visible_top_k(scores, k, jnp.int32(bound))
    expected = jnp.sort(jax.lax.top_k(scores, k)[1], axis=-1)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


@pytest.mark.parametrize("lengths", [(256,), (100, 3, 1, 152), (37, 90, 129)])
def test_attend_window_branch_free_matches_conditional(lengths):
    # Contiguous requests (chunk tails, decode tokens) packed into one batch,
    # padded to 256 tokens; W = 128 window records per token.
    rng = np.random.default_rng(sum(lengths))
    tokens, heads, dim, window = 256, 2, 512, 128
    pool = jnp.asarray(rng.integers(0, 2**31, (4096, 128)), jnp.int32) & jnp.int32(
        0x3EFF3EFF
    )
    records = np.full((tokens, window), -1, np.int32)
    start = 0
    for request, length in enumerate(lengths):
        base = 1000 * request + int(rng.integers(0, 300))
        for t in range(length):
            position = base + t
            logical = position - window + 1 + np.arange(window)
            records[start + t] = np.where(logical >= 0, logical, -1)
        start += length
    q = jnp.asarray(rng.standard_normal((tokens, heads, dim)), jnp.bfloat16)
    state = _attention_state(q)
    # fallback=tokens: the conditional version patches instead of redoing
    # the whole batch with `_attend`.
    expected = _attend_window(
        q, pool, jnp.asarray(records), state, dim**-0.5, fallback=tokens
    )
    got = _attend_window(
        q, pool, jnp.asarray(records), state, dim**-0.5, max_requests=len(lengths)
    )
    for e, g in zip(expected, got):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(e))


@pytest.mark.parametrize("ranks", [2, 4, 8])
def test_query_split_parts_are_even_and_covered(ranks):
    for count in range(1, 2 * 2048 + 1):
        part, padding = _query_split(count, ranks)
        assert part >= 2 and part % 2 == 0
        assert 0 <= ranks * part - count <= padding
        assert ranks * (part - 2) < count  # no whole spare part per rank


def test_query_split_traces_one_graph_for_all_buckets():
    # A 1-row part (ceil split at T=4/8 on 8 ranks) 0/1-specialized and traced
    # a second graph; the base 16-row split traced one.
    from torch._dynamo.testing import CompileCounter

    def split(x):
        part, padding = _query_split(x.shape[0], 8)
        mine = 3 * part + torch.arange(part, dtype=torch.int32)
        rows = torch.cat((x, x.new_zeros((padding, *x.shape[1:])))).index_select(
            0, mine
        )
        return rows.repeat(8, 1)[: x.shape[0]]

    torch._dynamo.reset()
    counter = CompileCounter()
    compiled = torch.compile(split, backend=counter)
    for tokens in (2048, 4, 8, 16):
        x = torch.randn(tokens, 4)
        torch._dynamo.mark_dynamic(x, 0)
        compiled(x)
    torch._dynamo.reset()
    assert counter.frame_count == 1


@pytest.mark.parametrize(
    "tokens,heads,queries,width,layout",
    [
        (128, 8, 1, 128, "shared"),
        (128, 8, 1, 128, "boundary"),
        (128, 8, 1, 128, "decode"),
        (288, 64, 6, 128, "grouped"),
        (96, 32, 6, 128, "grouped"),
        (48, 16, 6, 128, "short"),
        (24, 8, 6, 128, "reordered"),
        (24, 64, 6, 128, "padded"),
        (15, 64, 5, 128, "grouped"),
        (21, 16, 7, 128, "grouped"),
        (18, 8, 3, 128, "grouped"),
        (32, 32, 4, 128, "grouped"),
        (16, 64, 8, 128, "grouped"),
        (256, 8, 16, 128, "grouped"),
        (24, 32, 6, 256, "grouped"),
        (24, 64, 2, 128, "grouped"),
        (25, 16, 6, 128, "grouped"),
        (6, 64, 6, 128, "masked"),
        (24, 16, 6, 128, "repeated"),
        (128, 32, 4, 128, "prefill_padded"),
        (258, 8, 129, 128, "grouped"),
        (260, 8, 130, 128, "grouped"),
    ],
)
@pytest.mark.parametrize("banked", [False, True])
def test_pure_swa_pallas_preserves_cache_addressing(
    monkeypatch, tokens, heads, queries, width, layout, banked
):
    """Check union reuse, request-tail patching and per-token windows end to end."""
    import functools

    from vllm_torchtpu.kernels.mla import swa_attention
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import _pack_bf16_words

    if jax.default_backend() != "tpu":
        monkeypatch.setattr(swa_attention, "supports_swa", lambda q, records: True)
        monkeypatch.setattr(
            swa_attention,
            "attend_swa",
            functools.partial(swa_attention.attend_swa, interpret=True),
        )
    rng = np.random.default_rng(17)
    dim = 512
    values = jnp.asarray(rng.normal(size=(512, dim)), jnp.bfloat16)
    pool = _pack_bf16_words(values).reshape(-1, 128)
    if banked:
        pool = (pool[:512], pool[512:])
    positions = np.arange(tokens) + width - 1
    if queries > 1 and layout != "prefill_padded":
        positions = (np.arange(tokens) // queries) * 7 + width - 1
        if layout != "repeated":
            positions += np.arange(tokens) % queries
        if layout == "short":
            positions[:queries] = np.arange(queries)
    elif layout == "boundary":
        positions[37:] = np.arange(tokens - 37) + 300
    elif layout == "decode":
        positions = rng.integers(127, 512, tokens)
    records = positions[:, None] - width + 1 + np.arange(width)[None]
    if queries > 1:
        # Shuffle physical pages, including across the two cache banks.
        pages = rng.permutation(16)
        records = np.where(
            records >= 0, pages[np.maximum(records, 0) // 32] * 32 + records % 32, -1
        )
        records[-queries:] = -1
        if layout == "reordered":
            records[[2, 3]] = records[[3, 2]]
        elif layout in ("padded", "prefill_padded"):
            records[3] = -1
        elif layout == "masked":
            records[:] = -1
    else:
        records[-1] = -1
    gathered_sizes = []
    original_gather = attention_kernels.gather_cache_pieces

    def checked_gather(pool, records, elements):
        jax.debug.callback(lambda size: gathered_sizes.append(int(size)), records.size)
        return original_gather(pool, records, elements)

    monkeypatch.setattr(attention_kernels, "gather_cache_pieces", checked_gather)
    records = jnp.asarray(records, jnp.int32)
    q = jnp.asarray(rng.normal(size=(tokens, heads, dim)), jnp.bfloat16)
    sink = jnp.asarray(rng.normal(size=heads), jnp.float32)
    got = paged_attention_jax(
        q,
        pool,
        pool,
        records,
        jnp.empty((tokens, 0), jnp.int32),
        sink,
        dim**-0.5,
        query_group_size=queries,
    )
    kv = values[jnp.maximum(records, 0)].astype(jnp.float32)
    scores = jnp.einsum("thd,tkd->thk", q.astype(jnp.float32), kv) * dim**-0.5
    scores = jnp.where((records >= 0)[:, None], scores, -jnp.inf)
    maximum = jnp.maximum(scores.max(-1), sink)
    prob = jnp.exp(scores - maximum[..., None])
    expected = jnp.einsum(
        "thk,tkd->thd", prob.astype(jnp.bfloat16).astype(jnp.float32), kv
    )
    expected /= (prob.sum(-1) + jnp.exp(sink - maximum))[..., None]
    np.testing.assert_allclose(
        np.asarray(got, np.float32), expected, atol=3e-3, rtol=1e-2
    )
    np.testing.assert_array_equal(got[-1], 0)
    if queries > 1:
        jax.effects_barrier()
        shares = (
            3 <= queries <= 129
            and width == 128
            and tokens % queries == 0
            and layout in ("grouped", "short", "masked")
        )
        expected_rows = tokens // queries * 256 if shares else tokens * width
        if layout == "prefill_padded":
            expected_rows = tokens // 64 * 256
        assert gathered_sizes == [expected_rows]
        baseline = paged_attention_jax(
            q, pool, pool, records, jnp.empty((tokens, 0), jnp.int32), sink, dim**-0.5
        )
        np.testing.assert_allclose(
            np.asarray(baseline, np.float32), expected, atol=3e-3, rtol=1e-2
        )
        np.testing.assert_allclose(
            np.asarray(got, np.float32),
            np.asarray(baseline, np.float32),
            atol=3e-3,
            rtol=1e-2,
        )
        if not shares:
            np.testing.assert_array_equal(got, baseline)


@pytest.mark.parametrize("banked", [False, True])
def test_swa_query_group_bridge_keeps_token_count_dynamic(monkeypatch, banked):
    from types import SimpleNamespace

    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels

    lib = torch.library.Library("swa_group_bridge_test", "FRAGMENT")
    lib.define("attend(Tensor q, int group_size) -> Tensor")
    lib.impl("attend", lambda q, group_size: q.clone(), "CPU")
    lib.impl("attend", lambda q, group_size: torch.empty_like(q), "Meta")
    op = torch.ops.swa_group_bridge_test.attend.default

    def bridge(*args):
        return op(args[0], args[-1])

    monkeypatch.setattr(attention_kernels, "_attention_op", bridge)
    monkeypatch.setattr(attention_kernels, "_attention_2bank_op", bridge)
    monkeypatch.setattr(attention_kernels, "_active_gathered_kv", None)
    cache = SimpleNamespace(raw=torch.zeros((32, 128), dtype=torch.int32))
    cache.raw_hi = cache.raw.clone() if banked else None
    graphs = []

    def backend(gm, inputs):
        graphs.append(gm)
        return gm.forward

    def forward(q, records):
        return attention_kernels.paged_attention(
            q,
            cache,
            cache,
            records,
            records[:, :0],
            q.new_zeros(8),
            512**-0.5,
            query_group_size=5,
        )

    compiled = torch.compile(forward, backend=backend, fullgraph=True)
    for tokens in (15, 25):
        q = torch.ones((tokens, 8, 512), dtype=torch.bfloat16)
        records = torch.zeros((tokens, 128), dtype=torch.int32)
        torch._dynamo.mark_dynamic(q, 0)
        torch._dynamo.mark_dynamic(records, 0)
        torch.testing.assert_close(compiled(q, records), q)
    assert len(graphs) == 1
    calls = [node for node in graphs[0].graph.nodes if node.target == op]
    assert len(calls) == 1 and calls[0].args[-1] == 5
