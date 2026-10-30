# SPDX-License-Identifier: Apache-2.0
"""(b) DCP-2 block-sparse prefill: 2048 / 8192-token chunks, unaligned cached
prefixes and tails, every row's QSA selection split by B-page owner.

Oracle at chunk scale: the float32 replicated reference on the full pool (the
production block-sparse kernel runs in Pallas interpret mode far too slowly
for 2048+ rows on CPU). ``test_reference_matches_production_prefill`` ties
that reference to the production replicated kernel at a small shape, where
the production DCP adapter (set6: ``_prefill_attention(dcp, return_lse)``)
is also compared against the production replicated kernel once it lands.
"""

import inspect

import jax.numpy as jnp
import numpy as np
import pytest
from dcp2_harness import (
    D,
    check_gated,
    check_pregate,
    combine_plain_average,
    combine_rank0_only,
    fault_attend,
    merge_pair,
    owner_flipped,
    owner_token_parity,
    prefill_case,
    replicated_gated,
    run_dcp_attention,
)

from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa
from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
    block_sparse_gqa_attention,
)

B = 512

CHUNKS = {
    # 2048-token chunk on a 1234-token prefix (not a multiple of B).
    "c2048_p1234": dict(chunks=[(1234, 2048)], budget=512),
    # Two requests: a fresh 700-token piece and an unaligned 513-token tail.
    "pieces_p0_p2047": dict(chunks=[(0, 700), (2047, 513)], budget=512),
    # 8192-token chunk on a 3001-token prefix (>20 pages). The block budget is
    # cut to 64 groups to stay within the CPU time budget; ownership and
    # compaction logic is budget-independent.
    "c8192_p3001": dict(chunks=[(3001, 8192)], budget=64),
}


def check_prefill(case, attend=None, combine=None, check_partials=True):
    kw = {} if attend is None else {"attend": attend}
    result, (full_out, full_lse) = run_dcp_attention(
        case, combine=combine, check_partials=check_partials, **kw
    )
    live = np.broadcast_to((case.lengths > 0)[:, None], (len(case.lengths), case.heads))
    ref = replicated_gated(case, full_out, full_lse)
    reports = dict(result.reports)
    for r in range(D):
        reports[f"dcp_vs_ref{r}"] = check_gated(
            result.gated[r], ref[r], live, what=f"prefill DCP rank {r}"
        )
    return reports


@pytest.mark.parametrize("name", sorted(CHUNKS))
def test_prefill_dcp_matches_replicated(name):
    spec = CHUNKS[name]
    case = prefill_case(
        np.random.default_rng(len(name)), spec["chunks"], budget=spec["budget"]
    )
    # Both ranks own tokens, and owned tail groups land on both ranks.
    print(name, check_prefill(case, check_partials=name != "c8192_p3001"))


PREFILL_FAULTS = {
    "owner_flipped": dict(attend=fault_attend(owner_flipped)),
    "owner_token_parity": dict(attend=fault_attend(owner_token_parity)),
    "missing_combine_rank0_only": dict(combine=combine_rank0_only),
    "dropped_lse_plain_average": dict(combine=combine_plain_average),
}


@pytest.mark.parametrize("fault", sorted(PREFILL_FAULTS))
def test_prefill_fault_controls_are_detected(fault):
    case = prefill_case(np.random.default_rng(11), [(1234, 2048)], budget=128)
    with pytest.raises(AssertionError):
        check_prefill(case, check_partials=False, **PREFILL_FAULTS[fault])


# --- production kernels at a small shape (Pallas interpret mode)

SMALL = [(613, 96), (1500, 40)]  # two requests, unaligned prefixes and tails


def production_replicated(case, *, pregate=False):
    """Replicated served prefill route per TP rank: gated BF16, or with
    ``pregate`` its f32 (out, lse) from ``return_lse=True``."""
    h = case.heads
    out = []
    for r in range(D):
        res = qsa_rpa._prefill_attention(
            case.q[:, r * h : (r + 1) * h],
            case.gate[:, r * h : (r + 1) * h],
            jnp.asarray(case.blocks),
            jnp.asarray(case.lengths),
            jnp.asarray(case.full_rows),
            case.full_pool,
            jnp.asarray(case.starts),
            **case.opts,
            **({"return_lse": True} if pregate else {}),
        )
        out.append(res[:2] if pregate else res[0])
    return out


def production_prefill_attend(case):
    """set6's DCP prefill: _prefill_attention(dcp, return_lse) (served tiles:
    kv_tile = 2B, paired pages) -> (out, lse)."""

    def attend(q, gate, blocks, lengths, table, cache, **kw):
        return qsa_rpa._prefill_attention(
            q,
            gate,
            jnp.asarray(blocks),
            jnp.asarray(lengths),
            jnp.asarray(table),
            cache,
            jnp.asarray(case.starts),
            return_lse=True,
            **kw,
        )[:2]

    return attend


def check_production_prefill(case, tol=None):
    """Production DCP prefill vs the production replicated prefill, gated and
    pre-gate; per-rank partials vs the f32 reference."""
    kw = {} if tol is None else {"tol": tol}
    result, _ = run_dcp_attention(case, production_prefill_attend(case), **kw)
    live = np.broadcast_to((case.lengths > 0)[:, None], (len(case.lengths), case.heads))
    gated = production_replicated(case)
    pre = production_replicated(case, pregate=True)
    out_pair = jnp.stack([p[0] for p in result.partial])
    lse_pair = jnp.stack([p[1] for p in result.partial])
    reports = dict(result.reports)
    for r in range(D):
        reports[f"pregate{r}"] = check_pregate(
            *merge_pair(out_pair, lse_pair, r, case.heads),
            *pre[r],
            live,
            *([] if tol is None else [tol]),
            what=f"prefill pre-gate rank {r}",
        )
        reports[f"gated{r}"] = check_gated(
            result.gated[r],
            gated[r],
            live,
            *([] if tol is None else [tol]),
            what=f"prefill production DCP rank {r}",
        )
    return reports


def test_reference_matches_production_prefill(interpret):
    """The f32 reference vs the production replicated block-sparse kernel."""
    case = prefill_case(np.random.default_rng(3), SMALL, budget=64)
    _, (full_out, full_lse) = run_dcp_attention(case, check_partials=False)
    ref = replicated_gated(case, full_out, full_lse)
    live = np.broadcast_to((case.lengths > 0)[:, None], (len(case.lengths), case.heads))
    prod = production_replicated(case)
    for r in range(D):
        print(check_gated(ref[r], prod[r], live, what=f"reference vs production {r}"))


def _prefill_lse_implemented():
    return "return_lse" in inspect.signature(block_sparse_gqa_attention).parameters


@pytest.mark.skipif(
    not _prefill_lse_implemented(),
    reason="block_sparse_gqa_attention(dcp, return_lse) not landed (set6)",
)
def test_production_prefill_dcp_matches_production_replicated(interpret):
    case = prefill_case(np.random.default_rng(4), SMALL, budget=64)
    print(check_production_prefill(case))


@pytest.mark.parametrize("pages_per_query", [512, 1024])
def test_non_dcp_bounded_metadata_matches_scatter_bitset(monkeypatch, pages_per_query):
    """Non-DCP paired-page Q256 prefill at >=16 MiB bitset with
    VLLM_TPU_QSA_PREFILL_BITSET_KERNEL=0 routes through
    _prepare_metadata_bounded without undefined DCP globals and matches
    _scatter_bitset exactly."""
    from jax.experimental import pallas as pl

    from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as bsp

    rng = np.random.default_rng(17 + pages_per_query)
    tokens, query_block, kv_tile, page_size, ratio, budget = 4096, 256, 512, 256, 4, 64
    groups = pages_per_query * page_size // ratio
    num_kv_tiles = (groups * ratio + kv_tile - 1) // kv_tile
    words_per_tile = kv_tile // (ratio * 32)
    words = ((num_kv_tiles * words_per_tile + 127) // 128) * 128
    assert tokens * words * 4 >= 16 * 1024**2

    lengths = rng.integers(0, groups * ratio, size=(tokens,), dtype=np.int32)
    lengths[0] = 0
    blocks = random_blocks_for_test = np.full((tokens, budget), -1, np.int32)
    for i, n in enumerate(lengths):
        complete = int(n) // ratio
        count = min(complete, budget)
        if count:
            random_blocks_for_test[i, :count] = rng.choice(
                complete, count, replace=False
            )
    block_table_row = rng.integers(1, 16, size=(1, pages_per_query), dtype=np.int32)
    page_table = np.repeat(block_table_row, tokens, axis=0)

    captured: list[tuple] = []

    def capture_pallas_call(*args, **kwargs):
        out_shape = kwargs["out_shape"]

        def runner(*operands):
            captured.append(operands)
            return jnp.zeros(out_shape.shape, out_shape.dtype)

        return runner

    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_BITSET_KERNEL", "0")
    monkeypatch.setattr(pl, "pallas_call", capture_pallas_call)

    query = jnp.zeros((tokens, 6, 256), jnp.bfloat16)
    cache = jnp.zeros((16, page_size, 1, 4, 256), jnp.float8_e4m3fn)
    jit_out = bsp.block_sparse_gqa_prefill(
        query,
        jnp.asarray(blocks),
        jnp.asarray(lengths),
        jnp.asarray(page_table),
        cache,
        ratio=ratio,
        kv_heads=1,
        query_block=query_block,
        kv_tile=kv_tile,
        bf16_io=True,
    )
    assert jit_out.shape == query.shape
    captured.clear()
    out = bsp.block_sparse_gqa_prefill.__wrapped__(
        query,
        jnp.asarray(blocks),
        jnp.asarray(lengths),
        jnp.asarray(page_table),
        cache,
        ratio=ratio,
        kv_heads=1,
        query_block=query_block,
        kv_tile=kv_tile,
        bf16_io=True,
    )
    assert out.shape == query.shape
    assert len(captured) == 1
    got_table, got_tiles, got_counts, _, got_bits, _, _ = captured[0]

    # Reference metadata from _scatter_bitset on the same inputs.
    bitset = bsp._scatter_bitset(
        jnp.asarray(blocks),
        jnp.asarray(lengths),
        ratio=ratio,
        groups=groups,
        words=words,
    )
    nblocks = tokens // query_block
    block_bits = bitset.reshape(nblocks, query_block, words)
    active = jnp.any(
        block_bits[:, :, : words_per_tile * num_kv_tiles].reshape(
            nblocks, query_block, num_kv_tiles, words_per_tile
        )
        != 0,
        axis=(1, 3),
    )
    want_tiles = bsp._compact_active(active).reshape(-1)
    want_counts = jnp.sum(active, axis=1, dtype=jnp.int32)
    want_bits = block_bits.transpose(0, 2, 1)
    want_table = (
        jnp.asarray(page_table).reshape(nblocks, query_block, -1)[:, 0].reshape(-1)
    )

    np.testing.assert_array_equal(np.asarray(got_table), np.asarray(want_table))
    np.testing.assert_array_equal(np.asarray(got_counts), np.asarray(want_counts))
    np.testing.assert_array_equal(np.asarray(got_tiles), np.asarray(want_tiles))
    np.testing.assert_array_equal(np.asarray(got_bits), np.asarray(want_bits))
