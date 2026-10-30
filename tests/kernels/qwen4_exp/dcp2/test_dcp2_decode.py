# SPDX-License-Identifier: Apache-2.0
"""(a) DCP-2 decode (RPAd rows) over QSA-selected blocks, and the owner split.

Oracle: the replicated production route ``qsa_rpa.rpa_attention`` on the full
pool (B-token table), with the CPU RPA stand-in the existing QSA tests use.
DCP path: each rank attends its owned selected tokens on its NaN-poisoned
pool (2B-token table) through the attend adapter, the pair is merged with
``qsa_rpa.dcp_combine`` per TP rank, and compared within the tolerance policy
of dcp2_harness. Rows: MTP verify seats of 4 rows (lengths L..L+3), idle
seats, NaN-padded bucket rows; lengths hit every tail phase (0/1/B-1/B/B+1
tokens past a page) and long contexts (3-40 pages of B=512).
"""

import dataclasses
import inspect

import jax
import numpy as np
import pytest
from dcp2_harness import (
    DEFAULT_TOL,
    D,
    ON_TPU,
    check_gated,
    check_metadata,
    check_pregate,
    combine_plain_average,
    combine_rank0_only,
    decode_case,
    fault_attend,
    merge_pair,
    owner_flipped,
    owner_of_group,
    owner_token_parity,
    rank_operand,
    reference_attend,
    reference_metadata,
    replicated_gated,
    run_dcp_attention,
    selected_token_ids,
)

from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa

B = 512
# Seats (base lengths; 0 = idle seat). Rows per seat = 4 (MTP verify).
SEATS = {
    4: [3 * B - 1],
    8: [B - 1, 2 * B + 1],
    32: [1, B - 2, B, 0, 2 * B - 1, 3 * B + 3, 7 * B + 1, 12 * B - 1],
    96: [1, 3, B - 2, B - 1, B, B + 1, 2 * B - 1, 2 * B, 2 * B + 1, 3 * B + 3, 0,
         4 * B - 1, 5 * B + 2, 7 * B + 1, 10 * B - 1, 13 * B, 17 * B + 3, 20 * B + 1,
         25 * B - 1, 31 * B + 2, 36 * B, 40 * B - 1, 0, 39 * B + 5],
}
PAD_ROWS = 4


def _rpa_lse_implemented():
    return "return_lse" in inspect.signature(qsa_rpa.rpa_attention).parameters


def production_rpa_attend(q, gate, blocks, lengths, table, cache, **kw):
    """set6's rpa_attention(dcp=..., return_lse=True) -> (out, lse, cache),
    in chunks of rpa_query_batch_size rows (eager; see ``oracle``)."""
    step = len(q) if ON_TPU else qsa_rpa.rpa_query_batch_size(blocks.shape[1] + 1)
    outs, lses = [], []
    for s in range(0, len(q), step):
        rows = slice(s, min(len(q), s + step))
        out, lse, _ = qsa_rpa.rpa_attention(
            q[rows], gate[rows], jax.numpy.asarray(blocks[rows]),
            jax.numpy.asarray(lengths[rows]), jax.numpy.asarray(table[rows]), cache,
            return_lse=True, **kw)
        outs.append(out)
        lses.append(lse)
    return jax.numpy.concatenate(outs), jax.numpy.concatenate(lses)


ADAPTERS = {
    "production": pytest.param(
        production_rpa_attend,
        marks=pytest.mark.skipif(not _rpa_lse_implemented(),
                                 reason="rpa_attention(dcp, return_lse) not landed (set6)"),
    ),
    "reference": reference_attend,
}


def oracle(case, *, pregate=False):
    """Replicated production route per TP rank (H heads each): gated BF16, or
    with ``pregate`` its f32 (out, lse) from ``return_lse=True``."""
    h = case.heads
    n = len(case.lengths)
    # Chunks within rpa_query_batch_size run eagerly (no lax.scan trace of the
    # CPU stand-in's per-sequence loop); decode rows are independent.
    step = n if ON_TPU else qsa_rpa.rpa_query_batch_size(case.blocks.shape[1] + 1)
    out = []
    for r in range(D):
        parts = []
        for s in range(0, n, step):
            rows = slice(s, min(n, s + step))
            res = qsa_rpa.rpa_attention(
                case.q[rows, r * h : (r + 1) * h], case.gate[rows, r * h : (r + 1) * h],
                jax.numpy.asarray(case.blocks[rows]), jax.numpy.asarray(case.lengths[rows]),
                jax.numpy.asarray(case.full_rows[rows]), case.full_pool, **case.opts,
                **({"return_lse": True} if pregate else {}))
            parts.append(res[:2] if pregate else res[:1])
        out.append([jax.numpy.concatenate(x) for x in zip(*parts)])
    return out if pregate else [o[0] for o in out]


def oracle_gated(case):
    return oracle(case)


def check_case(case, attend=reference_attend, combine=None, check_partials=True):
    result, (full_out, full_lse) = run_dcp_attention(
        case, attend, combine, check_partials=check_partials)
    live = np.broadcast_to((case.lengths > 0)[:, None], (len(case.lengths), case.heads))
    reports = dict(result.reports)
    gated = oracle_gated(case)
    ref_gated = replicated_gated(case, full_out, full_lse)
    if check_partials:
        pre = oracle(case, pregate=True)
        out_pair = jax.numpy.stack([p[0] for p in result.partial])
        lse_pair = jax.numpy.stack([p[1] for p in result.partial])
    for r in range(D):
        if check_partials:
            # Before the gate: merged pair vs the replicated route's f32 output.
            merged = merge_pair(out_pair, lse_pair, r, case.heads)
            reports[f"pregate_dcp_vs_oracle{r}"] = check_pregate(
                *merged, *pre[r], live, what=f"pre-gate DCP vs oracle rank {r}")
        # The f32 reference agrees with the production replicated route...
        # (Report-only on TPU: that compares the replicated kernel's own
        # precision with float32, not DCP.)
        reports[f"oracle_vs_ref{r}"] = check_gated(
            ref_gated[r], gated[r], live,
            dataclasses.replace(DEFAULT_TOL, max_ulp=np.inf) if ON_TPU else DEFAULT_TOL,
            what=f"reference vs oracle rank {r}")
        # ...and the DCP output agrees with the production replicated route.
        reports[f"dcp_vs_oracle{r}"] = check_gated(
            result.gated[r], gated[r], live, what=f"DCP vs oracle rank {r}")
    return reports


@pytest.mark.parametrize("adapter", list(ADAPTERS.values()), ids=list(ADAPTERS))
@pytest.mark.parametrize("rows", sorted(SEATS))
def test_decode_dcp_matches_replicated(rpa_stand_in, interpret, rows, adapter):
    case = decode_case(np.random.default_rng(rows), SEATS[rows], pad_rows=PAD_ROWS)
    assert len(case.lengths) == rows + PAD_ROWS
    reports = check_case(case, adapter)
    print(rows, {k: v for k, v in reports.items()})


@pytest.mark.parametrize("adapter", list(ADAPTERS.values()), ids=list(ADAPTERS))
def test_decode_rank_without_keys(rpa_stand_in, interpret, adapter):
    """Contexts shorter than B: rank 1 owns nothing (lse = -inf), rank 0 all."""
    case = decode_case(np.random.default_rng(1), [1, 5, B - 4, 0])
    _, kv_lens, _, _ = reference_metadata(case.blocks, case.lengths, case.dcp_rows,
                                          ratio=4, page_size=B, dcp=(1, D))
    assert not kv_lens.any()
    check_case(case, adapter)


@pytest.mark.parametrize("adapter", list(ADAPTERS.values()), ids=list(ADAPTERS))
def test_decode_tail_group_on_either_owner(rpa_stand_in, interpret, adapter):
    """Partial tail groups (len % 4 != 0) owned by rank 0 and by rank 1."""
    bases = [B + 1, 2 * B + 1, 3 * B + 2, 4 * B + 3]
    case = decode_case(np.random.default_rng(2), bases)
    owners = {int(owner_of_group(n // 4, B, 4)) for n in case.lengths if n % 4}
    assert owners == {0, 1}
    check_case(case, adapter)


DECODE_FAULTS = {
    "owner_flipped": dict(attend=fault_attend(owner_flipped)),
    "owner_token_parity": dict(attend=fault_attend(owner_token_parity)),
    "missing_combine_rank0_only": dict(attend=production_rpa_attend,
                                       combine=combine_rank0_only),
    "dropped_lse_plain_average": dict(attend=production_rpa_attend,
                                      combine=combine_plain_average),
}


@pytest.mark.parametrize("fault", sorted(DECODE_FAULTS))
def test_decode_fault_controls_are_detected(rpa_stand_in, interpret, fault):
    case = decode_case(np.random.default_rng(32), SEATS[32], pad_rows=PAD_ROWS)
    # Only the final gated comparison against the replicated oracle decides.
    with pytest.raises(AssertionError):
        check_case(case, check_partials=False, **DECODE_FAULTS[fault])


# --- (a') selection split and per-rank virtual-page metadata: EXACT


def test_reference_metadata_equals_replicated_virtual_pages(interpret):
    """The harness's metadata model, at dcp=None, is the production function."""
    case = decode_case(np.random.default_rng(96), SEATS[96], pad_rows=PAD_ROWS)
    pages, kv_lens, cu, _ = qsa_rpa.virtual_page_metadata(
        jax.numpy.asarray(case.blocks), jax.numpy.asarray(case.lengths),
        jax.numpy.asarray(case.full_rows), ratio=4, page_size=B)
    e_pages, e_lens, _, _ = reference_metadata(case.blocks, case.lengths, case.full_rows,
                                               ratio=4, page_size=B, order="stable")
    np.testing.assert_array_equal(np.asarray(pages), e_pages)
    np.testing.assert_array_equal(np.asarray(kv_lens), e_lens)


def test_owner_split_partitions_the_selection():
    """Each selected token on exactly one rank, by B-page parity; tail too."""
    case = decode_case(np.random.default_rng(7), SEATS[96])
    tokens = selected_token_ids(case.blocks, case.lengths, 4)
    total = np.zeros(len(tokens), np.int64)
    for r in range(D):
        _, kv_lens, counts, tail = reference_metadata(
            case.blocks, case.lengths, case.dcp_rows, ratio=4, page_size=B, dcp=(r, D))
        mine = (tokens >= 0) & ((np.maximum(tokens, 0) // B) % D == r)
        np.testing.assert_array_equal(kv_lens, mine.sum(1))
        total += kv_lens
    np.testing.assert_array_equal(total, (tokens >= 0).sum(1))


def _dcp_metadata_implemented():
    return "dcp" in inspect.signature(qsa_rpa.virtual_page_metadata).parameters


@pytest.mark.skipif(not _dcp_metadata_implemented(),
                    reason="virtual_page_metadata(dcp=...) not landed (set6)")
@pytest.mark.parametrize("rows", [32, 96])
def test_production_dcp_metadata_is_exact(interpret, rows):
    case = decode_case(np.random.default_rng(rows + 1), SEATS[rows], pad_rows=PAD_ROWS)
    for r in range(D):
        actual = qsa_rpa.virtual_page_metadata(
            jax.numpy.asarray(case.blocks), jax.numpy.asarray(case.lengths),
            jax.numpy.asarray(case.dcp_rows), ratio=4, page_size=B,
            dcp=(rank_operand(r), D))
        expected = reference_metadata(case.blocks, case.lengths, case.dcp_rows,
                                      ratio=4, page_size=B, dcp=(r, D))
        check_metadata(actual, expected)


@pytest.mark.parametrize("wrong", ["stable_order", "flipped_owner", "token_parity"])
def test_metadata_fault_controls_are_detected(wrong):
    case = decode_case(np.random.default_rng(9), SEATS[32])
    expected = reference_metadata(case.blocks, case.lengths, case.dcp_rows, ratio=4,
                                  page_size=B, dcp=(0, D))
    if wrong == "stable_order":
        actual = reference_metadata(case.blocks, case.lengths, case.dcp_rows, ratio=4,
                                    page_size=B, dcp=(0, D), order="stable")
    elif wrong == "flipped_owner":
        actual = reference_metadata(case.blocks, case.lengths, case.dcp_rows, ratio=4,
                                    page_size=B, dcp=(1, D))
    else:
        # Owner by token parity: every group's even tokens on rank 0 -> all
        # groups and all of the tail on rank 0, kv_lens doubled-ish.
        tokens = selected_token_ids(case.blocks, case.lengths, 4)
        actual = (expected[0], ((tokens >= 0) & (tokens % 2 == 0)).sum(1))
    with pytest.raises(AssertionError):
        check_metadata(actual, expected)
