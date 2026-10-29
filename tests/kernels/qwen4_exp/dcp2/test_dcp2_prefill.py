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
    combine_plain_average,
    combine_rank0_only,
    fault_attend,
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
        case, combine=combine, check_partials=check_partials, **kw)
    live = np.broadcast_to((case.lengths > 0)[:, None], (len(case.lengths), case.heads))
    ref = replicated_gated(case, full_out, full_lse)
    reports = dict(result.reports)
    for r in range(D):
        reports[f"dcp_vs_ref{r}"] = check_gated(result.gated[r], ref[r], live,
                                                what=f"prefill DCP rank {r}")
    return reports


@pytest.mark.parametrize("name", sorted(CHUNKS))
def test_prefill_dcp_matches_replicated(name):
    spec = CHUNKS[name]
    case = prefill_case(np.random.default_rng(len(name)), spec["chunks"],
                        budget=spec["budget"])
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


def production_replicated(case):
    h = case.heads
    out = []
    for r in range(D):
        o, _ = qsa_rpa._prefill_attention(
            case.q[:, r * h : (r + 1) * h], case.gate[:, r * h : (r + 1) * h],
            jnp.asarray(case.blocks), jnp.asarray(case.lengths),
            jnp.asarray(case.full_rows), case.full_pool, jnp.asarray(case.starts),
            kv_heads=1, ratio=case.ratio, k_scale=case.k_scale, v_scale=case.v_scale)
        out.append(o)
    return out


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


@pytest.mark.skipif(not _prefill_lse_implemented(),
                    reason="block_sparse_gqa_attention(dcp, return_lse) not landed (set6)")
def test_production_prefill_dcp_matches_production_replicated(interpret):
    case = prefill_case(np.random.default_rng(4), SMALL, budget=64)

    def attend(q, gate, blocks, lengths, table, cache, **kw):
        return qsa_rpa._prefill_attention(
            q, gate, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table),
            cache, jnp.asarray(case.starts), return_lse=True, **kw)[:2]

    result, _ = run_dcp_attention(case, attend)
    live = np.broadcast_to((case.lengths > 0)[:, None], (len(case.lengths), case.heads))
    prod = production_replicated(case)
    for r in range(D):
        print(check_gated(result.gated[r], prod[r], live, what=f"production DCP {r}"))
