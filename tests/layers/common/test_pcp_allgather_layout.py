# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the all-gather PCP head-tail token layout."""
import numpy as np
import pytest

from vllm_torchtpu.layers.common.pcp_allgather_layout import (
    build_headtail_pcp_meta, headtail_inv_row, headtail_row_perm,
    headtail_slot_for_global_token, to_rank_order)


@pytest.mark.parametrize("pcp", [2, 4, 8])
def test_row_perm_is_a_permutation_pairing_head_with_tail(pcp):
    perm = headtail_row_perm(pcp)
    assert sorted(perm) == list(range(2 * pcp))
    # rank r contributes chunks (r, 2P-1-r) -> each pair sums to 2P-1
    for r in range(pcp):
        head, tail = perm[2 * r], perm[2 * r + 1]
        assert head == r and tail == 2 * pcp - 1 - r
        assert head + tail == 2 * pcp - 1


@pytest.mark.parametrize("pcp", [2, 4, 8])
def test_inv_row_inverts_row_perm(pcp):
    inv = headtail_inv_row(pcp)
    perm = headtail_row_perm(pcp)
    for slot, chunk in enumerate(perm):
        assert inv[chunk] == slot


@pytest.mark.parametrize("pcp,chunk", [(2, 4), (4, 3), (8, 1)])
def test_slot_lookup_matches_rank_order_permutation(pcp, chunk):
    """headtail_slot_for_global_token must locate each token exactly where
    to_rank_order actually put it."""
    n = 2 * pcp * chunk
    tokens = np.arange(n, dtype=np.int64).reshape(n, 1)
    packed = to_rank_order(tokens, pcp, chunk)
    for g in range(n):
        slot = headtail_slot_for_global_token(g, chunk, pcp)
        assert int(packed[slot, 0]) == g


@pytest.mark.parametrize("pcp", [2, 4])
def test_to_rank_order_rejects_wrong_length(pcp):
    with pytest.raises(ValueError, match="rows"):
        to_rank_order(np.zeros((2 * pcp * 4 + 1, 2)), pcp, 4)


@pytest.mark.parametrize("pcp", [2, 4])
def test_meta_full_tokens_all_chunks_real(pcp):
    chunk, max_seqs = 8, 8
    num_current = 2 * pcp * chunk  # fully real
    cu, qpos = build_headtail_pcp_meta(pcp, chunk, num_current, max_seqs)
    assert cu.shape == (pcp, max_seqs + 1)
    assert qpos.shape == (pcp, max_seqs)
    for r in range(pcp):
        assert cu[r, 1] == chunk                 # head always full
        assert cu[r, 2] == 2 * chunk             # tail also full
        assert qpos[r, 0] == r * chunk
        assert qpos[r, 1] == (2 * pcp - 1 - r) * chunk


@pytest.mark.parametrize("pcp", [2, 4])
def test_meta_partial_tail_clamps_per_rank(pcp):
    """With num_current < 2*pcp*C the late chunks are partly padding, and how
    much differs per rank -- the reason cu_q_lens must be pcp-sharded."""
    chunk, max_seqs = 8, 8
    padded = 2 * pcp * chunk
    num_current = padded - chunk - 3  # kills the last chunk, clips the next
    cu, qpos = build_headtail_pcp_meta(pcp, chunk, num_current, max_seqs)
    for r in range(pcp):
        tail_off = (2 * pcp - 1 - r) * chunk
        expected_tail = int(np.clip(num_current - tail_off, 0, chunk))
        assert cu[r, 1] == chunk
        assert cu[r, 2] == chunk + expected_tail
    # rank 0 owns the very last chunk, which is entirely padding here
    assert cu[0, 2] == chunk
    # and some rank must still have a partially-real tail
    assert any(chunk < cu[r, 2] < 2 * chunk for r in range(pcp))


def test_meta_rejects_bad_sizes():
    with pytest.raises(ValueError):
        build_headtail_pcp_meta(0, 4, 8, 8)
    with pytest.raises(ValueError):
        build_headtail_pcp_meta(2, 0, 8, 8)
