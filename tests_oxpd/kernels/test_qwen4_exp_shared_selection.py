# SPDX-License-Identifier: Apache-2.0
"""Pooled QSA attending another layer's selection (VLLM_TPU_QSA_DRAFT_SHARED_SELECTION).

Given the selection a step made itself, the shared step must reproduce that
step exactly: same pool writes, output and selected tokens, for all tokens
and for the draft's rows. It skips only the indexer scoring.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_qwen4_exp_mtp_draft_rows as draft_rows

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa
from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import selected_tokens

CONTEXT, OPTIONS, STARTS, _inputs = (
    draft_rows.CONTEXT,
    draft_rows.OPTIONS,
    draft_rows.STARTS,
    draft_rows._inputs,
)
rowwise_attention = draft_rows.rowwise_attention

# 64 blocks per row: fewer than the ~415 completed groups of CONTEXT, so the
# rows take the selection rather than every visible block.
SHARED_OPTIONS = dict(OPTIONS, token_topk=256)


def _bytes(x):
    return np.asarray(x).view(np.uint8)


@pytest.mark.parametrize("fp8", [False, True])
def test_shared_step_reproduces_the_step_from_its_own_selection(rowwise_attention, fp8):
    head, pool, tail = _inputs(fp8, STARTS, CONTEXT)
    full = jax.jit(lambda *a: pooled_qsa.pooled_qsa_step(*a, **SHARED_OPTIONS))
    shared = jax.jit(lambda *a: pooled_qsa.pooled_qsa_shared_step(*a, **SHARED_OPTIONS))
    want_pool, want_out, want_selected = full(*head, pool, *tail)
    got_pool, got_out, got_selected = shared(*head, pool, *tail, want_selected)
    np.testing.assert_array_equal(_bytes(got_pool), _bytes(want_pool))
    np.testing.assert_array_equal(np.asarray(got_out), np.asarray(want_out))
    np.testing.assert_array_equal(np.asarray(got_selected), np.asarray(want_selected))


@pytest.mark.parametrize("rows", [[2, 6, -1], [1, 6, 6]])
def test_shared_rows_step_reproduces_the_rows_step(rowwise_attention, rows):
    head, pool, tail = _inputs(False, STARTS, CONTEXT)
    rows = jnp.asarray(rows, jnp.int32)
    plain = jax.jit(lambda *a: pooled_qsa.pooled_qsa_rows_step(*a, **SHARED_OPTIONS))
    shared = jax.jit(
        lambda *a: pooled_qsa.pooled_qsa_shared_rows_step(*a, **SHARED_OPTIONS)
    )
    want_pool, want_out, want_selected = plain(*head, pool, *tail, rows)
    got_pool, got_out, got_selected = shared(*head, pool, *tail, rows, want_selected)
    np.testing.assert_array_equal(_bytes(got_pool), _bytes(want_pool))
    np.testing.assert_array_equal(np.asarray(got_out), np.asarray(want_out))
    np.testing.assert_array_equal(np.asarray(got_selected), np.asarray(want_selected))


def test_shared_blocks_take_the_selection_or_every_visible_block():
    ratio, topk = 4, 16
    budget = topk // ratio
    lengths = jnp.asarray([10, 64, 64], jnp.int32)  # 2 groups fit; 16 do not
    blocks = jnp.asarray([[0, 1, -1, -1], [9, 3, 12, 5], [7, -1, 2, 0]], jnp.int32)
    shared = selected_tokens(blocks, lengths, topk, ratio)
    got = np.asarray(pooled_qsa.shared_blocks(shared, lengths, ratio, topk))
    np.testing.assert_array_equal(
        got[0], np.asarray(pooled_qsa.visible_blocks(lengths, ratio, budget))[0]
    )
    np.testing.assert_array_equal(got[1], [9, 3, 12, 5])
    np.testing.assert_array_equal(got[2], [7, 0, 2, 0])  # an uncovered slot: group 0
