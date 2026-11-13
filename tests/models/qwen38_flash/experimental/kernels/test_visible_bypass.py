# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_VISIBLE_BYPASS: the unsharded indexer path takes every visible
group without scoring when all rows fit the budget, and scores otherwise."""

import jax
import numpy as np
import pytest

import tests.models.qwen38_flash.experimental.kernels.test_mtp_draft_rows as mtp_rows
from tests.models.qwen38_flash.experimental.kernels.test_mtp_draft_rows import (
    rowwise_attention,  # noqa: F401
)
from vllm_torchtpu.models.qwen38_flash.experimental.kernels import pooled_qsa


def _step(monkeypatch, flag, token_topk):
    monkeypatch.setenv("VLLM_TPU_QSA_VISIBLE_BYPASS", flag)
    head, pool, tail = mtp_rows._inputs(False, mtp_rows.STARTS, mtp_rows.CONTEXT)
    options = dict(mtp_rows.OPTIONS, token_topk=token_topk)
    return jax.jit(lambda *args: pooled_qsa.pooled_qsa_step(*args, **options))(
        *head, pool, *tail
    )


def _same(a, b):
    for x, y in zip(a, b):
        np.testing.assert_array_equal(
            np.asarray(x).view(np.uint8), np.asarray(y).view(np.uint8)
        )


def test_bypass_when_everything_fits(monkeypatch, rowwise_attention):  # noqa: F811
    calls = []
    stand_in = pooled_qsa.select_blocks

    def visible(query, pool, lengths, table, req, starts, context, **kw):
        calls.append(1)
        return pooled_qsa.visible_blocks(lengths, kw["ratio"], kw["block_topk"])

    # Contexts ~1.66K tokens: 416 groups fit a 2048-token (512-group) budget.
    monkeypatch.setattr(pooled_qsa, "select_blocks", visible)
    want = _step(monkeypatch, "0", 2048)
    assert calls
    # The scorer is traced (lax.cond) but the visible branch must win: a
    # scorer returning other blocks would change the outputs.
    monkeypatch.setattr(pooled_qsa, "select_blocks", stand_in)
    got = _step(monkeypatch, "1", 2048)
    _same(got, want)


@pytest.mark.parametrize("token_topk", [256])
def test_scores_when_a_row_exceeds_the_budget(
    monkeypatch,
    rowwise_attention,  # noqa: F811
    token_topk,  # noqa: F811
):
    _same(_step(monkeypatch, "1", token_topk), _step(monkeypatch, "0", token_topk))
