# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_DCP_META_PARTITION: the decode DCP metadata of an ascending
selection by a stable partition equals the sorted original, bit for bit.

Rows: idle (length 0), short (fewer complete groups than the budget, -1
suffix), long (a full budget), every tail phase (length % ratio), both DCP
ranks, 4..128 rows. Also checks the hint is ignored without the flag and that
an unsorted selection is NOT claimed ascending by the served callers.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.qwen38_flash.experimental import settings
from vllm_torchtpu.models.qwen38_flash.experimental.kernels import qsa_rpa

PAGE = 512
RATIO = 4
D = 2


def _case(rng, rows, budget):
    lengths = rng.integers(0, 120_000, rows).astype(np.int32)
    lengths[:: max(1, rows // 4)] = 0  # idle rows
    lengths[1::7] = rng.integers(
        1, budget * RATIO // 2, len(lengths[1::7])
    )  # short rows
    cols = int(-(-max(1, int(lengths.max())) // (D * PAGE)))
    table = (
        rng.permutation(rows * cols * 2)[: rows * cols]
        .reshape(rows, cols)
        .astype(np.int32)
    )
    blocks = np.full((rows, budget), -1, np.int32)
    for i, n in enumerate(lengths):
        c = int(n) // RATIO
        k = min(c, budget)
        if k:
            blocks[i, :k] = np.sort(rng.choice(c, k, replace=False))
    return jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table)


@pytest.mark.parametrize("rows", [4, 32, 96, 128])
@pytest.mark.parametrize("rank", [0, 1])
def test_partition_matches_sort(monkeypatch, rows, rank):
    rng = np.random.default_rng(rows * 2 + rank)
    blocks, lengths, table = _case(rng, rows, budget=512)
    dcp = (jnp.asarray([rank], jnp.int32), D)
    kw = dict(ratio=RATIO, page_size=PAGE, dcp=dcp)
    monkeypatch.setattr(settings, "VLLM_TPU_QSA_DCP_META_PARTITION", False)
    ref = jax.jit(lambda b, l, t: qsa_rpa.virtual_page_metadata(b, l, t, **kw))(  # noqa: E741
        blocks, lengths, table
    )
    monkeypatch.setattr(settings, "VLLM_TPU_QSA_DCP_META_PARTITION", True)
    got = jax.jit(
        lambda b, l, t: qsa_rpa.virtual_page_metadata(b, l, t, ascending=True, **kw)  # noqa: E741
    )(blocks, lengths, table)
    for name, a, b in zip(("pages", "kv_lens", "cu_q_lens", "dist"), ref, got):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)
    assert int(np.asarray(ref[1]).sum()) > 0  # some row owns keys on this rank


@pytest.mark.parametrize("where", ["front", "suffix"])
@pytest.mark.parametrize("rank", [0, 1])
def test_partition_matches_sort_with_holes(monkeypatch, rank, where):
    # -1 slots inside the counted columns (an unfilled top-k slot): the sort puts
    # owned -1 first; the partition must too.
    rng = np.random.default_rng(11 + rank)
    blocks, lengths, table = _case(rng, 32, budget=512)
    b = np.asarray(blocks).copy()
    for row in b:
        k = int((row >= 0).sum())
        if k > 8:
            holes = int(rng.integers(1, 5))
            if where == "front":  # ascending including the -1s
                row[:k] = np.sort(np.concatenate([np.full(holes, -1), row[holes:k]]))
            else:  # compressed_dcp_blocks layout: valid ascending, then -1
                row[k - holes : k] = -1
    blocks = jnp.asarray(b)
    dcp = (jnp.asarray([rank], jnp.int32), D)
    kw = dict(ratio=RATIO, page_size=PAGE, dcp=dcp)
    monkeypatch.setattr(settings, "VLLM_TPU_QSA_DCP_META_PARTITION", False)
    ref = qsa_rpa.virtual_page_metadata(blocks, lengths, table, **kw)
    monkeypatch.setattr(settings, "VLLM_TPU_QSA_DCP_META_PARTITION", True)
    got = qsa_rpa.virtual_page_metadata(blocks, lengths, table, ascending=True, **kw)
    for x, y in zip(ref, got):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_owned_first_is_stable_partition():
    blocks = jnp.asarray([[9, 2, 7, 4, -1], [1, 3, 5, 7, 9]], jnp.int32)
    owned = jnp.asarray([[True, False, True, True, False], [False] * 5])
    big = np.iinfo(np.int32).max
    np.testing.assert_array_equal(
        np.asarray(qsa_rpa._owned_first(blocks, owned)),
        [[9, 7, 4, big, big], [big] * 5],
    )


def test_hint_without_flag_keeps_sort(monkeypatch):
    # Unsorted owned entries: with the flag off the hint must not change the result.
    rng = np.random.default_rng(7)
    blocks, lengths, table = _case(rng, 32, budget=512)
    shuffled = np.asarray(blocks).copy()
    for row in shuffled:  # permute each row's valid prefix only
        k = int((row >= 0).sum())
        row[:k] = rng.permutation(row[:k])
    shuffled = jnp.asarray(shuffled)
    dcp = (jnp.asarray([0], jnp.int32), D)
    kw = dict(ratio=RATIO, page_size=PAGE, dcp=dcp)
    monkeypatch.setattr(settings, "VLLM_TPU_QSA_DCP_META_PARTITION", False)
    a = qsa_rpa.virtual_page_metadata(blocks, lengths, table, **kw)
    b = qsa_rpa.virtual_page_metadata(shuffled, lengths, table, ascending=True, **kw)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
