# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_MAIN_KV_DCP: per-rank decode RPA page metadata keeps exactly
the owned selected groups (and the owned partial tail) of each row, with page
ids from the shared ownership rule (entry t // (D*B), row t % B)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import virtual_page_metadata

RATIO, B, D, BUDGET = 4, 32, 2, 16  # small page: 8 groups per page

# The page-id lookup (sparse MLA's gather_page_ids) is a Pallas kernel.
pytestmark = pytest.mark.skipif(
    jax.default_backend() != "tpu", reason="gather_page_ids runs on TPU"
)


@pytest.mark.parametrize("seed", range(3))
def test_dcp_metadata_keeps_owned_groups(seed):
    rng = np.random.default_rng(seed)
    rows, entries = 12, 6
    table = rng.permutation(100)[: rows * entries].reshape(rows, entries).astype(np.int32)
    lengths = rng.integers(0, entries * D * B, size=rows).astype(np.int32)
    lengths[0] = 0
    blocks = np.full((rows, BUDGET), -5, np.int32)  # junk past count
    for i, L in enumerate(lengths):
        vis = L // RATIO
        k = min(vis, BUDGET)
        if k:
            blocks[i, :k] = rng.choice(vis, k, replace=False)
    page_groups = B // RATIO
    for rank in range(D):
        pages, kv_lens, cu, _ = (
            np.asarray(x)
            for x in virtual_page_metadata(
                jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table),
                ratio=RATIO, page_size=B, dcp=(jnp.array([rank], jnp.int32), D),
            )
        )
        for i, L in enumerate(lengths):
            complete = L // RATIO
            sel = blocks[i, : min(complete, BUDGET)]
            owned = sorted(g for g in sel if (g // page_groups) % D == rank)
            tail = L % RATIO and (complete // page_groups) % D == rank
            groups = owned + ([complete] if tail else [])
            expect_pages = [
                table[i, g // (D * page_groups)] * page_groups + g % page_groups
                for g in groups
            ]
            assert list(pages[i, : len(groups)]) == expect_pages, (rank, i)
            assert kv_lens[i] == len(owned) * RATIO + (L % RATIO if tail else 0)
        assert list(cu) == [0, *np.cumsum(kv_lens > 0)]


def test_dcp_off_is_unchanged():
    blocks = jnp.asarray([[3, 1, 0, -1]], jnp.int32)
    lengths = jnp.asarray([18], jnp.int32)
    table = jnp.asarray([[7, 9]], jnp.int32)
    a = virtual_page_metadata(blocks, lengths, table, ratio=RATIO, page_size=B)
    b = virtual_page_metadata(blocks, lengths, table, ratio=RATIO, page_size=B, dcp=None)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
