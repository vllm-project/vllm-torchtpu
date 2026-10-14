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
"""CPU tests for the sparse-MLA padding-slot hash."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla

# Enough slots to cover a 4096-token x 2048-topk prefill (int32 overflow regime).
_NUM_SLOTS = 1 << 16
_BASES = (0, 4096 * 2048 - _NUM_SLOTS)


def _hash(total_num_pages, page_size, base):
    idx = jnp.arange(_NUM_SLOTS, dtype=jnp.int32) + jnp.int32(base)
    fn = jax.jit(
        lambda x: sparse_mla.scatter_padding_slots(x, total_num_pages, page_size)
    )
    page, offset = fn(idx)
    return np.asarray(page), np.asarray(offset)


def _chi2_ok(values, num_buckets):
    counts = np.bincount(values, minlength=num_buckets)
    expected = values.size / num_buckets
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    dof = num_buckets - 1
    # ~6 sigma above the chi-square mean; a strided alias blows far past this.
    return chi2 < dof + 6 * np.sqrt(2 * dof), chi2


@pytest.mark.parametrize("total_num_pages", [256, 1000, 1536, 2048, 3001])
@pytest.mark.parametrize("page_size", [256, 64, 100])
@pytest.mark.parametrize("base", _BASES)
def test_pad_hash_in_range_and_uniform(total_num_pages, page_size, base):
    page, offset = _hash(total_num_pages, page_size, base)
    assert page.dtype == np.int32 and offset.dtype == np.int32
    assert page.min() >= 0 and page.max() < total_num_pages
    assert offset.min() >= 0 and offset.max() < page_size
    ok, chi2 = _chi2_ok(page, total_num_pages)
    assert ok, f"page chi2={chi2:.1f} over {total_num_pages} buckets"
    ok, chi2 = _chi2_ok(offset, page_size)
    assert ok, f"offset chi2={chi2:.1f} over {page_size} buckets"
    # Consecutive slots must not collapse onto few rows (random: >=88% unique).
    rows = page.astype(np.int64) * page_size + offset
    assert np.unique(rows[:4096]).size > 0.8 * 4096


def test_pad_hash_deterministic_and_eager_matches_jit():
    idx = jnp.arange(_NUM_SLOTS, dtype=jnp.int32) * jnp.int32(127)
    eager = sparse_mla.scatter_padding_slots(idx, 1536, 256)
    jitted = jax.jit(lambda x: sparse_mla.scatter_padding_slots(x, 1536, 256))(idx)
    again = jax.jit(lambda x: sparse_mla.scatter_padding_slots(x, 1536, 256))(idx)
    for a, b, c in zip(eager, jitted, again):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        np.testing.assert_array_equal(np.asarray(b), np.asarray(c))


@pytest.mark.parametrize(
    "use_page_table",
    [False, pytest.param(True, marks=pytest.mark.tpu_test)],
)
def test_prepare_gather_indices_pads_use_hash(use_page_table):
    """Valid slots are untouched; -1 slots land on in-range, spread-out rows."""
    if use_page_table and jax.devices()[0].platform != "tpu":
        pytest.skip("Pallas page-table lookup requires a TPU")
    rng = np.random.default_rng(0)
    page_size, total_num_pages, num_tokens, topk = 256, 1536, 64, 512
    topk_idx = rng.integers(0, 4 * page_size, (num_tokens, topk)).astype(np.int32)
    topk_idx[:, topk // 2 :] = -1
    page_indices = rng.permutation(total_num_pages)[:8].astype(np.int32)
    kv_lens, phys = sparse_mla.prepare_gather_indices(
        jnp.asarray(topk_idx),
        jnp.asarray(page_indices) if use_page_table else None,
        jnp.asarray(np.array([0, num_tokens], np.int32)),
        page_size=page_size,
        total_num_pages=total_num_pages,
    )
    phys = np.asarray(phys)
    np.testing.assert_array_equal(np.asarray(kv_lens), topk // 2)
    valid = topk_idx[:, : topk // 2]
    if use_page_table:
        valid = page_indices[valid // page_size] * page_size + valid % page_size
    np.testing.assert_array_equal(phys[:, : topk // 2], valid)
    pads = phys[:, topk // 2 :]
    flat = np.arange(num_tokens * topk, dtype=np.int32).reshape(num_tokens, topk)
    page, offset = sparse_mla.scatter_padding_slots(
        jnp.asarray(flat[:, topk // 2 :]), total_num_pages, page_size
    )
    np.testing.assert_array_equal(pads, np.asarray(page) * page_size + offset)
    assert pads.min() >= 0 and pads.max() < total_num_pages * page_size
    ok, chi2 = _chi2_ok((pads // page_size).ravel(), total_num_pages)
    assert ok, f"pad page chi2={chi2:.1f}"
