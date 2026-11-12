# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_SORTED_SEARCH_MASK: the compare-and-sum search equals jnp.searchsorted."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import sorted_search


def _boundaries(rng, n, hi):
    # Request starts: non-decreasing, repeats for empty requests and padding.
    lens = rng.integers(0, 64, size=n)
    lens[rng.random(n) < 0.3] = 0
    b = np.concatenate([[0], np.cumsum(lens)])
    return np.minimum(b, hi).astype(np.int32)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("nreq,tokens", [(1, 8), (24, 256), (24, 1024), (33, 3072), (64, 8192)])
def test_matches_searchsorted(monkeypatch, side, nreq, tokens):
    monkeypatch.setenv("VLLM_TPU_SORTED_SEARCH_MASK", "1")
    rng = np.random.default_rng(nreq * 7 + tokens)
    for _ in range(5):
        b = jnp.asarray(_boundaries(rng, nreq, tokens))
        x = jnp.arange(tokens + 16, dtype=jnp.int32) - 8  # below, inside and past the range
        want = jnp.searchsorted(b, x, side=side)
        got = sorted_search.searchsorted(b, x, side=side)
        assert got.dtype == want.dtype
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        jitted = jax.jit(lambda b, x: sorted_search.searchsorted(b[1:], x, side=side))
        np.testing.assert_array_equal(
            np.asarray(jitted(b, x)), np.asarray(jnp.searchsorted(b[1:], x, side=side))
        )


def test_off_and_fallbacks_use_searchsorted(monkeypatch):
    b = jnp.asarray([0, 3, 3, 9], jnp.int32)
    x = jnp.arange(12, dtype=jnp.int32)
    for flag in ("0", "1"):
        monkeypatch.setenv("VLLM_TPU_SORTED_SEARCH_MASK", flag)
        np.testing.assert_array_equal(
            sorted_search.searchsorted(b, x, side="right"), jnp.searchsorted(b, x, side="right")
        )
    monkeypatch.setenv("VLLM_TPU_SORTED_SEARCH_MASK", "1")
    fb = jnp.asarray([0.0, 1.5, 4.0])  # float boundaries keep the search
    np.testing.assert_array_equal(
        sorted_search.searchsorted(fb, x), jnp.searchsorted(fb, x)
    )
    big = jnp.arange(sorted_search.MAX_MASK_ELEMENTS // 1000 + 1, dtype=jnp.int32)
    xs = jnp.arange(1000, dtype=jnp.int32)
    np.testing.assert_array_equal(sorted_search.searchsorted(big, xs), jnp.searchsorted(big, xs))
    with pytest.raises(ValueError):
        sorted_search.searchsorted(b, x, side="middle")


def test_patched_modules_import():
    import vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk  # noqa: F401
    import vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill  # noqa: F401
    import vllm_torchtpu.kernels.qwen4_exp.qsa_union_dense  # noqa: F401
