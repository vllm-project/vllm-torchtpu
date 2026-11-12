"""Decode indexer coverage: indexer_select._visible_top_k SparseCore and
fallback branches."""

import jax.numpy as jnp

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
    indexer_select,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_select import _visible_top_k


class TestAttentionKernelsDecodeIndexer:
    """Coverage for v2/attention_kernels.py / changes."""

    def test_visible_top_k_sparsecore_and_fallback_branches(self, monkeypatch):
        """Covers _visible_top_k unpadded rows < 16, rows >= 16, and fallback."""
        sparsecore_calls = []

        def mock_sc_top_k(scores, k, lengths):
            sparsecore_calls.append((scores.shape, k))
            return jnp.zeros((scores.shape[0], k), dtype=jnp.int32)

        monkeypatch.setattr(indexer_select, "_sparsecore_top_k", mock_sc_top_k)

        bound = jnp.int32(2048)

        # 1. SparseCore branch with rows < 16 (e.g. rows = 3, unpadded)
        scores_3 = jnp.zeros((3, 2048), dtype=jnp.float32)
        lengths_3 = jnp.array([100, 200, 300], dtype=jnp.int32)
        out_3 = _visible_top_k(scores_3, 16, bound, lengths_3, unordered=True)
        assert out_3.shape == (3, 16)
        assert sparsecore_calls[-1] == ((3, 2048), 16)

        # 2. SparseCore branch with rows >= 16 (e.g. rows = 16)
        scores_16 = jnp.zeros((16, 2048), dtype=jnp.float32)
        lengths_16 = jnp.full((16,), 100, dtype=jnp.int32)
        out_16 = _visible_top_k(scores_16, 32, bound, lengths_16, unordered=True)
        assert out_16.shape == (16, 32)
        assert sparsecore_calls[-1] == ((16, 2048), 32)

        # 3. Fallback branch: unordered=False (calls select via switch)
        scores_fb = jnp.zeros((2, 2048), dtype=jnp.float32)
        lengths_fb = jnp.array([10, 20], dtype=jnp.int32)
        out_fb = _visible_top_k(scores_fb, 16, bound, lengths_fb, unordered=False)
        assert out_fb.shape == (2, 16)

        # 4. Fallback branch: width < 2048
        scores_narrow = jnp.zeros((2, 1024), dtype=jnp.float32)
        out_narrow = _visible_top_k(
            scores_narrow, 16, jnp.int32(1024), lengths_fb, unordered=True
        )
        assert out_narrow.shape == (2, 16)

        # 5. Fallback branch: width % 16 != 0
        scores_unaligned_w = jnp.zeros((2, 2050), dtype=jnp.float32)
        out_unaligned_w = _visible_top_k(
            scores_unaligned_w, 16, jnp.int32(2050), lengths_fb, unordered=True
        )
        assert out_unaligned_w.shape == (2, 16)

        # 6. Fallback branch: k % 16 != 0
        out_unaligned_k = _visible_top_k(
            scores_fb, 10, bound, lengths_fb, unordered=True
        )
        assert out_unaligned_k.shape == (2, 10)

        # 7. Fallback branch: lengths is None
        out_no_len = _visible_top_k(scores_fb, 16, bound, None, unordered=True)
        assert out_no_len.shape == (2, 16)

        # 8. Fallback branch: scores.shape[0] >= 16 and columns >= 2048
        # (triggers _threshold_top_k)
        scores_thresh = jnp.zeros((16, 2048), dtype=jnp.float32)
        out_thresh = _visible_top_k(
            scores_thresh, 16, bound, lengths_16, unordered=False
        )
        assert out_thresh.shape == (16, 16)
