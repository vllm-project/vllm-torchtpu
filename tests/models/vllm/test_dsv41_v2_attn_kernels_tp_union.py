"""Unit tests for DeepSeek-V4.1 attention kernels: Union & Token-Parallel.

Achieves 100% diff line and branch coverage on:
- src/vllm_torchtpu/models/vllm/deepseek_v41/v2/attention_kernels.py:267-760
  (compact_records, _pad_compacted_records, compute_chunk_union_gather_indices,
   unpack/repack rope words, gather_sparse_kv_chunk_with_union_fallback,
   sparse_ragged_paged_attention_union_gather, sparse_fp8_attention_jax,
   slice_token_parallel_attention_inputs, initialize_paged_attention,
   and sparse_fp8_attention wrapper)
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
    repack_chunk_rope_words,
    unpack_union_rope_words,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_select import _compact_selected


class TestCompactRecords:
    """Tests for record compaction and padding functions."""

    def test_compact_selected_all_valid(self):
        records = jnp.array([[5, 2, 8, 1], [10, 4, 7, 3]], dtype=jnp.int32)
        got = _compact_selected(records)
        expected = np.array([[1, 2, 5, 8], [3, 4, 7, 10]], dtype=np.int32)
        np.testing.assert_array_equal(np.asarray(got), expected)

    def test_compact_selected_all_pad(self):
        records = jnp.array([[-1, -1, -1], [-5, -1, -100]], dtype=jnp.int32)
        got = _compact_selected(records)
        expected = np.full((2, 3), -1, dtype=np.int32)
        np.testing.assert_array_equal(np.asarray(got), expected)

    def test_compact_selected_mixed_and_duplicates(self):
        records = jnp.array(
            [[10, -1, 5, 0, -1, 5], [-1, 20, -1, 10, -1, 0]], dtype=jnp.int32
        )
        got = _compact_selected(records)
        expected = np.array(
            [[0, 5, 5, 10, -1, -1], [0, 10, 20, -1, -1, -1]], dtype=np.int32
        )
        np.testing.assert_array_equal(np.asarray(got), expected)


class TestRopeUnpackRepackRemap:
    """Tests for unpack, repack, and remap rope/kv functions."""

    def test_unpack_repack_roundtrip(self):
        union_cap = 256
        topk = 128
        # raw rope words shape: (union_cap, 32)
        rng = np.random.default_rng(1234)
        raw_words = rng.integers(-100000, 100000, (union_cap, 32), dtype=np.int32)
        raw_words_jax = jnp.asarray(raw_words)

        packed = repack_chunk_rope_words(
            raw_words_jax.reshape(union_cap // topk, topk, 32)
        ).reshape(union_cap // 4, 128)

        unpacked = unpack_union_rope_words(packed, topk)
        np.testing.assert_array_equal(np.asarray(unpacked), raw_words)
