"""Unit tests for DeepSeek-V4.1 record compaction (`compact_records`,
`_pad_compacted_records`)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

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
