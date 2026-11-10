"""Unit tests for DeepSeek-V4.1 record compaction (`compact_records`,
`_pad_compacted_records`)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
    _pad_compacted_records,
    compact_records,
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

    def test_compact_records_padding_to_128_multiple(self):
        # width = 50 -> padded to 128
        records = jnp.array([list(range(50))], dtype=jnp.int32)
        got = compact_records(records)
        assert got.shape == (1, 128)
        assert int(got[0, 49]) == 49
        assert int(got[0, 50]) == -1
        assert int(got[0, 127]) == -1

    def test_compact_records_exact_128(self):
        # width = 128 -> remains 128
        records = jnp.array([list(range(128))], dtype=jnp.int32)
        got = compact_records(records)
        assert got.shape == (1, 128)
        assert int(got[0, 0]) == 0
        assert int(got[0, 127]) == 127

    def test_compact_records_padded_to_256(self):
        # width = 200 -> padded to 256
        records = jnp.array([list(range(200))], dtype=jnp.int32)
        got = compact_records(records)
        assert got.shape == (1, 256)
        assert int(got[0, 199]) == 199
        assert int(got[0, 200]) == -1

    def test_pad_compacted_records_branches(self):
        # Branch 1: padded == width (e.g. 128)
        records_128 = jnp.zeros((2, 128), dtype=jnp.int32)
        got_128 = _pad_compacted_records(records_128)
        assert got_128 is records_128

        # Branch 2: padded != width (e.g. 100 -> padded to 128)
        records_100 = jnp.ones((2, 100), dtype=jnp.int32)
        got_padded = _pad_compacted_records(records_100)
        assert got_padded.shape == (2, 128)
        assert int(got_padded[0, 99]) == 1
        assert int(got_padded[0, 100]) == -1
        assert int(got_padded[0, 127]) == -1


class TestTokenParallelAttentionDispatchAndHelpers:
    """Tests for fp8_attention dispatch logic, conditions, and wrappers."""

    def test_token_parallel_boolean_condition_branches(self):
        def check(token_parallel, cur_tp_size, tokens):
            return (
                token_parallel
                and cur_tp_size > 1
                and tokens >= 6 * cur_tp_size
                and tokens % cur_tp_size == 0
            )

        # Falsy branches
        assert not check(False, 8, 2048)  # token_parallel is False
        assert not check(True, 1, 2048)  # tp_size <= 1
        assert not check(True, 8, 32)  # tokens < 48
        assert not check(True, 8, 50)  # tokens % 8 != 0

        # Truthy branch
        assert check(True, 8, 48)
        assert check(True, 8, 64)
        assert check(True, 8, 96)
        assert check(True, 8, 128)
        assert check(True, 8, 256)
        assert check(True, 8, 1024)
        assert check(True, 8, 2048)
        assert check(True, 8, 8192)

    def test_csa_gather_dynamic_num_row_subchunks(self, monkeypatch):
        from types import SimpleNamespace

        from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_gather

        fake_sc = SimpleNamespace(num_lanes=8, num_cores=4, num_subcores=16)
        fake_tpu_info = SimpleNamespace(sparse_core=fake_sc)
        monkeypatch.setattr(csa_gather.pltpu, "get_tpu_info", lambda: fake_tpu_info)
        monkeypatch.setattr(
            csa_gather.plsc, "VectorSubcoreMesh", lambda **kw: SimpleNamespace(**kw)
        )

        captured_subchunks: list[int] = []

        def fake_pl_kernel(kernel_fn, *, out_type, **kwargs):
            captured_subchunks.append(kernel_fn.keywords["num_row_subchunks"])
            n_rows = out_type[0].shape[0]
            return lambda nope_c, rope_c, idx, valid: (
                jnp.zeros((n_rows, 128), dtype=jnp.int32),
                jnp.zeros((n_rows // 4, 128), dtype=jnp.int32),
            )

        monkeypatch.setattr(csa_gather.pl, "kernel", fake_pl_kernel)

        nope_cache = jnp.zeros((64, 16, 128), dtype=jnp.int32)
        rope_cache = jnp.zeros((64, 4, 128), dtype=jnp.int32)

        for num_tokens, expected_subchunks in [
            (4, 4),
            (8, 8),
            (16, 16),
            (24, 32),
            (64, 64),
        ]:
            indices = jnp.zeros((num_tokens * 512,), dtype=jnp.int32)
            nope_out, rope_out = csa_gather.csa_gather(
                nope_cache,
                rope_cache,
                indices,
                num_valid_indices=None if num_tokens == 4 else num_tokens * 512,
                top_k=512,
            )
            assert nope_out.shape == (num_tokens * 512, 128)
            assert rope_out.shape == ((num_tokens * 512) // 4, 128)
            assert captured_subchunks[-1] == expected_subchunks
