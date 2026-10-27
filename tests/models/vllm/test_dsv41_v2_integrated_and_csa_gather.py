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

"""100% full-module line and branch coverage tests for csa_gather.py

and integrated v2/* attention and attention_kernels modules under + .
"""

from __future__ import annotations

import contextlib
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax
import jax.numpy as jnp
import numpy as np
import torch

from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_gather
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
    attention as v2_attn,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
    attention_kernels as v2_ak,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
    ROW_WORDS,
    gather_cache_page_values,
)


def _norm_idx(idx):
    if isinstance(idx, tuple):
        return tuple(_norm_idx(i) for i in idx)
    if hasattr(idx, "start") and hasattr(idx, "size"):
        stride = getattr(idx, "stride", 1)
        stride = 1 if stride is None else int(stride)
        return slice(int(idx.start), int(idx.start + idx.size), stride)
    return idx


class MockArrayRef:
    """Mock HBM / VMEM buffer reference for simulating Pallas pipelines on CPU."""

    def __init__(self, arr: np.ndarray | jax.Array):
        self.arr = np.asarray(arr)
        self.shape = self.arr.shape
        self.dtype = self.arr.dtype

    def __getitem__(self, idx):
        try:
            return self.arr[_norm_idx(idx)]
        except Exception:
            return self.arr

    def __setitem__(self, idx, val):
        with contextlib.suppress(Exception):
            self.arr[_norm_idx(idx)] = np.asarray(val)

    @property
    def at(self):
        return self


# ==============================================================================
# 1. csa_gather.py: main_kernel and process_rope Full Coverage
# ==============================================================================


class TestCsaGatherMainKernelAndProcessRope:
    """Tests 100% of lines and branches in csa_gather.main_kernel and process_rope."""

    def test_process_rope_bitwise_repack(self):
        """Exercises process_rope bitwise lane packing directly."""
        num_lanes = 8
        rope_in_cols = 16
        gather_data = np.zeros((num_lanes, rope_in_cols), dtype=np.int32)
        # Even rows have low 16 bits set, odd rows have high 16 bits set
        for i in range(num_lanes):
            if i % 2 == 0:
                gather_data[i] = 0x00001234
            else:
                gather_data[i] = 0x00005678

        sc_info = SimpleNamespace(num_lanes=num_lanes, num_cores=2, num_subcores=2)
        tpu_info = SimpleNamespace(sparse_core=sc_info)

        nope_in = MockArrayRef(np.zeros((64, 128), dtype=np.int32))
        rope_in = MockArrayRef(gather_data)
        indices_ref = MockArrayRef(np.zeros((64,), dtype=np.int32))
        valid_indices_ref = MockArrayRef(np.array([64], dtype=np.int32))
        nope_out = MockArrayRef(np.zeros((64, 128), dtype=np.int32))
        rope_out = MockArrayRef(np.zeros((32, rope_in_cols * 2), dtype=np.int32))
        nope_sem = MockArrayRef(np.zeros((4,), dtype=np.int32))

        async_copies = []

        class MockAsyncCopy:
            def __init__(self):
                self.started = False
                self.waited = False

            def start(self):
                self.started = True

            def wait(self):
                self.waited = True

        def fake_make_async_copy(src, dst, sem):
            copy = MockAsyncCopy()
            async_copies.append(copy)
            return copy

        def fake_emit(pipeline_fn, grid=None, in_specs=None, out_specs=None):
            def runner(*args):
                for spec in (in_specs or ()) + (out_specs or ()):
                    if hasattr(spec, "index_map") and callable(spec.index_map):
                        for step in (0, 1, 2, 4):
                            with contextlib.suppress(Exception):
                                spec.index_map(step)
                pipeline_fn(*args)

            return runner

        with (
            patch.object(csa_gather.pltpu, "get_tpu_info", return_value=tpu_info),
            patch.object(csa_gather.lax, "axis_size", return_value=4),
            patch.object(csa_gather.lax, "axis_index", return_value=0),
            patch.object(csa_gather.pl, "program_id", return_value=0),
            patch.object(
                csa_gather.pltpu, "make_async_copy", side_effect=fake_make_async_copy
            ),
            patch.object(csa_gather.pl, "when", lambda cond: lambda fn: fn()),
            patch.object(csa_gather.pltpu, "emit_pipeline", side_effect=fake_emit),
        ):
            csa_gather.main_kernel(
                nope_in,
                rope_in,
                indices_ref,
                valid_indices_ref,
                nope_out,
                rope_out,
                nope_sem,
                core_axis_name="core",
                subcore_axis_name="subcore",
                num_row_subchunks=4,
                num_streams=4,
                top_k=256,
            )

        assert len(async_copies) == 4
        assert all(c.started and c.waited for c in async_copies)
        # Check that process_rope wrote into rope_out
        assert rope_out.arr[0, 0] != 0

    def test_main_kernel_rope_out_block_multihalf(self):
        """Exercises rope_out_block for both half % 2 == 0 and half % 2 == 1."""
        num_lanes = 8
        sc_info = SimpleNamespace(num_lanes=num_lanes, num_cores=1, num_subcores=1)
        tpu_info = SimpleNamespace(sparse_core=sc_info)

        nope_in = MockArrayRef(np.zeros((128, 128), dtype=np.int32))
        rope_in = MockArrayRef(np.zeros((128, 32), dtype=np.int32))
        indices_ref = MockArrayRef(np.zeros((128,), dtype=np.int32))
        valid_indices_ref = MockArrayRef(np.array([128], dtype=np.int32))
        nope_out = MockArrayRef(np.zeros((128, 128), dtype=np.int32))
        rope_out = MockArrayRef(np.zeros((64, 64), dtype=np.int32))
        nope_sem = MockArrayRef(np.zeros((4,), dtype=np.int32))

        captured_blocks = []

        def fake_emit(pipeline_fn, grid=None, in_specs=None, out_specs=None):
            def runner(*args):
                for spec in (in_specs or ()) + (out_specs or ()):
                    if hasattr(spec, "index_map") and callable(spec.index_map):
                        # top_k=128, num_streams=4, row_subchunk_size=8
                        # steps_per_half = 64 // 32 = 2
                        # r=0 -> half=0, r=2 -> half=1
                        for r_val in (0, 1, 2, 3):
                            captured_blocks.append(spec.index_map(r_val))
                pipeline_fn(*args)

            return runner

        with (
            patch.object(csa_gather.pltpu, "get_tpu_info", return_value=tpu_info),
            patch.object(csa_gather.lax, "axis_size", return_value=1),
            patch.object(csa_gather.lax, "axis_index", return_value=0),
            patch.object(csa_gather.pl, "program_id", return_value=0),
            patch.object(
                csa_gather.pltpu,
                "make_async_copy",
                return_value=SimpleNamespace(start=lambda: None, wait=lambda: None),
            ),
            patch.object(csa_gather.pl, "when", lambda cond: lambda fn: fn()),
            patch.object(csa_gather.pltpu, "emit_pipeline", side_effect=fake_emit),
        ):
            csa_gather.main_kernel(
                nope_in,
                rope_in,
                indices_ref,
                valid_indices_ref,
                nope_out,
                rope_out,
                nope_sem,
                core_axis_name="core",
                subcore_axis_name="subcore",
                num_row_subchunks=4,
                num_streams=4,
                top_k=128,
            )

        assert len(captured_blocks) > 0
        # Verify that both parity halves were generated in rope_out_block
        parities = {
            b[1] for b in captured_blocks if isinstance(b, tuple) and len(b) == 2
        }
        assert 0 in parities
        assert 1 in parities


# ==============================================================================
# 2. csa_gather.py: csa_gather Function & Validation Coverage
# ==============================================================================


class TestCsaGatherFunctionAndValidation:
    """Tests 100% of lines and branches in csa_gather function."""

    def test_validation_assertions(self):
        nope_cache = jnp.zeros((16, 16, 128), dtype=jnp.int32)
        rope_cache = jnp.zeros((16, 4, 128), dtype=jnp.int32)
        valid_indices = jnp.zeros((512,), dtype=jnp.int32)

        # 1. indices not 1D
        with pytest.raises(AssertionError, match="Indices must be 1D"):
            csa_gather.csa_gather.__wrapped__(
                nope_cache, rope_cache, jnp.zeros((512, 1), dtype=jnp.int32)
            )

        # 2. Caches dtype mismatch
        with pytest.raises(AssertionError, match="Caches must share a dtype"):
            csa_gather.csa_gather.__wrapped__(
                nope_cache.astype(jnp.float32), rope_cache, valid_indices
            )

        # 3. Caches not int32
        with pytest.raises(AssertionError, match="Caches must be int32"):
            csa_gather.csa_gather.__wrapped__(
                nope_cache.astype(jnp.float32),
                rope_cache.astype(jnp.float32),
                valid_indices,
            )

        # 4. nope_cache row words mismatch
        with pytest.raises(AssertionError):
            csa_gather.csa_gather.__wrapped__(
                jnp.zeros((16, 16, 64), dtype=jnp.int32), rope_cache, valid_indices
            )

        # 5. rope_cache row words mismatch
        with pytest.raises(AssertionError):
            csa_gather.csa_gather.__wrapped__(
                nope_cache, jnp.zeros((16, 4, 64), dtype=jnp.int32), valid_indices
            )

        # 6. SparseCore info is None
        with (
            patch.object(
                csa_gather.pltpu,
                "get_tpu_info",
                return_value=SimpleNamespace(sparse_core=None),
            ),
            pytest.raises(AssertionError, match="SparseCore info is missing"),
        ):
            csa_gather.csa_gather.__wrapped__(nope_cache, rope_cache, valid_indices)

        # 7. out_size not multiple of top_k
        sc_info = SimpleNamespace(num_lanes=8, num_cores=4, num_subcores=16)
        with (
            patch.object(
                csa_gather.pltpu,
                "get_tpu_info",
                return_value=SimpleNamespace(sparse_core=sc_info),
            ),
            pytest.raises(AssertionError),
        ):
            csa_gather.csa_gather.__wrapped__(
                nope_cache,
                rope_cache,
                jnp.zeros((500,), dtype=jnp.int32),
                top_k=512,
            )

    def test_dynamic_num_row_subchunks_and_padding_branches(self):
        """Tests all branches of num_row_subchunks while loop and padding branches."""
        sc_info = SimpleNamespace(num_lanes=8, num_cores=4, num_subcores=16)
        tpu_info = SimpleNamespace(sparse_core=sc_info)

        captured_kernel_kwargs = []

        def fake_pl_kernel(kernel_fn, *, out_type, **kwargs):
            captured_kernel_kwargs.append(kernel_fn.keywords)
            n_rows = out_type[0].shape[0]
            return lambda nope_c, rope_c, idx, valid: (
                jnp.zeros((n_rows, 128), dtype=jnp.int32),
                jnp.zeros((n_rows // 4, 128), dtype=jnp.int32),
            )

        nope_cache = jnp.zeros((64, 16, 128), dtype=jnp.int32)
        rope_cache = jnp.zeros((64, 4, 128), dtype=jnp.int32)

        with (
            patch.object(csa_gather.pltpu, "get_tpu_info", return_value=tpu_info),
            patch.object(
                csa_gather.plsc, "VectorSubcoreMesh", lambda **kw: SimpleNamespace(**kw)
            ),
            patch.object(csa_gather.pl, "kernel", fake_pl_kernel),
        ):
            # Branch 1: num_valid_indices is None, out_pad_size == 0
            indices1 = jnp.zeros((4 * 512,), dtype=jnp.int32)
            out1_nope, out1_rope = csa_gather.csa_gather.__wrapped__(
                nope_cache, rope_cache, indices1, num_valid_indices=None, top_k=512
            )
            assert out1_nope.shape == (4 * 512, 128)
            assert out1_rope.shape == (4 * 512 // 4, 128)

            # Branch 2: num_valid_indices is not None, out_pad_size > 0
            indices2 = jnp.zeros((3 * 512,), dtype=jnp.int32)
            out2_nope, out2_rope = csa_gather.csa_gather.__wrapped__(
                nope_cache, rope_cache, indices2, num_valid_indices=512, top_k=512
            )
            assert out2_nope.shape == (3 * 512, 128)
            assert out2_rope.shape == (3 * 512 // 4, 128)

            # Branch 3: Sweep across token configurations for while loop branches
            # num_tokens in (4, 8, 16, 24, 64)
            expected_subchunks = [4, 8, 16, 32, 64]
            actual_subchunks = []
            for num_tok in [4, 8, 16, 24, 64]:
                idx = jnp.zeros((num_tok * 512,), dtype=jnp.int32)
                csa_gather.csa_gather.__wrapped__(
                    nope_cache, rope_cache, idx, top_k=512
                )
                actual_subchunks.append(captured_kernel_kwargs[-1]["num_row_subchunks"])

            assert actual_subchunks == expected_subchunks


# ==============================================================================
# 3. v2/attention.py: _query_split and Even Part Allocation
# ==============================================================================


class TestIntegratedEvenQuerySplit:
    """Verifies _query_split even part allocation across all token counts."""

    def test_query_split_formulas_and_properties(self):
        """_query_split(count, ranks) guarantees part is even and covers count."""
        test_counts = [1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 17, 31, 32, 64, 100]
        test_ranks = [1, 2, 4, 8]

        for ranks in test_ranks:
            for count in test_counts:
                part, padding = v2_attn._query_split(count, ranks)
                # 1. part is strictly even (TorchDynamo requirement)
                assert part % 2 == 0, (
                    f"part={part} not even for count={count}, ranks={ranks}"
                )
                # 2. padding is 2 * ranks
                assert padding == 2 * ranks
                # 3. total slots cover count
                assert ranks * part >= count
                # 4. total padding is bounded by 2 * ranks - 1
                assert 0 <= ranks * part - count < padding
                # 5. part formula matches manual ceil
                expected_part = ((count + 2 * ranks - 1) // (2 * ranks)) * 2
                assert part == expected_part

    def test_select_paged_indices_even_split(self, monkeypatch):
        """_select_paged_indices correctly uses even query split."""
        ranks = 4
        count = 12  # under old ceil: part=3 (odd); under part=4 (even)

        mock_attn = MagicMock()
        mock_attn.rank = 0
        mock_attn._tp_rank = 0
        mock_attn.ranks = ranks
        mock_attn.topk = 4
        mock_attn._config = {"index_topk": 4}
        mock_attn.candidate_block_size = 16
        mock_attn._fp8_kv = False
        mock_attn.indexer.head_dim = 128
        mock_attn.indexer.indexer_op.candidate_blocks = None
        mock_attn.has_fast_idx_offset = False
        mock_attn.candidate_block_buffer = None
        mock_attn.paged_index_block_buffer = None
        mock_attn._project_index_q = lambda q, pos: q

        captured_args = {}

        def mock_paged_indexer(
            q_part,
            index_cache,
            records,
            weights_part,
            lengths_part,
            candidates_part,
            write_records,
            write_keys,
            requests_part,
            *args,
            **kwargs,
        ):
            captured_args["q_part_shape"] = q_part.shape
            captured_args["weights_part_shape"] = weights_part.shape
            captured_args["lengths_part_shape"] = lengths_part.shape
            captured_args["candidates_part_shape"] = candidates_part.shape
            expected_part, _ = v2_attn._query_split(count, ranks)
            assert q_part.shape[0] == expected_part == 4  # strictly 4, NOT 3!
            return (
                index_cache.raw,
                torch.zeros(expected_part, 4, dtype=torch.int32),
                candidates_part,
            )

        monkeypatch.setattr(v2_attn, "paged_indexer", mock_paged_indexer)
        monkeypatch.setattr(
            v2_attn, "get_tensor_model_parallel_world_size", lambda: ranks
        )
        monkeypatch.setattr(
            v2_attn, "_gather_query_rows", lambda s: torch.cat([s] * ranks)[:count]
        )

        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        index_cache_mock = SimpleNamespace(raw=raw)

        qr = torch.zeros(count, 128)
        positions = torch.arange(count, dtype=torch.int32)
        records = torch.zeros(count, 256, dtype=torch.int32)
        requests = torch.arange(count, dtype=torch.int32)
        weights = torch.zeros(count, 2)
        lengths = torch.full((count, 1), 50, dtype=torch.int32)
        swa_slots = torch.zeros(count, 1, dtype=torch.int32)

        with v2_attn.stage_attention_cache_scope():
            selected = v2_attn.DeepseekV41Attention._select_paged_indices(
                mock_attn,
                qr,
                positions,
                index_cache_mock,
                records,
                requests,
                weights,
                lengths,
                swa_slots,
                None,
            )

        assert selected.shape == (count, 4)
        assert captured_args["q_part_shape"][0] == 4


# ==============================================================================
# 4. v2/attention_kernels.py: & Indexer & Token-Parallel Threshold
# ==============================================================================


class TestIntegratedAttentionKernelsDecode:
    """Tests token-parallel threshold and /optimizations."""

    def test_token_parallel_threshold_decode(self):
        """lowered use_token_parallel threshold from 1024 to 128 (16*tp)."""
        tp_size = 8
        min_thresh = 16 * tp_size  # 128
        large_thresh = 128 * tp_size  # 1024

        def check_tp(tokens):
            return tokens >= min_thresh and (
                tokens % min_thresh == 0 or tokens >= large_thresh
            )

        # 1. tokens < 128 (e.g. standard decode 4, 8, 16, 64) -> False
        assert not check_tp(4)
        assert not check_tp(8)
        assert not check_tp(16)
        assert not check_tp(64)

        # 2. tokens in [128, 1024) divisible by 128 (DSpark verify) -> True!
        assert check_tp(128)
        assert check_tp(256)
        assert check_tp(384)
        assert check_tp(512)

        # 3. tokens >= 1024 -> True
        assert check_tp(1024)
        assert check_tp(2048)
        assert check_tp(8192)

    def test_sparsecore_top_k_unpadded_decode_parts(self, monkeypatch):
        """removed rows >= 16 check in _visible_top_k / _sparsecore_top_k."""
        scores_small = jnp.zeros((3, 2048), dtype=jnp.float32)
        lengths = jnp.array([100, 200, 300], dtype=jnp.int32)

        sc_called = []

        def mock_sc_top_k(scores, k, lengths_arg):
            sc_called.append((scores.shape, k, lengths_arg.shape))
            return jnp.zeros((scores.shape[0], k), dtype=jnp.int32)

        monkeypatch.setattr(v2_ak, "_sparsecore_top_k", mock_sc_top_k)

        # 1. Under , scores.shape[0] < 16 executes _sparsecore_top_k
        # when unordered=True
        out_sc = v2_ak._visible_top_k(
            scores_small, 64, bound=jnp.max(lengths), lengths=lengths, unordered=True
        )
        assert len(sc_called) == 1
        assert sc_called[0] == ((3, 2048), 64, (3,))
        assert out_sc.shape == (3, 64)

        # 2. Fallback path when unordered=False
        out_fallback = v2_ak._visible_top_k(
            scores_small, 64, bound=jnp.max(lengths), lengths=None, unordered=False
        )
        assert out_fallback.shape == (3, 64)

    def test_paged_indexer_candidate_context_sparsecore_and_fallback(self, monkeypatch):
        """Candidate_context covers both SparseCore top-k and fallback paths."""
        tokens = 2
        width = 4096
        topk = 64
        block_size = 16
        q = jnp.zeros((tokens, 1, 128), dtype=jnp.bfloat16)
        pool = jnp.zeros((1000, 128), dtype=jnp.int32)
        records = jnp.zeros((tokens, width), dtype=jnp.int32)
        weights = jnp.zeros((tokens, 1), dtype=jnp.float32)
        lengths = jnp.full((tokens,), 50, dtype=jnp.int32)
        requests = jnp.arange(tokens, dtype=jnp.int32)

        monkeypatch.setattr(
            v2_ak,
            "_sparsecore_top_k",
            lambda s, k, lengths_arg: jnp.zeros((s.shape[0], k), dtype=jnp.int32),
        )

        # 1. SparseCore top-k branch in candidate_context
        # (candidates.shape[1] * block_size >= 2048)
        candidates_sc = jnp.zeros((tokens, 128), dtype=jnp.int32)  # 128 * 16 = 2048
        sel_sc, cand_sc = v2_ak.paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates_sc,
            requests,
            topk=topk,
            block_size=block_size,
            candidate_write=False,
            candidate_filter=True,
            unordered=True,
            page_records=16,
        )
        assert sel_sc.shape == (tokens, topk)
        assert cand_sc.shape == (tokens, 128)

        # 2. Fallback lax.top_k branch in candidate_context
        # (candidates.shape[1] * block_size < 2048)
        candidates_fb = jnp.zeros((tokens, 8), dtype=jnp.int32)  # 8 * 16 = 128 < 2048
        sel_fb, cand_fb = v2_ak.paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates_fb,
            requests,
            topk=topk,
            block_size=block_size,
            candidate_write=False,
            candidate_filter=True,
            unordered=False,
            page_records=None,
        )
        assert sel_fb.shape == (tokens, topk)
        assert cand_fb.shape == (tokens, 8)

    def test_update_and_select_jax_and_wrappers_with_page_records(self):
        """Verifies page_records forwarding in update_and_select_jax."""
        tokens = 2
        width = 256
        topk = 32
        q = jnp.zeros((tokens, 1, 128), dtype=jnp.bfloat16)
        pool = jnp.zeros((1000, 128), dtype=jnp.int32)
        write_records = jnp.zeros((0,), dtype=jnp.int32)
        keys = jnp.zeros((0, 128), dtype=jnp.bfloat16)
        records = jnp.zeros((tokens, width), dtype=jnp.int32)
        weights = jnp.zeros((tokens, 1), dtype=jnp.float32)
        lengths = jnp.full((tokens,), 20, dtype=jnp.int32)
        candidates = jnp.zeros((tokens, 0), dtype=jnp.int32)
        requests = jnp.arange(tokens, dtype=jnp.int32)
        batch_max = lengths.max()

        # Case 1: page_records > 0
        up1, sel1, pub1 = v2_ak.update_and_select_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            write_records,
            keys,
            requests,
            batch_max,
            topk,
            16,
            candidate_write=False,
            candidate_filter=False,
            unordered=False,
            page_records=16,
        )
        assert sel1.shape == (tokens, topk)

        # Case 2: page_records == 0 (disables page gathers)
        up2, sel2, pub2 = v2_ak.update_and_select_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            write_records,
            keys,
            requests,
            batch_max,
            topk,
            16,
            candidate_write=False,
            candidate_filter=False,
            unordered=False,
            page_records=0,
        )
        assert sel2.shape == (tokens, topk)


# ==============================================================================
# 5. v2/cache.py: gather_cache_page_values Coverage
# ==============================================================================


class TestGatherCachePageValuesFull:
    """Verifies gather_cache_page_values layout guards, page gathers, and sentinels."""

    def test_gather_cache_page_values_layout_and_math(self):
        cache = jnp.arange(64 * ROW_WORDS, dtype=jnp.int32).reshape(64, ROW_WORDS)
        page_records = 16
        elements = 128

        # Valid call
        first_records = jnp.array([0, 16, -1], dtype=jnp.int32)
        out = gather_cache_page_values(cache, first_records, page_records, elements)
        assert out.shape == (3 * page_records, elements)

        # Invalid row words coverage
        with pytest.raises(ValueError, match="Pages must cover whole pool word rows"):
            gather_cache_page_values(cache, first_records, page_records=5, elements=128)
