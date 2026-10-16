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
"""Unit tests for TPU_PARALLEL_PRECOMPILE rotation, patch activation, and rank invariance."""

import os
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.compilation.piecewise_backend import PiecewiseBackend


class TestParallelPrecompileRotationMath:
    """Validates the mathematical rotation algorithm used in compile_all_ranges."""

    @staticmethod
    def _compute_indexes(range_entry_len: int, local_rank: int) -> list[int]:
        """Replicates the rotation algorithm from vllm_torchtpu._patch_vllm_compile_all_ranges."""
        expanded_indexes = list(range((range_entry_len + 7) // 8 * 8))
        d = deque(sorted(expanded_indexes, key=lambda x: (x % 8, x)))
        d.rotate(-local_rank * (len(d) // 8))
        indexes = [i for i in d if i < range_entry_len]
        return indexes

    def test_rotation_matrix_dp8_eight_buckets(self):
        """For 8 buckets on DP8, every rank must start compiling a distinct bucket at Step 0."""
        step_0_assignments = []
        full_schedules = {}

        for rank in range(8):
            indexes = self._compute_indexes(range_entry_len=8, local_rank=rank)
            assert len(
                indexes) == 8, f"Rank {rank} did not schedule all 8 buckets"
            assert set(indexes) == set(
                range(8)), f"Rank {rank} missed or duplicated buckets"
            step_0_assignments.append(indexes[0])
            full_schedules[rank] = indexes

        # Critical: Step 0 assignments must be all 8 distinct buckets [0..7] (zero collision)
        assert sorted(step_0_assignments) == list(range(8)), (
            f"Step 0 collision detected! Ranks started on duplicate buckets: {step_0_assignments}"
        )

        # Confirm that no two ranks have the exact same schedule
        unique_schedules = {tuple(sched) for sched in full_schedules.values()}
        assert len(
            unique_schedules
        ) == 8, "Expected 8 distinct rotation schedules across DP8 ranks"

    @pytest.mark.parametrize("num_buckets", [1, 2, 4, 5, 7, 8, 12, 16, 24])
    def test_arbitrary_bucket_counts_cover_all_indices(self, num_buckets: int):
        """Validates that for any bucket count, all ranks schedule every bucket exactly once."""
        for rank in range(8):
            indexes = self._compute_indexes(range_entry_len=num_buckets,
                                            local_rank=rank)
            assert len(indexes) == num_buckets
            assert sorted(indexes) == list(range(num_buckets))
            assert all(0 <= idx < num_buckets for idx in indexes)

    def test_single_bucket_edge_case(self):
        """Single bucket shape should schedule index 0 across all ranks without error."""
        for rank in range(8):
            indexes = self._compute_indexes(range_entry_len=1, local_rank=rank)
            assert indexes == [0]


class TestPatchCompileAllRangesActivation:
    """Verifies that TPU_PARALLEL_PRECOMPILE correctly patches PiecewiseBackend."""

    def test_patch_applies_when_enabled(self):
        """When TPU_PARALLEL_PRECOMPILE=1, PiecewiseBackend.compile_all_ranges is patched."""
        import vllm_torchtpu
        import vllm_torchtpu.envs as tpu_envs

        original = PiecewiseBackend.compile_all_ranges

        with patch.object(tpu_envs, "TPU_PARALLEL_PRECOMPILE", True):
            vllm_torchtpu._patch_vllm_compile_all_ranges()
            patched = PiecewiseBackend.compile_all_ranges
            assert patched != original, "PiecewiseBackend.compile_all_ranges was not patched when enabled"

        # Restore original
        PiecewiseBackend.compile_all_ranges = original

    def test_patch_skips_when_disabled(self):
        """When TPU_PARALLEL_PRECOMPILE=0, PiecewiseBackend.compile_all_ranges remains unchanged."""
        import vllm_torchtpu
        import vllm_torchtpu.envs as tpu_envs

        original = PiecewiseBackend.compile_all_ranges

        with patch.object(tpu_envs, "TPU_PARALLEL_PRECOMPILE", False):
            vllm_torchtpu._patch_vllm_compile_all_ranges()
            assert PiecewiseBackend.compile_all_ranges == original, (
                "PiecewiseBackend.compile_all_ranges should not be patched when disabled"
            )


class TestCompileAllRangesExecution:
    """Tests the execution flow and order of compile_all_ranges."""

    @staticmethod
    def _create_mock_backend(num_buckets: int = 8):
        """Builds a mock PiecewiseBackend with mock range entries."""
        backend = MagicMock(spec=PiecewiseBackend)
        backend.graph = MagicMock()
        backend.piecewise_compile_index = 0
        backend.total_piecewise_compiles = 1
        backend.vllm_backend = MagicMock()
        backend.compilation_config = MagicMock()

        entries = {}
        for i in range(num_buckets):
            entry = SimpleNamespace(
                compiled=False,
                compile_range=SimpleNamespace(
                    is_single_size=lambda: True,
                    start=16 * (i + 1),
                ),
                runnable=None,
            )
            entries[f"range_{i}"] = entry

        backend.range_entries = entries
        return backend

    def test_execution_order_follows_rank_rotation(self):
        """Asserts that compiler_manager.compile is invoked in the rotated order for the rank."""
        import vllm_torchtpu
        import vllm_torchtpu.envs as tpu_envs

        original = PiecewiseBackend.compile_all_ranges
        with patch.object(tpu_envs, "TPU_PARALLEL_PRECOMPILE", True):
            vllm_torchtpu._patch_vllm_compile_all_ranges()

            for rank in range(8):
                backend = self._create_mock_backend(num_buckets=8)
                compiled_sizes = []

                def mock_compile(*args,
                                 compile_range=None,
                                 sizes=compiled_sizes,
                                 **kwargs):
                    sizes.append(compile_range.start)
                    return MagicMock()

                backend.vllm_backend.compiler_manager.compile = mock_compile

                with patch.dict(os.environ, {"LOCAL_RANK": str(rank)}):
                    PiecewiseBackend.compile_all_ranges(backend)

                # All 8 entries should now be marked compiled
                for entry in backend.range_entries.values():
                    assert entry.compiled is True
                    assert entry.runnable is not None

                # The first size compiled by this rank must correspond to its rotated bucket
                expected_start_size = 16 * (rank + 1)
                assert compiled_sizes[0] == expected_start_size, (
                    f"Rank {rank} expected to compile size {expected_start_size} first, got {compiled_sizes[0]}"
                )

        PiecewiseBackend.compile_all_ranges = original

    def test_already_compiled_entries_are_skipped(self):
        """Asserts that range entries already compiled (e.g. loaded from cache) are not recompiled."""
        import vllm_torchtpu
        import vllm_torchtpu.envs as tpu_envs

        original = PiecewiseBackend.compile_all_ranges
        with patch.object(tpu_envs, "TPU_PARALLEL_PRECOMPILE", True):
            vllm_torchtpu._patch_vllm_compile_all_ranges()

            backend = self._create_mock_backend(num_buckets=4)
            # Pre-mark bucket 0 and bucket 1 as compiled
            backend.range_entries["range_0"].compiled = True
            backend.range_entries["range_1"].compiled = True

            compile_calls = []
            backend.vllm_backend.compiler_manager.compile = lambda *a, **k: compile_calls.append(
                k) or MagicMock()

            with patch.dict(os.environ, {"LOCAL_RANK": "0"}):
                PiecewiseBackend.compile_all_ranges(backend)

            # Only the 2 uncompiled buckets should have triggered compile()
            assert len(compile_calls) == 2

        PiecewiseBackend.compile_all_ranges = original


class TestRankCacheKeyInvariance:
    """Verifies that compilation cache keys are strictly invariant across LOCAL_RANK values."""

    def test_tpu_hash_is_invariant_to_local_rank(self):
        """Different LOCAL_RANK environment settings must NOT produce different cache keys."""
        from vllm_torchtpu.compilation.tpu_compiler import \
            compute_tpu_compilation_hash

        mock_config = SimpleNamespace(
            cache_config=SimpleNamespace(
                gpu_memory_utilization=0.9,
                num_gpu_blocks=1024,
                num_gpu_blocks_override=None,
            ),
            model_config=SimpleNamespace(max_model_len=4096),
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=2048,
                max_num_seqs=16,
            ),
            parallel_config=SimpleNamespace(data_parallel_size=8),
            speculative_config=None,
        )

        with patch(
                "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
                return_value=[]):
            hashes = {}
            for rank in range(8):
                with patch.dict(os.environ, {
                        "LOCAL_RANK": str(rank),
                        "RANK": str(rank)
                }):
                    hashes[rank] = compute_tpu_compilation_hash(mock_config)

            # All 8 ranks must have the exact same hash
            unique_hashes = set(hashes.values())
            assert len(unique_hashes) == 1, (
                f"Rank divergence detected! Generated {len(unique_hashes)} distinct hashes across ranks: {hashes}"
            )
