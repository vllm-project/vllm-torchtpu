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

import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class TestTpuCompilerCache:

    @staticmethod
    def _config(*, max_num_seqs=16, num_gpu_blocks=1024):
        return SimpleNamespace(
            cache_config=SimpleNamespace(
                gpu_memory_utilization=0.9,
                num_gpu_blocks=num_gpu_blocks,
                num_gpu_blocks_override=None,
            ),
            model_config=SimpleNamespace(max_model_len=4096),
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=2048,
                max_num_seqs=max_num_seqs,
            ),
            parallel_config=SimpleNamespace(data_parallel_size=1),
            speculative_config=None,
        )

    def test_tpu_hash_covers_derived_shapes(self):
        from vllm_torchtpu.compilation.tpu_compiler import \
            compute_tpu_compilation_hash

        with patch(
                "vllm_torchtpu.compilation.tpu_compiler."
                "_iter_runtime_cache_key_files",
                return_value=[]):
            base = compute_tpu_compilation_hash(self._config())
            different_batch = compute_tpu_compilation_hash(
                self._config(max_num_seqs=32))
            different_kv = compute_tpu_compilation_hash(
                self._config(num_gpu_blocks=2048))

        assert base != different_batch
        assert base != different_kv

    def test_tpu_hash_covers_plugin_compile_env(self):
        from vllm_torchtpu.compilation.tpu_compiler import \
            compute_tpu_compilation_hash

        with patch(
                "vllm_torchtpu.compilation.tpu_compiler."
                "_iter_runtime_cache_key_files",
                return_value=[]):
            with patch.dict(os.environ, {"ONEHOT_MOE_PERMUTE_THRESHOLD": "0"}):
                disabled = compute_tpu_compilation_hash(self._config())
            with patch.dict(os.environ,
                            {"ONEHOT_MOE_PERMUTE_THRESHOLD": "4096"}):
                enabled = compute_tpu_compilation_hash(self._config())

        assert disabled != enabled

    def test_tpu_hash_ignores_the_phased_profiler_switch(self):
        """Selecting a profiler must not bust the compile cache: otherwise a
        profiled run recompiles everything, and so does the next plain one."""
        from vllm_torchtpu.compilation.tpu_compiler import \
            compute_tpu_compilation_hash

        with patch(
                "vllm_torchtpu.compilation.tpu_compiler."
                "_iter_runtime_cache_key_files",
                return_value=[]):
            with patch.dict(os.environ, {"USE_PHASED_PROFILER": "false"}):
                unprofiled = compute_tpu_compilation_hash(self._config())
            with patch.dict(os.environ, {"USE_PHASED_PROFILER": "true"}):
                phased = compute_tpu_compilation_hash(self._config())

        assert unprofiled == phased

    def test_tpu_hash_covers_native_compile_env(self):
        from vllm_torchtpu.compilation.tpu_compiler import \
            compute_tpu_compilation_hash

        with patch(
                "vllm_torchtpu.compilation.tpu_compiler."
                "_iter_runtime_cache_key_files",
                return_value=[]):
            with patch.dict(os.environ, {"XLA_FLAGS": "--xla_dump_to=/tmp/a"}):
                first = compute_tpu_compilation_hash(self._config())
            with patch.dict(os.environ, {"XLA_FLAGS": "--xla_dump_to=/tmp/b"}):
                second = compute_tpu_compilation_hash(self._config())

        assert first != second

    def test_tpu_hash_covers_compiler_versions(self):
        from vllm_torchtpu.compilation.tpu_compiler import \
            compute_tpu_compilation_hash

        versions = {
            "torch-tpu": "1",
            "jax": "1",
            "jaxlib": "1",
            "libtpu": "1",
        }
        with patch(
                "vllm_torchtpu.compilation.tpu_compiler."
                "_iter_runtime_cache_key_files",
                return_value=[]):
            with patch(
                    "vllm_torchtpu.compilation.tpu_compiler."
                    "importlib_metadata.version",
                    side_effect=versions.get):
                base = compute_tpu_compilation_hash(self._config())

            for package in versions:
                changed_versions = {**versions, package: "2"}
                with patch(
                        "vllm_torchtpu.compilation.tpu_compiler."
                        "importlib_metadata.version",
                        side_effect=changed_versions.get):
                    changed = compute_tpu_compilation_hash(self._config())
                assert base != changed

    def test_tpu_hash_covers_effective_data_parallel_size(self):
        from vllm_torchtpu.compilation.tpu_compiler import \
            compute_tpu_compilation_hash

        with patch(
                "vllm_torchtpu.compilation.tpu_compiler."
                "_iter_runtime_cache_key_files",
                return_value=[]):
            with patch.dict(os.environ, {"TORCH_TPU_DP_SIZE": "1"}):
                dp1 = compute_tpu_compilation_hash(self._config())
            with patch.dict(os.environ, {"TORCH_TPU_DP_SIZE": "4"}):
                dp4 = compute_tpu_compilation_hash(self._config())

        assert dp1 != dp4

    def test_runtime_hash_covers_custom_op_boundaries(self):
        from vllm_torchtpu.compilation import tpu_compiler

        source_root = Path(tpu_compiler.__file__).resolve().parents[2]
        hashed = {
            path.relative_to(source_root).as_posix()
            for path in tpu_compiler._iter_runtime_cache_key_files(source_root)
        }
        custom_op_boundaries = {
            path.relative_to(source_root).as_posix()
            for path in (source_root / "vllm_torchtpu").rglob("*.py")
            if "pallas.jax_op" in path.read_text()
        }
        assert custom_op_boundaries <= hashed
        assert {
            "vllm_torchtpu/kernels/quantized_matmul/blockwise_kernel.py",
            "vllm_torchtpu/kernels/pool_adapters.py",
        } <= hashed

    def test_aot_hash_includes_tpu_compiler_hash(self):
        from vllm.compilation import caching

        from vllm_torchtpu import _patch_vllm_aot_compile_cache_key

        def upstream_factors(_):
            return ["upstream"]

        with patch.object(caching, "aot_compile_hash_factors",
                          upstream_factors):
            _patch_vllm_aot_compile_cache_key()
            with patch(
                    "vllm_torchtpu.compilation.tpu_compiler."
                    "compute_tpu_compilation_hash",
                    return_value="tpu"):
                factors = caching.aot_compile_hash_factors(self._config())

        assert factors == ["upstream", "tpu"]

    def test_env_override_tier3_default_when_cache_enabled(self):
        """Verify that when compile cache is enabled, TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT defaults to VLLM_CACHE_ROOT/xla_cache."""
        env_mock = {
            "VLLM_DISABLE_COMPILE_CACHE": "0",
            "VLLM_CACHE_ROOT": "/tmp/custom_vllm_cache",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            import vllm_torchtpu.env_override
            importlib.reload(vllm_torchtpu.env_override)

            assert os.environ.get("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"
                                  ) == "/tmp/custom_vllm_cache/xla_cache"
            assert os.environ.get(
                "TORCH_TPU_TIER2_COMPILATION_CACHE") == "tpu_tier2_cache"

    def test_env_override_tier3_respects_vllm_xla_cache_path(self):
        """Verify that VLLM_XLA_CACHE_PATH overrides the default VLLM_CACHE_ROOT/xla_cache path."""
        env_mock = {
            "VLLM_DISABLE_COMPILE_CACHE": "0",
            "VLLM_CACHE_ROOT": "/tmp/custom_vllm_cache",
            "VLLM_XLA_CACHE_PATH": "/mnt/gcs_xla_cache",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            import vllm_torchtpu.env_override
            importlib.reload(vllm_torchtpu.env_override)

            assert os.environ.get("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"
                                  ) == "/mnt/gcs_xla_cache"

    def test_env_override_tier3_disabled_when_vllm_disable_compile_cache_1(
            self):
        """Verify that TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT is not set when VLLM_DISABLE_COMPILE_CACHE=1."""
        env_mock = {
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "VLLM_CACHE_ROOT": "/tmp/custom_vllm_cache",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            import vllm_torchtpu.env_override
            importlib.reload(vllm_torchtpu.env_override)

            assert os.environ.get(
                "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT") is None

    def test_tpu_compilation_handle_serialization(self):
        """Verify serialization and deserialization of TpuCompilationHandle and legacy fallbacks."""
        import pickle

        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilationHandle

        # 1. Test dataclass handle serialization
        handle = TpuCompilationHandle(key="test_key_123",
                                      tier3_cache_active=True)
        dumped = pickle.dumps(handle)
        loaded = pickle.loads(dumped)
        assert isinstance(loaded, TpuCompilationHandle)
        assert loaded.key == "test_key_123"
        assert loaded.tier3_cache_active is True
        assert loaded.executable is None

        # 2. Test legacy dict deserialization fallback
        legacy_dict = {"tier3_cache_active": True, "key": "test_key_dict"}
        loaded_dict = pickle.loads(pickle.dumps(legacy_dict))
        assert isinstance(loaded_dict, dict)
        assert loaded_dict.get("tier3_cache_active") is True

        # 3. Test non-tier3 handle with executable
        handle_with_exe = TpuCompilationHandle(key="test_key_456",
                                               tier3_cache_active=False,
                                               executable="mock_exe")
        loaded_exe_handle = pickle.loads(pickle.dumps(handle_with_exe))
        assert loaded_exe_handle.tier3_cache_active is False
        assert loaded_exe_handle.executable == "mock_exe"

    def test_compiler_initialize_cache(self):
        """Verify that TpuCompilerAdaptor sets cache_dir accurately."""
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor
        compiler = TpuCompilerAdaptor()
        compiler.initialize_cache(cache_dir="/tmp/test_cache_dir")
        assert compiler.cache_dir == "/tmp/test_cache_dir"

    def test_tpu_compiler_adaptor_load_direct_executable(self, tmp_path):
        """Verify that TpuCompilerAdaptor.load directly returns the unpickled executable without re-tracing."""
        import pickle

        from vllm_torchtpu.compilation.tpu_compiler import (
            TpuCompilationHandle, TpuCompilerAdaptor)

        class MockExecutable:

            def __call__(self, x):
                return x * 2

        mock_exe = MockExecutable()
        handle_path = tmp_path / "mock_handle.pkl"
        handle_data = TpuCompilationHandle(
            key="test_key",
            tier3_cache_active=True,
            executable=mock_exe,
        )
        with open(handle_path, "wb") as f:
            pickle.dump(handle_data, f)

        adaptor = TpuCompilerAdaptor()
        adaptor.initialize_cache(cache_dir=str(tmp_path))

        # Test loading unwrapped
        loaded_fn = adaptor.load(
            handle=("test_key", str(handle_path), False),
            graph=None,
            example_inputs=[],
            graph_index=0,
            compile_range=(512, 512),
        )
        assert loaded_fn(10) == 20

        # Test loading with tuple unwrapping (was_wrapped=True)
        class MockTupleExecutable:

            def __call__(self, x):
                return (x * 3, )

        tuple_handle_path = tmp_path / "mock_tuple_handle.pkl"
        with open(tuple_handle_path, "wb") as f:
            pickle.dump(
                TpuCompilationHandle(key="tuple_key",
                                     tier3_cache_active=True,
                                     executable=MockTupleExecutable()), f)

        loaded_tuple_fn = adaptor.load(
            handle=("tuple_key", str(tuple_handle_path), True),
            graph=None,
            example_inputs=[],
            graph_index=0,
            compile_range=(512, 512),
        )
        assert loaded_tuple_fn(10) == 30

        # Test loading with tier3_key (lightweight handle where executable=None)
        tier3_handle_path = tmp_path / "mock_tier3_handle.pkl"
        with open(tier3_handle_path, "wb") as f:
            pickle.dump(
                TpuCompilationHandle(
                    key="tier3_test_key",
                    tier3_cache_active=True,
                    executable=None,
                    tier3_key=
                    "1234567890abcdef_1234567890abcdef_1234567890abcdef",
                ), f)

        with patch(
                "torch_tpu._internal.compile.tpu_torch_compile.load_executable_from_tier3_key",
                return_value=MockExecutable(),
                create=True,
        ):
            loaded_tier3_fn = adaptor.load(
                handle=("tier3_test_key", str(tier3_handle_path), False),
                graph=None,
                example_inputs=[],
                graph_index=0,
                compile_range=(512, 512),
            )
            assert loaded_tier3_fn(15) == 30

    def test_patch_vllm_compile_all_ranges(self):
        """Verify that TPU_PARALLEL_PRECOMPILE patches PiecewiseBackend and partitions ranges across ranks."""
        import os
        from unittest.mock import MagicMock, patch

        from vllm_torchtpu import _patch_vllm_compile_all_ranges

        class MockRange:

            def __init__(self, size):
                self.start = size
                self.end = size

            def is_single_size(self):
                return True

        class MockRangeEntry:

            def __init__(self, size):
                self.compile_range = MockRange(size)
                self.compiled = False
                self.runnable = None

        class MockPiecewiseBackend:

            def __init__(self):
                self.graph = MagicMock()
                self.range_entries = {
                    512: MockRangeEntry(512),
                    1024: MockRangeEntry(1024),
                    2048: MockRangeEntry(2048),
                    4096: MockRangeEntry(4096),
                }
                self.piecewise_compile_index = 0
                self.total_piecewise_compiles = 1
                self.vllm_backend = MagicMock()
                self.compilation_config = MagicMock()
                self.returns_tuple = False

            def _log_compile_start(self, r):
                pass

        compiled_order = []

        def mock_compile(graph, args, inductor_config, compilation_config,
                         compile_range, **kwargs):
            compiled_order.append(compile_range.start)
            return MagicMock()

        with patch("vllm_torchtpu.envs.TPU_PARALLEL_PRECOMPILE", True):
            _patch_vllm_compile_all_ranges()

            from vllm.compilation.piecewise_backend import PiecewiseBackend

            # Test Rank 0 partition
            backend_rank0 = MockPiecewiseBackend()
            backend_rank0.vllm_backend.compiler_manager.compile = mock_compile
            compiled_order.clear()

            with patch.dict(os.environ, {
                    "LOCAL_RANK": "0",
                    "WORLD_SIZE": "4"
            }):
                PiecewiseBackend.compile_all_ranges(backend_rank0)
                # First compiled should be 512 (index 0)
                assert compiled_order[0] == 512
                assert len(compiled_order) == 4
                assert all(e.compiled
                           for e in backend_rank0.range_entries.values())

            # Test Rank 1 partition
            backend_rank1 = MockPiecewiseBackend()
            backend_rank1.vllm_backend.compiler_manager.compile = mock_compile
            compiled_order.clear()

            with patch.dict(os.environ, {
                    "LOCAL_RANK": "1",
                    "WORLD_SIZE": "4"
            }):
                PiecewiseBackend.compile_all_ranges(backend_rank1)
                # First compiled should be 1024 (index 1)
                assert compiled_order[0] == 1024
                assert len(compiled_order) == 4
                assert all(e.compiled
                           for e in backend_rank1.range_entries.values())
