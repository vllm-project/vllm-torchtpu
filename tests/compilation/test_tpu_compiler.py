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

    def test_env_override_mapping(self):
        """Verify that VLLM_XLA_CACHE_PATH sets the corresponding native variables."""
        env_mock = {
            "VLLM_XLA_CACHE_PATH": "/tmp/test_xla_cache_env",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            # Reload env_override to re-execute its top-level statements
            import vllm_torchtpu.env_override
            importlib.reload(vllm_torchtpu.env_override)

            assert os.environ.get(
                "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT"
            ) == "/tmp/test_xla_cache_env/torch_tpu_tier3"
            assert os.environ.get(
                "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"
            ) == "/tmp/test_xla_cache_env/torch_tpu_tier3"

            assert os.environ.get("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"
                                  ) == "tpu_tier2_cache"
            assert os.environ.get(
                "TORCH_TPU_TIER2_COMPILATION_CACHE") == "tpu_tier2_cache"

    def test_env_override_disables_breakable_cudagraph(self):
        """Verify that the TPU plugin always disables breakable cudagraph."""
        with patch.dict(os.environ, {"VLLM_USE_BREAKABLE_CUDAGRAPH": "1"}):
            import vllm_torchtpu.env_override
            importlib.reload(vllm_torchtpu.env_override)

            assert os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "0"

    def test_env_override_no_clobber(self):
        """Verify that existing native variables are not overwritten by env_override."""
        env_mock = {
            "VLLM_XLA_CACHE_PATH": "/tmp/test_xla_cache_env",
            "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT":
            "/custom/tier3/path",
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "/custom/tier3/path",
            "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": "custom_tier2",
            "TORCH_TPU_TIER2_COMPILATION_CACHE": "custom_tier2",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            import vllm_torchtpu.env_override
            importlib.reload(vllm_torchtpu.env_override)

            assert os.environ.get(
                "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT"
            ) == "/custom/tier3/path"
            assert os.environ.get("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"
                                  ) == "/custom/tier3/path"

            assert os.environ.get("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"
                                  ) == "/custom/tier3/path"

            assert os.environ.get(
                "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE") == "custom_tier2"
            assert os.environ.get(
                "TORCH_TPU_TIER2_COMPILATION_CACHE") == "custom_tier2"

    def test_compiler_cache_dir_relocation(self):
        """Verify that TpuCompilerAdaptor relocates cache_dir under VLLM_XLA_CACHE_PATH."""
        with patch.dict(os.environ,
                        {"VLLM_XLA_CACHE_PATH": "/tmp/test_xla_cache_dir"}):
            # Import adaptor
            from vllm_torchtpu.compilation.tpu_compiler import \
                TpuCompilerAdaptor

            compiler = TpuCompilerAdaptor()

            # Simulate initialize_cache call with default path
            from vllm.envs import VLLM_CACHE_ROOT
            default_cache_dir = os.path.join(VLLM_CACHE_ROOT,
                                             "torch_compile_cache")

            compiler.initialize_cache(cache_dir=default_cache_dir)

            expected_cache_dir = os.path.join("/tmp/test_xla_cache_dir",
                                              "torch_compile_cache")
            assert compiler.cache_dir == expected_cache_dir

    def test_compiler_cache_dir_fallback_on_error(self):
        """Verify that initialize_cache falls back to default cache_dir on exception."""
        with patch.dict(os.environ,
                        {"VLLM_XLA_CACHE_PATH": "/tmp/test_xla_cache_dir"}):
            from vllm_torchtpu.compilation.tpu_compiler import \
                TpuCompilerAdaptor
            compiler = TpuCompilerAdaptor()

            # Force relpath exception by passing incompatible path types or mocking relpath
            with patch("os.path.relpath",
                       side_effect=ValueError("mock error")):
                compiler.initialize_cache(cache_dir="/original/cache/dir")
                # Should fallback to the original directory
                assert compiler.cache_dir == "/original/cache/dir"
