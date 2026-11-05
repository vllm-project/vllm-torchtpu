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

import concurrent.futures
import importlib
import logging
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch_tpu._internal.compile.split_compiler import _SplitCompiledExecutable
from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
    AsyncCompiledArtifact,
    NoOpCompiledArtifact,
    TorchTpuCompiledExecutable,
)

from vllm_torchtpu.env_override import _TPU_VMEM_BYTES_PER_TENSOR_CORE


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
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            base = compute_tpu_compilation_hash(self._config())
            different_batch = compute_tpu_compilation_hash(
                self._config(max_num_seqs=32)
            )
            different_kv = compute_tpu_compilation_hash(
                self._config(num_gpu_blocks=2048)
            )

        assert base != different_batch
        assert base != different_kv

    def test_tpu_hash_covers_plugin_compile_env(self):
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            with patch.dict(os.environ, {"ONEHOT_MOE_PERMUTE_THRESHOLD": "0"}):
                disabled = compute_tpu_compilation_hash(self._config())
            with patch.dict(os.environ, {"ONEHOT_MOE_PERMUTE_THRESHOLD": "4096"}):
                enabled = compute_tpu_compilation_hash(self._config())

        assert disabled != enabled

    def test_tpu_hash_ignores_the_phased_profiler_switch(self):
        """Selecting a profiler must not bust the compile cache: otherwise a
        profiled run recompiles everything, and so does the next plain one."""
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            with patch.dict(os.environ, {"USE_PHASED_PROFILER": "false"}):
                unprofiled = compute_tpu_compilation_hash(self._config())
            with patch.dict(os.environ, {"USE_PHASED_PROFILER": "true"}):
                phased = compute_tpu_compilation_hash(self._config())

        assert unprofiled == phased

    def test_tpu_hash_ignores_reload_compiled_executable_switch(self):
        """Reloading newly compiled executables strips host-only metadata
        without changing the serialized executable bytes or disk cache key."""
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            with patch.dict(os.environ, {"VLLM_TPU_RELOAD_COMPILED_EXECUTABLE": "0"}):
                reload_off = compute_tpu_compilation_hash(self._config())
            with patch.dict(os.environ, {"VLLM_TPU_RELOAD_COMPILED_EXECUTABLE": "1"}):
                reload_on = compute_tpu_compilation_hash(self._config())

        assert reload_off == reload_on

    def test_tpu_hash_covers_native_compile_env(self):
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            with patch.dict(os.environ, {"XLA_FLAGS": "--xla_dump_to=/tmp/a"}):
                first = compute_tpu_compilation_hash(self._config())
            with patch.dict(os.environ, {"XLA_FLAGS": "--xla_dump_to=/tmp/b"}):
                second = compute_tpu_compilation_hash(self._config())

        assert first != second

    def test_tpu_hash_covers_compiler_versions(self):
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        versions = {
            "torch-tpu": "1",
            "jax": "1",
            "jaxlib": "1",
            "libtpu": "1",
        }
        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            with patch(
                "vllm_torchtpu.compilation.tpu_compiler.importlib_metadata.version",
                side_effect=versions.get,
            ):
                base = compute_tpu_compilation_hash(self._config())

            for package in versions:
                changed_versions = {**versions, package: "2"}
                with patch(
                    "vllm_torchtpu.compilation.tpu_compiler.importlib_metadata.version",
                    side_effect=changed_versions.get,
                ):
                    changed = compute_tpu_compilation_hash(self._config())
                assert base != changed

    @pytest.mark.parametrize(
        "libtpu_version",
        (pytest.param(None, id="no-package"), pytest.param("0.0.50", id="package")),
    )
    def test_tpu_hash_follows_custom_library_bytes_not_path(
        self, tmp_path, libtpu_version
    ):
        # JAX loads a custom TPU_LIBRARY_PATH in preference to the pip package,
        # so the hash follows that library's bytes whether or not libtpu is
        # installed, and not the path it was loaded from.
        from importlib import metadata as importlib_metadata

        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        def version(name):
            if name != "libtpu":
                return "1"
            if libtpu_version is None:
                raise importlib_metadata.PackageNotFoundError(name)
            return libtpu_version

        def compilation_hash(library):
            with (
                patch(
                    "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
                    return_value=[],
                ),
                patch(
                    "vllm_torchtpu.compilation.tpu_compiler.importlib_metadata.version",
                    side_effect=version,
                ),
                patch.dict(os.environ, {"TPU_LIBRARY_PATH": str(library)}),
            ):
                return compute_tpu_compilation_hash(self._config())

        build_a = b"build-a" * 4096
        # A different size as well as different bytes, so the cached
        # fingerprint of the first build at this path is not reused.
        build_b = b"build-b" * 8192
        library = tmp_path / "a" / "libtpu.so"
        library.parent.mkdir()
        library.write_bytes(build_a)
        hash_a = compilation_hash(library)
        library.write_bytes(build_b)
        hash_b = compilation_hash(library)
        assert hash_a != hash_b

        # The same build mounted at another path.
        moved = tmp_path / "b" / "libtpu.so"
        moved.parent.mkdir()
        moved.write_bytes(build_b)
        assert compilation_hash(moved) == hash_b

    def test_tpu_hash_covers_effective_data_parallel_size(self):
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
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
        excluded_files = set(tpu_compiler._RUNTIME_CACHE_KEY_EXCLUDED_PATHS)
        excluded_dirs = tuple(
            f"{rel_path}/"
            for rel_path in tpu_compiler._RUNTIME_CACHE_KEY_EXCLUDED_PATHS
        )
        omni_only = {
            path
            for path in custom_op_boundaries
            if path in excluded_files or path.startswith(excluded_dirs)
        }
        assert custom_op_boundaries - omni_only <= hashed
        assert not {
            path
            for path in hashed
            if path in excluded_files or path.startswith(excluded_dirs)
        }
        assert {
            "vllm_torchtpu/kernels/quantized_matmul/blockwise_kernel.py",
            "vllm_torchtpu/kernels/pool_adapters.py",
            "vllm_torchtpu/models/vllm/kimi_k3/collective_ops.py",
            "vllm_torchtpu/models/vllm/kimi_k3/layers.py",
        } <= hashed

    def test_runtime_hash_excluded_kernels_stay_omni_only(self):
        """Excluded diffusion kernels must have no caller in hashed code.

        An LLM path importing one would compile it into a cached executable
        whose key no longer covers the kernel's source.
        """
        from vllm_torchtpu.compilation import tpu_compiler

        source_root = Path(tpu_compiler.__file__).resolve().parents[2]
        excluded_files = set(tpu_compiler._RUNTIME_CACHE_KEY_EXCLUDED_PATHS)
        excluded_dirs = tuple(
            f"{rel_path}/"
            for rel_path in tpu_compiler._RUNTIME_CACHE_KEY_EXCLUDED_PATHS
        )
        importers = {
            path.relative_to(source_root).as_posix()
            for path in (source_root / "vllm_torchtpu").rglob("*.py")
            if not (
                path.relative_to(source_root).as_posix() in excluded_files
                or path.relative_to(source_root).as_posix().startswith(excluded_dirs)
            )
            and re.search(
                r"kernels\.diffusion_attention|kernels import diffusion_attention",
                path.read_text(),
            )
        }
        assert not importers

    def test_serving_only_kv_knobs_stay_out_of_the_compile_cache_key(self):
        """Serving-time knobs must stay out of the compile-cache key.

        Every `envs.py` entry not in `_TPU_COMPILE_ENV_IGNORED` is hashed, so
        a knob added there recompiles the world once on upgrade and again
        every time an operator retunes it.
        """
        from vllm_torchtpu.compilation.tpu_compiler import (
            _TPU_COMPILE_ENV_IGNORED,
            _tpu_compile_env_factors,
        )

        serving_only = {
            "TPU_IPC_SOCKET_DIR",
            "TPU_KV_CHANNEL_EXECUTOR_MAX_WORKERS",
            "TPU_KV_COORD_EXECUTOR_MAX_WORKERS",
            "TPU_KV_LATENCY_LOG_INTERVAL",
            "TPU_KV_PIN_SHM",
            "TPU_KV_SHM_POOL_GB",
            "TPU_KV_STAGE_WAITER_POOL_SIZE",
            "TPU_KV_STAGE_WAIT_TIMEOUT_SECS",
            "TPU_KV_TRANSFER_CHANNEL_NUMBER",
            "TPU_KV_TRANSFER_NAMESPACE",
            "TPU_KV_TRANSFER_PORT",
            "TPU_KV_WARMUP_ENABLED",
            "TPU_NODE_ID",
            "TPU_P2P_WAIT_PULL_TIMEOUT",
            "TPU_RAIDEN_INLINE_LOAD",
            "TPU_RAIDEN_POOL_STAGING_LEASES",
            "TPU_RAIDEN_STAGE3_DEFERRED_SUBMIT",
            "TPU_RAIDEN_STAGE3_REGISTRATION_WAIT_S",
            "TPU_RAIDEN_STAGE3_STATUS_PROBE_S",
            "TPU_RAIDEN_TEST_REGISTRATION_DELAY_S",
            "TPU_RAIDEN_TRANSFER_NUM_SLOTS",
            "TPU_SIDE_CHANNEL_PORT",
            "TPU_USE_RAIDEN_CONNECTOR",
            "DEBUG_TPU_LOCAL_RANK_OFFSET",
            "EXPERIMENTAL_RAIDEN_STORAGE_SKIP_PARALLELISM_CHECK",
            "RAIDEN_DISABLE_SINGLETON_WORKER",
            "RAIDEN_SHM_KEY",
            "RAIDEN_STORAGE_CAPACITY_GB",
            "RAIDEN_STORAGE_PATH",
            "RAIDEN_STORAGE_PROPERTIES",
            "RAIDEN_STORAGE_TYPE",
            "SPEC_WARMUP",
            "TORCH_TPU_BASE_PORT",
            "TORCH_TPU_MP_RENDEZVOUS_PORT",
            "TPU_LOCAL_RANK_OFFSET",
            "TPU_SHARDED_LOAD_SYNC_EVERY",
            "VLLM_TORCHTPU_IPC_KEY",
            "VLLM_TPU_DEBUG_PCP_LAYOUT",
            "VLLM_TPU_OFFLOAD_SAVE_RETRIES",
            "VLLM_TPU_OFFLOAD_WAIT_TIMEOUT_S",
            "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES",
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT",
            "TPU_LIBRARY_PATH",
        }

        assert serving_only <= _TPU_COMPILE_ENV_IGNORED
        assert not serving_only & set(_tpu_compile_env_factors())

    def test_every_registry_entry_is_hashed_or_deliberately_ignored(self):
        """The two sets must partition the registry, so a new entry forces a
        deliberate hash-or-ignore choice, and a stale exemption cannot sit
        waiting for a later knob to reuse its name."""
        from vllm_torchtpu import envs
        from vllm_torchtpu.compilation.tpu_compiler import (
            _TPU_COMPILE_ENV_IGNORED,
            _tpu_compile_env_factors,
        )

        registry = set(envs.environment_variables)
        stale = _TPU_COMPILE_ENV_IGNORED - registry
        assert not stale, f"ignored but no longer in envs.py: {sorted(stale)}"
        assert set(_tpu_compile_env_factors()) | _TPU_COMPILE_ENV_IGNORED == registry

    def test_aot_hash_includes_tpu_compiler_hash(self):
        from vllm.compilation import caching

        from vllm_torchtpu import _patch_vllm_aot_compile_cache_key

        def upstream_factors(_):
            return ["upstream"]

        with patch.object(caching, "aot_compile_hash_factors", upstream_factors):
            _patch_vllm_aot_compile_cache_key()
            with patch(
                "vllm_torchtpu.compilation.tpu_compiler.compute_tpu_compilation_hash",
                return_value="tpu",
            ):
                factors = caching.aot_compile_hash_factors(self._config())

        assert factors == ["upstream", "tpu"]

    def test_env_override_mapping(self):
        """Verify that setting TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT sets
        TORCH_TPU_TIER2_COMPILATION_CACHE default.
        """
        env_mock = {
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "/tmp/test_tier3_cache_env",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            # Reload env_override to re-execute its top-level statements
            import vllm_torchtpu.env_override

            importlib.reload(vllm_torchtpu.env_override)

            assert (
                os.environ.get("TORCH_TPU_TIER2_COMPILATION_CACHE") == "tpu_tier2_cache"
            )

    @staticmethod
    def _sc_offload_flags() -> list[str]:
        return [
            token
            for token in os.environ.get("LIBTPU_INIT_ARGS", "").split()
            if "offload_min_size_in_bytes" in token
        ]

    @pytest.mark.parametrize(
        "value,expected_threshold",
        [
            ("12345", 12345),
            ("auto", _TPU_VMEM_BYTES_PER_TENSOR_CORE["v6e"]),
            ("", _TPU_VMEM_BYTES_PER_TENSOR_CORE["v6e"]),
            ("0", None),
        ],
    )
    def test_env_override_sc_offload_threshold(self, value, expected_threshold):
        """An explicit value is used as given, unset/empty/auto resolves
        from the chip family, and 0 sets no flag at all."""
        env_mock = {
            "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES": value,
            "TPU_ACCELERATOR_TYPE": "v6e-8",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            import vllm_torchtpu.env_override

            importlib.reload(vllm_torchtpu.env_override)

            expected = (
                []
                if expected_threshold is None
                else [
                    f"--xla_tpu_sparse_core_{op}_offload_min_size_in_bytes="
                    f"{expected_threshold}"
                    for op in ("all_reduce", "all_gather")
                ]
            )
            assert self._sc_offload_flags() == expected

    @pytest.mark.parametrize("value", ["banana", "-1"])
    def test_env_override_sc_offload_bad_value_raises(self, value):
        """A value that is not "auto" or a non-negative integer stops the
        package import with an error that names the variable, rather than
        silently running without the SC-offload flag."""
        env_mock = {
            "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES": value,
            "TPU_ACCELERATOR_TYPE": "v6e-8",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            import vllm_torchtpu.env_override

            with pytest.raises(
                ValueError, match="SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES"
            ):
                importlib.reload(vllm_torchtpu.env_override)

    def test_experimental_libtpu_init_args_lists_the_flags_libtpu_will_read(self):
        """The user's flags, env_override's defaults, and flags set later."""
        env_mock = {
            "LIBTPU_INIT_ARGS": "--xla_user_flag=1 --tpu_user_flag",
            "TPU_ACCELERATOR_TYPE": "v6e-8",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            import vllm_torchtpu.env_override

            importlib.reload(vllm_torchtpu.env_override)
            vllm_torchtpu.env_override.set_libtpu_flag_default("--xla_later", "2")

            args = vllm_torchtpu.env_override.experimental_libtpu_init_args()
            assert {
                "--xla_user_flag=1",
                "--tpu_user_flag",
                "--xla_tpu_use_dynamic_smem_negotiation=true",
                "--xla_later=2",
            } <= set(args)
            assert args == os.environ["LIBTPU_INIT_ARGS"].split()

    def test_envs_module_imports_only_the_standard_library(self):
        """env_override.py imports envs while the package's own __init__ is
        still running, which is only safe while envs.py depends on nothing
        in this package."""
        import ast
        import sys

        import vllm_torchtpu.envs

        tree = ast.parse(Path(vllm_torchtpu.envs.__file__).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        assert imported <= sys.stdlib_module_names, (
            f"envs.py imports non-stdlib modules: "
            f"{sorted(imported - sys.stdlib_module_names)}"
        )

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
            "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT": "/custom/tier3/path",
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "/custom/tier3/path",
            "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": "custom_tier2",
            "TORCH_TPU_TIER2_COMPILATION_CACHE": "custom_tier2",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            import vllm_torchtpu.env_override

            importlib.reload(vllm_torchtpu.env_override)

            assert (
                os.environ.get("TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT")
                == "/custom/tier3/path"
            )
            assert (
                os.environ.get("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT")
                == "/custom/tier3/path"
            )

            assert (
                os.environ.get("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE")
                == "custom_tier2"
            )
            assert os.environ.get("TORCH_TPU_TIER2_COMPILATION_CACHE") == "custom_tier2"

    def test_tpu_compilation_handle_serialization(self):
        """Round-trip the artifact payload TpuCompilerAdaptor.compile writes."""
        import pickle

        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilationHandle

        handle = TpuCompilationHandle(
            key="test_key_123",
            entry="mock_entry",
            signature=(("torch.int32", (16,)),),
            was_wrapped=True,
        )
        loaded = pickle.loads(pickle.dumps(handle))
        assert isinstance(loaded, TpuCompilationHandle)
        assert loaded.key == "test_key_123"
        assert loaded.entry == "mock_entry"
        assert loaded.signature == (("torch.int32", (16,)),)
        assert loaded.was_wrapped is True

    def test_replay_refuses_artifacts_it_cannot_vouch_for(self, tmp_path):
        """Stale or damaged artifacts recompile instead of being replayed.

        The older format stored a bare executable picked off a shared list by
        index, with nothing tying it to the graph it was saved under, so
        replaying one can run the wrong program. Restoring such a pickle leaves
        `entry` at its class default of None, and it is refused on that basis.
        """
        import pickle

        from vllm_torchtpu.compilation.tpu_compiler import (
            TpuCompilationHandle,
            TpuCompilerAdaptor,
        )

        compiler = TpuCompilerAdaptor()
        path = tmp_path / "artifact"

        legacy = object.__new__(TpuCompilationHandle)
        legacy.__dict__.update(key="k", executable="mock_exe")
        assert legacy.entry is None
        path.write_bytes(pickle.dumps(legacy))
        assert compiler._replay(str(path), graph=None, was_wrapped=False) is None

        path.write_bytes(pickle.dumps({"key": "k", "executable": "mock_exe"}))
        assert compiler._replay(str(path), graph=None, was_wrapped=False) is None

        path.write_bytes(b"not a pickle")
        assert compiler._replay(str(path), graph=None, was_wrapped=False) is None

        assert (
            compiler._replay(str(tmp_path / "missing"), graph=None, was_wrapped=False)
            is None
        )

    def test_compiler_initialize_cache(self):
        """Verify that TpuCompilerAdaptor sets cache_dir accurately."""
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        compiler = TpuCompilerAdaptor()
        compiler.initialize_cache(cache_dir="/tmp/test_cache_dir")
        assert compiler.cache_dir == "/tmp/test_cache_dir"

    def test_runtime_hash_excludes_omni_rope(self):
        """Verify that diffusion/Omni specific paths are excluded from the hash."""
        from vllm_torchtpu.compilation.tpu_compiler import (
            _RUNTIME_CACHE_KEY_EXCLUDED_PATHS,
        )

        assert (
            "vllm_torchtpu/kernels/diffusion_attention"
            in _RUNTIME_CACHE_KEY_EXCLUDED_PATHS
        )
        assert "vllm_torchtpu/omni/rope.py" in _RUNTIME_CACHE_KEY_EXCLUDED_PATHS

    def test_run_backend_restarts_scheduling_group_ids(self, monkeypatch):
        """A bucket's HLO, and so its cache key, must not depend on what the
        rank compiled before it."""
        from vllm_torchtpu.compilation import tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor
        from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import config

        firsts = []

        def fake_backend(graph, example_inputs):
            firsts.append(config.reserve_scheduling_group_ids(3))
            return lambda *args: args

        monkeypatch.setattr(tpu_compiler, "_tpu_backend", fake_backend)
        compiler = TpuCompilerAdaptor()
        compiler._run_backend(None, [])
        compiler._run_backend(None, [])
        assert firsts == [1, 1]

    @staticmethod
    def _peer_compilers(tmp_path, monkeypatch, n=2, peer_replay="1"):
        """Compilers for local ranks 0..n-1 of one host, with a fake backend
        that records which ranks really compiled. peer_replay=None leaves
        TPU_PRECOMPILE_PEER_REPLAY unset."""
        import torch
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation import tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        if peer_replay is None:
            monkeypatch.delenv("TPU_PRECOMPILE_PEER_REPLAY", raising=False)
        else:
            monkeypatch.setenv("TPU_PRECOMPILE_PEER_REPLAY", peer_replay)
        compiled, replayed = [], []

        class Compiled:
            def __init__(self, rank):
                self.rank = rank

            def __call__(self, *args):
                return (self.rank,)

            def serialize(self):
                return f"entry-from-rank-{self.rank}"

        def compilers():
            for rank in range(n):
                compiler = TpuCompilerAdaptor()
                compiler.initialize_cache(
                    cache_dir=str(tmp_path / "hash" / f"rank_{rank}_0" / "backbone")
                )
                compiler.rank = rank
                yield compiler

        current = {}

        def fake_backend(graph, example_inputs):
            compiled.append(current["rank"])
            return Compiled(current["rank"])

        def fake_deserialize(entry):
            replayed.append(entry)
            return lambda *args: (entry,)

        monkeypatch.setattr(tpu_compiler, "_tpu_backend", fake_backend)
        monkeypatch.setattr(tpu_compiler, "_deserialize_entry", fake_deserialize)
        monkeypatch.setattr(tpu_compiler, "_PEER_POLL_S", 0.01)

        def compile(compiler, graph=None):
            current["rank"] = compiler.rank
            if graph is None:
                graph = torch.fx.symbolic_trace(lambda x: x * 2 + 1)
            return compiler.compile(
                graph, [], {}, Range(start=1, end=4096), key="artifact_bucket"
            )

        return list(compilers()), compile, compiled, replayed

    def test_a_peer_artifact_is_replayed_instead_of_compiled(
        self, tmp_path, monkeypatch
    ):
        """The second rank of a host replays what the first one saved, and its
        own cache entry is the same file, not a copy."""
        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        compile(a)
        fn, handle = compile(b)

        assert compiled == [0]
        assert replayed == ["entry-from-rank-0"]
        assert fn() == "entry-from-rank-0"
        key, path, _ = handle
        assert path == os.path.join(b.cache_dir, key)
        assert os.path.samefile(path, os.path.join(a.cache_dir, key))
        assert sorted(os.listdir(tmp_path / "hash")) == ["rank_0_0", "rank_1_0"]
        assert os.listdir(b.cache_dir) == [key]

    def test_a_peer_artifact_of_another_graph_is_not_replayed(
        self, tmp_path, monkeypatch
    ):
        """vLLM's key names a bucket, not a graph: a rank that traced
        something else, e.g. with its rank baked in, compiles its own."""
        import torch

        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        compile(a, torch.fx.symbolic_trace(lambda x: x * 2 + 1))
        compile(b, torch.fx.symbolic_trace(lambda x: x * 2 + 2))

        assert compiled == [0, 1]
        assert replayed == []
        assert not os.path.samefile(
            os.path.join(a.cache_dir, "artifact_bucket"),
            os.path.join(b.cache_dir, "artifact_bucket"),
        )

    def test_a_rank_waits_for_the_peer_that_compiles_a_bucket(
        self, tmp_path, monkeypatch
    ):
        import threading

        from vllm_torchtpu.compilation.tpu_compiler import wait_for_peer_artifact

        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        result = {}

        def waiter():
            with wait_for_peer_artifact(30.0):
                result["fn"], _ = compile(b)

        thread = threading.Thread(target=waiter)
        thread.start()
        thread.join(0.2)
        assert thread.is_alive()
        compile(a)
        thread.join(30.0)

        assert not thread.is_alive()
        assert compiled == [0]
        assert result["fn"]() == "entry-from-rank-0"

    def test_a_rank_compiles_itself_when_no_peer_saves_in_time(
        self, tmp_path, monkeypatch
    ):
        from vllm_torchtpu.compilation.tpu_compiler import wait_for_peer_artifact

        (_, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        with wait_for_peer_artifact(0.05):
            compile(b)
        assert compiled == [1]
        assert replayed == []

    def test_a_rank_stops_waiting_when_the_owner_cannot_save(
        self, tmp_path, monkeypatch
    ):
        """An owner with nothing to save leaves a marker; its waiter compiles
        the bucket at once instead of waiting out the full timeout."""
        import time

        from vllm_torchtpu.compilation import tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import wait_for_peer_artifact

        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        backend = tpu_compiler._tpu_backend
        # No serialize(): nothing the owner can hand over.
        monkeypatch.setattr(
            tpu_compiler,
            "_tpu_backend",
            lambda graph, inputs: (lambda *args: backend(graph, inputs)(*args)),
        )
        compile(a)
        assert os.path.exists(os.path.join(a.cache_dir, "artifact_bucket.failed"))
        start = time.monotonic()
        with wait_for_peer_artifact(30.0):
            fn, _ = compile(b)
        assert time.monotonic() - start < 10.0
        assert replayed == []
        # The lazy backend compiles on the first call: b built its own.
        fn()
        assert compiled == [1]

    def test_a_rank_stops_waiting_when_the_owner_save_fails(
        self, tmp_path, monkeypatch
    ):
        """An owner whose save fails (e.g. a full disk) leaves a marker; its
        waiter compiles the bucket at once instead of waiting out the timeout."""
        import time

        from vllm_torchtpu.compilation.tpu_compiler import (
            TpuCompilerAdaptor,
            wait_for_peer_artifact,
        )

        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        monkeypatch.setattr(
            TpuCompilerAdaptor, "_save_handle", lambda self, *args: False
        )
        _, handle = compile(a)
        assert handle is None
        assert os.path.exists(os.path.join(a.cache_dir, "artifact_bucket.failed"))
        start = time.monotonic()
        with wait_for_peer_artifact(30.0):
            compile(b)
        assert time.monotonic() - start < 10.0
        assert compiled == [0, 1]
        assert replayed == []

    def test_a_marker_for_another_graph_is_ignored(self, tmp_path, monkeypatch):
        """A marker from a rank that traced a different graph does not cut the
        wait short: the rank compiling this graph may still save it."""
        import time

        import torch

        from vllm_torchtpu.compilation.tpu_compiler import (
            TpuCompilerAdaptor,
            wait_for_peer_artifact,
        )

        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        monkeypatch.setattr(
            TpuCompilerAdaptor, "_save_handle", lambda self, *args: False
        )
        compile(a, graph=torch.fx.symbolic_trace(lambda x: x * 3 - 1))
        assert os.path.exists(os.path.join(a.cache_dir, "artifact_bucket.failed"))
        start = time.monotonic()
        with wait_for_peer_artifact(0.3):
            compile(b)
        assert time.monotonic() - start >= 0.3
        assert compiled == [0, 1]

    def test_a_marker_from_an_earlier_run_is_ignored(self, tmp_path, monkeypatch):
        """A marker older than this process does not cut the wait short."""
        import time

        from vllm_torchtpu.compilation import tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import wait_for_peer_artifact

        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        os.makedirs(a.cache_dir)
        marker = os.path.join(a.cache_dir, "artifact_bucket.failed")
        open(marker, "w").close()
        old = tpu_compiler._PROCESS_START - 3600
        os.utime(marker, (old, old))
        start = time.monotonic()
        with wait_for_peer_artifact(0.3):
            compile(b)
        assert time.monotonic() - start >= 0.3
        assert compiled == [1]

    def test_a_saved_artifact_clears_an_old_marker(self, tmp_path, monkeypatch):
        (a, _), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        os.makedirs(a.cache_dir)
        marker = os.path.join(a.cache_dir, "artifact_bucket.failed")
        open(marker, "w").close()
        compile(a)
        assert not os.path.exists(marker)
        assert os.path.exists(os.path.join(a.cache_dir, "artifact_bucket"))

    def test_no_peer_replay_outside_a_rank_cache_dir(self, tmp_path, monkeypatch):
        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        a.initialize_cache(cache_dir=str(tmp_path / "x" / "backbone"))
        b.initialize_cache(cache_dir=str(tmp_path / "y" / "backbone"))
        compile(a)
        compile(b)
        assert compiled == [0, 1]

    @pytest.mark.parametrize("peer_replay", ["0", None])
    def test_peer_replay_off_compiles_every_rank_itself(
        self, tmp_path, monkeypatch, peer_replay
    ):
        """TPU_PRECOMPILE_PEER_REPLAY off, explicitly or by default (None): no
        peer lookup, no wait, no link, and no fingerprint recorded; each rank
        compiles and saves its own."""
        import pickle

        from vllm_torchtpu.compilation import tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import (
            TpuCompilerAdaptor,
            wait_for_peer_artifact,
        )

        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch, peer_replay=peer_replay
        )

        def no_peer_lookup(*args, **kwargs):
            raise AssertionError("looked for a peer artifact")

        monkeypatch.setattr(TpuCompilerAdaptor, "_peer_paths", no_peer_lookup)
        monkeypatch.setattr(TpuCompilerAdaptor, "_link_artifact", no_peer_lookup)
        monkeypatch.setattr(tpu_compiler, "_graph_fingerprint", no_peer_lookup)

        compile(a)
        with wait_for_peer_artifact(30.0):
            _, handle = compile(b)

        assert compiled == [0, 1]
        assert replayed == []
        key, path, _ = handle
        assert not os.path.samefile(path, os.path.join(a.cache_dir, key))
        with open(path, "rb") as f:
            assert pickle.load(f).fingerprint is None


def test_deserialize_entry_does_not_copy_the_entry():
    """Replay rebuilds the runnable from the entry itself: a deepcopy would load
    a second TPU executable."""
    from vllm_torchtpu.compilation.tpu_compiler import _deserialize_entry

    class Entry:
        def __init__(self):
            self.guards_expr = "guards"
            self.sanitized_aot_config = object()

        def wrap_post_compile(self, example_inputs, aot_config, options):
            return lambda boxed_args: sum(boxed_args)

        def __deepcopy__(self, memo):
            raise AssertionError("replay must not deepcopy the entry")

    assert _deserialize_entry(Entry())(2, 3, 7) == 12


_RELOAD_EVENTS: list[str] = []
_RELOAD_REQUIRE_RELEASED = False
_RELOAD_FAIL_UNPICKLE = False
_LIVE_COMPILE_LEAVES: list["_ReloadLeafArtifact"] = []


def _reconstruct_reload_leaf(name: str, scale: int) -> "_ReloadLeafArtifact":
    if _RELOAD_FAIL_UNPICKLE:
        raise RuntimeError("corrupt serialized executable")
    if _RELOAD_REQUIRE_RELEASED:
        assert all(leaf.released for leaf in _LIVE_COMPILE_LEAVES), (
            "CompileAndLoad leaves must be released before LoadSerializedExecutable"
        )
        assert "clear_tier1_cache" in _RELOAD_EVENTS, (
            "Tier-1 cache must be cleared before LoadSerializedExecutable"
        )
    _RELOAD_EVENTS.append(f"pickle_loads:{name}")
    return _ReloadLeafArtifact(name=name, scale=scale, from_compile_and_load=False)


class _ReloadLeafArtifact(TorchTpuCompiledExecutable):
    def __init__(
        self, name: str, scale: int, *, from_compile_and_load: bool = True
    ) -> None:
        self.name = name
        self.scale = scale
        self.from_compile_and_load = from_compile_and_load
        self._exec_handle: object | None = None
        self._runner_handle: object | None = None
        super().__init__(
            executable=object(),
            reconstruct_fx_outputs_fn=None,
            updates_default_generator_state=False,
        )
        self._runner_handle = object()
        if from_compile_and_load:
            _LIVE_COMPILE_LEAVES.append(self)

    @property
    def _executable(self) -> object | None:
        return self._exec_handle

    @_executable.setter
    def _executable(self, value: object | None) -> None:
        if value is None and self._exec_handle is not None:
            _RELOAD_EVENTS.append(f"clear_exec:{self.name}")
        self._exec_handle = value

    @property
    def _fast_runner(self) -> object | None:
        return self._runner_handle

    @_fast_runner.setter
    def _fast_runner(self, value: object | None) -> None:
        if value is None and self._runner_handle is not None:
            _RELOAD_EVENTS.append(f"clear_runner:{self.name}")
        self._runner_handle = value

    @property
    def released(self) -> bool:
        return self._exec_handle is None and self._runner_handle is None

    def __call__(self, x):
        assert not self.released, "released executable must not be called"
        return x * self.scale

    def __reduce__(self):
        assert not self.released, "must serialize before releasing executable"
        _RELOAD_EVENTS.append(f"pickle_dumps:{self.name}")
        return (_reconstruct_reload_leaf, (self.name, self.scale))


def _reconstruct_reload_split(leaves: list[Any]) -> "_ReloadSplitArtifact":
    return _ReloadSplitArtifact(leaves)


class _ReloadSplitArtifact(_SplitCompiledExecutable):
    def __init__(self, leaves: list[Any]) -> None:
        super().__init__(torch.fx.GraphModule({}, torch.fx.Graph()))
        self._leaves = leaves

    @property
    def compiled_executables(self) -> list[Any]:
        return self._leaves

    def __call__(self, boxed_args):
        out = boxed_args[0]
        for leaf in self._leaves:
            out = leaf(out)
        return out

    def __reduce__(self):
        return (_reconstruct_reload_split, (self._leaves,))


class _ReloadLoadable:
    def __init__(self, result) -> None:
        self.result = result


class _ReloadBundledEntry:
    def __init__(
        self,
        compiled_fw: _ReloadLoadable | None,
        compiled_bw: _ReloadLoadable | None = None,
        *,
        fail_wrap: bool = False,
    ) -> None:
        self.guards_expr = "guards"
        self.sanitized_aot_config = None
        self.compiled_fw = compiled_fw
        self.compiled_bw = compiled_bw
        self.fail_wrap = fail_wrap

    def wrap_post_compile(self, example_inputs, aot_config, options):
        _RELOAD_EVENTS.append("deserialize_entry")
        if self.fail_wrap:
            raise RuntimeError("wrap_post_compile failed")
        fw = self.compiled_fw
        assert fw is not None
        return lambda boxed_args: (fw.result(boxed_args),)


class TestReloadCompiledExecutable:
    @pytest.fixture(autouse=True)
    def _reset_reload_globals(self, monkeypatch):
        global _RELOAD_REQUIRE_RELEASED, _RELOAD_FAIL_UNPICKLE
        from vllm_torchtpu import envs

        monkeypatch.delitem(
            vars(envs), "VLLM_TPU_RELOAD_COMPILED_EXECUTABLE", raising=False
        )
        monkeypatch.delitem(vars(envs), "VLLM_XLA_CHECK_RECOMPILATION", raising=False)
        monkeypatch.delitem(vars(envs), "TPU_PRECOMPILE_PEER_REPLAY", raising=False)
        _RELOAD_EVENTS.clear()
        _LIVE_COMPILE_LEAVES.clear()
        _RELOAD_REQUIRE_RELEASED = False
        _RELOAD_FAIL_UNPICKLE = False
        yield
        _RELOAD_EVENTS.clear()
        _LIVE_COMPILE_LEAVES.clear()
        _RELOAD_REQUIRE_RELEASED = False
        _RELOAD_FAIL_UNPICKLE = False

    @staticmethod
    def _install_fake_tpu_stats(monkeypatch, counts: list[int]):
        import torch

        idx = {"i": 0}

        def get_stats():
            val = counts[min(idx["i"], len(counts) - 1)]
            idx["i"] += 1
            return SimpleNamespace(num_compilations=val)

        def clear_cache():
            _RELOAD_EVENTS.append("clear_tier1_cache")

        monkeypatch.setattr(
            torch,
            "tpu",
            SimpleNamespace(_get_cache_stats=get_stats, _clear_cache=clear_cache),
            raising=False,
        )

    def test_compile_reloads_in_strict_order_and_persists_disk_handle(
        self, tmp_path, monkeypatch
    ):
        """With disk caching active, cold compile saves the handle *once*,
        synchronously releases all CompileAndLoad leaves + clears Tier-1 cache,
        and replays directly from the saved file without a second serialize."""
        global _RELOAD_REQUIRE_RELEASED
        import torch
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation import shape_variants, tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        _RELOAD_REQUIRE_RELEASED = True
        monkeypatch.setenv("VLLM_TPU_RELOAD_COMPILED_EXECUTABLE", "1")
        monkeypatch.delenv("TPU_PRECOMPILE_PEER_REPLAY", raising=False)
        self._install_fake_tpu_stats(monkeypatch, [0, 2])
        monkeypatch.setattr(shape_variants, "_session", None)

        def fake_backend(graph, example_inputs):
            fw0 = _ReloadLeafArtifact("fw_0", 2)
            fw1 = _ReloadLeafArtifact("fw_1", 5)
            bw0 = _ReloadLeafArtifact("bw_0", 1)
            split_fw = _ReloadSplitArtifact([fw0, fw1])
            entry = _ReloadBundledEntry(_ReloadLoadable(split_fw), _ReloadLoadable(bw0))

            def runnable(x):
                return (split_fw([x]),)

            runnable.serialize = lambda: entry
            return runnable

        monkeypatch.setattr(tpu_compiler, "_tpu_backend", fake_backend)

        compiler = TpuCompilerAdaptor()
        compiler.initialize_cache(cache_dir=str(tmp_path / "cache"))
        graph = torch.fx.symbolic_trace(lambda x: x * 10)
        example_in = [torch.tensor([3, 4])]

        compiled_fn, handle = compiler.compile(
            graph, example_in, {}, Range(start=2, end=2), key="bucket_2"
        )
        assert handle is not None
        assert all(leaf.released for leaf in _LIVE_COMPILE_LEAVES)

        # Verify strict single-serialize ordering:
        # 1. _save_handle pickle.dump (1x) -> 2. clear _fast_runner/_executable
        # -> 3. _clear_cache() -> 4. _replay pickle.load -> 5. _deserialize_entry
        assert _RELOAD_EVENTS == [
            "pickle_dumps:fw_0",
            "pickle_dumps:fw_1",
            "pickle_dumps:bw_0",
            "clear_runner:fw_0",
            "clear_exec:fw_0",
            "clear_runner:fw_1",
            "clear_exec:fw_1",
            "clear_runner:bw_0",
            "clear_exec:bw_0",
            "clear_tier1_cache",
            "pickle_loads:fw_0",
            "pickle_loads:fw_1",
            "pickle_loads:bw_0",
            "deserialize_entry",
        ]

        cold_out = compiled_fn(torch.tensor([3, 4]))
        assert torch.equal(cold_out, torch.tensor([30, 40]))

        # Warm-start load from the handle written before reload must succeed
        # without invoking _tpu_backend again.
        _RELOAD_REQUIRE_RELEASED = False
        monkeypatch.setattr(
            tpu_compiler,
            "_tpu_backend",
            lambda *a, **kw: pytest.fail("warm load should not recompile"),
        )
        warm_compiler = TpuCompilerAdaptor()
        warm_compiler.initialize_cache(cache_dir=str(tmp_path / "cache"))
        warm_fn = warm_compiler.load(
            handle,
            graph,
            example_in,
            graph_index=0,
            compile_range=Range(start=2, end=2),
        )
        assert torch.equal(warm_fn(torch.tensor([3, 4])), cold_out)

    def test_compile_without_disk_cache_reloads_from_in_memory_bytes(self, monkeypatch):
        """When disk caching is inactive, `_reload_compiled` falls back to an
        in-memory `pickle.dumps` -> release -> `pickle.loads` round-trip."""
        global _RELOAD_REQUIRE_RELEASED
        import torch
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation import shape_variants, tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        _RELOAD_REQUIRE_RELEASED = True
        monkeypatch.setenv("VLLM_TPU_RELOAD_COMPILED_EXECUTABLE", "1")
        self._install_fake_tpu_stats(monkeypatch, [3, 4])
        monkeypatch.setattr(shape_variants, "_session", None)

        def fake_backend(graph, example_inputs):
            fw0 = _ReloadLeafArtifact("fw_mem", 6)
            entry = _ReloadBundledEntry(_ReloadLoadable(_ReloadSplitArtifact([fw0])))

            def runnable(x):
                return (fw0(x),)

            runnable.serialize = lambda: entry
            return runnable

        monkeypatch.setattr(tpu_compiler, "_tpu_backend", fake_backend)

        compiler = TpuCompilerAdaptor()
        graph = torch.fx.symbolic_trace(lambda x: x * 6)
        compiled_fn, handle = compiler.compile(
            graph, [torch.tensor([2, 5])], {}, Range(start=2, end=2), key="no_cache"
        )
        assert handle is None
        assert all(leaf.released for leaf in _LIVE_COMPILE_LEAVES)
        assert _RELOAD_EVENTS == [
            "pickle_dumps:fw_mem",
            "clear_runner:fw_mem",
            "clear_exec:fw_mem",
            "clear_tier1_cache",
            "pickle_loads:fw_mem",
            "deserialize_entry",
        ]
        assert torch.equal(compiled_fn(torch.tensor([2, 5])), torch.tensor([12, 30]))

    @pytest.mark.parametrize(
        "reload_env,stats_counts,serializable",
        [
            pytest.param("0", [0, 1], True, id="env-disabled"),
            pytest.param(None, [0, 1], True, id="env-default-off"),
            pytest.param("1", [5, 5], True, id="tier2-or-tier3-cache-hit"),
            pytest.param("1", [0, 1], False, id="unserializable-graph"),
        ],
    )
    def test_compile_skips_reload_when_not_applicable(
        self, monkeypatch, reload_env, stats_counts, serializable
    ):
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation import shape_variants, tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        if reload_env is None:
            monkeypatch.delenv("VLLM_TPU_RELOAD_COMPILED_EXECUTABLE", raising=False)
        else:
            monkeypatch.setenv("VLLM_TPU_RELOAD_COMPILED_EXECUTABLE", reload_env)

        self._install_fake_tpu_stats(monkeypatch, stats_counts)
        monkeypatch.setattr(shape_variants, "_session", None)

        leaf = _ReloadLeafArtifact("fw_0", 7)
        entry = _ReloadBundledEntry(_ReloadLoadable(_ReloadSplitArtifact([leaf])))

        def fake_backend(graph, example_inputs):
            def runnable(x):
                return (leaf(x),)

            if serializable:
                runnable.serialize = lambda: entry
            return runnable

        monkeypatch.setattr(tpu_compiler, "_tpu_backend", fake_backend)

        compiler = TpuCompilerAdaptor()
        graph = torch.fx.symbolic_trace(lambda x: x * 7)
        fn, handle = compiler.compile(
            graph, [torch.tensor([6])], {}, Range(start=1, end=1), key=None
        )
        assert handle is None
        assert not leaf.released
        assert _RELOAD_EVENTS == []
        assert torch.equal(fn(torch.tensor([6])), torch.tensor([42]))

    @pytest.mark.parametrize(
        "broken_stats,expected_exc",
        [
            pytest.param("missing_tpu_attr", AttributeError, id="missing-torch-tpu"),
            pytest.param("runtime_error", RuntimeError, id="broken-get-cache-stats"),
        ],
    )
    def test_get_num_compilations_fails_fast_when_stats_unavailable(
        self, monkeypatch, broken_stats, expected_exc
    ):
        """With `VLLM_TPU_RELOAD_COMPILED_EXECUTABLE=1`, missing or broken
        `torch.tpu._get_cache_stats()` propagates immediately instead of
        silently disabling reload."""
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation import shape_variants
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        monkeypatch.setenv("VLLM_TPU_RELOAD_COMPILED_EXECUTABLE", "1")
        monkeypatch.setattr(shape_variants, "_session", None)
        if broken_stats == "missing_tpu_attr":
            monkeypatch.delattr(torch, "tpu", raising=False)
        else:

            def fail_stats():
                raise RuntimeError("stats unavailable")

            monkeypatch.setattr(
                torch,
                "tpu",
                SimpleNamespace(_get_cache_stats=fail_stats),
                raising=False,
            )

        compiler = TpuCompilerAdaptor()
        graph = torch.fx.symbolic_trace(lambda x: x * 2)
        with pytest.raises(expected_exc):
            compiler.compile(
                graph, [torch.tensor([1])], {}, Range(start=1, end=1), key=None
            )

    def test_clear_tpu_tier1_cache_warns_once_with_check_recompilation(
        self, monkeypatch
    ):
        """`_clear_tpu_tier1_cache` emits `warning_once` only when
        `VLLM_XLA_CHECK_RECOMPILATION` is enabled."""
        from vllm_torchtpu.compilation import tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import _clear_tpu_tier1_cache

        self._install_fake_tpu_stats(monkeypatch, [0])
        tpu_compiler.logger._resolve()

        messages: list[str] = []
        handler = logging.Handler(logging.WARNING)
        handler.emit = lambda record: messages.append(record.getMessage())
        vllm_logger = logging.getLogger("vllm")
        old_level = vllm_logger.level
        old_disable = logging.root.manager.disable
        vllm_logger.addHandler(handler)
        vllm_logger.setLevel(logging.WARNING)
        logging.disable(logging.NOTSET)
        try:
            monkeypatch.setenv("VLLM_XLA_CHECK_RECOMPILATION", "0")
            _clear_tpu_tier1_cache()
            assert messages == []

            monkeypatch.setenv("VLLM_XLA_CHECK_RECOMPILATION", "1")
            _clear_tpu_tier1_cache()
            _clear_tpu_tier1_cache()
        finally:
            vllm_logger.removeHandler(handler)
            vllm_logger.setLevel(old_level)
            logging.disable(old_disable)

        assert len(messages) == 1
        assert "VLLM_XLA_CHECK_RECOMPILATION" in messages[0]
        assert "per_entry_stats" in messages[0]
        assert _RELOAD_EVENTS == [
            "clear_tier1_cache",
            "clear_tier1_cache",
            "clear_tier1_cache",
        ]

    def test_stage1_pickle_failure_keeps_compile_and_load_executable(
        self, tmp_path, monkeypatch
    ):
        """If pickling `entry` fails before release, keep the original runnable
        and do not release leaves or clear the Tier-1 cache."""
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation import shape_variants, tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        monkeypatch.setenv("VLLM_TPU_RELOAD_COMPILED_EXECUTABLE", "1")
        self._install_fake_tpu_stats(monkeypatch, [2, 3])
        monkeypatch.setattr(shape_variants, "_session", None)

        leaf = _ReloadLeafArtifact("fw_0", 4)
        # Local lambda makes `entry` unpicklable in both `_save_handle` and Stage 1.
        unpicklable_entry = SimpleNamespace(
            compiled_fw=_ReloadLoadable(_ReloadSplitArtifact([leaf])),
            compiled_bw=None,
            bad_closure=lambda: None,
        )

        def fake_backend(graph, example_inputs):
            def runnable(x):
                return (leaf(x),)

            runnable.serialize = lambda: unpicklable_entry
            return runnable

        monkeypatch.setattr(tpu_compiler, "_tpu_backend", fake_backend)

        compiler = TpuCompilerAdaptor()
        compiler.initialize_cache(cache_dir=str(tmp_path / "cache"))
        graph = torch.fx.symbolic_trace(lambda x: x * 4)
        fn, handle = compiler.compile(
            graph, [torch.tensor([5])], {}, Range(start=1, end=1), key="unpicklable"
        )
        assert handle is None
        assert not leaf.released
        assert "clear_tier1_cache" not in _RELOAD_EVENTS
        assert torch.equal(fn(torch.tensor([5])), torch.tensor([20]))

    @pytest.mark.parametrize(
        "use_cache,fail_stage",
        [
            pytest.param(False, "pickle_loads", id="mem-unpickle-fails"),
            pytest.param(False, "deserialize_entry", id="mem-deserialize-fails"),
            pytest.param(True, "deserialize_entry", id="disk-replay-resaves-fallback"),
            pytest.param(
                True,
                "deserialize_and_fallback_serialize",
                id="disk-replay-drops-corrupt-handle",
            ),
        ],
    )
    def test_stage3_failure_falls_back_with_pristine_orig_graph(
        self, tmp_path, monkeypatch, use_cache, fail_stage
    ):
        """If reload fails after releasing the original executable, fallback
        recompiles from an unmutated copy of `orig_graph` and repairs or drops
        the on-disk artifact."""
        global _RELOAD_FAIL_UNPICKLE
        import os

        import torch
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation import shape_variants, tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor
        from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import config

        monkeypatch.setenv("VLLM_TPU_RELOAD_COMPILED_EXECUTABLE", "1")
        if use_cache:
            monkeypatch.setenv("TPU_PRECOMPILE_PEER_REPLAY", "1")
        else:
            monkeypatch.delenv("TPU_PRECOMPILE_PEER_REPLAY", raising=False)
        self._install_fake_tpu_stats(monkeypatch, [0, 1])
        monkeypatch.setattr(shape_variants, "_session", None)
        if fail_stage == "pickle_loads":
            _RELOAD_FAIL_UNPICKLE = True

        backend_calls = []
        group_starts = []
        saw_mutated_flag = []

        def fake_backend(graph, example_inputs):
            call_idx = len(backend_calls)
            backend_calls.append(call_idx)
            group_starts.append(config.reserve_scheduling_group_ids(2))
            saw_mutated_flag.append(getattr(graph, "_pre_grad_mutated", False))
            # Simulate TpuBackend.execute_pre_grad_passes mutating `graph` in place.
            graph._pre_grad_mutated = True
            leaf = _ReloadLeafArtifact(f"call_{call_idx}", 9)
            entry = _ReloadBundledEntry(
                _ReloadLoadable(_ReloadSplitArtifact([leaf])),
                fail_wrap=(fail_stage != "pickle_loads" and call_idx == 0),
            )

            def runnable(x):
                return (leaf(x),)

            if fail_stage == "deserialize_and_fallback_serialize" and call_idx == 1:

                def broken_serialize():
                    raise RuntimeError("fallback serialize failed")

                runnable.serialize = broken_serialize
            else:
                runnable.serialize = lambda: entry
            return runnable

        monkeypatch.setattr(tpu_compiler, "_tpu_backend", fake_backend)

        compiler = TpuCompilerAdaptor()
        cache_dir = tmp_path / "rank_0_0" / "cache"
        if use_cache:
            cache_dir.mkdir(parents=True)
            (cache_dir / "bucket_1.failed").write_text("stale")
            compiler.initialize_cache(cache_dir=str(cache_dir))
        orig_graph = torch.fx.symbolic_trace(lambda x: x * 9)
        fn, handle = compiler.compile(
            orig_graph,
            [torch.tensor([3])],
            {},
            Range(start=1, end=1),
            key="bucket_1" if use_cache else None,
        )
        assert backend_calls == [0, 1]
        assert group_starts == [1, 1]
        # Both calls must see a fresh copy without `_pre_grad_mutated`.
        assert saw_mutated_flag == [False, False]
        assert not hasattr(orig_graph, "_pre_grad_mutated")
        assert _LIVE_COMPILE_LEAVES[0].released is True
        assert _LIVE_COMPILE_LEAVES[1].released is False
        assert torch.equal(fn(torch.tensor([3])), torch.tensor([27]))
        if use_cache and fail_stage == "deserialize_entry":
            assert handle is not None
            assert os.path.exists(handle[1])
            assert not os.path.exists(str(cache_dir / "bucket_1.failed"))
        else:
            assert handle is None
            if use_cache:
                assert not os.path.exists(str(cache_dir / "bucket_1"))
                assert os.path.exists(str(cache_dir / "bucket_1.failed"))

    def test_load_stale_self_heal_reloads_and_cleans_broken_fallback(
        self, tmp_path, monkeypatch
    ):
        """When `load()` encounters a corrupt cache file, its self-heal compile
        also saves once + reloads (and removes the file if fallback cannot save)."""
        import os

        import torch
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation import shape_variants, tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        monkeypatch.setenv("VLLM_TPU_RELOAD_COMPILED_EXECUTABLE", "1")
        monkeypatch.delenv("TPU_PRECOMPILE_PEER_REPLAY", raising=False)
        monkeypatch.setattr(shape_variants, "_session", None)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        artifact_path = cache_dir / "stale_bucket"
        artifact_path.write_bytes(b"corrupt")

        # Part 1: Normal self-heal with single-serialize reload.
        self._install_fake_tpu_stats(monkeypatch, [0, 1])

        def healthy_backend(graph, example_inputs):
            leaf = _ReloadLeafArtifact("heal_0", 5)
            entry = _ReloadBundledEntry(_ReloadLoadable(_ReloadSplitArtifact([leaf])))

            def runnable(x):
                return (leaf(x),)

            runnable.serialize = lambda: entry
            return runnable

        monkeypatch.setattr(tpu_compiler, "_tpu_backend", healthy_backend)
        compiler = TpuCompilerAdaptor()
        compiler.initialize_cache(cache_dir=str(cache_dir))
        graph = torch.fx.symbolic_trace(lambda x: x * 5)
        handle = ("stale_bucket", str(artifact_path), True)
        fn = compiler.load(
            handle, graph, [torch.tensor([4])], graph_index=0, compile_range=Range(1, 1)
        )
        assert torch.equal(fn(torch.tensor([4])), torch.tensor([20]))
        assert _RELOAD_EVENTS.count("pickle_dumps:heal_0") == 1
        assert os.path.exists(artifact_path)

        # Part 2: Corrupt again, and fail both reload replay and fallback serialize.
        _RELOAD_EVENTS.clear()
        _LIVE_COMPILE_LEAVES.clear()
        artifact_path.write_bytes(b"corrupt")
        monkeypatch.setattr(shape_variants, "_session", None)
        self._install_fake_tpu_stats(monkeypatch, [1, 2])
        calls = {"n": 0}

        def fallback_unserializable_backend(graph, example_inputs):
            idx = calls["n"]
            calls["n"] += 1
            leaf = _ReloadLeafArtifact(f"heal_fb_{idx}", 8)
            entry = _ReloadBundledEntry(
                _ReloadLoadable(_ReloadSplitArtifact([leaf])), fail_wrap=(idx == 0)
            )

            def runnable(x):
                return (leaf(x),)

            if idx == 0:
                runnable.serialize = lambda: entry
            return runnable

        monkeypatch.setattr(
            tpu_compiler, "_tpu_backend", fallback_unserializable_backend
        )
        fn2 = compiler.load(
            handle, graph, [torch.tensor([4])], graph_index=0, compile_range=Range(1, 1)
        )
        assert torch.equal(fn2(torch.tensor([4])), torch.tensor([32]))
        assert not os.path.exists(artifact_path)

    def test_reload_compiled_fallback_without_orig_graph(self, monkeypatch):
        """Direct `_reload_compiled` drops `reloaded_entry` on serialized-bytes
        success and falls back to `graph` when `orig_graph=None`."""
        global _RELOAD_FAIL_UNPICKLE

        from vllm_torchtpu.compilation import tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        monkeypatch.setenv("VLLM_TPU_RELOAD_COMPILED_EXECUTABLE", "1")
        self._install_fake_tpu_stats(monkeypatch, [0, 1])

        ok_leaf = _ReloadLeafArtifact("direct_ok", 5)
        ok_entry = _ReloadBundledEntry(_ReloadLoadable(_ReloadSplitArtifact([ok_leaf])))
        compiler = TpuCompilerAdaptor()
        graph = torch.fx.symbolic_trace(lambda x: (x * 11,))
        ok_fn, ok_fb_entry, ok_reloaded = compiler._reload_compiled(
            lambda x: (ok_leaf(x),),
            ok_entry,
            compilations=(0, 1),
            graph=graph,
            orig_graph=None,
            example_inputs=[torch.tensor([2])],
            was_wrapped=False,
            saved_path=None,
        )
        assert ok_reloaded is True
        assert ok_fb_entry is None
        assert torch.equal(ok_fn(torch.tensor([2]))[0], torch.tensor([10]))

        _RELOAD_EVENTS.clear()
        _RELOAD_FAIL_UNPICKLE = True
        leaf0 = _ReloadLeafArtifact("direct_0", 3)
        entry0 = _ReloadBundledEntry(_ReloadLoadable(_ReloadSplitArtifact([leaf0])))

        def fake_backend(graph, example_inputs):
            leaf1 = _ReloadLeafArtifact("direct_1", 11)
            return lambda x: (leaf1(x),)

        monkeypatch.setattr(tpu_compiler, "_tpu_backend", fake_backend)
        fn, fb_entry, reloaded = compiler._reload_compiled(
            lambda x: (leaf0(x),),
            entry0,
            compilations=(0, 1),
            graph=graph,
            orig_graph=None,
            example_inputs=[torch.tensor([2])],
            was_wrapped=False,
            saved_path=None,
        )
        assert reloaded is False
        assert fb_entry is None
        assert "clear_tier1_cache" in _RELOAD_EVENTS
        assert torch.equal(fn(torch.tensor([2]))[0], torch.tensor([22]))

    def test_release_compiled_artifact_dispatches_all_supported_types_and_cycles(
        self,
    ):
        """`_release_entry_executables` resolves `AsyncCompiledArtifact`, visits
        shared/cyclic `_SplitCompiledExecutable` nodes once, skips
        `NoOpCompiledArtifact` and `None`, and clears leaf handles."""
        from vllm_torchtpu.compilation.tpu_compiler import (
            _release_entry_executables,
        )

        shared_leaf = _ReloadLeafArtifact("shared", 2, from_compile_and_load=False)
        async_target = _ReloadLeafArtifact("async_leaf", 3, from_compile_and_load=False)
        fut: concurrent.futures.Future = concurrent.futures.Future()
        fut.set_result(async_target)
        async_artifact = AsyncCompiledArtifact(fut)
        noop_artifact = NoOpCompiledArtifact(reconstruct_fx_outputs_fn=None)
        empty_split = _ReloadSplitArtifact([])

        cyclic_split = _ReloadSplitArtifact([])
        cyclic_split._leaves.extend(
            [
                cyclic_split,
                shared_leaf,
                shared_leaf,
                async_artifact,
                noop_artifact,
                empty_split,
            ]
        )
        entry = _ReloadBundledEntry(
            compiled_fw=_ReloadLoadable(cyclic_split),
            compiled_bw=_ReloadLoadable(None),
        )

        _release_entry_executables(entry)

        assert shared_leaf.released
        assert isinstance(async_artifact, _ReloadLeafArtifact)
        assert async_artifact.released
        assert _RELOAD_EVENTS == [
            "clear_runner:shared",
            "clear_exec:shared",
            "clear_runner:async_leaf",
            "clear_exec:async_leaf",
        ]

    @pytest.mark.parametrize(
        "failure_kind,expected_exc,match",
        [
            pytest.param(
                "unsupported_type",
                TypeError,
                "Unsupported CompiledArtifact type",
                id="unsupported-artifact-type",
            ),
            pytest.param(
                "readonly_exec",
                AttributeError,
                None,
                id="readonly-executable-property",
            ),
            pytest.param(
                "readonly_runner",
                AttributeError,
                None,
                id="readonly-fast-runner-property",
            ),
        ],
    )
    def test_release_compiled_artifact_fails_fast_on_incompatible_artifacts(
        self, failure_kind, expected_exc, match
    ):
        """`_release_compiled_artifact` raises immediately on unknown artifact
        types or read-only handle descriptors."""
        from vllm_torchtpu.compilation.tpu_compiler import _release_compiled_artifact

        class _ReadOnlyExecArtifact(TorchTpuCompiledExecutable):
            def __init__(self) -> None:
                self._fast_runner = object()

            @property
            def _executable(self) -> object:
                return "readonly_handle"

        class _ReadOnlyRunnerArtifact(TorchTpuCompiledExecutable):
            def __init__(self) -> None:
                self._executable = object()

            @property
            def _fast_runner(self) -> object:
                return "readonly_runner_handle"

        artifacts = {
            "unsupported_type": SimpleNamespace(
                _executable=object(), _fast_runner=object()
            ),
            "readonly_exec": _ReadOnlyExecArtifact(),
            "readonly_runner": _ReadOnlyRunnerArtifact(),
        }

        with pytest.raises(expected_exc, match=match):
            _release_compiled_artifact(artifacts[failure_kind])

    def test_reload_env_var_registered_for_ray_workers(self):
        """Ray worker propagation copies `TpuPlatform.additional_env_vars`."""
        from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

        assert "VLLM_TPU_RELOAD_COMPILED_EXECUTABLE" in TpuPlatform.additional_env_vars
