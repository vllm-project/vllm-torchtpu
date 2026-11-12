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

import dataclasses
import importlib
import os
import pickle
import re
import sys
import weakref
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch
from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
from torch_tpu._internal.compile.split_compiler import _SplitCompiledExecutable
from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
    NoOpCompiledArtifact,
    TorchTpuCompiledExecutable,
)

from tests.compilation.test_graph_fingerprint import (
    _trace_all_reduce_graph,
    install_group_stand_ins,
)
from vllm_torchtpu.env_override import _TPU_VMEM_BYTES_PER_TENSOR_CORE


@pytest.fixture
def group_stand_ins(monkeypatch):
    return install_group_stand_ins(monkeypatch)


class TestTpuCompilerCache:
    @staticmethod
    def _config(
        *,
        max_num_seqs=16,
        num_gpu_blocks=1024,
        mamba_cache_mode="none",
        prefix_match_unit=None,
    ):
        return SimpleNamespace(
            cache_config=SimpleNamespace(
                gpu_memory_utilization=0.9,
                num_gpu_blocks=num_gpu_blocks,
                num_gpu_blocks_override=None,
                mamba_cache_mode=mamba_cache_mode,
                mamba_block_size=3072,
                prefix_match_unit=prefix_match_unit,
            ),
            model_config=SimpleNamespace(
                max_model_len=4096, architecture="Qwen3_5MoeForCausalLM"
            ),
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=2048,
                max_num_seqs=max_num_seqs,
            ),
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
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

    def test_tpu_hash_covers_gdn_prefill_checkpoint_gate_only(self):
        """The hash follows prefill_checkpoint_enabled, not its raw inputs."""
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            disabled = compute_tpu_compilation_hash(self._config())
            enabled = compute_tpu_compilation_hash(
                self._config(mamba_cache_mode="align")
            )
            enabled_block_unit = compute_tpu_compilation_hash(
                self._config(mamba_cache_mode="align", prefix_match_unit=3072)
            )
            disabled_fine_unit = compute_tpu_compilation_hash(
                self._config(mamba_cache_mode="align", prefix_match_unit=1024)
            )

        assert disabled != enabled
        assert enabled == enabled_block_unit
        assert disabled_fine_unit == disabled

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

    def test_tpu_hash_ignores_the_raiden_advertise_host(self):
        """The advertised host is the pod IP: hashing it gives every pod its
        own compile cache, so a cache restored from an earlier pod misses."""
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            with patch.dict(os.environ, {"TPU_RAIDEN_ADVERTISE_HOST": "10.0.0.1"}):
                first_pod = compute_tpu_compilation_hash(self._config())
            with patch.dict(os.environ, {"TPU_RAIDEN_ADVERTISE_HOST": "10.0.0.2"}):
                second_pod = compute_tpu_compilation_hash(self._config())

        assert first_pod == second_pod

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

    def test_tpu_hash_ignores_the_expert_placement(self):
        """A placement permutes weights and router tables at load and leaves
        every program as it was, so a placed server reuses the cache."""
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        def placed(value):
            with patch.dict(os.environ, {"TPU_EXPERT_PLACEMENT": value}):
                return compute_tpu_compilation_hash(self._config())

        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            assert placed("") == placed("/tmp/placement.json")

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
            "vllm_torchtpu/gdn_pool_layout.py",
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
            "TPU_RAIDEN_TRANSFER_STREAMS",
            "TPU_RAIDEN_ADVERTISE_HOST",
            "TPU_RAIDEN_RESHARD_PORT_BASE",
            "TPU_RAIDEN_STORE_DISPATCH_PORT_BASE",
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

    def test_a_peer_artifact_is_replayed_across_c10d_subgroups(
        self, tmp_path, monkeypatch, group_stand_ins
    ):
        """Ranks in distinct DP/TP subgroups trace different rank-local c10d
        group_name strings for the same collective role, yet replay the first
        rank's compiled artifact."""
        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        compile(a, _trace_all_reduce_graph("1"))
        fn, handle = compile(b, _trace_all_reduce_graph("7"))

        assert compiled == [0]
        assert replayed == ["entry-from-rank-0"]
        assert fn() == "entry-from-rank-0"
        key, path, _ = handle
        assert a.cache_dir is not None
        assert os.path.samefile(path, os.path.join(a.cache_dir, key))


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


@dataclasses.dataclass
class _ReloadState:
    """What the reload fakes record. Module-level because pickled leaves are
    rebuilt by `_reconstruct_reload_leaf`, which only sees module globals."""

    events: list[str] = dataclasses.field(default_factory=list)
    num_compilations: int = 0
    # Leaves and runnables the fake backend built, as CompileAndLoad would.
    live_leaves: list["_ReloadLeafArtifact"] = dataclasses.field(default_factory=list)
    runnables: list[weakref.ref] = dataclasses.field(default_factory=list)
    # When set, loading a leaf asserts nothing compiled is still alive.
    check_released_on_load: bool = False
    fail_unpickle: bool = False


_reload_state = _ReloadState()


def _reconstruct_reload_leaf(name: str, scale: int) -> "_ReloadLeafArtifact":
    state = _reload_state
    if state.fail_unpickle:
        raise RuntimeError("corrupt serialized executable")
    if state.check_released_on_load:
        assert all(leaf.released for leaf in state.live_leaves), (
            "CompileAndLoad leaves must be released before LoadSerializedExecutable"
        )
        assert all(ref() is None for ref in state.runnables), (
            "the CompileAndLoad runnable must be dropped before "
            "LoadSerializedExecutable"
        )
        assert "clear_tier1_cache" in state.events
    state.events.append(f"pickle_loads:{name}")
    return _ReloadLeafArtifact(name, scale)


class _ReloadLeafArtifact(TorchTpuCompiledExecutable):
    """Leaf executable computing `x * scale` that logs being pickled."""

    def __init__(self, name: str, scale: int) -> None:
        super().__init__(
            executable=object(),
            reconstruct_fx_outputs_fn=None,
            updates_default_generator_state=False,
        )
        self._fast_runner = object()
        self.name = name
        self.scale = scale

    @property
    def released(self) -> bool:
        return self._executable is None and self._fast_runner is None

    def __call__(self, x):
        assert not self.released, "released executable must not be called"
        return x * self.scale

    def __reduce__(self):
        assert not self.released, "must serialize before releasing executable"
        _reload_state.events.append(f"pickle_dumps:{self.name}")
        return (_reconstruct_reload_leaf, (self.name, self.scale))


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
        return (_ReloadSplitArtifact, (self._leaves,))


class _ReloadLoadable:
    def __init__(self, result) -> None:
        self.result = result


class _ReloadBundledEntry:
    """Bundled AOTAutograd entry, as far as `_deserialize_entry` reads it."""

    def __init__(self, compiled_fw, compiled_bw=None, *, fail_wrap=False) -> None:
        self.guards_expr = "guards"
        self.sanitized_aot_config = None
        self.compiled_fw = compiled_fw
        self.compiled_bw = compiled_bw
        self.fail_wrap = fail_wrap

    def wrap_post_compile(self, example_inputs, aot_config, options):
        _reload_state.events.append("deserialize_entry")
        if self.fail_wrap:
            raise RuntimeError("wrap_post_compile failed")
        fw = self.compiled_fw.result
        return lambda boxed_args: (fw(boxed_args),)


class _FakeTpuBackend:
    """TpuBackend minus torch_tpu: call `i` builds a forward split of leaves
    `c<i>_fw0` (x * scale) and `c<i>_fw1` (x * 1), plus a `c<i>_bw` leaf."""

    def __init__(
        self,
        state: _ReloadState,
        scale: int,
        *,
        cache_hit: bool = False,
        unserializable: frozenset[int] = frozenset(),
        broken_serialize: frozenset[int] = frozenset(),
        fail_wrap: frozenset[int] = frozenset(),
    ) -> None:
        self.state = state
        self.scale = scale
        self.cache_hit = cache_hit
        self.unserializable = unserializable
        self.broken_serialize = broken_serialize
        self.fail_wrap = fail_wrap
        self.calls = 0
        self.group_starts: list[int] = []
        self.saw_mutated: list[bool] = []

    def __call__(self, graph, example_inputs):
        from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import config

        idx = self.calls
        self.calls += 1
        self.group_starts.append(config.reserve_scheduling_group_ids(2))
        # TpuBackend's pre-grad passes mutate the graph it compiles in place.
        self.saw_mutated.append(getattr(graph, "_pre_grad_mutated", False))
        graph._pre_grad_mutated = True
        if not self.cache_hit:
            self.state.num_compilations += 1

        fw = [
            _ReloadLeafArtifact(f"c{idx}_fw0", self.scale),
            _ReloadLeafArtifact(f"c{idx}_fw1", 1),
        ]
        bw = _ReloadLeafArtifact(f"c{idx}_bw", 1)
        self.state.live_leaves.extend([*fw, bw])
        split = _ReloadSplitArtifact(fw)
        entry = _ReloadBundledEntry(
            _ReloadLoadable(split),
            _ReloadLoadable(bw),
            fail_wrap=idx in self.fail_wrap,
        )
        # torch saves the entry here, as TpuBackend's injected cache_info does.
        AOTAutogradCache.save("aot-key", entry, remote=False)

        def runnable(x):
            return (split([x]),)

        def broken():
            raise RuntimeError("serialize failed")

        if idx in self.broken_serialize:
            runnable.serialize = broken
        elif idx not in self.unserializable:
            runnable.serialize = lambda: entry
        self.state.runnables.append(weakref.ref(runnable))
        return runnable


def _compile(compiler, *, key="bucket", graph=None):
    from vllm.config.utils import Range

    graph = graph if graph is not None else torch.fx.symbolic_trace(lambda x: x)
    return compiler.compile(
        graph, [torch.tensor([3, 4])], {}, Range(start=2, end=2), key=key
    )


def _adaptor(cache_dir=None):
    from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

    compiler = TpuCompilerAdaptor()
    if cache_dir is not None:
        compiler.initialize_cache(cache_dir=str(cache_dir))
    return compiler


class TestReloadCompiledExecutable:
    @pytest.fixture
    def reload_state(self, monkeypatch):
        """A fresh `_ReloadState`, a fake `torch.tpu` cache, and reload on."""
        from vllm_torchtpu import envs
        from vllm_torchtpu.compilation import shape_variants

        state = _ReloadState()
        monkeypatch.setattr(sys.modules[__name__], "_reload_state", state)
        for name in (
            "VLLM_TPU_RELOAD_COMPILED_EXECUTABLE",
            "VLLM_XLA_CHECK_RECOMPILATION",
            "TPU_PRECOMPILE_PEER_REPLAY",
        ):
            monkeypatch.delitem(vars(envs), name, raising=False)
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("VLLM_TPU_RELOAD_COMPILED_EXECUTABLE", "1")
        monkeypatch.setattr(shape_variants, "_session", None)
        monkeypatch.setattr(
            torch,
            "tpu",
            SimpleNamespace(
                _get_cache_stats=lambda: SimpleNamespace(
                    num_compilations=state.num_compilations
                ),
                _clear_cache=lambda: state.events.append("clear_tier1_cache"),
            ),
            raising=False,
        )
        monkeypatch.setattr(
            AOTAutogradCache,
            "save",
            staticmethod(lambda *a, **kw: state.events.append("aot_cache_save")),
        )
        return state

    @staticmethod
    def _install(monkeypatch, backend):
        from vllm_torchtpu.compilation import tpu_compiler

        monkeypatch.setattr(tpu_compiler, "_tpu_backend", backend)
        return backend

    @pytest.mark.parametrize("use_disk_cache", [True, False], ids=["disk", "memory"])
    def test_compile_reloads_released_executable(
        self, tmp_path, monkeypatch, reload_state, captured_report, use_disk_cache
    ):
        """A new CompileAndLoad is serialized once (to the artifact, or to bytes
        without a disk cache), released with the runnable dropped and Tier-1
        cleared, and only then loaded back. torch's own AOTAutogradCache save
        does not run."""
        self._install(monkeypatch, _FakeTpuBackend(reload_state, 10))
        reload_state.check_released_on_load = True
        compiler = _adaptor(tmp_path / "cache" if use_disk_cache else None)

        with captured_report() as messages:
            fn, handle = _compile(compiler)

        assert (handle is not None) == use_disk_cache
        assert all(leaf.released for leaf in reload_state.live_leaves)
        assert reload_state.events == [
            "pickle_dumps:c0_fw0",
            "pickle_dumps:c0_fw1",
            "pickle_dumps:c0_bw",
            "clear_tier1_cache",
            "pickle_loads:c0_fw0",
            "pickle_loads:c0_fw1",
            "pickle_loads:c0_bw",
            "deserialize_entry",
        ]
        assert any("compilations 0 -> 1" in m for m in messages)
        assert any("Reloaded compiled executable" in m for m in messages)
        assert torch.equal(fn(torch.tensor([3, 4])), torch.tensor([30, 40]))

    def test_warm_load_replays_reloaded_handle_without_recompile(
        self, tmp_path, monkeypatch, reload_state
    ):
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation import shape_variants

        self._install(monkeypatch, _FakeTpuBackend(reload_state, 10))
        graph = torch.fx.symbolic_trace(lambda x: x)
        _, handle = _compile(_adaptor(tmp_path / "cache"), graph=graph)
        assert handle is not None

        monkeypatch.setattr(shape_variants, "_session", None)
        self._install(
            monkeypatch, lambda *a: pytest.fail("warm load must not recompile")
        )
        warm_fn = _adaptor(tmp_path / "cache").load(
            handle, graph, [torch.tensor([3, 4])], 0, Range(start=2, end=2)
        )
        assert torch.equal(warm_fn(torch.tensor([3, 4])), torch.tensor([30, 40]))
        assert reload_state.num_compilations == 1

    @pytest.mark.parametrize(
        "reload_env,cache_hit,serializable",
        [
            pytest.param("0", False, True, id="env-disabled"),
            pytest.param(None, False, True, id="env-default-off"),
            pytest.param("1", True, True, id="tier2-or-tier3-cache-hit"),
            pytest.param("1", False, False, id="unserializable-graph"),
        ],
    )
    def test_compile_skips_reload_when_not_applicable(
        self, tmp_path, monkeypatch, reload_state, reload_env, cache_hit, serializable
    ):
        """Without a new CompileAndLoad, or anything to reload from, the
        compiled executable is kept, and with reload off torch's
        AOTAutogradCache save still runs untouched."""
        if reload_env is None:
            monkeypatch.delenv("VLLM_TPU_RELOAD_COMPILED_EXECUTABLE")
        else:
            monkeypatch.setenv("VLLM_TPU_RELOAD_COMPILED_EXECUTABLE", reload_env)
        self._install(
            monkeypatch,
            _FakeTpuBackend(
                reload_state,
                7,
                cache_hit=cache_hit,
                unserializable=frozenset() if serializable else frozenset({0}),
            ),
        )

        fn, handle = _compile(_adaptor(tmp_path / "cache"))

        assert (handle is not None) == serializable
        assert not any(leaf.released for leaf in reload_state.live_leaves)
        assert "clear_tier1_cache" not in reload_state.events
        assert ("aot_cache_save" in reload_state.events) == (reload_env != "1")
        assert torch.equal(fn(torch.tensor([3, 4])), torch.tensor([21, 28]))

    @pytest.mark.parametrize(
        "tpu_module,expected_exc",
        [
            pytest.param(None, AttributeError, id="missing-torch-tpu"),
            pytest.param(
                SimpleNamespace(_get_cache_stats=Mock(side_effect=RuntimeError)),
                RuntimeError,
                id="broken-get-cache-stats",
            ),
        ],
    )
    def test_reload_fails_fast_when_cache_stats_are_unavailable(
        self, monkeypatch, reload_state, tpu_module, expected_exc
    ):
        """A missing or broken `torch.tpu._get_cache_stats()` raises instead of
        silently disabling reload."""
        if tpu_module is None:
            monkeypatch.delattr(torch, "tpu")
        else:
            monkeypatch.setattr(torch, "tpu", tpu_module)
        with pytest.raises(expected_exc):
            _compile(_adaptor(), key=None)

    def test_clear_tpu_tier1_cache_warns_once_with_check_recompilation(
        self, monkeypatch, reload_state, captured_report
    ):
        from vllm_torchtpu.compilation.torch_tpu_release import clear_tpu_tier1_cache

        with captured_report() as messages:
            monkeypatch.setenv("VLLM_XLA_CHECK_RECOMPILATION", "0")
            clear_tpu_tier1_cache()
            assert messages == []
            monkeypatch.setenv("VLLM_XLA_CHECK_RECOMPILATION", "1")
            clear_tpu_tier1_cache()
            clear_tpu_tier1_cache()

        assert len(messages) == 1
        assert "VLLM_XLA_CHECK_RECOMPILATION" in messages[0]
        assert reload_state.events == ["clear_tier1_cache"] * 3

    def test_unpicklable_entry_keeps_compile_and_load_executable(
        self, tmp_path, monkeypatch, reload_state
    ):
        """If the entry can be neither saved nor pickled, nothing is released
        and the compiled executable keeps serving."""
        backend = self._install(monkeypatch, _FakeTpuBackend(reload_state, 4))

        def refuse(self):
            raise pickle.PicklingError("not serializable")

        monkeypatch.setattr(_ReloadLeafArtifact, "__reduce__", refuse)

        fn, handle = _compile(_adaptor(tmp_path / "cache"))

        assert backend.calls == 1
        assert handle is None
        assert not any(leaf.released for leaf in reload_state.live_leaves)
        assert "clear_tier1_cache" not in reload_state.events
        assert torch.equal(fn(torch.tensor([3, 4])), torch.tensor([12, 16]))

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
    def test_failed_reload_recompiles_from_pristine_orig_graph(
        self, tmp_path, monkeypatch, reload_state, use_cache, fail_stage
    ):
        """A reload failing after the release recompiles from an unmutated copy
        of `orig_graph`, and repairs or drops the on-disk artifact."""
        if use_cache:
            monkeypatch.setenv("TPU_PRECOMPILE_PEER_REPLAY", "1")
        reload_state.fail_unpickle = fail_stage == "pickle_loads"
        backend = self._install(
            monkeypatch,
            _FakeTpuBackend(
                reload_state,
                9,
                fail_wrap=frozenset()
                if fail_stage == "pickle_loads"
                else frozenset({0}),
                broken_serialize=frozenset({1})
                if fail_stage == "deserialize_and_fallback_serialize"
                else frozenset(),
            ),
        )
        cache_dir = tmp_path / "rank_0_0" / "cache"
        if use_cache:
            cache_dir.mkdir(parents=True)
            (cache_dir / "bucket.failed").write_text("stale")
        orig_graph = torch.fx.symbolic_trace(lambda x: x)

        fn, handle = _compile(
            _adaptor(cache_dir if use_cache else None),
            key="bucket" if use_cache else None,
            graph=orig_graph,
        )

        assert backend.calls == 2
        assert backend.group_starts == [1, 1]
        assert backend.saw_mutated == [False, False]
        assert not hasattr(orig_graph, "_pre_grad_mutated")
        first, second = reload_state.live_leaves[:3], reload_state.live_leaves[3:]
        assert all(leaf.released for leaf in first)
        assert not any(leaf.released for leaf in second)
        assert torch.equal(fn(torch.tensor([3, 4])), torch.tensor([27, 36]))
        if fail_stage == "deserialize_entry" and use_cache:
            assert handle is not None
            assert os.path.exists(handle[1])
            assert not (cache_dir / "bucket.failed").exists()
        else:
            assert handle is None
            if use_cache:
                assert not (cache_dir / "bucket").exists()
                assert (cache_dir / "bucket.failed").exists()

    def test_load_self_heal_saves_once_and_reloads(
        self, tmp_path, monkeypatch, reload_state
    ):
        """`load()` recompiling a corrupt artifact saves it once and reloads."""
        from vllm.config.utils import Range

        self._install(monkeypatch, _FakeTpuBackend(reload_state, 5))
        artifact = tmp_path / "stale_bucket"
        artifact.write_bytes(b"corrupt")
        compiler = _adaptor(tmp_path)

        fn = compiler.load(
            ("stale_bucket", str(artifact), True),
            torch.fx.symbolic_trace(lambda x: x),
            [torch.tensor([3, 4])],
            0,
            Range(start=2, end=2),
        )

        assert reload_state.events.count("pickle_dumps:c0_fw0") == 1
        assert all(leaf.released for leaf in reload_state.live_leaves)
        assert torch.equal(fn(torch.tensor([3, 4])), torch.tensor([15, 20]))
        assert artifact.read_bytes() != b"corrupt"

    def test_release_entry_executables_clears_every_leaf_once(self, reload_state):
        """Shared and cyclic splits are visited once; `NoOpCompiledArtifact`
        and empty results are skipped."""
        from vllm_torchtpu.compilation.torch_tpu_release import (
            release_entry_executables,
        )

        shared = _ReloadLeafArtifact("shared", 2)
        cyclic = _ReloadSplitArtifact([])
        cyclic._leaves.extend(
            [
                cyclic,
                shared,
                shared,
                NoOpCompiledArtifact(reconstruct_fx_outputs_fn=None),
                _ReloadSplitArtifact([]),
            ]
        )
        bw = _ReloadLeafArtifact("bw", 1)

        release_entry_executables(_ReloadBundledEntry(_ReloadLoadable(cyclic)))
        release_entry_executables(
            _ReloadBundledEntry(_ReloadLoadable(None), _ReloadLoadable(bw))
        )

        assert shared.released
        assert bw.released

    def test_release_entry_executables_rejects_unsupported_types_unreleased(
        self, reload_state
    ):
        """An unsupported artifact anywhere in the entry raises `TypeError`
        before any leaf, earlier in the tree or in the forward, is cleared."""
        from vllm_torchtpu.compilation.torch_tpu_release import (
            release_entry_executables,
        )

        unsupported = SimpleNamespace(_executable=object(), _fast_runner=object())
        earlier = _ReloadLeafArtifact("earlier", 2)
        fw_leaf = _ReloadLeafArtifact("fw", 3)
        for entry in (
            _ReloadBundledEntry(
                _ReloadLoadable(_ReloadSplitArtifact([earlier, unsupported]))
            ),
            _ReloadBundledEntry(_ReloadLoadable(fw_leaf), _ReloadLoadable(unsupported)),
        ):
            with pytest.raises(TypeError, match="Unsupported CompiledArtifact type"):
                release_entry_executables(entry)
        assert not earlier.released
        assert not fw_leaf.released

    def test_reload_env_var_registered_for_ray_workers(self):
        """Ray worker propagation copies `TpuPlatform.additional_env_vars`."""
        from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

        assert "VLLM_TPU_RELOAD_COMPILED_EXECUTABLE" in TpuPlatform.additional_env_vars


def _compile_and_reload_on_tpu(root: str) -> None:
    """Child-process body of `test_reload_compiled_executable_on_tpu`."""
    # Neither persistent torch_tpu cache may serve the graph: the cold compile
    # has to be a new CompileAndLoad for the reload to run.
    os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"] = "disabled"
    os.environ["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] = "disabled"
    os.environ.pop("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", None)
    os.environ.pop("TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT", None)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(root, "inductor")
    os.environ["VLLM_TPU_RELOAD_COMPILED_EXECUTABLE"] = "1"
    import torch_tpu  # noqa: F401
    from vllm.config.utils import Range

    from vllm_torchtpu.compilation import shape_variants, torch_tpu_release
    from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

    released = []
    release = torch_tpu_release.release_entry_executables

    def tracked_release(entry):
        released.append(entry)
        release(entry)

    torch_tpu_release.release_entry_executables = tracked_release
    cache_dir = os.path.join(root, "backbone")
    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=cache_dir)
    compiled = []

    def adaptor_backend(graph, example_inputs):
        fn, handle = compiler.compile(
            graph, example_inputs, {}, Range(start=8, end=8), key="bucket_8"
        )
        assert fn is not None and handle is not None
        compiled.append((graph, example_inputs, handle))
        return fn

    @torch.compile(backend=adaptor_backend, fullgraph=True, dynamic=False)
    def step(x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 + 1.0

    x = torch.arange(8, dtype=torch.float32, device="tpu")
    expected = torch.arange(8, dtype=torch.float32) * 2.0 + 1.0
    assert torch.equal(step(x).cpu(), expected)
    assert len(compiled) == 1 and len(released) == 1
    result = released[0].compiled_fw.result
    leaves = (
        list(result.compiled_executables)
        if isinstance(result, _SplitCompiledExecutable)
        else [result]
    )
    assert leaves and all(leaf._executable is None for leaf in leaves)
    assert not os.path.exists(os.path.join(root, "inductor", "aotautograd"))

    # A warm start replays the saved artifact without compiling again.
    shape_variants._session = None
    graph, example_inputs, handle = compiled[0]
    before = torch.tpu._get_cache_stats().num_compilations
    warm = TpuCompilerAdaptor()
    warm.initialize_cache(cache_dir=cache_dir)
    out = warm.load(handle, graph, example_inputs, 0, Range(start=8, end=8))(x)
    out = out[0] if isinstance(out, (tuple, list)) else out
    assert torch.equal(out.cpu(), expected)
    assert torch.tpu._get_cache_stats().num_compilations == before


@pytest.mark.tpu_test
def test_reload_compiled_executable_on_tpu(tmp_path):
    """Real TpuBackend: a cold compile under VLLM_TPU_RELOAD_COMPILED_EXECUTABLE
    releases and reloads its executable, writes no AOTAutogradCache entry, and
    the saved artifact warm-loads. Spawned, so PJRT never binds pytest itself."""
    from tests.distributed.tpu_test_utils import run_in_isolated_process

    run_in_isolated_process(partial(_compile_and_reload_on_tpu, str(tmp_path)))
