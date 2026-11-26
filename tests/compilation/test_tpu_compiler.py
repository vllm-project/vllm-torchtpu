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

import contextlib
import importlib
import importlib.util
import os
import re
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.compilation.test_graph_fingerprint import (
    _trace_all_reduce_graph,
    install_group_stand_ins,
)
from vllm_torchtpu.env_override import _TPU_VMEM_BYTES_PER_TENSOR_CORE


@pytest.fixture
def group_stand_ins(monkeypatch):
    return install_group_stand_ins(monkeypatch)


def _install_fake_backend(monkeypatch):
    """Swap `_tpu_backend` and `_deserialize_entry` as a matched pair."""
    import threading

    from vllm_torchtpu.compilation import tpu_compiler

    compiled = []
    replayed = []
    state = {"serializable": True}
    local_tag = threading.local()

    class Compiled:
        def __init__(self, tag):
            self.tag = tag

        def __call__(self, *args):
            return (self.tag,)

        def serialize(self):
            return f"entry-from-rank-{self.tag}"

    def set_tag(tag):
        local_tag.value = tag

    def backend(graph, example_inputs):
        tag = getattr(local_tag, "value", 0)
        compiled.append(tag)
        if not state["serializable"]:
            return lambda *args: (tag,)
        return Compiled(tag)

    def deserialize(entry):
        replayed.append(entry)
        return lambda *args: (entry,)

    monkeypatch.setattr(tpu_compiler, "_tpu_backend", backend)
    monkeypatch.setattr(tpu_compiler, "_deserialize_entry", deserialize)
    return SimpleNamespace(
        compiled=compiled, replayed=replayed, state=state, set_tag=set_tag
    )


@pytest.fixture
def fake_backend(monkeypatch):
    return _install_fake_backend(monkeypatch)


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
            "vllm_torchtpu/kv_cache_spec_normalizer.py",
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

    @staticmethod
    @contextlib.contextmanager
    def _capture_vllm_info():
        import logging

        import vllm.logger  # noqa: F401

        messages: list[str] = []
        handler = logging.Handler(logging.INFO)
        handler.emit = lambda record: messages.append(record.getMessage())
        vllm_logger = logging.getLogger("vllm")
        old_level = vllm_logger.level
        old_disable = logging.root.manager.disable
        vllm_logger.addHandler(handler)
        if vllm_logger.level == logging.NOTSET or vllm_logger.level > logging.INFO:
            vllm_logger.setLevel(logging.INFO)
        logging.disable(logging.NOTSET)
        try:
            yield messages
        finally:
            vllm_logger.removeHandler(handler)
            vllm_logger.setLevel(old_level)
            logging.disable(old_disable)

    @pytest.mark.parametrize(
        ("init_cache", "disable_cache", "key"),
        [
            (False, False, "bucket_1"),
            (True, False, None),
            (True, True, "bucket_1"),
        ],
    )
    def test_compile_without_cache_skips_saving(
        self, tmp_path, fake_backend, init_cache, disable_cache, key
    ):
        """`compile` returns a runnable and `handle=None` without writing to disk
        before `initialize_cache`, when `key=None`, and when `disable_cache=True`."""
        import torch
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        fake_backend.set_tag("compiled")
        graph = torch.fx.symbolic_trace(lambda x: x + 1)
        cache_dir = tmp_path / "cache"

        c = TpuCompilerAdaptor()
        if init_cache:
            c.initialize_cache(cache_dir=str(cache_dir), disable_cache=disable_cache)
        fn, handle = c.compile(graph, [], {}, Range(start=1, end=1), key=key)

        assert handle is None
        assert fn() == "compiled"
        assert not cache_dir.exists()
        assert fake_backend.compiled == ["compiled"]

    def test_compile_logs_and_skips_saving_when_unserializable(
        self, tmp_path, fake_backend
    ):
        """When `compile` has an active cache and the backend output has no
        `serialize`, `handle` is `None`, no artifact is written, and the log
        names the artifact path."""
        import torch
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        fake_backend.set_tag("compiled")
        fake_backend.state["serializable"] = False
        cache_dir = tmp_path / "cache"
        artifact = cache_dir / "bucket_1"
        c = TpuCompilerAdaptor()
        c.initialize_cache(cache_dir=str(cache_dir))
        graph = torch.fx.symbolic_trace(lambda x: x + 1)

        with self._capture_vllm_info() as messages:
            fn, handle = c.compile(graph, [], {}, Range(start=1, end=1), key="bucket_1")

        assert handle is None
        assert fn() == "compiled"
        assert not artifact.exists()
        assert any(
            f"[TpuCompilerAdaptor] No serializable artifact for {artifact}; it "
            "will be recompiled on the next start" in m
            for m in messages
        )

    def test_load_self_heal_overwrites_stale_artifact(self, tmp_path, fake_backend):
        """When `load` encounters an unreadable artifact, it recompiles,
        overwrites the file so the next `load` replays from disk, and runs the
        artifact it just saved rather than the in-process result."""
        import pickle

        import torch
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation.tpu_compiler import (
            TpuCompilationHandle,
            TpuCompilerAdaptor,
        )

        fake_backend.set_tag("compiled")
        c = TpuCompilerAdaptor()
        c.initialize_cache(cache_dir=str(tmp_path / "cache"))
        stale = tmp_path / "cache" / "stale_bucket"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"corrupt")
        graph = torch.fx.symbolic_trace(lambda x: x * 2 + 1)

        healed = c.load(
            ("stale_bucket", str(stale), True), graph, [], 0, Range(start=2, end=2)
        )
        assert fake_backend.compiled == ["compiled"]
        assert fake_backend.replayed == ["entry-from-rank-compiled"]
        assert healed() == "entry-from-rank-compiled"
        with open(stale, "rb") as f:
            saved = pickle.load(f)
        assert isinstance(saved, TpuCompilationHandle)
        assert saved.entry == "entry-from-rank-compiled"

        warm = c.load(
            ("stale_bucket", str(stale), True), graph, [], 0, Range(start=2, end=2)
        )
        assert fake_backend.compiled == ["compiled"]
        assert fake_backend.replayed == ["entry-from-rank-compiled"] * 2
        assert warm() == "entry-from-rank-compiled"

    def test_load_with_disabled_cache_keeps_stale_file(self, tmp_path, fake_backend):
        """When `disable_cache=True`, `load` recompiles an unreadable artifact
        without overwriting the stale file."""
        import torch
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        fake_backend.set_tag("compiled")
        c = TpuCompilerAdaptor()
        c.initialize_cache(cache_dir=str(tmp_path / "cache"), disable_cache=True)
        stale = tmp_path / "cache" / "stale_bucket"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"corrupt")
        graph = torch.fx.symbolic_trace(lambda x: x * 2 + 1)

        fn = c.load(
            ("stale_bucket", str(stale), True), graph, [], 0, Range(start=2, end=2)
        )

        assert fake_backend.compiled == ["compiled"]
        assert fake_backend.replayed == []
        assert fn() == "compiled"
        assert stale.read_bytes() == b"corrupt"

    def test_load_self_heal_logs_and_keeps_stale_file_when_unserializable(
        self, tmp_path, fake_backend
    ):
        """When `load` recompiles a stale artifact whose backend output has no
        `serialize`, it leaves the stale file untouched and logs the path."""
        import torch
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        fake_backend.set_tag("compiled")
        fake_backend.state["serializable"] = False
        c = TpuCompilerAdaptor()
        c.initialize_cache(cache_dir=str(tmp_path / "cache"))
        stale = tmp_path / "cache" / "stale_bucket"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"corrupt")
        graph = torch.fx.symbolic_trace(lambda x: x * 2 + 1)

        with self._capture_vllm_info() as messages:
            fn = c.load(
                ("stale_bucket", str(stale), True), graph, [], 0, Range(start=2, end=2)
            )

        assert fn() == "compiled"
        assert stale.read_bytes() == b"corrupt"
        assert any(
            f"[TpuCompilerAdaptor] No serializable artifact for {stale}; it "
            "will be recompiled on the next start" in m
            for m in messages
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

    @staticmethod
    def _first_ids_per_compile(monkeypatch, *reservers, compiles=2, count=3):
        """Compile `compiles` programs with a stub backend; in each, every
        reserver asks for `count` scheduling group ids. Returns, per program,
        the first id each reserver got."""
        from vllm_torchtpu.compilation import tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        firsts = []

        def fake_backend(graph, example_inputs):
            firsts.append(tuple(reserve(count) for reserve in reservers))
            return lambda *args: args

        monkeypatch.setattr(tpu_compiler, "_tpu_backend", fake_backend)
        compiler = TpuCompilerAdaptor()
        for _ in range(compiles):
            compiler._run_backend(None, [])
        return firsts

    def test_run_backend_restarts_scheduling_group_ids(self, monkeypatch):
        """A bucket's HLO, and so its cache key, must not depend on what the
        rank compiled before it."""
        from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import config

        # As if this process had already traced other programs.
        config.reserve_scheduling_group_ids(5)
        firsts = self._first_ids_per_compile(
            monkeypatch, config.reserve_scheduling_group_ids
        )
        assert firsts == [(1,), (1,)]

    def test_kernels_share_one_scheduling_group_counter(self, monkeypatch):
        """Two kernels in one program (say the reference kernel and a
        per-model copy) get disjoint ids, and every program starts at 1."""
        from vllm_torchtpu.kernels import scheduling_groups
        from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import config

        scheduling_groups.reserve_scheduling_group_ids(5)
        firsts = self._first_ids_per_compile(
            monkeypatch,
            config.reserve_scheduling_group_ids,
            scheduling_groups.reserve_scheduling_group_ids,
        )
        assert firsts == [(1, 4), (1, 4)]

    def test_streamindex_config_reexports_shared_scheduling_groups(self, monkeypatch):
        """The reference config hands out ids, slots included, from the shared
        module, so a copy importing either gets one fallback slot table."""
        from vllm_torchtpu.kernels import scheduling_groups
        from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import config

        for name in (
            "reserve_scheduling_group_ids",
            "reset_scheduling_group_ids",
            "slot_scheduling_group_ids",
            "layer_scheduling_group_slot",
            "SLOT_ID_BASE",
            "MAX_IDS_PER_SLOT",
            "MAX_SCHEDULING_GROUP_SLOT",
        ):
            assert getattr(config, name) is getattr(scheduling_groups, name), name
        assert not hasattr(config, "_fallback_slots")

        monkeypatch.setattr(scheduling_groups, "_fallback_slots", {})
        first = config.layer_scheduling_group_slot("model.mtp.indexer")
        second = scheduling_groups.layer_scheduling_group_slot("model.norm.indexer")
        assert second == first + 1
        assert scheduling_groups._fallback_slots == {
            "model.mtp.indexer": first,
            "model.norm.indexer": second,
        }

    def test_only_scheduling_groups_defines_id_allocators(self):
        """Every kernel takes scheduling-group ids from `scheduling_groups`.

        A module that defines its own allocator (say a per-model copy of
        `streamindex_topk/config.py`) keeps a counter or slot table the
        compiler never resets, so its ids collide with the shared ones and
        depend on compile order again. Re-exporting the shared names is fine.
        """
        from vllm_torchtpu.compilation import tpu_compiler

        source_root = Path(tpu_compiler.__file__).resolve().parents[2]
        allocator_def = re.compile(
            r"^\s*def\s+(reserve_scheduling_group_ids|reset_scheduling_group_ids"
            r"|slot_scheduling_group_ids|layer_scheduling_group_slot)\b",
            re.MULTILINE,
        )
        definers = {
            path.relative_to(source_root).as_posix(): sorted(
                set(allocator_def.findall(path.read_text()))
            )
            for path in (source_root / "vllm_torchtpu").rglob("*.py")
            if allocator_def.search(path.read_text())
        }
        assert definers == {
            "vllm_torchtpu/kernels/scheduling_groups.py": [
                "layer_scheduling_group_slot",
                "reserve_scheduling_group_ids",
                "reset_scheduling_group_ids",
                "slot_scheduling_group_ids",
            ]
        }

    def test_runtime_hash_covers_model_kernel_dirs(self, tmp_path, monkeypatch):
        """Adding or editing a file in any `models/**/kernels/` or
        `models/**/ops/` directory changes the compile cache key; a model
        without one is a no-op. In kernel-iteration mode only the hot-reloaded
        module drops out."""
        from vllm_torchtpu.compilation import tpu_compiler

        repo_root = tmp_path.resolve()
        package = repo_root / "vllm_torchtpu"
        (package / "compilation").mkdir(parents=True)
        (package / "kernels").mkdir()
        (package / "kernels" / "reference.py").write_text("REF = 1\n")
        (package / "models" / "plain_model").mkdir(parents=True)
        (package / "models" / "plain_model" / "model.py").write_text("M = 1\n")
        # A plain file named like a discovered directory is not a package.
        (package / "models" / "plain_model" / "ops").write_text("not python\n")
        # compute_tpu_compilation_hash finds the source root from __file__.
        monkeypatch.setattr(
            tpu_compiler,
            "__file__",
            str(package / "compilation" / "tpu_compiler.py"),
        )

        def hashed_relpaths():
            return [
                path.relative_to(repo_root).as_posix()
                for path in tpu_compiler._iter_runtime_cache_key_files(repo_root)
            ]

        def cache_key():
            return tpu_compiler.compute_tpu_compilation_hash(self._config())

        keys = [cache_key()]
        assert not [p for p in hashed_relpaths() if "/models/" in p]

        nested = package / "models" / "vllm" / "fake_model" / "kernels" / "topk"
        nested.mkdir(parents=True)
        (nested / "config.py").write_text("TOP_K = 512\n")
        keys.append(cache_key())

        (nested / "config.py").write_text("TOP_K = 256\n")
        keys.append(cache_key())

        flat = package / "models" / "flat_model" / "kernels"
        flat.mkdir(parents=True)
        (flat / "kernel.py").write_text("K = 1\n")
        keys.append(cache_key())

        ops = package / "models" / "vllm" / "fake_model" / "ops"
        ops.mkdir(parents=True)
        (ops / "topk_op.py").write_text("OP = 1\n")
        keys.append(cache_key())

        assert len(set(keys)) == len(keys)
        relpaths = hashed_relpaths()
        assert relpaths == sorted(relpaths)
        model_files = {
            "vllm_torchtpu/models/flat_model/kernels/kernel.py",
            "vllm_torchtpu/models/vllm/fake_model/kernels/topk/config.py",
            "vllm_torchtpu/models/vllm/fake_model/ops/topk_op.py",
        }
        assert model_files <= set(relpaths)
        assert "vllm_torchtpu/models/plain_model/model.py" not in relpaths
        assert "vllm_torchtpu/models/plain_model/ops" not in relpaths

        # Kernel-iteration mode: the module listed for hot reload leaves the
        # key, every other discovered file stays.
        reloaded = "vllm_torchtpu.models.vllm.fake_model.kernels.topk.config"
        monkeypatch.setenv("TPU_KERNEL_ITER_MODE", "1")
        monkeypatch.setenv("TPU_KERNEL_RELOAD_MODULES", reloaded)
        real_find_spec = importlib.util.find_spec

        def find_spec(name, *args, **kwargs):
            if name != reloaded:
                return real_find_spec(name, *args, **kwargs)
            return SimpleNamespace(
                origin=str(repo_root / (name.replace(".", "/") + ".py"))
            )

        monkeypatch.setattr(importlib.util, "find_spec", find_spec)
        iter_relpaths = hashed_relpaths()
        assert set(relpaths) - set(iter_relpaths) == {
            "vllm_torchtpu/models/vllm/fake_model/kernels/topk/config.py"
        }
        assert model_files - set(iter_relpaths) == {
            "vllm_torchtpu/models/vllm/fake_model/kernels/topk/config.py"
        }

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
        fb = _install_fake_backend(monkeypatch)

        def compilers():
            for rank in range(n):
                compiler = TpuCompilerAdaptor()
                compiler.initialize_cache(
                    cache_dir=str(tmp_path / "hash" / f"rank_{rank}_0" / "backbone")
                )
                compiler.rank = rank
                yield compiler

        monkeypatch.setattr(tpu_compiler, "_PEER_POLL_S", 0.01)

        def compile(compiler, graph=None):
            fb.set_tag(compiler.rank)
            if graph is None:
                graph = torch.fx.symbolic_trace(lambda x: x * 2 + 1)
            return compiler.compile(
                graph, [], {}, Range(start=1, end=4096), key="artifact_bucket"
            )

        return list(compilers()), compile, fb.compiled, fb.replayed

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
        # Each compiling rank also reloads the artifact it saved.
        assert replayed == ["entry-from-rank-0", "entry-from-rank-0"]
        assert fn() == "entry-from-rank-0"
        key, path, _ = handle
        assert path == os.path.join(b.cache_dir, key)
        assert os.path.samefile(path, os.path.join(a.cache_dir, key))
        assert sorted(os.listdir(tmp_path / "hash")) == ["rank_0_0", "rank_1_0"]
        assert os.listdir(b.cache_dir) == [key]

    def test_a_compiled_bucket_runs_from_its_saved_artifact(
        self, tmp_path, monkeypatch
    ):
        """The compiling rank runs the runnable rebuilt from the artifact it
        saved, as replaying ranks and warm starts do, and drops the in-process
        compile result, whose device tensors would otherwise count against
        the KV budget."""
        from vllm_torchtpu.compilation import tpu_compiler

        (a,), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch, n=1, peer_replay="0"
        )
        built = []
        real_backend = tpu_compiler._tpu_backend

        def backend(graph, example_inputs):
            result = real_backend(graph, example_inputs)
            built.append(weakref.ref(result))
            return result

        monkeypatch.setattr(tpu_compiler, "_tpu_backend", backend)
        fn, handle = compile(a)

        assert compiled == [0]
        assert replayed == ["entry-from-rank-0"]
        assert fn() == "entry-from-rank-0"
        assert handle is not None
        assert built[0]() is None

    def test_a_stale_artifact_recompiled_in_load_runs_from_the_new_one(
        self, tmp_path, monkeypatch
    ):
        """`load`'s self-heal path also runs the artifact it re-saved, not the
        in-process result, so a warm start that recompiles a stale artifact
        holds no extra HBM either."""
        import torch
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation import tpu_compiler

        (a,), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch, n=1, peer_replay="0"
        )
        _, handle = compile(a)
        with open(handle[1], "wb") as f:
            f.write(b"stale")

        built = []
        real_backend = tpu_compiler._tpu_backend

        def backend(graph, example_inputs):
            result = real_backend(graph, example_inputs)
            built.append(weakref.ref(result))
            return result

        monkeypatch.setattr(tpu_compiler, "_tpu_backend", backend)
        fn = a.load(
            handle,
            torch.fx.symbolic_trace(lambda x: x * 2 + 1),
            [],
            0,
            Range(start=1, end=4096),
        )

        assert compiled == [0, 0]
        assert fn() == "entry-from-rank-0"
        assert replayed[-1] == "entry-from-rank-0"
        assert built[0]() is None

    def test_the_in_process_result_is_kept_when_the_reload_fails(
        self, tmp_path, monkeypatch
    ):
        from vllm_torchtpu.compilation import tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        (a,), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch, n=1, peer_replay="0"
        )

        def cannot_rebuild(entry):
            raise TypeError("unsupported entry")

        monkeypatch.setattr(
            tpu_compiler, "_rebuild_sharing_executables", cannot_rebuild
        )
        monkeypatch.setattr(TpuCompilerAdaptor, "_replay", lambda *args, **kw: None)
        fn, handle = compile(a)

        assert compiled == [0]
        assert fn() == 0
        assert handle is not None

    def test_a_compiled_bucket_reuses_its_executable_instead_of_reloading(
        self, tmp_path, monkeypatch
    ):
        """The runnable is rebuilt from the in-process entry, so the saved
        artifact is not read back: torch_tpu's Tier-1 cache keeps the
        executable the backend returned, and loading the artifact would add a
        second copy of it."""
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        (a,), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch, n=1, peer_replay="0"
        )
        read_back = []
        real_replay = TpuCompilerAdaptor._replay

        def replay(self, path, *args, **kwargs):
            read_back.append(path)
            return real_replay(self, path, *args, **kwargs)

        monkeypatch.setattr(TpuCompilerAdaptor, "_replay", replay)
        fn, handle = compile(a)

        assert compiled == [0]
        assert read_back == []
        assert replayed == ["entry-from-rank-0"]
        assert fn() == "entry-from-rank-0"
        assert handle is not None

    def test_a_bucket_that_cannot_be_rebuilt_replays_its_artifact(
        self, tmp_path, monkeypatch
    ):
        from vllm_torchtpu.compilation import tpu_compiler

        (a,), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch, n=1, peer_replay="0"
        )

        def cannot_rebuild(entry):
            raise TypeError("unsupported entry")

        monkeypatch.setattr(
            tpu_compiler, "_rebuild_sharing_executables", cannot_rebuild
        )
        fn, handle = compile(a)

        assert compiled == [0]
        assert replayed == ["entry-from-rank-0"]
        assert fn() == "entry-from-rank-0"
        assert handle is not None

    def test_nothing_is_reloaded_without_a_saved_artifact(self, tmp_path, monkeypatch):
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        (a,), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch, n=1, peer_replay="0"
        )
        monkeypatch.setattr(TpuCompilerAdaptor, "_save_handle", lambda *a, **k: False)
        fn, handle = compile(a)

        assert compiled == [0]
        assert replayed == []
        assert fn() == 0
        assert handle is None

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
        # Each compiling rank also reloads the artifact it saved.
        assert replayed == ["entry-from-rank-0", "entry-from-rank-1"]
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
        # Each compiling rank also reloads the artifact it saved.
        assert replayed == ["entry-from-rank-1"]

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
        # Each compiling rank also reloads the artifact it saved.
        assert replayed == ["entry-from-rank-0", "entry-from-rank-1"]
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
        # Each compiling rank also reloads the artifact it saved.
        assert replayed == ["entry-from-rank-0", "entry-from-rank-0"]
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


def test_rebuild_sharing_executables_gives_new_leaves_on_the_same_executable(
    monkeypatch,
):
    """Every object of the entry is new except each leaf's loaded executable,
    which is neither serialized nor loaded again."""
    from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
        TorchTpuCompiledExecutable,
    )

    from vllm_torchtpu.compilation import tpu_compiler

    class Executable:
        def __reduce__(self):
            raise AssertionError("the executable must not be serialized")

    executable = Executable()
    leaf = TorchTpuCompiledExecutable(
        executable=executable,
        reconstruct_fx_outputs_fn=None,
        updates_default_generator_state=True,
        dynamic_outputs=[False],
        unique_output_indices=[0],
    )
    other = TorchTpuCompiledExecutable(
        executable=Executable(),
        reconstruct_fx_outputs_fn=None,
        updates_default_generator_state=False,
    )
    entry = SimpleNamespace(fw=leaf, same_fw=leaf, bw=other, meta={"k": [1, 2]})
    monkeypatch.setattr(tpu_compiler, "_deserialize_entry", lambda e: e)

    rebuilt = tpu_compiler._rebuild_sharing_executables(entry)

    assert rebuilt is not entry
    assert rebuilt.meta == entry.meta and rebuilt.meta is not entry.meta
    assert rebuilt.fw is not leaf and rebuilt.fw is rebuilt.same_fw
    assert rebuilt.fw._executable is executable
    assert rebuilt.fw._updates_default_generator_state is True
    assert rebuilt.fw._dynamic_outputs == [False]
    assert rebuilt.fw._unique_output_indices == [0]
    assert rebuilt.bw is not other and rebuilt.bw._executable is other._executable
    assert rebuilt.bw._updates_default_generator_state is False


def test_rebuild_sharing_executables_matches_pickle_leaf_attributes(monkeypatch):
    """A rebuilt `TorchTpuCompiledExecutable` carries the exact same attribute
    set and values as `pickle.loads(pickle.dumps(leaf))`, except `_executable`
    is shared instead of reloaded from serialized bytes."""
    import pickle

    from torch_tpu._internal.compile import tpu_torch_compile
    from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
        TorchTpuCompiledExecutable,
    )

    from vllm_torchtpu.compilation import tpu_compiler

    monkeypatch.setattr(
        tpu_torch_compile,
        "serialize_executable",
        lambda ex: ex.serialize_executable(),
    )
    monkeypatch.setattr(
        tpu_torch_compile,
        "load_serialized_executable",
        lambda serialized, subslice_origin=None: _FakePjrtExecutable("reloaded"),
    )
    monkeypatch.setattr(tpu_compiler, "_deserialize_entry", lambda e: e)

    leaf = TorchTpuCompiledExecutable(
        executable=_FakePjrtExecutable("original"),
        reconstruct_fx_outputs_fn=_passthrough_fx_outputs,
        updates_default_generator_state=True,
        dynamic_outputs=[False, True],
        unique_output_indices=[0, 1],
    )
    unpickled = pickle.loads(pickle.dumps(leaf))
    rebuilt = tpu_compiler._rebuild_sharing_executables(leaf)

    assert rebuilt is not leaf
    assert rebuilt._executable is leaf._executable
    assert unpickled._executable is not leaf._executable
    assert set(rebuilt.__dict__) == set(unpickled.__dict__)
    for name in rebuilt.__dict__:
        if name == "_executable":
            continue
        actual = getattr(rebuilt, name)
        expected = getattr(unpickled, name)
        if name == "_cached_dynamic_output_shapes":
            assert [(s.dimensions, s.is_dynamic) for s in actual] == [
                (s.dimensions, s.is_dynamic) for s in expected
            ]
        else:
            assert actual == expected, name


def _rebuild_subclass_marker(tag: str) -> tuple[str, str]:
    return ("subclass-reduced", tag)


def test_rebuild_sharing_executables_keeps_subclass_reduce(monkeypatch):
    """Subclasses of `TorchTpuCompiledExecutable` bypass `persistent_id` and
    keep their own `__reduce__`, while exact instances still share their
    loaded executable."""
    from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
        TorchTpuCompiledExecutable,
    )

    from vllm_torchtpu.compilation import tpu_compiler

    class Executable:
        def __reduce__(self):
            raise AssertionError("exact leaf executable must not be serialized")

    reduced = []

    class SubLeaf(TorchTpuCompiledExecutable):
        def __init__(self, tag: str):
            super().__init__(
                executable=Executable(),
                reconstruct_fx_outputs_fn=None,
                updates_default_generator_state=False,
            )
            self.tag = tag

        def __reduce__(self):
            reduced.append(self.tag)
            return (_rebuild_subclass_marker, (self.tag,))

    executable = Executable()
    plain = TorchTpuCompiledExecutable(
        executable=executable,
        reconstruct_fx_outputs_fn=None,
        updates_default_generator_state=False,
    )
    entry = SimpleNamespace(plain=plain, sub=SubLeaf("custom"))
    monkeypatch.setattr(tpu_compiler, "_deserialize_entry", lambda e: e)

    rebuilt = tpu_compiler._rebuild_sharing_executables(entry)

    assert rebuilt.plain is not plain
    assert rebuilt.plain._executable is executable
    assert reduced == ["custom"]
    assert rebuilt.sub == ("subclass-reduced", "custom")


def _rebuild_on_loaded_executable_on_tpu(root: str) -> None:
    """Child-process body of `test_compiled_bucket_reuses_its_executable_on_tpu`."""
    import torch
    from torch_tpu._internal.compile.split_compiler import _SplitCompiledExecutable
    from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
        TorchTpuCompiledExecutable,
    )
    from vllm.config.utils import Range

    from vllm_torchtpu import patch_registry
    from vllm_torchtpu.compilation import tpu_compiler
    from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor
    from vllm_torchtpu.tpu_info import get_num_chips

    if get_num_chips() == 0:
        pytest.skip("TPU hardware required")
    patch_registry.apply("worker_init")

    def leaves_of(entry):
        result = entry.compiled_fw.result
        subs = (
            result.compiled_executables
            if isinstance(result, _SplitCompiledExecutable)
            else [result]
        )
        return [sub for sub in subs if isinstance(sub, TorchTpuCompiledExecutable)]

    built = []
    dropped_refs = []
    real_backend = tpu_compiler._tpu_backend

    def tracked_backend(graph, example_inputs):
        compiled_fn = real_backend(graph, example_inputs)
        entry = compiled_fn.serialize()
        leaves = leaves_of(entry)
        built.append([(leaf, leaf._executable) for leaf in leaves])
        dropped_refs.append((weakref.ref(compiled_fn), weakref.ref(entry)))
        return compiled_fn

    rebuilt = []
    real_deserialize = tpu_compiler._deserialize_entry

    def tracked_deserialize(entry):
        rebuilt.append([(leaf, leaf._executable) for leaf in leaves_of(entry)])
        return real_deserialize(entry)

    read_back = []
    real_replay = TpuCompilerAdaptor._replay

    def tracked_replay(self, path, *args, **kwargs):
        read_back.append(path)
        return real_replay(self, path, *args, **kwargs)

    tpu_compiler._tpu_backend = tracked_backend
    tpu_compiler._deserialize_entry = tracked_deserialize
    TpuCompilerAdaptor._replay = tracked_replay

    cache_dir = os.path.join(root, "cache")
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

    compiled_step = torch.compile(
        lambda x: x * 2.0 + 1.0, backend=adaptor_backend, fullgraph=True, dynamic=False
    )
    x = torch.arange(8, dtype=torch.float32, device="tpu")
    expected = torch.arange(8, dtype=torch.float32) * 2.0 + 1.0
    assert torch.equal(compiled_step(x).cpu(), expected)

    # The cold compile ran the runnable rebuilt from the in-process entry: new
    # leaves on the very executables the backend returned, no artifact read.
    assert read_back == []
    assert len(built) == 1 and len(rebuilt) == 1
    assert len(built[0]) >= 1 and len(rebuilt[0]) == len(built[0])
    for (old_leaf, old_exe), (new_leaf, new_exe) in zip(built[0], rebuilt[0]):
        assert new_leaf is not old_leaf
        assert new_exe is old_exe
    # The dropped in-process backend runnable and entry were reclaimed.
    assert len(dropped_refs) == 1
    assert dropped_refs[0][0]() is None
    assert dropped_refs[0][1]() is None

    # A warm start still replays the artifact the cold start saved.
    graph, example_inputs, handle = compiled[0]
    before = torch.tpu._get_cache_stats().num_compilations
    warm = TpuCompilerAdaptor()
    warm.initialize_cache(cache_dir=cache_dir)
    (out,) = warm.load(handle, graph, example_inputs, 0, Range(start=8, end=8))(x)
    assert torch.equal(out.cpu(), expected)
    assert torch.tpu._get_cache_stats().num_compilations == before
    assert read_back == [handle[1]]


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
def test_compiled_bucket_reuses_its_executable_on_tpu(tmp_path, monkeypatch):
    """Real TpuBackend: a cold compile runs a runnable rebuilt on the loaded
    executable the backend returned (no second load of the saved artifact),
    computes the right result, and the artifact still warm-loads."""
    from functools import partial

    from tests.distributed.tpu_test_utils import run_in_isolated_process

    monkeypatch.setenv("TORCH_TPU_TIER2_COMPILATION_CACHE", "disabled")
    monkeypatch.delenv("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", raising=False)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setenv("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")

    run_in_isolated_process(
        partial(_rebuild_on_loaded_executable_on_tpu, str(tmp_path)), timeout=180
    )


class _FakePjrtExecutable:
    """A loaded executable, as far as `TorchTpuCompiledExecutable` reads one."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def serialize_executable(self) -> bytes:
        return f"pjrt:{self.tag}".encode()

    def subslice_origin(self) -> None:
        return None

    def fingerprint_executable(self) -> str:
        return f"fp:{self.tag}"


def _passthrough_fx_outputs(orig_args, tpu_outputs):
    return tpu_outputs


def _fake_execute(executable, args, *rest):
    """Runs `_double` for whichever leaf calls it, boxed inputs or not."""
    import torch
    from torch.utils._pytree import tree_leaves

    (x,) = [a for a in tree_leaves(args) if isinstance(a, torch.Tensor)]
    return (x * 2,)


def _fake_leaf(tag: str = "leaf"):
    from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
        TorchTpuCompiledExecutable,
    )

    return TorchTpuCompiledExecutable(
        executable=_FakePjrtExecutable(tag),
        reconstruct_fx_outputs_fn=_passthrough_fx_outputs,
        updates_default_generator_state=False,
    )


class _FakeBaseCompiler:
    """torch_tpu's per-graph compiler minus XLA: each graph it is handed
    becomes a leaf on a fake loaded executable."""

    compilation_context = None
    _debug = None

    def execute_pre_grad_passes(self, graph_module) -> None:
        pass

    def __call__(self, graph_module, example_inputs, is_fwd=True, **kwargs):
        return _fake_leaf(str(kwargs.get("module_name")))


class _UnknownArtifact:
    """An artifact `_release_compiled_graph_cycles` does not know."""


def _double(x):
    return x * 2


def _leaves_of(result):
    from torch_tpu._internal.compile.split_compiler import _SplitCompiledExecutable

    if isinstance(result, _SplitCompiledExecutable):
        return list(result.compiled_executables)
    return [result]


def _artifacts_cell(fn):
    """The backend runnable's `compiled_artifacts` closure list."""
    cells = dict(zip(fn.__code__.co_freevars, fn.__closure__))
    return cells["compiled_artifacts"].cell_contents


@pytest.fixture(params=[True, False], ids=["split", "single-leaf"])
def cpu_tpu_backend(request, monkeypatch):
    """The real TpuBackend and aot_autograd, compiling each graph -- whole, or
    split by torch_tpu's SplitCompiler -- to leaves on fake executables, so
    `compile` and `load` handle the very objects they do on a TPU host.

    Records what each backend call built, and the executables of the leaves
    each runnable is deserialized on. `loads` lists every executable loaded
    back from serialized bytes, the second copy a replay costs."""
    import gc

    import torch
    from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
    from torch_tpu._internal.compile import _backend, tpu_torch_compile
    from torch_tpu._internal.compile.split_compiler import SplitCompiler

    from vllm_torchtpu.compilation import tpu_compiler

    split = request.param
    state = SimpleNamespace(
        split=split, loads=[], built=[], deserialized=[], extra_artifacts=[]
    )

    def load_serialized_executable(serialized, subslice_origin=None):
        state.loads.append(serialized)
        return _FakePjrtExecutable("loaded")

    monkeypatch.setattr(AOTAutogradCache, "save", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(
        tpu_torch_compile, "serialize_executable", lambda ex: ex.serialize_executable()
    )
    monkeypatch.setattr(
        tpu_torch_compile, "load_serialized_executable", load_serialized_executable
    )
    monkeypatch.setattr(tpu_torch_compile, "execute", _fake_execute)
    monkeypatch.setattr(
        tpu_torch_compile, "get_handshake_stage_env_var_once", lambda: 0
    )
    monkeypatch.setattr(
        _backend,
        "make_backend_compiler",
        lambda *a, **kw: (
            SplitCompiler(_FakeBaseCompiler()) if split else _FakeBaseCompiler()
        ),
    )
    real_backend = _backend.TpuBackend()

    def backend(graph, example_inputs):
        fn = real_backend(graph, example_inputs)
        # Sweep dead Gen-1 closures left by PyTorch's CodeGen._gen_python_code
        # when automatic GC is active (promoted_to_gen2 already runs collect(1)).
        if gc.isenabled():
            gc.collect(1)
        entry = fn.serialize()
        result = entry.compiled_fw.result
        leaves = _leaves_of(result)
        tracked = {
            "fn": weakref.ref(fn),
            "entry": weakref.ref(entry),
            "result": weakref.ref(result),
        }
        tracked.update({f"leaf{i}": weakref.ref(leaf) for i, leaf in enumerate(leaves)})
        if split:
            # A constant the split graph holds, as sunk weights would be, plus
            # unshared node metadata to verify erased nodes and the fx.Graph
            # itself are freed by refcounting rather than stranded in Gen-2.
            constant = torch.zeros(16)
            result._split_gm.sunk_constant = constant
            split_graph = result._split_gm.graph
            tracked["split_gm"] = weakref.ref(result._split_gm)
            tracked["split_graph"] = weakref.ref(split_graph)
            tracked["constant"] = weakref.ref(constant)
            for idx, node in enumerate(split_graph.nodes):
                meta_tensor = torch.zeros(4)
                node.meta["probe_tensor"] = meta_tensor
                tracked[f"node_{idx}"] = weakref.ref(node)
                tracked[f"node_meta_{idx}"] = weakref.ref(meta_tensor)
        state.built.append(
            SimpleNamespace(
                tracked=tracked,
                executables=[leaf._executable for leaf in leaves],
            )
        )
        _artifacts_cell(fn).extend(state.extra_artifacts)
        return fn

    real_deserialize = tpu_compiler._deserialize_entry

    def deserialize(entry):
        leaves = _leaves_of(entry.compiled_fw.result)
        state.deserialized.append([leaf._executable for leaf in leaves])
        return real_deserialize(entry)

    monkeypatch.setattr(tpu_compiler, "_tpu_backend", backend)
    monkeypatch.setattr(tpu_compiler, "_deserialize_entry", deserialize)
    return state


@pytest.fixture
def promoted_to_gen2(monkeypatch):
    """Automatic GC off; once `_save_handle` returns, a Gen-1 pass moves
    everything the in-process compile result holds to Gen-2, where a Gen-0
    sweep cannot reach it. Yields the generations `gc.collect` is asked for."""
    import gc

    from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

    real_collect = gc.collect
    real_save = TpuCompilerAdaptor._save_handle

    def promoting_save(self, *args, **kwargs):
        saved = real_save(self, *args, **kwargs)
        real_collect(1)
        return saved

    generations = []

    def collect(generation=2):
        generations.append(generation)
        return real_collect(generation)

    monkeypatch.setattr(TpuCompilerAdaptor, "_save_handle", promoting_save)
    monkeypatch.setattr(gc, "collect", collect)
    real_collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield generations
    finally:
        if was_enabled:
            gc.enable()


class TestInProcessResultReclaim:
    """A compiled bucket runs a runnable rebuilt on the executables the backend
    returned, and the in-process result goes without a full collection."""

    @staticmethod
    def _graph():
        import torch

        return torch.fx.symbolic_trace(_double)

    @staticmethod
    def _compile(tmp_path, graph):
        import torch
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        compiler = TpuCompilerAdaptor()
        compiler.initialize_cache(cache_dir=str(tmp_path / "cache"))
        return compiler.compile(
            graph, [torch.tensor([3, 4])], {}, Range(start=2, end=2), key="bucket"
        )

    @staticmethod
    def _load_stale(tmp_path, graph):
        import torch
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        compiler = TpuCompilerAdaptor()
        compiler.initialize_cache(cache_dir=str(tmp_path / "cache"))
        stale = tmp_path / "cache" / "bucket"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"stale")
        fn = compiler.load(
            ("bucket", str(stale), True),
            graph,
            [torch.tensor([3, 4])],
            0,
            Range(start=2, end=2),
        )
        assert stale.read_bytes() != b"stale"
        return fn

    @staticmethod
    def _assert_reclaimed(backend):
        (built,) = backend.built
        alive = {name for name, ref in built.tracked.items() if ref() is not None}
        assert alive == set()

    @staticmethod
    def _assert_runs(fn):
        import torch

        assert torch.equal(fn(torch.tensor([3, 4])), torch.tensor([6, 8]))

    @pytest.mark.parametrize("path", ["compile", "load"])
    def test_runs_on_the_same_executables_and_reclaims_with_gen0_only(
        self, tmp_path, cpu_tpu_backend, promoted_to_gen2, path
    ):
        """Even with the in-process result promoted to Gen-2 during the save,
        a Gen-0 sweep reclaims it: its wrapper, entry, leaves and (split)
        graph and constants. The runnable that replaces it is new, but runs
        on the executables the backend returned; none is loaded again."""
        graph = self._graph()
        if path == "compile":
            fn, handle = self._compile(tmp_path, graph)
            assert handle is not None
        else:
            fn = self._load_stale(tmp_path, graph)

        assert promoted_to_gen2 == [0]
        assert cpu_tpu_backend.loads == []
        (built,) = cpu_tpu_backend.built
        (deserialized,) = cpu_tpu_backend.deserialized
        assert len(deserialized) == len(built.executables) >= 1
        assert all(new is old for new, old in zip(deserialized, built.executables))
        self._assert_reclaimed(cpu_tpu_backend)
        self._assert_runs(fn)

    @pytest.mark.parametrize("cpu_tpu_backend", [True], indirect=True)
    @pytest.mark.parametrize("path", ["compile", "load"])
    def test_gen2_split_graph_survives_gen0_without_cycle_breaking(
        self, tmp_path, monkeypatch, cpu_tpu_backend, promoted_to_gen2, path
    ):
        """Negative control: with `_release_compiled_graph_cycles` stubbed out,
        the Gen-2 promoted `_split_gm`, its `fx.Graph`, and constants survive
        `gc.collect(0)` and are only freed by a full `gc.collect()`."""
        import gc

        from vllm_torchtpu.compilation import tpu_compiler

        monkeypatch.setattr(
            tpu_compiler, "_release_compiled_graph_cycles", lambda *a, **k: True
        )
        graph = self._graph()
        if path == "compile":
            fn, handle = self._compile(tmp_path, graph)
            assert handle is not None
        else:
            fn = self._load_stale(tmp_path, graph)

        assert promoted_to_gen2 == [0]
        (built,) = cpu_tpu_backend.built
        alive = {name for name, ref in built.tracked.items() if ref() is not None}
        assert {"split_gm", "split_graph", "constant"} <= alive
        self._assert_runs(fn)
        gc.collect()
        self._assert_reclaimed(cpu_tpu_backend)

    @pytest.mark.parametrize("path", ["compile", "load"])
    def test_leaves_the_caller_graph_intact(self, tmp_path, cpu_tpu_backend, path):
        """The graph vLLM passed in -- from which `shape_variants` reads the
        placeholder signature -- is not dismantled with the in-process result."""
        from vllm_torchtpu.compilation import shape_variants

        remembered = []
        real_remember = shape_variants.remember

        def remember(cache_dir, key, graph, fn):
            remembered.append(graph)
            return real_remember(cache_dir, key, graph, fn)

        with (
            shape_variants.warmup(),
            patch.object(shape_variants, "remember", remember),
        ):
            graph = self._graph()
            code = graph.code
            nodes = [(n.name, n.op, n.args) for n in graph.graph.nodes]
            metas = [n.meta for n in graph.graph.nodes]
            meta_items = [dict(m) for m in metas]
            if path == "compile":
                fn, _ = self._compile(tmp_path, graph)
            else:
                fn = self._load_stale(tmp_path, graph)
            handed = shape_variants.runnable_for(
                str(tmp_path / "cache"), "bucket", self._graph(), 2
            )

        assert graph.code == code
        assert [(n.name, n.op, n.args) for n in graph.graph.nodes] == nodes
        assert [n.meta for n in graph.graph.nodes] == meta_items
        assert all(a is b for a, b in zip([n.meta for n in graph.graph.nodes], metas))
        assert graph.graph.owning_module is graph
        (kept,) = remembered
        assert kept is graph
        assert handed is fn
        if cpu_tpu_backend.split:
            # ... while the in-process split graph was dismantled and reclaimed.
            self._assert_reclaimed(cpu_tpu_backend)
        self._assert_runs(fn)

    @pytest.mark.parametrize("path", ["compile", "load"])
    def test_a_changed_leaf_constructor_replays_the_artifact(
        self, tmp_path, monkeypatch, cpu_tpu_backend, path
    ):
        """If torch_tpu's leaf constructor no longer takes the arguments a
        leaf is rebuilt from, the bucket replays its artifact (a second load)
        instead of building leaves that miss one."""
        from vllm_torchtpu.compilation import tpu_compiler

        monkeypatch.setattr(tpu_compiler, "_TORCH_TPU_EXEC_INIT_MATCHES", False)
        graph = self._graph()
        with TestTpuCompilerCache._capture_vllm_info() as messages:
            if path == "compile":
                fn, handle = self._compile(tmp_path, graph)
                assert handle is not None
            else:
                fn = self._load_stale(tmp_path, graph)

        assert any(
            "Could not rebuild the runnable on its loaded executable" in m
            and "TorchTpuCompiledExecutable.__init__ takes" in m
            for m in messages
        )
        (built,) = cpu_tpu_backend.built
        assert len(cpu_tpu_backend.loads) == len(built.executables)
        (deserialized,) = cpu_tpu_backend.deserialized
        assert all(new is not old for new, old in zip(deserialized, built.executables))
        self._assert_reclaimed(cpu_tpu_backend)
        self._assert_runs(fn)

    def test_a_changed_leaf_constructor_is_reported_once(
        self, tmp_path, monkeypatch, cpu_tpu_backend
    ):
        """A changed leaf constructor fails the rebuild of every bucket the
        same way, so it is reported once per process, not once per bucket."""
        import torch
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation import tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        monkeypatch.setattr(tpu_compiler, "_TORCH_TPU_EXEC_INIT_MATCHES", False)
        compiler = TpuCompilerAdaptor()
        compiler.initialize_cache(cache_dir=str(tmp_path / "cache"))
        with TestTpuCompilerCache._capture_vllm_info() as messages:
            for key in ("bucket_a", "bucket_b"):
                fn, handle = compiler.compile(
                    self._graph(),
                    [torch.tensor([3, 4])],
                    {},
                    Range(start=2, end=2),
                    key=key,
                )
                assert handle is not None
                self._assert_runs(fn)

        assert (
            sum("TorchTpuCompiledExecutable.__init__ takes" in m for m in messages) == 1
        )
        assert len(cpu_tpu_backend.built) == 2
        assert len(cpu_tpu_backend.loads) == sum(
            len(built.executables) for built in cpu_tpu_backend.built
        )

    @pytest.mark.parametrize(
        "reason", ["reload_failed", "cache_disabled"], ids=["reload-failed", "no-save"]
    )
    @pytest.mark.parametrize("path", ["compile", "load"])
    def test_keeps_the_in_process_result_intact_when_not_reloaded(
        self, tmp_path, monkeypatch, cpu_tpu_backend, path, reason
    ):
        """When no rebuilt or replayed runnable replaces the in-process compile
        result, its split FX graph, metadata and backend closure stay intact."""
        import torch
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation import tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        graph = self._graph()
        if reason == "reload_failed":

            def cannot_rebuild(entry):
                raise TypeError("unsupported entry")

            monkeypatch.setattr(
                tpu_compiler, "_rebuild_sharing_executables", cannot_rebuild
            )
            monkeypatch.setattr(TpuCompilerAdaptor, "_replay", lambda *a, **kw: None)
            if path == "compile":
                fn, handle = self._compile(tmp_path, graph)
                assert handle is not None
            else:
                fn = self._load_stale(tmp_path, graph)
        else:
            compiler = TpuCompilerAdaptor()
            compiler.initialize_cache(
                cache_dir=str(tmp_path / "cache"), disable_cache=True
            )
            if path == "compile":
                fn, handle = compiler.compile(
                    graph,
                    [torch.tensor([3, 4])],
                    {},
                    Range(start=2, end=2),
                    key="bucket",
                )
                assert handle is None
            else:
                stale = tmp_path / "cache" / "bucket"
                stale.parent.mkdir(parents=True, exist_ok=True)
                stale.write_bytes(b"stale")
                fn = compiler.load(
                    ("bucket", str(stale), True),
                    graph,
                    [torch.tensor([3, 4])],
                    0,
                    Range(start=2, end=2),
                )
                assert stale.read_bytes() == b"stale"

        assert cpu_tpu_backend.loads == []
        assert cpu_tpu_backend.deserialized == []
        (built,) = cpu_tpu_backend.built
        backend_fn = built.tracked["fn"]()
        result = built.tracked["result"]()
        assert backend_fn is not None and result is not None
        assert len(_artifacts_cell(backend_fn)) >= 1
        if cpu_tpu_backend.split:
            split_gm = built.tracked["split_gm"]()
            assert split_gm is not None
            assert len(list(split_gm.graph.nodes)) >= 1
            assert split_gm.graph.owning_module is split_gm
            assert result._graph_module_str is not None
        self._assert_runs(fn)

    @pytest.mark.parametrize("path", ["compile", "load"])
    def test_an_unknown_artifact_falls_back_to_a_full_collection(
        self, tmp_path, cpu_tpu_backend, promoted_to_gen2, path
    ):
        """An artifact of a type it does not know may hold cycles of its own:
        the known ones are still dismantled, a warning is logged once naming
        the unexpected type, and a full collection runs."""
        cpu_tpu_backend.extra_artifacts.append(_UnknownArtifact())
        graph = self._graph()
        with TestTpuCompilerCache._capture_vllm_info() as messages:
            if path == "compile":
                fn, _ = self._compile(tmp_path, graph)
            else:
                fn = self._load_stale(tmp_path, graph)

        assert promoted_to_gen2 == [2]
        assert any(
            "Could not dismantle all compiled graph cycles" in m
            and "_UnknownArtifact" in m
            for m in messages
        )
        self._assert_reclaimed(cpu_tpu_backend)
        self._assert_runs(fn)


def test_the_pinned_torch_tpu_leaf_constructor_is_the_one_rebuilt_from():
    from vllm_torchtpu.compilation import tpu_compiler

    assert tpu_compiler._TORCH_TPU_EXEC_INIT_MATCHES


def _tag_first_outputs(orig_args, tpu_outputs):
    return ("first", tpu_outputs)


def _tag_second_outputs(orig_args, tpu_outputs):
    return ("second", tpu_outputs)


def test_rebuild_sharing_executables_keeps_distinct_leaves_on_shared_executable(
    monkeypatch,
):
    """Two distinct leaves backed by the same Tier-1-cached `_executable` keep
    their own output reconstruction and flags while sharing `_executable`."""
    from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
        TorchTpuCompiledExecutable,
    )

    from vllm_torchtpu.compilation import tpu_compiler

    class Executable:
        def __reduce__(self):
            raise AssertionError("the executable must not be serialized")

    shared = Executable()
    first = TorchTpuCompiledExecutable(
        executable=shared,
        reconstruct_fx_outputs_fn=_tag_first_outputs,
        updates_default_generator_state=False,
        dynamic_outputs=[False],
        unique_output_indices=[0],
    )
    second = TorchTpuCompiledExecutable(
        executable=shared,
        reconstruct_fx_outputs_fn=_tag_second_outputs,
        updates_default_generator_state=True,
        dynamic_outputs=[True, False],
        unique_output_indices=[1, 0],
    )
    entry = SimpleNamespace(a=first, same_a=first, b=second)
    monkeypatch.setattr(tpu_compiler, "_deserialize_entry", lambda e: e)

    rebuilt = tpu_compiler._rebuild_sharing_executables(entry)

    assert rebuilt.a is not first and rebuilt.a is rebuilt.same_a
    assert rebuilt.b is not second and rebuilt.b is not rebuilt.a
    assert rebuilt.a._executable is shared and rebuilt.b._executable is shared
    assert rebuilt.a._reconstruct_fx_outputs_fn([], ["x"]) == ("first", ["x"])
    assert rebuilt.b._reconstruct_fx_outputs_fn([], ["x"]) == ("second", ["x"])
    assert rebuilt.a._updates_default_generator_state is False
    assert rebuilt.b._updates_default_generator_state is True
    assert rebuilt.a._dynamic_outputs == [False]
    assert rebuilt.b._dynamic_outputs == [True, False]
    assert rebuilt.a._unique_output_indices == [0]
    assert rebuilt.b._unique_output_indices == [1, 0]


def test_rebuild_sharing_executables_refuses_a_changed_leaf_constructor(monkeypatch):
    from vllm_torchtpu.compilation import tpu_compiler

    monkeypatch.setattr(tpu_compiler, "_TORCH_TPU_EXEC_INIT_MATCHES", False)
    monkeypatch.setattr(tpu_compiler, "_deserialize_entry", lambda e: e)
    with pytest.raises(TypeError, match=r"TorchTpuCompiledExecutable.__init__ takes"):
        tpu_compiler._rebuild_sharing_executables(SimpleNamespace(fw=_fake_leaf()))


def _split_artifact(graph_module, leaves=(), recompile_fn=None):
    """A `_SplitCompiledExecutable` over `graph_module`, with `leaves` attached
    to it the way SplitCompiler attaches compiled submodules."""
    from torch_tpu._internal.compile.split_compiler import (
        _SplitCompiledExecutable,
        _WrapperModule,
    )

    for i, leaf in enumerate(leaves):
        graph_module.add_submodule(f"submod_{i}", _WrapperModule(leaf, False))
    return _SplitCompiledExecutable(
        target=graph_module, recompile_fn=recompile_fn, graph_module_str="graph"
    )


def _bundled(fw, bw=None, *, has_bw=True):
    entry = SimpleNamespace(compiled_fw=SimpleNamespace(result=fw))
    entry.compiled_bw = SimpleNamespace(result=bw) if has_bw else None
    return entry


def test_release_compiled_graph_cycles_frees_split_graphs_by_refcount():
    """The split graph, its fx.Graph/nodes/meta/codegen, a graph its
    `recompile_fn` holds, and the backend's closure list are dismantled, so
    refcounting alone frees them; leaf executables stay for the rebuilt
    runnable."""
    import collections
    import functools
    import gc

    import torch
    from torch_tpu._internal.compile.split_compiler import (
        _register_output_namedtuple_globals,
    )

    from vllm_torchtpu.compilation.tpu_compiler import _release_compiled_graph_cycles

    leaf = _fake_leaf()
    executable = leaf._executable
    NamedOut = collections.namedtuple("NamedOut", ["y"])
    fw_gm = torch.fx.symbolic_trace(lambda x: NamedOut(y=x * 2))
    _register_output_namedtuple_globals(fw_gm)
    fw_gm.recompile()
    unshared_meta_tensor = torch.zeros(4)
    first_node = next(iter(fw_gm.graph.nodes))
    first_node.meta["probe"] = unshared_meta_tensor
    recompile_gm = torch.fx.symbolic_trace(_double)
    split = _split_artifact(
        fw_gm,
        [leaf, leaf],
        recompile_fn=functools.partial(lambda gm, tag: gm, recompile_gm, "tag"),
    )
    closure_gm = torch.fx.symbolic_trace(_double)
    closure_split = _split_artifact(closure_gm)
    compiled_artifacts = [split, closure_split, None, leaf]

    def backend_fn():
        return compiled_artifacts

    refs = [
        weakref.ref(obj)
        for obj in (
            fw_gm,
            fw_gm.graph,
            fw_gm.graph._codegen,
            first_node,
            unshared_meta_tensor,
            recompile_gm,
            recompile_gm.graph,
            closure_gm,
            closure_gm.graph,
        )
    ]
    entry = _bundled(split, None)
    del fw_gm, first_node, unshared_meta_tensor, recompile_gm, closure_gm

    # Sweep cyclic fx.Tracer instances from symbolic_trace setup before
    # disabling GC so the assertion tests pure refcounting on the artifacts.
    gc.collect()
    gc.disable()
    try:
        assert _release_compiled_graph_cycles(backend_fn, entry) is True
        assert compiled_artifacts == []
        assert split.recompile_fn is None
        assert split._graph_module_str is None
        assert split.compiled_executables == [leaf, leaf]
        assert leaf._executable is executable
        del split, closure_split, entry
        assert [ref() for ref in refs] == [None] * len(refs)
    finally:
        gc.enable()


def test_release_compiled_graph_cycles_leaves_preserved_graphs_and_meta():
    """GraphModules under `preserve` keep their nodes, and a node `meta` dict
    shared with a dismantled graph is not emptied."""
    import functools

    import torch

    from vllm_torchtpu.compilation.tpu_compiler import _release_compiled_graph_cycles

    caller = torch.fx.symbolic_trace(_double)
    split_gm = torch.fx.symbolic_trace(_double)
    for shared, node in zip(caller.graph.nodes, split_gm.graph.nodes):
        shared.meta["val"] = "shared"
        node.meta = shared.meta
    child = torch.fx.symbolic_trace(_double)
    split_gm.add_submodule("child", child)
    split_gm.add_submodule("plain", torch.nn.Linear(1, 1))
    caller.add_submodule("child", child)
    split = _split_artifact(
        split_gm, recompile_fn=functools.partial(lambda gm: gm, caller)
    )

    assert _release_compiled_graph_cycles(
        None, _bundled(split, has_bw=False), preserve=(caller, "not a module")
    )

    assert list(split_gm.graph.nodes) == []
    assert split_gm.graph.owning_module is None
    assert not split_gm.graph._graph_namespace._obj_to_name
    for module in (caller, child):
        assert [n.op for n in module.graph.nodes] == [
            "placeholder",
            "call_function",
            "output",
        ]
        assert module.graph.owning_module is module
    assert all(n.meta == {"val": "shared"} for n in caller.graph.nodes)


@pytest.mark.parametrize(
    "where", ["entry", "split", "closure"], ids=["in-entry", "in-split", "in-closure"]
)
def test_release_compiled_graph_cycles_reports_an_async_artifact(where):
    """An unresolved `AsyncCompiledArtifact` (or any unknown artifact) makes it
    return False, but the split graphs it knows are still dismantled."""
    import concurrent.futures

    import torch
    from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
        AsyncCompiledArtifact,
        NoOpCompiledArtifact,
    )

    from vllm_torchtpu.compilation.tpu_compiler import _release_compiled_graph_cycles

    pending = AsyncCompiledArtifact(concurrent.futures.Future())
    gm = torch.fx.symbolic_trace(_double)
    split = _split_artifact(
        gm, [_fake_leaf(), pending] if where == "split" else [_fake_leaf()]
    )
    compiled_artifacts = [pending] if where == "closure" else []

    def backend_fn():
        return compiled_artifacts

    bw = pending if where == "entry" else NoOpCompiledArtifact(None)
    assert _release_compiled_graph_cycles(backend_fn, _bundled(split, bw)) is False
    assert list(gm.graph.nodes) == []
    assert gm.graph.owning_module is None
    assert not pending.is_resolved


def test_saving_the_entry_resolves_an_async_artifact(monkeypatch):
    """`compile` and `load` release the in-process result only after
    `_save_handle` pickled its entry. Pickling resolves an
    `AsyncCompiledArtifact` into the leaf it wraps, so the release then
    counts it as known."""
    import concurrent.futures
    import pickle

    from torch_tpu._internal.compile import tpu_torch_compile
    from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
        AsyncCompiledArtifact,
        TorchTpuCompiledExecutable,
    )

    from vllm_torchtpu.compilation.tpu_compiler import _release_compiled_graph_cycles

    monkeypatch.setattr(
        tpu_torch_compile, "serialize_executable", lambda ex: ex.serialize_executable()
    )
    done = concurrent.futures.Future()
    done.set_result(_fake_leaf())
    pending = AsyncCompiledArtifact(done)
    entry = _bundled(pending, has_bw=False)
    assert _release_compiled_graph_cycles(None, entry) is False

    pending = AsyncCompiledArtifact(done)
    entry = _bundled(pending, has_bw=False)
    pickle.dumps(entry)

    assert type(pending) is TorchTpuCompiledExecutable
    assert _release_compiled_graph_cycles(None, entry) is True


def test_release_compiled_graph_cycles_needs_a_forward_artifact():
    from vllm_torchtpu.compilation.tpu_compiler import _release_compiled_graph_cycles

    with TestTpuCompilerCache._capture_vllm_info() as messages:
        assert _release_compiled_graph_cycles(None, None) is False
        assert _release_compiled_graph_cycles(None, "entry") is False
        assert _release_compiled_graph_cycles(None, _bundled(None)) is False
        assert (
            _release_compiled_graph_cycles(None, SimpleNamespace(compiled_fw=object()))
            is False
        )
        assert _release_compiled_graph_cycles(None, _bundled(_fake_leaf())) is True

    assert (
        sum(
            "Could not dismantle all compiled graph cycles" in m
            and "missing compiled_fw.result" in m
            for m in messages
        )
        == 1
    )


def test_release_compiled_graph_cycles_reports_unknown_wrapped_submod():
    """A `_WrapperModule` holding a non-`CompiledArtifact` is skipped by
    `_SplitCompiledExecutable.compiled_executables`, yet must still force a
    full collection while known graphs are dismantled."""
    import torch

    from vllm_torchtpu.compilation.tpu_compiler import _release_compiled_graph_cycles

    gm = torch.fx.symbolic_trace(_double)
    split = _split_artifact(gm, [_fake_leaf(), _UnknownArtifact()])
    assert _release_compiled_graph_cycles(None, _bundled(split)) is False
    assert list(gm.graph.nodes) == []
    assert gm.graph.owning_module is None


@pytest.mark.parametrize("gen0_suffices", [True, False])
def test_collect_dropped_result_sweeps_gen0_only_when_it_suffices(
    monkeypatch, gen0_suffices
):
    import gc

    from vllm_torchtpu.compilation import tpu_compiler

    calls = []
    monkeypatch.setattr(gc, "collect", lambda *args: calls.append(args) or 0)
    tpu_compiler._collect_dropped_result(gen0_suffices)
    assert calls == ([(0,)] if gen0_suffices else [()])


@pytest.mark.parametrize("was_wrapped", [True, False], ids=["wrapped", "tuple-output"])
def test_compile_unwraps_once_and_releases_backend_fn_before_peer_clear(
    tmp_path, monkeypatch, cpu_tpu_backend, was_wrapped
):
    """`compile` wraps only the final runnable (never wrapping `backend_fn` and
    then discarding that closure), so `backend_fn` and `entry` are already freed
    by refcounting before `_clear_failed` runs."""
    import gc

    import torch
    from vllm.config.utils import Range

    from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

    monkeypatch.setenv("TPU_PRECOMPILE_PEER_REPLAY", "1")
    unwrap_calls = []
    real_unwrap = TpuCompilerAdaptor._unwrap_compiled_fn

    def spy_unwrap(fn):
        unwrap_calls.append(fn)
        return real_unwrap(fn)

    alive_at_clear = []
    real_clear = TpuCompilerAdaptor._clear_failed

    def spy_clear(self, key):
        (built,) = cpu_tpu_backend.built
        alive_at_clear.append(
            {k for k in ("fn", "entry") if built.tracked[k]() is not None}
        )
        return real_clear(self, key)

    monkeypatch.setattr(
        TpuCompilerAdaptor, "_unwrap_compiled_fn", staticmethod(spy_unwrap)
    )
    monkeypatch.setattr(TpuCompilerAdaptor, "_clear_failed", spy_clear)

    graph = torch.fx.symbolic_trace(_double if was_wrapped else lambda x: (_double(x),))
    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(
        cache_dir=str(tmp_path / "hash" / "rank_0_0" / "backbone")
    )

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        fn, handle = compiler.compile(
            graph, [torch.tensor([3, 4])], {}, Range(start=2, end=2), key="bucket"
        )
    finally:
        if was_enabled:
            gc.enable()

    assert handle == (
        "bucket",
        str(tmp_path / "hash" / "rank_0_0" / "backbone" / "bucket"),
        was_wrapped,
    )
    assert len(unwrap_calls) == (1 if was_wrapped else 0)
    assert alive_at_clear == [set()]
    out = fn(torch.tensor([3, 4]))
    if was_wrapped:
        assert torch.equal(out, torch.tensor([6, 8]))
    else:
        assert isinstance(out, (tuple, list)) and len(out) == 1
        assert torch.equal(out[0], torch.tensor([6, 8]))


def test_reclaims_the_temporary_graph_deepcopy(
    tmp_path, monkeypatch, cpu_tpu_backend, promoted_to_gen2
):
    """The temporary `deepcopy(orig_graph)` used for tuple-output rewriting is
    dismantled and freed without a full collection across cold `compile`,
    warm `load`, stale-artifact `load`, and peer-replayed `compile`."""
    import copy
    import logging

    import torch
    from vllm.config.utils import Range

    from vllm_torchtpu.compilation import tpu_compiler
    from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

    helper = TestInProcessResultReclaim()
    orig_graph = helper._graph()
    deepcopies = []
    real_deepcopy = copy.deepcopy

    def spy_deepcopy(obj, *args, **kwargs):
        res = real_deepcopy(obj, *args, **kwargs)
        if obj is orig_graph:
            deepcopies.append(
                (
                    weakref.ref(res),
                    weakref.ref(res.graph),
                    [weakref.ref(n) for n in res.graph.nodes],
                )
            )
        return res

    monkeypatch.setattr(copy, "deepcopy", spy_deepcopy)

    # 1. Cold compile (promoted to Gen-2 during _save_handle)
    fn_cold, handle_cold = helper._compile(tmp_path / "cold", orig_graph)
    assert handle_cold is not None
    w_gm, w_graph, w_nodes = deepcopies.pop()
    assert promoted_to_gen2 == [0]
    assert w_gm() is None and w_graph() is None
    assert all(w() is None for w in w_nodes)
    helper._assert_runs(fn_cold)

    # 2. Warm load (replayed from disk; refcount drops dismantled copy)
    warm = TpuCompilerAdaptor()
    warm.initialize_cache(cache_dir=str(tmp_path / "cold" / "cache"))
    fn_warm = warm.load(
        handle_cold, orig_graph, [torch.tensor([3, 4])], 0, Range(start=2, end=2)
    )
    w_gm, w_graph, _ = deepcopies.pop()
    assert w_gm() is None and w_graph() is None
    helper._assert_runs(fn_warm)

    # 3. Stale load (self-heal recompile promoted to Gen-2 during _save_handle)
    fn_stale = helper._load_stale(tmp_path / "stale", orig_graph)
    w_gm, w_graph, w_nodes = deepcopies.pop()
    assert promoted_to_gen2 == [0, 0]
    assert w_gm() is None and w_graph() is None
    assert all(w() is None for w in w_nodes)
    helper._assert_runs(fn_stale)

    # 4. Peer-replayed compile (rank 1 replays rank 0's artifact)
    monkeypatch.setenv("TPU_PRECOMPILE_PEER_REPLAY", "1")
    built_before = len(cpu_tpu_backend.built)
    rank0 = TpuCompilerAdaptor()
    rank0.initialize_cache(cache_dir=str(tmp_path / "peer" / "rank_0_0" / "backbone"))
    _, h0 = rank0.compile(
        orig_graph, [torch.tensor([3, 4])], {}, Range(start=2, end=2), key="b_peer"
    )
    assert h0 is not None
    w_gm, w_graph, w_nodes = deepcopies.pop()
    assert w_gm() is None and w_graph() is None
    assert all(w() is None for w in w_nodes)

    rank1 = TpuCompilerAdaptor()
    rank1.initialize_cache(cache_dir=str(tmp_path / "peer" / "rank_1_0" / "backbone"))
    fn_peer, h1 = rank1.compile(
        orig_graph, [torch.tensor([3, 4])], {}, Range(start=2, end=2), key="b_peer"
    )
    assert h1 is not None
    assert len(cpu_tpu_backend.built) == built_before + 1
    w_gm, w_graph, _ = deepcopies.pop()
    assert w_gm() is None and w_graph() is None
    helper._assert_runs(fn_peer)

    # Attach a handler that retains LogRecords (and their args) so cases 5 & 6
    # genuinely prove logged exception messages cannot pin graph/entry via
    # retained tracebacks.
    retained_records: list[logging.LogRecord] = []

    class _RetainingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            retained_records.append(record)

    retaining_handler = _RetainingHandler()
    tpu_compiler.logger.addHandler(retaining_handler)
    try:
        # 5. Replay deserialize failure in load() does not pin graph/entry
        real_deserialize = tpu_compiler._deserialize_entry
        calls = 0

        def fail_once(entry):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("corrupt entry")
            return real_deserialize(entry)

        with monkeypatch.context() as m:
            m.setattr(tpu_compiler, "_deserialize_entry", fail_once)
            fn_healed = warm.load(
                handle_cold,
                orig_graph,
                [torch.tensor([3, 4])],
                0,
                Range(start=2, end=2),
            )
        w_gm, w_graph, w_nodes = deepcopies.pop()
        assert w_gm() is None and w_graph() is None
        assert all(w() is None for w in w_nodes)
        helper._assert_runs(fn_healed)

        # 6. Save failure in _save_handle does not pin entry via retained LogRecord
        import pickle

        def fail_dump(*_args, **_kwargs):
            raise OSError("disk full")

        with monkeypatch.context() as m:
            m.setattr(pickle, "dump", fail_dump)
            fn_nosave, h_nosave = helper._compile(tmp_path / "nosave", orig_graph)
        assert h_nosave is None
        deepcopies.pop()
        helper._assert_runs(fn_nosave)
        del fn_nosave
        assert cpu_tpu_backend.built[-1].tracked["entry"]() is None
        assert any("corrupt entry" in r.getMessage() for r in retained_records)
        assert any("disk full" in r.getMessage() for r in retained_records)
    finally:
        tpu_compiler.logger.removeHandler(retaining_handler)


def test_break_fx_graph_cycles_is_idempotent_and_safe_to_iterate():
    """Iterating or re-dismantling a graph after `_break_fx_graph_cycles` must
    remain safe (no C++ `_NodeIter` segfault on `_root` links), including when
    `_release_compiled_graph_cycles` dismantles a `graph` copy that
    `_drop_graph_copy` later visits."""
    import copy
    import functools
    import gc

    import torch

    from vllm_torchtpu.compilation.tpu_compiler import (
        _break_fx_graph_cycles,
        _drop_graph_copy,
        _release_compiled_graph_cycles,
    )

    gm = torch.fx.symbolic_trace(_double)
    _break_fx_graph_cycles(gm, set())
    assert list(gm.graph.nodes) == []
    assert isinstance(str(gm.graph), str)
    _break_fx_graph_cycles(gm, set())
    assert list(gm.graph.nodes) == []

    orig_graph = torch.fx.symbolic_trace(_double)
    graph = copy.deepcopy(orig_graph)
    split_gm = torch.fx.symbolic_trace(_double)
    split = _split_artifact(
        split_gm, recompile_fn=functools.partial(lambda g: g, graph)
    )
    w_gm = weakref.ref(graph)
    w_graph = weakref.ref(graph.graph)
    w_split_gm = weakref.ref(split_gm)
    w_split_graph = weakref.ref(split_gm.graph)
    del split_gm

    gc.collect()
    gc.disable()
    try:
        assert (
            _release_compiled_graph_cycles(
                None, _bundled(split, has_bw=False), preserve=(orig_graph,)
            )
            is True
        )
        assert list(graph.graph.nodes) == []
        _drop_graph_copy(graph, orig_graph)
        assert list(graph.graph.nodes) == []
        del split, graph
        assert w_gm() is None and w_graph() is None
        assert w_split_gm() is None and w_split_graph() is None
    finally:
        gc.enable()

    assert [n.op for n in orig_graph.graph.nodes] == [
        "placeholder",
        "call_function",
        "output",
    ]
