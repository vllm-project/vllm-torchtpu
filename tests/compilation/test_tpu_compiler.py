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

    def test_tpu_hash_covers_the_ep_device_placement(self):
        """A fused EP MoE program has the EP placement compiled in, so a start
        whose EP ranks sit on other devices must not load it."""
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash
        from vllm_torchtpu.distributed import ep_mesh

        ranks = tuple(range(16))
        # Two 8-chip hosts; local ranks 0-7 sit on devices 0, 1, 4, 5, 6, 7, 2, 3.
        host = (0, 1, 4, 5, 6, 7, 2, 3)
        upper = tuple(d + 8 for d in host)
        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            with patch.dict(
                ep_mesh._GATHERED_DEVICE_IDS, {ranks: host + upper}, clear=True
            ):
                first = compute_tpu_compilation_hash(self._config())
                first_again = compute_tpu_compilation_hash(self._config())
            with patch.dict(
                ep_mesh._GATHERED_DEVICE_IDS, {ranks: upper + host}, clear=True
            ):
                hosts_swapped = compute_tpu_compilation_hash(self._config())

        assert first == first_again
        assert first != hosts_swapped

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
        assert sorted(os.listdir(tmp_path / "hash")) == [
            "peer_claims",
            "rank_0_0",
            "rank_1_0",
        ]
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
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        (a,), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch, n=1, peer_replay="0"
        )
        monkeypatch.setattr(TpuCompilerAdaptor, "_replay", lambda *args, **kw: None)
        fn, handle = compile(a)

        assert compiled == [0]
        assert fn() == 0
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

    def test_a_rank_tracing_another_graph_claims_the_bucket_at_once(
        self, tmp_path, monkeypatch
    ):
        """A claim, or a marker, for a different graph does not hold a waiter
        back: the waiter claims the bucket for its own graph and compiles it
        at once instead of waiting out the timeout."""
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
        with wait_for_peer_artifact(30.0):
            compile(b)
        assert time.monotonic() - start < 10.0
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

        monkeypatch.setattr(TpuCompilerAdaptor, "_claims", no_peer_lookup)
        monkeypatch.setattr(TpuCompilerAdaptor, "_claim", no_peer_lookup)
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

    def test_a_peer_artifact_is_replayed_for_another_handle_of_the_same_group(
        self, tmp_path, monkeypatch, group_stand_ins
    ):
        """Ranks trace different rank-local c10d group_name strings for the
        same process group, yet replay the first rank's compiled artifact."""
        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        compile(a, _trace_all_reduce_graph("1"))
        fn, handle = compile(b, _trace_all_reduce_graph("17"))

        assert compiled == [0]
        # Each compiling rank also reloads the artifact it saved.
        assert replayed == ["entry-from-rank-0", "entry-from-rank-0"]
        assert fn() == "entry-from-rank-0"
        key, path, _ = handle
        assert a.cache_dir is not None
        assert os.path.samefile(path, os.path.join(a.cache_dir, key))

    def test_another_c10d_subgroup_compiles_its_own_artifact(
        self, tmp_path, monkeypatch, group_stand_ins
    ):
        """torch_tpu >= 0.3.0 compiles a group's collective for the group's own
        devices, so a TP subgroup on other ranks must not replay the artifact
        of the first one: it fingerprints differently and compiles."""
        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        compile(a, _trace_all_reduce_graph("1"))
        fn, handle = compile(b, _trace_all_reduce_graph("7"))

        assert compiled == [0, 1]
        # Each compiling rank reloads only the artifact it saved itself.
        assert replayed == ["entry-from-rank-0", "entry-from-rank-1"]
        assert fn() == "entry-from-rank-1"
        key, path, _ = handle
        assert b.cache_dir is not None
        assert os.path.samefile(path, os.path.join(b.cache_dir, key))

    def test_a_subgroup_replays_its_own_owner_past_another_subgroups_artifact(
        self, tmp_path, monkeypatch, group_stand_ins
    ):
        """Ranks 1 and 2, of TP subgroup "7", wait while rank 0, of subgroup
        "1", compiles the bucket: they do not replay rank 0's artifact or wait
        out the timeout, one of them claims and compiles the bucket for their
        subgroup, and the other replays that one's artifact."""
        import threading
        import time

        from vllm_torchtpu.compilation.tpu_compiler import wait_for_peer_artifact

        (a, b, c), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch, n=3
        )
        compile(a, _trace_all_reduce_graph("1"))
        results = {}

        def waiter(compiler):
            with wait_for_peer_artifact(30.0):
                results[compiler.rank] = compile(compiler, _trace_all_reduce_graph("7"))

        start = time.monotonic()
        threads = [threading.Thread(target=waiter, args=(r,)) for r in (b, c)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30.0)
        assert time.monotonic() - start < 10.0
        assert not any(thread.is_alive() for thread in threads)

        assert compiled[0] == 0 and len(compiled) == 2 and compiled[1] in (1, 2)
        owner = compiled[1]
        tag = f"entry-from-rank-{owner}"
        assert replayed.count("entry-from-rank-0") == 1
        assert replayed.count(tag) == 2
        for rank in (1, 2):
            fn, (key, path, _) = results[rank]
            assert fn() == tag
            owner_dir = (b, c)[owner - 1].cache_dir
            assert owner_dir is not None
            assert os.path.samefile(path, os.path.join(owner_dir, key))

    def test_the_third_rank_replays_the_second_ranks_artifact(
        self, tmp_path, monkeypatch, group_stand_ins
    ):
        """Ranks on groups "1", "7", "7": the third rank does not stop at the
        first rank's artifact, which is for another subgroup, and replays the
        second rank's."""
        (a, b, c), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch, n=3
        )
        compile(a, _trace_all_reduce_graph("1"))
        compile(b, _trace_all_reduce_graph("7"))
        fn, (key, path, _) = compile(c, _trace_all_reduce_graph("7"))

        assert compiled == [0, 1]
        assert replayed == [
            "entry-from-rank-0",
            "entry-from-rank-1",
            "entry-from-rank-1",
        ]
        assert fn() == "entry-from-rank-1"
        assert b.cache_dir is not None
        assert os.path.samefile(path, os.path.join(b.cache_dir, key))

    def test_a_claim_from_an_earlier_run_is_taken_over(self, tmp_path, monkeypatch):
        """A claim older than this process names an owner of an earlier run:
        a waiting rank does not wait for that owner's artifact, and once the
        wait for an owner of this run runs out it claims and compiles the
        bucket itself."""
        import time

        from vllm_torchtpu.compilation import tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import wait_for_peer_artifact

        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        compile(a)
        claims = tmp_path / "hash" / "peer_claims" / "backbone"
        [claim] = os.listdir(claims)
        old = tpu_compiler._PROCESS_START - 3600
        os.utime(claims / claim, (old, old))
        os.remove(os.path.join(a.cache_dir, "artifact_bucket"))
        start = time.monotonic()
        with wait_for_peer_artifact(0.3):
            compile(b)
        assert 0.3 <= time.monotonic() - start < 10.0
        assert compiled == [0, 1]
        with open(claims / claim) as f:
            assert f.read() == os.path.join(b.cache_dir, "artifact_bucket")


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
