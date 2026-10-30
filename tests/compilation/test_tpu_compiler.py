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
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

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


def _fingerprint_at_rank(monkeypatch, fn, rank):
    import torch.distributed as dist
    from torch import fx

    from vllm_torchtpu.compilation import tpu_compiler

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda *a, **k: rank)
    return tpu_compiler._graph_fingerprint(fx.symbolic_trace(fn))


def test_a_graph_slicing_this_ranks_rows_is_never_replayed_by_another_rank(
    monkeypatch,
):
    # sequence_parallel_chunk reads the TP rank inside the op: the graph code
    # is the same on every rank, the lowered program is not. Peer replay of
    # it served every rank another rank's shard (GSM8K 2% with prefill SP).
    import torch
    import vllm.model_executor.models.utils  # noqa: F401  registers the op

    def sharded(x):
        return torch.ops.vllm.sequence_parallel_chunk_impl(x) * 2

    def replicated(x):
        return x * 2

    assert _fingerprint_at_rank(monkeypatch, sharded, 0) != _fingerprint_at_rank(
        monkeypatch, sharded, 1
    )
    assert _fingerprint_at_rank(monkeypatch, replicated, 0) == _fingerprint_at_rank(
        monkeypatch, replicated, 1
    )


class TestPrecompileProgress:
    """A waiter reads its bucket owner's precompile progress: it compiles the
    bucket itself when the owner failed it, and fails the boot when the owner
    skipped the subgraph or reported nothing before the backstop."""

    def _dirs(self, tmp_path):
        root = tmp_path / "hash"
        own = root / "rank_1_0" / "backbone"
        peer = root / "rank_0_0" / "backbone"
        own.mkdir(parents=True)
        peer.mkdir(parents=True)
        return str(own), str(peer)

    def test_owner_status_pending_failed_skipped(self, tmp_path):
        import time

        from vllm_torchtpu.compilation.tpu_compiler import PrecompileProgress

        own, peer = self._dirs(tmp_path)
        waiter = PrecompileProgress(own, subgraph=1)
        since = time.time() - 1
        assert waiter.owner_status(0, 512, since) == "pending"  # nothing yet
        owner = PrecompileProgress(peer, subgraph=1)
        owner.started()
        assert waiter.owner_status(0, 512, since) == "pending"
        owner.saved.append(256)
        owner.failed[512] = "MemoryError('vmem')"
        owner.finished()
        assert waiter.owner_status(0, 256, since) == "pending"  # saved: wait for file
        assert waiter.owner_status(0, 512, since).startswith("failed")
        assert waiter.owner_status(0, 1024, since).startswith("failed")
        # Another subgraph: the owner went on to subgraph 2 without starting 1.
        other = PrecompileProgress(own, subgraph=1)
        PrecompileProgress(peer, subgraph=2).started()
        assert other.owner_status(0, 512, since) != "skipped"  # it did start 1
        # Subgraph 0 was never started although the owner is past it: skipped.
        skipped = PrecompileProgress(own, subgraph=0)
        assert skipped.owner_status(0, 512, since) == "skipped"
        lagging = PrecompileProgress(str(tmp_path / "hash" / "rank_1_0" / "mtp"), 0)
        assert lagging.owner_status(0, 512, since) == "pending"  # other module
        # Records from another pod (a cache synced in from shared storage) are
        # ignored.
        import json
        import os

        with open(
            os.path.join(os.path.dirname(peer), ".precompile_progress"), "a"
        ) as f:
            f.write(
                json.dumps(
                    dict(
                        event="start",
                        prefix="backbone",
                        subgraph=9,
                        t=time.time(),
                        host="another-pod",
                    )
                )
                + "\n"
            )
        fresh = PrecompileProgress(own, subgraph=5)
        assert fresh.owner_status(0, 512, time.time() - 0.5) == "pending"

    def test_wait_fails_fast_on_a_skipped_subgraph_and_at_the_backstop(
        self, tmp_path, monkeypatch
    ):
        import time

        import pytest

        from vllm_torchtpu.compilation import tpu_compiler as tc

        own, peer = self._dirs(tmp_path)
        monkeypatch.setattr(tc, "_PEER_POLL_S", 0.01)
        adaptor = tc.TpuCompilerAdaptor.__new__(tc.TpuCompilerAdaptor)
        adaptor.cache_dir = own
        waiter = tc.PrecompileProgress(own, subgraph=1)

        # Owner moved on to subgraph 2 after we started waiting: diverged.
        def owner_skips():
            time.sleep(0.05)
            tc.PrecompileProgress(peer, subgraph=2).started()

        import threading

        threading.Thread(target=owner_skips).start()
        with tc.wait_for_peer_artifact(5.0, waiter):
            tc.set_peer_owner(0, 512)
            with pytest.raises(tc.PeerDivergedError, match="rank 0 skipped"):
                adaptor._replay_from_peer("artifact_x", None, False, "fp")
        # Owner says it failed the bucket: compile here (None), no error.
        tc.PrecompileProgress(peer, subgraph=3).started()
        failed = tc.PrecompileProgress(peer, subgraph=3)
        failed.failed[512] = "boom"
        failed.finished()
        with tc.wait_for_peer_artifact(5.0, tc.PrecompileProgress(own, subgraph=3)):
            tc.set_peer_owner(0, 512)
            assert adaptor._replay_from_peer("artifact_y", None, False, "fp") is None
        # Nothing reported at all: the backstop fails the boot, naming the rank.
        with tc.wait_for_peer_artifact(0.05, tc.PrecompileProgress(own, subgraph=4)):
            tc.set_peer_owner(0, 512)
            with pytest.raises(tc.PeerDivergedError, match="rank 0 reported nothing"):
                adaptor._replay_from_peer("artifact_z", None, False, "fp")

    def test_owner_records_isolates_dp_ranks_with_tuple_owner(
        self, tmp_path, monkeypatch
    ):
        import threading
        import time

        from vllm_torchtpu.compilation import tpu_compiler as tc

        root = tmp_path / "hash"
        dp1 = str(root / "rank_0_1" / "backbone")
        dp2 = str(root / "rank_0_2" / "backbone")
        dp3 = str(root / "rank_0_3" / "backbone")
        for d in (dp1, dp2, dp3):
            os.makedirs(d, exist_ok=True)
        monkeypatch.setattr(tc, "_PEER_POLL_S", 0.01)

        since = time.time() - 0.1
        # DP rank 1 already moved on to subgraph 2 without starting subgraph 1,
        # while DP rank 3 is actively compiling subgraph 1.
        tc.PrecompileProgress(dp1, subgraph=2).started()
        tc.PrecompileProgress(dp3, subgraph=1).started()

        waiter = tc.PrecompileProgress(dp2, subgraph=1)
        assert waiter.owner_status((0, 3), 512, since) == "pending"
        assert waiter.owner_status((0, 1), 512, since) == "skipped"

        def dp1_moves_ahead():
            time.sleep(0.05)
            tc.PrecompileProgress(dp1, subgraph=3).started()

        threading.Thread(target=dp1_moves_ahead).start()
        adaptor = tc.TpuCompilerAdaptor.__new__(tc.TpuCompilerAdaptor)
        adaptor.cache_dir = dp2
        with tc.wait_for_peer_artifact(2.0, waiter):
            tc.set_peer_owner((0, 1), 512)
            with pytest.raises(
                tc.PeerDivergedError, match=r"rank 0 \(dp 1\) skipped subgraph 1"
            ):
                adaptor._replay_from_peer("artifact_dp", None, False, "fp")


def test_load_refuses_cross_rank_or_unsalted_artifact_for_rank_dependent_graph(
    tmp_path, monkeypatch
):
    """Warm-start `load()` must reject a rank-dependent handle saved by another
    rank (or saved unsalted before rank salting existed), recompile for this
    rank, and overwrite the poisoned cache file so later boots replay cleanly."""
    import pickle
    import shutil

    import torch
    import torch.distributed as dist
    import vllm.model_executor.models.utils  # noqa: F401  registers the op
    from vllm.config.utils import Range

    from vllm_torchtpu.compilation import tpu_compiler

    (a, b), _, compiled, replayed = TestTpuCompilerCache._peer_compilers(
        tmp_path, monkeypatch, peer_replay="0"
    )
    active_rank = {"r": 0}
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda *args, **kwargs: active_rank["r"])

    class RankCompiled:
        def __init__(self, r):
            self.r = r

        def __call__(self, *args):
            return (self.r,)

        def serialize(self):
            return f"entry-from-rank-{self.r}"

    def active_backend(graph, example_inputs):
        compiled.append(active_rank["r"])
        return RankCompiled(active_rank["r"])

    monkeypatch.setattr(tpu_compiler, "_tpu_backend", active_backend)

    def sharded(x):
        return torch.ops.vllm.sequence_parallel_chunk_impl(x) * 2

    compile_range = Range(start=1, end=4096)

    # Rank 0 compiles and saves its rank-0-salted artifact.
    active_rank["r"] = 0
    graph_r0 = torch.fx.symbolic_trace(sharded)
    _, handle_r0 = a.compile(graph_r0, [], {}, compile_range, key="artifact_sp")
    assert compiled == [0]
    key, path_r0, was_wrapped = handle_r0

    # Poison rank 1's cache dir with rank 0's saved artifact (simulating a
    # pre-existing cross-rank hard-link in a persistent cache).
    os.makedirs(b.cache_dir, exist_ok=True)
    path_r1 = os.path.join(b.cache_dir, key)
    shutil.copyfile(path_r0, path_r1)

    active_rank["r"] = 1
    graph_r1 = torch.fx.symbolic_trace(sharded)
    fn_r1 = b.load((key, path_r1, was_wrapped), graph_r1, [], 0, compile_range)
    assert replayed == []
    assert compiled == [0, 1]
    assert fn_r1() == 1

    # Self-healed: a second warm start on rank 1 now replays rank 1's own handle.
    fn_r1_warm = b.load((key, path_r1, was_wrapped), graph_r1, [], 0, compile_range)
    assert compiled == [0, 1]
    assert replayed == ["entry-from-rank-1"]
    assert fn_r1_warm() == "entry-from-rank-1"

    # Also reject an older unsalted handle (`fingerprint=None`) on warm start.
    with open(path_r1, "rb") as f:
        unsalted = pickle.load(f)
    unsalted.fingerprint = None
    with open(path_r1, "wb") as f:
        pickle.dump(unsalted, f)

    b.load((key, path_r1, was_wrapped), graph_r1, [], 0, compile_range)
    assert compiled == [0, 1, 1]
    assert replayed == ["entry-from-rank-1"]
