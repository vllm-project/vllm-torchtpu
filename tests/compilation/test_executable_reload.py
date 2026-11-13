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
"""Unit and TPU hardware tests for VLLM_TPU_RELOAD_COMPILED_EXECUTABLE.

Kept in its own file so removing the `executable_reload.py` workaround deletes
this file cleanly without churning `test_tpu_compiler.py`.
"""

import contextlib
import dataclasses
import logging
import os
import pickle
import sys
import weakref
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from torch_tpu._internal.compile.split_compiler import _SplitCompiledExecutable
from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
    NoOpCompiledArtifact,
    TorchTpuCompiledExecutable,
)


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


class _UnknownArtifact:
    pass


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
        extra_fw_artifact: Any | None = None,
    ) -> None:
        self.state = state
        self.scale = scale
        self.cache_hit = cache_hit
        self.unserializable = unserializable
        self.broken_serialize = broken_serialize
        self.fail_wrap = fail_wrap
        self.extra_fw_artifact = extra_fw_artifact
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

        fw_leaves = [
            _ReloadLeafArtifact(f"c{idx}_fw0", self.scale),
            _ReloadLeafArtifact(f"c{idx}_fw1", 1),
        ]
        bw = _ReloadLeafArtifact(f"c{idx}_bw", 1)
        self.state.live_leaves.extend([*fw_leaves, bw])
        split_items: list[Any] = list(fw_leaves)
        if self.extra_fw_artifact is not None:
            split_items.append(self.extra_fw_artifact)
        split = _ReloadSplitArtifact(split_items)
        entry = _ReloadBundledEntry(
            _ReloadLoadable(split),
            _ReloadLoadable(bw),
            fail_wrap=idx in self.fail_wrap,
        )

        def runnable(x):
            out = x
            for leaf in fw_leaves:
                out = leaf(out)
            return (out,)

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
    def captured_vllm_log(self):
        """Capture `vllm` logger messages (`propagate=False` bypasses `caplog`)."""
        import vllm.logger  # noqa: F401

        @contextlib.contextmanager
        def capture(level: int = logging.INFO):
            messages: list[str] = []
            handler = logging.Handler(level)
            handler.emit = lambda record: messages.append(record.getMessage())
            vllm_logger = logging.getLogger("vllm")
            old_level = vllm_logger.level
            old_disable = logging.root.manager.disable
            vllm_logger.addHandler(handler)
            if vllm_logger.level == logging.NOTSET or vllm_logger.level > level:
                vllm_logger.setLevel(level)
            logging.disable(logging.NOTSET)
            try:
                yield messages
            finally:
                vllm_logger.removeHandler(handler)
                vllm_logger.setLevel(old_level)
                logging.disable(old_disable)

        return capture

    @pytest.fixture
    def reload_state(self, monkeypatch):
        """A fresh `_ReloadState`, a fake `torch.tpu` cache, and reload on."""
        from vllm_torchtpu import envs

        state = _ReloadState()
        monkeypatch.setattr(sys.modules[__name__], "_reload_state", state)
        for name in (
            "VLLM_TPU_RELOAD_COMPILED_EXECUTABLE",
            "TPU_PRECOMPILE_PEER_REPLAY",
        ):
            monkeypatch.delitem(vars(envs), name, raising=False)
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("VLLM_TPU_RELOAD_COMPILED_EXECUTABLE", "1")
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
        return state

    @staticmethod
    def _install(monkeypatch, backend):
        from vllm_torchtpu.compilation import tpu_compiler

        monkeypatch.setattr(tpu_compiler, "_tpu_backend", backend)
        return backend

    @pytest.mark.parametrize("use_disk_cache", [True, False], ids=["disk", "memory"])
    def test_compile_reloads_released_executable(
        self, tmp_path, monkeypatch, reload_state, captured_vllm_log, use_disk_cache
    ):
        """A new CompileAndLoad is serialized once (to the artifact, or to bytes
        without a disk cache), released with the runnable dropped and Tier-1
        cleared, and only then loaded back."""
        self._install(monkeypatch, _FakeTpuBackend(reload_state, 10))
        reload_state.check_released_on_load = True
        compiler = _adaptor(tmp_path / "cache" if use_disk_cache else None)

        with captured_vllm_log() as messages:
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

        self._install(monkeypatch, _FakeTpuBackend(reload_state, 10))
        graph = torch.fx.symbolic_trace(lambda x: x)
        _, handle = _compile(_adaptor(tmp_path / "cache"), graph=graph)
        assert handle is not None

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
        compiled executable is kept untouched."""
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

    def test_unsupported_artifact_warns_once_and_keeps_executable(
        self, tmp_path, monkeypatch, reload_state, captured_vllm_log
    ):
        """An unknown CompiledArtifact subclass keeps the CompileAndLoad
        executable intact and logs one actionable warning."""
        backend = self._install(
            monkeypatch,
            _FakeTpuBackend(reload_state, 6, extra_fw_artifact=_UnknownArtifact()),
        )
        compiler = _adaptor(tmp_path / "cache")

        with captured_vllm_log() as messages:
            fn1, handle1 = _compile(compiler, key="b1")
            fn2, handle2 = _compile(compiler, key="b2")

        assert backend.calls == 2
        assert handle1 is not None and handle2 is not None
        assert not any(leaf.released for leaf in reload_state.live_leaves)
        assert "clear_tier1_cache" not in reload_state.events
        warnings = [
            m for m in messages if "Unset VLLM_TPU_RELOAD_COMPILED_EXECUTABLE" in m
        ]
        assert len(warnings) == 1
        assert torch.equal(fn1(torch.tensor([3, 4])), torch.tensor([18, 24]))
        assert torch.equal(fn2(torch.tensor([3, 4])), torch.tensor([18, 24]))

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

    @pytest.mark.parametrize(
        "fallback_serializes", [True, False], ids=["updates-file", "deletes-file"]
    )
    def test_load_failed_reload_recompiles_and_updates_or_deletes_corrupt_file(
        self, tmp_path, monkeypatch, reload_state, fallback_serializes
    ):
        """When `load()` self-heals a corrupt file and the reload fails after
        release, `load()` recompiles from `orig_graph`, returns a working
        runnable, and either overwrites or deletes the corrupt file on disk."""
        from vllm.config.utils import Range

        backend = self._install(
            monkeypatch,
            _FakeTpuBackend(
                reload_state,
                8,
                fail_wrap=frozenset({0}),
                broken_serialize=frozenset() if fallback_serializes else frozenset({1}),
            ),
        )
        artifact = tmp_path / "stale_bucket"
        artifact.write_bytes(b"corrupt")
        orig_graph = torch.fx.symbolic_trace(lambda x: x)
        compiler = _adaptor(tmp_path)

        fn = compiler.load(
            ("stale_bucket", str(artifact), True),
            orig_graph,
            [torch.tensor([3, 4])],
            0,
            Range(start=2, end=2),
        )

        assert backend.calls == 2
        assert backend.saw_mutated == [False, False]
        assert not hasattr(orig_graph, "_pre_grad_mutated")
        assert torch.equal(fn(torch.tensor([3, 4])), torch.tensor([24, 32]))
        if fallback_serializes:
            assert artifact.exists()
            assert artifact.read_bytes() != b"corrupt"
            self._install(
                monkeypatch, lambda *a: pytest.fail("repaired handle must warm-load")
            )
            warm = _adaptor(tmp_path).load(
                ("stale_bucket", str(artifact), True),
                orig_graph,
                [torch.tensor([3, 4])],
                0,
                Range(start=2, end=2),
            )
            assert torch.equal(warm(torch.tensor([3, 4])), torch.tensor([24, 32]))
        else:
            assert not artifact.exists()

    def test_release_entry_executables_clears_every_leaf_once(self):
        """Shared and cyclic splits are visited once; `NoOpCompiledArtifact`
        and empty results are skipped."""
        from vllm_torchtpu.compilation.executable_reload import (
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

    def test_release_entry_executables_rejects_unsupported_types_unreleased(self):
        """An unsupported artifact anywhere in the entry raises `TypeError`
        before any leaf, earlier in the tree or in the forward, is cleared."""
        from vllm_torchtpu.compilation.executable_reload import (
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


def _compile_and_reload_on_tpu(root: str, inductor_dir: str) -> None:
    """Child-process body of `test_reload_compiled_executable_on_tpu`."""
    import torch._dynamo.utils
    from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
    from torch.compiler._cache import CacheArtifactManager
    from vllm.config.utils import Range

    from vllm_torchtpu import patch_registry
    from vllm_torchtpu.compilation import executable_reload, tpu_compiler
    from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor
    from vllm_torchtpu.tpu_info import get_num_chips

    if get_num_chips() == 0:
        pytest.skip("TPU hardware required")

    expected_aot_dir = os.path.join(inductor_dir, "aotautograd")
    assert AOTAutogradCache._get_tmp_dir() == expected_aot_dir

    patch_registry.apply("worker_init")
    real_backend = tpu_compiler._tpu_backend

    def tracked_backend(graph, example_inputs):
        compiled_fn = real_backend(graph, example_inputs)
        # Invariant required by `release_entry_executables`: `entry.compiled_fw`
        # must hold the exact `CompiledArtifact` object closed over by
        # `compiled_fn` (preserving this identity is why `AOTAutogradCache.save`
        # and its `entry.pre_save()` copy are skipped).
        freevars = dict(
            zip(
                compiled_fn.__code__.co_freevars,
                (c.cell_contents for c in compiled_fn.__closure__ or ()),
            )
        )
        assert "compiled_artifacts" in freevars, (
            "TpuBackend.__call__ closure no longer exposes 'compiled_artifacts'; "
            "re-validate executable_reload.release_entry_executables against the "
            "new torch_tpu."
        )
        assert (
            compiled_fn.serialize().compiled_fw.result
            is freevars["compiled_artifacts"][0]
        )
        return compiled_fn

    tpu_compiler._tpu_backend = tracked_backend

    released = []
    release = executable_reload.release_entry_executables

    def tracked_release(entry):
        released.append(entry)
        release(entry)

    executable_reload.release_entry_executables = tracked_release

    def verify_case(name: str, step_fn, expected: torch.Tensor, min_leaves: int):
        released.clear()
        cache_dir = os.path.join(root, name)
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
            step_fn, backend=adaptor_backend, fullgraph=True, dynamic=False
        )
        x = torch.arange(8, dtype=torch.float32, device="tpu")
        assert torch.equal(compiled_step(x).cpu(), expected)
        assert len(compiled) == 1 and len(released) == 1
        result = released[0].compiled_fw.result
        leaves = [
            sub
            for sub in (
                result.compiled_executables
                if isinstance(result, _SplitCompiledExecutable)
                else [result]
            )
            if isinstance(sub, TorchTpuCompiledExecutable)
        ]
        assert len(leaves) >= min_leaves, (
            f"expected >= {min_leaves} TPU leaves for {name}, got {len(leaves)}"
        )
        assert all(
            leaf._executable is None and leaf._fast_runner is None for leaf in leaves
        )

        # A warm start replays the saved artifact without compiling again.
        graph, example_inputs, handle = compiled[0]
        before = torch.tpu._get_cache_stats().num_compilations
        warm = TpuCompilerAdaptor()
        warm.initialize_cache(cache_dir=cache_dir)
        (out,) = warm.load(handle, graph, example_inputs, 0, Range(start=8, end=8))(x)
        assert torch.equal(out.cpu(), expected)
        assert torch.tpu._get_cache_stats().num_compilations == before

    # 1. Single-leaf TPU graph.
    base = torch.arange(8, dtype=torch.float32)
    verify_case("single_leaf", lambda x: x * 2.0 + 1.0, base * 2.0 + 1.0, min_leaves=1)

    # 2. Multi-leaf `_SplitCompiledExecutable` (3 TPU partitions on one chip).
    from unittest.mock import patch

    from torch_tpu._internal.compile import split_compiler, tpu_torch_compile
    from torch_tpu._internal.distributed import collective_ops

    orig_is_collective = collective_ops.is_collective_op
    orig_get_bool_env = split_compiler.env.get_bool_env_once
    with (
        patch.object(
            collective_ops,
            "is_collective_op",
            side_effect=lambda op: (op == torch.ops.aten.neg or orig_is_collective(op)),
        ),
        patch.object(
            split_compiler.env,
            "get_bool_env_once",
            side_effect=lambda k, default_value=False: (
                True
                if k == "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS"
                else orig_get_bool_env(k, default_value=default_value)
            ),
        ),
        patch.object(
            tpu_torch_compile,
            "get_handshake_stage_env_var_once",
            return_value=tpu_torch_compile.HandshakeStage.OFF,
        ),
    ):
        verify_case(
            "multi_leaf_split",
            lambda x: torch.neg(x * 2.0 + 1.0) * 3.0,
            -(base * 2.0 + 1.0) * 3.0,
            min_leaves=3,
        )
    assert not os.path.exists(expected_aot_dir)
    assert torch._dynamo.utils.counters["aot_autograd"]["autograd_cache_saved"] == 0
    assert CacheArtifactManager.serialize() is None


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
def test_reload_compiled_executable_on_tpu(tmp_path, monkeypatch):
    """Real TpuBackend: single-leaf and multi-leaf `_SplitCompiledExecutable`
    cold compiles under VLLM_TPU_RELOAD_COMPILED_EXECUTABLE release and reload
    every TPU leaf, write no AOTAutogradCache entry, and warm-load cleanly."""
    from tests.distributed.tpu_test_utils import run_in_isolated_process

    inductor_dir = str(tmp_path / "inductor")
    # Neither persistent torch_tpu cache may serve the graph: the cold compile
    # has to be a new CompileAndLoad for the reload to run.
    monkeypatch.setenv("TORCH_TPU_TIER2_COMPILATION_CACHE", "disabled")
    monkeypatch.delenv("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", raising=False)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", inductor_dir)
    monkeypatch.setenv("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    monkeypatch.setenv("VLLM_TPU_RELOAD_COMPILED_EXECUTABLE", "1")

    run_in_isolated_process(
        partial(_compile_and_reload_on_tpu, str(tmp_path), inductor_dir), timeout=180
    )
