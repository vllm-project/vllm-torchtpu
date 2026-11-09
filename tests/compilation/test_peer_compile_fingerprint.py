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

import fcntl
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.config.utils import Range

from vllm_torchtpu.compilation import tpu_compiler

KEY = "artifact_bucket"


@pytest.fixture
def peer_compilers(tmp_path, monkeypatch):
    monkeypatch.setenv("TPU_PRECOMPILE_PEER_REPLAY", "1")
    monkeypatch.setattr(tpu_compiler, "_PEER_POLL_S", 0.001)
    monkeypatch.setattr(
        tpu_compiler, "_deserialize_entry", lambda entry: lambda *args: (entry,)
    )
    compiled = []

    def make(rank, namespace="serving"):
        compiler = tpu_compiler.TpuCompilerAdaptor()
        compiler.initialize_cache(
            str(tmp_path / namespace / f"rank_0_{rank}" / "model")
        )

        def backend(graph, inputs):
            compiled.append(rank)
            return lambda *args: (rank,), rank

        monkeypatch.setattr(compiler, "_run_backend", backend)
        return compiler

    def compile(compiler, graph, owner_slot=None):
        if owner_slot is None:
            return compiler.compile(graph, [], {}, Range(1, 32), key=KEY)
        with tpu_compiler.wait_for_peer_artifact(0.0, owner_slot=owner_slot):
            return compiler.compile(graph, [], {}, Range(1, 32), key=KEY)

    return make, compile, compiled


def graph(offset=0):
    return torch.fx.symbolic_trace(lambda x: x.narrow(0, offset, 2))


def publish_advice(compiler, fingerprint, slot):
    with compiler._publish_peer_fingerprint(KEY, fingerprint, slot):
        pass


def test_identical_graph_replays_after_delayed_owner(peer_compilers, monkeypatch):
    make, compile, compiled = peer_compilers
    owner, peer = make(0), make(1)
    other = make(2)
    publish_advice(other, tpu_compiler._graph_fingerprint(graph(2)), 2)
    started, release = threading.Event(), threading.Event()
    results, errors = {}, []
    original = owner._run_backend

    def delayed(*args):
        started.set()
        assert release.wait(2.0)
        return original(*args)

    monkeypatch.setattr(owner, "_run_backend", delayed)

    def run_owner():
        try:
            compile(owner, graph(), owner_slot=0)
        except Exception as exc:
            errors.append(exc)

    def run_peer():
        try:
            with tpu_compiler.wait_for_peer_artifact(1.0, owner_slot=0):
                results["fn"], _ = compile(peer, graph())
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run_owner), threading.Thread(target=run_peer)]
    threads[0].start()
    assert started.wait(2.0)
    threads[1].start()
    try:
        threads[1].join(0.02)
        assert threads[1].is_alive()
    finally:
        release.set()
        for thread in threads:
            thread.join(2.0)
    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert compiled == [0]
    assert results["fn"]() == 0
    assert Path(peer.cache_dir, KEY).samefile(Path(owner.cache_dir, KEY))


def test_incompatible_owner_does_not_delay_peer(peer_compilers, monkeypatch):
    make, compile, compiled = peer_compilers
    owner, peer = make(0), make(1)
    started, release, finished = (threading.Event() for _ in range(3))
    errors, results = [], {}
    original = owner._run_backend

    def delayed(*args):
        started.set()
        assert release.wait(2.0)
        return original(*args)

    monkeypatch.setattr(owner, "_run_backend", delayed)

    def run_owner():
        try:
            compile(owner, graph(), owner_slot=0)
        except Exception as exc:
            errors.append(exc)

    def run_peer():
        try:
            with tpu_compiler.wait_for_peer_artifact(1.0, owner_slot=0):
                results["fn"], _ = compile(peer, graph(2))
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()

    threads = [threading.Thread(target=run_owner), threading.Thread(target=run_peer)]
    threads[0].start()
    assert started.wait(2.0)
    threads[1].start()
    try:
        completed_before_owner = finished.wait(0.3)
    finally:
        release.set()
        for thread in threads:
            thread.join(2.0)
    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert completed_before_owner
    assert sorted(compiled) == [0, 1]
    assert results["fn"]() == 1
    assert not Path(peer.cache_dir, KEY).samefile(Path(owner.cache_dir, KEY))


def test_delayed_assigned_owner_advice_ignores_other_slot(peer_compilers, monkeypatch):
    make, compile, compiled = peer_compilers
    owner, peer, other = make(0), make(1), make(2)
    publish_advice(other, tpu_compiler._graph_fingerprint(graph(2)), 2)
    writing, publish, compiling, finish = (threading.Event() for _ in range(4))
    errors, results = [], {}
    replace = tpu_compiler.os.replace
    backend = owner._run_backend

    def delayed_publish(src, dst):
        if str(dst).endswith(".fingerprint"):
            writing.set()
            assert publish.wait(2.0)
        return replace(src, dst)

    def delayed_backend(*args):
        compiling.set()
        assert finish.wait(2.0)
        return backend(*args)

    monkeypatch.setattr(tpu_compiler.os, "replace", delayed_publish)
    monkeypatch.setattr(owner, "_run_backend", delayed_backend)

    def run_owner():
        try:
            compile(owner, graph(), owner_slot=0)
        except Exception as exc:
            errors.append(exc)

    def run_peer():
        try:
            with tpu_compiler.wait_for_peer_artifact(1.0, owner_slot=0):
                results["fn"], _ = compile(peer, graph())
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run_owner), threading.Thread(target=run_peer)]
    threads[0].start()
    assert writing.wait(2.0)
    assert list(Path(owner.cache_dir).glob("*.tmp"))
    threads[1].start()
    try:
        threads[1].join(0.03)
        waited_for_advice = threads[1].is_alive()
        publish.set()
        assert compiling.wait(2.0)
        threads[1].join(0.03)
        waited_for_artifact = threads[1].is_alive()
    finally:
        publish.set()
        finish.set()
        for thread in threads:
            thread.join(2.0)
    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert waited_for_advice and waited_for_artifact
    assert compiled == [0]
    assert results["fn"]() == 0


def test_ready_artifact_published_during_advice_read_is_replayed(
    peer_compilers, monkeypatch
):
    make, compile, compiled = peer_compilers
    owner, peer = make(0), make(1)
    publish_advice(owner, tpu_compiler._graph_fingerprint(graph(2)), 0)
    peer_paths = peer._peer_paths
    published = False

    def publish_while_reading(key):
        nonlocal published
        paths = peer_paths(key)
        if key.endswith(".fingerprint") and not published:
            published = True
            compile(owner, graph())
        return paths

    monkeypatch.setattr(peer, "_peer_paths", publish_while_reading)
    with tpu_compiler.wait_for_peer_artifact(0.1, owner_slot=0):
        fn, _ = compile(peer, graph())
    assert published
    assert fn() == 0
    assert compiled == [0]


@pytest.mark.parametrize("compatible_first", [True, False])
def test_multiple_ready_artifacts_keep_existing_selection(
    peer_compilers, compatible_first
):
    make, compile, compiled = peer_compilers
    first, second, peer = make(0), make(1), make(2)
    compile(first, graph(0 if compatible_first else 2))
    compile(second, graph(2 if compatible_first else 0))
    publish_advice(second, tpu_compiler._graph_fingerprint(graph(2)), 1)
    with tpu_compiler.wait_for_peer_artifact(0.1, owner_slot=1):
        fn, _ = compile(peer, graph())
    # Existing replay tests only the first sorted artifact, independently of advice.
    assert fn() == (0 if compatible_first else 2)
    assert compiled == ([0, 1] if compatible_first else [0, 1, 2])


def test_partial_artifact_is_rejected(peer_compilers):
    make, compile, compiled = peer_compilers
    owner, peer = make(0), make(1)
    directory = Path(owner.cache_dir)
    directory.mkdir(parents=True)
    (directory / KEY).write_bytes(b"\x80")
    publish_advice(owner, tpu_compiler._graph_fingerprint(graph()), 0)
    with tpu_compiler.wait_for_peer_artifact(0.03, owner_slot=0):
        fn, _ = compile(peer, graph())
    assert fn() == 1
    assert compiled == [1]


def test_stale_assigned_advice_only_forces_local_compile(peer_compilers):
    make, compile, compiled = peer_compilers
    owner, peer = make(0), make(1)
    publish_advice(owner, tpu_compiler._graph_fingerprint(graph(2)), 0)
    with tpu_compiler.wait_for_peer_artifact(0.1, owner_slot=0):
        fn, _ = compile(peer, graph())
    assert fn() == 1
    assert compiled == [1]


def test_wait_context_restores_assigned_owner():
    with tpu_compiler.wait_for_peer_artifact(0.1, owner_slot=3):
        with tpu_compiler.wait_for_peer_artifact(0.0, owner_slot=0):
            assert tpu_compiler._peer_wait.owner_slot == 0
        assert tpu_compiler._peer_wait.owner_slot == 3
        assert tpu_compiler._peer_wait.timeout_s == 0.1
    assert tpu_compiler._peer_wait.owner_slot is None
    assert tpu_compiler._peer_wait.timeout_s == 0.0


@pytest.mark.parametrize(
    "namespace, sizes, slot",
    [("profile", [32], 0), ("serving", [4, 32], 0), ("serving", [4, 32], 1)],
)
def test_range_allocator_publishes_owner_before_backend(
    peer_compilers, monkeypatch, namespace, sizes, slot
):
    from vllm.compilation import piecewise_backend
    from vllm.distributed import parallel_state

    from vllm_torchtpu import _patch_vllm_compile_all_ranges

    make, _, _ = peer_compilers
    compiler = make(slot, namespace)
    monkeypatch.setenv("TPU_PARALLEL_PRECOMPILE", "1")
    monkeypatch.setenv("TPU_PRECOMPILE_BIGGEST_FIRST", "0")
    monkeypatch.setenv("LOCAL_RANK", str(slot))
    monkeypatch.setenv("TPU_LOCAL_RANK_OFFSET", "0")
    monkeypatch.setattr(
        parallel_state, "get_world_group", lambda: SimpleNamespace(world_size=2)
    )
    monkeypatch.setattr(parallel_state, "get_node_count", lambda: 1)
    monkeypatch.setattr(piecewise_backend, "create_concrete_args", lambda *args: [])
    monkeypatch.setattr(
        piecewise_backend.PiecewiseBackend,
        "compile_all_ranges",
        piecewise_backend.PiecewiseBackend.compile_all_ranges,
    )
    monkeypatch.setattr(
        tpu_compiler.shape_variants, "register_deferred_compiler", lambda fn: None
    )
    _patch_vllm_compile_all_ranges()
    traced = graph()
    backend_calls, contexts = [], {}

    def backend(graph, inputs):
        expected = tpu_compiler._graph_fingerprint(graph)
        advice = Path(compiler.cache_dir, f"{KEY}.owner_{slot}.fingerprint")
        assert advice.read_text() == expected
        with advice.open() as lease, pytest.raises(BlockingIOError):
            fcntl.flock(lease, fcntl.LOCK_SH | fcntl.LOCK_NB)
        backend_calls.append(slot)
        return lambda *args: (slot,), slot

    def manager_compile(graph, *args, compile_range, **kwargs):
        contexts[compile_range.end] = (
            tpu_compiler._peer_wait.timeout_s,
            tpu_compiler._peer_wait.owner_slot,
        )
        if tpu_compiler._peer_wait.timeout_s == 0.0:
            return compiler.compile(graph, [], {}, compile_range, key=KEY)[0]
        return lambda *args: args

    monkeypatch.setattr(compiler, "_run_backend", backend)
    wrapped = SimpleNamespace(
        graph=traced,
        range_entries={
            size: SimpleNamespace(compile_range=Range(size, size), compiled=False)
            for size in sizes
        },
        vllm_backend=SimpleNamespace(
            compiler_manager=SimpleNamespace(
                compile=manager_compile, disable_cache=False
            ),
            inductor_config={},
            is_encoder=False,
        ),
        compilation_config=None,
        piecewise_compile_index=0,
        total_piecewise_compiles=1,
        _log_compile_start=lambda compile_range: None,
    )
    piecewise_backend.PiecewiseBackend.compile_all_ranges(wrapped)
    assert backend_calls == [slot]
    assert contexts[sizes[slot]] == (0.0, slot)
    for index, size in enumerate(sizes):
        if index != slot:
            assert contexts[size] == (tpu_compiler.PEER_WAIT_S, index)


@pytest.mark.parametrize("failure", ["missing", "save", "backend"])
def test_unavailable_owner_has_bounded_fallback(peer_compilers, monkeypatch, failure):
    make, compile, compiled = peer_compilers
    owner, peer = make(0), make(1)
    if failure == "save":
        monkeypatch.setattr(owner, "_save_handle", lambda *args: False)
        compile(owner, graph(), owner_slot=0)
    elif failure == "backend":

        def fail(*args):
            raise RuntimeError("compiler failed")

        monkeypatch.setattr(owner, "_run_backend", fail)
        with pytest.raises(RuntimeError, match="compiler failed"):
            compile(owner, graph(), owner_slot=0)
    start = time.monotonic()
    with tpu_compiler.wait_for_peer_artifact(0.03, owner_slot=0):
        fn, _ = compile(peer, graph())
    assert time.monotonic() - start < 1.0
    assert fn() == 1
    assert compiled[-1] == 1
    assert not Path(owner.cache_dir, KEY).exists()


@pytest.mark.parametrize("namespace", ["profile", "serving"])
def test_fingerprint_and_artifact_stay_in_namespace(peer_compilers, namespace):
    make, compile, compiled = peer_compilers
    owner = make(0, namespace)
    same = make(1, namespace)
    other = make(2, "serving" if namespace == "profile" else "profile")
    compile(owner, graph(), owner_slot=0)
    fn, _ = compile(same, graph())
    assert fn() == 0
    with tpu_compiler.wait_for_peer_artifact(0.03, owner_slot=0):
        fn, _ = compile(other, graph())
    assert fn() == 2
    assert compiled == [0, 2]


def test_advisory_fingerprint_cannot_authorize_different_graph(
    peer_compilers, monkeypatch
):
    make, compile, compiled = peer_compilers
    owner, peer = make(0), make(1)
    compile(owner, graph(), owner_slot=0)
    Path(owner.cache_dir, f"{KEY}.owner_0.fingerprint").write_text(
        tpu_compiler._graph_fingerprint(graph(2))
    )
    with tpu_compiler.wait_for_peer_artifact(0.1, owner_slot=0):
        fn, _ = compile(peer, graph(2))
    assert fn() == 1
    assert compiled == [0, 1]


def test_ready_artifact_still_validates_with_stale_advice(peer_compilers):
    make, compile, compiled = peer_compilers
    owner, peer = make(0), make(1)
    compile(owner, graph(), owner_slot=0)
    Path(owner.cache_dir, f"{KEY}.owner_0.fingerprint").write_text(
        tpu_compiler._graph_fingerprint(graph(2))
    )
    with tpu_compiler.wait_for_peer_artifact(0.1, owner_slot=0):
        fn, _ = compile(peer, graph())
    assert fn() == 0
    assert compiled == [0]


def test_advice_write_failure_does_not_fail_compile(peer_compilers, monkeypatch):
    make, compile, compiled = peer_compilers
    owner = make(0)

    replace = tpu_compiler.os.replace

    def fail(src, dst):
        if str(dst).endswith(".fingerprint"):
            raise OSError("cannot write advice")
        return replace(src, dst)

    monkeypatch.setattr(tpu_compiler.os, "replace", fail)
    fn, handle = compile(owner, graph(), owner_slot=0)
    assert fn() == 0
    assert compiled == [0]
    assert handle is not None
    assert not list(Path(owner.cache_dir).glob("*.tmp"))


@pytest.mark.parametrize("contents", [b"bad", b"\xff", b"a" * 12])
def test_malformed_advice_keeps_bounded_fallback(peer_compilers, contents):
    make, compile, compiled = peer_compilers
    owner, peer = make(0), make(1)
    directory = Path(owner.cache_dir)
    directory.mkdir(parents=True)
    (directory / f"{KEY}.owner_0.fingerprint").write_bytes(contents)
    with tpu_compiler.wait_for_peer_artifact(0.03, owner_slot=0):
        fn, _ = compile(peer, graph())
    assert fn() == 1
    assert compiled == [1]
