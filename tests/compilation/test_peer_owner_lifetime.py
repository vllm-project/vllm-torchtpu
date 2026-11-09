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
import dataclasses
import fcntl
import multiprocessing
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm import envs as vllm_envs
from vllm.config.utils import Range

from vllm_torchtpu.compilation import tpu_compiler

KEY = "artifact_bucket"


def _graph(offset=0):
    return torch.fx.symbolic_trace(lambda x: x.narrow(0, offset, 2) + 17)


def _compiler(root, rank, namespace):
    compiler = tpu_compiler.TpuCompilerAdaptor()
    compiler.initialize_cache(str(Path(root) / namespace / f"rank_0_{rank}" / "model"))
    return compiler


def _owner(root, ready, release, mode, namespace, offset, hold_at):
    """Use real publication/save/exception paths with a CPU graph backend."""
    compiler = _compiler(root, 0, namespace)
    save = compiler._save_handle
    replace = os.replace

    def hold():
        ready.set()
        assert release.wait(30), "owner test release missing"

    def backend(graph, inputs):
        if hold_at == "backend":
            hold()
        if mode == "crash":
            os._exit(23)
        if mode == "exit":
            os._exit(0)
        if mode == "error":
            raise ValueError("owner backend failed")
        return graph, None if mode == "unserializable" else graph

    def publish(src, dst):
        if hold_at == "publication" and str(dst).endswith(".fingerprint"):
            hold()
        return replace(src, dst)

    def save_entry(key, path, entry, graph, wrapped, fingerprint):
        if hold_at == "save":
            hold()
        if mode == "save_failure":
            return False
        if mode == "partial":
            Path(path).write_bytes(b"\x80")
            return False
        if mode == "wrong_wrapper":
            wrapped = not wrapped
        if mode == "wrong_fingerprint":
            fingerprint = "0" * 64
        return save(key, path, entry, graph, wrapped, fingerprint)

    compiler._run_backend = backend
    compiler._save_handle = save_entry
    if hold_at == "publication":
        os.replace = publish
    try:
        with tpu_compiler.wait_for_peer_artifact(0.0, owner_slot=0):
            compiler.compile(_graph(offset), [], {}, Range(1, 32), key=KEY)
    except ValueError:
        assert mode == "error"
        Path(root, "backend-error-observed").touch()


@pytest.fixture
def owners(tmp_path, monkeypatch):
    monkeypatch.setenv("TPU_PRECOMPILE_PEER_REPLAY", "1")
    context = multiprocessing.get_context("spawn")

    @contextlib.contextmanager
    def start(mode="success", namespace="profile", offset=0, hold_at="backend"):
        ready, release = context.Event(), context.Event()
        process = context.Process(
            target=_owner,
            args=(str(tmp_path), ready, release, mode, namespace, offset, hold_at),
        )
        process.start()
        try:
            assert ready.wait(30), (
                f"owner failed before backend/save: {process.exitcode}"
            )
            yield SimpleNamespace(process=process, release=release)
        finally:
            release.set()
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join(5)
            assert not process.is_alive()

    return start


@dataclasses.dataclass
class _Clock:
    elapsed: float = 0.0
    advance: Callable[[float], None] | None = None

    def monotonic(self):
        return self.elapsed

    def sleep(self, seconds):
        self.elapsed += seconds
        if self.advance is not None:
            self.advance(self.elapsed)


@pytest.fixture
def peer(tmp_path, monkeypatch):
    monkeypatch.setenv("TPU_PRECOMPILE_PEER_REPLAY", "1")
    monkeypatch.setattr(vllm_envs, "VLLM_ENGINE_READY_TIMEOUT_S", 8)
    monkeypatch.setattr(tpu_compiler, "_PEER_POLL_S", 1)
    monkeypatch.setattr(tpu_compiler, "_deserialize_entry", lambda entry: entry)
    calls = []

    def compile(namespace="profile", offset=0):
        compiler = _compiler(tmp_path, 1, namespace)

        def backend(graph, inputs):
            calls.append(namespace)
            return graph, graph

        monkeypatch.setattr(compiler, "_run_backend", backend)
        with tpu_compiler.wait_for_peer_artifact(2, owner_slot=0):
            fn, _ = compiler.compile(_graph(offset), [], {}, Range(1, 32), key=KEY)
        values = torch.arange(8, dtype=torch.int64)
        assert torch.equal(fn(values), values[offset : offset + 2] + 17)
        return compiler

    return compile, calls


def _finish(owner):
    owner.release.set()
    owner.process.join(10)
    assert not owner.process.is_alive()


@pytest.mark.parametrize("hold_at", ["backend", "save"])
def test_live_owner_replays_beyond_discovery(
    owners, peer, tmp_path, monkeypatch, hold_at
):
    compile, calls = peer
    with owners(hold_at=hold_at) as owner:
        clock = _Clock(advance=lambda elapsed: _finish(owner) if elapsed >= 3 else None)
        monkeypatch.setattr(tpu_compiler, "time", clock)
        compiler = compile()
        print(
            f"compiler_source={tpu_compiler.__file__} "
            f"virtual_wait_s={clock.elapsed} peer_backend_calls={len(calls)}"
        )
        assert not calls, "live compatible owner must survive discovery timeout"
        assert clock.elapsed == 3
        assert owner.process.exitcode == 0
        assert Path(compiler.cache_dir, KEY).samefile(
            tmp_path / "profile/rank_0_0/model" / KEY
        )


@pytest.mark.parametrize(
    "mode", ["crash", "exit", "error", "unserializable", "save_failure"]
)
def test_owner_termination_releases_wait(owners, peer, tmp_path, monkeypatch, mode):
    compile, calls = peer
    with owners(mode=mode) as owner:
        clock = _Clock(advance=lambda elapsed: _finish(owner))
        monkeypatch.setattr(tpu_compiler, "time", clock)
        compile()
        assert calls == ["profile"]
        assert clock.elapsed == 1
        assert owner.process.exitcode == (23 if mode == "crash" else 0)
        if mode == "error":
            assert (tmp_path / "backend-error-observed").exists()


@pytest.mark.parametrize("mode", ["partial", "wrong_wrapper", "wrong_fingerprint"])
def test_owner_artifact_is_strictly_validated(owners, peer, monkeypatch, mode):
    compile, calls = peer
    with owners(mode=mode) as owner:
        clock = _Clock(advance=lambda elapsed: _finish(owner))
        monkeypatch.setattr(tpu_compiler, "time", clock)
        compile()
        assert calls == ["profile"]
        assert clock.elapsed == 1
        assert owner.process.exitcode == 0


@pytest.mark.parametrize("seconds", [1, 8])
def test_live_owner_hard_deadline(owners, peer, monkeypatch, seconds):
    compile, calls = peer
    with owners() as owner:
        clock = _Clock()
        monkeypatch.setattr(tpu_compiler, "time", clock)
        monkeypatch.setattr(vllm_envs, "VLLM_ENGINE_READY_TIMEOUT_S", seconds)
        compile()
        assert calls == ["profile"]
        assert clock.elapsed == seconds
        assert owner.process.is_alive()


@pytest.mark.parametrize("case", ["incompatible", "other_namespace"])
def test_live_owner_does_not_extend_unrelated_wait(owners, peer, monkeypatch, case):
    compile, calls = peer
    with owners(offset=2 if case == "incompatible" else 0) as owner:
        clock = _Clock()
        monkeypatch.setattr(tpu_compiler, "time", clock)
        compile(namespace="serving" if case == "other_namespace" else "profile")
        assert len(calls) == 1
        assert clock.elapsed == (0 if case == "incompatible" else 2)
        assert owner.process.is_alive()


def test_competing_publication_cannot_replace_live_lease(owners, tmp_path):
    with owners() as owner:
        compiler = _compiler(tmp_path, 0, "profile")
        advice = Path(compiler.cache_dir, f"{KEY}.owner_0.fingerprint")
        original = advice.read_bytes()
        inode = advice.stat().st_ino
        with compiler._publish_peer_fingerprint(KEY, "0" * 64, 0):
            assert advice.read_bytes() == original
            assert advice.stat().st_ino == inode
        with advice.open() as lease, pytest.raises(BlockingIOError):
            fcntl.flock(lease, fcntl.LOCK_SH | fcntl.LOCK_NB)
        _finish(owner)
        with compiler._publish_peer_fingerprint(KEY, "0" * 64, 0):
            assert advice.read_text() == "0" * 64
        assert not list(Path(compiler.cache_dir).glob("*.tmp"))


@pytest.mark.parametrize("stale", ["same_owner", "other_owner", "missing"])
def test_pending_publication_does_not_use_stale_advice(
    owners, peer, tmp_path, monkeypatch, stale
):
    compile, calls = peer
    if stale != "missing":
        rank = 0 if stale == "same_owner" else 2
        path = tmp_path / f"profile/rank_0_{rank}/model" / f"{KEY}.owner_0.fingerprint"
        path.parent.mkdir(parents=True)
        path.write_text("0" * 64)
    with owners(hold_at="publication") as owner:
        clock = _Clock(advance=lambda elapsed: _finish(owner))
        monkeypatch.setattr(tpu_compiler, "time", clock)
        compile()
        assert not calls
        assert clock.elapsed == 1


@pytest.mark.parametrize("live", [True, False])
def test_other_publication_does_not_mask_live_compatible_owner(
    owners, peer, tmp_path, monkeypatch, live
):
    compile, calls = peer
    other = _compiler(tmp_path, 2, "profile")
    with owners() as owner:
        clock = _Clock(advance=lambda elapsed: _finish(owner) if elapsed >= 3 else None)
        monkeypatch.setattr(tpu_compiler, "time", clock)
        with contextlib.ExitStack() as stack:
            publication = other._publish_peer_fingerprint(KEY, "0" * 64, 0)
            if live:
                stack.enter_context(publication)
            else:
                with publication:
                    pass
            compile()
        assert not calls
        assert clock.elapsed == 3


@pytest.mark.parametrize("advice", [None, b"bad", b"\xff", "compatible", b"0" * 64])
def test_unheld_advice_cannot_extend_discovery(peer, tmp_path, monkeypatch, advice):
    compile, calls = peer
    if advice is not None:
        path = tmp_path / "profile/rank_0_0/model" / f"{KEY}.owner_0.fingerprint"
        path.parent.mkdir(parents=True)
        if advice == "compatible":
            graph, _ = tpu_compiler._ensure_tuple_output(_graph())
            path.write_text(tpu_compiler._graph_fingerprint(graph))
        else:
            path.write_bytes(advice)
    clock = _Clock()
    monkeypatch.setattr(tpu_compiler, "time", clock)
    compile()
    assert calls == ["profile"]
    assert clock.elapsed <= 2


def test_reader_guard_does_not_cancel_owner_publication(tmp_path, monkeypatch):
    compiler = _compiler(tmp_path, 0, "profile")
    path = Path(compiler.cache_dir, f"{KEY}.owner_0.fingerprint")
    path.parent.mkdir(parents=True)
    retry, release = threading.Event(), threading.Event()
    errors = []

    def sleep(seconds):
        retry.set()
        assert release.wait(2)

    def publish():
        try:
            with compiler._publish_peer_fingerprint(KEY, "0" * 64, 0):
                assert path.read_text() == "0" * 64
        except Exception as error:
            errors.append(error)

    monkeypatch.setattr(
        tpu_compiler, "time", SimpleNamespace(monotonic=time.monotonic, sleep=sleep)
    )
    with open(f"{path}.lock", "a") as reader:
        fcntl.flock(reader, fcntl.LOCK_SH)
        thread = threading.Thread(target=publish)
        thread.start()
        try:
            assert retry.wait(2)
        finally:
            fcntl.flock(reader, fcntl.LOCK_UN)
            release.set()
            thread.join(2)
    assert not thread.is_alive()
    assert not errors
    assert path.read_text() == "0" * 64


def test_publisher_guard_wait_is_bounded(tmp_path, monkeypatch):
    compiler = _compiler(tmp_path, 0, "profile")
    path = Path(compiler.cache_dir, f"{KEY}.owner_0.fingerprint")
    path.parent.mkdir(parents=True)
    clock = _Clock()
    monkeypatch.setattr(tpu_compiler, "time", clock)
    monkeypatch.setattr(tpu_compiler, "_PEER_POLL_S", 1)
    monkeypatch.setattr(vllm_envs, "VLLM_ENGINE_READY_TIMEOUT_S", 1)
    with open(f"{path}.lock", "a") as reader:
        fcntl.flock(reader, fcntl.LOCK_SH)
        with compiler._publish_peer_fingerprint(KEY, "0" * 64, 0):
            assert not path.exists()
    assert clock.elapsed == 1
    assert not list(path.parent.glob("*.tmp"))
