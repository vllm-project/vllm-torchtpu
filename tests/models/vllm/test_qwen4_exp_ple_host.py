# SPDX-License-Identifier: Apache-2.0
"""PLE shared storage must not expose partial weights or orphan a large file."""

import fcntl
import importlib.util
import multiprocessing
import os
from pathlib import Path

import pytest


def backend_module():
    path = (
        Path(__file__).parents[3] / "src/vllm_torchtpu/models/vllm/"
        "qwen4_exp/ple_host.py"
    )
    assert path.exists(), "PLE host storage backend is not implemented"
    spec = importlib.util.spec_from_file_location("ple_host_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def backend():
    return backend_module().SharedPLETable


def write(table, offset, value):
    with table.writable_buffer() as buffer:
        buffer[offset : offset + len(value)] = value


def read(table, offset, size):
    with table.buffer() as buffer:
        return bytes(buffer[offset : offset + size])


def writer(directory, connection, *, die=False):
    table = backend()(
        directory=directory,
        key="service-a/ple/tp0",
        identity={"checkpoint": "fixture", "dtype": "int16"},
        nbytes=8192,
        participants=2,
        timeout=10,
    )
    try:
        assert table.is_writer
        write(table, 0, b"original")
        connection.send((os.getpid(), table.data_fd))
        assert connection.recv() == "continue-loading"
        if die:
            os._exit(7)
        write(table, 8, b"-weights")
        table.finish()
        connection.send("ready")
        assert connection.recv() == "exit"
    finally:
        table.close()


def create(directory, **overrides):
    options = dict(
        directory=directory,
        key="service-a/ple/tp0",
        identity={"checkpoint": "fixture", "dtype": "int16"},
        nbytes=8192,
        participants=2,
        timeout=10,
    )
    options.update(overrides)
    return backend()(**options)


def test_publish_seals_table_and_releases_control_files(tmp_path):
    with create(tmp_path, participants=1) as table:
        write(table, 0, b"checkpoint")
        with pytest.raises(RuntimeError, match="ready"):
            read(table, 0, 10)
        table.finish()
        assert read(table, 0, 10) == b"checkpoint"
        assert fcntl.fcntl(table.data_fd, fcntl.F_GET_SEALS) & fcntl.F_SEAL_WRITE
        with pytest.raises(RuntimeError, match="read.only|published"):
            write(table, 0, b"changed")
        assert not list(tmp_path.iterdir())


def test_reader_attaches_before_writer_finishes_remaining_weights(tmp_path):
    # Blocking in attach would deadlock a model loader's later collectives.
    backend()
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=writer, args=(str(tmp_path), child))
    process.start()
    table = None
    try:
        assert parent.poll(30)
        writer_pid, writer_fd = parent.recv()
        table = create(tmp_path)
        assert not table.is_writer
        assert (
            os.fstat(table.data_fd).st_ino
            == os.stat(f"/proc/{writer_pid}/fd/{writer_fd}").st_ino
        )
        with pytest.raises(RuntimeError, match="ready"):
            read(table, 0, 16)
        parent.send("continue-loading")
        table.finish()
        assert parent.poll(30) and parent.recv() == "ready"
        assert read(table, 0, 16) == b"original-weights"
        parent.send("exit")
        process.join(10)
        assert process.exitcode == 0
        assert read(table, 0, 16) == b"original-weights"
        assert not list(tmp_path.iterdir())
    finally:
        if process.is_alive():
            process.kill()
        process.join(10)
        if table is not None:
            table.close()


def test_writer_death_fails_wait_and_has_no_named_weight_file(tmp_path):
    backend()
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(
        target=writer, args=(str(tmp_path), child), kwargs={"die": True}
    )
    process.start()
    table = None
    try:
        assert parent.poll(30)
        parent.recv()
        table = create(tmp_path)
        parent.send("continue-loading")
        process.join(10)
        assert process.exitcode == 7
        with pytest.raises(RuntimeError, match="writer.*(exit|died)"):
            table.finish()
        assert all(p.stat().st_size < 8192 for p in tmp_path.iterdir())
    finally:
        if process.is_alive():
            process.kill()
        process.join(10)
        if table is not None:
            table.close()
    with pytest.raises(RuntimeError, match="writer.*(exit|died)"):
        create(tmp_path)


def test_identity_mismatch_cannot_join_existing_table(tmp_path):
    with create(tmp_path) as original:
        with pytest.raises(ValueError, match="identity"):
            create(tmp_path, identity={"checkpoint": "other", "dtype": "int16"})
        original.abort("fixture shutdown")
    with pytest.raises(RuntimeError, match="fixture shutdown"):
        create(tmp_path)


def test_services_do_not_share_storage(tmp_path):
    with (
        create(tmp_path, participants=1) as first,
        create(tmp_path, key="service-b/ple/tp0", participants=1) as second,
    ):
        write(first, 0, b"first")
        write(second, 0, b"other")
        first.finish()
        second.finish()
        assert os.fstat(first.data_fd).st_ino != os.fstat(second.data_fd).st_ino
        assert read(first, 0, 5) == b"first"
        assert read(second, 0, 5) == b"other"


def test_missing_participant_has_bounded_wait(tmp_path):
    with create(tmp_path, timeout=0.05) as table:
        write(table, 0, b"loaded")
        with pytest.raises(TimeoutError, match="participants"):
            table.finish()
    # Retain a small failed-generation marker so a late worker cannot create
    # another partial table under the same service identity.
    with pytest.raises(RuntimeError, match="participants"):
        create(tmp_path)
    assert all(p.stat().st_size < 8192 for p in tmp_path.iterdir())


def failing_reader(directory, connection, failure):
    with create(directory) as table:
        connection.send("attached")
        assert connection.recv() == "continue"
        if failure == "death":
            os._exit(7)
        try:
            table.finish(metadata={"scale_bits": 1})
        except ValueError as exc:
            connection.send(str(exc))
        else:
            raise AssertionError("Mismatched checkpoint scale was published")


@pytest.mark.parametrize("failure", ["death", "scale"])
def test_reader_failure_cannot_publish_shared_checkpoint(tmp_path, failure):
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    with create(tmp_path) as table:
        write(table, 0, b"weights")
        process = ctx.Process(
            target=failing_reader, args=(str(tmp_path), child, failure)
        )
        process.start()
        try:
            assert parent.poll(30) and parent.recv() == "attached"
            parent.send("continue")
            message = (
                "participant.*exited" if failure == "death" else "metadata mismatch"
            )
            with pytest.raises(RuntimeError, match=message):
                table.finish(metadata={"scale_bits": 2})
            with pytest.raises(RuntimeError, match="not ready"):
                read(table, 0, 7)
            if failure == "scale":
                assert parent.poll(30)
                assert "metadata mismatch" in parent.recv()
            process.join(10)
            assert process.exitcode == (7 if failure == "death" else 0)
            with pytest.raises(RuntimeError, match=message):
                create(tmp_path)
        finally:
            if process.is_alive():
                process.kill()
            process.join(10)


@pytest.mark.parametrize("error", [FileNotFoundError, ProcessLookupError])
def test_process_identity_disappeared_during_proc_read(monkeypatch, error):
    module = backend_module()

    def disappeared(path):
        raise error("process disappeared")

    monkeypatch.setattr(Path, "read_text", disappeared)
    assert module._process_identity(123) is None
    with pytest.raises(RuntimeError, match="writer exited"):
        module.SharedPLETable._check_writer({"writer": {"pid": 123, "start": "10"}})


@pytest.mark.parametrize("error", [PermissionError, OSError])
def test_process_identity_does_not_hide_other_read_errors(monkeypatch, error):
    module = backend_module()

    def failed(path):
        raise error("not a disappearance")

    monkeypatch.setattr(Path, "read_text", failed)
    with pytest.raises(error, match="not a disappearance"):
        module._process_identity(123)


@pytest.mark.parametrize(
    "state,start", [("S", "10"), ("Z", "10"), ("X", "10"), ("S", "11")]
)
def test_process_identity_checks_liveness_and_pid_reuse(monkeypatch, state, start):
    module = backend_module()
    fields = [state, *(["0"] * 18), start]
    monkeypatch.setattr(
        Path, "read_text", lambda path: "123 (name with ) spaces) " + " ".join(fields)
    )
    expected = start if state == "S" else None
    assert module._process_identity(123) == expected
    record = {"writer": {"pid": 123, "start": "10"}}
    if expected == "10":
        module.SharedPLETable._check_writer(record)
    else:
        with pytest.raises(RuntimeError, match="writer exited"):
            module.SharedPLETable._check_writer(record)
