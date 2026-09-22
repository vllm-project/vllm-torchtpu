# SPDX-License-Identifier: Apache-2.0
"""PLE shared storage must not expose partial weights or orphan a large file."""
import fcntl
import importlib.util
import multiprocessing
import os
from pathlib import Path

import pytest


def backend():
    path = (Path(__file__).parents[3] / "src/vllm_torchtpu/models/vllm/"
            "qwen4_exp/ple_host.py")
    assert path.exists(), "PLE host storage backend is not implemented"
    spec = importlib.util.spec_from_file_location("ple_host_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SharedPLETable


def test_host_consumer_keeps_prepared_graph_input_immutable():
    # External raw rows become a compiled graph argument. In-place collectives
    # on a dtype view of that argument corrupt the mixed TP/DP BF16 path.
    # The consumer must own its mutable reduction buffer, independent of the
    # adapter's input (CPU mocks alone cannot check the device's raw bits).
    from types import SimpleNamespace as NS

    import torch

    from vllm_torchtpu.models.vllm.qwen4_exp.ple import Qwen4ExpPLE

    prepared = torch.full((1, 2, 2560), 11, dtype=torch.int16, device="cpu")
    hidden = torch.zeros((2, 2560), dtype=torch.bfloat16, device="cpu")
    weight = NS(weight=torch.zeros(1, device="cpu"))
    owner = NS(_finalized=True,
               cpu_offload=True,
               kv_cache=(hidden, ),
               ple_embedding=NS(ngram_embedding=NS(
                   decode=lambda raw: raw.bfloat16())),
               parallel=NS(combine=lambda raw, tokens: raw.add_(7)),
               _values=lambda embeddings, query, *args: (embeddings, query),
               _conv=lambda *args: torch.zeros_like(hidden),
               key_proj=weight,
               value_proj=weight,
               norm_key=weight,
               norm_query=weight,
               norm_conv=weight,
               conv1d=NS(weight=torch.zeros((2560, 1, 4), device="cpu")))
    output = Qwen4ExpPLE.forward(
        owner, hidden, torch.zeros(2, dtype=torch.int32, device="cpu"),
        torch.tensor([0, 1, 1], dtype=torch.int32, device="cpu"), None,
        torch.tensor([1, -1], dtype=torch.int32, device="cpu"), None, None,
        prepared)
    assert (prepared == 11).all()
    assert (output[0] == 18).all() and not output[1].any()


def writer(directory, connection, *, die=False):
    table = backend()(directory=directory,
                      key="service-a/ple/tp0",
                      identity={
                          "checkpoint": "fixture",
                          "dtype": "int16"
                      },
                      nbytes=8192,
                      participants=2,
                      timeout=10)
    try:
        assert table.is_writer
        table.write(0, b"original")
        connection.send((os.getpid(), table.data_fd))
        assert connection.recv() == "continue-loading"
        if die:
            os._exit(7)
        table.write(8, b"-weights")
        table.finish()
        connection.send("ready")
        assert connection.recv() == "exit"
    finally:
        table.close()


def create(directory, **overrides):
    options = dict(directory=directory,
                   key="service-a/ple/tp0",
                   identity={
                       "checkpoint": "fixture",
                       "dtype": "int16"
                   },
                   nbytes=8192,
                   participants=2,
                   timeout=10)
    options.update(overrides)
    return backend()(**options)


def test_publish_seals_table_and_releases_control_files(tmp_path):
    with create(tmp_path, participants=1) as table:
        table.write(0, b"checkpoint")
        with pytest.raises(RuntimeError, match="ready"):
            table.read(0, 10)
        table.finish()
        assert table.read(0, 10) == b"checkpoint"
        assert fcntl.fcntl(table.data_fd,
                           fcntl.F_GET_SEALS) & fcntl.F_SEAL_WRITE
        with pytest.raises(RuntimeError, match="read.only|published"):
            table.write(0, b"changed")
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
        assert os.fstat(table.data_fd).st_ino == os.stat(
            f"/proc/{writer_pid}/fd/{writer_fd}").st_ino
        with pytest.raises(RuntimeError, match="ready"):
            table.read(0, 16)
        parent.send("continue-loading")
        table.finish()
        assert parent.poll(30) and parent.recv() == "ready"
        assert table.read(0, 16) == b"original-weights"
        parent.send("exit")
        process.join(10)
        assert process.exitcode == 0
        assert table.read(0, 16) == b"original-weights"
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
    process = ctx.Process(target=writer,
                          args=(str(tmp_path), child),
                          kwargs={"die": True})
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
            create(tmp_path,
                   identity={
                       "checkpoint": "other",
                       "dtype": "int16"
                   })
        original.abort("fixture shutdown")
    with pytest.raises(RuntimeError, match="fixture shutdown"):
        create(tmp_path)


def test_services_do_not_share_storage(tmp_path):
    with create(tmp_path,
                participants=1) as first, create(tmp_path,
                                                 key="service-b/ple/tp0",
                                                 participants=1) as second:
        first.write(0, b"first")
        second.write(0, b"other")
        first.finish()
        second.finish()
        assert os.fstat(first.data_fd).st_ino != os.fstat(
            second.data_fd).st_ino
        assert first.read(0, 5) == b"first"
        assert second.read(0, 5) == b"other"


def test_missing_participant_has_bounded_wait(tmp_path):
    with create(tmp_path, timeout=0.05) as table:
        table.write(0, b"loaded")
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
        table.write(0, b"weights")
        process = ctx.Process(target=failing_reader,
                              args=(str(tmp_path), child, failure))
        process.start()
        try:
            assert parent.poll(30) and parent.recv() == "attached"
            parent.send("continue")
            message = "participant.*exited" if failure == "death" else "metadata mismatch"
            with pytest.raises(RuntimeError, match=message):
                table.finish(metadata={"scale_bits": 2})
            with pytest.raises(RuntimeError, match="not ready"):
                table.read(0, 7)
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


@pytest.mark.parametrize("fp8", [False, True])
def test_shared_host_loader_keeps_only_one_table_and_original_bits(
        tmp_path, fp8):
    import torch

    from vllm_torchtpu.models.vllm.qwen4_exp.ple import HostPLEEmbedding

    carrier = torch.uint8 if fp8 else torch.int16
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    bits = torch.arange(32 * 160, dtype=torch.int32,
                        device="cpu").to(carrier).reshape(32, 160)
    embedding = HostPLEEmbedding(tp_rank=1,
                                 tp_size=2,
                                 total_rows=32,
                                 checkpoint_shards=2,
                                 device="cpu",
                                 shared=dict(
                                     directory=str(tmp_path),
                                     key="service/tp1",
                                     identity={"checkpoint": "fixture"},
                                     participants=1))
    try:
        embedding.load_weights([(f"shard_{i}.weight",
                                 bits[i * 16:(i + 1) * 16].view(dtype))
                                for i in range(2)])
        if fp8:
            embedding.load_weights([("weight_scale",
                                     torch.tensor([0.25],
                                                  dtype=torch.bfloat16,
                                                  device="cpu"))])
        embedding.finalize_weights()
        assert embedding._staging is None
        assert embedding.table.numel() == 0
        ids = torch.tensor([[0, 15, 16, 31] * 4],
                           dtype=torch.int32,
                           device="cpu")
        expected = bits[[0, 15, 16, 31] * 4].clone()
        expected[torch.tensor([True, True, False, False] * 4)] = 0
        assert torch.equal(
            embedding.lookup_host(ids).view(carrier),
            expected.reshape(1, 2560))
        assert not list(tmp_path.iterdir())
    finally:
        embedding.close()


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("fp8", [False, True])
def test_private_host_lookup_keeps_bits_and_checkpoint_ownership(tp, fp8):
    import torch

    from vllm_torchtpu.models.vllm.qwen4_exp import ple

    assert hasattr(ple,
                   "HostPLEEmbedding"), "Host PLE loader is not implemented"
    carrier = torch.uint8 if fp8 else torch.int16
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    bits = torch.arange(32 * 160, dtype=torch.int32,
                        device="cpu").to(carrier).reshape(32, 160)
    weights = bits.view(dtype)
    # Unequal checkpoint partitions cross the runtime owner boundaries.
    shards = [(f"shard_{i}.weight", weights[i * 11:min((i + 1) * 11, 32)])
              for i in range(3)]
    if fp8:
        shards.append(("weight_scale",
                       torch.tensor([0.25], dtype=torch.bfloat16,
                                    device="cpu")))
    ids = torch.tensor(
        [[-1, 0, 1, 3, 4, 7, 8, 11, 15, 16, 23, 24, 28, 30, 31, 32]],
        dtype=torch.int32,
        device="cpu")
    for rank in range(tp):
        embedding = ple.HostPLEEmbedding(tp_rank=rank,
                                         tp_size=tp,
                                         total_rows=32,
                                         checkpoint_shards=3,
                                         device="cpu")
        embedding.load_weights(shards)
        embedding.finalize_weights()
        result = embedding.lookup_host(ids)
        expected = torch.zeros((1, 16, 160), dtype=carrier, device="cpu")
        for column, row in enumerate(ids[0].tolist()):
            if rank * (32 // tp) <= row < (rank + 1) * (32 // tp):
                expected[0, column] = bits[row]
        actual_bits = result.view(carrier).reshape(1, 16, 160)
        assert torch.equal(actual_bits, expected)
        assert result.dtype == (torch.int8 if fp8 else torch.int16)
        assert embedding.table.numel() == 0
        assert embedding.lookup_host(ids[:0]).shape == (0, 2560)
        embedding.close()


def test_host_loader_rejects_missing_shards_and_scale():
    import torch

    from vllm_torchtpu.models.vllm.qwen4_exp import ple

    assert hasattr(ple,
                   "HostPLEEmbedding"), "Host PLE loader is not implemented"
    embedding = ple.HostPLEEmbedding(tp_rank=0,
                                     tp_size=1,
                                     total_rows=8,
                                     checkpoint_shards=2,
                                     device="cpu")
    weights = torch.zeros(4, 160, dtype=torch.float8_e4m3fn, device="cpu")
    embedding.load_weights([("shard_0.weight", weights)])
    with pytest.raises(ValueError, match="Missing PLE checkpoint shards"):
        embedding.finalize_weights()
    embedding.load_weights([("shard_1.weight", weights)])
    with pytest.raises(ValueError, match="Missing PLE global weight_scale"):
        embedding.finalize_weights()
    embedding.close()
