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
"""CPU host lookup uses disjoint bounded tasks without outliving shared bytes."""

import contextlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import vllm.model_executor.parameter as parameter
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbeddingMethod
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_torchtpu.models.vllm.qwen4_exp import ple_storage
from vllm_torchtpu.models.vllm.qwen4_exp.ple_host import SharedPLEOptions


@pytest.fixture
def make_table(monkeypatch, tmp_path):
    # Real constructors/loaders, with only the distributed rank supplied locally.
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 2)
    tables = []

    def create(*, fp8=False, shared=False, participants=1, finalize=True):
        options = (
            SharedPLEOptions(
                tmp_path, f"table-{len(tables)}", {"test": "gather"}, participants
            )
            if shared
            else None
        )
        method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(
            None, "ple", "float8_e4m3fn" if fp8 else None
        )
        with (
            set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))),
            set_default_torch_dtype(torch.bfloat16),
        ):
            table = ple_storage.HostPLETable(
                512,
                160,
                embedding_method=method,
                padding_size=64,
                prefix="ple",
                parallel_group=SimpleNamespace(rank_in_group=1, world_size=2),
                shared=options,
            )
        tables.append(table)
        dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
        checkpoint = (torch.arange(512 * 160).reshape(512, 160) % 17 - 8).to(dtype)
        if finalize:
            for start in range(0, 512, 170):
                table.weight_loader(
                    table.weight, checkpoint[start : start + 170], start
                )
            if fp8:
                table.weight_scale.data.fill_(0.25)
            table.finalize_weights()
        return table, checkpoint

    yield create
    for table in tables:
        table.close()


def expected_bits(checkpoint, ids):
    dtype = torch.uint8 if checkpoint.dtype == torch.float8_e4m3fn else torch.int16
    bits = checkpoint.view(dtype).numpy()
    result = bits[np.clip(ids.numpy(), 0, 511)].copy()
    result[(ids.numpy() < 256) | (ids.numpy() >= 512)] = 0
    return result


@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("shared", [False, True], ids=["owned", "shared"])
@pytest.mark.parametrize("tokens", [0, 1, 8192], ids=["empty", "decode", "prefill"])
def test_host_lookup_exact_bits_and_storage_lifetime(
    make_table, monkeypatch, fp8, shared, tokens
):
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    table, checkpoint = make_table(fp8=fp8, shared=shared)
    ids = (torch.arange(tokens * 16, dtype=torch.int32) % 800 - 100).reshape(tokens, 16)
    expected = expected_bits(checkpoint, ids)
    keys = tuple(table.state_dict())
    shared_table = table._shared_table
    actual = table.lookup_host(ids)
    carrier = torch.uint8 if fp8 else torch.int16
    assert actual.dtype == (torch.int8 if fp8 else torch.int16)
    assert actual.shape == (tokens, 16 * 160)
    np.testing.assert_array_equal(
        actual.view(carrier).numpy().reshape(tokens, 16, 160), expected
    )
    assert tuple(table.state_dict()) == keys
    assert (
        table._gather_pool is None if tokens < 8192 else table._gather_pool is not None
    )
    pool = table._gather_pool
    table.close()
    table.close()
    assert table._gather_pool is None
    assert pool is None or pool._shutdown
    assert shared_table is None or shared_table._closed
    assert not table._finalized and table.weight.numel() == 0
    with pytest.raises(AssertionError):
        table.lookup_host(ids)


def test_large_lookup_dispatches_and_reuses_pool(make_table, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    table, checkpoint = make_table(shared=True)
    ids = (torch.arange(8192 * 16, dtype=torch.int32) % 256 + 256).reshape(8192, 16)
    original_take = np.take
    workers = set()
    spans = []
    lock = threading.Lock()

    def take(source, indices, *, axis, out, mode):
        with lock:
            workers.add(threading.current_thread().name)
            spans.append((out.__array_interface__["data"][0], out.nbytes))
        return original_take(source, indices, axis=axis, out=out, mode=mode)

    monkeypatch.setattr(ple_storage.np, "take", take)
    first = table.lookup_host(ids)
    pool = getattr(table, "_gather_pool", None)
    assert pool is not None, "Large lookup lost its parallel executor"
    assert len(spans) == 8
    ordered = sorted(spans)
    assert all(a + size <= b for (a, size), (b, _) in zip(ordered, ordered[1:]))
    assert sum(size for _, size in spans) == first.numel() * first.element_size()
    assert all(name.startswith("q4exp-ple-gather") for name in workers)
    second = table.lookup_host(ids.flip(0))
    assert table._gather_pool is pool and len(spans) == 16
    np.testing.assert_array_equal(
        first.numpy().reshape(8192, 16, 160), expected_bits(checkpoint, ids)
    )
    np.testing.assert_array_equal(
        second.numpy().reshape(8192, 16, 160), expected_bits(checkpoint, ids.flip(0))
    )


@pytest.mark.parametrize(
    "cpus,participants,expected", [(128, 8, 8), (32, 4, 4), (8, 8, 1), (None, 1, 1)]
)
def test_worker_budget_is_shared_by_dp_participants(
    make_table, monkeypatch, cpus, participants, expected
):
    monkeypatch.setattr(os, "cpu_count", lambda: cpus)
    table, _ = make_table(shared=True, participants=participants, finalize=False)
    assert table._gather_workers == expected
    assert table._gather_pool is None


@pytest.mark.parametrize("failure", ["worker", "submit"])
def test_failed_partition_drains_before_mapping_release(
    make_table, monkeypatch, failure
):
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    table, _ = make_table(shared=True)
    ids = torch.full((2048, 16), 256, dtype=torch.int32)
    started = threading.Event()
    failed = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    exited = threading.Event()
    original_take = np.take
    original_buffer = table._shared_table.buffer
    calls = 0
    lock = threading.Lock()

    def take(source, indices, *, axis, out, mode):
        nonlocal calls
        with lock:
            ordinal = calls
            calls += 1
        if failure == "worker" and ordinal == 0:
            assert started.wait(5)
            failed.set()
            raise RuntimeError("partition failure")
        started.set()
        assert release.wait(5)
        result = original_take(source, indices, axis=axis, out=out, mode=mode)
        finished.set()
        return result

    class FailingSubmitPool(ThreadPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):
            if self._work_queue.qsize() or started.is_set():
                failed.set()
                raise RuntimeError("submit failure")
            future = super().submit(fn, *args, **kwargs)
            assert started.wait(5)
            return future

    @contextlib.contextmanager
    def buffer():
        try:
            with original_buffer() as view:
                yield view
        finally:
            assert finished.is_set(), (
                "Borrowed mapping escaped while a task was running"
            )
            exited.set()

    monkeypatch.setattr(ple_storage.np, "take", take)
    monkeypatch.setattr(table._shared_table, "buffer", buffer)
    if failure == "submit":
        monkeypatch.setattr(ple_storage, "ThreadPoolExecutor", FailingSubmitPool)
    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(table.lookup_host, ids)
        try:
            assert failed.wait(5)
            assert not exited.wait(0.05)
        finally:
            release.set()
        with pytest.raises(
            RuntimeError,
            match=f"{failure if failure == 'submit' else 'partition'} failure",
        ):
            future.result(timeout=5)
    assert finished.is_set() and exited.is_set()
    table.close()
    table.close()


def test_constructor_failure_releases_acquired_mapping(monkeypatch, tmp_path):
    captured = []
    original = ple_storage.SharedPLETable

    def shared_table(**kwargs):
        table = original(**kwargs)
        captured.append(table)
        return table

    monkeypatch.setattr(ple_storage, "SharedPLETable", shared_table)
    method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(None, "ple")
    with (
        set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))),
        set_default_torch_dtype(torch.float32),
        pytest.raises(ValueError, match="BF16 model activations"),
    ):
        # Allocate shared BF16 bytes, then trigger post-allocation dtype validation.
        method.create_weights_original = method.create_weights

        def create_weights(layer, *args, **kwargs):
            kwargs["params_dtype"] = torch.bfloat16
            return method.create_weights_original(layer, *args, **kwargs)

        monkeypatch.setattr(method, "create_weights", create_weights)
        ple_storage.HostPLETable(
            64,
            160,
            embedding_method=method,
            padding_size=64,
            prefix="ple",
            parallel_group=SimpleNamespace(rank_in_group=0, world_size=1),
            shared=SharedPLEOptions(tmp_path, "failed", {}, 1),
        )
    assert len(captured) == 1 and captured[0]._closed
    assert captured[0].data_fd == -1 and captured[0]._control_fd == -1


def test_storage_instances_do_not_share_executors(make_table, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    first, _ = make_table(shared=True)
    second, checkpoint = make_table(shared=True)
    ids = torch.full((8192, 16), 300, dtype=torch.int32)
    first.lookup_host(ids)
    second.lookup_host(ids)
    assert first._gather_pool is not second._gather_pool
    first.close()
    actual = second.lookup_host(ids)
    np.testing.assert_array_equal(
        actual.numpy().reshape(8192, 16, 160), expected_bits(checkpoint, ids)
    )


def test_executor_setup_failure_releases_borrowed_view(make_table, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    table, _ = make_table(shared=True)

    def fail_setup(*args, **kwargs):
        raise RuntimeError("executor setup failure")

    monkeypatch.setattr(ple_storage, "ThreadPoolExecutor", fail_setup)
    ids = torch.full((8192, 16), 256, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="executor setup failure") as captured:
        table.lookup_host(ids)
    assert captured.value.args == ("executor setup failure",)
    assert table._gather_pool is None
    table.close()
    table.close()


def test_serial_failure_releases_borrowed_view(make_table, monkeypatch):
    table, _ = make_table(shared=True)

    def fail_take(source, *args, **kwargs):
        raise RuntimeError("serial gather failure")

    monkeypatch.setattr(ple_storage.np, "take", fail_take)
    with pytest.raises(RuntimeError, match="serial gather failure") as captured:
        table.lookup_host(torch.tensor([[256]], dtype=torch.int32))
    assert captured.value.args == ("serial gather failure",)
    table.close()
