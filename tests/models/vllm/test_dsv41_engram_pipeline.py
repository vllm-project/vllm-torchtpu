# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of DMA lifetimes and prompt prefetch without a TPU runtime.

Execute the production methods with a delayed transport. Completing a transfer
reads the host memory at that point, so premature reuse corrupts the result.
Device/compiled coverage lives in the Engram upload probe.
"""

import ast
import ctypes
import importlib.util
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "src/vllm_torchtpu/models/vllm/deepseek_v41"


def load_nodes(path, names, namespace):
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(body) == len(names)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return types.SimpleNamespace(**namespace)


@pytest.fixture
def transport(monkeypatch):
    events = []

    class HostBuffer:
        def __init__(self, size):
            self.size_bytes = size
            self.memory = (ctypes.c_uint8 * size)()
            self.data_ptr = ctypes.addressof(self.memory)

    class Transfer:
        def __init__(self, device, host, unsafe_skip_buffer_lock):
            self.device = device
            self.host = torch.frombuffer(host.memory, dtype=torch.uint8)

        def d2h(self):
            self.host[: self.device.numel() * 4].copy_(self.device.view(torch.uint8))

        def h2d_async(self):
            events.append("upload-start")

            def wait():
                events.append("upload-done")
                self.device.copy_(
                    self.host[: self.device.numel() * 4].view(torch.int32)
                )

            return types.SimpleNamespace(wait=wait)

    def synchronize(tensors, wait):
        assert wait is True
        events.append(("consumer-done", tensors))

    sync = types.SimpleNamespace(synchronize=synchronize)
    monkeypatch.setitem(
        sys.modules, "torch_tpu._internal", types.SimpleNamespace(sync=sync)
    )
    module = load_nodes(
        SOURCE / "common/host_transfer.py",
        {"PreparedEngramUpload"},
        {
            "torch": torch,
            "ctypes": ctypes,
            "Any": Any,
            "raw_transfer": types.SimpleNamespace(
                RawHostBuffer=HostBuffer, PreparedTorchRawTransfer=Transfer
            ),
            "tpu_torch_compile": types.SimpleNamespace(
                get_device_layout_if_materialized=lambda _: ([0], [[4]], 0)
            ),
        },
    )
    return module.PreparedEngramUpload, events


def test_upload_waits_before_reusing_host_and_device(transport):
    Upload, events = transport
    slot = Upload(5, torch.device("cpu"))
    slot.acquire().fill_(17)
    slot.start()
    slot.acquire()  # Must finish DMA before the caller can rewrite host memory.
    assert slot.device.tolist() == [17] * 5
    slot.activate()
    slot.acquire().fill_(29)  # Host preparation can overlap a device reader.
    assert slot.device.tolist() == [17] * 5
    with pytest.raises(RuntimeError, match="Record the Engram consumer"):
        slot.start()
    result = torch.tensor([23])
    slot.record_consumers([result])
    assert events == ["upload-start", "upload-done"]
    slot.start()
    assert events[-2:] == [("consumer-done", [result]), "upload-start"]
    slot.wait()
    assert slot.device.tolist() == [29] * 5


def test_upload_failure_preserves_pending_transfer(transport):
    Upload, _ = transport
    slot = Upload(4, torch.device("cpu"))

    def fail():
        raise RuntimeError("DMA failed")

    pending = types.SimpleNamespace(wait=fail)
    slot.pending = pending
    with pytest.raises(RuntimeError, match="DMA failed"):
        slot.acquire()
    assert slot.pending is pending


@pytest.fixture
def staging(transport, monkeypatch):
    Upload, events = transport
    path = ROOT / "scripts/dsv41_bench/bench_engram_upload.py"
    spec = importlib.util.spec_from_file_location("engram_probe", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    package = types.ModuleType("engram_upload_fixture")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(
        sys.modules,
        "engram_upload_fixture.host_transfer",
        types.SimpleNamespace(PreparedEngramUpload=Upload),
    )
    unpack, dequant = probe.load_consumer(SOURCE / "common/engram.py", torch)
    Staging = probe.load_staging(SOURCE / "common/engram.py", torch, unpack, dequant)
    weight = torch.arange(32 * 32, dtype=torch.int64).remainder(120).to(torch.uint8)
    weight = weight.reshape(32, 32)
    scales = torch.full((32, 1), 127, dtype=torch.uint8)

    def gather(ids, raw, scale):
        torch.index_select(weight, 0, ids.flatten(), out=raw.view(-1, 32))
        torch.index_select(scales, 0, ids.flatten(), out=scale.view(-1, 1))

    layer = Staging()
    layer.embed_tokens = types.SimpleNamespace(
        host_offload=True,
        part_n_hash_cols=2,
        dp_size=1,
        block_size=32,
        weight=weight,
        gather_raw=gather,
    )
    layer._init_staging(4, 32)
    layer.enable_prepared_upload()
    return layer, events, weight, scales, dequant


def test_prefetch_uses_other_slot_and_preserves_current_values(staging):
    layer, events, weight, scales, dequant = staging
    identities = set()
    for step in range(6):
        ids = torch.full((3, 2), step + 1)
        pending = layer.start_prepare_embeddings(ids)
        # start issues DMA without waiting for the current consumer or upload.
        assert events[-1] == "upload-start"
        old = layer._active_raw.clone() if layer._active_upload is not None else None
        layer.finish_prepare_embeddings(pending)
        identities.add(id(layer._active_upload))
        result = layer._ready_rows(3)
        assert torch.equal(result, dequant(weight[ids], scales[ids], 32))
        layer.record_upload_consumers([result])
        if old is not None:
            assert not torch.equal(old, layer._active_raw)
    assert len(identities) == 2
    # One upload per iteration and only the prior consumer of the reused slot.
    assert sum(event == "upload-start" for event in events) == 6
    assert sum(isinstance(event, tuple) for event in events) == 4


def test_bucket_switch_releases_dormant_activation_outputs(staging):
    layer, _, _, _, _ = staging
    layer.prepare_embeddings(torch.ones((3, 2), dtype=torch.int64))
    old = layer._active_upload
    layer.record_upload_consumers([layer._ready_rows(3)])
    assert old.consumers
    layer.start_prepare_embeddings(torch.ones((4, 2), dtype=torch.int64))
    assert not old.consumers


def test_upload_capacity_covers_all_columns_after_dp_exchange(staging):
    layer, _, _, _, _ = staging
    layer.embed_tokens.dp_size = 2
    layer._init_staging(4, 32)
    layer.enable_prepared_upload()
    slot = layer._reserve_upload(3)
    assert slot.host_words.numel() == 3 * (4 * 32 // 4 + 1)


@pytest.fixture
def prompt_inputs():
    module = load_nodes(
        SOURCE / "v2/host_engram.py",
        {"HostEngramInputs", "next_prompt_inputs"},
        {"torch": torch, "dataclass": dataclass, "Callable": Callable},
    )
    current = module.HostEngramInputs(
        input_ids=torch.tensor([10, 11, 30, 31, 0]),
        positions=torch.tensor([0, 1, 2, 3, 0]),
        query_start_loc=torch.tensor([0, 2, 4, 4]),
        lookback_token_ids=torch.tensor([[-1, -1], [29, 28], [-1, -1]]),
        all_prompt=True,
    )
    runner = types.SimpleNamespace(
        input_batch=types.SimpleNamespace(
            num_prompt_tokens=[8, 8],
            token_ids_cpu=torch.tensor([list(range(10, 18)), list(range(28, 36))]),
        )
    )
    return module, current, runner


def test_prefetch_advances_each_prompt_and_lookback_without_mutating_runner(
    prompt_inputs,
):
    module, current, runner = prompt_inputs
    following = module.next_prompt_inputs(runner, 2, current)
    assert following.input_ids.tolist() == [12, 13, 32, 33, 0]
    assert following.positions.tolist() == [2, 3, 4, 5, 0]
    assert following.lookback_token_ids.tolist() == [[11, 10], [31, 30], [-1, -1]]
    assert current.input_ids.tolist() == [10, 11, 30, 31, 0]
    assert following.matches(following.snapshot(5, 3, 4))
    assert not following.matches(current)


@pytest.mark.parametrize(
    "field", ["input_ids", "positions", "query_start_loc", "lookback_token_ids"]
)
def test_prefetch_rejects_changed_actual_inputs(prompt_inputs, field):
    module, current, runner = prompt_inputs
    following = module.next_prompt_inputs(runner, 2, current)
    actual = following.snapshot(5, 3, 4)
    getattr(actual, field).view(-1)[0] += 1
    assert not following.matches(actual)


def test_prefetch_stops_at_short_prompt_tail(prompt_inputs):
    module, current, runner = prompt_inputs
    runner.input_batch.num_prompt_tokens[0] = 3
    assert module.next_prompt_inputs(runner, 2, current) is None
