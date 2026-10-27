# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise production DMA lifetimes with a deliberately delayed transport."""

import ast
import ctypes
import importlib.util
import logging
import math
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from itertools import islice
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch.utils._pytree import tree_leaves

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "src/vllm_torchtpu/models/vllm/deepseek_v41"
pytestmark = pytest.mark.cpu_test


def load_nodes(path, names, namespace):
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(body) == len(names)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return types.SimpleNamespace(**namespace)


@pytest.fixture
def transport():
    events = []

    class HostBuffer:
        def __init__(self, size):
            self.size_bytes = size
            self.memory = None

        @property
        def data_ptr(self):
            return ctypes.addressof(self.memory) if self.memory is not None else 0

    class Transfer:
        def __init__(self, device, host, unsafe_skip_buffer_lock):
            assert not unsafe_skip_buffer_lock
            self.device = device
            if host.memory is None:
                host.memory = (ctypes.c_uint8 * host.size_bytes)()
            self.host = torch.frombuffer(host.memory, dtype=torch.uint8)
            events.append(("bind", device, host))

        def d2h(self):
            source = (
                self.device
                if isinstance(self.device, torch.Tensor)
                else self.device.data
            )
            if source.ndim == 3:
                self.host.view(torch.int32).view(source.shape[0], 2, 128)[
                    :, : source.shape[1], : source.shape[2]
                ].copy_(source)
            else:
                self.host[: source.numel() * 4].copy_(source.view(torch.uint8))

        def h2d_async(self):
            events.append("upload-start")

            def wait():
                events.append("upload-done")
                self.device.copy_(
                    self.host[: self.device.numel() * 4].view(torch.int32)
                )

            return types.SimpleNamespace(wait=wait)

    layout = types.SimpleNamespace(
        get_device_layout_if_materialized=lambda tensor: (
            ([0], [[4]], 32) if len(tensor.shape) == 1 else ([2, 1, 0], [[2, 128]], 32)
        )
    )
    sync = types.SimpleNamespace(
        synchronize=lambda tensors, wait: events.append(("sync", wait, tensors))
    )
    module = load_nodes(
        SOURCE / "common/host_transfer.py",
        {
            "_physical_words",
            "_host_words",
            "_hash_layout",
            "PreparedEngramUpload",
            "_DownloadBuffer",
            "EngramHashDownload",
        },
        dict(
            torch=torch,
            ctypes=ctypes,
            math=math,
            Any=Any,
            dataclass=dataclass,
            cached_property=cached_property,
            sync=sync,
            tpu_torch_compile=layout,
            raw_transfer=types.SimpleNamespace(
                RawHostBuffer=HostBuffer, PreparedTorchRawTransfer=Transfer
            ),
        ),
    )
    original_host_words = module._host_words

    def host_words(buffer, num_words):
        assert buffer.data_ptr != 0, "Bind the lazy host buffer before viewing it"
        return original_host_words(buffer, num_words)

    original_host_words.__globals__["_host_words"] = host_words
    return module, events, layout


def test_host_and_device_overwrite_wait_for_different_lifetimes(transport):
    module, events, _ = transport
    slot = module.PreparedEngramUpload(5, torch.device("cpu"))
    events.clear()
    slot.acquire().fill_(17)
    slot.start()
    slot.acquire()
    assert slot.device.tolist() == [17] * 5
    slot.activate()
    slot.acquire().fill_(29)
    assert slot.device.tolist() == [17] * 5
    with pytest.raises(RuntimeError, match="Record the Engram consumer"):
        slot.start()
    result = torch.tensor([23])
    slot.record_consumers([result])
    assert events == ["upload-start", "upload-done"]
    slot.start()
    assert events[-2:] == [("sync", True, [result]), "upload-start"]
    slot.wait()
    assert slot.device.tolist() == [29] * 5


def test_failed_dma_preserves_pending_handle(transport):
    module, _, _ = transport
    slot = module.PreparedEngramUpload(4, torch.device("cpu"))

    def fail():
        raise RuntimeError("DMA failed")

    pending = types.SimpleNamespace(wait=fail)
    slot.pending = pending
    with pytest.raises(RuntimeError, match="DMA failed"):
        slot.acquire()
    assert slot.pending is pending


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_model_records_device_consumers_before_slot_reuse(transport, version):
    module, events, _ = transport
    slot = module.PreparedEngramUpload(4, torch.device("cpu"))
    slot.acquire().fill_(17)
    slot.start()
    slot.activate()
    tree = ast.parse((SOURCE / version / "model.py").read_text())
    cls = next(
        node for node in tree.body if getattr(node, "name", None) == "DeepseekV41Model"
    )
    method = next(
        node
        for node in cls.body
        if getattr(node, "name", None) == "_record_engram_consumers"
    )
    engram_type = type("Engram", (), {})
    layer = engram_type()
    layer._active_upload = slot
    owner = types.SimpleNamespace(
        _raw_dma_enabled=True, layers=[types.SimpleNamespace(engram=layer)]
    )
    namespace = dict(
        torch=torch,
        islice=islice,
        tree_leaves=tree_leaves,
        Engram=engram_type,
        upstream=types.SimpleNamespace(
            IntermediateTensors=type("Intermediate", (), {})
        ),
        synchronize_tensors=lambda tensors, wait: events.append(("dispatch", tensors)),
    )
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "<consumer>", "exec"),
        namespace,
    )
    output = torch.tensor([23])
    namespace["_record_engram_consumers"](owner, 0, 1, {"output": output})
    assert not slot.needs_consumer
    assert slot.consumers == [output]
    slot.acquire().fill_(29)
    slot.start()
    assert events[-2:] == [("sync", True, [output]), "upload-start"]


@pytest.mark.parametrize(
    "value",
    [None, ([1, 0], [], 32), ([0], [[2, 2]], 32), ([0], [[0]], 32), ([0], [], 16)],
)
def test_raw_layout_rejects_unverified_storage(transport, value):
    module, _, layout = transport
    layout.get_device_layout_if_materialized = lambda _: value
    with pytest.raises((RuntimeError, NotImplementedError)):
        module._physical_words(torch.zeros(5, dtype=torch.int32))


def test_download_reuses_host_storage_but_rebinds_each_device_producer(transport):
    module, events, _ = transport

    class DeviceTensor:
        device = torch.device("xla:0")
        dtype = torch.int32

        def __init__(self, values):
            self.data = values
            self.shape = values.shape
            self.ndim = values.ndim

        def numel(self):
            return self.data.numel()

        def reshape(self, *shape):
            return DeviceTensor(self.data.reshape(*shape))

        def to(self, device):
            assert device == "cpu"
            return self.data.clone()

    download = module.EngramHashDownload()
    outputs = []
    for step in range(6):
        expected = torch.arange(48, dtype=torch.int32).view(3, 2, 8) + step * 100
        actual = download.copy_to_host(DeviceTensor(expected))
        assert torch.equal(actual, expected)
        outputs.append(actual.data_ptr())
    bindings = [event for event in events if event[0] == "bind"]
    assert len(bindings) == 6
    assert len({id(event[1]) for event in bindings}) == 6
    assert len({id(event[2]) for event in bindings}) == 1
    assert len(set(outputs)) == 1
    assert bindings[0][2].size_bytes == 3 * 2 * 128 * 4


@pytest.mark.parametrize("tokens", [1, 128, 288])
@pytest.mark.parametrize("order,tile", [([2, 1, 0], [2, 128]), ([0, 2, 1], [8, 128])])
def test_hash_layout_decodes_padded_physical_tiles(transport, tokens, order, tile):
    module, _, _ = transport
    shape = (tokens, 2, 8)
    physical, permutation, padded = module._hash_layout(shape, (order, [tile], 0))
    words = torch.full(physical, -1, dtype=torch.int32)
    expected = torch.arange(math.prod(shape)).view(shape)
    major = list(reversed(order))
    tiled = list(reversed(order[:2]))
    for token in range(tokens):
        for layer in range(2):
            for head in range(8):
                logical = [token, layer, head]
                index = [logical[dim] for dim in major]
                for dim, size in zip(tiled, tile):
                    index[major.index(dim)] //= size
                index += [logical[dim] % size for dim, size in zip(tiled, tile)]
                words[tuple(index)] = expected[token, layer, head]
    actual = words.permute(permutation).reshape(padded)[:tokens, :2, :8]
    assert torch.equal(actual, expected)


@pytest.fixture
def staging(transport, monkeypatch):
    module, events, _ = transport
    package = types.ModuleType("engram_transfer_fixture")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, package.__name__ + ".host_transfer", module)
    path = SOURCE / "common/engram.py"
    tree = ast.parse(path.read_text())
    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Engram"
    )
    methods = {
        "_init_staging",
        "_reserve_upload",
        "start_prepared_embeddings",
        "activate_prepared_embeddings",
        "_staged_views",
        "_ready_rows",
        "prepare_embeddings",
    }
    cls = ast.ClassDef(
        name="Staging",
        bases=[],
        keywords=[],
        decorator_list=[],
        body=[
            node
            for node in original.body
            if isinstance(node, ast.FunctionDef) and node.name in methods
        ],
    )
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_unpack_bytes", "dequant_fp8_rows"}
    ]
    namespace = dict(
        __package__=package.__name__,
        torch=torch,
        cast=cast,
        PreparedEngramUpload=module.PreparedEngramUpload,
        _INVALID_SCALE=255,
        get_current_vllm_config_or_none=lambda: None,
        logger=logging.getLogger(__name__),
    )
    adapter = ROOT / "src/vllm_torchtpu/layers/adapter/custom_ops/packed_suffix_op.py"
    load_nodes(adapter, {"packed_suffix"}, namespace)
    code = ast.fix_missing_locations(ast.Module(body=helpers + [cls], type_ignores=[]))
    exec(compile(code, str(path), "exec"), namespace)
    weight = torch.arange(64 * 32).remainder(120).to(torch.uint8).view(64, 32)
    scales = torch.arange(64).remainder(7).add(124).to(torch.uint8).view(64, 1)
    spec = importlib.util.spec_from_file_location(
        "vllm_torchtpu.models.vllm.deepseek_v41.common.host_gather",
        SOURCE / "common/host_gather.py",
    )
    host_gather = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, host_gather)
    spec.loader.exec_module(host_gather)

    def gather(ids, raw, scale):
        host_gather.gather_weight_scale(
            weight, scales, ids, raw.view(-1, 32), scale.view(-1, 1)
        )

    layer = namespace["Staging"]()
    layer.layer_hash_index = 0
    layer.embed_tokens = types.SimpleNamespace(
        host_offload=True,
        part_n_hash_cols=2,
        dp_size=1,
        block_size=32,
        gather_raw=gather,
    )
    layer._init_staging(4, 32)
    return layer, events, weight, scales, namespace["dequant_fp8_rows"]


def test_prefetch_alternates_slots_preserving_current_chunk(staging):
    layer, events, weight, scales, dequant = staging
    slots = set()
    for step in range(6):
        ids = torch.full((3, 2), step + 1)
        old = layer._active_raw.clone() if layer._active_upload is not None else None
        upload = layer.start_prepared_embeddings(ids)
        assert events[-1] == "upload-start"
        if old is not None:
            assert torch.equal(old, layer._active_raw)
        layer.activate_prepared_embeddings(3, upload)
        result = layer._ready_rows(3)
        assert torch.equal(result, dequant(weight[ids], scales[ids], 32))
        upload.record_consumers([result])
        slots.add(id(upload))
    assert len(slots) == 2
    assert events.count("upload-start") == 6
    assert (
        sum(
            isinstance(event, tuple) and event[:2] == ("sync", True) for event in events
        )
        == 4
    )


def test_direct_host_staging_uses_dma_and_handles_empty_tokens(staging):
    layer, events, weight, scales, dequant = staging
    for step in range(6):
        ids = torch.full((3, 2), step + 1)
        layer.prepare_embeddings(ids)
        result = layer._ready_rows(3)
        assert torch.equal(result, dequant(weight[ids], scales[ids], 32))
        layer._active_upload.record_consumers([result])
    assert events.count("upload-start") == 6
    layer.prepare_embeddings(torch.empty((0, 2), dtype=torch.int32))
    assert layer._active_raw.shape == (0, 17)
    assert layer._active_upload is None
    assert events.count("upload-start") == 6


def test_bucket_switch_and_prefetch_miss_do_not_reuse_pending_host_data(staging):
    layer, _, weight, scales, dequant = staging
    old = layer.start_prepared_embeddings(torch.ones((3, 2), dtype=torch.int32))
    layer.activate_prepared_embeddings(3, old)
    old.record_consumers([layer._ready_rows(3)])
    missed = layer.start_prepared_embeddings(torch.full((3, 2), 2, dtype=torch.int32))
    replacement = layer.start_prepared_embeddings(
        torch.full((3, 2), 3, dtype=torch.int32)
    )
    assert replacement is missed
    layer.activate_prepared_embeddings(3, replacement)
    expected = dequant(weight[torch.full((3, 2), 3)], scales[torch.full((3, 2), 3)], 32)
    assert torch.equal(layer._ready_rows(3), expected)
    replacement.record_consumers([layer._ready_rows(3)])
    layer.start_prepared_embeddings(torch.ones((4, 2), dtype=torch.int32))
    assert not old.consumers and not replacement.consumers


@pytest.fixture
def prompt_inputs():
    module = load_nodes(
        SOURCE / "v2/host_engram.py",
        {"HostEngramInputs", "next_prompt_inputs"},
        dict(torch=torch, dataclass=dataclass, Callable=Callable),
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


def test_next_prompt_tracks_positions_and_lookback_without_mutation(prompt_inputs):
    module, current, runner = prompt_inputs
    following = module.next_prompt_inputs(runner, 2, current)
    assert following.input_ids.tolist() == [12, 13, 32, 33, 0]
    assert following.positions.tolist() == [2, 3, 4, 5, 0]
    assert following.lookback_token_ids.tolist() == [[11, 10], [31, 30], [-1, -1]]
    assert current.input_ids.tolist() == [10, 11, 30, 31, 0]
    assert following.matches(following.snapshot(5, 3, 4))


@pytest.mark.parametrize(
    "field", ["input_ids", "positions", "query_start_loc", "lookback_token_ids"]
)
def test_prediction_must_match_every_actual_hash_input(prompt_inputs, field):
    module, current, runner = prompt_inputs
    following = module.next_prompt_inputs(runner, 2, current)
    actual = following.snapshot(5, 3, 4)
    getattr(actual, field).view(-1)[0] += 1
    assert not following.matches(actual)


def test_prediction_excludes_decode_and_short_tail(prompt_inputs):
    module, current, runner = prompt_inputs
    runner.input_batch.num_prompt_tokens[0] = 3
    assert module.next_prompt_inputs(runner, 2, current) is None
    decode = module.HostEngramInputs(
        current.input_ids,
        current.positions,
        current.query_start_loc,
        current.lookback_token_ids,
        False,
        True,
    )
    assert module.next_prompt_inputs(runner, 2, decode) is None
    assert not current.matches(decode)
