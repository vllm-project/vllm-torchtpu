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
"""TPULoRAModelManager writes each adapter through a host buffer. These tests
guard that every slot still ends up exactly as vLLM's own manager writes it,
and, on the chip, that the write compiles once, holds at most one extra slot,
and leaves the stacked tensors' device layout alone."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch._dynamo.utils
import vllm.lora.layers
from torch import nn
from torch_tpu._internal import execution_mode
from vllm.config.lora import LoRAConfig
from vllm.lora.layers import (
    BaseLayerWithLoRA,
    ClassificationHeadWithLoRA,
    ColumnParallelLinearWithLoRA,
    ColumnParallelLinearWithShardedLoRA,
    FusedMoE3DWithLoRA,
    FusedMoEWithLoRA,
    LogitsProcessorWithLoRA,
    MergedColumnParallelLinearVariableSliceWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithLoRA,
    MergedQKVParallelLinearWithShardedLoRA,
    QKVParallelLinearWithLoRA,
    QKVParallelLinearWithShardedLoRA,
    ReplicatedLinearWithLoRA,
    RowParallelLinearWithLoRA,
    RowParallelLinearWithShardedLoRA,
    VocabParallelEmbeddingWithLoRA,
)
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.model_manager import AdapterLRUCache, LRUCacheLoRAModelManager
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

from tests.distributed.tpu_test_utils import run_in_isolated_process
from vllm_torchtpu.layers.adapter.custom_ops import lora_slot_update_op
from vllm_torchtpu.lora import lora_manager
from vllm_torchtpu.lora.lora_manager import (
    TPULoRAModelManager,
    TPUWorkerLoRAManager,
    _AdapterSlotWriter,
)

NUM_SLOTS, MAX_RANK, MAX_CPU_LORAS = 3, 8, 8
HIDDEN, Q, KV, INTERMEDIATE, VOCAB = 16, 32, 8, 24, 40
CPU = torch.device("cpu")
# The tpu device type exists only once torch_tpu loads in full, which CPU-only
# runs skip.
TPU = "tpu"


@pytest.fixture(autouse=True)
def _seed():
    # The CPU generator only: torch.manual_seed also seeds the TPU, which opens
    # the chip in this process and locks the spawned TPU cases out of it.
    torch.default_generator.manual_seed(0)


@pytest.fixture(autouse=True)
def slot_writes(monkeypatch):
    """Runs the slot write's JAX function on the host in place of the TPU
    program, and records the slot each write targets."""
    slots = []

    def write_adapter_to_slot(stacked_weights, slot_weights, slot_index):
        slots.append(int(slot_index[0]))
        updated = lora_slot_update_op._compute_slot_update(
            [jnp.asarray(weights.numpy()) for weights in stacked_weights],
            jnp.asarray(slot_weights.numpy()),
            jnp.asarray(slot_index.numpy()),
        )
        for weights, new_weights in zip(stacked_weights, updated):
            weights.copy_(torch.from_numpy(np.array(new_weights)))

    monkeypatch.setattr(lora_manager, "write_adapter_to_slot", write_adapter_to_slot)
    return slots


def _bare_layer(cls):
    """A vLLM LoRA layer without a base layer: reset_lora and set_lora only
    need the stacked tensors, so no model or distributed setup is built."""
    layer = cls.__new__(cls)
    nn.Module.__init__(layer)
    return layer


def _linear_layer(cls, lora_a_shapes, lora_b_shapes, device=CPU, dtype=torch.float32):
    def zeros(shape):
        return torch.zeros(NUM_SLOTS, *shape, device=device, dtype=dtype)

    layer = _bare_layer(cls)
    layer.lora_a_stacked = tuple(zeros(shape) for shape in lora_a_shapes)
    layer.lora_b_stacked = tuple(zeros(shape) for shape in lora_b_shapes)
    layer.n_slices = len(lora_a_shapes)
    layer.output_slices = tuple(shape[1] for shape in lora_b_shapes)
    layer.tp_size = 1
    return layer


def _vocab_layer(cls, lora_a_shape, lora_b_shape, device=CPU, dtype=torch.float32):
    layer = _bare_layer(cls)
    layer.lora_a_stacked = torch.zeros(
        NUM_SLOTS, *lora_a_shape, device=device, dtype=dtype
    )
    layer.lora_b_stacked = torch.zeros(
        NUM_SLOTS, *lora_b_shape, device=device, dtype=dtype
    )
    return layer


def _variable_slice_layer(device, dtype):
    """A merged layer of three slices with different widths, as Mamba2's
    in_proj. Its set_lora splits one whole adapter weight by the base layer's
    output_sizes."""
    output_sizes = (INTERMEDIATE, KV, Q)
    layer = _linear_layer(
        MergedColumnParallelLinearVariableSliceWithLoRA,
        [(1, MAX_RANK, HIDDEN)] * len(output_sizes),
        [(1, size, MAX_RANK) for size in output_sizes],
        device,
        dtype,
    )
    layer.base_layer = SimpleNamespace(output_sizes=output_sizes)
    return layer


def _layers(device=CPU, dtype=torch.float32) -> dict[str, BaseLayerWithLoRA]:
    """One layer of every supported type, sized like a tiny decoder."""
    from_hidden = (1, MAX_RANK, HIDDEN)
    embed = _vocab_layer(
        VocabParallelEmbeddingWithLoRA,
        (VOCAB, MAX_RANK),
        (1, HIDDEN, MAX_RANK),
        device,
        dtype,
    )
    # The view the embedding's forward reads, as create_lora_weights makes it.
    embed.lora_a_stacked_2d = embed.lora_a_stacked.view(NUM_SLOTS * VOCAB, MAX_RANK)
    return {
        "column": _linear_layer(
            ColumnParallelLinearWithLoRA,
            [from_hidden],
            [(1, Q, MAX_RANK)],
            device,
            dtype,
        ),
        "gate_up": _linear_layer(
            MergedColumnParallelLinearWithLoRA,
            [from_hidden] * 2,
            [(1, INTERMEDIATE, MAX_RANK)] * 2,
            device,
            dtype,
        ),
        "qkv": _linear_layer(
            QKVParallelLinearWithLoRA,
            [from_hidden],
            [(1, Q + 2 * KV, MAX_RANK)],
            device,
            dtype,
        ),
        "merged_qkv": _linear_layer(
            MergedQKVParallelLinearWithLoRA,
            [from_hidden] * 3,
            [(1, Q, MAX_RANK), (1, KV, MAX_RANK), (1, KV, MAX_RANK)],
            device,
            dtype,
        ),
        "in_proj": _variable_slice_layer(device, dtype),
        "row": _linear_layer(
            RowParallelLinearWithLoRA,
            [(1, MAX_RANK, Q)],
            [(1, HIDDEN, MAX_RANK)],
            device,
            dtype,
        ),
        "replicated": _linear_layer(
            ReplicatedLinearWithLoRA,
            [from_hidden],
            [(1, HIDDEN, MAX_RANK)],
            device,
            dtype,
        ),
        "embed": embed,
        "logits": _vocab_layer(
            LogitsProcessorWithLoRA,
            (1, MAX_RANK, HIDDEN),
            (1, VOCAB, MAX_RANK),
            device,
            dtype,
        ),
    }


def _slices(stacked):
    return stacked if isinstance(stacked, tuple) else (stacked,)


def _adapter(lora_id, layers, rank=MAX_RANK, skip=(), drop_second_slice=False):
    """A LoRAModel with random weights, in the layers' dtype, for every layer
    not in skip."""
    loras: dict[str, LoRALayerWeights] = {}
    for name, layer in layers.items():
        if name in skip:
            continue
        stacked_a = _slices(layer.lora_a_stacked)
        stacked_b = _slices(layer.lora_b_stacked)
        dtype = stacked_a[0].dtype
        if isinstance(layer, VocabParallelEmbeddingWithLoRA):
            # The embedding's lora_a comes as [rank, vocab]; set_lora
            # transposes it.
            in_sizes = [stacked_a[0].shape[1]]
        else:
            in_sizes = [stacked.shape[-1] for stacked in stacked_a]
        out_sizes = [stacked.shape[2] for stacked in stacked_b]
        if len(in_sizes) == 1 or isinstance(
            layer, MergedColumnParallelLinearVariableSliceWithLoRA
        ):
            # One weight for the whole module, as a checkpoint saves a layer
            # it does not know is split; set_lora splits it by output_sizes.
            lora_a = torch.randn(rank, in_sizes[0], dtype=dtype)
            lora_b = torch.randn(sum(out_sizes), rank, dtype=dtype)
            loras[name] = LoRALayerWeights(name, rank, rank, lora_a, lora_b)
            continue
        slices_a: list[torch.Tensor | None] = [
            torch.randn(rank, size, dtype=dtype) for size in in_sizes
        ]
        slices_b: list[torch.Tensor | None] = [
            torch.randn(size, rank, dtype=dtype) for size in out_sizes
        ]
        if drop_second_slice:
            slices_a[1] = slices_b[1] = None
        loras[name] = PackedLoRALayerWeights(
            name, rank, [rank] * len(in_sizes), slices_a, slices_b
        )
    return LoRAModel(lora_id, rank, loras)


def _manager(cls, layers):
    """A LoRA model manager over hand-built layers. LoRAModelManager.__init__
    needs a whole model and distributed setup, so this sets only the state
    activate_adapter uses."""
    manager = cls.__new__(cls)
    manager.lora_config = LoRAConfig(
        max_lora_rank=MAX_RANK, max_loras=NUM_SLOTS, max_cpu_loras=MAX_CPU_LORAS
    )
    manager.modules = layers
    manager.is_pooling_model = False
    manager.lora_index_to_id = [None] * NUM_SLOTS
    manager._registered_adapters = AdapterLRUCache(
        MAX_CPU_LORAS, manager.deactivate_adapter
    )
    manager._active_adapters = AdapterLRUCache(NUM_SLOTS, manager._deactivate_adapter)
    return manager


def _managers(make_layers=_layers, device=CPU):
    """TPULoRAModelManager and vLLM's LRUCacheLoRAModelManager, each over its
    own zeroed layers from make_layers."""
    vllm_manager = _manager(LRUCacheLoRAModelManager, make_layers())
    # Built again rather than copied: a copy of a TPU tensor shares its device
    # buffer until written, and the slot write donates that buffer.
    tpu_layers = make_layers()
    tpu_manager = _manager(TPULoRAModelManager, tpu_layers)
    tpu_manager._slot_writer = _AdapterSlotWriter.create(
        list(tpu_layers.values()), NUM_SLOTS, device
    )
    return tpu_manager, vllm_manager


def _activate(managers, adapter):
    for manager in managers:
        manager._registered_adapters[adapter.id] = adapter
        manager.activate_adapter(adapter.id)


def _stacked(layer):
    return [
        weights
        for stacked in (layer.lora_a_stacked, layer.lora_b_stacked)
        for weights in _slices(stacked)
    ]


def _assert_same_slots(tpu_manager, vllm_manager):
    assert tpu_manager.lora_index_to_id == vllm_manager.lora_index_to_id
    for name, layer in vllm_manager.modules.items():
        tpu_layer = tpu_manager.modules[name]
        for expected, actual in zip(_stacked(layer), _stacked(tpu_layer)):
            assert torch.equal(actual.cpu(), expected.cpu()), name


def test_host_redirected_writes_match_vllm_bit_for_bit():
    """Covers a rank below the max, an empty slice, the embedding's transposed
    lora_a, and layers an adapter leaves out."""
    tpu_manager, vllm_manager = _managers()
    layers = vllm_manager.modules
    for adapter in (
        _adapter(1, layers),
        _adapter(2, layers, rank=4, drop_second_slice=True),
        _adapter(3, layers, skip=("embed", "logits")),
    ):
        _activate((tpu_manager, vllm_manager), adapter)
        _assert_same_slots(tpu_manager, vllm_manager)


def test_every_slot_region_is_rewritten():
    """The host buffer is reused for every adapter, so nothing the previous
    adapter left there may reach the next slot."""
    tpu_manager, _ = _managers()
    writer = tpu_manager._slot_writer
    assert writer is not None
    layers = tpu_manager.modules
    for adapter in (
        _adapter(1, layers),
        _adapter(2, layers, rank=4, drop_second_slice=True, skip=("embed",)),
    ):
        writer._host_slot_weights.fill_(float("nan"))
        _activate((tpu_manager,), adapter)
        assert not writer._host_slot_weights.isnan().any()


def test_active_adapter_uploads_nothing(slot_writes, monkeypatch):
    """vLLM activates every adapter of the batch at every step, so an active
    one must cost neither an upload nor a swap of the layers' tensors."""
    tpu_manager, _ = _managers()
    adapter = _adapter(1, tpu_manager.modules)
    _activate((tpu_manager,), adapter)
    writer = tpu_manager._slot_writer
    assert writer is not None
    monkeypatch.setattr(writer, "redirect_writes_to_host", None)
    assert not tpu_manager.activate_adapter(adapter.id)
    assert slot_writes == [0]


def test_lru_eviction_writes_the_freed_slot(slot_writes):
    tpu_manager, vllm_manager = _managers()
    adapters = {
        lora_id: _adapter(lora_id, vllm_manager.modules)
        for lora_id in range(1, NUM_SLOTS + 2)
    }
    # 1, 2 and 3 fill the slots; touching 1 again leaves 2 the oldest, so 4
    # takes the slot 2 frees.
    for lora_id in (1, 2, 3, 1, 4):
        _activate((tpu_manager, vllm_manager), adapters[lora_id])
    assert slot_writes == [0, 1, 2, 1]
    assert tpu_manager.lora_index_to_id == [1, 4, 3]
    _assert_same_slots(tpu_manager, vllm_manager)


def test_failed_activation_restores_tensors_and_frees_the_slot(slot_writes):
    """A set_lora that raises leaves the layers on their TPU tensors, and the
    slot vLLM gave the adapter is free again."""
    tpu_manager, _ = _managers()
    device_weights = {
        name: (layer.lora_a_stacked, layer.lora_b_stacked)
        for name, layer in tpu_manager.modules.items()
    }
    too_wide = _adapter(1, tpu_manager.modules, rank=MAX_RANK + 1)
    with pytest.raises(RuntimeError):
        _activate((tpu_manager,), too_wide)
    for name, layer in tpu_manager.modules.items():
        assert layer.lora_a_stacked is device_weights[name][0], name
        assert layer.lora_b_stacked is device_weights[name][1], name
    assert slot_writes == []
    assert tpu_manager.lora_index_to_id == [None] * NUM_SLOTS


def test_failed_slot_write_frees_the_slot(slot_writes, monkeypatch):
    """vLLM gives the adapter its slot before the weights land. When the write
    fails, e.g. the upload runs out of HBM, the slot must not keep the adapter,
    or its requests would read stale weights; the next activation writes it."""
    tpu_manager, vllm_manager = _managers()
    adapter = _adapter(1, vllm_manager.modules)
    write_on_host = lora_manager.write_adapter_to_slot

    def failing_write(*args):
        raise RuntimeError("RESOURCE_EXHAUSTED")

    monkeypatch.setattr(lora_manager, "write_adapter_to_slot", failing_write)
    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        _activate((tpu_manager,), adapter)
    assert tpu_manager.lora_index_to_id == [None] * NUM_SLOTS

    monkeypatch.setattr(lora_manager, "write_adapter_to_slot", write_on_host)
    _activate((tpu_manager, vllm_manager), adapter)
    assert slot_writes == [0]
    _assert_same_slots(tpu_manager, vllm_manager)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_adapter_with_nan_or_inf_weights_fails_to_load(monkeypatch, bad):
    """NaN or Inf in one adapter's LoRA B would reach every request, since
    lora_ops' expand sums over every slot, so an adapter with NaN or Inf in
    either matrix fails to load."""
    lora_a = torch.randn(MAX_RANK, HIDDEN)
    lora_b = torch.randn(Q, MAX_RANK)
    weights = LoRALayerWeights("column", MAX_RANK, MAX_RANK, lora_a, lora_b)
    adapter = LoRAModel(1, MAX_RANK, {"column": weights})
    monkeypatch.setattr(
        LRUCacheWorkerLoRAManager, "_load_adapter", lambda self, request: adapter
    )
    worker = TPUWorkerLoRAManager.__new__(TPUWorkerLoRAManager)
    request = LoRARequest("adapter", 1, "/unused")
    assert worker._load_adapter(request) is adapter

    for matrix in (lora_a, lora_b):
        matrix[0, 0] = bad
        with pytest.raises(ValueError, match="NaN or Inf weights in column"):
            worker._load_adapter(request)
        matrix[0, 0] = 0


class _CustomColumnLoRA(ColumnParallelLinearWithLoRA):
    """A layer type whose writes the host buffer was never checked against."""


def _add_custom_layer(layers):
    layers["custom"] = _linear_layer(
        _CustomColumnLoRA, [(1, MAX_RANK, HIDDEN)], [(1, Q, MAX_RANK)]
    )


def _mix_dtypes(layers):
    row = layers["row"]
    row.lora_b_stacked = (row.lora_b_stacked[0].double(),)


def _share_a_layer(layers):
    layers["column_again"] = layers["column"]


@pytest.mark.parametrize(
    "make_unsupported", [_add_custom_layer, _mix_dtypes, _share_a_layer]
)
def test_unsupported_layers_fall_back_to_vllm_writes(slot_writes, make_unsupported):
    def unsupported_layers():
        layers = _layers()
        make_unsupported(layers)
        return layers

    tpu_manager, vllm_manager = _managers(unsupported_layers)
    assert tpu_manager._slot_writer is None
    _activate((tpu_manager, vllm_manager), _adapter(1, vllm_manager.modules))
    assert slot_writes == []
    _assert_same_slots(tpu_manager, vllm_manager)


def test_every_vllm_lora_layer_class_is_classified():
    """A layer class joins _SUPPORTED_LAYERS only after its reset_lora and
    set_lora were read to write through lora_a_stacked[index] and
    lora_b_stacked[index] alone. When vLLM adds a class, this fails until it is
    added there or listed here, with the reason it keeps vLLM's own writes."""
    takes_vllm_writes = {
        # Built only with fully_sharded_loras, which the TPU platform rejects.
        ColumnParallelLinearWithShardedLoRA,
        MergedColumnParallelLinearWithShardedLoRA,
        MergedQKVParallelLinearWithShardedLoRA,
        QKVParallelLinearWithShardedLoRA,
        RowParallelLinearWithShardedLoRA,
        # MoE models are rejected; the expert weights live in w13 and w2
        # tensors, not in lora_a_stacked and lora_b_stacked.
        FusedMoEWithLoRA,
        FusedMoE3DWithLoRA,
        # Writes the saved classification head into full_weight_stacked and
        # full_bias_stacked, which the slot write does not carry.
        ClassificationHeadWithLoRA,
    }
    exported = (getattr(vllm.lora.layers, name) for name in vllm.lora.layers.__all__)
    layer_classes = {
        cls
        for cls in exported
        if isinstance(cls, type) and issubclass(cls, BaseLayerWithLoRA)
    } - {BaseLayerWithLoRA}

    assert layer_classes == set(lora_manager._SUPPORTED_LAYERS) | takes_vllm_writes


def test_worker_without_lora_layers_activates_without_writes(slot_writes):
    """A worker can hold none of the LoRA layers, e.g. a pipeline stage that
    no target module falls in."""
    tpu_manager, vllm_manager = _managers(dict)
    assert tpu_manager._slot_writer is None
    _activate((tpu_manager, vllm_manager), _adapter(1, {}))
    assert slot_writes == []
    assert tpu_manager.lora_index_to_id == vllm_manager.lora_index_to_id
    assert tpu_manager.slot_write_hbm_bytes() == 0


def _all_stacked(manager):
    return [
        weights for layer in manager.modules.values() for weights in _stacked(layer)
    ]


def test_slot_write_reserves_one_slot():
    """The slot writer uploads one slot of every stacked tensor."""
    tpu_manager, _ = _managers()

    assert tpu_manager.slot_write_hbm_bytes() == sum(
        weights[0].nbytes for weights in _all_stacked(tpu_manager)
    )


def test_vllm_writes_reserve_two_copies_and_one_slot():
    """vLLM's writes copy each whole stacked tensor while the old buffer still
    exists: up to two new copies, plus the adapter's slot."""

    def unsupported_layers():
        layers = _layers()
        _add_custom_layer(layers)
        return layers

    tpu_manager, _ = _managers(unsupported_layers)
    assert tpu_manager._slot_writer is None

    assert tpu_manager.slot_write_hbm_bytes() == sum(
        2 * weights.nbytes + weights[0].nbytes for weights in _all_stacked(tpu_manager)
    )


# The tests below run on the chip, each in a spawned child. The child imports
# the real slot write, not the host stand-in that slot_writes installs here,
# and spawns_tpu_workers runs them before tests that hold the chip in-process.


def _tpu_layers():
    return _layers(TPU, torch.bfloat16)


def _large_logits_layer():
    """A layer whose slot is 128 MiB: its upload outlasts writing the next
    adapter on the host, and a second copy of all its slots would show in the
    peak memory."""
    return {
        "logits": _vocab_layer(
            LogitsProcessorWithLoRA,
            (1, MAX_RANK, HIDDEN),
            (1, 1 << 23, MAX_RANK),
            TPU,
            torch.bfloat16,
        )
    }


def _tpu_managers(make_layers=_tpu_layers):
    # The engine runs eager ops in this mode.
    execution_mode.eager_mode = execution_mode.EagerMode.DEFER_AND_FUSE
    torch.default_generator.manual_seed(0)
    return _managers(make_layers, torch.device(TPU))


def _dynamo_graphs():
    return torch._dynamo.utils.counters["stats"]["unique_graphs"]


def _slot_write_matches_vllm_writes_on_tpu():
    tpu_manager, vllm_manager = _tpu_managers()
    layers = vllm_manager.modules
    for adapter in (
        _adapter(1, layers),
        _adapter(2, layers, rank=4, drop_second_slice=True),
        _adapter(3, layers, skip=("embed", "logits")),
        _adapter(4, layers),  # Evicts 1 and overwrites slot 0.
    ):
        _activate((tpu_manager, vllm_manager), adapter)
        _assert_same_slots(tpu_manager, vllm_manager)
    expected = vllm_manager.modules["embed"].lora_a_stacked
    assert torch.equal(
        tpu_manager.modules["embed"].lora_a_stacked_2d.cpu(),
        expected.reshape(NUM_SLOTS * VOCAB, MAX_RANK).cpu(),
    )


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
def test_slot_write_matches_vllm_writes_on_tpu():
    """The real program writes what vLLM's eager writes do, including through
    the embedding's 2D view."""
    run_in_isolated_process(_slot_write_matches_vllm_writes_on_tpu)


def _back_to_back_adapters_both_land_on_tpu():
    tpu_manager, vllm_manager = _tpu_managers(_large_logits_layer)
    # As at startup, so the first write does not compile while its upload runs.
    tpu_manager.precompile_activation()
    layers = vllm_manager.modules
    first, second = _adapter(1, layers), _adapter(2, layers)
    # Nothing syncs between the two, so the second rewrites the host buffer
    # right after the first upload returns.
    _activate((tpu_manager,), first)
    _activate((tpu_manager,), second)
    _activate((vllm_manager,), first)
    _activate((vllm_manager,), second)
    _assert_same_slots(tpu_manager, vllm_manager)


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
def test_back_to_back_adapters_both_land_on_tpu():
    """One step can bring in two adapters. The first upload must be done
    reading the host buffer before the second adapter is written into it."""
    run_in_isolated_process(_back_to_back_adapters_both_land_on_tpu)


def _slot_write_compiles_once_for_every_slot():
    tpu_manager, _ = _tpu_managers()
    layers = tpu_manager.modules
    tpu_manager.precompile_activation()
    torch.accelerator.synchronize()
    graphs, programs = _dynamo_graphs(), torch.tpu._get_cache_stats().num_cache_reqs
    # 1, 2 and 3 use each slot for the first time and 4 overwrites slot 0.
    # Grad mode alternates, since the add_lora RPC runs with grad enabled.
    for lora_id in range(1, NUM_SLOTS + 2):
        with torch.set_grad_enabled(lora_id % 2 == 0):
            _activate((tpu_manager,), _adapter(lora_id, layers))
    torch.accelerator.synchronize()
    assert _dynamo_graphs() == graphs
    assert torch.tpu._get_cache_stats().num_cache_reqs == programs


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
def test_slot_write_compiles_once_for_every_slot():
    """After the startup precompile, no adapter swap traces again or runs an
    eager program."""
    run_in_isolated_process(_slot_write_compiles_once_for_every_slot)


def _slot_write_holds_at_most_one_extra_slot():
    tpu_manager, _ = _tpu_managers(_large_logits_layer)
    writer = tpu_manager._slot_writer
    assert writer is not None
    tpu_manager.precompile_activation()
    adapter = _adapter(1, tpu_manager.modules)
    torch.accelerator.synchronize()
    before = torch.accelerator.memory_allocated(TPU)
    torch.accelerator.reset_peak_memory_stats(TPU)
    _activate((tpu_manager,), adapter)
    torch.accelerator.synchronize()
    extra = torch.accelerator.max_memory_allocated(TPU) - before
    assert extra <= writer._host_slot_weights.nbytes + 16 * 2**20, extra


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
def test_slot_write_holds_at_most_one_extra_slot():
    """A swap may hold the uploaded slot, but not a second copy of the stacked
    tensors, which a compiled torch write without donation would."""
    run_in_isolated_process(_slot_write_holds_at_most_one_extra_slot)


def _compiled_reader_runs_no_fixups_after_activation():
    tpu_manager, _ = _tpu_managers()
    stacked = [
        weights for layer in tpu_manager.modules.values() for weights in _stacked(layer)
    ]

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def read(stacked):
        return torch.stack([weights.float().sum() for weights in stacked]).sum()

    # Compiled before any adapter is loaded, as the backbone is at startup.
    read(stacked).cpu()
    for lora_id in range(1, NUM_SLOTS + 2):
        _activate((tpu_manager,), _adapter(lora_id, tpu_manager.modules))
    torch.accelerator.synchronize()
    programs = torch.tpu._get_cache_stats().num_cache_reqs
    read(stacked).cpu()
    assert torch.tpu._get_cache_stats().num_cache_reqs == programs


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
def test_compiled_reader_runs_no_fixups_after_activation():
    """Writes that change the stacked tensors' device layout make every later
    compiled program relayout them first; the slot write keeps the layout."""
    run_in_isolated_process(_compiled_reader_runs_no_fixups_after_activation)
