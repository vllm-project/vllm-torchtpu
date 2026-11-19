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

from unittest.mock import Mock

import pytest
import torch

from tests.distributed.tpu_test_utils import run_in_isolated_process
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.distributed import tpu_communicator as comm


@pytest.fixture(autouse=True)
def _tpu7x(monkeypatch):
    """Packing is gated to tpu7x; CPU tests have no chip to ask."""
    monkeypatch.setattr(comm, "_chip_is_tpu7x", lambda: True)


def _dispatch_tensors(count):
    """bf16 hidden states, then fp32 weights / logits and int32 ids."""
    dtypes = [torch.bfloat16, torch.float32, torch.int32][:count]
    return [torch.tensor([[i, i + 1]]).to(dtype) for i, dtype in enumerate(dtypes)]


def _assert_one_packed_gather(group, result, tensors):
    """Dispatch issued one packed all-gather and unpacked it exactly.

    The fake all-gather appends the input with its rows reversed, standing in
    for a second rank's distinct rows.
    """
    assert group.all_gather.call_count == 1
    packed = group.all_gather.call_args.args[0]
    assert packed.dtype == comm._PACK_CARRIER
    assert group.all_gather.call_args.kwargs == {"dim": 0}
    assert len(result) == len(tensors)
    for actual, expected in zip(result, tensors):
        torch.testing.assert_close(
            actual, torch.cat([expected, expected.flip(0)]), rtol=0, atol=0
        )
    group.reduce_scatter.assert_not_called()


@pytest.fixture
def communicator(monkeypatch):
    def init(instance, *args, **kwargs):
        instance.use_all2all = False

    monkeypatch.setattr(comm.DeviceCommunicatorBase, "__init__", init)
    return comm.TpuDeviceCommunicator()


@pytest.mark.parametrize("use_all2all", [False, True])
def test_all2all_probe_manager(monkeypatch, use_all2all):
    cpu_group = object()

    def init(instance, *args, **kwargs):
        instance.use_all2all = use_all2all
        instance.cpu_group = cpu_group

    monkeypatch.setattr(comm.DeviceCommunicatorBase, "__init__", init)
    manager = Mock()
    monkeypatch.setattr(comm, "TpuNullAll2AllManager", manager)
    result = comm.TpuDeviceCommunicator()
    if use_all2all:
        manager.assert_called_once_with(cpu_group)
        assert result.all2all_manager is manager.return_value
    else:
        manager.assert_not_called()


@pytest.mark.parametrize(
    "method,count", [("dispatch", 3), ("dispatch_router_logits", 2), ("combine", 1)]
)
@pytest.mark.parametrize("dp_size", [1, 2])
def test_dispatch_and_combine_use_dp_group(
    monkeypatch, communicator, method, count, dp_size
):
    group = Mock(world_size=dp_size)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    tensors = _dispatch_tensors(count)
    group.all_gather.side_effect = lambda x, dim: torch.cat([x, x.flip(0)], dim=dim)
    group.reduce_scatter.return_value = torch.tensor([[11, 12]])
    result = getattr(communicator, method)(*tensors)
    if method == "combine":
        if dp_size == 1:
            assert result is tensors[0]
            group.reduce_scatter.assert_not_called()
        else:
            group.reduce_scatter.assert_called_once_with(tensors[0], dim=0)
            assert result is group.reduce_scatter.return_value
        group.all_gather.assert_not_called()
    elif dp_size == 1:
        assert len(result) == count
        assert all(actual is expected for actual, expected in zip(result, tensors))
        group.all_gather.assert_not_called()
    else:
        _assert_one_packed_gather(group, result, tensors)


@pytest.mark.parametrize(
    "method,count", [("dispatch", 3), ("dispatch_router_logits", 2)]
)
def test_extra_tensors_rejected_before_collectives(
    monkeypatch, communicator, method, count
):
    get_group = Mock()
    monkeypatch.setattr(comm, "get_dp_group", get_group)
    with pytest.raises(NotImplementedError, match="extra_tensors"):
        getattr(communicator, method)(*[torch.ones(1)] * count, extra_tensors=[])
    get_group.assert_not_called()


def test_moe_group_with_sequence_parallel(monkeypatch, communicator):
    ep_group = Mock(world_size=8)
    dp_group = Mock(world_size=1)
    monkeypatch.setattr(comm, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(comm, "get_dp_group", lambda: dp_group)
    assert communicator._moe_group(is_sequence_parallel=True) is ep_group


def test_moe_group_without_sequence_parallel(monkeypatch, communicator):
    dp_group_1 = Mock(world_size=1)
    monkeypatch.setattr(comm, "get_dp_group", lambda: dp_group_1)
    assert communicator._moe_group(is_sequence_parallel=False) is None

    dp_group_4 = Mock(world_size=4)
    monkeypatch.setattr(comm, "get_dp_group", lambda: dp_group_4)
    assert communicator._moe_group(is_sequence_parallel=False) is dp_group_4


@pytest.mark.parametrize(
    "method,count", [("dispatch", 3), ("dispatch_router_logits", 2), ("combine", 1)]
)
def test_dispatch_and_combine_with_sequence_parallel(
    monkeypatch, communicator, method, count
):
    ep_group = Mock(world_size=8)
    monkeypatch.setattr(comm, "get_ep_group", lambda: ep_group)
    tensors = _dispatch_tensors(count)
    ep_group.all_gather.side_effect = lambda x, dim: torch.cat([x, x.flip(0)], dim=dim)
    ep_group.reduce_scatter.return_value = torch.tensor([[11, 12]])

    result = getattr(communicator, method)(*tensors, is_sequence_parallel=True)
    if method == "combine":
        ep_group.reduce_scatter.assert_called_once_with(tensors[0], dim=0)
        assert result is ep_group.reduce_scatter.return_value
        ep_group.all_gather.assert_not_called()
    else:
        _assert_one_packed_gather(ep_group, result, tensors)


def test_packed_gather_is_bit_exact_across_dtypes_and_shapes(monkeypatch, communicator):
    """bf16 hidden states (with inf/nan), fp32 weights and int32 ids, with a
    trailing extra dim, survive the int16 carrier bit for bit."""
    group = Mock(world_size=2)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    group.all_gather.side_effect = lambda x, dim: torch.cat([x, x.flip(0)], dim=dim)
    hidden = torch.randn(3, 2, 4).to(torch.bfloat16)
    hidden[0, 0, 0] = float("inf")
    hidden[1, 1, 3] = float("nan")
    weights = torch.rand(3, 8, dtype=torch.float32)
    ids = torch.randint(0, 256, (3, 8), dtype=torch.int32)

    out = communicator.dispatch(hidden, weights, ids)

    assert group.all_gather.call_count == 1
    for actual, expected in zip(out, (hidden, weights, ids)):
        reference = torch.cat([expected, expected.flip(0)])
        assert actual.dtype == expected.dtype
        assert actual.shape == reference.shape
        assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))


def test_packed_gather_handles_zero_tokens(monkeypatch, communicator):
    group = Mock(world_size=2)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    group.all_gather.side_effect = lambda x, dim: torch.cat([x, x], dim=dim)
    hidden = torch.empty(0, 4, dtype=torch.bfloat16)
    logits = torch.empty(0, 6, dtype=torch.float32)

    out_hidden, out_logits = communicator.dispatch_router_logits(hidden, logits)

    assert out_hidden.shape == (0, 4) and out_hidden.dtype == torch.bfloat16
    assert out_logits.shape == (0, 6) and out_logits.dtype == torch.float32


def test_packed_gather_rejects_mismatched_token_counts(monkeypatch, communicator):
    group = Mock(world_size=2)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    with pytest.raises(ValueError, match="same token count"):
        communicator.dispatch_router_logits(torch.ones(2, 4), torch.ones(3, 4))
    group.all_gather.assert_not_called()


def test_packed_gather_carrier_is_int16(monkeypatch, communicator):
    """The carrier has the bf16 hidden states' width but is an integer type,
    so fp32 logits ride as two int16 lanes per value."""
    group = Mock(world_size=2)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    group.all_gather.side_effect = lambda x, dim: torch.cat([x, x], dim=dim)
    hidden = torch.randn(4, 6).to(torch.bfloat16)
    logits = torch.randn(4, 5, dtype=torch.float32)

    communicator.dispatch_router_logits(hidden, logits)

    packed = group.all_gather.call_args.args[0]
    assert packed.dtype == torch.int16
    assert packed.shape == (4, 6 + 2 * 5)


def test_packed_gather_falls_back_when_sizes_do_not_divide(monkeypatch, communicator):
    """A 16-bit carrier cannot hold 1-byte elements; each tensor is then
    gathered on its own, with a warning naming the dtypes."""
    group = Mock(world_size=2)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    group.all_gather.side_effect = lambda x, dim: torch.cat([x, x], dim=dim)
    warnings = []
    monkeypatch.setattr(comm, "_log_unpackable", warnings.append)
    hidden = torch.randn(4, 6).to(torch.bfloat16)
    logits = torch.randint(0, 255, (4, 5), dtype=torch.uint8)

    out_hidden, out_logits = communicator.dispatch_router_logits(hidden, logits)

    assert group.all_gather.call_count == 2
    torch.testing.assert_close(out_hidden, torch.cat([hidden, hidden]))
    torch.testing.assert_close(out_logits, torch.cat([logits, logits]))
    assert warnings == ["torch.bfloat16, torch.uint8"]


@pytest.mark.parametrize("hidden_dtype", [torch.float32, torch.float8_e4m3fn])
def test_non_bf16_hidden_states_are_gathered_tensor_by_tensor(
    monkeypatch, communicator, hidden_dtype
):
    """Only bf16 hidden states (the measured case) pack; fp32 or fp8 hidden
    states keep per-tensor gathers."""
    group = Mock(world_size=2)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    group.all_gather.side_effect = lambda x, dim: torch.cat([x, x], dim=dim)
    hidden = torch.zeros(4, 6).to(hidden_dtype)
    logits = torch.zeros(4, 5, dtype=torch.float32)

    communicator.dispatch_router_logits(hidden, logits)

    assert group.all_gather.call_count == 2


def test_other_chips_are_gathered_tensor_by_tensor(monkeypatch, communicator):
    """The cutoff and the A/B data are from tpu7x only."""
    monkeypatch.setattr(comm, "_chip_is_tpu7x", lambda: False)
    group = Mock(world_size=2)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    group.all_gather.side_effect = lambda x, dim: torch.cat([x, x], dim=dim)

    communicator.dispatch(*_dispatch_tensors(3))

    assert group.all_gather.call_count == 3


@pytest.mark.parametrize(("aot_cache_tag", "packed"), [(None, False), ("trace0", True)])
def test_traces_outside_warmup_do_not_branch_on_token_count(
    monkeypatch, communicator, aot_cache_tag, packed
):
    """A module traced once for every bucket (outside `shape_variants.warmup`)
    keeps per-tensor gathers; a warmup trace may specialize and pack."""
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(shape_variants, "aot_cache_tag", lambda: aot_cache_tag)
    group = Mock(world_size=2)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    group.all_gather.side_effect = lambda x, dim: torch.cat([x, x], dim=dim)

    communicator.dispatch(*_dispatch_tensors(3))

    assert group.all_gather.call_count == (1 if packed else 3)


def test_prefill_sized_steps_are_gathered_tensor_by_tensor(monkeypatch, communicator):
    """Above the decode-size cap the copies outweigh the saved collective."""
    group = Mock(world_size=2)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    group.all_gather.side_effect = lambda x, dim: torch.cat([x, x], dim=dim)
    num_tokens = comm._PACK_MAX_TOKENS + 1
    hidden = torch.zeros(num_tokens, 4, dtype=torch.bfloat16)
    logits = torch.zeros(num_tokens, 3, dtype=torch.float32)

    communicator.dispatch_router_logits(hidden, logits)

    assert group.all_gather.call_count == 2
    assert group.all_gather.call_args_list[0].args[0] is hidden
    assert group.all_gather.call_args_list[1].args[0] is logits


def _packed_gather_bit_exact_on_device():
    """Body of the on-device test; runs in a spawned process (see below)."""
    import torch_tpu  # noqa: F401

    device = torch.device("tpu")
    # Pretend to be a warmup trace on tpu7x, where packing is allowed.
    shape_variants.aot_cache_tag = lambda: "trace0"
    comm._chip_is_tpu7x = lambda: True
    gathers = []

    class FakeGroup:
        def all_gather(self, x, dim):
            gathers.append(x.dtype)
            return torch.cat([x, x.flip(0)], dim=dim)

    group = FakeGroup()

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def dispatch(hidden, weights, ids):
        return tuple(comm._packed_all_gather(group, [hidden, weights, ids]))

    num_tokens, top_k = 256, 8
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(num_tokens, 64, generator=generator).to(torch.bfloat16)
    hidden[0, 0], hidden[1, 1], hidden[2, 2] = float("nan"), float("inf"), -0.0
    weights = torch.rand(num_tokens, top_k, generator=generator)
    weights[0, :4] = torch.tensor([1e-40, float("nan"), float("-inf"), -0.0])
    ids = (torch.arange(num_tokens * top_k, dtype=torch.int32) % 256).reshape(
        num_tokens, top_k
    )
    inputs = (hidden, weights, ids)

    outputs = dispatch(*(t.to(device) for t in inputs))

    assert gathers == [torch.int16]  # one packed collective
    for actual, expected in zip(outputs, inputs):
        reference = torch.cat([expected, expected.flip(0)])
        actual = actual.cpu()
        assert actual.dtype == reference.dtype and actual.shape == reference.shape
        assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
def test_packed_gather_is_bit_exact_on_device_under_compile():
    """On TPU, inside a fullgraph compile, int32 ids (small ids have bf16
    subnormal lanes), fp32 weights with subnormals / NaN / inf / -0 and bf16
    hidden states with NaN / inf come back bit-identical to per-tensor
    gathers. A bf16 carrier fails this: TPU flushes the subnormal lanes.

    Runs in a spawned process so its TPU handles do not block later tests
    that start their own engine processes.
    """
    run_in_isolated_process(_packed_gather_bit_exact_on_device)
