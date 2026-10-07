# SPDX-License-Identifier: Apache-2.0
"""100% Diff coverage test suite for TPU Communicator and SP MoE routing.

Covers all lines and branches in:
- vllm_torchtpu.distributed.tpu_communicator (tpu_communicator.py:15-99)
- vllm_torchtpu.layers.adapter.quantization.deepseek_v41 (deepseek_v41.py:15-57)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_torchtpu.distributed import tpu_communicator as comm
from vllm_torchtpu.layers.adapter.quantization import deepseek_v41 as moe_quant

# ==============================================================================
# 1. Tests for TpuDeviceCommunicator & TpuNullAll2AllManager
# ==============================================================================


@pytest.fixture
def communicator(monkeypatch):
    """Fixture providing an initialized TpuDeviceCommunicator with use_all2all=False."""

    def init(instance, *args, **kwargs):
        instance.use_all2all = False

    monkeypatch.setattr(comm.DeviceCommunicatorBase, "__init__", init)
    return comm.TpuDeviceCommunicator()


@pytest.mark.parametrize("use_all2all", [False, True])
def test_all2all_probe_manager(monkeypatch, use_all2all):
    """Verify all2all_manager initialization when use_all2all is True or False."""
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
    """Verify dispatch, dispatch_router_logits, combine under DP groups."""
    group = Mock(world_size=dp_size)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    tensors = [torch.tensor([[i, i + 1]]) for i in range(count)]
    group.all_gather.side_effect = lambda x, dim: torch.cat([x, x + 10], dim=dim)
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
        assert len(result) == count
        for actual, expected, invocation in zip(
            result, tensors, group.all_gather.call_args_list
        ):
            torch.testing.assert_close(actual, torch.cat([expected, expected + 10]))
            assert invocation.args[0] is expected
            assert invocation.kwargs == {"dim": 0}
        assert group.all_gather.call_count == count
        group.reduce_scatter.assert_not_called()


@pytest.mark.parametrize(
    "method,count", [("dispatch", 3), ("dispatch_router_logits", 2)]
)
def test_extra_tensors_rejected_before_collectives(
    monkeypatch, communicator, method, count
):
    """Verify extra_tensors raises NotImplementedError immediately."""
    get_group = Mock()
    monkeypatch.setattr(comm, "get_dp_group", get_group)
    with pytest.raises(NotImplementedError, match="extra_tensors"):
        getattr(communicator, method)(*[torch.ones(1)] * count, extra_tensors=[])
    get_group.assert_not_called()


@pytest.mark.parametrize(
    "method,count", [("dispatch", 3), ("dispatch_router_logits", 2), ("combine", 1)]
)
def test_sequence_parallel_uses_ep_group(monkeypatch, communicator, method, count):
    """Verify is_sequence_parallel=True directs collectives to EP group."""
    dp, ep = Mock(world_size=1), Mock(world_size=2)
    monkeypatch.setattr(comm, "get_dp_group", lambda: dp)
    monkeypatch.setattr(comm, "get_ep_group", lambda: ep)
    ep.all_gather.side_effect = lambda x, dim: torch.cat([x, x + 10], dim=dim)
    ep.reduce_scatter.return_value = torch.tensor([[11, 12]])
    tensors = [torch.tensor([[i, i + 1]]) for i in range(count)]
    result = getattr(communicator, method)(*tensors, is_sequence_parallel=True)
    if method == "combine":
        ep.reduce_scatter.assert_called_once_with(tensors[0], dim=0)
        assert result is ep.reduce_scatter.return_value
    else:
        assert ep.all_gather.call_count == count
        for actual, expected in zip(result, tensors):
            torch.testing.assert_close(actual, torch.cat([expected, expected + 10]))
    dp.all_gather.assert_not_called()
    dp.reduce_scatter.assert_not_called()


def test_router_logits_alias_gathered_once(monkeypatch, communicator):
    """Verify router_logits is hidden_states gathers only once."""
    ep = Mock(world_size=2)
    monkeypatch.setattr(comm, "get_ep_group", lambda: ep)
    ep.all_gather.side_effect = lambda x, dim: torch.cat([x, x + 10], dim=dim)
    hidden = torch.tensor([[1, 2]])
    gathered, logits = communicator.dispatch_router_logits(
        hidden, hidden, is_sequence_parallel=True
    )
    assert logits is gathered
    ep.all_gather.assert_called_once_with(hidden, dim=0)


def test_router_logits_distinct_tensor_gathered_twice(monkeypatch, communicator):
    """Verify distinct router_logits calls all_gather for both tensors."""
    ep = Mock(world_size=2)
    monkeypatch.setattr(comm, "get_ep_group", lambda: ep)
    ep.all_gather.side_effect = lambda x, dim: torch.cat([x, x + 10], dim=dim)
    hidden = torch.tensor([[1, 2]])
    logits = torch.tensor([[3, 4]])
    g_hidden, g_logits = communicator.dispatch_router_logits(
        hidden, logits, is_sequence_parallel=True
    )
    assert ep.all_gather.call_count == 2
    torch.testing.assert_close(g_hidden, torch.tensor([[1, 2], [11, 12]]))
    torch.testing.assert_close(g_logits, torch.tensor([[3, 4], [13, 14]]))


# ==============================================================================
# 2. Tests for VllmDeepseekV41Fp8Config & DeepseekV41MoE
# ==============================================================================


def test_v41_fp8_config_name():
    """Verify quantization config registration and get_name."""
    assert moe_quant.VllmDeepseekV41Fp8Config.get_name() == "deepseek_v41_fp8"


def test_moe_init_rejects_eplb():
    """Verify DeepseekV41MoE raises when enable_eplb is True."""
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_eplb=True),
        kernel_config=SimpleNamespace(moe_backend="native"),
    )
    with pytest.raises(NotImplementedError, match="does not support EPLB"):
        moe_quant.DeepseekV41MoE(vllm_config)


def test_moe_init_rejects_mega_moe_backend(monkeypatch):
    """Verify DeepseekV41MoE raises when moe_backend is in MEGA_MOE_BACKENDS."""
    monkeypatch.setattr(
        moe_quant.v4_upstream, "MEGA_MOE_BACKENDS", ["mega_v1", "mega_v2"]
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_eplb=False),
        kernel_config=SimpleNamespace(moe_backend="mega_v1"),
    )
    with pytest.raises(NotImplementedError, match="requires the native V4 MoE backend"):
        moe_quant.DeepseekV41MoE(vllm_config)


def test_moe_init_success_wraps_factory(monkeypatch):
    """Verify DeepseekV41MoE __init__ wraps FusedMoEFactory with scale option."""
    captured_options = {}

    def mock_super_init(self, vllm_config, *args, **kwargs):
        pass

    monkeypatch.setattr(moe_quant.v4_upstream, "MEGA_MOE_BACKENDS", ["mega_backend"])
    monkeypatch.setattr(
        moe_quant.DeepseekV4MoE,
        "__init__",
        mock_super_init,
    )

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_eplb=False),
        kernel_config=SimpleNamespace(moe_backend="native"),
    )

    class DummyFactory:
        def __call__(self, **options):
            captured_options.update(options)
            return "factory_result"

    monkeypatch.setattr(moe_quant.v4_upstream, "FusedMoEFactory", DummyFactory())

    orig_device_hooks = moe_quant.device_hooks

    def mock_device_hooks(target, **replacements):
        if "FusedMoEFactory" in replacements:
            replacements["FusedMoEFactory"](foo="bar")
        return orig_device_hooks(target, **replacements)

    monkeypatch.setattr(moe_quant, "device_hooks", mock_device_hooks)

    moe = moe_quant.DeepseekV41MoE(vllm_config)
    assert isinstance(moe, moe_quant.DeepseekV41MoE)
    assert captured_options.get("apply_routed_scale_to_output") is True
    assert captured_options.get("foo") == "bar"


@pytest.mark.parametrize(
    "use_sp,has_input_ids,has_tid2eid,do_naive,expect_gather",
    [
        (True, True, True, True, True),  # All 4 conditions met -> all_gather
        (False, True, True, True, False),  # Not SP -> skip
        (True, False, True, True, False),  # input_ids is None -> skip
        (True, True, False, True, False),  # tid2eid is None -> skip
        (True, True, True, False, False),  # do_naive is False -> skip
    ],
)
def test_moe_forward_sp_input_ids_gathering(
    monkeypatch, use_sp, has_input_ids, has_tid2eid, do_naive, expect_gather
):
    """Verify input_ids is all-gathered under SP hash-routing when do_naive is True."""
    ep_group = Mock()
    ep_group.all_gather.side_effect = lambda x, dim: torch.cat([x, x], dim=dim)
    monkeypatch.setattr(moe_quant, "get_ep_group", lambda: ep_group)

    captured_args = {}

    def mock_super_forward(self, hidden_states, input_ids=None):
        captured_args["hidden_states"] = hidden_states
        captured_args["input_ids"] = input_ids
        return "super_forward_out"

    monkeypatch.setattr(moe_quant.DeepseekV4MoE, "forward", mock_super_forward)

    moe = object.__new__(moe_quant.DeepseekV41MoE)
    moe.use_sequence_parallel = use_sp
    moe.gate = SimpleNamespace(tid2eid=torch.tensor([0, 1, 2]) if has_tid2eid else None)
    moe.experts = SimpleNamespace(do_naive_dispatch_combine=do_naive)

    hidden = torch.randn(4, 16)
    input_ids = torch.tensor([10, 20]) if has_input_ids else None

    out = moe.forward(hidden, input_ids=input_ids)
    assert out == "super_forward_out"
    assert captured_args["hidden_states"] is hidden

    if expect_gather:
        ep_group.all_gather.assert_called_once_with(input_ids, dim=0)
        assert captured_args["input_ids"] is not None
        torch.testing.assert_close(
            captured_args["input_ids"], torch.tensor([10, 20, 10, 20])
        )
    else:
        ep_group.all_gather.assert_not_called()
        assert captured_args["input_ids"] is input_ids
