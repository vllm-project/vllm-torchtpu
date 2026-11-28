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

from vllm_torchtpu.distributed import tpu_communicator as comm


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
    tensors = [torch.tensor([[i, i + 1]]) for i in range(count)]
    ep_group.all_gather.side_effect = lambda x, dim: torch.cat([x, x + 10], dim=dim)
    ep_group.reduce_scatter.return_value = torch.tensor([[11, 12]])

    result = getattr(communicator, method)(*tensors, is_sequence_parallel=True)
    if method == "combine":
        ep_group.reduce_scatter.assert_called_once_with(tensors[0], dim=0)
        assert result is ep_group.reduce_scatter.return_value
        ep_group.all_gather.assert_not_called()
    else:
        assert len(result) == count
        for actual, expected, invocation in zip(
            result, tensors, ep_group.all_gather.call_args_list
        ):
            torch.testing.assert_close(actual, torch.cat([expected, expected + 10]))
            assert invocation.args[0] is expected
            assert invocation.kwargs == {"dim": 0}
        assert ep_group.all_gather.call_count == count
        ep_group.reduce_scatter.assert_not_called()


class _FakeAllReduce:
    """Serves the group of ranks (0, 1), bf16 only."""

    def __init__(self, ranks=(0, 1)):
        self.ranks = ranks
        self.setups = []
        self.calls = []

    def setup(self, model, vllm_config, device):
        self.setups.append((model, vllm_config, device))
        return self.ranks

    def all_reduce(self, ranks, x):
        self.calls.append((tuple(ranks), tuple(x.shape)))
        if x.dtype != torch.bfloat16:
            return None
        return x + 1


class _Model(torch.nn.Module):
    """A top-level model that supplies its own all-reduce."""

    def __init__(self, impl):
        super().__init__()
        self.impl = impl

    def get_model_all_reduce(self):
        return self.impl


class _ModuleAllReduce(torch.nn.Module, _FakeAllReduce):
    """An implementation that is itself an `nn.Module` (a submodule)."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        _FakeAllReduce.__init__(self)


@pytest.fixture
def base_all_reduce(monkeypatch):
    calls = []

    def all_reduce(instance, x):
        calls.append(tuple(x.shape))
        return x * 2

    monkeypatch.setattr(comm.DeviceCommunicatorBase, "all_reduce", all_reduce)
    monkeypatch.setattr(comm, "_model_all_reduce", None)
    return calls


def test_model_all_reduce_off_by_default(communicator, base_all_reduce):
    communicator.ranks = [0, 1]
    assert not comm.install_model_all_reduce(Mock(spec=[]), None, "cpu")
    # A module without the capability, and one that declares no implementation.
    plain = torch.nn.Linear(1, 1)
    assert not isinstance(plain, comm.SupportsModelAllReduce)
    assert not comm.install_model_all_reduce(plain, None, "cpu")
    declines = _Model(None)
    assert isinstance(declines, comm.SupportsModelAllReduce)
    assert not comm.install_model_all_reduce(declines, None, "cpu")
    x = torch.ones(8, 128, dtype=torch.bfloat16)
    assert torch.equal(communicator.all_reduce(x), x * 2)
    assert base_all_reduce == [(8, 128)]


def test_model_all_reduce_dispatch(communicator, base_all_reduce):
    impl = _FakeAllReduce()
    model = _Model(impl)
    assert comm.install_model_all_reduce(model, "cfg", "dev")
    assert impl.setups == [(model, "cfg", "dev")]
    x = torch.ones(8, 128, dtype=torch.bfloat16)
    communicator.ranks = [0, 1]
    assert torch.equal(communicator.all_reduce(x), x + 1)
    # The implementation declines: vLLM's all-reduce.
    assert torch.equal(communicator.all_reduce(x.float()), x.float() * 2)
    # Another group (e.g. PCP): never offered to the implementation.
    communicator.ranks = [0, 2]
    assert torch.equal(communicator.all_reduce(x), x * 2)
    assert impl.calls == [((0, 1), (8, 128))] * 2
    assert base_all_reduce == [(8, 128)] * 2


def test_model_all_reduce_module_implementation(communicator, base_all_reduce):
    """An implementation that is an `nn.Module` is found through the method."""
    impl = _ModuleAllReduce()
    assert comm.install_model_all_reduce(_Model(impl), None, "cpu")
    communicator.ranks = [0, 1]
    x = torch.ones(8, 128, dtype=torch.bfloat16)
    assert torch.equal(communicator.all_reduce(x), x + 1)


def test_model_all_reduce_not_installed_when_setup_declines(
    communicator, base_all_reduce
):
    impl = _FakeAllReduce(ranks=None)
    assert not comm.install_model_all_reduce(_Model(impl), None, "cpu")
    communicator.ranks = [0, 1]
    x = torch.ones(8, 128, dtype=torch.bfloat16)
    assert torch.equal(communicator.all_reduce(x), x * 2)
    assert impl.calls == []


def test_model_all_reduce_needs_every_rank(monkeypatch, base_all_reduce):
    """If another rank declines, this rank does not install either."""
    from vllm_torchtpu.distributed import hier_reduce_scatter as hrs

    monkeypatch.setattr(
        hrs, "_gather_reasons", lambda reason: [reason, "setup() declined"]
    )
    assert not comm.install_model_all_reduce(_Model(_FakeAllReduce()), None, "cpu")
    assert comm._model_all_reduce is None


def test_model_all_reduce_reinstall_clears_previous(base_all_reduce):
    assert comm.install_model_all_reduce(_Model(_FakeAllReduce()), None, "cpu")
    assert not comm.install_model_all_reduce(Mock(spec=[]), None, "cpu")
    assert comm._model_all_reduce is None


def test_model_all_reduce_cleared_on_destroy(communicator, base_all_reduce):
    assert comm.install_model_all_reduce(_Model(_FakeAllReduce()), None, "cpu")
    communicator.destroy()
    assert comm._model_all_reduce is None
    communicator.ranks = [0, 1]
    x = torch.ones(8, 128, dtype=torch.bfloat16)
    assert torch.equal(communicator.all_reduce(x), x * 2)


def test_model_all_reduce_compiles_without_a_graph_break(communicator, base_all_reduce):
    impl = _FakeAllReduce()
    assert comm.install_model_all_reduce(_Model(impl), None, "cpu")
    communicator.ranks = [0, 1]
    fn = torch.compile(communicator.all_reduce, backend="eager", fullgraph=True)
    x = torch.ones(8, 128, dtype=torch.bfloat16)
    assert torch.equal(fn(x), x + 1)
    assert impl.calls == [((0, 1), (8, 128))]


def test_load_model_installs_through_the_runner(monkeypatch):
    import inspect

    from vllm_torchtpu.runner import tpu_runner

    runner_cls = tpu_runner.TPUModelRunner
    assert "self._install_model_all_reduce()" in inspect.getsource(
        runner_cls.load_model
    )
    seen = []
    monkeypatch.setattr(
        comm,
        "install_model_all_reduce",
        lambda model, vllm_config, device: seen.append((model, vllm_config, device)),
    )
    runner = object.__new__(runner_cls)
    runner.model, runner.vllm_config, runner.device = "model", "cfg", "dev"
    runner._install_model_all_reduce()
    assert seen == [("model", "cfg", "dev")]
